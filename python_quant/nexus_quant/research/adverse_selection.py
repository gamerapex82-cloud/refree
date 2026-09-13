"""Adverse selection after passive fills (plan_2.md Phase 3, E6).

A passive fill is *adversely selected* when the mid moves against the filled
position right after the fill: you bought on the bid and the price kept
falling, or sold on the ask and it kept rising. We measure it as the
**signed post-fill mark-to-market drift**

    drift_h = s · (mid[t+h] − mid[t])        s = +1 bid fill (bought), −1 ask fill (sold)

in price units, so a **negative** value is adverse. ``P_adverse = P(drift_h < 0)``
(ties excluded — an unchanged mid is neither). Drifts are reported over
``h ∈ {1, 5, 25}`` events by default, grouped by fill side, by the sign of the
order-flow imbalance at the fill, and by the order's queue position at
placement, with a **Newey–West** t-stat on the mean drift (labels at overlapping
horizons are autocorrelated) and a matched pre-fill control: the same signed
drift over the ``h`` events *before* the fill, so ``post − pre`` is the
fill-conditional excess drift, not the tape's own trend.

Inputs are plain sequences so the module runs on the real tape
(``OrderLevelTracker.completed`` + the replayed mid series) and on the synthetic
env alike; nothing here touches the book.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import erf, sqrt
from typing import Any

import numpy as np

from ..book_state import Side


@dataclass(slots=True, frozen=True)
class PassiveFill:
    """One passive execution: tape index of the fill, side of the *resting* order."""

    idx: int
    side: Side
    price: int = 0
    size: int = 0
    ofi: float = 0.0  # order-flow imbalance observed at/just before the fill
    queue_frac: float = float("nan")  # queue position at placement (0 = front)


def _mids(mids: Sequence[float]) -> np.ndarray:
    return np.asarray(mids, dtype=np.float64).reshape(-1)


def post_fill_drift(
    fills: Sequence[PassiveFill],
    mids: Sequence[float],
    h: int,
) -> np.ndarray:
    """Signed post-fill drift ``s·(mid[t+h] − mid[t])`` per fill (NaN if no future).

    ``mids`` is the mid series indexed by tape event (one entry per event the
    book was replayed through). A fill at ``idx`` beyond ``len(mids) − h − 1``
    has no ``t+h`` state and yields NaN — it is dropped by the aggregators.
    """
    m = _mids(mids)
    n = m.size
    out = np.full(len(fills), np.nan, dtype=np.float64)
    for i, f in enumerate(fills):
        t = int(f.idx)
        if t < 0 or t + h >= n or m[t] <= 0 or m[t + h] <= 0:
            continue
        s = 1.0 if f.side == Side.Bid else -1.0
        out[i] = s * (m[t + h] - m[t])
    return out


def nw_tstat(x: Sequence[float], *, lag: int) -> dict[str, float]:
    """Mean, Newey–West (Bartlett) standard error, t-stat and two-sided p-value."""
    a = np.asarray(x, dtype=np.float64)
    a = a[~np.isnan(a)]
    n = a.size
    if n < 3:
        return {"n": float(n), "mean": float(a.mean()) if n else float("nan"),
                "se": float("nan"), "t": float("nan"), "p": float("nan")}
    d = a - a.mean()
    L = int(max(0, min(lag, n - 1)))
    gamma0 = float(np.dot(d, d) / n)
    var = gamma0
    for j in range(1, L + 1):
        gj = float(np.dot(d[j:], d[:-j]) / n)
        var += 2.0 * (1.0 - j / (L + 1)) * gj
    var = max(var, 0.0)
    se = sqrt(var / n) if var > 0 else float("nan")
    mean = float(a.mean())
    t = mean / se if se and se > 0 else float("nan")
    p = 2.0 * (1.0 - _ncdf(abs(t))) if not np.isnan(t) else float("nan")
    return {"n": float(n), "mean": mean, "se": float(se), "t": float(t), "p": float(p)}


def _ncdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def p_adverse(drifts: Sequence[float]) -> float:
    """P(drift < 0) over non-NaN, non-zero drifts (ties are neither adverse nor favorable)."""
    a = np.asarray(drifts, dtype=np.float64)
    a = a[~np.isnan(a)]
    a = a[a != 0.0]
    if a.size == 0:
        return float("nan")
    return float(np.mean(a < 0.0))


def pre_fill_drift(
    fills: Sequence[PassiveFill],
    mids: Sequence[float],
    h: int,
) -> np.ndarray:
    """Matched control: the same signed drift over the ``h`` events *before* the fill.

    ``s·(mid[t−1] − mid[t−1−h])`` — the window ends strictly before the fill
    event, so it excludes the fill's own mechanical BBO move (a fully consumed
    level shifts the mid at ``t`` itself). ``post − pre`` is the fill-conditional
    excess drift: it nets out a tape that was already trending into the fill,
    so E6 reports selection, not momentum.
    """
    m = _mids(mids)
    out = np.full(len(fills), np.nan, dtype=np.float64)
    for i, f in enumerate(fills):
        t = int(f.idx)
        a, b = t - 1 - h, t - 1
        if a < 0 or b >= m.size or m[a] <= 0 or m[b] <= 0:
            continue
        s = 1.0 if f.side == Side.Bid else -1.0
        out[i] = s * (m[b] - m[a])
    return out


def _bucket_ofi(v: float) -> str:
    if np.isnan(v) or v == 0.0:
        return "ofi=0"
    return "ofi>0" if v > 0 else "ofi<0"


def _bucket_queue(q: float) -> str:
    if np.isnan(q):
        return "queue=n/a"
    if q < 1.0 / 3.0:
        return "queue=front"
    if q < 2.0 / 3.0:
        return "queue=mid"
    return "queue=back"


def adverse_selection_report(
    fills: Sequence[PassiveFill],
    mids: Sequence[float],
    *,
    horizons: Sequence[int] = (1, 5, 25),
    min_group: int = 20,
) -> dict[str, Any]:
    """E6 bundle: per horizon, overall + grouped drift stats and P(adverse).

    Returns ``{"n_fills": int, "horizons": {h: {"overall": {...}, "pre_fill":
    {...}, "post_minus_pre": {...}, "groups": {label: {...}}}}}`` where each
    stats dict carries ``n, mean_drift, se_nw, t_nw, p_value, p_adverse``.
    Groups with fewer than ``min_group`` fills are omitted (no claims on 5 fills).
    """
    fills = list(fills)
    result: dict[str, Any] = {"n_fills": len(fills), "horizons": {}}
    for h in horizons:
        d = post_fill_drift(fills, mids, h)
        c = pre_fill_drift(fills, mids, h)
        excess = d - c
        block: dict[str, Any] = {
            "overall": _stats(d, h),
            "pre_fill": _stats(c, h),
            "post_minus_pre": _stats(excess, h),
            "groups": {},
        }
        labels = {
            "side=bid": np.array([f.side == Side.Bid for f in fills]),
            "side=ask": np.array([f.side == Side.Ask for f in fills]),
        }
        for name in ("ofi<0", "ofi=0", "ofi>0"):
            labels[name] = np.array([_bucket_ofi(f.ofi) == name for f in fills])
        for name in ("queue=front", "queue=mid", "queue=back"):
            labels[name] = np.array([_bucket_queue(f.queue_frac) == name for f in fills])
        for name, mask in labels.items():
            if mask.size and int(np.sum(mask & ~np.isnan(d))) >= min_group:
                block["groups"][name] = _stats(d[mask], h)
        result["horizons"][int(h)] = block
    return result


def _stats(d: np.ndarray, h: int) -> dict[str, float]:
    t = nw_tstat(d, lag=int(h))
    return {
        "n": int(t["n"]),
        "mean_drift": t["mean"],
        "se_nw": t["se"],
        "t_nw": t["t"],
        "p_value": t["p"],
        "p_adverse": p_adverse(d),
    }


def fills_from_tracker(completed: Sequence[Any], *, ofi_at: Sequence[float] | None = None) -> list[PassiveFill]:
    """Turn ``OrderLevelTracker.completed`` records with outcome ``"filled"`` into fills.

    The fill index is the tape event of the order's *first* execution
    (``idx_first_fill``); the queue fraction is ``ahead_at_add / (ahead_at_add +
    size0)`` at placement (0 = front of the queue). ``ofi_at[i]`` (optional) supplies the OFI observed at event
    ``i`` so fills can be bucketed by flow sign.
    """
    out: list[PassiveFill] = []
    for o in completed:
        if getattr(o, "outcome", "") != "filled":
            continue
        idx = o.idx_first_fill if getattr(o, "idx_first_fill", None) is not None else o.idx_end
        if idx is None:
            continue
        denom = float(o.ahead_at_add + max(1, o.size0))
        q = float(o.ahead_at_add) / denom if denom > 0 else float("nan")
        ofi_v = float(ofi_at[idx]) if ofi_at is not None and idx < len(ofi_at) else 0.0
        out.append(PassiveFill(idx=int(idx), side=Side(int(o.side)), price=int(o.price),
                               size=int(o.filled), ofi=ofi_v, queue_frac=q))
    return out
