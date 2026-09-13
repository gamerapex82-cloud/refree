"""Stateless LOB features over a ``View`` (a ``BOOK_STATE_DTYPE`` record).

Contract (plan_2.md Phase 1): every function is a pure function of state at
``t``, zero instance state, and reads *only* what is observable at call time.
``test_research.py::test_features_do_not_peek`` mutates the book after the
call and asserts the returned value is unchanged — the leakage lock.

The ``View`` shape mirrors ``StubOrderBook.view()`` / the C++ ``Engine``:

    view["bid_px"], view["bid_sz"], view["bid_ct"]   np.ndarray[DEPTH]
    view["ask_px"], view["ask_sz"], view["ask_ct"]   np.ndarray[DEPTH]
    view["mid"]/["spread"] are NOT stored by the engine — mid/spread are
    derived here from the best bid/ask levels only.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

# A shallow copy of the view dict so pure functions never keep a reference.
from typing import Any

import numpy as np

_View = Any  # dict[str, np.ndarray | int | float ...] -- kept loose on purpose


# ---------------------------------------------------------------------------
# helpers over the view
# ---------------------------------------------------------------------------
def _best(view: _View, side: str) -> tuple[int, int]:
    """Best (price, size) for a side; (0, 0) if the level is empty."""
    px = int(view[f"{side}_px"][0])
    sz = int(view[f"{side}_sz"][0])
    return px, sz


def mid(view: _View) -> float:
    """Tick mid from the BBO (0 if either side is empty)."""
    bp, _ = _best(view, "bid")
    ap, _ = _best(view, "ask")
    if bp <= 0 or ap <= 0:
        return 0.0
    return (bp + ap) / 2.0


def spread_ticks(view: _View) -> int:
    """Best-spread in ticks (0 if either side is empty)."""
    bp, _ = _best(view, "bid")
    ap, _ = _best(view, "ask")
    if bp <= 0 or ap <= 0:
        return 0
    return ap - bp


# ---------------------------------------------------------------------------
# level-1 features
# ---------------------------------------------------------------------------
def lob_imbalance(view: _View, k: int = 1) -> float:
    """(V_bid − V_ask) / (V_bid + V_ask) over the top ``k`` levels."""
    vb = float(np.sum(view["bid_sz"][:k]))
    va = float(np.sum(view["ask_sz"][:k]))
    if vb + va <= 0:
        return 0.0
    return (vb - va) / (vb + va)


def microprice(view: _View, k: int = 1) -> float:
    """VWAP of the BBO prices sized by resting volume — a bias-corrected mid.

    microprice = (V_a·P_b + V_b·P_a) / (V_a + V_b). When ``k > 1`` the ladder
    (up to depth ``k``) is volume-weighted with price-weighted rows.
    """
    bp, _ = _best(view, "bid")
    ap, _ = _best(view, "ask")
    vb = float(np.sum(view["bid_sz"][:k]))
    va = float(np.sum(view["ask_sz"][:k]))
    if vb + va <= 0 or bp <= 0 or ap <= 0:
        return float(mid(view))
    if k == 1:
        return (va * bp + vb * ap) / (vb + va)
    bpx = view["bid_px"][:k].astype(np.float64)
    apx = view["ask_px"][:k].astype(np.float64)
    bsz = view["bid_sz"][:k].astype(np.float64)
    asz = view["ask_sz"][:k].astype(np.float64)
    return float((np.sum(asz * bpx) + np.sum(bsz * apx)) / (bsz.sum() + asz.sum()))


def deep_imbalance(view: _View, k: int = 5, weighted: bool = True) -> float:
    """Multi-level imbalance, distance-weighted if ``weighted``.

    Weighted form down-weights far levels by 1/level index:
        ΔV(t) = Σ_i (B_i − A_i) / i ;  normalized by the total weighted size.
    """
    k = max(1, min(int(k), 10))
    bsz = view["bid_sz"][:k].astype(np.float64)
    asz = view["ask_sz"][:k].astype(np.float64)
    if weighted:
        w = 1.0 / np.arange(1, k + 1, dtype=np.float64)
        num = float(np.sum((bsz - asz) * w))
        den = float(np.sum((bsz + asz) * w))
    else:
        num = float(np.sum(bsz - asz))
        den = float(np.sum(bsz + asz))
    if den <= 0:
        return 0.0
    return num / den


def spread_bps(view: _View) -> float:
    """Best spread in basis points of the mid (0 if no BBO)."""
    m = mid(view)
    s = spread_ticks(view)
    if m <= 0 or s <= 0:
        return 0.0
    return 10_000.0 * s / m


# ---------------------------------------------------------------------------
# order-flow approximation + history features
# ---------------------------------------------------------------------------
def ofi(prev_view: _View | None, cur_view: _View, k: int = 1) -> float:
    """Order-flow imbalance across a book update — Cont–Kukanov–Stoikov (2014).

    Level-1 form (``k=1``): with best bid/ask ``(P_b, V_b)`` / ``(P_a, V_a)``::

        e = 1{P_b ≥ P_b'}·V_b − 1{P_b ≤ P_b'}·V_b' − 1{P_a ≤ P_a'}·V_a + 1{P_a ≥ P_a'}·V_a'

    (primes = previous view). A bid that steps UP contributes its full new size
    (+V_b) and a bid that steps DOWN removes the whole old queue (−V_b');
    symmetric for the ask. When the touch price is unchanged this collapses to
    ``ΔV_b − ΔV_a``. This is the *L2 ladder approximation*: it cannot see
    inside a level, so a cancel and an execution of the same size look alike.
    ``scripts/run_research.py::order_level_ofi`` is the exact order-level
    version (needs the ITCH order stream). For ``k > 1`` the size deltas over
    the top ``k`` levels are used (no price-move term).
    """
    if prev_view is None:
        return 0.0
    if k == 1:
        pb0, vb0 = int(prev_view["bid_px"][0]), float(prev_view["bid_sz"][0])
        pa0, va0 = int(prev_view["ask_px"][0]), float(prev_view["ask_sz"][0])
        pb1, vb1 = int(cur_view["bid_px"][0]), float(cur_view["bid_sz"][0])
        pa1, va1 = int(cur_view["ask_px"][0]), float(cur_view["ask_sz"][0])
        e = 0.0
        if pb0 > 0 and pb1 > 0:
            e += (vb1 if pb1 >= pb0 else 0.0) - (vb0 if pb1 <= pb0 else 0.0)
        if pa0 > 0 and pa1 > 0:
            e -= (va1 if pa1 <= pa0 else 0.0) - (va0 if pa1 >= pa0 else 0.0)
        return e
    vb_prev = float(np.sum(prev_view["bid_sz"][:k]))
    va_prev = float(np.sum(prev_view["ask_sz"][:k]))
    vb_cur = float(np.sum(cur_view["bid_sz"][:k]))
    va_cur = float(np.sum(cur_view["ask_sz"][:k]))
    return (vb_cur - vb_prev) - (va_cur - va_prev)


def realized_vol(mid_hist: Sequence[float], n: int = 10) -> float:
    """Annualized realized vol of the last ``n`` mid moves (zero if < 2 pts)."""
    xs = np.asarray([float(x) for x in mid_hist[-n:]], dtype=np.float64)
    if xs.size < 2:
        return 0.0
    if np.all(xs <= 0):
        return 0.0
    r = np.diff(xs) / xs[:-1]
    mu = r.mean()
    sig = float(np.sqrt(np.sum((r - mu) ** 2) / max(1, r.size - 1)))
    return sig * np.sqrt(252.0)


def momentum(mid_hist: Sequence[float], k: int = 5) -> float:
    """Fractional mid move over ``k`` samples: (m_t/m_{t-k} − 1)."""
    xs = [float(x) for x in mid_hist]
    if len(xs) <= k or xs[-1 - k] <= 0:
        return 0.0
    return (xs[-1] / xs[-1 - k]) - 1.0


def flow_intensity(events: Iterable[Any], tau: int = 10) -> float:
    """Signed event rate over the last ``tau`` events.

    ``events`` are order-level tuples/records with a ``side``-like field and a
    volume; the sign is +1 for bid adds / ask cancels, −1 for the reverse.
    Falls back to 0 when no events or when the records have no side/size.
    """
    s = 0.0
    seen = 0
    for ev in events:
        side = _get_side(ev)
        if side == 0:  # bid
            s += 1.0
        else:
            s -= 1.0
        seen += 1
        if seen >= tau:
            break
    return s / max(1, seen)


def _get_side(ev: Any) -> int:
    """Bid=0 / ask=1 from a dict, dataclass, or tuple-shaped event."""
    if isinstance(ev, dict):
        return int(ev.get("side", 0))
    if hasattr(ev, "side"):
        return int(ev.side)
    # tuple/namedtuple with side as the third field, matching NormalizedEvent.
    if isinstance(ev, tuple):
        return int(ev[2]) if len(ev) >= 3 else 0
    return 0