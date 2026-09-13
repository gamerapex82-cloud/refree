"""Order-level queue tracker + passive fill-probability models (plan_2.md Phase 3, E5).

``OrderLevelTracker`` consumes the normalized ITCH stream (``itch_parser``
events, or the ``(kind, side, price, size, order_id)`` shape the env/replay
already emit) and maintains, **per resting order**, the FIFO queue at its price
level: how much displayed size is *ahead* of it, how much is *behind*, and what
happened to it (filled / cancelled / still resting). That is the ground truth
the L2 ladder can never give — an L2 snapshot knows the level size, not where
*your* order sits in it.

Two estimators sit on top:

* ``fill_prob_survival`` — Kaplan–Meier survival of the *time-to-fill* where a
  cancel is a **competing risk** (treated as right-censoring for the fill
  hazard). Returns the survival curve and its complement, P(fill by τ).
* ``logistic_fill_model`` — a NumPy logistic regression (Newton–Raphson, no
  sklearn) mapping order-level features at placement (queue ahead, level
  depth, spread, imbalance, ...) to P(fill within the horizon), plus the
  calibration slope / Brier score that E5 reports. Calibration slope < 0.8 is
  the plan's "queue model is fiction" failure criterion.

Everything is stateless across calls except the tracker itself; the tracker
never looks at future events (it is fed strictly in tape order).
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..book_state import Side
from ..itch_parser import EventType, NormalizedEvent


@dataclass(slots=True)
class TrackedOrder:
    """One resting order and its queue context (all sizes in shares)."""

    order_id: int
    side: Side
    price: int
    size: int  # REMAINING displayed size (0 once the order has left the book)
    size0: int  # original size at placement
    ts_add: int
    idx_add: int
    ahead_at_add: int  # displayed size ahead in the FIFO when the order arrived
    ahead: int  # displayed size currently ahead (decreases as older orders leave)
    same_best_dist: int = 0  # ticks behind the same-side best at placement (<0 = new best)
    opp_dist: int = -1  # ticks to the opposite best at placement (-1 = no opposite side)
    filled: int = 0
    ts_first_fill: int | None = None
    idx_first_fill: int | None = None
    ts_end: int | None = None  # when the order left the book (fill-out / cancel)
    idx_end: int | None = None
    outcome: str = "resting"  # "resting" | "filled" (>=1 execution) | "cancelled" (never traded)


@dataclass(slots=True)
class _Level:
    """FIFO queue at one (side, price): order ids in arrival order."""

    queue: list[int] = field(default_factory=list)
    size: int = 0


class OrderLevelTracker:
    """Track queue position of every resting order on an ITCH stream.

    Feed events in tape order via ``on_event`` (or the typed ``on_add`` /
    ``on_execute`` / ``on_cancel_delete`` / ``on_replace`` / ``on_trade``).
    ``ahead_qty(oid)`` / ``behind_qty(oid)`` answer the queue question for any
    live order; ``completed`` accumulates the (filled | cancelled) records the
    survival and logistic models consume.
    """

    def __init__(self) -> None:
        self.orders: dict[int, TrackedOrder] = {}
        self.levels: dict[tuple[int, int], _Level] = {}
        # lazy-deletion heaps for O(log n) best-price lookups: bids keyed by -price
        self._heaps: dict[int, list[int]] = {int(Side.Bid): [], int(Side.Ask): []}
        self.completed: list[TrackedOrder] = []
        self.index = 0  # event counter (monotone tape position)
        self.stats = {"adds": 0, "executes": 0, "cancels": 0, "deletes": 0,
                      "replaces": 0, "trades": 0, "unknown_id": 0}

    # ------------------------------------------------------------------ api
    def on_event(self, ev: NormalizedEvent) -> None:
        k = ev.kind
        if k in (EventType.ADD, EventType.ADD_MPID):
            self.on_add(ev)
        elif k in (EventType.EXECUTE, EventType.EXECUTE_PX):
            self.on_execute(ev)
        elif k in (EventType.CANCEL, EventType.DELETE):
            self.on_cancel_delete(ev)
        elif k == EventType.REPLACE:
            self.on_replace(ev)
        elif k == EventType.TRADE:
            self.on_trade(ev)
        self.index += 1

    def on_add(self, ev: NormalizedEvent) -> TrackedOrder:
        self.stats["adds"] += 1
        side = Side(int(ev.side))
        price = int(ev.price_ticks)
        opp = Side.Ask if side == Side.Bid else Side.Bid
        same_best = self.best_price(side)
        opp_best = self.best_price(opp)
        sgn = 1 if side == Side.Bid else -1
        same_dist = sgn * (same_best - price) if same_best else 0
        opp_dist = sgn * (opp_best - price) if opp_best else -1
        key = (int(side), price)
        lvl = self.levels.get(key)
        if lvl is None:
            lvl = self.levels[key] = _Level()
            heapq.heappush(self._heaps[int(side)], -price if side == Side.Bid else price)
        o = TrackedOrder(
            order_id=int(ev.order_id), side=side, price=price,
            size=int(ev.size), size0=int(ev.size), ts_add=int(ev.ts_ns), idx_add=self.index,
            ahead_at_add=lvl.size, ahead=lvl.size,
            same_best_dist=int(same_dist), opp_dist=int(opp_dist),
        )
        lvl.queue.append(o.order_id)
        lvl.size += o.size
        self.orders[o.order_id] = o
        return o

    def on_execute(self, ev: NormalizedEvent) -> None:
        self.stats["executes"] += 1
        o = self.orders.get(int(ev.order_id))
        if o is None:
            self.stats["unknown_id"] += 1
            return
        qty = min(o.size, int(ev.size))
        if o.filled == 0 and qty > 0:
            o.ts_first_fill = int(ev.ts_ns)
            o.idx_first_fill = self.index
        o.filled += qty
        self._shrink(o, qty)
        if o.size <= 0:
            self._finish(o, "filled", ev.ts_ns)

    def on_cancel_delete(self, ev: NormalizedEvent) -> None:
        o = self.orders.get(int(ev.order_id))
        if ev.kind == EventType.DELETE:
            self.stats["deletes"] += 1
        else:
            self.stats["cancels"] += 1
        if o is None:
            self.stats["unknown_id"] += 1
            return
        qty = o.size if ev.kind == EventType.DELETE or ev.size <= 0 else min(o.size, int(ev.size))
        self._shrink(o, qty)
        if o.size <= 0:
            # a fully-cancelled order that never traded is the competing risk
            self._finish(o, "cancelled" if o.filled == 0 else "filled", ev.ts_ns)

    def on_replace(self, ev: NormalizedEvent) -> None:
        """ITCH ``U``: old order leaves its queue, new id joins the back of the new level."""
        self.stats["replaces"] += 1
        old = self.orders.get(int(ev.order_id))
        if old is None:
            self.stats["unknown_id"] += 1
            return
        side = old.side
        self._shrink(old, old.size)
        self._finish(old, "cancelled" if old.filled == 0 else "filled", ev.ts_ns)
        new_ev = NormalizedEvent(
            EventType.ADD, ev.ts_ns, ev.new_order_id or ev.order_id + 1, side,
            ev.price_ticks, ev.size, raw_type="U",
        )
        self.on_add(new_ev)

    def on_trade(self, ev: NormalizedEvent) -> None:
        """``P`` prints hit hidden liquidity — no displayed queue changes."""
        self.stats["trades"] += 1

    # ------------------------------------------------------------- queries
    def ahead_qty(self, order_id: int) -> int:
        o = self.orders.get(int(order_id))
        return 0 if o is None else int(o.ahead)

    def behind_qty(self, order_id: int) -> int:
        o = self.orders.get(int(order_id))
        if o is None:
            return 0
        lvl = self.levels.get((int(o.side), o.price))
        if lvl is None:
            return 0
        return int(lvl.size - o.ahead - o.size)

    def level_size(self, side: Side, price: int) -> int:
        lvl = self.levels.get((int(side), int(price)))
        return 0 if lvl is None else int(lvl.size)

    def best_price(self, side: Side) -> int:
        """Best displayed price on ``side`` (0 if empty). Amortized O(log n)."""
        heap = self._heaps[int(side)]
        while heap:
            price = -heap[0] if side == Side.Bid else heap[0]
            lvl = self.levels.get((int(side), price))
            if lvl is not None and lvl.size > 0:
                return price
            heapq.heappop(heap)
        return 0

    # ------------------------------------------------------------ internals
    def _shrink(self, o: TrackedOrder, qty: int) -> None:
        if qty <= 0:
            return
        lvl = self.levels.get((int(o.side), o.price))
        o.size -= qty
        if lvl is None:
            return
        lvl.size -= qty
        # everything behind ``o`` in the FIFO has ``qty`` less ahead of it
        try:
            pos = lvl.queue.index(o.order_id)
        except ValueError:
            return
        for oid in lvl.queue[pos + 1 :]:
            other = self.orders.get(oid)
            if other is not None:
                other.ahead = max(0, other.ahead - qty)
        if o.size <= 0:
            lvl.queue.pop(pos)
            if lvl.size <= 0 and not lvl.queue:
                self.levels.pop((int(o.side), o.price), None)

    def _finish(self, o: TrackedOrder, outcome: str, ts: int) -> None:
        o.outcome = outcome
        o.ts_end = int(ts)
        o.idx_end = self.index
        self.completed.append(o)
        self.orders.pop(o.order_id, None)


# ---------------------------------------------------------------------------
# E5 — passive fill probability
# ---------------------------------------------------------------------------
def fill_prob_survival(
    orders: Iterable[TrackedOrder],
    *,
    horizons: Sequence[float],
    clock: str = "events",
    cancel_is_censoring: bool = True,
) -> dict[str, Any]:
    """Kaplan–Meier P(fill by τ) with cancels as the competing risk.

    ``clock="events"`` measures time-to-outcome in tape events (``idx_end −
    idx_add``); ``"ns"`` uses the ITCH timestamp. With
    ``cancel_is_censoring=True`` a cancelled order is right-censored at its
    cancel time for the *fill* hazard — the standard competing-risk treatment
    that gives the fill probability *conditional on the order staying in the
    book*. Orders still resting at the end of the stream are censored at the
    stream end (their ``idx_end``/``ts_end`` is None → excluded; pass them
    through ``censor_open_orders`` first if you want them counted).

    Returns ``{"tau": [...], "survival": [...], "p_fill": [...], "n": int,
    "n_fill": int, "n_cancel": int, "median_fill_time": float | None}``.
    """
    times: list[float] = []
    fills: list[bool] = []
    n_cancel = 0
    for o in orders:
        if o.idx_end is None or o.ts_end is None:
            continue
        if o.outcome == "filled":
            # time-to-FIRST-fill: a partial fill is the event of interest
            i_end = o.idx_first_fill if o.idx_first_fill is not None else o.idx_end
            t_end = o.ts_first_fill if o.ts_first_fill is not None else o.ts_end
            t = float(i_end - o.idx_add) if clock == "events" else float(t_end - o.ts_add)
            times.append(t)
            fills.append(True)
            continue
        t = float(o.idx_end - o.idx_add) if clock == "events" else float(o.ts_end - o.ts_add)
        if o.outcome == "cancelled":
            n_cancel += 1
            if cancel_is_censoring:
                times.append(t)
                fills.append(False)
    t_arr = np.asarray(times, dtype=np.float64)
    f_arr = np.asarray(fills, dtype=bool)
    n = int(t_arr.size)
    out: dict[str, Any] = {"tau": [float(h) for h in horizons], "n": n,
                           "n_fill": int(f_arr.sum()), "n_cancel": n_cancel}
    if n == 0:
        out.update({"survival": [1.0] * len(horizons), "p_fill": [0.0] * len(horizons),
                    "median_fill_time": None})
        return out
    order = np.argsort(t_arr, kind="mergesort")
    t_sorted = t_arr[order]
    f_sorted = f_arr[order]
    uniq = np.unique(t_sorted[f_sorted]) if f_sorted.any() else np.empty(0)
    surv_steps: list[tuple[float, float]] = []
    s = 1.0
    for t in uniq:
        at_risk = int(np.sum(t_sorted >= t))
        d = int(np.sum((t_sorted == t) & f_sorted))
        if at_risk > 0:
            s *= 1.0 - d / at_risk
        surv_steps.append((float(t), s))
    surv = []
    for h in horizons:
        val = 1.0
        for t, sv in surv_steps:
            if t <= h:
                val = sv
            else:
                break
        surv.append(float(val))
    median = None
    for t, sv in surv_steps:
        if sv <= 0.5:
            median = t
            break
    out.update({"survival": surv, "p_fill": [1.0 - v for v in surv], "median_fill_time": median})
    return out


def censor_open_orders(tracker: OrderLevelTracker, ts_end: int) -> list[TrackedOrder]:
    """Snapshot copies of still-resting orders, censored at ``ts_end`` / current index."""
    out: list[TrackedOrder] = []
    for o in tracker.orders.values():
        c = TrackedOrder(**{f: getattr(o, f) for f in TrackedOrder.__slots__})
        c.ts_end = int(ts_end)
        c.idx_end = tracker.index
        c.outcome = "cancelled"  # censored: contributes to the at-risk set only
        out.append(c)
    return out


def order_features(o: TrackedOrder, *, level_size_at_add: int | None = None) -> dict[str, float]:
    """Placement-time features for the logistic fill model (all observable at add).

    ``same_best_dist`` / ``opp_dist`` are signed-log-compressed tick distances
    (an order deep in the book fills far less often than one at the touch).
    """
    ahead = float(o.ahead_at_add)
    lvl = float(level_size_at_add if level_size_at_add is not None else o.ahead_at_add + o.size0)
    d_same = float(o.same_best_dist)
    d_opp = float(o.opp_dist)
    return {
        "log_ahead": float(np.log1p(ahead)),
        "log_size": float(np.log1p(o.size0)),
        "queue_frac": ahead / max(1.0, lvl),
        "same_best_dist": float(np.sign(d_same) * np.log1p(abs(d_same))),
        "opp_dist": float(np.log1p(d_opp)) if d_opp >= 0 else 0.0,
        "no_opposite": 1.0 if d_opp < 0 else 0.0,
    }


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def logistic_fill_model(
    features: Mapping[str, Sequence[float]] | np.ndarray,
    filled: Sequence[bool] | Sequence[int],
    *,
    l2: float = 1e-3,
    max_iter: int = 50,
    tol: float = 1e-8,
    holdout: float = 0.3,
) -> dict[str, Any]:
    """Fit P(fill | features) by ridge-regularized logistic regression (Newton).

    Rows are time-ordered orders; the LAST ``holdout`` fraction is the
    out-of-sample block (walk-forward, never shuffled). Reports the
    out-of-sample **Brier score**, the **calibration slope** (logit of the
    prediction regressed on the outcome — 1.0 = perfectly calibrated, < 0.8 is
    the E5 failure criterion), and a decile calibration table.
    """
    if isinstance(features, np.ndarray):
        X = np.asarray(features, dtype=np.float64)
        names = [f"x{j}" for j in range(X.shape[1])]
    else:
        names = list(features.keys())
        X = np.column_stack([np.asarray(features[n], dtype=np.float64) for n in names])
    y = np.asarray(filled, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError("features rows must match filled length")
    n = y.size
    if n < 20:
        raise ValueError("logistic_fill_model needs at least 20 orders")
    n_train = max(10, int(n * (1.0 - holdout)))
    mu = X[:n_train].mean(axis=0)
    sd = X[:n_train].std(axis=0)
    sd[sd <= 1e-12] = 1.0
    Z = (X - mu) / sd
    D = np.column_stack([np.ones(n), Z])
    Dtr, ytr = D[:n_train], y[:n_train]
    w = np.zeros(D.shape[1])
    reg = np.full(D.shape[1], l2)
    reg[0] = 0.0
    for _ in range(max_iter):
        p = _sigmoid(Dtr @ w)
        g = Dtr.T @ (p - ytr) + reg * w
        W = p * (1.0 - p)
        H = (Dtr * W[:, None]).T @ Dtr + np.diag(reg)
        step = np.linalg.solve(H + 1e-9 * np.eye(H.shape[0]), g)
        w -= step
        if float(np.max(np.abs(step))) < tol:
            break
    p_all = _sigmoid(D @ w)
    p_te, y_te = p_all[n_train:], y[n_train:]
    brier = float(np.mean((p_te - y_te) ** 2)) if y_te.size else float("nan")
    base_rate = float(ytr.mean())
    brier_base = float(np.mean((base_rate - y_te) ** 2)) if y_te.size else float("nan")
    slope = _calibration_slope(p_te, y_te) if y_te.size >= 10 else float("nan")
    return {
        "coefs": dict(zip(["intercept", *names], w.tolist())),
        "n_train": int(n_train),
        "n_test": int(y_te.size),
        "base_rate": base_rate,
        "brier": brier,
        "brier_base_rate": brier_base,
        "brier_skill": (1.0 - brier / brier_base) if brier_base > 0 else float("nan"),
        "calibration_slope": slope,
        "calibration_table": calibration_table(p_te, y_te) if y_te.size else [],
        "predict": lambda Xnew: _sigmoid(
            np.column_stack([np.ones(len(Xnew)), (np.asarray(Xnew, dtype=np.float64) - mu) / sd]) @ w
        ),
    }


def _calibration_slope(p: np.ndarray, y: np.ndarray) -> float:
    """Slope of a logistic recalibration ``y ~ a + b·logit(p)`` (Cox 1958). 1.0 = calibrated."""
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    if np.std(lp) <= 1e-12:
        return float("nan")
    D = np.column_stack([np.ones(p.size), lp])
    w = np.zeros(2)
    for _ in range(50):
        q = _sigmoid(D @ w)
        g = D.T @ (q - y)
        H = (D * (q * (1 - q))[:, None]).T @ D + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        w -= step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return float(w[1])


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    """Decile calibration: mean predicted vs realized fill rate per prediction bin."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if p.size == 0:
        return []
    order = np.argsort(p, kind="mergesort")
    rows = []
    for chunk in np.array_split(order, min(bins, p.size)):
        if chunk.size == 0:
            continue
        rows.append({"n": int(chunk.size), "pred": float(p[chunk].mean()), "realized": float(y[chunk].mean())})
    return rows


def brier_score(p: Sequence[float], y: Sequence[float]) -> float:
    pa = np.asarray(p, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if pa.size != ya.size or pa.size == 0:
        return float("nan")
    return float(np.mean((pa - ya) ** 2))
