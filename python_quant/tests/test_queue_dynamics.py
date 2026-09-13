"""Phase 3 / E5 — order-level queue tracker + fill-probability models.

The queue is KNOWN by construction in every test (hand-built ITCH streams), so
the assertions are exact: ahead/behind sizes, outcomes, Kaplan–Meier values,
and the calibration of the logistic fill model on a stream whose true fill
probability is a known logistic function of the features.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexus_quant.book_state import Side
from nexus_quant.itch_parser import (
    EventType,
    NormalizedEvent,
    encode_event,
    iter_itch_events,
)
from nexus_quant.research.queue_dynamics import (
    OrderLevelTracker,
    brier_score,
    calibration_table,
    censor_open_orders,
    fill_prob_survival,
    logistic_fill_model,
    order_features,
)


def _ev(kind, oid, *, side=Side.Bid, px=0, sz=0, ts=0, new_id=0) -> NormalizedEvent:
    return NormalizedEvent(kind, ts, oid, side, px, sz, new_order_id=new_id, raw_type="")


def _feed(tr: OrderLevelTracker, events) -> None:
    for e in events:
        tr.on_event(e)


# --------------------------------------------------------------------------- tracker
def test_fifo_queue_position_is_exact():
    tr = OrderLevelTracker()
    _feed(tr, [
        _ev(EventType.ADD, 1, px=100, sz=100, ts=1),
        _ev(EventType.ADD, 2, px=100, sz=50, ts=2),
        _ev(EventType.ADD, 3, px=100, sz=30, ts=3),
    ])
    assert (tr.ahead_qty(1), tr.ahead_qty(2), tr.ahead_qty(3)) == (0, 100, 150)
    assert (tr.behind_qty(1), tr.behind_qty(2), tr.behind_qty(3)) == (80, 30, 0)
    assert tr.level_size(Side.Bid, 100) == 180
    assert tr.best_price(Side.Bid) == 100 and tr.best_price(Side.Ask) == 0

    # partial execution at the front shrinks everyone's "ahead" behind it
    _feed(tr, [_ev(EventType.EXECUTE, 1, sz=40, ts=4)])
    assert (tr.ahead_qty(2), tr.ahead_qty(3)) == (60, 110)
    assert tr.orders[1].filled == 40 and tr.orders[1].size == 60
    assert tr.orders[1].idx_first_fill == 3  # 4th event, zero-based

    # deleting the middle order removes its remaining 50 from those behind
    _feed(tr, [_ev(EventType.DELETE, 2, ts=5)])
    assert tr.ahead_qty(3) == 60
    assert 2 not in tr.orders and tr.completed[-1].outcome == "cancelled"

    # finishing order 1 puts order 3 at the front
    _feed(tr, [_ev(EventType.EXECUTE, 1, sz=60, ts=6)])
    assert tr.ahead_qty(3) == 0 and tr.behind_qty(3) == 0
    done = {o.order_id: o for o in tr.completed}
    assert done[1].outcome == "filled" and done[1].filled == 100
    assert tr.level_size(Side.Bid, 100) == 30


def test_partial_cancel_replace_and_unknown_ids():
    tr = OrderLevelTracker()
    _feed(tr, [
        _ev(EventType.ADD, 7, side=Side.Ask, px=105, sz=80, ts=1),
        _ev(EventType.ADD, 8, side=Side.Ask, px=105, sz=20, ts=2),
        _ev(EventType.CANCEL, 7, sz=30, ts=3),  # partial: 80 -> 50
    ])
    assert tr.orders[7].size == 50 and tr.ahead_qty(8) == 50
    # replace moves 7 to a new price with a new id, at the back of that level
    _feed(tr, [_ev(EventType.ADD, 9, side=Side.Ask, px=106, sz=10, ts=4)])
    _feed(tr, [_ev(EventType.REPLACE, 7, px=106, sz=25, ts=5, new_id=70)])
    assert 7 not in tr.orders and 70 in tr.orders
    assert tr.orders[70].price == 106 and tr.ahead_qty(70) == 10 and tr.ahead_qty(8) == 0
    assert tr.completed[-1].order_id == 7 and tr.completed[-1].outcome == "cancelled"
    assert tr.level_size(Side.Ask, 105) == 20 and tr.level_size(Side.Ask, 106) == 35
    # events for ids we never saw (e.g. a sliced tape) are counted, never raise
    _feed(tr, [_ev(EventType.EXECUTE, 999, sz=5, ts=6), _ev(EventType.DELETE, 998, ts=7)])
    assert tr.stats["unknown_id"] == 2
    # trades (P) never touch displayed queues
    _feed(tr, [_ev(EventType.TRADE, 0, side=Side.Bid, px=105, sz=5, ts=8)])
    assert tr.level_size(Side.Ask, 105) == 20 and tr.stats["trades"] == 1


def test_touch_distance_features_at_placement():
    tr = OrderLevelTracker()
    _feed(tr, [
        _ev(EventType.ADD, 1, side=Side.Bid, px=100, sz=10, ts=1),
        _ev(EventType.ADD, 2, side=Side.Ask, px=104, sz=10, ts=2),
        _ev(EventType.ADD, 3, side=Side.Bid, px=98, sz=10, ts=3),   # 2 ticks behind best bid
        _ev(EventType.ADD, 4, side=Side.Bid, px=101, sz=10, ts=4),  # new best bid
        _ev(EventType.ADD, 5, side=Side.Ask, px=103, sz=10, ts=5),  # new best ask
    ])
    o3, o4, o5 = tr.orders[3], tr.orders[4], tr.orders[5]
    assert o3.same_best_dist == 2 and o3.opp_dist == 6
    assert o4.same_best_dist == -1 and o4.opp_dist == 3
    assert o5.same_best_dist == -1 and o5.opp_dist == 2
    assert tr.best_price(Side.Bid) == 101 and tr.best_price(Side.Ask) == 103
    f = order_features(o3)
    assert set(f) == {"log_ahead", "log_size", "queue_frac", "same_best_dist", "opp_dist", "no_opposite"}
    assert f["queue_frac"] == 0.0 and f["no_opposite"] == 0.0
    # the very first order had no opposite side at all
    assert tr.orders[1].opp_dist == -1 and order_features(tr.orders[1])["no_opposite"] == 1.0


def test_tracker_consumes_encoded_itch_bytes():
    """End-to-end: encode → parse → track, the path the real tape takes."""
    evs = [
        _ev(EventType.ADD, 11, px=15000, sz=100, ts=1),
        _ev(EventType.ADD_MPID, 12, px=15000, sz=40, ts=2),
        _ev(EventType.EXECUTE, 11, sz=100, ts=3),
        _ev(EventType.DELETE, 12, ts=4),
    ]
    blob = b"".join(encode_event(e) for e in evs)
    tr = OrderLevelTracker()
    for e in iter_itch_events(blob):
        tr.on_event(e)
    outcomes = {o.order_id: o.outcome for o in tr.completed}
    assert outcomes == {11: "filled", 12: "cancelled"}
    assert tr.stats["adds"] == 2 and tr.stats["executes"] == 1 and tr.stats["deletes"] == 1


# --------------------------------------------------------------------------- Kaplan–Meier
def test_kaplan_meier_matches_hand_computation():
    """5 orders with known event-clock outcomes; KM computed by hand below."""
    tr = OrderLevelTracker()
    tr.on_event(_ev(EventType.ADD, 1, px=100, sz=10, ts=1))    # idx 0
    tr.on_event(_ev(EventType.ADD, 2, px=100, sz=10, ts=2))    # idx 1
    tr.on_event(_ev(EventType.EXECUTE, 1, sz=10, ts=3))        # idx 2 -> order1 fill t=2
    tr.on_event(_ev(EventType.ADD, 3, px=100, sz=10, ts=4))    # idx 3
    tr.on_event(_ev(EventType.DELETE, 2, ts=5))                # idx 4 -> order2 cancel t=3
    tr.on_event(_ev(EventType.ADD, 4, px=100, sz=10, ts=6))    # idx 5
    tr.on_event(_ev(EventType.ADD, 5, px=100, sz=10, ts=7))    # idx 6
    tr.on_event(_ev(EventType.EXECUTE, 3, sz=10, ts=8))        # idx 7 -> order3 fill t=4
    tr.on_event(_ev(EventType.DELETE, 4, ts=9))                # idx 8 -> order4 cancel t=3
    tr.on_event(_ev(EventType.EXECUTE, 5, sz=10, ts=10))       # idx 9 -> order5 fill t=3
    times = {o.order_id: (o.idx_end - o.idx_add, o.outcome) for o in tr.completed}
    assert times == {1: (2, "filled"), 2: (3, "cancelled"), 3: (4, "filled"),
                     4: (3, "cancelled"), 5: (3, "filled")}
    # sorted: 2F, 3C, 3C, 3F, 4F  -> S(2)=4/5=0.8 ; at t=3 at-risk=4, d=1 -> 0.8*3/4=0.6 ;
    # at t=4 at-risk=1, d=1 -> 0
    km = fill_prob_survival(tr.completed, horizons=[1, 2, 3, 4, 10])
    assert km["n"] == 5 and km["n_fill"] == 3 and km["n_cancel"] == 2
    np.testing.assert_allclose(km["survival"], [1.0, 0.8, 0.6, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(km["p_fill"], [0.0, 0.2, 0.4, 1.0, 1.0], atol=1e-12)
    assert km["median_fill_time"] == 4


def test_kaplan_meier_uses_first_fill_time_and_censors_open_orders():
    tr = OrderLevelTracker()
    tr.on_event(_ev(EventType.ADD, 1, px=100, sz=10, ts=1))  # idx 0
    tr.on_event(_ev(EventType.ADD, 2, px=100, sz=10, ts=2))  # idx 1 (stays open)
    tr.on_event(_ev(EventType.EXECUTE, 1, sz=4, ts=3))       # idx 2 -> first fill at t=2
    tr.on_event(_ev(EventType.ADD, 3, px=100, sz=10, ts=4))  # idx 3
    tr.on_event(_ev(EventType.EXECUTE, 1, sz=6, ts=5))       # idx 4 -> fully filled at t=4
    open_ = censor_open_orders(tr, ts_end=99)
    assert {o.order_id for o in open_} == {2, 3} and all(o.outcome == "cancelled" for o in open_)
    km = fill_prob_survival(list(tr.completed) + open_, horizons=[1, 2, 3, 4])
    # order1 event of interest at t=2 (first fill), orders 2/3 censored at t=4 and t=2
    assert km["n"] == 3 and km["n_fill"] == 1
    # at t=2: at risk = 3 (t>=2 for all), d=1 -> S=2/3
    np.testing.assert_allclose(km["survival"], [1.0, 2 / 3, 2 / 3, 2 / 3], atol=1e-12)
    # hard exclusion of cancels (not treated as censoring) changes the denominator
    km2 = fill_prob_survival(tr.completed, horizons=[2], cancel_is_censoring=False)
    assert km2["n"] == 1 and km2["p_fill"] == [1.0]


def test_kaplan_meier_empty_input_is_safe():
    km = fill_prob_survival([], horizons=[1, 5])
    assert km["n"] == 0 and km["p_fill"] == [0.0, 0.0] and km["median_fill_time"] is None


# --------------------------------------------------------------------------- logistic
def test_logistic_fill_model_recovers_a_known_logistic_queue():
    """Truth: P(fill) = σ(1.0 − 0.9·log_ahead). Calibration slope must be ≈ 1 out of sample."""
    rng = np.random.default_rng(0x51ED)
    n = 6000
    log_ahead = rng.uniform(0.0, 6.0, size=n)
    log_size = rng.uniform(1.0, 5.0, size=n)  # pure noise feature
    p_true = 1.0 / (1.0 + np.exp(-(1.0 - 0.9 * log_ahead)))
    y = rng.random(n) < p_true
    m = logistic_fill_model({"log_ahead": log_ahead, "log_size": log_size}, y, holdout=0.3)
    assert m["n_train"] == 4200 and m["n_test"] == 1800
    assert 0.85 <= m["calibration_slope"] <= 1.15, m["calibration_slope"]
    assert m["brier"] < m["brier_base_rate"] and m["brier_skill"] > 0.15
    # standardized coefficient sign: more queue ahead -> lower fill probability
    assert m["coefs"]["log_ahead"] < 0 and abs(m["coefs"]["log_size"]) < 0.1
    table = m["calibration_table"]
    assert len(table) == 10 and sum(r["n"] for r in table) == 1800
    # monotone decile table: realized rate tracks prediction within noise
    pred = np.array([r["pred"] for r in table])
    real = np.array([r["realized"] for r in table])
    assert np.all(np.diff(pred) >= 0) and np.max(np.abs(pred - real)) < 0.12
    # the fitted predictor is usable on new rows
    p_new = m["predict"](np.column_stack([[0.0, 6.0], [3.0, 3.0]]))
    assert p_new[0] > p_new[1]


def test_logistic_fill_model_walk_forward_not_shuffled():
    """A regime shift in the last 30% must show up out of sample — proof the
    holdout is the time-ordered tail, not a random subsample."""
    rng = np.random.default_rng(7)
    n = 4000
    x = rng.uniform(0, 4, size=n)
    p = 1.0 / (1.0 + np.exp(-(1.5 - 1.0 * x)))
    y = rng.random(n) < p
    y[int(n * 0.7):] = ~y[int(n * 0.7):]  # flip the tail
    m = logistic_fill_model({"x": x}, y)
    assert m["brier"] > m["brier_base_rate"]  # the fitted model is WORSE than base rate on the flipped tail


def test_logistic_rejects_bad_shapes_and_tiny_samples():
    import pytest

    with pytest.raises(ValueError):
        logistic_fill_model({"a": [0.0] * 10}, [True] * 10)
    with pytest.raises(ValueError):
        logistic_fill_model({"a": [0.0] * 30}, [True] * 29)


def test_brier_and_calibration_helpers():
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0
    assert brier_score([0.5, 0.5], [1, 0]) == 0.25
    assert np.isnan(brier_score([], []))
    rows = calibration_table(np.array([0.1, 0.9, 0.2, 0.8]), np.array([0, 1, 0, 1]), bins=2)
    assert [r["n"] for r in rows] == [2, 2] and rows[0]["realized"] == 0.0 and rows[1]["realized"] == 1.0
