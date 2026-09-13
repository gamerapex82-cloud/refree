"""Phase 3 / E6 — adverse selection after passive fills.

Sign conventions are pinned on hand-built mid paths (drift < 0 == adverse),
the Newey–West t-stat is checked against a hand computation, and the report
is wired end-to-end from an ``OrderLevelTracker`` on a synthetic stream whose
post-fill drift is adverse by construction.
"""
from __future__ import annotations

import sys
from math import sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexus_quant.book_state import Side
from nexus_quant.itch_parser import EventType, NormalizedEvent
from nexus_quant.research.adverse_selection import (
    PassiveFill,
    adverse_selection_report,
    fills_from_tracker,
    nw_tstat,
    p_adverse,
    post_fill_drift,
    pre_fill_drift,
)
from nexus_quant.research.queue_dynamics import OrderLevelTracker


def test_post_fill_drift_sign_convention():
    mids = [100.0, 101.0, 102.0, 103.0, 104.0]  # rising tape
    fills = [PassiveFill(idx=1, side=Side.Bid), PassiveFill(idx=1, side=Side.Ask)]
    d = post_fill_drift(fills, mids, h=2)
    # bought on the bid, price rose -> favorable (+2); sold on the ask, price rose -> adverse (-2)
    np.testing.assert_allclose(d, [2.0, -2.0])
    assert p_adverse(d) == 0.5
    # no future state -> NaN, dropped by the aggregators
    assert np.isnan(post_fill_drift([PassiveFill(idx=4, side=Side.Bid)], mids, h=1))[0]
    assert np.isnan(post_fill_drift([PassiveFill(idx=3, side=Side.Bid)], mids, h=5))[0]


def test_pre_fill_control_window_ends_before_the_fill():
    mids = [100.0, 100.0, 100.0, 105.0, 90.0, 90.0]
    f = [PassiveFill(idx=4, side=Side.Bid)]
    # pre window for h=2 is mids[1] -> mids[3] = +5 (strictly before the fill event at 4)
    np.testing.assert_allclose(pre_fill_drift(f, mids, h=2), [5.0])
    # post window is mids[4] -> mids[6]: out of range -> NaN
    assert np.isnan(post_fill_drift(f, mids, h=2))[0]
    np.testing.assert_allclose(post_fill_drift(f, mids, h=1), [0.0])
    assert np.isnan(pre_fill_drift([PassiveFill(idx=1, side=Side.Bid)], mids, h=2))[0]


def test_p_adverse_excludes_ties_and_nans():
    assert p_adverse([-1.0, -1.0, 1.0, 0.0, np.nan]) == 2 / 3
    assert np.isnan(p_adverse([0.0, np.nan]))


def test_newey_west_matches_hand_computation():
    x = np.array([1.0, 3.0, 2.0, 4.0, 3.0, 5.0])
    n = x.size
    d = x - x.mean()
    g0 = float(d @ d) / n
    g1 = float(d[1:] @ d[:-1]) / n
    var = g0 + 2 * (1 - 1 / 2) * g1  # Bartlett weight for lag=1
    r = nw_tstat(x, lag=1)
    assert r["n"] == 6 and r["mean"] == x.mean()
    np.testing.assert_allclose(r["se"], sqrt(var / n))
    np.testing.assert_allclose(r["t"], x.mean() / sqrt(var / n))
    assert 0.0 <= r["p"] <= 1.0
    # lag 0 collapses to the plain (population) standard error
    r0 = nw_tstat(x, lag=0)
    np.testing.assert_allclose(r0["se"], x.std() / sqrt(n))
    # too few points -> NaNs, never raises
    assert np.isnan(nw_tstat([1.0, 2.0], lag=1)["t"])


def test_newey_west_widens_se_under_positive_autocorrelation():
    rng = np.random.default_rng(3)
    e = rng.standard_normal(2000)
    ar = np.empty_like(e)
    ar[0] = e[0]
    for i in range(1, e.size):
        ar[i] = 0.8 * ar[i - 1] + e[i]
    assert nw_tstat(ar, lag=10)["se"] > 2.0 * nw_tstat(ar, lag=0)["se"]


def _adverse_stream(n_orders: int = 60, seed: int = 11):
    """Bid fills are followed by a mid drop, ask fills by a mid rise (adverse by construction).

    Returns (events, mids) where mids[i] is the mid AFTER event i.
    """
    rng = np.random.default_rng(seed)
    events: list[NormalizedEvent] = []
    mids: list[float] = []
    mid = 10_000.0
    oid = 1
    ts = 1
    for _ in range(n_orders):
        side = Side.Bid if rng.random() < 0.5 else Side.Ask
        px = int(mid) - 1 if side == Side.Bid else int(mid) + 1
        events.append(NormalizedEvent(EventType.ADD, ts, oid, side, px, 100))
        mids.append(mid)
        ts += 1
        # a few neutral events in between (tape noise with zero drift)
        for _ in range(int(rng.integers(0, 3))):
            events.append(NormalizedEvent(EventType.TRADE, ts, 0, Side.Bid, int(mid), 1))
            mids.append(mid)
            ts += 1
        events.append(NormalizedEvent(EventType.EXECUTE, ts, oid, Side.NONE, 0, 100))
        mids.append(mid)  # the fill event itself leaves the mid unchanged
        ts += 1
        # adverse move of 2..4 ticks after the fill, then a flat stretch
        move = float(rng.integers(2, 5))
        mid += -move if side == Side.Bid else move
        for _ in range(3):
            events.append(NormalizedEvent(EventType.TRADE, ts, 0, Side.Ask, int(mid), 1))
            mids.append(mid)
            ts += 1
        oid += 1
    return events, mids


def test_report_detects_constructed_adverse_selection():
    events, mids = _adverse_stream()
    tr = OrderLevelTracker()
    for e in events:
        tr.on_event(e)
    ofis = np.zeros(len(events))
    fills = fills_from_tracker(tr.completed, ofi_at=ofis)
    assert len(fills) == 60 and all(f.queue_frac == 0.0 for f in fills)
    rep = adverse_selection_report(fills, mids, horizons=(1, 3), min_group=10)
    assert rep["n_fills"] == 60
    for h in (1, 3):
        blk = rep["horizons"][h]
        ov = blk["overall"]
        assert ov["n"] == 60
        assert -4.0 <= ov["mean_drift"] <= -2.0  # every fill is adverse by 2..4 ticks
        assert ov["p_adverse"] == 1.0 and ov["t_nw"] < -5 and ov["p_value"] < 1e-6
        # the pre-fill window is (mostly) flat -> excess ≈ post
        assert blk["post_minus_pre"]["mean_drift"] < -2.0
        groups = blk["groups"]
        assert "side=bid" in groups and "side=ask" in groups and "queue=front" in groups
        assert groups["side=bid"]["p_adverse"] == 1.0 and groups["side=ask"]["p_adverse"] == 1.0
        assert "ofi=0" in groups and "ofi>0" not in groups  # only groups with >= min_group fills


def test_report_is_null_on_a_flat_tape():
    fills = [PassiveFill(idx=i, side=Side.Bid if i % 2 else Side.Ask) for i in range(5, 200, 3)]
    mids = [500.0] * 400
    rep = adverse_selection_report(fills, mids, horizons=(1, 5))
    for blk in rep["horizons"].values():
        assert blk["overall"]["mean_drift"] == 0.0
        assert np.isnan(blk["overall"]["p_adverse"])  # all ties -> undefined, not 0 or 1


def test_fills_from_tracker_uses_first_fill_index_and_queue_fraction():
    tr = OrderLevelTracker()
    tr.on_event(NormalizedEvent(EventType.ADD, 1, 1, Side.Ask, 105, 100))  # idx 0, front
    tr.on_event(NormalizedEvent(EventType.ADD, 2, 2, Side.Ask, 105, 100))  # idx 1, 100 ahead
    tr.on_event(NormalizedEvent(EventType.EXECUTE, 3, 1, Side.NONE, 0, 40))  # idx 2 first fill of 1
    tr.on_event(NormalizedEvent(EventType.EXECUTE, 4, 1, Side.NONE, 0, 60))  # idx 3 fill-out of 1
    tr.on_event(NormalizedEvent(EventType.DELETE, 5, 2, Side.NONE, 0, 0))    # idx 4 cancel 2
    fills = fills_from_tracker(tr.completed, ofi_at=[0.0, 0.0, -2.0, 1.0, 0.0])
    assert len(fills) == 1
    f = fills[0]
    assert f.idx == 2 and f.side == Side.Ask and f.size == 100 and f.ofi == -2.0 and f.queue_frac == 0.0
