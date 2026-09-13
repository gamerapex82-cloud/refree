#!/usr/bin/env python3
"""Phase 4 — E1–E6 on a REAL NASDAQ ITCH tape (plan_2.md §4, work package §4).

Pipeline (per symbol file produced by ``scripts/fetch_itch.py``)::

    .itch bytes ─► iter_itch_events ─► ReplayEngine(StubBookAdapter)   [L2 book]
                                     └► OrderLevelTracker                [order-level queue]
         ─► features per event, computed inline from the live view (lob_imbalance,
            microprice_off, deep_imbalance, spread_bps, L2-approx OFI, and ORDER-LEVEL
            OFI from the tracker — the E3 upgrade the L2 approximation was flagged for)
         ─► forward_mid_move labels at h ∈ {1, 5, 10, 25} events
         ─► walk-forward split (60/20/20, gap = max h) ─► rank IC per split; block-bootstrap
            95% CI on the TEST split; OLS-combined predictor fit on train, scored on test;
            Diebold–Mariano pairwise tests                                         [E1–E4]
         ─► Kaplan–Meier fill probability + logistic fill model (Brier, calibration)  [E5]
         ─► post-fill drift / P(adverse), NW t-stats, pre-fill matched control       [E6]

Only the **regular session** (09:30–16:00 ET) is used for features/labels — the
pre-market book is thin and mostly stale orders. Events are sampled on the
*event* clock (every book-affecting message is a row), the plan's
regularization against volume-hour bias.

Honesty rules baked in: features are pure functions of the current view;
labels use only the ``t+h`` mid; the split is time-ordered with a gap; the
combined model is fit on train only. Nothing is tuned to a target.

Run (from repo root)::

    python python_quant/scripts/fetch_itch.py --day 12302019 --symbols AAPL,QQQ
    python python_quant/scripts/run_research.py --day 12302019 --symbols AAPL,QQQ
    python python_quant/scripts/run_research.py --symbols AAPL --max-events 100000   # quick look

Output: ``docs/results/real_tape_<day>_<SYMBOL>.json`` + ``docs/results/real_tape_<day>.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 safe

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "python_quant"))

from nexus_quant.book_port import StubBookAdapter
from nexus_quant.itch_parser import EventType, ItchParseStats, iter_itch_events
from nexus_quant.replay import ReplayEngine, check_integrity
from nexus_quant.research import diebold_mariano, make_split, rank_ic, run_experiment
from nexus_quant.research.adverse_selection import (
    PassiveFill,
    adverse_selection_report,
    fills_from_tracker,
)
from nexus_quant.research.dataset import Row
from nexus_quant.research.features import (
    deep_imbalance,
    lob_imbalance,
    microprice,
    mid,
    ofi,
    spread_bps,
)
from nexus_quant.research.queue_dynamics import (
    OrderLevelTracker,
    censor_open_orders,
    fill_prob_survival,
    logistic_fill_model,
    order_features,
)

NS_PER_H = 3.6e12
REGULAR_OPEN_NS = int(9.5 * NS_PER_H)  # 09:30 ET
REGULAR_CLOSE_NS = int(16.0 * NS_PER_H)  # 16:00 ET
OFI_WINDOW = 20  # events; the order-level OFI is a flow, not a single print

_SINGLE_VIEW_FNS = {
    "lob_imbalance": lob_imbalance,
    "microprice_off": lambda v: microprice(v) - mid(v),
    "deep_imbalance": lambda v: deep_imbalance(v, k=5),
    "spread_bps": spread_bps,
}
FEATURES = (*_SINGLE_VIEW_FNS, "ofi_l2", "ofi_order", "ofi_order_w20")


def order_level_ofi(ev, live_before) -> float:
    """Order-level OFI contribution of ONE event (Cont–Kukanov–Stoikov ``e_n``).

    +size for a bid add / ask cancel-or-execute, −size for an ask add / bid
    cancel-or-execute. ``live_before`` is the tracked order the event acts on
    (None for adds / prints). Replaces the L2-ladder approximation in E3.
    """
    k = ev.kind
    if k in (EventType.ADD, EventType.ADD_MPID):
        return float(ev.size) if int(ev.side) == 0 else -float(ev.size)
    if live_before is None:
        return 0.0
    if k in (EventType.EXECUTE, EventType.EXECUTE_PX, EventType.CANCEL):
        qty = float(min(live_before.size, ev.size))
    elif k == EventType.DELETE:
        qty = float(live_before.size)
    elif k == EventType.REPLACE:
        old, new = float(live_before.size), float(ev.size)
        return (-old + new) if int(live_before.side) == 0 else (old - new)
    else:
        return 0.0
    return -qty if int(live_before.side) == 0 else qty


def replay_symbol(path: Path, *, max_events: int | None = None, log=print) -> dict:
    """Parse + replay one symbol file, computing per-event features inline."""
    stats = ItchParseStats()
    t0 = time.time()
    events = list(iter_itch_events(path, stats=stats))
    t_parse = time.time() - t0
    book = StubBookAdapter()
    rep = ReplayEngine(events, book)
    tracker = OrderLevelTracker()
    feats: dict[str, list[float]] = {name: [] for name in FEATURES if name != "ofi_order_w20"}
    mids: list[float] = []
    ts: list[int] = []
    tape_idx: list[int] = []  # tape event index of each regular-session row
    issues: Counter = Counter()
    prev_view = None
    t0 = time.time()
    for i, ev in enumerate(events):
        live_before = None
        if ev.kind not in (EventType.ADD, EventType.ADD_MPID, EventType.TRADE):
            live_before = tracker.orders.get(int(ev.order_id))
        e_n = order_level_ofi(ev, live_before)
        rep.apply(ev)
        tracker.on_event(ev)
        if not (REGULAR_OPEN_NS <= ev.ts_ns < REGULAR_CLOSE_NS):
            continue
        v = book.snapshot()
        for iss in check_integrity(v):
            issues[iss.code] += 1
        for name, fn in _SINGLE_VIEW_FNS.items():
            feats[name].append(float(fn(v)))
        feats["ofi_l2"].append(float(ofi(prev_view, v)) if prev_view is not None else 0.0)
        feats["ofi_order"].append(e_n)
        mids.append(mid(v))
        ts.append(int(ev.ts_ns))
        tape_idx.append(i)
        prev_view = v
        if max_events is not None and len(mids) >= max_events:
            break
    t_replay = time.time() - t0
    arr = {k: np.asarray(v, dtype=np.float64) for k, v in feats.items()}
    csum = np.concatenate([[0.0], np.cumsum(arr["ofi_order"])])
    n = len(mids)
    idx = np.arange(n)
    arr["ofi_order_w20"] = csum[idx + 1] - csum[np.maximum(0, idx + 1 - OFI_WINDOW)]
    log(
        f"  {path.name}: {stats.messages:,} msgs → {stats.emitted:,} events, truncated={stats.truncated}, "
        f"parse {t_parse:.1f}s, replay+track+features {t_replay:.1f}s; regular-session rows {n:,}; "
        f"integrity issues {dict(issues) or 'none'}; tracker unknown ids {tracker.stats['unknown_id']}"
    )
    return {
        "events": events, "stats": stats, "features": arr, "mids": np.asarray(mids), "ts": np.asarray(ts),
        "tape_idx": np.asarray(tape_idx), "tracker": tracker, "issues": dict(issues), "n_regular": n,
        "t_parse": t_parse, "t_replay": t_replay,
    }


def _ols_fit_predict(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray) -> np.ndarray:
    """Univariate OLS fit on train, predictions on test (a scale for DM squared errors)."""
    X = np.column_stack([np.ones(x_tr.size), x_tr])
    beta, *_ = np.linalg.lstsq(X, y_tr, rcond=None)
    return beta[0] + beta[1] * x_te


def ic_study(rep: dict, *, horizons: tuple[int, ...], train: float, val: float, n_boot: int) -> dict:
    """E1–E4 on the regular-session rows."""
    feats, mids, ts = rep["features"], rep["mids"], rep["ts"]
    n = mids.size
    out: dict = {"rows": [], "combined": [], "dm": []}
    gap = max(horizons)
    for h in horizons:
        m = n - h
        if m < 500:
            continue
        ok = (mids[:m] > 0) & (mids[h : m + h] > 0)
        y = np.where(ok, mids[h : m + h] - mids[:m], 0.0)
        # walk-forward split on the plan's own primitive (time-ordered, gapped)
        rows = [Row(ts=int(ts[i]), features={}, label=float(y[i])) for i in range(1, m)]
        make_split(rows, train=train, val=val, gap=gap)
        tags = np.array([r.split for r in rows])
        y = y[1:m]
        nz = y != 0.0
        masks = {t: tags == t for t in ("train", "val", "test")}
        test_preds: dict[str, np.ndarray] = {}
        for name in FEATURES:
            x = feats[name][1:m]
            res = run_experiment(y[masks["test"]], x[masks["test"]], n_boot=n_boot)["overall"]
            # sign agreement where BOTH label and feature take a side (a zero feature
            # is "no view", not a wrong call — sparse features like per-event OFI are 0 most of the time)
            sel = masks["test"] & nz & (x != 0.0)
            hit = float(np.mean(np.sign(x[sel]) == np.sign(y[sel]))) if sel.sum() >= 10 else None
            out["rows"].append({
                "horizon_h": h, "feature": name,
                "n_test": int(masks["test"].sum()),
                "ic_train": rank_ic(y[masks["train"]], x[masks["train"]]),
                "ic_val": rank_ic(y[masks["val"]], x[masks["val"]]),
                "ic_test": res["rank_ic"], "ci95_lo": res["ci95"]["lo"], "ci95_hi": res["ci95"]["hi"],
                "hit_rate_nonzero_test": hit, "hit_coverage_test": float(sel.sum() / max(1, (masks["test"] & nz).sum())),
                "decile_spread_test": res["decile_spread"],
                "frac_label_nonzero_test": float(np.mean(nz[masks["test"]])),
            })
            test_preds[name] = _ols_fit_predict(x[masks["train"]], y[masks["train"]], x[masks["test"]])
        # E4: linear combination fit on TRAIN only, scored on TEST
        names = list(FEATURES)
        Xtr = np.column_stack([np.ones(masks["train"].sum()), *[_z(feats[k][1:m][masks["train"]], feats[k][1:m][masks["train"]]) for k in names]])
        Xte = np.column_stack([np.ones(masks["test"].sum()), *[_z(feats[k][1:m][masks["test"]], feats[k][1:m][masks["train"]]) for k in names]])
        beta, *_ = np.linalg.lstsq(Xtr, y[masks["train"]], rcond=None)
        pred_te = Xte @ beta
        res = run_experiment(y[masks["test"]], pred_te, n_boot=n_boot)["overall"]
        sel = masks["test"] & nz
        yt_nz = y[sel]
        pred_nz = pred_te[nz[masks["test"]]]
        out["combined"].append({
            "horizon_h": h, "n_test": int(masks["test"].sum()), "ic_test": res["rank_ic"],
            "ci95_lo": res["ci95"]["lo"], "ci95_hi": res["ci95"]["hi"],
            "hit_rate_nonzero_test": float(np.mean(np.sign(pred_nz) == np.sign(yt_nz))) if sel.sum() >= 10 else None,
            "coefs": dict(zip(["intercept", *names], beta.tolist())),
            "ic_train": rank_ic(y[masks["train"]], Xtr @ beta),
        })
        test_preds["combined"] = pred_te
        yt = y[masks["test"]]
        for a, b in (("microprice_off", "lob_imbalance"), ("ofi_order_w20", "ofi_l2"),
                     ("ofi_order_w20", "lob_imbalance"), ("combined", "lob_imbalance")):
            dm = diebold_mariano(yt - test_preds[a], yt - test_preds[b], h=h)
            out["dm"].append({"horizon_h": h, "a": a, "b": b, **dm})
    return out


def _z(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    s = ref.std()
    return (x - ref.mean()) / s if s > 1e-12 else np.zeros_like(x)


def fill_study(rep: dict, *, horizons_events: tuple[int, ...]) -> dict:
    """E5: KM survival + logistic fill model on the tracker's regular-session orders."""
    tracker: OrderLevelTracker = rep["tracker"]
    def in_session(o) -> bool:
        return REGULAR_OPEN_NS <= o.ts_add < REGULAR_CLOSE_NS

    completed = [o for o in tracker.completed if in_session(o)]
    last_ts = rep["events"][-1].ts_ns if rep["events"] else 0
    censored = [o for o in censor_open_orders(tracker, last_ts) if in_session(o)]
    km = fill_prob_survival(completed + censored, horizons=horizons_events)
    out: dict = {
        "n_orders_regular": len(completed) + len(censored),
        "outcomes": dict(Counter(o.outcome for o in completed)),
        "n_still_open_at_close": len(censored),
        "km": {"tau_events": list(km["tau"]), "p_fill": km["p_fill"], "n": km["n"],
               "n_fill": km["n_fill"], "n_cancel": km["n_cancel"],
               "median_fill_time_events": km["median_fill_time"]},
    }
    X: dict[str, list[float]] = {}
    y: list[bool] = []
    for o in completed:
        for k, v in order_features(o).items():
            X.setdefault(k, []).append(v)
        y.append(o.outcome == "filled")
    if len(y) >= 200 and 5 <= sum(y) <= len(y) - 5:
        m = logistic_fill_model(X, y, holdout=0.3)
        out["logistic"] = {k: m[k] for k in ("n_train", "n_test", "base_rate", "brier", "brier_base_rate",
                                             "brier_skill", "calibration_slope", "coefs", "calibration_table")}
    return out


def adverse_study(rep: dict, *, horizons: tuple[int, ...]) -> dict:
    """E6 on regular-session passive fills; row index = position in the regular-session series."""
    tracker: OrderLevelTracker = rep["tracker"]
    tape_idx: np.ndarray = rep["tape_idx"]
    n_rows = tape_idx.size
    if n_rows == 0:
        return {"n_fills": 0}
    pos = {int(t): j for j, t in enumerate(tape_idx)}
    ofi_w = rep["features"]["ofi_order_w20"]
    fills = []
    for f in fills_from_tracker(tracker.completed):
        j = pos.get(int(f.idx))
        if j is None:
            continue
        fills.append(PassiveFill(idx=j, side=f.side, price=f.price, size=f.size,
                                 ofi=float(ofi_w[j - 1]) if j > 0 else 0.0, queue_frac=f.queue_frac))
    return adverse_selection_report(fills, rep["mids"], horizons=horizons, min_group=30)


def fmt(x, scale=1.0, w=7, d=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-" * w
    return f"{x * scale:>{w}.{d}f}"


def render_md(day: str, per_symbol: dict[str, dict]) -> str:
    lines = [f"# Real-tape research results — NASDAQ ITCH 5.0, {day[:2]}/{day[2:4]}/{day[4:]}", ""]
    lines.append(
        "Source: `emi.nasdaq.com` public sample day (provenance in `python_quant/scripts/fetch_itch.py`). "
        "Regular session only (09:30–16:00), event clock, walk-forward 60/20/20 split with a gap of the largest "
        "horizon. Labels: forward mid move in ticks ($0.0001) at `h` events. IC = Spearman rank IC; the 95% CI is a "
        "block bootstrap on the test split; hit% is sign agreement on test rows where the label AND the feature are "
        "non-zero (an unchanged mid is neither hit nor miss; a zero feature is no view), with the share of non-zero-label "
        "rows the feature takes a view on in `cov`. `ofi_order` is the order-level OFI of the current event, "
        f"`ofi_order_w20` its rolling {OFI_WINDOW}-event sum; `ofi_l2` is the L2-ladder approximation it replaces."
    )
    lines.append("")
    for sym, res in per_symbol.items():
        tape = res["tape"]
        lines += [
            f"## {sym}", "",
            (f"{tape['messages']:,} messages → {tape['events']:,} events, {tape['truncated']} truncated, "
             f"{tape['regular_events']:,} regular-session rows; integrity: {tape['integrity_issues'] or 'clean'}; "
             f"tracker unknown ids: {tape['tracker_unknown_id']}; parse {tape['t_parse_s']:.1f}s, replay "
             f"{tape['t_replay_s']:.1f}s."), "",
            "### E1–E4 — signal → forward mid move", "",
            "| feature | h | IC train | IC val | **IC test** | 95% CI (test) | hit% (test) | cov | decile spread (ticks) | label≠0 |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in res["ic"]["rows"]:
            lines.append(
                f"| {r['feature']} | {r['horizon_h']} | {fmt(r['ic_train'])} | {fmt(r['ic_val'])} | "
                f"**{fmt(r['ic_test'])}** | [{fmt(r['ci95_lo'])}, {fmt(r['ci95_hi'])}] | "
                f"{fmt(r['hit_rate_nonzero_test'], 100, 6, 1)} | {fmt(r.get('hit_coverage_test'), 100, 5, 0)}% | "
                f"{fmt(r['decile_spread_test'], 1, 8, 1)} | {fmt(r['frac_label_nonzero_test'], 100, 5, 1)}% |"
            )
        lines += ["", "OLS combination of all features (fit on train only, z-scored on train), scored on test:", "",
                  "| h | IC train | **IC test** | 95% CI | hit% (nz) |", "|---|---|---|---|---|"]
        for c in res["ic"]["combined"]:
            lines.append(f"| {c['horizon_h']} | {fmt(c['ic_train'])} | **{fmt(c['ic_test'])}** | "
                         f"[{fmt(c['ci95_lo'])}, {fmt(c['ci95_hi'])}] | {fmt(c['hit_rate_nonzero_test'], 100, 6, 1)} |")
        lines += ["", ("Diebold–Mariano on test-split squared errors of train-fit linear predictors "
                       "(negative `mean_diff` = A has lower loss; HAC lag = h):"), "",
                  "| h | A | B | DM | p | mean(loss A − loss B) |", "|---|---|---|---|---|---|"]
        for d in res["ic"]["dm"]:
            lines.append(f"| {d['horizon_h']} | {d['a']} | {d['b']} | {fmt(d['dm'], 1, 7, 2)} | {fmt(d['p_value'], 1, 6, 3)} | "
                         f"{fmt(d['mean_diff'], 1, 9, 4)} |")
        lines.append("")
        e5 = res["fill"]
        lines += [
            "### E5 — passive fill probability (order level)", "",
            (f"{e5['n_orders_regular']:,} regular-session orders; outcomes {e5['outcomes']}; "
             f"{e5['n_still_open_at_close']:,} still open at close (censored)."), "",
            "| τ (events after placement) | " + " | ".join(str(int(t)) for t in e5["km"]["tau_events"]) + " |",
            "|---|" + "---|" * len(e5["km"]["tau_events"]),
            "| P(first fill by τ), Kaplan–Meier, cancel = competing risk | "
            + " | ".join(f"{p:.4f}" for p in e5["km"]["p_fill"]) + " |", "",
        ]
        if "logistic" in e5:
            lg = e5["logistic"]
            lines += [
                (f"Logistic fill model (placement features → P(fill), time-ordered 70/30 holdout): base rate "
                 f"{lg['base_rate']:.4f}, Brier {lg['brier']:.5f} vs base-rate Brier {lg['brier_base_rate']:.5f} "
                 f"(skill {lg['brier_skill']:+.3f}), **calibration slope {lg['calibration_slope']:.2f}** "
                 "(plan §4 E5 failure criterion: slope < 0.8 → the queue model is fiction)."), "",
                "| decile of predicted P(fill) | n | mean predicted | realized fill rate |", "|---|---|---|---|",
            ]
            for i, row in enumerate(lg["calibration_table"]):
                lines.append(f"| {i + 1} | {row['n']} | {row['pred']:.4f} | {row['realized']:.4f} |")
            lines += ["", "Standardized coefficients: " + ", ".join(f"`{k}` {v:+.2f}" for k, v in lg["coefs"].items()), ""]
        e6 = res["adverse"]
        lines += [
            "### E6 — adverse selection after passive fills", "",
            (f"{e6.get('n_fills', 0):,} passive fills (first execution of a tracked resting order). Drift is "
             "`s·(mid[t+h] − mid[t])` in ticks, s = +1 for a filled bid (bought), −1 for a filled ask; negative = "
             "adverse. `post − pre` nets out the drift over the h events *before* the fill (matched control on the "
             "same tape)."), "",
            "| h | n | mean post drift | NW t | P(adverse) | mean pre drift | post − pre | NW t |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for h, blk in e6.get("horizons", {}).items():
            o, p, x = blk["overall"], blk["pre_fill"], blk["post_minus_pre"]
            lines.append(f"| {h} | {o['n']} | {fmt(o['mean_drift'], 1, 8, 2)} | {fmt(o['t_nw'], 1, 6, 2)} | "
                         f"{fmt(o['p_adverse'], 1, 6, 3)} | {fmt(p['mean_drift'], 1, 8, 2)} | "
                         f"{fmt(x['mean_drift'], 1, 8, 2)} | {fmt(x['t_nw'], 1, 6, 2)} |")
        lines.append("")
        hz = e6.get("horizons", {})
        blk5 = hz.get(5) or hz.get("5") or (next(iter(hz.values())) if hz else None)
        if blk5 and blk5["groups"]:
            lines += ["Conditioned on side / rolling order-flow imbalance sign / queue position at placement (h = 5):", "",
                      "| group | n | mean drift | NW t | P(adverse) |", "|---|---|---|---|---|"]
            for g, st in blk5["groups"].items():
                lines.append(f"| {g} | {st['n']} | {fmt(st['mean_drift'], 1, 8, 2)} | {fmt(st['t_nw'], 1, 6, 2)} | "
                             f"{fmt(st['p_adverse'], 1, 6, 3)} |")
            lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", default="12302019")
    ap.add_argument("--symbols", default="AAPL,QQQ")
    ap.add_argument("--data", type=Path, default=Path("data/itch"))
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 25])
    ap.add_argument("--train", type=float, default=0.6)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--max-events", type=int, default=None, help="cap regular-session rows (quick look)")
    ap.add_argument("--out-dir", type=Path, default=_ROOT / "docs" / "results")
    args = ap.parse_args(argv)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    per_symbol: dict[str, dict] = {}
    t_all = time.time()
    for sym in symbols:
        path = args.data / args.day / f"{sym}.itch"
        if not path.exists():
            print(f"missing {path} — run scripts/fetch_itch.py --day {args.day} --symbols {sym}")
            continue
        print(f"== {sym} ==")
        rep = replay_symbol(path, max_events=args.max_events)
        t0 = time.time()
        ic = ic_study(rep, horizons=tuple(args.horizons), train=args.train, val=args.val, n_boot=args.n_boot)
        print(f"  E1–E4 done in {time.time() - t0:.1f}s")
        for r in ic["rows"]:
            print(f"    h={r['horizon_h']:<3}{r['feature']:<16} IC train/val/test {fmt(r['ic_train'])}/{fmt(r['ic_val'])}/"
                  f"{fmt(r['ic_test'])} CI [{fmt(r['ci95_lo'])},{fmt(r['ci95_hi'])}] hit%(nz) {fmt(r['hit_rate_nonzero_test'], 100, 6, 1)}")
        for c in ic["combined"]:
            print(f"    h={c['horizon_h']:<3}{'combined(OLS)':<16} IC train/test {fmt(c['ic_train'])}/{fmt(c['ic_test'])} "
                  f"CI [{fmt(c['ci95_lo'])},{fmt(c['ci95_hi'])}]")
        t0 = time.time()
        fill = fill_study(rep, horizons_events=(10, 50, 100, 500, 1000, 5000))
        print(f"  E5 done in {time.time() - t0:.1f}s: KM p_fill {[f'{p:.3f}' for p in fill['km']['p_fill']]}"
              + (f", calibration slope {fill['logistic']['calibration_slope']:.2f}, Brier skill "
                 f"{fill['logistic']['brier_skill']:+.3f}" if "logistic" in fill else ""))
        t0 = time.time()
        adverse = adverse_study(rep, horizons=(1, 5, 25))
        print(f"  E6 done in {time.time() - t0:.1f}s: {adverse.get('n_fills', 0)} passive fills")
        for h, blk in adverse.get("horizons", {}).items():
            o, x = blk["overall"], blk["post_minus_pre"]
            print(f"    h={h:<3} post drift {o['mean_drift']:+.2f} (t {o['t_nw']:+.2f}, P(adv) {o['p_adverse']:.3f}); "
                  f"post−pre {x['mean_drift']:+.2f} (t {x['t_nw']:+.2f})")
        st = rep["stats"]
        per_symbol[sym] = {
            "tape": {"messages": st.messages, "events": st.emitted, "truncated": st.truncated,
                     "skipped_type": st.skipped_type, "regular_events": rep["n_regular"],
                     "integrity_issues": rep["issues"], "tracker_unknown_id": rep["tracker"].stats["unknown_id"],
                     "t_parse_s": rep["t_parse"], "t_replay_s": rep["t_replay"]},
            "ic": ic, "fill": fill, "adverse": adverse,
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / f"real_tape_{args.day}_{sym}.json").write_text(
            json.dumps(per_symbol[sym], indent=1, default=float) + "\n"
        )
        del rep
    if per_symbol:
        md = args.out_dir / f"real_tape_{args.day}.md"
        md.write_text(render_md(args.day, per_symbol))
        print(f"\nwrote {md} ({time.time() - t_all:.0f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
