#!/usr/bin/env python3
"""Synthetic E1–E4 IC harness (plan_2.md §4) — establishes the research spine.

Builds a seeded L2 book around a mid random-walk, computes features per event
(lob_imbalance, microprice-offset, OFI, deep_imbalance), labels forward mid
moves at h ∈ {1, 5, 10}, and reports rank IC / ICIR / hit rate / decile spread
with block-bootstrap CIs on a walk-forward split.

READ THE RESULT HONESTLY: this synthetic flow is a random walk — the honest
expected outcome is IC ≈ 0 for every feature. That is a *harness* result, not a
*predictive* result; claims come from the real NASDAQ tape in Phase 4.

Run (from repo root):
    python python_quant/scripts/research_vignette.py --steps 2000 --seed 0x51ED

The script inserts ``python_quant/`` on ``sys.path`` itself, so no ``PYTHONPATH``
or install is required.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 safe
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python_quant"))

from nexus_quant.book_state import Side, StubOrderBook
from nexus_quant.research import event_frame, hit_rate, make_split, run_experiment
from nexus_quant.research.features import (
    deep_imbalance,
    lob_imbalance,
    microprice,
    mid,
    ofi,
    spread_bps,
)
from nexus_quant.research.labels import forward_mid_move

_FEATURE_FNS = {
    "lob_imbalance": lob_imbalance,
    "microprice_off": lambda v: microprice(v) - mid(v),
    "deep_imbalance": lambda v: deep_imbalance(v, k=5),
    "spread_bps": spread_bps,
}
_PAIR_FNS = {"ofi": ofi}


def _hit_nonzero(y_true: list[float], y_pred: list[float]) -> float | None:
    """hit rate on rows where the label is nonzero — the meaningful metric
    when integer-tick labels produce many exact-zero ties."""
    pairs = [(a, b) for a, b in zip(y_true, y_pred) if a != 0.0]
    if len(pairs) < 10:
        return None
    yt, yp = zip(*pairs)
    return hit_rate(list(yt), list(yp))


def synthetic_events(*, steps: int, seed: int, offset_scale: int = 8):
    """Seeded adds/cancels around a mid random-walk (uncorrelated by design)."""
    rng = np.random.default_rng(seed)
    m = 15_000.0
    for i in range(steps):
        m += rng.normal(0.0, 1.0)
        px = round(m) + int(rng.integers(-offset_scale, offset_scale + 1))
        side = int(rng.integers(0, 2))
        sz = int(rng.integers(5, 60))
        yield {"kind": "add", "side": side, "px": px, "sz": sz}
        if i % 3 == 0:  # occasional partial cancel to move queue markers
            yield {"kind": "cancel", "side": side, "px": px, "sz": max(1, int(sz * 0.4))}


def make_apply(book: StubOrderBook):
    def _apply(ev: dict) -> dict:
        side = Side(ev["side"])
        if ev["kind"] == "add":
            book.add(side, ev["px"], ev["sz"])
        elif ev["kind"] == "cancel":
            book.cancel(side, ev["px"], ev["sz"])
        return book.view()

    return _apply


def run_horizon(events: list[dict], *, h: int, train: float, val: float) -> list[dict]:
    book = StubOrderBook()
    rows = event_frame(
        events,
        make_apply(book),
        feature_fns=_FEATURE_FNS,
        prev_feature_fns=_PAIR_FNS,
        h=h,
        label_fn=forward_mid_move,
    )
    make_split(rows, train=train, val=val, gap=h)  # stamps row.split (blocks for split_tags)
    y_all = [r.label for r in rows]
    tags = [r.split for r in rows]
    results: list[dict] = []
    for feat in list(_FEATURE_FNS) + list(_PAIR_FNS):
        pred = [r.features[feat] for r in rows]
        bundle = run_experiment(y_all, pred, split_tags=tags)
        o = bundle["overall"]
        split_mask = {tag: [t == tag for t in tags] for tag in ("train", "val", "test")}
        results.append(
            {
                "horizon_h": h,
                "feature": feat,
                "n": o["n"],
                "rank_ic": o["rank_ic"],
                "ci95_lo": None if o["ci95"] is None else o["ci95"]["lo"],
                "ci95_hi": None if o["ci95"] is None else o["ci95"]["hi"],
                "hit_rate": _hit_nonzero(y_all, pred),
                "decile_spread": o["decile_spread"],
                "icir": o["icir"],
                "split": "overall",
            }
        )
        for tag in ("train", "val", "test"):
            ps = bundle["per_split"].get(tag)
            if not ps:
                continue
            m = split_mask[tag]
            results.append(
                {
                    "horizon_h": h,
                    "feature": feat,
                    "n": ps["n"],
                    "rank_ic": ps["rank_ic"],
                    "ci95_lo": None if ps["ci95"] is None else ps["ci95"]["lo"],
                    "ci95_hi": None if ps["ci95"] is None else ps["ci95"]["hi"],
                    "hit_rate": _hit_nonzero(
                        [y for y, keep in zip(y_all, m) if keep],
                        [p for p, keep in zip(pred, m) if keep],
                    ),
                    "decile_spread": ps["decile_spread"],
                    "icir": ps["icir"],
                    "split": tag,
                }
            )
    return results


def fmt(x: float | None, scale: float = 1.0, w: int = 7) -> str:
    if x is None or math.isnan(x):
        return "-" * w
    return f"{x * scale:>{w}.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=2000, help="synthetic events")
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=0x51ED)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--train", type=float, default=0.6)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=None, help="write JSON results")
    args = ap.parse_args()

    events = list(synthetic_events(steps=args.steps, seed=args.seed))
    all_rows: list[dict] = []
    print(f"synthetic E1–E4 · {len(events)} events · seed {args.seed}")
    print("honest read: this is a random-walk flow — IC≈0 is the expected result\n"
          "* hit% = sign-match on rows where the label is nonzero (zeros-tie-free)\n")
    for h in args.horizons:
        res = run_horizon(events, h=h, train=args.train, val=args.val)
        print(f"--- horizon h={h} (forward mid move, ticks) ---")
        print(f"{'feature':<16}{'rank_ic':>8}{'ci95':>16}{'hit%*':>7}{'decile':>9}")
        for r in res:
            if r["split"] != "overall":
                continue
            ci = f"[{fmt(r['ci95_lo'])},{fmt(r['ci95_hi'])}]" if r["ci95_lo"] is not None else "-"
            print(
                f"{r['feature']:<16}{fmt(r['rank_ic'], 1, 8):>8}{ci:>16}"
                f"{fmt(None if r['hit_rate'] is None else r['hit_rate'], 100, 6):>7}"
                f"{fmt(r['decile_spread'], 1, 9):>9}"
            )
        all_rows.extend(res)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(all_rows, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()