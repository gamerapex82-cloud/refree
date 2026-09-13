#!/usr/bin/env python3
"""Phase 3 — fair re-verification of the PPO-vs-VWAP slippage headline (plan_2.md §6).

The Part-1 "+50.4% lower slippage than VWAP" number was produced with (1) a
regime indicator only the agent could see, (2) evaluation on the training
distribution, (3) slippage vs self-VWAP, (4) no fees / queue, (5) one seed
family and no CI, (6) a regime tuned until the number appeared. This script
re-runs the comparison with every one of those fixed:

* **Symmetric information** — two modes: ``novol`` (nobody sees the regime
  flag, obs dim 44) and ``volsym`` (the flag is in ``obs[44]`` AND every
  baseline reads the same flag via ``baselines.regime_indicator``).
* **Hold-out regimes** — the agent trains on ``highvol`` only and is scored on
  ``calm``, ``lowvol``, ``highvol_null`` (random-walk null arm), ``trending``,
  ``liquidity_shock`` as well.
* **Fees + queue ON** for every reported number (``envs.regimes.COSTS_ON``).
* **Fair baselines** — ``schedule_twap`` / ``adaptive_pov`` / ``is_aware``
  next to the legacy ``twap`` / ``vwap`` / ``pov`` / ``passive``.
* **≥5 training seeds × 5 eval seed families**, block-bootstrap 95% CIs, and
  a **paired** per-episode CI of ``best baseline − agent`` on identical tapes.
* Slippage is reported vs arrival (IS) **and** vs the tape's **market VWAP**.

The honest outcome may be "not significantly better" — that is recorded, not
hidden. Output: ``docs/results/rl_fairness.json`` + ``docs/results/rl_fairness.md``.

Run (from repo root; ~10–20 min at the defaults on one core)::

    python python_quant/scripts/rl_fairness_study.py --iters 600 --train-seeds 5
    python python_quant/scripts/rl_fairness_study.py --quick     # smoke (minutes)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 safe

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "python_quant"))

from nexus_quant.agents import (
    PPOConfig,
    PPOPolicy,
    ci_from_rows,
    format_regime_table,
    paired_difference_ci,
    run_regime_episodes,
    train_ppo,
)
from nexus_quant.baselines import ALL_BASELINES
from nexus_quant.envs.regimes import (
    HOLDOUT_REGIMES,
    TRAIN_REGIME,
    regime_factories,
)

MODES = ("novol", "volsym")
METRICS = ("shortfall_bps", "vwap_slip_bps", "completion", "reward")


def _train(mode: str, seed: int, iters: int, episodes: int, out_dir: Path) -> PPOPolicy:
    path = out_dir / f"policy_{mode}_seed{seed}.npz"
    if path.exists():
        return PPOPolicy.load(str(path))
    factory = regime_factories((TRAIN_REGIME,), costs=True, vol_feature=(mode == "volsym"))[TRAIN_REGIME]
    cfg = PPOConfig(iterations=iters, episodes=episodes, epochs=4, eval_every=0,
                    seed=0xACE + seed, unit_seed=0x2717 + 100_000 * seed)
    t0 = time.time()
    policy, _ = train_ppo(factory, cfg)
    policy.save(str(path))
    print(f"  trained {mode} seed {seed}: {iters} iters in {time.time() - t0:.0f}s -> {path.name}")
    return policy


def _ci_dict(ci) -> dict:
    return {"mean": ci.mean, "lo": ci.ci95[0], "hi": ci.ci95[1], "n": ci.n_episodes,
            "per_seed_mean": ci.per_seed_mean}


def run(args: argparse.Namespace) -> dict:
    out_dir = Path(args.artifacts)
    out_dir.mkdir(parents=True, exist_ok=True)
    regimes_all = (TRAIN_REGIME, *HOLDOUT_REGIMES)
    result: dict = {
        "config": {
            "iters": args.iters, "episodes": args.episodes, "train_seeds": args.train_seeds,
            "eval_seeds": args.eval_seeds, "episodes_per_seed": args.episodes_per_seed,
            "train_regime": TRAIN_REGIME, "holdout_regimes": list(HOLDOUT_REGIMES),
            "costs_on": True, "baselines": list(ALL_BASELINES),
        },
        "modes": {},
    }
    for mode in MODES:
        print(f"\n=== mode {mode} ({'agent + baselines see the regime flag' if mode == 'volsym' else 'nobody sees a regime flag'}) ===")
        vol = mode == "volsym"
        factories = regime_factories(regimes_all, costs=True, vol_feature=vol)
        policies = [_train(mode, s, args.iters, args.episodes, out_dir) for s in range(args.train_seeds)]
        common = {"seeds": args.eval_seeds, "episodes_per_seed": args.episodes_per_seed}
        # every strategy runs the SAME seeded episodes once; all metrics derive from those rows
        base_eps = run_regime_episodes(None, factories, baselines=ALL_BASELINES, **common)
        agent_eps = [run_regime_episodes(pol, factories, agent_name="ppo", **common) for pol in policies]
        mode_res: dict = {"per_metric": {}, "paired": {}}
        for metric in METRICS:
            mode_res["per_metric"][metric] = {}
            for r in regimes_all:
                seeds_ci = [_ci_dict(ci_from_rows(ae[r]["ppo"], metric=metric, name="ppo", regime=r,
                                                  n_boot=args.n_boot)) for ae in agent_eps]
                mode_res["per_metric"][metric][r] = {
                    "ppo_seeds": seeds_ci,
                    "ppo_pooled_mean": float(np.mean([d["mean"] for d in seeds_ci])),
                    "ppo_seed_std": float(np.std([d["mean"] for d in seeds_ci])),
                    "baselines": {
                        b: _ci_dict(ci_from_rows(base_eps[r][b], metric=metric, name=b, regime=r, n_boot=args.n_boot))
                        for b in ALL_BASELINES
                    },
                }
        table = {
            r: {b: ci_from_rows(base_eps[r][b], metric="shortfall_bps", name=b, regime=r, n_boot=args.n_boot)
                for b in ALL_BASELINES}
            for r in regimes_all
        }
        print(format_regime_table(table))
        # paired per-episode difference vs the best baseline and vs VWAP, per training seed
        for r in regimes_all:
            bl = mode_res["per_metric"]["shortfall_bps"][r]["baselines"]
            best = min(bl, key=lambda b: bl[b]["mean"])
            entry = {"best_baseline": best, "best_baseline_mean": bl[best]["mean"],
                     "vwap_mean": bl["vwap"]["mean"], "vs_best": [], "vs_vwap": []}
            for pol, ae in zip(policies, agent_eps):
                merged = {r: {"ppo": ae[r]["ppo"], **base_eps[r]}}
                for key, target in (("vs_best", best), ("vs_vwap", "vwap")):
                    d = paired_difference_ci(pol, target, {r: factories[r]}, metric="shortfall_bps",
                                             n_boot=args.n_boot, episodes=merged, **common)[r]
                    entry[key].append(d)
            mode_res["paired"][r] = entry
            ppo_mean = mode_res["per_metric"]["shortfall_bps"][r]["ppo_pooled_mean"]
            sig = sum(1 for d in entry["vs_best"] if d["lo"] > 0)
            worse = sum(1 for d in entry["vs_best"] if d["hi"] < 0)
            print(f"  {r:<16} ppo {ppo_mean:6.3f} vs best baseline {best} {bl[best]['mean']:6.3f} | "
                  f"paired Δ(best−ppo) per seed: "
                  + " ".join(f"{d['mean']:+.2f}[{d['lo']:+.2f},{d['hi']:+.2f}]" for d in entry["vs_best"])
                  + f" | sig better {sig}/{len(policies)}, sig worse {worse}/{len(policies)}")
        result["modes"][mode] = mode_res
    return result


def render_markdown(res: dict) -> str:
    cfg = res["config"]
    lines = [
        "# RL fairness re-verification (Phase 3, plan_2.md §6)",
        "",
        (
            f"Training regime **{cfg['train_regime']}**, hold-outs {', '.join(cfg['holdout_regimes'])}. "
            f"{cfg['train_seeds']} training seeds × {cfg['eval_seeds']} eval seed families × "
            f"{cfg['episodes_per_seed']} episodes; PPO {cfg['iters']} iters × {cfg['episodes']} episodes/iter. "
            "Fees + queue model **on** for every number. Slippage positive = a cost (bps)."
        ),
        "",
        (
            "`Δ` is the **paired** per-episode difference `baseline − PPO` on identical seeded tapes "
            "(positive = PPO better) with a block-bootstrap 95% CI; `sig` counts training seeds whose CI "
            "excludes 0 in PPO's favour / against it."
        ),
        "",
    ]
    for mode, mr in res["modes"].items():
        title = ("nobody observes the regime flag (obs dim 44)" if mode == "novol"
                 else "regime flag visible to the agent AND every baseline (symmetric)")
        lines += [f"## Mode `{mode}` — {title}", ""]
        lines += ["| regime | PPO shortfall (seed-mean ± seed-std) | best baseline | Δ vs best [95% CI] per training seed | sig better / worse | Δ vs VWAP (mean over seeds) | PPO completion | PPO slip vs market VWAP |",
                  "|---|---|---|---|---|---|---|---|"]
        for r, pm in mr["per_metric"]["shortfall_bps"].items():
            pr = mr["paired"][r]
            deltas = " ".join(f"{d['mean']:+.2f} [{d['lo']:+.2f},{d['hi']:+.2f}]" for d in pr["vs_best"])
            sig_b = sum(1 for d in pr["vs_best"] if d["lo"] > 0)
            sig_w = sum(1 for d in pr["vs_best"] if d["hi"] < 0)
            dv = np.mean([d["mean"] for d in pr["vs_vwap"]])
            pv = np.mean([d["pct_vs_baseline"] for d in pr["vs_vwap"]])
            comp = mr["per_metric"]["completion"][r]["ppo_pooled_mean"]
            mv = mr["per_metric"]["vwap_slip_bps"][r]["ppo_pooled_mean"]
            lines.append(
                f"| {r} | {pm['ppo_pooled_mean']:.3f} ± {pm['ppo_seed_std']:.3f} | "
                f"{pr['best_baseline']} {pr['best_baseline_mean']:.3f} | {deltas} | {sig_b} / {sig_w} | "
                f"{dv:+.3f} bps ({pv:+.1f}%) | {comp:.3f} | {mv:+.3f} |"
            )
        lines += ["", "Baselines (shortfall_bps, mean [95% CI]):", ""]
        regimes = list(mr["per_metric"]["shortfall_bps"].keys())
        bnames = list(next(iter(mr["per_metric"]["shortfall_bps"].values()))["baselines"].keys())
        lines.append("| baseline | " + " | ".join(regimes) + " |")
        lines.append("|---|" + "---|" * len(regimes))
        for b in bnames:
            cells = []
            for r in regimes:
                d = mr["per_metric"]["shortfall_bps"][r]["baselines"][b]
                cells.append(f"{d['mean']:.3f} [{d['lo']:.3f},{d['hi']:.3f}]")
            lines.append(f"| {b} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--train-seeds", type=int, default=5)
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--episodes-per-seed", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--artifacts", default="python_quant/artifacts/fairness", help="policy cache (gitignored .npz)")
    ap.add_argument("--out", type=Path, default=_ROOT / "docs" / "results" / "rl_fairness.json")
    ap.add_argument("--quick", action="store_true", help="smoke settings (2 seeds, 40 iters, 4 episodes)")
    args = ap.parse_args(argv)
    if args.quick:
        args.iters, args.train_seeds, args.eval_seeds, args.episodes_per_seed, args.n_boot = 40, 2, 2, 4, 200
        args.artifacts = str(Path(args.artifacts) / "quick")
    t0 = time.time()
    res = run(args)
    res["config"]["wall_seconds"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    md = args.out.with_suffix(".md")
    md.write_text(render_markdown(res))
    print(f"\nwrote {args.out} and {md} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
