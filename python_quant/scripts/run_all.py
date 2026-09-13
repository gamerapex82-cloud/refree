#!/usr/bin/env python3
"""One-command reproduce of the Part 2 research layer (plan_2.md Phase 5).

Steps (each is skippable; everything runs from the repo root, pure Python)::

    1. tests      python -m pytest python_quant/tests -q
    2. vignette   scripts/research_vignette.py        (synthetic null harness)
    3. fetch      scripts/fetch_itch.py --day … --symbols …   (public NASDAQ ITCH day → data/)
    4. research   scripts/run_research.py --day … --symbols … (E1–E6 on the real tape)
    5. fairness   scripts/rl_fairness_study.py         (fair PPO re-verification, E7)

Defaults reproduce ``docs/results/`` exactly (same day, symbols, seeds, n_boot).
``--quick`` runs a smoke of every stage in a few minutes: 64 MB of the gzip,
one symbol, 100k rows, 2 training seeds.

    python python_quant/scripts/run_all.py            # full (≈ 1 h on one core; fetch ≈ 14 min)
    python python_quant/scripts/run_all.py --quick
    python python_quant/scripts/run_all.py --skip fetch,fairness
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "python_quant" / "scripts"
STAGES = ("tests", "vignette", "fetch", "research", "fairness")


def _run(label: str, cmd: list[str]) -> float:
    print(f"\n### {label}\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=_ROOT, check=True)
    dt = time.time() - t0
    print(f"### {label} done in {dt:.0f}s", flush=True)
    return dt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", default="12302019")
    ap.add_argument("--symbols", default="AAPL,QQQ")
    ap.add_argument("--skip", default="", help="comma-separated stages to skip: " + ",".join(STAGES))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    unknown = skip - set(STAGES)
    if unknown:
        ap.error(f"unknown stage(s) {sorted(unknown)}; known: {STAGES}")
    py = sys.executable
    symbols = args.symbols.split(",")[0] if args.quick else args.symbols
    timings: dict[str, float] = {}

    if "tests" not in skip:
        timings["tests"] = _run("tests", [py, "-m", "pytest", "python_quant/tests", "-q"])
    if "vignette" not in skip:
        timings["vignette"] = _run("vignette", [py, str(_SCRIPTS / "research_vignette.py"), "--steps", "800" if args.quick else "2000"])
    if "fetch" not in skip:
        cmd = [py, str(_SCRIPTS / "fetch_itch.py"), "--day", args.day, "--symbols", symbols]
        if args.quick:
            cmd += ["--max-gz-bytes", str(64 << 20)]
        timings["fetch"] = _run("fetch", cmd)
    if "research" not in skip:
        cmd = [py, str(_SCRIPTS / "run_research.py"), "--day", args.day, "--symbols", symbols, "--n-boot", "300"]
        if args.quick:
            cmd += ["--max-events", "100000", "--n-boot", "100", "--out-dir", str(_ROOT / "docs" / "results" / "quick")]
        timings["research"] = _run("research", cmd)
    if "fairness" not in skip:
        cmd = [py, str(_SCRIPTS / "rl_fairness_study.py")]
        if args.quick:
            cmd += ["--quick", "--out", str(_ROOT / "docs" / "results" / "quick" / "rl_fairness.json")]
        timings["fairness"] = _run("fairness", cmd)

    print("\n=== run_all summary ===")
    for k, v in timings.items():
        print(f"  {k:<10}{v:8.0f}s")
    print(f"  {'total':<10}{sum(timings.values()):8.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
