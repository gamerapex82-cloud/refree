# Person B Handoff: Python Quantitative Layer

**Updated:** 2026-09-04
**Checkout:** `C:\Users\Shrikar\nexus\parent-transfer`
**Branch:** `main` at `d40bf59` (tracking `origin/main`)

## Scope

Person B owns the Python quantitative layer: ITCH decoding, L2 replay, the
Gymnasium execution environment, book adapters, baselines, and their tests.
The C++ matching engine, pybind bridge, ABI parity, and shared contract belong
to Person A.

## Current Status

The Person-B implementation is now present on upstream `main`. PR #2 ("Add
Person B execution environment and ITCH replay") was merged into `main` on
2026-09-03 as
`bf08948c6ba48e0a76b8f2ac515be5721f33b9ca`; it is closed, with no review or
issue comments. GitHub `main` and local `origin/main` are now synchronized at
`d40bf59` (2026-09-04).

Lokesh's `d40bf59` follow-up adds the missing `EngineAdapter.lookup()` and
`EngineAdapter.cancel_id()` methods in `book_port.py`, adds
`bindings/tests/test_diff_engine_stub.py`, and updates the handoff docs. The
partial-cancel path uses same-price `engine.modify()` to preserve priority.
That fix is present in the checked-out `main` branch.

## Completed Files

The original Person-B branch contributed 11 Python-layer files and 1,511 added
lines:

- `python_quant/nexus_quant/itch_parser.py`
- `python_quant/nexus_quant/replay.py`
- `python_quant/nexus_quant/book_port.py`
- `python_quant/nexus_quant/envs/order_book_env.py`
- `python_quant/nexus_quant/envs/__init__.py`
- `python_quant/nexus_quant/baselines.py`
- `python_quant/nexus_quant/__init__.py`
- `python_quant/requirements.txt`
- `python_quant/tests/test_itch_parser.py`
- `python_quant/tests/test_replay.py`
- `python_quant/tests/test_order_book_env.py`

## Frozen Boundaries

Do not modify `cpp_engine/**`, `bindings/pybind_wrapper.cpp`,
`bindings/CONTRACT.md`, `python_quant/nexus_quant/book_state.py`,
`BOOK_STATE_DTYPE`, `DEPTH`, `Side`, field order/names, integer-tick pricing,
or the `view()` zero-copy contract. No frontend, TypeScript, or React work is
part of this repository task.

## ITCH Parser

`itch_parser.py` is a lazy, bounded-chunk NASDAQ TotalView-ITCH 5.0 parser.
It decodes A/F/E/C/X/D/U/P messages, big-endian fields, 6-byte timestamps to
`ts_ns`, 8-byte order IDs, and integer `price_ticks` without float conversion.
It accepts raw concatenated messages and 2-byte length-prefixed messages.
Unknown types and a truncated tail are counted in `ItchParseStats` and do not
raise. Events are normalized as `NormalizedEvent` values.

## Replay

`ReplayEngine` applies normalized events to an injectable adapter and emits
owning `ReplayFrame` snapshots. A/F rest orders; E/C execute and reduce/remove
orders; X partially cancels; D fully cancels; U cancels and replaces; P records
a trade without displayed-size removal. `check_integrity()` checks crossed or
locked BBO, empty BBO, negative sizes, and ladder ordering.

## Book Adapter

`book_port.py` defines the view/execution protocols, `StubBookAdapter` over the
frozen `StubOrderBook`, and the `EngineAdapter` swap seam for
`nexus_engine.Engine`. It supports rest/take/cancel, zero-copy view vs owning
snapshot semantics, and the `lookup()`/`cancel_id()` methods required by
real-engine replay. Partial cancellation uses same-price `engine.modify()` to
preserve priority.

## OrderBookEnv

`OrderBookEnv` is a long-inventory liquidation environment with defaults
`Q=2000`, `T=40`, a continuous `Box([-1, 1], shape=(1,))` action, and an exact
44-element float32 observation:

- 0-9 bid distance from mid; 10-19 bid size
- 20-29 ask distance from mid; 30-39 ask size
- 40 inventory/Q; 41 remaining time/T
- 42 mark-to-market PnL normalized by `Q*10` and clipped to `[-3, 3]`
- 43 spread normalized by 20 ticks

Sell semantics: `a <= -0.92` sends a market sell; other actions map to
`round(a*12)` ticks around mid, with crossing limits capped to market behavior.
The previous residual is canceled each step. Child size is a clamped
ceil(inventory/time-left), with minimum 20 and configurable maximum. Horizon
leftovers are market-dumped. Arrival mid at reset benchmarks implementation
shortfall; reward combines normalized shortfall, inventory/time pressure,
adverse-mid movement, and an extra leftover penalty at truncation. Metrics
include VWAP, shortfall bps, filled/leftover quantity, PnL ticks, reward, and
steps.

## Baselines

`baselines.py` supplies TWAP, VWAP, POV, and Passive policies, plus
`run_episode()` and `compare()` returning execution metrics compatible with the
environment.

## Verification

Latest checks in this shell:

- `python -m compileall -q python_quant`: **PASS**
- Direct parser/replay smoke (encode one ADD, decode, replay into stub): **PASS**
- `python -m pytest python_quant/tests -q`: **NOT RUN**; Python 3.10.2 is
  available, but `pytest` is not installed (`No module named pytest`).
- Dependency probe: `numpy` installed; `gymnasium` and `pytest` absent.

There are 26 pytest test functions across the four Python test modules,
including the existing contract smoke test. The previously reported 26-pass
result is historical and was not reproducible in this checkout without the
missing dependencies. Run `pip install -r python_quant/requirements.txt` and
pytest in WSL/venv.

## Integration Blockers and Limitations

- The compiled `nexus_engine` module and ABI parity tests still require the
  Linux/WSL CMake + pybind environment described in `CLAUDE.md`.
- Run `pytest python_quant/tests bindings/tests/test_abi_parity.py
  bindings/tests/test_diff_engine_stub.py -v` after building the module.
- The real-engine replay fix is now on `main`; the remaining requirement is to
  build `nexus_engine` and run the parity/diff tests.
- No PPO/GRPO training agent, CUDA VaR/CVaR engine, or Python dashboard is
  implemented yet.
- Replay is an aggregate L2 reconstruction and intentionally does not retain a
  full ITCH market-state model beyond the adapter's tracked resting orders.

## Next Steps

1. In WSL, build the pybind module and run ABI, Python, and Engine-vs-Stub
   diff-test suites.
2. Confirm the remote-main `EngineAdapter` fix and diff-test pass against the
   real engine without changing frozen files.
3. Continue with the separate RL agent, risk analytics, and dashboard work when
   those milestones are scheduled.

## Files Changed This Turn

- `progress_b.md` (this handoff only)

---

## Update 2026-09-07 — C++ integration on Linux

**Branch:** `feature/person-b-polish` (PR #7)
**Engine build:** `cmake -S . -B /tmp/nexus_build -DCMAKE_BUILD_TYPE=Release -DNEXUS_BUILD_PYBIND=ON -DNEXUS_ENABLE_CUDA=OFF` then `cmake --build`. Module: `bindings/nexus_engine.cpython-310-x86_64-linux-gnu.so` (gitignored).

### Completed this session
- Built `nexus_engine` on Linux (g++ 12, CMake 4.4, pybind11, Python 3.10). No C++ / pybind / contract edits.
- CTest 5/5: abi_check, lob_test, ring_test, id_map_test, risk_test.
- `abi_check`: sizeof 448, alignof 8, contract v1.
- Python+bindings pytest: **75 passed, 0 skipped** (was 59 passed / 3 skipped before the .so existed).
- Risk parity `test_risk_parity.py`: **3/3**. Project check is `pytest.approx(..., abs=1e-12)`, not raw `==`. Example GBM 60k×120: engine var `0.3253218005627372` vs NumPy `0.3253218005627373`.
- Engine-vs-Stub `test_diff_engine_stub.py`: **2/2**.
- ABI `test_abi_parity.py`: **6/6**.
- `EngineAdapter.reset()` added so `OrderBookEnv.reset()` no longer silently swaps in `StubBookAdapter`. Limit / market / cancel / fills / env inventory identity verified on the real engine.
- Dashboard `read_shm_ring_latest()` decodes Person A's control block (48 B) + latest 448 B slot. Live check against `ring_producer /nex_lob_b`: seq/BBO/spread decoded; producer unlinks the segment on exit (unchanged C++ behavior).

### GPU
Blocked. No `nvcc` / `nvidia-smi` on this machine. `risk_bench` CPU path: 200k×252 in 1603.1 ms, `var=0.324574 cvar=0.389231`. GPU line: "not compiled (no CUDA toolkit)".

### Tests run
```
ctest --test-dir /tmp/nexus_build --output-on-failure     # 5/5
PYTHONPATH=python_quant:bindings python -m pytest python_quant/tests bindings/tests -q
# 75 passed
PYTHONPATH=python_quant:bindings python -m pytest \
  python_quant/tests/test_risk_parity.py \
  bindings/tests/test_diff_engine_stub.py \
  bindings/tests/test_abi_parity.py -v
# 11 passed
/tmp/nexus_build/risk_bench   # CPU only
```

### Remaining
- GPU `risk_bench` + ~40× claim on a CUDA box.
- Keep a producer alive if the dashboard should follow a live ring (C++ demo calls `ShmRing::destroy` on exit).
- Push/merge PR #7 on `Lokeshrao69/Nexus_LOB` after Person A review.

---

## Update 2026-09-13 — Part 2 Phases 3 + 4 (queue dynamics, fair RL re-verification, real ITCH tape)

**Branch:** `hoplite/kranioi-5b44d8a8` (Person B session; Linux sandbox, Python 3.12, numpy 2.5, gymnasium 1.3).
**Plan of record:** `plan_2.md` Phase 3 (E5/E6 + §6 RL fairness rework) and Phase 4 (real tape). Work package: `docs/work_package_b_phases_2_4.md` §3–§4.

### Landed (commits `f1f07f9`, `f31c8c1`, + this docs/results commit)

| Piece | File | State |
|---|---|---|
| Order-level queue tracker (ahead/behind qty, first-fill time, touch distances) | `research/queue_dynamics.py::OrderLevelTracker` | ✅ 0 unknown ids on real AAPL/QQQ tape |
| E5 — Kaplan–Meier P(fill by τ), cancel = competing risk | `research/queue_dynamics.py::fill_prob_survival` | ✅ hand-computed KM pinned in tests |
| E5 — logistic fill model (walk-forward holdout, Brier, calibration slope) | `research/queue_dynamics.py::logistic_fill_model` | ✅ recovers a known logistic queue (slope ≈ 1) |
| E6 — post-fill drift, NW t-stats, pre-fill matched control, P(adverse) | `research/adverse_selection.py` | ✅ |
| Named regimes + random-walk null arm + fees/queue overlay | `envs/regimes.py` | ✅ `highvol_null` = highvol with `gap_down_prob=0.5`, nothing else |
| Fair baselines `schedule_twap` / `adaptive_pov` / `is_aware` + symmetric `regime_indicator` | `baselines.py` | ✅ legacy 4 byte-identical |
| Per-regime multi-seed CI harness + paired CI | `agents/evaluate.py` (`run_regime_episodes`, `ci_from_rows`, `evaluate_regime_ci`, `paired_difference_ci`) | ✅ |
| Fairness study CLI + results | `scripts/rl_fairness_study.py` → `docs/results/rl_fairness.{md,json}` | ✅ run at 5×5×20, 600 iters |
| Public NASDAQ ITCH fetcher + per-symbol slicer | `scripts/fetch_itch.py` | ✅ streamed 3.5 GB gz → 268.7 M msgs in 837 s |
| E1–E6 real-tape runner | `scripts/run_research.py` → `docs/results/real_tape_12302019*.{md,json}` | ✅ |
| CKS level-1 OFI (touch-move aware) | `research/features.py::ofi` | ✅ tests pin the definition |
| Vectorized Spearman ranks / block bootstrap; decile_spread on true bins | `research/experiments.py` | ✅ output-identical, ~50× faster |
| Tests | `test_queue_dynamics.py` (11), `test_adverse_selection.py` (8), `test_rl_fairness.py` (18), `test_offline_real_tape.py` (3) | ✅ Tier 1 **152 passed** |

### Verified here
- `python -m pytest python_quant/tests` → **152 passed** (was 110 / 1 skipped); `ruff check python_quant/nexus_quant/ bindings/` clean.
- Engine build (g++ 13, CMake 4.4, pybind11): CTest **5/5**; Tier 2 `bindings/tests` still green.
- **Real bytes:** `12302019.NASDAQ_ITCH50.gz` (public sample) → AAPL 1,519,370 msgs / 0 truncated; replay integrity clean except the one `empty_bbo` on the first pre-open message; **Engine-vs-Stub L2 ladder parity exact over 7,037 real pre-market frames** (engine built with a $100–$500 price band — the default `Engine()` band is 1..100 000 ticks, far below real ITCH prices; adds outside the band are rejected, so run the real-tape diff with `ne.Engine(1_000_000, 5_000_000, 1<<20)`).

### The honest headline re-characterization (plan_2.md §6 — done)
`docs/results/rl_fairness.md`: PPO trained on `highvol` only, fees+queue on, 5 training seeds × 5 eval seed families × 20 episodes, paired vs the **best fair baseline** on identical tapes:
- **highvol / highvol_null / trending:** not significantly different from `adaptive_pov` (0/5 seeds significant either way, |Δ| ≲ 0.3 bps).
- **calm / lowvol hold-outs:** PPO significantly **worse** than the best baseline in 5/5 seeds (−0.05 … −0.16 bps).
- **liquidity_shock:** PPO significantly **better** in 5/5 (novol) / 4/5 (volsym) seeds, +0.4 … +1.4 bps.
- The old "+50.4 % vs VWAP" compared against a 2-line heuristic with asymmetric information; against `adaptive_pov` the edge is ~0. **Not claimed.**

### Blocked / not done
- GPU (`nvcc`) still absent — Person A item, unchanged.
- The real-tape E1–E6 numbers are from ONE day (12/30/2019) and two symbols; more days are one `fetch_itch.py --day` away (each full day ≈ 14 min to stream).

## Update 2026-09-13 (final) — Part 2 Phase 5 (report, one-command reproduce, honest docs). Research half complete.

**Branch:** `hoplite/kranioi-5b44d8a8` (third commit of the session; the two before it are Phases 3 + 4 above).
**Plan of record:** `plan_2.md` Phase 5 — all three items landed; §9 DoD ticks updated; §10 log appended.

### Landed

| Piece | File | State |
|---|---|---|
| Full-day E1–E6 on both symbols (`--n-boot 300`, 1,289 s) | `docs/results/real_tape_12302019.md` + `_AAPL.json` / `_QQQ.json` | ✅ AAPL 1,484,259 / QQQ 2,209,131 regular-session rows, 0 truncated, integrity clean, 0 unknown ids |
| Hit-rate definition fix in the runner | `scripts/run_research.py` | ✅ sign agreement only where label **and** feature are non-zero + `cov` (share of non-zero-label rows the feature takes a view on) — a zero per-event OFI is "no view", not a miss |
| Full research report incl. **negative results** (§8) | `docs/RESEARCH.md` | ✅ §1–§10; every table number script-checked against the final JSON (0 mismatches) |
| One-command reproduce | `scripts/run_all.py` | ✅ `tests → vignette → fetch → research → fairness`, `--skip`, `--quick` (gitignored `docs/results/quick/`) |
| Honest docs | `README.md`, `PROGRESS.md`, `CLAUDE.md`, `plan_2.md`, `docs/work_package_b_phases_2_4.md` | ✅ headline table = point estimates with CIs; PPO line = "not reproduced fairly" |

### What the real day says (test split, out of sample — details in `docs/RESEARCH.md`)
- **E1/E4:** L1 imbalance rank IC **0.140 → 0.229** (AAPL) and **0.155 → 0.462** (QQQ) from h=1 to h=25; CIs ±≤0.014; sign stable across train/val/test. Train-fit OLS of all features: 0.164 → 0.288 / 0.157 → 0.461.
- **E2:** microprice ≈ imbalance (within 0.01 everywhere; DM signs flip by symbol/horizon) — **E2 fails its own success criterion.**
- **E3:** order-level OFI (rolling 20) is a weaker, partly independent signal (0.09–0.21); it adds to the combination on AAPL, nothing on QQQ.
- **E5:** logistic fill model calibration slope **1.09 / 1.03** (criterion ≥ 0.8 passes), Brier skill only +0.080 / +0.016; 94–98 % of resting orders cancel before trading.
- **E6:** passive fills adversely selected in **97–99 %** of cases at h=1 (0.90–0.96 through h=25), drift −20/−50/−93 ticks (AAPL) and −11/−26/−44 (QQQ) at h=1/5/25, NW |t| 68–147; pre-fill control ≈ 0 → selection, not momentum; OFI sign does not separate adverse from benign fills.
- **E7:** unchanged from the Phase 3 entry — no significant PPO edge vs `adaptive_pov` except under `liquidity_shock`; worse on calm/lowvol. The +50.4 % headline is retired.

### Verified here
- `python -m pytest python_quant/tests bindings/tests` → **160 passed** (152 Tier 1 + 8 Tier 2 with the built engine); `ruff check python_quant/nexus_quant/ bindings/` (CI lint scope) clean.
- `run_all.py --help` and every flag it forwards exist on the target scripts (`fetch_itch --max-gz-bytes`, `run_research --max-events/--n-boot/--out-dir`, `rl_fairness_study --quick/--out`, `research_vignette --steps`).

### Remaining (research half)
- More tape days / symbols — one `fetch_itch.py --day … --symbols …` each (≈ 14 min to stream the public day).
- Optional: add fill % / MDD columns to the E7 table from `execution.backtest.summarize` (metrics exist, not in the committed table).
- Person A: GPU (`nvcc`) and hardware throughput/latency numbers — unchanged.
