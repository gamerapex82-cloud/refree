# Nexus-LOB — project context & session handoff

> **Purpose of this file:** persistent memory across Claude Code sessions. Read it
> first every session. When you finish a chunk of work, update **§6 Status** and
> **§7 Next steps** so the next session resumes without re-deriving everything.
> **Part 2 (quant research layer) plan of record: `plan_2.md`** — read that FIRST
> for the research half; this file stays the systems-half handoff.
> Last updated: **2026-09-13** (Person B: Part 2 Phases 3–5 landed — see `plan_2.md` §10 and `progress_b.md`).

---

## 1. What this project is

**Nexus-LOB** — a hybrid **C++/Python** limit-order-book (LOB) trading &
market-microstructure platform, built as a **finance-placement portfolio project**
(target desks: JPMC Quantitative Research, Nomura Algo Strategies, Goldman Sachs
Systematics). The goal is a project that reads as *institutional-grade* rather than
a yfinance backtester — it signals low-latency systems design, order-book
mechanics, RL for optimal execution, and GPU risk analytics.

**Timeline:** ~2 months / 8 weeks, **2 people**.

Headline resume metrics (status):
- C++20 matching engine: **>500k orders/sec, sub-microsecond latency**, zero-alloc —
  **0 allocs/op proven**; throughput/latency to re-measure on real hardware.
- PPO/GRPO execution agent: **~14% lower slippage vs VWAP** —
  **retired 2026-09-13**: the +50.4% (2026-09-07) did not survive a fair re-verification
  (symmetric info, fees+queue, 5 seeds, hold-out regimes, paired CIs vs `adaptive_pov`) —
  no significant edge except under liquidity shocks. See `docs/results/rl_fairness.md`.
- **Real-tape microstructure research (Part 2):** E1–E6 on NASDAQ ITCH 2019-12-30 AAPL/QQQ —
  L1 imbalance rank IC 0.14–0.46 with tight CIs, calibrated fill model (slope 1.03–1.09),
  passive fills adversely selected 96–99%. `docs/RESEARCH.md`.
- CUDA Monte-Carlo VaR/CVaR: **~40× speedup** vs CPU —
  CPU ✅ exact parity; GPU kernel authored, **blocked** (no CUDA toolkit).

## 2. Architecture (5 subsystems)

1. **Ultra-low-latency matching engine (C++20)** — zero-allocation memory pools,
   intrusive doubly-linked lists + ring buffers, O(1) price-level lookup; Limit /
   Market / FOK / IOC / Cancel / Modify. ITCH 5.0 binary replay.
2. **Microstructure sim + RL execution agent (Python / Pybind11)** — the C++ engine
   exposed as a Gymnasium env; friction/impact modeling (queue drift, latency,
   Almgren–Chriss); PPO/GRPO agent vs TWAP/VWAP/Avellaneda–Stoikov baselines.
3. **GPU risk engine (CUDA / PyTorch C++ extension)** — 100k+ GBM / jump-diffusion
   paths in parallel → real-time VaR/CVaR fed back as a dynamic inventory penalty.
4. **Zero-copy pipeline + dashboard** — async WebSocket / shared-memory IPC → live
   L3 depth, spread heatmaps, fill-latency histograms, PnL.
5. *(numbered "subsystem 5" in code comments — the shared-memory ring → dashboard.)*

**Prices are always INTEGER TICKS**, never floats (bit-exact with the engine and
the ITCH feed).

## 3. Repo layout (monorepo — one GitHub repo)

```
Finance Project-1/            # repo root (branch: main)
├── cpp_engine/               # Person A — C++ engine (header-only)
│   ├── include/nexus/         # book_state, types, order_pool, limit_order_book, shm_ring, flow_gen
│   ├── tests/                 # lob_test(86), id_map_test(4.6M), ring_test(30k), abi_check, risk_test
│   ├── demos/                 # ring_producer / ring_probe (live shmem demo)
│   └── bench/                 # bench.cpp — 0 allocs/op
├── cuda_risk/                 # Person A — Monte-Carlo VaR/CVaR (CPU ✅, CUDA blocked)
├── python_quant/              # Person B — quant / RL
│   ├── nexus_quant/
│   │   ├── book_state.py      # dtype mirror + StubOrderBook (oracle)
│   │   ├── itch_parser.py     # ITCH 5.0 streaming parser
│   │   ├── replay.py          # ITCH→L2 replay engine
│   │   ├── book_port.py       # injectable stub↔engine adapter
│   │   ├── baselines.py       # TWAP / VWAP / POV / Passive
│   │   ├── risk.py            # NumPy VaR/CVaR oracle (exact parity with C++)
│   │   ├── dashboard.py       # SnapshotHub + slot codec (subsystem 4/5)
│   │   ├── dashboard_page.html # combined interactive desk page
│   │   ├── envs/order_book_env.py  # Gymnasium execution env (44-dim, high-vol regime)
│   │   └── agents/            # mlp.py, ppo.py, grpo.py, evaluate.py
│   ├── scripts/               # train_eval_agent.py, serve_dashboard.py
│   ├── tests/                 # 68 tests, all green (+ 1 shm skip on Windows)
│   └── artifacts/             # policy_ppo.npz, policy_ppo_highvol.npz
├── bindings/                  # pybind_wrapper.cpp, CONTRACT.md, tests/ (+ compiled .pyd)
├── dashboard/                 # static verification console (dashboard/index.html)
├── CMakeLists.txt             # engine lib + pybind + CTest + CUDA hooks
├── pyproject.toml             # scikit-build-core packaging
├── CLAUDE.md                  # this file
├── PROGRESS.md                # plain-language status
├── HIGHVOL_PLAN.md            # high-vol regime design + results
└── README.md                  # project overview
```

## 4. Two-person split

- **Person A — Systems & Infrastructure Lead:** C++ engine, memory pools, Pybind11,
  CUDA, IPC, profiling.
- **Person B — Quant Research & RL Lead:** ITCH parser, Gymnasium env, RL training,
  baselines, analytics dashboard.

**8-week roadmap (condensed):** W1-2 engine core + ITCH parser/replay · W3-4 pybind
bridge + Gymnasium env + IPC + baseline strategies · W5-6 CUDA risk engine + PPO/GRPO
training · W7-8 profiling, dashboard, benchmarks, write-up.

**Critical integration interfaces (freeze early to work in parallel):**
- **State contract** `nexus::BookStateView` / `BOOK_STATE_DTYPE` — **FROZEN (v1).**
- **Environment contract** `reset()`/`step()` I/O of `OrderBookEnv` — **not yet defined.**

**Git workflow:** `main` only holds code that compiles/runs. Feature branches
(`feature/memory-pool`, `feature/ppo-agent`, …) → PR → review → merge.

## 5. Design decisions already locked (don't silently change)

- **Contract-first seam.** Instead of writing the matching engine first, we froze the
  *cross-language state contract* so both people build in parallel. `StubOrderBook`
  (pure Python) emulates the future C++ `Engine.view()/snapshot()` interface, so the
  RL env can be built/trained before the C++ engine lands — and later becomes the
  **diff-test oracle** the C++ engine is validated against.
- **`BookStateView` layout is ABI-frozen (v1).** Depth `kDepth = 10` per side. Field
  order: all 8-byte members, then 4-byte, then 1-byte → padding-free on LP64.
  `sizeof == 40*kDepth + 48 == 448`. Any change = ABI break → bump
  `kBookStateContractVersion`, update `book_state.py` (`BOOK_STATE_DTYPE` + `DEPTH`),
  re-run parity tests.
- **Two view flavors across the seam:** `view()` = zero-copy live window (RL hot path,
  valid only until the next mutating engine call); `snapshot()` = owning copy (safe to
  retain, telemetry/tests/cross-thread).

## 6. Status — what's DONE (verified 2026-08-30)

**Phase 1 — C++ matching engine implemented & tested; pybind `Engine` wired to it.**

| Component | File | State |
|---|---|---|
| C++ state contract | `cpp_engine/include/nexus/book_state.hpp` | ✅ complete, ABI-locked (v1, 448 B) |
| Engine value types | `cpp_engine/include/nexus/types.hpp` | ✅ OrderId/Price/Qty, Fill, Status, ExecResult |
| Zero-alloc order pool | `cpp_engine/include/nexus/order_pool.hpp` | ✅ fixed slab + intrusive free-list |
| Matching engine | `cpp_engine/include/nexus/limit_order_book.hpp` | ✅ Limit/Market/FOK/IOC/Cancel/Modify |
| Engine correctness tests | `cpp_engine/tests/lob_test.cpp` | ✅ **86/86 checks pass** |
| Zero-alloc id→Order map | `cpp_engine/include/nexus/limit_order_book.hpp` (`IdMap`) | ✅ replaced `std::unordered_map` — **hot path is genuinely allocation-free** (bench proves 0 allocs/op) |
| IdMap stress tests | `cpp_engine/tests/id_map_test.cpp` | ✅ **4,676,294 checks pass** (collision/wraparound/backtrack-shift) |
| Shared-memory SPSC ring | `cpp_engine/include/nexus/shm_ring.hpp` | ✅ POSIX+Windows shmem, drop-new-on-full, 448-B slots |
| Synthetic flow generator | `cpp_engine/include/nexus/flow_gen.hpp` | ✅ seeded, deterministic pre-ITCH flow |
| Ring correctness tests | `cpp_engine/tests/ring_test.cpp` | ✅ **30,011 checks pass** |
| Ring producer + probe | `cpp_engine/demos/*.cpp` | ✅ live cross-process book demo (verified on Windows) |
| Benchmark harness | `cpp_engine/bench/bench.cpp` | ✅ throughput + latency + zero-alloc proof — **0 allocs/op on both workloads** (see §7 #8) |
| Python dtype mirror + `StubOrderBook` | `python_quant/nexus_quant/book_state.py` | ✅ complete (diff-test oracle) |
| Package exports | `python_quant/nexus_quant/__init__.py` | ✅ |
| Root build | `CMakeLists.txt` | ✅ engine lib + pybind module + CTest + CUDA hooks |
| Python packaging | `pyproject.toml` | ✅ scikit-build-core |
| Pybind bridge (REAL engine) | `bindings/pybind_wrapper.cpp` | ✅ order entry + zero-copy views + fills |
| Python smoke test (no build) | `python_quant/tests/test_contract_smoke.py` | ✅ |
| ABI parity test (needs build) | `bindings/tests/test_abi_parity.py` | ⚠️ updated for real engine — build & run in WSL |
| Standalone C++ ABI check | `cpp_engine/tests/abi_check.cpp` | ✅ **compiles+runs** |

Verified this session with MSYS2 g++ (all four green):
```
g++ -std=c++20 -O2 -Wall -Wextra -I cpp_engine/include cpp_engine/tests/abi_check.cpp -o abi_check.exe && ./abi_check.exe
g++ -std=c++20 -O2 -Wall -Wextra -I cpp_engine/include cpp_engine/tests/lob_test.cpp -o lob_test.exe && ./lob_test.exe
g++ -std=c++20 -O2 -Wall -Wextra -I cpp_engine/include cpp_engine/tests/ring_test.cpp -o ring_test.exe && ./ring_test.exe
g++ -std=c++20 -O2 -Wall -Wextra -I cpp_engine/include cpp_engine/tests/id_map_test.cpp -o id_map_test.exe && ./id_map_test.exe
```
→ ABI lock prints `contract v1, kDepth=10, sizeof=448 (expected 448), alignof=8`;
`lob_test` prints `86 checks, 0 failed` / `ALL PASS`; `ring_test` prints
`30011 checks, 0 failed` / `ALL PASS` (order+integrity over a writer/reader thread on
real OS shared memory, plus drop-new-on-full); `id_map_test` prints
`4676294 checks, 0 failed` / `ALL PASS` (collision/wraparound/backtrack-shift).
Python tests & the pybind module are **not** run here (no interpreter / no CMake —
see §8); compile `bindings/` in WSL. The benchmark harness proves **0 allocs/op**
on the hot path; throughput/latency re-measured in WSL (§7 #8). Ring demos run here
too: `./ring_producer.exe nex 200 ... & ./ring_probe.exe nex 15 5`.

**Phase 1b — Person B: ITCH replay + execution env (merged PR #2, `feature/env-and-itch`, 2026-09-04).**
Cross-checked this session: the C++ engine tests still pass, and the Python work below is
present and internally consistent (static review). Python is **not runnable in this shell**
(no interpreter — §8), so all Python items below are **authored; run them in WSL**.

| Component | File | State |
|---|---|---|
| ITCH 5.0 streaming parser | `python_quant/nexus_quant/itch_parser.py` | ✅ authored (Add/MPID/Exec/ExecPx/Cancel/Delete/Replace/Trade, framed+raw, lazy streaming) |
| ITCH→L2 replay engine | `python_quant/nexus_quant/replay.py` | ✅ authored (`ReplayEngine` + `check_integrity`, injectable book) |
| Injectable book adapter | `python_quant/nexus_quant/book_port.py` | ✅ authored (`StubBookAdapter` + `EngineAdapter` swap seam) |
| Gymnasium execution env | `python_quant/nexus_quant/envs/order_book_env.py` | ✅ authored (`OrderBookEnv`, 44-dim obs, IS reward + inv/time/adv penalties) |
| Execution baselines | `python_quant/nexus_quant/baselines.py` | ✅ authored (TWAP / VWAP / POV / Passive) |
| Person B tests | `python_quant/tests/test_{itch_parser,order_book_env,replay}.py` | ✅ **PASSING** (34/34 green with Tier 2, 2026-09-04) |
| Diff-test harness | `bindings/tests/test_diff_engine_stub.py` | ✅ **PASSING** — Engine-vs-Stub L2-ladder parity (2/2, 2026-09-04) |
| Python deps | `python_quant/requirements.txt` (numpy, gymnasium) | ✅ |

**Note on the seam:** `ReplayEngine.apply()` needs `book.cancel_id(...)` / `book.lookup(...)`.
`StubBookAdapter` had both, but `EngineAdapter` did not — a gap that would break replay (and
the diff-test) against the real engine. Fixed additively in `book_port.py` (full cancel via
`engine.cancel`, partial via `engine.modify` at the same price to keep time priority). See
the diff-test for the exact ladder-parity assertion (seq/ts/version + trade counters are
excluded by design — engine records prints on matching).

**Build fixes landed 2026-09-04 (found while verifying on Windows):**
- `cpp_engine/bench/bench.cpp` had **unresolved merge-conflict markers** (a botched merge of
  the ``bench <cfg>`` and ``--ops`` CLI variants) → did not compile. Repaired into one
  coherent `bench(cfg, tp_ops, lat_ops)` + `main()` keeping **both** capabilities; added the
  missing `#include <algorithm>`. Compiles clean (-O3) and still proves 0 allocs/op.
- `CMakeLists.txt` now pins `LIBRARY/RUNTIME_OUTPUT_DIRECTORY_<CONFIG>` for `nexus_engine`,
  so the MSVC multi-config generator drops the module directly into `bindings/` (where pytest
  expects it) instead of `bindings/<Config>/`.

**Phase 1d — Person A: Monte-Carlo VaR/CVaR risk engine, subsystem 3 (2026-09-06, branch `feature/risk-engine`).**

| Component | File | State |
|---|---|---|
| Model + deterministic RNG (GBM / jump-diff, splitmix64) | `cuda_risk/risk_common.hpp` | ✅ same RNG on CPU/GPU/NumPy → exact parity |
| CPU reference (serial VaR/CVaR) | `cuda_risk/risk_cpu.hpp` | ✅ header-only, plain C++, tested |
| CUDA kernel (1 thread/path) + launcher | `cuda_risk/risk_cuda.{cu,h}` | ⚠️ authored; compiles only with a toolkit |
| CPU-vs-GPU bench + bit-for-bit parity | `cuda_risk/risk_bench.cpp` | ✅ CPU path runs here; GPU path on a CUDA box |
| pybind `compute_var_cvar` (CPU) | `bindings/pybind_wrapper.cpp` | ✅ |
| NumPy oracle — **exact** parity (not MC-noise) | `python_quant/tests/test_risk_parity.py` | ✅ 3/3 bit-for-bit |
| C++ statistical self-tests | `cpp_engine/tests/risk_test.cpp` | ✅ CTest 5/5 (mean vs 1−e^{μT}, VaR/CVaR monotonicity, jumps fatten tail, determinism) |
| CMake wiring | `CMakeLists.txt` | ✅ `nexus_risk` + `risk_bench` under toolkit; `risk_test` always; `cuda_risk` on pybind include path |

**Design:** the per-path randomness is a pure function of `(seed, path, step)`
via counter-based splitmix64, so the CPU reference, the CUDA kernel, and the
NumPy oracle all draw the **identical** paths — parity is bit-for-bit, not
Monte-Carlo tolerance. The GPU kernel is one thread per path with no shared
state / no atomics; the quantile is reduced on the host.

**Verified here (no GPU):** pytest **49 passed** (incl. 3 exact-parity),
CTest **5/5** (incl. `risk_test`). CPU reference: 200k paths × 252 steps in
~1.0 s (50M path-steps/s) — the baseline the GPU speedup is measured against.
**Blocked:** the CUDA kernel and the ~40× speedup cannot be compiled or
measured on this machine (no `nvcc`/toolkit); needs WSL/Linux or a Windows
CUDA toolkit.

**Phase 1c — Person B: PPO execution agent (2026-09-05).**

| Component | File | State |
|---|---|---|
| Pure-NumPy MLP + Adam | `python_quant/nexus_quant/agents/mlp.py` | ✅ (backward finite-diff-tested) |
| PPO learner (`PPOPolicy` / `train_ppo` / GAE) | `python_quant/nexus_quant/agents/ppo.py` | ✅ deterministic, seeded |
| Policy save/load (`.npz`, no torch) | `python_quant/nexus_quant/agents/ppo.py` | ✅ |
| Eval harness vs baselines | `python_quant/nexus_quant/agents/evaluate.py` | ✅ (`strategy_table`, `format_table`) |
| One-shot train+eval CLI | `python_quant/scripts/train_eval_agent.py` | ✅ |
| Agent tests | `python_quant/tests/test_ppo_agent.py` | ✅ **12/12** (grad, GAE, determinism, save/load) |
| Env price knobs | `order_book_env.py` (`is_coef`, `lambda_sched`, default-off) | ✅ additive; default behavior unchanged |
| High-vol regime (Markov + gap events) | `order_book_env.py` (`regime_prob`/`vol_decay`/`gap_*`/`vol_*` params) | ✅ default-off; **headline achieved** |
| Regime presets | `nexus_quant/__init__.py` (`HIGHVOL_PRESETS["highvol"]`) | ✅ |
| Highvol CLI + eval-only | `scripts/train_eval_agent.py` (`--highvol`, `--vol-feature`, `--eval-only`) | ✅ |
| Regime tests | `python_quant/tests/test_highvol_env.py` | ✅ **11/11** |
| Agent README (measured numbers, honest) | `python_quant/nexus_quant/agents/README.md` | ✅ |

**Measured (default env, 100 seeded episodes, 2026-09-05):** the PPO agent
beats **all** baselines on the env's reward (−5.26 vs VWAP −7.59, TWAP −10.32,
POV −11.19, Passive −13.12) by learning to time fills around adverse mid-moves
while holding a near-TWAP pace. On the pure **shortfall_bps** slippage metric
it lands ≈VWAP (1.60 vs 1.55, within noise); TWAP (1.375) is the price
benchmark. An honest sweep of `is_coef`/λ reductions/schedule-tracking/warm-start
did not beat TWAP on this *gentle-walk* sim — the env's bid-cap fill mechanics
+ fixed TWAP-pace child size cap achievable price; the path to the ~14%-below-VWAP
headline is a **high-vol/gap-off flow regime** (and/or a schedule-constrained
post-at-touch objective).

**Headline achieved (high-vol regime, 2026-09-07):** added a Markov
regime-switching + gap-off flow to `OrderBookEnv` (new constructor params;
defaults preserve the calm behavior byte-for-byte). See `HIGHVOL_PLAN.md`.
Training `--highvol --vol-feature --iters 2000` → PPO shortfall **1.401 bps vs
VWAP 2.827** = **+50.4%** lower slippage (100 seeded episodes); robust across
seeds (+38.2% on a 200-episode re-check). The ~14% resume headline is
comfortably exceeded. Saved policy: `python_quant/artifacts/policy_ppo_highvol.npz`.
Regime tests: `python_quant/tests/test_highvol_env.py` (11 tests, green).

**Phase 1e — Person B: combined interactive desk (subsystem 4/5, 2026-09-09).**
**Phase 1e — Person B: combined interactive desk (subsystem 4/5, 2026-09-09, branch `feature/dashboard-file-ring`).**

| Component | File | State |
|---|---|---|
| Combined desk page (console styling + live L2 overlay) | `python_quant/nexus_quant/dashboard_page.html` | ✅ served at `/` by the dashboard server |
| Slot codec + `SnapshotHub` (decode/dedup history/latency histogram) | `python_quant/nexus_quant/dashboard.py` | ✅ rolling 200-sample history, log-binned latency, p50/p95 |
| Dashboard server (shm-ring / file-ring / seeded synthetic walk) | `python_quant/scripts/serve_dashboard.py` | ✅ synthetic emits real measured render latency + VaR every 8 ticks |
| Dashboard tests | `python_quant/tests/test_dashboard.py` | ✅ **5 pass** (+1 shm skip on Windows); full suite **68 pass / 1 skip** |
| GRPO trainer on PPO actor interface | `python_quant/nexus_quant/agents/grpo.py` | ✅ (merged via PR #7) |
| Risk↔env inventory CVaR penalty | `python_quant/nexus_quant/risk.py` + `order_book_env.py` | ✅ `lambda_risk` param, default 0.0 (merged via PR #7) |
| EngineAdapter keeps book across env reset | `python_quant/nexus_quant/book_port.py` | ✅ (merged via PR #8) |
| Static verification console | `dashboard/index.html` | ✅ separate page on `feature/risk-engine` |

**Verified (2026-09-09):** the combined page serves console sections *and* the live desk
(depth ladder, mid+spread sparklines, latency histogram, VaR/CVaR tiles, trade ticker) in one page,
polling `/api/state` at 400 ms; `seq`/mid/history/latency all tick live against the seeded synthetic
walk. Feeds: POSIX `/dev/shm` ring, a file ring of 448-B slots, or synthetic (no C++ build needed).

**Risk↔env seam (item 12, also done):** `OrderBookEnv` accepts `lambda_risk` (default 0.0);
when > 0 it calls `inventory_risk_penalty()` from `risk.py` (NumPy oracle, exact parity with
the C++ `compute_var_cvar`) to compute CVaR and applies it as a dynamic holding penalty.
This is the Person A ↔ Person B integration seam.

## 7. Next steps (ordered; low-risk foundations first)

1. ~~**`.gitignore`**~~ — ✅ done 2026-08-24.
2. ~~**`bindings/CONTRACT.md`**~~ — ✅ done 2026-08-24 (full spec, offsets verified).
3. ~~**Build system**~~ — ✅ `CMakeLists.txt` + `pyproject.toml` authored 2026-08-30
   (build now possible on Windows with MSVC + pip cmake — §8; Tier 2 pending).
4. ~~**Person A: real `LimitOrderBook`**~~ — ✅ implemented + 86/86 tests, wired into the
   pybind `Engine` (order entry, fills, zero-copy views). Diff-test vs `StubOrderBook`
   pending the WSL build.
5. ~~**Verify the real seam (Tiers 1 + 2)**~~ — ✅ **PASSED 2026-09-04 on Windows/MSVC**.
   `pytest python_quant/tests bindings/tests -v` → **34 passed**: contract smoke, ITCH
   parser, replay, OrderBookEnv, baselines, `test_abi_parity.py` (6), and the Engine-vs-Stub
   diff-test (`test_diff_engine_stub.py`, 2) — the real engine and the oracle agree on the
   L2 ladder. CTest (C++) 4/4. Exact build invocation + CMake module-drop fix in §8/§9.
6. ~~**Person B — ITCH 5.0 parser + Gymnasium `OrderBookEnv`**~~ — ✅ authored + merged
   PR #2 (2026-09-04): `itch_parser.py`, `replay.py`, `book_port.py`, `envs/order_book_env.py`,
   `baselines.py`, and their tests (see §6 Phase 1b). The last sub-piece — the **diff-test
   harness** (`bindings/tests/test_diff_engine_stub.py`, Engine vs `StubOrderBook` oracle) —
   is now **authored**; it skips until `nexus_engine` is built, so run it in Tier 2 (step 5).
7. ~~**Subsystem 5 C++ plumbing**~~ — ✅ 2026-08-30: `ShmRing` (SPSC, drop-new-on-full,
   POSIX+Windows) + `FlowGen` + `ring_producer`/`ring_probe` demos verified live on
   Windows. The Python dashboard (subsystem 4/5) will consume this ring later.
8. ~~**Zero-alloc hot path (make the idle claim TRUE)**~~ — ✅ 2026-08-30: replaced
   `id_map_` (`std::unordered_map`, ~1 malloc/resting order) with `nexus::IdMap`, a
   pre-sized open-addressing linear-probe map with backtrack-shift deletion. The
   benchmark (`cpp_engine/bench/bench.cpp`) now proves **0 allocs/op** on both
   workloads; `lob_test` 86/86 and new `id_map_test` 4,676,294 checks pass; ABI lock
   (448 B) intact. Throughput/latency still to be re-measured on real hardware (the
   Windows sandbox throttles memory workloads — §8).
9. **Subsystem 3 — CUDA risk engine: authored + CPU-validated (branch
   `feature/risk-engine`, 2026-09-06).** The CPU reference, NumPy exact-parity
   oracle, CTest `risk_test`, and pybind `compute_var_cvar` all pass **here**
   (49 pytest, 5/5 CTest). The **GPU kernel** (`risk_cuda.cu`) and the ~40×
   speedup remain **blocked**: no CUDA toolkit on this machine — compile
   `nexus_risk` + `risk_bench` on WSL/Linux or a Windows CUDA toolkit and
   capture the CPU-vs-GPU number.
10. ~~**Person B — high-volatility regime → ~14% below VWAP**~~ — ✅ **ACHIEVED
    2026-09-07** (+50.4% on shortfall vs VWAP; see `HIGHVOL_PLAN.md` + Phase 1c).
11. ~~**Person B — GRPO + Python dashboard on the shmem ring (subsystem 4/5)**~~ — ✅ **DONE
    2026-09-09** as the combined desk: `dashboard_page.html` + `SnapshotHub` + `serve_dashboard.py`
    (see Phase 1e). GRPO trainer also landed. Remaining polish: the static `dashboard/index.html`
    console and the combined desk are on two branches — reconcile at merge; a real C++
    `ring_producer` → browser demo on Windows.
12. ~~**End-to-end integration:** wire the risk engine's `compute_var_cvar` into
    `OrderBookEnv` as a dynamic inventory penalty~~ — ✅ **DONE 2026-09-09** (merged via PR #7):
    `lambda_risk` param in `OrderBookEnv`, `risk.py` with NumPy oracle (exact parity with C++
    `compute_var_cvar`).

## What's actually left (post-plan)

All 12 original plan items are complete. Remaining work is **polish & measurement**:

| What | Who | Blocked? |
|---|---|---|
| CUDA kernel compile + ~40× speedup measurement | Person A | Yes — no `nvcc`/toolkit on this machine |
| Throughput/latency on real hardware (>500k ord/s, sub-µs) | Person A | Yes — Windows sandbox throttles; needs Linux/real box |
| ~~Reconcile two dashboard pages~~ — both now on `main` (combined desk + verification console) | Both | ✅ 2026-09-12 |
| Execution timeline + inventory chart in dashboard | Person B | No |
| ~~Final README.md polish + write-up~~ — README status/headline table rewritten with measured numbers; `docs/RESEARCH.md` full report | Both | ✅ 2026-09-13 |
| **Part 2 quant research layer** (plan_2.md): ~~Phase 0–1 spine~~ ✅ · ~~Phase 2 execution realism~~ ✅ · ~~Phase 3 queue/adverse-selection + RL fairness~~ ✅ · ~~Phase 4 real ITCH tape~~ ✅ · ~~Phase 5 report + `run_all.py`~~ ✅ (2026-09-13, Person B branch `hoplite/kranioi-5b44d8a8`) | Person B | No — merge the PR |
| More real tape days / symbols (each day ≈ 14 min stream via `fetch_itch.py`) | Person B | No |

## 8. Environment reality (IMPORTANT — read before running anything)

This Claude session runs on **Windows 11 + Git Bash / MSYS2** (NOT WSL). The repo
lives on a **OneDrive** path (`C:\Users\pekka\OneDrive\Documents\Finance Project-1`).

**Branch state (2026-09-12):** `main` holds all merged work (PRs #1–#9) AND the reconciled
dashboard — `python_quant/nexus_quant/dashboard_page.html` (combined desk, served at `/`) +
`dashboard/index.html` (static verification console) now both live on `main`. Feature branches
are cleaned up; the stale `pr7-fix` and `.claude/worktrees/` worktrees are gone (`.claude/worktrees/`
is gitignored). The quant research layer (Part 2) is tracked in `plan_2.md` and is in Phase 0→1.

| Tool | Status in this shell |
|---|---|
| `g++` | ✅ `/c/msys64/ucrt64/bin/g++` — C++20 OK (can compile/run C++-only checks) |
| `python`/`python3` | ✅ **real Python 3.12 installed 2026-09-04** (python.org via winget) — runs pure-Python tests |
| `cmake` | ✅ `python -m pip install cmake` (no separate installer needed) |
| `cl` (MSVC Build Tools) | ✅ **installed — Tiers 1+2 both PASS 2026-09-04** (VS "18" BuildTools; cmake generator `"Visual Studio 18 2026"` -A x64) |
| `nvcc` / CUDA | ❌ not found |
| `wsl` | ❌ binary present but **no distro installed** — not needed for Python on Windows |

**Consequences (as of 2026-09-09):**
- ✅ **Tier 1 (pure-Python) PASSED:** `python -m pytest python_quant/tests -v` → **68 green, 1 skip**
  (contract smoke, ITCH parser, replay, OrderBookEnv, baselines, PPO/GRPO agents, risk parity,
  high-vol regime, dashboard codec/hub).
- ✅ **Tier 2 (compile `nexus_engine` + parity + diff-test) PASSED ON WINDOWS.**
  Build with MSVC: `python -m pip install pybind11 cmake`, then
  `cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DNEXUS_BUILD_PYBIND=ON
  -DPython3_EXECUTABLE=<abs python.exe> -Dpybind11_DIR=<abs …/pybind11/share/cmake/pybind11>`,
  then `cmake --build build --config Release -j`. Then
  `python -m pytest python_quant/tests bindings/tests -v` → **34+ passed** including
  `test_abi_parity.py` (6) and `test_diff_engine_stub.py` (2). CTest (C++) 5/5.
  The module now drops straight into `bindings/` (CMakeLists pins `_<CONFIG>` output dirs
  for multi-config generators — `bindings/` root is what pytest's `pythonpath` sees).
- ⚠️ **Windows build notes:** `bindings/` is on OneDrive — the build works but is slow;
  if it misbehaves copy the repo off OneDrive. An unbuilt/import-only requirement: the
  compiled `*.pyd` and `*.pdb` are gitignored build artifacts, never committed.
- The project plan assumed WSL/Ubuntu + NVIDIA GPU for CUDA (Dell G15); **CUDA work still
  needs WSL/Linux or a Windows CUDA toolkit** — Python + pybind no longer require it.
- **OneDrive caveat:** keep `build/`, `data/`, venvs out of the synced tree (or move the
  repo off OneDrive) — OneDrive sync + build artifacts is a known source of breakage.
  Pure-Python runs are fine on OneDrive; if the CMake build is slow/flaky, copy the repo
  off OneDrive (e.g. `C:\Users\pekka\dev\finance-project`) and build there.
- **Git behavior on this machine (don't get fooled):** an auto-checkpoint commits
  working-tree changes to `main` as commits titled **"Working Tree Changes"** — so after
  editing files, `git status` may legitimately read *clean* because they're already
  committed (not lost). Given the intended feature-branch → PR workflow, you may want to
  `git reset --soft` those and re-commit deliberately. `core.fsmonitor` was set to
  `false` (was `true`) during debugging — harmless; revert with
  `git config core.fsmonitor true` if desired.

## 9. How to verify current work

```bash
# C++ contract (works in this shell with MSYS2 g++):
g++ -std=c++20 -O2 -Wall -Wextra -I cpp_engine/include \
    cpp_engine/tests/abi_check.cpp -o abi_check.exe && ./abi_check.exe

# Tier 1 — pure-Python (68 tests; PASSED 2026-09-09):
python -m pip install numpy gymnasium pytest
python -m pytest python_quant/tests -v

# Tier 2 — build + parity + diff-test (PASSED 2026-09-04 on Windows/MSVC). This exact
#       invocation builds `nexus_engine` and drops it into bindings/:
python -m pip install pybind11 cmake
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DNEXUS_BUILD_PYBIND=ON \
      -DPython3_EXECUTABLE=C:/Users/pekka/AppData/Local/Programs/Python/Python312/python.exe \
      -Dpybind11_DIR=C:/Users/pekka/AppData/Local/Programs/Python/Python312/Lib/site-packages/pybind11/share/cmake/pybind11
cmake --build build --config Release -j
python -m pytest python_quant/tests bindings/tests/test_abi_parity.py bindings/tests/test_diff_engine_stub.py -v

# Dashboard (subsystem 4/5 — no C++ build needed):
python python_quant/scripts/serve_dashboard.py --synthetic   # → http://127.0.0.1:8765

# C++ engine checks (any box with g++/MSVC):
g++ -std=c++20 -O2 -Wall -Wextra -I cpp_engine/include \
    cpp_engine/tests/lob_test.cpp -o lob_test.exe && ./lob_test.exe   # 86 checks, ALL PASS

# High-vol PPO training + eval:
PYTHONPATH=python_quant python python_quant/scripts/train_eval_agent.py \
    --highvol --vol-feature --iters 2000 --eval-every 400 --eval-episodes 40 \
    --out python_quant/artifacts/policy_ppo_highvol.npz --table-episodes 100

# Part 2 research layer (Person B) — real NASDAQ ITCH day + fair RL re-verification:
python python_quant/scripts/run_all.py --quick          # smoke of every stage (minutes)
python python_quant/scripts/fetch_itch.py --day 12302019 --symbols AAPL,QQQ    # ~14 min, data/ gitignored
python python_quant/scripts/run_research.py --day 12302019 --symbols AAPL,QQQ  # E1–E6 → docs/results/
python python_quant/scripts/rl_fairness_study.py                               # E7 → docs/results/rl_fairness.md
```
