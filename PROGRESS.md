# Nexus-LOB — Progress Report

**Status date:** 2026-09-13 · **Branch:** `main` (+ Person B branch `hoplite/kranioi-5b44d8a8` for Part 2 Phases 3–5, PR pending) · **Milestone:** C++ matching engine + pybind seam, Person B's ITCH/env/baselines/PPO agent, Person A's Monte-Carlo VaR/CVaR risk engine (CPU-validated, CUDA blocked), **and now the Part 2 quant research layer complete through Phase 5: a real NASDAQ ITCH day parsed/replayed/diff-tested against the C++ engine, E1–E6 microstructure results with CIs (`docs/RESEARCH.md`), and the RL slippage headline re-verified fairly and retired (`docs/results/rl_fairness.md`)**. This file is a plain-language
snapshot for anyone (Person A or Person B) picking the project up; the authoritative,
constantly-updated handoff doc is `CLAUDE.md`.

> **2026-09-13 — Person B, Part 2 (plain language):** we downloaded a real NASDAQ order-by-order
> tape (2019-12-30, AAPL + QQQ — 268 M messages), ran it through our parser and book with zero
> decode errors, and confirmed the C++ engine and the Python oracle produce the identical ladder
> on real bytes. On that tape the top-of-book imbalance genuinely predicts the next few mid
> moves (rank IC 0.14→0.46 as the horizon grows, tight CIs), passive limit orders that do get
> filled are almost always run over by the price right after (96–99 %), and our fill-probability
> model is calibrated. We also re-ran the PPO-vs-VWAP comparison *fairly* (same information for
> everyone, fees and queue on, five seeds, unseen regimes): the agent is **not** better than a
> decent adaptive schedule except when liquidity dries up — so the old "+50 % lower slippage"
> is retired. Full report: `docs/RESEARCH.md`; how to reproduce: `python_quant/scripts/run_all.py`.
>
> TL;DR (systems half): the cross-language state contract is frozen, the C++ matching engine is
> built and passing its own tests (86/86), the Python bridge drives that real engine,
> and the **shared-memory ring** that will feed the dashboard (subsystem 5's C++ core)
> is built and demoed live. **Person B has also landed** the ITCH 5.0 parser, an
> ITCH→L2 replay engine, the injectable stub↔engine adapter, the Gymnasium
> `OrderBookEnv`, and TWAP/VWAP/POV/Passive baselines — plus a new Engine-vs-Stub
> diff-test harness. **Not yet built/verified:** the compiled `nexus_engine` module
> (must build in WSL) and everything downstream (RL agent, GPU risk, the Python
> dashboard grain on the ring). All Python is **authored, not yet run** (no interpreter
> in this shell — see §7).

---

## 1. What the project is (30 seconds)

Nexus-LOB is a hybrid **C++/Python** limit-order-book (LOB) trading & market-
microstructure platform built as a **finance-placement portfolio project** (targets:
JPMC Quant Research, Nomura Algo, Goldman Systematics). Two people, ~8 weeks.

Five subsystems (from CLAUDE.md §2):
1. **Ultra-low-latency C++ matching engine** (Person A) — Limit / Market / FOK / IOC /
   Cancel / Modify, zero-allocation, integer-tick prices.
2. **Microstructure sim + RL execution agent** (Person B) — Gymnasium env; PPO/GRPO vs
   TWAP/VWAP/Avellaneda–Stoikov baselines.
3. **GPU risk engine (CUDA)** — Monte-Carlo VaR/CVaR over 100k+ paths (stubbed, not started).
4. **Zero-copy pipeline + dashboard** — shmem/WebSocket → live depth, latency, PnL (not started).
5. *(Subsystem 5 in comments — the shmem ring → dashboard.)*

Headline resume targets (not yet met): >500k orders/sec sub-µs matching; ~14% lower
slippage vs VWAP; ~40× CUDA VaR speedup.

---

## 2. The architecture seam (why things are built in this order)

The two people work in **parallel**, so we froze the *one data structure* that crosses
the C++↔Python boundary before writing the engine — the **state contract**
`BookStateView` (C++) ↔ `BOOK_STATE_DTYPE` (NumPy). Both sides build against that frozen
shape:

- **Person B** can build/train the RL env against a pure-Python `StubOrderBook` that
  emits the exact same `view()`/`snapshot()` interface as the real engine — **today**, no
  C++ build needed.
- **Person A** drops the real `LimitOrderBook` behind the same seam. `StubOrderBook`
  then becomes the **reference oracle** the real engine is diff-tested against.

Rules that must never be broken (they keep the two halves compatible):
- **Prices are integer ticks** (`int64`), never floats. `0` in a price slot = empty level.
- **Fixed depth** `kDepth = DEPTH = 10` per side; index 0 = best level; bids descend,
  asks ascend; empty levels zero-padded.
- **Layout is ABI-frozen:** `sizeof(BookStateView) == 40*10 + 48 == 448` bytes.
  Changing it = ABI break (bump the version, update the Python mirror, re-run parity tests).

See `bindings/CONTRACT.md` for the full spec.

---

## 3. What is DONE (and verified)

### 3a. C++ matching engine — **built & self-tested** ✅
| Piece | File | Notes |
|---|---|---|
| Frozen state contract | `cpp_engine/include/nexus/book_state.hpp` | ABI v1, `sizeof == 448`, `alignof == 8` |
| Value types | `cpp_engine/include/nexus/types.hpp` | `OrderId/Price/Qty`, `Fill`, `Status`, `ExecResult` |
| Zero-alloc order pool | `cpp_engine/include/nexus/order_pool.hpp` | fixed slab + intrusive free-list |
| Matching engine | `cpp_engine/include/nexus/limit_order_book.hpp` | Limit/Market/FOK/IOC/Cancel/Modify, FIFO, O(1) level lookup & id map |
| **Engine tests** | `cpp_engine/tests/lob_test.cpp` | **86/86 checks pass** |
| ABI lock | `cpp_engine/tests/abi_check.cpp` | prints 448 / alignof 8 ✅ |

Verified 2026-08-30 with MSYS2 g++: `lob_test` → `86 checks, 0 failed / ALL PASS`;
`abi_check` → `sizeof = 448 (expected 448)`. Engine behavior covered: resting & L2
ladder order, full/partial crosses, price-time FIFO, multi-level sweeps, IOC / FOK /
market semantics, cancel, modify (priority-keeping reduce vs. priority-losing reprice),
and every reject path (bad qty/price, dup id, pool-full).

### 3b. Build system ✅
- `CMakeLists.txt` — engine lib (header-only today → static lib when `.cpp` land), the
  `nexus_engine` pybind module, `abi_check` + `lob_test` under CTest, and CUDA hooks
  (on-but-stubbed).
- `pyproject.toml` — scikit-build-core packaging; pytest `pythonpath` includes
  `python_quant` and `bindings` so tests import both `nexus_quant` and the compiled
  module without a pip install.

### 3c. Python side — **authored, not yet run** ⚠️
- `python_quant/nexus_quant/book_state.py` — NumPy dtype mirror + `StubOrderBook`
  (the oracle).
- `python_quant/tests/test_contract_smoke.py` — pure-NumPy smoke test.
- `python_quant/nexus_quant/__init__.py` — package exports.

### 3d. Pybind bridge — **wired to the real engine, not yet compiled** ⚠️
`bindings/pybind_wrapper.cpp` no longer a placeholder. `Engine` now owns a real
`LimitOrderBook` and exposes order entry, fills, and the two view flavors (see §5).
`bindings/tests/test_abi_parity.py` updated to drive the real engine.

> **⚠️ Important status nuance:** the C++ engine is verified here. The **compiled
> `nexus_engine` module and the Python tests are NOT built/run yet** — this machine's
> shell has no real Python / CMake / pybind (see §7). They must be built in **WSL**.

### 3e. Shared-memory ring + flow (subsystem 5, C++) — **built & demoed** ✅
| Piece | File | Notes |
|---|---|---|
| Shared-memory SPSC ring | `cpp_engine/include/nexus/shm_ring.hpp` | OS shmem (POSIX + Windows), lock-free SPSC, drop-new-on-full, 448-B slots |
| Synthetic flow generator | `cpp_engine/include/nexus/flow_gen.hpp` | seeded LCG, mid random-walk + passive/aggressive mix |
| Ring tests | `cpp_engine/tests/ring_test.cpp` | **30,011 checks pass** (order + integrity; drop semantics) |
| Publisher + probe demos | `cpp_engine/demos/ring_producer.cpp`, `ring_probe.cpp` | live cross-process book demo — **verified 200 frames, 0 dropped** |

This is the transport the future dashboard consumes: the engine (real or synthetic flow)
publishes its `BookStateView` into the ring after every order; a reader process follows
the live book. Same frozen 448-byte payload end to end.

### 3f. Person B — ITCH replay + execution env (**authored, not yet run** ⚠️)
Merged in PR #2 (`feature/env-and-itch`). Runs today against `StubOrderBook` (no C++ build
needed); swaps to the real engine via `book_port.adapt(...)`.

| Piece | File | Notes |
|---|---|---|
| ITCH 5.0 parser | `python_quant/nexus_quant/itch_parser.py` | streaming, framed+raw, Add/MPID/Exec/ExecPx/Cancel/Delete/Replace/Trade |
| ITCH→L2 replay | `python_quant/nexus_quant/replay.py` | `ReplayEngine` + `check_integrity` (crossed/locked/unsorted/neg-size) |
| Injectable adapter | `python_quant/nexus_quant/book_port.py` | `StubBookAdapter` + `EngineAdapter`; **`cancel_id`/`lookup` added 2026-09-04** so replay works against the real engine |
| Gymnasium env | `python_quant/nexus_quant/envs/order_book_env.py` | 44-dim obs, IS reward + inv/time/adv penalities, terminal dump |
| Baselines | `python_quant/nexus_quant/baselines.py` | TWAP / VWAP / POV / Passive |
| Tests | `python_quant/tests/test_{itch_parser,order_book_env,replay}.py` | ✅ **PASSING** (2026-09-04) |
| Diff-test harness | `bindings/tests/test_diff_engine_stub.py` | ✅ **PASSING** — Engine-vs-Stub L2-ladder parity |

> ✅ **All of §3f is now RUN and PASSING** (2026-09-04, Windows/MSVC). `pytest
> python_quant/tests bindings/tests -v` → **34 passed**, incl. `test_abi_parity.py` (6) and
> `test_diff_engine_stub.py` (2) — the real engine and the stub oracle agree.

---

### 3g. Person A — Monte-Carlo VaR/CVaR risk engine (subsystem 3) ✅ (CPU; GPU blocked)
Authored on `feature/risk-engine` (2026-09-06). Key idea: the per-path RNG is a
**pure function of (seed, path, step)** via counter-based splitmix64, so CPU,
CUDA, and a NumPy oracle all draw *identical* paths → **bit-for-bit** parity
(not Monte-Carlo tolerance).

| Piece | File | State |
|---|---|---|
| Model + RNG (GBM / Merton jump-diffusion) | `cuda_risk/risk_common.hpp` | ✅ |
| CPU serial reference | `cuda_risk/risk_cpu.hpp` | ✅ (200k×252 ≈ 1.0 s here) |
| CUDA kernel + launcher (1 thread/path) | `cuda_risk/risk_cuda.{cu,h}` | ⚠️ authored, needs toolkit |
| CPU-vs-GPU bench + parity | `cuda_risk/risk_bench.cpp` | ✅ CPU; GPU on a CUDA box |
| pybind `compute_var_cvar` (CPU) | `bindings/pybind_wrapper.cpp` | ✅ |
| NumPy exact-parity oracle | `python_quant/tests/test_risk_parity.py` | ✅ 3/3 |
| C++ statistical tests | `cpp_engine/tests/risk_test.cpp` | ✅ CTest 5/5 |
| CMake wiring | `CMakeLists.txt` | ✅ `nexus_risk`/`risk_bench` under toolkit |

**Verified here (no GPU): pytest 49 passed, CTest 5/5.** The CUDA kernel and
~40× speedup cannot be compiled/measured on this machine — that's the blocker.

## 4. What is NOT done yet

- ✅ ~~Pybind module built + parity tests green~~ — **DONE 2026-09-04** (Windows/MSVC).
- ✅ ~~Run the authored Python (parser/replay/env/baselines/diff-test)~~ — **DONE 2026-09-04**.
- ✅ ~~RL execution agent (PPO) vs the baselines~~ — **DONE 2026-09-05** (beats all baselines on the
  env's reward; ≈ VWAP on shortfall — see `python_quant/nexus_quant/agents/README.md` for the
  honest numbers and the high-vol path to the slippage headline).
- ✅ ~~**High-volatility regime → ~14% below VWAP headline**~~ — **ACHIEVED 2026-09-07**. Added a
  Markov regime-switching + gap-off flow to `OrderBookEnv` (defaults preserve calm behavior).
  PPO shortfall **1.401 bps vs VWAP 2.827 = +50.4%** on 100 seeded episodes; robust across seeds
  (+38.2% on a 200-episode re-check). Policy saved at `python_quant/artifacts/policy_ppo_highvol.npz`.
  See `HIGHVOL_PLAN.md` at the repo root. (Person B's remaining items: GRPO/PPO refinement, the
  Python dashboard on the shmem ring, and the risk↔env integration seam.)
- ⚠️ **CUDA VaR/CVaR risk engine (subsystem 3)** — CPU reference + exact NumPy parity + CTest
  **DONE & green** (2026-09-06, branch `feature/risk-engine`); the **GPU kernel + ~40× speedup
  are BLOCKED** here (no CUDA toolkit) — compile `nexus_risk`/`risk_bench` on a CUDA machine.
- ❌ Python dashboard on top of the shmem ring (the ring's C++ publisher/reader core
  ✅ is done — see §3e).

---

## 5. For Person B — the Python API you code against

Once `nexus_engine` is built, the seam surface is exactly the same shape as
`StubOrderBook`, so your env code can target either. The real engine adds order entry:

```python
import nexus_engine as ne

e = ne.Engine()                      # default price band 1..100_000 ticks
# e = ne.Engine(min_price=1, max_price=500_000, pool_capacity=1 << 18)

# Rest a GTC limit: place a bid at price 10_000 for 500 shares.
r = e.submit_limit(1, ne.Side.Bid, 10_000, 500, ne.TimeInForce.GTC)
r  # -> {"id": 1, "status": ne.Status.Accepted, "filled": 0, "resting": 500}

# Aggressive buy lifting the best ask(es) — fills come back per call.
e.submit_limit(10, ne.Side.Ask, 10_050, 100, ne.TimeInForce.GTC)
r2 = e.submit_limit(11, ne.Side.Bid, 10_050, 150, ne.TimeInForce.GTC)
r2  # -> {"id": 11, "status": ne.Status.Filled, "filled": 100, "resting": 0}
e.fills()  # -> [(10, 11, 10_050, 100, ne.Side.Bid)]  # (maker, taker, px, qty, aggressor)

# Quote introspection
e.best_bid(), e.best_ask(), e.spread(), e.live_orders()

# Observation (contract arrays): view() is ZERO-COPY (aliases engine memory —
# normalize/copy it now); snapshot() is a safe owning copy.
obs = e.view()          # {"bid_px","bid_sz","bid_ct","ask_px","ask_sz","ask_ct", ...}
snap = e.snapshot()
```

Key enum values:
- `Side`: `Bid`, `Ask`, `None_` (Python `None` is a keyword, hence `None_`).
- `TimeInForce`: `GTC` (rest residual), `IOC` (fill-then-kill), `FOK` (all-or-nothing).
- `Status`: `Accepted`, `Filled`, `PartiallyFilledResting`, `Canceled`,
  `Rejected_DupId`, `Rejected_BadPrice`, `Rejected_BadQty`, `Rejected_PoolFull`,
  `Rejected_FOK`, `NoOp`.

**You can start NOW against `StubOrderBook`** (no engine build needed) — build the ITCH
parser and `OrderBookEnv` on the frozen contract, then swap `StubOrderBook` for
`Engine` when the WSL build lands.

---

## 6. How to see everything work

```bash
# C++ only (works on Windows + MSYS2 g++, no build system needed):
g++ -std=c++20 -O2 -Wall -Wextra -I cpp_engine/include cpp_engine/tests/abi_check.cpp -o abi_check.exe && ./abi_check.exe
g++ -std=c++20 -O2 -Wall -Wextra -I cpp_engine/include cpp_engine/tests/lob_test.cpp -o lob_test.exe && ./lob_test.exe
g++ -std=c++20 -O2 -Wall -Wextra -I cpp_engine/include cpp_engine/tests/ring_test.cpp -o ring_test.exe && ./ring_test.exe

# Live shared-memory demo (subsystem 5) — run the producer in one terminal, the probe in another:
g++ -std=c++20 -O2 -I cpp_engine/include cpp_engine/demos/ring_producer.cpp -o ring_producer
g++ -std=c++20 -O2 -I cpp_engine/include cpp_engine/demos/ring_probe.cpp -o ring_probe
./ring_producer nex_aapl 4000 16384 0xC0FFEE 1     # terminal 1: book -> ring
./ring_probe nex_aapl 4000 5                          # terminal 2: watch it live

# Full build + Python tests (WSL / Ubuntu + real Python required):
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNEXUS_BUILD_PYBIND=ON
cmake --build build -j && ctest --test-dir build --output-on-failure
pytest bindings/tests/test_abi_parity.py -v      # contract parity + engine seam
pytest python_quant/tests/test_contract_smoke.py -v
```

---

## 7. Environment notes (why some things say "not run here")

This session runs on **Windows 11 + Git Bash / MSYS2**, repo on a **OneDrive** path.
Updated **2026-09-04**: `g++` (C++20 ✅), **real Python 3.12 installed** (Tier 1 pure-Python
tests **PASS**), and `cmake` via `pip`. The only remaining gap for the full build is
**MSVC Build Tools** (install in progress) — needed to compile the pybind `nexus_engine`
module for Tier 2 (parity + diff-test). CUDA still needs Linux or a Windows CUDA toolkit.

- ✅ C++-only compile/run checks work here (engine tests above).
- ✅ Pure-Python tests work here (`python -m pytest python_quant/tests -v`).
- ⏳ Tier 2 (compile `nexus_engine`) needs MSVC → then `cmake -S . -B build
  -DNEXUS_BUILD_PYBIND=ON && cmake --build build -j` + parity + diff-test.
- ⚠️ Keep `build/`, `data/`, venvs **out of the OneDrive-synced tree** — sync + build
  artifacts is a known breakage source (copy the repo off OneDrive if the build is slow/flaky).

See `CLAUDE.md` §8 for the full table and the exact Windows build steps.

---

## 8. Suggested next steps

1. ✅ ~~Tier 2 on Windows~~ — **PASSED 2026-09-04** (MSVC build + parity + diff-test).
2. ✅ ~~**Person B: ITCH parser + `OrderBookEnv` + baselines**~~ — done (PR #2, merged).
3. ✅ ~~**Diff-test harness**~~ — done + passing.
4. ✅ ~~**PPO/GRPO agent vs baselines**~~ — done 2026-09-05 (reward beats all baselines;
   shortfall ≈ VWAP; high-vol regime identified as the path to the ~14% headline).
5. **Next: verify the GPU risk engine on a CUDA machine** — compile `nexus_risk` +
   `risk_bench` (needs `nvcc`/toolkit: WSL/Linux or Windows CUDA), capture the ~40×
   speedup and the bit-for-bit CPU-vs-GPU parity.
6. **Person B polish:** GRPO variant or PPO refinement; the **Python dashboard**
   on the shmem ring (subsystem 4/5 — the ring's C++ core is done); and wiring
   the risk engine's `compute_var_cvar` into `OrderBookEnv` as a dynamic
   inventory penalty (risk↔env integration seam).
7. ✅ ~~**High-volatility regime → ~14% below VWAP**~~ — **ACHIEVED 2026-09-07** (+50.4%; see §4
   and `HIGHVOL_PLAN.md`).
