# Nexus-LOB — Part 2: Quant Research Layer (Plan of Record)

**Last updated:** 2026-09-13 (final) · **Status:** Phases 0–2 on `main` (PRs #15, #16); **Phases 3, 4 and 5 landed on the Person B branch `hoplite/kranioi-5b44d8a8` (PR pending)** — Phase 3 (queue dynamics E5/E6 + RL fairness rework, §6 — DONE, headline re-characterized), Phase 4 (real NASDAQ ITCH tape: `fetch_itch.py` + `run_research.py`, E1–E6 on 12/30/2019 AAPL/QQQ), Phase 5 (`docs/RESEARCH.md` full report incl. negative results, `scripts/run_all.py` one-command reproduce, honest README rewrite). Research half **complete**; remaining are more tape days and Person A's GPU/hardware items. Person B work-package for Phases 2–4: `docs/work_package_b_phases_2_4.md`.
**Read this FIRST each session, then `CLAUDE.md`.** This is the single source of truth for the research-half roadmap. Keep updating it as work progresses (§ Running log at the bottom).

---

## 0. Why this file exists

The project is now cleanly split in two:

- **Part 1 (DONE, verified):** C++ matching engine, frozen 448-byte ABI contract, pybind seam, shmem ring, ITCH parser + replay, `OrderBookEnv`, PPO/GRPO, risk engine (CPU + exact parity). All 12 original plan items complete.
- **Part 2 (NOW, this file):** the **quant research layer** — turning Nexus-LOB from "systems project with a toy RL agent" into *"systems + real market-microstructure research."* This matches what Jane Street / Citadel / Optiver / DRW interviewers actually evaluate.

The full 16-part audit (repository audit → design → experiment → rigor → execution → queue → RL → C++ perf → CUDA → architecture → file plan → prioritization → 6-week roadmap → resume → definition of done) was delivered in the 2026-09-12 session. **This file is the distilled plan of record derived from it.**

### Ground rules (a violation of any = progressive failure)
1. **No feature is used before its leakage is proven absent.** Features are pure functions of state at `t`; labels are strictly `t → t+h`; splits are walk-forward by wall-clock time. A test enforces it.
2. **No headline tuned to a target.** The Part-1 "+50.4% vs VWAP" was produced by tuning the sim until it hit the target. We do not extend that pattern — we fix it (§6).
3. **Every number gets a CI.** IC, ICIR, IS, slippage: block-bootstrap / multiple seeds, t-stats, Newey–West where horizons overlap.
4. **Preserve the architecture.** Frozen `BookStateView` contract, `StubOrderBook`↔engine parity oracle, deterministic PPO, `cuda_risk` bit-for-bit parity design — all stay as-is.
5. **Real data is the end goal.** Synthetic flow establishes the harness; research *claims* are made on a real NASDAQ ITCH tape.
6. **Negative results are reported.** A project with zero negative results is research theater.

---

## 1. Audit snapshot — where the repo genuinely stands (2026-09-12)

| Side | Verdict |
|---|---|
| C++ LOB / matching / ABI / pybind / shmem | Genuinely strong, well-tested, production-shaped. **8/10 systems.** |
| ITCH parser / replay | Correct but **never exercised on real bytes** (synthetic fixtures only). |
| Python research layer | **Essentially absent** — no features, no signals, no IC, no backtest, synthetic-only price process. |
| RL/PPO | Engineering solid (deterministic, seedable). **Research validity weak** — see §6. |
| VaR/CVaR | Correct, exactly-parity. Parametric-MC only; CUDA never compiled. |
| Dashboard / branches / CI / docs | **Broken state** — see §1a. |
| **Overall quant signal** | **≈3.5/10.** Systems ≥ interview-ready; research half ≈ six weeks away. |

### 1a. Repo-state problems that must be fixed at the start (Phase 0)
- Working tree is on **stale `feature/risk-engine`** (missing `grpo.py`, `risk.py`, `dashboard.py`, `serve_dashboard.py`); GitHub default **`main`** has them. `dashboard_page.html` is stranded on **`feature/dashboard-file-ring`**; `dashboard/index.html` only exists on **`feature/risk-engine`**.
- Local origin still points at **`Finance-Project-1`** (repo was renamed to **`Nexus_LOB`** — GitHub redirects the URL, but update it: `git remote set-url origin https://github.com/Lokeshrao69/Nexus_LOB.git`).
- **No CI** (no `.github/` anywhere).
- `README.md` contains **two concatenated full READMEs** and describes the wrong branch state.
- `%TEMP%\pr7-fix` worktree is leftover; `feature/person-b-polish` is fully merged into `main`.

> **Decision pending:** reconcile so `main` == docs == checkout, then build the research layer on `main`. (Recommend: `git checkout main`, drop the stale `feature/risk-engine` / `feature/dashboard-file-ring` unmerged dashboard commits into one reconciled dashboard or drop them; keep `policy_ppo_highvol.npz` artifact.)

### 1b. The one credibility issue fixed before anything else
The **"+50.4% lower slippage than VWAP"** headline. Ground truth from the code:
- PPO was trained with an explicit `vol_feature` **regime indicator** (`obs[44]`) that TWAP/VWAP/POV/Passive **could not observe** → asymmetric information in the comparison.
- Eval ran on the **training distribution** (same env config + seed family), no holdout regimes.
- **No CI / no t-stat** on the vs-VWAP% (two seed checks: +50.4%, +38.2%).
- The high-vol regime was **tuned until the metric hit its target** (`HIGHVOL_PLAN.md §9` lists "tuning levers if it doesn't hit 14%").

We do **not** delete it, and we **do not** claim it. We **re-characterize and fairly re-verify** it (§6). Until then, no resume/README uses the number.

---

## 2. Target architecture (research half)

```
REAL MARKET DATA ─► ITCH/L2 PARSER (reuse itch_parser.py, validate on real bytes)
     ▼
EVENT REPLAY (reuse replay.py) ─► L2 / ORDER-LEVEL BOOK     (stub today; C++ engine = diff oracle)
     ▼
FEATURE ENGINE (NEW research/features.py)          [imbalance, microprice, OFI, deep, spread, vol, flow, momentum]
     ▼
DATASET + LABELS (NEW research/dataset.py)          [event frames, forward labels, leak-free walk-forward split]
     ▼
EXPERIMENTS (NEW research/experiments.py)           [IC/ICIR/hit-rate, bootstraps, E1–E4]
     ▼
EXECUTION SIM (reuse envs/order_book_env.py; add cost+queue knobs)
     ▼
COST/IMPACT/QUEUE (NEW execution/cost_model.py + research/queue_dynamics.py)
     ▼
BACKTEST + METRICS (NEW execution/backtest.py)      [IS, market-VWAP slip, fill %, completion, MDD]
     ▼
RISK (reuse risk.py/cuda_risk; add historical + stress VaR)
     ▼
REPORT (NEW research/report.py + docs/RESEARCH.md)
```

---

## 3. Build order — phases

**Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5.** Each phase has CREATE/MODIFY/TEST/BENCH/DOC entries below.

### Phase 0 — Repo hygiene (day 1, few hours)
- **FIX** branch state so checkout == `main` == docs (§1a decision).
- **CREATE** `.github/workflows/ci.yml` — on PR: CMake build + CTest (Linux, Windows), `pytest python_quant/tests bindings/tests`, lint job. (Zero CI today.)
- **MODIFY** `README.md` — remove the duplicated second copy; status section updated to match `main`.
- **CREATE** `requirements-dev.txt` — pinned `pytest`, `mypy`, `ruff` (runtime deps stay in `python_quant/requirements.txt`).
- **MODIFY** `python_quant/requirements.txt` — pin versions (`numpy>=1.24`, `gymnasium>=0.29` today are deliberately loose — fine, but add nothing heavy).

### Phase 1 — Research spine (P0, weeks 1–2)
- **CREATE `python_quant/nexus_quant/research/__init__.py`** — export the subpackage; wire into `nexus_quant/__init__.py`.
- **CREATE `python_quant/nexus_quant/research/features.py`** — pure, stateless functions over a `View` (dict) or order-level events:
  - `lob_imbalance(view) -> float`  = (V_b − V_a)/(V_b + V_a)
  - `microprice(view) -> float`     = (V_a·P_b + V_b·P_a)/(V_a + V_b)
  - `ofi(prev_events, cur) -> float` (needs order-level deltas — build approximation from ladder first, upgrade when order-level tracker lands in Phase 3/4)
  - `deep_imbalance(view, k=5|10) -> float` (weighted multi-level)
  - `spread_bps(view) -> float`
  - `realized_vol(mid_hist, n) -> float`
  - `momentum(mid_hist, k) -> float`
  - `flow_intensity(events, tau) -> float`
  - **Contract:** no instance state; a test mutes the book after the call and asserts the feature is unchanged (the leakage lock).
- **CREATE `python_quant/nexus_quant/research/labels.py`** — `forward_return(state_t, future_state, h)`, `forward_mid_move(...)`; strictly uses prices at `t+h` or later, never `t`.
- **CREATE `python_quant/nexus_quant/research/dataset.py`** —
  - `event_frame(events, book) -> rows (feature, label, ts, split_tag)`
  - `make_split(rows, mode="walk_forward", train=0.6, val=0.2)` — time-ordered disjoint blocks (k-fold **walk-forward**, not shuffled CV).
- **CREATE `python_quant/nexus_quant/research/experiments.py`** —
  - `rank_ic(y_true, y_pred)`, `icir(ic_series)`, `hit_rate`, `decile_spread`
  - `bootstrap_ci(values, n_boot=2000, kind="block", block=…)`
  - `diebold_mariano(a, b)` for pairwise feature comparisons
  - `run_experiment(feature_fn, label_h, split) -> result` (JSON-serializable)
- **CREATE `python_quant/nexus_quant/research/models.py`** — OLS on z-scored features only (IC benchmark). **No trees/boosters/neural nets until the linear baseline stands.**
- **TEST `python_quant/tests/test_research.py`** — (a) features don't read future state (mutation-after-call); (b) labels are strictly forward; (c) walk-forward blocks are disjoint in time; (d) IC≈1 on a hand-built predictable series; (e) IC≈0 (within CI) on white noise; (f) bootstrap CI covers the true IC on a known distribution.
- **DOC `docs/RESEARCH.md`** — skeleton: scope, methods section, template tables.

### Phase 2 — Execution realism (P0/P1, weeks 2–3) — **LANDED 2026-09-13** on `feature/part2-phase2-cost-queue`
Exact shipped API: **`docs/work_package_b_phases_2_4.md §2.2`** (frozen). Signatures finalized there; two meaningful deltas from the draft below: `run_backtest` is **env-factory based** (the env IS the tape + book) and reports **pre-cost** slippage — costs fold into the env's own `reward`; and the env fee/rebate is **maker/taker-aware** (`is_ticks` is a cost, so fees add, rebates subtract — the draft had the signs inverted).
- **CREATE `python_quant/nexus_quant/execution/cost_model.py`** — `CostParams(fee_bps, rebate_bps, spread_ecn, impact_coef, impact_mode="sqrt")`; `net_pnl(fills, *, side_cost=None, gross)` (net = gross − fee_ticks + rebate_ticks), `impact(qty_participation, *, sigma, coef, mode)` (square-root law). Costs validate: maker rebate sign, taker fee sign, impact monotone in participation.
- **CREATE `python_quant/nexus_quant/execution/metrics.py`** — `implementation_shortfall`, `vwap_slippage` (**vs MARKET VWAP, not self-executed VWAP** — the Part-1 flaw fix), `arrival_slippage`, `fill_rate`, `completion_rate`, `inv_risk(sigma, inv, T)`, `max_drawdown(pnl_path)`. **All slippage reads positive = a cost** (same sign as env `shortfall_bps`).
- **CREATE `python_quant/nexus_quant/execution/backtest.py`** — `run_backtest(env_factories, *, policy, n_episodes, seed, name)` + `summarize(rows)`; per-regime breakdown; E7 harness skeleton.
- **MODIFY `envs/order_book_env.py`** — accept `fee_bps`, `rebate_bps`, `impact_coef`, `impact_participation` (0.1), `queue_model` (`""`/`"uniform"`, validated; default `""` = today's mechanics) — **byte-identical to today** when defaulted. Add **`market_vwap` to `info`** in every `step` (volume-weighted over the tape's own prints; the agent's own active fills are excluded from the benchmark).
- **TEST `tests/test_cost_model.py` (+10), `tests/test_exec_backtest.py` (+14)** — fee/rebate signs, square-root-law monotonicity, MDD on a hand-built path, slippage-vs-market & sign pins, cost-folds-into-reward-not-slippage, env byte-parity with defaults, market-VWAP volume-weighting. Full suite: **110 passed / 1 skipped**; ruff clean.

### Phase 3 — RL fairness + queue studies (P0/P1, weeks 3–4) — **LANDED 2026-09-13** (`research/queue_dynamics.py`, `research/adverse_selection.py`, `envs/regimes.py`, fair baselines, `evaluate_regime_ci` / `paired_difference_ci`, `scripts/rl_fairness_study.py` → `docs/results/rl_fairness.md`)
- **CREATE `python_quant/nexus_quant/research/queue_dynamics.py`** — order-level tracker over an ITCH stream: per resting order `ahead_qty`, `behind_qty`, cancel-cursor, consumed-at-level, fill/delete events; survival: `fill_prob_survival(...)` (Kaplan–Meier, cancel = competing risk), `logistic_fill_model(features) -> P(fill)`.
- **CREATE `python_quant/nexus_quant/research/adverse_selection.py`** — `post_fill_drift(fills, book, h)` over {1,5,25} events, conditioned on (fill side, OFI, queue position); `P(adverse | passive fill)`.
- **MODIFY `agents/evaluate.py`** — `evaluate_regime_ci(policy, regimes, seeds=5)` returns per-regime mean ± bootstrap-95%-CI of IS; add a **fair `volFeat` toggle**: baselines get the same indicator or the feature is disabled — the comparison must be symmetric.
- **MODIFY `agents/baselines.py` (or add `agents/baselines_rl.py`)** — stronger, defensible baselines: adaptive-POV (reacts to spread/vol), schedule-TWAP with a volume curve, IS-aware rule. Retire the 2-line "VWAP" heuristic as the reference.
- **TEST `tests/test_queue_dynamics.py`, `tests/test_adverse_selection.py`** — fill-prob calibration on a synthetic stream with a *known* queue; adverse-drift sign test; E5/E6 wiring.

### Phase 4 — Real data (P0, weeks 4–5 — the credibility unlock) — **LANDED 2026-09-13** (`scripts/fetch_itch.py`, `scripts/run_research.py`, `tests/test_offline_real_tape.py`; results `docs/results/real_tape_12302019.md`)
- **CREATE `scripts/fetch_itch.py`** — download a public NASDAQ ITCH sample (e.g. classic 2020-02-28 `S030220-v50.txt.gz`, or a published SPY/NVDA day) into **`data/`** (already gitignored). Record provenance + license note in the script header.
- **MODIFY `itch_parser.py`** — fix any real-file decode issues that surface; expose order-level events for OFI/queue (this is where the parser earns its keep).
- **CREATE `scripts/run_research.py`** — runs E1–E4 on the real tape; outputs IC/ICIR table + bootstrap CIs + figures.
- **TEST `tests/test_offline_real_tape.py`** — smoke: parser consumes the real tape with 0 truncated, integrity checks clean.

### Phase 5 — Report + write-up (P1/P2, week 5–6) — **LANDED 2026-09-13** (`docs/RESEARCH.md`, `scripts/run_all.py`, README/PROGRESS/CLAUDE rewritten to measured numbers)
- **CREATE `docs/RESEARCH.md`** (full) — hypotheses, methods, split discipline, IC tables, per-regime execution results, adverse-selection findings, **negative results section**. ✅ §1–§10: data provenance + parser validation, split/rigor discipline, E1–E4 tables (AAPL + QQQ, test-split IC with CIs), E5 KM + logistic calibration, E6 signed drift with pre-fill control, E7 fair RL table, §8 seven negative results, §9 reproduce, §10 DoD.
- **CREATE `scripts/run_all.py`** — one-command reproduce: fetch → build → test → research vignettes → exec comparisons → report. ✅ stages `tests, vignette, fetch, research, fairness`, each skippable; `--quick` smoke of every stage (64 MB of the gzip, one symbol, 100k rows, 2 seeds) writes to the gitignored `docs/results/quick/`; defaults reproduce `docs/results/` exactly.
- **MODIFY `README.md`** — final honest rewrite using only measured numbers. ✅ headline table restated as point estimates with CIs; PPO line reads "not reproduced fairly".

---

## 4. Experiment registry (E1–E7)

| # | Experiment | Primary metric | Split | Tests | Failure criterion |
|---|---|---|---|---|---|
| E1 | LOB imbalance → Δmid | rank IC by horizon {1,5,10,25} | walk-forward | IC≠0 (bootstrap CI 95%), DM vs 0 | IC ≤ 0.01 or unstable sign across folds |
| E2 | Microprice → Δmid | rank IC, vs E1 | walk-forward | pairwise DM, CI on ΔIC | microprice not ≥ bid/ask alone |
| E3 | OFI → Δmid | rank IC (nested vs imbalance) | walk-forward | LR / nested IC | no IC gain |
| E4 | Compare imbalance/microprice/OFI/combined | IC, ICIR, hit rate | identical split | permutation test | best feature changes per fold |
| E5 | Passive fill probability | Brier, survival AUC | time-ordered | calibration slope ≥ 0.8 | slope < 0.8 → queue model is fiction |
| E6 | Adverse selection after passive fills | signed post-fill drift | matched pairs | t-stat (NW) on drift | drift not ≠ 0 once conditioned on OFI/queue |
| E7 | Execution strategies (TWAP/VWAP/POV/Passive/Adaptive/PPO) | IS vs **market** VWAP, fill %, completion, MDD | per-regime × seeds≥5 | bootstrap CIs; fairness symmetric | any edge vanishes under fees+queue+adverse |

---

## 5. Statistical rigor — standing checklist

- Walk-forward time split (never shuffled CV on tape data).
- Event-based sampling (regulate on event cadence; kill volume-hour biases).
- Block / stationary bootstrap for CIs on IC and IS (never i.i.d. bootstrap on autocorrelated series).
- Diebold–Mariano for pairwise comparisons; Newey–West when labels overlap.
- Rank IC + ICIR primary; hit rate + decile spread robustness.
- AUC where appropriate (fill models); multiple seeds ≥5 for RL; per-regime eval.
- Sensitivity analysis: report IC/IS vs (horizon, tick size, λ, participation) — not just point estimates.
- **Leakage points identified in Part-1 code:** (1) `vol_feature` regime indicator asymmetry; (2) eval on training distribution; (3) shortfall vs self VWAP; (4) reward/eval objective mismatch; (5) regime tuned to target; (6) no CIs on headline; (7) "free" passive fills (no queue); (8) fixed arrival mid 15,000 with no drift/regime diversity; (9) `lambda_risk` seam default-off. All must be addressed before any claim leaves the repo.

---

## 6. RL fairness — the standing rework spec (do this before the headline is re-used)

1. Train/eval across regimes: low-vol, normal, high-vol, liquidity-shock, trending, mean-reverting.
2. ≥5 seeds; report mean ± bootstrap-95%-CI of IS, not one seed family.
3. **Symmetric information:** either baselines consume the same `vol_feature` indicator or the feature is removed from the agent.
4. Fair baselines: adaptive-POV, schedule-TWAP with a volume curve, IS-aware rule. Evaluate all strategies under a **shared objective** (same IS metric) and with fees + queue on.
5. Report IS vs **market** VWAP, completion%, MDD per regime.
6. Then, and only then, one line in README: "PPO vs best baseline: [X] bps IS improvement (CI [±Y]) under [regime]." If it's not significantly better, say so.

**Outcome (2026-09-13, `docs/results/rl_fairness.md`):** items 1–5 executed (6 regimes incl. the random-walk null arm, 5 training seeds × 5 eval seed families × 20 episodes, symmetric information in both modes, fair baselines, fees + queue on, IS vs arrival and vs market VWAP). **PPO is not significantly better than the best fair baseline (`adaptive_pov`) on highvol / highvol_null / trending; it is significantly worse on the calm and lowvol hold-outs (5/5 seeds); it is significantly better only under `liquidity_shock` (+0.4…+1.4 bps, 5/5 novol, 4/5 volsym).** The +50.4 % number is retired; the README line reads "no significant edge except under liquidity shocks".

---

## 7. Prioritization

| Rank | Task |
|---|---|
| P0 | Reconcile branches; checkout == main == docs; one dashboard |
| P0 | CI (GitHub Actions) |
| P0 | Research spine: `features.py` + `labels.py` + `dataset.py` + `experiments.py` + `models.py` |
| P0 | Statistical rigor module (walk-forward, block-bootstrap CI, DM, IC/ICIR) |
| P0 | RL fairness rework (§6) — re-characterize the +50.4% headline |
| P0 | Execution cost/queue model + market-VWAP metrics |
| P0 | Real ITCH tape ingestion + parser validation |
| P0 | Benchmark hardening + real-HW numbers (don't publish before) |
| P1 | Queue dynamics + adverse-selection studies (E5/E6) |
| P1 | Per-regime RL eval + adaptive baselines (E7) |
| P1 | Historical/parametric/stress VaR + scenario bank |
| P1 | `docs/RESEARCH.md` + honest README rewrite |
| P2 | CUDA toolkit + `risk_bench` measurement (fast once decided) |
| P2 | Execution timeline + inventory chart in dashboard |
| P2 | VWAP volume-curve forecast (real baseline) |
| P3 | Notebooks, plotting polish, ruff/mypy config, lockfile, RSS-style writeup |

**DO NOT BUILD:** live market connectivity/FIX; WebSocket fan-out; swapping PPO for a fancier RL to "fix" results; autoML feature stacking; NN predictors before linear IC baseline; cloud deployment; a custom plotting framework; **another tuned regime to chase another headline.**

---

## 8. Six-week calendar (starts Mon 2026-09-14)

| Week | Dates | Focus | Deliverables | Commit names |
|---|---|---|---|---|
| W1 | 09-14 → 09-20 | Phase 0 + Phase 1 start | Green CI; branch reconcile; `features.py`+`labels.py`+`dataset.py` with leak tests | `chore: reconcile branches to main`, `ci: build+test+lint`, `feat: research features + leak-proof datasets`, `bench: workload matrix + cache counters` |
| W2 | 09-21 → 09-27 | Phase 1 + E1/E2 | IC tables with CIs on synthetic; `docs/RESEARCH.md` skeleton | `feat: signal experiments E1/E2 + bootstrap CIs` |
| W3 | 09-28 → 10-04 | Phase 2 + Phase 3 start | Cost/queue model; queue dynamics + adverse selection (E5/E6); fair RL eval harness | `feat: execution cost+queue model`, `feat: queue/adverse-selection studies`, `fix: fair RL eval + regime CIs` |
| W4 | 10-05 → 10-11 | Phase 4 | Real ITCH tape ingested; parser validated; E1–E5 on real tape | `feat: ingest NASDAQ ITCH tape`, `test: real-tape parser smoke` |
| W5 | 10-12 → 10-18 | Phase 4/5 | E7 on real + synthetic with fees/queue/CIs; historical + stress VaR | `feat: E7 exec comparison on real tape`, `feat: historical/stress VaR` |
| W6 | 10-19 → 10-25 | Phase 5 | `docs/RESEARCH.md` (incl. negative results); README rewrite; resume bullets; final benchmarks | `docs: research report + honest README`, `chore: final benchmark numbers` |

---

## 9. Definition of Done — research half

- [ ] Working tree == `main` == README; no stale branches behind the docs.
- [ ] CI green (Linux + Windows): build, CTest, pytest, lint.
- [x] At least one real NASDAQ ITCH tape parsed + replayed cleanly (12/30/2019 AAPL/QQQ, 0 truncated, integrity clean, Engine-vs-Stub parity exact on the pre-market prefix); `test_offline_real_tape.py` asserts it whenever `data/` is present (CI has no tape — skips).
- [x] E1–E6 with walk-forward splits, rank IC, block-bootstrap CIs on a real day (`docs/results/real_tape_12302019.md`); negative results: `spread_bps` has no IC anywhere, microprice is not better than L1 imbalance, the per-event order-level OFI is weaker than the L2 approximation at h=1 (it only wins as a rolling flow at h ≥ 10), and the synthetic vignette stays null.
- [x] Execution eval: per-regime, ≥5 seeds, CI-reported, symmetric information, fair baselines, fees+queue on; headline **re-characterized** (`docs/results/rl_fairness.md`).
- [x] Cost/queue/impact model present (Phase 2 `execution/`); IS reported vs **market** VWAP with completion in the fairness study (`docs/results/rl_fairness.md`: `PPO slip vs market VWAP`, `PPO completion`); fill % and MDD are available via `execution.metrics` / `backtest.summarize` but are not in the committed E7 table.
- [ ] Zero-alloc + throughput/latency measured on a documented Linux box (compiler, flags, CPU, ≥30 runs, CI/IQR).
- [ ] CUDA risk measured (or clearly parked with the exact blocker).
- [x] `docs/RESEARCH.md` exists: hypotheses, methods, tables, and a "what we tried that failed" section (§8, seven items).

---

## 10. Running log (append per session)

- **2026-09-12** — Full 16-part repo audit delivered; `plan_2.md` created. Part-1 work (systems half) confirmed complete. Decision flagged: reconcile branches to `main` before building the research half.
- **2026-09-12 (later)** — **Phase 0 + Phase 1 spine landed on `main`** (`e630ad7`; ruff clean; Tier 1 85 passed / 1 skipped).
  - **Phase 0:** checkout `main` == docs; dashboard reconciled (combined desk `dashboard_page.html` + verification console `dashboard/index.html` + enhanced `SnapshotHub`); README deduped (was 2 full READMEs) + status matches main + slippage headline marked re-verification-pending (§1b/§6); CLAUDE.md brought to main's reality + points here; stale `.claude/worktrees/` + `pr7-fix` removed; `.github/workflows/ci.yml` (CMake+CTest+pytest on Linux/Windows, ruff lint); `requirements-dev.txt` (pytest/mypy/ruff pinned); Python layer ruff-clean (67 legacy findings, zero behavior change).
  - **Phase 1:** `research/{features,labels,dataset,experiments,models}.py` + `test_research.py` (17 tests: leak locks, walk-forward disjointness, IC≈1/IC≈0, bootstrap CI coverage, DM, OLS slope recovery) + `docs/RESEARCH.md` skeleton. Wired as `nexus_quant.research`.
  - **2026-09-12 (final)** — **Phase 1 E1–E4 synthetic IC vignette landed** (`research_vignette.py`): the spine is exercised end-to-end from synthetic flow → `StubOrderBook` → single- **and** pairwise (`ofi`) features → `event_frame` → `make_split` → `run_experiment` (rank IC / ICIR / hit rate / decile spread / block-bootstrap CI) → JSON. **Honest result: IC ≈ 0 with CIs straddling 0** for `ofi` / `deep_imbalance` / `microprice_off` across h ∈ {1,5,10} (correct null for a random walk). `lob_imbalance` shows a minor spurious drift (~0.1–0.2 IC at h≥5) — an artifact of random adds landing around the walk, recorded as a negative result, not a predictor. Also landed: `event_frame` now supports `prev_feature_fns` (row 0 dropped — docstring contract enforced in code) + test `test_event_frame_pairwise_feature_drops_first_row`; hit-rate reported tie-free (zeros excluded). Tier 1 stays **86 passed / 1 skipped**. **Person B work-package for Phases 2–4** is at `docs/work_package_b_phases_2_4.md` (exact verified interfaces: research spine, `OrderBookEnv`, baselines, `strategy_table`, PPO/GRPO, replay/book-port seam).
  - **Remaining:** Phase 2 (execution cost/queue model, market-VWAP metrics) → Phase 3 (queue dynamics + RL fairness rework, random-walk null arm) → Phase 4 (real tape; `fetch_itch.py` + `run_research.py` + order-level parser events).- **2026-09-13 — Phase 2 (execution realism) landed on `feature/part2-phase2-cost-queue`** (draft state was reviewed & corrected; not yet merged — pending the usual PR + real merge commit).
  - **Note:** `feat/part2-phases-0-1` (Phase 0+1, `4092cb2`) was already merged to `main` via PR #15 (`167b31a`) — the "merge pending" note is closed. Phase 2 branches off that merge.
  - **`execution/` package shipped:** `cost_model.py` (`CostParams`, `impact` square-root law, `net_pnl` = gross − fee + rebate), `metrics.py` (`implementation_shortfall`, `vwap_slippage` vs **market** VWAP — the Part-1 self-VWAP flaw fix, `arrival_slippage`, `fill_rate`, `completion_rate`, `inv_risk`, `max_drawdown`), `backtest.py` (`run_backtest` env-factory based + `summarize`, pre-cost slippage, per-regime), wired as `nexus_quant.execution`.
  - **Env knobs (all default-off, byte-identical today):** `fee_bps`/`rebate_bps` (applied maker/taker-aware: passive limit fills get the rebate, market takes pay the fee), `impact_coef`/`impact_participation`, `queue_model` (`""`/`"uniform"`, validated). **`market_vwap` added to `info`** — volume-weighted over the tape's own prints; the agent's own active fills excluded from the benchmark.
  - **Honesty fixes found while verifying the draft (would have corrupted E7 numbers if shipped):** (1) the drafted `net_pnl` code/docstring/test disagreed on sign — pinned to fee-lowers/rebate-raises, tested exactly; (2) `vwap_slippage`/`implementation_shortfall` had positive = *better* — inverted to positive = **a cost**, matching env `shortfall_bps`; (3) the env fee/impact reward penalty had inverted signs (a fee *improved* reward) — corrected, with the original cost tests shown to be vacuous (never called `reset()` → empty book → knobs never bit) and replaced with real fill-based tests.
  - **Tests:** `test_cost_model.py` (10) + `test_exec_backtest.py` (14) — fee/rebate signs, impact monotonicity, MDD on a hand-built path, slippage sign + market-not-self benchmark, cost-folds-into-reward-not-slippage, env byte-parity, market-VWAP volume-weighting. **Full suite 110 passed / 1 skipped**; ruff clean; `research_vignette.py` still prints the honest null IC table.
  - **Interfaces:** frozen in `docs/work_package_b_phases_2_4.md §2.2` (bumped for the two intended deltas: env-factory backtest + maker/taker-aware env costs).
  - **Remaining (unchanged):** Phase 3 (queue dynamics + RL fairness rework, random-walk null arm) → Phase 4 (real tape; `fetch_itch.py` + `run_research.py` + order-level parser events).
- **2026-09-13 (later) — Phases 3 + 4 landed (Person B, branch `hoplite/kranioi-5b44d8a8`).**
  - **Phase 3 / E5–E6:** `research/queue_dynamics.py` (`OrderLevelTracker` with exact FIFO ahead/behind, first-fill time, touch distances via lazy-heap BBO; Kaplan–Meier `fill_prob_survival` with cancels as competing risk; ridge-logistic `logistic_fill_model` with walk-forward holdout, Brier skill and calibration slope) + `research/adverse_selection.py` (signed post-fill drift, Newey–West t, pre-fill matched control, P(adverse), grouped by side / OFI sign / queue position). 19 tests, all exact (hand-computed KM, known-logistic recovery, constructed adverse stream).
  - **Phase 3 / §6 fairness:** `envs/regimes.py` (6 regimes incl. `highvol_null` random-walk null arm, `COSTS_ON`), fair baselines (`schedule_twap`, `adaptive_pov`, `is_aware`) + `regime_indicator` for symmetric information, `agents/evaluate.py` `run_regime_episodes` / `ci_from_rows` / `evaluate_regime_ci` / `paired_difference_ci`, `scripts/rl_fairness_study.py`. **Result:** no significant PPO edge vs `adaptive_pov` except under liquidity shocks; worse on calm/lowvol hold-outs. Committed at `docs/results/rl_fairness.{md,json}`.
  - **Phase 4 / real tape:** `scripts/fetch_itch.py` streams the public `emi.nasdaq.com` day and slices per symbol (`data/`, gitignored, manifest with SHA-256); `scripts/run_research.py` runs E1–E6 with order-level OFI (Cont–Kukanov–Stoikov `e_n`), test-split block-bootstrap CIs, train-fit OLS combination, DM tests, KM/logistic fill model, adverse-selection report → `docs/results/real_tape_12302019.md`. Parser 0 truncated on 1.5 M real messages; replay integrity clean; Engine-vs-Stub ladder parity exact on 7,037 real frames (engine needs a real price band, e.g. `Engine(1_000_000, 5_000_000, 1<<20)`).
  - **Spine fixes found on real data:** `features.ofi` upgraded to the CKS level-1 definition (touch moves count the full queue); `_rank`/block-bootstrap vectorized (output-identical, 50× faster — 200k-row bootstrap now 38 s); `decile_spread` averaged the 10 extreme points instead of decile bins — fixed; `research_vignette.py` sys.path pointed one level too shallow — fixed.
  - Tier 1: **152 passed** (was 110 / 1 skipped). ruff clean.
- **2026-09-13 (final) — Phase 5 landed (Person B, branch `hoplite/kranioi-5b44d8a8`). Research half complete.**
  - **Full-day E1–E6 rerun** on both symbols (`run_research.py --day 12302019 --symbols AAPL,QQQ --n-boot 300`, 1,289 s): AAPL 1,484,259 / QQQ 2,209,131 regular-session rows, 0 truncated, integrity clean, tracker 0 unknown ids. Results replaced the smoke-run numbers in `docs/results/real_tape_12302019.{md,json}`; hit rate now counts sign agreement only where label AND feature are non-zero and reports the coverage (`cov`) — a zero per-event OFI is "no view", not a miss.
  - **`docs/RESEARCH.md`** — full report: §2 data/provenance + parser validation on real bytes, §3 split & rigor discipline, §4 E1–E4 (AAPL: L1-imbalance IC 0.140→0.229 h=1→25, QQQ 0.155→0.462, all CIs ±≤0.014; OLS combination 0.164→0.288 / 0.157→0.461), §5 E5 (KM fill curves; logistic slope 1.09 / 1.03, Brier skill +0.080 / +0.016), §6 E6 (post-fill drift −20/−50/−93 ticks AAPL, −11/−26/−44 QQQ, NW |t| 68–147, P(adverse) 0.90–0.99, pre-fill control confirms selection not momentum), §7 E7 fair RL table, **§8 seven negative results** (headline not reproduced; microprice ≯ imbalance; spread has no IC; per-event OFI weaker than the L2 approx at h=1; fill model has little skill beyond base rate; OFI sign does not explain away adverse selection; synthetic vignette stays null), §9 reproduce, §10 DoD. Every number cross-checked against the final JSON (script-verified, 0 mismatches).
  - **`scripts/run_all.py`** — one-command reproduce (`tests → vignette → fetch → research → fairness`), `--skip`, `--quick` (writes to gitignored `docs/results/quick/`).
  - README/PROGRESS/CLAUDE/progress_b rewritten to the measured numbers; `docs/work_package_b_phases_2_4.md` closed out.
  - **Remaining (research half):** more tape days / symbols (`fetch_itch.py --day …`, ≈ 14 min each); the E7 table could add fill % / MDD from `execution.backtest.summarize`. Person A: GPU + hardware numbers.
