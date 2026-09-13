# Work package — Person B: Phases 2–4 (execution realism → RL fairness → real tape)

> **Plan of record:** `plan_2.md` §3–§8. This file is the Person B hand-off for Phases 2–4.
> Every interface named here is **exact as of 2026-09-12** (verified against `main` + the
> Phase 0/1 spine). Where a signature is **TO FREEZE** (bold), Person B freezes it and Person A
> consumes it — no renames without bumping this doc and `plan_2.md`.
>
> Predecessor work already landed: Phase 0 (repo hygiene, CI) + Phase 1 spine
> (`research/{features,labels,dataset,experiments,models}.py`, `test_research.py`, `docs/RESEARCH.md`)
> on `main`; the synthetic E1–E4 IC vignette (`scripts/research_vignette.py`) now runs
> end-to-end and prints the first honest IC table (null result, by design — see §R below).

Owner: **Person B (Quant Research & RL Lead)**. Integration counterpart: **Person A** (C++ engine, shmem, CUDA).

---

## 0. TL;DR

| # | Phase | Goal | Deliverables | A/B seam |
|---|-------|------|--------------|----------|
| WP2 | **Phase 2 — execution realism** | Make the sim's cost/queue realism honest enough that *accuracy* (not rank) means something | `execution/{cost_model,metrics,backtest}.py`; `env` fee/rebate/impact/queue knobs; market-VWAP metrics | Person A reviews the fill/impact realism; B keeps `reset()/step()` contract frozen |
| WP3 | **Phase 3 — RL fairness** | Re-verify (or re-characterize) the +50.4% headline under symmetric information, fair baselines, per-regime CIs | `research/{queue_dynamics,adverse_selection}.py`; `evaluate_regime_ci`; fair `strategy_table` runs | **A provides the random-walk control env** (B's null arm) |
| WP4 | **Phase 4 — real tape** | Real NASDAQ ITCH bytes → replay → features → E1–E7 tables | `scripts/fetch_itch.py`, `scripts/run_research.py`; `itch_parser.py` order-level events; `test_offline_real_tape.py` | A extends `EngineAdapter`/parser for order-level events if needed |

Each phase ends with a **MERGE** (real merge commit, per the git workflow — feature branch, PR, review).

---

## R. What the Phase-1 vignette already establishes (recorded, honest)

`python_quant/scripts/research_vignette.py` (synthetic E1–E4, seeded random-walk flow, h ∈ {1,5,10}):

- The spine works end-to-end: `synthetic_events` → `StubOrderBook` → `features` (single-view **and** pairwise `ofi`) → `event_frame` → `make_split` (walk-forward) → `run_experiment` (rank IC, ICIR, hit rate, decile spread, block-bootstrap CI) → JSON.
- **Honest result:** a random-walk flow gives IC ≈ 0 (CIs straddle 0) for `ofi`, `deep_imbalance`, `microprice_off` across horizons — the null harness behaves as designed. `lob_imbalance` shows a small spurious drift (~0.2 IC at h=5–10, CI excluding 0) — an artifact of random *adds* landing around the walk, not a predictor. Recorded as a **negative/harness result**; real claims wait for Phase 4 tape.
- Vignette CLI: `--steps` `--seed` (int or hex) `--horizons` `--train` `--val` `--out <json>`.
- 18 research tests now cover the spine incl. the `prev_feature_fns` **row-0-drop + exact OFI** contract (`test_event_frame_pairwise_feature_drops_first_row`).

**Do not** tune the synthetic flow to produce nonzero IC — that repeats the Part-1 headline-tuning sin (`plan_2.md §0 rule 2`). The null result is the correct result here.

---

## 1. Exact interfaces (verified 2026-09-12 — freeze these)

### 1.1 Research spine (already on `main`) — Person B owns, runs *as-is*

```python
from nexus_quant.research import (               # __init__ re-exports
    event_frame, make_split,                     # event_frame: see below
    # experiments/rigor
    rank_ic, icir, hit_rate, decile_spread,
    bootstrap_ci, diebold_mariano, run_experiment,
    # features
    lob_imbalance, microprice, deep_imbalance, spread_bps,
    ofi, realized_vol, momentum, flow_intensity,
    # labels
    forward_mid_move, forward_return,
    # models (linear-only, per §3 Phase 1 rule)
    fit_ols_ic, zscore,
)
from nexus_quant.research.dataset import Row     # Row is NOT in package __all__

# event_frame — pairwise prev-features supported (new since spine):
def event_frame(
    events, apply, *,
    feature_fns: Mapping[str, Callable[[View], float]],
    prev_feature_fns: Mapping[str, Callable[[View, View], float]] | None = None,
    h: int = 5,
    label_fn: Callable[[View, View, int], float] = forward_mid_move,
    ts_fn: Callable[[View], int] = lambda v: int(v["seq"]),
) -> list[Row]: ...
#   Row 0 is DROPPED when prev_feature_fns is set (no pairwise predecessor).
#   apply(ev) mutates the book and returns the current view (mirrors ReplayEngine.apply).

# run_experiment — the E1–E4 result bundle:
def run_experiment(y_true, y_pred, *, split_tags=None, n_boot=2000, seed=0x51ED) -> dict: ...
#   {"n": int, "overall": {...}, "per_split": {"train"/"val"/"test": {...}}}
#   per-split fields: rank_ic, icir, hit_rate, decile_spread, ci95{"lo","hi","mean"}
```

### 1.2 Env + baselines + agent harness (from Phase 1c/1e — freeze)

```python
# envs/order_book_env.py
class OrderBookEnv(gym.Env):            # obs dim = 44, Gymnasium reset()/step()
    def __init__(self, *, seed=0, is_coef=0.0, lambda_sched=..., lambda_risk=0.0,
                 regime_prob=0.0, vol_decay=0.0, gap_*=..., vol_*=..., ...): ...
    def reset(self, *, seed=None, options=None) -> (np.ndarray, dict): ...
    def step(self, action) -> (obs, reward, terminated, truncated, info): ...
    #   high-vol regime params default-off (byte-identical to calm env)
    #   lambda_risk>0 -> CVaR inventory penalty from risk.py (exact-parity oracle)

# baselines.py
def run_episode(env, name: AgentId, *, seed=None, policy=None) -> EpisodeResult: ...
def compare(seed=20973, **env_kw) -> list[EpisodeResult]: ...
EpisodeResult: dataclass(name, reward, shortfall_bps, vwap, arrival, leftover, filled, steps)

# agents/evaluate.py
def strategy_table(agent=None, *, agent_name="ppo", n_episodes=50, seed=0,
                   baselines=("twap","vwap","pov","passive"),
                   env_factory=OrderBookEnv) -> list[dict]: ...
def format_table(rows: list[dict]) -> str: ...

# agents/ppo.py
def train_ppo(env_factory=OrderBookEnv, cfg=None, *, tracker=None) -> (PPOPolicy, list[TrainHistory]): ...
class PPOPolicy(obs_dim=44, hidden=(64,32), *, seed=0, std0=0.4): ...
PPOPolicy.save(path) / PPOPolicy.load(path)   # .npz, no torch
# agents/grpo.py — GRPO trainer on the same actor interface
```

### 1.3 Tape/engine seam (from Phase 1b/1e — freeze)

```python
# replay.py
class ReplayEngine: __init__(book, ...); apply(msg, ...) -> view-ish; check_integrity(...) 
# book_port.py — the injectable book seam
StubBookAdapter  (oracle)  <->  EngineAdapter  (real C++ engine, via pybind)
#   EngineAdapter has cancel_id()/lookup() (added for replay) — full cancel via
#   engine.cancel, partial via engine.modify at same price (time priority). [CLAUDE.md §6]
```

---

## 2. WP2 — Phase 2: execution realism (weeks 2–3)

### 2.1 Files
- **CREATE `python_quant/nexus_quant/execution/__init__.py`** + `cost_model.py` + `metrics.py` + `backtest.py`
- **MODIFY `envs/order_book_env.py`** — additive, default-off
- **CREATE `python_quant/tests/test_cost_model.py`**, `python_quant/tests/test_exec_backtest.py`
- **CREATE `docs/RESEARCH.md`** (§ methods + template tables; skeleton exists)

### 2.2 Interfaces — FINAL, as shipped on `feature/part2-phase2-cost-queue` (bumped 2026-09-13)
The drafted signatures below were revised during implementation; these are the exact, frozen shapes Person B consumes. Wrap A/B: both already use `net_pnl`, `impact`, and `vpwap_slippage` unrescaled (Phase 5 confirmations).

```python
# execution/cost_model.py
@dataclass(frozen=True)
class CostParams(fee_bps=0.0, rebate_bps=0.0, spread_ecn=0.0, impact_coef=0.0, impact_mode="sqrt")
impact(qty_participation, *, sigma, coef, mode="sqrt") -> float   # market impact, bps; monotone in participation
net_pnl(fills, *, side_cost: CostParams | None = None, gross=0.0) -> float
    #  net = gross − fee_ticks + rebate_ticks   (fee lowers, rebate raises the net)

# execution/metrics.py  — ALL slippage metrics read POSITIVE = A COST (worse),
#   identical in sign to env `info["shortfall_bps"]`; `side`: +1 sell, −1 buy.
implementation_shortfall(fills, arrival_mid, *, side=1) -> float
vwap_slippage(fills, market_vwap, *, side=1) -> float  # vs MARKET vwap, NOT self-executed vwap (Phase-1 flaw fix)
arrival_slippage(fills, arrival_mid, *, side=1) -> float   # alias of implementation_shortfall
execution_vwap(fills) -> float                              # internal only — never a benchmark
fill_rate(fills, target_qty) -> float
completion_rate(fills, target_qty, leftover) -> float
inv_risk(sigma, inv, t_fraction) -> float
max_drawdown(pnl_path) -> float

# execution/backtest.py  — variance vs the draft: no `policy,tape,book,cost` args. The env
#   IS the tape + book; costs fold into the env's own reward via the cost knobs, and the
#   slippage columns stay pre-cost so every strategy compares on the same timing skill.
run_backtest(env_factories, *, policy=None, n_episodes=50, seed=0x51ED, name="policy") -> list[dict]
    # env_factories: Callable[[], OrderBookEnv] | (name, factory) | [(name, factory), ...]
summarize(rows) -> list[dict]   # per-(regime, policy) means
```
- **env additions (defaults keep today byte-identical — same discipline as regime params):** `fee_bps`, `rebate_bps`, `impact_coef`, `impact_participation` (default 0.1), `queue_model` (`""` = off, `"uniform"` = placeholder queue degradation); **`market_vwap` is added to `info`** in every `step`, volume-weighted over the tape's own prints only — the agent's own active fills are excluded (never used as the benchmark).
- **Fee/rebate are maker/taker-aware:** a resting limit filled by flow gets the maker rebate; a market / capped-market child pays the taker fee. `is_ticks` is a cost, so fees ADD and rebates subtract — verified against the original (inverted, was silently rewarding fees) implementation.
- **Contract-freeze rule:** `reset()/step()` shapes, obs dim 44(45 with vol_feature), reward type — **unchanged**. New knobs are keyword-only with defaults that reproduce the 2026-09 byte-identical traces.
- **Acceptance (all in `test_cost_model.py` + `test_exec_backtest.py`):** (a) maker-rebate sign, taker-fee sign; (b) impact monotone in participation; (c) MDD on a hand-built path; (d) **env byte-parity when all new knobs default** — identical 20-step reward trace; (e) slippage positive = a cost and benchmarked vs MARKET VWAP; (f) cost knobs move `reward`, never the pre-cost slippage columns.

---

## 3. WP3 — Phase 3: RL fairness + queue studies (weeks 3–4) — **SHIPPED 2026-09-13** (interfaces below updated to what landed)

### 3.1 Goal
Re-verify the +50.4% headline under the §6 rework spec **before** it re-enters any README:
symmetric info, fair baselines, per-regime CIs, random-walk null arm. Expected honest outcome
may be "PPO not significantly better than best baseline under fees+queue" — that is an
acceptable, honest result.

### 3.2 Files
- **CREATE `research/queue_dynamics.py`**, `research/adverse_selection.py`
- **MODIFY `agents/evaluate.py`** (`evaluate_regime_ci`), **MODIFY `baselines.py`** (adaptive-POV, schedule-TWAP, IS-aware rule) or **CREATE `agents/baselines_rl.py`**
- **CREATE tests**: `test_queue_dynamics.py`, `test_adverse_selection.py`

### 3.3 Interfaces (exact)
```python
# research/queue_dynamics.py  (as shipped)
class OrderLevelTracker:            # feed NormalizedEvent in tape order
    on_event(ev); on_add/on_execute/on_cancel_delete/on_replace/on_trade(ev)
    ahead_qty(oid) / behind_qty(oid) / level_size(side, px) / best_price(side)
    orders: dict[oid, TrackedOrder]; completed: list[TrackedOrder]   # outcome "filled" | "cancelled"
TrackedOrder(order_id, side, price, size, size0, ts_add, idx_add, ahead_at_add, ahead,
             same_best_dist, opp_dist, filled, ts_first_fill, idx_first_fill, ts_end, idx_end, outcome)
fill_prob_survival(orders, *, horizons, clock="events"|"ns", cancel_is_censoring=True) -> {tau, survival, p_fill, n, n_fill, n_cancel, median_fill_time}
censor_open_orders(tracker, ts_end) -> list[TrackedOrder]          # still-resting orders as censored records
order_features(o) -> {log_ahead, log_size, queue_frac, same_best_dist, opp_dist, no_opposite}
logistic_fill_model(features, filled, *, l2=1e-3, holdout=0.3) -> {coefs, brier, brier_base_rate, brier_skill, calibration_slope, calibration_table, predict, ...}
# research/adverse_selection.py  (as shipped)
PassiveFill(idx, side, price, size, ofi, queue_frac)
post_fill_drift(fills, mids, h) / pre_fill_drift(fills, mids, h) -> np.ndarray   # s·Δmid, NaN when no future
nw_tstat(x, *, lag) -> {n, mean, se, t, p};  p_adverse(drifts) -> float (ties excluded)
adverse_selection_report(fills, mids, *, horizons=(1,5,25), min_group=20) -> {n_fills, horizons: {h: {overall, pre_fill, post_minus_pre, groups}}}
fills_from_tracker(completed, *, ofi_at=None) -> list[PassiveFill]
# envs/regimes.py  (NEW)
REGIMES, COSTS_ON, TRAIN_REGIME="highvol", HOLDOUT_REGIMES
regime_kwargs(name, *, costs=True, vol_feature=False, **overrides) / make_regime(...) / regime_factories(names=None, ...)
# baselines.py  (MODIFY, additive)
AgentId += "schedule_twap" | "adaptive_pov" | "is_aware";  LEGACY_BASELINES / FAIR_BASELINES / ALL_BASELINES
regime_indicator(env) -> bool | None          # == obs[44] when vol_feature else None (symmetric information)
volume_curve_target(t_frac, *, u_weight=0.6)  # CDF of 1 + w·cos(2πt)
# agents/evaluate.py  (MODIFY — strategy_table intact)
run_regime_episodes(policy, regimes, *, seeds=5, episodes_per_seed=20, seed0, baselines=(), agent_name="ppo") -> {regime: {strategy: [row,...]}}
ci_from_rows(rows, *, metric, name, regime, n_boot=2000) -> RegimeCI(name, regime, metric, mean, ci95, n_episodes, n_seeds, per_seed_mean)
evaluate_regime_ci(policy, regimes, *, seeds=5, episodes_per_seed=20, metric="shortfall_bps", baselines=(), ...) -> {regime: {strategy: RegimeCI}}
paired_difference_ci(policy, baseline, regimes, *, ..., episodes=None) -> {regime: {mean, lo, hi, n, frac_agent_better, pct_vs_baseline}}
format_regime_table(result) -> str
# scripts/rl_fairness_study.py  → docs/results/rl_fairness.{md,json}
```
**Result (recorded):** PPO not significantly better than `adaptive_pov` on highvol / highvol_null / trending; significantly worse on calm / lowvol; significantly better only under `liquidity_shock`. The random-walk null arm behaves as designed (PPO ≈ baselines).
- **Fairness fixes that must land here (§5 leakage list 1,2,3,5):**
  1. **vol_feat symmetric:** baselines observe the same regime indicator, or the feature is removed — never asymmetric.
  2. **eval ≠ train distribution:** hold out a regime set the agent never trains in.
  3. **VWAP vs market VWAP** (`vwap_slippage`), not self-VWAP.
  4. **fees + queue ON** for every reported number (Phase 2).
  5. **≥5 seeds, per-regime, CI from block bootstrap** — one seed family never reported alone.
- **Random-walk null arm (Person A):** env subclass whose mid follows a seeded random walk with identical fill/cost/penalty mechanics. PPO ≈ baseline on shortfall is the correct result.

---

## 4. WP4 — Phase 4: real tape (weeks 4–5, the credibility unlock) — **SHIPPED 2026-09-13** (`fetch_itch.py`, `run_research.py`, `test_offline_real_tape.py`; parser needed NO decode fixes on 268 M real messages)

### 4.1 Files
- **CREATE `scripts/fetch_itch.py`** (public NASDAQ ITCH sample → `data/`, gitignored; provenance + license in header), **CREATE `scripts/run_research.py`**
- **MODIFY `itch_parser.py`** (real-file decode fixes; expose order-level events for OFI/queue)
- **CREATE `python_quant/tests/test_offline_real_tape.py`**
- **MODIFY `research_vignette.py`** if it should accept a real day (`--tape`) — optional; `run_research.py` is the tape runner.

### 4.2 Exact seams
```python
# scripts/fetch_itch.py  (as shipped)
fetch(*, day, symbols, out_root, base=DEFAULT_BASE, max_gz_bytes=None) -> manifest dict
#   streams https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/<day>.NASDAQ_ITCH50.gz (HTTP Range optional),
#   inflates on the fly, writes data/itch/<day>/<SYMBOL>.itch (S + R + A F E C X D U P for that locate)
#   + manifest.json (source URL, bytes, SHA-256, per-type counts). data/ is gitignored.
# scripts/run_research.py  (as shipped — variance vs the draft: features are computed INLINE during the
#   replay instead of via event_frame, because a full day is ~1.5 M views and event_frame materializes them;
#   the split still uses make_split on the plan's Row primitive)
replay_symbol(path, *, max_events=None) -> {features, mids, ts, tape_idx, tracker, stats, issues, ...}
order_level_ofi(ev, live_before) -> float       # Cont–Kukanov–Stoikov e_n from the order stream
ic_study(rep, *, horizons, train, val, n_boot) -> {rows, combined, dm}   # E1–E4
fill_study(rep, *, horizons_events) -> {km, logistic, outcomes, ...}      # E5
adverse_study(rep, *, horizons) -> adverse_selection_report(...)           # E6
#   → docs/results/real_tape_<day>_<SYMBOL>.json + docs/results/real_tape_<day>.md
# tests/test_offline_real_tape.py
test_symbol_slicer_keeps_exactly_the_symbol_stream()   # always runs (hand-built framed stream, byte-exact)
test_real_tape_parses_clean()                           # skips unless data/itch/<day>/<SYMBOL>.itch exists
```
- **Parser order-level upgrade (A/B) — DONE:** the parser already emitted order-level events; `run_research.py::order_level_ofi` computes the exact per-event OFI from the tracker's live order state, and `features.ofi` (the L2 approximation) was upgraded to the Cont–Kukanov–Stoikov level-1 definition. On the real day the rolling order-level OFI beats the L2 approximation at h ≥ 10 on both names (AAPL 0.189 vs 0.149, QQQ 0.151 vs 0.121 at h=10) but is *weaker* at h=1, where a single event's OFI is zero most of the time (`docs/results/real_tape_12302019.md`, `docs/RESEARCH.md` §4 E3).

---

## 5. Merge/workflow rules (both halves)

- Work on `feature/part2-*` branches, small PRs, **real merge commits** (project rule — commit count visible, never squash).
- Each PR: Tier-1 pytest green (`152 passed` today), ruff clean, test count ticks up.
- End-of-token handoff: update **`plan_2.md` §10 running log** (append one bullet per session) so the next session resumes without re-derivation.

## 6. Definition of done (checklist — copy into the WP4 PR body)

- [x] `python -m pytest python_quant/tests` green (today: 152 passed)
- [x] `scripts/research_vignette.py` runs, prints honest null IC table (≈0, CIs straddling 0)
- [x] Phase 2: `test_cost_model.py` + `test_exec_backtest.py` green; env byte-parity test with defaults
- [x] Phase 3: `evaluate_regime_ci` ≥5 seeds per regime; fairness toggles verified symmetric (`test_rl_fairness.py`); random-walk null arm result recorded (`docs/results/rl_fairness.md`)
- [x] Phase 4: `fetch_itch.py` + `run_research.py` produce E1–E6 tables from a real day (`docs/results/real_tape_12302019.md`); `test_offline_real_tape.py` green (E7 = the fairness study, synthetic regimes)
- [x] Headline re-characterized with CI (not reproduced); README uses only measured numbers
- [x] `plan_2.md` running log updated; this doc's interfaces match the code (this bump)