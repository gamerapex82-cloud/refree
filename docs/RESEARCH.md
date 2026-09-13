# Nexus-LOB — Research Report (Part 2)

> **Status (2026-09-13):** Phases 0–5 complete. Every number below is measured
> by a script in this repo (`python_quant/scripts/`) and reproducible with
> `python python_quant/scripts/run_all.py`. Ground rules: `plan_2.md` §0.
> Negative results are in §8. Nothing here was tuned to a target.

## 1. Scope

Does an L2/L3 order book predict short-horizon mid moves, and can an execution
strategy turn that into lower implementation shortfall? We test the canonical
microstructure signals — LOB imbalance, microprice, order-flow imbalance (OFI,
both the L2-ladder approximation and the exact order-level version), deep
imbalance, spread — against strictly forward tick-move labels on walk-forward
splits with block-bootstrap CIs (E1–E4); measure passive fill probability (E5)
and adverse selection after passive fills (E6) from the order-level queue; and
re-verify the RL execution agent fairly against defensible baselines (E7).

## 2. Data

| Dataset | Source | Use |
|---|---|---|
| **NASDAQ TotalView-ITCH 5.0, 2019-12-30** | public sample `emi.nasdaq.com/ITCH/Nasdaq ITCH/12302019.NASDAQ_ITCH50.gz` (3.5 GB gz, 268.7 M messages) — `scripts/fetch_itch.py` streams it and slices per symbol; `data/` is gitignored, a `manifest.json` records URL, byte counts and SHA-256 | E1–E6 |
| AAPL slice | 1,519,370 msgs → 1,519,357 events, **0 truncated**, 1,484,259 regular-session rows | |
| QQQ slice | 2,376,260 msgs → 2,376,241 events, **0 truncated**, 2,209,131 regular-session rows | |
| Seeded synthetic flow (`OrderBookEnv`, six regimes) | `envs/regimes.py` | E7 (RL) + the null harness |

Parser/replay validation on the real bytes: 0 truncated messages, every event
applies to the book (`ReplayEngine.skipped == 0`), integrity clean during the
regular session (crossed / locked / negative / unsorted never fire; the only
flag all day is `empty_bbo` on the first pre-open message), order-level tracker
0 unknown ids, and the C++ engine agrees with the Python oracle **bit-exactly on
the L2 ladder over 7,037 real pre-market frames** (`Engine(1_000_000,
5_000_000, 1<<20)` — real ITCH prices are $×10⁴ ticks, above the default band).

## 3. Split & rigor discipline

- **Regular session only** (09:30–16:00 ET); pre-market books are thin and stale.
- **Event clock**: every book-affecting message is a row (1.5–2.2 M rows/day).
- **Walk-forward 60/20/20** by tape time with a gap of the largest horizon (25
  events) between blocks; the OLS combination is fit on *train* only and scored
  on *test*; the bootstrap CI is computed on the *test* block.
- **Labels**: forward mid move in ticks ($0.0001) at `h ∈ {1, 5, 10, 25}`
  events; many labels are exactly 0 (9 % non-zero at h=1 for AAPL, 67 % at
  h=25), so hit rate is reported on rows where both label and feature are
  non-zero, with the coverage.
- **Statistics**: Spearman rank IC; moving-block bootstrap (block = √n) 95 %
  CI; Diebold–Mariano with Newey–West HAC (lag = h) on squared errors of
  train-fit linear predictors; Newey–West t-stats on post-fill drift.
- **Leak locks** (`tests/test_research.py`): features are pure functions of the
  view at `t` (mutation-after-call test); labels use only the `t+h` endpoint;
  `event_frame` drops row 0 for pairwise features.

## 4. E1–E4 — signals → forward mid move (real tape, test split, out of sample)

Rank IC on the test block with 95 % block-bootstrap CI (`docs/results/real_tape_12302019.md` has train/val/test, hit rates, decile spreads, DM tables).

**AAPL (2019-12-30)**

| feature | h=1 | h=5 | h=10 | h=25 |
|---|---|---|---|---|
| lob_imbalance (L1) | **0.140** [0.135, 0.143] | **0.220** [0.213, 0.228] | **0.238** [0.229, 0.248] | **0.229** [0.215, 0.243] |
| microprice − mid | 0.139 [0.135, 0.144] | 0.216 [0.207, 0.224] | 0.229 [0.219, 0.239] | 0.217 [0.204, 0.230] |
| deep_imbalance (k=5, 1/i weights) | 0.101 [0.095, 0.106] | 0.172 [0.162, 0.183] | 0.196 [0.182, 0.209] | 0.200 [0.181, 0.217] |
| OFI, order-level, rolling 20 events | 0.088 [0.084, 0.092] | 0.156 [0.148, 0.164] | 0.189 [0.178, 0.200] | 0.207 [0.191, 0.219] |
| OFI, order-level, single event | 0.076 [0.071, 0.080] | 0.103 [0.098, 0.108] | 0.112 [0.107, 0.116] | 0.109 [0.105, 0.113] |
| OFI, L2-ladder approx. (CKS L1) | 0.123 [0.116, 0.131] | 0.147 [0.142, 0.154] | 0.149 [0.143, 0.155] | 0.139 [0.134, 0.144] |
| spread_bps | 0.003 [−0.002, 0.008] | 0.002 [−0.008, 0.012] | −0.001 [−0.014, 0.011] | 0.004 [−0.017, 0.022] |
| **OLS of all (train-fit)** | **0.164** [0.160, 0.169] | **0.255** [0.248, 0.263] | **0.284** [0.275, 0.294] | **0.288** [0.274, 0.300] |

**QQQ (2019-12-30)**

| feature | h=1 | h=5 | h=10 | h=25 |
|---|---|---|---|---|
| lob_imbalance (L1) | **0.155** [0.152, 0.158] | **0.296** [0.290, 0.301] | **0.375** [0.367, 0.382] | **0.462** [0.453, 0.472] |
| microprice − mid | 0.160 [0.157, 0.163] | 0.301 [0.294, 0.307] | 0.378 [0.369, 0.386] | 0.460 [0.450, 0.471] |
| deep_imbalance | 0.134 [0.131, 0.138] | 0.266 [0.259, 0.272] | 0.340 [0.332, 0.351] | 0.421 [0.409, 0.435] |
| OFI, order-level, rolling 20 | 0.067 [0.065, 0.070] | 0.124 [0.119, 0.131] | 0.151 [0.143, 0.160] | 0.180 [0.168, 0.192] |
| OFI, order-level, single event | 0.071 [0.067, 0.074] | 0.113 [0.108, 0.117] | 0.127 [0.122, 0.131] | 0.133 [0.128, 0.137] |
| OFI, L2-ladder approx. | 0.082 [0.077, 0.085] | 0.115 [0.110, 0.119] | 0.121 [0.117, 0.125] | 0.116 [0.111, 0.120] |
| spread_bps | 0.016 [0.011, 0.021] | 0.024 [0.015, 0.033] | 0.032 [0.021, 0.042] | 0.028 [0.016, 0.040] |
| **OLS of all (train-fit)** | 0.157 [0.153, 0.159] | 0.298 [0.292, 0.304] | 0.376 [0.367, 0.383] | 0.461 [0.451, 0.470] |

Readings (E1–E4 conclusions):

- **E1 — imbalance predicts the next mid move.** IC is positive at every horizon
  on both names, CIs far from 0, sign stable across train/val/test folds (the
  E1 failure criterion — sign flips across folds — does not trigger). IC grows
  with horizon (integer-tick labels at h=1 are mostly 0; more labels resolve by
  h=25). QQQ's ETF book (tight, deep, 1-tick spread) is far more predictable
  than AAPL's (IC 0.46 vs 0.23 at h=25).
- **E2 — microprice ≈ imbalance, not better.** The microprice offset and L1
  imbalance are the same information on a 1-tick book (rank IC within 0.01
  everywhere). DM tests give mixed signs by horizon and symbol (AAPL: microprice
  loss *higher* at h=5, p<0.001; QQQ: *lower* at h=5, p<0.001) — no consistent
  winner. **E2 fails its own success criterion**: microprice is not ≥ imbalance.
- **E3 — order-level OFI is an independent, weaker signal.** Order-level OFI is
  well below imbalance on its own (0.09–0.21 vs 0.14–0.46) but adds to the OLS
  combination on AAPL (0.288 vs 0.229 alone at h=25; DM combined vs imbalance
  p<0.001 at every horizon). On QQQ the combination adds nothing over imbalance
  (0.461 vs 0.462) — the book state already carries the flow information.
  Against the L2 approximation: the exact order-level OFI wins clearly only as
  a rolling 20-event flow at h ≥ 10 (AAPL 0.189 vs 0.149; QQQ 0.151 vs 0.121);
  per-event it is *weaker* than the approximation at h=1 because a single
  event is sparse (zero most of the time).
- **E4 — best feature is stable within a name.** L1 imbalance / microprice lead
  on every fold and horizon; OFI ranks third; spread has no information.

## 5. E5 — passive fill probability (order level)

`OrderLevelTracker` reconstructs every resting order's FIFO position from the
ITCH stream. Kaplan–Meier P(first fill by τ events after placement), with
cancellation as the competing risk (censoring):

| τ (events) | 10 | 50 | 100 | 500 | 1000 | 5000 |
|---|---|---|---|---|---|---|
| AAPL (791,477 regular-session orders; 5.8 % ever fill) | 0.011 | 0.034 | 0.049 | 0.095 | 0.114 | 0.163 |
| QQQ (1,141,142 orders; 1.5 % ever fill) | 0.003 | 0.012 | 0.018 | 0.039 | 0.053 | 0.081 |

Logistic fill model on placement-time features (`log_ahead`, `log_size`,
`queue_frac`, distance to same-side best, distance to the opposite best), fit
on the first 70 % of the day, scored on the last 30 %:

| | base rate | Brier | Brier (base rate) | skill | **calibration slope** |
|---|---|---|---|---|---|
| AAPL | 0.057 | 0.0537 | 0.0584 | +0.080 | **1.09** |
| QQQ | 0.015 | 0.01687 | 0.01715 | +0.016 | **1.03** |

The E5 failure criterion (slope < 0.8 → "the queue model is fiction") does
**not** trigger: the model is calibrated on both names (decile tables in the
results file are monotone). Skill is modest: fill probability at placement is
dominated by *where* the order is placed (distance to the touch — the strongest
coefficient on QQQ is `opp_dist −1.05`) more than by queue position; 94–98 % of
orders are cancelled before they ever trade.

## 6. E6 — adverse selection after passive fills

Signed post-fill drift `s·(mid[t+h] − mid[t])` in ticks (s = +1 for a filled
bid, −1 for a filled ask; **negative = adverse**), Newey–West t (lag h), with a
matched pre-fill control (the same drift over the h events *before* the fill):

| | h | n fills | post drift | NW t | P(adverse) | post − pre | NW t |
|---|---|---|---|---|---|---|---|
| AAPL | 1 | 46,357 | −20.3 | −131 | 0.970 | −19.5 | −114 |
| AAPL | 5 | 46,357 | −50.0 | −147 | 0.957 | −45.8 | −117 |
| AAPL | 25 | 46,356 | −92.9 | −114 | 0.895 | −61.3 | −71 |
| QQQ | 1 | 17,662 | −11.4 | −71 | 0.988 | −11.4 | −68 |
| QQQ | 5 | 17,662 | −25.7 | −80 | 0.958 | −26.4 | −75 |
| QQQ | 25 | 17,661 | −43.8 | −68 | 0.912 | −49.0 | −61 |

Passive fills are adversely selected essentially always: 97–99 % of first fills
are followed by a mid move against the filled position at h=1 (96–99 % through
h=5), and the drift
keeps growing to h=25 (AAPL −93 ticks ≈ −0.93 ¢ on a $290 stock ≈ 0.3 bps; QQQ
−44 ticks ≈ 0.2 bps). The pre-fill control confirms it is selection, not
momentum: the drift *before* the fill is small and mostly positive (the price
was moving toward the order), so `post − pre` is as negative as `post` itself.
Conditioning (results file, h=5): back-of-queue fills are slightly more adverse
than front-of-queue (AAPL P(adverse) 0.972 vs 0.932); the sign of order-flow
imbalance at the fill does **not** separate adverse from benign fills (0.955 vs
0.959) — E6's "drift ≠ 0 once conditioned on OFI/queue" is confirmed: it is not
explained away.

Part of the h=1 effect is mechanical: a passive fill is by construction the
moment the opposite side traded through the level, and if the level is consumed
the mid steps against the order at that very event. The h=5/h=25 persistence is
the economically relevant part.

## 7. E7 — execution: fair RL re-verification (`docs/results/rl_fairness.md`)

Setup (plan_2.md §6, all six items): PPO trained **only** on the `highvol`
regime; evaluated on `highvol` + five hold-outs (`calm`, `lowvol`,
`highvol_null` — the random-walk null arm with symmetric gaps, `trending`,
`liquidity_shock`); **fees + queue model on**; **symmetric information** in both
modes (`novol`: nobody sees a regime flag; `volsym`: the flag is in `obs[44]`
*and* every baseline reads it); fair baselines `schedule_twap` (U-shaped volume
curve with catch-up), `adaptive_pov`, `is_aware` beside the legacy
`twap`/`vwap`/`pov`/`passive`; **5 training seeds × 5 evaluation seed families
× 20 episodes**; block-bootstrap CIs; a **paired** per-episode difference
`best baseline − PPO` on identical seeded tapes.

| regime (novol) | PPO shortfall (bps) | best baseline | paired Δ vs best, seeds significant better / worse |
|---|---|---|---|
| highvol (train) | 2.60 ± 0.08 | adaptive_pov 2.65 | 0 / 0 (Δ −0.06 … +0.17) |
| highvol_null | 2.55 ± 0.15 | adaptive_pov 2.59 | 0 / 0 |
| trending | 2.89 ± 0.13 | adaptive_pov 2.91 | 0 / 0 |
| calm | 1.61 ± 0.02 | twap 1.47 | **0 / 5** (PPO worse, −0.12 … −0.16) |
| lowvol | 0.96 ± 0.01 | adaptive_pov 0.90 | **0 / 5** (PPO worse, −0.05 … −0.08) |
| liquidity_shock | 3.05 ± 0.31 | adaptive_pov 3.74 | **5 / 0** (PPO better, +0.40 … +1.10) |

The `volsym` mode gives the same picture (1/5 better and 1/5 worse on highvol,
4/5 better on liquidity_shock, 5/5 worse on calm/lowvol).

**Conclusion.** Against a defensible baseline with symmetric information and
costs on, the PPO agent has **no significant edge** in the regime it was trained
on or on the null arm, is **significantly worse** on the calm hold-outs, and is
significantly better **only** when liquidity evaporates (it learned to cross
early when the book thins — the one regime where a static schedule is
genuinely wrong). The Part-1 "+50.4 % lower slippage than VWAP" compared
against a 2-line heuristic that could not see the regime; that number is
retired. The random-walk null arm behaves as designed (PPO ≈ baselines).

## 8. Negative results (what we tried that did not work)

1. **The headline RL number does not survive fairness** (§7). Not a bug — an
   asymmetric-information, single-seed, self-VWAP comparison against a weak
   baseline.
2. **Microprice is not better than imbalance** (E2) on 1-tick books; the two
   are the same signal.
3. **Spread has no predictive content** for the mid direction (IC ≈ 0 with CIs
   straddling 0 on AAPL; 0.02–0.03 on QQQ).
4. **Single-event order-level OFI is weaker than the L2 approximation at h=1**
   (it is zero on most events); it only earns its keep as a rolling flow at
   longer horizons, and on QQQ adds nothing to imbalance in the combination.
5. **The logistic fill model has little skill beyond the base rate** (+0.08 /
   +0.016 Brier skill) even though it is well calibrated — passive fill
   probability at placement is mostly about distance to the touch and is
   dominated by cancellations.
6. **Conditioning on order-flow sign does not explain away adverse selection**
   (E6): P(adverse) is ≈ 0.96 in every OFI bucket.
7. **The synthetic random-walk vignette stays null** (`research_vignette.py`):
   IC ≈ 0 with CIs straddling 0 for OFI / microprice / deep imbalance; the
   small spurious L1-imbalance drift (~0.1) is an artifact of random adds
   landing around the walk, and is what the harness is expected to show.

## 9. Reproduce

```bash
python python_quant/scripts/run_all.py                     # everything (fetch ≈ 14 min, E1–E6 ≈ 22 min, E7 ≈ 18 min)
python python_quant/scripts/run_all.py --quick             # smoke of every stage in minutes
python python_quant/scripts/fetch_itch.py --day 12302019 --symbols AAPL,QQQ
python python_quant/scripts/run_research.py --day 12302019 --symbols AAPL,QQQ
python python_quant/scripts/rl_fairness_study.py
```

Results land in `docs/results/` (`real_tape_<day>.md` + per-symbol JSON,
`rl_fairness.md` + JSON). One day, two symbols — more days are one
`fetch_itch.py --day` away.

## 10. Definitions of done that must be true for any claim to leave this repo

1. Walk-forward split, event-indexed sampling, block-bootstrap CIs. ✅
2. Symmetric information in every RL comparison. ✅ (`test_rl_fairness.py`)
3. IS vs **market** VWAP (never self-executed VWAP) with fees + queue on. ✅
4. At least one negative/unstable result reported. ✅ (§8)
5. Every headline restated as `point estimate (CI)` — never a bare number. ✅
