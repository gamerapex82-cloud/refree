"""Phase 3 — RL fairness harness: regimes, fair baselines, symmetric information, CIs.

Every check here guards one item of the plan_2.md §6 rework spec:
  1/2. named regimes are plain env kwargs; the null arm differs from highvol
       ONLY in gap direction; the training regime is excluded from hold-outs;
  3.   information is symmetric — baselines read the same regime flag the
       agent gets in obs[44], and read nothing when the agent gets nothing;
  4.   fair baselines complete the parent order and finish on time;
  5.   evaluation runs identical seeded tapes for every strategy and reports
       block-bootstrap CIs; the paired CI is exactly zero for self-vs-self.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexus_quant.agents import (
    ci_from_rows,
    evaluate_regime_ci,
    format_regime_table,
    paired_difference_ci,
    run_regime_episodes,
)
from nexus_quant.baselines import (
    ALL_BASELINES,
    FAIR_BASELINES,
    LEGACY_BASELINES,
    policy_action,
    regime_indicator,
    run_episode,
    volume_curve_target,
)
from nexus_quant.envs.order_book_env import OBS_DIM, OrderBookEnv
from nexus_quant.envs.regimes import (
    COSTS_ON,
    HOLDOUT_REGIMES,
    REGIMES,
    TRAIN_REGIME,
    make_regime,
    regime_factories,
    regime_kwargs,
)


# ------------------------------------------------------------------ regimes
def test_regime_catalogue_shape():
    assert TRAIN_REGIME == "highvol" and TRAIN_REGIME not in HOLDOUT_REGIMES
    assert set(HOLDOUT_REGIMES) == set(REGIMES) - {TRAIN_REGIME}
    assert {"calm", "lowvol", "highvol", "highvol_null", "trending", "liquidity_shock"} <= set(REGIMES)
    # the null arm is the headline regime with symmetric gaps and NOTHING else changed
    diff = {k for k in set(REGIMES["highvol"]) | set(REGIMES["highvol_null"])
            if REGIMES["highvol"].get(k) != REGIMES["highvol_null"].get(k)}
    assert diff == {"gap_down_prob"} and REGIMES["highvol_null"]["gap_down_prob"] == 0.5
    with pytest.raises(KeyError):
        regime_kwargs("nope")


def test_regime_kwargs_costs_and_vol_feature_toggle():
    kw = regime_kwargs("calm", costs=True)
    assert all(kw[k] == v for k, v in COSTS_ON.items()) and kw["vol_feature"] is False
    kw = regime_kwargs("calm", costs=False, vol_feature=True, seed=7)
    assert "fee_bps" not in kw and kw["vol_feature"] is True and kw["seed"] == 7
    env = make_regime("highvol", vol_feature=True)()
    assert env.observation_space.shape == (OBS_DIM + 1,) and env.queue_model == "uniform"
    env44 = make_regime("highvol")()
    assert env44.observation_space.shape == (OBS_DIM,)
    fs = regime_factories(("calm", "trending"))
    assert list(fs) == ["calm", "trending"] and all(isinstance(f(), OrderBookEnv) for f in fs.values())
    assert next(iter(regime_factories())) == TRAIN_REGIME


def test_regimes_share_mechanics_and_differ_only_in_flow():
    """Identical seed + identical actions on calm vs highvol_null: same arrival
    book (flow differs only from step 1), so mechanics are shared."""
    a = make_regime("calm", costs=True)()
    b = make_regime("highvol_null", costs=True)()
    oa, _ = a.reset(seed=3)
    ob, _ = b.reset(seed=3)
    np.testing.assert_array_equal(oa, ob)  # same seeded arrival book
    assert a.fee_bps == b.fee_bps == COSTS_ON["fee_bps"]


# ------------------------------------------------------------------ baselines
def test_volume_curve_is_a_cdf():
    assert volume_curve_target(0.0) == pytest.approx(0.0)
    assert volume_curve_target(1.0) == pytest.approx(1.0)
    assert volume_curve_target(0.5) == pytest.approx(0.5)
    xs = np.linspace(0.0, 1.0, 201)
    ys = np.array([volume_curve_target(x) for x in xs])
    assert np.all(np.diff(ys) > 0)  # strictly increasing -> positive density
    # front-loaded: more than a flat TWAP would have done by 25 %
    assert volume_curve_target(0.25) > 0.25
    assert volume_curve_target(0.3, u_weight=0.0) == pytest.approx(0.3)


@pytest.mark.parametrize("name", ALL_BASELINES)
def test_every_baseline_completes_on_time(name):
    assert set(LEGACY_BASELINES) | set(FAIR_BASELINES) == set(ALL_BASELINES)
    env = make_regime("calm", costs=True)()
    res = run_episode(env, name, seed=11)
    assert res.filled + res.leftover == env.inventory0
    assert res.steps <= env.horizon
    assert res.leftover == 0  # the calm regime is liquid enough to finish
    a = policy_action(name, env)
    assert -1.0 <= a <= 1.0


def test_fair_baselines_beat_or_match_passive_completion_in_highvol():
    """The fair baselines exist to be *defensible*: they must not leave large
    residuals behind where the 2-line heuristics do."""
    env = make_regime("highvol", costs=True)()
    left = {n: np.mean([run_episode(env, n, seed=200 + i).leftover for i in range(10)]) for n in ALL_BASELINES}
    assert max(left[n] for n in FAIR_BASELINES) <= left["passive"]
    assert left["schedule_twap"] < 100.0


def test_regime_indicator_is_symmetric_with_the_agent_observation():
    env = make_regime("highvol", vol_feature=True)()
    env.reset(seed=5)
    for _ in range(12):
        obs, *_ = env.step(0.1)
        flag = regime_indicator(env)
        assert flag is not None and float(obs[44]) == (1.0 if flag else 0.0)
    env44 = make_regime("highvol", vol_feature=False)()
    env44.reset(seed=5)
    env44.step(0.1)
    assert regime_indicator(env44) is None  # nothing to read when the agent sees nothing


def test_baseline_actions_ignore_the_flag_when_the_agent_cannot_see_it():
    """With vol_feature off, a baseline's decision must not depend on the
    hidden regime state — we flip it by hand and assert the action is unchanged."""
    env = make_regime("highvol", vol_feature=False)()
    env.reset(seed=9)
    for _ in range(5):
        env.step(0.0)
    for name in FAIR_BASELINES:
        env._volatile = False
        a0 = policy_action(name, env)
        env._volatile = True
        a1 = policy_action(name, env)
        assert a0 == a1, name


# ------------------------------------------------------------------ evaluation
class _Const:
    """Deterministic 'policy' with the PPOPolicy.act signature."""

    def __init__(self, a: float) -> None:
        self.a = a

    def act(self, obs, *, deterministic: bool = True) -> float:
        return self.a


def test_run_regime_episodes_identical_tapes_and_row_schema():
    fs = regime_factories(("calm",))
    eps = run_regime_episodes(_Const(0.1), fs, seeds=2, episodes_per_seed=3, baselines=("twap",), agent_name="agent")
    rows_a, rows_b = eps["calm"]["agent"], eps["calm"]["twap"]
    assert len(rows_a) == len(rows_b) == 6
    assert [r["seed"] for r in rows_a] == [r["seed"] for r in rows_b]
    assert [r["seed_family"] for r in rows_a] == [0, 0, 0, 1, 1, 1]
    assert {"seed", "reward", "shortfall_bps", "is_bps", "vwap_slip_bps", "leftover", "completion", "seed_family"} <= set(rows_a[0])
    # is_bps (execution.metrics) and the env's shortfall_bps are the same number
    for r in rows_a + rows_b:
        assert r["is_bps"] == pytest.approx(r["shortfall_bps"], abs=1e-9)
        assert 0.0 <= r["completion"] <= 1.0
    with pytest.raises(ValueError):
        run_regime_episodes(None, fs, seeds=1, episodes_per_seed=1)


def test_ci_from_rows_block_bootstrap_and_regime_table():
    rows = [{"x": float(v), "seed_family": i // 5} for i, v in enumerate(range(20))]
    ci = ci_from_rows(rows, metric="x", name="s", regime="r", n_boot=300)
    assert ci.mean == pytest.approx(9.5) and ci.n_episodes == 20 and ci.n_seeds == 4
    assert ci.per_seed_mean == [2.0, 7.0, 12.0, 17.0]
    assert ci.ci95[0] <= ci.mean <= ci.ci95[1]
    txt = format_regime_table({"r": {"s": ci}})
    assert "regime r" in txt and "9.500" in txt


def test_evaluate_regime_ci_and_paired_difference_are_consistent():
    fs = regime_factories(("calm", "highvol_null"))
    pol = _Const(0.12)
    res = evaluate_regime_ci(pol, fs, seeds=2, episodes_per_seed=4, baselines=("twap",), n_boot=200)
    for regime in fs:
        for name in ("ppo", "twap"):
            ci = res[regime][name]
            assert ci.n_episodes == 8 and ci.n_seeds == 2 and ci.ci95[0] <= ci.mean <= ci.ci95[1]
    # paired self-vs-self difference is exactly zero with a zero-width CI
    eps = run_regime_episodes(pol, fs, seeds=2, episodes_per_seed=4, baselines=("twap",))
    for regime in fs:
        eps[regime]["ppo_copy"] = [dict(r) for r in eps[regime]["ppo"]]
    d = paired_difference_ci(pol, "ppo_copy", fs, seeds=2, episodes_per_seed=4, n_boot=100, episodes=eps)
    for regime in fs:
        assert d[regime] == {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 8.0, "frac_agent_better": 0.0,
                             "pct_vs_baseline": pytest.approx(0.0)}
    # and the real paired number equals the difference of the two pooled means
    d2 = paired_difference_ci(pol, "twap", fs, seeds=2, episodes_per_seed=4, n_boot=100, episodes=eps)
    for regime in fs:
        assert d2[regime]["mean"] == pytest.approx(res[regime]["twap"].mean - res[regime]["ppo"].mean, abs=1e-9)
        assert d2[regime]["lo"] <= d2[regime]["mean"] <= d2[regime]["hi"]


def test_paired_difference_rejects_mismatched_tapes():
    fs = regime_factories(("calm",))
    eps = run_regime_episodes(_Const(0.0), fs, seeds=1, episodes_per_seed=3, baselines=("twap",))
    eps["calm"]["twap"][0]["seed"] += 1
    with pytest.raises(ValueError):
        paired_difference_ci(_Const(0.0), "twap", fs, seeds=1, episodes_per_seed=3, episodes=eps)
