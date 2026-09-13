"""Evaluation harness: trained PPO agent vs. the execution baselines.

The headline resume metric is **lower implementation-shortfall slippage than
VWAP**. ``shortfall_bps`` (from ``OrderBookEnv`` info) is ``(arrival_mid −
vwap) / arrival_mid × 1e4`` — how much the child execution conceded relative
to the arrival mid, in basis points. Lower is better.

To make the comparison fair every strategy runs the **same seeded episodes**:
episode *i* = ``seed + i``, which reproduces identical initial books and
exogenous flow for TWAP/VWAP/POV/Passive and the agent alike, so the only
difference in results is the policy, not the tape.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from ..envs.order_book_env import OrderBookEnv

BaselineId = Literal[
    "twap", "vwap", "pov", "passive",
    "schedule_twap", "adaptive_pov", "is_aware",
]


class Policy(Protocol):
    """Anything with ``act(obs, deterministic=True) -> float`` (PPOPolicy)."""

    def act(self, obs: np.ndarray, *, deterministic: bool = True) -> float: ...


@dataclass
class EvalSummary:
    name: str
    reward_mean: float
    shortfall_bps_mean: float
    shortfall_bps_std: float
    leftover_mean: float
    n: int


def _episode_shortfall(
    env: OrderBookEnv, act_fn: Callable[[np.ndarray], float], seed: int
) -> tuple[float, float, int]:
    """One full episode: reset(seed), then follow ``act_fn`` to the end."""
    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float64)
    total = 0.0
    while True:
        a = act_fn(obs)
        obs, r, term, trunc, info = env.step(a)
        obs = np.asarray(obs, dtype=np.float64)
        total += float(r)
        if term or trunc:
            return total, float(info["shortfall_bps"]), int(env.inventory)


def evaluate_policy(
    policy: Policy,
    *,
    n_episodes: int = 50,
    seed: int = 0,
    env_factory: Callable[[], OrderBookEnv] = OrderBookEnv,
    deterministic: bool = True,
) -> tuple[list[dict], EvalSummary]:
    """Run ``policy`` over ``n_episodes`` seeded episodes; return rows + summary.

    Rows are dicts for easy tabulation; the summary carries the mean
    shortfall (the slippage metric) and its scatter.
    """
    env = env_factory()
    rows: list[dict] = []
    rewards: list[float] = []
    sfs: list[float] = []
    leftovers: list[int] = []
    for i in range(n_episodes):
        total, sf, leftover = _episode_shortfall(
            env, lambda ob: policy.act(ob, deterministic=deterministic), int(seed) + i
        )
        rewards.append(total)
        sfs.append(sf)
        leftovers.append(leftover)
        rows.append({"name": policy.__class__.__name__, "reward": total, "shortfall_bps": sf, "leftover": leftover})
    summary = EvalSummary(
        name=policy.__class__.__name__,
        reward_mean=float(np.mean(rewards)),
        shortfall_bps_mean=float(np.mean(sfs)),
        shortfall_bps_std=float(np.std(sfs)),
        leftover_mean=float(np.mean(leftovers)),
        n=n_episodes,
    )
    return rows, summary


def _baseline_summary(
    name: BaselineId,
    n_episodes: int,
    seed: int,
    env_factory: Callable[[], OrderBookEnv] = OrderBookEnv,
) -> EvalSummary:
    from ..baselines import run_episode

    env = env_factory()
    rewards: list[float] = []
    sfs: list[float] = []
    leftovers: list[int] = []
    for i in range(n_episodes):
        res = run_episode(env, name, seed=int(seed) + i)
        rewards.append(res.reward)
        sfs.append(res.shortfall_bps)
        leftovers.append(res.leftover)
    return EvalSummary(
        name=name,
        reward_mean=float(np.mean(rewards)),
        shortfall_bps_mean=float(np.mean(sfs)),
        shortfall_bps_std=float(np.std(sfs)),
        leftover_mean=float(np.mean(leftovers)),
        n=n_episodes,
    )


def strategy_table(
    agent: Policy | None = None,
    *,
    agent_name: str = "ppo",
    n_episodes: int = 50,
    seed: int = 0,
    baselines: tuple[BaselineId, ...] = ("twap", "vwap", "pov", "passive"),
    env_factory: Callable[[], OrderBookEnv] = OrderBookEnv,
) -> list[dict]:
    """Compare agent + baselines on the same seeded episodes.

    Returns one dict per strategy:
    ``name, reward_mean, shortfall_bps_mean, shortfall_bps_std, vs_vwap_bps``
    where ``vs_vwap_bps`` is the signed *reduction* in shortfall relative to
    VWAP (positive = agent/baseline is *better* than VWAP).
    """
    rows: list[dict] = []
    vwap_sf: float | None = None
    if agent is not None:
        _, a_sum = evaluate_policy(
            agent, n_episodes=n_episodes, seed=seed,
            env_factory=env_factory, deterministic=True,
        )
        a_sum.name = agent_name
        rows.append(a_sum)
    for name in baselines:
        b = _baseline_summary(name, n_episodes, seed, env_factory=env_factory)
        if name == "vwap":
            vwap_sf = b.shortfall_bps_mean
        rows.append(b)
    out = []
    for r in rows:
        d = r.__dict__.copy()
        if vwap_sf is not None:
            d["vs_vwap_bps"] = vwap_sf - r.shortfall_bps_mean
            d["vs_vwap_pct"] = (vwap_sf - r.shortfall_bps_mean) / max(vwap_sf, 1e-9) * 100.0
        else:
            d["vs_vwap_bps"] = 0.0
            d["vs_vwap_pct"] = 0.0
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Phase 3 — per-regime, multi-seed, CI-reported evaluation (plan_2.md §6)
# ---------------------------------------------------------------------------
@dataclass
class RegimeCI:
    """Mean ± block-bootstrap 95% CI of one metric for one strategy in one regime."""

    name: str
    regime: str
    metric: str
    mean: float
    ci95: tuple[float, float]
    n_episodes: int
    n_seeds: int
    per_seed_mean: list[float]


def _episode_rows(
    env: OrderBookEnv,
    act_fn: Callable[[OrderBookEnv, np.ndarray], float],
    seeds: Sequence[int],
) -> list[dict]:
    """Full episodes for every seed; one row per episode with the honest metric set."""
    from ..execution.metrics import implementation_shortfall, vwap_slippage

    rows: list[dict] = []
    for sd in seeds:
        obs, _ = env.reset(seed=int(sd))
        obs = np.asarray(obs, dtype=np.float64)
        total = 0.0
        while True:
            a = act_fn(env, obs)
            obs, r, term, trunc, info = env.step(a)
            obs = np.asarray(obs, dtype=np.float64)
            total += float(r)
            if term or trunc:
                break
        market_vwap = float(info.get("market_vwap", 0.0) or 0.0)
        rows.append(
            {
                "seed": int(sd),
                "reward": total,
                "shortfall_bps": float(info["shortfall_bps"]),
                "is_bps": implementation_shortfall(env.fills, env.arrival_mid, side=1),
                "vwap_slip_bps": vwap_slippage(env.fills, market_vwap, side=1),
                "leftover": int(env.inventory),
                "completion": 1.0 - env.inventory / max(1, env.inventory0),
            }
        )
    return rows


def _policy_act(policy: Policy, deterministic: bool) -> Callable[[OrderBookEnv, np.ndarray], float]:
    return lambda _env, ob: policy.act(ob, deterministic=deterministic)


def _baseline_act(name: str) -> Callable[[OrderBookEnv, np.ndarray], float]:
    from ..baselines import policy_action

    return lambda env, _ob: policy_action(name, env)  # type: ignore[arg-type]


def run_regime_episodes(
    policy: Policy | None,
    regimes: Mapping[str, Callable[[], OrderBookEnv]],
    *,
    seeds: int = 5,
    episodes_per_seed: int = 20,
    seed0: int = 0x5EED,
    baselines: Sequence[str] = (),
    agent_name: str = "ppo",
    deterministic: bool = True,
) -> dict[str, dict[str, list[dict]]]:
    """Run every strategy over identical seeded episodes in every regime.

    Seed family ``k`` (``k < seeds``) covers episode seeds ``seed0 + k·10_000 +
    i`` for ``i < episodes_per_seed``; the agent AND every named baseline see
    the same tapes, so a difference between two strategies is never a
    difference in the flow. Returns ``{regime: {strategy: [row, ...]}}`` with
    one row per episode (``seed, reward, shortfall_bps, is_bps, vwap_slip_bps,
    leftover, completion``), ordered by seed family then episode.
    """
    strategies: list[tuple[str, Callable[[OrderBookEnv, np.ndarray], float]]] = []
    if policy is not None:
        strategies.append((agent_name, _policy_act(policy, deterministic)))
    for b in baselines:
        strategies.append((str(b), _baseline_act(str(b))))
    if not strategies:
        raise ValueError("need a policy and/or at least one baseline")
    seed_families = [
        [int(seed0) + k * 10_000 + i for i in range(episodes_per_seed)] for k in range(int(seeds))
    ]
    out: dict[str, dict[str, list[dict]]] = {}
    for regime, factory in regimes.items():
        out[regime] = {}
        for name, act in strategies:
            env = factory()
            rows: list[dict] = []
            for k, fam in enumerate(seed_families):
                for r in _episode_rows(env, act, fam):
                    r["seed_family"] = k
                    rows.append(r)
            out[regime][name] = rows
    return out


def ci_from_rows(
    rows: Sequence[dict],
    *,
    metric: str,
    name: str,
    regime: str,
    n_boot: int = 2000,
) -> RegimeCI:
    """Mean ± moving-block-bootstrap 95% CI of ``metric`` over episode rows.

    The block length is one seed family, so seed-family dependence is
    respected instead of assumed away (plan_2.md §5: never i.i.d. bootstrap on
    dependent draws).
    """
    from ..research.experiments import bootstrap_ci

    vals = [float(r[metric]) for r in rows]
    fams = sorted({int(r.get("seed_family", 0)) for r in rows})
    per_seed = [
        float(np.mean([float(r[metric]) for r in rows if int(r.get("seed_family", 0)) == k])) for k in fams
    ]
    block = max(1, len(vals) // max(1, len(fams)))
    ci = bootstrap_ci(vals, n_boot=n_boot, kind="block", block=block, seed=0x51ED)
    return RegimeCI(
        name=name, regime=regime, metric=metric, mean=float(np.mean(vals)),
        ci95=(ci["lo"], ci["hi"]), n_episodes=len(vals), n_seeds=len(fams), per_seed_mean=per_seed,
    )


def evaluate_regime_ci(
    policy: Policy | None,
    regimes: Mapping[str, Callable[[], OrderBookEnv]],
    *,
    seeds: int = 5,
    episodes_per_seed: int = 20,
    seed0: int = 0x5EED,
    metric: str = "shortfall_bps",
    baselines: Sequence[str] = (),
    agent_name: str = "ppo",
    n_boot: int = 2000,
    deterministic: bool = True,
) -> dict[str, dict[str, RegimeCI]]:
    """Per-regime mean ± 95% block-bootstrap CI of ``metric`` (plan_2.md §6 items 1, 2, 5).

    ``regimes`` maps a label to an env factory (see ``envs.regimes``). Every
    strategy runs the **same** seeded episodes (``run_regime_episodes``); the
    CI is a moving-block bootstrap with one seed family per block
    (``ci_from_rows``). Returns ``{regime: {strategy: RegimeCI}}``.

    ``metric`` is one of ``shortfall_bps`` (vs arrival, positive = cost),
    ``is_bps`` (same math via ``execution.metrics``), ``vwap_slip_bps`` (vs
    **market** VWAP), ``reward``, ``leftover``, ``completion``.
    """
    episodes = run_regime_episodes(
        policy, regimes, seeds=seeds, episodes_per_seed=episodes_per_seed, seed0=seed0,
        baselines=baselines, agent_name=agent_name, deterministic=deterministic,
    )
    return {
        regime: {
            name: ci_from_rows(rows, metric=metric, name=name, regime=regime, n_boot=n_boot)
            for name, rows in strategies.items()
        }
        for regime, strategies in episodes.items()
    }


def paired_difference_ci(
    policy: Policy,
    baseline: str,
    regimes: Mapping[str, Callable[[], OrderBookEnv]],
    *,
    seeds: int = 5,
    episodes_per_seed: int = 20,
    seed0: int = 0x5EED,
    metric: str = "shortfall_bps",
    n_boot: int = 2000,
    episodes: Mapping[str, Mapping[str, Sequence[dict]]] | None = None,
    agent_name: str = "ppo",
) -> dict[str, dict[str, float]]:
    """Per-regime CI of the **paired** per-episode difference ``baseline − agent``.

    Positive = the agent has lower ``metric`` (better, for cost metrics) on the
    same tapes. This is the number a README line may quote: it uses identical
    seeded episodes for both sides, so tape noise cancels and the CI reflects
    policy differences only. Pass ``episodes`` (from ``run_regime_episodes``,
    containing both ``agent_name`` and ``baseline``) to avoid re-running.
    Returns ``{regime: {"mean", "lo", "hi", "n", "frac_agent_better",
    "pct_vs_baseline"}}``.
    """
    from ..research.experiments import bootstrap_ci

    if episodes is None:
        episodes = run_regime_episodes(
            policy, regimes, seeds=seeds, episodes_per_seed=episodes_per_seed, seed0=seed0,
            baselines=(baseline,), agent_name=agent_name,
        )
    out: dict[str, dict[str, float]] = {}
    for regime in regimes:
        ra = episodes[regime][agent_name]
        rb = episodes[regime][baseline]
        if len(ra) != len(rb) or any(a["seed"] != b["seed"] for a, b in zip(ra, rb)):
            raise ValueError("paired comparison needs identical episode seeds for both strategies")
        diffs = [float(b[metric]) - float(a[metric]) for a, b in zip(ra, rb)]
        base_vals = [float(b[metric]) for b in rb]
        fams = len({int(r.get("seed_family", 0)) for r in ra})
        ci = bootstrap_ci(diffs, n_boot=n_boot, kind="block", block=max(1, len(diffs) // max(1, fams)), seed=0x51ED)
        base_mean = float(np.mean(base_vals))
        out[regime] = {
            "mean": float(np.mean(diffs)),
            "lo": ci["lo"],
            "hi": ci["hi"],
            "n": float(len(diffs)),
            "frac_agent_better": float(np.mean(np.asarray(diffs) > 0.0)),
            "pct_vs_baseline": float(np.mean(diffs) / base_mean * 100.0) if abs(base_mean) > 1e-12 else float("nan"),
        }
    return out


def format_regime_table(result: Mapping[str, Mapping[str, RegimeCI]]) -> str:
    """Monospace table: one block per regime, ``mean [lo, hi]`` per strategy."""
    lines: list[str] = []
    for regime, strategies in result.items():
        any_ci = next(iter(strategies.values()))
        lines.append(
            f"--- regime {regime} · {any_ci.metric} · {any_ci.n_seeds} seeds × "
            f"{any_ci.n_episodes // max(1, any_ci.n_seeds)} episodes ---"
        )
        lines.append(f"{'strategy':<15}{'mean':>9}{'ci95_lo':>10}{'ci95_hi':>10}{'per-seed means':>34}")
        for name, ci in strategies.items():
            seeds = " ".join(f"{v:6.2f}" for v in ci.per_seed_mean)
            lines.append(f"{name:<15}{ci.mean:>9.3f}{ci.ci95[0]:>10.3f}{ci.ci95[1]:>10.3f}  {seeds:>32}")
    return "\n".join(lines)


def format_table(rows: list[dict]) -> str:
    """Render ``strategy_table`` output as a monospace summary line per row."""
    header = f"{'strategy':<10}{'reward':>10}{'shortfall_bps':>14}{'vs_vwap%':>10}{'leftover':>10}"
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['name']:<10}{r['reward_mean']:>10.2f}{r['shortfall_bps_mean']:>14.3f}"
            f"{r['vs_vwap_pct']:>9.1f}%{r['leftover_mean']:>10.2f}"
        )
    return "\n".join(lines)