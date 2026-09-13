"""Execution baselines that share OrderBookEnv's action interface.

Two families:

* the Phase-1c heuristics ``twap`` / ``vwap`` / ``pov`` / ``passive``
  (kept byte-identical — they are the historical reference);
* the Phase-3 **fair baselines** (plan_2.md §6 item 4): ``schedule_twap``
  (volume-curve schedule with catch-up), ``adaptive_pov`` (participation
  reacting to spread / flow / regime), ``is_aware`` (implementation-shortfall
  rule: lock in favorable mid moves, slow down on adverse ones, never fall
  behind the schedule). All three read only what the agent's observation
  exposes — the ladder, inventory, time left, mark-to-market and spread —
  plus ``regime_indicator(env)``, which is the SAME regime flag the agent
  sees at ``obs[44]`` when ``vol_feature`` is on (and ``None`` when it is off),
  so information is symmetric in both modes (§6 item 3).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import pi, sin
from typing import Literal

from .envs.order_book_env import OrderBookEnv

AgentId = Literal[
    "twap", "vwap", "pov", "passive",
    "schedule_twap", "adaptive_pov", "is_aware",
]
LEGACY_BASELINES: tuple[str, ...] = ("twap", "vwap", "pov", "passive")
FAIR_BASELINES: tuple[str, ...] = ("schedule_twap", "adaptive_pov", "is_aware")
ALL_BASELINES: tuple[str, ...] = LEGACY_BASELINES + FAIR_BASELINES


def regime_indicator(env: OrderBookEnv) -> bool | None:
    """The regime flag the agent observes (``obs[44]``) — or ``None`` when the
    env does not expose it. Baselines may only use what the agent can see."""
    if getattr(env, "vol_feature", False):
        return bool(getattr(env, "_volatile", False))
    return None


def volume_curve_target(t_frac: float, *, u_weight: float = 0.6) -> float:
    """Fraction of the parent order a U-shaped intraday volume curve has done
    by ``t_frac`` of the horizon (0 → 0, 1 → 1).

    Closed-form CDF of the density ``1 + w·cos(2πt)`` on [0, 1] (heavier at
    the open and the close, integrates to 1). ``u_weight=0`` is a flat TWAP
    schedule; ``|u_weight| < 1`` keeps the density positive.
    """
    t = min(1.0, max(0.0, float(t_frac)))
    return t + u_weight * sin(2.0 * pi * t) / (2.0 * pi)


def policy_action(name: AgentId, env: OrderBookEnv) -> float:
    s = env.book.view()
    spr = 2
    if int(s["bid_px"][0]) and int(s["ask_px"][0]):
        spr = int(s["ask_px"][0]) - int(s["bid_px"][0])
    t_frac = env.t / env.horizon
    last_sz = int(s.get("last_trade_sz") or 0)
    if name == "twap":
        return -1.0 if t_frac > 0.72 else 0.12
    if name == "vwap":
        vol = min(1.0, last_sz / 120.0)
        return -1.0 if t_frac > 0.8 else 0.35 - vol * 0.9
    if name == "pov":
        if spr <= 1 and t_frac > 0.25:
            return -0.55
        return -0.95 if t_frac > 0.7 else 0.2
    if name == "passive":
        return -1.0 if t_frac > 0.92 else 0.7
    if name in FAIR_BASELINES:
        return _fair_action(name, env, s, spr, t_frac, last_sz)
    raise ValueError(name)


def _fair_action(name: str, env: OrderBookEnv, s, spr: int, t_frac: float, last_sz: int) -> float:
    inv_frac = env.inventory / max(1, env.inventory0)
    done_frac = 1.0 - inv_frac
    volatile = regime_indicator(env)
    if name == "schedule_twap":
        # U-shaped volume curve; behind schedule -> cross, ahead -> rest at touch
        target = volume_curve_target((env.t + 1) / env.horizon, u_weight=0.6)
        behind = target - done_frac
        if t_frac > 0.9 or behind > 0.08:
            return -1.0
        if behind > 0.02:
            return -0.4  # inside the spread / at the bid: capped market
        return 0.05 if spr > 1 else 0.1
    if name == "adaptive_pov":
        # participation reacts to observed flow and spread; a wide spread or a
        # volatile flag means taking is expensive -> post; thin spread -> take
        flow = min(1.0, last_sz / 150.0)
        aggressive = flow > 0.5 or spr <= 1
        if volatile is True:
            aggressive = aggressive or t_frac > 0.5
        target = min(1.0, t_frac * (1.35 if aggressive else 0.9))
        behind = target - done_frac
        if t_frac > 0.85 or behind > 0.10:
            return -1.0
        if aggressive and behind > 0.0:
            return -0.5
        return 0.25 if spr > 2 else 0.1
    if name == "is_aware":
        # implementation-shortfall rule: mid above arrival -> lock it in (sell
        # faster); mid below arrival -> slow down unless behind the linear schedule
        mid = env._mid(s) or env.arrival_mid
        edge_bps = (mid - env.arrival_mid) / max(1.0, env.arrival_mid) * 1e4
        behind = t_frac - done_frac
        if t_frac > 0.9 or behind > 0.15:
            return -1.0
        if edge_bps > 1.0:
            return -0.45 if behind > -0.1 else -0.1
        if edge_bps < -1.0 and behind < 0.05:
            return 0.35  # adverse move: rest deeper, wait for reversion
        return -0.2 if behind > 0.05 else 0.1
    raise ValueError(name)


@dataclass
class EpisodeResult:
    name: str
    reward: float
    shortfall_bps: float
    vwap: float
    arrival: int
    leftover: int
    filled: int
    steps: int


def run_episode(
    env: OrderBookEnv,
    name: AgentId,
    *,
    seed: int | None = None,
    policy: Callable[[OrderBookEnv], float] | None = None,
) -> EpisodeResult:
    env.reset(seed=seed)
    total = 0.0
    act = policy or (lambda e: policy_action(name, e))
    while True:
        a = act(env)
        _obs, r, term, trunc, info = env.step(a)
        total += float(r)
        if term or trunc:
            filled = env.inventory0 - env.inventory
            return EpisodeResult(
                name=name,
                reward=total,
                shortfall_bps=float(info["shortfall_bps"]),
                vwap=float(info["vwap"]),
                arrival=int(info["arrival_mid"]),
                leftover=int(env.inventory),
                filled=int(filled),
                steps=int(env.t),
            )


def compare(seed: int = 0x51ED, names: tuple[str, ...] = LEGACY_BASELINES, **env_kw) -> list[EpisodeResult]:
    rows = []
    for name in names:
        rows.append(run_episode(OrderBookEnv(seed=seed, **env_kw), name, seed=seed))  # type: ignore[arg-type]
    return rows
