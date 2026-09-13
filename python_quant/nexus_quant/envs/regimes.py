"""Named market regimes for per-regime RL evaluation (plan_2.md §6 items 1–2).

Every regime is a plain ``OrderBookEnv`` kwargs dict built from the existing
constructor knobs — no new mechanics, so the fill / cost / penalty code path
is **identical** across regimes and only the exogenous flow differs. The
agent is trained on ONE regime (``highvol`` by default) and evaluated on all
of them; everything except the training regime is a hold-out.

``highvol_null`` is the **random-walk null arm**: the high-vol regime with
symmetric gap direction (``gap_down_prob=0.5``). The Part-1 headline regime
gapped *down* 75% of the time — a directional drift a seller can learn to
front-run. Under the null arm there is no drift to exploit, so an agent whose
edge is real *timing skill* keeps it while one whose edge is *drift
anticipation* falls back to the baselines. "PPO ≈ baselines" is the correct
result here, not a failure.

``COSTS_ON`` is the fees + queue overlay every reported number must use
(work package §3.3 item 4).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .order_book_env import OrderBookEnv

_HIGHVOL: dict[str, Any] = {
    "regime_prob": 0.08,
    "vol_decay": 0.15,
    "gap_prob": 0.20,
    "gap_min": 800,
    "gap_max": 1800,
    "gap_down_prob": 0.75,
    "vol_take_prob": 0.55,
    "vol_take_min": 60,
    "vol_take_max": 220,
    "vol_add_min": 15,
    "vol_add_max": 70,
    "vol_add_offset_min": 2,
    "vol_add_offset_max": 12,
    "vol_events_min": 3,
    "vol_events_max": 8,
}

REGIMES: dict[str, dict[str, Any]] = {
    # the original gentle-walk sim (Phase 1c "calm")
    "calm": {},
    # quiet tape: always in the "volatile" code path but with small takes,
    # thick near-touch adds and no gaps -> lower realized vol than calm
    "lowvol": {
        "regime_prob": 1.0, "vol_decay": 0.0, "gap_prob": 0.0,
        "vol_take_prob": 0.20, "vol_take_min": 10, "vol_take_max": 40,
        "vol_add_min": 60, "vol_add_max": 220,
        "vol_add_offset_min": 1, "vol_add_offset_max": 5,
        "vol_events_min": 2, "vol_events_max": 5,
    },
    # the Part-1 headline regime (Markov switching + 75%-down gap events)
    "highvol": dict(_HIGHVOL),
    # random-walk null arm: identical mechanics, symmetric gaps -> no drift
    "highvol_null": {**_HIGHVOL, "gap_down_prob": 0.5},
    # persistent one-way pressure: every gap is a sell-off
    "trending": {**_HIGHVOL, "gap_prob": 0.30, "gap_down_prob": 1.0},
    # thin book under heavy taking, no gaps: liquidity evaporates near touch
    "liquidity_shock": {
        "regime_prob": 0.25, "vol_decay": 0.10, "gap_prob": 0.0,
        "vol_take_prob": 0.60, "vol_take_min": 60, "vol_take_max": 180,
        "vol_add_min": 20, "vol_add_max": 90,
        "vol_add_offset_min": 3, "vol_add_offset_max": 14,
        "vol_events_min": 3, "vol_events_max": 7,
    },
}

# Fees + queue model ON — the honest-reporting overlay (work package §3.3 #4).
COSTS_ON: dict[str, Any] = {
    "fee_bps": 0.3,
    "rebate_bps": 0.2,
    "impact_coef": 1.0,
    "impact_participation": 0.1,
    "queue_model": "uniform",
}

TRAIN_REGIME = "highvol"
HOLDOUT_REGIMES: tuple[str, ...] = tuple(r for r in REGIMES if r != TRAIN_REGIME)


def regime_kwargs(name: str, *, costs: bool = True, vol_feature: bool = False, **overrides: Any) -> dict[str, Any]:
    """Constructor kwargs for regime ``name`` (+ cost overlay, + overrides)."""
    if name not in REGIMES:
        raise KeyError(f"unknown regime {name!r}; known: {sorted(REGIMES)}")
    kw: dict[str, Any] = dict(REGIMES[name])
    if costs:
        kw.update(COSTS_ON)
    kw["vol_feature"] = bool(vol_feature)
    kw.update(overrides)
    return kw


def make_regime(name: str, *, costs: bool = True, vol_feature: bool = False, **overrides: Any) -> Callable[[], OrderBookEnv]:
    """Env factory for ``name``; every call builds a fresh env with identical kwargs."""
    kw = regime_kwargs(name, costs=costs, vol_feature=vol_feature, **overrides)
    return lambda: OrderBookEnv(**kw)


def regime_factories(
    names: tuple[str, ...] | list[str] | None = None,
    *,
    costs: bool = True,
    vol_feature: bool = False,
    **overrides: Any,
) -> dict[str, Callable[[], OrderBookEnv]]:
    """``{name: factory}`` for ``names`` (default: every regime, training one first)."""
    names = tuple(names) if names is not None else (TRAIN_REGIME, *HOLDOUT_REGIMES)
    return {n: make_regime(n, costs=costs, vol_feature=vol_feature, **overrides) for n in names}
