"""Experiment + statistical-rigor toolbox (plan_2.md §5).

Everything is NumPy-only and deterministic (seeded RNG). The three comparison
primitives used across E1–E7:

  * rank_ic / icir — primary signal metrics;
  * bootstrap_ci (block)  — CIs that respect autocorrelation (never i.i.d.
    bootstrap on a tape);
  * diebold_mariano      — paired forecast comparison with HAC variance when
    horizons overlap.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import erf, isinf, isnan, sqrt

import numpy as np


# ---------------------------------------------------------------------------
# signal metrics
# ---------------------------------------------------------------------------
def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (standard Spearman) — ties are common on integer-tick labels.

    Vectorized: a tie block occupying sorted positions ``[i, j)`` gets rank
    ``(i + j − 1) / 2`` for every member (identical to the scalar definition).
    """
    n = x.size
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    new_block = np.concatenate(([True], xs[1:] != xs[:-1]))
    starts = np.flatnonzero(new_block)
    ends = np.concatenate((starts[1:], [n]))
    block_rank = (starts + ends - 1) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = block_rank[np.cumsum(new_block) - 1]
    return ranks


def rank_ic(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Spearman rank correlation between prediction and realized outcome."""
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    if a.size != b.size or a.size < 2:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    denom = np.sqrt(((ra - ra.mean()) ** 2).sum() * ((rb - rb.mean()) ** 2).sum())
    if denom <= 0:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).sum() / denom)


def icir(ic_series: Sequence[float], annualize: float = 1.0) -> float:
    """Information ratio of an IC series: mean / std * sqrt(n)."""
    x = np.asarray(ic_series, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    s = x.std(ddof=1)
    if s <= 0:
        return float("nan")
    return float(x.mean() / s * np.sqrt(x.size) * annualize)


def hit_rate(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Fraction of predictions whose sign matches the realized sign."""
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    if a.size != b.size or a.size == 0:
        return float("nan")
    return float(np.mean(np.sign(b) == np.sign(a)))


def decile_spread(y_true: Sequence[float], y_pred: Sequence[float], n: int = 10) -> float:
    """mean(label | top decile of prediction) − mean(label | bottom decile).

    ``n`` is the number of quantile bins (10 = deciles): the top and bottom
    ``size // n`` observations by prediction are averaged, so the statistic
    is a *bin* mean, not the mean of ``n`` extreme points. NaN if fewer than
    ``2n`` observations.
    """
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    if a.size != b.size or a.size < 2 * n:
        return float("nan")
    order = np.argsort(b, kind="mergesort")
    a_sorted = a[order]
    k = max(1, a.size // n)
    top = a_sorted[-k:].mean()
    bot = a_sorted[:k].mean()
    return float(top - bot)


# ---------------------------------------------------------------------------
# bootstrap CIs
# ---------------------------------------------------------------------------
def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    kind: str = "block",
    block: int = 0,
    alpha: float = 0.05,
    seed: int = 0x51ED,
    stat_fn: object | None = None,
) -> dict[str, float]:
    """Bootstrap (1−alpha) CI on the mean of ``values``.

    ``kind="block"`` (default) runs the moving-block bootstrap over
    non-overlapping blocks of length ``block`` (default ``int(sqrt(n))``) to
    preserve autocorrelation. ``kind="iid"`` samples single observations.
    ``stat_fn`` overrides the statistic (default: the sample mean).

    Returns {"lo", "hi", "mean", "se"}.
    """
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if n < 2:
        raise ValueError("bootstrap_ci needs at least 2 observations")
    rng = np.random.default_rng(seed)
    fn = stat_fn if stat_fn is not None else _mean
    blen = block if block > 0 else max(1, int(np.sqrt(n)))
    if kind == "block":
        n_blocks = max(1, (n + blen - 1) // blen)
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        if kind == "block":
            starts = rng.integers(0, n, size=n_blocks)
            idx = _block_indices(starts, blen, n)
            if idx.size == 0:
                idx = np.arange(n)
            sample = x[idx]
        else:  # iid
            sample = x[rng.integers(0, n, size=n)]
        stats[b] = fn(sample)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"lo": float(lo), "hi": float(hi), "mean": float(np.mean(stats)), "se": float(np.std(stats))}


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x))


def _block_indices(starts: np.ndarray, blen: int, n: int) -> np.ndarray:
    """Concatenate ``[s, min(s+blen, n))`` for every block start, truncated to ``n`` draws.

    Same draw as the scalar loop it replaces (block-by-block, clipped at the
    series end), computed without a Python-level list of ranges.
    """
    idx = (starts[:, None] + np.arange(blen, dtype=np.int64)[None, :]).ravel()
    return idx[idx < n][:n] if blen > 0 else np.arange(n)


# ---------------------------------------------------------------------------
# paired comparison
# ---------------------------------------------------------------------------
def diebold_mariano(
    errors_a: Sequence[float],
    errors_b: Sequence[float],
    *,
    h: int = 1,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Diebold–Mariano test that the loss of A is not equal to the loss of B.

    Loss here is the *squared error* of the forecast. The variance of the loss
    differential is HAC-adjusted for overlap (Newey–West with ``lag = h``), so
    the statistic is valid when horizons overlap.

    Returns {"dm", "p_value", "mean_diff"} where ``mean_diff = mean(loss_a −
    loss_b)`` — a negative value means A has lower loss on average.
    """
    a = np.asarray(errors_a, dtype=np.float64)
    b = np.asarray(errors_b, dtype=np.float64)
    if a.size != b.size or a.size < 2:
        raise ValueError("diebold_mariano needs equal-length series of >= 2")
    d = a**2 - b**2
    d = d - d.mean()
    n = d.size
    lag = int(max(1, min(h, n - 1)))
    gamma = np.array([np.mean(d[i:] * d[: n - i]) for i in range(lag + 1)])
    var = gamma[0] + 2 * np.sum(gamma[1:] * (1 - np.arange(1, lag + 1) / (lag + 1)))
    if var <= 0:
        var = np.var(d, ddof=1)
    se = sqrt(var / n)
    mean_diff = float(np.mean(a**2 - b**2))
    dm = float(mean_diff / se) if se > 0 else float("nan")
    if alternative in ("two-sided", "two_sided"):
        p = 2.0 * (1.0 - _normal_cdf(abs(dm)))
    elif alternative == "less":
        p = _normal_cdf(dm)
    else:  # greater
        p = 1.0 - _normal_cdf(dm)
    return {"dm": dm, "p_value": float(p), "mean_diff": mean_diff}


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


# ---------------------------------------------------------------------------
# experiment runner
# ---------------------------------------------------------------------------
def run_experiment(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    split_tags: Sequence[str] | None = None,
    n_boot: int = 2000,
    seed: int = 0x51ED,
) -> dict:
    """E1–E4 result bundle for one feature on one label: JSON-serializable.

    Returns per-split (or overall) rank IC, ICIR, hit rate, decile spread, and
    a block-bootstrap CI of the rank IC.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.size != yp.size:
        raise ValueError("y_true and y_pred must be equal length")

    result: dict = {"n": int(yt.size), "overall": _one_result(yt, yp, n_boot, seed)}
    if split_tags is not None:
        tags = np.asarray([str(t) for t in split_tags])
        per_split: dict[str, dict] = {}
        for tag in np.unique(tags):
            m = tags == tag
            tag_seed = seed ^ (sum(int(c) for c in tag.encode()) & 0xFFFF)  # deterministic
            per_split[str(tag)] = _one_result(yt[m], yp[m], n_boot, tag_seed)
        result["per_split"] = per_split
    return result


def _one_result(yt: np.ndarray, yp: np.ndarray, n_boot: int, seed: int) -> dict:
    n = int(yt.size)
    if n < 4:
        return {"n": n, "rank_ic": None, "icir": None, "hit_rate": None,
                "decile_spread": None, "ci95": None}
    ic = rank_ic(yt, yp)
    ci = _rank_ic_bootstrap(yt, yp, n_boot=n_boot, seed=seed)
    ic_clean = None if (isnan(ic) or isinf(ic)) else float(ic)
    return {
        "n": n,
        "rank_ic": ic_clean,
        "icir": None if ic_clean is None else icir([ic]),
        "hit_rate": hit_rate(yt, yp),
        "decile_spread": decile_spread(yt, yp),
        "ci95": {"lo": ci["lo"], "hi": ci["hi"], "mean": ci["mean"]},
    }


def _rank_ic_bootstrap(
    yt: np.ndarray,
    yp: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    block: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Block-bootstrap CI of the rank IC, resampling (y, pred) pairs jointly.

    This is the CI that E1–E4 report: it drives the "is IC ≠ 0" decision. The
    bootstrap preserves the joint pair distribution AND, via moving blocks,
    the autocorrelation of the tape.
    """
    n = yt.size
    rng = np.random.default_rng(seed)
    blen = block if block > 0 else max(1, int(np.sqrt(n)))
    n_blocks = max(1, (n + blen - 1) // blen)
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = _block_indices(starts, blen, n)
        stats[b] = rank_ic(yt[idx], yp[idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"lo": float(lo), "hi": float(hi), "mean": float(np.mean(stats))}