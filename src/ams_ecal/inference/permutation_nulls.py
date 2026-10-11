"""Nulls and floors for two-sample comparisons and multiscale statistics (validity gate SV-5; review C4, C5, C6, plan item A3).

What was wrong. The development KS verdicts compared a stratified KS with a bootstrap lower quantile against a different statistic
(the median half-sample KS), resampled with replacement (ties, 100 draws), skipped small strata silently, and never built the
energy-preserving cell-shuffle null the pre-registration names. Here:

* ``permutation_pvalue``: the label-permutation test, with labels permuted **within strata** (energy x occupancy, say) so that the
  null keeps the stratum composition. p = (1 + #{null >= observed}) / (1 + B), never zero.
* ``stratum_ks``: a stratum-weighted KS distance that reports how many strata and events it used (nothing skipped silently).
* ``matched_size_split_floor``: the sample-to-sample floor of a statistic from disjoint subsets of ONE class with the sizes of the
  two classes being compared (a floor of the same statistic, not of a different one).
* ``shuffle_cells_within_layers`` and ``paired_shuffle_excess``: the cell-shuffle null. Every layer keeps its own set of cell
  energies (so layer profile, occupancy per layer and the energy values are unchanged) while the positions inside the layer are
  permuted, destroying lateral clustering. A multiscale statistic that is the same on the shuffled copy of an event is not
  measuring lateral structure.

All functions take an explicit ``numpy.random.Generator``; nothing here reads data files.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats

MIN_PER_STRATUM = 20


# --- strata ----------------------------------------------------------------------------------------------------------------


def equal_count_strata(key: np.ndarray, n_strata: int) -> np.ndarray:
    """Stratum index 0..n_strata-1 of each value by equal-count quantile bins of the pooled ``key``."""

    edges = np.quantile(key, np.linspace(0.0, 1.0, n_strata + 1)[1:-1])
    return np.digitize(key, edges)


def joint_strata(*keys: np.ndarray, n_strata: int = 5) -> np.ndarray:
    """One integer stratum per event from the equal-count bins of several keys (for example energy x occupied cells)."""

    out = np.zeros(len(keys[0]), dtype=np.int64)
    for key in keys:
        out = out * n_strata + equal_count_strata(np.asarray(key, dtype=float), n_strata)
    return out


# --- permutation of labels within strata -----------------------------------------------------------------------------------


def permute_labels_within_strata(labels: np.ndarray, strata: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    """A copy of ``labels`` shuffled inside every stratum (all together if ``strata`` is None)."""

    labels = np.asarray(labels)
    out = labels.copy()
    groups = [np.arange(len(labels))] if strata is None else [np.flatnonzero(strata == s) for s in np.unique(strata)]
    for idx in groups:
        out[idx] = labels[rng.permutation(idx)]
    return out


@dataclass(frozen=True)
class PermutationResult:
    observed: float
    null_mean: float
    null_95th: float
    p_value: float  # (1 + #{null >= observed}) / (1 + n_permutations)
    n_permutations: int


Statistic = Callable[[np.ndarray, np.ndarray, "np.ndarray | None"], float]
"""``statistic(values, labels, strata) -> float``; larger means a larger difference between label 1 and label 0."""


def permutation_pvalue(
    statistic: Statistic,
    values: np.ndarray,
    labels: np.ndarray,
    strata: np.ndarray | None,
    rng: np.random.Generator,
    n_permutations: int = 1000,
) -> PermutationResult:
    """Permutation test of ``statistic`` with labels permuted within strata. Add 1 to numerator and denominator: p > 0."""

    values, labels = np.asarray(values, dtype=float), np.asarray(labels)
    observed = statistic(values, labels, strata)
    null = np.array(
        [statistic(values, permute_labels_within_strata(labels, strata, rng), strata) for _ in range(n_permutations)]
    )
    finite = null[np.isfinite(null)]
    exceed = int((finite >= observed).sum()) if np.isfinite(observed) else len(finite)
    return PermutationResult(
        observed=float(observed),
        null_mean=float(finite.mean()) if len(finite) else float("nan"),
        null_95th=float(np.percentile(finite, 95)) if len(finite) else float("nan"),
        p_value=(1.0 + exceed) / (1.0 + len(finite)),
        n_permutations=len(finite),
    )


def stratum_ks(values: np.ndarray, labels: np.ndarray, strata: np.ndarray | None) -> float:
    """KS distance between label 1 and label 0, averaged over strata weighted by the smaller class count; NaN values are not events.

    Strata with fewer than ``MIN_PER_STRATUM`` finite values in a class are left out and the caller can count them with
    ``stratum_ks_report``. Returns NaN when no stratum qualifies.
    """

    return stratum_ks_report(values, labels, strata)["ks"]


def stratum_ks_report(values: np.ndarray, labels: np.ndarray, strata: np.ndarray | None) -> dict[str, float]:
    """``stratum_ks`` with the number of strata used and skipped and the events behind it."""

    values, labels = np.asarray(values, dtype=float), np.asarray(labels)
    groups = [np.ones(len(values), bool)] if strata is None else [strata == s for s in np.unique(strata)]
    ks, weights, used, skipped, events = [], [], 0, 0, 0
    for mask in groups:
        a = values[mask & (labels == 1)]
        b = values[mask & (labels == 0)]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < MIN_PER_STRATUM or len(b) < MIN_PER_STRATUM:
            skipped += 1
            continue
        ks.append(stats.ks_2samp(a, b).statistic)
        weights.append(min(len(a), len(b)))
        used += 1
        events += len(a) + len(b)
    return {
        "ks": float(np.average(ks, weights=weights)) if ks else float("nan"),
        "strata_used": used,
        "strata_skipped": skipped,
        "events_used": events,
    }


# --- the sample-to-sample floor ---------------------------------------------------------------------------------------------


def matched_size_split_floor(
    values: np.ndarray,
    n_first: int,
    n_second: int,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    rng: np.random.Generator,
    repeats: int = 200,
) -> np.ndarray:
    """``statistic(first, second)`` over disjoint random subsets of ONE class with the sizes of the two groups being compared.

    If the class is too small for both sizes the two sizes are scaled down by the same factor (the floor is then slightly
    higher than the true one: say so when reporting). Returns the ``repeats`` values; use the median and the 95th percentile.
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if n_first + n_second > len(values):
        scale = len(values) / (n_first + n_second)
        n_first, n_second = max(int(n_first * scale), 2), max(int(n_second * scale), 2)
    out = []
    for _ in range(repeats):
        order = rng.permutation(len(values))
        out.append(statistic(values[order[:n_first]], values[order[n_first : n_first + n_second]]))
    return np.array(out)


# --- the cell-shuffle null (plan item A3) ----------------------------------------------------------------------------------


def shuffle_cells_within_layers(grids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute the cells of every layer independently: ``(layers, cells)`` or ``(events, layers, cells)`` input.

    Keeps, per layer, the multiset of cell energies (layer energy, occupied cells per layer, the energy values); destroys where
    in the layer they sit. The layer is one view (x or y) of the alternating readout, so no view is ever mixed.
    """

    grids = np.asarray(grids)
    order = np.argsort(rng.random(grids.shape), axis=-1)
    return np.take_along_axis(grids, order, axis=-1)


@dataclass(frozen=True)
class ShuffleExcess:
    real: np.ndarray  # statistic per event
    shuffled_mean: np.ndarray  # mean over shuffles per event
    excess: np.ndarray  # real - shuffled_mean
    event_p_value: np.ndarray  # share of shuffles at least as extreme as the real value, per event (two-sided, plus-one)
    median_excess: float
    median_excess_95: tuple[float, float]  # bootstrap over events


def paired_shuffle_excess(
    grids: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_shuffles: int,
    rng: np.random.Generator,
    n_bootstrap: int = 500,
) -> ShuffleExcess:
    """Per event: the statistic of the real grid against the same statistic of ``n_shuffles`` cell-shuffled copies of it."""

    grids = np.asarray(grids)
    real = np.array([statistic(g) for g in grids], dtype=float)
    shuffled = np.empty((n_shuffles, len(grids)))
    for i in range(n_shuffles):
        copy = shuffle_cells_within_layers(grids, rng)
        shuffled[i] = [statistic(g) for g in copy]
    mean = np.nanmean(shuffled, axis=0)
    excess = real - mean
    centred = np.abs(shuffled - mean)
    p = (1.0 + np.sum(centred >= np.abs(real - mean), axis=0)) / (1.0 + n_shuffles)
    finite = excess[np.isfinite(excess)]
    boots = [np.median(rng.choice(finite, len(finite))) for _ in range(n_bootstrap)] if len(finite) else [np.nan]
    return ShuffleExcess(
        real=real,
        shuffled_mean=mean,
        excess=excess,
        event_p_value=p,
        median_excess=float(np.median(finite)) if len(finite) else float("nan"),
        median_excess_95=(float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
    )
