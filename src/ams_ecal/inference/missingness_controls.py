"""Controls for features that are undefined for some events (validity gate SV-4; review C3, C4, C11).

A multiscale statistic is NaN when an event has too few lit cells. The development ladder fed the NaN to the classifier, so the
classifier could read "was this event eligible?", which is a threshold on the occupied-cell count (a conventional feature). The
S0 audit measured the consequence: at 10 GeV and on the readout at 20-50 GeV the multiscale rung equals the indicator alone to
0.014 AUC. These tools make that visible and give the analyses that do not suffer from it:

* ``indicator_only_separation``: the separation (max(AUC, 1 - AUC)) of "this feature was defined" between two classes. Always
  reported beside a multiscale rung (SV-4).
* ``add_missing_indicators`` / ``impute_for_model``: an EXPLICIT missing indicator, so the model's use of missingness is a named
  column that can be dropped, instead of a side effect of NaN handling.
* ``eligible_in_both``: the events for which the feature is defined, with the classes matched on a key (occupied cells) so the
  comparison is among events that were eligible and similarly sized, with the numbers per stratum reported.
* ``testability``: per cell (energy x representation x window) whether the comparison is testable at all; cells with too few
  eligible events in either class are declared "not testable" rather than analysed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

MIN_ELIGIBLE_PER_CLASS = 100


def indicator_only_separation(defined_a: np.ndarray, defined_b: np.ndarray) -> dict[str, float]:
    """How well "the feature is defined" alone separates class A from class B: AUC, separation and the two eligible fractions."""

    a, b = np.asarray(defined_a, dtype=bool), np.asarray(defined_b, dtype=bool)
    scores = np.concatenate([a, b]).astype(float)
    labels = np.concatenate([np.ones(len(a), int), np.zeros(len(b), int)])
    auc = float(roc_auc_score(labels, scores)) if 0 < scores.sum() < len(scores) else 0.5
    return {
        "auc": auc,
        "separation": max(auc, 1.0 - auc),
        "eligible_fraction_a": float(a.mean()),
        "eligible_fraction_b": float(b.mean()),
    }


def add_missing_indicators(features: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Append one 0/1 column per feature column that contains a NaN; returns the new table and the indicator column indices."""

    features = np.asarray(features, dtype=float)
    holes = np.isnan(features).any(axis=0)
    indicators = np.isnan(features[:, holes]).astype(float)
    start = features.shape[1]
    return np.column_stack([features, indicators]), list(range(start, start + indicators.shape[1]))


def impute_for_model(features: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Replace NaN by a constant, for models that cannot take NaN; always use together with ``add_missing_indicators``."""

    return np.where(np.isnan(features), fill, features)


@dataclass(frozen=True)
class EligibleInBoth:
    index_a: np.ndarray
    index_b: np.ndarray
    n_per_stratum: list[tuple[int, int]]  # (class A, class B) events kept in each matched stratum
    n_eligible_a: int
    n_eligible_b: int


def eligible_in_both(
    values_a: np.ndarray,
    values_b: np.ndarray,
    key_a: np.ndarray,
    key_b: np.ndarray,
    n_strata: int,
    rng: np.random.Generator,
) -> EligibleInBoth:
    """Events with a finite value in each class, matched in equal-count strata of ``key`` (occupied cells), down to the smaller class.

    Within each stratum the larger class is subsampled to the size of the smaller, so the kept events have the same key
    distribution in both classes; the strata are the quantile bins of the pooled eligible keys.
    """

    ok_a, ok_b = np.flatnonzero(np.isfinite(values_a)), np.flatnonzero(np.isfinite(values_b))
    pooled = np.concatenate([np.asarray(key_a)[ok_a], np.asarray(key_b)[ok_b]])
    if len(pooled) == 0:
        return EligibleInBoth(ok_a, ok_b, [], 0, 0)
    edges = np.quantile(pooled, np.linspace(0, 1, n_strata + 1)[1:-1])
    sa, sb = np.digitize(np.asarray(key_a)[ok_a], edges), np.digitize(np.asarray(key_b)[ok_b], edges)
    keep_a, keep_b, counts = [], [], []
    for s in range(n_strata):
        ia, ib = ok_a[sa == s], ok_b[sb == s]
        n = min(len(ia), len(ib))
        counts.append((n, n))
        if n:
            keep_a.append(rng.choice(ia, n, replace=False))
            keep_b.append(rng.choice(ib, n, replace=False))
    return EligibleInBoth(
        index_a=np.concatenate(keep_a) if keep_a else np.array([], int),
        index_b=np.concatenate(keep_b) if keep_b else np.array([], int),
        n_per_stratum=counts,
        n_eligible_a=len(ok_a),
        n_eligible_b=len(ok_b),
    )


def testability(values_a: np.ndarray, values_b: np.ndarray, minimum: int = MIN_ELIGIBLE_PER_CLASS) -> dict:
    """Declare a comparison testable only if both classes keep at least ``minimum`` events with a defined value."""

    na, nb = int(np.isfinite(values_a).sum()), int(np.isfinite(values_b).sum())
    testable = na >= minimum and nb >= minimum
    return {
        "testable": testable,
        "eligible_a": na,
        "eligible_b": nb,
        "minimum": minimum,
        "verdict": "testable" if testable else "not testable (too few eligible events in a class)",
    }
