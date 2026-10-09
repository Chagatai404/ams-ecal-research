"""The registered statistics of the first multiscale test (``research/plans/2026-10-09_multiscale_test_preregistration.md``).

Per event: the generalised dimensions D0, D1, D2 and D0 - D2 over the six registered scale windows, computed only where the
event has at least the occupied cells of the control image at which that window was found usable (336 for the 1000-hit
windows, 572 for the 3000-hit windows); elsewhere NaN (ineligible, reported, never dropped silently). Per pair of classes:
the KS distance of each statistic, overall and within five equal-count strata of the occupied cells (or of the deposited
energy), with a 95% bootstrap interval and the Geant4 split-half floor (two random halves of one class).

This module computes statistics; it decides nothing. Development use is on exposed events only; the registered opening of the
sealed sets is done once, elsewhere, through ``ams_ecal.validation.sealed_set``.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ams_ecal.multiscale_estimators import multiscale as ms

# window -> minimum occupied cells (the median cells of the control image at the hit count where the window passed)
REGISTERED_WINDOWS: dict[tuple[int, ...], int] = {
    (1, 2, 3, 6, 9, 18): 336,
    (2, 3, 6, 9): 336,
    (1, 2, 3, 6): 572,
    (1, 2, 3, 6, 9): 572,
    (2, 3): 572,
    (2, 3, 6): 572,
}
STAT_NAMES = ("D0", "D1", "D2", "D0-D2")
MATERIAL_KS = 0.10
STRATA = 5
BOOTSTRAPS = 200
MIN_PER_STRATUM = 20


def feature_names() -> list[str]:
    return [f"{s}@{'-'.join(map(str, w))}" for w in REGISTERED_WINDOWS for s in STAT_NAMES]


def event_multiscale(image: np.ndarray) -> np.ndarray:
    """Registered statistics of one image, in ``feature_names`` order; NaN where the event is ineligible for the window."""

    cells = ms.n_eff(image)
    profile = ms.profile(image)
    out = []
    for window, minimum in REGISTERED_WINDOWS.items():
        if cells < minimum:
            out += [np.nan] * len(STAT_NAMES)
            continue
        d = {q: ms.window_dimension(profile[q], ms.AMS_BOX_SIDES, q, window) for q in ms.Q_VALUES}
        out += [d[0.0], d[1.0], d[2.0], d[0.0] - d[2.0]]
    return np.array(out)


def multiscale_table(grids: np.ndarray) -> np.ndarray:
    """``(events, features)`` for ``(events, 18, 72)`` grids; events with no energy give all NaN."""

    rows = []
    for grid in grids:
        rows.append(event_multiscale(grid) if grid.sum() > 0 else np.full(len(feature_names()), np.nan))
    return np.array(rows)


def conventional_features(grids: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Conventional calorimeter observables of ``(events, layers, cells)`` grids (the first rung of the ladder)."""

    n, layers, cells = grids.shape
    layer_e = grids.sum(axis=2)
    total = layer_e.sum(axis=1)
    safe_total = np.where(total > 0, total, 1.0)
    fractions = layer_e / safe_total[:, None]
    cog = (fractions * np.arange(layers)[None, :]).sum(axis=1)
    idx = np.arange(cells)[None, None, :]
    safe_layer = np.where(layer_e > 0, layer_e, 1.0)[..., None]
    centre = (grids * idx).sum(axis=2, keepdims=True) / safe_layer
    variance = (grids * (idx - centre) ** 2).sum(axis=2) / safe_layer[..., 0]
    width = (fractions * np.sqrt(variance)).sum(axis=1)
    centre_cell = np.clip(np.rint(centre[..., 0]).astype(int), 1, cells - 2)
    core = np.zeros((n, layers))
    for offset in (-1, 0, 1):
        core += np.take_along_axis(grids, (centre_cell + offset)[..., None], axis=2)[..., 0]
    core_fraction = (core.sum(axis=1)) / safe_total
    occupied = (grids > 0).reshape(n, -1).sum(axis=1)
    max_cell = grids.reshape(n, -1).max(axis=1) / safe_total
    names = ["log_total_energy", "occupied_cells", "layer_cog", "width_cells", "core_fraction", "max_cell_fraction"] + [
        f"layer_fraction_{i}" for i in range(layers)
    ]
    table = np.column_stack([np.log(safe_total), occupied, cog, width, core_fraction, max_cell, fractions])
    return names, table


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < MIN_PER_STRATUM or len(b) < MIN_PER_STRATUM:
        return float("nan")
    return float(stats.ks_2samp(a, b, method="asymp").statistic)


def stratum_edges(pooled: np.ndarray, strata: int = STRATA) -> np.ndarray:
    return np.quantile(pooled, np.linspace(0, 1, strata + 1)[1:-1])


def matched_ks(a: np.ndarray, b: np.ndarray, key_a: np.ndarray, key_b: np.ndarray) -> float:
    """KS of ``a`` against ``b`` averaged over equal-count strata of the matching key (pooled quantiles), weighted by the
    smaller class count of each stratum; strata with fewer than ``MIN_PER_STRATUM`` finite values in a class are skipped."""

    edges = stratum_edges(np.concatenate([key_a, key_b]))
    sa, sb = np.digitize(key_a, edges), np.digitize(key_b, edges)
    values, weights = [], []
    for s in range(STRATA):
        da, db = ks_distance(a[sa == s], b[sb == s]), min((sa == s).sum(), (sb == s).sum())
        if np.isfinite(da):
            values.append(da)
            weights.append(db)
    return float(np.average(values, weights=weights)) if values else float("nan")


def split_half_floor(x: np.ndarray, rng: np.random.Generator, repeats: int = BOOTSTRAPS) -> float:
    """Median KS between two random halves of one class: the Geant4 sample-to-sample floor of the statistic."""

    finite = x[np.isfinite(x)]
    if len(finite) < 2 * MIN_PER_STRATUM:
        return float("nan")
    values = []
    for _ in range(repeats):
        perm = rng.permutation(len(finite))
        values.append(stats.ks_2samp(finite[perm[: len(perm) // 2]], finite[perm[len(perm) // 2 :]]).statistic)
    return float(np.median(values))


def compare_feature(
    a: np.ndarray,
    b: np.ndarray,
    key_a: np.ndarray,
    key_b: np.ndarray,
    rng: np.random.Generator,
    bootstraps: int = BOOTSTRAPS,
) -> dict[str, float]:
    """One statistic of two classes: eligible fractions, KS, matched KS, bootstrap interval, floor and the registered verdict."""

    raw, matched = ks_distance(a, b), matched_ks(a, b, key_a, key_b)
    draws = []
    for _ in range(bootstraps):
        ia, ib = rng.integers(0, len(a), len(a)), rng.integers(0, len(b), len(b))
        draws.append(matched_ks(a[ia], b[ib], key_a[ia], key_b[ib]))
    low, high = (
        (float(v) for v in np.nanpercentile(draws, [2.5, 97.5])) if np.isfinite(draws).any() else (np.nan, np.nan)
    )
    floors = [f for f in (split_half_floor(a, rng), split_half_floor(b, rng)) if np.isfinite(f)]
    floor = max(floors) if floors else float("nan")
    material = bool(np.isfinite(matched) and matched >= MATERIAL_KS and low > floor)
    return {
        "eligible_fraction_a": float(np.isfinite(a).mean()),
        "eligible_fraction_b": float(np.isfinite(b).mean()),
        "ks": raw,
        "ks_matched": matched,
        "ks_matched_low": low,
        "ks_matched_high": high,
        "split_half_floor": floor,
        "material_by_the_registered_rule": material,
    }
