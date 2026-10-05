"""Event observables and statistics for the Geant4 proton pilot.

Everything here is a pure function of arrays, so it is tested on synthetic
events whose answers are known. Definitions are fixed here, before any pilot
result is looked at; thresholds are ANALYSIS CHOICES expressed in units of a
measured MIP scale, and every one of them is varied in the report.

OBSERVABLES act on an alternating-view grid ``(n_events, n_layers, 72)``: the
18-layer AMS readout, or the 270-layer extended readout, with identical code,
so thin and extended observations of the same event are directly comparable.

* energy       - summed grid energy (scintillator energy for the readout grid).
* long_cog     - energy-weighted mean layer depth, mm from the front face.
* long_rms     - energy-weighted RMS of layer depth about long_cog, mm.
* max_layer    - layer holding the most energy (depth of maximum observed).
* active / last_active_layer - layers above the layer threshold.
* width        - energy-weighted RMS of the measured coordinate about the
                 incident track, over all cells, mm.
* core / containment - energy fraction within +-1 / +-2 cells of the track cell.
* n_hit_cells  - cells above the cell threshold.
* max_cell, max_cell_fraction - hottest cell and its share of the energy.
* participation - E^2 / sum(E_cell^2), the effective number of lit cells.

STATISTICS

* ``conditional_variance_fraction`` - S_D = Var(E[Y|D]) / Var(Y) with E[Y|D]
  estimated in equal-count bins of D. Reported both raw (eta^2, biased upward
  by bin noise) and bias-adjusted (epsilon^2). It is the fraction of variance
  STATISTICALLY ASSOCIATED with D under this binning - not a causal Sobol
  index, because D is not manipulated.
* ``bootstrap_interval`` - percentile bootstrap over events.
* ``wilson_interval``    - binomial fractions.
* ``censored_exponential_rate`` - maximum-likelihood interaction rate from
  first-interaction depths inside a slab plus the count that crossed it.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy import stats

from ams_ecal.detector.geometry import ECALGeometry
from ams_ecal.detector.projection import cell_indices, layer_measures_x

OBSERVABLES = (
    "energy_mev",
    "long_cog_mm",
    "long_rms_mm",
    "max_layer",
    "n_active_layers",
    "last_active_layer",
    "width_mm",
    "core_fraction",
    "containment_fraction",
    "n_hit_cells",
    "max_cell_mev",
    "max_cell_fraction",
    "participation",
)


def cell_centres_mm(geometry: ECALGeometry) -> np.ndarray:
    pitch = geometry.cell_pitch_mm
    return -geometry.cells_per_layer * pitch / 2 + (np.arange(geometry.cells_per_layer) + 0.5) * pitch


def layer_centres_mm(geometry: ECALGeometry, n_layers: int) -> np.ndarray:
    return (np.arange(n_layers) + 0.5) * geometry.mean_readout_slice_thickness_mm


def track_cells(
    geometry: ECALGeometry, n_layers: int, entry_x_mm: np.ndarray, entry_y_mm: np.ndarray
) -> np.ndarray:
    """Return the cell each normally incident track crosses, per layer."""

    measures_x = layer_measures_x(geometry, n_layers)
    coordinate = np.where(
        measures_x[None, :], np.asarray(entry_x_mm)[:, None], np.asarray(entry_y_mm)[:, None]
    )
    return cell_indices(coordinate, geometry)


def event_observables(
    grids: np.ndarray,
    geometry: ECALGeometry,
    entry_x_mm: np.ndarray,
    entry_y_mm: np.ndarray,
    *,
    cell_threshold_mev: float,
    layer_threshold_mev: float,
) -> dict[str, np.ndarray]:
    """Return every observable for every event of a normally incident sample."""

    g = np.asarray(grids, dtype=float)
    _, n_layers, n_cells = g.shape
    energy = g.sum(axis=(1, 2))
    layer = g.sum(axis=2)
    z = layer_centres_mm(geometry, n_layers)

    measures_x = layer_measures_x(geometry, n_layers)
    track = np.where(
        measures_x[None, :], np.asarray(entry_x_mm)[:, None], np.asarray(entry_y_mm)[:, None]
    )
    offset_mm = cell_centres_mm(geometry)[None, None, :] - track[:, :, None]
    offset_cells = np.abs(
        np.arange(n_cells)[None, None, :] - cell_indices(track, geometry)[:, :, None]
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        cog = (layer * z).sum(axis=1) / energy
        rms = np.sqrt((layer * (z[None, :] - cog[:, None]) ** 2).sum(axis=1) / energy)
        width = np.sqrt((g * offset_mm**2).sum(axis=(1, 2)) / energy)
        core = (g * (offset_cells <= 1)).sum(axis=(1, 2)) / energy
        containment = (g * (offset_cells <= 2)).sum(axis=(1, 2)) / energy
        max_cell = g.max(axis=(1, 2))
        max_fraction = max_cell / energy
        participation = energy**2 / (g**2).sum(axis=(1, 2))

    active = layer > layer_threshold_mev
    last_active = np.where(
        active.any(axis=1), n_layers - 1 - np.argmax(active[:, ::-1], axis=1), -1
    )
    return {
        "energy_mev": energy,
        "long_cog_mm": cog,
        "long_rms_mm": rms,
        "max_layer": np.where(energy > 0, layer.argmax(axis=1), -1),
        "n_active_layers": active.sum(axis=1),
        "last_active_layer": last_active,
        "width_mm": width,
        "core_fraction": core,
        "containment_fraction": containment,
        "n_hit_cells": (g > cell_threshold_mev).sum(axis=(1, 2)),
        "max_cell_mev": max_cell,
        "max_cell_fraction": max_fraction,
        "participation": participation,
    }


@dataclass(frozen=True, slots=True)
class MipScale:
    """Typical deposit of a through-going primary, measured, not assumed."""

    cell_mev: float  # median energy in the crossed cell of one layer
    layer_mev: float  # median energy of one layer


def mip_scale(
    grids: np.ndarray,
    geometry: ECALGeometry,
    entry_x_mm: np.ndarray,
    entry_y_mm: np.ndarray,
) -> MipScale:
    """Measure the MIP scale from events known (by truth) not to interact."""

    g = np.asarray(grids, dtype=float)
    n, n_layers, _ = g.shape
    if n == 0:
        raise ValueError("a MIP scale needs at least one non-interacting event")
    cells = track_cells(geometry, n_layers, entry_x_mm, entry_y_mm)
    crossed = np.take_along_axis(g, np.clip(cells, 0, None)[:, :, None], axis=2)[:, :, 0]
    return MipScale(
        cell_mev=float(np.median(crossed)), layer_mev=float(np.median(g.sum(axis=2)))
    )


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Interval:
    estimate: float
    low: float
    high: float


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial fraction."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    z = stats.norm.ppf(0.5 + confidence / 2)
    p = successes / trials
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    half = z * sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return Interval(p, centre - half, centre + half)


def bootstrap_interval(
    statistic: Callable[..., float],
    arrays: Sequence[np.ndarray],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> Interval:
    """Percentile bootstrap, resampling events (rows) jointly across arrays."""

    columns = [np.asarray(a) for a in arrays]
    n = len(columns[0])
    rng = np.random.default_rng(seed)
    estimate = float(statistic(*columns))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        index = rng.integers(0, n, n)
        draws[b] = statistic(*(c[index] for c in columns))
    alpha = (1 - confidence) / 2
    low, high = np.nanquantile(draws, [alpha, 1 - alpha])
    return Interval(estimate, float(low), float(high))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(stats.pearsonr(x, y).statistic)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(stats.spearmanr(x, y).statistic)


@dataclass(frozen=True, slots=True)
class ConditionalVariance:
    eta_squared: float  # raw between-bin variance share, biased upward
    epsilon_squared: float  # bias-adjusted
    n: int
    n_bins: int
    edges: np.ndarray
    bin_means: np.ndarray
    bin_counts: np.ndarray


def conditional_variance_fraction(
    d: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> ConditionalVariance:
    """Estimate S_D = Var(E[Y|D]) / Var(Y) with equal-count bins of D.

    Law of total variance: Var(Y) = Var(E[Y|D]) + E[Var(Y|D)]. The binned
    between-group share (eta^2) overestimates S_D by roughly (K-1)/N x (1-S_D)
    because bin means are noisy; epsilon^2 removes that expected excess.
    """

    d = np.asarray(d, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(d) & np.isfinite(y)
    d, y = d[keep], y[keep]
    n = len(y)
    if n < 2 * n_bins:
        raise ValueError("too few events for the requested number of bins")

    edges = np.quantile(d, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges[1:-1], d, side="right"), 0, n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins)
    sums = np.bincount(bins, weights=y, minlength=n_bins)
    used = counts > 0
    means = np.where(used, sums / np.maximum(counts, 1), np.nan)

    grand = y.mean()
    ss_total = float(((y - grand) ** 2).sum())
    ss_between = float((counts[used] * (means[used] - grand) ** 2).sum())
    k = int(used.sum())
    ss_within = ss_total - ss_between
    if ss_total == 0.0:
        eta, epsilon = float("nan"), float("nan")
    else:
        eta = ss_between / ss_total
        epsilon = (ss_between - (k - 1) * ss_within / (n - k)) / ss_total
    return ConditionalVariance(eta, epsilon, n, k, edges, means, counts)


@dataclass(frozen=True, slots=True)
class ExponentialRate:
    rate_per_mm: float
    standard_error: float
    n_interacting: int
    n_crossing: int

    def survival(self, length_mm: float) -> float:
        return float(np.exp(-self.rate_per_mm * length_mm))


def censored_exponential_rate(
    depths_mm: np.ndarray, n_crossing: int, length_mm: float
) -> ExponentialRate:
    """Maximum-likelihood rate of an exponential depth truncated by a slab.

    Interacting events contribute their depth; events that cross the slab
    contribute its full length as exposure. rate = n_int / total exposure,
    with standard error rate / sqrt(n_int).
    """

    depths = np.asarray(depths_mm, dtype=float)
    if np.any(depths < 0) or np.any(depths > length_mm):
        raise ValueError("depths must lie inside the slab")
    exposure = depths.sum() + n_crossing * length_mm
    n_int = len(depths)
    rate = n_int / exposure
    return ExponentialRate(rate, rate / sqrt(max(n_int, 1)), n_int, n_crossing)
