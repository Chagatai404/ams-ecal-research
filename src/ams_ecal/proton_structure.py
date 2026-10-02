"""Burst latent and lateral spill for crossing protons (EXP-005; DEC-008, DEC-009).

WORDING. Everything here is a Geant4-DERIVED PHENOMENOLOGY of a proton that crosses the thin
ECAL without a hadronic interaction: a calibration of Geant4 11.4.1, FTFP_BERT, in this
project's material model, over 10-100 GeV at normal incidence. The "burst" is an empirical
description of multi-layer coherent excess; it is NOT identified here with a mechanism.

WHY. The first crossing validation (``research/plans/2026-09-29_proton_dependency_analysis.md``
section 11) failed on event total, hit multiplicity, maximum cell and containment, because the
crossing branch drew its 18 layers independently and put every layer's energy in the crossed
cells. Two repairs were accepted: a structural burst latent (probability, onset, energy,
downstream extent) and a per-layer lateral spill.

THE BURST. A burst starts at the first layer whose all-material energy exceeds
``BURST_RATIO`` times the chord-conditioned reference median. Its latent is: presence
(a probability per energy), onset layer ``o``, a SHAPE CLASS (one of three equally populated
classes by peak size, each with its own mean peak-normalised profile ``q`` in the layer offset
``k``), an amplitude ``Y`` and a downstream EXTENT ``s`` (a stretch of the profile in ``k``).
The excess over the bulk mean at layer ``o + k`` is ``Y * q(k / s) * J_k`` with ``J_k`` a
mean-one log-normal jitter. ``Y`` and ``s`` are fitted per calibration burst by least squares
on the part of the burst inside the ECAL, so they describe the whole burst and not the part a
late onset left visible; fitting the peak of the visible part instead biased early bursts low.
Energy that would fall behind the last layer is lost, which is why a late burst deposits less
inside the ECAL. The FIBRE excess is the deposition excess times a share drawn from its own
quantile table: in the calibration events the two bursts are the same bursts (rank correlation
0.5 at 10 GeV rising to 0.9 at 100 GeV). ``Y``, ``s`` and the onset are independent to the
precision of the calibration (|rho| <= 0.1), so they are drawn independently.

THE SPILL. Spill is a conditional SPLIT of the layer's total energy, so layer and event totals
are untouched. Given the layer's ratio to its reference median (7 bins), the layer spills with
a calibrated probability; a spilling layer sends a calibrated fraction of its energy to ``K``
cells (K <= ``K_MAX``) at calibrated distances from the nearest crossed cell. A layer that is
no larger than ordinary almost never spills; one far above its median nearly always does.

THE BULK. The per-layer table that the burst is added to is rebuilt from the layers of
burst-free EVENTS. Using the full-sample table would count every burst twice; pooling the layers
in front of a burst's onset made burst-free events 2% too energetic, because those layers are
measurably heavier (a burst and ordinary excess share a cause).

BULK COUPLING (NOT one of the two accepted repairs; awaiting the researcher, plan Q7). In
burst-free events the two readout layers of one superlayer are correlated (fibre lag-1 about
0.15 within a superlayer, about 0.02 across), which independent bulk draws miss. The bulk draws
are therefore joined by a three-parameter Gaussian copula: the within-superlayer pair, adjacent
layers of neighbouring superlayers, and layers two apart.

Calibration inputs are plain arrays (``CrossingCalibrationInputs``) so that the builder can be
tested without Geant4 data. Draws are ordered and fixed in number per event, so a seed always
names the same event.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy import stats
from scipy.special import ndtr

REPRESENTATIONS = ("readout", "deposition")
N_LAYERS = 18
BURST_RATIO = 3.0
BURST_AMPLITUDE_BINS = 3
K_MAX = 6
SPILL_CELL_MEV = 0.01
# interior edges of the 7 spill bins in r = layer energy / reference median
SPILL_R_EDGES = np.array([1.0, 1.5, 2.2, 3.0, 5.0, 10.0])
# the spill-distance bins, in cells from the nearest crossed cell: [1,2) [2,3) ... [25,72)
DISTANCE_EDGES = np.array([1, 2, 3, 4, 5, 7, 10, 15, 25, 72])
N_DISTANCE_BINS = len(DISTANCE_EDGES) - 1
# r >= this uses the "shower-like" distance law
SPILL_DISTANCE_SPLIT_R = 3.0
STRUCTURE_LEVELS = np.unique(
    np.concatenate(
        [
            [0.0, 0.01, 0.02, 0.05],
            np.arange(0.1, 0.91, 0.1),
            [0.95, 0.98, 0.99, 0.995, 1.0],
        ]
    )
)
# draws per layer and representation for the spill: presence, K, fraction, then 4 per cell
SPILL_DRAWS = 3 + 4 * K_MAX
# uniform draws per event for the burst latent: presence, onset, shape class, amplitude, extent, share;
# plus one standard normal per layer for the jitter
BURST_UNIFORMS = 6
# the stretch factors of the downstream extent tried at calibration (log-spaced)
EXTENT_GRID = np.exp(np.linspace(np.log(0.2), np.log(6.0), 81))
# penalty on |ln stretch| in the template fit: a window of one or two layers cannot tell a
# stretch from a scale, and without it the fit would drift to the edge of the grid
EXTENT_PENALTY = 0.01


# ----------------------------------------------------------------------
# Small numerical helpers
# ----------------------------------------------------------------------


def ln_energy_weights(
    anchors: np.ndarray, energies_gev: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(lower, upper, w)`` so that ``ln E`` interpolates the anchors.

    Raises outside the anchor range: nothing is extrapolated.
    """

    anchors = np.asarray(anchors, dtype=float)
    energy = np.asarray(energies_gev, dtype=float)
    if np.any(energy < anchors[0]) or np.any(energy > anchors[-1]):
        raise ValueError(
            f"energy outside the calibrated range [{anchors[0]:g}, {anchors[-1]:g}] GeV; "
            "nothing is extrapolated"
        )
    upper = np.clip(np.searchsorted(anchors, energy), 1, len(anchors) - 1)
    lower = upper - 1
    span = np.log(anchors[upper]) - np.log(anchors[lower])
    return lower, upper, (np.log(energy) - np.log(anchors[lower])) / span


def interpolate_levels(levels: np.ndarray, values: np.ndarray, uniform: np.ndarray) -> np.ndarray:
    """Evaluate quantile functions ``values`` (``(..., Q)``) at ``uniform`` (``(...)``).

    Linear between levels; monotone in ``uniform`` when ``values`` is non-decreasing.
    """

    u = np.asarray(uniform, dtype=float)
    if np.any((u < 0) | (u > 1)):
        raise ValueError("uniform variates must lie in [0, 1]")
    index = np.clip(np.searchsorted(levels, u, side="right") - 1, 0, len(levels) - 2)
    low, high = levels[index], levels[index + 1]
    weight = (u - low) / (high - low)
    v0 = np.take_along_axis(values, index[..., None], axis=-1)[..., 0]
    v1 = np.take_along_axis(values, index[..., None] + 1, axis=-1)[..., 0]
    return v0 + weight * (v1 - v0)


def _pmf_draw(pmf: np.ndarray, uniform: np.ndarray) -> np.ndarray:
    """Draw category indices from ``pmf`` (``(..., C)``) by inversion."""

    cumulative = np.cumsum(pmf, axis=-1)
    cumulative[..., -1] = 1.0
    return np.minimum((uniform[..., None] > cumulative).sum(axis=-1), pmf.shape[-1] - 1)


# ----------------------------------------------------------------------
# Calibration inputs
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CrossingCalibrationInputs:
    """Per-event layer quantities of the calibration crossing events at one energy.

    ``spill_*`` describe, per representation, the cells OUTSIDE the cells the track crosses
    that hold more than ``SPILL_CELL_MEV``: how many (capped at ``K_MAX``), the share of the
    layer energy they hold, and their distance in cells from the nearest crossed cell (0 pads
    unused slots).
    """

    energy_gev: float
    chord_mm: np.ndarray
    layer_energy_mev: dict[str, np.ndarray]
    spill_count: dict[str, np.ndarray]
    spill_fraction: dict[str, np.ndarray]
    spill_distance: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        n = len(self.chord_mm)
        if self.chord_mm.shape != (n, N_LAYERS):
            raise ValueError(f"chord_mm must have shape (n, {N_LAYERS})")
        for name in REPRESENTATIONS:
            if self.layer_energy_mev[name].shape != (n, N_LAYERS):
                raise ValueError(f"layer_energy_mev[{name!r}] must have shape (n, {N_LAYERS})")
            if self.spill_count[name].shape != (n, N_LAYERS):
                raise ValueError(f"spill_count[{name!r}] must have shape (n, {N_LAYERS})")
            if self.spill_fraction[name].shape != (n, N_LAYERS):
                raise ValueError(f"spill_fraction[{name!r}] must have shape (n, {N_LAYERS})")
            if self.spill_distance[name].shape != (n, N_LAYERS, K_MAX):
                raise ValueError(f"spill_distance[{name!r}] must have shape (n, {N_LAYERS}, {K_MAX})")

    @property
    def n_events(self) -> int:
        return len(self.chord_mm)


# ----------------------------------------------------------------------
# The tables
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BurstTable:
    """Calibrated burst latent (deposition excess; fibre excess by a share)."""

    energies_gev: np.ndarray  # (A,)
    probability: np.ndarray  # (A,) P(an event has a burst)
    onset_pmf: np.ndarray  # (N_LAYERS,)
    levels: np.ndarray  # (Q,)
    amplitude_quantiles_mev: np.ndarray  # (A, BINS, Q) template amplitude Y of each shape class
    profile: np.ndarray  # (A, BINS, N_LAYERS) peak-normalised profile in layer offset
    extent_quantiles: np.ndarray  # (A, BINS, Q) stretch of the profile in layer offset
    jitter_sigma: np.ndarray  # (A,)
    share_quantiles: np.ndarray  # (A, Q) fibre excess / deposition excess
    counts: np.ndarray  # (A,) bursts calibrated per energy

    def __post_init__(self) -> None:
        a, q = len(self.energies_gev), len(self.levels)
        shapes = {
            "probability": (a,),
            "onset_pmf": (N_LAYERS,),
            "amplitude_quantiles_mev": (a, BURST_AMPLITUDE_BINS, q),
            "profile": (a, BURST_AMPLITUDE_BINS, N_LAYERS),
            "extent_quantiles": (a, BURST_AMPLITUDE_BINS, q),
            "jitter_sigma": (a,),
            "share_quantiles": (a, q),
            "counts": (a,),
        }
        for name, shape in shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {getattr(self, name).shape}")
        for name in (
            "probability",
            "onset_pmf",
            "amplitude_quantiles_mev",
            "profile",
            "extent_quantiles",
            "share_quantiles",
        ):
            if not np.all(np.isfinite(getattr(self, name))) or np.any(getattr(self, name) < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if abs(self.onset_pmf.sum() - 1.0) > 1e-9:
            raise ValueError("onset_pmf must sum to 1")
        if np.any(self.probability > 1):
            raise ValueError("probability must not exceed 1")
        for name in ("amplitude_quantiles_mev", "extent_quantiles", "share_quantiles"):
            if np.any(np.diff(getattr(self, name), axis=-1) < 0):
                raise ValueError(f"{name} must be non-decreasing")


@dataclass(frozen=True, slots=True)
class SpillTable:
    """Calibrated lateral spill of one representation."""

    levels: np.ndarray  # (Q,)
    probability: np.ndarray  # (R,) P(layer spills | r bin)
    cells_pmf: np.ndarray  # (R, K_MAX) P(K = k + 1 | spills, r bin)
    fraction_quantiles: np.ndarray  # (R, Q) share of the layer energy that spills
    distance_pmf: np.ndarray  # (2, N_DISTANCE_BINS) ordinary / shower-like layers
    counts: np.ndarray  # (R,) calibration layers per r bin

    def __post_init__(self) -> None:
        r, q = len(SPILL_R_EDGES) + 1, len(self.levels)
        shapes = {
            "probability": (r,),
            "cells_pmf": (r, K_MAX),
            "fraction_quantiles": (r, q),
            "distance_pmf": (2, N_DISTANCE_BINS),
            "counts": (r,),
        }
        for name, shape in shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {getattr(self, name).shape}")
        for name in ("probability", "cells_pmf", "fraction_quantiles", "distance_pmf"):
            values = getattr(self, name)
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if np.any(self.probability > 1) or np.any(self.fraction_quantiles > 1):
            raise ValueError("probabilities and spill fractions must not exceed 1")
        if np.any(np.diff(self.fraction_quantiles, axis=-1) < 0):
            raise ValueError("fraction_quantiles must be non-decreasing")
        for name in ("cells_pmf", "distance_pmf"):
            if not np.allclose(getattr(self, name).sum(axis=-1), 1.0):
                raise ValueError(f"{name} rows must each sum to 1")


@dataclass(frozen=True, slots=True)
class CrossingStructure:
    """Everything the repaired crossing branch needs beyond the geometry."""

    reference_median_mev: dict[str, np.ndarray]  # rep -> (A, chord bins)
    burst: BurstTable
    spill: dict[str, SpillTable]
    # rep -> (3,) Gaussian-copula correlations of the BULK layer draws: the two readout layers
    # of one superlayer, neighbouring layers across superlayers, and layers two apart
    bulk_coupling: dict[str, np.ndarray]

    def arrays(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name in REPRESENTATIONS:
            out[f"reference_median_{name}"] = np.asarray(self.reference_median_mev[name], dtype=float)
            out[f"bulk_coupling_{name}"] = np.asarray(self.bulk_coupling[name], dtype=float)
            spill = self.spill[name]
            out[f"spill_probability_{name}"] = np.asarray(spill.probability, dtype=float)
            out[f"spill_cells_pmf_{name}"] = np.asarray(spill.cells_pmf, dtype=float)
            out[f"spill_fraction_quantiles_{name}"] = np.asarray(spill.fraction_quantiles, dtype=float)
            out[f"spill_distance_pmf_{name}"] = np.asarray(spill.distance_pmf, dtype=float)
            out[f"spill_counts_{name}"] = np.asarray(spill.counts, dtype=np.int64)
        burst = self.burst
        out["spill_levels"] = np.asarray(self.spill[REPRESENTATIONS[0]].levels, dtype=float)
        out["burst_levels"] = np.asarray(burst.levels, dtype=float)
        out["burst_probability"] = np.asarray(burst.probability, dtype=float)
        out["burst_onset_pmf"] = np.asarray(burst.onset_pmf, dtype=float)
        out["burst_amplitude_quantiles_mev"] = np.asarray(burst.amplitude_quantiles_mev, dtype=float)
        out["burst_profile"] = np.asarray(burst.profile, dtype=float)
        out["burst_extent_quantiles"] = np.asarray(burst.extent_quantiles, dtype=float)
        out["burst_jitter_sigma"] = np.asarray(burst.jitter_sigma, dtype=float)
        out["burst_share_quantiles"] = np.asarray(burst.share_quantiles, dtype=float)
        out["burst_counts"] = np.asarray(burst.counts, dtype=np.int64)
        return out

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray], energies_gev: np.ndarray) -> CrossingStructure:
        burst = BurstTable(
            energies_gev=np.asarray(energies_gev, dtype=float),
            probability=arrays["burst_probability"],
            onset_pmf=arrays["burst_onset_pmf"],
            levels=arrays["burst_levels"],
            amplitude_quantiles_mev=arrays["burst_amplitude_quantiles_mev"],
            profile=arrays["burst_profile"],
            extent_quantiles=arrays["burst_extent_quantiles"],
            jitter_sigma=arrays["burst_jitter_sigma"],
            share_quantiles=arrays["burst_share_quantiles"],
            counts=arrays["burst_counts"],
        )
        spill = {
            name: SpillTable(
                levels=arrays["spill_levels"],
                probability=arrays[f"spill_probability_{name}"],
                cells_pmf=arrays[f"spill_cells_pmf_{name}"],
                fraction_quantiles=arrays[f"spill_fraction_quantiles_{name}"],
                distance_pmf=arrays[f"spill_distance_pmf_{name}"],
                counts=arrays[f"spill_counts_{name}"],
            )
            for name in REPRESENTATIONS
        }
        return cls(
            reference_median_mev={n: arrays[f"reference_median_{n}"] for n in REPRESENTATIONS},
            burst=burst,
            spill=spill,
            bulk_coupling={n: arrays[f"bulk_coupling_{n}"] for n in REPRESENTATIONS},
        )

    def coupling_factor(self, representation: str) -> np.ndarray:
        """Lower Cholesky factor of the bulk-draw correlation matrix of one representation."""

        return _coupling_factor(tuple(float(x) for x in self.bulk_coupling[representation]))


@lru_cache(maxsize=32)
def _coupling_factor(parameters: tuple[float, ...]) -> np.ndarray:
    factor = np.linalg.cholesky(coupling_matrix(np.array(parameters)))
    factor.setflags(write=False)
    return factor


def coupling_matrix(parameters: np.ndarray) -> np.ndarray:
    """Correlation matrix of the 18 bulk layer draws from its three parameters.

    ``parameters`` are the correlations of (the two layers of a superlayer, adjacent layers of
    neighbouring superlayers, layers two apart); everything else is zero. Shrunk towards the
    identity until positive definite, which for the calibrated values it already is.
    """

    within, across, second = (float(x) for x in parameters)
    matrix = np.eye(N_LAYERS)
    for layer in range(N_LAYERS - 1):
        value = within if layer % 2 == 0 else across
        matrix[layer, layer + 1] = matrix[layer + 1, layer] = value
    for layer in range(N_LAYERS - 2):
        matrix[layer, layer + 2] = matrix[layer + 2, layer] = second
    shrink = 1.0
    off = matrix - np.eye(N_LAYERS)
    while np.linalg.eigvalsh(matrix).min() < 1e-6:
        shrink *= 0.9
        matrix = np.eye(N_LAYERS) + shrink * off
    return matrix


def estimate_bulk_coupling(layer_energy_mev: np.ndarray) -> np.ndarray:
    """Gaussian-copula correlations of the bulk layer energies of burst-free events.

    Spearman correlations averaged over the layer pairs of each kind, converted to the
    correlation of the underlying normals by ``2 sin(pi rho / 6)``.
    """

    ranks = stats.spearmanr(layer_energy_mev).statistic
    within = float(np.mean([ranks[i, i + 1] for i in range(0, N_LAYERS - 1, 2)]))
    across = float(np.mean([ranks[i, i + 1] for i in range(1, N_LAYERS - 1, 2)]))
    second = float(np.mean([ranks[i, i + 2] for i in range(N_LAYERS - 2)]))
    return 2.0 * np.sin(np.pi * np.array([within, across, second]) / 6.0)


def coupled_uniforms(factor: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Uniform variates whose normal scores have the correlation ``factor @ factor.T``."""

    return np.clip(ndtr(normals @ factor.T), 0.0, 1.0)


def stretch_profile(profile: np.ndarray, stretch: np.ndarray) -> np.ndarray:
    """Evaluate peak-normalised profiles ``profile`` (``(n, L)`` or ``(L,)``) at ``k / stretch``.

    Linear in the layer offset ``k``; zero beyond the last tabulated layer. ``stretch`` is
    ``(n,)``; a stretch above 1 lengthens the burst downstream, below 1 shortens it.
    """

    stretch = np.atleast_1d(np.asarray(stretch, dtype=float))
    table = np.broadcast_to(profile, (len(stretch), N_LAYERS))
    position = np.arange(N_LAYERS)[None, :] / stretch[:, None]
    lower = np.floor(position).astype(int)
    fraction = position - lower
    low = np.take_along_axis(table, np.clip(lower, 0, N_LAYERS - 1), axis=1)
    high = np.take_along_axis(table, np.clip(lower + 1, 0, N_LAYERS - 1), axis=1)
    out = (1.0 - fraction) * low + fraction * high
    return np.where(position > N_LAYERS - 1, 0.0, out)


# ----------------------------------------------------------------------
# The builder
# ----------------------------------------------------------------------


def chord_bin_edges(inputs: Sequence[CrossingCalibrationInputs], n_chord_bins: int) -> np.ndarray:
    """Interior edges of equal-count chord bins over all calibration layers."""

    pooled = np.concatenate([entry.chord_mm.ravel() for entry in inputs])
    return np.quantile(pooled, np.linspace(0.0, 1.0, n_chord_bins + 1)[1:-1])


def reference_medians(
    inputs: Sequence[CrossingCalibrationInputs], chord_edges: np.ndarray
) -> dict[str, np.ndarray]:
    """Median layer energy per (energy anchor, chord bin) over ALL calibration layers."""

    n_bins = len(chord_edges) + 1
    out = {name: np.empty((len(inputs), n_bins)) for name in REPRESENTATIONS}
    for a, entry in enumerate(inputs):
        bins = np.searchsorted(chord_edges, entry.chord_mm, side="right")
        for name in REPRESENTATIONS:
            for b in range(n_bins):
                out[name][a, b] = np.median(entry.layer_energy_mev[name][bins == b])
    return out


def detect_bursts(
    deposition_energy_mev: np.ndarray, reference_mev: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(has_burst, onset)``: the first layer above ``BURST_RATIO`` times the reference."""

    above = deposition_energy_mev > BURST_RATIO * reference_mev
    return above.any(axis=1), np.argmax(above, axis=1)


def _aligned(excess: np.ndarray, onset: np.ndarray) -> np.ndarray:
    """Return excess aligned at each event's onset layer, NaN behind the last layer."""

    n = len(excess)
    out = np.full((n, N_LAYERS), np.nan)
    for i in range(n):
        span = N_LAYERS - onset[i]
        out[i, :span] = excess[i, onset[i] :]
    return out


def _bulk_tables(
    inputs: Sequence[CrossingCalibrationInputs],
    chord_edges: np.ndarray,
    flags: list[np.ndarray],
    bulk_levels: np.ndarray,
    min_layers_per_bin: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Per-layer quantile tables and means of the layers of burst-free EVENTS.

    Layers in front of a burst's onset are deliberately NOT included: they are measurably
    heavier than the layers of burst-free events (a burst and ordinary excess share a cause),
    and pooling them made the generated burst-free events 2% too energetic. The excess of a
    burst is referenced to this table's mean, so burst events keep an unbiased total.
    """

    n_bins = len(chord_edges) + 1
    quantiles = {name: np.empty((len(inputs), n_bins, len(bulk_levels))) for name in REPRESENTATIONS}
    means = {name: np.empty((len(inputs), n_bins)) for name in REPRESENTATIONS}
    counts = np.zeros((len(inputs), n_bins), dtype=np.int64)
    for a, entry in enumerate(inputs):
        keep = np.broadcast_to(~flags[a][:, None], (len(flags[a]), N_LAYERS))
        bins = np.searchsorted(chord_edges, entry.chord_mm, side="right")
        for b in range(n_bins):
            selected = (bins == b) & keep
            counts[a, b] = int(selected.sum())
            if counts[a, b] < min_layers_per_bin:
                raise ValueError(
                    f"only {counts[a, b]} burst-free layers at {entry.energy_gev:g} GeV in chord bin {b}"
                )
            for name in REPRESENTATIONS:
                values = entry.layer_energy_mev[name][selected]
                quantiles[name][a, b] = np.quantile(values, bulk_levels)
                means[name][a, b] = values.mean()
    return quantiles, means, counts


def _fit_template(
    profile: np.ndarray, aligned: np.ndarray, available: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit ``Y * profile(k / s)`` to each burst's visible excess by least squares.

    Only the layers inside the ECAL (``available`` of them) are used, so the amplitude ``Y`` is
    the amplitude of the whole burst and not of the part a late onset left visible. ``Y`` is
    solved linearly for every stretch ``s`` of ``EXTENT_GRID``; the stretch with the smallest
    relative residual (plus ``EXTENT_PENALTY * ln(s)**2``) is kept. Returns ``(Y, s, expected)``
    where ``expected`` is the fitted profile, ``(n, N_LAYERS)``.
    """

    n = len(aligned)
    window = np.arange(N_LAYERS)[None, :] < available[:, None]
    x = np.where(window & np.isfinite(aligned), aligned, 0.0)
    scale = np.maximum((x**2).sum(axis=1), 1e-9)
    best_cost = np.full(n, np.inf)
    best_amp = np.zeros(n)
    best_stretch = np.ones(n)
    for stretch in EXTENT_GRID:
        shape = stretch_profile(profile, np.full(n, stretch)) * window
        denominator = np.maximum((shape**2).sum(axis=1), 1e-12)
        amplitude = np.clip((shape * x).sum(axis=1) / denominator, 0.0, None)
        cost = ((x - amplitude[:, None] * shape) ** 2).sum(axis=1) / scale
        cost = cost + EXTENT_PENALTY * np.log(stretch) ** 2
        better = cost < best_cost
        best_cost = np.where(better, cost, best_cost)
        best_amp = np.where(better, amplitude, best_amp)
        best_stretch = np.where(better, stretch, best_stretch)
    expected = best_amp[:, None] * stretch_profile(profile, best_stretch) * window
    return best_amp, best_stretch, expected


def _burst_table(
    inputs: Sequence[CrossingCalibrationInputs],
    chord_edges: np.ndarray,
    bulk_mean: dict[str, np.ndarray],
    flags: list[np.ndarray],
    onsets: list[np.ndarray],
    levels: np.ndarray,
) -> BurstTable:
    """Calibrate the burst table from the excess over the BULK MEAN of each layer.

    Each burst is first assigned to one of ``BURST_AMPLITUDE_BINS`` equally populated SHAPE
    CLASSES by the size of its in-ECAL peak, and the class's profile is the mean peak-normalised
    excess. Each burst's amplitude and extent are then fitted jointly on its visible window,
    which makes them independent of where the burst starts.
    """

    n_a = len(inputs)
    probability = np.empty(n_a)
    counts = np.empty(n_a, dtype=np.int64)
    amplitude_q = np.empty((n_a, BURST_AMPLITUDE_BINS, len(levels)))
    profile = np.zeros((n_a, BURST_AMPLITUDE_BINS, N_LAYERS))
    extent_q = np.empty((n_a, BURST_AMPLITUDE_BINS, len(levels)))
    share_q = np.empty((n_a, len(levels)))
    onset_hist = np.full(N_LAYERS, 0.5)
    residuals: list[list[np.ndarray]] = []
    quantile_edges = np.linspace(0, 1, BURST_AMPLITUDE_BINS + 1)[1:-1]

    for a, entry in enumerate(inputs):
        has, onset = flags[a], onsets[a]
        bins = np.searchsorted(chord_edges, entry.chord_mm, side="right")
        probability[a] = has.mean()
        counts[a] = int(has.sum())
        if counts[a] < 15:
            raise ValueError(f"only {counts[a]} bursts at {entry.energy_gev:g} GeV; too few to calibrate")
        onset_hist += np.bincount(onset[has], minlength=N_LAYERS)
        excess_dep = entry.layer_energy_mev["deposition"] - bulk_mean["deposition"][a][bins]
        excess_ro = entry.layer_energy_mev["readout"] - bulk_mean["readout"][a][bins]
        dep = _aligned(excess_dep[has], onset[has])
        ro = _aligned(excess_ro[has], onset[has])
        available = N_LAYERS - onset[has]
        peak = np.maximum(np.nanmax(dep, axis=1), 1e-3)
        # the share is taken over the layers where the deposition excess is positive, so the
        # denominator is at least the onset layer's excess and the ratio cannot blow up when
        # later layers sit below their mean and cancel the sum
        positive = np.where(np.isfinite(dep) & (dep > 0), dep, 0.0)
        fibre = np.where(np.isfinite(ro), np.clip(ro, 0.0, None), 0.0) * (positive > 0)
        share = fibre.sum(axis=1) / np.maximum(positive.sum(axis=1), 1e-9)
        share_q[a] = np.quantile(share, levels)
        shape_class = np.searchsorted(np.quantile(peak, quantile_edges), peak, side="right")
        normalised = dep / peak[:, None]
        per_class = []
        for b in range(BURST_AMPLITUDE_BINS):
            chosen = shape_class == b
            rows = normalised[chosen]
            contributors = np.isfinite(rows).sum(axis=0)
            with np.errstate(invalid="ignore"):
                mean = np.where(contributors >= 5, np.nanmean(rows, axis=0), 0.0)
            profile[a, b] = np.clip(np.nan_to_num(mean), 0.0, None)
            amplitude, stretch, expected = _fit_template(profile[a, b], dep[chosen], available[chosen])
            amplitude_q[a, b] = np.quantile(amplitude, levels)
            extent_q[a, b] = np.quantile(stretch, levels)
            observed = dep[chosen]
            strong = np.isfinite(observed) & (expected > 20.0) & (observed > 0)
            per_class.append(np.log(observed[strong] / expected[strong]))
        residuals.append(per_class)

    pooled = np.concatenate([r for per_class in residuals for r in per_class])
    pooled_sigma = float(np.clip(pooled.std(), 0.05, 1.5)) if len(pooled) >= 20 else 0.3
    sigma = np.empty(n_a)
    for a in range(n_a):
        own = np.concatenate(residuals[a])
        sigma[a] = float(np.clip(own.std(), 0.05, 1.5)) if len(own) >= 20 else pooled_sigma
    return BurstTable(
        energies_gev=np.array([e.energy_gev for e in inputs]),
        probability=probability,
        onset_pmf=onset_hist / onset_hist.sum(),
        levels=np.asarray(levels, dtype=float),
        amplitude_quantiles_mev=amplitude_q,
        profile=profile,
        extent_quantiles=extent_q,
        jitter_sigma=sigma,
        share_quantiles=share_q,
        counts=counts,
    )


def _spill_table(
    inputs: Sequence[CrossingCalibrationInputs],
    reference: dict[str, np.ndarray],
    chord_edges: np.ndarray,
    name: str,
    levels: np.ndarray,
) -> SpillTable:
    n_r = len(SPILL_R_EDGES) + 1
    r_all, count_all, fraction_all, distance_all = [], [], [], []
    for a, entry in enumerate(inputs):
        bins = np.searchsorted(chord_edges, entry.chord_mm, side="right")
        r_all.append((entry.layer_energy_mev[name] / reference[name][a][bins]).ravel())
        count_all.append(entry.spill_count[name].ravel())
        fraction_all.append(entry.spill_fraction[name].ravel())
        distance_all.append(entry.spill_distance[name].reshape(-1, K_MAX))
    r = np.concatenate(r_all)
    count = np.concatenate(count_all)
    fraction = np.concatenate(fraction_all)
    distance = np.concatenate(distance_all)
    r_bin = np.searchsorted(SPILL_R_EDGES, r, side="right")

    spills = count > 0
    probability = np.zeros(n_r)
    cells = np.zeros((n_r, K_MAX))
    quantiles = np.zeros((n_r, len(levels)))
    n_per_bin = np.zeros(n_r, dtype=np.int64)
    fallback_cells = np.bincount(count[spills] - 1, minlength=K_MAX)[:K_MAX] + 0.5
    fallback_cells = fallback_cells / fallback_cells.sum()
    fallback_q = np.quantile(fraction[spills], levels) if spills.any() else np.zeros(len(levels))
    for b in range(n_r):
        in_bin = r_bin == b
        n_per_bin[b] = int(in_bin.sum())
        spilling = in_bin & spills
        probability[b] = spilling.sum() / max(in_bin.sum(), 1)
        if spilling.sum() >= 20:
            c = np.bincount(count[spilling] - 1, minlength=K_MAX)[:K_MAX] + 0.5
            cells[b] = c / c.sum()
            quantiles[b] = np.quantile(fraction[spilling], levels)
        else:
            cells[b] = fallback_cells
            quantiles[b] = fallback_q
    quantiles = np.maximum.accumulate(np.clip(quantiles, 0.0, 1.0), axis=1)

    distance_pmf = np.zeros((2, N_DISTANCE_BINS))
    shower_like = r >= SPILL_DISTANCE_SPLIT_R
    for group, selected in enumerate((~shower_like, shower_like)):
        d = distance[selected]
        d = d[d > 0]
        histogram = np.histogram(d, bins=DISTANCE_EDGES)[0] + 0.5
        distance_pmf[group] = histogram / histogram.sum()
    return SpillTable(
        levels=np.asarray(levels, dtype=float),
        probability=probability,
        cells_pmf=cells,
        fraction_quantiles=quantiles,
        distance_pmf=distance_pmf,
        counts=n_per_bin,
    )


def build_structure(
    inputs: Sequence[CrossingCalibrationInputs],
    *,
    n_chord_bins: int,
    bulk_levels: np.ndarray,
    levels: np.ndarray = STRUCTURE_LEVELS,
    min_layers_per_bin: int = 20,
) -> tuple[CrossingStructure, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Calibrate the structure from calibration crossing events.

    Returns ``(structure, bulk_quantiles, chord_edges, bulk_counts)``. ``bulk_quantiles`` maps
    each representation to the ``(A, chord bins, len(bulk_levels))`` per-layer quantile table
    built from burst-free layers, which replaces the full-sample table.
    """

    inputs = sorted(inputs, key=lambda entry: entry.energy_gev)
    chord_edges = chord_bin_edges(inputs, n_chord_bins)
    reference = reference_medians(inputs, chord_edges)
    detected = []
    for a, entry in enumerate(inputs):
        bins = np.searchsorted(chord_edges, entry.chord_mm, side="right")
        detected.append(detect_bursts(entry.layer_energy_mev["deposition"], reference["deposition"][a][bins]))
    flags = [d[0] for d in detected]
    onsets = [d[1] for d in detected]
    bulk, bulk_mean, counts = _bulk_tables(inputs, chord_edges, flags, bulk_levels, min_layers_per_bin)
    burst = _burst_table(inputs, chord_edges, bulk_mean, flags, onsets, levels)
    spill = {
        name: _spill_table(inputs, reference, chord_edges, name, levels) for name in REPRESENTATIONS
    }
    coupling = {}
    for name in REPRESENTATIONS:
        free = np.concatenate(
            [entry.layer_energy_mev[name][~flag] for entry, flag in zip(inputs, flags, strict=True)]
        )
        coupling[name] = estimate_bulk_coupling(free)
    structure = CrossingStructure(
        reference_median_mev=reference, burst=burst, spill=spill, bulk_coupling=coupling
    )
    return structure, bulk, chord_edges, counts


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BurstDraw:
    """The burst latent of a batch of events, in both representations."""

    has_burst: np.ndarray  # (n,)
    onset: np.ndarray  # (n,)
    excess_mev: dict[str, np.ndarray]  # rep -> (n, N_LAYERS), zero where there is no burst


def sample_bursts(
    table: BurstTable, energies_gev: np.ndarray, uniforms: np.ndarray, normals: np.ndarray
) -> BurstDraw:
    """Draw the burst latent.

    ``uniforms`` is ``(n, BURST_UNIFORMS)``: presence, onset, shape class, amplitude, extent,
    share. ``normals`` is ``(n, N_LAYERS)``: the layer jitter, indexed by absolute layer.
    """

    energies = np.atleast_1d(np.asarray(energies_gev, dtype=float))
    n = len(energies)
    if uniforms.shape != (n, BURST_UNIFORMS) or normals.shape != (n, N_LAYERS):
        raise ValueError(f"uniforms must be (n, {BURST_UNIFORMS}) and normals (n, {N_LAYERS})")
    lower, upper, w = ln_energy_weights(table.energies_gev, energies)

    def blend(array: np.ndarray) -> np.ndarray:
        shape = (n,) + (1,) * (array.ndim - 1)
        return (1.0 - w).reshape(shape) * array[lower] + w.reshape(shape) * array[upper]

    rows = np.arange(n)
    has = uniforms[:, 0] < blend(table.probability)
    onset = _pmf_draw(np.broadcast_to(table.onset_pmf, (n, N_LAYERS)).copy(), uniforms[:, 1])
    which = np.minimum((uniforms[:, 2] * BURST_AMPLITUDE_BINS).astype(int), BURST_AMPLITUDE_BINS - 1)
    amplitude = interpolate_levels(
        table.levels, blend(table.amplitude_quantiles_mev)[rows, which], uniforms[:, 3]
    )
    stretch = interpolate_levels(
        table.levels, blend(table.extent_quantiles)[rows, which], uniforms[:, 4]
    )
    share = interpolate_levels(table.levels, blend(table.share_quantiles), uniforms[:, 5])
    profile = stretch_profile(blend(table.profile)[rows, which], stretch)  # (n, N_LAYERS) by offset
    sigma = blend(table.jitter_sigma)

    offset = np.arange(N_LAYERS)[None, :] - onset[:, None]
    shape = np.where(
        offset >= 0, np.take_along_axis(profile, np.clip(offset, 0, N_LAYERS - 1), axis=1), 0.0
    )
    jitter = np.exp(sigma[:, None] * normals - 0.5 * sigma[:, None] ** 2)
    deposition = np.where(has[:, None], amplitude[:, None] * shape * jitter, 0.0)
    return BurstDraw(
        has_burst=has,
        onset=onset,
        excess_mev={"deposition": deposition, "readout": share[:, None] * deposition},
    )


def reference_for(
    reference_mev: np.ndarray, anchors_gev: np.ndarray, energies_gev: np.ndarray, chord_bins: np.ndarray
) -> np.ndarray:
    """Reference median per layer at each event's energy: ``(A, chord bins)`` blended in ``ln E``."""

    lower, upper, w = ln_energy_weights(anchors_gev, np.atleast_1d(energies_gev))
    return (1.0 - w)[:, None] * reference_mev[lower[:, None], chord_bins] + w[:, None] * reference_mev[
        upper[:, None], chord_bins
    ]


def sample_bulk(
    quantiles_mev: np.ndarray,
    levels: np.ndarray,
    anchors_gev: np.ndarray,
    chord_bins: np.ndarray,
    energies_gev: np.ndarray,
    uniforms: np.ndarray,
) -> np.ndarray:
    """Draw bulk layer energies: ``quantiles_mev`` is ``(A, chord bins, Q)``; ``chord_bins`` ``(n, 18)``."""

    energies = np.atleast_1d(np.asarray(energies_gev, dtype=float))
    lower, upper, w = ln_energy_weights(anchors_gev, energies)
    n = len(energies)
    if uniforms.shape != (n, N_LAYERS) or chord_bins.shape != (n, N_LAYERS):
        raise ValueError("uniforms and chord_bins must be (n, 18)")
    low = quantiles_mev[lower[:, None], chord_bins]
    high = quantiles_mev[upper[:, None], chord_bins]
    vectors = (1.0 - w)[:, None, None] * low + w[:, None, None] * high
    return interpolate_levels(levels, vectors, uniforms)


@dataclass(frozen=True, slots=True)
class SpillDraw:
    """The lateral spill of a batch of layers, in one representation."""

    fraction: np.ndarray  # (n, N_LAYERS) share of the layer energy that leaves the crossed cells
    cells: np.ndarray  # (n, N_LAYERS) number of spill cells, 0 = no spill
    offset: np.ndarray  # (n, N_LAYERS, K_MAX) signed distance in cells from the reference cell
    weight: np.ndarray  # (n, N_LAYERS, K_MAX) shares of the spilled energy, summing to 1 over used cells


def sample_spill(table: SpillTable, ratio: np.ndarray, uniforms: np.ndarray) -> SpillDraw:
    """Draw the spill of each layer from its ratio ``r`` to the reference median.

    ``uniforms`` has shape ``(n, N_LAYERS, SPILL_DRAWS)``: presence, K, fraction, then per cell
    distance bin, position inside the bin, side, split weight.
    """

    n = ratio.shape[0]
    if uniforms.shape != (n, N_LAYERS, SPILL_DRAWS):
        raise ValueError(f"uniforms must have shape (n, {N_LAYERS}, {SPILL_DRAWS})")
    r_bin = np.searchsorted(SPILL_R_EDGES, ratio, side="right")
    spills = uniforms[..., 0] < table.probability[r_bin]
    k = 1 + _pmf_draw(table.cells_pmf[r_bin], uniforms[..., 1])
    fraction = interpolate_levels(table.levels, table.fraction_quantiles[r_bin], uniforms[..., 2])
    cells = np.where(spills, k, 0)

    per_cell = uniforms[..., 3:].reshape(n, N_LAYERS, K_MAX, 4)
    group = (ratio >= SPILL_DISTANCE_SPLIT_R).astype(int)
    pmf = np.broadcast_to(
        table.distance_pmf[group][:, :, None, :], (n, N_LAYERS, K_MAX, N_DISTANCE_BINS)
    ).copy()
    bins = _pmf_draw(pmf, per_cell[..., 0])
    low, high = DISTANCE_EDGES[bins], DISTANCE_EDGES[bins + 1]
    distance = low + np.floor(per_cell[..., 1] * (high - low)).astype(int)
    side = np.where(per_cell[..., 2] < 0.5, -1, 1)
    gamma = -np.log1p(-per_cell[..., 3] * (1.0 - 1e-12))  # standard exponential: Dirichlet(1, ..., 1)
    used = np.arange(K_MAX)[None, None, :] < cells[..., None]
    gamma = np.where(used, gamma, 0.0)
    total = gamma.sum(axis=-1, keepdims=True)
    weight = np.where(total > 0, gamma / np.where(total > 0, total, 1.0), 0.0)
    offset = np.where(used, side * distance, 0)
    return SpillDraw(
        fraction=np.where(spills, fraction, 0.0),
        cells=cells,
        offset=offset.astype(int),
        weight=weight,
    )


def describe(structure: CrossingStructure) -> dict[str, Any]:
    """A compact summary for the artifact manifest."""

    burst = structure.burst
    return {
        "burst_ratio": BURST_RATIO,
        "amplitude_bins": BURST_AMPLITUDE_BINS,
        "burst_probability": {
            f"{e:g}": float(p) for e, p in zip(burst.energies_gev, burst.probability, strict=True)
        },
        "bursts_calibrated": {
            f"{e:g}": int(c) for e, c in zip(burst.energies_gev, burst.counts, strict=True)
        },
        "spill": {
            "k_max": K_MAX,
            "cell_threshold_mev": SPILL_CELL_MEV,
            "ratio_edges": [float(x) for x in SPILL_R_EDGES],
            "distance_edges_cells": [int(x) for x in DISTANCE_EDGES],
            "layers_per_ratio_bin": {
                name: [int(c) for c in structure.spill[name].counts] for name in REPRESENTATIONS
            },
        },
    }
