"""Interacting protons: layer energies after and before the first inelastic interaction.

WORDING. A Geant4-DERIVED PHENOMENOLOGY (11.4.1, FTFP_BERT, this project's material model,
10-100 GeV, normal incidence), not a true proton shower model and not the AMS proton response.

WHAT THIS MODULE IS. Step 1 of the interacting model (DEC-005, EXP-006): the energy of every
readout layer of a proton that interacts at depth ``D`` inside the ECAL, in both
representations. Cell-level placement (lateral scale, quanta) is the next step and is NOT here.

THE ACCEPTED STRUCTURE AND HOW IT IS WRITTEN. DEC-005 factorises an interacting event into
amplitude x universal profile in the layer offset ``k = l - l_D`` from the interaction layer,
a correlated residual in LOG-energy space (its family tested before being named), an upstream
albedo component and a back-edge factor. Any amplitude defined from the layers it scales
forces those layers' residuals to sum to zero, and fitting a copula to constrained residuals
double counts the constraint. The equivalent, constraint-free form used here is:

* per layer offset ``k = 0 .. 17`` an EMPIRICAL quantile table of ``ln e_k`` per energy: this
  carries the universal profile AND the non-Gaussian residual law (the residual was measured
  skewed and heavy-tailed, so no parametric family is imposed);
* a Gaussian copula over the offsets with one COMMON factor and an AR(1) chain,
  ``rho(h) = lam + (1 - lam) * rho_ar ** h``: the common factor is the amplitude, the chain is
  the lag correlation;
* a BACK-EDGE shift of the normal scores per depth group (the amplitude is lower when the
  interaction is near the back, because returning particles come from material that is not
  there);
* a layer beyond the back simply does not exist, so truncation is geometry, as in the plan.

UPSTREAM. Layers in front of the interaction layer carry a MIP-like crossing response plus an
albedo excess (backward secondaries correlated with the shower's size). For ``j = 1 .. 6``
layers upstream there is an empirical quantile table of ``ln e`` per energy, like the layers
after the interaction. The six are tied to the shower through the normal score of the event's
AMPLITUDE (the geometric mean of its layers ``k = 0 .. 8``; measured rank correlation 0.5-0.7)
and to each other through one shared albedo latent. They are NOT scaled by the amplitude:
dividing by it made a tiny shower, whose amplitude is below the ordinary scatter of a crossing
layer, produce enormous ratios. Layers further upstream follow the crossing table.

REPRESENTATIONS. Readout and deposition share latents through a Gaussian coupling of their
normal scores (correlation measured in the calibration events), not an identical draw.

CALIBRATION touches CALIBRATION events only (the caller enforces the split).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from scipy import stats
from scipy.special import ndtr, ndtri

from ams_ecal.detector.crossing import FibreCrossingGeometry
from ams_ecal.proton_model.proton_structure import (
    N_LAYERS,
    REPRESENTATIONS,
    interpolate_levels,
    ln_energy_weights,
    sample_bulk,
)

WINDOW = 9  # offsets 0 .. 8 define the amplitude proxy
UPSTREAM = 6  # layers in front of the interaction layer that carry an albedo excess
FLOOR_MEV = 0.01  # a layer below this is recorded as exactly this in log space and generated as 0
MIN_EVENTS_PER_OFFSET = 30
MAX_LAG = 8
DEPTH_GROUP_EDGES_FRACTION = (1.0 / 3.0, 2.0 / 3.0)  # of the ECAL depth: front, middle, back third
N_DEPTH_GROUPS = len(DEPTH_GROUP_EDGES_FRACTION) + 1
INTERACTION_LEVELS = np.unique(
    np.concatenate(
        [
            [0.0, 0.002, 0.005, 0.01, 0.02, 0.035, 0.05],
            np.arange(0.1, 0.91, 0.05),
            [0.935, 0.95, 0.965, 0.98, 0.99, 0.995, 0.998, 1.0],
        ]
    )
)
# standard normals per event: for each of primary and secondary: factor (1) + 18 chain
# innovations; then for each: albedo factor (1) + 6 upstream innovations
N_NORMALS = 2 * (1 + N_LAYERS) + 2 * (1 + UPSTREAM)


# ----------------------------------------------------------------------
# Calibration inputs
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InteractingCalibrationInputs:
    """Layer quantities of the calibration events that interacted, at one energy."""

    energy_gev: float
    depth_mm: np.ndarray  # (n,) first-inelastic depth from the front face
    chord_mm: np.ndarray  # (n, N_LAYERS) the straight track's chord per layer
    layer_energy_mev: dict[str, np.ndarray]  # rep -> (n, N_LAYERS)

    def __post_init__(self) -> None:
        n = len(self.depth_mm)
        if self.chord_mm.shape != (n, N_LAYERS):
            raise ValueError(f"chord_mm must have shape (n, {N_LAYERS})")
        for name in REPRESENTATIONS:
            if self.layer_energy_mev[name].shape != (n, N_LAYERS):
                raise ValueError(f"layer_energy_mev[{name!r}] must have shape (n, {N_LAYERS})")
            if np.any(self.layer_energy_mev[name] < 0):
                raise ValueError("layer energies must be nonnegative")

    @property
    def n_events(self) -> int:
        return len(self.depth_mm)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def interaction_layer(depth_mm: np.ndarray, ecal_depth_mm: float) -> np.ndarray:
    """Readout layer containing each interaction depth (uniform layers)."""

    layer = np.floor(np.asarray(depth_mm, dtype=float) / (ecal_depth_mm / N_LAYERS)).astype(np.int64)
    return np.clip(layer, 0, N_LAYERS - 1)


def depth_group(depth_mm: np.ndarray, ecal_depth_mm: float) -> np.ndarray:
    """Front, middle or back third of the ECAL depth: 0, 1, 2."""

    edges = np.array(DEPTH_GROUP_EDGES_FRACTION) * ecal_depth_mm
    return np.searchsorted(edges, depth_mm, side="right")


def aligned_log_energy(layer_energy: np.ndarray, first_layer: np.ndarray) -> np.ndarray:
    """``ln max(e[l_D + k], floor)`` for k = 0 .. N_LAYERS - 1, NaN behind the last layer."""

    n = len(layer_energy)
    out = np.full((n, N_LAYERS), np.nan)
    for i in range(n):
        span = N_LAYERS - first_layer[i]
        out[i, :span] = np.log(np.maximum(layer_energy[i, first_layer[i] :], FLOOR_MEV))
    return out


def amplitude_proxy(log_energy: np.ndarray) -> np.ndarray:
    """Geometric mean of the layers k = 0 .. WINDOW - 1 that exist."""

    return np.exp(np.nanmean(log_energy[:, :WINDOW], axis=1))


def normal_scores(values: np.ndarray) -> np.ndarray:
    """Per-column normal scores of the finite entries (rank / (n + 1)); NaN stays NaN."""

    out = np.full(values.shape, np.nan)
    for column in range(values.shape[1]):
        finite = np.isfinite(values[:, column])
        if finite.sum() >= 3:
            ranks = stats.rankdata(values[finite, column])
            out[finite, column] = ndtri(ranks / (finite.sum() + 1.0))
    return out


def pairwise_lag_correlation(scores: np.ndarray, max_lag: int = MAX_LAG) -> np.ndarray:
    """Mean correlation of the columns ``h`` apart over events where both exist, h = 1 .. max_lag."""

    out = np.full(max_lag, np.nan)
    for lag in range(1, max_lag + 1):
        values = []
        for column in range(scores.shape[1] - lag):
            both = np.isfinite(scores[:, column]) & np.isfinite(scores[:, column + lag])
            if both.sum() >= MIN_EVENTS_PER_OFFSET:
                values.append(np.corrcoef(scores[both, column], scores[both, column + lag])[0, 1])
        if values:
            out[lag - 1] = float(np.mean(values))
    return out


def fit_factor_chain(lag_correlation: np.ndarray) -> tuple[float, float]:
    """Fit ``rho(h) = lam + (1 - lam) rho_ar ** h`` to the lag correlations by least squares."""

    lags = np.arange(1, len(lag_correlation) + 1)
    ok = np.isfinite(lag_correlation)
    best = (np.inf, 0.0, 0.0)
    for rho_ar in np.linspace(0.0, 0.99, 100):
        basis = rho_ar ** lags[ok]
        # rho(h) = lam (1 - basis) + basis  ->  linear in lam
        weight = 1.0 - basis
        target = lag_correlation[ok] - basis
        lam = float(np.clip((weight * target).sum() / max((weight**2).sum(), 1e-12), 0.0, 0.98))
        cost = float(((lam * weight + basis - lag_correlation[ok]) ** 2).sum())
        if cost < best[0]:
            best = (cost, lam, float(rho_ar))
    return best[1], best[2]


def gaussian_correlation(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Correlation of two arrays of normal scores over the entries where both are finite."""

    both = np.isfinite(scores_a) & np.isfinite(scores_b)
    if both.sum() < MIN_EVENTS_PER_OFFSET:
        return 0.0
    return float(np.clip(np.corrcoef(scores_a[both], scores_b[both])[0, 1], 0.0, 0.995))


def pairwise_mean_correlation(scores: np.ndarray) -> float:
    """Mean correlation over all column pairs of normal scores (pairwise complete)."""

    values = []
    for i in range(scores.shape[1]):
        for j in range(i + 1, scores.shape[1]):
            both = np.isfinite(scores[:, i]) & np.isfinite(scores[:, j])
            if both.sum() >= MIN_EVENTS_PER_OFFSET:
                values.append(np.corrcoef(scores[both, i], scores[both, j])[0, 1])
    return float(np.mean(values)) if values else 0.0


# ----------------------------------------------------------------------
# The table
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InteractingTable:
    """Calibrated layer-energy structure of interacting protons."""

    energies_gev: np.ndarray  # (A,)
    levels: np.ndarray  # (Q,)
    log_quantiles: dict[str, np.ndarray]  # rep -> (A, N_LAYERS, Q): ln e at offset k
    factor: np.ndarray  # (A, 2) common-factor loading lam, per representation
    chain: np.ndarray  # (A, 2) AR coefficient rho_ar
    backedge_shift: np.ndarray  # (2, N_DEPTH_GROUPS) normal-score shift per depth group
    amp_data_quantiles: np.ndarray  # (A, 2, G, Q): ln amplitude of the calibration events, per depth group
    amp_model_quantiles: np.ndarray  # (A, 2, G, Q): the same for the copula alone, before the mapping
    representation_coupling: np.ndarray  # (A,) correlation of the two representations' scores
    upstream_quantiles: dict[str, np.ndarray]  # rep -> (A, UPSTREAM, Q): ln e, j layers upstream
    amplitude_quantiles: np.ndarray  # (A, 2, Q): ln of the amplitude proxy, for its normal score
    amplitude_loading: np.ndarray  # (A, 2) correlation of an upstream score with the amplitude score
    albedo_factor: np.ndarray  # (A, 2) shared albedo latent, correlation left after the amplitude
    upstream_coupling: np.ndarray  # (A,) correlation of the representations' upstream scores
    counts: np.ndarray  # (A,) interacting calibration events per energy

    def __post_init__(self) -> None:
        a, q = len(self.energies_gev), len(self.levels)
        for name in REPRESENTATIONS:
            if self.log_quantiles[name].shape != (a, N_LAYERS, q):
                raise ValueError(f"log_quantiles[{name!r}] must have shape {(a, N_LAYERS, q)}")
            if self.upstream_quantiles[name].shape != (a, UPSTREAM, q):
                raise ValueError(f"upstream_quantiles[{name!r}] must have shape {(a, UPSTREAM, q)}")
            for table in (self.log_quantiles[name], self.upstream_quantiles[name]):
                if not np.all(np.isfinite(table)) or np.any(np.diff(table, axis=-1) < 0):
                    raise ValueError("quantile tables must be finite and non-decreasing")
        shapes = {
            "factor": (a, 2),
            "chain": (a, 2),
            "backedge_shift": (2, N_DEPTH_GROUPS),
            "amp_data_quantiles": (a, 2, N_DEPTH_GROUPS, q),
            "amp_model_quantiles": (a, 2, N_DEPTH_GROUPS, q),
            "representation_coupling": (a,),
            "albedo_factor": (a, 2),
            "amplitude_loading": (a, 2),
            "amplitude_quantiles": (a, 2, q),
            "upstream_coupling": (a,),
            "counts": (a,),
        }
        for name, shape in shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {getattr(self, name).shape}")
        for name in (
            "factor",
            "chain",
            "albedo_factor",
            "amplitude_loading",
            "representation_coupling",
            "upstream_coupling",
        ):
            values = getattr(self, name)
            if np.any(values < 0) or np.any(values >= 1):
                raise ValueError(f"{name} must lie in [0, 1)")

    def arrays(self) -> dict[str, np.ndarray]:
        out = {
            "inter_energies_gev": np.asarray(self.energies_gev, dtype=float),
            "inter_levels": np.asarray(self.levels, dtype=float),
            "inter_factor": self.factor,
            "inter_chain": self.chain,
            "inter_backedge_shift": self.backedge_shift,
            "inter_amp_data_quantiles": self.amp_data_quantiles,
            "inter_amp_model_quantiles": self.amp_model_quantiles,
            "inter_representation_coupling": self.representation_coupling,
            "inter_albedo_factor": self.albedo_factor,
            "inter_amplitude_loading": self.amplitude_loading,
            "inter_amplitude_quantiles": self.amplitude_quantiles,
            "inter_upstream_coupling": self.upstream_coupling,
            "inter_counts": np.asarray(self.counts, dtype=np.int64),
        }
        for name in REPRESENTATIONS:
            out[f"inter_log_quantiles_{name}"] = self.log_quantiles[name]
            out[f"inter_upstream_quantiles_{name}"] = self.upstream_quantiles[name]
        return out

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> InteractingTable:
        return cls(
            energies_gev=arrays["inter_energies_gev"],
            levels=arrays["inter_levels"],
            log_quantiles={n: arrays[f"inter_log_quantiles_{n}"] for n in REPRESENTATIONS},
            factor=arrays["inter_factor"],
            chain=arrays["inter_chain"],
            backedge_shift=arrays["inter_backedge_shift"],
            amp_data_quantiles=arrays["inter_amp_data_quantiles"],
            amp_model_quantiles=arrays["inter_amp_model_quantiles"],
            representation_coupling=arrays["inter_representation_coupling"],
            upstream_quantiles={n: arrays[f"inter_upstream_quantiles_{n}"] for n in REPRESENTATIONS},
            albedo_factor=arrays["inter_albedo_factor"],
            amplitude_loading=arrays["inter_amplitude_loading"],
            amplitude_quantiles=arrays["inter_amplitude_quantiles"],
            upstream_coupling=arrays["inter_upstream_coupling"],
            counts=arrays["inter_counts"],
        )


# ----------------------------------------------------------------------
# The builder
# ----------------------------------------------------------------------


def _quantile_rows(values: np.ndarray, levels: np.ndarray, what: str) -> np.ndarray:
    """Quantile table per column; a column with too few entries copies the last good one."""

    table = np.empty((values.shape[1], len(levels)))
    last = None
    for column in range(values.shape[1]):
        finite = values[:, column][np.isfinite(values[:, column])]
        if len(finite) >= MIN_EVENTS_PER_OFFSET:
            last = np.quantile(finite, levels)
        if last is None:
            raise ValueError(f"too few interacting events to calibrate the first {what}")
        table[column] = last
    return table


def build_interacting(
    inputs: Sequence[InteractingCalibrationInputs],
    *,
    ecal_depth_mm: float,
    levels: np.ndarray = INTERACTION_LEVELS,
) -> InteractingTable:
    """Calibrate the interacting layer structure from calibration events."""

    inputs = sorted(inputs, key=lambda entry: entry.energy_gev)
    n_a = len(inputs)
    log_q = {name: np.empty((n_a, N_LAYERS, len(levels))) for name in REPRESENTATIONS}
    up_q = {name: np.empty((n_a, UPSTREAM, len(levels))) for name in REPRESENTATIONS}
    factor, chain, albedo = np.empty((n_a, 2)), np.empty((n_a, 2)), np.empty((n_a, 2))
    amp_loading = np.empty((n_a, 2))
    amp_q = np.empty((n_a, 2, len(levels)))
    coupling, up_coupling = np.empty(n_a), np.empty(n_a)
    counts = np.empty(n_a, dtype=np.int64)
    window_means: list[list[tuple[np.ndarray, np.ndarray]]] = []
    ln_amplitudes: list[list[np.ndarray]] = []
    groups: list[np.ndarray] = []

    for a, entry in enumerate(inputs):
        l_d = interaction_layer(entry.depth_mm, ecal_depth_mm)
        group = depth_group(entry.depth_mm, ecal_depth_mm)
        counts[a] = entry.n_events
        scores: list[np.ndarray] = []
        up_scores: list[np.ndarray] = []
        ln_amps: list[np.ndarray] = []
        per_rep: list[tuple[np.ndarray, np.ndarray]] = []
        for r, name in enumerate(REPRESENTATIONS):
            ln_e = aligned_log_energy(entry.layer_energy_mev[name], l_d)
            log_q[name][a] = _quantile_rows(ln_e, levels, "offset")
            z = normal_scores(ln_e)
            scores.append(z)
            factor[a, r], chain[a, r] = fit_factor_chain(pairwise_lag_correlation(z))
            ln_amplitude = np.log(amplitude_proxy(ln_e))
            ln_amps.append(ln_amplitude)
            amp_q[a, r] = np.quantile(ln_amplitude, levels)
            amp_score = normal_scores(ln_amplitude[:, None])[:, 0]
            per_rep.append((np.nanmean(z[:, :WINDOW], axis=1), group))

            # upstream: ln e of the layer j in front of the interaction layer
            ln_up = np.full((entry.n_events, UPSTREAM), np.nan)
            for j in range(1, UPSTREAM + 1):
                rows = np.flatnonzero(l_d - j >= 0)
                ln_up[rows, j - 1] = np.log(
                    np.maximum(entry.layer_energy_mev[name][rows, l_d[rows] - j], FLOOR_MEV)
                )
            up_q[name][a] = _quantile_rows(ln_up, levels, "upstream layer")
            zu = normal_scores(ln_up)
            up_scores.append(zu)
            loading = float(
                np.mean([gaussian_correlation(zu[:, j], amp_score) for j in range(UPSTREAM)])
            )
            amp_loading[a, r] = loading
            mean_pair = pairwise_mean_correlation(zu)
            albedo[a, r] = np.clip((mean_pair - loading**2) / max(1.0 - loading**2, 1e-6), 0.0, 0.98)
        window_means.append(per_rep)
        ln_amplitudes.append(ln_amps)
        groups.append(group)
        coupling[a] = gaussian_correlation(scores[0], scores[1])
        up_coupling[a] = gaussian_correlation(up_scores[0], up_scores[1])

    backedge = np.zeros((2, N_DEPTH_GROUPS))
    for r in range(2):
        pooled = np.concatenate([per_rep[r][0] for per_rep in window_means])
        pooled_group = np.concatenate([per_rep[r][1] for per_rep in window_means])
        finite = np.isfinite(pooled)
        overall = pooled[finite].mean()
        for g in range(N_DEPTH_GROUPS):
            selected = finite & (pooled_group == g)
            backedge[r, g] = pooled[selected].mean() - overall if selected.any() else 0.0
    preliminary = InteractingTable(
        energies_gev=np.array([e.energy_gev for e in inputs]),
        levels=np.asarray(levels, dtype=float),
        log_quantiles=log_q,
        factor=np.clip(factor, 0.0, 0.98),
        chain=np.clip(chain, 0.0, 0.99),
        backedge_shift=backedge,
        amp_data_quantiles=np.zeros((n_a, 2, N_DEPTH_GROUPS, len(levels))),
        amp_model_quantiles=np.zeros((n_a, 2, N_DEPTH_GROUPS, len(levels))),
        representation_coupling=np.clip(coupling, 0.0, 0.995),
        upstream_quantiles=up_q,
        albedo_factor=np.clip(albedo, 0.0, 0.98),
        amplitude_loading=np.clip(amp_loading, 0.0, 0.98),
        amplitude_quantiles=amp_q,
        upstream_coupling=np.clip(up_coupling, 0.0, 0.995),
        counts=counts,
    )
    # the amplitude MAPPING: the copula alone, simulated with the calibration depths, gives the
    # model's own amplitude law; the mapping sends it onto the calibration events' law
    rng = np.random.default_rng(20261004)
    data_q = np.empty((n_a, 2, N_DEPTH_GROUPS, len(levels)))
    model_q = np.empty_like(data_q)
    for a, entry in enumerate(inputs):
        n_sim = 20000
        depth = entry.depth_mm[rng.integers(0, entry.n_events, n_sim)]
        l_d = interaction_layer(depth, ecal_depth_mm)
        group = depth_group(depth, ecal_depth_mm)
        simulated = _post_log_energies(
            preliminary,
            np.full(n_sim, entry.energy_gev),
            l_d,
            group,
            rng.standard_normal((n_sim, N_NORMALS)),
            apply_map=False,
        )
        for r, name in enumerate(REPRESENTATIONS):
            for g in range(N_DEPTH_GROUPS):
                observed = ln_amplitudes[a][r][groups[a] == g]
                if len(observed) < MIN_EVENTS_PER_OFFSET:
                    observed = ln_amplitudes[a][r]
                modelled = simulated[name][1][group == g]
                data_q[a, r, g] = np.quantile(observed, levels)
                model_q[a, r, g] = np.quantile(modelled, levels)
    return replace(preliminary, amp_data_quantiles=data_q, amp_model_quantiles=model_q)


def invert_quantiles(levels: np.ndarray, values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Level at which each row's quantile function ``values`` (``(n, Q)``) reaches ``x`` (``(n,)``)."""

    index = np.clip((values < x[:, None]).sum(axis=1), 1, values.shape[1] - 1)
    rows = np.arange(len(x))
    low, high = values[rows, index - 1], values[rows, index]
    span = np.where(high > low, high - low, 1.0)
    fraction = np.clip((x - low) / span, 0.0, 1.0)
    return levels[index - 1] + fraction * (levels[index] - levels[index - 1])


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------


def _chain(innovations: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Unit-variance AR(1) chain: ``z_0 = e_0``, ``z_k = rho z_{k-1} + sqrt(1 - rho^2) e_k``."""

    n, length = innovations.shape
    out = np.empty((n, length))
    out[:, 0] = innovations[:, 0]
    scale = np.sqrt(1.0 - rho**2)
    for k in range(1, length):
        out[:, k] = rho * out[:, k - 1] + scale * innovations[:, k]
    return out


def _post_log_energies(
    table: InteractingTable,
    energies: np.ndarray,
    l_d: np.ndarray,
    group: np.ndarray,
    normals: np.ndarray,
    *,
    apply_map: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``ln e`` by layer offset ``(n, N_LAYERS)`` and ``ln`` amplitude ``(n,)``, per representation.

    With ``apply_map`` every layer of an event is shifted together so that the event's amplitude
    follows the calibration events' law of its depth group (the copula alone gives a Gaussian
    common level, which cannot make a whole shower weak at once).
    """

    n = len(energies)
    lower, upper, w = ln_energy_weights(table.energies_gev, energies)

    def blend(array: np.ndarray) -> np.ndarray:
        shape = (n,) + (1,) * (array.ndim - 1)
        return (1.0 - w).reshape(shape) * array[lower] + w.reshape(shape) * array[upper]

    primary_factor = normals[:, 0]
    primary_chain = normals[:, 1 : 1 + N_LAYERS]
    secondary_factor = normals[:, 1 + N_LAYERS]
    secondary_chain = normals[:, 2 + N_LAYERS : 2 + 2 * N_LAYERS]
    lam, rho_ar = blend(table.factor), blend(table.chain)
    coupling = blend(table.representation_coupling)
    z_primary = (
        np.sqrt(lam[:, 0])[:, None] * primary_factor[:, None]
        + np.sqrt(1 - lam[:, 0])[:, None] * _chain(primary_chain, rho_ar[:, 0])
    )
    z_secondary = (
        np.sqrt(lam[:, 1])[:, None] * secondary_factor[:, None]
        + np.sqrt(1 - lam[:, 1])[:, None] * _chain(secondary_chain, rho_ar[:, 1])
    )
    z_by_rep = {
        "readout": z_primary,
        "deposition": coupling[:, None] * z_primary + np.sqrt(1 - coupling**2)[:, None] * z_secondary,
    }
    k = np.arange(N_LAYERS)[None, :]
    window = (k < (N_LAYERS - l_d)[:, None]) & (k < WINDOW)
    rows = np.arange(n)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for r, name in enumerate(REPRESENTATIONS):
        z = z_by_rep[name] + table.backedge_shift[r][group][:, None]
        log_q = blend(table.log_quantiles[name])
        ln_e = np.empty((n, N_LAYERS))
        for offset in range(N_LAYERS):
            ln_e[:, offset] = interpolate_levels(table.levels, log_q[:, offset, :], ndtr(z[:, offset]))
        ln_amp = np.where(window, ln_e, 0.0).sum(axis=1) / np.maximum(window.sum(axis=1), 1)
        if apply_map:
            model_q = blend(table.amp_model_quantiles)[rows, r, group]
            data_q = blend(table.amp_data_quantiles)[rows, r, group]
            level = invert_quantiles(table.levels, model_q, ln_amp)
            target = interpolate_levels(table.levels, data_q, level)
            ln_e = ln_e + (target - ln_amp)[:, None]
            ln_amp = target
        out[name] = (ln_e, ln_amp)
    return out


def sample_interacting_layers(
    table: InteractingTable,
    crossing_table,
    energies_gev: np.ndarray,
    depth_mm: np.ndarray,
    ecal_depth_mm: float,
    chord_bins: np.ndarray,
    normals: np.ndarray,
    bulk_uniforms: np.ndarray,
) -> dict[str, np.ndarray]:
    """Layer energies ``(n, N_LAYERS)`` of interacting events in both representations.

    ``normals`` is ``(n, N_NORMALS)`` and ``bulk_uniforms`` ``(n, 2, N_LAYERS)`` (readout, then
    deposition); the order is fixed so that a seed always names the same event.
    """

    energies = np.atleast_1d(np.asarray(energies_gev, dtype=float))
    n = len(energies)
    if normals.shape != (n, N_NORMALS) or bulk_uniforms.shape != (n, 2, N_LAYERS):
        raise ValueError(f"normals must be (n, {N_NORMALS}) and bulk_uniforms (n, 2, {N_LAYERS})")
    lower, upper, w = ln_energy_weights(table.energies_gev, energies)

    def blend(array: np.ndarray) -> np.ndarray:
        shape = (n,) + (1,) * (array.ndim - 1)
        return (1.0 - w).reshape(shape) * array[lower] + w.reshape(shape) * array[upper]

    l_d = interaction_layer(depth_mm, ecal_depth_mm)
    group = depth_group(depth_mm, ecal_depth_mm)
    rows = np.arange(n)
    offset = np.arange(N_LAYERS)[None, :] - l_d[:, None]  # k of each layer l
    k_index = np.clip(offset, 0, N_LAYERS - 1)

    cursor = 2 * (1 + N_LAYERS)
    albedo_primary = normals[:, cursor]
    upstream_primary = normals[:, cursor + 1 : cursor + 1 + UPSTREAM]
    cursor += 1 + UPSTREAM
    albedo_secondary = normals[:, cursor]
    upstream_secondary = normals[:, cursor + 1 : cursor + 1 + UPSTREAM]
    post = _post_log_energies(table, energies, l_d, group, normals, apply_map=True)

    albedo = blend(table.albedo_factor)
    loading = blend(table.amplitude_loading)
    up_coupling = blend(table.upstream_coupling)

    out: dict[str, np.ndarray] = {}
    z_up_readout = np.zeros((n, UPSTREAM))  # set by the readout pass, used by the deposition pass
    for r, name in enumerate(REPRESENTATIONS):
        ln_e, ln_amplitude = post[name]
        by_layer = np.take_along_axis(ln_e, k_index, axis=1)  # in layer index
        energy = np.exp(by_layer)
        energy = np.where(energy <= FLOOR_MEV * 1.0001, 0.0, energy)

        # --- the amplitude's normal score, from its calibrated distribution ---------------
        amp_levels = blend(table.amplitude_quantiles)[:, r, :]  # (n, Q) of ln amp
        rank = np.array(
            [np.interp(ln_amplitude[i], amp_levels[i], table.levels) for i in range(n)]
        )
        amplitude_score = ndtri(np.clip(rank, 1e-4, 1 - 1e-4))

        # --- upstream layers: crossing response, plus an albedo tied to the shower ------
        bulk = sample_bulk(
            crossing_table.quantiles_mev[name],
            crossing_table.levels,
            crossing_table.energies_gev,
            chord_bins,
            energies,
            bulk_uniforms[:, r, :],
        )
        energy = np.where(offset < 0, bulk, energy)  # in front: crossing only ...
        load, rho_alb = loading[:, r], albedo[:, r]
        own = np.sqrt(np.maximum(1.0 - load**2, 0.0))
        shared = np.sqrt(rho_alb)[:, None] * (albedo_primary if r == 0 else albedo_secondary)[:, None]
        single = np.sqrt(1.0 - rho_alb)[:, None] * (upstream_primary if r == 0 else upstream_secondary)
        z_up = load[:, None] * amplitude_score[:, None] + own[:, None] * (shared + single)
        if name == "readout":
            z_up_readout = z_up
        else:  # the deposition's upstream scores are coupled to the readout's
            z_up = up_coupling[:, None] * z_up_readout + np.sqrt(1 - up_coupling**2)[:, None] * z_up
        up_q = blend(table.upstream_quantiles[name])  # (n, UPSTREAM, Q)
        for j in range(1, UPSTREAM + 1):  # ... plus an albedo in the nearest layers
            ln_x = interpolate_levels(table.levels, up_q[:, j - 1, :], ndtr(z_up[:, j - 1]))
            target = l_d - j
            valid = target >= 0
            value = np.exp(ln_x[valid])
            energy[rows[valid], target[valid]] = np.where(value <= FLOOR_MEV * 1.0001, 0.0, value)
        out[name] = energy
    return out


_GRID_KEY = {"readout": "readout_grid_mev", "deposition": "deposit_grid_mev"}


def extract_interacting_inputs(
    energy_gev: float, arrays: dict[str, np.ndarray], crossing_geometry: FibreCrossingGeometry
) -> InteractingCalibrationInputs:
    """Inputs from the CALIBRATION events of one batch whose truth says an inelastic interaction occurred."""

    from ams_ecal.detector.tracking import TrackState

    rows = np.flatnonzero(arrays["truth_occurred"])
    chord = np.empty((len(rows), N_LAYERS))
    for i, event in enumerate(rows):
        track = TrackState(
            x0_mm=float(arrays["entry_x_mm"][event]),
            y0_mm=float(arrays["entry_y_mm"][event]),
            z0_mm=0.0,
            theta_rad=0.0,
            phi_rad=0.0,
        )
        chord[i] = crossing_geometry.cross(track).layer_path_mm(N_LAYERS)
    return InteractingCalibrationInputs(
        energy_gev=float(energy_gev),
        depth_mm=np.asarray(arrays["truth_z_mm"][rows], dtype=float),
        chord_mm=chord,
        layer_energy_mev={
            name: arrays[_GRID_KEY[name]][rows].sum(axis=2).astype(float) for name in REPRESENTATIONS
        },
    )
