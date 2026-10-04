"""Interacting protons: how a layer's energy is spread over its cells (lateral scale and quanta).

WORDING. A Geant4-DERIVED PHENOMENOLOGY (11.4.1, FTFP_BERT, this project's material model,
10-100 GeV, normal incidence), not a true proton shower model.

WHAT THIS IS. Step 2 of the interacting model (DEC-005, DEC-009): given the energy of a layer
that lies at or behind the interaction (``ams_ecal.proton_interacting``), place it on the 72
cells. Layers in front of the interaction are crossing-like and use the crossing spill.

MEASURED STRUCTURE (calibration events only). A layer's energy is centred on the track cell
with a core (``|d| <= 1`` cells, 53-73% of the energy) and a power-law halo (3-6% beyond 12
cells). The width of a layer is independent of its energy and about 72% of the variance of
``ln`` width is BETWEEN events: one event-level lateral scale latent, as the accepted design
says. The number of hit cells grows with the layer energy from 1 to about 21 and saturates.

THE MODEL. Per layer at offset ``k = l - l_D`` from the interaction layer:

* the core fraction ``c`` is ``sigmoid(mu_k + b (ln e - ln e_ref) + s_event z_event + s_layer eps)``:
  a mean by offset, an energy slope, one event-level normal shared by all layers (the
  lateral-scale latent) and a layer term;
* the energy is cut into ``N = max(1, Poisson(e / q(e)))`` QUANTA, with a quantum that grows
  with the layer's energy, ``q(e) = q (e / e_q)^kappa`` (measured: a layer of 10-30 MeV lights
  several times more cells than one quantum size fitted to the strong layers gives); each lands in the core with
  probability ``c`` (on the track cell with probability ``sigmoid(a + b logit c)`` - measured:
  the more concentrated the layer, the more of its core sits on one cell - else on a
  neighbour) or in the halo, whose distance follows an empirical pmf; the side is random;
* every quantum carries a Pareto weight with tail index ``alpha`` (a few quanta dominate a cell,
  as a single energetic particle does) and a cell's weight is the sum over its quanta; the
  layer's energy is shared in proportion to the cell weights, so energy is conserved exactly
  layer by layer.

Sparsity and hit multiplicity therefore EMERGE from the energy and the lateral spread: a weak
layer has few quanta and lights few cells. ``q`` and ``alpha`` are the two granularity
parameters, calibrated by matching the hit-cell, occupied-cell and top-cell-share statistics of
the calibration layers. The core fraction also depends on the layer's energy (measured: a
stronger layer is more concentrated), through a calibrated slope in ``ln e``. Finally the two
offsets (core fraction, centre-cell share) are shifted by a short fixed-point simulation so
that the ENERGY-WEIGHTED share on the centre cell and in the core match the calibration
layers: the validation observables are energy-weighted, and matching layer-by-layer means
alone left the centre cell over- and its neighbours under-weighted.

CALIBRATION touches CALIBRATION events only (the caller enforces the split).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit

from ams_ecal.crossing import FibreCrossingGeometry
from ams_ecal.proton_interacting import interaction_layer
from ams_ecal.proton_structure import N_LAYERS, REPRESENTATIONS, ln_energy_weights
from ams_ecal.tracking import TrackState

N_CELLS = 72
HALO_DISTANCES = N_CELLS - 2  # d = 2 .. 71
CELL_THRESHOLD_MEV = 0.2798003852367401  # half a measured MIP: the validation's hit threshold
OCCUPIED_MEV = 0.01
MIN_FIT_ENERGY_MEV = {"readout": 1.0, "deposition": 10.0}
MIN_LAYERS = 100
QUANTUM_GRID_MEV = {
    "readout": (0.15, 0.25, 0.4, 0.6, 0.9, 1.4),
    "deposition": (1.5, 2.5, 4.0, 6.0, 9.0, 14.0),
}
EQUAL_WEIGHTS = 1.0e6  # a tail index this large means every quantum weighs the same
TAIL_GRID = (0.5, 0.7, 1.0, 1.4, 2.0, 3.0, EQUAL_WEIGHTS)
GRANULARITY_LAYERS = 4000
CORE_CLIP = 0.02
MATCH_ITERATIONS = 8
QUANTUM_SLOPE_GRID = (0.0, 0.2, 0.4, 0.6, 0.8)
MIN_USABLE_MEV = 0.5


@dataclass(frozen=True, slots=True)
class LateralCalibrationInputs:
    """Cell grids of the calibration events that interacted, at one energy."""

    energy_gev: float
    depth_mm: np.ndarray  # (n,)
    grid_mev: dict[str, np.ndarray]  # rep -> (n, N_LAYERS, N_CELLS)
    track_cell: np.ndarray  # (n, N_LAYERS) the cell of the track coordinate in each layer

    def __post_init__(self) -> None:
        n = len(self.depth_mm)
        if self.track_cell.shape != (n, N_LAYERS):
            raise ValueError(f"track_cell must have shape (n, {N_LAYERS})")
        for name in REPRESENTATIONS:
            if self.grid_mev[name].shape != (n, N_LAYERS, N_CELLS):
                raise ValueError(f"grid_mev[{name!r}] must have shape (n, {N_LAYERS}, {N_CELLS})")

    @property
    def n_events(self) -> int:
        return len(self.depth_mm)


@dataclass(frozen=True, slots=True)
class LateralTable:
    """Calibrated lateral structure of the layers at and behind the interaction."""

    energies_gev: np.ndarray  # (A,)
    centre_intercept: np.ndarray  # (A, 2) logit P(a core quantum is on the track cell) at logit c = 0
    centre_slope: np.ndarray  # (A, 2) slope of that logit in the logit core fraction
    centre_first: np.ndarray  # (A, 2) extra logit at the interaction layer itself (offset 0)
    halo_pmf: np.ndarray  # (A, 2, HALO_DISTANCES) distance d = 2 .. 71, one side
    logit_mean: np.ndarray  # (A, 2, N_LAYERS) mean logit core fraction by offset k
    event_sd: np.ndarray  # (A, 2) sd of the event-level lateral-scale latent
    layer_sd: np.ndarray  # (A, 2) sd of the layer term
    quantum_mev: np.ndarray  # (2,) energy of one quantum
    tail_index: np.ndarray  # (2,) Pareto tail index of a quantum's weight
    quantum_slope: np.ndarray  # (2,) kappa: the quantum grows as (layer energy) ** kappa
    quantum_reference_mev: np.ndarray  # (2,) the layer energy at which the quantum is quantum_mev
    energy_slope: np.ndarray  # (A, 2) slope of the logit core fraction in ln layer energy
    log_energy_ref: np.ndarray  # (A, 2) ln energy at which the slope term vanishes
    counts: np.ndarray  # (A,) layers used per energy

    def __post_init__(self) -> None:
        a = len(self.energies_gev)
        shapes = {
            "centre_intercept": (a, 2),
            "centre_slope": (a, 2),
            "centre_first": (a, 2),
            "halo_pmf": (a, 2, HALO_DISTANCES),
            "logit_mean": (a, 2, N_LAYERS),
            "event_sd": (a, 2),
            "layer_sd": (a, 2),
            "quantum_mev": (2,),
            "tail_index": (2,),
            "quantum_slope": (2,),
            "quantum_reference_mev": (2,),
            "energy_slope": (a, 2),
            "log_energy_ref": (a, 2),
            "counts": (a,),
        }
        for name, shape in shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {getattr(self, name).shape}")
        for name in ("halo_pmf", "event_sd", "layer_sd", "quantum_mev", "tail_index"):
            values = getattr(self, name)
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.allclose(self.halo_pmf.sum(axis=-1), 1.0):
            raise ValueError("every halo pmf must sum to 1")
        if np.any(self.quantum_mev <= 0) or np.any(self.tail_index <= 0):
            raise ValueError("quantum_mev and tail_index must be positive")
        if not np.all(np.isfinite(self.quantum_slope)) or not np.all(
            np.isfinite(self.quantum_reference_mev)
        ):
            raise ValueError("quantum_slope and quantum_reference_mev must be finite")
        if np.any(self.quantum_reference_mev <= 0) or np.any(self.quantum_slope < 0):
            raise ValueError("quantum_reference_mev must be positive and quantum_slope nonnegative")
        for name in ("energy_slope", "log_energy_ref", "centre_intercept", "centre_slope", "centre_first"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"{name} must be finite")

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "lateral_energies_gev": np.asarray(self.energies_gev, dtype=float),
            "lateral_centre_intercept": self.centre_intercept,
            "lateral_centre_slope": self.centre_slope,
            "lateral_centre_first": self.centre_first,
            "lateral_halo_pmf": self.halo_pmf,
            "lateral_logit_mean": self.logit_mean,
            "lateral_event_sd": self.event_sd,
            "lateral_layer_sd": self.layer_sd,
            "lateral_quantum_mev": self.quantum_mev,
            "lateral_tail_index": self.tail_index,
            "lateral_quantum_slope": self.quantum_slope,
            "lateral_quantum_reference_mev": self.quantum_reference_mev,
            "lateral_energy_slope": self.energy_slope,
            "lateral_log_energy_ref": self.log_energy_ref,
            "lateral_counts": np.asarray(self.counts, dtype=np.int64),
        }

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> LateralTable:
        return cls(
            energies_gev=arrays["lateral_energies_gev"],
            centre_intercept=arrays["lateral_centre_intercept"],
            centre_slope=arrays["lateral_centre_slope"],
            centre_first=arrays["lateral_centre_first"],
            halo_pmf=arrays["lateral_halo_pmf"],
            logit_mean=arrays["lateral_logit_mean"],
            event_sd=arrays["lateral_event_sd"],
            layer_sd=arrays["lateral_layer_sd"],
            quantum_mev=arrays["lateral_quantum_mev"],
            tail_index=arrays["lateral_tail_index"],
            quantum_slope=np.asarray(arrays.get("lateral_quantum_slope", np.zeros(2)), dtype=float),
            quantum_reference_mev=np.asarray(
                arrays.get("lateral_quantum_reference_mev", np.ones(2)), dtype=float
            ),
            energy_slope=arrays["lateral_energy_slope"],
            log_energy_ref=arrays["lateral_log_energy_ref"],
            counts=arrays["lateral_counts"],
        )


# ----------------------------------------------------------------------
# Placing quanta
# ----------------------------------------------------------------------


def cell_probabilities(
    core_fraction: np.ndarray, core_centre: np.ndarray, halo_pmf: np.ndarray, centre_cell: np.ndarray
) -> np.ndarray:
    """Probability of a quantum landing in each of the 72 cells, ``(m, N_CELLS)``.

    A quantum is in the core with probability ``core_fraction``: on the centre cell with
    probability ``core_centre``, else on a neighbour (either side equally). Otherwise it is in
    the halo at distance ``d = 2 ..`` from the centre, either side equally. Probability that
    would fall off the grid is dropped and the rest renormalised.
    """

    cells = np.arange(N_CELLS)[None, :]
    distance = np.abs(cells - centre_cell[:, None])
    c = core_fraction[:, None]
    centre = np.asarray(core_centre, dtype=float)[:, None]
    halo = np.zeros(N_CELLS)
    halo[2:] = halo_pmf[: N_CELLS - 2]
    profile = np.where(distance == 0, c * centre, 0.0)
    profile = np.where(distance == 1, c * (1.0 - centre) / 2.0, profile)
    profile = np.where(distance >= 2, (1.0 - c) * halo[np.clip(distance, 0, N_CELLS - 1)] / 2.0, profile)
    total = profile.sum(axis=1, keepdims=True)
    return profile / np.where(total > 0, total, 1.0)


def place_quanta(
    rng: np.random.Generator,
    energy_mev: np.ndarray,
    core_fraction: np.ndarray,
    centre_cell: np.ndarray,
    core_centre: np.ndarray,
    halo_pmf: np.ndarray,
    quantum_mev: float,
    tail_index: float,
    quantum_slope: float = 0.0,
    quantum_reference_mev: float = 1.0,
) -> np.ndarray:
    """Cell energies ``(m, N_CELLS)`` of ``m`` layers; each row sums exactly to its layer energy.

    A layer of energy ``e`` is cut into Poisson(``e / q(e)``) quanta with
    ``q(e) = quantum_mev (e / quantum_reference_mev) ** quantum_slope``.
    """

    energy = np.asarray(energy_mev, dtype=float)
    probability = cell_probabilities(
        np.asarray(core_fraction, dtype=float),
        np.asarray(core_centre, dtype=float),
        halo_pmf,
        np.asarray(centre_cell),
    )
    out = np.zeros((len(energy), N_CELLS))
    for i in range(len(energy)):
        if energy[i] <= 0.0:
            continue
        quantum = quantum_mev * (energy[i] / quantum_reference_mev) ** quantum_slope
        n_quanta = max(1, int(rng.poisson(energy[i] / quantum)))
        counts = rng.multinomial(n_quanta, probability[i])
        cell_of_quantum = np.repeat(np.arange(N_CELLS), counts)
        if tail_index >= EQUAL_WEIGHTS:
            weights = counts.astype(float)
        else:  # Pareto weights: a few quanta dominate the cell they land in
            weights = np.bincount(
                cell_of_quantum,
                weights=(1.0 - rng.random(n_quanta)) ** (-1.0 / tail_index),
                minlength=N_CELLS,
            )
        total = weights.sum()
        if total > 0:
            out[i] = energy[i] * weights / total
    return out


# ----------------------------------------------------------------------
# The builder
# ----------------------------------------------------------------------


def layer_statistics(grid: np.ndarray, centre: np.ndarray) -> dict[str, np.ndarray]:
    """Per-layer statistics of ``(m, N_CELLS)`` grids about their centre cells."""

    energy = grid.sum(axis=1)
    distance = np.abs(np.arange(N_CELLS)[None, :] - centre[:, None])
    safe = np.where(energy > 0, energy, 1.0)
    return {
        "energy": energy,
        "core": np.where(distance <= 1, grid, 0.0).sum(axis=1) / safe,
        "hits": (grid > CELL_THRESHOLD_MEV).sum(axis=1),
        "occupied": (grid > OCCUPIED_MEV).sum(axis=1),
        "top": grid.max(axis=1) / safe,
    }


def _solve_mean(core: np.ndarray, shift: np.ndarray, sd: float) -> float:
    """The logit mean ``mu`` for which ``E[sigmoid(mu + shift + sd z)]`` equals the mean core fraction.

    The mean of a sigmoid of a noisy logit is not the sigmoid of the mean, so matching the mean
    logit alone leaves the generated core fraction biased.
    """

    nodes, weights = np.polynomial.hermite_e.hermegauss(15)
    weights = weights / weights.sum()
    target = float(core.mean())
    low, high = -9.0, 9.0
    for _ in range(40):
        mid = 0.5 * (low + high)
        value = (weights[None, :] * expit(mid + shift[:, None] + sd * nodes[None, :])).sum(axis=1).mean()
        if value < target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _energy_weighted_shifts(
    rng: np.random.Generator,
    grids: np.ndarray,
    centres: np.ndarray,
    offsets: np.ndarray,
    events: np.ndarray,
    parameters: dict[str, object],
) -> tuple[float, float]:
    """Shifts of the logit core mean and of the centre-cell logit that match the energy-weighted shares.

    ``grids`` are the calibration layers at and behind the interaction. The shares are the
    energy-weighted fractions in the core (``|d| <= 1``) and on the centre cell; the shifts are
    found in ``MATCH_ITERATIONS`` fixed-point steps of a simulation with the table's own latents.
    """

    layer = layer_statistics(grids, centres)
    energy = layer["energy"]
    keep = energy > 0
    grids, centres, offsets, events, energy = (
        grids[keep],
        centres[keep],
        offsets[keep],
        events[keep],
        energy[keep],
    )
    distance = np.abs(np.arange(N_CELLS)[None, :] - centres[:, None])

    def shares(cell_grids: np.ndarray) -> tuple[float, float]:
        total = cell_grids.sum()
        core = np.where(distance <= 1, cell_grids, 0.0).sum() / total
        centre = np.where(distance == 0, cell_grids, 0.0).sum() / total
        return float(core), float(centre)

    def tilt(target: float, got: float) -> float:
        return float(logit(np.clip(target, 1e-3, 1 - 1e-3)) - logit(np.clip(got, 1e-3, 1 - 1e-3)))

    target_core, target_centre = shares(grids)
    unique, inverse = np.unique(events, return_inverse=True)
    event_normal = rng.standard_normal(len(unique))[inverse]
    layer_normal = rng.standard_normal(len(energy))
    ln_e = np.log(np.maximum(energy, 1e-6))
    mu = np.asarray(parameters["logit_mean"])[offsets]
    delta_core = delta_centre = 0.0
    for _ in range(MATCH_ITERATIONS):
        core = expit(
            mu
            + delta_core
            + float(parameters["slope"]) * (ln_e - float(parameters["reference"]))
            + float(parameters["event_sd"]) * event_normal
            + float(parameters["layer_sd"]) * layer_normal
        )
        centre_logit = (
            float(parameters["centre_intercept"])
            + delta_centre
            + float(parameters["centre_slope"]) * logit(np.clip(core, CORE_CLIP, 1 - CORE_CLIP))
            + np.where(offsets == 0, float(parameters["centre_first"]), 0.0)
        )
        simulated = place_quanta(
            rng,
            energy,
            core,
            centres,
            expit(centre_logit),
            np.asarray(parameters["halo"]),
            float(parameters["quantum"]),
            float(parameters["tail"]),
            float(parameters["quantum_slope"]),
            float(parameters["quantum_reference"]),
        )
        got_core, got_centre = shares(simulated)
        delta_core += tilt(target_core, got_core)
        delta_centre += tilt(
            target_centre / max(target_core, 1e-9), got_centre / max(got_core, 1e-9)
        )
    return delta_core, delta_centre


def _collect_layers(
    entry: LateralCalibrationInputs, ecal_depth_mm: float, name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grids, centres, offsets and event index of the layers at and behind the interaction."""

    l_d = interaction_layer(entry.depth_mm, ecal_depth_mm)
    grids, centres, offsets, events = [], [], [], []
    for i in range(entry.n_events):
        for layer in range(l_d[i], N_LAYERS):
            grids.append(entry.grid_mev[name][i, layer])
            centres.append(entry.track_cell[i, layer])
            offsets.append(layer - l_d[i])
            events.append(i)
    return np.array(grids), np.array(centres), np.array(offsets), np.array(events)


def _granularity(
    rng: np.random.Generator,
    name: str,
    pool: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    centre_relation: tuple[float, float],
    halo_pmf: np.ndarray,
) -> tuple[float, float, float, float]:
    """Pick ``(quantum, shape, slope, reference)`` by matching hits, occupied cells and top-cell share.

    The quantum is the one at the reference energy (the median of the usable layers); ``slope`` is
    the exponent of its growth with the layer's energy.
    """

    grids = np.concatenate([p[0] for p in pool])
    centres = np.concatenate([p[1] for p in pool])
    take = rng.choice(len(grids), size=min(GRANULARITY_LAYERS, len(grids)), replace=False)
    grids, centres = grids[take], centres[take]
    data = layer_statistics(grids, centres)
    usable = data["energy"] > MIN_USABLE_MEV
    reference = float(np.median(data["energy"][usable]))
    edges = np.quantile(data["energy"][usable], np.linspace(0, 1, 7))
    which = np.clip(np.searchsorted(edges, data["energy"], side="right") - 1, 0, 5)
    core = np.clip(data["core"], CORE_CLIP, 1 - CORE_CLIP)
    centre_probability = expit(centre_relation[0] + centre_relation[1] * logit(core))
    best = (np.inf, 0.0, 0.0, 0.0)
    for quantum, shape, slope in itertools.product(QUANTUM_GRID_MEV[name], TAIL_GRID, QUANTUM_SLOPE_GRID):
        sim = layer_statistics(
            place_quanta(
                rng,
                data["energy"],
                core,
                centres,
                centre_probability,
                halo_pmf,
                quantum,
                shape,
                slope,
                reference,
            ),
            centres,
        )
        cost = 0.0
        for b in range(6):
            m = usable & (which == b)
            if m.sum() < 20:
                continue
            hits_data = np.median(data["hits"][m])
            occ_data = np.mean(data["occupied"][m])
            cost += abs(np.median(sim["hits"][m]) - hits_data) / (1 + hits_data)
            cost += abs(np.mean(sim["occupied"][m]) - occ_data) / (1 + occ_data)
            cost += abs(np.median(sim["top"][m]) - np.median(data["top"][m]))
            cost += abs(np.quantile(sim["top"][m], 0.9) - np.quantile(data["top"][m], 0.9))
        if cost < best[0]:
            best = (cost, quantum, shape, slope)
    return best[1], best[2], best[3], reference


def build_lateral(
    inputs: Sequence[LateralCalibrationInputs], *, ecal_depth_mm: float, seed: int = 20261004
) -> LateralTable:
    """Calibrate the lateral structure from calibration events."""

    entries = sorted(inputs, key=lambda entry: entry.energy_gev)
    n_a = len(entries)
    rng = np.random.default_rng(seed)
    centre_intercept, centre_slope = np.empty((n_a, 2)), np.empty((n_a, 2))
    centre_first = np.zeros((n_a, 2))
    halo = np.empty((n_a, 2, HALO_DISTANCES))
    logit_mean = np.empty((n_a, 2, N_LAYERS))
    event_sd, layer_sd = np.empty((n_a, 2)), np.empty((n_a, 2))
    counts = np.zeros(n_a, dtype=np.int64)
    quantum, shape = np.empty(2), np.empty(2)
    quantum_slope, quantum_reference = np.empty(2), np.empty(2)
    energy_slope, log_energy_ref = np.empty((n_a, 2)), np.empty((n_a, 2))

    for r, name in enumerate(REPRESENTATIONS):
        pool: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for a, entry in enumerate(entries):
            grids, centres, offsets, events = _collect_layers(entry, ecal_depth_mm, name)
            pool.append((grids, centres, offsets, events))
            layer = layer_statistics(grids, centres)
            fit = layer["energy"] >= MIN_FIT_ENERGY_MEV[name]
            if fit.sum() < MIN_LAYERS:
                raise ValueError(
                    f"only {int(fit.sum())} well-populated layers at {entry.energy_gev:g} GeV"
                )
            if r == 0:
                counts[a] = int(fit.sum())
            distance = np.abs(np.arange(N_CELLS)[None, :] - centres[:, None])
            share = grids / np.where(layer["energy"] > 0, layer["energy"], 1.0)[:, None]
            centre_share = np.where(distance == 0, share, 0.0).sum(axis=1)
            usable = fit & (layer["core"] > 0.05)
            y = logit(np.clip(centre_share[usable] / layer["core"][usable], 0.02, 0.98))
            x = logit(np.clip(layer["core"][usable], CORE_CLIP, 1 - CORE_CLIP))
            slope_c = float(np.cov(x, y)[0, 1] / max(x.var(ddof=1), 1e-9))
            centre_slope[a, r] = slope_c
            centre_intercept[a, r] = float(y.mean() - slope_c * x.mean())
            first = offsets[usable] == 0
            if first.sum() >= 30:
                centre_first[a, r] = float((y[first] - centre_intercept[a, r] - slope_c * x[first]).mean())
            pmf = np.array([share[fit][distance[fit] == d].sum() for d in range(2, N_CELLS)]) + 1e-6
            halo[a, r] = pmf / pmf.sum()

            logit_core = logit(np.clip(layer["core"], CORE_CLIP, 1 - CORE_CLIP))
            ln_e = np.log(np.maximum(layer["energy"], 1e-6))
            reference = float(ln_e[fit].mean())
            # slope of the logit core fraction in ln energy, from the variation WITHIN each offset
            x_centred, y_centred = np.zeros(len(ln_e)), np.zeros(len(ln_e))
            for k in range(N_LAYERS):
                chosen = fit & (offsets == k)
                if chosen.sum() >= 30:
                    x_centred[chosen] = ln_e[chosen] - ln_e[chosen].mean()
                    y_centred[chosen] = logit_core[chosen] - logit_core[chosen].mean()
            slope = float((x_centred * y_centred).sum() / max((x_centred**2).sum(), 1e-9))
            energy_slope[a, r], log_energy_ref[a, r] = slope, reference
            adjusted = logit_core - slope * (ln_e - reference)
            last = 0.0
            for k in range(N_LAYERS):
                chosen = fit & (offsets == k)
                if chosen.sum() >= 30:
                    last = float(adjusted[chosen].mean())
                logit_mean[a, r, k] = last
            residual = adjusted[fit] - logit_mean[a, r][offsets[fit]]
            event = events[fit]
            ids, sizes = np.unique(event, return_counts=True)
            means = np.array([residual[event == e].mean() for e in ids])
            multi = sizes >= 2
            within = np.mean(
                [residual[event == e].var(ddof=1) for e, ok in zip(ids, multi, strict=True) if ok]
            )
            # Var(event mean) = s_event^2 + s_layer^2 / n_layers_in_event
            noise = np.mean(within / sizes)
            event_sd[a, r] = float(np.sqrt(max(means.var() - noise, 1e-4)))
            layer_sd[a, r] = float(np.sqrt(max(within, 1e-4)))
            total_sd = float(np.hypot(event_sd[a, r], layer_sd[a, r]))
            shift = slope * (ln_e - reference)
            for k in range(N_LAYERS):
                chosen = fit & (offsets == k)
                if chosen.sum() >= 30:
                    logit_mean[a, r, k] = _solve_mean(layer["core"][chosen], shift[chosen], total_sd)
                elif k > 0:
                    logit_mean[a, r, k] = logit_mean[a, r, k - 1]
        relation = (float(centre_intercept[:, r].mean()), float(centre_slope[:, r].mean()))
        pooled_halo = halo[:, r].mean(axis=0)
        quantum[r], shape[r], quantum_slope[r], quantum_reference[r] = _granularity(
            rng, name, pool, relation, pooled_halo / pooled_halo.sum()
        )
        for a in range(n_a):
            grids_a, centres_a, offsets_a, events_a = pool[a]
            delta_core, delta_centre = _energy_weighted_shifts(
                rng,
                grids_a,
                centres_a,
                offsets_a,
                events_a,
                {
                    "logit_mean": logit_mean[a, r],
                    "slope": energy_slope[a, r],
                    "reference": log_energy_ref[a, r],
                    "event_sd": event_sd[a, r],
                    "layer_sd": layer_sd[a, r],
                    "centre_intercept": centre_intercept[a, r],
                    "centre_slope": centre_slope[a, r],
                    "centre_first": centre_first[a, r],
                    "halo": halo[a, r],
                    "quantum": quantum[r],
                    "tail": shape[r],
                    "quantum_slope": quantum_slope[r],
                    "quantum_reference": quantum_reference[r],
                },
            )
            logit_mean[a, r] += delta_core
            centre_intercept[a, r] += delta_centre
    return LateralTable(
        energies_gev=np.array([e.energy_gev for e in entries]),
        centre_intercept=centre_intercept,
        centre_slope=centre_slope,
        centre_first=centre_first,
        halo_pmf=halo,
        logit_mean=logit_mean,
        event_sd=event_sd,
        layer_sd=layer_sd,
        quantum_mev=quantum,
        tail_index=shape,
        quantum_slope=quantum_slope,
        quantum_reference_mev=quantum_reference,
        energy_slope=energy_slope,
        log_energy_ref=log_energy_ref,
        counts=counts,
    )


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------


def core_fractions(
    table: LateralTable,
    representation: int,
    energies_gev: np.ndarray,
    offset_k: np.ndarray,
    layer_energy_mev: np.ndarray,
    event_normal: np.ndarray,
    layer_normals: np.ndarray,
) -> np.ndarray:
    """Core fraction ``sigmoid(mu_k + b (ln e - ln e_ref) + s_event z + s_layer eps)`` for ``(n, m)`` layers."""

    lower, upper, w = ln_energy_weights(table.energies_gev, energies_gev)

    def blend(array: np.ndarray) -> np.ndarray:
        shape = (len(energies_gev),) + (1,) * (array.ndim - 1)
        return (1.0 - w).reshape(shape) * array[lower] + w.reshape(shape) * array[upper]

    mean = blend(table.logit_mean[:, representation])  # (n, N_LAYERS) by offset
    mu = np.take_along_axis(mean, np.clip(offset_k, 0, N_LAYERS - 1), axis=1)
    s_event = blend(table.event_sd[:, representation])[:, None]
    s_layer = blend(table.layer_sd[:, representation])[:, None]
    slope = blend(table.energy_slope[:, representation])[:, None]
    reference = blend(table.log_energy_ref[:, representation])[:, None]
    ln_e = np.log(np.maximum(layer_energy_mev, 1e-6))
    return expit(mu + slope * (ln_e - reference) + s_event * event_normal[:, None] + s_layer * layer_normals)


def blended_kernel(
    table: LateralTable, representation: int, energy_gev: float
) -> tuple[tuple[float, float, float], np.ndarray]:
    """Centre relation ``(a, b, first-layer shift)`` and halo pmf at one energy (interpolated in ``ln E``)."""

    lower, upper, w = ln_energy_weights(table.energies_gev, np.array([energy_gev]))
    lo, hi, weight = lower[0], upper[0], w[0]
    a = (1 - weight) * table.centre_intercept[lo, representation]
    a += weight * table.centre_intercept[hi, representation]
    b = (1 - weight) * table.centre_slope[lo, representation] + weight * table.centre_slope[hi, representation]
    shift = (1 - weight) * table.centre_first[lo, representation] + weight * table.centre_first[hi, representation]
    halo = (1 - weight) * table.halo_pmf[lo, representation] + weight * table.halo_pmf[hi, representation]
    return (float(a), float(b), float(shift)), halo / halo.sum()


def event_grid(
    table: LateralTable,
    representation: int,
    rng: np.random.Generator,
    energy_gev: float,
    layer_energy_mev: np.ndarray,
    first_layer: int,
    centre_cells: np.ndarray,
    event_normal: float,
    layer_normals: np.ndarray,
) -> np.ndarray:
    """Cell energies ``(N_LAYERS, N_CELLS)`` of the layers ``first_layer ..`` of one event."""

    layers = np.arange(first_layer, N_LAYERS)
    out = np.zeros((N_LAYERS, N_CELLS))
    if len(layers) == 0:
        return out
    fractions = core_fractions(
        table,
        representation,
        np.array([energy_gev]),
        (layers - first_layer)[None, :],
        layer_energy_mev[None, layers],
        np.array([event_normal]),
        layer_normals[None, layers],
    )[0]
    relation, halo = blended_kernel(table, representation, energy_gev)
    extra = np.zeros(len(layers))
    extra[0] = relation[2]  # the interaction layer itself
    centre = expit(relation[0] + relation[1] * logit(np.clip(fractions, CORE_CLIP, 1 - CORE_CLIP)) + extra)
    out[layers] = place_quanta(
        rng,
        layer_energy_mev[layers],
        fractions,
        centre_cells[layers],
        centre,
        halo,
        float(table.quantum_mev[representation]),
        float(table.tail_index[representation]),
        float(table.quantum_slope[representation]),
        float(table.quantum_reference_mev[representation]),
    )
    return out


def place_excess(
    table: LateralTable,
    representation: int,
    rng: np.random.Generator,
    energy_gev: float,
    excess_mev: np.ndarray,
    centre_cells: np.ndarray,
    event_normal: float,
    layer_normals: np.ndarray,
) -> np.ndarray:
    """Cell energies ``(N_LAYERS, N_CELLS)`` of energy placed laterally in layers IN FRONT of the interaction.

    ``excess_mev`` is, per layer, the energy above what the crossing proton itself deposits
    (backsplash from the shower behind). It is laid out with the lateral law of the interaction
    layer (offset 0) and the event's own lateral latents; a layer with no excess stays empty.
    """

    out = np.zeros((N_LAYERS, N_CELLS))
    layers = np.flatnonzero(excess_mev > 0.0)
    if len(layers) == 0:
        return out
    fractions = core_fractions(
        table,
        representation,
        np.array([energy_gev]),
        np.zeros((1, len(layers)), dtype=int),
        excess_mev[None, layers],
        np.array([event_normal]),
        layer_normals[None, layers],
    )[0]
    relation, halo = blended_kernel(table, representation, energy_gev)
    centre = expit(relation[0] + relation[1] * logit(np.clip(fractions, CORE_CLIP, 1 - CORE_CLIP)))
    out[layers] = place_quanta(
        rng,
        excess_mev[layers],
        fractions,
        centre_cells[layers],
        centre,
        halo,
        float(table.quantum_mev[representation]),
        float(table.tail_index[representation]),
        float(table.quantum_slope[representation]),
        float(table.quantum_reference_mev[representation]),
    )
    return out


def track_cells(crossing: FibreCrossingGeometry, track: TrackState) -> np.ndarray:
    """The cell of the track coordinate in every readout layer."""

    return np.array([crossing.track_cell(track, layer) for layer in range(N_LAYERS)])


_GRID_KEY = {"readout": "readout_grid_mev", "deposition": "deposit_grid_mev"}


def extract_lateral_inputs(
    energy_gev: float, arrays: dict[str, np.ndarray], crossing: FibreCrossingGeometry
) -> LateralCalibrationInputs:
    """Inputs from the CALIBRATION events of one batch whose truth says an interaction occurred."""

    rows = np.flatnonzero(arrays["truth_occurred"])
    cells = np.empty((len(rows), N_LAYERS), dtype=np.int64)
    for i, event in enumerate(rows):
        cells[i] = track_cells(
            crossing,
            TrackState(
                x0_mm=float(arrays["entry_x_mm"][event]),
                y0_mm=float(arrays["entry_y_mm"][event]),
                z0_mm=0.0,
                theta_rad=0.0,
                phi_rad=0.0,
            ),
        )
    return LateralCalibrationInputs(
        energy_gev=float(energy_gev),
        depth_mm=np.asarray(arrays["truth_z_mm"][rows], dtype=float),
        grid_mev={name: arrays[_GRID_KEY[name]][rows].astype(float) for name in REPRESENTATIONS},
        track_cell=cells,
    )
