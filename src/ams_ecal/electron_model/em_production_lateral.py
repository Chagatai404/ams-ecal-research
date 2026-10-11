"""The ``em_production`` electron generator, Slice 2: the Grindhammer-Peters lateral spot model (candidate, development).

Decision record ``research/DECISIONS.md`` (DEC-014, F2 and the 2026-10-07 rows), note
``research/plans/2026-10-07_em_lateral_structure_note.md`` (literature, measurement and the approved formulation).
Slice 1 (``em_production.py``, the locked gamma longitudinal family) and DEC-001 are NOT changed; this module adds the
lateral structure on top of Slice 1's longitudinal draws. Words: calibrated, candidate, development; never validated.

WHAT ONE EVENT IS. Slice 1 draws the event's longitudinal shape ``(T_i, alpha_i)`` and the layer energies ``E_l``. This
module then gives every layer a radial law and a spot structure:

    tau_i(l)  = (t_l / T_ref) <t>_ref / <t>_i ,   <t> = T alpha / (alpha - 1)             (Grindhammer-Peters Eq. 34)
    f(r)      = p 2 r R_C^2 / (r^2 + R_C^2)^2 + (1 - p) 2 r R_T^2 / (r^2 + R_T^2)^2       (Eq. 23; AMS Eq. 2)
    R_C       = z1 + z2 tau,  z1 = z1a + z1b ln E                                          (Eq. 24)
    R_T       = k1 {exp(k3 (tau - k2)) + exp(k4 (tau - k2))},  k4 = k4a + k4b ln E        (Eq. 25)
    p         = p1 exp{(p2 - tau)/p3 - exp((p2 - tau)/p3)},  p3 = p3a + p3b ln E          (Eq. 26)

with the radii in units of the Moliere radius R_M = 18 mm (the geometry's nominal value; the researcher's choice,
2026-10-07) and E in GeV. The layer's ``N_l`` spots of equal energy ``E_l / N_l`` are placed independently from ``f``;
the count in a one-coordinate cell strip is multinomial with the closed-form strip fractions
``F(x) = (1 + x / sqrt(x^2 + R^2)) / 2`` of each component, which is distributionally identical to drawing radii and
azimuths and is vectorised. Spots that fall outside the 72 cells are lateral leakage and are not renormalised.

    N_spot = A E^b ,   N_l = N_spot x (mass of layer l under a gamma of shape kappa_a alpha_ref, scale kappa_T T_ref)

PROVENANCE. The functional forms and the spot structure are externally established (Grindhammer and Peters,
hep-ex/0001020) and used by AMS (Kounine et al. NIM A 869 Eqs. 2-3; Chin. Phys. C 40 096204 Eq. 2): a transferred
structure. The twelve radial constants and the four spot constants are CALIBRATED to the exposed Geant4 electrons
(project phenomenological values; the published constants are not used, because they are 4-11 points off at depth).
Not modelled: layer-to-layer residual noise (plan stage 4), the 2D structure of the fibre direction (one coordinate
per layer), tilted showers beyond the straight-line projection, the detector response.

SCOPE AND REFUSALS. 10-100 GeV (Slice 1's range, enforced by its longitudinal model); a radial-law width parameter
that is not positive, a Moliere radius that differs from the geometry's, or a longitudinal artifact that differs from
the one the lateral artifact was calibrated on are refused.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from math import isfinite, log
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import betainc, gammainc

from ams_ecal.detector.event import ECALEvent, EnergyGrid, EventProvenance
from ams_ecal.detector.geometry import ECALGeometry
from ams_ecal.detector.readout import measured_axis_for_fiber
from ams_ecal.detector.tracking import TrackState, project_track_to_z
from ams_ecal.electron_model.em_production import (
    DEFAULT_ARTIFACT as LONGITUDINAL_ARTIFACT,
)
from ams_ecal.electron_model.em_production import (
    MIN_SHAPE,
    PROJECT_ROOT,
    EMProductionLongitudinalModel,
    git_provenance,
    spawn_event_seeds,
)
from ams_ecal.electron_model.em_production import (
    build_model as build_longitudinal_model,
)
from ams_ecal.electron_model.em_production import (
    read_parameters_artifact as read_longitudinal_artifact,
)

MODEL_NAME = "em-production-gp-lateral"
SCHEMA_VERSION = 1
DEFAULT_ARTIFACT = PROJECT_ROOT / "data" / "calibration" / "em_production" / "gp_lateral_v1" / "parameters.json"
MIN_RADIUS_MM = 0.5  # a 9 mm cell does not resolve a core below about 1 mm
MIN_SPOT_COUNT = 1
FRONT_DOUBLING_PER_LAYER = 2.0  # extension E1: the number of independent particles at most doubles per layer (~1 X0)


@dataclass(frozen=True, slots=True)
class LateralParameters:
    """The named constants of the lateral slice. Radial constants are in R_M units with E in GeV."""

    core_intercept: float  # z1a
    core_log_energy_slope: float  # z1b
    core_depth_slope: float  # z2
    tail_scale: float  # k1
    tail_centre: float  # k2
    tail_decay_rate: float  # k3 (negative)
    tail_growth_intercept: float  # k4a
    tail_growth_log_energy_slope: float  # k4b
    weight_amplitude: float  # p1
    weight_centre: float  # p2
    weight_width_intercept: float  # p3a
    weight_width_log_energy_slope: float  # p3b
    spot_amplitude: float  # A
    spot_energy_exponent: float  # b
    spot_depth_scale: float  # kappa_T
    spot_shape_scale: float  # kappa_alpha
    moliere_radius_mm: float = 18.0
    # Extension E1 (opt-in; the defaults reproduce Slice 2 exactly): the tail component's spots carry 1 / kappa of the
    # core spot energy (kappa >= 1), and the spot number of layer l is capped by n0 * 2^l (the number of independent
    # particles of a cascade at most doubles per layer, about one radiation length); n0 = 0 switches the cap off.
    spot_tail_refinement: float = 1.0
    front_spot_limit: float = 0.0
    # Faster-falling tail (opt-in; 1 reproduces Slice 2 exactly): the tail component's survival function is
    # (R_T^2 / (r^2 + R_T^2))^s with s = tail_exponent; s = 1 is the Grindhammer-Peters Cauchy-like tail.
    tail_exponent: float = 1.0
    # Extension E2 (opt-in; 0 reproduces Slice 2 exactly): the tail component's spots carry a fixed absolute energy
    # ``tail_quantum_mev`` (independent of the primary energy and the layer) instead of an equal share of the layer
    # energy; the core component is unchanged. The tail's energy share ``(1 - p) E_l`` is carried by
    # ``(1 - p) E_l / tail_quantum_mev`` spots (at least one), each sharing that energy equally.
    tail_quantum_mev: float = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.tail_quantum_mev < 0.0:
            raise ValueError("tail_quantum_mev must not be negative")
        if self.tail_quantum_mev > 0.0 and self.spot_tail_refinement > 1.0:
            raise ValueError("tail_quantum_mev and spot_tail_refinement (extension E1) are alternatives, not both")
        if self.tail_exponent < 1.0:
            raise ValueError("tail_exponent must be at least 1 (a tail not heavier than the Cauchy-like one)")
        if self.spot_tail_refinement < 1.0:
            raise ValueError("spot_tail_refinement must be at least 1 (tail spots are not coarser than core spots)")
        if self.front_spot_limit < 0.0:
            raise ValueError("front_spot_limit must not be negative")
        if self.tail_scale <= 0 or self.moliere_radius_mm <= 0:
            raise ValueError("tail_scale and moliere_radius_mm must be positive")
        if self.weight_amplitude <= 0:
            raise ValueError("weight_amplitude must be positive")
        if self.spot_amplitude <= 0 or self.spot_depth_scale <= 0 or self.spot_shape_scale <= 0:
            raise ValueError("spot_amplitude, spot_depth_scale and spot_shape_scale must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LateralParameters:
        return cls(**data)

    def vector(self, names: Sequence[str]) -> np.ndarray:
        return np.array([getattr(self, name) for name in names], dtype=float)

    def with_values(self, names: Sequence[str], values: Sequence[float]) -> LateralParameters:
        updated = self.as_dict()
        updated.update({name: float(value) for name, value in zip(names, values, strict=True)})
        return LateralParameters(**updated)


RADIAL_NAMES = (
    "core_intercept",
    "core_log_energy_slope",
    "core_depth_slope",
    "tail_scale",
    "tail_centre",
    "tail_decay_rate",
    "tail_growth_intercept",
    "tail_growth_log_energy_slope",
    "weight_amplitude",
    "weight_centre",
    "weight_width_intercept",
    "weight_width_log_energy_slope",
)
SPOT_NAMES = (
    "spot_amplitude",
    "spot_energy_exponent",
    "spot_depth_scale",
    "spot_shape_scale",
)


def lateral_parameters_digest(parameters: LateralParameters) -> str:
    canonical = json.dumps(parameters.as_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def strip_cdf(x: np.ndarray, radius: np.ndarray, exponent: float = 1.0) -> np.ndarray:
    """CDF of the strip marginal of the radial law with survival function ``(R^2 / (r^2 + R^2))^s``.

    ``s = 1`` is the Grindhammer-Peters density ``2 r R^2 / (r^2 + R^2)^2`` and gives the closed form
    ``(1 + x / sqrt(x^2 + R^2)) / 2``; ``s > 1`` falls faster. The marginal density is
    ``Gamma(s + 1/2) / (sqrt(pi) Gamma(s) R) (1 + x^2 / R^2)^(-(s + 1/2))`` (a scaled Student-t), whose CDF is
    ``1/2 + sign(x) I_z(1/2, s) / 2`` with ``z = x^2 / (x^2 + R^2)``, which reduces to the closed form at ``s = 1``.
    """

    if exponent == 1.0:
        return 0.5 * (1.0 + x / np.sqrt(x * x + radius * radius))
    z = x * x / (x * x + radius * radius)
    return 0.5 + 0.5 * np.sign(x) * betainc(0.5, exponent, z)


def conditional_binomial_counts(rng: np.random.Generator, totals: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Multinomial counts over the last axis of ``probabilities`` (sums to one) by sequential binomials.

    ``totals`` broadcasts against the leading axes. The sequential form is exact and vectorised over events and layers.
    """

    categories = probabilities.shape[-1]
    remaining = np.broadcast_to(totals, probabilities.shape[:-1]).astype(np.int64).copy()
    left = np.ones(probabilities.shape[:-1])
    counts = np.empty(probabilities.shape, dtype=np.int64)
    for c in range(categories - 1):
        share = np.clip(probabilities[..., c] / np.maximum(left, 1e-300), 0.0, 1.0)
        counts[..., c] = rng.binomial(remaining, share)
        remaining -= counts[..., c]
        left = np.maximum(left - probabilities[..., c], 0.0)
    counts[..., categories - 1] = remaining
    return counts


@dataclass(frozen=True, slots=True)
class PreparedBatch:
    """The deterministic part of a batch: the draws of Slice 1 and the strip probabilities, before the spots."""

    energy_mev: float
    layer_fractions: np.ndarray  # (n, layers)
    ln_t: np.ndarray
    ln_alpha: np.ndarray
    tau: np.ndarray  # (n, layers)
    probabilities: np.ndarray  # (n, layers, cells + 1); the last category is lateral leakage
    core_probabilities: np.ndarray | None = None  # only with the two-population spots of extension E1
    tail_probabilities: np.ndarray | None = None
    core_weight: np.ndarray | None = None  # (n, layers)


@dataclass(frozen=True, slots=True)
class LateralBatch:
    cell_energy_mev: np.ndarray  # (n, layers, cells)
    layer_fractions: np.ndarray
    ln_t: np.ndarray
    ln_alpha: np.ndarray
    tau: np.ndarray
    spot_counts: np.ndarray  # (layers,)


@dataclass(frozen=True, slots=True, eq=False)
class EMProductionLateralModel:
    """Slice 1's longitudinal draws with the Grindhammer-Peters radial law and spot fluctuations."""

    longitudinal: EMProductionLongitudinalModel
    parameters: LateralParameters
    geometry: ECALGeometry
    longitudinal_sha256: str = ""
    simulation_version: str = "em-production-slice2-dev"

    def __post_init__(self) -> None:
        if abs(self.parameters.moliere_radius_mm - self.geometry.nominal_moliere_radius_mm) > 1e-9:
            raise ValueError(
                f"moliere_radius_mm {self.parameters.moliere_radius_mm} differs from the geometry's "
                f"{self.geometry.nominal_moliere_radius_mm}"
            )

    # ---- geometry helpers

    @property
    def cell_edges_mm(self) -> np.ndarray:
        pitch, cells = self.geometry.cell_pitch_mm, self.geometry.cells_per_layer
        return -cells * pitch / 2.0 + np.arange(cells + 1) * pitch

    @property
    def measures_x(self) -> np.ndarray:
        return np.array(
            [measured_axis_for_fiber(axis) == "x" for axis in self.geometry.layer_fiber_axes],
            dtype=bool,
        )

    def axis_for_entry(self, entry_x_mm: np.ndarray, entry_y_mm: np.ndarray) -> np.ndarray:
        """``(n, layers)`` measured-coordinate axis of normally incident showers at the given entry points."""

        x, y = np.asarray(entry_x_mm, dtype=float), np.asarray(entry_y_mm, dtype=float)
        return np.where(self.measures_x[None, :], x[:, None], y[:, None])

    def axis_for_track(self, track: TrackState) -> np.ndarray:
        """The measured-coordinate axis of one track in every layer (straight-line projection)."""

        axis = np.empty(self.geometry.number_of_layers)
        for layer, z_mm in enumerate(self.geometry.uniform_layer_centers_z_mm):
            x, y, _ = project_track_to_z(track, target_z_mm=z_mm)
            axis[layer] = x if self.measures_x[layer] else y
        return axis

    def layer_depth_x0(self) -> np.ndarray:
        """Layer-centre depth from the calibrated origin, in X0 (the longitudinal model's own coordinate)."""

        bounds = np.asarray(self.longitudinal.layer_bounds_x0, dtype=float)
        return bounds.mean(axis=1) - self.longitudinal.parameters.origin_x0

    # ---- the laws

    def depth_coordinate(self, energy_mev: float, ln_t: np.ndarray, ln_alpha: np.ndarray) -> np.ndarray:
        """``tau_i`` per event and layer (Grindhammer and Peters Eq. 34), shape ``(n, layers)``."""

        beta = self.longitudinal.parameters.beta
        t_ref = self.longitudinal.median_depth_from_origin_x0(energy_mev)
        alpha_ref = 1.0 + beta * t_ref
        mean_ref = t_ref * alpha_ref / (alpha_ref - 1.0)
        alpha = np.maximum(np.exp(np.atleast_1d(ln_alpha)), MIN_SHAPE)
        mean_i = np.exp(np.atleast_1d(ln_t)) * alpha / (alpha - 1.0)
        return (self.layer_depth_x0()[None, :] / t_ref) * (mean_ref / mean_i)[:, None]

    def radial_parameters(self, energy_mev: float, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(R_C, R_T, p)`` in mm, mm and weight, for an array of tau."""

        q = self.parameters
        ln_e = log(energy_mev / 1000.0)
        p3 = q.weight_width_intercept + q.weight_width_log_energy_slope * ln_e
        if p3 <= 0:
            raise ValueError("the weight width p3 is not positive at this energy")
        tau = np.asarray(tau, dtype=float)
        core = (q.core_intercept + q.core_log_energy_slope * ln_e + q.core_depth_slope * tau) * q.moliere_radius_mm
        k4 = q.tail_growth_intercept + q.tail_growth_log_energy_slope * ln_e
        tail = (
            q.tail_scale
            * (np.exp(q.tail_decay_rate * (tau - q.tail_centre)) + np.exp(k4 * (tau - q.tail_centre)))
            * q.moliere_radius_mm
        )
        argument = (q.weight_centre - tau) / p3
        weight = np.clip(q.weight_amplitude * np.exp(argument - np.exp(argument)), 0.0, 1.0)
        return np.maximum(core, MIN_RADIUS_MM), np.maximum(tail, MIN_RADIUS_MM), weight

    def strip_probabilities(self, energy_mev: float, axis_mm: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """``(n, layers, cells + 1)``: the cell fractions and, last, the fraction outside the 72 cells."""

        core, tail, weight = self.radial_parameters(energy_mev, tau)
        offsets = self.cell_edges_mm[None, None, :] - np.asarray(axis_mm, dtype=float)[:, :, None]
        cdf = weight[..., None] * strip_cdf(offsets, core[..., None]) + (1.0 - weight[..., None]) * strip_cdf(
            offsets, tail[..., None], self.parameters.tail_exponent
        )
        cells = np.diff(cdf, axis=-1)
        outside = np.clip(1.0 - cells.sum(axis=-1, keepdims=True), 0.0, 1.0)
        return np.concatenate([cells, outside], axis=-1)

    def component_probabilities(
        self, energy_mev: float, axis_mm: np.ndarray, tau: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(core, tail, p)``: the cell fractions (plus leakage) of each radial component and the core weight."""

        core, tail, weight = self.radial_parameters(energy_mev, tau)
        offsets = self.cell_edges_mm[None, None, :] - np.asarray(axis_mm, dtype=float)[:, :, None]
        out = []
        for radius, exponent in ((core, 1.0), (tail, self.parameters.tail_exponent)):
            cells = np.diff(strip_cdf(offsets, radius[..., None], exponent), axis=-1)
            outside = np.clip(1.0 - cells.sum(axis=-1, keepdims=True), 0.0, 1.0)
            out.append(np.concatenate([cells, outside], axis=-1))
        return out[0], out[1], weight

    def spot_counts(self, energy_mev: float) -> np.ndarray:
        """Spots per layer for a primary energy (deterministic, at least one per layer)."""

        q = self.parameters
        total = q.spot_amplitude * (energy_mev / 1000.0) ** q.spot_energy_exponent
        t_ref = self.longitudinal.median_depth_from_origin_x0(energy_mev)
        alpha_ref = 1.0 + self.longitudinal.parameters.beta * t_ref
        shape = q.spot_shape_scale * alpha_ref
        if shape <= 1.0:
            raise ValueError("the spot depth density needs a shape above one")
        rate = (shape - 1.0) / (q.spot_depth_scale * t_ref)
        bounds = np.asarray(self.longitudinal.layer_bounds_x0, dtype=float) - self.longitudinal.parameters.origin_x0
        mass = gammainc(shape, rate * np.maximum(bounds[:, 1], 0.0)) - gammainc(
            shape, rate * np.maximum(bounds[:, 0], 0.0)
        )
        counts = total * mass
        if q.front_spot_limit > 0.0:
            cap = q.front_spot_limit * FRONT_DOUBLING_PER_LAYER ** np.arange(len(counts))
            counts = 1.0 / (1.0 / np.maximum(counts, 1e-12) + 1.0 / cap)
        return np.maximum(np.rint(counts).astype(np.int64), MIN_SPOT_COUNT)

    # ---- the draw

    def prepare_batch(
        self,
        energy_mev: float,
        axis_mm: np.ndarray,
        rng: np.random.Generator,
        *,
        allow_extrapolation: bool = False,
    ) -> PreparedBatch:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        axis = np.asarray(axis_mm, dtype=float)
        if axis.ndim != 2 or axis.shape[1] != self.geometry.number_of_layers:
            raise ValueError("axis_mm must have shape (events, layers)")
        z = rng.standard_normal((len(axis), 2))
        ln_t, ln_alpha = self.longitudinal.draw_shape(
            energy_mev, z[:, 0], z[:, 1], allow_extrapolation=allow_extrapolation
        )
        fractions = self.longitudinal.layer_fractions(energy_mev, ln_t, ln_alpha)
        tau = self.depth_coordinate(energy_mev, ln_t, ln_alpha)
        two_populations = self.parameters.spot_tail_refinement > 1.0 or self.parameters.tail_quantum_mev > 0.0
        core_p, tail_p, weight = self.component_probabilities(energy_mev, axis, tau) if two_populations else (None,) * 3
        return PreparedBatch(
            energy_mev=float(energy_mev),
            layer_fractions=fractions,
            ln_t=ln_t,
            ln_alpha=ln_alpha,
            tau=tau,
            probabilities=self.strip_probabilities(energy_mev, axis, tau),
            core_probabilities=core_p,
            tail_probabilities=tail_p,
            core_weight=weight,
        )

    def _two_population_cells(
        self, prepared: PreparedBatch, counts_per_layer: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Cell energies with a core and a tail population: the tail's spots carry 1 / kappa of a core spot's energy.

        The core population holds the energy share ``p`` in ``N_l p`` spots; the tail holds ``1 - p`` in
        ``kappa N_l (1 - p)`` spots. Each population's energy is shared equally by its own spots, so the layer energy is
        conserved apart from the spots beyond the 72 cells.
        """

        weight = prepared.core_weight
        kappa = self.parameters.spot_tail_refinement
        quantum = self.parameters.tail_quantum_mev
        layer_energy = prepared.energy_mev * prepared.layer_fractions
        n_core = np.where(weight > 1e-12, np.maximum(np.rint(counts_per_layer[None, :] * weight), 1), 0).astype(
            np.int64
        )
        if quantum > 0.0:  # extension E2: the tail's spot number follows from its energy and the fixed quantum
            tail_spots = layer_energy * (1.0 - weight) / quantum
        else:
            tail_spots = kappa * counts_per_layer[None, :] * (1.0 - weight)
        n_tail = np.where(1.0 - weight > 1e-12, np.maximum(np.rint(tail_spots), 1), 0).astype(np.int64)
        core_counts = conditional_binomial_counts(rng, n_core, prepared.core_probabilities)[..., :-1]
        tail_counts = conditional_binomial_counts(rng, n_tail, prepared.tail_probabilities)[..., :-1]
        core_energy = np.where(n_core > 0, layer_energy * weight / np.maximum(n_core, 1), 0.0)
        tail_energy = np.where(n_tail > 0, layer_energy * (1.0 - weight) / np.maximum(n_tail, 1), 0.0)
        return core_energy[..., None] * core_counts + tail_energy[..., None] * tail_counts

    def draw_cells(self, prepared: PreparedBatch, rng: np.random.Generator) -> LateralBatch:
        counts_per_layer = self.spot_counts(prepared.energy_mev)
        if prepared.core_probabilities is not None:
            cell_energy = self._two_population_cells(prepared, counts_per_layer, rng)
            return LateralBatch(
                cell_energy_mev=cell_energy,
                layer_fractions=prepared.layer_fractions,
                ln_t=prepared.ln_t,
                ln_alpha=prepared.ln_alpha,
                tau=prepared.tau,
                spot_counts=counts_per_layer,
            )
        counts = conditional_binomial_counts(rng, counts_per_layer[None, :], prepared.probabilities)
        layer_energy = prepared.energy_mev * prepared.layer_fractions
        cell_energy = layer_energy[:, :, None] / counts_per_layer[None, :, None] * counts[..., :-1]
        return LateralBatch(
            cell_energy_mev=cell_energy,
            layer_fractions=prepared.layer_fractions,
            ln_t=prepared.ln_t,
            ln_alpha=prepared.ln_alpha,
            tau=prepared.tau,
            spot_counts=counts_per_layer,
        )

    def sample_batch(
        self,
        energy_mev: float,
        axis_mm: np.ndarray,
        rng: np.random.Generator,
        *,
        allow_extrapolation: bool = False,
    ) -> LateralBatch:
        """Events at the given axes. One event is a batch of one: the draws are consumed in the same order."""

        prepared = self.prepare_batch(energy_mev, axis_mm, rng, allow_extrapolation=allow_extrapolation)
        return self.draw_cells(prepared, rng)

    def mean_cell_fractions(
        self,
        energy_mev: float,
        z: np.ndarray,
        axis_mm: np.ndarray,
        *,
        allow_extrapolation: bool = False,
    ) -> np.ndarray:
        """Ensemble-mean ``(layers, cells)`` fractions of the layer energy over quadrature draws (no spot noise).

        The mean is ENERGY-WEIGHTED, ``sum_i f_il p_ilc / sum_i f_il`` (a ratio of sums), the same quantity as the
        Geant4 profile ``sum_events E_cell / sum_events E_layer``. The weighting matters: an event's layer energy and its
        radial shape are coupled through tau_i, so an unweighted mean of per-event fractions would be biased.
        """

        z = np.asarray(z, dtype=float)
        self.longitudinal.check_energy(energy_mev, allow_extrapolation)
        ln_t, ln_alpha = self.longitudinal.draw_shape(energy_mev, z[:, 0], z[:, 1], allow_extrapolation=True)
        tau = self.depth_coordinate(energy_mev, ln_t, ln_alpha)
        weights = self.longitudinal.layer_fractions(energy_mev, ln_t, ln_alpha)
        probabilities = self.strip_probabilities(energy_mev, axis_mm, tau)[..., :-1]
        return (weights[..., None] * probabilities).sum(axis=0) / weights.sum(axis=0)[:, None]

    def mean_second_moments_mm2(
        self,
        energy_mev: float,
        z: np.ndarray,
        axis_mm: np.ndarray,
        *,
        allow_extrapolation: bool = False,
    ) -> np.ndarray:
        """Energy-weighted mean second moment of the measured coordinate about the track, per layer, in mm^2.

        ``sum_i f_il sum_c p_ilc d_ilc^2 / sum_i f_il`` with ``d`` the offset of the cell centre from the axis: the
        layer's mean contribution to the event width squared, the far-tail statistic the width is built from, with
        the same quadrature ensemble and the same energy weights as :meth:`mean_cell_fractions`.
        """

        z = np.asarray(z, dtype=float)
        self.longitudinal.check_energy(energy_mev, allow_extrapolation)
        ln_t, ln_alpha = self.longitudinal.draw_shape(energy_mev, z[:, 0], z[:, 1], allow_extrapolation=True)
        tau = self.depth_coordinate(energy_mev, ln_t, ln_alpha)
        weights = self.longitudinal.layer_fractions(energy_mev, ln_t, ln_alpha)
        probabilities = self.strip_probabilities(energy_mev, axis_mm, tau)[..., :-1]
        edges = self.cell_edges_mm
        offsets = 0.5 * (edges[:-1] + edges[1:])[None, None, :] - np.asarray(axis_mm, dtype=float)[:, :, None]
        moment = (probabilities * offsets**2).sum(axis=2)
        return (weights * moment).sum(axis=0) / weights.sum(axis=0)

    # ---- events

    def sample_cell_energies_mev(
        self, primary_energy_mev: float, track: TrackState, rng: np.random.Generator
    ) -> EnergyGrid:
        axis = self.axis_for_track(track)[None, :]
        batch = self.sample_batch(primary_energy_mev, axis, rng)
        return tuple(tuple(float(v) for v in layer) for layer in batch.cell_energy_mev[0])

    def generate_event(
        self,
        *,
        event_id: str,
        primary_energy_mev: float,
        track: TrackState,
        random_seed: int,
        configuration_sha256: str,
        particle_type: str = "electron",
    ) -> ECALEvent:
        if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")
        rng = np.random.default_rng(random_seed)
        return ECALEvent(
            event_id=event_id,
            particle_type=particle_type,
            primary_energy_mev=float(primary_energy_mev),
            track=track,
            geometry=self.geometry,
            cell_energies_mev=self.sample_cell_energies_mev(primary_energy_mev, track, rng),
            provenance=EventProvenance(
                simulation_backend="fastmc",
                simulation_version=self.simulation_version,
                configuration_sha256=configuration_sha256,
                random_seed=random_seed,
            ),
        )

    def generate_events(
        self,
        *,
        primary_energies_mev: Sequence[float],
        tracks: Sequence[TrackState],
        base_seed: int,
        configuration_sha256: str,
        particle_type: str = "electron",
        event_id_prefix: str = "fastmc-em-production-lateral",
    ) -> Iterator[ECALEvent]:
        energies, states = tuple(primary_energies_mev), tuple(tracks)
        if len(energies) != len(states):
            raise ValueError("primary_energies_mev and tracks must have the same length")
        for index, (energy, track, seed) in enumerate(
            zip(
                energies,
                states,
                spawn_event_seeds(base_seed, len(energies)),
                strict=True,
            )
        ):
            yield self.generate_event(
                event_id=f"{event_id_prefix}-{index:06d}",
                primary_energy_mev=energy,
                track=track,
                random_seed=seed,
                configuration_sha256=configuration_sha256,
                particle_type=particle_type,
            )


# ------------------------------------------------------------------ the parameter artifact


def write_parameters_artifact(
    parameters: LateralParameters,
    path: Path,
    calibration: dict[str, Any],
    module: str,
    longitudinal_sha256: str,
) -> Path:
    document = {
        "kind": "em_production_lateral_parameters",
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "status": "CANDIDATE DEVELOPMENT parameters calibrated on exposed Geant4 electrons; not validated",
        "parameters": parameters.as_dict(),
        "parameters_sha256": lateral_parameters_digest(parameters),
        "longitudinal_parameters_sha256": longitudinal_sha256,
        "builder": {"module": module, **git_provenance()},
        "calibration": calibration,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_parameters_artifact(
    path: Path = DEFAULT_ARTIFACT,
) -> tuple[LateralParameters, dict[str, Any]]:
    """Load and verify an artifact: the schema, the model name and the digest of the parameters."""

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("kind") != "em_production_lateral_parameters":
        raise ValueError("not an em_production lateral parameters artifact")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version {document.get('schema_version')!r}")
    if document.get("model") != MODEL_NAME:
        raise ValueError(f"unexpected model {document.get('model')!r}")
    stored = document["parameters"]
    parameters = LateralParameters.from_dict(stored)
    # the digest is checked on the STORED content, so an artifact written before an optional field existed still verifies
    canonical = json.dumps(stored, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != document.get("parameters_sha256"):
        raise ValueError("the parameters do not match the recorded sha256: the artifact was altered")
    return parameters, document


def build_model(
    parameters: LateralParameters,
    longitudinal_artifact: Path = LONGITUDINAL_ARTIFACT,
    expected_longitudinal_sha256: str | None = None,
) -> EMProductionLateralModel:
    """The generator for the given lateral parameters on the verified Slice 1 longitudinal artifact."""

    longitudinal_parameters, document = read_longitudinal_artifact(longitudinal_artifact)
    digest = document["parameters_sha256"]
    if expected_longitudinal_sha256 is not None and digest != expected_longitudinal_sha256:
        raise ValueError(
            "the longitudinal artifact differs from the one the lateral parameters were calibrated on "
            f"({digest[:12]} against {expected_longitudinal_sha256[:12]})"
        )
    slice1 = build_longitudinal_model(longitudinal_parameters)
    return EMProductionLateralModel(
        longitudinal=slice1.longitudinal,
        parameters=parameters,
        geometry=slice1.geometry,
        longitudinal_sha256=digest,
    )


def load_em_production_lateral(
    path: Path = DEFAULT_ARTIFACT, longitudinal_artifact: Path = LONGITUDINAL_ARTIFACT
) -> EMProductionLateralModel:
    """The calibrated Slice 2 model from its verified artifacts."""

    parameters, document = read_parameters_artifact(path)
    return build_model(parameters, longitudinal_artifact, document["longitudinal_parameters_sha256"])
