"""The ``em_production`` electron generator, Slice 1: a candidate DEVELOPMENT model (not validated).

Decision record ``research/DECISIONS.md`` (DEC-014 and the 2026-10-06 rows), plan
``research/plans/2026-10-05_em_production_generator_plan.md`` sections 8-13, note
``research/plans/2026-10-06_em_production_slice1_note.md``. DEC-001 (``stochastic.py``) is NOT changed: it stays
the historical baseline and the null control.

LOCKED BASELINE PHYSICS FAMILY (researcher's decision, 2026-10-06). The longitudinal profile of one event is a
GAMMA density. No alternative functional family is searched for; the gamma family stays unless later validation
gives strong evidence that it is inadequate.

WHAT ONE EVENT IS (``deposition`` representation, 18 layers, fractions of the primary energy):

    x_max(E)  = ln(E / E_c) + delta                     depth of the maximum from the front face (X0)
    T_o(E)    = x_max(E) - z0                           depth of the maximum from the calibrated origin
    ln T      ~ skew-normal, MEDIAN ln T_o, sd  s_T * x_max / T_o, skewness gamma_T, driven by z_t
    ln alpha  = ln(1 + beta T_o) + s_a (rho z_t + sqrt(1 - rho^2) z_perp)
    core      = gamma density in u = x - z0 with shape alpha and rate (alpha - 1) / T
    tail      = gamma density in u = x - z0 with shape a_2 and rate kappa
    fractions = (1 - w(E)) core + w(E) tail,   w(E) = w_0 + w_1 ln(E / 30 GeV)

``z_t`` and ``z_perp`` are two independent standard normals; ln T is a monotone function of ``z_t`` through the
normal CDF (a Gaussian copula), so ``rho`` is the correlation of the Gaussian copula. The fractions are never
renormalised to the finite depth: the missing part is longitudinal leakage. The lateral grid is DEC-001's
deterministic lateral model, unchanged and not part of this slice.

PROVENANCE of every ingredient:

* externally established in this regime: the gamma family (Grindhammer and Peters 2000 Eq. 2-4; PDG review 34.5);
  ``delta = -0.5`` for electron-induced showers (PDG review Eq. 34.36); the tail DECAY RATE ``kappa = 1/3.6`` per X0
  (Leroy and Rancoita 2000, Table 2, lead 3.3-3.9 X0), confirmed by the extended-depth Geant4 electrons (0.27-0.28);
* transferred approximation: the widths ``s_T = 1/(-2.5 + 1.25 ln y)`` and ``s_a = 1/(-0.82 + 0.79 ln y)`` with the
  covariant factor ``x_max / T_o`` (Grindhammer and Peters, sampling set, App. A.2.3). The correlation ``rho`` is the
  project's own line in ln y because the measured value is lower than theirs (0.57-0.62);
* project phenomenological, calibrated on the exposed Geant4 electrons: the origin ``z0`` (a COORDINATE CONVENTION,
  not a physical shower start), the mean scale ``beta``, the in-prefix tail correction (``w_0``, ``w_1``, ``a_2``:
  an EMPIRICAL GEANT4 CALIBRATION CORRECTION, not a physical tail, see the extended-depth note), ``rho`` and the ln T
  skewness ``gamma_T``;
* AMS-specific evidence: none is used. AMS b = 0.65 is treated as a depth-origin convention (profile shape study).

SCOPE AND REFUSALS. 10-100 GeV is the calibrated range; outside it the model refuses unless ``allow_extrapolation``
is set. The lateral structure, event-level tail weight, layer-to-layer residual noise and the first-layer floor are
NOT modelled (floor rejected by the evidence). The detector response is a separate step.
Words: calibrated, candidate, development; never validated.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from math import isfinite, log, sqrt
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from scipy.special import gammainc, ndtr

from ams_ecal.detector.event import ECALEvent, EnergyGrid, EventProvenance
from ams_ecal.detector.geometry import ECALGeometry, load_geometry
from ams_ecal.detector.tracking import TrackState
from ams_ecal.electron_model.fastmc_config import load_fastmc_config
from ams_ecal.electron_model.lateral import AMSLateralShowerModel
from ams_ecal.electron_model.longitudinal import AMSLongitudinalGammaModel

MODEL_NAME = "em-production-gamma-baseline"
SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[3]
GEOMETRY_CONFIG = PROJECT_ROOT / "configs" / "geometry.yaml"
FASTMC_CONFIG = PROJECT_ROOT / "configs" / "fastmc.yaml"
DEFAULT_ARTIFACT = PROJECT_ROOT / "data" / "calibration" / "em_production" / "gamma_baseline_v1" / "parameters.json"

# Grindhammer and Peters 2000, Appendix A.2.3 (sampling set), as (intercept, slope in ln y) of 1 / sigma
SIGMA_LN_T_COEFFICIENTS = (-2.5, 1.25)
SIGMA_LN_ALPHA_COEFFICIENTS = (-0.82, 0.79)
MAX_ABS_SKEWNESS = 0.99  # the skew-normal family cannot reach a skewness of 1
MAX_TAIL_WEIGHT = 0.95
MIN_SHAPE = 1.0 + 1e-6


def skew_normal_shape_and_scale(sd: float, skewness: float) -> tuple[float, float]:
    """``(shape, scale)`` of the skew-normal with standard deviation ``sd`` and the given skewness."""

    g = float(np.clip(skewness, -MAX_ABS_SKEWNESS, MAX_ABS_SKEWNESS))
    if g == 0.0:
        return 0.0, float(sd)
    ratio = (4.0 - np.pi) / 2.0
    delta = np.sign(g) * np.sqrt(
        (np.pi / 2.0) * abs(g) ** (2.0 / 3.0) / (abs(g) ** (2.0 / 3.0) + ratio ** (2.0 / 3.0))
    )
    delta = float(np.clip(delta, -0.999, 0.999))
    scale = float(sd / np.sqrt(1.0 - 2.0 * delta**2 / np.pi))
    return float(delta / np.sqrt(1.0 - delta**2)), scale


Z_GRID = np.linspace(-6.0, 6.0, 4001)


@lru_cache(maxsize=64)
def standard_skew_normal_table(shape: float) -> tuple[np.ndarray, np.ndarray, float]:
    """``(z grid, quantile of the standard skew-normal at the normal CDF of z, its median)`` for one shape.

    The quantile function of a skew-normal is evaluated by root finding, which is slow inside a fit; it depends
    only on the shape, so it is tabulated once per shape and interpolated (error about 1e-6 for the |z| < 6 used).
    """

    quantiles = stats.skewnorm.ppf(np.clip(ndtr(Z_GRID), 1e-300, 1.0 - 1e-16), shape)
    return Z_GRID, quantiles, float(stats.skewnorm.median(shape))


@dataclass(frozen=True, slots=True)
class EMProductionParameters:
    """The named parameters of Slice 1. The first block is calibrated, the second is fixed by the sources."""

    origin_x0: float
    beta: float
    tail_weight: float
    tail_weight_log_energy_slope: float
    tail_shape: float
    rho_intercept: float
    rho_log_slope: float
    ln_t_skewness: float
    x_max_offset_x0: float = -0.5
    tail_rate_per_x0: float = 1.0 / 3.6
    energy_reference_gev: float = 30.0
    calibrated_energy_range_gev: tuple[float, float] = (10.0, 100.0)

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            values = value if isinstance(value, tuple) else (value,)
            if not all(isfinite(float(v)) for v in values):
                raise ValueError(f"{name} must be finite")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.tail_shape <= 1.0:
            raise ValueError("tail_shape must exceed 1")
        if self.tail_rate_per_x0 <= 0:
            raise ValueError("tail_rate_per_x0 must be positive")
        if not 0.0 <= self.tail_weight <= MAX_TAIL_WEIGHT:
            raise ValueError("tail_weight must lie in [0, 0.95]")
        if abs(self.ln_t_skewness) >= 1.0:
            raise ValueError("ln_t_skewness must lie inside (-1, 1)")
        low, high = self.calibrated_energy_range_gev
        if not 0 < low < high:
            raise ValueError("calibrated_energy_range_gev must be an increasing positive pair")

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["calibrated_energy_range_gev"] = list(self.calibrated_energy_range_gev)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EMProductionParameters:
        data = dict(data)
        data["calibrated_energy_range_gev"] = tuple(data["calibrated_energy_range_gev"])
        return cls(**data)


def parameters_digest(parameters: EMProductionParameters) -> str:
    """SHA-256 of the canonical JSON of the parameters (what the artifact's manifest records)."""

    canonical = json.dumps(parameters.as_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class EMProductionLongitudinalModel:
    """The longitudinal part: sample the event's ``(ln T, ln alpha)`` and integrate the layers."""

    parameters: EMProductionParameters
    critical_energy_mev: float
    layer_bounds_x0: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not isfinite(self.critical_energy_mev) or self.critical_energy_mev <= 0:
            raise ValueError("critical_energy_mev must be positive")
        if len(self.layer_bounds_x0) == 0:
            raise ValueError("layer_bounds_x0 must not be empty")

    # ---- the energy law

    def check_energy(self, primary_energy_mev: float, allow_extrapolation: bool = False) -> float:
        energy = float(primary_energy_mev)
        if not isfinite(energy) or energy <= 0:
            raise ValueError("primary_energy_mev must be positive and finite")
        low, high = self.parameters.calibrated_energy_range_gev
        if not allow_extrapolation and not (low * 1000.0 * 0.999 <= energy <= high * 1000.0 * 1.001):
            raise ValueError(
                f"{energy / 1000.0:g} GeV is outside the calibrated range {low:g}-{high:g} GeV; "
                "pass allow_extrapolation=True to use the model outside it"
            )
        return energy

    def ln_y(self, primary_energy_mev: float) -> float:
        return log(float(primary_energy_mev) / self.critical_energy_mev)

    def shower_max_depth_x0(self, primary_energy_mev: float) -> float:
        """``x_max`` from the front face."""

        return self.ln_y(primary_energy_mev) + self.parameters.x_max_offset_x0

    def median_depth_from_origin_x0(self, primary_energy_mev: float) -> float:
        depth = self.shower_max_depth_x0(primary_energy_mev) - self.parameters.origin_x0
        if depth <= 0:
            raise ValueError("the shower maximum lies in front of the calibrated origin")
        return depth

    def widths(self, primary_energy_mev: float) -> tuple[float, float, float]:
        """``(sigma of ln T with the covariant factor, sigma of ln alpha, rho)``."""

        ln_y = self.ln_y(primary_energy_mev)
        t_denominator = SIGMA_LN_T_COEFFICIENTS[0] + SIGMA_LN_T_COEFFICIENTS[1] * ln_y
        a_denominator = SIGMA_LN_ALPHA_COEFFICIENTS[0] + SIGMA_LN_ALPHA_COEFFICIENTS[1] * ln_y
        if t_denominator <= 0 or a_denominator <= 0:
            raise ValueError("primary_energy_mev is below the domain of the width laws")
        x_max = self.shower_max_depth_x0(primary_energy_mev)
        sigma_t = (1.0 / t_denominator) * x_max / self.median_depth_from_origin_x0(primary_energy_mev)
        rho = self.parameters.rho_intercept + self.parameters.rho_log_slope * ln_y
        if not -0.999 < rho < 0.999:
            raise ValueError("the correlation rho left (-1, 1) at this energy")
        return float(sigma_t), float(1.0 / a_denominator), float(rho)

    def tail_weight(self, primary_energy_mev: float) -> float:
        p = self.parameters
        weight = p.tail_weight + p.tail_weight_log_energy_slope * log(
            float(primary_energy_mev) / 1000.0 / p.energy_reference_gev
        )
        return float(np.clip(weight, 0.0, MAX_TAIL_WEIGHT))

    # ---- the draw

    def draw_shape(
        self, primary_energy_mev: float, z_t: np.ndarray, z_perp: np.ndarray, *, allow_extrapolation: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(ln T, ln alpha)`` for arrays of the two standard normals."""

        energy = self.check_energy(primary_energy_mev, allow_extrapolation)
        z_t, z_perp = np.asarray(z_t, dtype=float), np.asarray(z_perp, dtype=float)
        sigma_t, sigma_a, rho = self.widths(energy)
        depth = self.median_depth_from_origin_x0(energy)
        shape, scale = skew_normal_shape_and_scale(sigma_t, self.parameters.ln_t_skewness)
        if shape == 0.0:
            ln_t = log(depth) + sigma_t * z_t
        else:
            grid, quantiles, median = standard_skew_normal_table(round(shape, 12))
            ln_t = log(depth) + scale * (np.interp(z_t, grid, quantiles) - median)
        alpha_median = 1.0 + self.parameters.beta * depth
        ln_alpha = log(alpha_median) + sigma_a * (rho * z_t + sqrt(1.0 - rho**2) * z_perp)
        return ln_t, ln_alpha

    def layer_fractions(self, primary_energy_mev: float, ln_t: np.ndarray, ln_alpha: np.ndarray) -> np.ndarray:
        """``(n, layers)`` energy fractions of the events with the given shape draws."""

        p = self.parameters
        bounds = np.asarray(self.layer_bounds_x0, dtype=float)
        lower = np.maximum(bounds[:, 0] - p.origin_x0, 0.0)[None, :]
        upper = np.maximum(bounds[:, 1] - p.origin_x0, 0.0)[None, :]
        alpha = np.maximum(np.exp(np.atleast_1d(ln_alpha)), MIN_SHAPE)[:, None]
        rate = (alpha - 1.0) / np.exp(np.atleast_1d(ln_t))[:, None]
        core = gammainc(alpha, rate * upper) - gammainc(alpha, rate * lower)
        weight = self.tail_weight(primary_energy_mev)
        if weight == 0.0:
            return core
        tail = gammainc(p.tail_shape, p.tail_rate_per_x0 * upper) - gammainc(p.tail_shape, p.tail_rate_per_x0 * lower)
        return (1.0 - weight) * core + weight * tail

    def mean_layer_fractions(self, primary_energy_mev: float, z: np.ndarray) -> np.ndarray:
        """Ensemble mean over ``z`` of shape ``(n, 2)`` (``z_t``, ``z_perp``) normals (a quadrature set)."""

        ln_t, ln_alpha = self.draw_shape(primary_energy_mev, z[:, 0], z[:, 1])
        return self.layer_fractions(primary_energy_mev, ln_t, ln_alpha).mean(axis=0)

    def sample_layer_energy_fractions_batch(
        self, primary_energy_mev: float, rng: np.random.Generator, n: int, *, allow_extrapolation: bool = False
    ) -> np.ndarray:
        """``(n, layers)`` fractions; the two normals of each event are consecutive draws of ``rng``."""

        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        z = rng.standard_normal((n, 2))
        ln_t, ln_alpha = self.draw_shape(primary_energy_mev, z[:, 0], z[:, 1], allow_extrapolation=allow_extrapolation)
        return self.layer_fractions(primary_energy_mev, ln_t, ln_alpha)

    def sample_layer_energy_fractions(
        self, primary_energy_mev: float, rng: np.random.Generator, *, allow_extrapolation: bool = False
    ) -> tuple[float, ...]:
        """One event's layer fractions (never renormalised: the deficit from one is longitudinal leakage)."""

        row = self.sample_layer_energy_fractions_batch(
            primary_energy_mev, rng, 1, allow_extrapolation=allow_extrapolation
        )[0]
        return tuple(float(v) for v in row)


def spawn_event_seeds(base_seed: int, count: int) -> tuple[int, ...]:
    """``count`` independent per-event seeds from one base seed (SeedSequence), as DEC-001 does."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    children = np.random.SeedSequence(base_seed).spawn(count)
    return tuple(int(child.generate_state(1, dtype=np.uint32)[0]) for child in children)


@dataclass(frozen=True, slots=True, eq=False)
class EMProductionShowerModel:
    """Event generator: the Slice 1 longitudinal model with DEC-001's deterministic lateral grid."""

    longitudinal: EMProductionLongitudinalModel
    lateral: AMSLateralShowerModel
    simulation_version: str = "em-production-slice1-dev"

    @property
    def geometry(self) -> ECALGeometry:
        return self.lateral.geometry

    @property
    def parameters(self) -> EMProductionParameters:
        return self.longitudinal.parameters

    def sample_layer_energies_mev(self, primary_energy_mev: float, rng: np.random.Generator) -> tuple[float, ...]:
        energy = float(primary_energy_mev)
        return tuple(energy * f for f in self.longitudinal.sample_layer_energy_fractions(energy, rng))

    def sample_cell_energies_mev(
        self, primary_energy_mev: float, track: TrackState, rng: np.random.Generator
    ) -> EnergyGrid:
        layer_energies = self.sample_layer_energies_mev(primary_energy_mev, rng)
        lateral_fractions = self.lateral.track_centered_cell_fractions(track, primary_energy_mev)
        return tuple(
            tuple(layer_energy * cell for cell in layer_fractions)
            for layer_energy, layer_fractions in zip(layer_energies, lateral_fractions, strict=True)
        )

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
        event_id_prefix: str = "fastmc-em-production",
    ) -> Iterator[ECALEvent]:
        energies, states = tuple(primary_energies_mev), tuple(tracks)
        if len(energies) != len(states):
            raise ValueError("primary_energies_mev and tracks must have the same length")
        for index, (energy, track, seed) in enumerate(
            zip(energies, states, spawn_event_seeds(base_seed, len(energies)), strict=True)
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


def git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {
            "git_commit": run("rev-parse", "HEAD"),
            "tracked_changes": bool(run("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "tracked_changes": None}


def write_parameters_artifact(
    parameters: EMProductionParameters, path: Path, calibration: dict[str, Any], module: str
) -> Path:
    document = {
        "kind": "em_production_parameters",
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "status": "CANDIDATE DEVELOPMENT parameters calibrated on exposed Geant4 electrons; not validated",
        "parameters": parameters.as_dict(),
        "parameters_sha256": parameters_digest(parameters),
        "builder": {"module": module, **git_provenance()},
        "calibration": calibration,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_parameters_artifact(path: Path = DEFAULT_ARTIFACT) -> tuple[EMProductionParameters, dict[str, Any]]:
    """Load and verify an artifact: the schema, the model name and the digest of the parameters."""

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("kind") != "em_production_parameters":
        raise ValueError("not an em_production parameters artifact")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version {document.get('schema_version')!r}")
    if document.get("model") != MODEL_NAME:
        raise ValueError(f"unexpected model {document.get('model')!r}")
    parameters = EMProductionParameters.from_dict(document["parameters"])
    if parameters_digest(parameters) != document.get("parameters_sha256"):
        raise ValueError("the parameters do not match the recorded sha256: the artifact was altered")
    return parameters, document


def build_model(parameters: EMProductionParameters) -> EMProductionShowerModel:
    """The generator for the given parameters, on the project geometry and DEC-001's lateral model."""

    geometry = load_geometry(GEOMETRY_CONFIG)
    config = load_fastmc_config(FASTMC_CONFIG)
    lateral = AMSLateralShowerModel(config=config.lateral_em, geometry=geometry)
    critical = AMSLongitudinalGammaModel(
        config=config.longitudinal_em, geometry=geometry, regime="deposition"
    ).critical_energy_mev
    longitudinal = EMProductionLongitudinalModel(
        parameters=parameters,
        critical_energy_mev=critical,
        layer_bounds_x0=tuple((float(a), float(b)) for a, b in geometry.uniform_layer_bounds_x0),
    )
    return EMProductionShowerModel(longitudinal=longitudinal, lateral=lateral)


def load_em_production(path: Path = DEFAULT_ARTIFACT) -> EMProductionShowerModel:
    """The calibrated Slice 1 model from its verified parameter artifact."""

    parameters, _ = read_parameters_artifact(path)
    return build_model(parameters)
