"""Joint mean-profile calibration of a common depth origin and mean beta (analysis only).

Decision record ``research/DECISIONS.md`` (DEC-014; the 2026-10-06 rows), note
``research/plans/2026-10-06_em_depth_origin_calibration_note.md``. Uses only the EXPOSED Geant4 electrons
(development / calibration data); no sealed data, no generator is changed or written.

COORDINATES. ``x`` is the depth in X0 from the ECAL front face. The profile is a gamma density in
``u = x - z0``: shape ``alpha``, rate ``beta``, mode ``T_o = (alpha - 1) / beta`` measured from the
origin, so the maximum sits at ``x_max = z0 + T_o``. A NEGATIVE ``z0`` puts the origin upstream of
the front face. ``z0`` is a calibrated coordinate-alignment parameter, not a physical shower start.

MEAN MODEL. The ensemble-mean profile of the stochastic model: ``ln T_o`` and ``ln alpha`` are
correlated normals around ``(T_o, alpha = 1 + beta T_o)`` with the Grindhammer-Peters SAMPLING
spreads and correlation (``rho``), averaged by Gauss-Hermite quadrature. The gamma profile is
nonlinear, so the ensemble mean differs from the profile at the mean parameters; a deterministic
variant (no fluctuation) is fitted too. The depth of the maximum follows DEC-001's law
``x_max(E) = ln(E / E_c) + delta`` (``delta = -0.5``, the deposition regime), so the first-stage
model has exactly two free parameters, ``z0`` and ``beta``. Because a shifted origin changes ``T``,
the GP width of ``ln T`` is applied COVARIANTLY, ``sigma_o = sigma_GP * x_max / T_o`` (the absolute
spread of the maximum position is coordinate independent); the naive version is fitted too.

OBJECTIVE. Chi-square of the four energies' mean layer fractions with the standard error of each
mean (event standard deviation over root n). Because a smooth model is imperfect the errors are
rescaled so that chi-square per degree of freedom is one at the optimum before contours are drawn;
parameter uncertainties are also taken from an event bootstrap.

THE FLOOR IS GATED (``GATE``): a first-layer floor is fitted only if the entrance residual of the
two-parameter model is stable, structured and not absorbed by the alternatives, and only if it then
improves held-out calibration scores.

    uv run python -m ams_ecal.electron_studies.em_depth_origin_calibration
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from scipy.optimize import least_squares

from ams_ecal.detector.geometry import load_geometry
from ams_ecal.electron_studies.em_longitudinal_fluctuations import (
    ENERGIES_GEV,
    LOWER,
    START,
    UPPER,
    layer_bounds_x0,
)
from ams_ecal.electron_studies.em_longitudinal_structure_analysis import (
    BETA_AMS,
    EnergyData,
    _clean,
    _model,
    beta_values,
    dec001_ensemble,
    distances,
    fit_free_with_origin,
    gp_formulae,
    joint_diagnostics,
    load_energy,
    longitudinal_metrics,
    marginal_diagnostics,
    profile_fractions_batch,
    removed_fraction,
)
from ams_ecal.geant4_simulation.geant4_backend import PROJECT_ROOT
from ams_ecal.geant4_simulation.pilot_analysis import layer_centres_mm
from ams_ecal.validation.dataset import GEOMETRY_CONFIG

RESULTS_DIR = PROJECT_ROOT / "results" / "em_generator"
CALIBRATION_JSON = RESULTS_DIR / "depth_origin_calibration.json"
CALIBRATION_PLOTS = RESULTS_DIR / "depth_origin_calibration"

DELTA_DEC001 = -0.5  # x_max = ln y + delta in the DEC-001 deposition regime
TAIL_ONSET_OFF = 1.0e9  # the tail is a gamma density unless an onset is given
ENERGY_REFERENCE_GEV = 30.0  # energy-dependent parameters are linear in ln(E / this)
GAUSS_HERMITE_NODES = 12
Z0_GRID = np.linspace(-6.0, 1.0, 71)
BETA_GRID = np.linspace(0.30, 1.10, 81)
CONTOUR_LEVELS = {"68%": 2.30, "95%": 5.99}  # two parameters
BOOTSTRAP_RESAMPLES = 200
BOOTSTRAP_RESAMPLES_SECONDARY = 100
ENSEMBLE_EVENTS = 4000
ESTIMATOR_CONTROL_EVENTS = 1500
ORIGINS_FOR_SKEWNESS = (0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0)
COHERENCE_BAND = (0.75, 1.25)  # measured / expected width counts as coherent inside this band
COHERENCE_RHO_TOLERANCE = 0.25

GATE = {
    "min_relative_layer0_residual": 0.10,  # |model - Geant4| / Geant4 in layer 0, at every energy
    "min_significance": 3.0,  # in standard errors of the Geant4 layer-0 mean, at every energy
    "max_magnitude_spread": 3.0,  # largest / smallest relative residual across energies
    "absorbed_fraction": 0.5,  # an alternative that removes more than this is "absorbing" the residual
    "max_contained_fraction_residual": 0.01,  # leakage accounting: the contained fraction is within 1%
    "cv_improvement_to_accept": 0.20,  # held-out score must improve by this fraction to accept a floor
    "max_layers_4_17_degradation": 0.05,
}

DEFAULTS = {
    "z0": 0.0,
    "beta": BETA_AMS,
    "delta": DELTA_DEC001,
    "floor_a": 0.0,
    "floor_b": 0.0,
    "z0_slope": 0.0,
    "beta_slope": 0.0,
    "depth_slope": 1.0,
    "tail_w": 0.0,
    "tail_w_slope": 0.0,
    "tail_alpha": 4.0,
    "tail_kappa": 0.278,
    "tail_onset": TAIL_ONSET_OFF,
}
PARAMETER_BOUNDS = {
    "z0": (-8.0, 2.0),
    "beta": (0.2, 1.6),
    "delta": (-4.0, 3.0),
    "floor_a": (0.0, 500.0),
    "floor_b": (0.0, 5.0),
    "z0_slope": (-3.0, 3.0),
    "beta_slope": (-0.5, 0.5),
    "depth_slope": (0.5, 1.5),
    "tail_w": (0.0, 0.5),
    "tail_w_slope": (-0.2, 0.2),
    "tail_alpha": (1.05, 12.0),
    "tail_kappa": (0.15, 0.6),
    "tail_onset": (-6.0, 20.0),
}


# ----------------------------------------------------------------------
# Targets
# ----------------------------------------------------------------------


@dataclass
class Targets:
    """Mean layer fractions of the Geant4 events at each energy, with their standard errors."""

    energies_gev: np.ndarray
    mean: np.ndarray  # (n_energies, 18)
    se: np.ndarray
    events: list[np.ndarray]
    critical_energy_mev: float
    bounds: np.ndarray

    @property
    def energies_mev(self) -> np.ndarray:
        return 1000.0 * self.energies_gev

    def subset(self, indices: Sequence[int]) -> Targets:
        idx = list(indices)
        return Targets(
            self.energies_gev[idx],
            self.mean[idx],
            self.se[idx],
            [self.events[i] for i in idx],
            self.critical_energy_mev,
            self.bounds,
        )

    def with_events(self, events: list[np.ndarray]) -> Targets:
        """The same energies with new events; the standard errors are kept from the full sample."""

        mean = np.array([e.mean(axis=0) for e in events])
        return Targets(self.energies_gev, mean, self.se, events, self.critical_energy_mev, self.bounds)

    def resample(self, rng: np.random.Generator) -> Targets:
        return self.with_events([e[rng.integers(0, len(e), len(e))] for e in self.events])

    def halves(self) -> tuple[Targets, Targets]:
        """Even-indexed and odd-indexed events (split-half calibration resampling)."""

        return (
            self.with_events([e[0::2] for e in self.events]),
            self.with_events([e[1::2] for e in self.events]),
        )


def make_targets(loaded: dict[float, EnergyData], representation: str, bounds: np.ndarray) -> Targets:
    events = [loaded[e].fractions[representation] for e in loaded]
    mean = np.array([f.mean(axis=0) for f in events])
    se = np.array([f.std(axis=0, ddof=1) / np.sqrt(len(f)) for f in events])
    return Targets(
        np.array(list(loaded), dtype=float),
        mean,
        se,
        events,
        _model("deposition").longitudinal.critical_energy_mev,
        bounds,
    )


# ----------------------------------------------------------------------
# The mean model
# ----------------------------------------------------------------------


def x_max_of_energy(energy_mev: float, critical_energy_mev: float, delta: float, slope: float = 1.0) -> float:
    """Depth of the profile maximum from the front face, ``slope * ln(E / E_c) + delta`` (X0).

    DEC-001 is ``slope = 1``; the slope is a free parameter only in the shape study.
    """

    return float(slope * np.log(energy_mev / critical_energy_mev) + delta)


def ensemble_mean_fractions(
    z0: float,
    beta: float,
    x_max: np.ndarray,
    energies_mev: np.ndarray,
    critical_energy_mev: float,
    bounds: np.ndarray,
    *,
    fluctuate: bool = True,
    covariant: bool = True,
) -> np.ndarray:
    """``(n_energies, layers)`` ensemble-mean layer fractions of the stochastic profile model."""

    nodes, weights = np.polynomial.hermite_e.hermegauss(GAUSS_HERMITE_NODES)
    weights = weights / weights.sum()
    grid_1, grid_2 = np.meshgrid(nodes, nodes, indexing="ij")
    grid_weight = np.outer(weights, weights).ravel()
    rows = []
    for xm, energy in zip(x_max, energies_mev, strict=True):
        depth_from_origin = max(float(xm) - z0, 0.2)
        alpha_median = 1.0 + beta * depth_from_origin
        if not fluctuate:
            rows.append(
                profile_fractions_batch(
                    np.array([np.log(depth_from_origin)]), np.array([np.log(alpha_median)]), bounds, z0
                )[0]
            )
            continue
        gp = gp_formulae(float(np.log(energy / critical_energy_mev)))["sampling"]
        sigma_t = gp["sigma_ln_t"] * (float(xm) / depth_from_origin if covariant else 1.0)
        sigma_a, rho = gp["sigma_ln_alpha"], gp["rho"]
        ln_t = np.log(depth_from_origin) + sigma_t * grid_1.ravel()
        ln_a = np.log(alpha_median) + sigma_a * (
            rho * grid_1.ravel() + np.sqrt(1.0 - rho**2) * grid_2.ravel()
        )
        rows.append(grid_weight @ profile_fractions_batch(ln_t, ln_a, bounds, z0))
    return np.array(rows)


@dataclass(frozen=True)
class ModelSpec:
    """Which parameters are free, what is fixed, and how the ensemble mean is formed."""

    name: str
    free: tuple[str, ...]
    fixed: dict[str, float] = field(default_factory=dict)
    fluctuate: bool = True
    covariant: bool = True
    amplitude: str = "none"  # "free": a scale per energy (the readout profile)


def _parameters(spec: ModelSpec, theta: np.ndarray) -> dict[str, float]:
    params = dict(DEFAULTS)
    params.update(spec.fixed)
    params.update(dict(zip(spec.free, theta, strict=True)))
    return params


def exponential_tail_fractions(bounds: np.ndarray, start: float, kappa: float) -> np.ndarray:
    """Layer fractions of a density ``kappa exp(-kappa (x - start))`` for ``x >= start`` (zero in front)."""

    front = np.maximum(bounds[:, 0], start) - start
    back = np.maximum(bounds[:, 1], start) - start
    return np.exp(-kappa * front) - np.exp(-kappa * back)


def predict(spec: ModelSpec, params: dict[str, float], targets: Targets) -> tuple[np.ndarray, np.ndarray]:
    """``(model fractions, per-energy amplitude)`` for the targets' energies."""

    energies = targets.energies_mev
    x_max = np.array(
        [
            params.get(
                f"xmax_{k}",
                x_max_of_energy(e, targets.critical_energy_mev, params["delta"], params["depth_slope"]),
            )
            for k, e in enumerate(energies)
        ]
    )
    if params["z0_slope"] == 0.0 and params["beta_slope"] == 0.0:
        model = ensemble_mean_fractions(
            params["z0"],
            params["beta"],
            x_max,
            energies,
            targets.critical_energy_mev,
            targets.bounds,
            fluctuate=spec.fluctuate,
            covariant=spec.covariant,
        )
    else:  # parameters that depend on the energy: one ensemble per energy
        ln_ratio = np.log(targets.energies_gev / ENERGY_REFERENCE_GEV)
        model = np.array(
            [
                ensemble_mean_fractions(
                    params["z0"] + params["z0_slope"] * ln_ratio[k],
                    params["beta"] + params["beta_slope"] * ln_ratio[k],
                    x_max[k : k + 1],
                    energies[k : k + 1],
                    targets.critical_energy_mev,
                    targets.bounds,
                    fluctuate=spec.fluctuate,
                    covariant=spec.covariant,
                )[0]
                for k in range(len(energies))
            ]
        )
    if params["tail_w"] != 0.0 or params["tail_w_slope"] != 0.0:
        # a slowly decaying tail component (a second gamma density with rate kappa) mixed into the core profile
        ln_ratio = np.log(targets.energies_gev / ENERGY_REFERENCE_GEV)
        weight = np.clip(params["tail_w"] + params["tail_w_slope"] * ln_ratio, 0.0, 0.95)[:, None]
        if params["tail_onset"] < 0.5 * TAIL_ONSET_OFF:
            # an exponential tail of rate kappa that starts ``tail_onset`` X0 behind the profile maximum
            tail = np.array(
                [
                    exponential_tail_fractions(targets.bounds, float(x_max[k]) + params["tail_onset"], params["tail_kappa"])
                    for k in range(len(energies))
                ]
            )
        else:
            tail_depth = (params["tail_alpha"] - 1.0) / params["tail_kappa"]
            tail = profile_fractions_batch(
                np.array([np.log(tail_depth)]), np.array([np.log(params["tail_alpha"])]), targets.bounds, params["z0"]
            )[0][None, :]
        model = (1.0 - weight) * model + weight * tail
    if params["floor_a"] > 0.0 or params["floor_b"] > 0.0:
        floor_mev = params["floor_a"] + params["floor_b"] * targets.energies_gev
        share = (floor_mev / energies)[:, None]
        model = (1.0 - share) * model
        model[:, 0] += share[:, 0]
    amplitude = np.ones(len(energies))
    if spec.amplitude == "free":
        weight = 1.0 / targets.se**2
        amplitude = (model * targets.mean * weight).sum(axis=1) / (model * model * weight).sum(axis=1)
        model = model * amplitude[:, None]
    return model, amplitude


def residuals(spec: ModelSpec, theta: np.ndarray, targets: Targets) -> np.ndarray:
    model, _ = predict(spec, _parameters(spec, theta), targets)
    return ((model - targets.mean) / targets.se).ravel()


def _bounds(spec: ModelSpec) -> tuple[list[float], list[float]]:
    low, high = [], []
    for name in spec.free:
        low_, high_ = (2.0, 12.0) if name.startswith("xmax_") else PARAMETER_BOUNDS[name]
        low.append(low_)
        high.append(high_)
    return low, high


def default_starts(spec: ModelSpec, targets: Targets) -> list[np.ndarray]:
    base: dict[str, list[float]] = {
        "z0": [-2.5, -0.5, 0.0],
        "beta": [0.65, 0.55, 0.5],
        "delta": [DELTA_DEC001],
        "floor_a": [20.0],
        "floor_b": [0.2],
        "z0_slope": [0.0],
        "beta_slope": [0.0],
        "depth_slope": [1.0],
        "tail_w": [0.05],
        "tail_w_slope": [0.0],
        "tail_alpha": [4.0],
        "tail_kappa": [0.278],
        "tail_onset": [4.0],
    }
    starts = []
    for i in range(3):
        start = []
        for name in spec.free:
            if name.startswith("xmax_"):
                k = int(name.split("_")[1])
                start.append(
                    x_max_of_energy(targets.energies_mev[k], targets.critical_energy_mev, DELTA_DEC001)
                )
            else:
                values = base[name]
                start.append(values[min(i, len(values) - 1)])
        starts.append(np.array(start, dtype=float))
    return starts


@dataclass
class Fit:
    spec: ModelSpec
    theta: np.ndarray
    chi2: float
    n_data: int
    jacobian: np.ndarray

    @property
    def dof(self) -> int:
        return self.n_data - len(self.theta)

    @property
    def scale(self) -> float:
        """Error rescaling: chi-square per degree of freedom at the optimum."""

        return self.chi2 / max(self.dof, 1)

    def parameters(self) -> dict[str, float]:
        return _parameters(self.spec, self.theta)


def fit_model(spec: ModelSpec, targets: Targets, starts: list[np.ndarray] | None = None) -> Fit:
    """Least-squares fit of the free parameters (all of them fixed: just evaluate)."""

    if not spec.free:
        r = residuals(spec, np.array([]), targets)
        return Fit(spec, np.array([]), float(r @ r), targets.mean.size, np.zeros((targets.mean.size, 0)))
    low, high = _bounds(spec)
    best = None
    for start in starts or default_starts(spec, targets):
        result = least_squares(
            lambda p: residuals(spec, p, targets), np.clip(start, low, high), bounds=(low, high)
        )
        if best is None or result.cost < best.cost:
            best = result
    assert best is not None
    return Fit(spec, best.x, float(2.0 * best.cost), targets.mean.size, best.jac)


# ----------------------------------------------------------------------
# Surface, degeneracy and bootstrap
# ----------------------------------------------------------------------

PRIMARY = ModelSpec("origin_beta", ("z0", "beta"))
DETERMINISTIC = ModelSpec("origin_beta_deterministic", ("z0", "beta"), fluctuate=False)
NAIVE_WIDTH = ModelSpec("origin_beta_naive_width", ("z0", "beta"), covariant=False)
ORIGIN_BETA_OFFSET = ModelSpec("origin_beta_offset", ("z0", "beta", "delta"))
AMPLITUDE_FREE = ModelSpec("origin_beta_amplitude_free", ("z0", "beta"), amplitude="free")
AMS_BETA_FREE_ORIGIN = ModelSpec("ams_beta_free_origin", ("z0",), {"beta": BETA_AMS})
FRONT_FACE_AMS = ModelSpec("front_face_ams", (), {"z0": 0.0, "beta": BETA_AMS})
FRONT_FACE_FREE_BETA = ModelSpec("front_face_free_beta", ("beta",), {"z0": 0.0})
FRONT_FACE_FREE_BETA_OFFSET = ModelSpec("front_face_free_beta_offset", ("beta", "delta"), {"z0": 0.0})
READOUT = ModelSpec("readout_origin_beta", ("z0", "beta"), amplitude="free")
READOUT_OFFSET = ModelSpec("readout_origin_beta_offset", ("z0", "beta", "delta"), amplitude="free")


def origin_beta_free_depth(n_energies: int) -> ModelSpec:
    return ModelSpec(
        "origin_beta_free_depth", ("z0", "beta", *[f"xmax_{k}" for k in range(n_energies)])
    )


def evaluate(spec: ModelSpec, theta: np.ndarray, targets: Targets) -> float:
    r = residuals(spec, theta, targets)
    return float(r @ r)


def chi2_surface(targets: Targets, spec: ModelSpec = PRIMARY) -> np.ndarray:
    """Chi-square on the ``(z0, beta)`` grid (the other parameters at the spec's defaults)."""

    surface = np.empty((len(Z0_GRID), len(BETA_GRID)))
    for i, z0 in enumerate(Z0_GRID):
        for j, beta in enumerate(BETA_GRID):
            surface[i, j] = evaluate(spec, np.array([z0, beta]), targets)
    return surface


def surface_summary(surface: np.ndarray, fit: Fit) -> dict[str, Any]:
    """Contours, profile valley and marginal ranges of the scaled chi-square surface."""

    delta = (surface - surface.min()) / fit.scale
    index = np.unravel_index(surface.argmin(), surface.shape)
    out: dict[str, Any] = {
        "scale_chi2_per_dof": fit.scale,
        "grid_minimum": {"z0": float(Z0_GRID[index[0]]), "beta": float(BETA_GRID[index[1]])},
        "region_extent": {},
    }
    for label, level in CONTOUR_LEVELS.items():
        inside = delta <= level
        out["region_extent"][label] = {
            "z0": [float(Z0_GRID[inside.any(axis=1)].min()), float(Z0_GRID[inside.any(axis=1)].max())],
            "beta": [
                float(BETA_GRID[inside.any(axis=0)].min()),
                float(BETA_GRID[inside.any(axis=0)].max()),
            ],
            "grid_points_inside": int(inside.sum()),
            "touches_grid_edge": bool(
                inside[0].any() or inside[-1].any() or inside[:, 0].any() or inside[:, -1].any()
            ),
        }
    profile_over_beta = delta.min(axis=1)
    profile_over_z0 = delta.min(axis=0)
    out["profile_valley"] = {
        "z0": Z0_GRID.tolist(),
        "best_beta_given_z0": BETA_GRID[delta.argmin(axis=1)].tolist(),
        "delta_chi2_scaled_given_z0": profile_over_beta.tolist(),
        "beta_grid": BETA_GRID.tolist(),
        "best_z0_given_beta": Z0_GRID[delta.argmin(axis=0)].tolist(),
        "delta_chi2_scaled_given_beta": profile_over_z0.tolist(),
    }
    out["marginal_one_sigma_profile"] = {
        "z0": [
            float(Z0_GRID[profile_over_beta <= 1.0].min()),
            float(Z0_GRID[profile_over_beta <= 1.0].max()),
        ],
        "beta": [
            float(BETA_GRID[profile_over_z0 <= 1.0].min()),
            float(BETA_GRID[profile_over_z0 <= 1.0].max()),
        ],
    }
    return out


def hessian_summary(fit: Fit) -> dict[str, Any]:
    """Local degeneracy from the Gauss-Newton Hessian: eigen-directions, condition number, covariance."""

    jtj = fit.jacobian.T @ fit.jacobian
    values, vectors = np.linalg.eigh(jtj)
    covariance = fit.scale * np.linalg.inv(jtj)
    sd = np.sqrt(np.diag(covariance))
    names = list(fit.spec.free)
    return {
        "parameters": names,
        "eigenvalues": values.tolist(),
        "condition_number": float(values.max() / max(values.min(), 1e-300)),
        "valley_direction": dict(zip(names, vectors[:, 0].tolist(), strict=True)),
        "stiff_direction": dict(zip(names, vectors[:, -1].tolist(), strict=True)),
        "scaled_covariance": covariance.tolist(),
        "scaled_sd": dict(zip(names, sd.tolist(), strict=True)),
        "correlation": float(covariance[0, 1] / (sd[0] * sd[1])) if len(names) >= 2 else None,
    }


def bootstrap(spec: ModelSpec, targets: Targets, start: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([fit_model(spec, targets.resample(rng), [start]).theta for _ in range(n)])


def bootstrap_summary(samples: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n": len(samples),
        "mean": dict(zip(names, samples.mean(axis=0).tolist(), strict=True)),
        "sd": dict(zip(names, samples.std(axis=0, ddof=1).tolist(), strict=True)),
        "percentiles_2.5_50_97.5": {
            n_: np.percentile(samples[:, i], [2.5, 50, 97.5]).tolist() for i, n_ in enumerate(names)
        },
    }
    if len(names) >= 2:
        values, vectors = np.linalg.eigh(np.cov(samples.T))
        out["correlation_z0_beta"] = float(np.corrcoef(samples[:, 0], samples[:, 1])[0, 1])
        out["valley_direction"] = dict(zip(names, vectors[:, 0].tolist(), strict=True))
        out["axis_ratio"] = float(np.sqrt(values.max() / max(values.min(), 1e-300)))
    return out


# ----------------------------------------------------------------------
# Fit summaries and cross-validation
# ----------------------------------------------------------------------


def fit_summary(fit: Fit, targets: Targets) -> dict[str, Any]:
    params = fit.parameters()
    model, amplitude = predict(fit.spec, params, targets)
    relative = (model - targets.mean) / targets.mean
    pull = (model - targets.mean) / targets.se
    return {
        "model": fit.spec.name,
        "free": list(fit.spec.free),
        "parameters": {k: float(v) for k, v in params.items()},
        "chi2": fit.chi2,
        "dof": fit.dof,
        "chi2_per_dof": fit.scale if fit.dof > 0 else None,
        "aic_scaled": float(fit.chi2 / max(fit.scale, 1e-12) + 2 * len(fit.theta)) if fit.dof > 0 else None,
        "energies_gev": targets.energies_gev.tolist(),
        "relative_residual_by_layer": relative.tolist(),
        "pull_by_layer": pull.tolist(),
        "chi2_by_energy": (pull**2).sum(axis=1).tolist(),
        "rms_relative_layers_0_3": np.sqrt((relative[:, :4] ** 2).mean(axis=1)).tolist(),
        "rms_relative_layers_4_17": np.sqrt((relative[:, 4:] ** 2).mean(axis=1)).tolist(),
        "model_mean_fraction": model.tolist(),
        "geant4_mean_fraction": targets.mean.tolist(),
        "contained_fraction_model": model.sum(axis=1).tolist(),
        "contained_fraction_geant4": targets.mean.sum(axis=1).tolist(),
        "amplitude": amplitude.tolist(),
    }


def held_out_score(
    spec: ModelSpec, train: Targets, test: Targets, start: np.ndarray | None = None
) -> dict[str, float]:
    """Fit on ``train``, score on ``test``: chi-square per point and relative rms by layer group."""

    fit = fit_model(spec, train, None if start is None else [start])
    model, _ = predict(spec, fit.parameters(), test)
    relative = (model - test.mean) / test.mean
    pull = (model - test.mean) / test.se
    return {
        "chi2_per_point": float((pull**2).mean()),
        "rms_relative_layers_0_3": float(np.sqrt((relative[:, :4] ** 2).mean())),
        "rms_relative_layers_4_17": float(np.sqrt((relative[:, 4:] ** 2).mean())),
    }


def leave_one_energy_out(spec: ModelSpec, targets: Targets) -> dict[str, Any]:
    """Fit the common parameters on three energies, predict the fourth (the energy-commonality test)."""

    rows = []
    for k in range(len(targets.energies_gev)):
        keep = [i for i in range(len(targets.energies_gev)) if i != k]
        rows.append(
            {
                "left_out_gev": float(targets.energies_gev[k]),
                **held_out_score(spec, targets.subset(keep), targets.subset([k])),
            }
        )
    return {
        "per_energy": rows,
        "mean_chi2_per_point": float(np.mean([r["chi2_per_point"] for r in rows])),
        "mean_rms_relative_layers_0_3": float(np.mean([r["rms_relative_layers_0_3"] for r in rows])),
        "mean_rms_relative_layers_4_17": float(np.mean([r["rms_relative_layers_4_17"] for r in rows])),
    }


def split_half(spec: ModelSpec, targets: Targets) -> dict[str, float]:
    a, b = targets.halves()
    forward, backward = held_out_score(spec, a, b), held_out_score(spec, b, a)
    return {key: 0.5 * (forward[key] + backward[key]) for key in forward}


def per_energy_fits(targets: Targets) -> list[dict[str, Any]]:
    """An independent (z0, beta) at each energy: identifiability and commonality diagnostic."""

    rows = []
    for k, energy in enumerate(targets.energies_gev):
        fit = fit_model(PRIMARY, targets.subset([k]))
        hess = hessian_summary(fit)
        rows.append(
            {
                "energy_gev": float(energy),
                "z0": float(fit.theta[0]),
                "beta": float(fit.theta[1]),
                "chi2": fit.chi2,
                "scaled_sd_z0": hess["scaled_sd"]["z0"],
                "scaled_sd_beta": hess["scaled_sd"]["beta"],
                "condition_number": hess["condition_number"],
                "correlation": hess["correlation"],
            }
        )
    return rows


# ----------------------------------------------------------------------
# The entrance-residual gate and the floor
# ----------------------------------------------------------------------


def entrance_gate(
    primary: Fit,
    alternatives: dict[str, Fit],
    targets: Targets,
    readout_targets: Targets,
) -> dict[str, Any]:
    """Decide, from pre-stated criteria (``GATE``), whether a first-layer floor may be tested."""

    model, _ = predict(primary.spec, primary.parameters(), targets)
    relative = (model[:, 0] - targets.mean[:, 0]) / targets.mean[:, 0]
    pull = (model[:, 0] - targets.mean[:, 0]) / targets.se[:, 0]
    alternative_residuals = {}
    for name, fit in alternatives.items():
        alt_model, _ = predict(fit.spec, fit.parameters(), targets)
        alternative_residuals[name] = ((alt_model[:, 0] - targets.mean[:, 0]) / targets.mean[:, 0]).tolist()
    absorbed = {
        name: float(1.0 - np.mean(np.abs(r)) / np.mean(np.abs(relative)))
        for name, r in alternative_residuals.items()
    }
    contained_residual = np.abs(model.sum(axis=1) / targets.mean.sum(axis=1) - 1.0)
    params = primary.parameters()
    readout_spec = ModelSpec(
        "readout_with_deposition_parameters",
        (),
        {"z0": params["z0"], "beta": params["beta"], "delta": params["delta"]},
        amplitude="free",
    )
    readout_model, _ = predict(readout_spec, _parameters(readout_spec, np.array([])), readout_targets)
    readout_relative = (readout_model[:, 0] - readout_targets.mean[:, 0]) / readout_targets.mean[:, 0]
    conditions = {
        "layer0_residual_large_at_every_energy": bool(
            np.all(np.abs(relative) >= GATE["min_relative_layer0_residual"])
        ),
        "layer0_residual_significant_at_every_energy": bool(np.all(np.abs(pull) >= GATE["min_significance"])),
        "same_sign_at_every_energy": bool(np.all(np.sign(relative) == np.sign(relative[0]))),
        "reproducible_magnitude_across_energy": bool(
            np.max(np.abs(relative)) / max(np.min(np.abs(relative)), 1e-12) <= GATE["max_magnitude_spread"]
        ),
        "not_absorbed_by_alternatives": bool(all(a <= GATE["absorbed_fraction"] for a in absorbed.values())),
        "not_a_leakage_accounting_problem": bool(
            np.all(contained_residual <= GATE["max_contained_fraction_residual"])
        ),
    }
    return {
        "criteria": GATE,
        "layer0_relative_residual": relative.tolist(),
        "layer0_pull": pull.tolist(),
        "layers_1_3_relative_residual": (
            (model[:, 1:4] - targets.mean[:, 1:4]) / targets.mean[:, 1:4]
        ).tolist(),
        "alternative_layer0_relative_residual": alternative_residuals,
        "fraction_of_layer0_residual_absorbed_by": absorbed,
        "contained_fraction_relative_residual": contained_residual.tolist(),
        "readout_layer0_relative_residual_with_deposition_parameters": readout_relative.tolist(),
        "normalisation_check": (
            "the `amplitude_free` alternative; finite-layer integration is exact (gamma CDF differences)"
        ),
        "conditions": conditions,
        "open": bool(all(conditions.values())),
    }


def floor_study(primary: Fit, targets: Targets) -> dict[str, Any]:
    """Run only if the gate is open: constant floor first; the linear term only if the residual demands it."""

    constant = ModelSpec("origin_beta_floor_constant", ("z0", "beta", "floor_a"))
    linear = ModelSpec("origin_beta_floor_linear", ("z0", "beta", "floor_a", "floor_b"))
    base_loeo = leave_one_energy_out(primary.spec, targets)
    base_half = split_half(primary.spec, targets)
    out: dict[str, Any] = {"baseline_held_out": {"leave_one_energy_out": base_loeo, "split_half": base_half}}
    candidates = {"constant": constant}
    fit_c = fit_model(constant, targets)
    out["constant"] = fit_summary(fit_c, targets)
    rel0 = np.array(out["constant"]["relative_residual_by_layer"])[:, 0]
    out["linear_term_considered"] = bool(np.all(np.abs(rel0) >= GATE["min_relative_layer0_residual"]))
    if out["linear_term_considered"]:
        candidates["linear"] = linear
        out["linear"] = fit_summary(fit_model(linear, targets), targets)
    verdicts = {}
    for name, spec in candidates.items():
        loeo, half = leave_one_energy_out(spec, targets), split_half(spec, targets)
        out[name]["held_out"] = {"leave_one_energy_out": loeo, "split_half": half}
        improvement_loeo = 1.0 - loeo["mean_chi2_per_point"] / base_loeo["mean_chi2_per_point"]
        improvement_half = 1.0 - half["chi2_per_point"] / base_half["chi2_per_point"]
        degradation = loeo["mean_rms_relative_layers_4_17"] / base_loeo["mean_rms_relative_layers_4_17"] - 1.0
        verdicts[name] = {
            "held_out_improvement_leave_one_energy_out": float(improvement_loeo),
            "held_out_improvement_split_half": float(improvement_half),
            "layers_4_17_degradation_leave_one_energy_out": float(degradation),
            "accepted": bool(
                improvement_loeo >= GATE["cv_improvement_to_accept"]
                and improvement_half >= GATE["cv_improvement_to_accept"]
                and degradation <= GATE["max_layers_4_17_degradation"]
            ),
        }
    out["verdicts"] = verdicts
    return out


# ----------------------------------------------------------------------
# Consistency at the calibrated coordinate (C) and recalibrated structure (D)
# ----------------------------------------------------------------------


def fit_cumulative_with_origin(fractions: np.ndarray, bounds: np.ndarray, origin: float) -> np.ndarray:
    target = np.cumsum(fractions)
    result = least_squares(
        lambda p: np.cumsum(profile_fractions_batch(p[0:1], p[1:2], bounds, origin)[0]) - target,
        START,
        bounds=(LOWER, UPPER),
    )
    return result.x


def per_event_fits(
    loaded: dict[float, EnergyData], bounds: np.ndarray, origin: float, estimator: str
) -> dict[float, np.ndarray]:
    out = {}
    for energy, data in loaded.items():
        fractions = data.fractions["deposition"]
        if estimator == "least_squares":
            out[energy] = np.array([fit_free_with_origin(f, bounds, origin)[0] for f in fractions])
        else:
            out[energy] = np.array([fit_cumulative_with_origin(f, bounds, origin) for f in fractions])
    return out


def stochastic_structure(
    fits: dict[float, np.ndarray], z0: float, delta: float, critical_energy_mev: float
) -> dict[str, Any]:
    """Part D: the per-event structure at one origin against the GP formulae (naive and covariant)."""

    out = {}
    for energy, fit in fits.items():
        ln_t, ln_alpha = fit[:, 0], fit[:, 1]
        beta = beta_values(ln_t, ln_alpha)
        energy_mev = 1000.0 * energy
        ln_y = float(np.log(energy_mev / critical_energy_mev))
        x_max = x_max_of_energy(energy_mev, critical_energy_mev, delta)
        t_origin = max(x_max - z0, 0.2)
        gp = gp_formulae(ln_y)["sampling"]
        sigma_t_covariant = gp["sigma_ln_t"] * x_max / t_origin
        marginal_t, marginal_a = marginal_diagnostics(ln_t), marginal_diagnostics(ln_alpha)
        joint = joint_diagnostics(ln_t, ln_alpha)
        ratios = {
            "sigma_ln_t_over_gp_naive": marginal_t["sd"] / gp["sigma_ln_t"],
            "sigma_ln_t_over_gp_covariant": marginal_t["sd"] / sigma_t_covariant,
            "sigma_ln_alpha_over_gp": marginal_a["sd"] / gp["sigma_ln_alpha"],
            "rho_minus_gp": joint["pearson"] - gp["rho"],
        }
        out[f"{energy:g}"] = {
            "ln_t": marginal_t,
            "ln_alpha": marginal_a,
            "beta": {
                **marginal_diagnostics(beta),
                "quantiles_5_16_50_84_95": np.quantile(beta, [0.05, 0.16, 0.5, 0.84, 0.95]).tolist(),
            },
            "joint": joint,
            "gp": gp,
            "gp_sigma_ln_t_covariant": sigma_t_covariant,
            "ratios": ratios,
            "coherent": {
                "sigma_ln_t_covariant": bool(
                    COHERENCE_BAND[0] <= ratios["sigma_ln_t_over_gp_covariant"] <= COHERENCE_BAND[1]
                ),
                "sigma_ln_alpha": bool(
                    COHERENCE_BAND[0] <= ratios["sigma_ln_alpha_over_gp"] <= COHERENCE_BAND[1]
                ),
                "rho": bool(abs(ratios["rho_minus_gp"]) <= COHERENCE_RHO_TOLERANCE),
            },
        }
    return out


def skewness_versus_origin(loaded: dict[float, EnergyData], bounds: np.ndarray) -> dict[str, Any]:
    """Does the ln T skewness depend on where the depth origin is put? (least-squares estimator)"""

    table: dict[str, Any] = {"origins_x0": list(ORIGINS_FOR_SKEWNESS)}
    for energy, data in loaded.items():
        rows = []
        for origin in ORIGINS_FOR_SKEWNESS:
            fit = np.array(
                [fit_free_with_origin(f, bounds, origin)[0] for f in data.fractions["deposition"]]
            )
            rows.append(
                {
                    "origin_x0": float(origin),
                    "mean_ln_t": float(fit[:, 0].mean()),
                    "sd_ln_t": float(fit[:, 0].std(ddof=1)),
                    "sd_ln_alpha": float(fit[:, 1].std(ddof=1)),
                    "skewness_ln_t": float(stats.skew(fit[:, 0])),
                    "skewness_ln_alpha": float(stats.skew(fit[:, 1])),
                    "excess_kurtosis_ln_t": float(stats.kurtosis(fit[:, 0])),
                    "pearson": float(np.corrcoef(fit.T)[0, 1]),
                    "beta_median": float(np.median(beta_values(fit[:, 0], fit[:, 1]))),
                }
            )
        table[f"{energy:g}"] = rows
    return table


def estimator_skewness_control(
    loaded: dict[float, EnergyData],
    fits_at_origin: dict[float, np.ndarray],
    bounds: np.ndarray,
    z0: float,
    n: int = ESTIMATOR_CONTROL_EVENTS,
) -> dict[str, Any]:
    """Planted-null control: can the per-event fit alone turn a symmetric ln T into a skewed one?

    Draw (ln T, ln alpha) from a bivariate NORMAL with the measured mean and covariance (skewness zero by
    construction), build the profiles, add the residual vectors (Geant4 event minus its own best-fit
    profile) of randomly chosen real events, refit, and compare the refitted skewness with the measured
    one. The residuals are borrowed from other events, so their dependence on the event's own shape is not
    reproduced: a qualitative control, not an exact calibration of the estimator bias.
    """

    out: dict[str, Any] = {}
    for energy, data in loaded.items():
        rng = np.random.default_rng(int(1000 * energy) + 17)
        fractions = data.fractions["deposition"]
        measured = fits_at_origin[energy]
        residual = fractions - profile_fractions_batch(measured[:, 0], measured[:, 1], bounds, z0)
        joint = rng.multivariate_normal(measured.mean(axis=0), np.cov(measured.T), size=n)
        synthetic = profile_fractions_batch(joint[:, 0], joint[:, 1], bounds, z0)
        synthetic = np.clip(synthetic + residual[rng.integers(0, len(residual), n)], 0.0, None)
        refit = np.array([fit_free_with_origin(f, bounds, z0)[0] for f in synthetic])
        out[f"{energy:g}"] = {
            "planted_skewness_ln_t": float(stats.skew(joint[:, 0])),
            "refitted_skewness_ln_t": float(stats.skew(refit[:, 0])),
            "measured_skewness_ln_t": float(stats.skew(measured[:, 0])),
            "planted_sd_ln_t": float(joint[:, 0].std(ddof=1)),
            "refitted_sd_ln_t": float(refit[:, 0].std(ddof=1)),
            "measured_sd_ln_t": float(measured[:, 0].std(ddof=1)),
            "planted_pearson": float(np.corrcoef(joint.T)[0, 1]),
            "refitted_pearson": float(np.corrcoef(refit.T)[0, 1]),
            "measured_pearson": float(np.corrcoef(measured.T)[0, 1]),
        }
    return out


def calibrated_ensembles(
    loaded: dict[float, EnergyData],
    params: dict[str, float],
    fits_at_origin: dict[float, np.ndarray],
    critical_energy_mev: float,
    bounds: np.ndarray,
    n: int,
) -> dict[float, dict[str, np.ndarray]]:
    """Stochastic ensembles at the calibrated coordinate: GP-prescribed spreads and measured spreads."""

    z0, beta, delta = params["z0"], params["beta"], params["delta"]
    out: dict[float, dict[str, np.ndarray]] = {}
    for energy in loaded:
        rng = np.random.default_rng(int(1000 * energy) + 6)
        energy_mev = 1000.0 * energy
        x_max = x_max_of_energy(energy_mev, critical_energy_mev, delta)
        t_origin = max(x_max - z0, 0.2)
        gp = gp_formulae(float(np.log(energy_mev / critical_energy_mev)))["sampling"]
        z_t, z_perp = rng.standard_normal(n), rng.standard_normal(n)
        ln_t = np.log(t_origin) + gp["sigma_ln_t"] * x_max / t_origin * z_t
        ln_alpha = np.log(1.0 + beta * t_origin) + gp["sigma_ln_alpha"] * (
            gp["rho"] * z_t + np.sqrt(1.0 - gp["rho"] ** 2) * z_perp
        )
        measured = fits_at_origin[energy]
        joint = rng.multivariate_normal(measured.mean(axis=0), np.cov(measured.T), size=n)
        out[energy] = {
            "calibrated_gp_prescribed": profile_fractions_batch(ln_t, ln_alpha, bounds, z0),
            "calibrated_measured_spread": profile_fractions_batch(joint[:, 0], joint[:, 1], bounds, z0),
        }
    return out


def consistency_report(
    loaded: dict[float, EnergyData],
    ensembles: dict[float, dict[str, np.ndarray]],
    dec001_ensembles: dict[float, np.ndarray],
    z_mm: np.ndarray,
) -> dict[str, Any]:
    """Part C: the calibrated-coordinate ensembles against Geant4 and against DEC-001 (same metrics as the longitudinal-structure analysis)."""

    out = {}
    for energy, data in loaded.items():
        reference = data.fractions["deposition"]
        metrics = {name: longitudinal_metrics(f, reference, z_mm) for name, f in ensembles[energy].items()}
        dec001_metrics = longitudinal_metrics(dec001_ensembles[energy], reference, z_mm)
        dist = {name: distances(m) for name, m in metrics.items()}
        dist_dec = distances(dec001_metrics)
        out[f"{energy:g}"] = {
            "metrics": metrics,
            "dec001_metrics": dec001_metrics,
            "distances_to_geant4": dist,
            "dec001_distances_to_geant4": dist_dec,
            "removed_vs_dec001": {name: removed_fraction(dist_dec, d) for name, d in dist.items()},
        }
    return out


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------


def model_family_specs(n_energies: int, amplitude: str) -> dict[str, ModelSpec]:
    """The model forms that move the fitted origin: beta fixed or free, depth law common or free, fluctuations or not."""

    depth = [f"xmax_{k}" for k in range(n_energies)]
    return {
        "beta_fixed_common_depth": ModelSpec("a", ("z0",), {"beta": BETA_AMS}, amplitude=amplitude),
        "beta_fixed_free_depth": ModelSpec("b", ("z0", *depth), {"beta": BETA_AMS}, amplitude=amplitude),
        "beta_fixed_free_depth_deterministic": ModelSpec(
            "c", ("z0", *depth), {"beta": BETA_AMS}, fluctuate=False, amplitude=amplitude
        ),
        "beta_free_common_depth": ModelSpec("d", ("z0", "beta"), amplitude=amplitude),
        "beta_free_free_depth": ModelSpec("e", ("z0", "beta", *depth), amplitude=amplitude),
        "beta_free_free_depth_deterministic": ModelSpec(
            "f", ("z0", "beta", *depth), fluctuate=False, amplitude=amplitude
        ),
    }


def model_family_fits(
    targets: Targets, amplitude: str = "none", families: Sequence[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Fit each model form on ``targets``; the origin is a coordinate convention tied to the form."""

    specs = model_family_specs(len(targets.energies_gev), amplitude)
    out = {}
    for name in families or specs:
        fit = fit_model(specs[name], targets)
        params = fit.parameters()
        out[name] = {
            "z0": float(params["z0"]),
            "beta": float(params["beta"]),
            "chi2": fit.chi2,
            "dof": fit.dof,
            "chi2_per_dof": fit.scale,
            "x_max_by_energy": [
                float(params.get(f"xmax_{k}", x_max_of_energy(e, targets.critical_energy_mev, params["delta"])))
                for k, e in enumerate(targets.energies_mev)
            ],
        }
    return out


def analyse(
    energies: Sequence[float] = ENERGIES_GEV,
    *,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    n_boot_secondary: int = BOOTSTRAP_RESAMPLES_SECONDARY,
    n_ensemble: int = ENSEMBLE_EVENTS,
    skewness: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run slices A-F. Returns ``(results, artefacts)``; ``artefacts`` hold arrays for the plots."""

    bounds = layer_bounds_x0()
    z_mm = layer_centres_mm(load_geometry(GEOMETRY_CONFIG), 18)
    loaded = {float(e): load_energy(e) for e in energies}
    deposition = make_targets(loaded, "deposition", bounds)
    readout = make_targets(loaded, "readout", bounds)
    n_e = len(deposition.energies_gev)
    free_depth = origin_beta_free_depth(n_e)
    results: dict[str, Any] = {}
    artefacts: dict[str, Any] = {"loaded": loaded, "targets": deposition, "readout_targets": readout}

    # ---- A: the fits
    specs = [
        FRONT_FACE_AMS,
        FRONT_FACE_FREE_BETA,
        FRONT_FACE_FREE_BETA_OFFSET,
        AMS_BETA_FREE_ORIGIN,
        PRIMARY,
        DETERMINISTIC,
        NAIVE_WIDTH,
        ORIGIN_BETA_OFFSET,
        free_depth,
        AMPLITUDE_FREE,
    ]
    fits = {spec.name: fit_model(spec, deposition) for spec in specs}
    artefacts["fits"] = fits
    results["A_fits"] = {name: fit_summary(f, deposition) for name, f in fits.items()}
    primary = fits[PRIMARY.name]
    params = primary.parameters()
    results["A_primary"] = {
        "model": PRIMARY.name,
        "z0": float(primary.theta[0]),
        "beta": float(primary.theta[1]),
        "delta": DELTA_DEC001,
        "x_max_by_energy": [
            x_max_of_energy(e, deposition.critical_energy_mev, DELTA_DEC001) for e in deposition.energies_mev
        ],
        "coordinate_convention": (
            "u = x - z0, x in X0 from the ECAL front face; z0 < 0 is upstream of the front face; "
            "gamma density in u with shape alpha = 1 + beta T_o and mode T_o = x_max - z0"
        ),
    }

    # ---- B: identifiability
    surface = chi2_surface(deposition, PRIMARY)
    artefacts["surface"] = surface
    results["B_surface"] = surface_summary(surface, primary)
    results["B_hessian"] = hessian_summary(primary)
    boot = bootstrap(PRIMARY, deposition, primary.theta, n_boot, seed=11)
    artefacts["bootstrap_primary"] = boot
    results["B_bootstrap"] = bootstrap_summary(boot, ["z0", "beta"])
    for alt in (DETERMINISTIC, NAIVE_WIDTH):
        alt_boot = bootstrap(alt, deposition, fits[alt.name].theta, n_boot_secondary, seed=12)
        artefacts[f"bootstrap_{alt.name}"] = alt_boot
        results[f"B_bootstrap_{alt.name}"] = bootstrap_summary(alt_boot, ["z0", "beta"])
    results["B_per_energy_fits"] = per_energy_fits(deposition)
    results["B_leave_one_energy_out"] = {
        PRIMARY.name: leave_one_energy_out(PRIMARY, deposition),
        ORIGIN_BETA_OFFSET.name: leave_one_energy_out(ORIGIN_BETA_OFFSET, deposition),
    }
    results["B_split_half"] = {PRIMARY.name: split_half(PRIMARY, deposition)}

    # ---- C and D: the calibrated coordinate
    z0 = float(primary.theta[0])
    fits_ls = per_event_fits(loaded, bounds, z0, "least_squares")
    fits_cum = per_event_fits(loaded, bounds, z0, "cumulative")
    artefacts["fits_at_origin"] = {"least_squares": fits_ls, "cumulative": fits_cum}
    results["D_stochastic_structure"] = {
        "origin_x0": z0,
        "least_squares": stochastic_structure(fits_ls, z0, DELTA_DEC001, deposition.critical_energy_mev),
        "cumulative": stochastic_structure(fits_cum, z0, DELTA_DEC001, deposition.critical_energy_mev),
    }
    results["D_estimator_skewness_control"] = estimator_skewness_control(loaded, fits_ls, bounds, z0)
    if skewness:
        results["D_skewness_versus_origin"] = skewness_versus_origin(loaded, bounds)
    ensembles = calibrated_ensembles(loaded, params, fits_ls, deposition.critical_energy_mev, bounds, n_ensemble)
    artefacts["ensembles"] = ensembles
    dec001_ensembles = {
        energy: dec001_ensemble(energy, "deposition", n_ensemble, int(1000 * energy)) for energy in loaded
    }
    artefacts["dec001_ensembles"] = dec001_ensembles
    results["C_consistency"] = consistency_report(loaded, ensembles, dec001_ensembles, z_mm)

    # ---- E: the gate
    alternatives = {
        "amplitude_free": fits[AMPLITUDE_FREE.name],
        "origin_beta_offset": fits[ORIGIN_BETA_OFFSET.name],
        "origin_beta_free_depth": fits[free_depth.name],
    }
    results["E_gate"] = entrance_gate(primary, alternatives, deposition, readout)
    if results["E_gate"]["open"]:
        results["E_floor"] = floor_study(primary, deposition)
    else:
        results["E_floor"] = {
            "status": "gate closed: no floor was fitted",
            "conditions": results["E_gate"]["conditions"],
        }

    # ---- F: deposition versus readout (interpretation only)
    readout_fit = fit_model(READOUT, readout)
    readout_offset = fit_model(READOUT_OFFSET, readout)
    readout_surface = chi2_surface(readout, READOUT)
    artefacts["readout_surface"] = readout_surface
    readout_boot = bootstrap(READOUT, readout, readout_fit.theta, n_boot_secondary, seed=13)
    artefacts["bootstrap_readout"] = readout_boot
    with_deposition = ModelSpec(
        "readout_with_deposition_parameters", (), {"z0": z0, "beta": float(primary.theta[1])}, amplitude="free"
    )
    fit_with_deposition = fit_model(with_deposition, readout)
    readout_sd = np.std(readout_boot, axis=0, ddof=1)
    surface_at_deposition = readout_surface[
        int(np.abs(Z0_GRID - primary.theta[0]).argmin()), int(np.abs(BETA_GRID - primary.theta[1]).argmin())
    ]
    results["F_readout"] = {
        "independent_fit": fit_summary(readout_fit, readout),
        "independent_fit_with_offset": fit_summary(readout_offset, readout),
        "with_deposition_parameters": fit_summary(fit_with_deposition, readout),
        "bootstrap": bootstrap_summary(readout_boot, ["z0", "beta"]),
        "surface": surface_summary(readout_surface, readout_fit),
        "difference_readout_minus_deposition": {
            "z0": float(readout_fit.theta[0] - primary.theta[0]),
            "beta": float(readout_fit.theta[1] - primary.theta[1]),
            "z0_in_readout_bootstrap_sd": float((readout_fit.theta[0] - primary.theta[0]) / readout_sd[0]),
            "beta_in_readout_bootstrap_sd": float((readout_fit.theta[1] - primary.theta[1]) / readout_sd[1]),
        },
        "deposition_point_delta_chi2_scaled_on_readout_surface": float(
            (surface_at_deposition - readout_surface.min()) / readout_fit.scale
        ),
        "deposition_point_inside_readout_95pct_region": bool(
            (surface_at_deposition - readout_surface.min()) / readout_fit.scale <= CONTOUR_LEVELS["95%"]
        ),
    }
    return results, artefacts


def write_tables(results: dict[str, Any], directory: Path, artefacts: dict[str, Any]) -> None:
    with (directory / "depth_origin_calibration_fits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "free", "z0", "beta", "delta", "chi2", "dof", "chi2_per_dof", "aic_scaled"])
        for name, fit in results["A_fits"].items():
            p = fit["parameters"]
            writer.writerow(
                [name, "+".join(fit["free"]), p["z0"], p["beta"], p["delta"], fit["chi2"], fit["dof"],
                 fit["chi2_per_dof"], fit["aic_scaled"]]
            )
    surface = artefacts["surface"]
    with (directory / "depth_origin_calibration_surface.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["z0", "beta", "chi2"])
        for i, z0 in enumerate(Z0_GRID):
            for j, beta in enumerate(BETA_GRID):
                writer.writerow([float(z0), float(beta), float(surface[i, j])])
    with (directory / "depth_origin_calibration_structure.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["estimator", "energy_gev", "mean_ln_t", "sd_ln_t", "skew_ln_t", "excess_kurtosis_ln_t",
             "mean_ln_alpha", "sd_ln_alpha", "skew_ln_alpha", "pearson", "spearman", "beta_median",
             "beta_sd", "sd_ln_t_over_gp_naive", "sd_ln_t_over_gp_covariant", "sd_ln_alpha_over_gp",
             "rho_minus_gp"]
        )
        for estimator in ("least_squares", "cumulative"):
            for energy, block in results["D_stochastic_structure"][estimator].items():
                r = block["ratios"]
                writer.writerow(
                    [estimator, energy, block["ln_t"]["mean"], block["ln_t"]["sd"], block["ln_t"]["skewness"],
                     block["ln_t"]["excess_kurtosis"], block["ln_alpha"]["mean"], block["ln_alpha"]["sd"],
                     block["ln_alpha"]["skewness"], block["joint"]["pearson"], block["joint"]["spearman"],
                     block["beta"]["median"], block["beta"]["sd"], r["sigma_ln_t_over_gp_naive"],
                     r["sigma_ln_t_over_gp_covariant"], r["sigma_ln_alpha_over_gp"], r["rho_minus_gp"]]
                )
    scan = results.get("D_skewness_versus_origin")
    if scan:
        with (directory / "depth_origin_calibration_skewness_versus_origin.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            columns = ["origin_x0", "mean_ln_t", "sd_ln_t", "sd_ln_alpha", "skewness_ln_t", "skewness_ln_alpha",
                       "excess_kurtosis_ln_t", "pearson", "beta_median"]
            writer.writerow(["energy_gev", *columns])
            for energy, rows in scan.items():
                if energy == "origins_x0":
                    continue
                for row in rows:
                    writer.writerow([energy, *[row[c] for c in columns]])
    with (directory / "depth_origin_calibration_residuals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "energy_gev", "layer", "relative_residual", "pull"])
        for name, fit in results["A_fits"].items():
            for k, energy in enumerate(fit["energies_gev"]):
                for layer in range(18):
                    writer.writerow(
                        [name, energy, layer, fit["relative_residual_by_layer"][k][layer],
                         fit["pull_by_layer"][k][layer]]
                    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=CALIBRATION_JSON)
    parser.add_argument("--plots", type=Path, default=CALIBRATION_PLOTS)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--no-skewness-scan", action="store_true")
    parser.add_argument(
        "--model-families",
        action="store_true",
        help="only fit the beta / depth-law / fluctuation model forms and write them next to the main results",
    )
    args = parser.parse_args(argv)
    if args.model_families:
        bounds = layer_bounds_x0()
        loaded = {float(e): load_energy(e) for e in ENERGIES_GEV}
        document = {
            "status": "ANALYSIS ONLY: the model forms that move the fitted depth origin (exposed electrons).",
            "deposition": model_family_fits(make_targets(loaded, "deposition", bounds)),
            "readout": model_family_fits(make_targets(loaded, "readout", bounds), amplitude="free"),
        }
        out = args.out.with_name("depth_origin_calibration_model_families.json")
        out.write_text(json.dumps(document, indent=2), encoding="utf-8")
        print(out)
        return
    results, artefacts = analyse(n_boot=args.bootstrap, skewness=not args.no_skewness_scan)
    document = {
        "status": (
            "ANALYSIS ONLY (depth-origin calibration) on the EXPOSED Geant4 electrons (development / calibration data). No "
            "generator was written or changed; no sealed data was read. Not a validation."
        ),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gate": GATE,
        **_clean(results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2), encoding="utf-8")
    write_tables(results, args.out.parent, artefacts)
    if not args.no_plots:
        from ams_ecal.electron_studies.em_depth_origin_calibration_plots import (
            make_plots,
        )

        make_plots(results, artefacts, args.plots)
    print(args.out)


if __name__ == "__main__":
    main()
