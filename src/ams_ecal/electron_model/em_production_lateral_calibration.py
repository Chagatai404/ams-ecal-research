"""Calibration of the ``em_production`` lateral slice (Slice 2) on the exposed baseline Geant4 electrons.

Decision record ``research/DECISIONS.md`` (2026-10-07 rows), note ``research/plans/2026-10-07_em_lateral_structure_note.md``.
The sealed electron sets are never read. Slice 1's longitudinal artifact is fixed; only the lateral constants move.

Two steps, each with a different target (the mean profile does not depend on the spot numbers, so they separate):

1. THE RADIAL LAW (12 constants of the Grindhammer-Peters structure, R_M = 18 mm). Target: the mean cell-fraction profile
   of every (energy, layer), as the cumulative profile over the 72 cells with equal weights (a shape-balanced residual: an
   error-weighted fit with tiny statistical errors fits only the central cells). Model: the ensemble mean over a fixed
   Sobol quadrature of Slice 1's draws, so each event's tau_i is part of the model and the energy dependence is a
   consequence, not an extra fit. Start: the published constants (lead, E in GeV).
2. THE SPOT STRUCTURE (4 constants: amplitude and energy exponent of the spot number, scale and shape of its depth
   density). Target: the measured relative variance of the central-three-cell fraction after removing its dependence on
   the event's own centre of gravity (``em_lateral_structure_study.spot_number``), reproduced with the SAME estimator on
   generated events (common random numbers, Nelder-Mead on the log ratios of the spot numbers).

Command: ``python -m ams_ecal.electron_model.em_production_lateral_calibration``
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.special import ndtri
from scipy.stats import qmc

from ams_ecal.electron_model import em_production_lateral as lateral
from ams_ecal.electron_model.em_production import PROJECT_ROOT
from ams_ecal.electron_studies.em_lateral_structure_study import (
    AXIS_CELL,
    DATA,
    ENERGIES_GEV,
    spot_number,
    total_variation,
)

RESULTS_JSON = PROJECT_ROOT / "results" / "em_generator" / "em_production_lateral_calibration.json"
SAMPLE_CONFIG = PROJECT_ROOT / "configs" / "geant4_electron_sample.yaml"
QUADRATURE_LOG2 = 8  # 256 points: a fixed Sobol set of (z_t, z_perp, entry_x, entry_y)
SPOT_EVENTS = 1500
SPOT_SEED = 20261007
SPOT_FIT_FIRST_LAYER = 2  # layers 0-1 hold an energy-independent spot number that the approved structure cannot reach
CENTRAL = slice(AXIS_CELL - 1, AXIS_CELL + 2)
# the outskirts: every cell at least five cells from the axis cell (the same split as the lateral structure study)
OUTER = np.r_[0 : AXIS_CELL - 4, AXIS_CELL + 5 : 72]
E1_RESULTS_JSON = PROJECT_ROOT / "results" / "em_generator" / "em_production_lateral_e1_calibration.json"
E1_ARTIFACT = PROJECT_ROOT / "data" / "calibration" / "em_production" / "gp_lateral_e1_v1" / "parameters.json"
E1_NAMES = (*lateral.SPOT_NAMES, "spot_tail_refinement", "front_spot_limit")
TAIL_RESULTS_JSON = PROJECT_ROOT / "results" / "em_generator" / "em_production_lateral_tail_calibration.json"
TAIL_ARTIFACT = PROJECT_ROOT / "data" / "calibration" / "em_production" / "gp_lateral_tail_v1" / "parameters.json"
# Tail-sensitive radial calibration: the relative error of the layer's second moment is added to the cumulative residual
# with this FIXED weight (a 10% moment error counts like a cumulative residual of 0.002, the size of the fit's own rms).
# It is stated here once, before the candidate was run, and is not tuned.
MOMENT_WEIGHT = 0.02

# Start of the radial fit: the converged solution of a first run that began at the published constants of Grindhammer
# and Peters Appendix A.1.3 (homogeneous, Z = 82; core 0.0251 + 0.00319 ln E + 0.085 tau, tail 0.4056 / 0.645 / -2.59 /
# 0.3585 + 0.0421 ln E, weight 2.555 / 0.554 / 1.313 - 0.0686 ln E), widened to a tail-decay bound of -60. The fit
# restarts from it at three scalings, so the result does not depend on the start; spots: order of magnitude from the study.
START = lateral.LateralParameters(
    core_intercept=0.002,
    core_log_energy_slope=-0.0051,
    core_depth_slope=0.1761,
    tail_scale=0.6845,
    tail_centre=0.1281,
    tail_decay_rate=-14.2926,
    tail_growth_intercept=0.4413,
    tail_growth_log_energy_slope=-0.0633,
    weight_amplitude=2.2903,
    weight_centre=0.7046,
    weight_width_intercept=1.5458,
    weight_width_log_energy_slope=-0.1884,
    spot_amplitude=100.0,
    spot_energy_exponent=0.85,
    spot_depth_scale=0.9,
    spot_shape_scale=0.9,
)
RADIAL_BOUNDS = {
    "core_intercept": (-0.5, 0.5),
    "core_log_energy_slope": (-0.1, 0.1),
    "core_depth_slope": (0.0, 1.0),
    "tail_scale": (0.02, 3.0),
    "tail_centre": (0.1, 2.0),
    "tail_decay_rate": (-60.0, -0.1),
    "tail_growth_intercept": (-1.0, 3.0),
    "tail_growth_log_energy_slope": (-0.5, 0.5),
    "weight_amplitude": (0.2, 6.0),
    "weight_centre": (-1.0, 2.0),
    "weight_width_intercept": (0.2, 5.0),
    "weight_width_log_energy_slope": (-1.0, 1.0),
    "tail_exponent": (1.0, 4.0),
}
# the faster-falling tail: the twelve constants plus the exponent of the tail component's survival function
FASTER_TAIL_NAMES = (*lateral.RADIAL_NAMES, "tail_exponent")
FASTER_TAIL_START_EXPONENT = 1.5


def load_observed(
    energies: tuple[float, ...] = ENERGIES_GEV, max_events: int | None = None
) -> dict[float, dict[str, Any]]:
    """Per energy: the mean cell profile, and the measured conditioned spot numbers (exposed baseline only)."""

    observed: dict[float, dict[str, Any]] = {}
    for energy in energies:
        grid = np.load(DATA / f"E{energy:g}GeV" / "events.npz")["deposit_grid_mev"].astype(float)
        if max_events is not None:
            grid = grid[:max_events]
        layer_energy = grid.sum(axis=2)
        layers = grid.shape[1]
        cog = (layer_energy * (np.arange(layers) + 0.5)).sum(axis=1) / layer_energy.sum(axis=1)
        spots, outer_spots = [], []
        for layer in range(layers):
            usable = layer_energy[:, layer] > 0
            fraction = grid[usable, layer, CENTRAL].sum(axis=1) / layer_energy[usable, layer]
            spots.append(spot_number(fraction, cog[usable])["spot_number_conditioned"])
            outer = grid[usable, layer][:, OUTER].sum(axis=1) / layer_energy[usable, layer]
            outer_spots.append(spot_number(outer, cog[usable])["spot_number_conditioned"])
        observed[energy] = {
            "events": len(grid),
            "profile": grid.sum(axis=0) / layer_energy.sum(axis=0)[:, None],
            "spot_number_conditioned": np.array(spots),
            "spot_number_outer": np.array(outer_spots),
        }
    return observed


def quadrature(
    model: lateral.EMProductionLateralModel,
) -> tuple[np.ndarray, np.ndarray]:
    """A fixed Sobol set: ``(z, axis_mm)`` with ``z`` of shape ``(n, 2)`` and ``axis_mm`` of shape ``(n, layers)``."""

    points = qmc.Sobol(d=4, scramble=True, seed=7).random_base2(QUADRATURE_LOG2)
    z = ndtri(np.clip(points[:, :2], 1e-9, 1 - 1e-9))
    lower = model.cell_edges_mm[AXIS_CELL]
    width = model.geometry.cell_pitch_mm
    entry_x, entry_y = lower + width * points[:, 2], lower + width * points[:, 3]
    return z, model.axis_for_entry(entry_x, entry_y)


def predicted_profile(
    model: lateral.EMProductionLateralModel,
    energy_gev: float,
    z: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    fractions = model.mean_cell_fractions(1000.0 * energy_gev, z, axis)
    return fractions / fractions.sum(axis=1, keepdims=True)


def fit_radial_law(
    template: lateral.EMProductionLateralModel,
    observed: dict[float, dict[str, Any]],
    max_nfev: int = 200,
    moments: dict[float, np.ndarray] | None = None,
    moment_weight: float = 0.0,
    names: tuple[str, ...] = lateral.RADIAL_NAMES,
) -> tuple[lateral.LateralParameters, dict[str, Any]]:
    """The radial constants (twelve; thirteen with the faster-tail exponent). With ``moments`` and a weight the
    relative error of the layer's second moment (the quantity the width is built from) is added to the cumulative
    residual; no parameter is added by that."""

    z, axis = quadrature(template)
    lower = np.array([RADIAL_BOUNDS[n][0] for n in names])
    upper = np.array([RADIAL_BOUNDS[n][1] for n in names])
    targets = {e: np.cumsum(o["profile"], axis=1) for e, o in observed.items()}
    use_moments = moments is not None and moment_weight > 0.0

    def residual(values: np.ndarray) -> np.ndarray:
        model = replace(template, parameters=template.parameters.with_values(names, values))
        parts = [(np.cumsum(predicted_profile(model, e, z, axis), axis=1) - targets[e]).ravel() for e in observed]
        if use_moments:
            parts += [
                moment_weight * (model.mean_second_moments_mm2(1000.0 * e, z, axis) / moments[e] - 1.0)
                for e in observed
            ]
        return np.concatenate(parts)

    start = template.parameters.vector(names)
    best = None
    for factor in (1.0, 0.8, 1.25):
        trial = np.clip(start * factor, lower + 1e-9, upper - 1e-9)
        result = least_squares(residual, trial, bounds=(lower, upper), x_scale="jac", max_nfev=max_nfev)
        if best is None or result.cost < best.cost:
            best = result
    parameters = template.parameters.with_values(names, best.x)
    at_bound = [
        n for n, v, lo, hi in zip(names, best.x, lower, upper, strict=True) if min(v - lo, hi - v) < 1e-3 * (hi - lo)
    ]
    return parameters, {
        "free_parameters": list(names),
        "cost": float(best.cost),
        "root_mean_square_cumulative_residual": float(np.sqrt(2 * best.cost / best.fun.size)),
        "evaluations": int(best.nfev),
        "parameters_at_bound": at_bound,
        "quadrature_points": len(z),
        "second_moment_weight": moment_weight if use_moments else 0.0,
    }


def observed_second_moments(
    model: lateral.EMProductionLateralModel,
    energies: tuple[float, ...] = ENERGIES_GEV,
    max_events: int | None = None,
) -> dict[float, np.ndarray]:
    """Per energy, the energy-weighted mean second moment of the cell offsets about the track, per layer, in mm^2.

    ``sum_events sum_cells E_cell (x_cell - axis)^2 / sum_events E_layer``: the layer's mean contribution to the
    event width squared (exposed baseline electrons only).
    """

    edges = model.cell_edges_mm
    centres = 0.5 * (edges[:-1] + edges[1:])
    out = {}
    for energy in energies:
        arrays = np.load(DATA / f"E{energy:g}GeV" / "events.npz")
        grid = arrays["deposit_grid_mev"].astype(float)
        entry_x, entry_y = arrays["entry_x_mm"], arrays["entry_y_mm"]
        if max_events is not None:
            grid, entry_x, entry_y = grid[:max_events], entry_x[:max_events], entry_y[:max_events]
        offsets = centres[None, None, :] - model.axis_for_entry(entry_x, entry_y)[:, :, None]
        out[energy] = (grid * offsets**2).sum(axis=(0, 2)) / grid.sum(axis=(0, 2))
    return out


def second_moment_errors(
    model: lateral.EMProductionLateralModel, moments: dict[float, np.ndarray]
) -> dict[str, dict[str, float]]:
    """Relative error of the model's layer second moments against Geant4: layers 3-17 mean and worst, layers 0-2 worst."""

    z, axis = quadrature(model)
    out = {}
    for energy, observed_moment in moments.items():
        relative = model.mean_second_moments_mm2(1000.0 * energy, z, axis) / observed_moment - 1.0
        out[f"{energy:g}"] = {
            "layers_0_2_max_abs": float(np.abs(relative[:3]).max()),
            "layers_3_17_max_abs": float(np.abs(relative[3:]).max()),
            "layers_3_17_mean": float(relative[3:].mean()),
        }
    return out


def profile_errors(
    model: lateral.EMProductionLateralModel, observed: dict[float, dict[str, Any]]
) -> dict[str, dict[str, float]]:
    z, axis = quadrature(model)
    out = {}
    for energy, o in observed.items():
        predicted = predicted_profile(model, energy, z, axis)
        tv = np.array([100 * total_variation(predicted[layer], o["profile"][layer]) for layer in range(len(predicted))])
        out[f"{energy:g}"] = {
            "layers_0_2_max": float(tv[:3].max()),
            "layers_3_17_max": float(tv[3:].max()),
            "layers_3_17_mean": float(tv[3:].mean()),
        }
    return out


def simulated_spot_statistics(
    model: lateral.EMProductionLateralModel,
    prepared: dict[float, lateral.PreparedBatch],
    seed: int,
) -> dict[float, dict[str, np.ndarray]]:
    """Conditioned spot numbers of the central three cells and of the outskirts, with the measured estimator."""

    rng = np.random.default_rng(seed)
    out = {}
    for energy, batch in prepared.items():
        drawn = model.draw_cells(batch, rng)
        layer_energy = drawn.cell_energy_mev.sum(axis=2)
        layers = layer_energy.shape[1]
        cog = (layer_energy * (np.arange(layers) + 0.5)).sum(axis=1) / layer_energy.sum(axis=1)
        central, outer = [], []
        for layer in range(layers):
            usable = layer_energy[:, layer] > 0
            fraction = drawn.cell_energy_mev[usable, layer, CENTRAL].sum(axis=1) / layer_energy[usable, layer]
            central.append(spot_number(fraction, cog[usable])["spot_number_conditioned"])
            far = drawn.cell_energy_mev[usable, layer][:, OUTER].sum(axis=1) / layer_energy[usable, layer]
            outer.append(_safe_spot_number(far, cog[usable]))
        out[energy] = {"central": np.array(central), "outer": np.array(outer)}
    return out


def _safe_spot_number(fraction: np.ndarray, cog: np.ndarray) -> float:
    """The conditioned spot number, with a large finite value where the fraction does not vary at all."""

    if fraction.std() == 0.0:
        return 1e9
    return spot_number(fraction, cog)["spot_number_conditioned"]


def simulated_spot_numbers(
    model: lateral.EMProductionLateralModel,
    prepared: dict[float, lateral.PreparedBatch],
    seed: int,
) -> dict[float, np.ndarray]:
    return {e: s["central"] for e, s in simulated_spot_statistics(model, prepared, seed).items()}


def prepare_spot_batches(
    model: lateral.EMProductionLateralModel,
    energies: tuple[float, ...],
    events: int = SPOT_EVENTS,
    seed: int = SPOT_SEED,
) -> dict[float, lateral.PreparedBatch]:
    rng = np.random.default_rng(seed)
    lower = model.cell_edges_mm[AXIS_CELL]
    width = model.geometry.cell_pitch_mm
    prepared = {}
    for energy in energies:
        entry = lower + width * rng.random((events, 2))
        axis = model.axis_for_entry(entry[:, 0], entry[:, 1])
        prepared[energy] = model.prepare_batch(1000.0 * energy, axis, rng)
    return prepared


def fit_spot_structure(
    template: lateral.EMProductionLateralModel,
    observed: dict[float, dict[str, Any]],
    max_iterations: int = 400,
) -> tuple[lateral.LateralParameters, dict[str, Any]]:
    prepared = prepare_spot_batches(template, tuple(observed))
    target = {e: np.log(o["spot_number_conditioned"]) for e, o in observed.items()}
    fit_layers = slice(SPOT_FIT_FIRST_LAYER, None)

    def unpack(q: np.ndarray) -> lateral.LateralParameters:
        return template.parameters.with_values(lateral.SPOT_NAMES, [np.exp(q[0]), q[1], np.exp(q[2]), np.exp(q[3])])

    def objective(q: np.ndarray) -> float:
        try:
            model = replace(template, parameters=unpack(q))
            simulated = simulated_spot_numbers(model, prepared, SPOT_SEED + 1)
        except ValueError:
            return 1e6
        return float(
            sum(np.mean((np.log(simulated[e][fit_layers]) - target[e][fit_layers]) ** 2) for e in observed)
            / len(observed)
        )

    p = template.parameters
    start = np.array(
        [
            np.log(p.spot_amplitude),
            p.spot_energy_exponent,
            np.log(p.spot_depth_scale),
            np.log(p.spot_shape_scale),
        ]
    )
    result = minimize(
        objective,
        start,
        method="Nelder-Mead",
        options={"maxiter": max_iterations, "xatol": 1e-3, "fatol": 1e-5},
    )
    parameters = unpack(result.x)
    fitted = simulated_spot_numbers(replace(template, parameters=parameters), prepared, SPOT_SEED + 2)
    return parameters, {
        "free_parameters": list(lateral.SPOT_NAMES),
        "fit_layers": f"{SPOT_FIT_FIRST_LAYER}-{len(next(iter(target.values()))) - 1}",
        "objective_mean_squared_log_ratio": float(result.fun),
        "root_mean_square_log_ratio": float(np.sqrt(result.fun)),
        "root_mean_square_log_ratio_all_layers": float(
            np.sqrt(np.mean([np.mean((np.log(fitted[e]) - target[e]) ** 2) for e in observed]))
        ),
        "front_layers_note": (
            "layers 0-1 are not fitted: the measured conditioned spot number there (about 13 and 24) is the same at "
            "all four energies, which a gamma depth density scaled by E^b cannot reach; reported as a limitation"
        ),
        "evaluations": int(result.nfev),
        "converged": bool(result.success),
        "measured": {f"{e:g}": observed[e]["spot_number_conditioned"].tolist() for e in observed},
        "generated": {f"{e:g}": fitted[e].tolist() for e in observed},
        "events_per_energy": SPOT_EVENTS,
    }


def fit_spot_extension(
    template: lateral.EMProductionLateralModel,
    observed: dict[float, dict[str, Any]],
    max_iterations: int = 600,
) -> tuple[lateral.LateralParameters, dict[str, Any]]:
    """Extension E1: the four spot constants of Slice 2 plus ``kappa`` and ``n0``; the radial law stays fixed.

    Targets: the conditioned spot number of the central three cells AND of the outskirts (cells at least five from the
    axis), every layer, same estimator on generated events. The outskirts statistic is a fluctuation measure at a
    second radius; the hit-cell count and the width are NOT targets, so they remain hold-out tests of the extension.
    """

    prepared = prepare_spot_batches(template, tuple(observed))
    central_target = {e: np.log(o["spot_number_conditioned"]) for e, o in observed.items()}
    outer_target = {e: np.log(o["spot_number_outer"]) for e, o in observed.items()}

    def unpack(q: np.ndarray) -> lateral.LateralParameters:
        values = [np.exp(q[0]), q[1], np.exp(q[2]), np.exp(q[3]), 1.0 + np.exp(q[4]), np.exp(q[5])]
        return template.parameters.with_values(E1_NAMES, values)

    def terms(model: lateral.EMProductionLateralModel, seed: int) -> dict[float, dict[str, np.ndarray]]:
        return simulated_spot_statistics(model, prepared, seed)

    def objective(q: np.ndarray) -> float:
        try:
            simulated = terms(replace(template, parameters=unpack(q)), SPOT_SEED + 1)
        except ValueError:
            return 1e6
        total = 0.0
        for e in observed:
            total += np.mean((np.log(simulated[e]["central"]) - central_target[e]) ** 2)
            total += np.mean((np.log(simulated[e]["outer"]) - outer_target[e]) ** 2)
        return float(total / (2 * len(observed)))

    p = template.parameters
    start = np.array(
        [
            np.log(p.spot_amplitude),
            p.spot_energy_exponent,
            np.log(p.spot_depth_scale),
            np.log(p.spot_shape_scale),
            np.log(2.0),  # kappa = 3
            np.log(15.0),  # n0 = 15
        ]
    )
    # Slice 2 itself (kappa = 1, no front cap) on the same two-statistic objective
    base = terms(template, SPOT_SEED + 1)
    base_objective = float(
        sum(
            np.mean((np.log(base[e]["central"]) - central_target[e]) ** 2)
            + np.mean((np.log(base[e]["outer"]) - outer_target[e]) ** 2)
            for e in observed
        )
        / (2 * len(observed))
    )
    result = minimize(
        objective,
        start,
        method="Nelder-Mead",
        options={"maxiter": max_iterations, "xatol": 1e-3, "fatol": 1e-5},
    )
    parameters = unpack(result.x)
    fitted = terms(replace(template, parameters=parameters), SPOT_SEED + 2)

    def rms(key: str, target: dict[float, np.ndarray], layers: slice) -> float:
        return float(
            np.sqrt(np.mean([np.mean((np.log(fitted[e][key][layers]) - target[e][layers]) ** 2) for e in observed]))
        )

    return parameters, {
        "free_parameters": list(E1_NAMES),
        "objective_mean_squared_log_ratio": float(result.fun),
        "slice2_objective_on_the_same_statistics": float(base_objective),
        "root_mean_square_log_ratio": {
            "central_all_layers": rms("central", central_target, slice(0, None)),
            "central_layers_2_17": rms("central", central_target, slice(2, None)),
            "outer_all_layers": rms("outer", outer_target, slice(0, None)),
            "outer_layers_2_17": rms("outer", outer_target, slice(2, None)),
        },
        "extension_values": {
            "spot_tail_refinement": parameters.spot_tail_refinement,
            "front_spot_limit": parameters.front_spot_limit,
        },
        "extension_collapsed": bool(parameters.spot_tail_refinement < 1.001 or parameters.front_spot_limit > 1e4),
        "evaluations": int(result.nfev),
        "converged": bool(result.success),
        "measured_central": {f"{e:g}": observed[e]["spot_number_conditioned"].tolist() for e in observed},
        "measured_outer": {f"{e:g}": observed[e]["spot_number_outer"].tolist() for e in observed},
        "generated_central": {f"{e:g}": fitted[e]["central"].tolist() for e in observed},
        "generated_outer": {f"{e:g}": fitted[e]["outer"].tolist() for e in observed},
        "events_per_energy": SPOT_EVENTS,
        "targets_not_used": ["n_hit_cells", "width_mm"],
    }


def calibrate_extension(
    observed: dict[float, dict[str, Any]],
    base: lateral.EMProductionLateralModel,
    max_iterations: int = 600,
) -> tuple[lateral.LateralParameters, dict[str, Any], lateral.EMProductionLateralModel]:
    """Fit extension E1 on top of a calibrated Slice 2 model (the radial law is not refitted)."""

    parameters, spot_record = fit_spot_extension(base, observed, max_iterations)
    model = replace(base, parameters=parameters)
    record = {
        "status": "extension E1 on a calibrated candidate; development; not validated",
        "sealed_data_read": False,
        "data": sample_provenance(observed),
        "method": {
            "moliere_radius_mm": parameters.moliere_radius_mm,
            "radial_law": "fixed: taken unchanged from the Slice 2 artifact",
            "extension": "tail spots carry 1 / kappa of the core spot energy; spot number capped by n0 * 2^layer",
            "spot_targets": "conditioned spot number of the central three cells and of the outskirts, all layers",
        },
        "spot_fit": spot_record,
        "profile_errors_points": profile_errors(model, observed),
        "longitudinal_parameters_sha256": base.longitudinal_sha256,
    }
    return parameters, record, model


def sample_provenance(observed: dict[float, dict[str, Any]]) -> dict[str, Any]:
    return {
        "directory": str(DATA.relative_to(PROJECT_ROOT)),
        "events": {f"{e:g}": o["events"] for e, o in observed.items()},
        "configuration_sha256": hashlib.sha256(SAMPLE_CONFIG.read_bytes()).hexdigest(),
        "sample": "baseline",
    }


def calibrate(
    observed: dict[float, dict[str, Any]],
    template: lateral.EMProductionLateralModel | None = None,
    max_nfev: int = 200,
    max_spot_iterations: int = 400,
    moments: dict[float, np.ndarray] | None = None,
    moment_weight: float = 0.0,
    radial_names: tuple[str, ...] = lateral.RADIAL_NAMES,
) -> tuple[lateral.LateralParameters, dict[str, Any], lateral.EMProductionLateralModel]:
    if template is None:
        template = lateral.build_model(START)
    radial, radial_record = fit_radial_law(template, observed, max_nfev, moments, moment_weight, radial_names)
    radial_model = replace(template, parameters=radial)
    final, spot_record = fit_spot_structure(radial_model, observed, max_spot_iterations)
    model = replace(template, parameters=final)
    record = {
        "status": "calibrated candidate for development on exposed Geant4 electrons; not validated",
        "sealed_data_read": False,
        "data": sample_provenance(observed),
        "method": {
            "moliere_radius_mm": final.moliere_radius_mm,
            "radial_structure": "Grindhammer and Peters Eqs. 24-26 with tau_i from Eq. 34",
            "radial_target": "cumulative mean cell profile per (energy, layer), equal weights",
            "spot_target": "conditioned relative variance of the central three cells, same estimator on generated events",
            "start": "converged first run from the published constants (Z = 82, E in GeV); see START",
        },
        "radial_fit": radial_record,
        "spot_fit": spot_record,
        "profile_errors_points": profile_errors(model, observed),
        "longitudinal_parameters_sha256": template.longitudinal_sha256,
    }
    if moments is not None:
        record["second_moment_relative_errors"] = second_moment_errors(model, moments)
        record["method"]["radial_target"] = (
            "cumulative mean cell profile per (energy, layer), equal weights, PLUS the relative error of the layer "
            f"second moment with the fixed weight {moment_weight} (tail-sensitive; no parameter added)"
        )
    return final, record, model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, default=lateral.DEFAULT_ARTIFACT)
    parser.add_argument("--results", type=Path, default=RESULTS_JSON)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument(
        "--extension",
        action="store_true",
        help="fit extension E1 on top of the Slice 2 artifact into its own artifact and results file",
    )
    parser.add_argument(
        "--tail-sensitive",
        action="store_true",
        help="refit the twelve radial constants with the layer second moment as a second target (own artifact)",
    )
    parser.add_argument(
        "--faster-tail",
        action="store_true",
        help="thirteen radial constants: add the tail exponent (needs --tail-sensitive)",
    )
    parser.add_argument("--moment-weight", type=float, default=MOMENT_WEIGHT, help="weight of the second-moment target")
    parser.add_argument("--tag", default="tail", help="name of the tail-sensitive variant (its artifact and results)")
    args = parser.parse_args(argv)
    observed = load_observed(max_events=args.max_events)
    if args.tail_sensitive:
        artifact = TAIL_ARTIFACT.parents[1] / f"gp_lateral_{args.tag}_v1" / "parameters.json"
        results_path = TAIL_RESULTS_JSON.parent / f"em_production_lateral_{args.tag}_calibration.json"
        template = lateral.build_model(START)
        names = FASTER_TAIL_NAMES if args.faster_tail else lateral.RADIAL_NAMES
        if args.faster_tail:
            template = replace(
                template, parameters=replace(template.parameters, tail_exponent=FASTER_TAIL_START_EXPONENT)
            )
        moments = observed_second_moments(template, tuple(observed), args.max_events)
        parameters, record, model = calibrate(
            observed, template, moments=moments, moment_weight=args.moment_weight, radial_names=names
        )
        lateral.write_parameters_artifact(
            parameters,
            artifact,
            record,
            "ams_ecal.electron_model.em_production_lateral_calibration --tail-sensitive",
            model.longitudinal_sha256,
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps({"parameters": parameters.as_dict(), **record}, indent=2), encoding="utf-8")
        print(json.dumps(record["profile_errors_points"], indent=2))
        print(json.dumps(record["second_moment_relative_errors"], indent=2))
        print("radial at bound:", record["radial_fit"]["parameters_at_bound"])
        print("spot rms log ratio", record["spot_fit"]["root_mean_square_log_ratio"])
        return
    if args.extension:
        base = lateral.load_em_production_lateral(args.artifact)
        parameters, record, model = calibrate_extension(observed, base)
        lateral.write_parameters_artifact(
            parameters,
            E1_ARTIFACT,
            record,
            "ams_ecal.electron_model.em_production_lateral_calibration --extension",
            model.longitudinal_sha256,
        )
        E1_RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
        E1_RESULTS_JSON.write_text(
            json.dumps({"parameters": parameters.as_dict(), **record}, indent=2), encoding="utf-8"
        )
        print(json.dumps(record["spot_fit"]["extension_values"], indent=2))
        print(json.dumps(record["spot_fit"]["root_mean_square_log_ratio"], indent=2))
        print(
            "objective",
            record["spot_fit"]["objective_mean_squared_log_ratio"],
            "against Slice 2",
            record["spot_fit"]["slice2_objective_on_the_same_statistics"],
        )
        return
    parameters, record, model = calibrate(observed)
    lateral.write_parameters_artifact(
        parameters,
        args.artifact,
        record,
        "ams_ecal.electron_model.em_production_lateral_calibration",
        model.longitudinal_sha256,
    )
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(
        json.dumps({"parameters": parameters.as_dict(), **record}, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "radial": record["radial_fit"],
                "profile_errors_points": record["profile_errors_points"],
            },
            indent=2,
        )
    )
    print("spot rms log ratio", record["spot_fit"]["root_mean_square_log_ratio"])


if __name__ == "__main__":
    main()
