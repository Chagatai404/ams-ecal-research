"""Calibration of the ``em_production`` Slice 1 parameters on the exposed Geant4 electrons.

Decision record ``research/DECISIONS.md`` (DEC-014, 2026-10-06 rows), note
``research/plans/2026-10-06_em_production_slice1_note.md``. The calibration data is the EXPOSED baseline electron
sample (``data/geant4_electron_sample/baseline``: 1000 events at each of 10, 20, 50, 100 GeV, ``deposition``
representation). The sealed electron set is not read. The extended-depth sample
(``data/geant4_electron_extended``) is NOT used here: it is the DEVELOPMENT HOLD-OUT of the development check
(earlier analyses looked at it descriptively, so it is a hold-out for these parameters, not a validation).

Two blocks of parameters, fitted in turn and iterated until the origin is stable because they depend on each other:

1. FLUCTUATION parameters (``rho`` as a line in ln y, the ln T skewness). Measured from per-event gamma fits of the
   Geant4 electrons at the current origin, the same estimator that later checks the generated events. The widths are
   NOT fitted: they are the Grindhammer-Peters sampling-set widths with the covariant factor (approved in principle).
2. CENTRAL parameters (origin ``z0``, ``beta``, the in-prefix tail weight ``w_0``, ``w_1`` and shape ``a_2``). Fitted by
   least squares of the generator's ENSEMBLE-MEAN layer fractions (a fixed 2048-point scrambled Sobol set of the two
   normals, so the objective is smooth and reproducible) to the Geant4 mean layer fractions, residuals in standard
   errors of the Geant4 means. ``delta = -0.5`` and the tail rate ``1/3.6`` are fixed by the sources.

No parameter is tuned against a classifier or a dataset-level threshold. Uncertainties are scaled by the
chi-square per degree of freedom and are model-conditional. Words: calibrated, candidate, development.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from scipy.optimize import least_squares
from scipy.stats import qmc

from ams_ecal.electron_model.em_production import (
    DEFAULT_ARTIFACT,
    PROJECT_ROOT,
    EMProductionLongitudinalModel,
    EMProductionParameters,
    build_model,
    write_parameters_artifact,
)
from ams_ecal.electron_studies.em_longitudinal_structure_analysis import (
    fit_free_with_origin,
)

ENERGIES_GEV = (10.0, 20.0, 50.0, 100.0)
SAMPLE_DIR = PROJECT_ROOT / "data" / "geant4_electron_sample" / "baseline"
RESULTS_JSON = PROJECT_ROOT / "results" / "em_generator" / "em_production_calibration.json"
QUADRATURE_LOG2 = 11  # 2048 points
QUADRATURE_SEED = 20261006
MAX_ITERATIONS = 8
ORIGIN_CONVERGENCE_X0 = 0.03  # the loop stops when the origin moves by less than this between iterations

START_PARAMETERS = EMProductionParameters(
    origin_x0=-1.0,
    beta=0.6,
    tail_weight=0.1,
    tail_weight_log_energy_slope=0.0,
    tail_shape=4.0,
    rho_intercept=0.45,
    rho_log_slope=0.0,
    ln_t_skewness=0.8,
)
CENTRAL_NAMES = ("origin_x0", "beta", "tail_weight", "tail_weight_log_energy_slope", "tail_shape")
CENTRAL_LOWER = (-4.0, 0.3, 0.0, -0.1, 1.05)
CENTRAL_UPPER = (1.0, 1.2, 0.5, 0.1, 12.0)
CENTRAL_STARTS = (
    (-1.5, 0.7, 0.15, 0.03, 4.4),
    (-1.0, 0.57, 0.05, 0.0, 3.0),
    (0.0, 0.55, 0.06, -0.03, 2.1),
)
# The depth origin is a COORDINATE CONVENTION. With it free, the calibration has no fixed point: the correlation and
# the skewness measured at one origin send the central fit to the other basin (a two-cycle, z0 about +0.08 and about
# -1.65, recorded in the note). It is therefore fixed at the front face, the convention of the PDG review and of
# Grindhammer and Peters, where the measured beta (0.52-0.53) and rho (0.57-0.62) agree with those sources.
FRONT_FACE_ORIGIN_X0 = 0.0
# The mean leakage is a named target of the contract. Its materiality floor in the project's comparison metrics is
# 0.5 points of E (em_longitudinal_structure_analysis.NOISE_FLOORS); the contained fraction of every calibration
# energy must stay within CONTAINED_TOLERANCE (below that floor). Exceeding it is penalised steeply, inside the fit.
CONTAINED_TOLERANCE = 0.003
CONTAINED_PENALTY_SCALE = 0.0005  # one unit of residual per 0.05 points beyond the tolerance


def load_sample(
    energies: Sequence[float] = ENERGIES_GEV, directory: Path = SAMPLE_DIR, max_events: int | None = None
) -> dict[str, Any]:
    """Layer fractions (all-material deposit over E) per energy and the provenance of the batches."""

    fractions: dict[float, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    for energy in energies:
        batch = directory / f"E{energy:g}GeV"
        arrays = np.load(batch / "events.npz")
        grid = arrays["deposit_grid_mev"][:max_events]
        fractions[float(energy)] = grid.sum(axis=2).astype(float) / (1000.0 * energy)
        metadata = json.loads((batch / "metadata.json").read_text(encoding="utf-8"))
        provenance[f"{energy:g}"] = {
            "events": len(grid),
            "configuration_sha256": metadata.get("configuration_sha256"),
            "base_seed": metadata.get("base_seed"),
            "sample": metadata.get("sample"),
        }
    return {"fractions": fractions, "provenance": provenance}


def quadrature_normals() -> np.ndarray:
    """The fixed ``(2048, 2)`` set of standard-normal pairs of the ensemble-mean objective."""

    u = qmc.Sobol(d=2, scramble=True, seed=QUADRATURE_SEED).random_base2(QUADRATURE_LOG2)
    return stats.norm.ppf(np.clip(u, 1e-12, 1.0 - 1e-12))


def _model(
    parameters: EMProductionParameters, critical_energy_mev: float, bounds: np.ndarray
) -> EMProductionLongitudinalModel:
    return EMProductionLongitudinalModel(
        parameters=parameters,
        critical_energy_mev=critical_energy_mev,
        layer_bounds_x0=tuple((float(a), float(b)) for a, b in bounds),
    )


def measured_fluctuations(fractions: dict[float, np.ndarray], bounds: np.ndarray, origin: float) -> dict[str, Any]:
    """Per-event gamma fits at ``origin``: the moments the generated events are later checked against."""

    rows: dict[str, Any] = {}
    for energy, data in fractions.items():
        fits = np.array([fit_free_with_origin(f, bounds, origin)[0] for f in data])
        ln_t, ln_alpha = fits[:, 0], fits[:, 1]
        rows[f"{energy:g}"] = {
            "mean_ln_t": float(ln_t.mean()),
            "sd_ln_t": float(ln_t.std(ddof=1)),
            "skewness_ln_t": float(stats.skew(ln_t)),
            "mean_ln_alpha": float(ln_alpha.mean()),
            "sd_ln_alpha": float(ln_alpha.std(ddof=1)),
            "pearson": float(np.corrcoef(ln_t, ln_alpha)[0, 1]),
            "beta_median": float(np.median((np.exp(ln_alpha) - 1.0) / np.exp(ln_t))),
        }
    return rows


def fit_fluctuation_parameters(moments: dict[str, Any], critical_energy_mev: float) -> dict[str, float]:
    """The line ``rho(ln y)`` through the measured correlations and the mean ln T skewness."""

    energies = np.array([float(e) for e in moments])
    ln_y = np.log(1000.0 * energies / critical_energy_mev)
    pearson = np.array([moments[f"{e:g}"]["pearson"] for e in energies])
    slope, intercept = np.polyfit(ln_y, pearson, 1)
    skew = float(np.mean([moments[f"{e:g}"]["skewness_ln_t"] for e in energies]))
    return {
        "rho_intercept": float(intercept),
        "rho_log_slope": float(slope),
        "ln_t_skewness": float(np.clip(skew, -0.95, 0.95)),
    }


def fit_central_parameters(
    fractions: dict[float, np.ndarray],
    parameters: EMProductionParameters,
    critical_energy_mev: float,
    bounds: np.ndarray,
    normals: np.ndarray,
    fixed_origin: float | None = None,
) -> tuple[EMProductionParameters, dict[str, Any]]:
    """Least squares of the ensemble-mean layer fractions of the generator against the Geant4 means."""

    energies = list(fractions)
    free = [i for i, name in enumerate(CENTRAL_NAMES) if not (fixed_origin is not None and name == "origin_x0")]
    names = [CENTRAL_NAMES[i] for i in free]
    lower = np.array(CENTRAL_LOWER)[free]
    upper = np.array(CENTRAL_UPPER)[free]
    mean = np.array([fractions[e].mean(axis=0) for e in energies])
    se = np.array([fractions[e].std(axis=0, ddof=1) / np.sqrt(len(fractions[e])) for e in energies])
    contained = mean.sum(axis=1)

    def build(theta: np.ndarray) -> EMProductionParameters:
        values = dict(zip(names, (float(v) for v in theta), strict=True))
        if fixed_origin is not None:
            values["origin_x0"] = float(fixed_origin)
        return replace(parameters, **values)

    def residual(theta: np.ndarray) -> np.ndarray:
        try:
            model = _model(build(theta), critical_energy_mev, bounds)
            predicted = np.array([model.mean_layer_fractions(1000.0 * e, normals) for e in energies])
        except ValueError:
            return np.full(mean.size + len(energies), 1e3)
        excess = np.maximum(np.abs(predicted.sum(axis=1) - contained) - CONTAINED_TOLERANCE, 0.0)
        return np.concatenate([((predicted - mean) / se).ravel(), excess / CONTAINED_PENALTY_SCALE])

    best = None
    scales = np.array((1.0, 0.1, 0.1, 0.05, 1.0))[free]
    for start in CENTRAL_STARTS:
        result = least_squares(
            residual, np.array(start)[free], bounds=(lower, upper), x_scale=scales
        )
        if best is None or result.cost < best.cost:
            best = result
    assert best is not None
    layer_chi2 = float((best.fun[: mean.size] ** 2).sum())
    chi2 = layer_chi2  # the penalty is a constraint, not data: the chi-square and its scale are the layers'
    dof = mean.size - len(names)
    scale = chi2 / dof
    jacobian = best.jac[: mean.size]
    covariance = scale * np.linalg.pinv(jacobian.T @ jacobian)
    sd = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    fitted = build(best.x)
    model = _model(fitted, critical_energy_mev, bounds)
    predicted = np.array([model.mean_layer_fractions(1000.0 * e, normals) for e in energies])
    relative = (predicted - mean) / mean
    at_bound = [
        name
        for name, value, low, high in zip(names, best.x, lower, upper, strict=True)
        if abs(value - low) < 1e-6 or abs(value - high) < 1e-6
    ]
    diagnostics = {
        "chi2": chi2,
        "dof": dof,
        "chi2_per_dof": scale,
        "contained_tolerance": CONTAINED_TOLERANCE,
        "contained_difference_model_minus_geant4": (predicted.sum(axis=1) - contained).tolist(),
        "contained_penalty_active": bool(np.any(np.abs(predicted.sum(axis=1) - contained) > CONTAINED_TOLERANCE + 1e-9)),
        "free_parameters": names,
        "origin_fixed_at_x0": fixed_origin,
        "scaled_sd": dict(zip(names, sd.tolist(), strict=True)),
        "at_bound": at_bound,
        "energies_gev": energies,
        "rms_relative_layers_0_3": np.sqrt((relative[:, :4] ** 2).mean(axis=1)).tolist(),
        "rms_relative_layers_4_17": np.sqrt((relative[:, 4:] ** 2).mean(axis=1)).tolist(),
        "contained_fraction_model": predicted.sum(axis=1).tolist(),
        "contained_fraction_geant4": mean.sum(axis=1).tolist(),
        "relative_residual_by_layer": relative.tolist(),
    }
    return fitted, diagnostics


def calibrate(
    sample: dict[str, Any],
    max_iterations: int = MAX_ITERATIONS,
    fixed_origin: float | None = FRONT_FACE_ORIGIN_X0,
) -> tuple[EMProductionParameters, dict[str, Any]]:
    """Run the two blocks in turn until the origin is stable; return the parameters and the record.

    The correlation and skewness are measured with per-event fits AT THE CURRENT ORIGIN, and both depend on that
    origin (the depth-origin convention), so the origin of the central fit is fed back until it stops moving.
    ``converged`` in the record says whether that happened within ``max_iterations``. With ``fixed_origin`` (the
    default, the front face) the origin is a convention, the moments are measured there once, and the loop ends
    after one pass; ``fixed_origin=None`` runs the free-origin iteration, which is known to two-cycle.
    """

    fractions = sample["fractions"]
    template = build_model(START_PARAMETERS)
    critical = template.longitudinal.critical_energy_mev
    bounds = np.asarray(template.longitudinal.layer_bounds_x0, dtype=float)
    normals = quadrature_normals()
    parameters = START_PARAMETERS if fixed_origin is None else replace(START_PARAMETERS, origin_x0=float(fixed_origin))
    history: list[dict[str, Any]] = []
    moments: dict[str, Any] = {}
    central: dict[str, Any] = {}
    converged = False
    for step in range(max_iterations):
        measured_at = parameters.origin_x0
        moments = measured_fluctuations(fractions, bounds, measured_at)
        parameters = replace(parameters, **fit_fluctuation_parameters(moments, critical))
        parameters, central = fit_central_parameters(fractions, parameters, critical, bounds, normals, fixed_origin)
        history.append(
            {
                "iteration": step + 1,
                "origin_at_which_moments_were_measured": measured_at,
                "parameters": parameters.as_dict(),
                "chi2": central["chi2"],
            }
        )
        if abs(parameters.origin_x0 - measured_at) < ORIGIN_CONVERGENCE_X0:  # always true with a fixed origin
            converged = True
            break
    record = {
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data": {
            "sample": "geant4_electron_sample/baseline (exposed development data; deposition representation)",
            "sealed_data_read": False,
            "batches": sample["provenance"],
        },
        "method": {
            "widths": "Grindhammer and Peters sampling set with the covariant factor (fixed, not fitted)",
            "fixed_by_sources": {
                "x_max_offset_x0": parameters.x_max_offset_x0,
                "tail_rate_per_x0": parameters.tail_rate_per_x0,
            },
            "quadrature": {"kind": "scrambled Sobol", "points": 2**QUADRATURE_LOG2, "seed": QUADRATURE_SEED},
            "iterations": len(history),
            "converged": converged,
            "origin_convention": "front face" if fixed_origin is not None else "free (two-cycles)",
            "origin_convergence_x0": ORIGIN_CONVERGENCE_X0,
        },
        "history": history,
        "measured_fluctuations_at_final_origin": measured_fluctuations(fractions, bounds, parameters.origin_x0),
        "central_fit": central,
    }
    return parameters, record


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=RESULTS_JSON)
    args = parser.parse_args(argv)
    sample = load_sample()
    parameters, record = calibrate(sample)
    write_parameters_artifact(parameters, args.artifact, record, "ams_ecal.electron_model.em_production_calibration")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"parameters": parameters.as_dict(), **record}, indent=2), encoding="utf-8")
    print(args.artifact)
    print(args.report)


if __name__ == "__main__":
    main()
