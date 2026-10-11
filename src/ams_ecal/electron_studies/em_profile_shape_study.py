"""Profile shape study of Geant4 electrons (analysis only): why a single gamma cannot reach the 100 GeV profile.

Decision record ``research/DECISIONS.md`` (DEC-014, 2026-10-06 rows), note
``research/plans/2026-10-06_em_profile_shape_study_note.md``. Uses only the EXPOSED Geant4 electrons
(development / calibration data); no sealed data, no generator is changed or written.

Four questions, in the order the researcher approved them:

1. WHY the gamma form cannot reach the profile. The residual structure of the per-event gamma fits
   (layers 0-1 above, the middle below, containment too high), the energy dependence of the first
   layers, the scintillator fraction by depth, and the held-out score of every extension of the mean
   profile: energy-dependent beta or origin, a free depth law, and a slowly decaying tail component
   whose rate is first taken from the literature (Leroy and Rancoita, Table 2: longitudinal
   attenuation length of lead 3.3-3.9 X0).
2. Does the SKEWNESS of ln T survive a profile family that can represent the tail? A planted-null
   control (a symmetric ln T plus a tail, refitted with and without the tail) tells whether the gamma-only
   fit can manufacture it; the consequence of a skewed marginal for the leakage tail and the contained
   fraction is measured against the sampling noise of the comparison.
3. The AMS statement (b = 0.65, constant) on the readout and the deposition profile: per-origin event
   beta of both representations, its energy slope, the origin at which it equals 0.65, and the
   mean-profile cost of fixing 0.65 with and without the tail.
4. The literature predictions read from the sources (Grindhammer and Peters, the PDG review) against the
   same electrons.

Nothing here changes DEC-001. Words: calibrated, candidate, development; never validated.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from scipy.optimize import least_squares

from ams_ecal.detector.geometry import load_geometry
from ams_ecal.electron_studies import em_depth_origin_calibration as calibration
from ams_ecal.electron_studies.em_depth_origin_calibration import (
    ENERGY_REFERENCE_GEV,
    ModelSpec,
    Targets,
    fit_model,
    fit_summary,
    leave_one_energy_out,
    make_targets,
    split_half,
)
from ams_ecal.electron_studies.em_longitudinal_fluctuations import (
    DATA,
    LOWER,
    START,
    UPPER,
    layer_bounds_x0,
)
from ams_ecal.electron_studies.em_longitudinal_structure_analysis import (
    EnergyData,
    _clean,
    beta_values,
    distances,
    fit_free_with_origin,
    gp_formulae,
    joint_diagnostics,
    load_energy,
    longitudinal_metrics,
    marginal_diagnostics,
    profile_fractions_batch,
)
from ams_ecal.geant4_simulation.geant4_backend import PROJECT_ROOT
from ams_ecal.geant4_simulation.pilot_analysis import layer_centres_mm
from ams_ecal.validation.dataset import GEOMETRY_CONFIG

ENERGIES_GEV = (10.0, 20.0, 50.0, 100.0)
RESULTS_DIR = PROJECT_ROOT / "results" / "em_generator"
STUDY_JSON = RESULTS_DIR / "profile_shape_study.json"
STUDY_PLOTS = RESULTS_DIR / "profile_shape_study"

# ----- numbers read from the sources (exact locations in the note)
LEAD_Z = 82
GP_T_HOM_OFFSET = -0.858  # Grindhammer and Peters 2000, App. A.1.1: T_hom = ln y - 0.858
GP_ALPHA_COEFFICIENTS = (0.21, 0.492, 2.38)  # alpha_hom = a1 + (a2 + a3 / Z) ln y
PDG_T_OFFSET_ELECTRON = -0.5  # PDG review 34.5, Eq. 34.36: t_max = ln y + C_e
PDG_B = 0.5
TAIL_ATTENUATION_LEAD_X0 = (3.3, 3.9)  # Leroy and Rancoita 2000, Table 2 (lead, 0.6-6 GeV)
TAIL_KAPPA_LITERATURE = 1.0 / 3.6
BETA_AMS = 0.65
ORIGINS = (0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0)
ENSEMBLE_EVENTS = 4000
ENSEMBLE_SEEDS = 5
CONTROL_EVENTS = 1200
FINE_VOXELS = 36
MESH_XY = 216 * 216
CHOSEN_TAIL_FAMILY = "tail_literature_rate"

# ------------------------------------------------------------------ the sources, as numbers


def gp_homogeneous(energy_gev: float, critical_energy_mev: float, z: int = LEAD_Z) -> dict[str, float]:
    """Grindhammer and Peters homogeneous-medium mean parameters at the front-face origin."""

    y = 1000.0 * energy_gev / critical_energy_mev
    t = np.log(y) + GP_T_HOM_OFFSET
    a1, a2, a3 = GP_ALPHA_COEFFICIENTS
    alpha = a1 + (a2 + a3 / z) * np.log(y)
    return {"T": float(t), "alpha": float(alpha), "beta": float((alpha - 1.0) / t)}


def pdg_electron(energy_gev: float, critical_energy_mev: float) -> dict[str, float]:
    y = 1000.0 * energy_gev / critical_energy_mev
    t = np.log(y) + PDG_T_OFFSET_ELECTRON
    return {"T": float(t), "alpha": float(1.0 + PDG_B * t), "beta": PDG_B}


def literature_comparison(
    loaded: dict[float, EnergyData], bounds: np.ndarray, critical_energy_mev: float, front_face_beta: float
) -> dict[str, Any]:
    """The same electrons against the two formula sets, at the front-face origin."""

    rows = {}
    for energy, data in loaded.items():
        fits = np.array([fit_free_with_origin(f, bounds, 0.0)[0] for f in data.fractions["deposition"]])
        rows[f"{energy:g}"] = {
            "data_median_T": float(np.median(np.exp(fits[:, 0]))),
            "data_median_alpha": float(np.median(np.exp(fits[:, 1]))),
            "data_median_beta": float(np.median(beta_values(fits[:, 0], fits[:, 1]))),
            "grindhammer_peters_homogeneous": gp_homogeneous(energy, critical_energy_mev),
            "pdg_electron": pdg_electron(energy, critical_energy_mev),
        }
    return {
        "origin_x0": 0.0,
        "per_energy": rows,
        "mean_profile_front_face_free_beta": front_face_beta,
        "note": (
            "the sampling-calorimeter corrections of Grindhammer and Peters (Eqs. 18-21) act on the visible "
            "signal; the deposition target (all material of the composite) is compared with the homogeneous set"
        ),
    }


# ------------------------------------------------------------------ residual structure


def load_fine_profile(energy_gev: float) -> dict[str, Any]:
    """Half-layer (36 voxel) mean profile and the scintillator fraction by layer, straight from the events."""

    arrays = np.load(DATA / f"E{energy_gev:g}GeV" / "events.npz")
    offsets, index, edep = arrays["fine_offsets"], arrays["fine_index"], arrays["fine_edep_mev"]
    n_events = len(offsets) - 1
    voxel_z = index // MESH_XY  # the mesh index is z-slowest; checked against the layer grid below
    event = np.repeat(np.arange(n_events), np.diff(offsets))
    fine = np.zeros((n_events, FINE_VOXELS))
    np.add.at(fine, (event, voxel_z), edep)
    deposit = arrays["deposit_grid_mev"].sum(axis=2)
    readout = arrays["readout_grid_mev"].sum(axis=2)
    first = slice(offsets[0], offsets[1])
    by_layer = np.bincount(voxel_z[first] // 2, weights=edep[first], minlength=18)
    return {
        "mean_fine_profile_mev": fine.mean(axis=0).tolist(),
        "first_voxel_quantiles_5_50_95_mev": np.quantile(fine[:, 0], [0.05, 0.5, 0.95]).tolist(),
        "scintillator_fraction_by_layer": (readout.mean(axis=0) / deposit.mean(axis=0)).tolist(),
        "mean_total_deposit_mev": float(arrays["edep_total_mev"].mean()),
        "layer_grid_matches_fine_sum_max_abs_mev": float(np.abs(by_layer - deposit[0]).max()),
    }


def tail_fractions(bounds: np.ndarray, origin: float, tail_alpha: float, tail_kappa: float) -> np.ndarray:
    """Layer fractions of the tail component: a gamma density of shape ``tail_alpha`` and rate ``tail_kappa``."""

    depth = (tail_alpha - 1.0) / tail_kappa
    return profile_fractions_batch(np.array([np.log(depth)]), np.array([np.log(tail_alpha)]), bounds, origin)[0]


def fit_gamma_plus_tail(
    fractions: np.ndarray, bounds: np.ndarray, origin: float, tail_alpha: float, tail_kappa: float
) -> np.ndarray:
    """``(ln T, ln alpha, w)``: a gamma core plus a tail component of fixed shape and rate, weight free."""

    tail = tail_fractions(bounds, origin, tail_alpha, tail_kappa)

    def residual(p: np.ndarray) -> np.ndarray:
        core = profile_fractions_batch(p[0:1], p[1:2], bounds, origin)[0]
        return (1.0 - p[2]) * core + p[2] * tail - fractions

    result = least_squares(
        residual,
        [START[0], START[1], 0.05],
        bounds=([LOWER[0], LOWER[1], 0.0], [UPPER[0], UPPER[1], 0.9]),
    )
    return result.x


def residual_structure(
    loaded: dict[float, EnergyData],
    gamma_fits: dict[float, np.ndarray],
    tail_fits: dict[float, np.ndarray],
    bounds: np.ndarray,
    origin: float,
    tail_alpha: float,
    tail_kappa: float,
) -> dict[str, Any]:
    """Mean (data - fit) by layer for the gamma-only and the gamma-plus-tail per-event fits."""

    tail = tail_fractions(bounds, origin, tail_alpha, tail_kappa)
    out: dict[str, Any] = {}
    layer0_mev = []
    for energy, data in loaded.items():
        fractions = data.fractions["deposition"]
        scale = 1000.0 * energy
        gamma = profile_fractions_batch(gamma_fits[energy][:, 0], gamma_fits[energy][:, 1], bounds, origin)
        core = profile_fractions_batch(tail_fits[energy][:, 0], tail_fits[energy][:, 1], bounds, origin)
        w = tail_fits[energy][:, 2][:, None]
        both = (1.0 - w) * core + w * tail[None, :]
        entry: dict[str, Any] = {}
        for label, model in (("gamma_only", gamma), ("gamma_plus_tail", both)):
            residual = (fractions - model) * scale
            mean = residual.mean(axis=0)
            se = residual.std(axis=0, ddof=1) / np.sqrt(len(residual))
            entry[label] = {
                "mean_residual_mev_by_layer": mean.tolist(),
                "pull_by_layer": (mean / se).tolist(),
                "relative_residual_by_layer": (np.mean(fractions - model, axis=0) / fractions.mean(axis=0)).tolist(),
                "contained_data": float(fractions.sum(axis=1).mean()),
                "contained_fit": float(model.sum(axis=1).mean()),
                "rms_event_residual_fraction": float(np.sqrt(((fractions - model) ** 2).sum(axis=1).mean())),
            }
        entry["layer0_data_mev"] = float(fractions[:, 0].mean() * scale)
        layer0_mev.append(entry["layer0_data_mev"])
        out[f"{energy:g}"] = entry
    slope, intercept = np.polyfit([float(e) for e in loaded], layer0_mev, 1)
    out["layer0_energy_line"] = {"intercept_mev": float(intercept), "slope_mev_per_gev": float(slope)}
    return out


# ------------------------------------------------------------------ extensions of the mean profile


def extension_specs() -> dict[str, ModelSpec]:
    kappa = {"tail_kappa": TAIL_KAPPA_LITERATURE}
    return {
        "primary": calibration.PRIMARY,
        "beta_energy_slope": ModelSpec("beta_slope", ("z0", "beta", "beta_slope")),
        "origin_energy_slope": ModelSpec("z0_slope", ("z0", "beta", "z0_slope")),
        "beta_and_origin_energy_slopes": ModelSpec("both_slopes", ("z0", "beta", "beta_slope", "z0_slope")),
        "depth_law_slope_and_offset": ModelSpec("depth_law", ("z0", "beta", "depth_slope", "delta")),
        CHOSEN_TAIL_FAMILY: ModelSpec("tail_lit", ("z0", "beta", "tail_w", "tail_alpha"), kappa),
        "tail_literature_rate_energy_weight": ModelSpec(
            "tail_lit_slope", ("z0", "beta", "tail_w", "tail_w_slope", "tail_alpha"), kappa
        ),
        "tail_free_rate": ModelSpec("tail_free", ("z0", "beta", "tail_w", "tail_alpha", "tail_kappa")),
        "ams_beta_fixed_with_tail": ModelSpec("ams_tail", ("z0", "tail_w", "tail_alpha"), {"beta": BETA_AMS, **kappa}),
        "ams_beta_fixed_gamma_only": calibration.AMS_BETA_FREE_ORIGIN,
    }


def extension_study(
    targets: Targets,
    families: Sequence[str] | None = None,
    *,
    amplitude: str = "none",
    held_out: bool = True,
) -> dict[str, Any]:
    """Fit every extension; with ``held_out``, score it by leave-one-energy-out and split-half resampling."""

    specs = {name: replace(spec, amplitude=amplitude) for name, spec in extension_specs().items()}
    base = fit_model(specs["primary"], targets)
    base_loeo = leave_one_energy_out(specs["primary"], targets) if held_out else None
    base_half = split_half(specs["primary"], targets) if held_out else None
    out: dict[str, Any] = {}
    for name in families or specs:
        spec = specs[name]
        fit = base if name == "primary" else fit_model(spec, targets)
        summary = fit_summary(fit, targets)
        row: dict[str, Any] = {
            "free": list(spec.free),
            "fixed": dict(spec.fixed),
            "parameters": {k: summary["parameters"][k] for k in (*spec.free, *spec.fixed)},
            "chi2": fit.chi2,
            "dof": fit.dof,
            "chi2_per_dof": fit.scale,
            "aic_scaled_by_primary": float(fit.chi2 / base.scale + 2 * len(fit.theta)),
            "rms_relative_layers_0_3": summary["rms_relative_layers_0_3"],
            "rms_relative_layers_4_17": summary["rms_relative_layers_4_17"],
            "relative_residual_by_layer": summary["relative_residual_by_layer"],
            "contained_fraction_model": summary["contained_fraction_model"],
            "contained_fraction_geant4": summary["contained_fraction_geant4"],
        }
        if held_out and base_loeo is not None and base_half is not None:
            loeo = base_loeo if name == "primary" else leave_one_energy_out(spec, targets)
            half = base_half if name == "primary" else split_half(spec, targets)
            row["leave_one_energy_out"] = loeo
            row["split_half"] = half
            row["held_out_improvement_over_primary"] = {
                "leave_one_energy_out": float(1.0 - loeo["mean_chi2_per_point"] / base_loeo["mean_chi2_per_point"]),
                "split_half": float(1.0 - half["chi2_per_point"] / base_half["chi2_per_point"]),
                "left_out_100_gev": float(
                    1.0 - loeo["per_energy"][-1]["chi2_per_point"] / base_loeo["per_energy"][-1]["chi2_per_point"]
                ),
            }
        out[name] = row
    return out


def depth_scale_check(loaded: dict[float, EnergyData], bounds: np.ndarray) -> dict[str, Any]:
    """The composite radiation length of the Geant4 prefix against the 17 X0 used for the layer bounds."""

    metadata = json.loads((DATA / "E100GeV" / "metadata.json").read_text(encoding="utf-8"))
    x0_mm = float(metadata["materials"]["composite_radiation_length_mm"])
    true_x0 = 166.5 / x0_mm
    scale = true_x0 / float(bounds[-1, 1])
    reference = fit_model(calibration.PRIMARY, make_targets(loaded, "deposition", bounds))
    scaled = fit_model(calibration.PRIMARY, make_targets(loaded, "deposition", bounds * scale))
    return {
        "composite_x0_mm": x0_mm,
        "prefix_depth_x0_geant4": true_x0,
        "prefix_depth_x0_in_layer_bounds": float(bounds[-1, 1]),
        "scale": scale,
        "primary_z0_beta_with_layer_bounds": [float(v) for v in reference.theta],
        "primary_z0_beta_with_geant4_depth": [float(v) for v in scaled.theta],
        "chi2_with_layer_bounds": reference.chi2,
        "chi2_with_geant4_depth": scaled.chi2,
    }


# ------------------------------------------------------------------ the ln T marginal (question 2)


def skew_normal_parameters(mean: float, sd: float, skewness: float) -> tuple[float, float, float]:
    """``(shape, location, scale)`` of the skew-normal with the given first three moments (|skew| capped)."""

    g = float(np.clip(skewness, -0.99, 0.99))
    ratio = (4.0 - np.pi) / 2.0
    delta = np.sign(g) * np.sqrt(
        (np.pi / 2.0) * abs(g) ** (2.0 / 3.0) / (abs(g) ** (2.0 / 3.0) + ratio ** (2.0 / 3.0))
    )
    delta = float(np.clip(delta, -0.999, 0.999))
    scale = sd / np.sqrt(1.0 - 2.0 * delta**2 / np.pi)
    location = mean - scale * delta * np.sqrt(2.0 / np.pi)
    return float(delta / np.sqrt(1.0 - delta**2)), float(location), float(scale)


def draw_joint(measured: np.ndarray, kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` draws of (ln T, ln alpha): ``gaussian``, ``empirical`` (bootstrap) or ``skew_normal`` marginals."""

    if kind == "empirical":
        return measured[rng.integers(0, len(measured), n)]
    if kind == "gaussian":
        return rng.multivariate_normal(measured.mean(axis=0), np.cov(measured.T), size=n)
    rho_spearman = float(stats.spearmanr(measured[:, 0], measured[:, 1]).statistic)
    rho = 2.0 * np.sin(np.pi * rho_spearman / 6.0)  # Gaussian-copula correlation from the rank correlation
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n)
    u = stats.norm.cdf(z)
    out = np.empty((n, 2))
    for column in (0, 1):
        x = measured[:, column]
        a, loc, scale = skew_normal_parameters(float(x.mean()), float(x.std(ddof=1)), float(stats.skew(x)))
        out[:, column] = stats.skewnorm.ppf(u[:, column], a, loc=loc, scale=scale)
    return out


def centred_leakage_ks(ensemble: np.ndarray, reference: np.ndarray) -> float:
    """KS distance between the mean-centred contained-fraction distributions (removes the mean offset)."""

    a, b = ensemble.sum(axis=1), reference.sum(axis=1)
    return float(stats.ks_2samp(a - a.mean(), b - b.mean(), method="asymp").statistic)


def leakage_tail(ensemble: np.ndarray) -> dict[str, float]:
    leak = 1.0 - ensemble.sum(axis=1)
    return {
        "mean": float(leak.mean()),
        "sd": float(leak.std(ddof=1)),
        "skewness": float(stats.skew(leak)),
        "q95_minus_mean": float(np.quantile(leak, 0.95) - leak.mean()),
        "q99_minus_mean": float(np.quantile(leak, 0.99) - leak.mean()),
    }


def ensemble_fractions(
    draws: np.ndarray, bounds: np.ndarray, origin: float, weight: float, tail_alpha: float, tail_kappa: float
) -> np.ndarray:
    core = profile_fractions_batch(draws[:, 0], draws[:, 1], bounds, origin)
    if weight <= 0.0:
        return core
    return (1.0 - weight) * core + weight * tail_fractions(bounds, origin, tail_alpha, tail_kappa)[None, :]


def skew_consequence(
    loaded: dict[float, EnergyData],
    fits: dict[float, np.ndarray],
    bounds: np.ndarray,
    origin: float,
    z_mm: np.ndarray,
    weight_by_energy: dict[float, float],
    tail_alpha: float,
    tail_kappa: float,
    label: str,
    n_events: int = ENSEMBLE_EVENTS,
) -> dict[str, Any]:
    """Gaussian, empirical and skew-normal (ln T, ln alpha) ensembles against Geant4, with sampling noise."""

    out: dict[str, Any] = {"profile_family": label}
    for energy, data in loaded.items():
        reference = data.fractions["deposition"]
        measured = fits[energy][:, :2]
        rows: dict[str, Any] = {"geant4_leakage": leakage_tail(reference)}
        for kind in ("gaussian", "skew_normal", "empirical"):
            collected: list[dict[str, float]] = []
            for seed in range(ENSEMBLE_SEEDS):
                rng = np.random.default_rng(10_000 * seed + int(energy))
                draws = draw_joint(measured, kind, n_events, rng)
                ensemble = ensemble_fractions(
                    draws, bounds, origin, weight_by_energy[energy], tail_alpha, tail_kappa
                )
                row = dict(distances(longitudinal_metrics(ensemble, reference, z_mm)))
                row["centred_leakage_ks"] = centred_leakage_ks(ensemble, reference)
                row.update({f"leakage_{k}": v for k, v in leakage_tail(ensemble).items()})
                collected.append(row)
            rows[kind] = {
                key: {
                    "mean": float(np.mean([c[key] for c in collected])),
                    "sd": float(np.std([c[key] for c in collected], ddof=1)),
                }
                for key in collected[0]
            }
        out[f"{energy:g}"] = rows
    return out


def tail_artifact_control(
    loaded: dict[float, EnergyData],
    tail_fits: dict[float, np.ndarray],
    bounds: np.ndarray,
    origin: float,
    tail_alpha: float,
    tail_kappa: float,
    n: int = CONTROL_EVENTS,
) -> dict[str, Any]:
    """Planted null for the ln T skewness: symmetric truth, a tail, borrowed residuals, refitted two ways.

    Draw (ln T, ln alpha) from a bivariate NORMAL with the measured moments of the tail-aware fits, add a
    tail with the measured mean weight and the residual vectors of real events, and refit with the gamma-only
    form and with the gamma-plus-tail form. If only the gamma-only refit is skewed, the skewness measured with
    the gamma-only fits is an artefact of the missing tail. The borrowed residuals do not reproduce an
    event's own shape-dependent noise: a qualitative control.
    """

    tail = tail_fractions(bounds, origin, tail_alpha, tail_kappa)
    out: dict[str, Any] = {}
    for energy, data in loaded.items():
        rng = np.random.default_rng(int(energy) + 29)
        fractions = data.fractions["deposition"]
        measured = tail_fits[energy]
        core = profile_fractions_batch(measured[:, 0], measured[:, 1], bounds, origin)
        w = measured[:, 2][:, None]
        residual = fractions - ((1.0 - w) * core + w * tail[None, :])
        joint = rng.multivariate_normal(measured[:, :2].mean(axis=0), np.cov(measured[:, :2].T), size=n)
        planted_w = float(measured[:, 2].mean())
        synthetic = (1.0 - planted_w) * profile_fractions_batch(joint[:, 0], joint[:, 1], bounds, origin)
        synthetic = synthetic + planted_w * tail[None, :]
        synthetic = np.clip(synthetic + residual[rng.integers(0, len(residual), n)], 0.0, None)
        gamma_only = np.array([fit_free_with_origin(f, bounds, origin)[0] for f in synthetic])
        with_tail = np.array(
            [fit_gamma_plus_tail(f, bounds, origin, tail_alpha, tail_kappa)[:2] for f in synthetic]
        )
        out[f"{energy:g}"] = {
            "planted_skewness_ln_t": float(stats.skew(joint[:, 0])),
            "gamma_only_refit_skewness_ln_t": float(stats.skew(gamma_only[:, 0])),
            "tail_refit_skewness_ln_t": float(stats.skew(with_tail[:, 0])),
            "planted_pearson": float(np.corrcoef(joint.T)[0, 1]),
            "gamma_only_refit_pearson": float(np.corrcoef(gamma_only.T)[0, 1]),
            "tail_refit_pearson": float(np.corrcoef(with_tail.T)[0, 1]),
            "planted_sd_ln_t": float(joint[:, 0].std(ddof=1)),
            "gamma_only_refit_sd_ln_t": float(gamma_only[:, 0].std(ddof=1)),
            "tail_refit_sd_ln_t": float(with_tail[:, 0].std(ddof=1)),
        }
    return out


def stochastic_summary(fits: dict[float, np.ndarray], origin: float, critical_energy_mev: float) -> dict[str, Any]:
    """Skewness, spreads, correlation and the covariant GP ratios of a set of per-event fits."""

    out = {}
    for energy, fit in fits.items():
        ln_t, ln_alpha = fit[:, 0], fit[:, 1]
        energy_mev = 1000.0 * energy
        x_max = calibration.x_max_of_energy(energy_mev, critical_energy_mev, calibration.DELTA_DEC001)
        depth = max(x_max - origin, 0.2)
        gp = gp_formulae(float(np.log(energy_mev / critical_energy_mev)))["sampling"]
        marginal_t, marginal_a = marginal_diagnostics(ln_t), marginal_diagnostics(ln_alpha)
        joint = joint_diagnostics(ln_t, ln_alpha)
        out[f"{energy:g}"] = {
            "ln_t": marginal_t,
            "ln_alpha": marginal_a,
            "joint": joint,
            "beta_median": float(np.median(beta_values(ln_t, ln_alpha))),
            "sigma_ln_t_over_gp_covariant": marginal_t["sd"] / (gp["sigma_ln_t"] * x_max / depth),
            "sigma_ln_alpha_over_gp": marginal_a["sd"] / gp["sigma_ln_alpha"],
            "gp_rho": gp["rho"],
            "rho_minus_gp": joint["pearson"] - gp["rho"],
            "tail_weight_mean": float(fit[:, 2].mean()) if fit.shape[1] > 2 else None,
            "tail_weight_sd": float(fit[:, 2].std(ddof=1)) if fit.shape[1] > 2 else None,
        }
    return out


# ------------------------------------------------------------------ the AMS statement (question 3)


def fit_with_amplitude_and_origin(fractions: np.ndarray, bounds: np.ndarray, origin: float) -> np.ndarray:
    """``(ln T, ln alpha, ln A)`` of an observed profile of unknown scale, depth origin held at ``origin``."""

    start = (START[0], START[1], float(np.log(max(fractions.sum(), 1e-9) / 0.9)))
    result = least_squares(
        lambda p: np.exp(p[2]) * profile_fractions_batch(p[0:1], p[1:2], bounds, origin)[0] - fractions,
        start,
        bounds=((LOWER[0], LOWER[1], -12.0), (UPPER[0], UPPER[1], 2.0)),
    )
    return result.x


def ams_beta_versus_origin(loaded: dict[float, EnergyData], bounds: np.ndarray) -> dict[str, Any]:
    """Event-level beta of both representations at each origin, its energy slope, and where it crosses 0.65."""

    table: dict[str, Any] = {"origins_x0": list(ORIGINS)}
    energies = np.array([float(e) for e in loaded])
    for representation in ("deposition", "readout"):
        medians = np.zeros((len(loaded), len(ORIGINS)))
        for k, data in enumerate(loaded.values()):
            for j, origin in enumerate(ORIGINS):
                if representation == "deposition":
                    fits = np.array(
                        [fit_free_with_origin(f, bounds, origin)[0] for f in data.fractions[representation]]
                    )
                else:
                    fits = np.array(
                        [fit_with_amplitude_and_origin(f, bounds, origin)[:2] for f in data.fractions[representation]]
                    )
                medians[k, j] = np.median(beta_values(fits[:, 0], fits[:, 1]))
        crossing = []
        for k in range(len(loaded)):
            order = np.argsort(medians[k])
            crossing.append(float(np.interp(BETA_AMS, medians[k][order], np.array(ORIGINS)[order])))
        table[representation] = {
            "median_beta_by_energy_and_origin": medians.tolist(),
            "slope_of_median_beta_per_ln_energy_by_origin": [
                float(np.polyfit(np.log(energies), medians[:, j], 1)[0]) for j in range(len(ORIGINS))
            ],
            "origin_where_median_beta_is_0_65_by_energy": crossing,
        }
    return table


# ------------------------------------------------------------------ orchestration


def analyse(
    energies: Sequence[float] = ENERGIES_GEV,
    *,
    families: Sequence[str] | None = None,
    ensemble_events: int = ENSEMBLE_EVENTS,
    control_events: int = CONTROL_EVENTS,
    origins: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run questions 1-4. ``families`` limits the extension study (it must include the chosen tail family)."""

    if families is not None and CHOSEN_TAIL_FAMILY not in families:
        raise ValueError(f"the families must include {CHOSEN_TAIL_FAMILY!r}: the later parts use its parameters")
    bounds = layer_bounds_x0()
    z_mm = layer_centres_mm(load_geometry(GEOMETRY_CONFIG), 18)
    loaded = {float(e): load_energy(e) for e in energies}
    deposition = make_targets(loaded, "deposition", bounds)
    readout = make_targets(loaded, "readout", bounds)
    results: dict[str, Any] = {}
    artefacts: dict[str, Any] = {"loaded": loaded, "targets": deposition}

    # 4. the sources
    front_face = fit_model(calibration.FRONT_FACE_FREE_BETA, deposition)
    results["L_literature_comparison"] = literature_comparison(
        loaded, bounds, deposition.critical_energy_mev, float(front_face.parameters()["beta"])
    )

    # 1. why
    extension = extension_study(deposition, families)
    results["A_extensions_deposition"] = extension
    artefacts["extension"] = extension
    results["A_depth_scale"] = depth_scale_check(loaded, bounds)
    readout_families = ["primary", CHOSEN_TAIL_FAMILY, "ams_beta_fixed_with_tail", "ams_beta_fixed_gamma_only"]
    results["A_extensions_readout"] = {
        "note": "amplitude free per energy (the readout is an observed signal); no held-out scoring",
        **extension_study(readout, readout_families, amplitude="free", held_out=False),
    }

    chosen = extension[CHOSEN_TAIL_FAMILY]["parameters"]
    origin = float(chosen["z0"])
    tail_alpha, tail_kappa = float(chosen["tail_alpha"]), float(chosen["tail_kappa"])
    results["B_tail_family"] = {"origin_x0": origin, "tail_alpha": tail_alpha, "tail_kappa": tail_kappa}
    gamma_fits = {
        e: np.array([fit_free_with_origin(f, bounds, origin)[0] for f in d.fractions["deposition"]])
        for e, d in loaded.items()
    }
    tail_fits = {
        e: np.array([fit_gamma_plus_tail(f, bounds, origin, tail_alpha, tail_kappa) for f in d.fractions["deposition"]])
        for e, d in loaded.items()
    }
    artefacts.update(gamma_fits=gamma_fits, tail_fits=tail_fits, origin=origin)
    results["B_residual_structure"] = residual_structure(
        loaded, gamma_fits, tail_fits, bounds, origin, tail_alpha, tail_kappa
    )
    results["B_fine_profile"] = {f"{e:g}": load_fine_profile(e) for e in energies}

    # 2. skewness
    results["C_stochastic_gamma_only"] = stochastic_summary(gamma_fits, origin, deposition.critical_energy_mev)
    results["C_stochastic_gamma_plus_tail"] = stochastic_summary(tail_fits, origin, deposition.critical_energy_mev)
    results["C_tail_artifact_control"] = tail_artifact_control(
        loaded, tail_fits, bounds, origin, tail_alpha, tail_kappa, control_events
    )
    weights = {e: float(tail_fits[e][:, 2].mean()) for e in loaded}
    results["C_skew_consequence_gamma_only"] = skew_consequence(
        loaded, gamma_fits, bounds, origin, z_mm, {e: 0.0 for e in loaded}, tail_alpha, tail_kappa, "gamma_only",
        ensemble_events,
    )
    results["C_skew_consequence_gamma_plus_tail"] = skew_consequence(
        loaded, tail_fits, bounds, origin, z_mm, weights, tail_alpha, tail_kappa, "gamma_plus_tail", ensemble_events
    )

    # 3. the AMS statement
    if origins:
        results["D_ams_beta_versus_origin"] = ams_beta_versus_origin(loaded, bounds)
    return results, artefacts


def write_tables(results: dict[str, Any], directory: Path) -> None:
    with (directory / "profile_shape_study_extensions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["family", "free", "chi2", "dof", "chi2_per_dof", "aic_scaled", "rms_layers_0_3_10", "rms_layers_0_3_20",
             "rms_layers_0_3_50", "rms_layers_0_3_100", "rms_layers_4_17_mean", "improvement_loeo",
             "improvement_split_half", "improvement_left_out_100"]
        )
        for name, row in results["A_extensions_deposition"].items():
            improvement = row.get("held_out_improvement_over_primary", {})
            writer.writerow(
                [name, "+".join(row["free"]), row["chi2"], row["dof"], row["chi2_per_dof"], row["aic_scaled_by_primary"],
                 *row["rms_relative_layers_0_3"], float(np.mean(row["rms_relative_layers_4_17"])),
                 improvement.get("leave_one_energy_out"), improvement.get("split_half"),
                 improvement.get("left_out_100_gev")]
            )
    with (directory / "profile_shape_study_residuals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["energy_gev", "profile", "layer", "mean_residual_mev", "pull", "relative_residual"])
        for energy, entry in results["B_residual_structure"].items():
            if energy == "layer0_energy_line":
                continue
            for label in ("gamma_only", "gamma_plus_tail"):
                block = entry[label]
                for layer in range(18):
                    writer.writerow(
                        [energy, label, layer, block["mean_residual_mev_by_layer"][layer],
                         block["pull_by_layer"][layer], block["relative_residual_by_layer"][layer]]
                    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=STUDY_JSON)
    parser.add_argument("--plots", type=Path, default=STUDY_PLOTS)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    results, artefacts = analyse()
    document = {
        "status": (
            "ANALYSIS ONLY on the EXPOSED Geant4 electrons (development / calibration data). No generator was "
            "written or changed; no sealed data was read. Not a validation."
        ),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "literature_inputs": {
            "gp_t_hom_offset": GP_T_HOM_OFFSET,
            "gp_alpha_coefficients": GP_ALPHA_COEFFICIENTS,
            "pdg_t_offset_electron": PDG_T_OFFSET_ELECTRON,
            "tail_attenuation_lead_x0": TAIL_ATTENUATION_LEAD_X0,
            "energy_reference_gev": ENERGY_REFERENCE_GEV,
        },
        **_clean(results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2), encoding="utf-8")
    write_tables(results, args.out.parent)
    if not args.no_plots:
        from ams_ecal.electron_studies.em_profile_shape_study_plots import make_plots

        make_plots(results, artefacts, args.plots)
    print(args.out)


if __name__ == "__main__":
    main()
