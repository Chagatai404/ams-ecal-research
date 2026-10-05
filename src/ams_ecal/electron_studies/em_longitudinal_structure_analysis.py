"""Analysis-only study of the longitudinal structure of Geant4 electrons (no generator change).

Decision record: ``research/DECISIONS.md`` (DEC-014, F1-F5), plan
``research/plans/2026-10-05_em_production_generator_plan.md`` sections 8-9, note
``research/plans/2026-10-05_em_longitudinal_structure_analysis_note.md``. Uses only the EXPOSED Geant4 electrons
(``data/geant4_electron_sample``, development / calibration data); no sealed data, no change to
any generator. DEC-001 is evaluated through its public sampling API and never modified.

A. Joint longitudinal structure: per energy, ``T``, ``alpha``, ``ln T``, ``ln alpha`` and
   ``beta_i = (alpha_i - 1) / T_i`` from per-event gamma fits (two estimators), with marginal and
   joint diagnostics and the Grindhammer-Peters (GP) sampling AND homogeneous formulae, and the
   values DEC-001 uses in each of its two regimes.
B. The event-level beta distribution; how much worse a fit with beta fixed at 0.65 is than a free
   fit; the same diagnostic on the ``readout`` profile (amplitude free), because the AMS statement
   that b is constant refers to a fit of OBSERVED signals.
C. Counterfactual longitudinal ensembles against DEC-001 and Geant4 on early-layer energy, leakage,
   shower maximum, width, layer variance and correlations. Pairs with the same ``ln T`` marginal
   isolate the effect of restoring ``alpha`` as a second, correlated degree of freedom.
D. The AMS mean backbone (``alpha = 1 + 0.65 T`` at the mean depth) against the ensemble-mean
   profile of the joint ensembles and against Geant4; structural mean-profile fits (free alpha,
   origin shift, fixed beta, free normalisation) for the early-layer deficit.
E. Lateral-coupling confounders: entry-cell phase, rear leakage and edge proximity, by partial
   Spearman correlation, stratification and bootstrap.

    uv run python -m ams_ecal.electron_studies.em_longitudinal_structure_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from scipy.optimize import least_squares
from scipy.special import gammainc

from ams_ecal.detector.geometry import load_geometry
from ams_ecal.detector.projection import cell_indices
from ams_ecal.electron_studies.em_longitudinal_fluctuations import (
    DATA,
    ENERGIES_GEV,
    LOWER,
    START,
    UPPER,
    fit_cumulative_fractions,
    layer_bounds_x0,
    profile_fractions,
)
from ams_ecal.geant4_simulation.geant4_backend import PROJECT_ROOT, load_batch
from ams_ecal.geant4_simulation.pilot_analysis import cell_centres_mm, layer_centres_mm
from ams_ecal.proton_model.proton_validation import crossing_observables
from ams_ecal.validation.added_checks import lag_averaged_correlation
from ams_ecal.validation.dataset import GEOMETRY_CONFIG, load_electron_model

RESULTS_DIR = PROJECT_ROOT / "results" / "em_generator"
ANALYSIS_JSON = RESULTS_DIR / "longitudinal_structure_analysis.json"
ANALYSIS_PLOTS = RESULTS_DIR / "longitudinal_structure"
BETA_AMS = 0.65
COUNTERFACTUAL_EVENTS = 4000
BOOTSTRAP = 200
REPRESENTATION_KEYS = {"deposition": "deposit_grid_mev", "readout": "readout_grid_mev"}
LATERAL_OBSERVABLES = ("width_mm", "n_hit_cells", "core_fraction", "containment_fraction")
CORRELATION_LAGS = (1, 2, 3, 5, 8, 12)
VARIANTS = (
    "dec001_deposition",
    "dec001_sampling",
    "fixed_beta_data_T",
    "fixed_beta_matched_mean",
    "joint_data",
    "fixed_beta_origin_data_T",
    "joint_origin_data",
    "joint_ams_mean_gp_fluct",
    "joint_ams_mean_gp_fluct_sampling_T",
)


# ----------------------------------------------------------------------
# Profiles and fits
# ----------------------------------------------------------------------


def _shifted(bounds: np.ndarray, origin: float) -> tuple[np.ndarray, np.ndarray]:
    """Layer edges measured from a depth origin ``origin`` X0 from the front face (negative: upstream)."""

    return np.clip(bounds[:, 0] - origin, 0.0, None), np.clip(bounds[:, 1] - origin, 0.0, None)


def profile_fractions_batch(
    ln_t: np.ndarray, ln_alpha: np.ndarray, bounds: np.ndarray, origin: float = 0.0
) -> np.ndarray:
    """``(n, layers)`` fractions for ``n`` gamma profiles with maximum depth ``T`` (from ``origin``) and shape ``alpha``."""

    lower, upper = _shifted(bounds, origin)
    alpha = np.maximum(np.exp(np.asarray(ln_alpha, dtype=float)), 1.0 + 1e-6)[:, None]
    depth_of_maximum = np.exp(np.asarray(ln_t, dtype=float))[:, None]
    rate = (alpha - 1.0) / depth_of_maximum
    return gammainc(alpha, rate * upper[None, :]) - gammainc(alpha, rate * lower[None, :])


def profile_fixed_beta(
    ln_t: np.ndarray, bounds: np.ndarray, beta: float = BETA_AMS, origin: float = 0.0
) -> np.ndarray:
    """The AMS form: ``alpha = 1 + beta T`` with ``beta`` fixed, for ``n`` depths ``T`` (from ``origin``)."""

    lower, upper = _shifted(bounds, origin)
    depth = np.exp(np.atleast_1d(np.asarray(ln_t, dtype=float)))[:, None]
    alpha = 1.0 + beta * depth
    return gammainc(alpha, beta * upper[None, :]) - gammainc(alpha, beta * lower[None, :])


def profile_general(
    bounds: np.ndarray,
    ln_t: float,
    ln_alpha: float | None = None,
    *,
    origin: float = 0.0,
    amplitude: float = 1.0,
    beta: float | None = None,
) -> np.ndarray:
    """A gamma profile with optional origin shift, free normalisation and fixed beta, for one event."""

    depth = np.exp(ln_t)
    if beta is None:
        if ln_alpha is None:
            raise ValueError("ln_alpha is required unless beta is fixed")
        alpha = np.exp(ln_alpha)
        rate = (alpha - 1.0) / depth
    else:
        alpha, rate = 1.0 + beta * depth, beta
    lower = np.clip(bounds[:, 0] - origin, 0.0, None)
    upper = np.clip(bounds[:, 1] - origin, 0.0, None)
    return amplitude * (gammainc(alpha, rate * upper) - gammainc(alpha, rate * lower))


def fit_fixed_beta(fractions: np.ndarray, bounds: np.ndarray, beta: float = BETA_AMS) -> tuple[float, float]:
    """``(ln T, residual norm)`` of one event fitted with ``beta`` held fixed."""

    result = least_squares(
        lambda p: profile_fixed_beta(p[0], bounds, beta)[0] - fractions,
        (START[0],),
        bounds=((LOWER[0],), (UPPER[0],)),
    )
    return float(result.x[0]), float(np.linalg.norm(result.fun))


def fit_free(fractions: np.ndarray, bounds: np.ndarray) -> tuple[np.ndarray, float]:
    """``((ln T, ln alpha), residual norm)`` of one event fitted by least squares on the layer fractions."""

    result = least_squares(
        lambda p: profile_fractions(p[0], p[1], bounds) - fractions, START, bounds=(LOWER, UPPER)
    )
    return result.x, float(np.linalg.norm(result.fun))


def fit_free_with_origin(
    fractions: np.ndarray, bounds: np.ndarray, origin: float
) -> tuple[np.ndarray, float]:
    """``((ln T, ln alpha), residual norm)`` with the depth origin held at ``origin`` X0."""

    result = least_squares(
        lambda p: profile_fractions_batch(p[0:1], p[1:2], bounds, origin)[0] - fractions,
        START,
        bounds=(LOWER, UPPER),
    )
    return result.x, float(np.linalg.norm(result.fun))


def fit_fixed_beta_with_origin(
    fractions: np.ndarray, bounds: np.ndarray, origin: float, beta: float = BETA_AMS
) -> tuple[float, float]:
    """``(ln T, residual norm)`` with ``beta`` fixed and the depth origin held at ``origin`` X0."""

    result = least_squares(
        lambda p: profile_fixed_beta(p[0], bounds, beta, origin)[0] - fractions,
        (START[0],),
        bounds=((LOWER[0],), (UPPER[0],)),
    )
    return float(result.x[0]), float(np.linalg.norm(result.fun))


def fit_with_amplitude(fractions: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """``(ln T, ln alpha, ln A)``: the shape of an OBSERVED profile whose scale is unknown (readout)."""

    start = (START[0], START[1], float(np.log(max(fractions.sum(), 1e-9) / 0.9)))
    result = least_squares(
        lambda p: np.exp(p[2]) * profile_fractions(p[0], p[1], bounds) - fractions,
        start,
        bounds=((LOWER[0], LOWER[1], -12.0), (UPPER[0], UPPER[1], 2.0)),
    )
    return result.x


def fit_fixed_beta_with_amplitude(
    fractions: np.ndarray, bounds: np.ndarray, beta: float = BETA_AMS
) -> tuple[float, float, float]:
    """``(ln T, ln A, residual norm)`` of an OBSERVED profile with beta held at the AMS value."""

    start = (START[0], float(np.log(max(fractions.sum(), 1e-9) / 0.9)))
    result = least_squares(
        lambda p: np.exp(p[1]) * profile_fixed_beta(p[0], bounds, beta)[0] - fractions,
        start,
        bounds=((LOWER[0], -12.0), (UPPER[0], 2.0)),
    )
    return float(result.x[0]), float(result.x[1]), float(np.linalg.norm(result.fun))


def beta_values(ln_t: np.ndarray, ln_alpha: np.ndarray) -> np.ndarray:
    """``beta_i = (alpha_i - 1) / T_i`` per event."""

    return (np.exp(ln_alpha) - 1.0) / np.exp(ln_t)


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------

NORMAL_LEVELS = (0.01, 0.05, 0.5, 0.95, 0.99)


def marginal_diagnostics(values: np.ndarray) -> dict[str, Any]:
    """Moments, robust spread, tail deviations from a normal and effect sizes (not only p-values)."""

    x = np.asarray(values, dtype=float)
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    z = (x - mean) / sd
    deviations = np.quantile(z, NORMAL_LEVELS) - stats.norm.ppf(NORMAL_LEVELS)
    return {
        "n": len(x),
        "mean": mean,
        "sd": sd,
        "median": float(np.median(x)),
        "robust_sd": float(1.4826 * np.median(np.abs(x - np.median(x)))),
        "skewness": float(stats.skew(x)),
        "excess_kurtosis": float(stats.kurtosis(x)),
        "quantile_levels": list(NORMAL_LEVELS),
        "standardised_quantile_minus_normal": deviations.tolist(),
        "max_abs_quantile_deviation": float(np.abs(deviations).max()),
        "ks_distance_to_fitted_normal": float(stats.kstest(z, "norm").statistic),
        "shapiro_p": float(stats.shapiro(x).pvalue),
    }


def joint_diagnostics(ln_t: np.ndarray, ln_alpha: np.ndarray) -> dict[str, Any]:
    """Covariance, Pearson and Spearman correlation, and the Mahalanobis check against chi-square(2)."""

    data = np.column_stack([ln_t, ln_alpha])
    cov = np.cov(data.T)
    centred = data - data.mean(axis=0)
    mahalanobis = np.einsum("ij,jk,ik->i", centred, np.linalg.inv(cov), centred)
    return {
        "covariance": cov.tolist(),
        "pearson": float(np.corrcoef(data.T)[0, 1]),
        "spearman": float(stats.spearmanr(ln_t, ln_alpha).statistic),
        "mahalanobis_ks_vs_chi2_2": float(stats.kstest(mahalanobis, "chi2", args=(2,)).statistic),
    }


def gp_formulae(ln_y: float) -> dict[str, dict[str, float]]:
    """Grindhammer-Peters fluctuation values: sampling and homogeneous sets (hep-ex/0001020, verified)."""

    return {
        "sampling": {
            "sigma_ln_t": 1.0 / (-2.5 + 1.25 * ln_y),
            "sigma_ln_alpha": 1.0 / (-0.82 + 0.79 * ln_y),
            "rho": 0.784 - 0.023 * ln_y,
        },
        "homogeneous": {
            "sigma_ln_t": 1.0 / (-1.4 + 1.26 * ln_y),
            "sigma_ln_alpha": 1.0 / (-0.58 + 0.86 * ln_y),
            "rho": 0.705 - 0.023 * ln_y,
        },
    }


@lru_cache(maxsize=2)
def _model(regime: str):
    """DEC-001 in one regime (the electron model's `deposition` or the signal-level `sampling` regime)."""

    return load_electron_model("deposition" if regime == "deposition" else "readout")


def model_regime_values(energy_mev: float) -> dict[str, dict[str, float]]:
    """What DEC-001 itself uses at one energy, in each of its two regimes (read, never changed)."""

    out = {}
    for regime in ("deposition", "sampling"):
        model = _model(regime)
        mean_depth = model.longitudinal.shower_max_depth_x0(energy_mev)
        out[regime] = {
            "mean_ln_t": float(model.lognormal_location(energy_mev)),
            "sigma_ln_t": float(model.fluctuation_width(energy_mev)),
            "mean_depth_x0": float(mean_depth),
            "backbone_alpha": float(1.0 + BETA_AMS * mean_depth),
        }
    return out


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------


@dataclass
class EnergyData:
    energy_gev: float
    fractions: dict[str, np.ndarray]  # representation -> (n, 18), layer energy / primary energy
    grids: np.ndarray  # deposition grid (n, 18, 72)
    entry_x_mm: np.ndarray
    entry_y_mm: np.ndarray
    fits: dict[str, np.ndarray] = field(default_factory=dict)
    origins: dict[str, float] = field(default_factory=dict)


def load_energy(energy_gev: float) -> EnergyData:
    arrays = load_batch(DATA / f"E{energy_gev:g}GeV").arrays
    fractions = {
        name: arrays[key].astype(float).sum(axis=2) / (1000.0 * energy_gev)
        for name, key in REPRESENTATION_KEYS.items()
    }
    return EnergyData(
        energy_gev,
        fractions,
        arrays["deposit_grid_mev"].astype(float),
        arrays["entry_x_mm"],
        arrays["entry_y_mm"],
    )


# ----------------------------------------------------------------------
# A and B
# ----------------------------------------------------------------------


def readout_fits_for_residual(data: EnergyData, bounds: np.ndarray) -> np.ndarray:
    """The amplitude-free readout fits already computed for this energy."""

    return data.fits["readout"]


def joint_structure(data: EnergyData, bounds: np.ndarray) -> dict[str, Any]:
    """Parts A and B for one energy, with the deposition profile (and the readout diagnostic)."""

    fractions = data.fractions["deposition"]
    energy_mev = 1000.0 * data.energy_gev
    critical = _model("deposition").longitudinal.critical_energy_mev
    ln_y = float(np.log(energy_mev / critical))
    free = [fit_free(f, bounds) for f in fractions]
    least_squares_fit = np.array([p for p, _ in free])
    free_residual = np.array([r for _, r in free])
    cumulative_fit = np.array([fit_cumulative_fractions(f, bounds) for f in fractions])
    data.fits = {"least_squares": least_squares_fit, "cumulative": cumulative_fit}
    fixed_residual = np.array([fit_fixed_beta(f, bounds)[1] for f in fractions])
    out: dict[str, Any] = {
        "n_events": len(fractions),
        "critical_energy_mev": critical,
        "ln_y": ln_y,
        "gp": gp_formulae(ln_y),
        "dec001": model_regime_values(energy_mev),
        "estimators": {},
    }
    gp = out["gp"]
    for name, fit in data.fits.items():
        ln_t, ln_alpha = fit[:, 0], fit[:, 1]
        beta = beta_values(ln_t, ln_alpha)
        entry: dict[str, Any] = {
            "ln_t": marginal_diagnostics(ln_t),
            "ln_alpha": marginal_diagnostics(ln_alpha),
            "t": marginal_diagnostics(np.exp(ln_t)),
            "alpha": marginal_diagnostics(np.exp(ln_alpha)),
            "beta": {
                **marginal_diagnostics(beta),
                "quantiles_5_16_50_84_95": np.quantile(beta, [0.05, 0.16, 0.5, 0.84, 0.95]).tolist(),
                "fraction_within_10pct_of_0.65": float(np.mean(np.abs(beta / BETA_AMS - 1.0) < 0.10)),
                "fraction_within_20pct_of_0.65": float(np.mean(np.abs(beta / BETA_AMS - 1.0) < 0.20)),
            },
            "joint": joint_diagnostics(ln_t, ln_alpha),
        }
        sd_t, sd_a = entry["ln_t"]["sd"], entry["ln_alpha"]["sd"]
        entry["ratio_to_gp"] = {
            kind: {
                "sigma_ln_t": sd_t / gp[kind]["sigma_ln_t"],
                "sigma_ln_alpha": sd_a / gp[kind]["sigma_ln_alpha"],
                "rho_minus_gp": entry["joint"]["pearson"] - gp[kind]["rho"],
            }
            for kind in ("sampling", "homogeneous")
        }
        entry["ratio_to_dec001"] = {
            regime: {
                "sigma_ln_t": sd_t / out["dec001"][regime]["sigma_ln_t"],
                "mean_ln_t_minus_model": entry["ln_t"]["mean"] - out["dec001"][regime]["mean_ln_t"],
            }
            for regime in ("deposition", "sampling")
        }
        out["estimators"][name] = entry
    ratio = fixed_residual / np.maximum(free_residual, 1e-12)
    out["fixed_beta_penalty"] = {
        "median_residual_ratio_fixed_over_free": float(np.median(ratio)),
        "q90_residual_ratio_fixed_over_free": float(np.quantile(ratio, 0.9)),
        "median_free_residual_norm": float(np.median(free_residual)),
        "median_fixed_residual_norm": float(np.median(fixed_residual)),
    }
    readout = np.array([fit_with_amplitude(f, bounds) for f in data.fractions["readout"]])
    data.fits["readout"] = readout
    fixed_readout = np.array([fit_fixed_beta_with_amplitude(f, bounds)[2] for f in data.fractions["readout"]])
    free_readout = np.array(
        [
            np.linalg.norm(np.exp(p[2]) * profile_fractions(p[0], p[1], bounds) - f)
            for p, f in zip(readout_fits_for_residual(data, bounds), data.fractions["readout"], strict=True)
        ]
    )
    out["readout_diagnostic"] = {
        "note": "diagnostic only (F3: calibration is deposition-only): amplitude-free fit of the OBSERVED fibre profile",
        "ln_t": marginal_diagnostics(readout[:, 0]),
        "ln_alpha": marginal_diagnostics(readout[:, 1]),
        "beta": marginal_diagnostics(beta_values(readout[:, 0], readout[:, 1])),
        "pearson_ln_t_ln_alpha": float(np.corrcoef(readout[:, 0], readout[:, 1])[0, 1]),
        "median_amplitude": float(np.median(np.exp(readout[:, 2]))),
        "fixed_beta_0.65_penalty_median_residual_ratio": float(
            np.median(fixed_readout / np.maximum(free_readout, 1e-12))
        ),
        "fixed_beta_0.65_penalty_q90_residual_ratio": float(
            np.quantile(fixed_readout / np.maximum(free_readout, 1e-12), 0.9)
        ),
    }
    out["origin_refit"] = origin_refit(data, bounds)
    return out


def origin_refit(data: EnergyData, bounds: np.ndarray) -> dict[str, Any]:
    """Per-event fits with the depth origin shifted to the value the MEAN profile asks for.

    The origin comes from the Geant4 mean profile (a fixed-beta fit with a free origin, and a free
    gamma fit with a free origin). A fixed front-face origin may be what makes beta look like 0.52;
    this refits every event at the shifted origin and reports where beta, the spreads and the
    correlation then sit.
    """

    mean_fraction = data.fractions["deposition"].mean(axis=0)
    origins = {
        "mean_profile_fixed_beta": fit_mean_profile(mean_fraction, bounds, "fixed_beta_origin")[
            "parameters"
        ][1],
        "mean_profile_free_gamma": fit_mean_profile(mean_fraction, bounds, "gamma_free_origin")[
            "parameters"
        ][2],
    }
    data.origins = origins
    out: dict[str, Any] = {"origins_x0": origins}
    for name, origin in origins.items():
        free = [fit_free_with_origin(f, bounds, origin) for f in data.fractions["deposition"]]
        fit = np.array([p for p, _ in free])
        free_residual = np.array([r for _, r in free])
        fixed = [fit_fixed_beta_with_origin(f, bounds, origin) for f in data.fractions["deposition"]]
        fixed_ln_t = np.array([x for x, _ in fixed])
        fixed_residual = np.array([r for _, r in fixed])
        data.fits[f"origin_{name}"] = fit
        data.fits[f"origin_{name}_fixed_beta_ln_t"] = fixed_ln_t
        beta = beta_values(fit[:, 0], fit[:, 1])
        ratio = fixed_residual / np.maximum(free_residual, 1e-12)
        out[name] = {
            "ln_t": marginal_diagnostics(fit[:, 0]),
            "ln_alpha": marginal_diagnostics(fit[:, 1]),
            "beta": {
                **marginal_diagnostics(beta),
                "quantiles_5_16_50_84_95": np.quantile(beta, [0.05, 0.16, 0.5, 0.84, 0.95]).tolist(),
                "fraction_within_10pct_of_0.65": float(np.mean(np.abs(beta / BETA_AMS - 1.0) < 0.10)),
                "fraction_within_20pct_of_0.65": float(np.mean(np.abs(beta / BETA_AMS - 1.0) < 0.20)),
            },
            "joint": joint_diagnostics(fit[:, 0], fit[:, 1]),
            "fixed_beta_penalty_median_residual_ratio": float(np.median(ratio)),
            "fixed_beta_penalty_q90_residual_ratio": float(np.quantile(ratio, 0.9)),
            "ln_t_with_beta_fixed": marginal_diagnostics(fixed_ln_t),
            "median_free_residual_norm": float(np.median(free_residual)),
        }
    return out


# ----------------------------------------------------------------------
# C
# ----------------------------------------------------------------------


def dec001_ensemble(energy_gev: float, regime: str, n: int, seed: int) -> np.ndarray:
    """Layer fractions of ``n`` DEC-001 events through its public sampling API."""

    model = _model(regime)
    rng = np.random.default_rng(seed)
    return np.array([model.sample_layer_energy_fractions(1000.0 * energy_gev, rng) for _ in range(n)])


def ams_mean_gp_fluctuation(
    energy_gev: float, regime: str, bounds: np.ndarray, n: int, seed: int
) -> np.ndarray:
    """DEC-001's own ``T`` distribution, ``alpha`` restored as a correlated second variable.

    ``ln T`` is DEC-001's (its regime's mean and width); ``ln alpha`` has the Grindhammer-Peters
    sampling sigma and rho, and its mean is set so that the DETERMINISTIC mean profile keeps the
    AMS backbone ``alpha = 1 + 0.65 T_mean`` (``<ln alpha> = ln(alpha_backbone) - sigma^2 / 2``).
    """

    values = model_regime_values(1000.0 * energy_gev)[regime]
    critical = _model("deposition").longitudinal.critical_energy_mev
    gp = gp_formulae(float(np.log(1000.0 * energy_gev / critical)))["sampling"]
    rng = np.random.default_rng(seed)
    z_t, z_perp = rng.standard_normal(n), rng.standard_normal(n)
    ln_t = values["mean_ln_t"] + values["sigma_ln_t"] * z_t
    mean_ln_alpha = np.log(values["backbone_alpha"]) - 0.5 * gp["sigma_ln_alpha"] ** 2
    ln_alpha = mean_ln_alpha + gp["sigma_ln_alpha"] * (
        gp["rho"] * z_t + np.sqrt(1.0 - gp["rho"] ** 2) * z_perp
    )
    return profile_fractions_batch(ln_t, ln_alpha, bounds)


def origin_variants(
    data: EnergyData, bounds: np.ndarray, n: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Variants whose depth origin is the mean-profile fixed-beta origin (AMS beta = 0.65 plus an origin offset)."""

    origin = data.origins["mean_profile_fixed_beta"]
    fixed_ln_t = data.fits["origin_mean_profile_fixed_beta_fixed_beta_ln_t"]
    free_fit = data.fits["origin_mean_profile_fixed_beta"]
    ln_t_only = fixed_ln_t.mean() + fixed_ln_t.std() * rng.standard_normal(n)
    joint = rng.multivariate_normal(free_fit.mean(axis=0), np.cov(free_fit.T), size=n)
    return {
        "fixed_beta_origin_data_T": profile_fixed_beta(ln_t_only, bounds, BETA_AMS, origin),
        "joint_origin_data": profile_fractions_batch(joint[:, 0], joint[:, 1], bounds, origin),
    }


def counterfactual_ensembles(data: EnergyData, bounds: np.ndarray, n: int, seed: int) -> dict[str, np.ndarray]:
    fit = data.fits["least_squares"]
    mean, cov = fit.mean(axis=0), np.cov(fit.T)
    rng = np.random.default_rng(seed)
    ln_t_only = mean[0] + np.sqrt(cov[0, 0]) * rng.standard_normal(n)
    joint = rng.multivariate_normal(mean, cov, size=n)
    return {
        "dec001_deposition": dec001_ensemble(data.energy_gev, "deposition", n, seed),
        "dec001_sampling": dec001_ensemble(data.energy_gev, "sampling", n, seed),
        "fixed_beta_data_T": profile_fixed_beta(ln_t_only, bounds),
        # the data's own median beta, held fixed per event: separates the effect of the MEAN beta
        # (0.65 against the measured value) from the effect of letting beta fluctuate
        "fixed_beta_matched_mean": profile_fixed_beta(
            ln_t_only, bounds, float(np.median(beta_values(fit[:, 0], fit[:, 1])))
        ),
        "joint_data": profile_fractions_batch(joint[:, 0], joint[:, 1], bounds),
        **origin_variants(data, bounds, n, rng),
        "joint_ams_mean_gp_fluct": ams_mean_gp_fluctuation(data.energy_gev, "deposition", bounds, n, seed),
        "joint_ams_mean_gp_fluct_sampling_T": ams_mean_gp_fluctuation(
            data.energy_gev, "sampling", bounds, n, seed
        ),
    }


def longitudinal_metrics(fractions: np.ndarray, reference: np.ndarray, z_mm: np.ndarray) -> dict[str, Any]:
    """The comparison of one ensemble of layer fractions with the Geant4 fractions."""

    def centre_and_width(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        total = np.maximum(f.sum(axis=1), 1e-12)
        centre = (f * z_mm).sum(axis=1) / total
        width = np.sqrt((f * (z_mm[None, :] - centre[:, None]) ** 2).sum(axis=1) / total)
        return centre, width

    contained, ref_contained = fractions.sum(axis=1), reference.sum(axis=1)
    centre, width = centre_and_width(fractions)
    ref_centre, ref_width = centre_and_width(reference)
    mean_ratio = fractions.mean(axis=0) / reference.mean(axis=0)
    relative_error = np.abs(mean_ratio - 1.0)
    correlation_difference = lag_averaged_correlation(
        fractions, CORRELATION_LAGS
    ) - lag_averaged_correlation(reference, CORRELATION_LAGS)
    return {
        "mean_fraction_by_layer": fractions.mean(axis=0).tolist(),
        "mean_ratio_to_geant4_by_layer": mean_ratio.tolist(),
        "layer0_mean_ratio": float(mean_ratio[0]),
        "early4_mean_ratio": float(fractions[:, :4].sum(axis=1).mean() / reference[:, :4].sum(axis=1).mean()),
        "profile_error_layers_0_3": float(relative_error[:4].mean()),
        "profile_error_layers_4_17": float(relative_error[4:].mean()),
        "leakage_mean": float(1.0 - contained.mean()),
        "leakage_mean_geant4": float(1.0 - ref_contained.mean()),
        "leakage_sd": float(contained.std()),
        "leakage_sd_geant4": float(ref_contained.std()),
        "contained_quantiles_5_50_95": np.quantile(contained, [0.05, 0.5, 0.95]).tolist(),
        "contained_quantiles_5_50_95_geant4": np.quantile(ref_contained, [0.05, 0.5, 0.95]).tolist(),
        "mean_argmax_layer": float(fractions.argmax(axis=1).mean()),
        "mean_argmax_layer_geant4": float(reference.argmax(axis=1).mean()),
        "cog_mm_mean_sd": [float(centre.mean()), float(centre.std())],
        "cog_mm_mean_sd_geant4": [float(ref_centre.mean()), float(ref_centre.std())],
        "rms_mm_mean_sd": [float(width.mean()), float(width.std())],
        "rms_mm_mean_sd_geant4": [float(ref_width.mean()), float(ref_width.std())],
        "layer_sd_ratio_to_geant4": (fractions.std(axis=0) / reference.std(axis=0)).tolist(),
        "lag_correlation_difference": correlation_difference.tolist(),
        "ks_contained": float(stats.ks_2samp(contained, ref_contained).statistic),
        "ks_layer0": float(stats.ks_2samp(fractions[:, 0], reference[:, 0]).statistic),
        "ks_cog": float(stats.ks_2samp(centre, ref_centre).statistic),
        "ks_rms": float(stats.ks_2samp(width, ref_width).statistic),
    }


DISTANCES = {
    "layer0_log_ratio": lambda m: abs(np.log(m["layer0_mean_ratio"])),
    "early4_log_ratio": lambda m: abs(np.log(m["early4_mean_ratio"])),
    "profile_error_layers_0_3": lambda m: m["profile_error_layers_0_3"],
    "profile_error_layers_4_17": lambda m: m["profile_error_layers_4_17"],
    "leakage_mean_difference": lambda m: abs(m["leakage_mean"] - m["leakage_mean_geant4"]),
    "leakage_sd_log_ratio": lambda m: abs(np.log(m["leakage_sd"] / m["leakage_sd_geant4"])),
    "ks_contained": lambda m: m["ks_contained"],
    "ks_cog": lambda m: m["ks_cog"],
    "ks_rms": lambda m: m["ks_rms"],
    "max_abs_lag_correlation_difference": lambda m: float(np.abs(m["lag_correlation_difference"]).max()),
}


def distances(metrics: dict[str, Any]) -> dict[str, float]:
    return {name: float(fn(metrics)) for name, fn in DISTANCES.items()}


NOISE_FLOORS = {
    # a two-sample KS between ~1000 and ~4000 events is 0.05 at the 95% level by chance alone
    "ks_contained": 0.05,
    "ks_cog": 0.05,
    "ks_rms": 0.05,
    "layer0_log_ratio": 0.05,
    "early4_log_ratio": 0.05,
    "profile_error_layers_0_3": 0.05,
    "profile_error_layers_4_17": 0.02,
    "leakage_mean_difference": 0.005,
    "leakage_sd_log_ratio": 0.05,
    "max_abs_lag_correlation_difference": 0.05,
}


def removed_fraction(base: dict[str, float], variant: dict[str, float]) -> dict[str, float | None]:
    """Fraction of the base discrepancy a variant removes (1 = all, 0 = none, negative = worse).

    ``None`` where the base discrepancy is already below its sampling-noise floor
    (``NOISE_FLOORS``): there is nothing to remove and a ratio would be noise.
    """

    return {
        name: (
            None
            if base[name] < max(NOISE_FLOORS.get(name, 0.0), 1e-12)
            else float(1.0 - variant[name] / base[name])
        )
        for name in base
    }


def counterfactual_comparison(
    data: EnergyData, bounds: np.ndarray, z_mm: np.ndarray, seed: int, n_events: int
) -> dict[str, Any]:
    ensembles = counterfactual_ensembles(data, bounds, n_events, seed)
    reference = data.fractions["deposition"]
    metrics = {name: longitudinal_metrics(f, reference, z_mm) for name, f in ensembles.items()}
    dist = {name: distances(m) for name, m in metrics.items()}
    return {
        "n_events_per_variant": n_events,
        "metrics": metrics,
        "distances_to_geant4": dist,
        "removed_vs_dec001_deposition": {
            name: removed_fraction(dist["dec001_deposition"], d) for name, d in dist.items()
        },
        # same ln T marginal, only the treatment of alpha differs: the causal pairs
        "alpha_effect_data_T": removed_fraction(dist["fixed_beta_data_T"], dist["joint_data"]),
        # of which: the mean-beta part and the fluctuation part (same ln T marginal throughout)
        "mean_beta_effect_only": removed_fraction(dist["fixed_beta_data_T"], dist["fixed_beta_matched_mean"]),
        "fluctuation_effect_only": removed_fraction(dist["fixed_beta_matched_mean"], dist["joint_data"]),
        "alpha_effect_dec001_T": removed_fraction(dist["dec001_deposition"], dist["joint_ams_mean_gp_fluct"]),
        "_ensembles": ensembles,
    }


# ----------------------------------------------------------------------
# D
# ----------------------------------------------------------------------

READOUT_MEAN_PROFILE_MODELS = (
    "gamma_free_amplitude",
    "fixed_beta_amplitude",
    "fixed_beta_amplitude_origin",
)
MEAN_PROFILE_MODELS = (
    "gamma_free",
    "gamma_free_origin",
    "gamma_free_amplitude",
    "fixed_beta",
    "fixed_beta_origin",
)


def fit_mean_profile(mean_fraction: np.ndarray, bounds: np.ndarray, model: str) -> dict[str, Any]:
    """Structural fits of the Geant4 ensemble-MEAN profile (F4 items: alpha, origin, normalisation, beta)."""

    def build(p: np.ndarray) -> np.ndarray:
        if model == "gamma_free":
            return profile_general(bounds, p[0], p[1])
        if model == "gamma_free_origin":
            return profile_general(bounds, p[0], p[1], origin=p[2])
        if model == "gamma_free_amplitude":
            return profile_general(bounds, p[0], p[1], amplitude=np.exp(p[2]))
        if model == "fixed_beta":
            return profile_general(bounds, p[0], beta=BETA_AMS)
        if model == "fixed_beta_amplitude":
            return profile_general(bounds, p[0], amplitude=np.exp(p[1]), beta=BETA_AMS)
        if model == "fixed_beta_amplitude_origin":
            return profile_general(bounds, p[0], origin=p[2], amplitude=np.exp(p[1]), beta=BETA_AMS)
        return profile_general(bounds, p[0], origin=p[1], beta=BETA_AMS)

    amplitude_start = float(np.log(max(mean_fraction.sum(), 1e-9) / 0.9))
    start, low, high = {
        "gamma_free": ((1.9, 1.6), (-1.0, 0.05), (4.0, 4.0)),
        "gamma_free_origin": ((1.9, 1.6, 0.0), (-1.0, 0.05, -4.0), (4.0, 4.0, 4.0)),
        "gamma_free_amplitude": ((1.9, 1.6, amplitude_start), (-1.0, 0.05, -12.0), (4.0, 4.0, 2.0)),
        "fixed_beta": ((1.9,), (-1.0,), (4.0,)),
        "fixed_beta_amplitude": ((1.9, amplitude_start), (-1.0, -12.0), (4.0, 2.0)),
        "fixed_beta_amplitude_origin": ((1.9, amplitude_start, 0.0), (-1.0, -12.0, -4.0), (4.0, 2.0, 4.0)),
        "fixed_beta_origin": ((1.9, 0.0), (-1.0, -4.0), (4.0, 4.0)),
    }[model]
    result = least_squares(lambda p: build(p) - mean_fraction, start, bounds=(low, high))
    fitted = build(result.x)
    ratio = mean_fraction / np.maximum(fitted, 1e-12)
    return {
        "parameters": result.x.tolist(),
        "rms_residual": float(np.sqrt(np.mean((fitted - mean_fraction) ** 2))),
        "geant4_over_fit_by_layer_0_5": ratio[:6].tolist(),
        "layer0_geant4_over_fit": float(ratio[0]),
    }


def mean_backbone(data: EnergyData, ensembles: dict[str, np.ndarray], bounds: np.ndarray) -> dict[str, Any]:
    """Part D: AMS backbone (deterministic, alpha = 1 + 0.65 T_mean) against ensemble means and Geant4."""

    energy_mev = 1000.0 * data.energy_gev
    backbone = np.array(_model("deposition").longitudinal.layer_energy_fractions(energy_mev))
    geant4 = data.fractions["deposition"].mean(axis=0)
    joint = ensembles["joint_ams_mean_gp_fluct"].mean(axis=0)
    dec001 = ensembles["dec001_deposition"].mean(axis=0)
    return {
        "energy_mev_layers_0_3_geant4_mean_sd": [
            [
                float(energy_mev * data.fractions["deposition"][:, i].mean()),
                float(energy_mev * data.fractions["deposition"][:, i].std()),
            ]
            for i in range(4)
        ],
        "energy_mev_layers_0_3_backbone": (energy_mev * backbone[:4]).tolist(),
        "backbone_fraction_by_layer": backbone.tolist(),
        "geant4_mean_fraction_by_layer": geant4.tolist(),
        "ensemble_mean_joint_fraction_by_layer": joint.tolist(),
        "ensemble_mean_dec001_fraction_by_layer": dec001.tolist(),
        "joint_over_backbone_by_layer": (joint / backbone).tolist(),
        "dec001_ensemble_over_backbone_by_layer": (dec001 / backbone).tolist(),
        "geant4_over_backbone_by_layer": (geant4 / backbone).tolist(),
        "geant4_over_joint_ensemble_by_layer": (geant4 / joint).tolist(),
        "max_abs_joint_over_backbone_minus_1": float(np.abs(joint / backbone - 1.0).max()),
        "structural_fits": {m: fit_mean_profile(geant4, bounds, m) for m in MEAN_PROFILE_MODELS},
        # the AMS statement that b = 0.65 refers to a fit of OBSERVED signals: the same test on the
        # readout (fibre) mean profile, whose scale is free (diagnostic only; F3)
        "readout_structural_fits": {
            m: fit_mean_profile(data.fractions["readout"].mean(axis=0), bounds, m)
            for m in READOUT_MEAN_PROFILE_MODELS
        },
    }


# ----------------------------------------------------------------------
# E
# ----------------------------------------------------------------------


def partial_spearman(a: np.ndarray, b: np.ndarray, covariates: np.ndarray | None) -> float:
    """Partial Spearman correlation of ``a`` and ``b`` given ``covariates``.

    All variables, the covariates too, are replaced by their ranks; the ranks of ``a`` and ``b`` are
    regressed on the ranked covariates and the residuals correlated (the textbook definition).
    """

    ra, rb = stats.rankdata(a).astype(float), stats.rankdata(b).astype(float)
    if covariates is None or covariates.size == 0:
        return float(np.corrcoef(ra, rb)[0, 1])
    ranked = np.column_stack(
        [stats.rankdata(column) for column in np.asarray(covariates, dtype=float).reshape(len(ra), -1).T]
    )
    design = np.column_stack([np.ones(len(ra)), ranked])
    ea = ra - design @ np.linalg.lstsq(design, ra, rcond=None)[0]
    eb = rb - design @ np.linalg.lstsq(design, rb, rcond=None)[0]
    return float(np.corrcoef(ea, eb)[0, 1])


def phase_covariates(data: EnergyData, geometry) -> dict[str, np.ndarray]:
    """Entry-cell phase (offset of the shower axis from its cell centre) and edge proximity, per event."""

    centres = cell_centres_mm(geometry)
    index_x = cell_indices(data.entry_x_mm, geometry)
    index_y = cell_indices(data.entry_y_mm, geometry)
    phase_x = data.entry_x_mm - centres[index_x]
    phase_y = data.entry_y_mm - centres[index_y]
    n_cells = len(centres)
    return {
        "phase_x_mm": phase_x,
        "phase_y_mm": phase_y,
        "cells_from_border": np.minimum.reduce(
            [index_x, n_cells - 1 - index_x, index_y, n_cells - 1 - index_y]
        ),
        "design": np.column_stack(
            [np.abs(phase_x), np.abs(phase_y), phase_x**2, phase_y**2, np.abs(phase_x) * np.abs(phase_y)]
        ),
    }


def confounder_analysis(
    data: EnergyData, geometry, rng: np.random.Generator, n_boot: int
) -> dict[str, Any]:
    fit = data.fits["least_squares"]
    observables = crossing_observables(data.grids, data.entry_x_mm, data.entry_y_mm, geometry)
    cov = phase_covariates(data, geometry)
    rear_leakage = 1.0 - data.fractions["deposition"].sum(axis=1)
    phase_only = cov["design"]
    with_leakage = np.column_stack([phase_only, rear_leakage])
    strength = np.abs(cov["phase_x_mm"]) + np.abs(cov["phase_y_mm"])
    strata = np.digitize(strength, np.quantile(strength, [1 / 3, 2 / 3]))
    out: dict[str, Any] = {
        "min_cells_from_border": int(cov["cells_from_border"].min()),
        "phase_range_mm": [float(cov["phase_x_mm"].min()), float(cov["phase_x_mm"].max())],
        "rear_leakage_mean_sd": [float(rear_leakage.mean()), float(rear_leakage.std())],
        "ln_t_leakage_spearman": float(stats.spearmanr(fit[:, 0], rear_leakage).statistic),
        "observables": {},
    }
    for name in LATERAL_OBSERVABLES:
        obs = observables[name]
        row: dict[str, Any] = {
            "observable_phase_spearman": float(stats.spearmanr(np.abs(cov["phase_x_mm"]), obs).statistic)
        }
        for label, column in (("ln_t", fit[:, 0]), ("ln_alpha", fit[:, 1])):
            other = fit[:, 1] if label == "ln_t" else fit[:, 0]
            boot = []
            for _ in range(n_boot):
                pick = rng.integers(0, len(obs), len(obs))
                boot.append(partial_spearman(column[pick], obs[pick], phase_only[pick]))
            row[label] = {
                "raw_spearman": partial_spearman(column, obs, None),
                "partial_phase": partial_spearman(column, obs, phase_only),
                "partial_phase_bootstrap_95": np.quantile(boot, [0.025, 0.975]).tolist(),
                "partial_phase_and_rear_leakage": partial_spearman(column, obs, with_leakage),
                "partial_phase_and_other_parameter": partial_spearman(
                    column, obs, np.column_stack([phase_only, other])
                ),
                "stratified_by_phase_spearman": [
                    float(stats.spearmanr(column[strata == s], obs[strata == s]).statistic)
                    for s in range(3)
                ],
            }
        out["observables"][name] = row
    return out


# ----------------------------------------------------------------------
# Orchestration and output
# ----------------------------------------------------------------------


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def energy_trend_of_beta(per_energy: dict[str, Any]) -> dict[str, Any]:
    """Part B: is the event-level beta distribution energy dependent?"""

    energies = np.array([float(e) for e in per_energy])
    medians = np.array(
        [per_energy[e]["estimators"]["least_squares"]["beta"]["median"] for e in per_energy]
    )
    if len(energies) < 2:
        return {"energies_gev": energies.tolist(), "median_beta_least_squares": medians.tolist(), "slope_per_ln_energy": None, "intercept": None}
    slope, intercept = np.polyfit(np.log(energies), medians, 1)
    return {
        "energies_gev": energies.tolist(),
        "median_beta_least_squares": medians.tolist(),
        "slope_per_ln_energy": float(slope),
        "intercept": float(intercept),
    }


def analyse(
    energies: Sequence[float] = ENERGIES_GEV,
    *,
    n_counterfactual: int = COUNTERFACTUAL_EVENTS,
    n_boot: int = BOOTSTRAP,
) -> tuple[dict[str, Any], dict[float, EnergyData], dict[float, dict[str, np.ndarray]]]:
    bounds = layer_bounds_x0()
    geometry = load_geometry(GEOMETRY_CONFIG)
    z_mm = layer_centres_mm(geometry, 18)
    results: dict[str, Any] = {}
    loaded: dict[float, EnergyData] = {}
    ensembles: dict[float, dict[str, np.ndarray]] = {}
    for energy in energies:
        data = load_energy(energy)
        entry: dict[str, Any] = {"A_B_joint_structure": joint_structure(data, bounds)}
        comparison = counterfactual_comparison(data, bounds, z_mm, int(1000 * energy), n_counterfactual)
        ensembles[energy] = comparison["_ensembles"]
        entry["C_counterfactual"] = comparison
        entry["D_mean_backbone"] = mean_backbone(data, comparison["_ensembles"], bounds)
        entry["E_confounders"] = confounder_analysis(
            data, geometry, np.random.default_rng(int(1000 * energy) + 1), n_boot
        )
        results[f"{energy:g}"] = entry
        loaded[energy] = data
    results["B_beta_energy_trend"] = energy_trend_of_beta(
        {k: v["A_B_joint_structure"] for k, v in results.items() if not k.startswith("B_")}
    )
    return results, loaded, ensembles


def write_tables(results: dict[str, Any], directory: Path) -> None:
    energies = [k for k in results if not k.startswith("B_")]
    with (directory / "longitudinal_structure_joint.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["energy_gev", "estimator", "mean_ln_t", "sd_ln_t", "mean_ln_alpha", "sd_ln_alpha", "pearson",
             "spearman", "skew_ln_t", "skew_ln_alpha", "beta_median", "beta_sd", "beta_q05", "beta_q95",
             "sd_ln_t_over_gp_sampling", "sd_ln_t_over_gp_homogeneous", "sd_ln_alpha_over_gp_sampling",
             "rho_minus_gp_sampling"]
        )
        for e in energies:
            block = results[e]["A_B_joint_structure"]
            for name, est in block["estimators"].items():
                q = est["beta"]["quantiles_5_16_50_84_95"]
                writer.writerow(
                    [e, name, est["ln_t"]["mean"], est["ln_t"]["sd"], est["ln_alpha"]["mean"],
                     est["ln_alpha"]["sd"], est["joint"]["pearson"], est["joint"]["spearman"],
                     est["ln_t"]["skewness"], est["ln_alpha"]["skewness"], est["beta"]["median"],
                     est["beta"]["sd"], q[0], q[4], est["ratio_to_gp"]["sampling"]["sigma_ln_t"],
                     est["ratio_to_gp"]["homogeneous"]["sigma_ln_t"],
                     est["ratio_to_gp"]["sampling"]["sigma_ln_alpha"],
                     est["ratio_to_gp"]["sampling"]["rho_minus_gp"]]
                )
    with (directory / "longitudinal_structure_counterfactual_distances.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        names = list(DISTANCES)
        writer.writerow(["energy_gev", "variant", *names])
        for e in energies:
            for variant, d in results[e]["C_counterfactual"]["distances_to_geant4"].items():
                writer.writerow([e, variant, *[d[n] for n in names]])
    with (directory / "longitudinal_structure_confounders.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["energy_gev", "observable", "parameter", "raw", "partial_phase", "ci_low", "ci_high",
             "partial_phase_leakage", "partial_phase_other_parameter", "stratum_low", "stratum_mid",
             "stratum_high"]
        )
        for e in energies:
            for obs, row in results[e]["E_confounders"]["observables"].items():
                for parameter in ("ln_t", "ln_alpha"):
                    r = row[parameter]
                    writer.writerow(
                        [e, obs, parameter, r["raw_spearman"], r["partial_phase"],
                         r["partial_phase_bootstrap_95"][0], r["partial_phase_bootstrap_95"][1],
                         r["partial_phase_and_rear_leakage"], r["partial_phase_and_other_parameter"],
                         *r["stratified_by_phase_spearman"]]
                    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ANALYSIS_JSON)
    parser.add_argument("--plots", type=Path, default=ANALYSIS_PLOTS)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--counterfactual-events", type=int, default=COUNTERFACTUAL_EVENTS)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    args = parser.parse_args(argv)
    results, loaded, ensembles = analyse(
        n_counterfactual=args.counterfactual_events, n_boot=args.bootstrap
    )
    document = {
        "status": (
            "ANALYSIS ONLY (longitudinal structure) on the EXPOSED Geant4 electrons (development / calibration data). No "
            "generator was changed; no sealed data was read. Not a validation."
        ),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "variants": list(VARIANTS),
        **_clean(results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2), encoding="utf-8")
    write_tables(results, args.out.parent)
    if not args.no_plots:
        from ams_ecal.electron_studies.em_longitudinal_structure_plots import make_plots

        make_plots(results, loaded, ensembles, args.plots)
    print(args.out)


if __name__ == "__main__":
    main()
