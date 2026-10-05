"""Scientific report of the Geant4 proton calibration pilot (steps 6-10).

Reads the stored batches and writes reproducible outputs - no notebook-only
analysis. Every threshold used here is an ANALYSIS CHOICE fixed before the
production data were read, expressed in units of a MIP scale MEASURED from
truth-non-interacting baseline events, and varied.

    uv run python -m ams_ecal.geant4_simulation.pilot_report

writes ``results/geant4_proton_pilot/summary.json``, CSV tables and figures.

Question -> section:

* interaction      - inelastic fraction, effective interaction length, and
                     whether the depth is exponential (vs exp(-0.6)).
* detector         - how non-interacting protons look; which interacting
                     events are indistinguishable from them (MIP band).
* p7               - share of each observable's variance associated with the
                     first-interaction depth D (S_D, law of total variance).
* truncation       - same-event correlations in the 0.6 lambda_I prefix vs the
                     extended calorimeter, event ordering, and backsplash.
* physics_list     - FTFP_BERT vs QBBC on the quantities that would shape proton model.
* high_energy_model - FTFP_BERT vs QGSP_BERT: QBBC shares FTFP for protons
                     above 3 GeV, so only this varies the first-interaction model.
* fixed_entry      - uniform vs fixed entry: entry phase against the fibres.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from ams_ecal.detector.geometry import load_geometry
from ams_ecal.geant4_simulation.geant4_backend import PROJECT_ROOT, Batch, load_batch
from ams_ecal.geant4_simulation.pilot_analysis import (
    bootstrap_interval,
    censored_exponential_rate,
    conditional_variance_fraction,
    event_observables,
    mip_scale,
    pearson,
    spearman,
    wilson_interval,
)

DATA = PROJECT_ROOT / "data" / "geant4_proton_pilot"
RESULTS = PROJECT_ROOT / "results" / "geant4_proton_pilot"
PREFIX_MM = 166.5
N_BOOT = 400

# Pre-registered analysis choices (2026-09-29, before reading production data).
CELL_THRESHOLD_MIP = 0.5  # cell counts as hit above half a MIP cell deposit
LAYER_THRESHOLD_MIP = 0.5  # layer counts as active above half a MIP layer deposit
THRESHOLD_VARIATIONS = (0.25, 0.5, 1.0)
MIP_BAND_QUANTILES = (0.95, 0.99, 0.999)  # of non-interacting events
PRIMARY_BAND_QUANTILE = 0.99
P7_OBSERVABLES = (
    "energy_mev",
    "log_energy",
    "long_cog_mm",
    "long_rms_mm",
    "width_mm",
    "n_hit_cells",
    "n_active_layers",
)
P7_BINS = (5, 10, 20)
P7_PRIMARY_BINS = 10
TRUNCATION_OBSERVABLES = ("energy_mev", "long_cog_mm", "long_rms_mm", "width_mm", "n_hit_cells")


def _interval(i) -> dict[str, float]:
    return {"estimate": i.estimate, "low": i.low, "high": i.high}


def _load(sample: str, data: Path) -> dict[float, Batch]:
    batches = {}
    for directory in sorted((data / sample).glob("E*GeV")):
        batch = load_batch(directory)
        batches[float(batch.metadata["energy_gev"])] = batch
    return batches


class Pilot:
    """All stored samples plus the observables computed once from them."""

    def __init__(self, data: Path = DATA) -> None:
        self.geometry = load_geometry(PROJECT_ROOT / "configs" / "geometry.yaml")
        self.samples = {
            name: _load(name, data)
            for name in ("baseline", "fixed_entry", "alternate", "extended", "high_energy_model")
        }
        baseline = self.samples["baseline"]
        if not baseline:
            raise SystemExit(f"no baseline batches under {data}")

        # One MIP scale for the whole analysis, pooled over baseline energies.
        grids, xs, ys = [], [], []
        for batch in baseline.values():
            a = batch.arrays
            crossing = ~a["truth_occurred"]
            grids.append(a["readout_grid_mev"][crossing])
            xs.append(a["entry_x_mm"][crossing])
            ys.append(a["entry_y_mm"][crossing])
        self.mip = mip_scale(np.concatenate(grids), self.geometry, np.concatenate(xs), np.concatenate(ys))
        self._cache: dict[tuple, dict[str, np.ndarray]] = {}

    def observables(
        self, sample: str, energy: float, grid_key: str = "readout_grid_mev",
        layers: slice | None = None, threshold_mip: float = CELL_THRESHOLD_MIP,
    ) -> dict[str, np.ndarray]:
        key = (sample, energy, grid_key, None if layers is None else (layers.start, layers.stop), threshold_mip)
        if key not in self._cache:
            a = self.samples[sample][energy].arrays
            grid = a[grid_key] if layers is None else a[grid_key][:, layers]
            obs = event_observables(
                grid, self.geometry, a["entry_x_mm"], a["entry_y_mm"],
                cell_threshold_mev=threshold_mip * self.mip.cell_mev,
                layer_threshold_mev=threshold_mip * self.mip.layer_mev,
            )
            with np.errstate(divide="ignore"):
                obs["log_energy"] = np.log(obs["energy_mev"])
            self._cache[key] = obs
        return self._cache[key]


# ----------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------


def interaction_section(p: Pilot) -> dict[str, Any]:
    out = {}
    for energy, batch in p.samples["baseline"].items():
        a = batch.arrays
        occurred = a["truth_occurred"]
        depth = a["truth_z_mm"][occurred]
        fit = censored_exponential_rate(depth, int((~occurred).sum()), PREFIX_MM)
        lam = 1 / fit.rate_per_mm
        lam_err = fit.standard_error / fit.rate_per_mm**2
        # Is the depth exponential? KS against the fitted truncated exponential.
        truncated = stats.truncexpon(b=PREFIX_MM * fit.rate_per_mm, scale=lam)
        ks = stats.kstest(depth, truncated.cdf)
        materials = a["truth_material"][occurred]
        leading = a["truth_leading_kinetic_energy_mev"][occurred] / a[
            "truth_primary_kinetic_energy_before_mev"
        ][occurred]
        out[f"{energy:g}"] = {
            "n_events": len(occurred),
            "n_interacting": int(occurred.sum()),
            "p_no_inelastic": _interval(wilson_interval(int((~occurred).sum()), len(occurred))),
            "effective_interaction_length_mm": {"estimate": lam, "standard_error": lam_err},
            "effective_depth_lambda": PREFIX_MM / lam,
            "p_no_inelastic_from_fit": fit.survival(PREFIX_MM),
            "reference_exp_minus_0p6": float(np.exp(-0.6)),
            "reference_geant4_composite": float(
                np.exp(-batch.metadata["materials"]["prefix_depth_lambda_i"])
            ),
            "depth_ks_vs_exponential": {"statistic": float(ks.statistic), "p_value": float(ks.pvalue)},
            "interaction_in_fibre_fraction": float(np.mean(materials == "G4_POLYSTYRENE")),
            "fibre_volume_fraction": batch.metadata["materials"]["fibre_volume_fraction"],
            "fibre_mass_fraction": batch.metadata["materials"]["fibre_mass_fraction"],
            "mean_primary_elastic_scatters": float(a["truth_primary_elastic_scatters"].mean()),
            "first_interaction_multiplicity_median": float(np.median(a["truth_n_secondaries"][occurred])),
            "leading_fraction_quantiles": np.quantile(leading, [0.1, 0.5, 0.9]).tolist(),
            "quasi_elastic_like_fraction": float(np.mean(leading > 0.9)),
        }
    return out


def detector_section(p: Pilot) -> dict[str, Any]:
    out: dict[str, Any] = {
        "mip_scale_mev": {"cell": p.mip.cell_mev, "layer": p.mip.layer_mev},
        "thresholds_mev": {
            "cell": CELL_THRESHOLD_MIP * p.mip.cell_mev,
            "layer": LAYER_THRESHOLD_MIP * p.mip.layer_mev,
        },
    }
    for energy, batch in p.samples["baseline"].items():
        a = batch.arrays
        obs = p.observables("baseline", energy)
        occurred = a["truth_occurred"]
        e = obs["energy_mev"]
        e0 = energy * 1000.0
        layers = a["readout_grid_mev"].sum(axis=2)
        crossing_layers = layers[~occurred]
        entry = {
            "non_interacting": {
                "energy_mev_quantiles": np.quantile(e[~occurred], [0.05, 0.5, 0.95]).tolist(),
                "energy_mev_mean_sd": [float(e[~occurred].mean()), float(e[~occurred].std())],
                "deposit_total_mev_mean": float(a["edep_total_mev"][~occurred].mean()),
                "scintillator_over_total": float(
                    (a["edep_scintillator_mev"][~occurred] / a["edep_total_mev"][~occurred]).mean()
                ),
                "layer_energy_mean_mev": float(crossing_layers.mean()),
                "layer_to_layer_cv": float(
                    np.mean(crossing_layers.std(axis=1) / crossing_layers.mean(axis=1))
                ),
                "n_hit_cells_quantiles": np.quantile(obs["n_hit_cells"][~occurred], [0.05, 0.5, 0.95]).tolist(),
                "containment_mean": float(np.nanmean(obs["containment_fraction"][~occurred])),
            },
            "interacting": {
                "energy_mev_quantiles": np.quantile(e[occurred], [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
                "scintillator_over_e0_mean": float(e[occurred].mean() / e0),
                "deposit_over_e0_quantiles": (
                    np.quantile(a["edep_total_mev"][occurred], [0.05, 0.5, 0.95]) / e0
                ).tolist(),
                "bimodality_coefficient_log_energy": _bimodality(np.log(e[occurred])),
            },
            "mip_band": {},
        }
        crossing_e = e[~occurred]
        crossing_hits = obs["n_hit_cells"][~occurred]
        for q in MIP_BAND_QUANTILES:
            e_max = float(np.quantile(crossing_e, q))
            h_max = float(np.quantile(crossing_hits, q))
            in_band = e <= e_max
            joint = in_band & (obs["n_hit_cells"] <= h_max)
            n_int = int(occurred.sum())
            entry["mip_band"][f"{q:g}"] = {
                "energy_upper_mev": e_max,
                "hits_upper": h_max,
                "fraction_all_events_mip_like": _interval(wilson_interval(int(in_band.sum()), len(e))),
                "fraction_interacting_mip_like": _interval(wilson_interval(int((in_band & occurred).sum()), n_int)),
                "fraction_interacting_mip_like_energy_and_hits": _interval(
                    wilson_interval(int((joint & occurred).sum()), n_int)
                ),
                "mip_like_interacting_share_of_mip_like": float(
                    (in_band & occurred).sum() / max(in_band.sum(), 1)
                ),
            }
        # How MIP-like are interactions as a function of the depth left behind them?
        band = entry["mip_band"][f"{PRIMARY_BAND_QUANTILE:g}"]["energy_upper_mev"]
        edges = np.linspace(0, PREFIX_MM, 10)
        depth = a["truth_z_mm"]
        rows = []
        for lo, hi in pairwise(edges):
            sel = occurred & (depth >= lo) & (depth < hi)
            if sel.sum() >= 10:
                interval = wilson_interval(int((e[sel] <= band).sum()), int(sel.sum()))
                rows.append({"depth_low_mm": lo, "depth_high_mm": hi, "n": int(sel.sum()),
                             "fraction_mip_like": _interval(interval)})
        entry["mip_like_vs_depth"] = rows
        out[f"{energy:g}"] = entry
    return out


def _bimodality(x: np.ndarray) -> float:
    """Sarle's bimodality coefficient; above 5/9 hints at more than one mode."""

    n = len(x)
    g = stats.skew(x)
    k = stats.kurtosis(x)  # excess
    return float((g**2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def p7_section(p: Pilot) -> dict[str, Any]:
    out = {}
    for energy, batch in p.samples["baseline"].items():
        a = batch.arrays
        occurred = a["truth_occurred"]
        d = a["truth_z_mm"][occurred]
        obs = p.observables("baseline", energy)
        entry = {}
        for name in P7_OBSERVABLES:
            y = obs[name][occurred].astype(float)
            primary = conditional_variance_fraction(d, y, P7_PRIMARY_BINS)
            interval = bootstrap_interval(
                lambda dd, yy: conditional_variance_fraction(dd, yy, P7_PRIMARY_BINS).epsilon_squared,
                [d, y], n_boot=N_BOOT, seed=7,
            )
            entry[name] = {
                "s_d_epsilon_squared": _interval(interval),
                "s_d_eta_squared": primary.eta_squared,
                "binning_sensitivity": {
                    str(k): conditional_variance_fraction(d, y, k).epsilon_squared for k in P7_BINS
                },
                "bin_means": primary.bin_means.tolist(),
                "bin_edges_mm": primary.edges.tolist(),
                "spearman_with_depth": spearman(d, y),
            }
        out[f"{energy:g}"] = {"n_interacting": int(occurred.sum()), "observables": entry}
    return out


def truncation_section(p: Pilot) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for energy, batch in p.samples["extended"].items():
        a = batch.arrays
        d = a["truth_z_mm"]
        in_prefix = a["truth_occurred"] & (d < PREFIX_MM)
        thin = p.observables("extended", energy, "extended_readout_grid_mev", slice(0, 18))
        full = p.observables("extended", energy, "extended_readout_grid_mev")
        entry: dict[str, Any] = {"n_events": len(d), "n_interacting_in_prefix": int(in_prefix.sum())}
        correlations = {}
        dd = d[in_prefix]
        for name in TRUNCATION_OBSERVABLES:
            yt, yf = thin[name][in_prefix].astype(float), full[name][in_prefix].astype(float)
            ok = np.isfinite(yt) & np.isfinite(yf)
            t = bootstrap_interval(spearman, [dd[ok], yt[ok]], n_boot=N_BOOT, seed=11)
            f = bootstrap_interval(spearman, [dd[ok], yf[ok]], n_boot=N_BOOT, seed=11)
            diff = bootstrap_interval(
                lambda x, u, v: spearman(x, u) - spearman(x, v),
                [dd[ok], yt[ok], yf[ok]], n_boot=N_BOOT, seed=11,
            )
            correlations[name] = {
                "spearman_thin": _interval(t),
                "spearman_full": _interval(f),
                "thin_minus_full": _interval(diff),
                "pearson_thin": pearson(dd[ok], yt[ok]),
                "pearson_full": pearson(dd[ok], yf[ok]),
                "s_d_thin": conditional_variance_fraction(dd[ok], yt[ok], P7_PRIMARY_BINS).epsilon_squared,
                "s_d_full": conditional_variance_fraction(dd[ok], yf[ok], P7_PRIMARY_BINS).epsilon_squared,
            }
        entry["depth_correlations"] = correlations
        # Event ordering: does the thin view rank events like the full shower?
        entry["energy_rank_agreement_thin_vs_full"] = _interval(
            bootstrap_interval(
                spearman, [thin["energy_mev"][in_prefix], full["energy_mev"][in_prefix]],
                n_boot=N_BOOT, seed=13,
            )
        )
        # Developed showers that look MIP-like when only their start is seen.
        crossing_baseline = p.samples["baseline"].get(energy)
        if crossing_baseline is not None:
            ba = crossing_baseline.arrays
            band = float(np.quantile(
                p.observables("baseline", energy)["energy_mev"][~ba["truth_occurred"]],
                PRIMARY_BAND_QUANTILE,
            ))
            full_fraction = full["energy_mev"][in_prefix] / (energy * 1000.0)
            looks_mip = thin["energy_mev"][in_prefix] <= band
            entry["thin_mip_like_but_developed"] = {
                "band_mev": band,
                "fraction_of_prefix_interacting_mip_like_in_thin_view": _interval(
                    wilson_interval(int(looks_mip.sum()), int(in_prefix.sum()))
                ),
                "median_full_scintillator_fraction_of_those": float(np.median(full_fraction[looks_mip]))
                if looks_mip.any() else None,
            }
            entry["backsplash"] = _backsplash(p, energy, band)
        out[f"{energy:g}"] = entry
    return out


def _backsplash(p: Pilot, energy: float, band: float) -> dict[str, Any]:
    """Does material behind the ECAL change what the prefix sees?"""

    ext = p.samples["extended"][energy].arrays
    base = p.samples["baseline"][energy].arrays
    thin = p.observables("extended", energy, "extended_readout_grid_mev", slice(0, 18))["energy_mev"]
    ams = p.observables("baseline", energy)["energy_mev"]
    downstream = ~(ext["truth_occurred"] & (ext["truth_z_mm"] < PREFIX_MM))
    crossing = ~base["truth_occurred"]
    ks = stats.ks_2samp(thin[downstream], ams[crossing])
    return {
        "note": "prefix-non-interacting events: extended geometry vs AMS-only",
        "mean_prefix_energy_extended_mev": float(thin[downstream].mean()),
        "mean_prefix_energy_ams_only_mev": float(ams[crossing].mean()),
        "fraction_above_mip_band_extended": float(np.mean(thin[downstream] > band)),
        "fraction_above_mip_band_ams_only": float(np.mean(ams[crossing] > band)),
        "ks_statistic": float(ks.statistic),
        "ks_p_value": float(ks.pvalue),
    }


def comparison_section(p: Pilot, other: str) -> dict[str, Any]:
    """Compare a control sample with the baseline on proton model-relevant quantities."""

    out = {}
    for energy, batch in p.samples[other].items():
        if energy not in p.samples["baseline"]:
            continue
        a, b = batch.arrays, p.samples["baseline"][energy].arrays
        oa, ob = p.observables(other, energy), p.observables("baseline", energy)
        ia, ib = a["truth_occurred"], b["truth_occurred"]

        def summary(arr, obs, occurred):
            depth = arr["truth_z_mm"][occurred]
            fit = censored_exponential_rate(depth, int((~occurred).sum()), PREFIX_MM)
            return {
                "n": len(occurred),
                "p_no_inelastic": _interval(wilson_interval(int((~occurred).sum()), len(occurred))),
                "effective_interaction_length_mm": 1 / fit.rate_per_mm,
                "interacting_energy_quantiles_mev": np.quantile(obs["energy_mev"][occurred], [0.1, 0.5, 0.9]).tolist(),
                "interacting_width_median_mm": float(np.nanmedian(obs["width_mm"][occurred])),
                "crossing_energy_mean_sd_mev": [
                    float(obs["energy_mev"][~occurred].mean()), float(obs["energy_mev"][~occurred].std())
                ],
                "s_d_energy": conditional_variance_fraction(depth, obs["energy_mev"][occurred], P7_PRIMARY_BINS).epsilon_squared,
                "s_d_long_cog": conditional_variance_fraction(depth, obs["long_cog_mm"][occurred], P7_PRIMARY_BINS).epsilon_squared,
                "s_d_width": conditional_variance_fraction(depth, obs["width_mm"][occurred], P7_PRIMARY_BINS).epsilon_squared,
            }

        entry = {"baseline": summary(b, ob, ib), other: summary(a, oa, ia)}
        entry["ks_interacting_energy"] = _ks(oa["energy_mev"][ia], ob["energy_mev"][ib])
        entry["ks_interaction_depth"] = _ks(a["truth_z_mm"][ia], b["truth_z_mm"][ib])
        entry["ks_crossing_energy"] = _ks(oa["energy_mev"][~ia], ob["energy_mev"][~ib])
        entry["crossing_energy_variance_ratio"] = _interval(
            _variance_ratio(oa["energy_mev"][~ia], ob["energy_mev"][~ib])
        )
        out[f"{energy:g}"] = entry
    return out


def _ks(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    result = stats.ks_2samp(x, y)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def _variance_ratio(x: np.ndarray, y: np.ndarray):
    rng = np.random.default_rng(17)
    draws = [
        np.var(rng.choice(x, len(x))) / np.var(rng.choice(y, len(y))) for _ in range(N_BOOT)
    ]
    from ams_ecal.geant4_simulation.pilot_analysis import Interval

    low, high = np.quantile(draws, [0.025, 0.975])
    return Interval(float(np.var(x) / np.var(y)), float(low), float(high))


def threshold_sensitivity(p: Pilot) -> dict[str, Any]:
    out = {}
    for energy, batch in p.samples["baseline"].items():
        occurred = batch.arrays["truth_occurred"]
        d = batch.arrays["truth_z_mm"][occurred]
        out[f"{energy:g}"] = {
            f"{t:g}xMIP": {
                "median_hits_interacting": float(np.median(p.observables("baseline", energy, threshold_mip=t)["n_hit_cells"][occurred])),
                "s_d_n_hit_cells": conditional_variance_fraction(
                    d, p.observables("baseline", energy, threshold_mip=t)["n_hit_cells"][occurred], P7_PRIMARY_BINS
                ).epsilon_squared,
                "s_d_n_active_layers": conditional_variance_fraction(
                    d, p.observables("baseline", energy, threshold_mip=t)["n_active_layers"][occurred], P7_PRIMARY_BINS
                ).epsilon_squared,
            }
            for t in THRESHOLD_VARIATIONS
        }
    return out


# ----------------------------------------------------------------------
# Exploratory inputs to the proton model model decision
# ----------------------------------------------------------------------

RESIDUAL_OBSERVABLES = ("log_energy", "long_cog_mm", "long_rms_mm", "width_mm", "log_hits")


def _depth_residual(d: np.ndarray, y: np.ndarray, n_bins: int = P7_PRIMARY_BINS) -> np.ndarray:
    """Y minus its mean in the equal-count D bin: the part D does not explain."""

    edges = np.quantile(d, np.linspace(0, 1, n_bins + 1))
    index = np.clip(np.searchsorted(edges[1:-1], d, side="right"), 0, n_bins - 1)
    means = np.array([np.nanmean(y[index == k]) for k in range(n_bins)])
    return y - means[index]


def model_inputs_section(p: Pilot) -> dict[str, Any]:
    """What is left after D, and how the crossing track looks layer by layer.

    EXPLORATORY: defined after the baseline had been read, to inform - not to
    decide - the choice of a stochastic family. Not pre-registered.
    """

    out: dict[str, Any] = {
        "status": "EXPLORATORY - defined after the baseline was read; not pre-registered",
    }
    for energy, batch in p.samples["baseline"].items():
        a = batch.arrays
        occurred = a["truth_occurred"]
        d = a["truth_z_mm"][occurred]
        obs = p.observables("baseline", energy)
        values = {
            "log_energy": obs["log_energy"][occurred],
            "long_cog_mm": obs["long_cog_mm"][occurred],
            "long_rms_mm": obs["long_rms_mm"][occurred],
            "width_mm": obs["width_mm"][occurred],
            "log_hits": np.log(np.maximum(obs["n_hit_cells"][occurred], 1)),
        }
        residuals = np.column_stack([_depth_residual(d, values[n]) for n in RESIDUAL_OBSERVABLES])
        rank_corr = stats.spearmanr(residuals).statistic
        eigen = np.sort(np.linalg.eigvalsh(rank_corr))[::-1]
        log_e = residuals[:, 0]

        remaining = PREFIX_MM - d
        edges = np.quantile(remaining, np.linspace(0, 1, 11))
        index = np.clip(np.searchsorted(edges[1:-1], remaining, side="right"), 0, 9)
        fraction = obs["energy_mev"][occurred] / (energy * 1000.0)
        visible_vs_remaining = [
            {
                "remaining_mm_median": float(np.median(remaining[index == k])),
                "visible_fraction_quantiles": np.quantile(fraction[index == k], [0.16, 0.5, 0.84]).tolist(),
            }
            for k in range(10)
        ]

        crossing_layers = a["readout_grid_mev"][~occurred].sum(axis=2)
        adjacent = [
            stats.spearmanr(crossing_layers[:, k], crossing_layers[:, k + 1]).statistic
            for k in range(crossing_layers.shape[1] - 1)
        ]
        out[f"{energy:g}"] = {
            "residual_names": list(RESIDUAL_OBSERVABLES),
            "residual_spearman": np.round(rank_corr, 3).tolist(),
            "residual_first_component_share": float(eigen[0] / eigen.sum()),
            "log_energy_residual": {
                "sd": float(np.std(log_e)),
                "skewness": float(stats.skew(log_e)),
                "excess_kurtosis": float(stats.kurtosis(log_e)),
                "quantiles_16_50_84": np.quantile(log_e, [0.16, 0.5, 0.84]).tolist(),
            },
            "visible_fraction_vs_remaining_depth": visible_vs_remaining,
            "crossing_layer_energy_quantiles_16_50_84": np.quantile(
                crossing_layers, [0.16, 0.5, 0.84]
            ).tolist(),
            "crossing_layer_mean_by_layer": crossing_layers.mean(axis=0).tolist(),
            "crossing_adjacent_layer_spearman_mean": float(np.mean(adjacent)),
        }
    return out


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------


def figures(p: Pilot, summary: dict[str, Any], out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    energies = sorted(p.samples["baseline"])
    colours = dict(zip(energies, plt.cm.viridis(np.linspace(0.1, 0.85, len(energies))), strict=True))

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=130)
        plt.close(fig)
        written.append(name)

    # 1. Where does the first inelastic interaction happen?
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins = np.linspace(0, PREFIX_MM, 19)
    for e in energies:
        a = p.samples["baseline"][e].arrays
        depth = a["truth_z_mm"][a["truth_occurred"]]
        ax.hist(depth, bins=bins, histtype="step", density=False, color=colours[e],
                weights=np.full(len(depth), 1 / len(a["truth_occurred"])), label=f"{e:g} GeV")
        lam = summary["interaction"][f"{e:g}"]["effective_interaction_length_mm"]["estimate"]
        width = bins[1] - bins[0]
        centres = 0.5 * (bins[1:] + bins[:-1])
        ax.plot(centres, width / lam * np.exp(-centres / lam), color=colours[e], lw=0.8, ls="--")
    ax.set_xlabel("first inelastic interaction depth D [mm]")
    ax.set_ylabel("fraction of all protons per bin")
    ax.set_title("First inelastic interaction in the 0.6 lambda_I prefix (dashed: fitted exponential)")
    ax.legend(fontsize=8)
    save(fig, "fig1_interaction_depth.png")

    # 2. Visible (scintillator) energy by truth class.
    fig, axes = plt.subplots(1, len(energies), figsize=(4 * len(energies), 3.6), sharey=True)
    for ax, e in zip(np.atleast_1d(axes), energies, strict=True):
        a = p.samples["baseline"][e].arrays
        energy = p.observables("baseline", e)["energy_mev"]
        bins = np.logspace(np.log10(max(energy.min(), 1)), np.log10(energy.max()), 60)
        ax.hist([energy[~a["truth_occurred"]], energy[a["truth_occurred"]]], bins=bins,
                stacked=True, label=["no inelastic", "inelastic"], color=["0.6", "C0"])
        band = summary["detector"][f"{e:g}"]["mip_band"][f"{PRIMARY_BAND_QUANTILE:g}"]["energy_upper_mev"]
        ax.axvline(band, color="k", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_title(f"{e:g} GeV")
        ax.set_xlabel("scintillator energy [MeV]")
    np.atleast_1d(axes)[0].set_ylabel("events")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Truth class vs detector-level energy (dotted: 99% of non-interacting)")
    save(fig, "fig2_visible_energy_by_truth.png")

    # 3. Observables against depth: binned means with spread.
    names = ("energy_mev", "long_cog_mm", "width_mm", "n_hit_cells")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, name in zip(axes.ravel(), names, strict=True):
        for e in energies:
            a = p.samples["baseline"][e].arrays
            occ = a["truth_occurred"]
            d = a["truth_z_mm"][occ]
            y = p.observables("baseline", e)[name][occ]
            edges = np.quantile(d, np.linspace(0, 1, 11))
            idx = np.clip(np.searchsorted(edges[1:-1], d, side="right"), 0, 9)
            centre = [np.median(d[idx == k]) for k in range(10)]
            mean = [np.nanmean(y[idx == k]) for k in range(10)]
            sd = [np.nanstd(y[idx == k]) for k in range(10)]
            ax.errorbar(centre, mean, yerr=sd, color=colours[e], marker="o", ms=3, lw=1,
                        capsize=2, label=f"{e:g} GeV")
        ax.set_xlabel("D [mm]")
        ax.set_ylabel(name)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Detector observables vs first-interaction depth (mean +- sd in D deciles)")
    save(fig, "fig3_observables_vs_depth.png")

    # 4. How MIP-like are interactions, by where they happen?
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for e in energies:
        rows = summary["detector"][f"{e:g}"]["mip_like_vs_depth"]
        x = [0.5 * (r["depth_low_mm"] + r["depth_high_mm"]) for r in rows]
        y = [r["fraction_mip_like"]["estimate"] for r in rows]
        lo = [r["fraction_mip_like"]["estimate"] - r["fraction_mip_like"]["low"] for r in rows]
        hi = [r["fraction_mip_like"]["high"] - r["fraction_mip_like"]["estimate"] for r in rows]
        ax.errorbar(x, y, yerr=[lo, hi], color=colours[e], marker="o", ms=3, label=f"{e:g} GeV")
    ax.set_xlabel("D [mm]")
    ax.set_ylabel("fraction inside MIP energy band")
    ax.set_title("Inelastic but MIP-like: late interactions hide")
    ax.legend(fontsize=8)
    save(fig, "fig4_mip_like_vs_depth.png")

    # 5. Depth-dominance hypothesis variance fractions.
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(energies)
    for j, e in enumerate(energies):
        entry = summary["p7"][f"{e:g}"]["observables"]
        est = [entry[n]["s_d_epsilon_squared"]["estimate"] for n in P7_OBSERVABLES]
        lo = [est[i] - entry[n]["s_d_epsilon_squared"]["low"] for i, n in enumerate(P7_OBSERVABLES)]
        hi = [entry[n]["s_d_epsilon_squared"]["high"] - est[i] for i, n in enumerate(P7_OBSERVABLES)]
        ax.bar(np.arange(len(P7_OBSERVABLES)) + j * width, est, width, yerr=[lo, hi],
               color=colours[e], capsize=2, label=f"{e:g} GeV")
    ax.set_xticks(np.arange(len(P7_OBSERVABLES)) + 0.4 - width / 2, P7_OBSERVABLES, rotation=20)
    ax.set_ylabel("S_D (epsilon^2, 95% bootstrap)")
    ax.set_title("Depth-dominance hypothesis: share of variance associated with first-interaction depth")
    ax.axhline(0.5, color="k", lw=0.5, ls=":")
    ax.legend(fontsize=8)
    save(fig, "fig5_p7_variance_fraction.png")

    # 6. Thin vs extended correlations.
    if summary.get("truncation"):
        ext_energies = sorted(float(k) for k in summary["truncation"])
        fig, axes = plt.subplots(1, len(ext_energies), figsize=(4 * len(ext_energies), 3.8), sharey=True)
        for ax, e in zip(np.atleast_1d(axes), ext_energies, strict=True):
            corr = summary["truncation"][f"{e:g}"]["depth_correlations"]
            for k, (key, label, shift) in enumerate((("spearman_thin", "0.6 lambda prefix", -0.18),
                                                      ("spearman_full", "extended", 0.18))):
                est = [corr[n][key]["estimate"] for n in TRUNCATION_OBSERVABLES]
                err = [[est[i] - corr[n][key]["low"] for i, n in enumerate(TRUNCATION_OBSERVABLES)],
                       [corr[n][key]["high"] - est[i] for i, n in enumerate(TRUNCATION_OBSERVABLES)]]
                ax.errorbar(np.arange(len(est)) + shift, est, yerr=err, fmt="o", ms=4,
                            color=f"C{k}", label=label, capsize=2)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(range(len(TRUNCATION_OBSERVABLES)), TRUNCATION_OBSERVABLES, rotation=30, fontsize=8)
            ax.set_title(f"{e:g} GeV")
        np.atleast_1d(axes)[0].set_ylabel("Spearman r(D, observable)")
        np.atleast_1d(axes)[0].legend(fontsize=8)
        fig.suptitle("Same events: correlation with D seen through the prefix vs the full shower")
        save(fig, "fig6_truncation_correlations.png")

    # 9. Exploratory model inputs: what is left after D.
    if "model_inputs" in summary:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for e in energies:
            entry = summary["model_inputs"][f"{e:g}"]
            rows = entry["visible_fraction_vs_remaining_depth"]
            x = [r["remaining_mm_median"] for r in rows]
            q = np.array([r["visible_fraction_quantiles"] for r in rows])
            axes[0].plot(x, q[:, 1], color=colours[e], marker="o", ms=3, label=f"{e:g} GeV")
            axes[0].fill_between(x, q[:, 0], q[:, 2], color=colours[e], alpha=0.15)
            a = p.samples["baseline"][e].arrays
            occ = a["truth_occurred"]
            residual = _depth_residual(a["truth_z_mm"][occ], p.observables("baseline", e)["log_energy"][occ])
            axes[1].hist(residual, bins=80, histtype="step", density=True, color=colours[e], label=f"{e:g} GeV")
        axes[0].set_xlabel("remaining ECAL depth after first interaction [mm]")
        axes[0].set_ylabel("scintillator energy / E0 (median, 16-84%)")
        axes[0].legend(fontsize=8)
        axes[1].set_xlabel("log visible energy minus its mean at fixed D")
        axes[1].set_ylabel("density")
        axes[1].set_title("not lognormal: a heavy low-visible tail")
        fig.suptitle("EXPLORATORY: inputs to the proton-model family choice")
        save(fig, "fig9_model_inputs.png")

    # 7. Physics-list and fixed-entry controls: interacting-energy CDFs.
    for sample, name, label in (("alternate", "fig7_physics_list.png", "QBBC"),
                                ("fixed_entry", "fig8_fixed_entry.png", "fixed entry"),
                                ("high_energy_model", "fig10_high_energy_model.png", "QGSP_BERT")):
        if not p.samples[sample]:
            continue
        common = [e for e in energies if e in p.samples[sample]]
        fig, axes = plt.subplots(1, len(common), figsize=(4 * len(common), 3.6), sharey=True)
        for ax, e in zip(np.atleast_1d(axes), common, strict=True):
            for s, style, lab in (("baseline", "-", "baseline (FTFP_BERT, uniform)"), (sample, "--", label)):
                a = p.samples[s][e].arrays
                energy = np.sort(p.observables(s, e)["energy_mev"])
                ax.plot(energy, np.linspace(0, 1, len(energy)), style, label=lab)
            ax.set_xscale("log")
            ax.set_title(f"{e:g} GeV")
            ax.set_xlabel("scintillator energy [MeV]")
        np.atleast_1d(axes)[0].set_ylabel("cumulative fraction")
        np.atleast_1d(axes)[0].legend(fontsize=7)
        save(fig, name)
    return written


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def _flat_rows(section: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    rows = []
    for key, value in section.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows += _flat_rows(value, name)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            for i, item in enumerate(value):
                rows += _flat_rows(item, f"{name}[{i}]")
        else:
            rows.append((name, value))
    return rows


def build_report(data: Path = DATA, out: Path = RESULTS) -> dict[str, Any]:
    p = Pilot(data)
    summary: dict[str, Any] = {
        "provenance": {
            sample: {
                f"{e:g}": {
                    k: b.metadata[k]
                    for k in ("git", "geant4_version", "physics_list", "n_events",
                              "configuration_sha256", "production_cut_mm", "created_utc")
                }
                for e, b in batches.items()
            }
            for sample, batches in p.samples.items()
        },
        "analysis_choices": {
            "cell_threshold_mip": CELL_THRESHOLD_MIP,
            "layer_threshold_mip": LAYER_THRESHOLD_MIP,
            "mip_band_quantiles": MIP_BAND_QUANTILES,
            "primary_band_quantile": PRIMARY_BAND_QUANTILE,
            "p7_bins": P7_BINS,
            "p7_primary_bins": P7_PRIMARY_BINS,
            "bootstrap_replicates": N_BOOT,
            "representation": "readout_grid_mev (scintillator energy) unless stated",
        },
        "interaction": interaction_section(p),
        "detector": detector_section(p),
        "p7": p7_section(p),
        "threshold_sensitivity": threshold_sensitivity(p),
        "model_inputs": model_inputs_section(p),
    }
    if p.samples["extended"]:
        summary["truncation"] = truncation_section(p)
    if p.samples["alternate"]:
        summary["physics_list"] = comparison_section(p, "alternate")
    if p.samples["fixed_entry"]:
        summary["fixed_entry"] = comparison_section(p, "fixed_entry")
    if p.samples["high_energy_model"]:
        summary["high_energy_model"] = comparison_section(p, "high_energy_model")

    out.mkdir(parents=True, exist_ok=True)
    summary["figures"] = figures(p, summary, out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    for section in ("interaction", "detector", "p7", "truncation", "physics_list", "fixed_entry",
                    "high_energy_model", "model_inputs"):
        if section in summary:
            with (out / f"{section}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["quantity", "value"])
                writer.writerows(_flat_rows(summary[section]))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--out", default=str(RESULTS))
    args = parser.parse_args()
    summary = build_report(Path(args.data), Path(args.out))
    print(json.dumps({k: summary[k] for k in ("interaction",)}, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
