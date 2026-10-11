"""Plots of the depth-origin calibration (``ams_ecal.electron_studies.em_depth_origin_calibration``): figures only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy import stats

from ams_ecal.electron_studies.em_depth_origin_calibration import (
    BETA_GRID,
    CONTOUR_LEVELS,
    Z0_GRID,
)

MODEL_COLOURS = {
    "front_face_ams": "#c0392b",
    "front_face_free_beta": "#e67e22",
    "front_face_free_beta_offset": "#f1c40f",
    "ams_beta_free_origin": "#d35400",
    "origin_beta": "#27ae60",
    "origin_beta_deterministic": "#16a085",
    "origin_beta_naive_width": "#8e44ad",
    "origin_beta_offset": "#2980b9",
    "origin_beta_free_depth": "#2c3e91",
    "origin_beta_amplitude_free": "#7f8c8d",
}
DPI = 100


def _save(figure: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / name, dpi=DPI)
    plt.close(figure)


def _delta(surface: np.ndarray, scale: float) -> np.ndarray:
    return (surface - surface.min()) / scale


def _energies(artefacts: dict[str, Any]) -> list[float]:
    return [float(e) for e in artefacts["targets"].energies_gev]


def _row(n: int, width: float, height: float, **kwargs):
    """One row of ``n`` panels; ``axes`` is always a 1-D array."""

    fig, axes = plt.subplots(1, n, figsize=(width * n, height), constrained_layout=True, **kwargs)
    return fig, np.atleast_1d(axes)


def surface_plot(results: dict[str, Any], artefacts: dict[str, Any], directory: Path) -> None:
    delta = _delta(artefacts["surface"], results["B_surface"]["scale_chi2_per_dof"])
    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    image = ax.pcolormesh(
        BETA_GRID, Z0_GRID, np.clip(delta, 0.05, None), norm=LogNorm(vmin=0.05, vmax=1000), cmap="viridis_r"
    )
    fig.colorbar(image, ax=ax, label="scaled delta chi-square")
    ax.contour(BETA_GRID, Z0_GRID, delta, levels=list(CONTOUR_LEVELS.values()), colors=["white", "yellow"])
    boot = artefacts["bootstrap_primary"]
    ax.scatter(boot[:, 1], boot[:, 0], s=6, color="black", alpha=0.35, label="event bootstrap")
    valley = results["B_surface"]["profile_valley"]
    ax.plot(
        valley["best_beta_given_z0"], valley["z0"], color="red", linewidth=1.2, label="valley: best beta at each z0"
    )
    primary = results["A_primary"]
    ax.plot(primary["beta"], primary["z0"], "r*", markersize=16, label="common optimum")
    ax.plot(0.65, 0.0, "wo", markeredgecolor="black", markersize=9, label="front face, beta = 0.65")
    for row in results["B_per_energy_fits"]:
        ax.plot(row["beta"], row["z0"], "s", color="orange", markeredgecolor="black", markersize=6)
    ax.plot([], [], "s", color="orange", markeredgecolor="black", label="separate fit per energy")
    ax.set_xlabel("beta (mean gamma rate)")
    ax.set_ylabel("z0 (origin, X0 from the front face; negative = upstream)")
    ax.set_title("Mean-profile objective over (z0, beta): contours 68% and 95% (scaled)")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, directory, "surface_z0_beta.png")


def valley_plot(results: dict[str, Any], directory: Path) -> None:
    fig, axes = _row(3, 5.3, 4.5)
    for label, block, colour in (
        ("deposition", results["B_surface"], "#27ae60"),
        ("readout", results["F_readout"]["surface"], "#2980b9"),
    ):
        valley = block["profile_valley"]
        axes[0].plot(valley["z0"], valley["delta_chi2_scaled_given_z0"], color=colour, label=label)
        axes[1].plot(valley["z0"], valley["best_beta_given_z0"], color=colour, label=label)
        axes[2].plot(valley["beta_grid"], valley["delta_chi2_scaled_given_beta"], color=colour, label=label)
    for ax in (axes[0], axes[2]):
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(3.84, color="gray", linestyle=":", linewidth=0.8)
        ax.set_ylim(0, 12)
    axes[0].set_xlabel("z0 (X0)")
    axes[0].set_ylabel("profile delta chi-square (scaled), beta profiled out")
    axes[1].set_xlabel("z0 (X0)")
    axes[1].set_ylabel("best beta at that z0")
    axes[2].set_xlabel("beta")
    axes[2].set_ylabel("profile delta chi-square (scaled), z0 profiled out")
    axes[0].legend()
    fig.suptitle("Identifiability: profile objective and the z0-beta valley (dashed 1 sigma, dotted 95%)")
    _save(fig, directory, "profile_valley.png")


def bootstrap_plot(results: dict[str, Any], artefacts: dict[str, Any], directory: Path) -> None:
    fig, axes = _row(3, 5.3, 4.5)
    series = (
        ("origin_beta (primary)", artefacts["bootstrap_primary"], "#27ae60"),
        ("deterministic", artefacts["bootstrap_origin_beta_deterministic"], "#16a085"),
        ("naive width", artefacts["bootstrap_origin_beta_naive_width"], "#8e44ad"),
    )
    for label, samples, colour in series:
        axes[0].scatter(samples[:, 1], samples[:, 0], s=8, alpha=0.5, color=colour, label=label)
        axes[1].hist(samples[:, 0], bins=25, alpha=0.5, color=colour, label=label)
        axes[2].hist(samples[:, 1], bins=25, alpha=0.5, color=colour, label=label)
    axes[0].set_xlabel("beta")
    axes[0].set_ylabel("z0 (X0)")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("z0 (X0)")
    axes[2].set_xlabel("beta")
    sd = results["B_bootstrap"]["sd"]
    fig.suptitle(
        f"Event bootstrap of the common fit: sd(z0) {sd['z0']:.2f}, sd(beta) {sd['beta']:.3f}, "
        f"correlation {results['B_bootstrap']['correlation_z0_beta']:+.2f}"
    )
    _save(fig, directory, "bootstrap_parameters.png")


def _models_for_residuals(results: dict[str, Any]) -> list[str]:
    order = [
        "front_face_ams",
        "front_face_free_beta",
        "ams_beta_free_origin",
        "origin_beta",
        "origin_beta_offset",
        "origin_beta_free_depth",
    ]
    return [m for m in order if m in results["A_fits"]]


def mean_profile_residuals(results: dict[str, Any], artefacts: dict[str, Any], directory: Path) -> None:
    energies = _energies(artefacts)
    fig, axes = _row(len(energies), 4.2, 4.5, sharey=True)
    layers = np.arange(18)
    floor = results.get("E_floor", {})
    floor_accepted = bool(floor.get("constant")) and floor.get("verdicts", {}).get("constant", {}).get("accepted")
    for k, (ax, energy) in enumerate(zip(axes, energies, strict=True)):
        for name in _models_for_residuals(results):
            ax.plot(
                layers,
                np.array(results["A_fits"][name]["relative_residual_by_layer"])[k],
                color=MODEL_COLOURS[name],
                label=name,
            )
        if floor_accepted:
            ax.plot(
                layers, np.array(floor["constant"]["relative_residual_by_layer"])[k], "k--", label="with floor"
            )
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_title(f"{energy:g} GeV: (model - Geant4) / Geant4")
        ax.set_xlabel("layer")
    axes[0].legend(fontsize=6)
    _save(fig, directory, "mean_profile_residuals.png")


def early_layer_residuals(results: dict[str, Any], artefacts: dict[str, Any], directory: Path) -> None:
    energies = _energies(artefacts)
    targets = artefacts["targets"]
    fig, axes = _row(len(energies), 4.2, 4.5, sharey=True)
    layers = np.arange(6)
    for k, (ax, energy) in enumerate(zip(axes, energies, strict=True)):
        band = 3.0 * targets.se[k, :6] / targets.mean[k, :6]
        ax.fill_between(layers, -band, band, color="#dddddd", label="3 standard errors of the Geant4 mean")
        for name in _models_for_residuals(results):
            ax.plot(
                layers,
                np.array(results["A_fits"][name]["relative_residual_by_layer"])[k, :6],
                "o-",
                color=MODEL_COLOURS[name],
                label=name,
            )
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_title(f"{energy:g} GeV: early layers")
        ax.set_xlabel("layer")
    axes[0].legend(fontsize=6)
    _save(fig, directory, "early_layer_residuals.png")


def leakage_plot(artefacts: dict[str, Any], directory: Path) -> None:
    energies = _energies(artefacts)
    loaded = artefacts["loaded"]
    fig, axes = _row(len(energies), 4.2, 4.2)
    for ax, energy in zip(axes, energies, strict=True):
        reference = 1.0 - loaded[energy].fractions["deposition"].sum(axis=1)
        bins = np.linspace(min(0.0, reference.min()), max(reference.max(), 0.2), 45)
        ax.hist(reference, bins=bins, density=True, histtype="stepfilled", color="#dddddd", label="Geant4")
        ax.hist(
            1.0 - artefacts["dec001_ensembles"][energy].sum(axis=1),
            bins=bins,
            density=True,
            histtype="step",
            color="#c0392b",
            label="DEC-001",
        )
        for name, colour in (("calibrated_gp_prescribed", "#27ae60"), ("calibrated_measured_spread", "#2c3e91")):
            ax.hist(
                1.0 - artefacts["ensembles"][energy][name].sum(axis=1),
                bins=bins,
                density=True,
                histtype="step",
                color=colour,
                label=name,
            )
        ax.set_title(f"{energy:g} GeV: energy beyond the last layer")
        ax.set_xlabel("1 - contained fraction")
    axes[0].legend(fontsize=6)
    _save(fig, directory, "leakage_comparison.png")


def recalibrated_marginals(results: dict[str, Any], artefacts: dict[str, Any], directory: Path) -> None:
    energies = _energies(artefacts)
    fits = artefacts["fits_at_origin"]["least_squares"]
    block = results["D_stochastic_structure"]["least_squares"]
    fig, axes = plt.subplots(
        2, len(energies), figsize=(3.8 * len(energies), 6.2), constrained_layout=True, squeeze=False
    )
    for column, energy in enumerate(energies):
        entry = block[f"{energy:g}"]
        for row, (label, index, gp_sigma) in enumerate(
            (("ln T", 0, entry["gp_sigma_ln_t_covariant"]), ("ln alpha", 1, entry["gp"]["sigma_ln_alpha"]))
        ):
            values = fits[energy][:, index]
            ax = axes[row, column]
            ax.hist(values, bins=40, density=True, color="#bbbbbb", edgecolor="white")
            grid = np.linspace(values.min(), values.max(), 200)
            ax.plot(grid, stats.norm.pdf(grid, values.mean(), values.std()), color="black", label="normal, fitted")
            ax.plot(
                grid, stats.norm.pdf(grid, values.mean(), gp_sigma), "--", color="#27ae60", label="normal, GP sigma"
            )
            ax.set_title(f"{label}, {energy:g} GeV (skew {stats.skew(values):+.2f})")
            if column == 0 and row == 0:
                ax.legend(fontsize=7)
    fig.suptitle(f"Per-event fits at the calibrated origin z0 = {results['A_primary']['z0']:+.2f} X0")
    _save(fig, directory, "recalibrated_ln_t_ln_alpha.png")


def recalibrated_betas(results: dict[str, Any], artefacts: dict[str, Any], directory: Path) -> None:
    energies = _energies(artefacts)
    fits = artefacts["fits_at_origin"]["least_squares"]
    beta_bar = results["A_primary"]["beta"]
    fig, axes = _row(len(energies), 4.2, 4.2, sharex=True)
    for ax, energy in zip(axes, energies, strict=True):
        beta = (np.exp(fits[energy][:, 1]) - 1.0) / np.exp(fits[energy][:, 0])
        ax.hist(beta, bins=np.linspace(0.2, 1.2, 46), color="#bbbbbb", edgecolor="white")
        ax.axvline(0.65, color="#c0392b", label="0.65")
        ax.axvline(beta_bar, color="#27ae60", label=f"calibrated beta {beta_bar:.2f}")
        ax.axvline(np.median(beta), color="black", linestyle="--", label=f"event median {np.median(beta):.2f}")
        ax.set_title(f"{energy:g} GeV: beta_i at the calibrated origin")
        ax.legend(fontsize=7)
    _save(fig, directory, "recalibrated_beta_distributions.png")


def structure_versus_origin(results: dict[str, Any], directory: Path) -> None:
    table = results.get("D_skewness_versus_origin")
    if not table:
        return
    origins = np.array(table["origins_x0"])
    energies = [k for k in table if k != "origins_x0"]
    fig, axes = _row(3, 5.3, 4.5)
    for energy in energies:
        rows = table[energy]
        axes[0].plot(origins, [r["skewness_ln_t"] for r in rows], "o-", label=f"{energy} GeV")
        axes[1].plot(origins, [r["pearson"] for r in rows], "o-", label=f"{energy} GeV")
        axes[2].plot(origins, [r["beta_median"] for r in rows], "o-", label=f"{energy} GeV")
    axes[0].set_ylabel("skewness of ln T")
    axes[1].set_ylabel("Pearson rho(ln T, ln alpha)")
    axes[2].set_ylabel("event-median beta")
    axes[2].axhline(0.65, color="#c0392b", linestyle="--", linewidth=0.8)
    for ax in axes:
        ax.axvline(results["A_primary"]["z0"], color="#27ae60", linestyle=":", label="calibrated z0")
        ax.set_xlabel("depth origin z0 (X0; negative = upstream)")
    axes[0].legend(fontsize=7)
    fig.suptitle("How the per-event structure depends on the origin convention")
    _save(fig, directory, "structure_versus_origin.png")


def gp_comparison(results: dict[str, Any], directory: Path) -> None:
    gp = results["D_stochastic_structure"]["least_squares"]
    energy_values = [float(e) for e in gp]
    fig, axes = _row(3, 5.0, 4.2)
    for label, key, colour in (
        ("naive", "sigma_ln_t_over_gp_naive", "#e67e22"),
        ("covariant", "sigma_ln_t_over_gp_covariant", "#27ae60"),
    ):
        axes[0].plot(energy_values, [gp[e]["ratios"][key] for e in gp], "o-", color=colour, label=label)
    axes[1].plot(energy_values, [gp[e]["ratios"]["sigma_ln_alpha_over_gp"] for e in gp], "o-", color="#2980b9")
    axes[2].plot(energy_values, [gp[e]["joint"]["pearson"] for e in gp], "o-", color="#27ae60", label="measured")
    axes[2].plot(energy_values, [gp[e]["gp"]["rho"] for e in gp], "k--", label="GP expectation")
    for ax in axes[:2]:
        ax.axhspan(0.75, 1.25, color="#eeeeee")
        ax.axhline(1.0, color="black", linewidth=0.6)
    axes[0].set_ylabel("measured / GP width of ln T")
    axes[1].set_ylabel("measured / GP width of ln alpha")
    axes[2].set_ylabel("Pearson rho")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("energy (GeV)")
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    z0 = results["A_primary"]["z0"]
    fig.suptitle(f"GP comparison at the calibrated origin z0 = {z0:+.2f} X0 (grey band: coherent)")
    _save(fig, directory, "gp_comparison_at_origin.png")


def floor_plot(results: dict[str, Any], artefacts: dict[str, Any], directory: Path) -> None:
    floor = results.get("E_floor", {})
    if "constant" not in floor:
        return
    energies = _energies(artefacts)
    fig, axes = _row(2, 6.0, 4.5)
    base = np.array(results["A_fits"]["origin_beta"]["relative_residual_by_layer"])
    for k, energy in enumerate(energies):
        axes[0].plot(np.arange(6), base[k, :6], "o-", label=f"{energy:g} GeV, no floor")
        axes[0].plot(
            np.arange(6),
            np.array(floor["constant"]["relative_residual_by_layer"])[k, :6],
            "x--",
            label=f"{energy:g} GeV, constant floor",
        )
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].set_yscale("symlog", linthresh=0.1)
    axes[0].set_xlabel("layer")
    axes[0].set_title("entrance residuals")
    axes[0].legend(fontsize=6)
    labels = [k for k in ("constant", "linear") if k in floor["verdicts"]]
    positions = np.arange(len(labels))
    axes[1].bar(
        positions - 0.18,
        [floor["verdicts"][k]["held_out_improvement_leave_one_energy_out"] for k in labels],
        0.36,
        label="leave one energy out",
    )
    axes[1].bar(
        positions + 0.18,
        [floor["verdicts"][k]["held_out_improvement_split_half"] for k in labels],
        0.36,
        label="split half",
    )
    axes[1].axhline(0.2, color="red", linestyle="--", label="acceptance (20%)")
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("held-out improvement over origin + beta")
    axes[1].legend(fontsize=7)
    _save(fig, directory, "floor_comparison.png")


def deposition_vs_readout(results: dict[str, Any], artefacts: dict[str, Any], directory: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    for label, surface, scale, colour, boot in (
        (
            "deposition",
            artefacts["surface"],
            results["B_surface"]["scale_chi2_per_dof"],
            "#27ae60",
            artefacts["bootstrap_primary"],
        ),
        (
            "readout (amplitude free)",
            artefacts["readout_surface"],
            results["F_readout"]["surface"]["scale_chi2_per_dof"],
            "#2980b9",
            artefacts["bootstrap_readout"],
        ),
    ):
        ax.contour(
            BETA_GRID,
            Z0_GRID,
            _delta(surface, scale),
            levels=list(CONTOUR_LEVELS.values()),
            colors=[colour, colour],
            linestyles=["-", "--"],
        )
        ax.scatter(boot[:, 1], boot[:, 0], s=6, color=colour, alpha=0.35)
        index = np.unravel_index(surface.argmin(), surface.shape)
        ax.plot(BETA_GRID[index[1]], Z0_GRID[index[0]], "*", color=colour, markersize=15, label=label)
    ax.plot(0.65, 0.0, "ko", markersize=8, label="front face, beta = 0.65")
    ax.set_xlabel("beta")
    ax.set_ylabel("z0 (X0)")
    ax.set_title("Deposition and readout: 68% (solid) and 95% (dashed) regions of the (z0, beta) objective")
    ax.legend(fontsize=8)
    _save(fig, directory, "deposition_vs_readout.png")


def make_plots(results: dict[str, Any], artefacts: dict[str, Any], directory: Path) -> None:
    surface_plot(results, artefacts, directory)
    valley_plot(results, directory)
    bootstrap_plot(results, artefacts, directory)
    mean_profile_residuals(results, artefacts, directory)
    early_layer_residuals(results, artefacts, directory)
    leakage_plot(artefacts, directory)
    recalibrated_marginals(results, artefacts, directory)
    recalibrated_betas(results, artefacts, directory)
    structure_versus_origin(results, directory)
    gp_comparison(results, directory)
    floor_plot(results, artefacts, directory)
    deposition_vs_readout(results, artefacts, directory)
