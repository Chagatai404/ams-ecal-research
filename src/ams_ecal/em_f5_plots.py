"""Plots of the F5 analysis (``ams_ecal.em_f5_analysis``): analysis figures only, no model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from ams_ecal.em_f5_analysis import DISTANCES, EnergyData

COLOURS = {
    "geant4": "#222222",
    "backbone": "#7a7a7a",
    "dec001_deposition": "#c0392b",
    "dec001_sampling": "#e67e22",
    "fixed_beta_data_T": "#8e44ad",
    "fixed_beta_matched_mean": "#2980b9",
    "joint_data": "#27ae60",
    "fixed_beta_origin_data_T": "#d35400",
    "joint_origin_data": "#1abc9c",
    "joint_ams_mean_gp_fluct": "#2c3e91",
    "joint_ams_mean_gp_fluct_sampling_T": "#7f8c8d",
}
DPI = 100


def _energies(results: dict[str, Any]) -> list[str]:
    return [k for k in results if not k.startswith("B_")]


def _grid(results: dict[str, Any], rows: int, width_per_energy: float, height: float, **kwargs):
    """A figure with one column per energy; ``axes`` is always 2-D (rows, energies)."""

    n = len(_energies(results))
    return plt.subplots(
        rows, n, figsize=(width_per_energy * n, height), constrained_layout=True, squeeze=False, **kwargs
    )


def _save(figure: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / name, dpi=DPI)
    plt.close(figure)


def _ellipse(ax, mean, cov, colour, label, style="-") -> None:
    values, vectors = np.linalg.eigh(cov)
    angle = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([np.cos(angle), np.sin(angle)])
    points = (vectors @ (2.0 * np.sqrt(values)[:, None] * circle)).T + mean
    ax.plot(points[:, 0], points[:, 1], style, color=colour, label=label)


def marginals(results: dict[str, Any], loaded: dict[float, EnergyData], directory: Path) -> None:
    fig, axes = _grid(results, 2, 3.75, 6)
    for column, (key, energy) in enumerate(zip(_energies(results), loaded, strict=True)):
        fit = loaded[energy].fits["least_squares"]
        gp = results[key]["A_B_joint_structure"]["gp"]["sampling"]
        for row, (label, index, gp_sigma) in enumerate(
            (("ln T", 0, gp["sigma_ln_t"]), ("ln alpha", 1, gp["sigma_ln_alpha"]))
        ):
            ax = axes[row, column]
            values = fit[:, index]
            ax.hist(values, bins=40, density=True, color="#bbbbbb", edgecolor="white")
            grid = np.linspace(values.min(), values.max(), 200)
            ax.plot(
                grid,
                stats.norm.pdf(grid, values.mean(), values.std()),
                color=COLOURS["geant4"],
                label="normal, fitted",
            )
            ax.plot(
                grid,
                stats.norm.pdf(grid, values.mean(), gp_sigma),
                "--",
                color=COLOURS["joint_data"],
                label="normal, GP sigma",
            )
            ax.set_title(f"{label}, {key} GeV  (skew {stats.skew(values):+.2f})")
            if column == 0 and row == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Geant4 electrons: per-event gamma-fit parameters (deposition, least squares)")
    _save(fig, directory, "marginals_ln_t_ln_alpha.png")


def joint(results: dict[str, Any], loaded: dict[float, EnergyData], directory: Path) -> None:
    fig, grid = _grid(results, 1, 3.75, 4)
    axes = grid[0]
    for ax, key, energy in zip(axes, _energies(results), loaded, strict=True):
        fit = loaded[energy].fits["least_squares"]
        gp = results[key]["A_B_joint_structure"]["gp"]["sampling"]
        ax.scatter(fit[:, 0], fit[:, 1], s=4, alpha=0.3, color="#999999")
        _ellipse(ax, fit.mean(axis=0), np.cov(fit.T), COLOURS["geant4"], "data, 2 sigma")
        cross = gp["rho"] * gp["sigma_ln_t"] * gp["sigma_ln_alpha"]
        gp_cov = np.array([[gp["sigma_ln_t"] ** 2, cross], [cross, gp["sigma_ln_alpha"] ** 2]])
        _ellipse(ax, fit.mean(axis=0), gp_cov, COLOURS["joint_data"], "GP sampling, data means", "--")
        ax.set_xlabel("ln T")
        ax.set_ylabel("ln alpha")
        ax.set_title(f"{key} GeV: rho = {np.corrcoef(fit.T)[0, 1]:.2f} (GP {gp['rho']:.2f})")
        ax.legend(fontsize=8)
    _save(fig, directory, "joint_ln_t_ln_alpha.png")


def betas(results: dict[str, Any], loaded: dict[float, EnergyData], directory: Path) -> None:
    fig, axes = _grid(results, 3, 3.75, 8.5, sharex=True)
    for column, (key, energy) in enumerate(zip(_energies(results), loaded, strict=True)):
        for row, (fit, title) in enumerate(
            (
                (loaded[energy].fits["least_squares"], "deposition (calibration profile)"),
                (loaded[energy].fits["readout"], "readout (diagnostic, amplitude free)"),
                (
                    loaded[energy].fits["origin_mean_profile_fixed_beta"],
                    f"deposition, origin {loaded[energy].origins['mean_profile_fixed_beta']:+.1f} X0",
                ),
            )
        ):
            beta = (np.exp(fit[:, 1]) - 1.0) / np.exp(fit[:, 0])
            ax = axes[row, column]
            ax.hist(beta, bins=np.linspace(0.2, 1.1, 46), color="#bbbbbb", edgecolor="white")
            ax.axvline(0.65, color=COLOURS["dec001_deposition"], label="AMS 0.65")
            ax.axvline(
                np.median(beta),
                color=COLOURS["geant4"],
                linestyle="--",
                label=f"median {np.median(beta):.2f}",
            )
            ax.set_title(f"{key} GeV, {title}", fontsize=9)
            ax.legend(fontsize=8)
    fig.suptitle("Event-level beta = (alpha - 1) / T")
    _save(fig, directory, "beta_distributions.png")


def mean_profiles(
    results: dict[str, Any],
    ensembles: dict[float, dict[str, np.ndarray]],
    loaded: dict[float, EnergyData],
    directory: Path,
) -> None:
    fig, axes = _grid(results, 2, 4, 7, sharex=True)
    layers = np.arange(18)
    for column, (key, energy) in enumerate(zip(_energies(results), loaded, strict=True)):
        d = results[key]["D_mean_backbone"]
        geant4 = np.array(d["geant4_mean_fraction_by_layer"])
        curves = {
            "AMS backbone (alpha = 1 + 0.65 T)": (
                np.array(d["backbone_fraction_by_layer"]),
                COLOURS["backbone"],
            ),
            "DEC-001 ensemble": (
                ensembles[energy]["dec001_deposition"].mean(axis=0),
                COLOURS["dec001_deposition"],
            ),
            "joint (AMS mean + GP fluct.)": (
                ensembles[energy]["joint_ams_mean_gp_fluct"].mean(axis=0),
                COLOURS["joint_ams_mean_gp_fluct"],
            ),
            "joint (data calibrated)": (
                ensembles[energy]["joint_data"].mean(axis=0),
                COLOURS["joint_data"],
            ),
        }
        top, bottom = axes[0, column], axes[1, column]
        top.semilogy(layers, geant4, "o-", color=COLOURS["geant4"], label="Geant4")
        for label, (curve, colour) in curves.items():
            top.semilogy(layers, curve, color=colour, label=label)
            bottom.plot(layers, curve / geant4, color=colour)
        bottom.axhline(1.0, color=COLOURS["geant4"], linewidth=0.8)
        bottom.set_ylim(0, 2.5)
        top.set_title(f"{key} GeV: mean layer fraction")
        bottom.set_title("ratio to Geant4")
        bottom.set_xlabel("layer")
        if column == 0:
            top.legend(fontsize=7)
    _save(fig, directory, "mean_profiles.png")


def early_layers(results: dict[str, Any], directory: Path) -> None:
    fig, grid = _grid(results, 1, 4, 4, sharey=True)
    axes = grid[0]
    for ax, key in zip(axes, _energies(results), strict=True):
        metrics = results[key]["C_counterfactual"]["metrics"]
        for variant, m in metrics.items():
            ax.semilogy(
                np.arange(6),
                np.array(m["mean_ratio_to_geant4_by_layer"])[:6],
                "o-",
                color=COLOURS[variant],
                label=variant,
            )
        ax.axhline(1.0, color=COLOURS["geant4"], linewidth=0.8)
        ax.set_title(f"{key} GeV: early layers, model / Geant4")
        ax.set_xlabel("layer")
    axes[0].legend(fontsize=6)
    _save(fig, directory, "early_layer_ratios.png")


def leakage(
    results: dict[str, Any],
    ensembles: dict[float, dict[str, np.ndarray]],
    loaded: dict[float, EnergyData],
    directory: Path,
) -> None:
    fig, grid = _grid(results, 1, 4, 4)
    axes = grid[0]
    for ax, key, energy in zip(axes, _energies(results), loaded, strict=True):
        reference = 1.0 - loaded[energy].fractions["deposition"].sum(axis=1)
        bins = np.linspace(min(0.0, reference.min()), max(reference.max(), 0.2), 50)
        ax.hist(reference, bins=bins, density=True, histtype="stepfilled", color="#dddddd", label="Geant4")
        for variant in (
            "dec001_deposition",
            "fixed_beta_matched_mean",
            "joint_data",
            "joint_ams_mean_gp_fluct",
        ):
            ax.hist(
                1.0 - ensembles[energy][variant].sum(axis=1),
                bins=bins,
                density=True,
                histtype="step",
                color=COLOURS[variant],
                label=variant,
            )
        ax.set_title(f"{key} GeV: energy beyond the last layer")
        ax.set_xlabel("1 - contained fraction")
    axes[0].legend(fontsize=6)
    _save(fig, directory, "leakage.png")


def counterfactual_heatmap(results: dict[str, Any], directory: Path) -> None:
    names = list(DISTANCES)
    fig, grid = _grid(results, 1, 4.25, 5, sharey=True)
    axes = grid[0]
    image = None
    for ax, key in zip(axes, _energies(results), strict=True):
        removed = results[key]["C_counterfactual"]["removed_vs_dec001_deposition"]
        variants = [v for v in removed if v != "dec001_deposition"]
        grid = np.array(
            [[np.nan if removed[v][n] is None else removed[v][n] for n in names] for v in variants]
        )
        image = ax.imshow(np.clip(grid, -1, 1), cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(names)), names, rotation=70, ha="right", fontsize=7)
        ax.set_yticks(range(len(variants)), variants, fontsize=7)
        ax.set_title(f"{key} GeV")
    fig.suptitle("Fraction of the DEC-001 discrepancy removed (blank: base already at its sampling-noise floor)")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.8)
    _save(fig, directory, "counterfactual_removed.png")


def structural_fits(results: dict[str, Any], directory: Path) -> None:
    fig, grid = _grid(results, 1, 4, 4, sharey=True)
    axes = grid[0]
    for ax, key in zip(axes, _energies(results), strict=True):
        for model, fit in results[key]["D_mean_backbone"]["structural_fits"].items():
            ax.semilogy(
                np.arange(6),
                fit["geant4_over_fit_by_layer_0_5"],
                "o-",
                label=f"{model} (rms {fit['rms_residual']:.4f})",
            )
        ax.axhline(1.0, color=COLOURS["geant4"], linewidth=0.8)
        ax.set_title(f"{key} GeV: Geant4 mean profile / fit")
        ax.set_xlabel("layer")
        ax.legend(fontsize=6)
    _save(fig, directory, "structural_mean_profile_fits.png")


def confounders(results: dict[str, Any], directory: Path) -> None:
    fig, grid = _grid(results, 1, 4.25, 4, sharey=True)
    axes = grid[0]
    for ax, key in zip(axes, _energies(results), strict=True):
        rows = results[key]["E_confounders"]["observables"]
        names = list(rows)
        x = np.arange(len(names))
        for offset, (label, field, colour) in enumerate(
            (
                ("raw", "raw_spearman", "#bbbbbb"),
                ("controlling entry phase", "partial_phase", "#2980b9"),
                ("phase and rear leakage", "partial_phase_and_rear_leakage", "#c0392b"),
            )
        ):
            ax.bar(
                x + (offset - 1) * 0.27,
                [rows[n]["ln_t"][field] for n in names],
                0.27,
                color=colour,
                label=label,
            )
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x, names, rotation=25, ha="right", fontsize=8)
        ax.set_title(f"{key} GeV: Spearman with ln T")
    axes[0].legend(fontsize=7)
    _save(fig, directory, "confounders.png")


def make_plots(
    results: dict[str, Any],
    loaded: dict[float, EnergyData],
    ensembles: dict[float, dict[str, np.ndarray]],
    directory: Path,
) -> None:
    marginals(results, loaded, directory)
    joint(results, loaded, directory)
    betas(results, loaded, directory)
    mean_profiles(results, ensembles, loaded, directory)
    early_layers(results, directory)
    leakage(results, ensembles, loaded, directory)
    counterfactual_heatmap(results, directory)
    structural_fits(results, directory)
    confounders(results, directory)
