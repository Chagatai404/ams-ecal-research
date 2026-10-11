"""Plots of the profile shape study (``ams_ecal.electron_studies.em_profile_shape_study``): figures only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 100
COLOURS = {"gaussian": "#e67e22", "skew_normal": "#2980b9", "empirical": "#27ae60"}
FAMILY_LABELS = {
    "primary": "gamma only (primary)",
    "beta_energy_slope": "+ beta slope in ln E",
    "origin_energy_slope": "+ origin slope in ln E",
    "beta_and_origin_energy_slopes": "+ both slopes",
    "depth_law_slope_and_offset": "+ depth law slope, offset",
    "tail_literature_rate": "+ tail, literature rate",
    "tail_literature_rate_energy_weight": "+ tail, literature rate, weight in ln E",
    "tail_free_rate": "+ tail, free rate",
    "ams_beta_fixed_with_tail": "beta = 0.65 fixed, with tail",
    "ams_beta_fixed_gamma_only": "beta = 0.65 fixed, gamma only",
}


def _energies(results: dict[str, Any]) -> list[str]:
    """Energies (as the keys used in the results) present in this run."""

    return [e for e in results["B_residual_structure"] if e != "layer0_energy_line"]


def _save(figure: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / name, dpi=DPI)
    plt.close(figure)


def extension_scores(results: dict[str, Any], directory: Path) -> None:
    rows = results["A_extensions_deposition"]
    names = list(rows)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    y = np.arange(len(names))
    axes[0].barh(y, [rows[n]["chi2_per_dof"] for n in names], color="#7f8c8d")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("chi-square per degree of freedom (in sample)")
    axes[0].set_yticks(y, [FAMILY_LABELS.get(n, n) for n in names], fontsize=8)
    axes[0].invert_yaxis()
    width = 0.27
    for k, (key, label, colour) in enumerate(
        (
            ("leave_one_energy_out", "leave one energy out", "#2980b9"),
            ("split_half", "split half", "#27ae60"),
            ("left_out_100_gev", "100 GeV left out", "#c0392b"),
        )
    ):
        axes[1].barh(
            y + (k - 1) * width,
            [rows[n].get("held_out_improvement_over_primary", {}).get(key, 0.0) for n in names],
            width,
            label=label,
            color=colour,
        )
    axes[1].axvline(0.0, color="black", linewidth=0.6)
    axes[1].axvline(0.2, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("held-out improvement over the gamma-only primary (dashed: 20%)")
    axes[1].set_yticks(y, [""] * len(names))
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8)
    energies = [float(e) for e in _energies(results)]
    for n in (
        "primary",
        "origin_energy_slope",
        "tail_literature_rate",
        "tail_literature_rate_energy_weight",
        "tail_free_rate",
    ):
        if n in rows:
            axes[2].plot(energies, np.array(rows[n]["rms_relative_layers_0_3"]) * 100, "o-", label=FAMILY_LABELS[n])
    axes[2].set_xscale("log")
    axes[2].set_xlabel("energy (GeV)")
    axes[2].set_ylabel("rms relative residual, layers 0-3 (%)")
    axes[2].legend(fontsize=7)
    fig.suptitle("Extensions of the mean profile: in-sample fit and held-out score (exposed development electrons)")
    _save(fig, directory, "extension_scores.png")


def residual_structure(results: dict[str, Any], directory: Path) -> None:
    block = results["B_residual_structure"]
    energies = [e for e in block if e != "layer0_energy_line"]
    fig, axes = plt.subplots(
        2, len(energies), figsize=(4.4 * len(energies), 7.5), constrained_layout=True, squeeze=False
    )
    layers = np.arange(18)
    for column, energy in enumerate(energies):
        for label, colour in (("gamma_only", "#c0392b"), ("gamma_plus_tail", "#27ae60")):
            axes[0, column].plot(
                layers,
                block[energy][label]["mean_residual_mev_by_layer"],
                "o-",
                color=colour,
                label=label.replace("_", " "),
            )
            axes[1, column].plot(
                layers, np.array(block[energy][label]["relative_residual_by_layer"]) * 100, "o-", color=colour
            )
        for row in (0, 1):
            axes[row, column].axhline(0.0, color="black", linewidth=0.6)
            axes[row, column].set_xlabel("layer")
        axes[0, column].set_title(f"{energy} GeV: mean (data - per-event fit), MeV")
    axes[1, 0].set_ylabel("percent of the mean layer energy")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Structure left by the per-event fits: layers 0-1 above, the middle below, containment too high")
    _save(fig, directory, "residual_structure.png")


def entrance_energy(results: dict[str, Any], directory: Path) -> None:
    block = results["B_residual_structure"]
    fine = results["B_fine_profile"]
    energies = [float(e) for e in fine]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    data = np.array([block[f"{e:g}"]["layer0_data_mev"] for e in energies])
    gamma_residual = np.array([block[f"{e:g}"]["gamma_only"]["mean_residual_mev_by_layer"][0] for e in energies])
    tail_residual = np.array([block[f"{e:g}"]["gamma_plus_tail"]["mean_residual_mev_by_layer"][0] for e in energies])
    axes[0].plot(energies, data, "ko-", label="Geant4 layer 0")
    axes[0].plot(energies, data - gamma_residual, "s--", color="#c0392b", label="per-event gamma fit")
    axes[0].plot(energies, data - tail_residual, "^--", color="#27ae60", label="gamma plus tail")
    line = block["layer0_energy_line"]
    axes[0].set_title(f"layer 0 energy: {line['intercept_mev']:.0f} MeV + {line['slope_mev_per_gev']:.2f} MeV/GeV x E")
    axes[0].set_xlabel("energy (GeV)")
    axes[0].set_ylabel("MeV")
    axes[0].legend(fontsize=8)
    for e in energies:
        axes[1].plot(np.arange(36) * 0.4635, fine[f"{e:g}"]["mean_fine_profile_mev"], label=f"{e:g} GeV")
    axes[1].set_xlim(0, 6)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("depth (X0 of the composite)")
    axes[1].set_ylabel("mean deposit per half-layer voxel (MeV)")
    axes[1].set_title("first six radiation lengths at half-layer resolution")
    axes[1].legend(fontsize=8)
    for e in energies:
        axes[2].plot(np.arange(18), fine[f"{e:g}"]["scintillator_fraction_by_layer"], "o-", label=f"{e:g} GeV")
    axes[2].set_ylim(0.05, 0.08)
    axes[2].set_xlabel("layer")
    axes[2].set_ylabel("scintillator / all-material deposit")
    axes[2].set_title("scintillator fraction by layer")
    axes[2].legend(fontsize=8)
    _save(fig, directory, "entrance_energy_and_fine_profile.png")


def tail_fit_residuals(results: dict[str, Any], directory: Path) -> None:
    rows = results["A_extensions_deposition"]
    energies = _energies(results)
    fig, axes = plt.subplots(1, len(energies), figsize=(4.5 * len(energies), 4.6), constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes)
    layers = np.arange(18)
    for k, energy in enumerate(energies):
        for name, colour in (
            ("primary", "#c0392b"),
            ("origin_energy_slope", "#e67e22"),
            ("tail_literature_rate", "#27ae60"),
            ("tail_free_rate", "#2c3e91"),
            ("ams_beta_fixed_with_tail", "#8e44ad"),
        ):
            if name not in rows:
                continue
            axes[k].plot(
                layers,
                np.array(rows[name]["relative_residual_by_layer"])[k] * 100,
                "o-",
                color=colour,
                label=FAMILY_LABELS[name],
                markersize=3,
            )
        axes[k].axhline(0, color="black", linewidth=0.6)
        axes[k].set_ylim(-45, 45)
        axes[k].set_title(f"{energy} GeV: (model - Geant4) / Geant4, %")
        axes[k].set_xlabel("layer")
    axes[0].legend(fontsize=7)
    _save(fig, directory, "mean_profile_residuals_with_tail.png")


def skew_consequence(results: dict[str, Any], directory: Path) -> None:
    metrics = (
        ("leakage_mean_difference", "mean leakage difference"),
        ("centred_leakage_ks", "centred leakage KS"),
        ("ks_contained", "contained-fraction KS"),
        ("leakage_q99_minus_mean", "99th leakage percentile - mean"),
        ("leakage_sd_log_ratio", "leakage sd |log ratio|"),
        ("ks_rms", "longitudinal width KS"),
    )
    families = (
        ("C_skew_consequence_gamma_only", "gamma only"),
        ("C_skew_consequence_gamma_plus_tail", "gamma plus tail"),
    )
    fig, axes = plt.subplots(len(families), len(metrics), figsize=(3.6 * len(metrics), 7.5), constrained_layout=True)
    for r, (key, title) in enumerate(families):
        block = results[key]
        energies = [e for e in block if e != "profile_family"]
        for c, (metric, label) in enumerate(metrics):
            ax = axes[r, c]
            for kind in ("gaussian", "skew_normal", "empirical"):
                means = [block[e][kind][metric]["mean"] for e in energies]
                sds = [block[e][kind][metric]["sd"] for e in energies]
                ax.errorbar(
                    [float(e) for e in energies], means, yerr=sds, fmt="o-", color=COLOURS[kind], label=kind.replace("_", " "), capsize=2
                )
            if metric == "leakage_q99_minus_mean":
                ax.plot(
                    [float(e) for e in energies],
                    [block[e]["geant4_leakage"]["q99_minus_mean"] for e in energies],
                    "k--",
                    label="Geant4",
                )
            ax.set_xscale("log")
            ax.set_title(f"{label}\n({title})", fontsize=8)
            ax.set_xlabel("energy (GeV)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        "Consequence of a skewed ln T: Gaussian, skew-normal and empirical joint draws (bars: sd over 5 ensemble seeds)"
    )
    _save(fig, directory, "skew_consequence.png")


def tail_artifact_control(results: dict[str, Any], directory: Path) -> None:
    block = results["C_tail_artifact_control"]
    energies = list(block)
    stoch_gamma = results["C_stochastic_gamma_only"]
    stoch_tail = results["C_stochastic_gamma_plus_tail"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    x = np.arange(len(energies))
    width = 0.2
    axes[0].bar(
        x - 1.5 * width,
        [block[e]["planted_skewness_ln_t"] for e in energies],
        width,
        label="planted (normal)",
        color="#bdc3c7",
    )
    axes[0].bar(
        x - 0.5 * width,
        [block[e]["gamma_only_refit_skewness_ln_t"] for e in energies],
        width,
        label="refit, gamma only",
        color="#c0392b",
    )
    axes[0].bar(
        x + 0.5 * width,
        [block[e]["tail_refit_skewness_ln_t"] for e in energies],
        width,
        label="refit, gamma + tail",
        color="#27ae60",
    )
    axes[0].set_title("planted-null control: skewness of ln T")
    axes[1].bar(
        x - width / 2,
        [stoch_gamma[e]["ln_t"]["skewness"] for e in energies],
        width,
        label="measured, gamma only",
        color="#c0392b",
    )
    axes[1].bar(
        x + width / 2,
        [stoch_tail[e]["ln_t"]["skewness"] for e in energies],
        width,
        label="measured, gamma + tail",
        color="#27ae60",
    )
    axes[1].set_title("measured skewness of ln T (Geant4 electrons)")
    for ax in axes:
        ax.set_xticks(x, [f"{e} GeV" for e in energies])
        ax.axhline(0, color="black", linewidth=0.6)
        ax.legend(fontsize=8)
    _save(fig, directory, "skewness_control.png")


def ams_beta(results: dict[str, Any], directory: Path) -> None:
    table = results.get("D_ams_beta_versus_origin")
    if not table:
        return
    origins = np.array(table["origins_x0"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True, sharey=True)
    for ax, representation in zip(axes, ("deposition", "readout"), strict=True):
        medians = np.array(table[representation]["median_beta_by_energy_and_origin"])
        for k, energy in enumerate(_energies(results)):
            ax.plot(origins, medians[k], "o-", label=f"{energy} GeV")
        ax.axhline(0.65, color="#c0392b", linestyle="--", label="0.65")
        ax.axhline(0.5, color="gray", linestyle=":", label="0.5 (PDG)")
        ax.set_xlabel("depth origin z0 (X0; negative = upstream)")
        ax.set_title(f"{representation}: median event beta")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("median of beta_i = (alpha_i - 1) / T_i")
    _save(fig, directory, "ams_beta_versus_origin.png")


def literature(results: dict[str, Any], directory: Path) -> None:
    block = results["L_literature_comparison"]["per_energy"]
    energies = [float(e) for e in block]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for ax, key, label in zip(axes, ("T", "alpha", "beta"), ("T (depth of the maximum, X0)", "alpha", "beta"), strict=True):
        ax.plot(
            energies,
            [block[f"{e:g}"][f"data_median_{key}"] for e in energies],
            "ko-",
            label="Geant4 electrons (front-face origin)",
        )
        ax.plot(
            energies,
            [block[f"{e:g}"]["grindhammer_peters_homogeneous"][key] for e in energies],
            "s--",
            color="#2980b9",
            label="Grindhammer-Peters homogeneous, lead",
        )
        ax.plot(
            energies,
            [block[f"{e:g}"]["pdg_electron"][key] for e in energies],
            "^--",
            color="#27ae60",
            label="PDG review (b = 0.5)",
        )
        ax.set_xscale("log")
        ax.set_xlabel("energy (GeV)")
        ax.set_title(label)
    axes[2].axhline(
        results["L_literature_comparison"]["mean_profile_front_face_free_beta"],
        color="#e67e22",
        label="Geant4 mean-profile fit",
    )
    axes[2].axhline(0.65, color="#c0392b", linestyle=":", label="AMS 0.65")
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    _save(fig, directory, "literature_comparison.png")


def make_plots(results: dict[str, Any], artefacts: dict[str, Any], directory: Path) -> None:
    extension_scores(results, directory)
    residual_structure(results, directory)
    entrance_energy(results, directory)
    tail_fit_residuals(results, directory)
    skew_consequence(results, directory)
    tail_artifact_control(results, directory)
    ams_beta(results, directory)
    literature(results, directory)
