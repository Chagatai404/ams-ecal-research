"""Plots of the Slice 1 development check (``em_production_development_check``): figures only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 100
METRICS = (
    ("layer0_log_ratio", "layer 0 mean |log ratio|"),
    ("early4_log_ratio", "layers 0-3 mean |log ratio|"),
    ("profile_error_layers_0_3", "profile error, layers 0-3"),
    ("profile_error_layers_4_17", "profile error, layers 4-17"),
    ("leakage_mean_difference", "mean leakage difference"),
    ("leakage_sd_log_ratio", "leakage sd |log ratio|"),
    ("ks_contained", "contained fraction KS"),
    ("ks_cog", "centre of gravity KS"),
    ("ks_rms", "longitudinal width KS"),
    ("max_abs_lag_correlation_difference", "max lag-correlation difference"),
)


def _save(figure: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / name, dpi=DPI)
    plt.close(figure)


def layer_errors(results: dict[str, Any], directory: Path) -> None:
    energies = list(results["per_energy"])
    fig, axes = plt.subplots(
        1, len(energies), figsize=(4.6 * len(energies), 4.6), constrained_layout=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    layers = np.arange(18)
    for ax, energy in zip(axes, energies, strict=True):
        block = results["per_energy"][energy]
        ax.plot(
            layers,
            100 * np.array(block["relative_layer_error_slice1_vs_baseline"]),
            "o-",
            color="#27ae60",
            label="Slice 1 vs calibration set",
        )
        ax.plot(
            layers,
            100 * np.array(block["relative_layer_error_slice1_vs_holdout"]),
            "s--",
            color="#2c3e91",
            label="Slice 1 vs development hold-out",
        )
        ax.plot(
            layers,
            100 * np.array(block["relative_layer_error_dec001_vs_baseline"]),
            "^:",
            color="#c0392b",
            label="DEC-001 vs calibration set",
        )
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_ylim(-40, 40)
        ax.set_title(f"{energy} GeV: (generator - Geant4) / Geant4, %")
        ax.set_xlabel("layer")
    axes[0].legend(fontsize=7)
    _save(fig, directory, "mean_layer_errors.png")


def distances(results: dict[str, Any], directory: Path) -> None:
    energies = list(results["per_energy"])
    fig, axes = plt.subplots(2, 5, figsize=(22, 8), constrained_layout=True)
    x = np.arange(len(energies))
    width = 0.2
    per = results["per_energy"]
    for ax, (key, label) in zip(axes.ravel(), METRICS, strict=True):
        hold = [per[e]["development_holdout"]["slice1"][key]["mean"] for e in energies]
        spread = [per[e]["development_holdout"]["slice1"][key]["sd"] for e in energies]
        control = [per[e]["development_holdout"]["dec001"][key] for e in energies]
        floor = [per[e]["sample_to_sample_floor"][key] for e in energies]
        calibration = [per[e]["calibration_set"]["slice1"][key]["mean"] for e in energies]
        ax.bar(x - 1.5 * width, floor, width, color="#bdc3c7", label="Geant4 sample vs sample (floor)")
        ax.bar(x - 0.5 * width, calibration, width, color="#27ae60", label="Slice 1, calibration set")
        ax.bar(
            x + 0.5 * width,
            hold,
            width,
            yerr=spread,
            color="#2c3e91",
            label="Slice 1, development hold-out",
            capsize=2,
        )
        ax.bar(x + 1.5 * width, control, width, color="#c0392b", label="DEC-001, development hold-out")
        ax.set_yscale("log")
        ax.set_title(label, fontsize=9)
        ax.set_xticks(x, [f"{e}" for e in energies])
        ax.set_xlabel("energy (GeV)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        "Distance from Geant4 (log scale; lower is closer): Slice 1 against DEC-001 and the sample-to-sample floor"
    )
    _save(fig, directory, "distances.png")


def leakage(artefacts: dict[str, Any], directory: Path) -> None:
    energies = list(artefacts["ensembles"])
    fig, axes = plt.subplots(1, len(energies), figsize=(4.6 * len(energies), 4.4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    bins = np.linspace(0.0, 0.35, 71)
    for ax, energy in zip(axes, energies, strict=True):
        ax.hist(
            1 - artefacts["baseline"][energy].sum(axis=1),
            bins=bins,
            histtype="stepfilled",
            color="#dddddd",
            density=True,
            label="Geant4 calibration set",
        )
        ax.hist(
            1 - artefacts["holdout"][energy].sum(axis=1),
            bins=bins,
            histtype="step",
            color="black",
            density=True,
            label="Geant4 development hold-out",
        )
        ax.hist(
            1 - artefacts["ensembles"][energy]["slice1"].sum(axis=1),
            bins=bins,
            histtype="step",
            color="#27ae60",
            density=True,
            label="Slice 1",
        )
        ax.hist(
            1 - artefacts["ensembles"][energy]["dec001"].sum(axis=1),
            bins=bins,
            histtype="step",
            color="#c0392b",
            density=True,
            label="DEC-001",
        )
        ax.set_yscale("log")
        ax.set_title(f"{energy:g} GeV: energy beyond the last layer")
        ax.set_xlabel("1 - contained fraction")
    axes[0].legend(fontsize=7)
    _save(fig, directory, "leakage_distributions.png")


def moments(results: dict[str, Any], directory: Path) -> None:
    energies = list(results["per_energy"])
    labels = (
        ("sd_ln_t", "sd of ln T"),
        ("skewness_ln_t", "skewness of ln T"),
        ("sd_ln_alpha", "sd of ln alpha"),
        ("pearson", "rho(ln T, ln alpha)"),
        ("beta_median", "median beta"),
    )
    fig, axes = plt.subplots(1, len(labels), figsize=(4.4 * len(labels), 4.2), constrained_layout=True)
    values = [float(e) for e in energies]
    for ax, (key, label) in zip(axes, labels, strict=True):
        for name, colour, style in (
            ("geant4_baseline", "black", "o-"),
            ("geant4_holdout", "#7f8c8d", "s--"),
            ("slice1", "#27ae60", "^-"),
            ("dec001", "#c0392b", "v:"),
        ):
            ax.plot(
                values,
                [results["per_energy"][e][f"moments_{name}"][key] for e in energies],
                style,
                color=colour,
                label=name.replace("_", " "),
            )
        ax.set_xscale("log")
        ax.set_title(label)
        ax.set_xlabel("energy (GeV)")
    axes[0].legend(fontsize=7)
    _save(fig, directory, "per_event_fit_moments.png")


def make_plots(results: dict[str, Any], artefacts: dict[str, Any], directory: Path) -> None:
    layer_errors(results, directory)
    distances(results, directory)
    leakage(artefacts, directory)
    moments(results, directory)
