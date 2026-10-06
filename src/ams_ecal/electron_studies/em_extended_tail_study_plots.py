"""Plots of the extended-depth tail study (``ams_ecal.electron_studies.em_extended_tail_study``): figures only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ams_ecal.electron_studies import em_extended_tail_study as study

DPI = 100
COLOURS = {
    "primary": "#c0392b",
    "tail_literature_rate": "#27ae60",
    "tail_literature_rate_energy_weight": "#2c3e91",
    "tail_free_rate": "#e67e22",
    "exponential_tail_literature_rate": "#8e44ad",
    "exponential_tail_free_rate": "#7f8c8d",
}
LABELS = {
    "primary": "gamma only",
    "tail_literature_rate": "tail, literature rate",
    "tail_literature_rate_energy_weight": "tail, literature rate, weight in ln E",
    "tail_free_rate": "tail, free rate",
    "exponential_tail_literature_rate": "exponential tail, literature rate",
    "exponential_tail_free_rate": "exponential tail, free rate",
}
LITERATURE_RATE = (1.0 / 3.9, 1.0 / 3.3)  # Leroy and Rancoita Table 2, lead


def _save(figure: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / name, dpi=DPI)
    plt.close(figure)


def _energies(results: dict[str, Any]) -> list[str]:
    return list(results["sample"]["events_per_energy"])


def profiles(results: dict[str, Any], artefacts: dict[str, Any], directory: Path) -> None:
    events = artefacts["events"]
    prediction = results["B_prefix_only_prediction"]
    specs = study.prefix_specs()
    fig, axes = plt.subplots(1, len(events), figsize=(4.6 * len(events), 4.8), constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes)
    depth = (np.arange(80) + 0.5) * study.LAYER_X0
    ec = 7.6
    for k, (energy, ext) in enumerate(events.items()):
        axes[k].semilogy(depth, np.maximum(ext.deposit.mean(axis=0)[:80], 1e-7), "ko", markersize=3, label="Geant4")
        for name in ("primary", "tail_literature_rate_energy_weight", "tail_free_rate"):
            if name not in prediction:
                continue
            params = prediction[name]["parameters"]
            model = study.predicted_layer_fractions(specs[name], params, np.array([energy]), ec, 80)[0]
            axes[k].semilogy(depth, np.maximum(model, 1e-7), color=COLOURS[name], label=LABELS[name])
        axes[k].axvline(study.PREFIX_LAYERS * study.LAYER_X0, color="gray", linestyle=":")
        axes[k].set_ylim(1e-6, 0.2)
        axes[k].set_title(f"{energy:g} GeV: fitted on the prefix only (dotted line)")
        axes[k].set_xlabel("depth (X0)")
    axes[0].set_ylabel("mean layer energy / E")
    axes[0].legend(fontsize=7)
    _save(fig, directory, "profiles_prefix_fits_extrapolated.png")


def decay_rates(results: dict[str, Any], directory: Path) -> None:
    local = results["C_local_decay_rates"]
    windows = local["windows_layers"]
    centres = [0.5 * (a + b) * study.LAYER_X0 for a, b in windows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for energy in _energies(results):
        axes[0].plot(centres, local[energy], "o-", label=f"{energy} GeV")
    axes[0].axhspan(*LITERATURE_RATE, color="#2ecc71", alpha=0.25, label="lead, Leroy and Rancoita (3.3-3.9 X0)")
    axes[0].set_xlabel("depth of the window (X0)")
    axes[0].set_ylabel("local decay rate of the mean layer energy (per X0)")
    axes[0].legend(fontsize=8)
    decay = results["C_decay_length"]
    energies = [float(e) for e in _energies(results)]
    rates = [decay[f"{e:g}"]["rate_per_x0"] for e in energies]
    errors = [decay[f"{e:g}"]["rate_sd_bootstrap"] for e in energies]
    axes[1].errorbar(energies, rates, yerr=errors, fmt="ko-", capsize=3, label="layers 22-60")
    axes[1].axhspan(*LITERATURE_RATE, color="#2ecc71", alpha=0.25)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("energy (GeV)")
    axes[1].set_ylabel("decay rate (per X0)")
    axes[1].set_title(f"slope per ln E: {decay['rate_slope_per_ln_energy']:+.4f}")
    axes[1].legend(fontsize=8)
    _save(fig, directory, "tail_decay_rate.png")


def prediction_ratios(results: dict[str, Any], directory: Path) -> None:
    prediction = results["B_prefix_only_prediction"]
    families = [k for k, v in prediction.items() if isinstance(v, dict)]
    energies = _energies(results)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    width = 0.8 / max(len(families), 1)
    x = np.arange(len(energies))
    measured_prefix = np.array(prediction["measured_contained_prefix"])
    for j, name in enumerate(families):
        row = prediction[name]
        offset = x + (j - len(families) / 2 + 0.5) * width
        axes[0].bar(
            offset, row["ratio_predicted_over_measured"], width, color=COLOURS.get(name), label=LABELS.get(name, name)
        )
        axes[1].bar(
            offset,
            np.array(row["predicted_layers_30_plus"]) / np.array(row["measured_layers_30_plus"]),
            width,
            color=COLOURS.get(name),
        )
        axes[2].bar(
            offset,
            100 * (np.array(row["predicted_contained_prefix"]) - measured_prefix),
            width,
            color=COLOURS.get(name),
        )
    for ax in axes[:2]:
        ax.axhline(1.0, color="black", linewidth=0.8)
    axes[1].set_yscale("log")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("energy behind layer 17: predicted / measured")
    axes[1].set_title("energy in layers 30+: predicted / measured")
    axes[2].set_title("contained in the prefix: predicted - measured (points)")
    for ax in axes:
        ax.set_xticks(x, [f"{e} GeV" for e in energies])
    axes[0].legend(fontsize=7)
    _save(fig, directory, "prefix_only_prediction.png")


def event_level(results: dict[str, Any], artefacts: dict[str, Any], directory: Path) -> None:
    events = artefacts["events"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for energy, ext in events.items():
        axes[0].hist(
            ext.behind_prefix, bins=np.linspace(0.0, 0.3, 61), histtype="step", density=True, label=f"{energy:g} GeV"
        )
        axes[1].hist(
            ext.deposit[:, 30:].sum(axis=1),
            bins=np.linspace(0.0, 0.02, 61),
            histtype="step",
            density=True,
            label=f"{energy:g} GeV",
        )
    axes[0].set_xlabel("energy behind layer 17 / E")
    axes[1].set_xlabel("energy in layers 30+ / E")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=8)
    block = results.get("D_event_level", {})
    if block:
        lines = [
            f"{e} GeV: r2 of the prefix gamma {block[e]['r2_of_prefix_gamma_for_behind']:.2f}, sd gamma/measured "
            f"{block[e]['sd_behind_prefix_from_prefix_gamma'] / block[e]['sd_behind_prefix_measured']:.2f}"
            for e in _energies(results)
        ]
        axes[0].set_title("; ".join(lines[:2]) + "\n" + "; ".join(lines[2:]), fontsize=7)
    _save(fig, directory, "behind_prefix_per_event.png")


def consistency(results: dict[str, Any], directory: Path) -> None:
    block = results["A_prefix_consistency"]
    fig, ax = plt.subplots(figsize=(8, 4.4), constrained_layout=True)
    for energy, row in block.items():
        ax.plot(np.arange(18), row["z_by_layer"], "o-", label=f"{energy} GeV (rms {row['rms_z']:.2f})")
    ax.axhspan(-2, 2, color="#dddddd")
    ax.set_xlabel("layer")
    ax.set_ylabel("(extended - baseline) / standard error")
    ax.set_title("the new sample against the baseline sample, prefix layers")
    ax.legend(fontsize=8)
    _save(fig, directory, "prefix_consistency.png")


def make_plots(results: dict[str, Any], artefacts: dict[str, Any], directory: Path) -> None:
    profiles(results, artefacts, directory)
    decay_rates(results, directory)
    prediction_ratios(results, directory)
    event_level(results, artefacts, directory)
    consistency(results, directory)
