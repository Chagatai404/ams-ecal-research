"""Figures for the Slice 2 lateral development check: profile error, spot numbers and distances."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LATERAL_KEYS = (
    "width_mm",
    "core_fraction",
    "containment_fraction",
    "n_hit_cells",
    "max_cell_fraction",
)


def make_plots(results: dict[str, Any], artefacts: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    energies = list(artefacts)
    layers = np.arange(18)

    fig, axes = plt.subplots(1, len(energies), figsize=(4 * len(energies), 3.6), sharey=True)
    for ax, energy in zip(np.atleast_1d(axes), energies, strict=True):
        row = artefacts[energy]
        ax.plot(
            layers,
            row["calibration_set"]["slice2"]["profile_error_per_layer"],
            "o-",
            label="calibration set",
        )
        ax.plot(
            layers,
            row["development_holdout"]["slice2"]["profile_error_per_layer"],
            "s-",
            label="hold-out",
        )
        ax.plot(
            layers,
            row["sample_to_sample_floor"]["profile_error_points"]["per_layer"],
            "k:",
            label="Geant4 floor",
        )
        ax.axhline(2.0, color="grey", lw=0.8, ls="--")
        ax.set_title(f"{energy} GeV")
        ax.set_xlabel("layer")
    np.atleast_1d(axes)[0].set_ylabel("mean cell-profile error (points of layer energy)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "profile_error_by_layer.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(energies), figsize=(4 * len(energies), 3.6), sharey=True)
    for ax, energy in zip(np.atleast_1d(axes), energies, strict=True):
        spots = artefacts[energy]["development_holdout"]["slice2"]["spot_number_conditioned"]
        ax.semilogy(layers, spots["geant4"], "o-", label="Geant4 (hold-out)")
        ax.semilogy(layers, spots["model"], "s-", label="Slice 2")
        ax.set_title(f"{energy} GeV")
        ax.set_xlabel("layer")
    np.atleast_1d(axes)[0].set_ylabel("conditioned spot number N_S")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "spot_numbers.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(energies), figsize=(4 * len(energies), 3.6), sharey=True)
    x = np.arange(len(LATERAL_KEYS))
    for ax, energy in zip(np.atleast_1d(axes), energies, strict=True):
        row = artefacts[energy]
        hold = row["development_holdout"]
        ax.bar(
            x - 0.27,
            [hold["slice2"]["ks"][k] for k in LATERAL_KEYS],
            0.27,
            label="Slice 2",
        )
        ax.bar(
            x,
            [hold["slice1_with_dec001_lateral"]["ks"][k] for k in LATERAL_KEYS],
            0.27,
            label="Slice 1 + DEC-001 lateral",
        )
        ax.bar(
            x + 0.27,
            [row["sample_to_sample_floor"]["ks"][k] for k in LATERAL_KEYS],
            0.27,
            label="Geant4 floor",
        )
        ax.set_xticks(x, [k.replace("_", "\n") for k in LATERAL_KEYS], fontsize=7)
        ax.set_title(f"{energy} GeV (hold-out)")
    np.atleast_1d(axes)[0].set_ylabel("KS distance")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(directory / "ks_distances.png", dpi=120)
    plt.close(fig)
