"""Where does the lateral width excess of the Slice 2 generator come from? (diagnostic, analysis only)

Decision record ``research/COMPLETED_WORK.md`` (2026-10-08 rows), note ``research/COMPLETED_WORK.md (was 2026-10-07_em_production_slice2_lateral_note.md)``.
The development check found the width distribution far from Geant4 (KS about 0.7, median 16.1 against 13.9 mm at 20 GeV)
and extension E1, which refines the spots, did not cure it. The width of an event is the energy-weighted rms of the
measured coordinate about the track over all 72 cells,

    width^2 = sum_cells E_cell d^2 / E ,

so it weights the far cells by d^2 and a Cauchy-like radial law (density about 1/d^3) makes it a tail statistic.
Three hypotheses are separated here, on exposed electrons only (the sealed sets are never read):

* H1, the far tail of the radial law: the MEAN profile of the model carries too much energy at large distance;
* H2, spot granularity: the mean profile is right but a few coarse far spots make the width noisy and biased;
* H3, the depth coupling: the width follows the wrong events.

For the Geant4 baseline and hold-out, for Slice 2 with its spots, and for Slice 2 with infinitely many spots (the
expectation of the same events, ``E_layer x strip fraction``), the diagnostic reports the width quantiles, the mean
contribution to width^2 by distance bin and by layer, and the energy fraction beyond several distances. If the
expectation already has the excess and the excess sits in the far bins, H1 holds; if only the spot version has it, H2.

Command: ``python -m ams_ecal.electron_studies.em_lateral_width_diagnostic``
"""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from ams_ecal.electron_model import em_production_lateral as lateral
from ams_ecal.electron_model.em_production import PROJECT_ROOT
from ams_ecal.electron_studies.em_production_lateral_check import (
    BASELINE_DIR,
    ENERGIES_GEV,
    ENTRY_SPOT_MM,
    EXTENDED_DIR,
    reference_sample,
)

RESULTS_JSON = PROJECT_ROOT / "results" / "em_generator" / "em_production_lateral_width_diagnostic.json"
PLOT_PATH = PROJECT_ROOT / "results" / "em_generator" / "em_production_lateral_width_diagnostic.png"
DISTANCE_BINS_MM = (0.0, 9.0, 27.0, 54.0, 108.0, 216.0, 700.0)
TAIL_DISTANCES_MM = (27.0, 54.0, 108.0, 216.0)
ENSEMBLE_EVENTS = 4000
SEED = 20261009
QUANTILES = (0.05, 0.5, 0.95)


def cell_offsets_mm(model: lateral.EMProductionLateralModel, axis_mm: np.ndarray) -> np.ndarray:
    """``(n, layers, cells)`` offset of every cell centre from the track, in mm."""

    edges = model.cell_edges_mm
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres[None, None, :] - np.asarray(axis_mm, dtype=float)[:, :, None]


def width_mm(grid: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    energy = grid.sum(axis=(1, 2))
    return np.sqrt((grid * offsets**2).sum(axis=(1, 2)) / energy)


def width_squared_by_distance(grid: np.ndarray, offsets: np.ndarray, bins_mm: tuple[float, ...]) -> np.ndarray:
    """Mean over events of each distance bin's contribution to width^2, in mm^2 (the bins sum to the mean width^2)."""

    energy = grid.sum(axis=(1, 2))[:, None, None]
    distance = np.abs(offsets)
    contributions = []
    for low, high in pairwise(bins_mm):
        inside = (distance >= low) & (distance < high)
        contributions.append(((grid * offsets**2 * inside) / energy).sum(axis=(1, 2)).mean())
    return np.array(contributions)


def width_squared_by_layer(grid: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    energy = grid.sum(axis=(1, 2))[:, None]
    return ((grid * offsets**2).sum(axis=2) / energy).mean(axis=0)


def energy_beyond(grid: np.ndarray, offsets: np.ndarray, distances_mm: tuple[float, ...]) -> np.ndarray:
    """Mean over events of the energy fraction at least ``d`` mm from the track, for each ``d``."""

    energy = grid.sum(axis=(1, 2))[:, None, None]
    distance = np.abs(offsets)
    return np.array([((grid * (distance >= d)) / energy).sum(axis=(1, 2)).mean() for d in distances_mm])


def summarise(grid: np.ndarray, offsets: np.ndarray) -> dict[str, Any]:
    w = width_mm(grid, offsets)
    beyond = energy_beyond(grid, offsets, TAIL_DISTANCES_MM).tolist()
    return {
        "width_quantiles_mm": np.quantile(w, QUANTILES).tolist(),
        "mean_width_squared_mm2": float((w**2).mean()),
        "width_squared_by_distance_mm2": width_squared_by_distance(grid, offsets, DISTANCE_BINS_MM).tolist(),
        "width_squared_by_layer_mm2": width_squared_by_layer(grid, offsets).tolist(),
        "energy_beyond": dict(zip((f"{d:g}" for d in TAIL_DISTANCES_MM), beyond, strict=True)),
    }


def model_grids(
    model: lateral.EMProductionLateralModel, energy_gev: float, events: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(spots grid, expectation grid, axis)`` of the same events: the expectation has infinitely many spots."""

    rng = np.random.default_rng(seed)
    entry = rng.uniform(*ENTRY_SPOT_MM, size=(events, 2))
    axis = model.axis_for_entry(entry[:, 0], entry[:, 1])
    prepared = model.prepare_batch(1000.0 * energy_gev, axis, rng)
    spots = model.draw_cells(prepared, rng).cell_energy_mev
    layer_energy = prepared.energy_mev * prepared.layer_fractions
    expectation = layer_energy[..., None] * prepared.probabilities[..., :-1]
    return spots, expectation, axis


def diagnose(
    energies: tuple[float, ...] = ENERGIES_GEV,
    *,
    artifact: Path = lateral.DEFAULT_ARTIFACT,
    baseline_dir: Path = BASELINE_DIR,
    extended_dir: Path = EXTENDED_DIR,
    events: int = ENSEMBLE_EVENTS,
) -> dict[str, Any]:
    model = lateral.load_em_production_lateral(artifact)
    out: dict[str, Any] = {
        "status": "diagnostic on exposed electrons; the sealed sets are not read; no model is changed",
        "sealed_data_read": False,
        "distance_bins_mm": list(DISTANCE_BINS_MM),
        "tail_distances_mm": list(TAIL_DISTANCES_MM),
        "per_energy": {},
    }
    for energy in energies:
        row: dict[str, Any] = {}
        for label, directory in (("geant4_baseline", baseline_dir), ("geant4_holdout", extended_dir)):
            sample = reference_sample(directory, energy)
            axis = model.axis_for_entry(sample.entry_x_mm, sample.entry_y_mm)
            row[label] = summarise(sample.grid, cell_offsets_mm(model, axis))
        spots, expectation, axis = model_grids(model, energy, events, SEED + int(energy))
        offsets = cell_offsets_mm(model, axis)
        row["slice2_with_spots"] = summarise(spots, offsets)
        row["slice2_expectation"] = summarise(expectation, offsets)
        reference = np.array(row["geant4_holdout"]["width_squared_by_distance_mm2"])
        excess = np.array(row["slice2_expectation"]["width_squared_by_distance_mm2"]) - reference
        row["expectation_excess_width_squared_by_distance_mm2"] = excess.tolist()
        row["fraction_of_expectation_excess_beyond_54_mm"] = (
            float(excess[3:].sum() / excess.sum()) if excess.sum() else None
        )
        out["per_energy"][f"{energy:g}"] = row
    return out


def make_plot(results: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    energies = list(results["per_energy"])
    labels = [f"{a:g}-{b:g}" for a, b in pairwise(DISTANCE_BINS_MM)]
    fig, axes = plt.subplots(1, len(energies), figsize=(4.2 * len(energies), 3.8), sharey=True)
    x = np.arange(len(labels))
    for ax, energy in zip(np.atleast_1d(axes), energies, strict=True):
        row = results["per_energy"][energy]
        for shift, key, name in (
            (-0.27, "geant4_holdout", "Geant4 hold-out"),
            (0.0, "slice2_with_spots", "Slice 2 (spots)"),
            (0.27, "slice2_expectation", "Slice 2 (expectation)"),
        ):
            ax.bar(x + shift, row[key]["width_squared_by_distance_mm2"], 0.27, label=name)
        ax.set_yscale("log")
        ax.set_xticks(x, labels, rotation=45, fontsize=7)
        ax.set_title(f"{energy} GeV")
        ax.set_xlabel("distance from the track (mm)")
    np.atleast_1d(axes)[0].set_ylabel("mean contribution to width^2 (mm^2)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS_JSON)
    parser.add_argument("--plot", type=Path, default=PLOT_PATH)
    parser.add_argument("--events", type=int, default=ENSEMBLE_EVENTS)
    parser.add_argument("--artifact", type=Path, default=lateral.DEFAULT_ARTIFACT)
    args = parser.parse_args(argv)
    results = diagnose(artifact=args.artifact, events=args.events)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    make_plot(results, args.plot)
    keys = ("geant4_holdout", "slice2_with_spots", "slice2_expectation")
    for energy, row in results["per_energy"].items():
        q = {k: [round(v, 1) for v in row[k]["width_quantiles_mm"]] for k in keys}
        print(f"{energy} GeV width q05/50/95: {q}")


if __name__ == "__main__":
    main()
