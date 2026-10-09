"""Cell occupancy of Geant4 electrons and of the lateral candidates, by distance from the track (diagnostic, analysis only).

The classifier diagnostic (``em_lateral_distinguishability``) found that the models are separated from Geant4 completely by
the cell occupancy alone: the number of cells above the cell threshold, in total and in every layer. This diagnostic asks
what the occupancy is, on the exposed electrons only (the sealed sets are never read), so the physics of the missing
occupancy can be formulated on its own terms and not by tuning to a classifier:

* by distance from the track: the fraction of cells above the threshold, the fraction exactly empty, the mean energy of an
  occupied cell and of the energy-weighted cell, the number of cells and the energy per event in each distance bin;
* by layer group (front 0-2, middle 3-11, rear 12-17) for the cells at least 27 mm from the track;
* the implied quantum: the energy in a distance bin divided by its number of occupied cells, to be compared with the
  spot energy of the model (a layer's energy over its spot number, tens of MeV).

Command: ``python -m ams_ecal.electron_studies.em_lateral_occupancy_diagnostic``
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
from ams_ecal.electron_studies.em_lateral_width_diagnostic import cell_offsets_mm
from ams_ecal.electron_studies.em_production_lateral_check import (
    BASELINE_DIR,
    ENERGIES_GEV,
    EXTENDED_DIR,
    reference_sample,
    slice2_sample,
)
from ams_ecal.proton_model.proton_validation import CELL_THRESHOLD_MEV

RESULTS_JSON = PROJECT_ROOT / "results" / "em_generator" / "em_production_lateral_occupancy_diagnostic.json"
FASTER_TAIL_ARTIFACT = (
    PROJECT_ROOT / "data" / "calibration" / "em_production" / "gp_lateral_faster_tail_v1" / "parameters.json"
)
DISTANCE_BINS_MM = (0.0, 9.0, 27.0, 54.0, 108.0, 216.0, 700.0)
LAYER_GROUPS = {"front_0_2": slice(0, 3), "middle_3_11": slice(3, 12), "rear_12_17": slice(12, 18)}
FAR_MM = 27.0
EVENTS = 2000
SEED = 20261012


def bin_masks(offsets: np.ndarray, bins_mm: tuple[float, ...]) -> list[np.ndarray]:
    distance = np.abs(offsets)
    return [(distance >= low) & (distance < high) for low, high in pairwise(bins_mm)]


def occupancy_row(grid: np.ndarray, mask: np.ndarray, threshold_mev: float = CELL_THRESHOLD_MEV) -> dict[str, float]:
    """Occupancy statistics of the cells selected by ``mask`` (same shape as ``grid``), averaged over events."""

    events = len(grid)
    selected = np.where(mask, grid, 0.0)
    occupied = mask & (grid > threshold_mev)
    n_occupied = int(occupied.sum())
    n_cells_total = int(mask.sum())
    energy_total = float(selected.sum())
    return {
        "cells_per_event": n_cells_total / events,
        "occupied_fraction": n_occupied / n_cells_total if n_cells_total else 0.0,
        "empty_fraction": float((mask & (grid <= 0.0)).sum() / n_cells_total) if n_cells_total else 0.0,
        "mean_energy_of_occupied_cell_mev": float(grid[occupied].mean()) if n_occupied else 0.0,
        "energy_per_event_mev": energy_total / events,
        "energy_per_occupied_cell_mev": energy_total / n_occupied if n_occupied else 0.0,
    }


def summarise(grid: np.ndarray, offsets: np.ndarray) -> dict[str, Any]:
    masks = bin_masks(offsets, DISTANCE_BINS_MM)
    labels = [f"{a:g}-{b:g}" for a, b in pairwise(DISTANCE_BINS_MM)]
    by_distance = {label: occupancy_row(grid, mask) for label, mask in zip(labels, masks, strict=True)}
    far = np.abs(offsets) >= FAR_MM
    by_layer_group = {}
    for name, layers in LAYER_GROUPS.items():
        layer_mask = np.zeros_like(far)
        layer_mask[:, layers, :] = True
        by_layer_group[name] = occupancy_row(grid, far & layer_mask)
    return {
        "by_distance_mm": by_distance,
        "cells_at_least_27_mm_by_layer_group": by_layer_group,
        "occupied_cells_per_event": float((grid > CELL_THRESHOLD_MEV).sum() / len(grid)),
    }


def diagnose(
    energies: tuple[float, ...] = ENERGIES_GEV,
    *,
    base_artifact: Path = lateral.DEFAULT_ARTIFACT,
    candidate_artifact: Path = FASTER_TAIL_ARTIFACT,
    baseline_dir: Path = BASELINE_DIR,
    extended_dir: Path = EXTENDED_DIR,
    events: int = EVENTS,
) -> dict[str, Any]:
    models = {
        "slice2": lateral.load_em_production_lateral(base_artifact),
        "faster_tail": lateral.load_em_production_lateral(candidate_artifact),
    }
    out: dict[str, Any] = {
        "status": "diagnostic on exposed electrons; the sealed sets are not read; no model is changed",
        "sealed_data_read": False,
        "cell_threshold_mev": CELL_THRESHOLD_MEV,
        "distance_bins_mm": list(DISTANCE_BINS_MM),
        "per_energy": {},
    }
    for energy in energies:
        row: dict[str, Any] = {}
        for label, directory in (("geant4_baseline", baseline_dir), ("geant4_holdout", extended_dir)):
            sample = reference_sample(directory, energy)
            axis = models["slice2"].axis_for_entry(sample.entry_x_mm, sample.entry_y_mm)
            row[label] = summarise(sample.grid, cell_offsets_mm(models["slice2"], axis))
        for name, model in models.items():
            sample = slice2_sample(model, energy, events, SEED + int(energy))
            axis = model.axis_for_entry(sample.entry_x_mm, sample.entry_y_mm)
            row[name] = summarise(sample.grid, cell_offsets_mm(model, axis))
        out["per_energy"][f"{energy:g}"] = row
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS_JSON)
    parser.add_argument("--events", type=int, default=EVENTS)
    args = parser.parse_args(argv)
    results = diagnose(events=args.events)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for energy, row in results["per_energy"].items():
        parts = {k: round(row[k]["occupied_cells_per_event"]) for k in ("geant4_holdout", "slice2", "faster_tail")}
        print(f"{energy} GeV occupied cells per event: {parts}")


if __name__ == "__main__":
    main()
