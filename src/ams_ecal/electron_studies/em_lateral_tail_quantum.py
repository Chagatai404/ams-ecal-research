"""Extension E2: measure the tail's energy quantum on the baseline electrons and build the two E2 candidates.

The occupancy diagnostic (``em_lateral_occupancy_diagnostic``) found a constant MeV-scale energy per occupied cell far from
the track in Geant4 (1.2-1.5 MeV at least 108 mm away) against 7.7-9.1 MeV in the model. E2 gives the radial TAIL's spots that
fixed energy. The quantum is MEASURED here, once, on the exposed baseline electrons (the calibration set), pooled over the
four energies, and is not fitted to the hit-cell count, the width or any classifier. The rule that decides adoption is fixed in
``em_production_lateral_e2_decision`` and was written before this module was run.

Two candidates, each with ONE added parameter over its base: ``e2`` on Slice 2 and ``faster_tail_e2`` on the faster-tail
candidate. Neither base is refitted. Outputs per tag: the parameters artifact, ``..._quantum.json`` and ``..._occupancy.json``.

Command: ``python -m ams_ecal.electron_studies.em_lateral_tail_quantum``
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ams_ecal.electron_model import em_production_lateral as lateral
from ams_ecal.electron_model.em_production import PROJECT_ROOT
from ams_ecal.electron_studies.em_lateral_occupancy_diagnostic import (
    FASTER_TAIL_ARTIFACT,
)
from ams_ecal.electron_studies.em_lateral_width_diagnostic import cell_offsets_mm
from ams_ecal.electron_studies.em_production_lateral_check import (
    BASELINE_DIR,
    ENERGIES_GEV,
    EXTENDED_DIR,
    reference_sample,
    slice2_sample,
)
from ams_ecal.proton_model.proton_validation import CELL_THRESHOLD_MEV

RESULTS = PROJECT_ROOT / "results" / "em_generator"
CALIBRATION_ROOT = PROJECT_ROOT / "data" / "calibration" / "em_production"
FAR_MM = 108.0
EVENTS = 2000
SEED = 20261013
CANDIDATES = {"e2": lateral.DEFAULT_ARTIFACT, "faster_tail_e2": FASTER_TAIL_ARTIFACT}


def occupied_energy_far(grid: np.ndarray, offsets: np.ndarray, far_mm: float = FAR_MM) -> tuple[float, int]:
    """``(energy, occupied cells)`` summed over the cells at least ``far_mm`` from the track."""

    mask = np.abs(offsets) >= far_mm
    occupied = mask & (grid > CELL_THRESHOLD_MEV)
    return float(np.where(occupied, grid, 0.0).sum()), int(occupied.sum())


def measure_tail_quantum(
    baseline_dir: Path = BASELINE_DIR, energies: tuple[float, ...] = ENERGIES_GEV, far_mm: float = FAR_MM
) -> dict[str, Any]:
    """The pooled mean energy of an occupied cell at least ``far_mm`` from the track, from the exposed baseline electrons.

    Pooled: total energy over total occupied cells, summed over the energies (not a mean of the per-energy means).
    """

    geometry_model = lateral.load_em_production_lateral()
    per_energy, energy_sum, cell_sum = {}, 0.0, 0
    for energy in energies:
        sample = reference_sample(baseline_dir, energy)
        axis = geometry_model.axis_for_entry(sample.entry_x_mm, sample.entry_y_mm)
        total, cells = occupied_energy_far(sample.grid, cell_offsets_mm(geometry_model, axis), far_mm)
        per_energy[f"{energy:g}"] = total / cells
        energy_sum, cell_sum = energy_sum + total, cell_sum + cells
    return {
        "status": "measured on the exposed baseline electrons (calibration set); not fitted; the sealed sets are not read",
        "far_mm": far_mm,
        "cell_threshold_mev": CELL_THRESHOLD_MEV,
        "tail_quantum_mev": energy_sum / cell_sum,
        "per_energy_mev": per_energy,
        "occupied_cells_used": cell_sum,
    }


def occupied_cells_per_event(model: lateral.EMProductionLateralModel, energy: float, events: int, seed: int) -> float:
    sample = slice2_sample(model, energy, events, seed)
    return float((sample.grid > CELL_THRESHOLD_MEV).sum() / len(sample.grid))


def occupancy_record(
    artifact: Path,
    *,
    baseline_dir: Path = BASELINE_DIR,
    extended_dir: Path = EXTENDED_DIR,
    energies: tuple[float, ...] = ENERGIES_GEV,
    events: int = EVENTS,
) -> dict[str, Any]:
    model = lateral.load_em_production_lateral(artifact)
    out: dict[str, Any] = {
        "artifact": str(artifact.relative_to(PROJECT_ROOT)),
        "sealed_data_read": False,
        "per_energy": {},
    }
    for energy in energies:
        holdout = reference_sample(extended_dir, energy).grid
        baseline = reference_sample(baseline_dir, energy).grid
        out["per_energy"][f"{energy:g}"] = {
            "model_occupied_cells": occupied_cells_per_event(model, energy, events, SEED + int(energy)),
            "geant4_holdout_occupied_cells": float((holdout > CELL_THRESHOLD_MEV).sum() / len(holdout)),
            "geant4_baseline_occupied_cells": float((baseline > CELL_THRESHOLD_MEV).sum() / len(baseline)),
        }
    return out


def build_candidate(tag: str, base_artifact: Path, quantum: dict[str, Any]) -> Path:
    """Write the artifact of one candidate: its base's parameters plus the measured quantum, nothing refitted."""

    parameters, document = lateral.read_parameters_artifact(base_artifact)
    candidate = replace(parameters, tail_quantum_mev=float(quantum["tail_quantum_mev"]))
    destination = CALIBRATION_ROOT / f"gp_lateral_{tag}_v1" / "parameters.json"
    record = {
        "status": "extension E2 on a calibrated candidate; development; not validated",
        "sealed_data_read": False,
        "base_artifact": str(base_artifact.relative_to(PROJECT_ROOT)),
        "base_parameters_sha256": document["parameters_sha256"],
        "method": "the tail component's spots carry the measured quantum; every other parameter is taken unchanged",
        "quantum": quantum,
    }
    return lateral.write_parameters_artifact(
        candidate,
        destination,
        record,
        "ams_ecal.electron_studies.em_lateral_tail_quantum",
        document["longitudinal_parameters_sha256"],
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--events", type=int, default=EVENTS)
    args = parser.parse_args(argv)
    quantum = measure_tail_quantum()
    for tag, base in CANDIDATES.items():
        artifact = build_candidate(tag, base, quantum)
        (RESULTS / f"em_production_lateral_{tag}_quantum.json").write_text(
            json.dumps(quantum, indent=2), encoding="utf-8"
        )
        record = occupancy_record(artifact, events=args.events)
        (RESULTS / f"em_production_lateral_{tag}_occupancy.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        print(tag, json.dumps(record["per_energy"], indent=1))
    print("tail quantum (MeV):", quantum["tail_quantum_mev"], quantum["per_energy_mev"])


if __name__ == "__main__":
    main()
