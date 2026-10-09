"""Development run of the registered multiscale statistics on EXPOSED events (electrons against truth-interacting protons).

NOT the registered test: it uses the exposed baseline electrons and the calibration protons (``event_index % 4 != 3``) to debug
the pipeline and to see the eligible fractions and the size of the differences before G-train exists. The sealed sets are not
read. Protons are truth-interacting events only (the crossing half, with about 20 cells, is never eligible).

Command: ``python -m ams_ecal.multiscale_estimators.multiscale_development``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ams_ecal.electron_model.em_production import PROJECT_ROOT
from ams_ecal.multiscale_estimators import multiscale_test as mt
from ams_ecal.multiscale_estimators.eligibility_count import ENERGIES_GEV, SAMPLES

RESULTS_JSON = PROJECT_ROOT / "results" / "multiscale_controls" / "multiscale_development.json"
SEED = 20261010


def load(particle: str, energy: float, representation: str) -> np.ndarray:
    arrays = np.load(SAMPLES[particle] / f"E{energy:g}GeV" / "events.npz")
    keep = np.ones(len(arrays["event_index"]), dtype=bool)
    if particle == "proton":
        keep = (arrays["event_index"] % 4 != 3) & arrays["truth_occurred"]
    return arrays[representation][keep].astype(float)


def run(bootstraps: int, energies: tuple[float, ...] = ENERGIES_GEV) -> dict:
    rng = np.random.default_rng(SEED)
    out: dict = {
        "status": "DEVELOPMENT on exposed events; not the registered test; sealed sets not read",
        "sealed_data_read": False,
        "bootstraps": bootstraps,
        "per_energy": {},
    }
    names = mt.feature_names()
    for energy in energies:
        row: dict = {}
        for representation in ("deposit_grid_mev", "readout_grid_mev"):
            grids = {p: load(p, energy, representation) for p in ("electron", "proton")}
            tables = {p: mt.multiscale_table(g) for p, g in grids.items()}
            keys = {
                "occupied_cells": {p: (g > 0).reshape(len(g), -1).sum(axis=1).astype(float) for p, g in grids.items()},
                "total_energy": {p: g.sum(axis=(1, 2)) for p, g in grids.items()},
            }
            block: dict = {"events": {p: len(g) for p, g in grids.items()}, "features": {}}
            for j, name in enumerate(names):
                block["features"][name] = {
                    key: mt.compare_feature(
                        tables["electron"][:, j],
                        tables["proton"][:, j],
                        values["electron"],
                        values["proton"],
                        rng,
                        bootstraps,
                    )
                    for key, values in keys.items()
                }
            row[representation] = block
        out["per_energy"][f"{energy:g}"] = row
        print(f"{energy:g} GeV done", flush=True)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS_JSON)
    parser.add_argument("--bootstraps", type=int, default=60)
    args = parser.parse_args(argv)
    results = run(args.bootstraps)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
