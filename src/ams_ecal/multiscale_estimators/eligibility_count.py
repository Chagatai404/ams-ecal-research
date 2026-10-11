"""Eligibility count: how many cells do real Geant4 events occupy, against what the estimator controls needed (phase 1 of the plan).

The estimator-validity controls (EXP-008) ran on equal-quanta images and found the estimators USABLE only from about 1000 hits
(2 of 13 windows) or 3000 hits (4 of 14). A real event is not a quanta image: what the saturation rule uses is the number of
occupied cells at the finest scale (``multiscale.n_eff``). This module counts, on EXPOSED development events only:

* the occupied cells of the deposition and the readout grid (any energy, and above the half-MIP cell threshold), per class,
  energy and, for protons, truth-interacting versus crossing;
* the reliable scale windows the saturation rule leaves each event (a necessary condition, not sufficient: the controls found it
  too lenient);
* the occupied cells of the control images at 100, 300, 1000 and 3000 hits, which is the conversion from "hits" to cells;
* the number of fibres and fine voxels the event has at the finer stored scales.

No multiscale value (no D_q) is computed. Held-out proton events (``event_index % 4 == 3``) are excluded. Sealed sets are not read.

Command: ``python -m ams_ecal.multiscale_estimators.eligibility_count``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ams_ecal.electron_model.em_production import PROJECT_ROOT
from ams_ecal.multiscale_estimators import multiscale
from ams_ecal.multiscale_estimators.multiscale_controls import (
    HIT_COUNTS,
    sample_hits,
    smooth_map,
)
from ams_ecal.proton_model.proton_validation import CELL_THRESHOLD_MEV

RESULTS_JSON = PROJECT_ROOT / "results" / "multiscale_controls" / "eligibility_count.json"
SAMPLES = {
    "electron": PROJECT_ROOT / "data" / "geant4_electron_sample" / "baseline",
    "proton": PROJECT_ROOT / "data" / "geant4_proton_pilot" / "baseline",
}
ENERGIES_GEV = (10.0, 20.0, 50.0, 100.0)
CONTROL_REPLICATES = 200
CONTROL_SEED = 20261009
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def occupied_cells(grids: np.ndarray, threshold_mev: float = 0.0) -> np.ndarray:
    """Occupied cells per event of ``(events, layers, cells)`` grids: strictly above ``threshold_mev``."""

    return (grids > threshold_mev).reshape(len(grids), -1).sum(axis=1)


def reliable_window_counts(counts: np.ndarray, shape: tuple[int, int] = (18, 72)) -> np.ndarray:
    """Number of scale windows the saturation rule leaves an event with ``n`` occupied cells (0 if ``n`` < 1)."""

    windows = multiscale.contiguous_windows()
    return np.array([sum(multiscale.is_reliable(shape, w, int(n)) for w in windows) if n >= 1 else 0 for n in counts])


def distribution(values: np.ndarray) -> dict[str, float]:
    out = {f"q{int(100 * q):02d}": float(np.quantile(values, q)) for q in QUANTILES}
    out["mean"] = float(np.mean(values))
    return out


def summarise(counts_any: np.ndarray, counts_threshold: np.ndarray) -> dict[str, Any]:
    windows = reliable_window_counts(counts_any)
    return {
        "events": len(counts_any),
        "occupied_cells_any": distribution(counts_any),
        "occupied_cells_above_threshold": distribution(counts_threshold),
        "events_with_at_least_one_reliable_window": float(np.mean(windows >= 1)),
        "reliable_windows_per_event": distribution(windows.astype(float)),
        "fraction_with_cells_at_least": {str(n): float(np.mean(counts_any >= n)) for n in (100, 300, 600, 1000)},
    }


def control_conversion(replicates: int = CONTROL_REPLICATES, seed: int = CONTROL_SEED) -> dict[str, Any]:
    """Occupied cells of the control images at each hit count: the conversion from hits to cells."""

    rng = np.random.default_rng(seed)
    prob = smooth_map()
    out = {}
    for hits in (100, *HIT_COUNTS):
        cells = np.array([multiscale.n_eff(sample_hits(prob, hits, rng)) for _ in range(replicates)])
        out[str(hits)] = distribution(cells.astype(float))
    return out


def fine_scale_counts(arrays: Any, keep: np.ndarray) -> dict[str, Any]:
    """Fibres and fine voxels per kept event (the stored finer scales)."""

    fibres = np.diff(arrays["fibre_offsets"])[keep]
    fine = np.diff(arrays["fine_offsets"])[keep]
    return {
        "fibres_per_event": distribution(fibres.astype(float)),
        "fine_voxels_per_event": distribution(fine.astype(float)),
    }


def study(samples: dict[str, Path] = SAMPLES, energies: tuple[float, ...] = ENERGIES_GEV) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "eligibility count on EXPOSED development events; no multiscale value computed; sealed sets not read",
        "sealed_data_read": False,
        "cell_threshold_mev": CELL_THRESHOLD_MEV,
        "control_images_occupied_cells_by_hits": control_conversion(),
        "per_class": {},
    }
    for particle, directory in samples.items():
        rows: dict[str, Any] = {}
        for energy in energies:
            arrays = np.load(directory / f"E{energy:g}GeV" / "events.npz")
            keep = np.ones(len(arrays["event_index"]), dtype=bool)
            if particle == "proton":
                keep = arrays["event_index"] % 4 != 3
            row: dict[str, Any] = {"fine_scales": fine_scale_counts(arrays, keep)}
            for representation in ("deposit_grid_mev", "readout_grid_mev"):
                grids = arrays[representation][keep].astype(float)
                any_c, thr_c = occupied_cells(grids), occupied_cells(grids, CELL_THRESHOLD_MEV)
                block = {"all": summarise(any_c, thr_c)}
                if particle == "proton":
                    interacted = arrays["truth_occurred"][keep]
                    for label, mask in (("interacting", interacted), ("crossing", ~interacted)):
                        if mask.sum():
                            block[label] = summarise(any_c[mask], thr_c[mask])
                row[representation] = block
            rows[f"{energy:g}"] = row
        out["per_class"][particle] = rows
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS_JSON)
    args = parser.parse_args(argv)
    results = study()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    controls = results["control_images_occupied_cells_by_hits"]
    print("control images, median occupied cells by hits:", {h: v["q50"] for h, v in controls.items()})
    for particle, rows in results["per_class"].items():
        for energy, row in rows.items():
            d = row["deposit_grid_mev"]["all"]
            print(
                f"{particle} {energy} GeV deposition: cells median {d['occupied_cells_any']['q50']:.0f} "
                f"(5-95%: {d['occupied_cells_any']['q05']:.0f}-{d['occupied_cells_any']['q95']:.0f}), "
                f"events with a reliable window {d['events_with_at_least_one_reliable_window']:.2f}"
            )


if __name__ == "__main__":
    main()
