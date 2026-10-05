"""The proton FastMC against Geant4 events of other hadronic physics lists (a SYSTEMATIC, not a validation).

The artifact is calibrated on FTFP_BERT calibration events. The pilot also holds QGSP_BERT
(``high_energy_model``) and QBBC (``alternate``) samples, 1000 events per energy, never used for
calibration. For each energy, class (crossing or interacting by the Geant4 truth flag) and
representation this reports

* ``model_vs_alternative``: KS distance of FastMC events against the alternative-list events;
* ``ftfp_vs_alternative``: KS distance of the FTFP_BERT CALIBRATION events (the model's own
  source) against the same alternative-list events: the physics-list effect itself, which a model
  tuned to FTFP_BERT cannot be expected to beat.

This is still Geant4, so it measures sensitivity to the hadronic model and is NOT independent
validation. It is also not the electron systematic: these lists share the standard EM physics.
Nothing here reads held-out or sealed events.

    uv run python -m ams_ecal.physics_list_systematic
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from ams_ecal.geant4_backend import PROJECT_ROOT, load_batch
from ams_ecal.proton import ProtonShowerModel
from ams_ecal.proton_structure_check import generate_interacting_sample
from ams_ecal.proton_validation import (
    ENERGIES_GEV,
    crossing_observables,
    generate_crossing_sample,
    split_batch,
)

DATA = PROJECT_ROOT / "data" / "geant4_proton_pilot"
RESULTS = PROJECT_ROOT / "results" / "proton_model" / "physics_list_systematic.json"
SAMPLES = {"QGSP_BERT": "high_energy_model", "QBBC": "alternate"}
OBSERVABLES = (
    "energy_mev",
    "n_hit_cells",
    "max_cell_fraction",
    "containment_fraction",
    "core_fraction",
    "width_mm",
)
GRID_KEY = {"readout": "readout_grid_mev", "deposition": "deposit_grid_mev"}
N_MODEL_EVENTS = 2000
BASE_SEED = 930_000


def _observables(arrays: dict[str, np.ndarray], mask: np.ndarray, key: str, geometry) -> dict:
    return crossing_observables(
        arrays[key][mask].astype(float),
        arrays["entry_x_mm"][mask],
        arrays["entry_y_mm"][mask],
        geometry,
    )


def _ks(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: float(stats.ks_2samp(a[name], b[name]).statistic) for name in OBSERVABLES}


def compare(
    model: ProtonShowerModel, list_name: str, energy_gev: float, n_model_events: int
) -> dict[str, Any]:
    alternative = load_batch(DATA / SAMPLES[list_name] / f"E{energy_gev:g}GeV").arrays
    ftfp, _, _ = split_batch("baseline", energy_gev)  # calibration half only; held-out is discarded
    out: dict[str, Any] = {}
    for klass, generate, want in (
        ("crossing", generate_crossing_sample, False),
        ("interacting", generate_interacting_sample, True),
    ):
        alt_mask = alternative["truth_occurred"] == want
        ftfp_mask = ftfp["truth_occurred"] == want
        for representation, key in GRID_KEY.items():
            made = generate(
                model.as_representation(representation), energy_gev, n_model_events, BASE_SEED
            )
            model_obs = crossing_observables(
                made["grids"], made["entry_x"], made["entry_y"], model.geometry
            )
            alt_obs = _observables(alternative, alt_mask, key, model.geometry)
            ftfp_obs = _observables(ftfp, ftfp_mask, key, model.geometry)
            out[f"{klass} {representation}"] = {
                "n_alternative": int(alt_mask.sum()),
                "model_vs_alternative": _ks(model_obs, alt_obs),
                "ftfp_vs_alternative": _ks(ftfp_obs, alt_obs),
                "median_energy_mev": {
                    "model": float(np.median(model_obs["energy_mev"])),
                    "ftfp": float(np.median(ftfp_obs["energy_mev"])),
                    "alternative": float(np.median(alt_obs["energy_mev"])),
                },
            }
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--events", type=int, default=N_MODEL_EVENTS)
    args = parser.parse_args(argv)
    model = ProtonShowerModel.from_config()
    result: dict[str, Any] = {
        "status": (
            "SYSTEMATIC against other Geant4 hadronic lists, not independent validation. The model is "
            "calibrated on FTFP_BERT calibration events; ftfp_vs_alternative is the physics-list "
            "effect itself."
        ),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_calibration_content_sha256": model.calibration.content_sha256,
    }
    for list_name in SAMPLES:
        result[list_name] = {
            f"{energy:g}": compare(model, list_name, energy, args.events) for energy in ENERGIES_GEV
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
