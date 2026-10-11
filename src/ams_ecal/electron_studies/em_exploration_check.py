"""The EM generator against the exploration Geant4 electrons: the contract rows, never run before.

``data/geant4_electron_sample`` (4 energies x 1000 electrons, base seed 20270000) is the
EXPLORATION set; the sealed electron set (``data/geant4_electron_sealed``) is not touched here.
The EM generator was parameterised from the literature and never fitted to this sample, so the
numbers are an honest out-of-sample comparison, but the sample is now an exploration set and any
later model change made after looking at it is a change after a look.

Only the ``deposition`` representation is compared. The EM generator emits true deposition
(DEC-006); the fibre-energy scale of the ``readout`` representation is the detector response
step R-A that does not exist yet, so a ``readout`` comparison would measure that missing step
(model 9.9 GeV against Geant4 0.6 GeV at 10 GeV), not the generator.

For each energy: a KS distance and the 5/50/95% quantiles of the contract observables, the mean
layer-energy ratio model / Geant4, and the added-checks harness rows
(``ams_ecal.validation.added_checks.run_cell``; reduced bootstrap and permutation counts, exploration only).

    uv run python -m ams_ecal.electron_studies.em_exploration_check
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

from ams_ecal.detector.geometry import load_geometry
from ams_ecal.geant4_simulation.geant4_backend import PROJECT_ROOT, load_batch
from ams_ecal.proton_model.proton_validation import crossing_observables
from ams_ecal.validation.added_checks import EventSample, run_cell
from ams_ecal.validation.dataset import (
    GEOMETRY_CONFIG,
    electron_grid,
    load_electron_model,
)

DATA = PROJECT_ROOT / "data" / "geant4_electron_sample" / "baseline"
RESULTS = PROJECT_ROOT / "results" / "em_generator" / "exploration_comparison.json"
ENERGIES_GEV = (10.0, 20.0, 50.0, 100.0)
MODEL_SEED_BASE = 7_000_000
OBSERVABLES = (
    "energy_mev",
    "long_cog_mm",
    "long_rms_mm",
    "max_layer",
    "width_mm",
    "core_fraction",
    "containment_fraction",
    "n_hit_cells",
    "max_cell_fraction",
)
QUANTILES = (0.05, 0.5, 0.95)


def compare_energy(energy_gev: float, *, n_boot: int, n_permutations: int) -> dict[str, Any]:
    geometry = load_geometry(GEOMETRY_CONFIG)
    batch = load_batch(DATA / f"E{energy_gev:g}GeV").arrays
    x, y = batch["entry_x_mm"], batch["entry_y_mm"]
    reference = batch["deposit_grid_mev"].astype(float)
    model = load_electron_model("deposition")
    grids = np.stack(
        [
            electron_grid(model, 1000.0 * energy_gev, float(x[i]), float(y[i]), MODEL_SEED_BASE + i)
            for i in range(len(x))
        ]
    )
    model_obs = crossing_observables(grids, x, y, geometry)
    reference_obs = crossing_observables(reference, x, y, geometry)
    rows = {
        name: {
            "ks": float(stats.ks_2samp(model_obs[name], reference_obs[name]).statistic),
            "model_quantiles": np.quantile(model_obs[name], QUANTILES).tolist(),
            "geant4_quantiles": np.quantile(reference_obs[name], QUANTILES).tolist(),
        }
        for name in OBSERVABLES
    }
    ratio = model_obs["layer_energy_mev"].mean(axis=0) / reference_obs["layer_energy_mev"].mean(axis=0)
    cell = run_cell(
        EventSample(grids, x, y),
        EventSample(reference, x, y),
        geometry,
        n_boot=n_boot,
        n_permutations=n_permutations,
    )
    return {
        "n_events": len(x),
        "observables": rows,
        "mean_layer_energy_ratio_model_over_geant4": ratio.tolist(),
        "added_checks": {
            "layer_correlation_material": cell["layer_correlation"]["material"],
            "layer_correlation_difference": cell["layer_correlation"]["difference"],
            "classifier_auc": cell["classifier"]["auc"],
            "classifier_null_threshold": cell["classifier"]["null_threshold"],
            "sparsity": [
                {k: row[k] for k in ("threshold_mev", "model", "reference", "difference", "material")}
                for row in cell["sparsity"]["thresholds"]
            ],
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--boot", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=20)
    args = parser.parse_args(argv)
    result: dict[str, Any] = {
        "status": (
            "EXPLORATION comparison of the EM generator (deposition) with Geant4 electrons that were "
            "not used to fit it. Not a validation: the sealed electron set is untouched."
        ),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "quantile_levels": list(QUANTILES),
    }
    for energy in ENERGIES_GEV:
        result[f"{energy:g}"] = compare_energy(
            energy, n_boot=args.boot, n_permutations=args.permutations
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
