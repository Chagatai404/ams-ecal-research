"""IN-SAMPLE fit-quality check of the crossing structure (burst, spill, bulk coupling).

THIS IS NOT A VALIDATION. The structure is calibrated on the CALIBRATION events
(``event_index % 4 != 3``) and this module compares generated events with those same events,
so it can only say whether the calibration fits what it was built from. It never reads the
held-out events (the 2026-09-29 crossing validation looked at those once) nor any sealed set:
the repaired crossing branch is validated once, on the sealed Geant4 set, under the contract
of ``research/plans/2026-09-29_proton_dependency_analysis.md`` section 8 and its added checks.

Compared per energy and representation, generated crossing events against the calibration
crossing events: the event total, hit cells, maximum-cell fraction and containment (as in the
validation), the layer-total variance ratio, the lag-averaged layer correlations, the burst
fraction. The KS distance between two finite samples is never zero; with ``n_model`` generated
and about 1550 calibration events a distance near 0.03-0.04 is the sampling floor.

    uv run python -m ams_ecal.proton_structure_check

writes ``results/proton_model/crossing_structure_in_sample.json``.
"""

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from ams_ecal.geant4_backend import PROJECT_ROOT, _git_state
from ams_ecal.proton import ProtonShowerModel
from ams_ecal.proton_calibration import DATA
from ams_ecal.proton_validation import (
    QUANTILES,
    compare_distribution,
    crossing_observables,
    generate_crossing_sample,
    split_batch,
)

RESULTS = PROJECT_ROOT / "results" / "proton_model"
ENERGIES_GEV = (10.0, 20.0, 50.0, 100.0)
LAGS = (1, 2, 5, 10)
N_MODEL_EVENTS = 4000
BASE_SEED = 20261002  # separate from the validation's 20260930


def lag_correlations(layer_energy: np.ndarray, lags: Sequence[int] = LAGS) -> list[float]:
    """Spearman correlation of layer energies averaged over layer pairs a given lag apart."""

    rho = stats.spearmanr(layer_energy).statistic
    n = rho.shape[0]
    return [float(np.mean([rho[i, i + lag] for i in range(n - lag)])) for lag in lags]


def variance_ratio(observables: dict[str, np.ndarray]) -> float:
    """Variance of the event total over the sum of the layer variances (1 for independent layers)."""

    return float(observables["energy_mev"].var() / observables["layer_energy_mev"].var(axis=0).sum())


def in_sample_section(
    model: ProtonShowerModel,
    representation: str,
    energy_gev: float,
    data: Path = DATA,
    n_events: int = N_MODEL_EVENTS,
    base_seed: int = BASE_SEED,
) -> dict[str, Any]:
    """Compare generated crossing events with the CALIBRATION crossing events of one energy."""

    calibration, _, _ = split_batch("baseline", energy_gev, data)  # the held-out half is discarded
    key = {"readout": "readout_grid_mev", "deposition": "deposit_grid_mev"}[representation]
    cal = ~calibration["truth_occurred"]
    generated = generate_crossing_sample(
        model.as_representation(representation), energy_gev, n_events, base_seed
    )
    geometry = model.geometry
    model_obs = crossing_observables(
        generated["grids"], generated["entry_x"], generated["entry_y"], geometry
    )
    cal_obs = crossing_observables(
        calibration[key][cal], calibration["entry_x_mm"][cal], calibration["entry_y_mm"][cal], geometry
    )
    out: dict[str, Any] = {
        "representation": representation,
        "n_model": n_events,
        "n_calibration": int(cal.sum()),
        "seconds_per_1000_events": 1000 * generated["seconds"] / n_events,
    }
    names = ["energy_mev"]
    if representation == "readout":
        names += ["n_hit_cells", "max_cell_fraction", "containment_fraction"]
    for name in names:
        out[name] = compare_distribution(model_obs[name], cal_obs[name])
    out["layer_energy_pooled"] = compare_distribution(
        model_obs["layer_energy_mev"].ravel(), cal_obs["layer_energy_mev"].ravel()
    )
    out["variance_ratio"] = {"model": variance_ratio(model_obs), "calibration": variance_ratio(cal_obs)}
    out["lag_correlation"] = {
        "lags": list(LAGS),
        "model": lag_correlations(model_obs["layer_energy_mev"]),
        "calibration": lag_correlations(cal_obs["layer_energy_mev"]),
    }
    out["mean_layer_energy_mev"] = {
        "model": model_obs["layer_energy_mev"].mean(axis=0).tolist(),
        "calibration": cal_obs["layer_energy_mev"].mean(axis=0).tolist(),
    }
    return out


def build_check(
    model: ProtonShowerModel,
    data: Path = DATA,
    energies: Sequence[float] = ENERGIES_GEV,
    n_events: int = N_MODEL_EVENTS,
) -> dict[str, Any]:
    if model.calibration.structure is None:
        raise ValueError("this calibration carries no crossing structure to check")
    git = _git_state()
    result: dict[str, Any] = {
        "status": (
            "IN-SAMPLE fit-quality check. Generated crossing events are compared with the "
            "CALIBRATION events the structure was built from. Not a validation."
        ),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": {"commit": git["commit"], "tracked_changes": git["tracked_changes"]},
        "model": {
            "model_version": model.model_version,
            "calibration_content_sha256": model.calibration.content_sha256,
            "physics_list": model.calibration.physics_list,
            "effective_length_mm": model.effective_length_mm,
        },
        "quantile_levels": list(QUANTILES),
        "reading": (
            "KS near 0.03-0.04 is the sampling floor for these sample sizes; the contract's "
            "material threshold (0.10 with p < 0.01) is shown for orientation only."
        ),
    }
    for energy in energies:
        result[f"{energy:g}"] = {
            representation: in_sample_section(model, representation, energy, data, n_events)
            for representation in ("readout", "deposition")
        }
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--events", type=int, default=N_MODEL_EVENTS)
    args = parser.parse_args(argv)
    model = ProtonShowerModel.from_config()
    args.out.mkdir(parents=True, exist_ok=True)
    result = build_check(model, args.data, ENERGIES_GEV, args.events)
    path = args.out / "crossing_structure_in_sample.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
