"""Development check of the ``em_production`` lateral slice (Slice 2) against Geant4 (NOT a validation).

Decision record ``research/DECISIONS.md`` (2026-10-07 rows), note ``research/plans/2026-10-07_em_lateral_structure_note.md``.
Compares ensembles of Slice 2 (Slice 1's longitudinal draws with the Grindhammer-Peters radial law and spot
fluctuations) with

* the CALIBRATION set: the exposed baseline electrons, on which the lateral constants were fitted;
* the DEVELOPMENT HOLD-OUT: the prefix layers of the extended-depth electrons (other seeds), not used for the constants.
  Earlier analyses looked at that sample descriptively, so it is a hold-out and NOT a fresh validation set;

and with Slice 1 with DEC-001's deterministic lateral law (the control). The reference for "as good as it can get" is
the SAMPLE-TO-SAMPLE distance: the same metrics between the two Geant4 samples. The sealed sets are never read.

Metrics: the lateral observables of the earlier exploration comparison (width, core fraction, containment, hit cells,
largest cell fraction; two-sample Kolmogorov-Smirnov distance), the mean cell-profile error per layer (total variation,
points of layer energy), the lateral-longitudinal coupling (Spearman of the lateral width with the energy centre of
gravity), and the conditioned spot number per layer (the calibration target, here also a hold-out test).

Proposed acceptance signs (the researcher sets the thresholds; they are REPORTED, not gated): mean cell-profile error
at most 2 points in layers 3-17; coupling within twice the floor or 0.05; conditioned spot number within a factor 1.28
(rms log ratio 0.25). Words: calibrated, candidate, development; never validated.

Command: ``python -m ams_ecal.electron_studies.em_production_lateral_check``
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from ams_ecal.detector.geometry import ECALGeometry, load_geometry
from ams_ecal.detector.tracking import TrackState
from ams_ecal.electron_model import em_production_lateral as lateral
from ams_ecal.electron_model.em_production import PROJECT_ROOT, load_em_production
from ams_ecal.electron_studies.em_lateral_structure_study import (
    AXIS_CELL,
    spot_number,
    total_variation,
)
from ams_ecal.proton_model.proton_validation import crossing_observables
from ams_ecal.validation.dataset import GEOMETRY_CONFIG

ENERGIES_GEV = (10.0, 20.0, 50.0, 100.0)
BASELINE_DIR = PROJECT_ROOT / "data" / "geant4_electron_sample" / "baseline"
EXTENDED_DIR = PROJECT_ROOT / "data" / "geant4_electron_extended" / "extended"
RESULTS_JSON = PROJECT_ROOT / "results" / "em_generator" / "em_production_lateral_check.json"
PLOTS_DIR = PROJECT_ROOT / "results" / "em_generator" / "em_production_lateral_check"
ENTRY_SPOT_MM = (0.0, 9.0)
ENSEMBLE_EVENTS = 4000
CONTROL_EVENTS = 1000
SEEDS = 3
CONTROL_SEED = 20261008
LATERAL_OBSERVABLES = (
    "width_mm",
    "core_fraction",
    "containment_fraction",
    "n_hit_cells",
    "max_cell_fraction",
)
CONTEXT_OBSERVABLES = ("long_cog_mm", "long_rms_mm", "max_layer")
CENTRAL = slice(AXIS_CELL - 1, AXIS_CELL + 2)
PROPOSED_SIGNS = {
    "profile_error_layers_3_17_max_points": 2.0,
    "coupling_difference_factor_of_floor": 2.0,
    "coupling_difference_minimum": 0.05,
    "spot_number_rms_log_ratio": 0.25,
}


@dataclass(frozen=True, slots=True)
class Sample:
    """Cell grids ``(n, layers, cells)`` in MeV with the entry point of every event."""

    grid: np.ndarray
    entry_x_mm: np.ndarray
    entry_y_mm: np.ndarray


def reference_sample(directory: Path, energy_gev: float) -> Sample:
    arrays = np.load(directory / f"E{energy_gev:g}GeV" / "events.npz")
    return Sample(
        arrays["deposit_grid_mev"].astype(float),
        arrays["entry_x_mm"],
        arrays["entry_y_mm"],
    )


def slice2_sample(model: lateral.EMProductionLateralModel, energy_gev: float, events: int, seed: int) -> Sample:
    rng = np.random.default_rng(seed)
    entry = rng.uniform(*ENTRY_SPOT_MM, size=(events, 2))
    axis = model.axis_for_entry(entry[:, 0], entry[:, 1])
    batch = model.sample_batch(1000.0 * energy_gev, axis, rng)
    return Sample(batch.cell_energy_mev, entry[:, 0], entry[:, 1])


def slice1_sample(energy_gev: float, events: int, seed: int) -> Sample:
    """The control: Slice 1's longitudinal model with DEC-001's deterministic lateral grid."""

    model = load_em_production()
    rng = np.random.default_rng(seed)
    entry = rng.uniform(*ENTRY_SPOT_MM, size=(events, 2))
    grids = np.empty((events, model.geometry.number_of_layers, model.geometry.cells_per_layer))
    for i in range(events):
        track = TrackState(
            x0_mm=float(entry[i, 0]),
            y0_mm=float(entry[i, 1]),
            z0_mm=0.0,
            theta_rad=0.0,
            phi_rad=0.0,
        )
        event = model.generate_event(
            event_id=f"control-{i}",
            primary_energy_mev=1000.0 * energy_gev,
            track=track,
            random_seed=seed + i,
            configuration_sha256="0" * 64,
        )
        grids[i] = np.array(event.cell_energies_mev)
    return Sample(grids, entry[:, 0], entry[:, 1])


def observables(sample: Sample, geometry: ECALGeometry) -> dict[str, np.ndarray]:
    return crossing_observables(sample.grid, sample.entry_x_mm, sample.entry_y_mm, geometry)


def mean_profile(grid: np.ndarray) -> np.ndarray:
    layer = grid.sum(axis=2)
    return grid.sum(axis=0) / layer.sum(axis=0)[:, None]


def profile_errors_points(model_grid: np.ndarray, reference_grid: np.ndarray) -> np.ndarray:
    """Total variation of the mean cell profile per layer, in points of layer energy."""

    model, reference = mean_profile(model_grid), mean_profile(reference_grid)
    return np.array([100 * total_variation(model[layer], reference[layer]) for layer in range(len(model))])


def conditioned_spot_numbers(grid: np.ndarray) -> np.ndarray:
    layer_energy = grid.sum(axis=2)
    layers = grid.shape[1]
    cog = (layer_energy * (np.arange(layers) + 0.5)).sum(axis=1) / layer_energy.sum(axis=1)
    out = []
    for layer in range(layers):
        usable = layer_energy[:, layer] > 0
        fraction = grid[usable, layer, CENTRAL].sum(axis=1) / layer_energy[usable, layer]
        out.append(spot_number(fraction, cog[usable])["spot_number_conditioned"])
    return np.array(out)


def coupling(obs: dict[str, np.ndarray]) -> float:
    return float(stats.spearmanr(obs["width_mm"], obs["long_cog_mm"]).statistic)


def comparison(model: Sample, reference: Sample, geometry: ECALGeometry) -> dict[str, Any]:
    model_obs, reference_obs = (
        observables(model, geometry),
        observables(reference, geometry),
    )
    ks = {
        name: float(stats.ks_2samp(model_obs[name], reference_obs[name], method="asymp").statistic)
        for name in LATERAL_OBSERVABLES + CONTEXT_OBSERVABLES
    }
    tv = profile_errors_points(model.grid, reference.grid)
    spots_model, spots_reference = (
        conditioned_spot_numbers(model.grid),
        conditioned_spot_numbers(reference.grid),
    )
    return {
        "ks": ks,
        "profile_error_points": {
            "layers_0_2_max": float(tv[:3].max()),
            "layers_3_17_mean": float(tv[3:].mean()),
            "layers_3_17_max": float(tv[3:].max()),
            "per_layer": tv.tolist(),
        },
        "coupling_spearman": {
            "model": coupling(model_obs),
            "geant4": coupling(reference_obs),
            "difference": coupling(model_obs) - coupling(reference_obs),
        },
        "spot_number_rms_log_ratio": float(np.sqrt(np.mean((np.log(spots_model) - np.log(spots_reference)) ** 2))),
        "spot_number_conditioned": {
            "model": spots_model.tolist(),
            "geant4": spots_reference.tolist(),
        },
    }


def _mean_over_seeds(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    keys = list(blocks[0]["ks"])
    mean_ks = {k: float(np.mean([b["ks"][k] for b in blocks])) for k in keys}
    profile = {
        k: float(np.mean([b["profile_error_points"][k] for b in blocks]))
        for k in ("layers_0_2_max", "layers_3_17_mean", "layers_3_17_max")
    }
    return {
        "ks": mean_ks,
        "profile_error_points": profile,
        "profile_error_per_layer": np.mean([b["profile_error_points"]["per_layer"] for b in blocks], axis=0).tolist(),
        "coupling_spearman": {
            k: float(np.mean([b["coupling_spearman"][k] for b in blocks])) for k in ("model", "geant4", "difference")
        },
        "spot_number_rms_log_ratio": float(np.mean([b["spot_number_rms_log_ratio"] for b in blocks])),
        "spot_number_conditioned": {
            k: np.mean([b["spot_number_conditioned"][k] for b in blocks], axis=0).tolist() for k in ("model", "geant4")
        },
        "seeds": len(blocks),
    }


def signs(block: dict[str, Any], floor: dict[str, Any]) -> dict[str, bool]:
    limit = max(
        PROPOSED_SIGNS["coupling_difference_factor_of_floor"] * abs(floor["coupling_spearman"]["difference"]),
        PROPOSED_SIGNS["coupling_difference_minimum"],
    )
    return {
        "profile_error_layers_3_17_max": block["profile_error_points"]["layers_3_17_max"]
        <= PROPOSED_SIGNS["profile_error_layers_3_17_max_points"],
        "coupling_within_floor": abs(block["coupling_spearman"]["difference"]) <= limit,
        "spot_number_within_factor": block["spot_number_rms_log_ratio"] <= PROPOSED_SIGNS["spot_number_rms_log_ratio"],
    }


def development_check(
    energies: Sequence[float] = ENERGIES_GEV,
    *,
    artifact: Path = lateral.DEFAULT_ARTIFACT,
    baseline_dir: Path = BASELINE_DIR,
    extended_dir: Path = EXTENDED_DIR,
    ensemble_events: int = ENSEMBLE_EVENTS,
    control_events: int = CONTROL_EVENTS,
    seeds: int = SEEDS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parameters, document = lateral.read_parameters_artifact(artifact)
    model = lateral.load_em_production_lateral(artifact)
    geometry = load_geometry(GEOMETRY_CONFIG)
    results: dict[str, Any] = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "development check of a calibrated candidate; not a validation; the sealed sets are not read",
        "parameters": parameters.as_dict(),
        "artifact_sha256": document["parameters_sha256"],
        "longitudinal_parameters_sha256": document["longitudinal_parameters_sha256"],
        "sealed_data_read": False,
        "proposed_signs": PROPOSED_SIGNS,
        "per_energy": {},
    }
    artefacts: dict[str, Any] = {}
    for energy in energies:
        baseline = reference_sample(baseline_dir, energy)
        holdout = reference_sample(extended_dir, energy)
        floor = comparison(baseline, holdout, geometry)
        control = slice1_sample(energy, control_events, CONTROL_SEED)
        row: dict[str, Any] = {
            "sample_to_sample_floor": {
                k: floor[k]
                for k in (
                    "ks",
                    "profile_error_points",
                    "coupling_spearman",
                    "spot_number_rms_log_ratio",
                )
            },
        }
        for label, reference in (
            ("calibration_set", baseline),
            ("development_holdout", holdout),
        ):
            blocks = [
                comparison(
                    slice2_sample(model, energy, ensemble_events, 70_000 + 17 * s + int(energy)),
                    reference,
                    geometry,
                )
                for s in range(seeds)
            ]
            slice2 = _mean_over_seeds(blocks)
            control_block = comparison(control, reference, geometry)
            row[label] = {
                "slice2": slice2,
                "slice1_with_dec001_lateral": {
                    "ks": control_block["ks"],
                    "profile_error_points": {
                        k: control_block["profile_error_points"][k]
                        for k in (
                            "layers_0_2_max",
                            "layers_3_17_mean",
                            "layers_3_17_max",
                        )
                    },
                    "coupling_spearman": control_block["coupling_spearman"],
                    "spot_number_rms_log_ratio": control_block["spot_number_rms_log_ratio"],
                },
                "proposed_signs_met": signs(slice2, floor),
            }
        results["per_energy"][f"{energy:g}"] = row
        artefacts[f"{energy:g}"] = row
    return results, artefacts


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS_JSON)
    parser.add_argument("--plots", type=Path, default=PLOTS_DIR)
    parser.add_argument("--events", type=int, default=ENSEMBLE_EVENTS)
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--artifact", type=Path, default=lateral.DEFAULT_ARTIFACT)
    args = parser.parse_args(argv)
    results, artefacts = development_check(artifact=args.artifact, ensemble_events=args.events, seeds=args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    from ams_ecal.electron_studies.em_production_lateral_check_plots import make_plots

    make_plots(results, artefacts, args.plots)
    for energy, row in results["per_energy"].items():
        for label in ("calibration_set", "development_holdout"):
            block = row[label]["slice2"]
            print(
                f"{energy} GeV {label}: profile 3-17 max {block['profile_error_points']['layers_3_17_max']:.2f}, "
                f"coupling diff {block['coupling_spearman']['difference']:+.3f}, "
                f"spots rms log {block['spot_number_rms_log_ratio']:.2f}, signs {row[label]['proposed_signs_met']}"
            )


if __name__ == "__main__":
    main()
