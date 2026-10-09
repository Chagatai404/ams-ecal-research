"""How well does the profile-error metric resolve a difference of a few tenths of a point? (analysis only)

The adoption rules of the lateral slice used a regression tolerance of 0.3 points of layer energy on the layers 3-17 mean
cell-profile error. That number was chosen by analogy (the project's 0.5-point materiality floor is registered for the mean
LEAKAGE, ``em_longitudinal_structure_analysis.NOISE_FLOORS``, and the profile tolerance was set below it); it was not derived
for this metric. The faster-tail candidate failed it by 0.11 points (+0.41). This module measures what the metric can
resolve, on the exposed extended-prefix hold-out only (the sealed sets are never read):

* the PAIRED bootstrap interval of the difference of two models' profile errors, resampling the Geant4 reference events and
  scoring both models against the same resampled reference (the reference noise, a positive bias of the total variation
  that equals the Geant4 sample-to-sample floor, then largely cancels in the difference);
* the seed-to-seed spread of one model's own ensemble, and the reference-only spread of the metric.

Command: ``python -m ams_ecal.electron_studies.em_lateral_metric_resolution``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ams_ecal.electron_model import em_production_lateral as lateral
from ams_ecal.electron_model.em_production import PROJECT_ROOT
from ams_ecal.electron_studies.em_lateral_structure_study import total_variation
from ams_ecal.electron_studies.em_production_lateral_check import (
    ENERGIES_GEV,
    EXTENDED_DIR,
    mean_profile,
    profile_errors_points,
    reference_sample,
    slice2_sample,
)

RESULTS_JSON = PROJECT_ROOT / "results" / "em_generator" / "em_production_lateral_metric_resolution.json"
FASTER_TAIL_ARTIFACT = (
    PROJECT_ROOT / "data" / "calibration" / "em_production" / "gp_lateral_faster_tail_v1" / "parameters.json"
)
BOOTSTRAPS = 300
ENSEMBLE_EVENTS = 4000
SEEDS = 5
SEED = 20261010


def layers_3_17_mean(model_grid: np.ndarray, reference_grid: np.ndarray) -> float:
    return float(profile_errors_points(model_grid, reference_grid)[3:].mean())


def paired_bootstrap_difference(
    grid_a: np.ndarray, grid_b: np.ndarray, reference: np.ndarray, n_boot: int, rng: np.random.Generator
) -> dict[str, float]:
    """Difference ``error(b) - error(a)`` of the layers 3-17 mean profile errors, resampling the reference events."""

    profile_a, profile_b = mean_profile(grid_a), mean_profile(grid_b)

    def error(model_profile: np.ndarray, reference_events: np.ndarray) -> float:
        reference_profile = mean_profile(reference_events)
        return float(np.mean([100 * total_variation(model_profile[k], reference_profile[k]) for k in range(3, 18)]))

    point = error(profile_b, reference) - error(profile_a, reference)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        resampled = reference[rng.integers(0, len(reference), len(reference))]
        draws[i] = error(profile_b, resampled) - error(profile_a, resampled)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "difference": point,
        "bootstrap_sd": float(draws.std(ddof=1)),
        "interval_low": float(low),
        "interval_high": float(high),
    }


def reference_only_spread(reference: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict[str, float]:
    """The metric between two independent halves of the reference: the floor and its spread."""

    values = []
    for _ in range(n_boot):
        order = rng.permutation(len(reference))
        half = len(reference) // 2
        values.append(layers_3_17_mean(reference[order[:half]], reference[order[half:]]))
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}


def study(
    energies: tuple[float, ...] = ENERGIES_GEV,
    *,
    base_artifact: Path = lateral.DEFAULT_ARTIFACT,
    candidate_artifact: Path = FASTER_TAIL_ARTIFACT,
    extended_dir: Path = EXTENDED_DIR,
    n_boot: int = BOOTSTRAPS,
    events: int = ENSEMBLE_EVENTS,
    seeds: int = SEEDS,
) -> dict[str, Any]:
    base = lateral.load_em_production_lateral(base_artifact)
    candidate = lateral.load_em_production_lateral(candidate_artifact)
    rng = np.random.default_rng(SEED)
    out: dict[str, Any] = {
        "status": "metric resolution on the exposed extended-prefix hold-out; the sealed sets are not read; no model changed",
        "sealed_data_read": False,
        "bootstraps": n_boot,
        "per_energy": {},
    }
    for energy in energies:
        reference = reference_sample(extended_dir, energy).grid
        models = (("base", base), ("candidate", candidate))
        grids = {name: slice2_sample(model, energy, events, SEED + int(energy)).grid for name, model in models}
        seed_errors = {
            name: [
                layers_3_17_mean(
                    slice2_sample(model, energy, events, SEED + 1000 * (s + 1) + int(energy)).grid, reference
                )
                for s in range(seeds)
            ]
            for name, model in models
        }
        out["per_energy"][f"{energy:g}"] = {
            "paired_difference_candidate_minus_base": paired_bootstrap_difference(
                grids["base"], grids["candidate"], reference, n_boot, rng
            ),
            "reference_events": len(reference),
            "reference_half_against_half": reference_only_spread(reference, n_boot, rng),
            "model_seed_spread": {
                k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for k, v in seed_errors.items()
            },
        }
    rows = list(out["per_energy"].values())
    differences = [r["paired_difference_candidate_minus_base"]["difference"] for r in rows]
    sds = [r["paired_difference_candidate_minus_base"]["bootstrap_sd"] for r in rows]
    out["mean_difference_over_energies"] = {
        "difference": float(np.mean(differences)),
        "approximate_sd": float(np.sqrt(np.sum(np.square(sds))) / len(sds)),
    }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS_JSON)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    args = parser.parse_args(argv)
    results = study(n_boot=args.bootstraps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for energy, row in results["per_energy"].items():
        d = row["paired_difference_candidate_minus_base"]
        floor = row["reference_half_against_half"]
        spread = row["model_seed_spread"]
        print(
            f"{energy} GeV: faster tail minus Slice 2 = {d['difference']:+.2f} points "
            f"[{d['interval_low']:+.2f}, {d['interval_high']:+.2f}] (sd {d['bootstrap_sd']:.2f}); "
            f"reference half-against-half {floor['mean']:.2f} +- {floor['sd']:.2f}; "
            f"model seed sd {spread['base']['sd']:.3f} / {spread['candidate']['sd']:.3f}"
        )
    m = results["mean_difference_over_energies"]
    print(f"mean over energies: {m['difference']:+.2f} +- {m['approximate_sd']:.2f}")


if __name__ == "__main__":
    main()
