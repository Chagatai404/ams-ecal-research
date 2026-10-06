"""Development check of the ``em_production`` Slice 1 generator against Geant4 (NOT a validation).

Decision record ``research/DECISIONS.md`` (DEC-014, 2026-10-06 rows), note
``research/plans/2026-10-06_em_production_slice1_note.md``. Compares ensembles of the Slice 1 generator with

* the CALIBRATION set: the exposed baseline electrons (``data/geant4_electron_sample/baseline``), on which the
  parameters were fitted;
* the DEVELOPMENT HOLD-OUT: the prefix layers of the extended-depth electrons (``data/geant4_electron_extended``,
  other seeds), not used for the parameters. Earlier analyses looked at that sample descriptively, so it is a
  hold-out, NOT a fresh validation set; the sealed electron set stays unopened and is the final validation;

and with the DEC-001 generator (the control). The reference for "as good as it can get" is the SAMPLE-TO-SAMPLE
distance: the same metrics between the two Geant4 samples. The sealed sets are never read.

Metrics are those of ``em_longitudinal_structure_analysis`` (mean layer fractions, early layers, leakage mean and
spread, contained-fraction and width distributions, lag correlations) plus the moments of per-event gamma fits (the
estimator the fluctuation parameters were calibrated with) and the leakage distribution.
Words: calibrated, candidate, development; never validated.
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
from ams_ecal.electron_model.em_production import (
    DEFAULT_ARTIFACT,
    PROJECT_ROOT,
    load_em_production,
    read_parameters_artifact,
)
from ams_ecal.electron_studies.em_longitudinal_structure_analysis import (
    NOISE_FLOORS,
    _clean,
    dec001_ensemble,
    distances,
    fit_free_with_origin,
    longitudinal_metrics,
    removed_fraction,
)
from ams_ecal.geant4_simulation.pilot_analysis import layer_centres_mm
from ams_ecal.validation.dataset import GEOMETRY_CONFIG

ENERGIES_GEV = (10.0, 20.0, 50.0, 100.0)
BASELINE_DIR = PROJECT_ROOT / "data" / "geant4_electron_sample" / "baseline"
EXTENDED_DIR = PROJECT_ROOT / "data" / "geant4_electron_extended" / "extended"
RESULTS_JSON = PROJECT_ROOT / "results" / "em_generator" / "em_production_development_check.json"
PLOTS_DIR = PROJECT_ROOT / "results" / "em_generator" / "em_production_development_check"
ENSEMBLE_EVENTS = 4000
SEEDS = 5


def load_prefix_fractions(directory: Path, energy: float, key: str = "deposit_grid_mev") -> np.ndarray:
    """``(n, 18)`` prefix layer fractions of E (all-material deposit)."""

    arrays = np.load(directory / f"E{energy:g}GeV" / "events.npz")
    grid = arrays[key][:, :18] if key.startswith("extended") else arrays[key]
    return grid.sum(axis=2).astype(float) / (1000.0 * energy)


def per_event_moments(fractions: np.ndarray, bounds: np.ndarray, origin: float) -> dict[str, float]:
    fits = np.array([fit_free_with_origin(f, bounds, origin)[0] for f in fractions])
    ln_t, ln_alpha = fits[:, 0], fits[:, 1]
    return {
        "mean_ln_t": float(ln_t.mean()),
        "sd_ln_t": float(ln_t.std(ddof=1)),
        "skewness_ln_t": float(stats.skew(ln_t)),
        "mean_ln_alpha": float(ln_alpha.mean()),
        "sd_ln_alpha": float(ln_alpha.std(ddof=1)),
        "pearson": float(np.corrcoef(ln_t, ln_alpha)[0, 1]),
        "beta_median": float(np.median((np.exp(ln_alpha) - 1.0) / np.exp(ln_t))),
    }


def leakage_summary(fractions: np.ndarray) -> dict[str, float]:
    leak = 1.0 - fractions.sum(axis=1)
    return {
        "mean": float(leak.mean()),
        "sd": float(leak.std(ddof=1)),
        "skewness": float(stats.skew(leak)) if leak.std() > 0 else 0.0,
        "q05": float(np.quantile(leak, 0.05)),
        "q50": float(np.quantile(leak, 0.5)),
        "q95": float(np.quantile(leak, 0.95)),
        "q99": float(np.quantile(leak, 0.99)),
    }


def distance_table(ensemble: np.ndarray, reference: np.ndarray, z_mm: np.ndarray) -> dict[str, float]:
    return dict(distances(longitudinal_metrics(ensemble, reference, z_mm)))


def _mean_sd(tables: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": float(np.mean([t[key] for t in tables])),
            "sd": float(np.std([t[key] for t in tables], ddof=1)),
        }
        for key in tables[0]
    }


def development_check(
    energies: Sequence[float] = ENERGIES_GEV,
    *,
    artifact: Path = DEFAULT_ARTIFACT,
    baseline_dir: Path = BASELINE_DIR,
    extended_dir: Path = EXTENDED_DIR,
    ensemble_events: int = ENSEMBLE_EVENTS,
    seeds: int = SEEDS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parameters, document = read_parameters_artifact(artifact)
    model = load_em_production(artifact)
    geometry = load_geometry(GEOMETRY_CONFIG)
    bounds = np.array(geometry.uniform_layer_bounds_x0, dtype=float)
    z_mm = layer_centres_mm(geometry, 18)
    origin = parameters.origin_x0
    results: dict[str, Any] = {
        "parameters": parameters.as_dict(),
        "artifact_sha256": document["parameters_sha256"],
        "sealed_data_read": False,
        "per_energy": {},
    }
    artefacts: dict[str, Any] = {"ensembles": {}, "baseline": {}, "holdout": {}}
    for energy in energies:
        baseline = load_prefix_fractions(baseline_dir, energy)
        holdout = load_prefix_fractions(extended_dir, energy, "extended_deposit_grid_mev")
        dec001 = dec001_ensemble(energy, "deposition", ensemble_events, int(1000 * energy) + 3)
        runs = []
        for seed in range(seeds):
            rng = np.random.default_rng(31_000 * (seed + 1) + int(energy))
            runs.append(model.longitudinal.sample_layer_energy_fractions_batch(1000.0 * energy, rng, ensemble_events))
        ensemble = runs[0]
        entry: dict[str, Any] = {}
        for label, reference in (("calibration_set", baseline), ("development_holdout", holdout)):
            slice1 = _mean_sd([distance_table(r, reference, z_mm) for r in runs])
            control = distance_table(dec001, reference, z_mm)
            entry[label] = {
                "slice1": slice1,
                "dec001": control,
                "removed_vs_dec001": removed_fraction(control, {k: v["mean"] for k, v in slice1.items()}),
            }
        entry["sample_to_sample_floor"] = distance_table(holdout, baseline, z_mm)
        entry["moments_geant4_baseline"] = per_event_moments(baseline, bounds, origin)
        entry["moments_geant4_holdout"] = per_event_moments(holdout, bounds, origin)
        entry["moments_slice1"] = per_event_moments(ensemble[:2000], bounds, origin)
        entry["moments_dec001"] = per_event_moments(dec001[:2000], bounds, origin)
        entry["leakage"] = {
            "geant4_baseline": leakage_summary(baseline),
            "geant4_holdout": leakage_summary(holdout),
            "slice1": leakage_summary(ensemble),
            "dec001": leakage_summary(dec001),
        }
        layers = {
            "geant4_baseline": baseline.mean(axis=0),
            "geant4_holdout": holdout.mean(axis=0),
            "slice1": np.mean([r.mean(axis=0) for r in runs], axis=0),
            "dec001": dec001.mean(axis=0),
        }
        entry["mean_layer_fractions"] = {k: v.tolist() for k, v in layers.items()}
        for name, model_key, reference_key in (
            ("relative_layer_error_slice1_vs_baseline", "slice1", "geant4_baseline"),
            ("relative_layer_error_slice1_vs_holdout", "slice1", "geant4_holdout"),
            ("relative_layer_error_dec001_vs_baseline", "dec001", "geant4_baseline"),
        ):
            entry[name] = ((layers[model_key] - layers[reference_key]) / layers[reference_key]).tolist()
        results["per_energy"][f"{energy:g}"] = entry
        artefacts["ensembles"][energy] = {"slice1": ensemble, "dec001": dec001}
        artefacts["baseline"][energy] = baseline
        artefacts["holdout"][energy] = holdout
    results["noise_floors"] = dict(NOISE_FLOORS)
    return results, artefacts


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--out", type=Path, default=RESULTS_JSON)
    parser.add_argument("--plots", type=Path, default=PLOTS_DIR)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    results, artefacts = development_check(artifact=args.artifact)
    document = {
        "status": (
            "DEVELOPMENT CHECK of a candidate generator on EXPOSED Geant4 electrons (calibration set and a development "
            "hold-out). Not a validation; the sealed electron set was not read."
        ),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **_clean(results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2), encoding="utf-8")
    if not args.no_plots:
        from ams_ecal.electron_studies.em_production_development_check_plots import (
            make_plots,
        )

        make_plots(results, artefacts, args.plots)
    print(args.out)


if __name__ == "__main__":
    main()
