"""The added validation checks, as registered in
``research/plans/2026-10-02_added_validation_checks_preregistration.md`` (DEC-007).

Four rows, each computed for one CELL of the validation (class x energy x representation) from
two arrays of events, the model's and Geant4's:

* layer-correlation row (section 1): lag-averaged Spearman correlation of the 18 layer energies;
  material iff a listed lag differs by >= 0.10 AND the model value is outside the 95% bootstrap
  interval of the Geant4 value;
* classifier two-sample test (section 2): cross-validated AUC of gradient-boosted trees on the
  declared features; material iff AUC >= 0.60 AND above the 99th percentile of a label-permutation
  null;
* standardised sparsity (section 3): layer-averaged cell occupancy at 0.1, 0.5, 1.0 x the pilot MIP
  cell scale; material iff it differs by >= 0.02 AND the model value is outside the Wilson 95%
  interval of the Geant4 value;
* multiscale panel (section 4): D0, D1, D2 on the combined image, REPORT ONLY (no verdict).

Interpretations the registration leaves open, fixed here and reported with every result:

* the Wilson interval of the sparsity row uses ``n = number of Geant4 events`` (cells of one event
  are strongly correlated, so ``events x 72`` would be too narrow);
* both samples of the classifier row are cut to the same number of events (the smaller), taking the
  first events of each, which are already in seed order;
* an event with no energy gives NaN features; the classifier handles NaN natively.

Nothing here reads a sealed file. ``dry_run`` uses the pilot's CALIBRATION events as the stand-in
for Geant4 (registration, rule 3); the final run is given arrays by the caller, which obtains them
only through ``ams_ecal.sealed_set.open_set``.
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

from ams_ecal.multiscale import (
    AMS_BOX_SIDES,
    Q_VALUES,
    profile,
    reliable_windows,
    window_dimension,
)
from ams_ecal.proton_validation import CELL_THRESHOLD_MEV, crossing_observables

LAGS = (1, 2, 3, 5, 8, 12)
LAG_MATERIAL = 0.10
AUC_MATERIAL = 0.60
NULL_PERCENTILE = 99.0
OCCUPANCY_MATERIAL = 0.02
MIP_CELL_MEV = 0.5596007704734802
OCCUPANCY_FRACTIONS = (0.1, 0.5, 1.0)
N_BOOTSTRAP = 400
N_PERMUTATIONS = 200
N_FOLDS = 5
PANEL_EVENTS = 2000
WILSON_Z = 1.959963984540054

CLASSIFIER_SETTINGS = {
    "max_iter": 100,
    "learning_rate": 0.1,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "early_stopping": False,
    "random_state": 0,
}
"""Fixed hyper-parameters; nothing is tuned (registration section 2)."""


@dataclass(frozen=True, slots=True)
class EventSample:
    """Events of one cell: grids ``(n, layers, cells)`` in MeV and the entry coordinates."""

    grids: np.ndarray
    entry_x: np.ndarray
    entry_y: np.ndarray

    def __post_init__(self) -> None:
        if self.grids.ndim != 3:
            raise ValueError("grids must be (events, layers, cells)")
        if not (len(self.grids) == len(self.entry_x) == len(self.entry_y)):
            raise ValueError("grids and entry coordinates must have the same length")

    def __len__(self) -> int:
        return len(self.grids)

    def first(self, n: int) -> EventSample:
        return EventSample(self.grids[:n], self.entry_x[:n], self.entry_y[:n])


# ----------------------------------------------------------------------
# Section 1: layer correlations
# ----------------------------------------------------------------------


def lag_averaged_correlation(layer_energy: np.ndarray, lags: Sequence[int] = LAGS) -> np.ndarray:
    """Spearman correlation of the layer energies, averaged over layer pairs a given lag apart."""

    rho = stats.spearmanr(layer_energy).statistic
    n = rho.shape[0]
    return np.array([np.mean([rho[i, i + lag] for i in range(n - lag)]) for lag in lags])


def correlation_flags(
    difference: np.ndarray, model: np.ndarray, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    """Material lags: a difference of at least ``LAG_MATERIAL`` AND a model value outside the interval."""

    return (np.abs(difference) >= LAG_MATERIAL) & ((model < low) | (model > high))


def layer_correlation_row(
    model_layers: np.ndarray,
    reference_layers: np.ndarray,
    *,
    noise_layers: np.ndarray | None = None,
    lags: Sequence[int] = LAGS,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 0,
) -> dict[str, Any]:
    """The layer-correlation row for ``(n, 18)`` layer energies of the model and of Geant4."""

    model = lag_averaged_correlation(model_layers, lags)
    reference = lag_averaged_correlation(reference_layers, lags)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, len(lags)))
    for b in range(n_boot):
        boot[b] = lag_averaged_correlation(
            reference_layers[rng.integers(0, len(reference_layers), len(reference_layers))], lags
        )
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    difference = model - reference
    flagged = correlation_flags(difference, model, low, high)
    full_model = np.nan_to_num(stats.spearmanr(model_layers).statistic)
    full_reference = np.nan_to_num(stats.spearmanr(reference_layers).statistic)
    row: dict[str, Any] = {
        "lags": list(lags),
        "model": model.tolist(),
        "reference": reference.tolist(),
        "reference_interval": [low.tolist(), high.tolist()],
        "difference": difference.tolist(),
        "material_by_lag": [bool(f) for f in flagged],
        "material": bool(flagged.any()),
        "largest_pairwise_difference": float(np.abs(full_model - full_reference).max()),
    }
    if noise_layers is not None:
        floor = lag_averaged_correlation(noise_layers, lags) - reference
        row["noise_floor_difference"] = floor.tolist()
    return row


# ----------------------------------------------------------------------
# Section 2: classifier two-sample test
# ----------------------------------------------------------------------


def event_features(sample: EventSample, geometry) -> tuple[np.ndarray, list[str]]:
    """The declared features (registration section 2), one row per event."""

    observables = crossing_observables(sample.grids, sample.entry_x, sample.entry_y, geometry)
    layer = observables["layer_energy_mev"]
    occupied = (sample.grids > CELL_THRESHOLD_MEV).sum(axis=2)
    columns = {
        "energy_mev": observables["energy_mev"],
        "n_hit_cells": observables["n_hit_cells"],
        "max_cell_fraction": observables["max_cell_fraction"],
        "containment_fraction": observables["containment_fraction"],
        "long_cog_mm": observables["long_cog_mm"],
        "long_rms_mm": observables["long_rms_mm"],
        "max_layer": observables["max_layer"],
        "width_mm": observables["width_mm"],
        "core_fraction": observables["core_fraction"],
    }
    names = list(columns)
    matrix = [np.asarray(v, dtype=float) for v in columns.values()]
    for i in range(layer.shape[1]):
        names.append(f"log1p_layer_energy_{i}")
        matrix.append(np.log1p(layer[:, i]))
    for i in range(occupied.shape[1]):
        names.append(f"occupied_cells_{i}")
        matrix.append(occupied[:, i].astype(float))
    return np.column_stack(matrix), names


def classifier_material(auc: float, null_threshold: float) -> bool:
    """Material iff the AUC reaches ``AUC_MATERIAL`` AND exceeds the permutation-null percentile."""

    return bool(auc >= AUC_MATERIAL and auc > null_threshold)


def classifier_two_sample_row(
    model_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    n_permutations: int = N_PERMUTATIONS,
    n_jobs: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    """Cross-validated AUC of model versus Geant4 events, with a label-permutation null."""

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, permutation_test_score

    n = min(len(model_features), len(reference_features))
    features = np.vstack([model_features[:n], reference_features[:n]])
    labels = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    classifier = HistGradientBoostingClassifier(**CLASSIFIER_SETTINGS)
    score, null, _ = permutation_test_score(
        classifier,
        features,
        labels,
        scoring="roc_auc",
        cv=cv,
        n_permutations=n_permutations,
        n_jobs=n_jobs,
        random_state=seed,
    )
    threshold = float(np.percentile(null, NULL_PERCENTILE))
    return {
        "n_per_class": int(n),
        "auc": float(score),
        "null_percentile": NULL_PERCENTILE,
        "null_threshold": threshold,
        "null_mean": float(np.mean(null)),
        "n_permutations": int(n_permutations),
        "material": classifier_material(float(score), threshold),
    }


# ----------------------------------------------------------------------
# Section 3: sparsity
# ----------------------------------------------------------------------


def wilson_interval(successes: float, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval of a proportion ``successes / n``."""

    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return float(centre - half), float(centre + half)


def occupancy(grids: np.ndarray, threshold_mev: float) -> np.ndarray:
    """Per-layer occupancy: the fraction of cells above the threshold, ``(layers,)``."""

    return (np.asarray(grids) > threshold_mev).mean(axis=(0, 2))


def occupancy_material(difference: float, model_mean: float, low: float, high: float) -> bool:
    """Material iff the difference is at least ``OCCUPANCY_MATERIAL`` AND the model is outside the interval."""

    return bool(abs(difference) >= OCCUPANCY_MATERIAL and not low <= model_mean <= high)


def sparsity_row(
    model_grids: np.ndarray,
    reference_grids: np.ndarray,
    *,
    fractions: Sequence[float] = OCCUPANCY_FRACTIONS,
) -> dict[str, Any]:
    """Layer-averaged occupancy at each threshold, with the material verdict."""

    n_reference = len(reference_grids)
    rows = []
    for fraction in fractions:
        threshold = fraction * MIP_CELL_MEV
        model_curve = occupancy(model_grids, threshold)
        reference_curve = occupancy(reference_grids, threshold)
        model_mean, reference_mean = float(model_curve.mean()), float(reference_curve.mean())
        low, high = wilson_interval(reference_mean * n_reference, n_reference)
        rows.append(
            {
                "threshold_mev": threshold,
                "model": model_mean,
                "reference": reference_mean,
                "reference_interval": [low, high],
                "difference": model_mean - reference_mean,
                "material": occupancy_material(model_mean - reference_mean, model_mean, low, high),
                "per_layer_model": model_curve.tolist(),
                "per_layer_reference": reference_curve.tolist(),
            }
        )
    return {"thresholds": rows, "material": any(r["material"] for r in rows)}


# ----------------------------------------------------------------------
# Section 4: multiscale panel (report only)
# ----------------------------------------------------------------------


def multiscale_panel(
    sample_grids: np.ndarray, *, n_events: int = PANEL_EVENTS
) -> dict[str, dict[str, float]]:
    """``D_q`` per (q, window) over the windows that are reliable for each event's hit count."""

    values: dict[tuple[float, tuple[int, ...]], list[float]] = {}
    for grid in sample_grids[:n_events]:
        if grid.sum() <= 0:
            continue
        windows = reliable_windows(grid)
        if not windows:
            continue
        measure = profile(grid)
        for q in Q_VALUES:
            for window in windows:
                values.setdefault((q, window), []).append(
                    window_dimension(measure[q], AMS_BOX_SIDES, q, window)
                )
    return {
        f"q={q:g} sides={'-'.join(map(str, window))}": {
            "n": len(v),
            "median": float(np.median(v)),
            "q25": float(np.quantile(v, 0.25)),
            "q75": float(np.quantile(v, 0.75)),
        }
        for (q, window), v in sorted(values.items())
    }


def multiscale_comparison(model_grids: np.ndarray, reference_grids: np.ndarray) -> dict[str, Any]:
    """The two panels side by side; no verdict (DEC-010)."""

    return {
        "status": "EXPLORATORY, ungated: no threshold, no verdict (registration section 4)",
        "model": multiscale_panel(model_grids),
        "reference": multiscale_panel(reference_grids),
    }


# ----------------------------------------------------------------------
# One cell and the whole run
# ----------------------------------------------------------------------


def run_cell(
    model: EventSample,
    reference: EventSample,
    geometry,
    *,
    noise: EventSample | None = None,
    n_boot: int = N_BOOTSTRAP,
    n_permutations: int = N_PERMUTATIONS,
    n_jobs: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    """Every added row for one cell (class x energy x representation)."""

    model_obs = crossing_observables(model.grids, model.entry_x, model.entry_y, geometry)
    reference_obs = crossing_observables(
        reference.grids, reference.entry_x, reference.entry_y, geometry
    )
    noise_layers = None
    if noise is not None:
        noise_layers = crossing_observables(noise.grids, noise.entry_x, noise.entry_y, geometry)[
            "layer_energy_mev"
        ]
    model_features, names = event_features(model, geometry)
    reference_features, _ = event_features(reference, geometry)
    return {
        "n_model": len(model),
        "n_reference": len(reference),
        "layer_correlation": layer_correlation_row(
            model_obs["layer_energy_mev"],
            reference_obs["layer_energy_mev"],
            noise_layers=noise_layers,
            n_boot=n_boot,
            seed=seed,
        ),
        "classifier": {
            **classifier_two_sample_row(
                model_features,
                reference_features,
                n_permutations=n_permutations,
                n_jobs=n_jobs,
                seed=seed,
            ),
            "features": names,
        },
        "sparsity": sparsity_row(model.grids, reference.grids),
        "multiscale": multiscale_comparison(model.grids, reference.grids),
    }


def verdicts(cell: dict[str, Any]) -> dict[str, bool]:
    """The three gated rows of one cell; the multiscale panel has no verdict."""

    return {
        "layer_correlation": cell["layer_correlation"]["material"],
        "classifier": cell["classifier"]["material"],
        "sparsity": cell["sparsity"]["material"],
    }


def summarise(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Flag counts per row over all cells; read collectively, never from one flag (rule 5)."""

    flags: dict[str, list[str]] = {name: [] for name in ("layer_correlation", "classifier", "sparsity")}
    for key, cell in cells.items():
        for name, material in verdicts(cell).items():
            if material:
                flags[name].append(key)
    return {
        "n_cells": len(cells),
        "flagged": flags,
        "flag_counts": {name: len(keys) for name, keys in flags.items()},
    }


def dry_run(
    model,
    energies_gev: Sequence[float] = (10.0, 20.0, 50.0, 100.0),
    *,
    n_model_events: int = 2000,
    n_boot: int = 100,
    n_permutations: int = 20,
    n_jobs: int = 1,
    base_seed: int = 910_000,
) -> dict[str, Any]:
    """Whole harness on stand-ins: pilot CALIBRATION events as 'Geant4', FastMC as the model.

    Never reads held-out or sealed events. Its purpose is to catch bugs in the harness, not to
    tune the model, so the numbers of bootstrap resamples and permutations are reduced and the
    verdicts are not a validation.
    """

    from ams_ecal.proton_structure_check import generate_interacting_sample
    from ams_ecal.proton_validation import generate_crossing_sample, split_batch

    cells: dict[str, dict[str, Any]] = {}
    for energy in energies_gev:
        calibration, _, _ = split_batch("baseline", energy)
        interacts = calibration["truth_occurred"]
        for representation, key in (
            ("readout", "readout_grid_mev"),
            ("deposition", "deposit_grid_mev"),
        ):
            variant = model.as_representation(representation)
            for klass, mask, generate in (
                ("interacting", interacts, generate_interacting_sample),
                ("crossing", ~interacts, generate_crossing_sample),
            ):
                reference = EventSample(
                    calibration[key][mask],
                    calibration["entry_x_mm"][mask],
                    calibration["entry_y_mm"][mask],
                )
                made = generate(variant, energy, n_model_events, base_seed)
                sample = EventSample(made["grids"], made["entry_x"], made["entry_y"])
                half = len(reference) // 2
                cells[f"{klass} {energy:g} GeV {representation}"] = run_cell(
                    sample,
                    reference.first(half),
                    model.geometry,
                    noise=EventSample(
                        reference.grids[half : 2 * half],
                        reference.entry_x[half : 2 * half],
                        reference.entry_y[half : 2 * half],
                    ),
                    n_boot=n_boot,
                    n_permutations=n_permutations,
                    n_jobs=n_jobs,
                )
    return {
        "status": (
            "HARNESS DRY RUN on stand-ins (pilot calibration events versus FastMC). Not a "
            "validation; reduced bootstrap and permutation counts."
        ),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": model.model_version,
        "calibration_content_sha256": model.calibration.content_sha256,
        "summary": summarise(cells),
        "cells": cells,
    }


def main(argv: Sequence[str] | None = None) -> None:
    from ams_ecal.proton import ProtonShowerModel

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("results/added_checks/dry_run.json"))
    parser.add_argument("--events", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)
    result = dry_run(ProtonShowerModel.from_config(), n_model_events=args.events, n_jobs=args.jobs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.out)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
