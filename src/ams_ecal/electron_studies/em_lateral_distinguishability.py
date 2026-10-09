"""Can a classifier tell the lateral candidates from Geant4 electrons? (diagnostic only, never a tuning target)

The project's registered classifier two-sample test (``ams_ecal.validation.added_checks``: fixed gradient-boosting settings,
cross-validated AUC, label-permutation null) is run on EXPOSED data only, the extended-depth prefix as the Geant4 reference,
for Slice 2 and for the faster-tail candidate, with the same classifier on two independent Geant4 samples as the FLOOR (an
AUC near 0.5). The AUC answers "is the model distinguishable"; it is NOT optimised: the researcher's rule is that the
generator is never trained against a classifier (it would learn to fool that one classifier). The feature-group ablations
(without the hit-cell features, without the width, without both, and each alone) only DIAGNOSE which observables carry the
separation, so that the physics of those observables can be improved on its own terms.

Command: ``python -m ams_ecal.electron_studies.em_lateral_distinguishability``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ams_ecal.detector.geometry import ECALGeometry, load_geometry
from ams_ecal.electron_model import em_production_lateral as lateral
from ams_ecal.electron_model.em_production import PROJECT_ROOT
from ams_ecal.electron_studies.em_production_lateral_check import (
    BASELINE_DIR,
    ENERGIES_GEV,
    EXTENDED_DIR,
    Sample,
    reference_sample,
    slice2_sample,
)
from ams_ecal.validation.added_checks import (
    CLASSIFIER_SETTINGS,
    N_FOLDS,
    EventSample,
    classifier_two_sample_row,
    event_features,
)
from ams_ecal.validation.dataset import GEOMETRY_CONFIG

RESULTS_JSON = PROJECT_ROOT / "results" / "em_generator" / "em_production_lateral_distinguishability.json"
FASTER_TAIL_ARTIFACT = (
    PROJECT_ROOT / "data" / "calibration" / "em_production" / "gp_lateral_faster_tail_v1" / "parameters.json"
)
EVENTS = 1000
PERMUTATIONS = 50
SEED = 20261011
GROUPS = {
    "all_features": lambda name: True,
    "without_hit_cells": lambda name: name != "n_hit_cells" and not name.startswith("occupied_cells_"),
    "without_width": lambda name: name != "width_mm",
    "without_hit_cells_and_width": lambda name: (
        name != "n_hit_cells" and not name.startswith("occupied_cells_") and name != "width_mm"
    ),
    "only_hit_cell_count": lambda name: name == "n_hit_cells",
    "only_width": lambda name: name == "width_mm",
}


def features(sample: Sample, geometry: ECALGeometry) -> tuple[np.ndarray, list[str]]:
    return event_features(EventSample(sample.grid, sample.entry_x_mm, sample.entry_y_mm), geometry)


def cross_validated_auc(model_features: np.ndarray, reference_features: np.ndarray, seed: int = 0) -> float:
    """Cross-validated AUC of model against reference with the registered fixed classifier (no tuning)."""

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    n = min(len(model_features), len(reference_features))
    x = np.vstack([model_features[:n], reference_features[:n]])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    return float(
        cross_val_score(HistGradientBoostingClassifier(**CLASSIFIER_SETTINGS), x, y, scoring="roc_auc", cv=cv).mean()
    )


def single_feature_separation(
    model_f: np.ndarray, reference_f: np.ndarray, names: list[str], top: int = 10
) -> list[dict[str, float | str]]:
    """The features that separate model from reference on their own, by the folded rank AUC ``max(a, 1 - a)``.

    A univariate diagnostic (no classifier): 0.5 is no separation, 1 is a complete one. It names the observables whose
    physics has to be improved on its own terms; it is not an objective.
    """

    from sklearn.metrics import roc_auc_score

    n = min(len(model_f), len(reference_f))
    labels = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    rows = []
    for i, name in enumerate(names):
        values = np.concatenate([model_f[:n, i], reference_f[:n, i]])
        if np.ptp(values) == 0.0:
            continue
        auc = float(roc_auc_score(labels, values))
        direction = "model higher" if auc < 0.5 else "model lower"
        rows.append({"feature": name, "folded_auc": max(auc, 1.0 - auc), "direction": direction})
    return sorted(rows, key=lambda r: -float(r["folded_auc"]))[:top]


def single_feature_report(
    energies: tuple[float, ...] = ENERGIES_GEV,
    *,
    base_artifact: Path = lateral.DEFAULT_ARTIFACT,
    candidate_artifact: Path = FASTER_TAIL_ARTIFACT,
    baseline_dir: Path = BASELINE_DIR,
    extended_dir: Path = EXTENDED_DIR,
    events: int = EVENTS,
) -> dict[str, Any]:
    """Per energy, the top single separating features of each model and of the Geant4 baseline against the hold-out."""

    geometry = load_geometry(GEOMETRY_CONFIG)
    models = {
        "slice2": lateral.load_em_production_lateral(base_artifact),
        "faster_tail": lateral.load_em_production_lateral(candidate_artifact),
    }
    out: dict[str, Any] = {
        "status": "univariate diagnostic on exposed electrons; not an objective; the sealed sets are not read",
        "sealed_data_read": False,
        "events_per_class": events,
        "per_energy": {},
    }
    for energy in energies:
        holdout, names = features(reference_sample(extended_dir, energy), geometry)
        baseline_f, _ = features(reference_sample(baseline_dir, energy), geometry)
        row: dict[str, Any] = {"geant4_baseline_vs_holdout": single_feature_separation(baseline_f, holdout, names, 6)}
        for name, model in models.items():
            model_f, _ = features(slice2_sample(model, energy, events, SEED + int(energy)), geometry)
            row[name] = single_feature_separation(model_f, holdout[:events], names)
        out["per_energy"][f"{energy:g}"] = row
    return out


def ablations(model_f: np.ndarray, reference_f: np.ndarray, names: list[str]) -> dict[str, float]:
    out = {}
    for label, keep in GROUPS.items():
        columns = [i for i, n in enumerate(names) if keep(n)]
        out[label] = cross_validated_auc(model_f[:, columns], reference_f[:, columns])
    return out


def distinguishability(
    energies: tuple[float, ...] = ENERGIES_GEV,
    *,
    base_artifact: Path = lateral.DEFAULT_ARTIFACT,
    candidate_artifact: Path = FASTER_TAIL_ARTIFACT,
    baseline_dir: Path = BASELINE_DIR,
    extended_dir: Path = EXTENDED_DIR,
    events: int = EVENTS,
    permutations: int = PERMUTATIONS,
) -> dict[str, Any]:
    geometry = load_geometry(GEOMETRY_CONFIG)
    models = {
        "slice2": lateral.load_em_production_lateral(base_artifact),
        "faster_tail": lateral.load_em_production_lateral(candidate_artifact),
    }
    out: dict[str, Any] = {
        "status": "diagnostic on exposed electrons; the classifier is never a tuning target; the sealed sets are not read",
        "sealed_data_read": False,
        "events_per_class": events,
        "per_energy": {},
    }
    for energy in energies:
        holdout, _ = features(reference_sample(extended_dir, energy), geometry)
        baseline_f, names = features(reference_sample(baseline_dir, energy), geometry)
        row: dict[str, Any] = {
            "geant4_floor": classifier_two_sample_row(
                baseline_f[:events], holdout[:events], n_permutations=permutations, seed=0
            )
        }
        for name, model in models.items():
            model_f, _ = features(slice2_sample(model, energy, events, SEED + int(energy)), geometry)
            row[name] = {
                "classifier": classifier_two_sample_row(model_f, holdout[:events], n_permutations=permutations, seed=0),
                "feature_group_auc": ablations(model_f, holdout[:events], names),
            }
        out["per_energy"][f"{energy:g}"] = row
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS_JSON)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--single-features", action="store_true", help="the univariate separation report only")
    args = parser.parse_args(argv)
    if args.single_features:
        report = single_feature_report()
        target = args.out.with_name("em_production_lateral_single_features.json")
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        for energy, row in report["per_energy"].items():
            for name in ("geant4_baseline_vs_holdout", "slice2", "faster_tail"):
                top = ", ".join(f"{r['feature']} {r['folded_auc']:.2f}" for r in row[name][:5])
                print(f"{energy} GeV {name}: {top}")
        return
    results = distinguishability(permutations=args.permutations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for energy, row in results["per_energy"].items():
        floor = row["geant4_floor"]
        line = f"{energy} GeV: Geant4 floor AUC {floor['auc']:.3f} (null 99% {floor['null_threshold']:.3f})"
        for name in ("slice2", "faster_tail"):
            c = row[name]["classifier"]
            g = row[name]["feature_group_auc"]
            line += (
                f" | {name}: AUC {c['auc']:.3f} (material {c['material']}); without hit cells "
                f"{g['without_hit_cells']:.3f}, without width {g['without_width']:.3f}, "
                f"without both {g['without_hit_cells_and_width']:.3f}"
            )
        print(line)


if __name__ == "__main__":
    main()
