"""The adoption rule for extension E1 of the lateral slice, stated BEFORE the extension was calibrated.

The researcher's condition (2026-10-08): "Do it only if it doesn't cause an overfitting and it doesn't complicate the
parameterization". It is made operational here, in code, before any E1 number existed, so the verdict cannot be shaped by
the result. E1 is ADOPTED only if every criterion holds; otherwise Slice 2 stays the candidate and E1 is recorded as a
negative result.

A. BENEFIT on the development hold-out (the extended-depth prefix, not used for the constants). The two observables the
   extension is meant to cure, the width and the hit-cell count, must both improve at every energy, and the mean relative
   reduction of their KS distances must be at least 30%.
B. NO OVERFITTING.
   B1. The improvement transfers: on the hold-out it is at least 70% of the improvement on the calibration set (for each
       of the two observables, averaged over the energies).
   B2. The spot-number error on the hold-out is at most 0.05 above the one on the calibration set (rms log ratio over all
       layers, averaged over the energies).
   B3. The extension did not collapse to a switched-off value (kappa near 1, a front cap above 10^4) and converged.
C. PARSIMONY AND NO REGRESSION.
   C1. At most two parameters are added to the spot structure.
   C2. On the hold-out, averaged over the energies, nothing that Slice 2 already did well gets worse: the cell-profile error
       (layers 3-17 mean) by no more than 0.2 points; the KS distances of the core fraction, containment and largest cell
       fraction by no more than 0.03; the absolute lateral-longitudinal coupling difference by no more than 0.05.

The hit-cell count and the width are not targets of the E1 calibration (it fits the fluctuation of the central cells and
of the outskirts), so they are genuine tests.

Command: ``python -m ams_ecal.electron_studies.em_production_lateral_e1_decision``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ams_ecal.electron_model.em_production import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results" / "em_generator"
BASELINE_JSON = RESULTS / "em_production_lateral_check.json"
EXTENSION_JSON = RESULTS / "em_production_lateral_e1_check.json"
CALIBRATION_JSON = RESULTS / "em_production_lateral_e1_calibration.json"
DECISION_JSON = RESULTS / "em_production_lateral_e1_decision.json"
TARGET_KEYS = ("width_mm", "n_hit_cells")
GUARD_KS_KEYS = ("core_fraction", "containment_fraction", "max_cell_fraction")
THRESHOLDS = {
    "mean_relative_reduction_minimum": 0.30,
    "transfer_fraction_minimum": 0.70,
    "spot_number_gap_maximum": 0.05,
    "added_parameters_maximum": 2,
    "profile_error_regression_points": 0.2,
    "ks_regression": 0.03,
    "coupling_regression": 0.05,
}
BASE_SPOT_PARAMETERS = 4


def _per_energy(results: dict[str, Any], label: str) -> list[dict[str, Any]]:
    return [row[label]["slice2"] for row in results["per_energy"].values()]


def _mean(items: list[Any], getter) -> float:
    return float(np.mean([getter(item) for item in items]))


def evaluate(baseline: dict[str, Any], extension: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-stated rule to the Slice 2 check, the E1 check and the E1 calibration record."""

    t = THRESHOLDS
    base_hold, ext_hold = _per_energy(baseline, "development_holdout"), _per_energy(extension, "development_holdout")
    base_cal, ext_cal = _per_energy(baseline, "calibration_set"), _per_energy(extension, "calibration_set")
    hold_pairs, cal_pairs = list(zip(base_hold, ext_hold, strict=True)), list(zip(base_cal, ext_cal, strict=True))

    # A: benefit on the hold-out
    every_energy = all(e["ks"][k] < b["ks"][k] for k in TARGET_KEYS for b, e in hold_pairs)
    reductions = [(b["ks"][k] - e["ks"][k]) / b["ks"][k] for k in TARGET_KEYS for b, e in hold_pairs]
    mean_reduction = float(np.mean(reductions))
    criteria: dict[str, Any] = {
        "A_improves_at_every_energy": bool(every_energy),
        "A_mean_relative_reduction": mean_reduction,
        "A_mean_relative_reduction_ok": bool(mean_reduction >= t["mean_relative_reduction_minimum"]),
    }

    # B1: the improvement transfers
    transfer = {}
    for k in TARGET_KEYS:
        hold = _mean(hold_pairs, lambda pair, k=k: pair[0]["ks"][k] - pair[1]["ks"][k])
        calib = _mean(cal_pairs, lambda pair, k=k: pair[0]["ks"][k] - pair[1]["ks"][k])
        transfer[k] = {
            "holdout_improvement": hold,
            "calibration_improvement": calib,
            "ok": bool(hold >= t["transfer_fraction_minimum"] * calib),
        }
    gap = _mean(ext_hold, lambda b: b["spot_number_rms_log_ratio"]) - _mean(
        ext_cal, lambda b: b["spot_number_rms_log_ratio"]
    )
    spot_fit = calibration["spot_fit"]
    criteria.update(
        {
            "B1_improvement_transfers": transfer,
            "B1_ok": all(v["ok"] for v in transfer.values()),
            "B2_spot_number_gap": gap,
            "B2_ok": bool(gap <= t["spot_number_gap_maximum"]),
            "B3_not_collapsed_and_converged": bool(not spot_fit["extension_collapsed"] and spot_fit["converged"]),
        }
    )

    # C: parsimony and no regression
    added = len(spot_fit["free_parameters"]) - BASE_SPOT_PARAMETERS
    regression = {
        "profile_error_layers_3_17_mean_points": _mean(
            ext_hold, lambda b: b["profile_error_points"]["layers_3_17_mean"]
        )
        - _mean(base_hold, lambda b: b["profile_error_points"]["layers_3_17_mean"]),
        "coupling_abs_difference": _mean(ext_hold, lambda b: abs(b["coupling_spearman"]["difference"]))
        - _mean(base_hold, lambda b: abs(b["coupling_spearman"]["difference"])),
    }
    for k in GUARD_KS_KEYS:
        regression[f"ks_{k}"] = _mean(ext_hold, lambda b, k=k: b["ks"][k]) - _mean(base_hold, lambda b, k=k: b["ks"][k])
    no_regression = (
        regression["profile_error_layers_3_17_mean_points"] <= t["profile_error_regression_points"]
        and regression["coupling_abs_difference"] <= t["coupling_regression"]
        and all(regression[f"ks_{k}"] <= t["ks_regression"] for k in GUARD_KS_KEYS)
    )
    criteria.update(
        {
            "C1_added_parameters": added,
            "C1_ok": bool(added <= t["added_parameters_maximum"]),
            "C2_changes_on_holdout": regression,
            "C2_ok": bool(no_regression),
        }
    )
    adopted = bool(
        criteria["A_improves_at_every_energy"]
        and criteria["A_mean_relative_reduction_ok"]
        and criteria["B1_ok"]
        and criteria["B2_ok"]
        and criteria["B3_not_collapsed_and_converged"]
        and criteria["C1_ok"]
        and criteria["C2_ok"]
    )
    return {
        "thresholds": t,
        "criteria": criteria,
        "adopted": adopted,
        "verdict": (
            "E1 ADOPTED: every pre-stated criterion holds"
            if adopted
            else "E1 NOT ADOPTED: Slice 2 stays the candidate; E1 is recorded as a result"
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, default=BASELINE_JSON)
    parser.add_argument("--extension", type=Path, default=EXTENSION_JSON)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_JSON)
    parser.add_argument("--out", type=Path, default=DECISION_JSON)
    args = parser.parse_args(argv)
    inputs = [json.loads(p.read_text(encoding="utf-8")) for p in (args.baseline, args.extension, args.calibration)]
    decision = evaluate(*inputs)
    args.out.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision["criteria"], indent=2))
    print(decision["verdict"])


if __name__ == "__main__":
    main()
