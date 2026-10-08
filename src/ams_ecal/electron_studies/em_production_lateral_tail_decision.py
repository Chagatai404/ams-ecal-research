"""The adoption rule for the tail-sensitive radial calibration, stated BEFORE it was run.

The researcher's condition (2026-10-08): run it "if it doesn't cause overfitting and doesn't complicate parameterization".
The width diagnostic found the width excess of Slice 2 in the far tail of the mean radial law, which the equal-weight
cumulative fit cannot see. The candidate keeps the SAME twelve radial constants and adds one target: the second moment of
the mean profile per (energy, layer), the quantity the width is built from. The rule is fixed here, in code, before any
number of the candidate existed; the candidate is ADOPTED only if every criterion holds, otherwise Slice 2 stays.

A. BENEFIT on the development hold-out (the extended-depth prefix): the width KS distance improves at every energy and
   its mean relative reduction is at least 30%.
B. NO OVERFITTING.
   B1. The width improvement transfers: on the hold-out it is at least 70% of the improvement on the calibration set.
   B2. The hold-out cell-profile error (layers 3-17 mean, averaged over the energies) is at most 0.2 points above the
       calibration-set one.
   B3. No radial constant ends at a bound.
C. NO COMPLICATION AND NO REGRESSION.
   C1. No parameter is added: the number of free radial constants equals that of Slice 2 (twelve).
   C2. On the hold-out, averaged over the energies, nothing that Slice 2 did well gets materially worse: the cell-profile
       error (layers 3-17 mean) by no more than 0.3 points; the KS distances of the core fraction, containment, largest
       cell fraction and hit-cell count by no more than 0.03, 0.03, 0.03 and 0.05; the absolute lateral-longitudinal
       coupling difference by no more than 0.05; the spot-number rms log ratio (all layers) by no more than 0.1.

FASTER-TAIL CANDIDATE (2026-10-08, the researcher's "try it" after the two tail-sensitive variants failed C2): the same
rule, with ONE change that the researcher approved: C1 allows exactly one added parameter (``--max-added-parameters 1``):
the exponent of the tail component's survival function. Everything else, including the no-overfitting criteria B and the
no-regression criterion C2, is unchanged and was fixed before the candidate was run.

Command: ``python -m ams_ecal.electron_studies.em_production_lateral_tail_decision``
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
CANDIDATE_JSON = RESULTS / "em_production_lateral_tail_check.json"
CALIBRATION_JSON = RESULTS / "em_production_lateral_tail_calibration.json"
DECISION_JSON = RESULTS / "em_production_lateral_tail_decision.json"
RADIAL_PARAMETERS = 12
KS_TOLERANCES = {"core_fraction": 0.03, "containment_fraction": 0.03, "max_cell_fraction": 0.03, "n_hit_cells": 0.05}
THRESHOLDS = {
    "mean_relative_reduction_minimum": 0.30,
    "transfer_fraction_minimum": 0.70,
    "profile_gap_maximum_points": 0.2,
    "profile_regression_points": 0.3,
    "coupling_regression": 0.05,
    "spot_number_regression": 0.1,
    "ks_regression": KS_TOLERANCES,
}


def _blocks(results: dict[str, Any], label: str) -> list[dict[str, Any]]:
    return [row[label]["slice2"] for row in results["per_energy"].values()]


def _mean(items: list[Any], getter) -> float:
    return float(np.mean([getter(item) for item in items]))


def evaluate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    calibration: dict[str, Any],
    max_added_parameters: int = 0,
) -> dict[str, Any]:
    """Apply the pre-stated rule to the Slice 2 check, the candidate check and the candidate calibration record."""

    t = {**THRESHOLDS, "added_parameters_maximum": max_added_parameters}
    base_hold, cand_hold = _blocks(baseline, "development_holdout"), _blocks(candidate, "development_holdout")
    base_cal, cand_cal = _blocks(baseline, "calibration_set"), _blocks(candidate, "calibration_set")
    hold_pairs, cal_pairs = list(zip(base_hold, cand_hold, strict=True)), list(zip(base_cal, cand_cal, strict=True))

    every_energy = all(c["ks"]["width_mm"] < b["ks"]["width_mm"] for b, c in hold_pairs)
    reductions = [(b["ks"]["width_mm"] - c["ks"]["width_mm"]) / b["ks"]["width_mm"] for b, c in hold_pairs]
    hold_gain = _mean(hold_pairs, lambda p: p[0]["ks"]["width_mm"] - p[1]["ks"]["width_mm"])
    cal_gain = _mean(cal_pairs, lambda p: p[0]["ks"]["width_mm"] - p[1]["ks"]["width_mm"])
    profile_gap = _mean(cand_hold, lambda b: b["profile_error_points"]["layers_3_17_mean"]) - _mean(
        cand_cal, lambda b: b["profile_error_points"]["layers_3_17_mean"]
    )
    radial = calibration["radial_fit"]
    criteria: dict[str, Any] = {
        "A_improves_at_every_energy": bool(every_energy),
        "A_mean_relative_reduction": float(np.mean(reductions)),
        "A_ok": bool(every_energy and np.mean(reductions) >= t["mean_relative_reduction_minimum"]),
        "B1_width_improvement": {"holdout": hold_gain, "calibration": cal_gain},
        "B1_ok": bool(hold_gain >= t["transfer_fraction_minimum"] * cal_gain),
        "B2_profile_error_gap_points": profile_gap,
        "B2_ok": bool(profile_gap <= t["profile_gap_maximum_points"]),
        "B3_parameters_at_bound": radial["parameters_at_bound"],
        "B3_ok": not radial["parameters_at_bound"],
    }
    added = len(radial["free_parameters"]) - RADIAL_PARAMETERS
    changes = {
        "profile_error_layers_3_17_mean_points": _mean(
            cand_hold, lambda b: b["profile_error_points"]["layers_3_17_mean"]
        )
        - _mean(base_hold, lambda b: b["profile_error_points"]["layers_3_17_mean"]),
        "coupling_abs_difference": _mean(cand_hold, lambda b: abs(b["coupling_spearman"]["difference"]))
        - _mean(base_hold, lambda b: abs(b["coupling_spearman"]["difference"])),
        "spot_number_rms_log_ratio": _mean(cand_hold, lambda b: b["spot_number_rms_log_ratio"])
        - _mean(base_hold, lambda b: b["spot_number_rms_log_ratio"]),
    }
    for key in KS_TOLERANCES:
        changes[f"ks_{key}"] = _mean(cand_hold, lambda b, k=key: b["ks"][k]) - _mean(
            base_hold, lambda b, k=key: b["ks"][k]
        )
    no_regression = (
        changes["profile_error_layers_3_17_mean_points"] <= t["profile_regression_points"]
        and changes["coupling_abs_difference"] <= t["coupling_regression"]
        and changes["spot_number_rms_log_ratio"] <= t["spot_number_regression"]
        and all(changes[f"ks_{k}"] <= tol for k, tol in KS_TOLERANCES.items())
    )
    criteria.update(
        {
            "C1_added_parameters": added,
            "C1_ok": added <= max_added_parameters,
            "C2_changes_on_holdout": changes,
            "C2_ok": bool(no_regression),
        }
    )
    adopted = bool(all(criteria[k] for k in ("A_ok", "B1_ok", "B2_ok", "B3_ok", "C1_ok", "C2_ok")))
    return {
        "thresholds": t,
        "criteria": criteria,
        "adopted": adopted,
        "verdict": (
            "TAIL-SENSITIVE CALIBRATION ADOPTED: every pre-stated criterion holds"
            if adopted
            else "TAIL-SENSITIVE CALIBRATION NOT ADOPTED: Slice 2 stays the candidate; the result is recorded"
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, default=BASELINE_JSON)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE_JSON)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_JSON)
    parser.add_argument("--out", type=Path, default=DECISION_JSON)
    parser.add_argument("--max-added-parameters", type=int, default=0)
    args = parser.parse_args(argv)
    inputs = [json.loads(p.read_text(encoding="utf-8")) for p in (args.baseline, args.candidate, args.calibration)]
    decision = evaluate(*inputs, max_added_parameters=args.max_added_parameters)
    args.out.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision["criteria"], indent=2))
    print(decision["verdict"])


if __name__ == "__main__":
    main()
