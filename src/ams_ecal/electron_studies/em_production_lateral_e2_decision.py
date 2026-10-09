"""The adoption rule for extension E2 (a fixed absolute energy quantum for the radial tail's spots), stated BEFORE it was run.

The researcher's decision (2026-10-09): build E2 "if [it satisfies] our goals" and keep following the Research OS procedure.
The goals, as the researcher stated them across the lateral work: no overfitting, no complication of the parameterization, and
a generator that is harder to tell from Geant4. The rule below was written before any E2 number existed and is applied to two
bases, Slice 2 and the faster-tail candidate, each against its OWN base. It is never relaxed after a result.

THE MECHANISM. The tail component's spots carry a fixed energy ``eps_t`` instead of an equal share of the layer energy. ``eps_t``
is MEASURED, not fitted: the pooled energy per occupied cell at least 108 mm from the track in the exposed baseline electrons
(``em_lateral_tail_quantum``). It is evaluated on the extended-depth hold-out and on nothing it was derived from.

A. BENEFIT on the hold-out.
   A1. The occupied cells per event are within 10% of the Geant4 hold-out at EVERY energy (two-sided: an overshoot fails).
   A2. The mean hit-cell KS distance is at most half that of the base.
B. NO OVERFITTING.
   B1. The hit-cell KS improvement transfers: the hold-out improvement is at least 70% of the calibration-set improvement.
   B2. The hold-out cell-profile error (layers 3-17 mean, averaged over the energies) is at most 0.2 points above the
       calibration-set one.
   B3. ``eps_t`` lies inside the range of its own per-energy measurements (it is measured, so it cannot sit at a fit bound).
C. NO COMPLICATION AND NO REGRESSION, against the base.
   C1. At most ONE added parameter (``eps_t``).
   C2. On the hold-out, averaged over the energies: the cell-profile error (layers 3-17 mean) by no more than 0.3 points;
       the KS distances of the core fraction, containment, largest cell fraction and width by no more than 0.03 each; the
       absolute lateral-longitudinal coupling difference by no more than 0.05; the spot-number rms log ratio by no more
       than 0.1.
   C3. At EVERY energy the cell-profile error (layers 3-17 mean) is no more than 0.5 points above the base (the project's
       registered 0.5-point materiality floor for the mean leakage, ``em_longitudinal_structure_analysis.NOISE_FLOORS``). This
       is the per-energy tolerance the metric-resolution study (note section 11) called for; it is new, and fixed here.

The classifier AUC and the single-feature separation are REPORTED outcomes, never criteria: the generator is not trained
against a discriminator.

Command: ``python -m ams_ecal.electron_studies.em_production_lateral_e2_decision --tag e2 --base-check <check.json>``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ams_ecal.electron_model.em_production import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results" / "em_generator"
KS_TOLERANCES = {"core_fraction": 0.03, "containment_fraction": 0.03, "max_cell_fraction": 0.03, "width_mm": 0.03}
THRESHOLDS = {
    "occupancy_relative_tolerance": 0.10,
    "hit_cell_ks_remaining_fraction_maximum": 0.50,
    "transfer_fraction_minimum": 0.70,
    "profile_gap_maximum_points": 0.2,
    "profile_regression_points": 0.3,
    "profile_regression_per_energy_points": 0.5,
    "coupling_regression": 0.05,
    "spot_number_regression": 0.1,
    "added_parameters_maximum": 1,
    "ks_regression": KS_TOLERANCES,
}


def _blocks(results: dict[str, Any], label: str) -> list[dict[str, Any]]:
    return [row[label]["slice2"] for row in results["per_energy"].values()]


def _mean(items: list[Any], getter) -> float:
    return float(np.mean([getter(item) for item in items]))


def _profile(block: dict[str, Any]) -> float:
    return block["profile_error_points"]["layers_3_17_mean"]


def evaluate(
    base: dict[str, Any],
    candidate: dict[str, Any],
    occupancy: dict[str, Any],
    quantum: dict[str, Any],
    added_parameters: int = 1,
) -> dict[str, Any]:
    """Apply the pre-stated rule to the base check, the candidate check, the occupancy record and the quantum record."""

    t = THRESHOLDS
    base_hold, cand_hold = _blocks(base, "development_holdout"), _blocks(candidate, "development_holdout")
    base_cal, cand_cal = _blocks(base, "calibration_set"), _blocks(candidate, "calibration_set")

    ratios = {
        e: row["model_occupied_cells"] / row["geant4_holdout_occupied_cells"]
        for e, row in occupancy["per_energy"].items()
    }
    hit_base = _mean(base_hold, lambda b: b["ks"]["n_hit_cells"])
    hit_cand = _mean(cand_hold, lambda b: b["ks"]["n_hit_cells"])
    cal_gain = _mean(base_cal, lambda b: b["ks"]["n_hit_cells"]) - _mean(cand_cal, lambda b: b["ks"]["n_hit_cells"])
    hold_gain = hit_base - hit_cand
    gap = _mean(cand_hold, _profile) - _mean(cand_cal, _profile)
    low, high = min(quantum["per_energy_mev"].values()), max(quantum["per_energy_mev"].values())

    per_energy_profile = {
        e: _profile(c) - _profile(b) for e, b, c in zip(base["per_energy"], base_hold, cand_hold, strict=True)
    }
    changes = {
        "profile_error_layers_3_17_mean_points": _mean(cand_hold, _profile) - _mean(base_hold, _profile),
        "coupling_abs_difference": _mean(cand_hold, lambda b: abs(b["coupling_spearman"]["difference"]))
        - _mean(base_hold, lambda b: abs(b["coupling_spearman"]["difference"])),
        "spot_number_rms_log_ratio": _mean(cand_hold, lambda b: b["spot_number_rms_log_ratio"])
        - _mean(base_hold, lambda b: b["spot_number_rms_log_ratio"]),
    }
    for key in KS_TOLERANCES:
        changes[f"ks_{key}"] = _mean(cand_hold, lambda b, k=key: b["ks"][k]) - _mean(
            base_hold, lambda b, k=key: b["ks"][k]
        )
    criteria: dict[str, Any] = {
        "A1_occupied_cell_ratio_to_holdout": ratios,
        "A1_ok": bool(all(abs(r - 1.0) <= t["occupancy_relative_tolerance"] for r in ratios.values())),
        "A2_hit_cell_ks": {"base": hit_base, "candidate": hit_cand},
        "A2_ok": bool(hit_cand <= t["hit_cell_ks_remaining_fraction_maximum"] * hit_base),
        "B1_hit_cell_ks_improvement": {"holdout": hold_gain, "calibration": cal_gain},
        "B1_ok": bool(cal_gain > 0 and hold_gain >= t["transfer_fraction_minimum"] * cal_gain),
        "B2_profile_error_gap_points": gap,
        "B2_ok": bool(gap <= t["profile_gap_maximum_points"]),
        "B3_quantum_mev": quantum["tail_quantum_mev"],
        "B3_measured_range_mev": [low, high],
        "B3_ok": bool(low <= quantum["tail_quantum_mev"] <= high),
        "C1_added_parameters": added_parameters,
        "C1_ok": added_parameters <= t["added_parameters_maximum"],
        "C2_changes_on_holdout": changes,
        "C2_ok": bool(
            changes["profile_error_layers_3_17_mean_points"] <= t["profile_regression_points"]
            and changes["coupling_abs_difference"] <= t["coupling_regression"]
            and changes["spot_number_rms_log_ratio"] <= t["spot_number_regression"]
            and all(changes[f"ks_{k}"] <= tol for k, tol in KS_TOLERANCES.items())
        ),
        "C3_profile_regression_per_energy_points": per_energy_profile,
        "C3_ok": bool(all(v <= t["profile_regression_per_energy_points"] for v in per_energy_profile.values())),
    }
    adopted = bool(all(criteria[k] for k in ("A1_ok", "A2_ok", "B1_ok", "B2_ok", "B3_ok", "C1_ok", "C2_ok", "C3_ok")))
    return {
        "thresholds": t,
        "criteria": criteria,
        "adopted": adopted,
        "verdict": (
            "E2 ADOPTED: every pre-stated criterion holds"
            if adopted
            else "E2 NOT ADOPTED: its base stays the candidate; the result is recorded"
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="e2 (on Slice 2) or faster_tail_e2 (on the faster tail)")
    parser.add_argument("--base-check", type=Path, required=True)
    args = parser.parse_args(argv)
    inputs = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in (
            args.base_check,
            RESULTS / f"em_production_lateral_{args.tag}_check.json",
            RESULTS / f"em_production_lateral_{args.tag}_occupancy.json",
            RESULTS / f"em_production_lateral_{args.tag}_quantum.json",
        )
    ]
    decision = evaluate(*inputs)
    (RESULTS / f"em_production_lateral_{args.tag}_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision["criteria"], indent=2))
    print(decision["verdict"])


if __name__ == "__main__":
    main()
