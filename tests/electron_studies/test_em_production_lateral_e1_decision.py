"""The pre-stated E1 adoption rule: adopted only if every criterion holds, with one failing case per criterion."""

import copy
from pathlib import Path

import pytest

from ams_ecal.electron_studies import em_production_lateral_e1_decision as decision

ENERGIES = ("10", "20", "50", "100")


def block(width: float, hits: float, spot_rms: float = 0.2) -> dict:
    return {
        "ks": {
            "width_mm": width,
            "n_hit_cells": hits,
            "core_fraction": 0.15,
            "containment_fraction": 0.3,
            "max_cell_fraction": 0.2,
        },
        "profile_error_points": {"layers_3_17_mean": 1.5},
        "coupling_spearman": {"difference": 0.05},
        "spot_number_rms_log_ratio": spot_rms,
    }


def results(
    width: float,
    hits: float,
    calibration_width: float | None = None,
    calibration_hits: float | None = None,
    spot_gap: float = 0.0,
) -> dict:
    out: dict = {"per_energy": {}}
    for energy in ENERGIES:
        out["per_energy"][energy] = {
            "development_holdout": {"slice2": block(width, hits, 0.2 + spot_gap)},
            "calibration_set": {
                "slice2": block(
                    width if calibration_width is None else calibration_width,
                    hits if calibration_hits is None else calibration_hits,
                    0.2,
                )
            },
        }
    return out


CALIBRATION = {
    "spot_fit": {
        "free_parameters": ["a", "b", "c", "d", "kappa", "n0"],
        "extension_collapsed": False,
        "converged": True,
    }
}
BASELINE = results(0.70, 1.0)


def run(extension: dict, calibration: dict | None = None) -> dict:
    return decision.evaluate(BASELINE, extension, CALIBRATION if calibration is None else calibration)


def test_a_clear_transferable_improvement_with_two_parameters_is_adopted() -> None:
    verdict = run(results(0.30, 0.40))

    assert verdict["adopted"] is True
    assert verdict["verdict"].startswith("E1 ADOPTED")


def test_no_benefit_means_not_adopted() -> None:
    verdict = run(results(0.69, 0.99))

    assert verdict["adopted"] is False
    assert verdict["criteria"]["A_mean_relative_reduction_ok"] is False


def test_an_improvement_that_does_not_transfer_to_the_holdout_is_overfitting() -> None:
    extension = results(0.30, 0.40, calibration_width=0.05, calibration_hits=0.05)

    verdict = run(extension)

    assert verdict["criteria"]["B1_ok"] is False
    assert verdict["adopted"] is False


def test_a_spot_number_gap_between_calibration_and_holdout_is_overfitting() -> None:
    verdict = run(results(0.30, 0.40, spot_gap=0.2))

    assert verdict["criteria"]["B2_ok"] is False
    assert verdict["adopted"] is False


def test_a_collapsed_or_unconverged_extension_is_not_adopted() -> None:
    collapsed = copy.deepcopy(CALIBRATION)
    collapsed["spot_fit"]["extension_collapsed"] = True

    assert run(results(0.30, 0.40), collapsed)["adopted"] is False


def test_more_than_two_added_parameters_complicate_the_parameterization() -> None:
    crowded = copy.deepcopy(CALIBRATION)
    crowded["spot_fit"]["free_parameters"] = ["a", "b", "c", "d", "e", "f", "g"]

    verdict = run(results(0.30, 0.40), crowded)

    assert verdict["criteria"]["C1_added_parameters"] == 3
    assert verdict["adopted"] is False


def test_a_regression_of_what_already_worked_is_not_adopted() -> None:
    extension = results(0.30, 0.40)
    for row in extension["per_energy"].values():
        for label in ("development_holdout", "calibration_set"):
            row[label]["slice2"]["ks"]["core_fraction"] = 0.40

    verdict = run(extension)

    assert verdict["criteria"]["C2_ok"] is False
    assert verdict["adopted"] is False


def test_the_thresholds_are_stated_in_the_result_and_the_module_never_reads_a_sealed_set() -> None:
    verdict = run(results(0.30, 0.40))
    text = Path(decision.__file__).read_text(encoding="utf-8").lower()

    assert verdict["thresholds"]["mean_relative_reduction_minimum"] == pytest.approx(0.30)
    assert "geant4_electron_sealed" not in text
    assert "open_set" not in text
