"""The pre-stated adoption rule of the tail-sensitive radial calibration: one failing case per criterion."""

import copy
from pathlib import Path

import pytest

from ams_ecal.electron_studies import em_production_lateral_tail_decision as decision

ENERGIES = ("10", "20", "50", "100")


def block(width: float, profile: float = 1.5, hits: float = 1.0, coupling: float = 0.05, spot: float = 0.5) -> dict:
    return {
        "ks": {
            "width_mm": width,
            "n_hit_cells": hits,
            "core_fraction": 0.15,
            "containment_fraction": 0.3,
            "max_cell_fraction": 0.2,
        },
        "profile_error_points": {"layers_3_17_mean": profile},
        "coupling_spearman": {"difference": coupling},
        "spot_number_rms_log_ratio": spot,
    }


def results(width: float, calibration_width: float | None = None, **hold_kwargs) -> dict:
    out: dict = {"per_energy": {}}
    for energy in ENERGIES:
        out["per_energy"][energy] = {
            "development_holdout": {"slice2": block(width, **hold_kwargs)},
            "calibration_set": {"slice2": block(width if calibration_width is None else calibration_width)},
        }
    return out


CALIBRATION = {"radial_fit": {"free_parameters": [f"p{i}" for i in range(12)], "parameters_at_bound": []}}
BASELINE = results(0.70)


def run(candidate: dict, calibration: dict | None = None) -> dict:
    return decision.evaluate(BASELINE, candidate, CALIBRATION if calibration is None else calibration)


def test_a_clear_transferable_width_improvement_without_regression_is_adopted() -> None:
    verdict = run(results(0.30))

    assert verdict["adopted"] is True
    assert verdict["verdict"].startswith("TAIL-SENSITIVE CALIBRATION ADOPTED")


def test_a_small_improvement_is_not_enough() -> None:
    verdict = run(results(0.65))

    assert verdict["criteria"]["A_ok"] is False
    assert verdict["adopted"] is False


def test_a_width_that_gets_worse_at_one_energy_fails_the_benefit() -> None:
    candidate = results(0.30)
    candidate["per_energy"]["100"]["development_holdout"]["slice2"]["ks"]["width_mm"] = 0.72

    assert run(candidate)["criteria"]["A_improves_at_every_energy"] is False


def test_an_improvement_only_on_the_calibration_set_is_overfitting() -> None:
    verdict = run(results(0.60, calibration_width=0.05))

    assert verdict["criteria"]["B1_ok"] is False
    assert verdict["adopted"] is False


def test_a_profile_error_gap_between_the_holdout_and_the_calibration_set_is_overfitting() -> None:
    candidate = results(0.30)
    for row in candidate["per_energy"].values():
        row["development_holdout"]["slice2"]["profile_error_points"]["layers_3_17_mean"] = 2.0

    verdict = run(candidate)

    assert verdict["criteria"]["B2_ok"] is False
    assert verdict["adopted"] is False


def test_a_constant_at_a_bound_is_not_adopted() -> None:
    bounded = copy.deepcopy(CALIBRATION)
    bounded["radial_fit"]["parameters_at_bound"] = ["tail_decay_rate"]

    assert run(results(0.30), bounded)["adopted"] is False


def test_an_added_parameter_complicates_the_parameterization() -> None:
    crowded = copy.deepcopy(CALIBRATION)
    crowded["radial_fit"]["free_parameters"].append("extra")

    verdict = run(results(0.30), crowded)

    assert verdict["criteria"]["C1_added_parameters"] == 1
    assert verdict["adopted"] is False


def test_a_regression_of_the_profile_or_the_coupling_is_not_adopted() -> None:
    worse_profile = run(results(0.30, profile=2.0))
    worse_coupling = run(results(0.30, coupling=0.2))

    assert worse_profile["criteria"]["C2_ok"] is False
    assert worse_coupling["criteria"]["C2_ok"] is False


def test_one_approved_added_parameter_is_allowed_only_when_the_rule_says_so() -> None:
    one_added = copy.deepcopy(CALIBRATION)
    one_added["radial_fit"]["free_parameters"].append("tail_exponent")
    candidate = results(0.30)

    strict = decision.evaluate(BASELINE, candidate, one_added)
    approved = decision.evaluate(BASELINE, candidate, one_added, max_added_parameters=1)

    assert strict["adopted"] is False and strict["criteria"]["C1_ok"] is False
    assert approved["adopted"] is True and approved["criteria"]["C1_added_parameters"] == 1
    assert approved["thresholds"]["added_parameters_maximum"] == 1


def test_two_added_parameters_exceed_the_approved_one() -> None:
    two_added = copy.deepcopy(CALIBRATION)
    two_added["radial_fit"]["free_parameters"] += ["a", "b"]

    verdict = decision.evaluate(BASELINE, results(0.30), two_added, max_added_parameters=1)

    assert verdict["criteria"]["C1_ok"] is False
    assert verdict["adopted"] is False


def test_the_rule_is_stated_in_the_result_and_the_module_never_reads_a_sealed_set() -> None:
    verdict = run(results(0.30))
    text = Path(decision.__file__).read_text(encoding="utf-8").lower()

    assert verdict["thresholds"]["mean_relative_reduction_minimum"] == pytest.approx(0.30)
    assert "geant4_electron_sealed" not in text
    assert "open_set" not in text
