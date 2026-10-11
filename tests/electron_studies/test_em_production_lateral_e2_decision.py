"""The pre-stated adoption rule of extension E2: one failing case per criterion."""

import copy

from ams_ecal.electron_studies import em_production_lateral_e2_decision as decision

ENERGIES = ("10", "20", "50", "100")


def block(
    profile: float = 1.5, hits: float = 1.0, width: float = 0.2, coupling: float = 0.05, spot: float = 0.5
) -> dict:
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


def results(hold: dict, calibration: dict | None = None) -> dict:
    return {
        "per_energy": {
            energy: {
                "development_holdout": {"slice2": copy.deepcopy(hold)},
                "calibration_set": {"slice2": copy.deepcopy(calibration or hold)},
            }
            for energy in ENERGIES
        }
    }


def occupancy(ratio: float = 0.97) -> dict:
    return {
        "per_energy": {
            energy: {"model_occupied_cells": 300.0 * ratio, "geant4_holdout_occupied_cells": 300.0}
            for energy in ENERGIES
        }
    }


QUANTUM = {"tail_quantum_mev": 1.36, "per_energy_mev": {"10": 1.18, "20": 1.2, "50": 1.3, "100": 1.47}}
BASE = results(block(hits=1.0))
GOOD = results(block(hits=0.2))


def run(candidate: dict = GOOD, occ: dict | None = None, quantum: dict | None = None, added: int = 1) -> dict:
    return decision.evaluate(BASE, candidate, occ or occupancy(), quantum or QUANTUM, added)


def test_a_transferable_occupancy_gain_without_regression_is_adopted() -> None:
    verdict = run()

    assert verdict["adopted"] is True
    assert verdict["verdict"].startswith("E2 ADOPTED")


def test_an_overshoot_of_the_occupancy_fails_as_much_as_an_undershoot() -> None:
    assert run(occ=occupancy(1.15))["criteria"]["A1_ok"] is False
    assert run(occ=occupancy(0.85))["criteria"]["A1_ok"] is False


def test_a_hit_cell_ks_that_is_not_halved_fails_the_benefit() -> None:
    verdict = run(results(block(hits=0.6)))

    assert verdict["criteria"]["A2_ok"] is False
    assert verdict["adopted"] is False


def test_an_improvement_only_on_the_calibration_set_is_overfitting() -> None:
    verdict = run(results(block(hits=0.55), block(hits=0.05)))

    assert verdict["criteria"]["B1_ok"] is False


def test_a_profile_error_gap_between_the_holdout_and_the_calibration_set_is_overfitting() -> None:
    verdict = run(results(block(hits=0.2, profile=1.5), block(hits=0.2, profile=1.0)))

    assert verdict["criteria"]["B2_ok"] is False


def test_a_quantum_outside_its_own_measurements_fails() -> None:
    verdict = run(quantum={**QUANTUM, "tail_quantum_mev": 2.0})

    assert verdict["criteria"]["B3_ok"] is False


def test_more_than_one_added_parameter_fails() -> None:
    assert run(added=2)["criteria"]["C1_ok"] is False


def test_a_mean_regression_beyond_a_tolerance_fails() -> None:
    assert run(results(block(hits=0.2, profile=1.9)))["criteria"]["C2_ok"] is False
    assert run(results(block(hits=0.2, width=0.26)))["criteria"]["C2_ok"] is False
    assert run(results(block(hits=0.2, coupling=0.12)))["criteria"]["C2_ok"] is False


def test_one_energy_that_regresses_by_more_than_the_floor_fails_even_if_the_mean_is_fine() -> None:
    candidate = results(block(hits=0.2))
    candidate["per_energy"]["10"]["development_holdout"]["slice2"]["profile_error_points"]["layers_3_17_mean"] = 2.1

    verdict = run(candidate)

    assert verdict["criteria"]["C2_ok"] is True
    assert verdict["criteria"]["C3_ok"] is False
    assert verdict["adopted"] is False
