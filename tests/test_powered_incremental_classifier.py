"""Tests for the EXP-019 powered incremental-classifier module (registration 4efe4d15d845e6aa)."""

from pathlib import Path

import numpy as np
import pytest

from ams_ecal.benchmarks import powered_incremental_classifier as pic

PILOT = Path("data/geant4_realistic_regime_pilot_electron")
PILOT_P = Path("data/geant4_realistic_regime_pilot_proton")


@pytest.mark.parametrize(
    ("lower", "upper", "outcome"),
    [
        (-0.030, -0.001, pic.OUTCOME_ADDS),
        (-0.004, 0.012, pic.OUTCOME_NO_GAIN),
        (-0.015, 0.004, pic.OUTCOME_INCONCLUSIVE),
        (0.001, 0.020, pic.OUTCOME_WORSE),
    ],
)
def test_decide_returns_the_four_registered_outcomes(lower, upper, outcome):
    assert pic.decide(lower, upper)["outcome"] == outcome


def test_decide_flags_a_gain_smaller_than_delta_min():
    assert pic.decide(-0.008, -0.001)["smaller_than_delta_min"] is True
    assert pic.decide(-0.020, -0.001)["smaller_than_delta_min"] is False


def test_boundary_lower_equal_to_minus_delta_min_is_inconclusive():
    assert pic.decide(-pic.DELTA_MIN, 0.005)["outcome"] == pic.OUTCOME_INCONCLUSIVE


def test_frozen_hard_subset_uses_absolute_caliper():
    # identical events are matched; events 0.3 SD apart in k1 are not
    cells = np.array([800.0, 800.0])
    energy = np.array([1e5, 1e5])
    keep_e, keep_p = pic.frozen_hard_subset(cells, energy, cells, energy)
    assert keep_e.all() and keep_p.all()
    shifted = cells * np.exp(0.3 * pic.FROZEN_SD[0])
    keep_e, keep_p = pic.frozen_hard_subset(cells, energy, shifted, energy)
    assert not keep_e.any() and not keep_p.any()


@pytest.mark.skipif(not (PILOT.exists() and PILOT_P.exists()), reason="pilot data not present")
def test_frozen_constants_reproduce_the_exp018_hard_subset_on_the_pilot():
    from ams_ecal.multiscale_estimators import realistic_regime_analysis as rra

    ev_e, ev_p = rra.load_class_events(PILOT, "electron"), rra.load_class_events(PILOT_P, "proton")
    pe, _ = pic.build_part(ev_e, np.ones(len(ev_e), bool), "electron", 1, 0, "pilot")
    pp, _ = pic.build_part(ev_p, ev_p.truth_occurred.astype(bool), "proton_interacting", 2, 0, "pilot")
    keep_e, keep_p = pic.frozen_hard_subset(pe.images.cells, pe.energy_mev, pp.images.cells, pp.energy_mev)
    assert (int(keep_e.sum()), int(keep_p.sum())) == (191, 206)


@pytest.mark.skipif(not PILOT.exists(), reason="pilot data not present")
def test_fibre_resum_reproduces_the_readout_grid():
    from ams_ecal.multiscale_estimators import realistic_regime_analysis as rra

    ev = rra.load_class_events(PILOT, "electron")
    keep = np.flatnonzero(rra.fiducial_of(ev))[:5]
    grids, gate = pic.fibre_grids(PILOT, ev, keep, None)
    assert gate["cells_outside_tolerance"] == 0
    assert gate["max_relative_difference"] < 1e-5
    saturated, _ = pic.fibre_grids(PILOT, ev, keep, pic.SATURATION_SCALE_MEV)
    assert (saturated <= grids + 1e-9).all() and saturated.sum() < grids.sum()
