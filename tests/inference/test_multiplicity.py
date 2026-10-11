"""Multiplicity: hand-computed adjusted p-values, then error-rate simulations under a global null and with real effects."""

import numpy as np
import pytest

from ams_ecal.inference import multiplicity as mp


def test_benjamini_hochberg_matches_a_hand_computation():
    # sorted 0.005, 0.01, 0.03, 0.04 -> p*m/rank = 0.02, 0.02, 0.04, 0.04
    q = mp.benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.005]))
    assert q == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_holm_matches_a_hand_computation():
    # sorted 0.005*4, 0.01*3, 0.03*2, 0.04*1 = 0.02, 0.03, 0.06, 0.04 -> running max 0.02, 0.03, 0.06, 0.06
    adjusted = mp.holm(np.array([0.01, 0.04, 0.03, 0.005]))
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_adjusted_values_never_fall_below_the_raw_ones_and_never_exceed_one():
    rng = np.random.default_rng(0)
    p = rng.random(50) ** 3
    for adjusted in (mp.benjamini_hochberg(p), mp.holm(p)):
        assert (adjusted >= p - 1e-12).all() and (adjusted <= 1.0).all()


def test_holm_and_westfall_young_control_the_family_wise_error_under_the_global_null():
    rng = np.random.default_rng(1)
    m, b, trials = 8, 199, 400
    holm_errors = wy_errors = 0
    for _ in range(trials):
        draws = rng.normal(size=(b + 1, m))  # the observed row is one more draw of the null: exchangeable
        observed, null = np.abs(draws[0]), np.abs(draws[1:])
        p_raw = np.array([(1 + np.sum(null[:, j] >= observed[j])) / (1 + b) for j in range(m)])
        holm_errors += mp.holm(p_raw).min() <= 0.05
        wy_errors += mp.westfall_young_stepdown(observed, null).min() <= 0.05
    assert holm_errors / trials <= 0.08 and wy_errors / trials <= 0.08


def test_westfall_young_keeps_power_for_a_real_effect_and_is_monotone_in_the_statistic():
    rng = np.random.default_rng(2)
    null = np.abs(rng.normal(size=(999, 6)))
    observed = np.array([6.0, 0.3, 0.5, 1.0, 0.2, 0.4])
    adjusted = mp.westfall_young_stepdown(observed, null)
    assert adjusted[0] == pytest.approx(1 / 1000) and adjusted[1:].min() > 0.2
    order = np.argsort(-observed)
    assert (np.diff(adjusted[order]) >= -1e-12).all()


def test_benjamini_hochberg_controls_the_false_discovery_rate():
    rng = np.random.default_rng(3)
    fdp = []
    for _ in range(300):
        null_p = rng.random(80)
        real_p = rng.random(20) * 0.0005
        q = mp.benjamini_hochberg(np.concatenate([null_p, real_p]))
        discoveries = q <= 0.05
        fdp.append(discoveries[:80].sum() / max(discoveries.sum(), 1))
    assert np.mean(fdp) <= 0.05 + 0.02


def test_secondaries_stay_exploratory_unless_their_familys_primary_is_confirmed():
    primaries = {"P1": ("D2 at 20 GeV, deposition", 0.001), "P2": ("paired degradation", 0.4), "P3": ("occupancy KS", 0.04)}
    secondaries = {"P1": {"D0": 0.001, "D1": 0.2}, "P2": {"D0": 0.0001}}
    decisions = mp.decide(primaries, secondaries)
    assert decisions["P1"].confirmed and not decisions["P2"].confirmed
    assert decisions["P1"].secondary_status == {"D0": "supported", "D1": "exploratory"}
    assert decisions["P2"].secondary_status == {"D0": "exploratory"}  # a tiny q does not rescue an unconfirmed primary
    assert decisions["P3"].primary_adjusted_p == pytest.approx(0.08)  # Holm: 0.001*3, 0.04*2, 0.4*1 -> 0.003, 0.08, 0.4
