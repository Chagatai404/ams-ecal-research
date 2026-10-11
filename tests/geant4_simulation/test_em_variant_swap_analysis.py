"""The decision rule of the EMV swap analysis, on constructed cases whose verdict is known (no Geant4, no data files)."""

import numpy as np
import pytest

from ams_ecal.geant4_simulation import em_variant_swap_analysis as swap


def sets(rng, share, n=2000):
    total = rng.normal(18_000, 300, n)
    return total * share * (1 + 0.05 * rng.normal(size=n)), total


def test_the_share_is_the_median_of_the_per_event_ratio():
    assert swap.share(np.array([1.0, 2.0, 30.0]), np.array([10.0, 10.0, 10.0])) == pytest.approx(0.2)


def test_restored_fraction_is_zero_for_no_change_and_one_for_full_restoration():
    assert swap.restored_fraction(0.041, 0.041, 0.065) == 0.0
    assert swap.restored_fraction(0.065, 0.041, 0.065) == 1.0
    assert swap.restored_fraction(0.053, 0.041, 0.065) == pytest.approx(0.5)


def test_the_closure_tolerance_is_three_percent():
    assert swap.closure_passes(0.0640, 0.0653) and not swap.closure_passes(0.0600, 0.0653)


def test_verdicts_follow_the_rule_fixed_in_advance():
    assert swap.verdict((0.6, 1.1)) == "mostly responsible"
    assert swap.verdict((-0.05, 0.15)) == "not responsible"
    assert swap.verdict((0.1, 0.9)) == "partly responsible / unresolved"


def test_a_planted_full_restoration_and_a_planted_null_are_recognised_by_the_bootstrap():
    rng = np.random.default_rng(0)
    reference, emv = sets(rng, 0.065, 5000), sets(rng, 0.041)
    restored, unchanged = sets(rng, 0.065), sets(rng, 0.041)
    low, high = swap.bootstrap_restored_fraction(restored, emv, reference, rng, draws=300)
    assert swap.verdict((low, high)) == "mostly responsible" and low > 0.8
    low, high = swap.bootstrap_restored_fraction(unchanged, emv, reference, rng, draws=300)
    assert swap.verdict((low, high)) == "not responsible" and abs((low + high) / 2) < 0.1


def test_layer_ratio_is_readout_over_deposit_per_layer():
    deposit = np.ones((5, 18, 72))
    readout = np.zeros_like(deposit)
    readout[:, :, :] = 0.05
    ratio = swap.layer_ratio(readout, deposit)
    assert ratio.shape == (18,) and np.allclose(ratio, 0.05)
