"""The coverage simulations pass on sound tools and FAIL on a broken setup (a check that cannot fail checks nothing)."""

import numpy as np
import pytest

from ams_ecal.inference import coverage_simulations as cov


def test_clopper_pearson_covers_including_when_most_draws_are_censored():
    rng = np.random.default_rng(0)
    for true_rate in (0.0, 1e-5, 1e-4, 1e-3, 1e-2):
        result = cov.clopper_pearson_coverage(true_rate, 9696, 0.95, 3000, rng)
        assert result["passed"] and result["coverage"] >= 0.94, result
    assert cov.clopper_pearson_coverage(1e-4, 9696, 0.95, 3000, rng)["share_of_draws_with_k_equal_0"] > 0.3


def test_the_rejection_lower_limit_never_exceeds_the_true_rejection_more_often_than_stated():
    rng = np.random.default_rng(1)
    for true_rejection in (1e3, 1e4, 1e5):
        assert cov.rejection_limit_coverage(true_rejection, 20_000, 0.95, 3000, rng)["passed"]


def test_the_permutation_test_holds_its_level_on_null_data_and_the_check_fails_when_a_difference_is_planted():
    rng = np.random.default_rng(2)

    def null_pair(g):
        return g.normal(size=300), g.normal(size=300)

    def shifted_pair(g):
        return g.normal(size=300), g.normal(size=300) + 0.5

    ok = cov.permutation_type_one_error(null_pair, trials=150, alpha=0.05, n_permutations=99, rng=rng)
    bad = cov.permutation_type_one_error(shifted_pair, trials=60, alpha=0.05, n_permutations=99, rng=rng)
    assert ok["passed"] and ok["rejection_rate"] <= 0.12
    assert not bad["passed"] and bad["rejection_rate"] > 0.5  # the harness can fail


def test_adjustment_is_needed_the_unadjusted_minimum_p_exceeds_alpha_and_holm_and_westfall_young_do_not():
    rng = np.random.default_rng(3)

    def null_pair(g):
        return g.normal(size=(200, 8)), g.normal(size=(200, 8))

    result = cov.familywise_type_one_error(null_pair, trials=120, alpha=0.05, n_permutations=99, rng=rng)
    assert result["unadjusted_minimum_p_rate"] > 0.15  # 8 looks at a null: about 1 - 0.95**8 = 0.34
    assert result["passed"] and result["holm_rate"] <= 0.1 and result["westfall_young_rate"] <= 0.1


def test_the_tolerance_is_three_monte_carlo_standard_errors():
    assert cov._mc_tolerance(0.05, 200) == pytest.approx(3 * np.sqrt(0.05 * 0.95 / 200))
