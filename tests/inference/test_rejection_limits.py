"""Rejection limits: the exact numbers of the review's table, the censoring rule and the role separation."""

import numpy as np
import pytest
from scipy import stats

from ams_ecal.inference import rejection_limits as rl


def test_zero_of_9696_gives_a_lower_limit_of_about_3237_and_no_point_estimate():
    result = rl.rejection_from_counts(0, 9696)
    assert result.censored and result.rejection_estimate is None and result.rejection_two_sided is None
    assert result.rejection_lower_limit == pytest.approx(3237, abs=2)  # review C1: the old code said 9,696
    assert "rejection > 3,23" in result.statement() and "0 of 9,696" in result.statement()


def test_zero_failures_follow_the_rule_of_three_closely():
    for n in (1000, 20_000, 300_000):
        assert rl.clopper_pearson_upper(0, n) == pytest.approx(3.0 / n, rel=0.01)


def test_a_measured_rejection_carries_an_interval_around_n_over_k():
    result = rl.rejection_from_counts(50, 100_000)
    assert result.rejection_estimate == 2000
    lo, hi = result.rejection_two_sided
    assert lo < 2000 < hi and lo == pytest.approx(1 / 0.000659, rel=0.02)


def test_the_sample_sizes_of_the_review_table_are_reproduced():
    table = {  # (target rejection, k): protons needed (review section 3, exact one-sided 95% Clopper-Pearson)
        (1e3, 0): 2995,
        (1e4, 0): 29_956,
        (1e5, 0): 299_572,
        (1e3, 1): 4742,
        (1e4, 1): 47_437,
        (1e3, 2): 6294,
        (1e3, 5): 10_511,
    }
    for (target, k), expected in table.items():
        assert rl.protons_needed(target, k) == pytest.approx(expected, abs=2), (target, k)


def test_protons_needed_is_the_smallest_n_that_proves_the_claim():
    n = rl.protons_needed(1e3, 2)
    assert rl.rejection_from_counts(2, n).rejection_lower_limit >= 1e3
    assert rl.rejection_from_counts(2, n - 1).rejection_lower_limit < 1e3


def test_bad_counts_are_refused():
    for k, n in ((-1, 10), (11, 10), (0, 0)):
        with pytest.raises(ValueError):
            rl.clopper_pearson_upper(k, n)


def test_all_passing_gives_a_rejection_limit_of_one():
    assert rl.rejection_from_counts(100, 100).rejection_lower_limit == 1.0


def test_event_keys_are_unique_per_particle_energy_and_index_and_roles_must_be_disjoint():
    keys_e = rl.event_key(1, 20.0, np.arange(5))
    keys_p = rl.event_key(2, 20.0, np.arange(5))
    assert len(np.union1d(keys_e, keys_p)) == 10
    rl.assert_disjoint_roles({"calibration": keys_e[:3], "evaluation": keys_e[3:]})
    with pytest.raises(rl.RolesOverlapError, match="calibration and evaluation share 1 events"):
        rl.assert_disjoint_roles({"calibration": keys_e[:3], "evaluation": keys_e[2:]})


def test_the_fresh_electron_efficiency_of_the_threshold_is_beta_distributed():
    """Simulation: the achieved efficiency of an order-statistic threshold on fresh electrons covers as the Beta law says."""

    rng = np.random.default_rng(0)
    m, efficiency = 400, 0.9
    lo, hi = rl.threshold_efficiency_interval(m, efficiency, 0.95)
    achieved = []
    for _ in range(3000):
        threshold = rl.electron_threshold(rng.normal(size=m), efficiency)
        achieved.append(1.0 - stats.norm.cdf(threshold))  # exact efficiency of that threshold on fresh N(0,1) electrons
    achieved = np.array(achieved)
    assert ((achieved >= lo) & (achieved <= hi)).mean() == pytest.approx(0.95, abs=0.02)


def test_the_conservative_threshold_reaches_the_target_efficiency_with_the_stated_confidence():
    rng = np.random.default_rng(1)
    m, efficiency, confidence = 300, 0.9, 0.95
    reached = []
    for _ in range(3000):
        threshold = rl.conservative_threshold(rng.normal(size=m), efficiency, confidence)
        reached.append(1.0 - stats.norm.cdf(threshold) >= efficiency)
    assert np.mean(reached) >= confidence - 0.01


def test_evaluation_scores_cannot_move_the_threshold():
    rng = np.random.default_rng(2)
    cal, ev_e = rng.normal(size=500), rng.normal(size=2000)
    a = rl.evaluate_at_efficiency(cal, ev_e, rng.normal(-3, 1, size=5000), 0.9)
    b = rl.evaluate_at_efficiency(cal, ev_e, rng.normal(+3, 1, size=5000), 0.9)
    assert a.threshold == b.threshold
    assert a.rejection.k < b.rejection.k


def test_evaluate_reports_counts_efficiency_and_a_bootstrap_range_of_the_limit():
    rng = np.random.default_rng(3)
    result = rl.evaluate_at_efficiency(
        rng.normal(size=1000), rng.normal(size=4000), rng.normal(-8, 1, size=20_000), 0.99, bootstrap=40, rng=rng
    )
    assert result.rejection.n == 20_000 and result.rejection.censored
    assert result.achieved_efficiency_interval[0] < 0.99 < result.achieved_efficiency_interval[1] + 0.01
    low, high = result.rejection_lower_limit_bootstrap
    assert low <= high <= 20_000
