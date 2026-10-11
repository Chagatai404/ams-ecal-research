"""Missingness controls: the indicator alone reproduces a 'feature' that is only missingness, and matching removes the confounder."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ams_ecal.inference import cross_fitting as cf
from ams_ecal.inference import missingness_controls as mc


def test_indicator_only_separation_is_exact_for_known_fractions():
    # class A eligible 80%, class B eligible 30%: AUC = P(A defined, B not) + 0.5 P(tie) = 0.8*0.7 + 0.5*(0.8*0.3 + 0.2*0.7)
    a = np.array([True] * 80 + [False] * 20)
    b = np.array([True] * 30 + [False] * 70)
    result = mc.indicator_only_separation(a, b)
    assert result["auc"] == pytest.approx(0.8 * 0.7 + 0.5 * (0.8 * 0.3 + 0.2 * 0.7))
    assert result["separation"] == pytest.approx(result["auc"]) and result["eligible_fraction_b"] == 0.3
    assert mc.indicator_only_separation(~a, ~b)["separation"] == pytest.approx(result["separation"])


def test_a_feature_that_is_only_missingness_scores_like_the_indicator():
    """The failure mode of the old ladder: a NaN-or-constant feature separates classes exactly as well as 'was it defined'."""

    rng = np.random.default_rng(0)
    cells = np.concatenate([rng.normal(500, 150, 3000), rng.normal(350, 150, 3000)])  # class A larger events
    y = np.concatenate([np.ones(3000, int), np.zeros(3000, int)])
    feature = np.where(cells >= 400, 1.0, np.nan)  # defined only for big events, and carrying no content
    auc_with_nan = cf.repeated_cross_fit(feature[:, None], y, seeds=range(2)).mean
    indicator = mc.indicator_only_separation(~np.isnan(feature[y == 1]), ~np.isnan(feature[y == 0]))
    assert auc_with_nan == pytest.approx(indicator["auc"], abs=0.02)


def test_missing_indicators_are_explicit_columns_and_imputation_keeps_the_rest():
    x = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan], [7.0, 8.0, 9.0]])
    table, indicators = mc.add_missing_indicators(x)
    assert table.shape == (3, 5) and indicators == [3, 4]
    assert table[:, 3].tolist() == [1.0, 0.0, 0.0] and table[:, 4].tolist() == [0.0, 1.0, 0.0]
    assert not np.isnan(mc.impute_for_model(table)).any() and mc.impute_for_model(table)[0, 0] == 1.0


def test_eligible_in_both_matches_the_key_distribution_and_reports_the_counts():
    rng = np.random.default_rng(1)
    key_a, key_b = rng.normal(500, 100, 4000), rng.normal(420, 100, 1500)
    values_a, values_b = np.where(key_a > 336, 1.0, np.nan), np.where(key_b > 336, 1.0, np.nan)
    result = mc.eligible_in_both(values_a, values_b, key_a, key_b, 5, rng)
    assert result.n_eligible_a > result.n_eligible_b
    assert len(result.index_a) == len(result.index_b) == sum(n for n, _ in result.n_per_stratum)
    assert np.isfinite(values_a[result.index_a]).all() and np.isfinite(values_b[result.index_b]).all()
    assert np.mean(key_a[result.index_a]) == pytest.approx(np.mean(key_b[result.index_b]), abs=8)  # before matching: ~80 apart


def test_testability_declares_cells_with_too_few_eligible_events():
    plenty, few = np.ones(500), np.concatenate([np.ones(26), np.full(974, np.nan)])
    assert mc.testability(plenty, plenty)["testable"]
    verdict = mc.testability(plenty, few)
    assert not verdict["testable"] and verdict["eligible_b"] == 26 and "not testable" in verdict["verdict"]


def test_the_roc_helper_input_conventions_are_the_expected_ones():
    assert roc_auc_score([1, 0], [1.0, 0.0]) == 1.0  # label 1 above label 0 gives AUC 1: the convention used above
