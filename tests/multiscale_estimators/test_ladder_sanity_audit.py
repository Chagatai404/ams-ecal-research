"""The helpers of the ladder sanity audit, on synthetic data: they must return the known answer of each control."""

import numpy as np

from ams_ecal.multiscale_estimators import ladder_sanity_audit as audit
from ams_ecal.multiscale_estimators import multiscale_test as mt


def test_row_hashes_find_identical_events_and_only_those():
    rng = np.random.default_rng(0)
    grids = rng.random((5, 18, 72)).astype(np.float32)
    grids = np.concatenate([grids, grids[[1, 3]]])  # events 1 and 3 appear twice
    hashes = audit.row_hashes(grids)
    assert len(hashes) - len(set(hashes)) == 2
    assert hashes[1] == hashes[5] and hashes[3] == hashes[6] and hashes[0] != hashes[2]


def test_univariate_separated_samples_give_auc_one_and_ks_one():
    low, high = np.linspace(0, 1, 200), np.linspace(2, 3, 200)
    result = audit.univariate(high, low)
    assert result["auc_a_above_b"] == 1.0 and result["ks"] == 1.0
    assert audit.univariate(low, high)["auc_a_above_b"] == 0.0


def test_a_two_sided_feature_has_large_ks_but_auc_one_half():
    """The counter-example behind the corrected expectation: AUC >= (1 + KS) / 2 is not a general inequality."""

    a = np.array([0.0] * 50 + [2.0] * 50)
    b = np.full(100, 1.0)
    result = audit.univariate(a, b)
    assert result["ks"] == 0.5 and result["auc_a_above_b"] == 0.5


def test_counts_above_the_electron_threshold_are_k_of_n():
    scores = np.concatenate([np.linspace(0.5, 1.0, 100), np.linspace(0.0, 0.6, 1000)])
    labels = np.concatenate([np.ones(100, int), np.zeros(1000, int)])
    counts = audit.counts_above_electron_threshold(scores, labels, 0.9)
    assert counts["n_protons"] == 1000 and 0 < counts["k_passing"] < 1000


def test_the_same_class_null_is_one_half_and_a_real_difference_is_not():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(1500, 4)), rng.normal(size=(1500, 4))
    x, y = audit.pooled(a, b)
    assert abs(audit.hgb_auc(x, y) - 0.5) < 0.04
    shifted = audit.pooled(a, b + np.array([1.5, 0, 0, 0]))
    assert audit.hgb_auc(*shifted) > 0.8
    assert audit.logistic_auc(*shifted) > 0.8


def test_families_cover_every_conventional_feature_exactly_once():
    names, _ = mt.conventional_features(np.random.default_rng(0).random((4, 18, 72)))
    covered = [n for family in audit.FAMILIES.values() for n in family]
    assert sorted(covered) == sorted(names)
    blocks = [n for block in audit.LAYER_BLOCKS.values() for n in block]
    assert sorted(blocks) == sorted(n for n in names if n.startswith("layer_fraction_"))
