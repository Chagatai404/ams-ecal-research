"""Permutation nulls, the stratified KS, the split floor and the cell-shuffle null, each against a case with a known answer."""

import numpy as np
import pytest

from ams_ecal.inference import permutation_nulls as pn
from ams_ecal.multiscale_estimators import multiscale as ms


def test_joint_strata_has_equal_counts_per_key_bin():
    rng = np.random.default_rng(0)
    strata = pn.joint_strata(rng.normal(size=5000), rng.normal(size=5000), n_strata=5)
    counts = np.bincount(strata)
    assert len(counts) == 25 and counts.min() > 100


def test_permuting_labels_within_strata_keeps_the_class_counts_of_every_stratum():
    rng = np.random.default_rng(1)
    strata = rng.integers(0, 4, size=400)
    labels = (rng.random(400) < 0.3).astype(int)
    permuted = pn.permute_labels_within_strata(labels, strata, rng)
    for s in range(4):
        assert permuted[strata == s].sum() == labels[strata == s].sum()
    assert not np.array_equal(permuted, labels)


def test_no_difference_gives_p_values_that_are_not_small_and_a_real_difference_gives_the_minimum():
    rng = np.random.default_rng(2)
    labels = np.repeat([1, 0], 300)
    same = rng.normal(size=600)
    assert pn.permutation_pvalue(pn.stratum_ks, same, labels, None, rng, 200).p_value > 0.05
    shifted = same + 2.0 * labels
    result = pn.permutation_pvalue(pn.stratum_ks, shifted, labels, None, rng, 200)
    assert result.p_value == pytest.approx(1 / 201) and result.observed > result.null_95th


def test_stratifying_removes_a_difference_that_only_a_confounder_creates():
    """Classes differ in the confounder (the key); the value depends on the key only. Unstratified: 'significant'. Stratified: not."""

    rng = np.random.default_rng(3)
    labels = np.repeat([1, 0], 600)
    key = np.concatenate([rng.normal(1.0, 1.0, 600), rng.normal(-1.0, 1.0, 600)])
    values = key + 0.2 * rng.normal(size=1200)
    strata = pn.equal_count_strata(key, 8)
    assert pn.permutation_pvalue(pn.stratum_ks, values, labels, None, rng, 200).p_value < 0.01
    assert pn.permutation_pvalue(pn.stratum_ks, values, labels, strata, rng, 200).p_value > 0.05


def test_small_strata_are_counted_not_skipped_silently():
    rng = np.random.default_rng(4)
    values = rng.normal(size=330)
    labels = np.concatenate([np.repeat([1, 0], 150), [1] * 15, [0] * 15])
    strata = np.concatenate([np.zeros(300, int), np.ones(30, int)])
    report = pn.stratum_ks_report(values, labels, strata)
    assert report["strata_used"] == 1 and report["strata_skipped"] == 1 and report["events_used"] == 300
    assert np.isnan(pn.stratum_ks(values, labels, np.arange(330) % 30))


def test_nan_values_are_not_events():
    values = np.array([np.nan] * 100 + list(np.linspace(0, 1, 100)) + list(np.linspace(0, 1, 100)))
    labels = np.concatenate([np.ones(100), np.ones(100), np.zeros(100)]).astype(int)
    assert pn.stratum_ks_report(values, labels, None)["events_used"] == 200


def _ks_of_two_samples(a, b):
    labels = np.concatenate([np.ones(len(a)), np.zeros(len(b))]).astype(int)
    return pn.stratum_ks(np.concatenate([a, b]), labels, None)


def test_the_split_floor_of_identical_distributions_is_small_and_shrinks_with_size():
    rng = np.random.default_rng(5)
    values = rng.normal(size=20_000)
    small = np.median(pn.matched_size_split_floor(values, 100, 100, _ks_of_two_samples, rng, 100))
    large = np.median(pn.matched_size_split_floor(values, 5000, 5000, _ks_of_two_samples, rng, 100))
    assert large < small < 0.25


def test_the_shuffle_keeps_each_layers_cell_energies_and_moves_their_positions():
    rng = np.random.default_rng(6)
    grids = np.where(rng.random((3, 18, 72)) < 0.2, rng.random((3, 18, 72)), 0.0)
    shuffled = pn.shuffle_cells_within_layers(grids, rng)
    assert np.allclose(np.sort(shuffled, axis=-1), np.sort(grids, axis=-1))
    assert np.allclose(shuffled.sum(axis=-1), grids.sum(axis=-1))
    assert ((shuffled > 0).sum(axis=-1) == (grids > 0).sum(axis=-1)).all()
    assert not np.array_equal(shuffled, grids)


def test_a_compact_cluster_is_detected_against_its_shuffled_copies_and_a_random_pattern_is_not():
    rng = np.random.default_rng(7)
    cluster = np.zeros((6, 18, 72))
    cluster[:, :6, 30:36] = 1.0  # a 6 x 6 block aligned with the 6-cell boxes

    def top_box_share(grid):
        return float(ms.box_shares(grid, 6).max())

    detected = pn.paired_shuffle_excess(cluster, top_box_share, 100, rng, 50)
    assert (detected.excess > 0.5).all() and (detected.event_p_value < 0.05).all()
    assert detected.median_excess_95[0] > 0.5

    noise = np.where(rng.random((40, 18, 72)) < 0.2, 1.0, 0.0)
    null = pn.paired_shuffle_excess(noise, top_box_share, 100, rng, 50)
    assert abs(null.median_excess) < 0.02 and (null.event_p_value < 0.05).mean() < 0.15
