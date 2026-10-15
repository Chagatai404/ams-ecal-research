import numpy as np
import pytest

from ams_ecal.inference import permutation_nulls as pn
from ams_ecal.multiscale_estimators import multiscale_verdict_reread as vr


def synthetic(seed=0, n=900):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n, 4))
    values[:, 1] = np.round(values[:, 1] * 3)  # ties
    values[rng.random(n) < 0.3, 2] = np.nan  # ineligible events
    values[:, 3] = np.where(np.arange(n) < 400, np.nan, values[:, 3])  # one stratum has no finite events at all
    strata = np.digitize(np.arange(n), [250, 500, 700])
    labels = (rng.random(n) < 0.4).astype(np.int8)
    values[labels == 1, 0] += 0.4  # a real difference in column 0
    return values, labels, strata, rng


def test_the_engine_reproduces_stratum_ks_on_the_observed_labelling():
    values, labels, strata, _ = synthetic()
    got = vr.stratified_ks_matrix(values, labels[None, :], strata)[0]
    for j in range(values.shape[1]):
        expected = pn.stratum_ks(values[:, j], labels, strata)
        assert (np.isnan(expected) and np.isnan(got[j])) or got[j] == pytest.approx(expected, abs=1e-6)


def test_the_engine_reproduces_stratum_ks_on_permuted_labellings():
    values, labels, strata, rng = synthetic(1)
    perms = vr.permuted_labels(labels, strata, 6, rng)
    got = vr.stratified_ks_matrix(values, perms, strata)
    for r in range(6):
        for j in range(values.shape[1]):
            expected = pn.stratum_ks(values[:, j], perms[r], strata)
            assert (np.isnan(expected) and np.isnan(got[r, j])) or got[r, j] == pytest.approx(expected, abs=1e-6)


def test_permuted_labels_keep_the_class_counts_inside_every_stratum():
    _, labels, strata, rng = synthetic(2)
    perms = vr.permuted_labels(labels, strata, 5, rng)
    for s in np.unique(strata):
        inside = strata == s
        assert (perms[:, inside].sum(axis=1) == labels[inside].sum()).all()
    assert not (perms == labels).all()


def test_the_block_test_detects_a_planted_difference_and_not_a_null_column():
    values, labels, _, rng = synthetic(3, n=1200)
    key = np.arange(1200, dtype=float)
    electrons, protons = values[labels == 1], values[labels == 0]
    result = vr.block_test(electrons, protons, key[labels == 1], key[labels == 0], permutations=399, rng=rng)
    p0 = (1 + (result["null"][:, 0] >= result["observed"][0]).sum()) / 400
    p1 = (1 + (result["null"][:, 1] >= result["observed"][1]).sum()) / 400
    assert p0 < 0.01 and p1 > 0.05
