"""Cross-fitting: spread across fold seeds, role-separated fitting, baselines and the same-class null on synthetic problems."""

import numpy as np
import pytest

from ams_ecal.inference import cross_fitting as cf
from ams_ecal.inference import rejection_limits as rl


def two_classes(rng, n_e=1500, n_p=1500, shift=2.0, dims=4):
    electrons = rng.normal(size=(n_e, dims))
    protons = rng.normal(size=(n_p, dims)) + np.array([shift] + [0.0] * (dims - 1))
    return electrons, protons


def test_repeated_cross_fit_reports_one_auc_per_seed_and_their_spread():
    rng = np.random.default_rng(0)
    e, p = two_classes(rng, shift=1.0)
    x, y = np.concatenate([e, p]), np.concatenate([np.ones(len(e), int), np.zeros(len(p), int)])
    spread = cf.repeated_cross_fit(x, y, seeds=range(4))
    assert len(spread.auc) == 4 and spread.minimum <= spread.mean <= spread.maximum
    assert 0.0 < spread.sd < 0.02 and spread.mean > 0.7


def test_the_same_class_null_passes_for_one_distribution_and_fails_for_a_shifted_one():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(1500, 4)), rng.normal(size=(1500, 4))
    assert cf.same_class_null(a, b)["within_tolerance"]
    assert not cf.same_class_null(a, b + np.array([1.0, 0, 0, 0]))["within_tolerance"]


def test_roles_that_overlap_are_refused_before_any_fitting():
    rng = np.random.default_rng(2)
    e, p = two_classes(rng, 300, 300)
    x = np.concatenate([e, p])
    y = np.concatenate([np.ones(300, int), np.zeros(300, int)])
    keys = {"train": np.arange(0, 100), "calibration": np.arange(90, 150), "evaluation": np.arange(150, 300)}
    with pytest.raises(rl.RolesOverlapError):
        cf.fit_calibrate_evaluate(x, y, e[:100], e[100:], p, 0.9, role_keys=keys)


def test_fit_calibrate_evaluate_gives_counts_from_events_the_model_never_saw():
    rng = np.random.default_rng(3)
    e, p = two_classes(rng, 4000, 4000, shift=3.0)
    train_x = np.concatenate([e[:1500], p[:1500]])
    train_y = np.concatenate([np.ones(1500, int), np.zeros(1500, int)])
    result = cf.fit_calibrate_evaluate(train_x, train_y, e[1500:2500], e[2500:], p[1500:], 0.9)
    assert result.rejection.n == 2500 and result.n_calibration_electrons == 1000
    assert result.achieved_efficiency == pytest.approx(0.9, abs=0.03)
    assert result.rejection.k < 250  # a clear separation: well under a tenth of the protons pass


def test_repeated_role_splits_carry_the_split_variance():
    rng = np.random.default_rng(4)
    e, p = two_classes(rng, 3000, 3000, shift=1.5)
    results = cf.repeated_role_splits(e, p, 0.9, n_repeats=5, seed=7)
    ks = [r.rejection.k for r in results]
    assert len(results) == 5 and len(set(ks)) > 1  # different splits, different counts
    assert all(r.rejection.n == 1800 for r in results)


def test_the_standard_baselines_expose_a_problem_that_energy_alone_solves():
    rng = np.random.default_rng(5)
    n = 2000
    energy = np.concatenate([rng.normal(5.0, 0.1, n), rng.normal(4.0, 0.5, n)])
    other = rng.normal(size=(2 * n, 2))
    features = np.column_stack([energy, other])
    y = np.concatenate([np.ones(n, int), np.zeros(n, int)])
    base = cf.standard_baselines(features, ["log_total_energy", "a", "b"], y)
    assert base["energy_rank_auc"] > 0.95 and base["logistic_all_features"] > 0.95 and base["energy_only_classifier"] > 0.95
