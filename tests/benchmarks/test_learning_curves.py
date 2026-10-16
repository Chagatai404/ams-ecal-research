import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ams_ecal.benchmarks import learning_curves as lc
from ams_ecal.inference import rejection_limits as rl


def test_the_balanced_log_loss_is_zero_for_certain_correct_scores_and_ln2_for_coin_flips():
    assert lc.balanced_log_loss(np.ones(50), np.zeros(80)) < 1e-9
    assert lc.balanced_log_loss(np.full(50, 0.5), np.full(80, 0.5)) == pytest.approx(
        np.log(2)
    )
    # class balance: many protons that are right cannot hide wrong electrons
    assert lc.balanced_log_loss(np.full(10, 0.01), np.zeros(10_000)) > 2.0


def test_metrics_counts_match_the_threshold_taken_from_the_calibration_electrons_only():
    rng = np.random.default_rng(0)
    cal, ev_e = rng.uniform(0.5, 1, 2000), rng.uniform(0.5, 1, 2000)
    ev_p = np.r_[rng.uniform(0, 0.4, 3980), np.full(20, 0.95)]
    m = lc.metrics(cal, ev_e, ev_p)
    r90 = m["efficiency_0.9"]
    assert r90["protons"] == 4000 and r90["protons_passing"] == 20
    assert 0.85 < r90["achieved_efficiency"] < 0.95
    assert r90["rejection_lower_limit"] == pytest.approx(
        rl.rejection_from_counts(20, 4000).rejection_lower_limit
    )
    assert m["auc"] > 0.99 and not r90["censored"]


def test_roles_are_disjoint_and_have_the_registered_sizes():
    roles = lc.split_roles(10_000, 20_000, seed=3)
    assert (
        len(roles["calibration_electrons"]) == lc.N_CALIBRATION
        and len(roles["evaluation_protons"]) == lc.N_EVAL_PROTONS
    )
    keys = np.concatenate(list(roles.values()))
    assert len(keys) == len(set(keys.tolist())) == 10_000 + 20_000


def test_boosted_trees_and_logistic_separate_a_signal_and_drop_all_nan_columns():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(600, 4))
    y = (rng.random(600) < 0.5).astype(float)
    x[:, 0] += 2.0 * y
    x[:, 3] = np.nan  # a column that carries nothing
    for kind in ("hgb", "logistic"):
        (p,) = lc.fit_predict_tabular(kind, x[:400], y[:400], [x[400:]])
        assert roc_auc_score(y[400:], p) > 0.9 and p.min() >= 0 and p.max() <= 1


def test_the_tabular_models_have_auc_one_half_when_there_is_nothing_to_learn():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2000, 5))
    y = (rng.random(2000) < 0.5).astype(float)
    (p,) = lc.fit_predict_tabular("hgb", x[:1000], y[:1000], [x[1000:]])
    assert abs(roc_auc_score(y[1000:], p) - 0.5) < 0.06


def test_the_network_separates_compact_from_broad_images_and_is_reproducible_on_the_cpu():
    torch = pytest.importorskip("torch")
    assert torch is not None
    rng = np.random.default_rng(3)

    def images(n, width):
        centre = np.exp(-0.5 * ((np.arange(72) - 36) / width) ** 2)
        return (rng.random((n, 18, 72)) < 0.1 + 0.8 * centre) * rng.exponential(
            2.0, (n, 18, 72)
        )

    x = np.concatenate([images(150, 4.0), images(150, 14.0)]).astype(np.float32)
    y = np.r_[np.ones(150), np.zeros(150)]
    test = np.concatenate([images(100, 4.0), images(100, 14.0)]).astype(np.float32)
    (p1,), cost = lc.fit_predict_cnn(x, y, [test], seed=5, device="cpu")
    (p2,), _ = lc.fit_predict_cnn(x, y, [test], seed=5, device="cpu")
    assert roc_auc_score(np.r_[np.ones(100), np.zeros(100)], p1) > 0.95
    assert np.allclose(p1, p2, atol=1e-5) and cost["seconds"] > 0


def test_paired_differences_report_the_spread_over_repeats_per_rung_and_size():
    def point(n, r, with_ms, without):
        models = {rung: {"log_loss": without + 0.01 * r} for rung in lc.RUNGS}
        models.update(
            {f"{rung}+ms": {"log_loss": with_ms + 0.01 * r} for rung in lc.RUNGS[1:]}
        )
        return {"size_per_class": n, "repeat": r, "models": models}

    points = [point(100, r, 0.30, 0.35) for r in range(4)]
    d = lc.paired_differences(points)["conv|100"]
    assert (
        d["mean"] == pytest.approx(-0.05)
        and d["sd_over_repeats"] == pytest.approx(0.0, abs=1e-12)
        and d["repeats"] == 4
    )
