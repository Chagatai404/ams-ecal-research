"""The information ladder: rates at fixed efficiency, the paired gain rule and a small end-to-end run."""

import numpy as np

from ams_ecal.multiscale_estimators import multiscale_ladder as ml


def test_false_positive_rate_at_a_fixed_electron_efficiency() -> None:
    labels = np.array([1] * 10 + [0] * 10)
    scores = np.concatenate([np.linspace(0.5, 1.0, 10), np.linspace(0.0, 0.7, 10)])

    fpr = ml.false_positive_rate(scores, labels, 0.9)

    threshold = np.quantile(scores[labels == 1], 0.1)
    assert fpr == (scores[labels == 0] >= threshold).mean()
    assert 0.0 <= fpr <= 1.0


def test_the_paired_gain_rule_needs_both_a_size_and_an_interval_above_zero() -> None:
    rng = np.random.default_rng(1)
    labels = np.array([1] * 400 + [0] * 400)
    noise = rng.normal(size=800)
    weak = labels * 0.5 + noise
    strong = labels * 1.5 + noise

    better = ml.paired_gain(weak, strong, labels, rng, 60)
    same = ml.paired_gain(weak, weak + rng.normal(0, 1e-6, 800), labels, rng, 60)

    assert better["adds_information_by_the_registered_rule"] is True
    assert same["adds_information_by_the_registered_rule"] is False
    assert same["gain_95"][0] <= 0.0 <= same["gain_95"][1] or abs(same["gain"]) < 0.01


def structured_grids(rng: np.random.Generator, n: int, spread: float) -> np.ndarray:
    grids = np.zeros((n, 18, 72))
    for i in range(n):
        cells = rng.integers(0, 72, size=(18, 40))
        centre = 36 + rng.normal(0, spread, size=(18, 1))
        cells = np.clip(np.rint(centre + rng.normal(0, 6 * spread, size=(18, 40))), 0, 71).astype(int)
        for layer in range(18):
            np.add.at(grids[i, layer], cells[layer], rng.exponential(1.0, 40))
    return grids


def test_the_ladder_separates_two_synthetic_classes_and_reports_every_rung() -> None:
    rng = np.random.default_rng(2)
    a, b = structured_grids(rng, 70, 1.0), structured_grids(rng, 70, 3.0)

    result = ml.ladder(a, b, sizes=(30,), bootstraps=10, rungs={k: v for k, v in ml.RUNGS.items() if "raw" not in v})

    assert set(result["rungs"]) == {"conventional", "multiscale", "conventional+multiscale"}
    assert result["rungs"]["conventional"]["auc"] > 0.8
    assert result["rungs"]["conventional"]["rejection_90"] >= 1.0
    assert "gain_conventional_plus_multiscale_over_conventional" in result
    assert set(result["sample_efficiency_auc"]["conventional"]) == {"30"}


def test_a_feature_column_that_is_nan_for_every_training_event_does_not_break_the_classifier() -> None:
    labels = np.array([1] * 40 + [0] * 40)
    x = np.column_stack([labels + np.random.default_rng(3).normal(0, 0.5, 80), np.full(80, np.nan)])

    scores = ml.out_of_fold_scores(x, labels, None)

    assert np.isfinite(scores).all()
    assert ml.roc_auc_score(labels, scores) > 0.8
    assert (ml.out_of_fold_scores(np.full((80, 2), np.nan), labels, None) == 0.5).all()
