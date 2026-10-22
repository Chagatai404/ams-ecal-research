import numpy as np

from ams_ecal.detector.sampling_response_noise_floor import classify, half_split_floor


def test_floor_is_zero_for_a_constant_ratio():
    rng = np.random.default_rng(1)
    deposit = rng.uniform(1.0, 5.0, size=(200, 18))
    floor = half_split_floor(deposit, 0.7 * deposit, rng, n_splits=20)
    assert floor.max() < 1e-12


def test_floor_shrinks_with_more_events():
    rng = np.random.default_rng(2)

    def p95(n: int) -> float:
        deposit = rng.uniform(1.0, 5.0, size=(n, 18))
        fibre = deposit * rng.normal(1.0, 0.3, size=(n, 18))
        return float(np.quantile(half_split_floor(deposit, fibre, rng, n_splits=100), 0.95))

    assert p95(4000) < p95(200)


def test_classify_uses_the_registered_thresholds():
    assert classify(0.03) == "structurally_unmeetable"
    assert classify(0.02) == "partly_noise"
    assert classify(0.01) == "real_model_error"
