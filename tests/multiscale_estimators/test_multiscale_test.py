"""Registered statistics of the first multiscale test: eligibility, matching, the split-half floor and the verdict rule."""

import numpy as np

from ams_ecal.multiscale_estimators import multiscale_test as mt


def dense_image(rng: np.random.Generator, cells: int) -> np.ndarray:
    image = np.zeros(18 * 72)
    image[rng.choice(18 * 72, cells, replace=False)] = rng.exponential(1.0, cells)
    return image.reshape(18, 72)


def test_an_event_below_the_cell_threshold_is_ineligible_not_dropped() -> None:
    rng = np.random.default_rng(1)

    few, many = mt.event_multiscale(dense_image(rng, 100)), mt.event_multiscale(dense_image(rng, 700))

    assert np.isnan(few).all()
    assert np.isfinite(many).all()
    assert len(many) == len(mt.feature_names()) == len(mt.REGISTERED_WINDOWS) * len(mt.STAT_NAMES)


def test_windows_with_the_higher_threshold_need_more_cells() -> None:
    rng = np.random.default_rng(2)

    middle = dict(zip(mt.feature_names(), mt.event_multiscale(dense_image(rng, 400)), strict=True))

    assert np.isfinite(middle["D0@2-3-6-9"])  # 336-cell window
    assert np.isnan(middle["D0@2-3"])  # 572-cell window


def test_ks_is_not_defined_for_tiny_samples() -> None:
    assert np.isnan(mt.ks_distance(np.arange(5.0), np.arange(5.0)))


def test_matching_removes_a_difference_that_only_the_matching_variable_carries() -> None:
    rng = np.random.default_rng(3)
    key_a, key_b = rng.normal(0, 1, 4000), rng.normal(1.5, 1, 4000)
    a, b = key_a + rng.normal(0, 0.2, 4000), key_b + rng.normal(0, 0.2, 4000)  # the statistic is the key plus noise

    assert mt.ks_distance(a, b) > 0.5
    assert mt.matched_ks(a, b, key_a, key_b) < 0.15


def test_a_real_difference_at_fixed_key_survives_matching() -> None:
    rng = np.random.default_rng(4)
    key = rng.normal(0, 1, 4000)

    assert mt.matched_ks(key + rng.normal(0, 0.2, 4000), key + 0.6 + rng.normal(0, 0.2, 4000), key, key) > 0.3


def test_the_split_half_floor_of_one_distribution_is_small() -> None:
    assert mt.split_half_floor(np.random.default_rng(5).normal(size=2000), np.random.default_rng(6)) < 0.08


def test_the_registered_rule_calls_a_shift_material_and_identical_classes_not() -> None:
    rng = np.random.default_rng(7)
    key = rng.normal(size=2000)
    same = mt.compare_feature(
        key + rng.normal(0, 0.3, 2000), key + rng.normal(0, 0.3, 2000), key, key, rng, bootstraps=40
    )
    shifted = mt.compare_feature(
        key + rng.normal(0, 0.3, 2000), key + 1.0 + rng.normal(0, 0.3, 2000), key, key, rng, bootstraps=40
    )

    assert same["material_by_the_registered_rule"] is False
    assert shifted["material_by_the_registered_rule"] is True


def test_conventional_features_of_a_single_cell_event() -> None:
    grids = np.zeros((1, 18, 72))
    grids[0, 4, 30] = 100.0

    names, table = mt.conventional_features(grids)
    row = dict(zip(names, table[0], strict=True))

    assert row["occupied_cells"] == 1
    assert row["layer_cog"] == 4
    assert row["max_cell_fraction"] == 1.0
    assert row["width_cells"] == 0.0
    assert row["core_fraction"] == 1.0
    assert table.shape == (1, len(names))
