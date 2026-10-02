"""Exact-answer tests for the multiscale estimators (estimator-validity gate, EXP-008)."""

from math import log

import numpy as np
import pytest

from ams_ecal import multiscale as ms

DYADIC = (1, 2, 4, 8, 16, 32)
SIERPINSKI_D = log(3) / log(2)


def _square():
    return np.ones((64, 64))


def _line():
    return np.eye(64)


def _sierpinski():
    i, j = np.indices((64, 64))
    return ((i & j) == 0).astype(float)


@pytest.mark.parametrize("q", ms.Q_VALUES)
@pytest.mark.parametrize("image, expected", [(_square, 2.0), (_line, 1.0), (_sierpinski, SIERPINSKI_D)])
def test_exact_patterns_recover_their_dimension_at_every_q(image, expected, q):
    assert ms.fitted_dimension(image(), q, DYADIC) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("q", ms.Q_VALUES)
def test_filled_ams_grid_has_dimension_two_on_the_ams_box_sides(q):
    assert ms.fitted_dimension(np.ones((18, 72)), q, ms.AMS_BOX_SIDES) == pytest.approx(2.0, abs=1e-9)


def test_partition_values_by_hand():
    p = np.array([0.5, 0.5, 0.0])
    assert ms.partition_value(p, 0) == pytest.approx(log(2))
    assert ms.partition_value(p, 1) == pytest.approx(log(2))
    assert ms.partition_value(p, 2) == pytest.approx(log(0.5))


def test_local_dimension_matches_the_closed_form_for_a_two_level_cascade():
    w = np.outer([0.7, 0.3], [0.7, 0.3])
    image = np.kron(w, w)  # 4 x 4, exactly self-similar over one level
    expected_d2 = -log(float(np.sum(w ** 2))) / log(2)
    assert ms.local_dimension(image, 2, 1, 2) == pytest.approx(expected_d2, abs=1e-12)
    expected_d1 = float(-np.sum(w * np.log(w))) / log(2)
    assert ms.local_dimension(image, 1, 1, 2) == pytest.approx(expected_d1, abs=1e-12)
    assert ms.local_dimension(image, 0, 1, 2) == pytest.approx(2.0, abs=1e-12)


def test_dimensions_are_ordered_for_an_uneven_pattern():
    w = np.outer([0.7, 0.3], [0.7, 0.3])
    image = np.kron(np.kron(w, w), w)
    d = [ms.fitted_dimension(image, q, (1, 2, 4)) for q in ms.Q_VALUES]
    assert d[0] >= d[1] >= d[2]
    assert d[0] - d[2] > 0.1


def test_energy_scale_does_not_change_the_estimate():
    image = _sierpinski()
    assert ms.fitted_dimension(image * 37.5, 2, DYADIC) == pytest.approx(ms.fitted_dimension(image, 2, DYADIC))


@pytest.mark.parametrize("bad", [np.ones(5), -np.ones((4, 4)), np.zeros((4, 4)), np.full((4, 4), np.nan)])
def test_invalid_images_are_rejected(bad):
    with pytest.raises(ValueError):
        ms.box_shares(bad, 1)


def test_a_box_side_that_does_not_divide_the_image_is_rejected():
    with pytest.raises(ValueError):
        ms.box_shares(np.ones((18, 72)), 4)


def test_fit_needs_two_scales_and_ordered_local_sides():
    with pytest.raises(ValueError):
        ms.fitted_dimension(np.ones((18, 72)), 0, (3,))
    with pytest.raises(ValueError):
        ms.local_dimension(np.ones((18, 72)), 0, 3, 3)


def test_ams_box_sides_divide_the_ams_grid_and_no_larger_square_side_does():
    assert all(18 % s == 0 and 72 % s == 0 for s in ms.AMS_BOX_SIDES)
    assert [s for s in range(1, 19) if 18 % s == 0 and 72 % s == 0] == list(ms.AMS_BOX_SIDES)


def test_windows_are_all_runs_of_at_least_two_scales():
    windows = ms.contiguous_windows()
    assert len(windows) == 15 and (1, 2) in windows and ms.AMS_BOX_SIDES in windows
    assert all(len(w) >= 2 for w in windows)


def test_saturation_rule_marks_fine_windows_of_a_sparse_image_unreliable():
    shape = (18, 72)
    assert not ms.is_reliable(shape, (1, 2), 300)  # fine scales, 300 hits: saturated
    assert ms.is_reliable(shape, (6, 9), 300)  # coarse scales: not saturated
    assert ms.is_reliable(shape, (1, 2), 1296)  # every cell occupied: nothing saturates


def test_saturation_x_decreases_as_the_hit_count_grows():
    x = [ms.saturation_x((18, 72), (2, 3), n) for n in (30, 100, 300, 1000)]
    assert x == sorted(x, reverse=True)


def test_reliable_windows_of_a_filled_image_are_all_windows():
    assert len(ms.reliable_windows(np.ones((18, 72)))) == 15
    sparse = np.zeros((18, 72))
    sparse.flat[np.random.default_rng(0).choice(18 * 72, 100, replace=False)] = 1
    assert 0 < len(ms.reliable_windows(sparse)) < 15
