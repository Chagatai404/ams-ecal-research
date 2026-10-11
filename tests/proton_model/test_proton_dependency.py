"""Proton model the dependency analysis (step 0) dependency diagnostics, on synthetic data with known answers."""

import json

import numpy as np
import pytest

from ams_ecal.geant4_simulation.geant4_backend import OUTPUT_SCHEMA_VERSION
from ams_ecal.proton_model.proton_calibration import validation_mask
from ams_ecal.proton_model.proton_dependency import (
    amplitude,
    correlation_interval,
    equal_count_bins,
    first_component_share,
    interaction_layer,
    layer_offset_matrix,
    load_calibration,
    partial_spearman,
    residual_within_cells,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260929)


# --- partial rank correlation ------------------------------------------


def test_a_shared_driver_creates_a_correlation_that_conditioning_removes(rng) -> None:
    r = rng.uniform(0, 166, 4000)
    u = r + rng.normal(0, 15, 4000)
    v = r + rng.normal(0, 15, 4000)

    assert partial_spearman(u, v) > 0.7
    assert abs(partial_spearman(u, v, [(r, 20)])) < 0.1


def test_a_genuine_dependence_survives_conditioning(rng) -> None:
    r = rng.uniform(0, 166, 4000)
    u = rng.normal(size=4000)
    v = u + rng.normal(0, 0.5, 4000)

    assert partial_spearman(u, v, [(r, 10)]) > 0.8


def test_the_partial_correlation_is_unchanged_by_a_monotone_transformation(rng) -> None:
    r = rng.uniform(0, 1, 3000)
    u = r + rng.normal(0, 0.2, 3000)
    v = r + rng.normal(0, 0.2, 3000)

    assert partial_spearman(np.exp(3 * u), v**3 + 10, [(r, 8)]) == pytest.approx(
        partial_spearman(u, v, [(r, 8)]), abs=1e-9
    )


def test_two_conditioning_variables_remove_more_than_one(rng) -> None:
    a = rng.uniform(size=4000)
    b = rng.uniform(size=4000)
    u = a + b + rng.normal(0, 0.1, 4000)
    v = a + b + rng.normal(0, 0.1, 4000)

    given_a = partial_spearman(u, v, [(a, 5)])
    given_a_and_b = partial_spearman(u, v, [(a, 5), (b, 5)])

    assert given_a_and_b < given_a
    # what is left is the variation of a and b INSIDE each 0.2 x 0.2 cell
    assert given_a_and_b == pytest.approx(0.4, abs=0.1)


def test_equal_count_bins_have_equal_populations(rng) -> None:
    counts = np.bincount(equal_count_bins(rng.normal(size=1000), 10), minlength=10)

    assert counts.tolist() == [100] * 10


def test_residuals_have_zero_mean_inside_every_cell(rng) -> None:
    cells = rng.integers(0, 5, 500)
    residual = residual_within_cells(rng.normal(size=500) + cells, cells)

    for cell in range(5):
        assert residual[cells == cell].mean() == pytest.approx(0.0, abs=1e-12)


def test_the_correlation_interval_brackets_the_estimate_and_narrows_with_n() -> None:
    low_small, high_small = correlation_interval(0.3, 100)
    low_large, high_large = correlation_interval(0.3, 10000)

    assert low_small < 0.3 < high_small
    assert (high_large - low_large) < (high_small - low_small)
    with pytest.raises(ValueError):
        correlation_interval(0.3, 3)


# --- layer bookkeeping -------------------------------------------------


def test_the_interaction_layer_follows_the_half_open_layer_rule() -> None:
    thickness = 166.5 / 18
    depth = np.array([0.0, thickness - 1e-6, thickness, 166.49])

    assert interaction_layer(depth, thickness).tolist() == [0, 0, 1, 17]


def test_layer_offsets_address_layers_relative_to_the_interaction_layer() -> None:
    energy = np.arange(18, dtype=float)[None, :] * np.ones((2, 1))
    first = np.array([3, 16])

    matrix = layer_offset_matrix(energy, first, [-4, -1, 0, 1, 3])

    assert np.isnan(matrix[0, 0])  # layer -1 does not exist
    assert matrix[0, 1:].tolist() == [2.0, 3.0, 4.0, 6.0]
    assert matrix[1, :4].tolist() == [12.0, 15.0, 16.0, 17.0]
    assert np.isnan(matrix[1, 4])  # layer 19 does not exist


def test_the_amplitude_is_the_sum_of_the_three_following_layers() -> None:
    energy = np.arange(18, dtype=float)[None, :] * np.ones((3, 1))

    a3 = amplitude(energy, np.array([0, 5, 15]))

    assert a3[0] == 1 + 2 + 3
    assert a3[1] == 6 + 7 + 8
    assert np.isnan(a3[2])


def test_a_rank_one_matrix_has_all_its_variance_in_one_component(rng) -> None:
    amplitude_factor = rng.normal(size=(500, 1))
    matrix = amplitude_factor * np.linspace(1, 3, 8)[None, :]

    assert first_component_share(matrix) == pytest.approx(1.0, abs=1e-9)


def test_independent_columns_share_the_variance_roughly_evenly(rng) -> None:
    share = first_component_share(rng.normal(size=(20000, 8)))

    assert 0.12 < share < 0.16


# --- the calibration-only loader ---------------------------------------


def test_no_validation_event_can_enter_the_calibration_arrays(tmp_path) -> None:
    n = 40
    directory = tmp_path / "baseline" / "E10GeV"
    directory.mkdir(parents=True)
    index = np.arange(n)
    np.savez(
        directory / "events.npz",
        event_index=index,
        truth_occurred=index % 2 == 0,
        first_secondary_offsets=np.zeros(n + 1, dtype=np.int64),
    )
    (directory / "metadata.json").write_text(
        json.dumps({"output_schema_version": OUTPUT_SCHEMA_VERSION}), encoding="utf-8"
    )

    arrays = load_calibration("baseline", 10.0, tmp_path)

    assert not validation_mask(arrays["event_index"]).any()
    assert len(arrays["event_index"]) == 30
    assert arrays["truth_occurred"].shape == (30,)
    # the sparse offsets array has n + 1 entries and is not an event array
    assert "first_secondary_offsets" not in arrays
    assert arrays["_kept_index"].tolist() == [i for i in range(n) if i % 4 != 3]
