from pathlib import Path

import numpy as np
import pytest

from ams_ecal.detector.geometry import load_geometry
from ams_ecal.detector.projection import (
    cell_indices,
    layer_indices,
    measured_axis_for_layer,
    project_deposits,
)
from ams_ecal.detector.readout import coordinate_to_cell_index, measured_axis_for_fiber

GEOMETRY = Path(__file__).parents[2] / "configs" / "geometry.yaml"


@pytest.fixture
def geometry():
    return load_geometry(GEOMETRY)


def test_vectorized_cells_agree_with_the_scalar_readout(geometry) -> None:
    boundaries = -324.0 + 9.0 * np.arange(73)
    points = np.concatenate(
        [
            boundaries,
            np.nextafter(boundaries, -np.inf),
            np.nextafter(boundaries, np.inf),
            np.random.default_rng(1).uniform(-330.0, 330.0, 5000),
        ]
    )
    vectorized = cell_indices(points, geometry)
    for point, cell in zip(points, vectorized, strict=True):
        scalar = coordinate_to_cell_index(float(point), geometry)
        assert cell == (-1 if scalar is None else scalar)


def test_layers_are_half_open_in_depth(geometry) -> None:
    t = geometry.mean_readout_slice_thickness_mm
    z = np.array([0.0, t - 1e-9, t, 17 * t, 18 * t - 1e-9, 18 * t, -1e-9])
    assert layer_indices(z, geometry, 18).tolist() == [0, 0, 1, 17, 17, -1, -1]


def test_measured_axis_matches_the_readout_and_keeps_alternating(geometry) -> None:
    for layer in range(geometry.number_of_layers):
        assert measured_axis_for_layer(layer, geometry) == measured_axis_for_fiber(
            geometry.layer_fiber_axes[layer]
        )
    # Superlayer 9 (layers 18, 19) continues the x/y alternation beyond the ECAL.
    assert [measured_axis_for_layer(l, geometry) for l in (16, 17, 18, 19, 20)] == [
        "y",
        "y",
        "x",
        "x",
        "y",
    ]


def test_a_hit_lands_in_the_expected_layer_and_cell(geometry) -> None:
    # Layer 0 measures y, layer 2 measures x.
    result = project_deposits(
        np.array([100.0, 100.0]),
        np.array([-50.0, -50.0]),
        np.array([1.0, 2 * 9.25 + 1.0]),
        np.array([3.0, 5.0]),
        geometry,
    )
    grid = result.grid_mev
    assert grid[0, coordinate_to_cell_index(-50.0, geometry)] == 3.0
    assert grid[2, coordinate_to_cell_index(100.0, geometry)] == 5.0
    assert grid.sum() == 8.0 and result.outside_mev == 0.0


def test_energy_is_conserved_and_the_outside_is_reported(geometry) -> None:
    rng = np.random.default_rng(7)
    n = 2000
    x, y = rng.uniform(-400, 400, n), rng.uniform(-400, 400, n)
    z = rng.uniform(-20, 200, n)
    e = rng.exponential(1.0, n)
    result = project_deposits(x, y, z, e, geometry)
    inside = (np.abs(x) < 324) & (np.abs(y) < 324) & (z >= 0) & (z < 166.5)
    # A deposit is lost only when outside its own layer's measured grid, so
    # the conserved total is exact regardless.
    assert result.grid_mev.sum() + result.outside_mev == pytest.approx(e.sum())
    assert result.grid_mev.sum() >= e[inside].sum() - 1e-9


def test_extended_layers_continue_the_same_thickness(geometry) -> None:
    result = project_deposits(
        np.array([0.0]), np.array([0.0]), np.array([19.5 * 9.25]), np.array([1.0]),
        geometry, n_layers=270,
    )
    assert result.grid_mev.shape == (270, 72)
    assert result.grid_mev[19].sum() == 1.0


def test_rejects_negative_energy(geometry) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        project_deposits(
            np.zeros(1), np.zeros(1), np.ones(1), np.array([-1.0]), geometry
        )
