from math import radians
from pathlib import Path

import pytest

from ams_ecal.detector.geometry import load_geometry
from ams_ecal.detector.readout import (
    cell_index_for_layer_projection,
    coordinate_to_cell_index,
    measured_axis_for_fiber,
    project_track_to_cell_indices,
)
from ams_ecal.detector.tracking import TrackState

CONFIG_PATH = Path(__file__).parents[2] / "configs" / "geometry.yaml"


@pytest.mark.parametrize(
    ("fiber_axis", "expected_measured_axis"),
    [
        ("x", "y"),
        ("y", "x"),
    ],
)
def test_derives_measured_axis_perpendicular_to_fibers(
    fiber_axis: str,
    expected_measured_axis: str,
) -> None:
    assert (
        measured_axis_for_fiber(fiber_axis)
        == expected_measured_axis
    )


@pytest.mark.parametrize(
    ("fiber_axis", "error_type", "message"),
    [
        (1, TypeError, "must be a string"),
        ("z", ValueError, "either 'x' or 'y'"),
    ],
)
def test_rejects_invalid_fiber_axis(
    fiber_axis: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        measured_axis_for_fiber(fiber_axis)


@pytest.mark.parametrize(
    ("coordinate_mm", "expected_index"),
    [
        (-324.0, 0),
        (-315.0, 1),
        (-0.001, 35),
        (0.0, 36),
        (8.999, 36),
        (9.0, 37),
        (323.999, 71),
    ],
)
def test_maps_coordinate_to_half_open_cell_grid(
    coordinate_mm: float,
    expected_index: int,
) -> None:
    geometry = load_geometry(CONFIG_PATH)

    assert (
        coordinate_to_cell_index(coordinate_mm, geometry)
        == expected_index
    )


@pytest.mark.parametrize(
    "coordinate_mm",
    [
        -324.001,
        324.0,
        400.0,
    ],
)
def test_returns_none_outside_active_cell_grid(
    coordinate_mm: float,
) -> None:
    geometry = load_geometry(CONFIG_PATH)

    assert coordinate_to_cell_index(coordinate_mm, geometry) is None


@pytest.mark.parametrize(
    ("coordinate_mm", "error_type", "message"),
    [
        (True, TypeError, "must be a real number"),
        ("0.0", TypeError, "must be a real number"),
        (float("nan"), ValueError, "must be finite"),
        (float("inf"), ValueError, "must be finite"),
        (float("-inf"), ValueError, "must be finite"),
    ],
)
def test_rejects_invalid_cell_coordinate(
    coordinate_mm: object,
    error_type: type[Exception],
    message: str,
) -> None:
    geometry = load_geometry(CONFIG_PATH)

    with pytest.raises(error_type, match=message):
        coordinate_to_cell_index(coordinate_mm, geometry)


def test_selects_coordinate_measured_by_each_layer() -> None:
    geometry = load_geometry(CONFIG_PATH)

    # Layer 0 has X-directed fibers, so it measures y.
    assert cell_index_for_layer_projection(
        projected_x_mm=22.4,
        projected_y_mm=-17.2,
        layer_index=0,
        geometry=geometry,
    ) == 34

    # Layer 2 has Y-directed fibers, so it measures x.
    assert cell_index_for_layer_projection(
        projected_x_mm=22.4,
        projected_y_mm=-17.2,
        layer_index=2,
        geometry=geometry,
    ) == 38


@pytest.mark.parametrize(
    ("layer_index", "error_type", "message"),
    [
        (True, TypeError, "must be an integer"),
        (1.5, TypeError, "must be an integer"),
        (-1, IndexError, "existing readout layer"),
        (18, IndexError, "existing readout layer"),
    ],
)
def test_rejects_invalid_layer_index(
    layer_index: object,
    error_type: type[Exception],
    message: str,
) -> None:
    geometry = load_geometry(CONFIG_PATH)

    with pytest.raises(error_type, match=message):
        cell_index_for_layer_projection(
            projected_x_mm=0.0,
            projected_y_mm=0.0,
            layer_index=layer_index,
            geometry=geometry,
        )


@pytest.mark.parametrize(
    ("projected_x_mm", "projected_y_mm", "error_type", "message"),
    [
        (
            True,
            0.0,
            TypeError,
            "projected_x_mm must be a real number",
        ),
        (
            0.0,
            "0.0",
            TypeError,
            "projected_y_mm must be a real number",
        ),
        (
            float("nan"),
            0.0,
            ValueError,
            "projected_x_mm must be finite",
        ),
        (
            0.0,
            float("inf"),
            ValueError,
            "projected_y_mm must be finite",
        ),
    ],
)
def test_rejects_invalid_projected_coordinates(
    projected_x_mm: object,
    projected_y_mm: object,
    error_type: type[Exception],
    message: str,
) -> None:
    geometry = load_geometry(CONFIG_PATH)

    with pytest.raises(error_type, match=message):
        cell_index_for_layer_projection(
            projected_x_mm=projected_x_mm,
            projected_y_mm=projected_y_mm,
            layer_index=0,
            geometry=geometry,
        )


def test_projects_vertical_track_to_alternating_cells() -> None:
    geometry = load_geometry(CONFIG_PATH)
    track = TrackState(
        x0_mm=22.4,
        y0_mm=-17.2,
        z0_mm=0.0,
        theta_rad=0.0,
        phi_rad=0.0,
    )

    cell_indices = project_track_to_cell_indices(track, geometry)

    assert cell_indices == (
        34,
        34,
        38,
        38,
        34,
        34,
        38,
        38,
        34,
        34,
        38,
        38,
        34,
        34,
        38,
        38,
        34,
        34,
    )


def test_projects_inclined_track_across_cell_boundaries() -> None:
    geometry = load_geometry(CONFIG_PATH)
    track = TrackState(
        x0_mm=10.0,
        y0_mm=-20.0,
        z0_mm=0.0,
        theta_rad=radians(10.0),
        phi_rad=radians(30.0),
    )

    cell_indices = project_track_to_cell_indices(track, geometry)

    assert cell_indices == (
        33,
        33,
        37,
        37,
        34,
        34,
        38,
        38,
        34,
        34,
        38,
        38,
        34,
        35,
        39,
        39,
        35,
        35,
    )


def test_preserves_layers_where_track_leaves_active_grid() -> None:
    geometry = load_geometry(CONFIG_PATH)
    track = TrackState(
        x0_mm=320.0,
        y0_mm=0.0,
        z0_mm=0.0,
        theta_rad=radians(10.0),
        phi_rad=0.0,
    )

    cell_indices = project_track_to_cell_indices(track, geometry)

    assert cell_indices == (
        36,
        36,
        None,
        None,
        36,
        36,
        None,
        None,
        36,
        36,
        None,
        None,
        36,
        36,
        None,
        None,
        36,
        36,
    )