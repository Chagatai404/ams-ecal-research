from math import floor, isfinite
from typing import Literal

from ams_ecal.detector.geometry import ECALGeometry, FiberAxis
from ams_ecal.detector.tracking import TrackState, project_track_to_z

MeasuredAxis = Literal["x", "y"]


def measured_axis_for_fiber(
    fiber_axis: FiberAxis,
) -> MeasuredAxis:
    """Return the coordinate measured across a set of parallel fibers."""

    if not isinstance(fiber_axis, str):
        raise TypeError("fiber_axis must be a string")

    if fiber_axis == "x":
        return "y"

    if fiber_axis == "y":
        return "x"

    raise ValueError("fiber_axis must be either 'x' or 'y'")


def coordinate_to_cell_index(
    coordinate_mm: float,
    geometry: ECALGeometry,
) -> int | None:
    """Map one transverse coordinate to a zero-based ECAL cell index.

    The cell grid uses half-open bounds. The lower boundary is included,
    while the upper boundary is excluded. Coordinates outside the active
    grid return None rather than being clamped to an edge cell.
    """

    if isinstance(coordinate_mm, bool) or not isinstance(
        coordinate_mm,
        int | float,
    ):
        raise TypeError("coordinate_mm must be a real number")

    if not isfinite(coordinate_mm):
        raise ValueError("coordinate_mm must be finite")

    coordinate_mm = float(coordinate_mm)

    grid_width_mm = (
        geometry.cells_per_layer * geometry.cell_pitch_mm
    )
    lower_bound_mm = -grid_width_mm / 2
    upper_bound_mm = grid_width_mm / 2

    if not lower_bound_mm <= coordinate_mm < upper_bound_mm:
        return None

    return floor(
        (coordinate_mm - lower_bound_mm)
        / geometry.cell_pitch_mm
    )


def cell_index_for_layer_projection(
    projected_x_mm: float,
    projected_y_mm: float,
    layer_index: int,
    geometry: ECALGeometry,
) -> int | None:
    """Return the cell crossed by a projected track in one readout layer."""

    if isinstance(layer_index, bool) or not isinstance(layer_index, int):
        raise TypeError("layer_index must be an integer")

    if not 0 <= layer_index < geometry.number_of_layers:
        raise IndexError(
            "layer_index must identify an existing readout layer"
        )

    for name, value in {
        "projected_x_mm": projected_x_mm,
        "projected_y_mm": projected_y_mm,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{name} must be a real number")

        if not isfinite(value):
            raise ValueError(f"{name} must be finite")

    fiber_axis = geometry.layer_fiber_axes[layer_index]
    measured_axis = measured_axis_for_fiber(fiber_axis)

    if measured_axis == "x":
        measured_coordinate_mm = projected_x_mm
    else:
        measured_coordinate_mm = projected_y_mm

    return coordinate_to_cell_index(
        measured_coordinate_mm,
        geometry,
    )


def project_track_to_cell_indices(
    track: TrackState,
    geometry: ECALGeometry,
) -> tuple[int | None, ...]:
    """Project a track to every ECAL layer and return its readout cells."""

    cell_indices: list[int | None] = []

    for layer_index, layer_z_mm in enumerate(
        geometry.uniform_layer_centers_z_mm
    ):
        projected_x_mm, projected_y_mm, _ = project_track_to_z(
            track,
            target_z_mm=layer_z_mm,
        )

        cell_index = cell_index_for_layer_projection(
            projected_x_mm=projected_x_mm,
            projected_y_mm=projected_y_mm,
            layer_index=layer_index,
            geometry=geometry,
        )

        cell_indices.append(cell_index)

    return tuple(cell_indices)