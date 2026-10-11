from dataclasses import FrozenInstanceError
from math import pi, radians, sqrt, tau

import pytest

from ams_ecal.detector.tracking import TrackState, project_track_to_z


def valid_track_values() -> dict[str, float]:
    return {
        "x0_mm": 10.0,
        "y0_mm": -20.0,
        "z0_mm": 0.0,
        "theta_rad": radians(10.0),
        "phi_rad": radians(30.0),
    }


def test_stores_reconstructed_track_state() -> None:
    track = TrackState(**valid_track_values())

    assert track.x0_mm == pytest.approx(10.0)
    assert track.y0_mm == pytest.approx(-20.0)
    assert track.z0_mm == pytest.approx(0.0)
    assert track.theta_rad == pytest.approx(radians(10.0))
    assert track.phi_rad == pytest.approx(radians(30.0))


def test_derives_unit_direction_vector() -> None:
    track = TrackState(**valid_track_values())

    direction_x, direction_y, direction_z = (
        track.direction_unit_vector
    )

    assert direction_x == pytest.approx(0.1503837332)
    assert direction_y == pytest.approx(0.0868240888)
    assert direction_z == pytest.approx(0.9848077530)

    magnitude = sqrt(
        direction_x**2 + direction_y**2 + direction_z**2
    )

    assert magnitude == pytest.approx(1.0)


def test_derives_transverse_slopes_wrt_z() -> None:
    track = TrackState(**valid_track_values())

    slope_x, slope_y = track.slopes_wrt_z

    assert slope_x == pytest.approx(0.1527036447)
    assert slope_y == pytest.approx(0.0881634904)


def test_vertical_track_has_zero_transverse_slopes() -> None:
    track = TrackState(
        x0_mm=10.0,
        y0_mm=-20.0,
        z0_mm=0.0,
        theta_rad=0.0,
        phi_rad=radians(120.0),
    )

    assert track.direction_unit_vector == pytest.approx(
        (0.0, 0.0, 1.0)
    )
    assert track.slopes_wrt_z == pytest.approx((0.0, 0.0))


def test_track_state_is_immutable() -> None:
    track = TrackState(**valid_track_values())

    with pytest.raises(FrozenInstanceError):
        track.x0_mm = 25.0


@pytest.mark.parametrize(
    ("field", "value", "error_type", "message"),
    [
        ("x0_mm", float("nan"), ValueError, "must be finite"),
        ("y0_mm", float("inf"), ValueError, "must be finite"),
        ("z0_mm", True, TypeError, "must be a real number"),
        ("theta_rad", -0.01, ValueError, "0 <= theta_rad"),
        ("theta_rad", pi / 2, ValueError, "0 <= theta_rad"),
        ("phi_rad", -0.01, ValueError, "0 <= phi_rad"),
        ("phi_rad", tau, ValueError, "0 <= phi_rad"),
    ],
)
def test_rejects_invalid_track_values(
    field: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    values = valid_track_values()
    values[field] = value

    with pytest.raises(error_type, match=message):
        TrackState(**values)

def test_projects_inclined_track_to_target_z() -> None:
    track = TrackState(**valid_track_values())

    projected_point = project_track_to_z(track, target_z_mm=4.625)

    assert projected_point == pytest.approx(
        (
            10.7062543566,
            -19.5922438571,
            4.625,
        )
    )


def test_projects_vertical_track_without_transverse_motion() -> None:
    track = TrackState(
        x0_mm=10.0,
        y0_mm=-20.0,
        z0_mm=0.0,
        theta_rad=0.0,
        phi_rad=radians(120.0),
    )

    projected_point = project_track_to_z(
        track,
        target_z_mm=161.875,
    )

    assert projected_point == pytest.approx(
        (10.0, -20.0, 161.875)
    )


def test_projects_from_an_upstream_reference_plane() -> None:
    track = TrackState(
        x0_mm=10.0,
        y0_mm=-20.0,
        z0_mm=-100.0,
        theta_rad=radians(10.0),
        phi_rad=radians(30.0),
    )

    projected_point = project_track_to_z(
        track,
        target_z_mm=0.0,
    )

    assert projected_point == pytest.approx(
        (
            25.2703644666,
            -11.1836509646,
            0.0,
        )
    )


def test_allows_backward_straight_line_extrapolation() -> None:
    track = TrackState(**valid_track_values())

    projected_point = project_track_to_z(
        track,
        target_z_mm=-10.0,
    )

    assert projected_point == pytest.approx(
        (
            8.4729635533,
            -20.8816349035,
            -10.0,
        )
    )


def test_rejects_boolean_target_z() -> None:
    track = TrackState(**valid_track_values())

    with pytest.raises(TypeError, match="must be a real number"):
        project_track_to_z(track, target_z_mm=True)


@pytest.mark.parametrize(
    "target_z_mm",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_rejects_nonfinite_target_z(
    target_z_mm: float,
) -> None:
    track = TrackState(**valid_track_values())

    with pytest.raises(ValueError, match="must be finite"):
        project_track_to_z(track, target_z_mm)