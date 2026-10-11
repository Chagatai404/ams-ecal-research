from dataclasses import dataclass
from math import cos, isfinite, pi, sin, tau


@dataclass(frozen=True, slots=True)
class TrackState:
    """A reconstructed incident track in the local ECAL coordinate system."""

    x0_mm: float
    y0_mm: float
    z0_mm: float
    theta_rad: float
    phi_rad: float

    def __post_init__(self) -> None:
        numeric_fields = {
            "x0_mm": self.x0_mm,
            "y0_mm": self.y0_mm,
            "z0_mm": self.z0_mm,
            "theta_rad": self.theta_rad,
            "phi_rad": self.phi_rad,
        }

        for name, value in numeric_fields.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be a real number")

            if not isfinite(value):
                raise ValueError(f"{name} must be finite")

        if not 0.0 <= self.theta_rad < pi / 2:
            raise ValueError(
                "theta_rad must satisfy 0 <= theta_rad < pi / 2 "
                "for a track directed into the ECAL"
            )

        if not 0.0 <= self.phi_rad < tau:
            raise ValueError(
                "phi_rad must satisfy 0 <= phi_rad < 2 * pi"
            )

    @property
    def direction_unit_vector(self) -> tuple[float, float, float]:
        """Return the dimensionless Cartesian direction components."""

        transverse_component = sin(self.theta_rad)

        return (
            transverse_component * cos(self.phi_rad),
            transverse_component * sin(self.phi_rad),
            cos(self.theta_rad),
        )

    @property
    def slopes_wrt_z(self) -> tuple[float, float]:
        """Return dx/dz and dy/dz for straight-line propagation."""

        direction_x, direction_y, direction_z = (
            self.direction_unit_vector
        )

        return (
            direction_x / direction_z,
            direction_y / direction_z,
        )

def project_track_to_z(
    track: TrackState,
    target_z_mm: float,
) -> tuple[float, float, float]:
    """Project a reconstructed track to a plane at a specified z-coordinate."""

    if isinstance(target_z_mm, bool) or not isinstance(
        target_z_mm,
        int | float,
    ):
        raise TypeError("target_z_mm must be a real number")

    if not isfinite(target_z_mm):
        raise ValueError("target_z_mm must be finite")

    target_z_mm = float(target_z_mm)
    delta_z_mm = target_z_mm - track.z0_mm
    slope_x, slope_y = track.slopes_wrt_z

    projected_x_mm = track.x0_mm + delta_z_mm * slope_x
    projected_y_mm = track.y0_mm + delta_z_mm * slope_y

    if not all(
        isfinite(value)
        for value in (projected_x_mm, projected_y_mm)
    ):
        raise ValueError("projected coordinates must be finite")

    return projected_x_mm, projected_y_mm, target_z_mm