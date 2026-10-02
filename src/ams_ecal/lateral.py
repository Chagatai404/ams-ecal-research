"""Mean lateral electromagnetic-shower development in the AMS ECAL.

This module implements the deterministic lateral counterpart to the Block-4
longitudinal gamma profile.  It uses the AMS test-beam parametrization

    rho(r) = 3 R^2 / (pi (r + R)^4)

for the normalized energy density in one transverse plane.  The scale ``R``
depends on primary energy and zero-based ECAL layer index.  The published fit
expresses ``R`` in cells, with one calibration cell approximately equal to half
a Moliere radius.

An ECAL layer measures only the coordinate perpendicular to its fibers.  Cell
fractions are therefore obtained from the one-dimensional marginal of the
radially symmetric density.  The marginal integrates over an effectively
infinite fiber direction.  This is the ideal fiducial-volume approximation;
finite fiber-end leakage remains a documented refinement for later validation.
No shower-to-shower fluctuations are generated here.  They belong to the stochastic generation.
"""

from dataclasses import dataclass
from itertools import pairwise
from math import isfinite, log, pi

import numpy as np

from ams_ecal.fastmc_config import LateralEMConfig
from ams_ecal.geometry import ECALGeometry
from ams_ecal.readout import measured_axis_for_fiber
from ams_ecal.tracking import TrackState, project_track_to_z

CALIBRATION_ENERGY_RANGE_MEV = (3_000.0, 180_000.0)

# Fixed Gauss-Legendre nodes make the universal projected-profile CDF fast and
# deterministic.  The transformed interval is theta in (0, pi/2); endpoints
# are excluded naturally, so sec(theta) remains finite at every node.
_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(64)
_THETA_NODES = 0.25 * pi * (_GAUSS_NODES + 1.0)
_THETA_WEIGHTS = 0.25 * pi * _GAUSS_WEIGHTS
_SEC_THETA = 1.0 / np.cos(_THETA_NODES)


def _validate_real(value: object, name: str) -> float:
    """Return one finite real value while rejecting booleans explicitly."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")

    if not isfinite(value):
        raise ValueError(f"{name} must be finite")

    return float(value)


def _validate_positive_real(value: object, name: str) -> float:
    """Return one finite strictly positive real value."""

    result = _validate_real(value, name)

    if result <= 0:
        raise ValueError(f"{name} must be positive")

    return result


def _validate_nonnegative_real(value: object, name: str) -> float:
    """Return one finite nonnegative real value."""

    result = _validate_real(value, name)

    if result < 0:
        raise ValueError(f"{name} must be nonnegative")

    return result


def _validate_layer_index(layer_index: object, geometry: ECALGeometry) -> int:
    """Return an index identifying one existing longitudinal sampling."""

    if isinstance(layer_index, bool) or not isinstance(layer_index, int):
        raise TypeError("layer_index must be an integer")

    if not 0 <= layer_index < geometry.number_of_layers:
        raise IndexError("layer_index must identify an existing readout layer")

    return layer_index


def _standard_projected_cdf(standard_coordinate: float) -> float:
    """Return the universal strip-marginal CDF for unit lateral scale.

    For positive ``z``, the upper-tail probability can be written as

        P(X > z) = (1 / pi) integral_0^(pi/2) S(z sec(theta)) dtheta,

    where ``S(u) = (1 + 3u) / (1 + u)^3`` is the radial survival
    function.  Reflection symmetry supplies the negative half of the CDF.
    """

    coordinate = _validate_real(standard_coordinate, "standard_coordinate")
    absolute_coordinate = abs(coordinate)

    with np.errstate(over="ignore"):
        scaled_radius = absolute_coordinate * _SEC_THETA

    inverse = 1.0 / (1.0 + scaled_radius)
    radial_survival = 3.0 * inverse**2 - 2.0 * inverse**3
    positive_tail = float(np.dot(_THETA_WEIGHTS, radial_survival) / pi)

    if not isfinite(positive_tail) or not 0.0 <= positive_tail <= 0.5:
        raise ArithmeticError("projected lateral CDF integration failed")

    if coordinate < 0:
        return positive_tail

    return 1.0 - positive_tail


@dataclass(frozen=True, slots=True)
class AMSLateralShowerModel:
    """AMS-specific deterministic mean lateral model for EM showers.

    The layer scale is

        A(E) = p0 ln(E / E_unit) + p1
        R_layer = A(E) layer_index^2 + B,

    with ``R_layer`` first evaluated in the calibration paper's cell units and
    then converted through the geometry's nominal Moliere radius.
    """

    config: LateralEMConfig
    geometry: ECALGeometry

    def __post_init__(self) -> None:
        if not isinstance(self.config, LateralEMConfig):
            raise TypeError("config must be a LateralEMConfig")

        if not isinstance(self.geometry, ECALGeometry):
            raise TypeError("geometry must be an ECALGeometry")

    @property
    def nominal_moliere_radius_mm(self) -> float:
        """Return the geometry-derived nominal Moliere radius."""

        return self.geometry.nominal_moliere_radius_mm

    def is_within_calibration_range(self, primary_energy_mev: float) -> bool:
        """Report whether energy lies in the published 3--180 GeV fit range."""

        energy_mev = _validate_positive_real(
            primary_energy_mev,
            "primary_energy_mev",
        )
        lower_mev, upper_mev = CALIBRATION_ENERGY_RANGE_MEV
        return lower_mev <= energy_mev <= upper_mev

    def scale_coefficient(self, primary_energy_mev: float) -> float:
        """Return the fitted quadratic layer coefficient ``A(E)``."""

        energy_mev = _validate_positive_real(
            primary_energy_mev,
            "primary_energy_mev",
        )
        coefficient = (
            self.config.scale_log_slope
            * log(energy_mev / self.config.calibration_energy_unit_mev)
            + self.config.scale_log_intercept
        )

        if not isfinite(coefficient) or coefficient <= 0:
            raise ValueError(
                "primary_energy_mev is outside the stable extrapolation "
                "domain of the lateral scale model"
            )

        return coefficient

    def layer_scale_cells(
        self,
        layer_index: int,
        primary_energy_mev: float,
    ) -> float:
        """Return lateral scale in the test-beam calibration-cell unit."""

        index = _validate_layer_index(layer_index, self.geometry)
        scale_cells = (
            self.scale_coefficient(primary_energy_mev) * index**2
            + self.config.entrance_scale_cells
        )

        if not isfinite(scale_cells) or scale_cells <= 0:
            raise ArithmeticError("computed lateral scale must be positive")

        return scale_cells

    def layer_scale_mm(
        self,
        layer_index: int,
        primary_energy_mev: float,
    ) -> float:
        """Return layer lateral scale in millimeters."""

        return (
            self.layer_scale_cells(layer_index, primary_energy_mev)
            * self.config.calibration_cell_moliere_fraction
            * self.nominal_moliere_radius_mm
        )

    def radial_energy_density_per_mm2(
        self,
        radius_mm: float,
        layer_index: int,
        primary_energy_mev: float,
    ) -> float:
        """Return normalized transverse energy density at radius ``r``."""

        radius = _validate_nonnegative_real(radius_mm, "radius_mm")
        scale_mm = self.layer_scale_mm(layer_index, primary_energy_mev)
        return 3.0 * scale_mm**2 / (pi * (radius + scale_mm) ** 4)

    def radial_contained_fraction(
        self,
        radius_mm: float,
        layer_index: int,
        primary_energy_mev: float,
    ) -> float:
        """Return ideal energy fraction inside a circle of radius ``r``."""

        radius = _validate_nonnegative_real(radius_mm, "radius_mm")
        scale_mm = self.layer_scale_mm(layer_index, primary_energy_mev)
        inverse = scale_mm / (radius + scale_mm)
        contained = 1.0 - 3.0 * inverse**2 + 2.0 * inverse**3

        if not isfinite(contained) or not 0.0 <= contained <= 1.0:
            raise ArithmeticError("computed radial containment is invalid")

        return contained

    def projected_cdf(
        self,
        coordinate_offset_mm: float,
        layer_index: int,
        primary_energy_mev: float,
    ) -> float:
        """Return strip-marginal CDF at an offset from the shower axis."""

        offset_mm = _validate_real(
            coordinate_offset_mm,
            "coordinate_offset_mm",
        )
        scale_mm = self.layer_scale_mm(layer_index, primary_energy_mev)
        return _standard_projected_cdf(offset_mm / scale_mm)

    def cell_energy_fractions(
        self,
        axis_coordinate_mm: float,
        layer_index: int,
        primary_energy_mev: float,
    ) -> tuple[float, ...]:
        """Integrate the projected mean profile over all 72 finite cells.

        Fractions are not renormalized to the active transverse width.  Their
        missing sum is the ideal lateral leakage in the measured coordinate.
        """

        axis_mm = _validate_real(axis_coordinate_mm, "axis_coordinate_mm")
        index = _validate_layer_index(layer_index, self.geometry)
        lower_bound_mm = (
            -(self.geometry.cells_per_layer * self.geometry.cell_pitch_mm) / 2.0
        )
        boundaries_mm = tuple(
            lower_bound_mm + cell_boundary * self.geometry.cell_pitch_mm
            for cell_boundary in range(self.geometry.cells_per_layer + 1)
        )
        cdf_values = tuple(
            self.projected_cdf(
                boundary_mm - axis_mm,
                index,
                primary_energy_mev,
            )
            for boundary_mm in boundaries_mm
        )
        fractions = tuple(
            upper_cdf - lower_cdf for lower_cdf, upper_cdf in pairwise(cdf_values)
        )

        if any(not isfinite(value) or value < 0 for value in fractions):
            raise ArithmeticError(
                "computed cell fractions must be finite and nonnegative"
            )

        if sum(fractions) > 1.0 + 1.0e-12:
            raise ArithmeticError("computed cell fractions exceed unity")

        return fractions

    def contained_fraction(
        self,
        axis_coordinate_mm: float,
        layer_index: int,
        primary_energy_mev: float,
    ) -> float:
        """Return fraction contained along one layer's measured coordinate."""

        return sum(
            self.cell_energy_fractions(
                axis_coordinate_mm,
                layer_index,
                primary_energy_mev,
            )
        )

    def mean_cell_energies_mev(
        self,
        layer_energy_mev: float,
        axis_coordinate_mm: float,
        layer_index: int,
        primary_energy_mev: float,
    ) -> tuple[float, ...]:
        """Distribute one mean longitudinal layer energy across its cells."""

        layer_energy = _validate_nonnegative_real(
            layer_energy_mev,
            "layer_energy_mev",
        )
        return tuple(
            layer_energy * fraction
            for fraction in self.cell_energy_fractions(
                axis_coordinate_mm,
                layer_index,
                primary_energy_mev,
            )
        )

    def track_centered_cell_fractions(
        self,
        track: TrackState,
        primary_energy_mev: float,
    ) -> tuple[tuple[float, ...], ...]:
        """Return the 18 x 72 mean lateral grid around a projected track."""

        if not isinstance(track, TrackState):
            raise TypeError("track must be a TrackState")

        _validate_positive_real(primary_energy_mev, "primary_energy_mev")
        layer_rows: list[tuple[float, ...]] = []

        for layer_index, layer_z_mm in enumerate(
            self.geometry.uniform_layer_centers_z_mm
        ):
            projected_x_mm, projected_y_mm, _ = project_track_to_z(
                track,
                target_z_mm=layer_z_mm,
            )
            measured_axis = measured_axis_for_fiber(
                self.geometry.layer_fiber_axes[layer_index]
            )
            axis_coordinate_mm = (
                projected_x_mm if measured_axis == "x" else projected_y_mm
            )
            layer_rows.append(
                self.cell_energy_fractions(
                    axis_coordinate_mm,
                    layer_index,
                    primary_energy_mev,
                )
            )

        return tuple(layer_rows)
