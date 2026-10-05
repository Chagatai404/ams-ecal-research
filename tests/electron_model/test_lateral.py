from itertools import pairwise
from math import isfinite, log, pi
from pathlib import Path

import pytest
from scipy.integrate import quad

from ams_ecal.detector.geometry import load_geometry
from ams_ecal.detector.tracking import TrackState
from ams_ecal.electron_model.fastmc_config import load_fastmc_config
from ams_ecal.electron_model.lateral import (
    CALIBRATION_ENERGY_RANGE_MEV,
    AMSLateralShowerModel,
)

PROJECT_ROOT = Path(__file__).parents[2]
GEOMETRY_CONFIG_PATH = PROJECT_ROOT / "configs" / "geometry.yaml"
FASTMC_CONFIG_PATH = PROJECT_ROOT / "configs" / "fastmc.yaml"


@pytest.fixture
def model() -> AMSLateralShowerModel:
    geometry = load_geometry(GEOMETRY_CONFIG_PATH)
    fastmc_config = load_fastmc_config(FASTMC_CONFIG_PATH)
    return AMSLateralShowerModel(
        config=fastmc_config.lateral_em,
        geometry=geometry,
    )


def test_uses_ams_fit_parameters_and_geometry_moliere_scale(
    model: AMSLateralShowerModel,
) -> None:
    assert model.nominal_moliere_radius_mm == pytest.approx(18.0)
    assert model.config.scale_log_slope == pytest.approx(-6.90e-4)
    assert model.config.scale_log_intercept == pytest.approx(6.60e-3)
    assert model.config.entrance_scale_cells == pytest.approx(0.176)
    assert model.config.calibration_cell_moliere_fraction == pytest.approx(0.5)


def test_exposes_published_calibration_range(
    model: AMSLateralShowerModel,
) -> None:
    assert CALIBRATION_ENERGY_RANGE_MEV == (3_000.0, 180_000.0)
    assert model.is_within_calibration_range(3_000.0)
    assert model.is_within_calibration_range(180_000.0)
    assert not model.is_within_calibration_range(2_999.0)
    assert not model.is_within_calibration_range(180_001.0)


def test_scale_coefficient_uses_natural_log_of_energy_in_gev(
    model: AMSLateralShowerModel,
) -> None:
    assert model.scale_coefficient(1_000.0) == pytest.approx(0.00660)
    assert model.scale_coefficient(100_000.0) == pytest.approx(
        -0.000690 * log(100.0) + 0.00660
    )


def test_lateral_scale_grows_with_depth_and_is_converted_to_mm(
    model: AMSLateralShowerModel,
) -> None:
    energy_mev = 100_000.0
    entrance_cells = model.layer_scale_cells(0, energy_mev)
    rear_cells = model.layer_scale_cells(17, energy_mev)

    assert entrance_cells == pytest.approx(0.176)
    assert rear_cells > entrance_cells
    assert model.layer_scale_mm(0, energy_mev) == pytest.approx(
        entrance_cells * 0.5 * 18.0
    )


def test_higher_energy_profile_widens_more_slowly_in_published_fit(
    model: AMSLateralShowerModel,
) -> None:
    assert model.layer_scale_cells(17, 10_000.0) > model.layer_scale_cells(
        17,
        100_000.0,
    )


def test_radial_density_is_normalized_over_the_transverse_plane(
    model: AMSLateralShowerModel,
) -> None:
    layer_index = 10
    energy_mev = 100_000.0

    integral, error = quad(
        lambda radius_mm: (
            2.0
            * pi
            * radius_mm
            * model.radial_energy_density_per_mm2(
                radius_mm,
                layer_index,
                energy_mev,
            )
        ),
        0.0,
        float("inf"),
    )

    assert error < 1.0e-8
    assert integral == pytest.approx(1.0, abs=1.0e-10)


def test_radial_containment_is_monotone_and_approaches_unity(
    model: AMSLateralShowerModel,
) -> None:
    radii_mm = (0.0, 9.0, 18.0, 36.0, 1.0e8)
    contained = tuple(
        model.radial_contained_fraction(radius, 10, 100_000.0) for radius in radii_mm
    )

    assert contained[0] == pytest.approx(0.0)
    assert all(later > earlier for earlier, later in pairwise(contained))
    assert contained[-1] == pytest.approx(1.0, abs=1.0e-12)


def test_projected_cdf_is_symmetric_monotone_and_normalized(
    model: AMSLateralShowerModel,
) -> None:
    offsets_mm = (-1.0e8, -18.0, 0.0, 18.0, 1.0e8)
    cdf = tuple(model.projected_cdf(offset, 10, 100_000.0) for offset in offsets_mm)

    assert cdf[0] == pytest.approx(0.0, abs=1.0e-12)
    assert cdf[-1] == pytest.approx(1.0, abs=1.0e-12)
    assert cdf[2] == pytest.approx(0.5, abs=1.0e-14)
    assert cdf[1] == pytest.approx(1.0 - cdf[3], abs=1.0e-14)
    assert all(later > earlier for earlier, later in pairwise(cdf))


def test_integrates_seventy_two_finite_nonnegative_cell_fractions(
    model: AMSLateralShowerModel,
) -> None:
    fractions = model.cell_energy_fractions(0.0, 10, 100_000.0)

    assert len(fractions) == model.geometry.cells_per_layer == 72
    assert all(isfinite(value) and value >= 0 for value in fractions)
    assert 0 < sum(fractions) < 1
    assert fractions == pytest.approx(tuple(reversed(fractions)))


def test_finite_transverse_width_preserves_lateral_leakage(
    model: AMSLateralShowerModel,
) -> None:
    central = model.contained_fraction(0.0, 17, 100_000.0)
    edge = model.contained_fraction(324.0, 17, 100_000.0)

    assert central < 1.0
    assert 0 < edge < central
    assert edge < 0.5


def test_mean_cell_energies_preserve_lateral_leakage(
    model: AMSLateralShowerModel,
) -> None:
    layer_energy_mev = 25_000.0
    cell_energies = model.mean_cell_energies_mev(
        layer_energy_mev,
        300.0,
        17,
        100_000.0,
    )

    assert len(cell_energies) == 72
    assert all(isfinite(value) and value >= 0 for value in cell_energies)
    assert sum(cell_energies) < layer_energy_mev
    assert sum(cell_energies) == pytest.approx(
        layer_energy_mev * model.contained_fraction(300.0, 17, 100_000.0)
    )


def test_projects_track_onto_each_layers_measured_axis(
    model: AMSLateralShowerModel,
) -> None:
    track = TrackState(
        x0_mm=10.0,
        y0_mm=-20.0,
        z0_mm=0.0,
        theta_rad=0.0,
        phi_rad=0.0,
    )
    grid = model.track_centered_cell_fractions(track, 100_000.0)

    assert len(grid) == 18
    assert all(len(row) == 72 for row in grid)
    assert grid[0].index(max(grid[0])) == 33  # x fibers measure y = -20 mm
    assert grid[2].index(max(grid[2])) == 37  # y fibers measure x = +10 mm


@pytest.mark.parametrize(
    ("energy_mev", "error_type", "message"),
    [
        (True, TypeError, "must be a real number"),
        ("1000", TypeError, "must be a real number"),
        (float("nan"), ValueError, "must be finite"),
        (float("inf"), ValueError, "must be finite"),
        (0.0, ValueError, "must be positive"),
        (-1.0, ValueError, "must be positive"),
    ],
)
def test_rejects_invalid_primary_energy(
    model: AMSLateralShowerModel,
    energy_mev: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        model.layer_scale_mm(0, energy_mev)


@pytest.mark.parametrize("layer_index", [True, 0.0, -1, 18])
def test_rejects_invalid_layer_index(
    model: AMSLateralShowerModel,
    layer_index: object,
) -> None:
    error_type = TypeError if layer_index in {True, 0.0} else IndexError

    with pytest.raises(error_type, match="layer_index"):
        model.layer_scale_mm(layer_index, 100_000.0)


@pytest.mark.parametrize(
    ("axis_mm", "error_type", "message"),
    [
        (True, TypeError, "must be a real number"),
        ("0", TypeError, "must be a real number"),
        (float("nan"), ValueError, "must be finite"),
        (float("inf"), ValueError, "must be finite"),
    ],
)
def test_rejects_invalid_axis_coordinate(
    model: AMSLateralShowerModel,
    axis_mm: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        model.cell_energy_fractions(axis_mm, 0, 100_000.0)


def test_rejects_invalid_model_components(model: AMSLateralShowerModel) -> None:
    with pytest.raises(TypeError, match="config must"):
        AMSLateralShowerModel(config=None, geometry=model.geometry)

    with pytest.raises(TypeError, match="geometry must"):
        AMSLateralShowerModel(config=model.config, geometry=None)

    with pytest.raises(TypeError, match="track must"):
        model.track_centered_cell_fractions(None, 100_000.0)
