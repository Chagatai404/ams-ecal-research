from itertools import pairwise
from math import isfinite
from pathlib import Path

import pytest

from ams_ecal.detector.geometry import load_geometry
from ams_ecal.electron_model.fastmc_config import load_fastmc_config
from ams_ecal.electron_model.longitudinal import AMSLongitudinalGammaModel

PROJECT_ROOT = Path(__file__).parents[2]
GEOMETRY_CONFIG_PATH = PROJECT_ROOT / "configs" / "geometry.yaml"
FASTMC_CONFIG_PATH = PROJECT_ROOT / "configs" / "fastmc.yaml"


@pytest.fixture
def model() -> AMSLongitudinalGammaModel:
    geometry = load_geometry(GEOMETRY_CONFIG_PATH)
    fastmc_config = load_fastmc_config(FASTMC_CONFIG_PATH)
    return AMSLongitudinalGammaModel(
        config=fastmc_config.longitudinal_em,
        geometry=geometry,
    )


def test_uses_geometry_critical_energy_and_ams_rate(
    model: AMSLongitudinalGammaModel,
) -> None:
    assert model.critical_energy_mev == pytest.approx(7.6)
    assert model.config.gamma_rate == pytest.approx(0.65)


def test_shower_maximum_moves_logarithmically_deeper_with_energy(
    model: AMSLongitudinalGammaModel,
) -> None:
    energies_mev = (1.0e3, 1.0e4, 1.0e5, 1.0e6)
    maxima_x0 = tuple(model.shower_max_depth_x0(e) for e in energies_mev)

    assert maxima_x0 == tuple(sorted(maxima_x0))
    expected_step = 2.302585092994046
    assert all(
        later - earlier == pytest.approx(expected_step)
        for earlier, later in pairwise(maxima_x0)
    )


def test_numerical_density_peak_matches_predicted_maximum(
    model: AMSLongitudinalGammaModel,
) -> None:
    energy_mev = 100_000.0
    predicted_peak_x0 = model.shower_max_depth_x0(energy_mev)
    step_x0 = 0.002
    depths_x0 = tuple(index * step_x0 for index in range(10_001))
    densities = tuple(model.energy_density(t, energy_mev) for t in depths_x0)
    numerical_peak_x0 = depths_x0[densities.index(max(densities))]

    assert numerical_peak_x0 == pytest.approx(
        predicted_peak_x0,
        abs=step_x0,
    )


def test_density_rises_then_falls_and_is_finite_nonnegative(
    model: AMSLongitudinalGammaModel,
) -> None:
    energy_mev = 10_000.0
    peak_x0 = model.shower_max_depth_x0(energy_mev)
    depths_x0 = (0.0, peak_x0 / 2.0, peak_x0, 2.0 * peak_x0)
    densities = tuple(model.energy_density(t, energy_mev) for t in depths_x0)

    assert all(isfinite(value) and value >= 0 for value in densities)
    assert densities[0] < densities[1] < densities[2]
    assert densities[3] < densities[2]


@pytest.mark.parametrize("energy_mev", [1.0e3, 1.0e4, 1.0e5, 1.0e6])
def test_integrates_eighteen_finite_nonnegative_layer_fractions(
    model: AMSLongitudinalGammaModel,
    energy_mev: float,
) -> None:
    fractions = model.layer_energy_fractions(energy_mev)

    assert len(fractions) == model.geometry.number_of_layers == 18
    assert all(isfinite(value) and value >= 0 for value in fractions)
    assert 0 < sum(fractions) < 1


def test_finite_depth_leakage_is_not_renormalized(
    model: AMSLongitudinalGammaModel,
) -> None:
    energy_mev = 1_000_000.0
    contained = model.contained_fraction(energy_mev)
    leakage = model.leakage_fraction(energy_mev)

    assert contained < 1.0
    assert leakage > 0.0
    assert contained + leakage == pytest.approx(1.0)


def test_mean_layer_energies_preserve_physical_leakage(
    model: AMSLongitudinalGammaModel,
) -> None:
    energy_mev = 100_000.0
    layer_energies = model.mean_layer_energies_mev(energy_mev)

    assert len(layer_energies) == 18
    assert all(isfinite(value) and value >= 0 for value in layer_energies)
    assert sum(layer_energies) < energy_mev
    assert sum(layer_energies) == pytest.approx(
        energy_mev * model.contained_fraction(energy_mev)
    )


def test_electron_and_positron_mean_profiles_are_identical(
    model: AMSLongitudinalGammaModel,
) -> None:
    electron = model.layer_energy_fractions(100_000.0, "electron")
    positron = model.layer_energy_fractions(100_000.0, "positron")

    assert electron == positron


@pytest.mark.parametrize(
    ("energy_mev", "error_type", "message"),
    [
        (True, TypeError, "must be a real number"),
        ("1000", TypeError, "must be a real number"),
        (float("nan"), ValueError, "must be finite"),
        (float("inf"), ValueError, "must be finite"),
        (float("-inf"), ValueError, "must be finite"),
        (0.0, ValueError, "must be positive"),
        (-1.0, ValueError, "must be positive"),
    ],
)
def test_rejects_invalid_primary_energy(
    model: AMSLongitudinalGammaModel,
    energy_mev: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        model.layer_energy_fractions(energy_mev)


def test_rejects_energy_below_high_energy_model_domain(
    model: AMSLongitudinalGammaModel,
) -> None:
    with pytest.raises(ValueError, match="below the valid domain"):
        model.shower_max_depth_x0(10.0)


@pytest.mark.parametrize(
    ("depth_x0", "error_type", "message"),
    [
        (True, TypeError, "must be a real number"),
        ("1.0", TypeError, "must be a real number"),
        (float("nan"), ValueError, "must be finite"),
        (float("inf"), ValueError, "must be finite"),
        (-0.1, ValueError, "must be nonnegative"),
    ],
)
def test_rejects_invalid_depth(
    model: AMSLongitudinalGammaModel,
    depth_x0: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        model.energy_density(depth_x0, 10_000.0)


@pytest.mark.parametrize(
    ("particle_type", "error_type", "message"),
    [
        (1, TypeError, "must be a string"),
        ("proton", ValueError, "must be 'electron' or 'positron'"),
    ],
)
def test_rejects_unsupported_particle_type(
    model: AMSLongitudinalGammaModel,
    particle_type: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        model.layer_energy_fractions(100_000.0, particle_type)


def test_rejects_invalid_model_components(
    model: AMSLongitudinalGammaModel,
) -> None:
    with pytest.raises(TypeError, match="config must"):
        AMSLongitudinalGammaModel(config=None, geometry=model.geometry)

    with pytest.raises(TypeError, match="geometry must"):
        AMSLongitudinalGammaModel(config=model.config, geometry=None)
