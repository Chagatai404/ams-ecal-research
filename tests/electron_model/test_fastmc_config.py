from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from ams_ecal.electron_model.fastmc_config import (
    EXPECTED_FASTMC_SCHEMA_VERSION,
    FastMCConfig,
    FastMCConfigError,
    LateralEMConfig,
    LongitudinalEMConfig,
    SamplingShowerMaxConfig,
    StochasticEMConfig,
    config_digest,
    load_fastmc_config,
)

CONFIG_PATH = Path(__file__).parents[2] / "configs" / "fastmc.yaml"


def make_lateral_config() -> LateralEMConfig:
    return LateralEMConfig(
        scale_log_slope=-0.000690,
        scale_log_intercept=0.00660,
        entrance_scale_cells=0.176,
        calibration_cell_moliere_fraction=0.5,
        calibration_energy_unit_mev=1000.0,
    )


def make_sampling_config() -> SamplingShowerMaxConfig:
    return SamplingShowerMaxConfig(
        sampling_frequency_coefficient=0.55,
        transition_effect_coefficient=0.69,
        absorber_atomic_number=82.0,
        active_atomic_number=5.3,
    )


def make_longitudinal_config() -> LongitudinalEMConfig:
    return LongitudinalEMConfig(
        gamma_rate=0.65,
        deposition_offset_x0=-0.5,
        sampling_offset_x0=-0.812,
        sampling_shower_max=make_sampling_config(),
    )


def make_stochastic_config() -> StochasticEMConfig:
    return StochasticEMConfig(
        deposition_width_intercept=-1.4,
        deposition_width_log_slope=1.26,
        sampling_width_intercept=-2.5,
        sampling_width_log_slope=1.25,
    )


def test_loads_versioned_fastmc_configuration() -> None:
    config = load_fastmc_config(CONFIG_PATH)

    assert isinstance(config, FastMCConfig)
    assert config.longitudinal_em.gamma_rate == pytest.approx(0.65)
    assert config.regime == "sampling"
    assert config.longitudinal_em.deposition_offset_x0 == pytest.approx(-0.5)
    assert config.longitudinal_em.sampling_offset_x0 == pytest.approx(-0.812)
    assert config.longitudinal_em.sampling_shower_max == make_sampling_config()
    assert config.lateral_em == make_lateral_config()
    assert config.stochastic_em == make_stochastic_config()


def test_longitudinal_configuration_is_immutable() -> None:
    config = load_fastmc_config(CONFIG_PATH)

    with pytest.raises(FrozenInstanceError):
        config.longitudinal_em.gamma_rate = 0.5


def test_lateral_configuration_is_immutable() -> None:
    config = load_fastmc_config(CONFIG_PATH)

    with pytest.raises(FrozenInstanceError):
        config.lateral_em.entrance_scale_cells = 0.2


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "error_type", "message"),
    [
        ("gamma_rate", True, TypeError, "must be a real number"),
        ("gamma_rate", "0.65", TypeError, "must be a real number"),
        ("gamma_rate", float("nan"), ValueError, "finite and positive"),
        ("gamma_rate", float("inf"), ValueError, "finite and positive"),
        ("gamma_rate", 0.0, ValueError, "finite and positive"),
        ("gamma_rate", -0.1, ValueError, "finite and positive"),
        (
            "sampling_offset_x0",
            True,
            TypeError,
            "must be a real number",
        ),
        (
            "sampling_offset_x0",
            float("nan"),
            ValueError,
            "must be finite",
        ),
    ],
)
def test_rejects_invalid_longitudinal_parameters(
    field_name: str,
    invalid_value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    config = load_fastmc_config(CONFIG_PATH).longitudinal_em

    with pytest.raises(error_type, match=message):
        replace(config, **{field_name: invalid_value})


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "error_type", "message"),
    [
        ("scale_log_slope", True, TypeError, "must be a real number"),
        ("scale_log_slope", float("nan"), ValueError, "must be finite"),
        ("scale_log_intercept", float("inf"), ValueError, "must be finite"),
        (
            "entrance_scale_cells",
            0.0,
            ValueError,
            "finite and positive",
        ),
        (
            "calibration_cell_moliere_fraction",
            -0.5,
            ValueError,
            "finite and positive",
        ),
        (
            "calibration_energy_unit_mev",
            "1000",
            TypeError,
            "must be a real number",
        ),
    ],
)
def test_rejects_invalid_lateral_parameters(
    field_name: str,
    invalid_value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    config = load_fastmc_config(CONFIG_PATH).lateral_em

    with pytest.raises(error_type, match=message):
        replace(config, **{field_name: invalid_value})


def test_rejects_invalid_top_level_component_type() -> None:
    with pytest.raises(
        TypeError,
        match="longitudinal_em must be a LongitudinalEMConfig",
    ):
        FastMCConfig(
            regime="sampling",
            longitudinal_em=None,
            lateral_em=make_lateral_config(),
            stochastic_em=make_stochastic_config(),
        )

    with pytest.raises(
        TypeError,
        match="lateral_em must be a LateralEMConfig",
    ):
        FastMCConfig(
            regime="sampling",
            longitudinal_em=make_longitudinal_config(),
            lateral_em=None,
            stochastic_em=make_stochastic_config(),
        )

    with pytest.raises(
        TypeError,
        match="stochastic_em must be a StochasticEMConfig",
    ):
        FastMCConfig(
            regime="sampling",
            longitudinal_em=make_longitudinal_config(),
            lateral_em=make_lateral_config(),
            stochastic_em=None,
        )


def test_rejects_missing_configuration_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="FastMC configuration"):
        load_fastmc_config(missing_path)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (
            "schema_version: 5\nregime: sampling\n",
            "missing keys:.*longitudinal_em",
        ),
        (
            "schema_version: 5\nregime: sampling\nlongitudinal_em: {}\nlateral_em: {}\nstochastic_em: {}\n",
            "missing keys",
        ),
        (
            "schema_version: 5\nregime: sampling\nlongitudinal_em: {gamma_rate: 0.65, deposition_offset_x0: -0.5, sampling_offset_x0: -0.812, sampling_shower_max: {sampling_frequency_coefficient: 0.55, transition_effect_coefficient: 0.69, absorber_atomic_number: 82.0, active_atomic_number: 5.3}, typo: 1}\nlateral_em: {}\nstochastic_em: {}\n",
            "unexpected keys:.*typo",
        ),
        (
            "schema_version: 5\nregime: sampling\nlongitudinal_em: {}\nlateral_em: {}\n",
            "missing keys:.*stochastic_em",
        ),
        (
            "schema_version: 4\nregime: sampling\nlongitudinal_em: {}\nlateral_em: {}\nstochastic_em: {}\n",
            "Unsupported FastMC schema_version 4",
        ),
    ],
)
def test_rejects_invalid_schema(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    config_path = tmp_path / "fastmc.yaml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(FastMCConfigError, match=message):
        load_fastmc_config(config_path)


def test_exposes_expected_schema_version() -> None:
    assert EXPECTED_FASTMC_SCHEMA_VERSION == 5
    assert isinstance(make_longitudinal_config(), LongitudinalEMConfig)
    assert isinstance(make_lateral_config(), LateralEMConfig)


def test_stochastic_configuration_is_immutable() -> None:
    config = load_fastmc_config(CONFIG_PATH)

    with pytest.raises(FrozenInstanceError):
        config.stochastic_em.sampling_width_log_slope = 2.0


def test_accepts_negative_stochastic_width_intercepts() -> None:
    # Both transferred width laws have negative intercepts, so these fields
    # must not be validated as positive quantities.
    config = make_stochastic_config()

    assert config.sampling_width_intercept == pytest.approx(-2.5)
    assert config.deposition_width_intercept == pytest.approx(-1.4)


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "error_type"),
    [
        ("sampling_width_intercept", True, TypeError),
        ("sampling_width_intercept", "-2.5", TypeError),
        ("sampling_width_intercept", float("nan"), ValueError),
        ("deposition_width_log_slope", float("inf"), ValueError),
    ],
)
def test_rejects_invalid_stochastic_parameters(
    field_name: str,
    invalid_value: object,
    error_type: type[Exception],
) -> None:
    config = make_stochastic_config()

    with pytest.raises(error_type, match=field_name):
        replace(config, **{field_name: invalid_value})


def test_config_digest_is_reproducible_and_content_sensitive(
    tmp_path: Path,
) -> None:
    first = tmp_path / "a.yaml"
    second = tmp_path / "b.yaml"
    first.write_text("alpha: 1\n", encoding="utf-8")
    second.write_text("beta: 2\n", encoding="utf-8")

    digest = config_digest([first, second])

    assert len(digest) == 64
    assert digest == config_digest([first, second])
    assert digest != config_digest([second, first])

    second.write_text("beta: 3\n", encoding="utf-8")

    assert digest != config_digest([first, second])


def test_config_digest_requires_existing_configurations(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one configuration"):
        config_digest([])

    with pytest.raises(FileNotFoundError, match="Configuration not found"):
        config_digest([tmp_path / "missing.yaml"])
