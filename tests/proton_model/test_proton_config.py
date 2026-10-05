"""Proton model configuration."""

from pathlib import Path

import pytest
import yaml

from ams_ecal.electron_model.fastmc_config import FastMCConfigError, config_digest
from ams_ecal.proton_model.proton_config import (
    ProtonDomainConfig,
    ProtonInteractionConfig,
    load_proton_config,
)

CONFIG = Path(__file__).parents[2] / "configs" / "fastmc_proton.yaml"


def variant(tmp_path: Path, edit) -> Path:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    edit(raw)
    path = tmp_path / "proton.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_the_committed_configuration_loads() -> None:
    config = load_proton_config(CONFIG)

    assert config.calibration.physics_list == "ftfp_bert"
    assert config.representation == "readout"
    assert config.interaction.effective_length_mm is None
    assert (config.domain.energy_min_gev, config.domain.energy_max_gev) == (10.0, 100.0)
    assert config.domain.max_theta_rad == 0.0


def test_the_configuration_can_be_hashed_for_provenance() -> None:
    digest = config_digest([CONFIG])

    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_the_domain_includes_its_endpoints_and_nothing_outside() -> None:
    domain = load_proton_config(CONFIG).domain

    assert domain.contains_energy_mev(10_000.0)
    assert domain.contains_energy_mev(100_000.0)
    assert domain.contains_energy_mev(55_000.0)
    assert not domain.contains_energy_mev(9_999.9)
    assert not domain.contains_energy_mev(100_000.1)


def test_a_numeric_interaction_length_overrides_the_calibrated_one(tmp_path) -> None:
    path = variant(tmp_path, lambda raw: raw["interaction"].update(effective_length_mm=240.0))

    assert load_proton_config(path).interaction.effective_length_mm == 240.0


@pytest.mark.parametrize("length", [0.0, -5.0])
def test_a_non_positive_interaction_length_is_rejected(length) -> None:
    with pytest.raises(ValueError, match="positive"):
        ProtonInteractionConfig(effective_length_mm=length)


def test_a_non_numeric_interaction_length_is_rejected() -> None:
    with pytest.raises(TypeError):
        ProtonInteractionConfig(effective_length_mm="long")


@pytest.mark.parametrize("representation", ["sampling", "deposit", ""])
def test_an_unknown_representation_is_rejected(tmp_path, representation) -> None:
    path = variant(tmp_path, lambda raw: raw.update(representation=representation))

    with pytest.raises(ValueError, match="representation"):
        load_proton_config(path)


def test_an_unknown_physics_list_is_rejected(tmp_path) -> None:
    path = variant(tmp_path, lambda raw: raw["calibration"].update(physics_list="qbbc"))

    with pytest.raises(ValueError, match="physics_list"):
        load_proton_config(path)


@pytest.mark.parametrize(
    ("low", "high"), [(100.0, 10.0), (10.0, 10.0), (0.0, 10.0), (-1.0, 10.0)]
)
def test_an_inverted_or_non_positive_energy_range_is_rejected(low, high) -> None:
    with pytest.raises(ValueError, match="energy_min_gev"):
        ProtonDomainConfig(low, high, 0.0, "forbid")


def test_extrapolation_must_be_forbidden() -> None:
    with pytest.raises(ValueError, match="forbid"):
        ProtonDomainConfig(10.0, 100.0, 0.0, "allow")


def test_a_negative_angle_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_theta_rad"):
        ProtonDomainConfig(10.0, 100.0, -0.1, "forbid")


def test_an_unexpected_key_is_rejected(tmp_path) -> None:
    path = variant(tmp_path, lambda raw: raw.update(surprise=1))

    with pytest.raises(FastMCConfigError, match="unexpected keys"):
        load_proton_config(path)


def test_a_missing_key_is_rejected(tmp_path) -> None:
    path = variant(tmp_path, lambda raw: raw.pop("domain"))

    with pytest.raises(FastMCConfigError, match="missing keys"):
        load_proton_config(path)


def test_an_unsupported_schema_version_is_rejected(tmp_path) -> None:
    path = variant(tmp_path, lambda raw: raw.update(schema_version=2))

    with pytest.raises(FastMCConfigError, match="schema_version"):
        load_proton_config(path)


def test_a_missing_file_is_reported() -> None:
    with pytest.raises(FileNotFoundError):
        load_proton_config(CONFIG.with_name("absent.yaml"))
