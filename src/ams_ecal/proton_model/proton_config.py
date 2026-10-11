"""Validated configuration for the proton model FastMC.

Kept apart from ``fastmc_config`` because that schema is the electromagnetic
one, locked to exact keys. The physics that can be calibrated (tables, the
effective interaction length) lives in the calibration artifact; this file
holds the CHOICES: which Geant4 scenario, which representation, which domain.
"""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal

import yaml

from ams_ecal.electron_model.fastmc_config import FastMCConfigError

EXPECTED_PROTON_SCHEMA_VERSION = 1

ProtonRepresentation = Literal["readout", "deposition"]
ProtonPhysicsList = Literal["ftfp_bert", "qgsp_bert"]

_REPRESENTATIONS = ("readout", "deposition")
_PHYSICS_LISTS = ("ftfp_bert", "qgsp_bert")


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class ProtonCalibrationRef:
    """Which calibration scenario, and where its artifact lives."""

    physics_list: ProtonPhysicsList
    artifact: str

    def __post_init__(self) -> None:
        if self.physics_list not in _PHYSICS_LISTS:
            raise ValueError(f"physics_list must be one of {list(_PHYSICS_LISTS)}")
        if not isinstance(self.artifact, str) or not self.artifact.strip():
            raise ValueError("artifact must be a non-empty path string")


@dataclass(frozen=True, slots=True)
class ProtonInteractionConfig:
    """The effective interaction length, or ``None`` to use the calibrated one."""

    effective_length_mm: float | None

    def __post_init__(self) -> None:
        if self.effective_length_mm is None:
            return
        length = _real(self.effective_length_mm, "effective_length_mm")
        if length <= 0:
            raise ValueError("effective_length_mm must be positive")


@dataclass(frozen=True, slots=True)
class ProtonDomainConfig:
    """The calibrated domain; nothing outside it is generated."""

    energy_min_gev: float
    energy_max_gev: float
    max_theta_rad: float
    extrapolation: Literal["forbid"]

    def __post_init__(self) -> None:
        low = _real(self.energy_min_gev, "energy_min_gev")
        high = _real(self.energy_max_gev, "energy_max_gev")
        theta = _real(self.max_theta_rad, "max_theta_rad")
        if not 0 < low < high:
            raise ValueError("require 0 < energy_min_gev < energy_max_gev")
        if theta < 0:
            raise ValueError("max_theta_rad must be nonnegative")
        if self.extrapolation != "forbid":
            raise ValueError(
                "extrapolation must be 'forbid': no extrapolation is implemented, "
                "and a silently extrapolated calibration would be an unlabelled one"
            )

    def contains_energy_mev(self, energy_mev: float) -> bool:
        return 1000.0 * self.energy_min_gev <= energy_mev <= 1000.0 * self.energy_max_gev


@dataclass(frozen=True, slots=True)
class ProtonConfig:
    """Top-level configuration of the proton model generator."""

    calibration: ProtonCalibrationRef
    representation: ProtonRepresentation
    interaction: ProtonInteractionConfig
    domain: ProtonDomainConfig

    def __post_init__(self) -> None:
        if self.representation not in _REPRESENTATIONS:
            raise ValueError(f"representation must be one of {list(_REPRESENTATIONS)}")


def _mapping(value: object, context: str) -> dict:
    if not isinstance(value, dict):
        raise FastMCConfigError(f"{context} must be a YAML mapping")
    return value


def _exact_keys(mapping: dict, expected: set[str], context: str) -> None:
    missing, unexpected = expected - set(mapping), set(mapping) - expected
    problems = []
    if missing:
        problems.append(f"missing keys: {sorted(missing)}")
    if unexpected:
        problems.append(f"unexpected keys: {sorted(map(str, unexpected))}")
    if problems:
        raise FastMCConfigError(f"{context} has " + "; ".join(problems))


def load_proton_config(config_path: str | Path) -> ProtonConfig:
    """Load and validate the versioned proton configuration."""

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Proton configuration not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise FastMCConfigError(f"Could not parse proton configuration {path}: {exc}") from exc

    root = _mapping(raw, "proton configuration root")
    _exact_keys(
        root,
        {"schema_version", "calibration", "representation", "interaction", "domain"},
        "proton configuration root",
    )
    version = root["schema_version"]
    if isinstance(version, bool) or version != EXPECTED_PROTON_SCHEMA_VERSION:
        raise FastMCConfigError(
            f"Unsupported proton schema_version {version!r}; "
            f"expected {EXPECTED_PROTON_SCHEMA_VERSION}"
        )

    calibration = _mapping(root["calibration"], "calibration")
    _exact_keys(calibration, {"physics_list", "artifact"}, "calibration")
    interaction = _mapping(root["interaction"], "interaction")
    _exact_keys(interaction, {"effective_length_mm"}, "interaction")
    domain = _mapping(root["domain"], "domain")
    _exact_keys(
        domain,
        {"energy_min_gev", "energy_max_gev", "max_theta_rad", "extrapolation"},
        "domain",
    )

    return ProtonConfig(
        calibration=ProtonCalibrationRef(
            physics_list=calibration["physics_list"], artifact=calibration["artifact"]
        ),
        representation=root["representation"],
        interaction=ProtonInteractionConfig(
            effective_length_mm=interaction["effective_length_mm"]
        ),
        domain=ProtonDomainConfig(
            energy_min_gev=domain["energy_min_gev"],
            energy_max_gev=domain["energy_max_gev"],
            max_theta_rad=domain["max_theta_rad"],
            extrapolation=domain["extrapolation"],
        ),
    )
