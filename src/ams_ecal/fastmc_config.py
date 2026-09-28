"""Validated configuration for the physics-informed FastMC.

Only parameters needed by implemented FastMC blocks belong here. Detector and
material properties remain in the geometry configuration so that scientific
quantities such as the effective critical energy have one source of truth.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Literal

import yaml

EXPECTED_FASTMC_SCHEMA_VERSION = 5

ShowerRegime = Literal["deposition", "sampling"]

_SUPPORTED_REGIMES = {"deposition", "sampling"}


class FastMCConfigError(ValueError):
    """Raised when a FastMC configuration is malformed or unsupported."""


def _validate_positive_real(value: object, name: str) -> None:
    """Require a finite, strictly positive real model parameter."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")

    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _validate_finite_real(value: object, name: str) -> None:
    """Require a finite real model parameter."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")

    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class SamplingShowerMaxConfig:
    """Sampling-geometry correction to the mean depth of shower maximum.

    In a sampling calorimeter the e/mip ratio falls with depth, so the signal
    maximum appears shallower than the true energy-deposition maximum. The
    correction subtracts

    ``sampling_frequency_coefficient / F_S
      + transition_effect_coefficient * (1 - e_hat)``

    where the sampling frequency ``F_S`` and the ratio ``e_hat`` are derived
    from the detector geometry rather than configured, so the correction tracks
    the detector description.

    Applied only under the ``sampling`` regime. Under ``deposition`` the
    longitudinal model describes true deposition and this correction is zero.
    """

    sampling_frequency_coefficient: float
    transition_effect_coefficient: float
    absorber_atomic_number: float
    active_atomic_number: float

    def __post_init__(self) -> None:
        _validate_finite_real(
            self.sampling_frequency_coefficient,
            "sampling_frequency_coefficient",
        )
        _validate_finite_real(
            self.transition_effect_coefficient,
            "transition_effect_coefficient",
        )
        _validate_positive_real(
            self.absorber_atomic_number,
            "absorber_atomic_number",
        )
        _validate_positive_real(
            self.active_atomic_number,
            "active_atomic_number",
        )


@dataclass(frozen=True, slots=True)
class LongitudinalEMConfig:
    """Scientific parameters for the mean electromagnetic shower profile.

    ``gamma_rate`` is the detector-specific beta parameter reported by AMS,
    held fixed for all showers and all energies as AMS itself does.

    One additive offset is held per regime, in ``t_max = ln(E / E_c) + offset``:
    ``deposition_offset_x0`` is the PDG electron energy-deposition value, and
    ``sampling_offset_x0`` is the homogeneous base to which
    ``sampling_shower_max`` then adds a geometry-derived shift. The model
    selects one according to its regime.
    """

    gamma_rate: float
    deposition_offset_x0: float
    sampling_offset_x0: float
    sampling_shower_max: SamplingShowerMaxConfig

    def __post_init__(self) -> None:
        _validate_positive_real(self.gamma_rate, "gamma_rate")
        _validate_finite_real(
            self.deposition_offset_x0,
            "deposition_offset_x0",
        )
        _validate_finite_real(
            self.sampling_offset_x0,
            "sampling_offset_x0",
        )

        if not isinstance(self.sampling_shower_max, SamplingShowerMaxConfig):
            raise TypeError(
                "sampling_shower_max must be a SamplingShowerMaxConfig"
            )


@dataclass(frozen=True, slots=True)
class LateralEMConfig:
    """Scientific parameters for the mean electromagnetic lateral profile.

    The parameters reproduce the AMS test-beam relation

    ``R_layer = (p0 * ln(E / E_unit) + p1) * layer_index**2 + B``.

    ``R_layer`` is expressed in calibration-cell units.  The calibration paper
    describes one such cell as approximately half a Moliere radius, which is
    recorded explicitly by ``calibration_cell_moliere_fraction``.
    """

    scale_log_slope: float
    scale_log_intercept: float
    entrance_scale_cells: float
    calibration_cell_moliere_fraction: float
    calibration_energy_unit_mev: float

    def __post_init__(self) -> None:
        _validate_finite_real(self.scale_log_slope, "scale_log_slope")
        _validate_finite_real(
            self.scale_log_intercept,
            "scale_log_intercept",
        )
        _validate_positive_real(
            self.entrance_scale_cells,
            "entrance_scale_cells",
        )
        _validate_positive_real(
            self.calibration_cell_moliere_fraction,
            "calibration_cell_moliere_fraction",
        )
        _validate_positive_real(
            self.calibration_energy_unit_mev,
            "calibration_energy_unit_mev",
        )


@dataclass(frozen=True, slots=True)
class StochasticEMConfig:
    """Shower-to-shower fluctuation parameters for electromagnetic events.

    One random variable per event, the depth of shower maximum ``T0``, carries
    all of the longitudinal fluctuation. The relative width of ``T0`` is

    ``s(E) = 1 / (intercept + slope * ln(E / E_c))``

    and is the standard deviation of ``ln(T0)``. One coefficient pair is held
    per regime, and the model selects the pair matching its own regime so that
    the mean depth and the width always come from the same parametrization:

    - ``deposition_*`` describes true energy deposition;
    - ``sampling_*`` describes signal-level shape, and is the wider of the two
      at every energy because the sampling process adds shape fluctuation.

    Both intercepts are negative in the parametrization these laws are
    transferred from, so no field may be constrained to positive values here.
    The combination must be positive at any energy actually used; that check
    belongs to the model, which knows the critical energy, rather than to the
    configuration.
    """

    deposition_width_intercept: float
    deposition_width_log_slope: float
    sampling_width_intercept: float
    sampling_width_log_slope: float

    def __post_init__(self) -> None:
        for name in (
            "deposition_width_intercept",
            "deposition_width_log_slope",
            "sampling_width_intercept",
            "sampling_width_log_slope",
        ):
            _validate_finite_real(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class FastMCConfig:
    """Top-level configuration for the implemented FastMC components."""

    regime: ShowerRegime
    longitudinal_em: LongitudinalEMConfig
    lateral_em: LateralEMConfig
    stochastic_em: StochasticEMConfig

    def __post_init__(self) -> None:
        if not isinstance(self.regime, str):
            raise TypeError("regime must be a string")

        if self.regime not in _SUPPORTED_REGIMES:
            raise ValueError(
                "regime must be either 'deposition' or 'sampling'"
            )

        if not isinstance(self.longitudinal_em, LongitudinalEMConfig):
            raise TypeError("longitudinal_em must be a LongitudinalEMConfig")

        if not isinstance(self.lateral_em, LateralEMConfig):
            raise TypeError("lateral_em must be a LateralEMConfig")

        if not isinstance(self.stochastic_em, StochasticEMConfig):
            raise TypeError("stochastic_em must be a StochasticEMConfig")


def _require_mapping(value: object, context: str) -> Mapping[object, object]:
    """Require a YAML mapping at one named configuration location."""

    if not isinstance(value, Mapping):
        raise FastMCConfigError(f"{context} must be a YAML mapping")

    return value


def _require_exact_keys(
    mapping: Mapping[object, object],
    expected_keys: set[str],
    context: str,
) -> None:
    """Reject missing or unknown keys in the strict scientific schema."""

    actual_keys = set(mapping)
    missing_keys = expected_keys - actual_keys
    unexpected_keys = actual_keys - expected_keys
    problems: list[str] = []

    if missing_keys:
        problems.append(f"missing keys: {sorted(missing_keys)}")

    if unexpected_keys:
        problems.append(f"unexpected keys: {sorted(map(str, unexpected_keys))}")

    if problems:
        raise FastMCConfigError(f"{context} has " + "; ".join(problems))


def load_fastmc_config(config_path: str | Path) -> FastMCConfig:
    """Load and validate the versioned FastMC configuration."""

    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"FastMC configuration not found: {path}")

    try:
        raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise FastMCConfigError(
            f"Could not parse FastMC configuration {path}: {exc}"
        ) from exc

    config = _require_mapping(raw_config, "FastMC configuration root")
    _require_exact_keys(
        config,
        {
            "schema_version",
            "regime",
            "longitudinal_em",
            "lateral_em",
            "stochastic_em",
        },
        "FastMC configuration root",
    )

    schema_version = config["schema_version"]

    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise FastMCConfigError("schema_version must be an integer")

    if schema_version != EXPECTED_FASTMC_SCHEMA_VERSION:
        raise FastMCConfigError(
            "Unsupported FastMC schema_version "
            f"{schema_version}; expected {EXPECTED_FASTMC_SCHEMA_VERSION}"
        )

    longitudinal_em = _require_mapping(
        config["longitudinal_em"],
        "longitudinal_em",
    )
    _require_exact_keys(
        longitudinal_em,
        {
            "gamma_rate",
            "deposition_offset_x0",
            "sampling_offset_x0",
            "sampling_shower_max",
        },
        "longitudinal_em",
    )

    sampling_shower_max = _require_mapping(
        longitudinal_em["sampling_shower_max"],
        "sampling_shower_max",
    )
    _require_exact_keys(
        sampling_shower_max,
        {
            "sampling_frequency_coefficient",
            "transition_effect_coefficient",
            "absorber_atomic_number",
            "active_atomic_number",
        },
        "sampling_shower_max",
    )

    lateral_em = _require_mapping(
        config["lateral_em"],
        "lateral_em",
    )
    _require_exact_keys(
        lateral_em,
        {
            "scale_log_slope",
            "scale_log_intercept",
            "entrance_scale_cells",
            "calibration_cell_moliere_fraction",
            "calibration_energy_unit_mev",
        },
        "lateral_em",
    )

    stochastic_em = _require_mapping(
        config["stochastic_em"],
        "stochastic_em",
    )
    _require_exact_keys(
        stochastic_em,
        {
            "deposition_width_intercept",
            "deposition_width_log_slope",
            "sampling_width_intercept",
            "sampling_width_log_slope",
        },
        "stochastic_em",
    )

    return FastMCConfig(
        regime=config["regime"],
        longitudinal_em=LongitudinalEMConfig(
            gamma_rate=longitudinal_em["gamma_rate"],
            deposition_offset_x0=longitudinal_em["deposition_offset_x0"],
            sampling_offset_x0=longitudinal_em["sampling_offset_x0"],
            sampling_shower_max=SamplingShowerMaxConfig(
                sampling_frequency_coefficient=sampling_shower_max[
                    "sampling_frequency_coefficient"
                ],
                transition_effect_coefficient=sampling_shower_max[
                    "transition_effect_coefficient"
                ],
                absorber_atomic_number=sampling_shower_max[
                    "absorber_atomic_number"
                ],
                active_atomic_number=sampling_shower_max[
                    "active_atomic_number"
                ],
            ),
        ),
        lateral_em=LateralEMConfig(
            scale_log_slope=lateral_em["scale_log_slope"],
            scale_log_intercept=lateral_em["scale_log_intercept"],
            entrance_scale_cells=lateral_em["entrance_scale_cells"],
            calibration_cell_moliere_fraction=lateral_em[
                "calibration_cell_moliere_fraction"
            ],
            calibration_energy_unit_mev=lateral_em["calibration_energy_unit_mev"],
        ),
        stochastic_em=StochasticEMConfig(
            deposition_width_intercept=stochastic_em[
                "deposition_width_intercept"
            ],
            deposition_width_log_slope=stochastic_em[
                "deposition_width_log_slope"
            ],
            sampling_width_intercept=stochastic_em[
                "sampling_width_intercept"
            ],
            sampling_width_log_slope=stochastic_em[
                "sampling_width_log_slope"
            ],
        ),
    )


def config_digest(config_paths: Iterable[str | Path]) -> str:
    """Return one SHA-256 digest over the given configuration files.

    Simulated events record the configuration that produced them. Hashing the
    raw bytes of every configuration file, in the order supplied, gives one
    reproducible identifier for that scientific input. Paths are hashed in the
    caller order, so the same files supplied in a different order yield a
    different digest; callers should therefore fix one order.
    """

    digest = sha256()
    paths = tuple(Path(config_path) for config_path in config_paths)

    if not paths:
        raise ValueError("config_digest requires at least one configuration")

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Configuration not found: {path}")

        digest.update(path.read_bytes())

    return digest.hexdigest()
