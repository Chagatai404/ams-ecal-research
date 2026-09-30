from dataclasses import dataclass
from math import isfinite
from typing import Literal

from ams_ecal.geometry import ECALGeometry
from ams_ecal.readout import project_track_to_cell_indices
from ams_ecal.tracking import TrackState

EVENT_SCHEMA_VERSION = 1

ParticleType = Literal["electron", "positron", "proton"]
SimulationBackend = Literal["fastmc", "geant4"]
EnergyGrid = tuple[tuple[float, ...], ...]

_SUPPORTED_PARTICLE_TYPES = {
    "electron",
    "positron",
    "proton",
}

_SUPPORTED_SIMULATION_BACKENDS = {
    "fastmc",
    "geant4",
}


def _validate_nonempty_string(value: object, name: str) -> None:
    """Require a nonempty string without modifying the supplied value."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")

    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _validate_sha256(value: object, name: str) -> None:
    """Require a lowercase hexadecimal SHA-256 digest."""

    _validate_nonempty_string(value, name)

    if len(value) != 64:
        raise ValueError(
            f"{name} must contain exactly 64 hexadecimal characters"
        )

    allowed_characters = set("0123456789abcdef")

    if any(character not in allowed_characters for character in value):
        raise ValueError(
            f"{name} must be a lowercase hexadecimal SHA-256 digest"
        )


@dataclass(frozen=True, slots=True)
class EventProvenance:
    """Reproducibility metadata for one simulated ECAL event."""

    simulation_backend: SimulationBackend
    simulation_version: str
    configuration_sha256: str
    random_seed: int
    # Optional named facts a backend needs to make the event reproducible and
    # interpretable beyond one version string: for a Block 6B proton, the model
    # version, calibration version, physics-list scenario and representation.
    # Sorted key/value pairs, so equal metadata compares and hashes equal.
    model_details: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.simulation_backend, str):
            raise TypeError("simulation_backend must be a string")

        if self.simulation_backend not in _SUPPORTED_SIMULATION_BACKENDS:
            raise ValueError(
                "simulation_backend must be either 'fastmc' or 'geant4'"
            )

        _validate_nonempty_string(
            self.simulation_version,
            "simulation_version",
        )
        _validate_sha256(
            self.configuration_sha256,
            "configuration_sha256",
        )

        if isinstance(self.random_seed, bool) or not isinstance(
            self.random_seed,
            int,
        ):
            raise TypeError("random_seed must be an integer")

        if self.random_seed < 0:
            raise ValueError("random_seed must be nonnegative")

        if not isinstance(self.model_details, tuple):
            raise TypeError("model_details must be a tuple of (key, value) pairs")

        for pair in self.model_details:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError("model_details must contain (key, value) pairs")

            _validate_nonempty_string(pair[0], "model_details key")
            _validate_nonempty_string(pair[1], "model_details value")

        keys = [key for key, _ in self.model_details]

        if keys != sorted(set(keys)):
            raise ValueError(
                "model_details keys must be unique and in sorted order"
            )

    def detail(self, key: str) -> str | None:
        """Return one named model detail, or ``None`` if it was not recorded."""

        return dict(self.model_details).get(key)


@dataclass(frozen=True, slots=True)
class ECALEvent:
    """Canonical representation of one simulated AMS-02 ECAL event.

    The ECAL energy grid follows the detector's alternating readout:

        cell_energies_mev[layer_index][cell_index]

    It therefore has shape

        number_of_layers x cells_per_layer

    which is 18 x 72 for the current simplified AMS-02 geometry.
    """

    event_id: str
    particle_type: ParticleType
    primary_energy_mev: float
    track: TrackState
    geometry: ECALGeometry
    cell_energies_mev: EnergyGrid
    provenance: EventProvenance
    schema_version: int = EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_nonempty_string(self.event_id, "event_id")

        if not isinstance(self.particle_type, str):
            raise TypeError("particle_type must be a string")

        if self.particle_type not in _SUPPORTED_PARTICLE_TYPES:
            raise ValueError(
                "particle_type must be 'electron', 'positron', or 'proton'"
            )

        if isinstance(self.primary_energy_mev, bool) or not isinstance(
            self.primary_energy_mev,
            int | float,
        ):
            raise TypeError("primary_energy_mev must be a real number")

        if not isfinite(self.primary_energy_mev):
            raise ValueError("primary_energy_mev must be finite")

        if self.primary_energy_mev <= 0:
            raise ValueError("primary_energy_mev must be positive")

        if not isinstance(self.track, TrackState):
            raise TypeError("track must be a TrackState")

        if not isinstance(self.geometry, ECALGeometry):
            raise TypeError("geometry must be an ECALGeometry")

        if not isinstance(self.provenance, EventProvenance):
            raise TypeError("provenance must be an EventProvenance")

        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version,
            int,
        ):
            raise TypeError("schema_version must be an integer")

        if self.schema_version != EVENT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported event schema_version "
                f"{self.schema_version}; expected {EVENT_SCHEMA_VERSION}"
            )

        self._validate_energy_grid()

    def _validate_energy_grid(self) -> None:
        """Validate the ECAL energy-grid structure and values."""

        if not isinstance(self.cell_energies_mev, tuple):
            raise TypeError("cell_energies_mev must be a tuple")

        if len(self.cell_energies_mev) != self.geometry.number_of_layers:
            raise ValueError(
                "cell_energies_mev must contain one row per ECAL layer"
            )

        for layer_index, layer_energies_mev in enumerate(
            self.cell_energies_mev
        ):
            if not isinstance(layer_energies_mev, tuple):
                raise TypeError(
                    "each layer in cell_energies_mev must be a tuple"
                )

            if (
                len(layer_energies_mev)
                != self.geometry.cells_per_layer
            ):
                raise ValueError(
                    "each ECAL energy row must contain exactly "
                    f"{self.geometry.cells_per_layer} cells; "
                    f"layer {layer_index} contains "
                    f"{len(layer_energies_mev)}"
                )

            for cell_index, energy_mev in enumerate(layer_energies_mev):
                if isinstance(energy_mev, bool) or not isinstance(
                    energy_mev,
                    int | float,
                ):
                    raise TypeError(
                        "ECAL cell energies must be real numbers; "
                        f"invalid value at layer {layer_index}, "
                        f"cell {cell_index}"
                    )

                if not isfinite(energy_mev):
                    raise ValueError(
                        "ECAL cell energies must be finite; "
                        f"invalid value at layer {layer_index}, "
                        f"cell {cell_index}"
                    )

                if energy_mev < 0:
                    raise ValueError(
                        "ECAL cell energies must be nonnegative; "
                        f"invalid value at layer {layer_index}, "
                        f"cell {cell_index}"
                    )

    @property
    def projected_cell_indices(self) -> tuple[int | None, ...]:
        """Return tracker-projected cells for every ECAL layer."""

        return project_track_to_cell_indices(
            track=self.track,
            geometry=self.geometry,
        )

    @property
    def layer_energies_mev(self) -> tuple[float, ...]:
        """Return the total ECAL energy recorded in each layer."""

        return tuple(
            sum(layer_energies_mev)
            for layer_energies_mev in self.cell_energies_mev
        )

    @property
    def total_ecal_energy_mev(self) -> float:
        """Return energy summed over all ECAL layers and cells."""

        return sum(self.layer_energies_mev)