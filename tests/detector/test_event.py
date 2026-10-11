from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from ams_ecal.detector.event import (
    EVENT_SCHEMA_VERSION,
    ECALEvent,
    EnergyGrid,
    EventProvenance,
)
from ams_ecal.detector.geometry import ECALGeometry, load_geometry
from ams_ecal.detector.readout import project_track_to_cell_indices
from ams_ecal.detector.tracking import TrackState

CONFIG_PATH = Path(__file__).parents[2] / "configs" / "geometry.yaml"
VALID_CONFIGURATION_SHA256 = "0" * 64


def make_track() -> TrackState:
    """Return a simple vertical track used by canonical-event tests."""

    return TrackState(
        x0_mm=22.4,
        y0_mm=-17.2,
        z0_mm=0.0,
        theta_rad=0.0,
        phi_rad=0.0,
    )


def make_provenance() -> EventProvenance:
    """Return valid reproducibility metadata for one FastMC event."""

    return EventProvenance(
        simulation_backend="fastmc",
        simulation_version="0.1.0",
        configuration_sha256=VALID_CONFIGURATION_SHA256,
        random_seed=12345,
    )


def make_energy_grid(
    geometry: ECALGeometry,
    track: TrackState,
) -> EnergyGrid:
    """Create a deterministic energy grid centered on the track.

    Layer zero receives 1 MeV, layer one receives 2 MeV, and so on.
    Each deposit is placed in the cell crossed by the projected track.
    """

    projected_cells = project_track_to_cell_indices(track, geometry)

    mutable_grid = [
        [0.0 for _ in range(geometry.cells_per_layer)]
        for _ in range(geometry.number_of_layers)
    ]

    for layer_index, cell_index in enumerate(projected_cells):
        if cell_index is not None:
            mutable_grid[layer_index][cell_index] = float(
                layer_index + 1
            )

    return tuple(
        tuple(layer_energies)
        for layer_energies in mutable_grid
    )


def make_event() -> ECALEvent:
    """Return one fully valid canonical ECAL event."""

    geometry = load_geometry(CONFIG_PATH)
    track = make_track()

    return ECALEvent(
        event_id="fastmc-event-000001",
        particle_type="electron",
        primary_energy_mev=100_000.0,
        track=track,
        geometry=geometry,
        cell_energies_mev=make_energy_grid(geometry, track),
        provenance=make_provenance(),
    )


def test_constructs_valid_canonical_event() -> None:
    event = make_event()

    assert event.event_id == "fastmc-event-000001"
    assert event.particle_type == "electron"
    assert event.primary_energy_mev == 100_000.0
    assert event.schema_version == EVENT_SCHEMA_VERSION

    assert len(event.cell_energies_mev) == 18
    assert all(
        len(layer_energies) == 72
        for layer_energies in event.cell_energies_mev
    )


def test_event_and_energy_grid_are_immutable() -> None:
    event = make_event()

    assert isinstance(event.cell_energies_mev, tuple)
    assert all(
        isinstance(layer_energies, tuple)
        for layer_energies in event.cell_energies_mev
    )

    with pytest.raises(FrozenInstanceError):
        event.event_id = "changed-event-id"


def test_derives_projected_cells_from_stored_track() -> None:
    event = make_event()

    assert event.projected_cell_indices == (
        34,
        34,
        38,
        38,
        34,
        34,
        38,
        38,
        34,
        34,
        38,
        38,
        34,
        34,
        38,
        38,
        34,
        34,
    )


def test_calculates_layer_and_total_ecal_energy() -> None:
    event = make_event()

    assert event.layer_energies_mev == tuple(
        float(layer_index + 1)
        for layer_index in range(18)
    )
    assert event.total_ecal_energy_mev == pytest.approx(171.0)


@pytest.mark.parametrize(
    ("event_id", "error_type", "message"),
    [
        (1, TypeError, "event_id must be a string"),
        ("", ValueError, "event_id must not be empty"),
        ("   ", ValueError, "event_id must not be empty"),
    ],
)
def test_rejects_invalid_event_id(
    event_id: object,
    error_type: type[Exception],
    message: str,
) -> None:
    event = make_event()

    with pytest.raises(error_type, match=message):
        replace(event, event_id=event_id)


@pytest.mark.parametrize(
    ("particle_type", "error_type", "message"),
    [
        (1, TypeError, "particle_type must be a string"),
        (
            "muon",
            ValueError,
            "must be 'electron', 'positron', or 'proton'",
        ),
        (
            "",
            ValueError,
            "must be 'electron', 'positron', or 'proton'",
        ),
    ],
)
def test_rejects_invalid_particle_type(
    particle_type: object,
    error_type: type[Exception],
    message: str,
) -> None:
    event = make_event()

    with pytest.raises(error_type, match=message):
        replace(event, particle_type=particle_type)


@pytest.mark.parametrize(
    ("primary_energy_mev", "error_type", "message"),
    [
        (True, TypeError, "must be a real number"),
        ("1000.0", TypeError, "must be a real number"),
        (float("nan"), ValueError, "must be finite"),
        (float("inf"), ValueError, "must be finite"),
        (float("-inf"), ValueError, "must be finite"),
        (0.0, ValueError, "must be positive"),
        (-1.0, ValueError, "must be positive"),
    ],
)
def test_rejects_invalid_primary_energy(
    primary_energy_mev: object,
    error_type: type[Exception],
    message: str,
) -> None:
    event = make_event()

    with pytest.raises(error_type, match=message):
        replace(event, primary_energy_mev=primary_energy_mev)


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("track", None, "track must be a TrackState"),
        ("geometry", None, "geometry must be an ECALGeometry"),
        (
            "provenance",
            None,
            "provenance must be an EventProvenance",
        ),
    ],
)
def test_rejects_invalid_canonical_component_types(
    field_name: str,
    invalid_value: object,
    message: str,
) -> None:
    event = make_event()

    with pytest.raises(TypeError, match=message):
        replace(event, **{field_name: invalid_value})


@pytest.mark.parametrize(
    ("schema_version", "error_type", "message"),
    [
        (True, TypeError, "schema_version must be an integer"),
        (1.5, TypeError, "schema_version must be an integer"),
        (
            2,
            ValueError,
            "unsupported event schema_version 2",
        ),
    ],
)
def test_rejects_invalid_or_unsupported_schema_version(
    schema_version: object,
    error_type: type[Exception],
    message: str,
) -> None:
    event = make_event()

    with pytest.raises(error_type, match=message):
        replace(event, schema_version=schema_version)


def test_requires_energy_grid_to_be_tuple() -> None:
    event = make_event()
    energy_grid_as_list = list(event.cell_energies_mev)

    with pytest.raises(
        TypeError,
        match="cell_energies_mev must be a tuple",
    ):
        replace(
            event,
            cell_energies_mev=energy_grid_as_list,
        )


def test_requires_one_energy_row_per_layer() -> None:
    event = make_event()
    incomplete_grid = event.cell_energies_mev[:-1]

    with pytest.raises(
        ValueError,
        match="one row per ECAL layer",
    ):
        replace(
            event,
            cell_energies_mev=incomplete_grid,
        )


def test_requires_each_energy_row_to_be_tuple() -> None:
    event = make_event()

    invalid_grid = (
        list(event.cell_energies_mev[0]),
        *event.cell_energies_mev[1:],
    )

    with pytest.raises(
        TypeError,
        match="each layer in cell_energies_mev must be a tuple",
    ):
        replace(
            event,
            cell_energies_mev=invalid_grid,
        )


def test_requires_exact_number_of_cells_in_each_layer() -> None:
    event = make_event()

    invalid_grid = (
        event.cell_energies_mev[0][:-1],
        *event.cell_energies_mev[1:],
    )

    with pytest.raises(
        ValueError,
        match="layer 0 contains 71",
    ):
        replace(
            event,
            cell_energies_mev=invalid_grid,
        )


@pytest.mark.parametrize(
    ("invalid_energy", "error_type", "message"),
    [
        (True, TypeError, "cell energies must be real numbers"),
        ("1.0", TypeError, "cell energies must be real numbers"),
        (float("nan"), ValueError, "cell energies must be finite"),
        (float("inf"), ValueError, "cell energies must be finite"),
        (-0.1, ValueError, "cell energies must be nonnegative"),
    ],
)
def test_rejects_invalid_cell_energy(
    invalid_energy: object,
    error_type: type[Exception],
    message: str,
) -> None:
    event = make_event()

    modified_first_layer = (
        invalid_energy,
        *event.cell_energies_mev[0][1:],
    )
    invalid_grid = (
        modified_first_layer,
        *event.cell_energies_mev[1:],
    )

    with pytest.raises(error_type, match=message):
        replace(
            event,
            cell_energies_mev=invalid_grid,
        )


@pytest.mark.parametrize(
    ("simulation_backend", "error_type", "message"),
    [
        (1, TypeError, "simulation_backend must be a string"),
        (
            "fast-mc",
            ValueError,
            "must be either 'fastmc' or 'geant4'",
        ),
    ],
)
def test_rejects_invalid_simulation_backend(
    simulation_backend: object,
    error_type: type[Exception],
    message: str,
) -> None:
    provenance = make_provenance()

    with pytest.raises(error_type, match=message):
        replace(
            provenance,
            simulation_backend=simulation_backend,
        )


@pytest.mark.parametrize(
    ("simulation_version", "error_type", "message"),
    [
        (1, TypeError, "simulation_version must be a string"),
        ("", ValueError, "simulation_version must not be empty"),
        ("   ", ValueError, "simulation_version must not be empty"),
    ],
)
def test_rejects_invalid_simulation_version(
    simulation_version: object,
    error_type: type[Exception],
    message: str,
) -> None:
    provenance = make_provenance()

    with pytest.raises(error_type, match=message):
        replace(
            provenance,
            simulation_version=simulation_version,
        )


@pytest.mark.parametrize(
    ("configuration_sha256", "error_type", "message"),
    [
        (
            1,
            TypeError,
            "configuration_sha256 must be a string",
        ),
        (
            "0" * 63,
            ValueError,
            "exactly 64 hexadecimal characters",
        ),
        (
            "G" * 64,
            ValueError,
            "lowercase hexadecimal SHA-256 digest",
        ),
        (
            "A" * 64,
            ValueError,
            "lowercase hexadecimal SHA-256 digest",
        ),
    ],
)
def test_rejects_invalid_configuration_hash(
    configuration_sha256: object,
    error_type: type[Exception],
    message: str,
) -> None:
    provenance = make_provenance()

    with pytest.raises(error_type, match=message):
        replace(
            provenance,
            configuration_sha256=configuration_sha256,
        )


@pytest.mark.parametrize(
    ("random_seed", "error_type", "message"),
    [
        (True, TypeError, "random_seed must be an integer"),
        (1.5, TypeError, "random_seed must be an integer"),
        (-1, ValueError, "random_seed must be nonnegative"),
    ],
)
def test_rejects_invalid_random_seed(
    random_seed: object,
    error_type: type[Exception],
    message: str,
) -> None:
    provenance = make_provenance()

    with pytest.raises(error_type, match=message):
        replace(
            provenance,
            random_seed=random_seed,
        )


def test_model_details_default_to_empty_and_keep_old_events_valid() -> None:
    provenance = make_provenance()

    assert provenance.model_details == ()
    assert provenance.detail("calibration_version") is None


def test_model_details_record_named_facts() -> None:
    provenance = replace(
        make_provenance(),
        model_details=(
            ("calibration_version", "1"),
            ("physics_list", "FTFP_BERT"),
        ),
    )

    assert provenance.detail("physics_list") == "FTFP_BERT"
    assert provenance.detail("calibration_version") == "1"
    assert provenance.detail("missing") is None


@pytest.mark.parametrize(
    ("model_details", "error_type", "message"),
    [
        ([("a", "b")], TypeError, "tuple of"),
        ((["a", "b"],), TypeError, "pairs"),
        ((("a", "b", "c"),), TypeError, "pairs"),
        ((("", "b"),), ValueError, "key must not be empty"),
        ((("a", ""),), ValueError, "value must not be empty"),
        ((("b", "1"), ("a", "2")), ValueError, "sorted"),
        ((("a", "1"), ("a", "2")), ValueError, "unique"),
    ],
)
def test_rejects_invalid_model_details(
    model_details: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        replace(make_provenance(), model_details=model_details)