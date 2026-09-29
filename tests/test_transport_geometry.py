from itertools import pairwise
from math import hypot, pi
from pathlib import Path

import numpy as np
import pytest

from ams_ecal.geometry import load_geometry
from ams_ecal.readout import coordinate_to_cell_index
from ams_ecal.transport_geometry import (
    FibreLayout,
    TransportConfigError,
    VoxelSize,
    extended_layer_count,
    extension_mesh,
    load_transport_config,
    prefix_mesh,
    superlayer_composition,
)

ROOT = Path(__file__).parents[1]
GEOMETRY = ROOT / "configs" / "geometry.yaml"
TRANSPORT = ROOT / "configs" / "geant4_transport.yaml"


@pytest.fixture
def geometry():
    return load_geometry(GEOMETRY)


@pytest.fixture
def layout(geometry):
    return FibreLayout(geometry)


@pytest.fixture
def transport():
    return load_transport_config(TRANSPORT)


# --- the fibre lattice --------------------------------------------------


def test_rows_alternate_full_and_staggered(layout) -> None:
    counts = [len(layout.fibre_centres_mm(row)) for row in range(10)]
    assert counts == [480, 479] * 5
    assert layout.fibres_per_superlayer == 4795


def test_every_fibre_lies_fully_inside_the_active_width(layout, geometry) -> None:
    half = geometry.width_x_mm / 2
    for row in range(layout.rows_per_superlayer):
        centres = np.array(layout.fibre_centres_mm(row))
        assert centres.min() - layout.fibre_radius_mm >= -half
        assert centres.max() + layout.fibre_radius_mm <= half


def test_fibres_follow_the_configured_pitch_and_stagger(layout, geometry) -> None:
    structure = geometry.sampling_structure
    full, staggered = layout.fibre_centres_mm(0), layout.fibre_centres_mm(1)
    assert np.allclose(np.diff(full), structure.fiber_horizontal_pitch_mm)
    assert staggered[0] - full[0] == pytest.approx(
        structure.adjacent_row_stagger_fraction * structure.fiber_horizontal_pitch_mm
    )
    offsets = [layout.row_offset_mm(row) for row in range(10)]
    assert np.allclose(np.diff(offsets), structure.fiber_row_spacing_mm)
    # The stack is centred in its superlayer.
    assert offsets[0] + offsets[-1] == pytest.approx(structure.superlayer_thickness_mm)


def test_no_two_fibres_overlap(layout, geometry) -> None:
    diameter = geometry.sampling_structure.fiber_diameter_mm
    for row, next_row in pairwise(range(layout.rows_per_superlayer)):
        a = np.array(layout.fibre_centres_mm(row))
        b = np.array(layout.fibre_centres_mm(next_row))
        dz = layout.row_offset_mm(next_row) - layout.row_offset_mm(row)
        assert np.diff(a).min() >= diameter
        nearest = np.abs(a[:, None] - b[None, :]).min()
        assert hypot(nearest, dz) >= diameter


def test_each_row_sits_inside_exactly_one_readout_sampling(layout) -> None:
    thickness = layout.sampling_thickness_mm
    for row in range(layout.rows_per_superlayer):
        sampling = layout.layer_index(0, row)
        centre = layout.row_offset_mm(row)
        assert sampling * thickness <= centre - layout.fibre_radius_mm
        assert centre + layout.fibre_radius_mm <= (sampling + 1) * thickness


def test_rows_split_evenly_between_the_two_samplings(layout) -> None:
    assert [layout.layer_index(3, row) for row in range(10)] == [6] * 5 + [7] * 5


def test_fibre_ids_round_trip_and_are_unique(layout, geometry) -> None:
    seen = set()
    for superlayer in range(geometry.number_of_superlayers):
        for row in range(layout.rows_per_superlayer):
            for index in range(len(layout.fibre_centres_mm(row))):
                fibre_id = layout.encode(superlayer, row, index)
                assert layout.decode(fibre_id) == (superlayer, row, index)
                seen.add(fibre_id)
    assert len(seen) == 9 * 4795


def test_fibre_cells_use_the_readout_convention(layout, geometry) -> None:
    mapping = layout.fibre_readout_map()
    for fibre_id in list(mapping)[::97]:
        superlayer, row, index = layout.decode(fibre_id)
        centre = layout.fibre_centres_mm(row)[index]
        assert mapping[fibre_id] == (
            layout.layer_index(superlayer, row),
            coordinate_to_cell_index(centre, geometry),
        )
    for layer in range(geometry.number_of_layers):
        cells = {cell for (lay, cell) in mapping.values() if lay == layer}
        assert cells == set(range(geometry.cells_per_layer))


def test_fibres_measure_the_coordinate_across_them(layout, geometry) -> None:
    assert [layout.measured_axis(s) for s in range(9)] == ["y", "x"] * 4 + ["y"]


def test_fibre_fraction_comes_from_the_explicit_lattice(layout, geometry) -> None:
    structure = geometry.sampling_structure
    expected = (
        4795 * pi * 0.5**2 / (geometry.width_x_mm * structure.superlayer_thickness_mm)
    )
    assert layout.fibre_volume_fraction == pytest.approx(expected)
    # Close to, but not equal to, the published 0.57 / 1.72 ratio: the ten
    # rows do not fill the full superlayer depth.
    assert abs(layout.fibre_volume_fraction - 0.57 / 1.72) < 0.02


# --- materials ----------------------------------------------------------


DENSITIES = {"lead_density": 11.35, "fibre_density": 1.06, "glue_density": 1.2}


def test_density_constraint_reproduces_the_published_density(layout) -> None:
    composition = superlayer_composition(layout, "average_density", **DENSITIES)
    assert composition.density_g_cm3 == pytest.approx(6.8)
    total = (
        composition.lead_volume_fraction
        + composition.fibre_volume_fraction
        + composition.glue_volume_fraction
    )
    assert total == pytest.approx(1.0)


def test_ratio_constraint_reproduces_the_published_lead_to_glue_ratio(layout) -> None:
    composition = superlayer_composition(layout, "relative_volume", **DENSITIES)
    assert composition.glue_volume_fraction / composition.lead_volume_fraction == (
        pytest.approx(0.15)
    )
    # The two published constraints disagree: the ratio implies a denser stack.
    assert composition.density_g_cm3 > 7.0


def test_mass_fractions_sum_to_one(layout) -> None:
    composition = superlayer_composition(layout, "average_density", **DENSITIES)
    fractions = composition.mass_fractions(11.35, 1.06, 1.2)
    assert sum(fractions) == pytest.approx(1.0)
    assert fractions[0] > 0.9  # the composite is mostly lead by mass


# --- scoring meshes -----------------------------------------------------


def test_meshes_tile_the_readout_exactly(geometry, transport) -> None:
    fine, coarse = prefix_mesh(geometry, transport), extension_mesh(geometry, transport)
    assert fine.shape == (216, 216, 36)
    assert coarse.shape == (72, 72, 252)
    assert coarse.origin_mm[2] == geometry.depth_z_mm
    assert extended_layer_count(geometry, transport) == 270


def test_voxel_centres_invert_the_flat_index(geometry, transport) -> None:
    grid = prefix_mesh(geometry, transport)
    nx, ny, _ = grid.shape
    ix, iy, iz = 5, 200, 17
    x, y, z = grid.centres_mm(np.array([(iz * ny + iy) * nx + ix]))
    assert (x[0], y[0], z[0]) == pytest.approx(
        (-324 + 5.5 * 3.0, -324 + 200.5 * 3.0, 17.5 * 4.625)
    )


def test_rejects_a_voxel_that_straddles_readout_cells(geometry, transport) -> None:
    from dataclasses import replace

    bad = replace(transport, prefix_voxel=VoxelSize(4.0, 3.0, 4.625))
    with pytest.raises(TransportConfigError, match="does not tile"):
        prefix_mesh(geometry, bad)


def test_sampling_extension_continues_the_superlayer_structure(
    layout, geometry, transport
) -> None:
    from ams_ecal.transport_geometry import extension_superlayer_count

    assert transport.extension_structure == "sampling"
    assert extension_superlayer_count(geometry, transport) == 126
    # The x/y alternation continues past superlayer 9 (index 8).
    assert [layout.fibre_axis(s) for s in range(7, 12)] == ["y", "x", "y", "x", "y"]
    layers, cells = layout.readout_lookup(135)
    assert (layers >= 0).sum() == 135 * 4795
    assert layers.max() == 269 and cells.min() == -1 and cells.max() == 71


def test_readout_lookup_agrees_with_the_fibre_map(layout) -> None:
    layers, cells = layout.readout_lookup()
    mapping = layout.fibre_readout_map()
    ids = np.flatnonzero(layers >= 0)
    assert set(ids.tolist()) == set(mapping)
    for fibre_id in ids[::131]:
        assert mapping[int(fibre_id)] == (layers[fibre_id], cells[fibre_id])


def test_rejects_unknown_configuration_keys(tmp_path) -> None:
    text = TRANSPORT.read_text(encoding="utf-8") + "\nsurprise: 1\n"
    path = tmp_path / "transport.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(TransportConfigError, match="unexpected"):
        load_transport_config(path)
