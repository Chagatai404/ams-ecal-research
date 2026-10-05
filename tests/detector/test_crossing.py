"""Exact fibre geometry of a straight track through the AMS ECAL."""

from itertools import pairwise
from math import pi
from pathlib import Path

import numpy as np
import pytest

from ams_ecal.detector.crossing import FibreCrossingGeometry
from ams_ecal.detector.geometry import load_geometry
from ams_ecal.detector.tracking import TrackState

GEOMETRY = Path(__file__).parents[2] / "configs" / "geometry.yaml"


@pytest.fixture(scope="module")
def crossing() -> FibreCrossingGeometry:
    return FibreCrossingGeometry(load_geometry(GEOMETRY))


def track(x: float, y: float, theta: float = 0.0) -> TrackState:
    return TrackState(x0_mm=x, y0_mm=y, z0_mm=0.0, theta_rad=theta, phi_rad=0.0)


# --- geometry of one row -------------------------------------------------


def test_a_track_through_a_fibre_centre_cuts_a_full_diameter(crossing) -> None:
    # Superlayer 0 has x-directed fibres, so it measures y.
    centre = crossing._centres[0][100]
    crossed = crossing.cross(track(x=3.0, y=float(centre)))

    first_row = crossed.depth_mm == crossing._row_depth[0, 0]

    assert first_row.sum() == 1
    assert crossed.chord_mm[first_row][0] == pytest.approx(2 * crossing.layout.fibre_radius_mm)


def test_a_track_between_fibres_crosses_none_in_that_row(crossing) -> None:
    pitch = crossing.geometry.sampling_structure.fiber_horizontal_pitch_mm
    gap = float(crossing._centres[0][100]) + pitch / 2  # midway between two fibres

    crossed = crossing.cross(track(x=3.0, y=gap))

    assert not (crossed.depth_mm == crossing._row_depth[0, 0]).any()


def test_at_most_one_fibre_per_row_and_five_per_layer(crossing) -> None:
    rng = np.random.default_rng(1)
    for _ in range(200):
        crossed = crossing.cross(track(rng.uniform(0, 9), rng.uniform(0, 9)))
        assert len(crossed) <= 90
        assert np.bincount(crossed.layer, minlength=18).max() <= 5


def test_the_mean_path_is_the_fibre_area_fraction(crossing) -> None:
    # A random line crosses a row of pitch p and radius r with mean chord
    # pi r^2 / p; ten rows per superlayer, nine superlayers.
    rng = np.random.default_rng(2)
    r = crossing.layout.fibre_radius_mm
    pitch = crossing.geometry.sampling_structure.fiber_horizontal_pitch_mm
    totals = [
        crossing.cross(track(rng.uniform(0, 9), rng.uniform(0, 9))).chord_mm.sum()
        for _ in range(1500)
    ]

    assert np.mean(totals) == pytest.approx(90 * pi * r**2 / pitch, rel=0.01)


def test_the_pattern_repeats_with_the_lattice_pitch(crossing) -> None:
    pitch = crossing.geometry.sampling_structure.fiber_horizontal_pitch_mm
    a = crossing.cross(track(1.1, 2.3)).layer_path_mm(18)
    b = crossing.cross(track(1.1 + pitch, 2.3 + pitch)).layer_path_mm(18)

    assert np.allclose(a, b, atol=1e-9)


def test_every_layer_is_crossed_by_something(crossing) -> None:
    rng = np.random.default_rng(3)
    minimum = min(
        crossing.cross(track(rng.uniform(0, 9), rng.uniform(0, 9))).layer_path_mm(18).min()
        for _ in range(500)
    )

    assert minimum > 1.0  # the pilot's smallest layer path was 1.87 mm


# --- cells and depth ------------------------------------------------------


def test_a_track_in_the_middle_of_a_cell_crosses_only_that_cell(crossing) -> None:
    crossed = crossing.cross(track(4.5, 4.5))

    assert set(crossed.cell.tolist()) == {36}


def test_a_track_on_a_cell_boundary_splits_between_the_neighbours(crossing) -> None:
    # Cell 35 | 36 lies at 0 mm. Unstaggered rows have fibres centred at
    # +-0.675 mm, staggered rows one centred exactly at 0 (which belongs to the
    # UPPER cell). At -0.3 mm the track crosses the -0.675 fibre (cell 35) and
    # the 0.0 fibre (cell 36).
    crossed = crossing.cross(track(-0.3, -0.3))

    assert set(crossed.cell.tolist()) == {35, 36}


def test_a_fibre_centred_exactly_on_a_boundary_goes_to_the_upper_cell(crossing) -> None:
    crossed = crossing.cross(track(0.0, 0.0))

    assert 35 not in set(crossed.cell.tolist())


def test_fibre_depths_lie_in_the_detector_and_increase_with_layer(crossing) -> None:
    crossed = crossing.cross(track(1.7, 6.2))
    mean_depth = [crossed.depth_mm[crossed.layer == layer].mean() for layer in range(18)]

    assert crossed.depth_mm.min() > 0
    assert crossed.depth_mm.max() < 166.5
    assert all(later > earlier for earlier, later in pairwise(mean_depth))


def test_upstream_selection_keeps_only_shallower_fibres(crossing) -> None:
    crossed = crossing.cross(track(1.7, 6.2))
    front = crossed.upstream_of(83.25)

    assert 0 < len(front) < len(crossed)
    assert front.depth_mm.max() < 83.25
    assert front.layer.max() <= 8


# --- spreading a layer's energy over cells --------------------------------


def test_layer_energy_is_conserved_and_stays_in_crossed_cells(crossing) -> None:
    t = track(-0.3, -0.3)
    crossed = crossing.cross(t)
    energy = np.linspace(0.3, 2.0, 18)

    grid = crossing.spread_layer_energies(t, crossed, energy)

    assert grid.shape == (18, 72)
    assert np.allclose(grid.sum(axis=1), energy)
    assert set(np.flatnonzero(grid.sum(axis=0)).tolist()) <= {35, 36}


def test_the_split_between_cells_follows_the_chord(crossing) -> None:
    t = track(-0.3, -0.3)
    crossed = crossing.cross(t)
    grid = crossing.spread_layer_energies(t, crossed, np.ones(18))

    layer = 5
    here = crossed.layer == layer
    expected = {
        int(cell): crossed.chord_mm[here & (crossed.cell == cell)].sum()
        / crossed.chord_mm[here].sum()
        for cell in set(crossed.cell[here].tolist())
    }
    for cell, share in expected.items():
        assert grid[layer, cell] == pytest.approx(share)


def test_a_layer_with_no_energy_stays_empty(crossing) -> None:
    t = track(4.5, 4.5)
    energy = np.ones(18)
    energy[3] = 0.0

    grid = crossing.spread_layer_energies(t, crossing.cross(t), energy)

    assert grid[3].sum() == 0.0


def test_energy_falls_back_to_the_track_cell_when_no_fibre_is_crossed(crossing) -> None:
    t = track(4.5, 4.5)
    crossed = crossing.cross(t)
    keep = crossed.layer != 7
    reduced = type(crossed)(
        crossed.layer[keep], crossed.cell[keep], crossed.chord_mm[keep], crossed.depth_mm[keep]
    )

    grid = crossing.spread_layer_energies(t, reduced, np.ones(18))

    assert grid[7, 36] == pytest.approx(1.0)


# --- refusals ---------------------------------------------------------------


def test_a_tilted_track_is_refused(crossing) -> None:
    with pytest.raises(ValueError, match="normally incident"):
        crossing.cross(track(1.0, 1.0, theta=0.1))


def test_a_track_outside_the_active_area_is_refused(crossing) -> None:
    with pytest.raises(ValueError, match="outside"):
        crossing.cross(track(400.0, 0.0))


def test_invalid_layer_energies_are_refused(crossing) -> None:
    t = track(4.5, 4.5)
    crossed = crossing.cross(t)

    with pytest.raises(ValueError, match="shape"):
        crossing.spread_layer_energies(t, crossed, np.ones(17))
    with pytest.raises(ValueError, match="nonnegative"):
        crossing.spread_layer_energies(t, crossed, -np.ones(18))
