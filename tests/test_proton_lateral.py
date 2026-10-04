"""Lateral structure of interacting protons: kernel, quanta, builder and event placement."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from ams_ecal.crossing import FibreCrossingGeometry
from ams_ecal.geometry import load_geometry
from ams_ecal.proton_interacting import interaction_layer
from ams_ecal.proton_lateral import (
    EQUAL_WEIGHTS,
    HALO_DISTANCES,
    N_CELLS,
    LateralCalibrationInputs,
    LateralTable,
    build_lateral,
    cell_probabilities,
    core_fractions,
    event_grid,
    extract_lateral_inputs,
    layer_statistics,
    place_quanta,
    track_cells,
)
from ams_ecal.proton_structure import N_LAYERS
from ams_ecal.tracking import TrackState

ROOT = Path(__file__).parents[1]
DEPTH = 166.5
ENERGIES = (10.0, 100.0)
HALO = np.arange(2, N_CELLS, dtype=float) ** -1.6
HALO = HALO / HALO.sum()


def planted_inputs(
    energy_gev: float, n: int = 700, seed: int = 0, tail_index: float = EQUAL_WEIGHTS
) -> LateralCalibrationInputs:
    """Events whose layers were placed with KNOWN lateral parameters."""

    rng = np.random.default_rng(seed)
    depth = rng.uniform(0.0, DEPTH, n)
    l_d = interaction_layer(depth, DEPTH)
    centre_cells = np.full((n, N_LAYERS), 36)
    grids = np.zeros((n, N_LAYERS, N_CELLS))
    for i in range(n):
        event = 0.6 * rng.standard_normal()  # the planted event-level latent
        for layer in range(l_d[i], N_LAYERS):
            energy = float(np.exp(rng.normal(4.0, 0.7)))
            core = float(
                expit(0.5 + event + 0.3 * rng.standard_normal() + 0.2 * (np.log(energy) - 4.0))
            )
            grids[i, layer] = place_quanta(
                rng,
                np.array([energy]),
                np.array([core]),
                np.array([36]),
                np.array([0.55]),
                HALO,
                0.5,
                tail_index,
            )[0]
    return LateralCalibrationInputs(
        energy_gev=energy_gev,
        depth_mm=depth,
        grid_mev={"readout": grids, "deposition": 11.0 * grids},
        track_cell=centre_cells,
    )


@pytest.fixture(scope="module")
def inputs():
    return [planted_inputs(e, seed=i) for i, e in enumerate(ENERGIES)]


@pytest.fixture(scope="module")
def table(inputs) -> LateralTable:
    return build_lateral(inputs, ecal_depth_mm=DEPTH, seed=1)


# --- the kernel -------------------------------------------------------------------------


def test_cell_probabilities_are_a_distribution_with_the_core_on_and_beside_the_centre() -> None:
    probability = cell_probabilities(np.array([0.6]), np.array([0.5]), HALO, np.array([36]))[0]

    assert probability.sum() == pytest.approx(1.0)
    # ratios are exact; the absolute values rise ~2% because halo mass beyond the grid is dropped
    assert probability[36] / probability[35] == pytest.approx(2.0)
    assert probability[35] == pytest.approx(probability[37])
    assert probability[34:39].sum() > 0.6  # the core and the first halo cells
    assert probability[34] == pytest.approx(probability[38])  # symmetric halo
    assert probability[34] > probability[30] > probability[10]  # a power-law halo


def test_probability_off_the_grid_is_dropped_and_the_rest_renormalised() -> None:
    probability = cell_probabilities(np.array([0.9]), np.array([0.5]), HALO, np.array([0]))[0]

    assert probability.sum() == pytest.approx(1.0)
    assert probability[0] > probability[1] > probability[2]
    assert np.all(probability >= 0)


def test_a_fully_concentrated_layer_puts_everything_in_the_core() -> None:
    probability = cell_probabilities(np.array([1.0]), np.array([1.0]), HALO, np.array([36]))[0]

    assert probability[36] == pytest.approx(1.0)


# --- placing quanta ---------------------------------------------------------------------


def place(rng, energy, tail_index=EQUAL_WEIGHTS, core=0.6, quantum=0.5):
    n = len(energy)
    return place_quanta(
        rng,
        np.asarray(energy, dtype=float),
        np.full(n, core),
        np.full(n, 36),
        np.full(n, 0.5),
        HALO,
        quantum,
        tail_index,
    )


def test_each_layer_keeps_exactly_its_energy() -> None:
    rng = np.random.default_rng(1)
    energy = np.array([0.3, 5.0, 80.0, 900.0])

    for tail in (EQUAL_WEIGHTS, 1.0, 0.5):
        grid = place(rng, energy, tail)
        assert grid.sum(axis=1) == pytest.approx(energy, rel=1e-12)
        assert np.all(grid >= 0.0)


def test_a_layer_without_energy_stays_empty() -> None:
    grid = place(np.random.default_rng(2), [0.0, 4.0])

    assert grid[0].sum() == 0.0 and grid[1].sum() == pytest.approx(4.0)


def test_the_placement_is_deterministic_in_its_generator() -> None:
    a = place(np.random.default_rng(3), [10.0, 20.0])
    b = place(np.random.default_rng(3), [10.0, 20.0])

    assert np.array_equal(a, b)
    assert not np.array_equal(a, place(np.random.default_rng(4), [10.0, 20.0]))


def test_a_weak_layer_lights_few_cells_and_a_strong_one_many() -> None:
    rng = np.random.default_rng(5)

    weak = place(rng, np.full(300, 0.4))
    strong = place(rng, np.full(300, 400.0))

    assert (weak > 0.01).sum(axis=1).mean() < 3.0
    assert (strong > 0.01).sum(axis=1).mean() > 10.0


def test_a_heavy_tail_makes_one_cell_dominate() -> None:
    rng = np.random.default_rng(6)
    energy = np.full(400, 200.0)

    equal = layer_statistics(place(rng, energy, EQUAL_WEIGHTS), np.full(400, 36))["top"]
    heavy = layer_statistics(place(rng, energy, 0.5), np.full(400, 36))["top"]

    assert np.median(heavy) > np.median(equal) + 0.1


def test_the_spread_follows_the_core_fraction() -> None:
    rng = np.random.default_rng(7)
    energy = np.full(300, 300.0)

    tight = layer_statistics(place(rng, energy, core=0.9), np.full(300, 36))["core"]
    wide = layer_statistics(place(rng, energy, core=0.3), np.full(300, 36))["core"]

    assert np.median(tight) > np.median(wide) + 0.3


# --- the builder -------------------------------------------------------------------------


def test_the_builder_recovers_the_planted_core_fraction(table) -> None:
    # logit core was planted as 0.5 + event + noise (+ 0.2 per unit ln energy, which is mean zero)
    assert np.all(np.abs(table.logit_mean[:, 0, 1:6] - 0.5) < 0.6)


def test_the_builder_recovers_the_planted_event_level_scatter(table) -> None:
    # the planted event latent has sd 0.6 and the layer term 0.3; quantisation adds layer noise
    assert np.all((table.event_sd > 0.3) & (table.event_sd < 1.0))
    assert np.all(table.layer_sd > 0.2)


def test_the_builder_finds_the_positive_energy_slope(table) -> None:
    assert np.all(table.energy_slope[:, 0] > 0.0)


def test_the_builder_prefers_equal_weights_when_the_planted_quanta_are_equal(table) -> None:
    assert table.tail_index[0] >= 2.0


def test_the_builder_finds_a_heavy_tail_when_one_is_planted() -> None:
    heavy = [planted_inputs(e, n=500, seed=10 + i, tail_index=0.7) for i, e in enumerate(ENERGIES)]

    fitted = build_lateral(heavy, ecal_depth_mm=DEPTH, seed=2)

    assert fitted.tail_index[0] <= 1.4


def test_the_table_survives_a_round_trip_through_its_arrays(table) -> None:
    again = LateralTable.from_arrays(table.arrays())

    for name, array in table.arrays().items():
        assert np.array_equal(array, again.arrays()[name]), name


def test_the_build_is_deterministic(inputs, table) -> None:
    again = build_lateral(inputs, ecal_depth_mm=DEPTH, seed=1)

    for name, array in table.arrays().items():
        assert np.array_equal(array, again.arrays()[name]), name


def test_malformed_tables_are_rejected(table) -> None:
    with pytest.raises(ValueError, match="halo pmf"):
        replace(table, halo_pmf=table.halo_pmf * 0.5)
    with pytest.raises(ValueError, match="positive"):
        replace(table, quantum_mev=np.zeros(2))
    with pytest.raises(ValueError, match="shape"):
        replace(table, counts=table.counts[:1])
    with pytest.raises(ValueError, match="finite"):
        replace(table, energy_slope=np.full_like(table.energy_slope, np.nan))
    assert HALO_DISTANCES == N_CELLS - 2


def test_too_few_populated_layers_are_refused() -> None:
    few = [planted_inputs(e, n=3, seed=20 + i) for i, e in enumerate(ENERGIES)]

    with pytest.raises(ValueError, match="well-populated layers"):
        build_lateral(few, ecal_depth_mm=DEPTH)


def test_inputs_with_the_wrong_shape_are_rejected(inputs) -> None:
    good = inputs[0]

    with pytest.raises(ValueError, match="track_cell"):
        replace(good, track_cell=good.track_cell[:, :5])
    with pytest.raises(ValueError, match="grid_mev"):
        replace(
            good,
            grid_mev={
                "readout": good.grid_mev["readout"][:, :, :5],
                "deposition": good.grid_mev["deposition"],
            },
        )


# --- sampling ----------------------------------------------------------------------------


def fractions(table, energy_mev, event=0.0, offsets=None):
    energy = np.atleast_2d(np.asarray(energy_mev, dtype=float))
    offsets = np.zeros(energy.shape, dtype=int) if offsets is None else offsets
    return core_fractions(
        table, 0, np.array([50.0]), offsets, energy, np.array([event]), np.zeros(energy.shape)
    )


def test_a_larger_layer_energy_gives_a_larger_core_fraction_with_a_positive_slope(table) -> None:
    assert np.all(fractions(table, [[500.0, 500.0]]) > fractions(table, [[5.0, 5.0]]))


def test_the_event_level_latent_moves_every_layer_together(table) -> None:
    offsets = np.arange(6)[None, :]
    energy = np.full((1, 6), 50.0)

    narrow = fractions(table, energy, event=-2.0, offsets=offsets)
    broad = fractions(table, energy, event=2.0, offsets=offsets)

    assert np.all(broad > narrow)


def test_the_event_grid_fills_only_the_layers_behind_the_interaction_and_keeps_their_energy(
    table,
) -> None:
    energy = np.linspace(5.0, 200.0, N_LAYERS)

    grid = event_grid(
        table, 0, np.random.default_rng(8), 50.0, energy, 7, np.full(N_LAYERS, 36), 0.3, np.zeros(N_LAYERS)
    )

    assert np.all(grid[:7] == 0.0)
    assert grid[7:].sum(axis=1) == pytest.approx(energy[7:])


def test_an_interaction_in_the_last_layer_places_only_that_layer(table) -> None:
    energy = np.full(N_LAYERS, 30.0)

    grid = event_grid(
        table, 1, np.random.default_rng(9), 20.0, energy, 17, np.full(N_LAYERS, 36), 0.0, np.zeros(N_LAYERS)
    )

    assert np.count_nonzero(grid.sum(axis=1)) == 1 and grid[17].sum() == pytest.approx(30.0)


def test_the_event_grid_refuses_an_energy_outside_the_calibrated_range(table) -> None:
    with pytest.raises(ValueError, match="nothing is extrapolated"):
        event_grid(
            table,
            0,
            np.random.default_rng(1),
            500.0,
            np.full(N_LAYERS, 5.0),
            3,
            np.full(N_LAYERS, 36),
            0.0,
            np.zeros(N_LAYERS),
        )


# --- geometry-bound helpers ---------------------------------------------------------------


@pytest.fixture(scope="module")
def crossing() -> FibreCrossingGeometry:
    return FibreCrossingGeometry(load_geometry(ROOT / "configs" / "geometry.yaml"))


def test_the_track_cell_of_every_layer_is_a_valid_cell(crossing) -> None:
    cells = track_cells(
        crossing, TrackState(x0_mm=4.5, y0_mm=-100.0, z0_mm=0.0, theta_rad=0.0, phi_rad=0.0)
    )

    assert cells.shape == (N_LAYERS,)
    assert np.all((cells >= 0) & (cells < N_CELLS))
    assert np.array_equal(cells[0::2], cells[1::2])  # the two layers of a superlayer share a view
    assert len(set(cells)) == 2  # and the superlayers alternate between the two views


def test_the_extractor_keeps_only_interacting_events_and_their_track_cells(crossing) -> None:
    n = 8
    rng = np.random.default_rng(11)
    arrays = {
        "truth_occurred": np.array([True, False] * 4),
        "truth_z_mm": rng.uniform(1, 160, n),
        "entry_x_mm": rng.uniform(0, 9, n),
        "entry_y_mm": rng.uniform(0, 9, n),
        "readout_grid_mev": rng.random((n, N_LAYERS, N_CELLS)).astype(np.float32),
        "deposit_grid_mev": rng.random((n, N_LAYERS, N_CELLS)).astype(np.float32),
    }

    inputs = extract_lateral_inputs(20.0, arrays, crossing)

    assert inputs.n_events == 4 and inputs.energy_gev == 20.0
    assert inputs.grid_mev["readout"].shape == (4, N_LAYERS, N_CELLS)
    assert np.all((inputs.track_cell >= 0) & (inputs.track_cell < N_CELLS))
    assert np.allclose(inputs.depth_mm, arrays["truth_z_mm"][arrays["truth_occurred"]])


def test_layer_statistics_describe_a_known_layer() -> None:
    grid = np.zeros((1, N_CELLS))
    grid[0, 36], grid[0, 35], grid[0, 40] = 6.0, 2.0, 2.0

    stats_ = layer_statistics(grid, np.array([36]))

    assert stats_["energy"][0] == pytest.approx(10.0)
    assert stats_["core"][0] == pytest.approx(0.8)
    assert stats_["top"][0] == pytest.approx(0.6)
    assert stats_["hits"][0] == 3 and stats_["occupied"][0] == 3
