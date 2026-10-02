"""Burst latent, lateral spill and bulk coupling of the crossing proton."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from ams_ecal.crossing import FibreCrossingGeometry
from ams_ecal.event import ECALEvent
from ams_ecal.geometry import load_geometry
from ams_ecal.proton import MODEL_VERSION, MODEL_VERSION_STRUCTURED, ProtonShowerModel
from ams_ecal.proton_calibration import CrossingTable, ProtonCalibration, content_sha256
from ams_ecal.proton_config import load_proton_config
from ams_ecal.proton_structure import (
    BURST_UNIFORMS,
    K_MAX,
    N_LAYERS,
    SPILL_DRAWS,
    SPILL_R_EDGES,
    STRUCTURE_LEVELS,
    CrossingCalibrationInputs,
    CrossingStructure,
    build_structure,
    coupled_uniforms,
    coupling_matrix,
    detect_bursts,
    estimate_bulk_coupling,
    interpolate_levels,
    ln_energy_weights,
    sample_bursts,
    sample_spill,
    stretch_profile,
)
from ams_ecal.tracking import TrackState

ROOT = Path(__file__).parents[1]
GEOMETRY = ROOT / "configs" / "geometry.yaml"
CONFIG = ROOT / "configs" / "fastmc_proton.yaml"
ENERGIES = (10.0, 100.0)
BULK_LEVELS = np.array([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])


# --- synthetic calibration events with a PLANTED structure ----------------------------


def synthetic_inputs(
    energy_gev: float, n: int = 1500, seed: int = 0, burst_probability: float = 0.2
) -> CrossingCalibrationInputs:
    """Bulk gamma layers (mean 8 MeV deposit, 0.6 MeV fibre), a planted burst, planted spill.

    The bulk never reaches three times its median, so every detected burst is a planted one.
    """

    rng = np.random.default_rng(seed)
    chord = rng.uniform(2.0, 4.0, (n, N_LAYERS))
    deposition = rng.gamma(25.0, 8.0 / 25.0, (n, N_LAYERS))
    readout = rng.gamma(40.0, 0.6 / 40.0, (n, N_LAYERS))
    has = rng.random(n) < burst_probability
    onset = rng.integers(0, N_LAYERS, n)
    offset = np.arange(N_LAYERS)[None, :] - onset[:, None]
    after = offset >= 0
    shape = np.where(after, (offset + 1.0) * np.exp(-offset / 3.0), 0.0)
    shape = shape / shape.max()
    amplitude = rng.uniform(80.0, 240.0, n)
    burst = np.where(has[:, None], amplitude[:, None] * shape, 0.0)
    deposition = deposition + burst
    readout = readout + 0.06 * burst

    lit = has[:, None] & after & (rng.random((n, N_LAYERS)) < 0.5)
    count = lit.astype(int)
    distance = np.zeros((n, N_LAYERS, K_MAX), dtype=np.int64)
    distance[..., 0] = 2 * count
    return CrossingCalibrationInputs(
        energy_gev=energy_gev,
        chord_mm=chord,
        layer_energy_mev={"deposition": deposition, "readout": readout},
        spill_count={"deposition": count, "readout": count},
        spill_fraction={"deposition": 0.3 * count, "readout": 0.3 * count},
        spill_distance={"deposition": distance, "readout": distance},
    )


@pytest.fixture(scope="module")
def built():
    inputs = [synthetic_inputs(e, seed=i) for i, e in enumerate(ENERGIES)]
    structure, bulk, edges, counts = build_structure(
        inputs, n_chord_bins=3, bulk_levels=BULK_LEVELS, min_layers_per_bin=20
    )
    return structure, bulk, edges, counts, inputs


# --- numerical helpers ------------------------------------------------------------------


def test_energy_weights_interpolate_in_the_logarithm_and_refuse_extrapolation() -> None:
    lower, upper, weight = ln_energy_weights(
        np.array([10.0, 100.0]), np.array([10.0, 31.6227766, 100.0])
    )

    assert list(lower) == [0, 0, 0] and list(upper) == [1, 1, 1]
    assert weight == pytest.approx([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="nothing is extrapolated"):
        ln_energy_weights(np.array([10.0, 100.0]), np.array([9.9]))


def test_interpolation_of_quantile_functions_is_monotone_and_bounded() -> None:
    levels = np.array([0.0, 0.5, 1.0])
    values = np.array([[1.0, 2.0, 6.0]])
    u = np.linspace(0.0, 1.0, 101)

    out = interpolate_levels(levels, np.broadcast_to(values, (101, 3)), u)

    assert out[0] == 1.0 and out[-1] == 6.0
    assert np.all(np.diff(out) >= 0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        interpolate_levels(levels, values, np.array([1.2]))


def test_a_stretch_of_one_is_the_identity_and_longer_means_longer() -> None:
    profile = np.exp(-np.arange(N_LAYERS) / 3.0)

    same = stretch_profile(profile, np.array([1.0]))[0]
    longer = stretch_profile(profile, np.array([2.0]))[0]
    shorter = stretch_profile(profile, np.array([0.5]))[0]

    assert np.allclose(same, profile)
    assert longer.sum() > same.sum() > shorter.sum()
    assert longer[0] == pytest.approx(profile[0])  # the peak does not move


def test_the_coupling_matrix_is_a_valid_correlation_matrix() -> None:
    matrix = coupling_matrix(np.array([0.2, 0.15, 0.09]))

    assert np.allclose(np.diag(matrix), 1.0)
    assert np.allclose(matrix, matrix.T)
    assert np.linalg.eigvalsh(matrix).min() > 0
    assert matrix[0, 1] == 0.2 and matrix[1, 2] == 0.15 and matrix[0, 2] == 0.09
    assert matrix[0, 5] == 0.0


def test_an_impossible_coupling_is_shrunk_until_it_is_positive_definite() -> None:
    matrix = coupling_matrix(np.array([0.95, 0.95, 0.95]))

    assert np.linalg.eigvalsh(matrix).min() >= 1e-6 - 1e-12


def test_the_estimated_coupling_recovers_a_planted_pair_correlation() -> None:
    rng = np.random.default_rng(1)
    z = rng.standard_normal((6000, N_LAYERS))
    for layer in range(0, N_LAYERS, 2):  # the two layers of a superlayer share a component
        shared = rng.standard_normal(6000)
        z[:, layer] = np.sqrt(0.7) * z[:, layer] + np.sqrt(0.3) * shared
        z[:, layer + 1] = np.sqrt(0.7) * z[:, layer + 1] + np.sqrt(0.3) * shared

    within, across, second = estimate_bulk_coupling(np.exp(z))

    assert within == pytest.approx(0.3, abs=0.04)
    assert abs(across) < 0.04 and abs(second) < 0.04


def test_coupled_uniforms_are_uniform_and_carry_the_correlation() -> None:
    rng = np.random.default_rng(2)
    factor = np.linalg.cholesky(coupling_matrix(np.array([0.5, 0.0, 0.0])))

    u = coupled_uniforms(factor, rng.standard_normal((20000, N_LAYERS)))

    assert stats.kstest(u[:, 3], "uniform").pvalue > 0.001
    assert stats.spearmanr(u[:, 0], u[:, 1]).statistic == pytest.approx(
        6 / np.pi * np.arcsin(0.25), abs=0.03
    )
    assert abs(stats.spearmanr(u[:, 1], u[:, 2]).statistic) < 0.03


# --- the builder ------------------------------------------------------------------------


def test_the_builder_recovers_the_planted_burst_probability(built) -> None:
    structure, *_ = built

    assert structure.burst.probability == pytest.approx([0.2, 0.2], abs=0.04)
    assert structure.burst.counts.min() > 200
    assert structure.burst.onset_pmf.sum() == pytest.approx(1.0)
    # a planted burst starts anywhere, but a detected burst cannot start behind the last layer
    assert structure.burst.onset_pmf.min() > 0


def test_the_bulk_table_is_built_from_burst_free_events_only(built) -> None:
    _, bulk, _, counts, inputs = built

    for name in ("deposition", "readout"):
        # the planted bursts add at least 80 MeV (deposition); the bulk tops out near 8 MeV x 1.5
        ceiling = 20.0 if name == "deposition" else 1.5
        assert bulk[name].max() < ceiling
    assert counts.min() >= 20
    total_layers = sum(entry.n_events * N_LAYERS for entry in inputs)
    assert counts.sum() < 0.9 * total_layers  # layers of burst events were left out


def test_the_spill_probability_rises_with_the_layers_excess(built) -> None:
    structure, *_ = built

    spill = structure.spill["readout"]
    assert spill.probability[0] < 0.01
    assert spill.probability[-1] > 0.3
    assert np.all(np.diff(spill.probability) >= -0.05)  # broadly increasing
    assert spill.cells_pmf.sum(axis=1) == pytest.approx(1.0)
    assert spill.fraction_quantiles[-1, -1] == pytest.approx(0.3)


def test_the_bulk_coupling_is_small_for_independent_planted_layers(built) -> None:
    structure, *_ = built

    for name in ("readout", "deposition"):
        assert np.all(np.abs(structure.bulk_coupling[name]) < 0.08)


def test_detection_starts_a_burst_at_the_first_layer_above_three_times_the_reference() -> None:
    energy = np.full((2, N_LAYERS), 8.0)
    energy[0, 5] = 30.0
    energy[0, 9] = 50.0

    has, onset = detect_bursts(energy, np.full((2, N_LAYERS), 8.0))

    assert list(has) == [True, False]
    assert onset[0] == 5


def test_too_few_bursts_are_refused() -> None:
    quiet = [synthetic_inputs(e, seed=3, burst_probability=0.001) for e in ENERGIES]

    with pytest.raises(ValueError, match="too few to calibrate"):
        build_structure(quiet, n_chord_bins=3, bulk_levels=BULK_LEVELS)


def test_too_little_burst_free_data_in_a_chord_bin_is_refused(built) -> None:
    *_, inputs = built

    with pytest.raises(ValueError, match="burst-free layers"):
        build_structure(inputs, n_chord_bins=3, bulk_levels=BULK_LEVELS, min_layers_per_bin=10_000)


def test_the_structure_survives_a_round_trip_through_its_arrays(built) -> None:
    structure, *_ = built

    again = CrossingStructure.from_arrays(structure.arrays(), structure.burst.energies_gev)

    for name, array in structure.arrays().items():
        assert np.array_equal(array, again.arrays()[name]), name


def test_the_builder_is_deterministic(built) -> None:
    structure, _, _, _, inputs = built

    again, *_ = build_structure(
        inputs, n_chord_bins=3, bulk_levels=BULK_LEVELS, min_layers_per_bin=20
    )

    assert content_sha256(again.arrays()) == content_sha256(structure.arrays())


def test_malformed_tables_are_rejected(built) -> None:
    structure, *_ = built

    with pytest.raises(ValueError, match="onset_pmf"):
        replace(structure.burst, onset_pmf=structure.burst.onset_pmf[:-1])
    with pytest.raises(ValueError, match="non-decreasing"):
        replace(structure.burst, share_quantiles=structure.burst.share_quantiles[:, ::-1])
    with pytest.raises(ValueError, match="sum to 1"):
        replace(structure.spill["readout"], cells_pmf=structure.spill["readout"].cells_pmf * 0.5)


# --- sampling ---------------------------------------------------------------------------


def draw_bursts(structure, energy_gev: float, n: int, seed: int):
    rng = np.random.default_rng(seed)
    return sample_bursts(
        structure.burst,
        np.full(n, energy_gev),
        rng.random((n, BURST_UNIFORMS)),
        rng.standard_normal((n, N_LAYERS)),
    )


def test_a_burst_adds_nothing_in_front_of_its_onset_or_when_absent(built) -> None:
    structure, *_ = built

    draw = draw_bursts(structure, 30.0, 4000, 1)

    excess = draw.excess_mev["deposition"]
    assert np.all(excess[~draw.has_burst] == 0.0)
    layer = np.arange(N_LAYERS)[None, :]
    assert np.all(excess[layer < draw.onset[:, None]] == 0.0)
    assert np.all(excess >= 0.0)
    assert draw.has_burst.mean() == pytest.approx(0.2, abs=0.03)


def test_a_late_burst_deposits_less_inside_the_ecal_than_an_early_one(built) -> None:
    structure, *_ = built

    draw = draw_bursts(structure, 30.0, 20000, 2)

    total = draw.excess_mev["deposition"].sum(axis=1)
    early = draw.has_burst & (draw.onset <= 4)
    late = draw.has_burst & (draw.onset >= 14)
    assert total[early].mean() > 1.5 * total[late].mean()


def test_the_fibre_excess_is_a_share_of_the_deposition_excess(built) -> None:
    structure, *_ = built

    draw = draw_bursts(structure, 30.0, 4000, 3)

    dep, ro = draw.excess_mev["deposition"], draw.excess_mev["readout"]
    both = dep.sum(axis=1) > 0
    ratio = ro[both].sum(axis=1) / dep[both].sum(axis=1)
    assert ratio.min() >= 0.0
    assert np.median(ratio) == pytest.approx(0.06, abs=0.02)
    # a layer's share is the burst's share: the two excesses are proportional layer by layer
    assert np.allclose(
        ro[both] * dep[both].sum(axis=1)[:, None], dep[both] * ro[both].sum(axis=1)[:, None]
    )


def test_the_burst_draw_is_a_function_of_its_variates(built) -> None:
    structure, *_ = built
    rng = np.random.default_rng(4)
    u, z = rng.random((50, BURST_UNIFORMS)), rng.standard_normal((50, N_LAYERS))
    energies = np.full(50, 30.0)

    a = sample_bursts(structure.burst, energies, u, z)
    b = sample_bursts(structure.burst, energies, u, z)

    assert np.array_equal(a.excess_mev["deposition"], b.excess_mev["deposition"])
    assert np.array_equal(a.onset, b.onset)


def test_a_larger_amplitude_variate_never_gives_a_smaller_burst(built) -> None:
    structure, *_ = built
    n = 51
    u = np.tile(np.array([0.0, 0.2, 0.5, 0.5, 0.5, 0.5]), (n, 1))  # always a burst, same shape class
    u[:, 3] = np.linspace(0.0, 1.0, n)

    draw = sample_bursts(structure.burst, np.full(n, 30.0), u, np.zeros((n, N_LAYERS)))

    peak = draw.excess_mev["deposition"].max(axis=1)
    assert np.all(np.diff(peak) >= -1e-9)


def test_the_burst_draw_refuses_energies_outside_the_calibrated_range(built) -> None:
    structure, *_ = built

    with pytest.raises(ValueError, match="nothing is extrapolated"):
        draw_bursts(structure, 200.0, 4, 5)


def spill_draw(structure, ratio: float, n: int, seed: int = 6):
    rng = np.random.default_rng(seed)
    return sample_spill(
        structure.spill["readout"],
        np.full((n, N_LAYERS), ratio),
        rng.random((n, N_LAYERS, SPILL_DRAWS)),
    )


def test_a_spill_is_a_bounded_split_of_a_layer(built) -> None:
    structure, *_ = built

    draw = spill_draw(structure, 6.0, 500)

    assert draw.fraction.min() >= 0.0 and draw.fraction.max() <= 1.0
    assert draw.cells.max() <= K_MAX
    assert np.all(draw.fraction[draw.cells == 0] == 0.0)
    used = np.arange(K_MAX)[None, None, :] < draw.cells[..., None]
    assert np.allclose(draw.weight.sum(axis=-1)[draw.cells > 0], 1.0)
    assert np.all(draw.weight[~used] == 0.0)
    assert np.all(draw.offset[~used] == 0)
    assert np.all(np.abs(draw.offset[used]) >= 1)


def test_big_layers_spill_more_often_than_ordinary_ones(built) -> None:
    structure, *_ = built

    ordinary = (spill_draw(structure, 0.9, 1000).cells > 0).mean()
    big = (spill_draw(structure, 12.0, 1000).cells > 0).mean()

    assert ordinary < 0.02
    assert big > 0.3


def test_the_spill_rejects_malformed_draws(built) -> None:
    structure, *_ = built

    with pytest.raises(ValueError, match="uniforms must have shape"):
        sample_spill(structure.spill["readout"], np.ones((2, N_LAYERS)), np.zeros((2, N_LAYERS, 3)))


# --- placement on the cells --------------------------------------------------------------


@pytest.fixture(scope="module")
def crossing() -> FibreCrossingGeometry:
    return FibreCrossingGeometry(load_geometry(GEOMETRY))


def track(x: float = 4.5, y: float = 4.5) -> TrackState:
    return TrackState(x0_mm=x, y0_mm=y, z0_mm=0.0, theta_rad=0.0, phi_rad=0.0)


def place(crossing, layer_energy, fraction, offset, weight, cells, tr=None):
    tr = tr or track()
    crossed = crossing.cross(tr)
    grid = crossing.place_layer_energies(tr, crossed, layer_energy, fraction, offset, weight, cells)
    return grid, crossed


def no_spill():
    return (
        np.zeros(N_LAYERS),
        np.zeros((N_LAYERS, K_MAX), dtype=int),
        np.zeros((N_LAYERS, K_MAX)),
        np.zeros(N_LAYERS, dtype=int),
    )


def test_without_spill_the_placement_is_the_original_spreading(crossing) -> None:
    energy = np.linspace(0.5, 2.0, N_LAYERS)
    tr = track()

    grid, crossed = place(crossing, energy, *no_spill(), tr)

    assert np.allclose(grid, crossing.spread_layer_energies(tr, crossed, energy))


def test_a_spill_conserves_every_layers_energy_and_leaves_the_crossed_cells_their_share(
    crossing,
) -> None:
    energy = np.full(N_LAYERS, 1.0)
    fraction = np.full(N_LAYERS, 0.25)
    cells = np.full(N_LAYERS, 2)
    offset = np.zeros((N_LAYERS, K_MAX), dtype=int)
    offset[:, 0], offset[:, 1] = 3, -5
    weight = np.zeros((N_LAYERS, K_MAX))
    weight[:, 0], weight[:, 1] = 0.6, 0.4

    grid, crossed = place(crossing, energy, fraction, offset, weight, cells)

    assert np.allclose(grid.sum(axis=1), 1.0)
    for layer in range(N_LAYERS):
        crossed_cells = np.unique(crossed.cell[(crossed.layer == layer) & (crossed.cell >= 0)])
        if len(crossed_cells) == 0:
            crossed_cells = np.array([crossing.track_cell(track(), layer)])
        assert grid[layer, crossed_cells].sum() == pytest.approx(0.75)
        assert (grid[layer] > 0).sum() == len(crossed_cells) + 2


def test_a_spill_off_the_edge_of_the_grid_is_reflected_and_stays_in_the_grid(crossing) -> None:
    tr = track(x=-323.0, y=-323.0)  # the first cell of every layer
    energy = np.full(N_LAYERS, 1.0)
    fraction = np.full(N_LAYERS, 0.3)
    cells = np.full(N_LAYERS, 1)
    offset = np.full((N_LAYERS, K_MAX), -4)
    weight = np.zeros((N_LAYERS, K_MAX))
    weight[:, 0] = 1.0

    grid, _ = place(crossing, energy, fraction, offset, weight, cells, tr)

    assert np.allclose(grid.sum(axis=1), 1.0)
    assert np.all(grid[:, 4:8].sum(axis=1) > 0.29)  # reflected to the positive side


def test_two_spill_cells_never_land_on_the_same_cell(crossing) -> None:
    energy = np.full(N_LAYERS, 1.0)
    fraction = np.full(N_LAYERS, 0.4)
    cells = np.full(N_LAYERS, 3)
    offset = np.full((N_LAYERS, K_MAX), 2)
    weight = np.zeros((N_LAYERS, K_MAX))
    weight[:, :3] = 1.0 / 3.0

    grid, crossed = place(crossing, energy, fraction, offset, weight, cells)

    assert np.allclose(grid.sum(axis=1), 1.0)
    for layer in range(N_LAYERS):
        crossed_cells = np.unique(crossed.cell[(crossed.layer == layer) & (crossed.cell >= 0)])
        assert (grid[layer] > 0).sum() == max(len(crossed_cells), 1) + 3


def test_the_placement_rejects_a_negative_layer_energy(crossing) -> None:
    energy = np.full(N_LAYERS, 1.0)
    energy[3] = -1.0

    with pytest.raises(ValueError, match="nonnegative"):
        place(crossing, energy, *no_spill())


# --- the generator with the structure ------------------------------------------------------


@pytest.fixture(scope="module")
def model(built) -> ProtonShowerModel:
    structure, bulk, edges, counts, _ = built
    table = CrossingTable(
        energies_gev=np.array(ENERGIES),
        chord_edges_mm=edges,
        levels=BULK_LEVELS,
        quantiles_mev=bulk,
        counts=counts,
    )
    manifest = {
        "schema_version": 2,
        "physics_list": "ftfp_bert",
        "content_sha256": "b" * 64,
        "source": {"geant4_version": "test-11.4"},
    }
    calibration = ProtonCalibration(manifest, table, 255.0, 3.0, 166.5, structure)
    return ProtonShowerModel(
        load_proton_config(CONFIG),
        calibration,
        FibreCrossingGeometry(load_geometry(GEOMETRY)),
        "0" * 64,
    )


def generate(model: ProtonShowerModel, seed: int, energy: float = 30_000.0) -> ECALEvent:
    return model.generate_crossing_event(
        event_id=f"s-{seed}", primary_energy_mev=energy, track=track(), random_seed=seed
    )


def test_a_structured_event_is_a_valid_event_and_says_so(model) -> None:
    event = generate(model, 1)

    assert np.array(event.cell_energies_mev).shape == (N_LAYERS, 72)
    assert np.all(np.array(event.cell_energies_mev) >= 0)
    details = dict(event.provenance.model_details)
    assert details["model_version"] == MODEL_VERSION_STRUCTURED
    assert event.provenance.simulation_version.endswith(MODEL_VERSION_STRUCTURED)
    assert "burst_onset_layer" in details and "spill_layers" in details


def test_the_same_seed_gives_the_same_structured_event(model) -> None:
    a, b = generate(model, 7), generate(model, 7)

    assert a.cell_energies_mev == b.cell_energies_mev
    assert generate(model, 8).cell_energies_mev != a.cell_energies_mev


def test_one_seed_names_the_same_burst_in_both_representations(model) -> None:
    for seed in range(40):
        readout = generate(model, seed)
        deposition = generate(model.as_representation("deposition"), seed)

        assert (
            dict(readout.provenance.model_details)["burst_onset_layer"]
            == dict(deposition.provenance.model_details)["burst_onset_layer"]
        )


def test_the_deposition_representation_is_the_larger_energy_scale(model) -> None:
    deposition = model.as_representation("deposition")

    readout_total = np.mean([generate(model, s).total_ecal_energy_mev for s in range(30)])
    deposition_total = np.mean([generate(deposition, s).total_ecal_energy_mev for s in range(30)])

    assert deposition_total > 8 * readout_total


def test_bursts_make_the_event_total_heavy_tailed_and_spill_adds_hit_cells(model) -> None:
    totals, hit_cells = [], []
    for seed in range(300):
        event = generate(model.as_representation("deposition"), seed)
        grid = np.array(event.cell_energies_mev)
        totals.append(grid.sum())
        hit_cells.append((grid > 0.28).sum())

    assert np.max(totals) > 1.3 * np.median(totals)
    assert np.max(hit_cells) > N_LAYERS  # more than one cell per layer somewhere


def test_a_schema_one_calibration_still_takes_the_independent_layer_path(model) -> None:
    plain = replace(model, calibration=replace(model.calibration, structure=None))

    event = generate(plain, 3)

    assert dict(event.provenance.model_details)["model_version"] == MODEL_VERSION
    assert "burst_onset_layer" not in dict(event.provenance.model_details)


def test_the_levels_cover_zero_to_one() -> None:
    assert STRUCTURE_LEVELS[0] == 0.0 and STRUCTURE_LEVELS[-1] == 1.0
    assert np.all(np.diff(STRUCTURE_LEVELS) > 0)
    assert SPILL_R_EDGES.tolist() == sorted(SPILL_R_EDGES.tolist())
