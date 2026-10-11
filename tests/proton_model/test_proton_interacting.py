"""Layer energies of interacting protons: builder, mapping and sampler on planted structure."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import stats

from ams_ecal.proton_model.proton_calibration import CrossingTable
from ams_ecal.proton_model.proton_interacting import (
    FLOOR_MEV,
    INTERACTION_LEVELS,
    N_NORMALS,
    UPSTREAM,
    WINDOW,
    InteractingCalibrationInputs,
    InteractingTable,
    _post_log_energies,
    aligned_log_energy,
    amplitude_proxy,
    build_interacting,
    depth_group,
    fit_factor_chain,
    gaussian_correlation,
    interaction_layer,
    invert_quantiles,
    normal_scores,
    pairwise_lag_correlation,
    sample_interacting_layers,
)
from ams_ecal.proton_model.proton_structure import N_LAYERS

DEPTH = 166.5
ENERGIES = (10.0, 100.0)
CROSSING_LEVELS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def synthetic_inputs(energy_gev: float, n: int = 1500, seed: int = 0) -> InteractingCalibrationInputs:
    """A shower whose layers share a skewed amplitude and an AR chain, with an albedo upstream."""

    rng = np.random.default_rng(seed)
    depth = rng.uniform(0.0, DEPTH, n)
    l_d = interaction_layer(depth, DEPTH)
    skewed = 1.5 - rng.gamma(2.0, 0.4, n)  # left-skewed log amplitude
    ln_amp = np.log(energy_gev) + skewed - 0.12 * (depth > 111.0)
    chain = np.empty((n, N_LAYERS))
    chain[:, 0] = rng.standard_normal(n)
    for k in range(1, N_LAYERS):
        chain[:, k] = 0.8 * chain[:, k - 1] + 0.6 * rng.standard_normal(n)
    profile = 0.3 * np.minimum(np.arange(N_LAYERS), 6)
    layer = np.arange(N_LAYERS)[None, :]
    offset = layer - l_d[:, None]
    shower = np.exp(ln_amp[:, None] + profile[np.clip(offset, 0, N_LAYERS - 1)] + 0.3 * chain)
    albedo = (
        0.12
        * np.exp(ln_amp)[:, None]
        * np.exp(-0.5 * (-offset))
        * np.exp(0.3 * rng.standard_normal((n, N_LAYERS)))
    )
    readout = np.where(offset >= 0, shower, 0.6 + np.where(offset < 0, albedo, 0.0))
    return InteractingCalibrationInputs(
        energy_gev=energy_gev,
        depth_mm=depth,
        chord_mm=rng.uniform(2.0, 4.0, (n, N_LAYERS)),
        layer_energy_mev={"readout": readout, "deposition": 11.0 * readout},
    )


@pytest.fixture(scope="module")
def inputs():
    return [synthetic_inputs(e, seed=i) for i, e in enumerate(ENERGIES)]


@pytest.fixture(scope="module")
def table(inputs) -> InteractingTable:
    return build_interacting(inputs, ecal_depth_mm=DEPTH)


@pytest.fixture(scope="module")
def crossing_table() -> CrossingTable:
    quantiles = np.broadcast_to(np.array([0.3, 0.5, 0.6, 0.7, 1.5]), (2, 3, 5)).copy()
    return CrossingTable(
        energies_gev=np.array(ENERGIES),
        chord_edges_mm=np.array([2.6, 3.3]),
        levels=CROSSING_LEVELS,
        quantiles_mev={"readout": quantiles, "deposition": 11.0 * quantiles},
        counts=np.full((2, 3), 1000),
    )


def draw(table, crossing_table, energy_gev, depth_mm, seed=0):
    rng = np.random.default_rng(seed)
    n = len(depth_mm)
    return sample_interacting_layers(
        table,
        crossing_table,
        np.full(n, energy_gev),
        np.asarray(depth_mm, dtype=float),
        DEPTH,
        np.full((n, N_LAYERS), 1),
        rng.standard_normal((n, N_NORMALS)),
        rng.random((n, 2, N_LAYERS)),
    )


# --- helpers ---------------------------------------------------------------------------


def test_the_interaction_layer_and_depth_group_follow_the_depth() -> None:
    depth = np.array([0.0, 9.24, 9.26, 100.0, 166.49])

    assert list(interaction_layer(depth, DEPTH)) == [0, 0, 1, 10, 17]
    assert list(depth_group(np.array([10.0, 60.0, 150.0]), DEPTH)) == [0, 1, 2]


def test_energies_are_aligned_at_the_interaction_layer() -> None:
    energy = np.tile(np.arange(1.0, N_LAYERS + 1), (2, 1))

    aligned = aligned_log_energy(energy, np.array([0, 15]))

    assert aligned[0, 0] == pytest.approx(np.log(1.0)) and aligned[0, 17] == pytest.approx(np.log(18.0))
    assert aligned[1, 0] == pytest.approx(np.log(16.0)) and aligned[1, 2] == pytest.approx(np.log(18.0))
    assert np.all(np.isnan(aligned[1, 3:]))


def test_a_zero_layer_is_recorded_at_the_floor() -> None:
    aligned = aligned_log_energy(np.zeros((1, N_LAYERS)), np.array([0]))

    assert np.allclose(aligned, np.log(FLOOR_MEV))


def test_the_amplitude_proxy_is_the_geometric_mean_of_the_first_layers() -> None:
    log_energy = np.full((1, N_LAYERS), np.nan)
    log_energy[0, :4] = np.log([1.0, 4.0, 4.0, 16.0])

    assert amplitude_proxy(log_energy)[0] == pytest.approx(4.0)
    assert WINDOW == 9


def test_normal_scores_are_standard_normal_and_leave_missing_values_alone() -> None:
    rng = np.random.default_rng(1)
    values = rng.gamma(2.0, 1.0, (4000, 2))
    values[::2, 1] = np.nan

    scores = normal_scores(values)

    assert np.isnan(scores[::2, 1]).all()
    assert stats.kstest(scores[:, 0], "norm").pvalue > 0.01
    assert np.isfinite(scores[1::2, 1]).all()


def test_the_factor_and_chain_are_recovered_from_their_lag_correlations() -> None:
    lam, rho = 0.3, 0.7
    lags = np.arange(1, 9)

    fitted = fit_factor_chain(lam + (1 - lam) * rho**lags)

    assert fitted == pytest.approx((lam, rho), abs=0.03)


def test_a_planted_chain_gives_the_expected_lag_correlation() -> None:
    rng = np.random.default_rng(2)
    z = np.empty((20000, N_LAYERS))
    z[:, 0] = rng.standard_normal(20000)
    for k in range(1, N_LAYERS):
        z[:, k] = 0.8 * z[:, k - 1] + 0.6 * rng.standard_normal(20000)

    lag = pairwise_lag_correlation(z)

    assert lag[0] == pytest.approx(0.8, abs=0.03) and lag[1] == pytest.approx(0.64, abs=0.03)


def test_the_correlation_of_scores_ignores_missing_entries() -> None:
    rng = np.random.default_rng(3)
    a = rng.standard_normal(500)
    b = 0.9 * a + 0.44 * rng.standard_normal(500)
    b[:50] = np.nan

    assert gaussian_correlation(a, b) == pytest.approx(0.9, abs=0.05)
    assert gaussian_correlation(a[:10], b[:10]) == 0.0  # too few pairs to say


def test_invert_quantiles_is_the_inverse_of_the_quantile_function() -> None:
    levels = np.array([0.0, 0.5, 1.0])
    values = np.tile(np.array([1.0, 2.0, 6.0]), (4, 1))

    level = invert_quantiles(levels, values, np.array([1.0, 1.5, 2.0, 4.0]))

    assert level == pytest.approx([0.0, 0.25, 0.5, 0.75])


# --- the builder -------------------------------------------------------------------------


def test_the_builder_finds_a_strong_persistent_structure_in_the_planted_layers(table) -> None:
    assert np.all(table.chain > 0.4)
    assert np.all(table.representation_coupling > 0.9)  # the deposition is a multiple of the readout
    assert np.all(table.counts > 1000)


def test_the_back_edge_shift_is_negative_for_the_planted_deep_interactions(table) -> None:
    for r in range(2):
        assert table.backedge_shift[r, 2] < table.backedge_shift[r, 0]


def test_upstream_layers_are_tied_to_the_amplitude(table) -> None:
    assert np.all(table.amplitude_loading > 0.3)


def test_the_amplitude_tables_have_one_row_per_depth_group(table) -> None:
    assert table.amp_data_quantiles.shape[:3] == (2, 2, 3)
    assert np.all(np.diff(table.amp_data_quantiles, axis=-1) >= 0)
    assert np.all(np.diff(table.amp_model_quantiles, axis=-1) >= 0)


def test_the_table_survives_a_round_trip_through_its_arrays(table) -> None:
    again = InteractingTable.from_arrays(table.arrays())

    for name, array in table.arrays().items():
        assert np.array_equal(array, again.arrays()[name]), name


def test_the_build_is_deterministic(inputs, table) -> None:
    again = build_interacting(inputs, ecal_depth_mm=DEPTH)

    for name, array in table.arrays().items():
        assert np.array_equal(array, again.arrays()[name]), name


def test_too_few_events_are_refused() -> None:
    few = [synthetic_inputs(e, n=10, seed=5) for e in ENERGIES]

    with pytest.raises(ValueError, match="too few interacting events"):
        build_interacting(few, ecal_depth_mm=DEPTH)


def test_malformed_tables_are_rejected(table) -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        replace(
            table,
            upstream_quantiles={
                n: table.upstream_quantiles[n][..., ::-1] for n in table.upstream_quantiles
            },
        )
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        replace(table, factor=np.full_like(table.factor, 1.2))
    with pytest.raises(ValueError, match="shape"):
        replace(table, counts=table.counts[:1])


def test_inputs_with_negative_energy_are_rejected() -> None:
    good = synthetic_inputs(10.0, n=40)
    bad = {name: array.copy() for name, array in good.layer_energy_mev.items()}
    bad["readout"][0, 0] = -1.0

    with pytest.raises(ValueError, match="nonnegative"):
        replace(good, layer_energy_mev=bad)


# --- the sampler -------------------------------------------------------------------------


def test_every_layer_of_an_interacting_event_has_a_finite_nonnegative_energy(
    table, crossing_table
) -> None:
    depth = np.random.default_rng(4).uniform(0.0, DEPTH, 2000)

    out = draw(table, crossing_table, 50.0, depth)

    for energy in out.values():
        assert energy.shape == (2000, N_LAYERS)
        assert np.all(np.isfinite(energy)) and np.all(energy >= 0.0)


def test_the_draw_is_a_function_of_its_variates(table, crossing_table) -> None:
    depth = np.linspace(5.0, 160.0, 50)

    a = draw(table, crossing_table, 30.0, depth, seed=7)
    b = draw(table, crossing_table, 30.0, depth, seed=7)

    for name in a:
        assert np.array_equal(a[name], b[name])
    assert not np.array_equal(a["readout"], draw(table, crossing_table, 30.0, depth, seed=8)["readout"])


def test_a_shower_is_much_larger_than_a_crossing_layer_after_the_interaction(
    table, crossing_table
) -> None:
    depth = np.full(2000, 20.0)  # interaction layer 2

    out = draw(table, crossing_table, 100.0, depth)["readout"]

    assert np.median(out[:, 6]) > 20 * np.median(out[:, 0])  # behind the interaction vs in front


def test_the_energy_grows_with_the_primary_energy(table, crossing_table) -> None:
    depth = np.full(2000, 30.0)

    low = draw(table, crossing_table, 10.0, depth)["readout"].sum(axis=1)
    high = draw(table, crossing_table, 100.0, depth)["readout"].sum(axis=1)

    assert np.median(high) > 3 * np.median(low)


def test_a_late_interaction_leaves_less_energy_in_the_ecal(table, crossing_table) -> None:
    early = draw(table, crossing_table, 100.0, np.full(2000, 20.0))["readout"].sum(axis=1)
    late = draw(table, crossing_table, 100.0, np.full(2000, 155.0))["readout"].sum(axis=1)

    assert np.median(early) > 3 * np.median(late)


def test_the_two_representations_are_strongly_but_not_perfectly_coupled(
    table, crossing_table
) -> None:
    out = draw(table, crossing_table, 50.0, np.full(4000, 40.0))

    rho = stats.spearmanr(out["readout"][:, 6], out["deposition"][:, 6]).statistic
    assert 0.85 < rho < 1.0


def test_the_mapping_reproduces_the_calibrated_amplitude_by_depth_group(
    inputs, table, crossing_table
) -> None:
    entry = inputs[1]
    rng = np.random.default_rng(9)
    depth = entry.depth_mm[rng.integers(0, entry.n_events, 8000)]

    out = draw(table, crossing_table, 100.0, depth, seed=10)["readout"]

    modelled = np.log(amplitude_proxy(aligned_log_energy(out, interaction_layer(depth, DEPTH))))
    observed = np.log(
        amplitude_proxy(
            aligned_log_energy(
                entry.layer_energy_mev["readout"], interaction_layer(entry.depth_mm, DEPTH)
            )
        )
    )
    for g in range(3):
        d_group = depth_group(depth, DEPTH) == g
        o_group = depth_group(entry.depth_mm, DEPTH) == g
        assert stats.ks_2samp(modelled[d_group], observed[o_group]).statistic < 0.05
    assert abs(stats.skew(modelled) - stats.skew(observed)) < 0.3  # the left skew of the amplitude


def test_the_back_edge_lowers_the_amplitude_of_a_deep_interaction(table, crossing_table) -> None:
    front = draw(table, crossing_table, 100.0, np.full(4000, 20.0), seed=11)["readout"]
    back = draw(table, crossing_table, 100.0, np.full(4000, 140.0), seed=12)["readout"]

    amp_front = np.median(amplitude_proxy(aligned_log_energy(front, np.full(4000, 2))))
    amp_back = np.median(amplitude_proxy(aligned_log_energy(back, np.full(4000, 15))))
    assert amp_back < amp_front


def test_upstream_layers_carry_an_albedo_that_follows_the_shower(table, crossing_table) -> None:
    depth = np.full(4000, 80.0)  # interaction layer 8

    out = draw(table, crossing_table, 100.0, depth, seed=13)["readout"]

    assert np.median(out[:, 7]) > np.median(out[:, 0])  # the layer next to the interaction is brighter
    shower = out[:, 9:12].sum(axis=1)
    assert stats.spearmanr(out[:, 7], shower).statistic > 0.2


def test_an_energy_outside_the_calibrated_range_is_refused(table, crossing_table) -> None:
    with pytest.raises(ValueError, match="nothing is extrapolated"):
        draw(table, crossing_table, 200.0, np.full(5, 50.0))


def test_the_sampler_rejects_malformed_variates(table, crossing_table) -> None:
    with pytest.raises(ValueError, match="normals must be"):
        sample_interacting_layers(
            table,
            crossing_table,
            np.full(3, 30.0),
            np.full(3, 50.0),
            DEPTH,
            np.full((3, N_LAYERS), 1),
            np.zeros((3, 5)),
            np.zeros((3, 2, N_LAYERS)),
        )


def test_the_constants_describe_the_layout() -> None:
    assert UPSTREAM == 6
    assert INTERACTION_LEVELS[0] == 0.0 and INTERACTION_LEVELS[-1] == 1.0
    assert np.all(np.diff(INTERACTION_LEVELS) > 0)


def test_the_unmapped_amplitude_is_less_skewed_than_the_calibrated_one(inputs, table) -> None:
    entry = inputs[1]
    rng = np.random.default_rng(14)
    n = 8000
    depth = entry.depth_mm[rng.integers(0, entry.n_events, n)]
    args = (
        table,
        np.full(n, 100.0),
        interaction_layer(depth, DEPTH),
        depth_group(depth, DEPTH),
        rng.standard_normal((n, N_NORMALS)),
    )

    raw = _post_log_energies(*args, apply_map=False)["readout"][1]
    mapped = _post_log_energies(*args, apply_map=True)["readout"][1]

    observed = np.log(
        amplitude_proxy(
            aligned_log_energy(
                entry.layer_energy_mev["readout"], interaction_layer(entry.depth_mm, DEPTH)
            )
        )
    )
    assert abs(stats.skew(mapped) - stats.skew(observed)) < abs(stats.skew(raw) - stats.skew(observed))


def test_the_back_edge_shift_acts_before_the_mapping(inputs, table) -> None:
    rng = np.random.default_rng(15)
    n = 6000
    normals = rng.standard_normal((n, N_NORMALS))
    front, back = np.full(n, 20.0), np.full(n, 150.0)

    def amplitude(depth):
        l_d = interaction_layer(depth, DEPTH)
        return _post_log_energies(
            table, np.full(n, 100.0), l_d, depth_group(depth, DEPTH), normals, apply_map=False
        )["readout"][1]

    # same normals, so the only difference is the layers that exist and the shift of the group
    assert np.median(amplitude(back)) < np.median(amplitude(front)) + 1e-9
    assert table.backedge_shift[0, 2] < 0 < table.backedge_shift[0, 0] + 0.5


def test_shifting_the_calibrated_amplitude_law_shifts_the_generated_amplitude(table, crossing_table) -> None:
    shifted = replace(table, amp_data_quantiles=table.amp_data_quantiles + 1.0)
    depth = np.full(4000, 40.0)

    base = draw(table, crossing_table, 100.0, depth, seed=16)["readout"]
    moved = draw(shifted, crossing_table, 100.0, depth, seed=16)["readout"]

    l_d = np.full(4000, 4)
    shift = np.median(np.log(amplitude_proxy(aligned_log_energy(moved, l_d)))) - np.median(
        np.log(amplitude_proxy(aligned_log_energy(base, l_d)))
    )
    assert shift == pytest.approx(1.0, abs=0.1)  # every layer of an event moves together


def test_a_back_edge_shift_lowers_the_unmapped_amplitude_of_that_depth_group(table) -> None:
    n = 6000
    depth = np.full(n, 150.0)
    l_d, group = interaction_layer(depth, DEPTH), depth_group(depth, DEPTH)
    normals = np.random.default_rng(17).standard_normal((n, N_NORMALS))
    lowered = replace(table, backedge_shift=np.full_like(table.backedge_shift, -1.5))

    base = _post_log_energies(table, np.full(n, 100.0), l_d, group, normals, apply_map=False)
    moved = _post_log_energies(lowered, np.full(n, 100.0), l_d, group, normals, apply_map=False)

    assert np.median(moved["readout"][1]) < np.median(base["readout"][1]) - 0.3
