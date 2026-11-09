"""Fast checks of the EXP-016 estimator and controls on synthetic events (registration sections 4-8, 2026-11-09)."""

import numpy as np
import pytest
from scipy import stats

from ams_ecal.inference import multiplicity as mp
from ams_ecal.multiscale_estimators import multiscale_3d as m3
from ams_ecal.multiscale_estimators import multiscale_3d_real_controls as rc


def block_cube(x0, y0, z0, nx, ny, nz):
    iz, iy, ix = np.indices((nz, ny, nx))
    return ix.ravel() + x0, iy.ravel() + y0, iz.ravel() + z0, np.ones(ix.size)


def random_event(n_voxels, seed, centre=108, spread=12.0):
    rng = np.random.default_rng(seed)
    ix = np.clip(rng.normal(centre, spread, 3 * n_voxels), 0, 215).astype(np.int64)
    iy = np.clip(rng.normal(centre, spread, 3 * n_voxels), 0, 215).astype(np.int64)
    iz = rng.integers(0, 36, 3 * n_voxels)
    flat = np.unique((iz * 216 + iy) * 216 + ix)[:n_voxels]
    x, y, z = m3.unravel(flat)
    return x, y, z, rng.exponential(1.0, len(flat)) + 0.01


# --- offsets, padded boxes, partial boxes -------------------------------------------------------------------------------------------


def test_the_eight_offsets_are_registered_integers_and_reproducible():
    offsets = rc.grid_offsets()
    assert offsets.shape == (8, 3)
    assert (
        (offsets >= 0).all()
        and (offsets[:, :2] < 54).all()
        and (offsets[:, 2] < 36).all()
    )
    np.testing.assert_array_equal(offsets, rc.grid_offsets())
    assert rc.OFFSET_SEED == 20261109


def test_padded_boxes_at_zero_offset_reproduce_the_mesh_aligned_shares():
    ix, iy, iz, e = random_event(500, 1)
    for kx, kz in m3.SCALES[1:]:
        padded = np.sort(rc.occupied_box_shares(ix, iy, iz, e, (0, 0, 0), kx, kz))
        aligned = np.sort(m3.box_shares(ix, iy, iz, e, kx, kz))
        np.testing.assert_allclose(padded, aligned)


def test_shifted_boxes_never_collide_and_conserve_energy():
    ix, iy, iz, e = random_event(800, 2)
    for offset in ((53, 53, 35), (7, 20, 3)):
        for kx, kz in m3.SCALES:
            p = rc.occupied_box_shares(ix, iy, iz, e, offset, kx, kz)
            assert p.sum() == pytest.approx(1.0)
            bx, by, bz = (
                (ix + offset[0]) // kx,
                (iy + offset[1]) // kx,
                (iz + offset[2]) // kz,
            )
            assert len(p) == len(set(zip(bx, by, bz, strict=True)))


def test_partial_box_fraction_is_lateral_small_for_a_central_event_and_total_with_depth():
    ix, iy, iz, e = block_cube(90, 90, 0, 36, 36, 36)
    kx, kz = m3.SCALES[-1]
    assert (
        rc.partial_energy_fraction(
            ix, iy, iz, e, (39, 16, 17), kx, kz, include_depth=False
        )
        == 0.0
    )
    assert (
        rc.partial_energy_fraction(
            ix, iy, iz, e, (39, 16, 17), kx, kz, include_depth=True
        )
        == 1.0
    )
    assert (
        rc.partial_energy_fraction(
            ix, iy, iz, e, (39, 16, 0), kx, kz, include_depth=True
        )
        == 0.0
    )
    edge = block_cube(0, 100, 5, 10, 10, 4)
    assert (
        rc.partial_energy_fraction(*edge, (3, 0, 0), kx, kz, include_depth=False) == 1.0
    )


def test_the_exclusion_rule_uses_the_selected_partial_fraction():
    table = {
        "n_occupied_voxels": np.array([10, 10, 0]),
        "partial_fraction_lateral": np.array([0.0, 0.5, 0.0]),
        "partial_fraction_with_depth": np.array([1.0, 1.0, 1.0]),
    }
    assert not rc.retained(table, "with_depth").any()
    np.testing.assert_array_equal(
        rc.retained(table, "lateral_only"), [True, False, False]
    )
    with pytest.raises(ValueError):
        rc.retained(table, "other")


# --- offset averaging ---------------------------------------------------------------------------------------------------------------


def test_offset_averaging_leaves_a_uniform_cube_at_dimension_three():
    profile = rc.offset_averaged_profile(*block_cube(54, 54, 0, 108, 108, 36))
    d = rc.window_dimensions(profile, [(0, 1, 2, 3, 4)])
    for row in range(
        3
    ):  # D_0, D_1, D_2 (a few per cent of boundary bias from the shifted partial boxes is expected)
        assert d[row, 0] == pytest.approx(3.0, abs=0.15)


def test_the_partition_quantities_are_averaged_before_the_logarithm():
    ix, iy, iz, e = random_event(300, 3)
    offsets = np.array([(0, 0, 0), (1, 5, 2)])
    profile = rc.offset_averaged_profile(ix, iy, iz, e, offsets)
    shares = [rc.occupied_box_shares(ix, iy, iz, e, o, 6, 4) for o in offsets]
    scale = m3.SCALES.index((6, 4))
    assert profile.mean_occupied_boxes[scale] == pytest.approx(
        np.mean([len(p) for p in shares])
    )
    assert profile.mean_squared_share_sum[scale] == pytest.approx(
        np.mean([np.sum(p**2) for p in shares])
    )
    assert profile.mean_entropy[scale] == pytest.approx(
        np.mean([-np.sum(p * np.log(p)) for p in shares])
    )
    assert profile.log_measures()[2][scale] == pytest.approx(
        np.log(profile.mean_squared_share_sum[scale])
    )
    assert profile.mean_occupied_boxes[0] == profile.n_occupied_voxels == len(e)


# --- the saturation fit -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "excess,d_f,n_eff", [(1.0, 2.0, 20_000), (1.5, 2.6, 5_000), (0.7, 1.7, 300)]
)
def test_the_fit_recovers_amplitude_and_dimension_from_noise_free_counts(
    excess, d_f, n_eff
):
    ln_eps0 = np.log(m3.EFFECTIVE_SIDE_MM[0])
    ln_a = (
        np.log(n_eff) + excess + d_f * ln_eps0
    )  # capacity at the finest scale is e^excess times N_eff
    capacity = np.exp(ln_a - d_f * np.log(m3.EFFECTIVE_SIDE_MM))
    fit = rc.fit_saturation(rc.saturated_box_count(capacity, n_eff), n_eff)
    assert fit.converged and fit.accepted
    assert fit.fractal_dimension == pytest.approx(d_f, abs=2e-3)
    assert fit.ln_amplitude_mm == pytest.approx(ln_a, abs=5e-3)
    expected_side = np.exp((ln_a - np.log(n_eff)) / d_f)
    assert fit.saturation_side_mm == pytest.approx(expected_side, rel=1e-2)


@pytest.mark.parametrize("d_f", [1.5, 2.0, 2.8])
def test_the_fit_recovers_the_dimension_of_simulated_balls_in_boxes_counts(d_f):
    rng = np.random.default_rng(int(10 * d_f))
    n_eff = 20_000
    ln_a = np.log(2.0 * n_eff) + d_f * np.log(m3.EFFECTIVE_SIDE_MM[0])
    capacity = np.maximum(
        np.round(np.exp(ln_a - d_f * np.log(m3.EFFECTIVE_SIDE_MM))), 1
    ).astype(np.int64)
    counts = np.array(
        [len(np.unique(rng.integers(0, b, n_eff))) for b in capacity], dtype=float
    )
    fit = rc.fit_saturation(counts, n_eff)
    assert fit.accepted
    assert fit.fractal_dimension == pytest.approx(d_f, abs=0.1)
    assert fit.ln_amplitude_mm == pytest.approx(ln_a, abs=0.5)


def test_the_fit_dimension_of_uniform_random_points_in_the_mesh_is_three_less_the_partial_box_bias():
    rng = np.random.default_rng(5)
    flat = np.unique(rng.integers(0, 216 * 216 * 36, 20_000))
    ix, iy, iz = m3.unravel(flat)
    profile = rc.offset_averaged_profile(ix, iy, iz, np.ones(len(flat)))
    fit = rc.fit_saturation(profile.mean_occupied_boxes, profile.n_occupied_voxels)
    assert fit.accepted
    # shifted partial boxes at the mesh edge add boxes at the coarse scales, which lowers the fitted exponent below 3
    assert 2.3 < fit.fractal_dimension < 3.2


@pytest.mark.parametrize("d_f", [0.2, 4.2])
def test_a_fit_with_dimension_outside_the_open_interval_fails_and_is_never_admissible(
    d_f,
):
    n_eff = 5000
    capacity = 4.0 * n_eff * (m3.EFFECTIVE_SIDE_MM / m3.EFFECTIVE_SIDE_MM[0]) ** -d_f
    fit = rc.fit_saturation(rc.saturated_box_count(capacity, n_eff), n_eff)
    assert not fit.accepted and fit.saturation_side_mm == float("inf")
    ok = rc.admissible_windows(
        np.array([fit.saturation_side_mm]), np.array([fit.accepted]), 1
    )
    assert not ok.any()


def test_admissibility_compares_the_finest_side_with_two_to_the_delta_times_the_saturation_side():
    for delta in (1, 2):
        eps_star = (
            m3.EFFECTIVE_SIDE_MM[3] / 2**delta * 1.0001
        )  # scale 3 is just inadmissible
        ok = rc.admissible_windows(np.array([eps_star]), np.array([True]), delta)[0]
        for k, w in enumerate(m3.WINDOWS):
            assert ok[k] == (w[0] >= 4)
    eps_star = np.array([m3.EFFECTIVE_SIDE_MM[3] / 2 * 1.0001])
    assert (
        rc.admissible_windows(eps_star, np.array([True]), 2).sum()
        < rc.admissible_windows(eps_star, np.array([True]), 1).sum()
    )


# --- window W* ----------------------------------------------------------------------------------------------------------------------


def hand_made_table(first_good_scale, n_sets=3):
    table = np.zeros((n_sets, len(m3.WINDOWS)))
    for k, w in enumerate(m3.WINDOWS):
        if w[0] >= first_good_scale:
            table[:, k] = 1.0
    return table


def test_wstar_is_the_widest_window_starting_at_the_finest_good_scale():
    k = rc.select_window(hand_made_table(3))
    assert m3.WINDOWS[k] == tuple(range(3, 9))


def test_wstar_requires_the_fraction_in_every_set_and_three_scales():
    table = hand_made_table(3)
    k_widest = m3.WINDOWS.index(tuple(range(3, 9)))
    table[1, k_widest] = 0.89  # one set just misses on the widest window of scale 3
    assert m3.WINDOWS[rc.select_window(table)] == tuple(range(3, 8))
    table[1, k_widest] = 0.90
    assert m3.WINDOWS[rc.select_window(table)] == tuple(range(3, 9))
    only_two = np.zeros((2, len(m3.WINDOWS)))
    only_two[:, m3.WINDOWS.index((7, 8))] = 1.0
    assert rc.select_window(only_two) is None
    assert rc.select_window(np.zeros((2, len(m3.WINDOWS)))) is None


def test_the_descriptive_fallback_window_maximises_the_minimum_fraction():
    table = hand_made_table(3) * 0.5
    table[0, m3.WINDOWS.index((5, 6, 7))] = 0.8
    table[1, m3.WINDOWS.index((5, 6, 7))] = 0.7
    table[2, m3.WINDOWS.index((5, 6, 7))] = 0.9
    assert m3.WINDOWS[rc.fallback_window(table)] == (5, 6, 7)


# --- planted disruptions ------------------------------------------------------------------------------------------------------------


def disrupt(variant, phi, seed=0, n=3000, block=18):
    event = random_event(n, 11, spread=8.0)
    rng = np.random.default_rng(seed)
    return event, rc.plant_disruption(*event, variant, phi, block, rng)


def test_variant_w_preserves_occupancy_total_energy_and_d0_but_changes_d2():
    (ix, iy, iz, e), (dx, dy, dz, de, info) = disrupt(rc.VARIANT_WEIGHTS, 1.0)
    np.testing.assert_array_equal((ix, iy, iz), (dx, dy, dz))  # support untouched
    assert de.sum() == pytest.approx(e.sum())
    assert np.sort(de).tolist() == pytest.approx(np.sort(e).tolist())
    before = rc.offset_averaged_profile(ix, iy, iz, e)
    after = rc.offset_averaged_profile(dx, dy, dz, de)
    np.testing.assert_array_equal(before.mean_occupied_boxes, after.mean_occupied_boxes)
    np.testing.assert_allclose(
        rc.window_dimensions(before)[0], rc.window_dimensions(after)[0]
    )
    assert not np.allclose(before.mean_squared_share_sum, after.mean_squared_share_sum)
    assert info["disrupted_blocks"] == info["occupied_blocks"]


def test_variant_s_keeps_occupancy_and_energy_and_moves_only_inside_the_block_and_layer_to_empty_voxels():
    block = 18
    (ix, iy, iz, e), (dx, dy, dz, de, info) = disrupt(
        rc.VARIANT_SUPPORT, 1.0, block=block
    )
    assert len(de) == len(e) and de.sum() == pytest.approx(e.sum())
    flat_new = (dz * 216 + dy) * 216 + dx
    assert len(np.unique(flat_new)) == len(flat_new)
    np.testing.assert_array_equal(dz, iz)  # same layer
    np.testing.assert_array_equal(dx // block, ix // block)  # same block
    np.testing.assert_array_equal(dy // block, iy // block)
    np.testing.assert_array_equal(de, e)  # each voxel keeps its energy
    flat_old = set(((iz * 216 + iy) * 216 + ix).tolist())
    moved = (dx != ix) | (dy != iy)
    assert moved.any()
    assert not any(
        f in flat_old for f in flat_new[moved]
    )  # moved only onto originally empty voxels
    assert info["blocks_with_fewer_empty_than_occupied_voxels"] == 0


def test_variant_s_in_a_full_block_still_preserves_occupancy():
    ix, iy, iz, e = block_cube(0, 0, 0, 18, 18, 1)
    rng = np.random.default_rng(0)
    dx, dy, dz, _de, info = rc.plant_disruption(
        ix, iy, iz, e, rc.VARIANT_SUPPORT, 1.0, 18, rng
    )
    assert len(np.unique((dz * 216 + dy) * 216 + dx)) == len(e)
    assert info["blocks_with_fewer_empty_than_occupied_voxels"] == 1


def test_a_fraction_phi_of_the_occupied_blocks_is_disrupted_and_the_rest_are_untouched():
    block = 18
    event = random_event(4000, 12, spread=30.0)
    rng = np.random.default_rng(1)
    _dx, _dy, _dz, de, info = rc.plant_disruption(
        *event, rc.VARIANT_WEIGHTS, 0.2, block, rng
    )
    ix, iy, iz, e = event
    block_id = (iz * 12 + iy // block) * 12 + ix // block
    n_blocks = len(np.unique(block_id))
    assert info["occupied_blocks"] == n_blocks
    assert info["disrupted_blocks"] == int(np.floor(0.2 * n_blocks + 0.5))
    changed_blocks = np.unique(block_id[np.flatnonzero(de != e)])
    assert 0 < len(changed_blocks) <= info["disrupted_blocks"]


def test_disruption_is_reproducible_from_its_seed():
    a = disrupt(rc.VARIANT_SUPPORT, 0.5, seed=4)[1]
    b = disrupt(rc.VARIANT_SUPPORT, 0.5, seed=4)[1]
    for x, y in zip(a[:4], b[:4], strict=True):
        np.testing.assert_array_equal(x, y)


# --- exact KS and the tests ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ties", [False, True])
def test_the_integer_ks_numerator_equals_the_scipy_ks_statistic(ties):
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=40), rng.normal(0.4, 1.0, size=55)
    if ties:
        a, b = np.round(a, 1), np.round(b, 1)
    values = np.concatenate([a, b])[:, None]
    labels = np.r_[np.ones(40, np.int8), np.zeros(55, np.int8)][None, :]
    num = rc.ks_numerators(values, labels, 40)[0, 0]
    assert num / (40 * 55) == pytest.approx(stats.ks_2samp(a, b).statistic)


def test_the_exact_ks_matches_the_stage_three_matrix_on_untied_values():
    from ams_ecal.multiscale_estimators import multiscale_verdict_reread as vr

    rng = np.random.default_rng(1)
    values = rng.normal(size=(60, 2))
    labels = np.array(
        [rng.permutation(np.r_[np.ones(25), np.zeros(35)]) for _ in range(20)]
    ).astype(np.int8)
    reference = vr.stratified_ks_matrix(values, labels, np.zeros(60, dtype=int), 2)
    mine = rc.ks_numerators(values, labels, 25) / (25 * 35)
    np.testing.assert_allclose(mine, reference, atol=1e-6)


def test_the_permutation_p_value_is_small_for_a_shift_and_never_zero():
    rng = np.random.default_rng(2)
    ks, p = rc.permutation_test(
        rng.normal(size=(100, 2)), rng.normal(1.0, 1.0, size=(100, 2)), 500, rng
    )
    assert (p <= 2 / 501).all() and (p > 0).all() and (ks > 0.3).all()


def test_the_pair_swap_p_value_detects_a_paired_shift_and_is_uniform_without_one():
    rng = np.random.default_rng(3)
    first = rng.normal(size=(150, 1))
    _, p_shift = rc.pair_swap_test(
        first, first + 1.0 + 0.1 * rng.normal(size=(150, 1)), 500, rng
    )
    assert p_shift[0] < 0.01
    pvals = [
        rc.pair_swap_test(rng.normal(size=(80, 1)), rng.normal(size=(80, 1)), 199, rng)[
            1
        ][0]
        for _ in range(60)
    ]
    assert np.mean(np.array(pvals) <= 0.05) <= 0.15


def test_the_machinery_check_rejects_about_alpha_on_exchangeable_data():
    rng = np.random.default_rng(4)
    result = rc.check_machinery(
        {"a": rng.normal(size=(200, 2))}, splits=200, permutations=200, seed=1
    )
    rates = result["per_set"]["a"]["rejection_rate"]
    assert result["accepted_range"][0] < 0.05 < result["accepted_range"][1]
    assert all(0.0 <= r <= 0.10 for r in rates.values())
    assert result["sane"]


def test_the_power_check_has_power_for_a_large_shift_and_none_without_one():
    rng = np.random.default_rng(5)
    original = rng.normal(size=(300, 2))
    shifts = np.array([0.0, 0.6])
    disrupted = (
        original[:, None, None, :]
        + shifts[None, None, :, None]
        + rng.normal(0, 0.2, (300, 2, 2, 2))
    )
    disrupted[:, :, 0] = original[:, None, :] + rng.normal(
        0, 0.0, (300, 2, 2)
    )  # phi 0: identical distribution
    power = rc.check_power(
        original, disrupted, arm=100, splits=20, permutations=100, seed=2
    )
    assert power.shape == (2, 2, 2)
    assert (power[:, 1, :] >= 0.9).all()
    assert (power[:, 0, :] <= 0.3).all()


def test_the_arm_sizes_are_the_realistic_size_and_the_largest_the_set_allows():
    assert rc.arm_sizes(300) == [150]
    assert rc.arm_sizes(491) == [150, 245]
    assert rc.arm_sizes(1000) == [150, 300]
    assert rc.arm_sizes(200) == [100]


# --- C1, C2, Holm and the verdict ---------------------------------------------------------------------------------------------------


def test_the_ordering_check_counts_events_with_d0_ge_d1_ge_d2_on_the_window():
    d = np.zeros((10, 3, len(m3.WINDOWS)))
    d[:, 0], d[:, 1], d[:, 2] = 3.0, 2.0, 1.0
    d[0, 1] = 4.0  # one violation in 10
    ok = rc.check_ordering({"a": d}, 5)
    assert ok["fraction_by_set"]["a"] == pytest.approx(0.9) and ok["pass"]
    d[1, 1] = 4.0
    assert not rc.check_ordering({"a": d}, 5)["pass"]


def test_holm_is_reused_across_the_testable_pairings_and_both_statistics():
    rng = np.random.default_rng(6)
    sets = {k: rng.normal(size=(300, 2)) for k in ("a", "b", "c")}
    keys = {
        k: np.column_stack([rng.integers(3000, 3400, 300), rng.normal(1e5, 5e3, 300)])
        for k in sets
    }
    result = rc.check_matched_null(
        sets,
        keys,
        pairings=(("a", "b"), ("b", "c")),
        caliper=1.0,
        permutations=300,
        seed=1,
    )
    testable = [r for r in result["pairings"] if r["testable"]]
    assert len(testable) == result["testable_pairings"] == 2
    raw = np.array([r["p_value"][n] for r in testable for n in rc.STATISTIC_NAMES])
    adjusted = np.array(
        [r["holm_adjusted_p"][n] for r in testable for n in rc.STATISTIC_NAMES]
    )
    np.testing.assert_allclose(adjusted, mp.holm(raw))
    assert result["pass"] == bool(not (adjusted <= 0.05).any())


def test_a_pairing_with_fewer_than_one_hundred_pairs_is_not_testable_and_not_a_pass():
    rng = np.random.default_rng(7)
    sets = {k: rng.normal(size=(40, 2)) for k in ("a", "b")}
    keys = {
        k: np.column_stack([rng.integers(3000, 3400, 40), rng.normal(1e5, 5e3, 40)])
        for k in sets
    }
    result = rc.check_matched_null(sets, keys, pairings=(("a", "b"),), permutations=100)
    assert (
        result["testable_pairings"] == 0
        and result["status"] == "not testable"
        and not result["pass"]
    )


@pytest.mark.parametrize(
    "window,c0,c1,c2,c3,expected",
    [
        (True, True, True, True, True, rc.VERDICT_USABLE),
        (True, True, True, False, True, rc.VERDICT_MATCHING),
        (True, True, False, True, True, rc.VERDICT_NOT_USABLE),
        (True, True, True, True, False, rc.VERDICT_NOT_USABLE),
        (True, True, True, False, False, rc.VERDICT_NOT_USABLE),
        (False, True, True, True, True, rc.VERDICT_NOT_USABLE),
        (False, True, False, False, False, rc.VERDICT_NOT_USABLE),
        (True, False, True, True, True, rc.VERDICT_STOPPED),
    ],
)
def test_the_verdict_logic_table(window, c0, c1, c2, c3, expected):
    assert rc.decide_verdict(window, c0, c1, c2, c3) == expected


@pytest.mark.parametrize(
    "window,c0,c1,c3,expected",
    [
        (True, True, True, True, "no registered verdict applies: C2 not testable"),
        (True, True, False, True, rc.VERDICT_NOT_USABLE),
        (True, True, True, False, rc.VERDICT_NOT_USABLE),
        (False, True, True, True, rc.VERDICT_NOT_USABLE),
        (True, False, True, True, rc.VERDICT_STOPPED),
    ],
)
def test_the_verdict_when_c2_is_not_testable(window, c0, c1, c3, expected):
    assert rc.decide_verdict(window, c0, c1, False, c3, c2_testable=False) == expected


def test_the_default_partial_box_rule_is_lateral_only_and_hashes_are_recorded():
    assert rc.DEFAULT_PARTIAL_BOX_RULE == "lateral_only"
    assert "with_depth" in rc.PARTIAL_BOX_RULES
    assert rc.AMENDMENT_1_HASH_PREFIX == "795c2d71add4c1b0"
    table = {
        "n_occupied_voxels": np.array([10, 10]),
        "partial_fraction_lateral": np.array([0.0, 0.5]),
        "partial_fraction_with_depth": np.array([1.0, 1.0]),
    }
    assert rc.retained(table).tolist() == [True, False]
    assert not rc.retained(table, "with_depth").any()


def test_the_registration_file_matches_its_recorded_hash_prefix_when_present():
    if not rc.REGISTRATION_FILE.is_file():
        pytest.skip("registration text not available")
    assert rc.registration_hash().startswith(rc.REGISTRATION_HASH_PREFIX)


def test_statistic_matrix_is_d2_and_d0_minus_d2():
    d = np.arange(2 * 3 * len(m3.WINDOWS), dtype=float).reshape(2, 3, -1)
    s = rc.statistic_matrix(d, 4)
    np.testing.assert_array_equal(s[:, 0], d[:, 2, 4])
    np.testing.assert_array_equal(s[:, 1], d[:, 0, 4] - d[:, 2, 4])
