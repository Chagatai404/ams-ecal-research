import numpy as np
import pytest

from ams_ecal.multiscale_estimators import multiscale_3d as m3


def full_mesh():
    iz, iy, ix = np.indices((m3.MESH[2], m3.MESH[1], m3.MESH[0]))
    return ix.ravel(), iy.ravel(), iz.ravel(), np.ones(ix.size)


def test_every_scale_tiles_the_mesh_exactly_and_the_effective_sides_increase():
    for kx, kz in m3.SCALES:
        assert m3.MESH[0] % kx == 0 and m3.MESH[1] % kx == 0 and m3.MESH[2] % kz == 0
    assert (np.diff(m3.EFFECTIVE_SIDE_MM) > 0).all()
    assert len(m3.WINDOWS) == 36 and all(len(w) >= 2 for w in m3.WINDOWS)


def test_unravel_inverts_the_flat_index_of_the_mesh():
    rng = np.random.default_rng(0)
    ix, iy, iz = (
        rng.integers(0, 216, 50),
        rng.integers(0, 216, 50),
        rng.integers(0, 36, 50),
    )
    flat = (iz * m3.MESH[1] + iy) * m3.MESH[0] + ix
    assert all(
        (a == b).all() for a, b in zip(m3.unravel(flat), (ix, iy, iz), strict=True)
    )


def test_a_uniformly_filled_mesh_has_dimension_three_at_every_window_and_q():
    stats = m3.statistics(*full_mesh())
    assert np.allclose(stats[:3], 3.0, atol=1e-9) and np.allclose(
        stats[3], 0.0, atol=1e-9
    )


def test_a_single_voxel_has_dimension_zero():
    stats = m3.statistics(
        np.array([100]), np.array([50]), np.array([7]), np.array([5.0])
    )
    assert np.allclose(stats, 0.0)


def test_repeated_voxels_add_their_energies_in_the_box_shares():
    p = m3.box_shares(
        np.array([1, 1, 200]),
        np.array([2, 2, 5]),
        np.array([3, 3, 30]),
        np.array([1.0, 2.0, 1.0]),
        1,
        1,
    )
    assert sorted(p.round(6)) == [0.25, 0.75]


def test_samplers_stay_inside_the_mesh():
    rng = np.random.default_rng(1)
    for sampler in (m3.sample_smooth, m3.sample_cascade):
        ix, iy, iz = sampler(5000, rng)
        assert (
            ix.min() >= 0
            and ix.max() < 216
            and iy.min() >= 0
            and iy.max() < 216
            and iz.min() >= 0
            and iz.max() < 36
        )


def test_the_cascade_recovers_its_exact_dimensions_on_the_coarse_isotropic_levels_and_saturates_on_the_finest():
    rows = m3.recovery_table(200_000, seed=3, replicates=3)
    coarse = [r for r in rows if r["fine_scale"] in ((27, 18), (9, 6))]
    assert len(coarse) == 6 and all(abs(r["relative_error"]) < 0.01 for r in coarse)
    # the finest levels have more boxes than hits (93,000 boxes of the (3,2) scale against 200,000 equal quanta on a heavy-tailed
    # cascade): the occupied-box count is undercounted, which is the saturation the controls must quantify, not a bug
    finest = next(
        r
        for r in rows
        if r["fine_scale"] == (0, 0) or r["fine_scale"] == (1, 1) and r["q"] == 0.0
    )
    assert finest["relative_error"] < -0.1


def test_the_smooth_map_is_not_flagged_as_a_cascade_by_the_spread_statistic():
    rng = np.random.default_rng(4)
    k = m3.WINDOWS.index((2, 3, 4, 5))
    smooth = [
        m3.statistics(*m3._event_from_hits(*m3.sample_smooth(3000, rng)))[3, k]
        for _ in range(40)
    ]
    cascade = [
        m3.statistics(*m3._event_from_hits(*m3.sample_cascade(3000, rng)))[3, k]
        for _ in range(40)
    ]
    assert np.mean(cascade) > np.percentile(smooth, 97.5)


@pytest.mark.parametrize("n_hits", [1000])
def test_evaluate_returns_one_row_per_window_and_statistic(n_hits):
    results = m3.evaluate(n_hits, seed=5, replicates=12)
    assert len(results) == 4 * len(m3.WINDOWS)
    spread = [r for r in results if r.statistic == "D0-D2"]
    assert all(r.power is not None and r.ordering_rate is not None for r in spread)
