import numpy as np

from ams_ecal.multiscale_estimators import multiscale_shuffle_excess as se


def random_grids(n=6, occupancy=0.6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.exponential(1.0, (n, 18, 72)) * (rng.random((n, 18, 72)) < occupancy)


def compact_grids(n=6, seed=1):
    """A lateral envelope: cells near the centre are far more likely occupied, with more cells overall than the eligibility minimum."""
    rng = np.random.default_rng(seed)
    centre = np.exp(-0.5 * ((np.arange(72) - 36) / 14.0) ** 2)
    p = 0.15 + 0.8 * centre
    return rng.exponential(1.0, (n, 18, 72)) * (rng.random((n, 18, 72)) < p)


def test_the_full_shuffle_keeps_each_layers_multiset_and_the_block_shuffle_keeps_each_blocks_energy():
    g = random_grids(3)
    rng = np.random.default_rng(5)
    full = se.shuffle_full(g, rng)
    assert np.allclose(np.sort(full, axis=-1), np.sort(g, axis=-1)) and (full != g).any()
    blocks = se.shuffle_blocks(g, rng)
    assert np.allclose(blocks.reshape(3, 18, 8, 9).sum(axis=-1), g.reshape(3, 18, 8, 9).sum(axis=-1))
    assert (blocks != g).any()


def test_the_shuffles_leave_the_occupied_cell_count_and_so_the_eligibility_unchanged():
    g = random_grids(4)
    rng = np.random.default_rng(6)
    for shuffled in (se.shuffle_full(g, rng), se.shuffle_blocks(g, rng)):
        assert ((shuffled > 0).sum(axis=(1, 2)) == (g > 0).sum(axis=(1, 2))).all()


def test_the_excess_is_near_zero_for_exchangeable_cells_and_large_for_a_lateral_envelope_under_the_full_shuffle():
    rng = np.random.default_rng(7)
    flat = se.excess_table(random_grids(12), "full", 3, rng)
    shaped = se.excess_table(compact_grids(12), "full", 3, rng)
    column = int(np.flatnonzero(np.isfinite(flat).all(axis=0) & np.isfinite(shaped).all(axis=0))[0])
    assert abs(np.median(flat[:, column])) < 0.5 * abs(np.median(shaped[:, column]))
    assert np.isfinite(shaped).any()


def test_the_block_shuffle_keeps_more_of_a_lateral_envelope_than_the_full_shuffle():
    g = compact_grids(12, seed=2)
    rng = np.random.default_rng(8)
    full = se.excess_table(g, "full", 3, rng)
    block = se.excess_table(g, "block9", 3, rng)
    column = int(np.flatnonzero(np.isfinite(full).all(axis=0) & np.isfinite(block).all(axis=0))[0])
    assert abs(np.median(block[:, column])) < abs(np.median(full[:, column]))


def test_the_median_interval_covers_the_true_median_of_a_known_sample():
    rng = np.random.default_rng(9)
    values = rng.normal(0.0, 1.0, 3000)
    median, low, high = se.median_with_interval(values, rng, draws=500)
    assert low <= 0.0 <= high and low <= median <= high
