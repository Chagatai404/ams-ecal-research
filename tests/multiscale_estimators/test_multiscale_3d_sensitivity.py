import numpy as np

from ams_ecal.multiscale_estimators import multiscale_3d as m3
from ams_ecal.multiscale_estimators.multiscale_3d_sensitivity import (
    K_SCALE,
    reliable_windows,
    saturation_level,
    separable_sampler,
    summarise,
)


def test_scale_index_falls_from_fine_to_coarse():
    assert np.all(np.diff(K_SCALE) < 0)
    assert abs(K_SCALE[0] - np.log2(216 * 216 * 36) / 3) < 1e-12


def test_reliable_window_counts_match_the_registration_table():
    # tabulated from geometry alone before registration: 19 of 36 at 9,143 and 24 at 23,721 occupied voxels
    assert sum(reliable_windows(9143)) == 19
    assert sum(reliable_windows(23721)) == 24
    assert saturation_level(23721) > saturation_level(9143)


def test_more_occupancy_never_removes_a_reliable_window():
    low, high = reliable_windows(1000), reliable_windows(30000)
    assert all(h or not lo for lo, h in zip(low, high, strict=True))


def test_separable_sampler_stays_on_the_mesh_and_b3_is_broader():
    rng = np.random.default_rng(1)
    narrow = separable_sampler(9.0, 6.0)(5000, rng)
    broad = separable_sampler(18.0, 9.0)(5000, rng)
    for ix, iy, iz in (narrow, broad):
        assert ix.max() < m3.MESH[0] and iy.max() < m3.MESH[1] and iz.max() < m3.MESH[2]
    # the 0.02 floor adds a wide uniform tail that dominates the standard deviation, so compare the core by its interquartile range
    iqr = lambda x: float(np.subtract(*np.percentile(x, [75, 25])))
    assert iqr(broad[0]) > 1.5 * iqr(narrow[0])
    assert broad[2].mean() > narrow[2].mean()


def _cell(baseline: str, n_hits: int, usable: list[bool]) -> dict:
    return {
        "baseline": baseline,
        "n_hits": n_hits,
        "median_occupied_voxels": {"B1": 100.0, "B2": 120.0, "B3": 90.0}[baseline],
        "windows": [{"usable": u} for u in usable],
    }


def test_summary_needs_every_baseline_and_takes_the_largest_threshold():
    n = len(m3.WINDOWS)
    yes = [True] + [False] * (n - 1)
    no = [False] * n
    cells = [_cell("B1", 1000, yes), _cell("B2", 1000, yes), _cell("B3", 1000, no)]
    assert summarise(cells)["n_usable"] == 0
    cells = [_cell("B1", 1000, yes), _cell("B2", 1000, yes), _cell("B3", 1000, yes)]
    summary = summarise(cells)
    assert summary["n_usable"] == 1
    threshold = next(iter(summary["usable_under_all_baselines"].values()))["threshold"]
    assert threshold == 120.0
