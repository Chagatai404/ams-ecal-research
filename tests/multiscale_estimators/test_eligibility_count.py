"""Eligibility count: occupied cells, reliable windows and the conversion from control hits to cells."""

import numpy as np

from ams_ecal.multiscale_estimators import eligibility_count as ec
from ams_ecal.multiscale_estimators import multiscale


def test_occupied_cells_counts_cells_strictly_above_the_threshold() -> None:
    grids = np.zeros((2, 18, 72))
    grids[0, 0, :5] = 1.0
    grids[1, 3, :3] = [0.1, 0.5, 2.0]

    assert ec.occupied_cells(grids).tolist() == [5, 3]
    assert ec.occupied_cells(grids, threshold_mev=0.28).tolist() == [5, 2]


def test_the_saturation_rule_leaves_more_windows_to_more_occupied_cells() -> None:
    counts = ec.reliable_window_counts(np.array([0, 20, 300, 1000]))

    assert counts[0] == 0
    assert (np.diff(counts) >= 0).all()
    assert counts[-1] == len(multiscale.contiguous_windows())


def test_the_control_images_occupy_more_cells_with_more_hits_and_never_more_than_the_grid() -> None:
    conversion = ec.control_conversion(replicates=20, seed=3)

    medians = [conversion[h]["q50"] for h in ("30", "100", "300", "1000", "3000")]
    assert medians == sorted(medians)
    assert conversion["3000"]["q95"] <= 18 * 72


def test_the_summary_reports_the_fraction_of_events_with_enough_cells() -> None:
    counts = np.array([20, 20, 400, 700])

    summary = ec.summarise(counts, counts)

    assert summary["events"] == 4
    assert summary["fraction_with_cells_at_least"]["300"] == 0.5
    assert summary["fraction_with_cells_at_least"]["600"] == 0.25
