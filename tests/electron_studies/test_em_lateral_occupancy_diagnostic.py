"""Occupancy diagnostic: the statistics on synthetic grids, the distance bins and the sealed-data rule."""

from pathlib import Path

import numpy as np
import pytest

from ams_ecal.electron_model import em_production_lateral as lat
from ams_ecal.electron_studies import em_lateral_occupancy_diagnostic as occ
from ams_ecal.electron_studies import em_production_lateral_check as check

needs_samples = pytest.mark.skipif(
    not (
        (check.EXTENDED_DIR / "E10GeV" / "events.npz").exists()
        and (check.BASELINE_DIR / "E10GeV" / "events.npz").exists()
        and lat.DEFAULT_ARTIFACT.exists()
        and occ.FASTER_TAIL_ARTIFACT.exists()
    ),
    reason="needs the git-ignored exposed samples (regenerate them from their configs) and the lateral artifacts",
)


def test_the_occupancy_row_counts_the_cells_above_the_threshold_and_their_energy() -> None:
    grid = np.zeros((2, 1, 4))
    grid[0, 0] = [10.0, 0.1, 0.0, 2.0]  # two occupied cells (10 and 2 MeV), one below threshold, one empty
    grid[1, 0] = [0.0, 0.0, 5.0, 0.0]
    mask = np.ones_like(grid, dtype=bool)

    row = occ.occupancy_row(grid, mask, threshold_mev=0.28)

    assert row["cells_per_event"] == pytest.approx(4.0)
    assert row["occupied_fraction"] == pytest.approx(3 / 8)
    assert row["empty_fraction"] == pytest.approx(4 / 8)  # exactly empty: one cell in event 0, three in event 1
    assert row["mean_energy_of_occupied_cell_mev"] == pytest.approx((10.0 + 2.0 + 5.0) / 3)
    assert row["energy_per_event_mev"] == pytest.approx((12.1 + 5.0) / 2)
    assert row["energy_per_occupied_cell_mev"] == pytest.approx(17.1 / 3)


def test_an_empty_mask_gives_zeros_and_not_a_division_error() -> None:
    grid = np.ones((2, 1, 3))

    row = occ.occupancy_row(grid, np.zeros_like(grid, dtype=bool))

    assert row["occupied_fraction"] == 0.0
    assert row["energy_per_occupied_cell_mev"] == 0.0


def test_the_distance_bins_partition_the_cells() -> None:
    offsets = np.linspace(-300.0, 300.0, 73)[None, None, :]

    masks = occ.bin_masks(offsets, occ.DISTANCE_BINS_MM)

    assert len(masks) == len(occ.DISTANCE_BINS_MM) - 1
    assert np.sum(masks, axis=0).max() == 1
    assert np.sum(masks, axis=0).min() == 1


def test_the_summary_has_the_distance_bins_and_the_layer_groups() -> None:
    rng = np.random.default_rng(0)
    grid = rng.exponential(1.0, (20, 18, 72))
    offsets = (np.arange(72) - 36 + 0.5)[None, None, :] * 9.0 * np.ones((20, 18, 1))

    summary = occ.summarise(grid, offsets)

    assert len(summary["by_distance_mm"]) == len(occ.DISTANCE_BINS_MM) - 1
    assert set(summary["cells_at_least_27_mm_by_layer_group"]) == set(occ.LAYER_GROUPS)
    assert summary["occupied_cells_per_event"] > 0


def test_the_diagnostic_never_reads_a_sealed_set() -> None:
    text = Path(occ.__file__).read_text(encoding="utf-8").lower()

    assert "geant4_electron_sealed" not in text
    assert "open_set" not in text


@needs_samples
def test_the_diagnostic_runs_end_to_end_on_a_small_slice() -> None:
    results = occ.diagnose((10.0,), events=200)

    row = results["per_energy"]["10"]
    assert results["sealed_data_read"] is False
    assert set(row) == {"geant4_baseline", "geant4_holdout", "slice2", "faster_tail"}
    assert row["geant4_holdout"]["occupied_cells_per_event"] > row["slice2"]["occupied_cells_per_event"]
