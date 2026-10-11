"""Width diagnostic: the decomposition identities, the expectation construction and the sealed-data rule."""

from pathlib import Path

import numpy as np
import pytest

from ams_ecal.electron_model import em_production_lateral as lat
from ams_ecal.electron_studies import em_lateral_width_diagnostic as diag
from ams_ecal.electron_studies import em_production_lateral_check as check

needs_samples = pytest.mark.skipif(
    not (
        (check.BASELINE_DIR / "E10GeV" / "events.npz").exists()
        and (check.EXTENDED_DIR / "E10GeV" / "events.npz").exists()
        and lat.DEFAULT_ARTIFACT.exists()
    ),
    reason="needs the git-ignored exposed samples (regenerate them from their configs) and the lateral artifact",
)


def synthetic(events: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4)
    offsets = (np.arange(72) - 36 + 0.5)[None, None, :] * 9.0 - rng.uniform(0, 9, (events, 18, 1))
    grid = rng.random((events, 18, 72)) * np.exp(-np.abs(offsets) / 30.0)
    return grid, offsets


def test_the_distance_bins_add_up_to_the_mean_width_squared() -> None:
    grid, offsets = synthetic()

    bins = diag.width_squared_by_distance(grid, offsets, diag.DISTANCE_BINS_MM)

    assert bins.sum() == pytest.approx((diag.width_mm(grid, offsets) ** 2).mean())
    assert (bins >= 0).all()


def test_the_layers_add_up_to_the_mean_width_squared() -> None:
    grid, offsets = synthetic()

    layers = diag.width_squared_by_layer(grid, offsets)

    assert layers.sum() == pytest.approx((diag.width_mm(grid, offsets) ** 2).mean())


def test_the_energy_beyond_a_distance_decreases_with_distance_and_starts_below_one() -> None:
    grid, offsets = synthetic()

    beyond = diag.energy_beyond(grid, offsets, (0.0, 27.0, 108.0, 216.0))

    assert beyond[0] == pytest.approx(1.0)
    assert (np.diff(beyond) <= 0).all()


def test_the_width_of_a_single_cell_is_its_offset() -> None:
    grid = np.zeros((1, 18, 72))
    offsets = np.zeros((1, 18, 72))
    grid[0, 3, 10] = 5.0
    offsets[0, 3, 10] = 40.0

    assert diag.width_mm(grid, offsets)[0] == pytest.approx(40.0)


def test_the_summary_has_the_quantiles_the_decomposition_and_the_tail() -> None:
    grid, offsets = synthetic()

    summary = diag.summarise(grid, offsets)

    assert len(summary["width_quantiles_mm"]) == 3
    assert len(summary["width_squared_by_distance_mm2"]) == len(diag.DISTANCE_BINS_MM) - 1
    assert len(summary["width_squared_by_layer_mm2"]) == 18
    assert set(summary["energy_beyond"]) == {"27", "54", "108", "216"}


def test_the_diagnostic_never_reads_a_sealed_set() -> None:
    text = Path(diag.__file__).read_text(encoding="utf-8").lower()

    assert "geant4_electron_sealed" not in text
    assert "open_set" not in text


@needs_samples
def test_the_expectation_has_the_same_mean_profile_as_the_spots() -> None:
    model = lat.load_em_production_lateral()

    spots, expectation, axis = diag.model_grids(model, 20.0, 1500, 5)

    assert spots.shape == expectation.shape == (1500, 18, 72)
    assert axis.shape == (1500, 18)
    layer_spots, layer_expectation = spots.sum(axis=(0, 2)), expectation.sum(axis=(0, 2))
    assert layer_spots == pytest.approx(layer_expectation, rel=0.05)
    offsets = diag.cell_offsets_mm(model, axis)
    assert diag.width_mm(expectation, offsets).std() < diag.width_mm(spots, offsets).std()


@needs_samples
def test_the_diagnostic_runs_end_to_end_and_writes_its_plot(tmp_path) -> None:
    results = diag.diagnose((10.0,), events=300)

    row = results["per_energy"]["10"]
    assert results["sealed_data_read"] is False
    assert set(row) >= {"geant4_baseline", "geant4_holdout", "slice2_with_spots", "slice2_expectation"}
    diag.make_plot(results, tmp_path / "w.png")
    assert (tmp_path / "w.png").exists()
