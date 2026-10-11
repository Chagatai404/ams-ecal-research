"""Metric resolution: the paired bootstrap resolves a real difference, sees none between identical models, and reads no sealed set."""

from pathlib import Path

import numpy as np
import pytest

from ams_ecal.electron_model import em_production_lateral as lat
from ams_ecal.electron_studies import em_lateral_metric_resolution as res
from ams_ecal.electron_studies import em_production_lateral_check as check

needs_samples = pytest.mark.skipif(
    not (
        (check.EXTENDED_DIR / "E10GeV" / "events.npz").exists()
        and lat.DEFAULT_ARTIFACT.exists()
        and res.FASTER_TAIL_ARTIFACT.exists()
    ),
    reason="needs the git-ignored extended-depth sample (regenerate it from its config) and the lateral artifacts",
)


def grids(width_mm: float, events: int, seed: int) -> np.ndarray:
    """Layer energies with a two-sided exponential cell profile of the given width; one cell is 9 mm."""

    rng = np.random.default_rng(seed)
    offsets = (np.arange(72) - 36) * 9.0
    shape = np.exp(-np.abs(offsets) / width_mm)
    layer = rng.uniform(0.5, 1.5, (events, 18, 1))
    return layer * shape[None, None, :] * rng.uniform(0.8, 1.2, (events, 18, 72))


def test_identical_models_have_a_difference_of_zero_inside_the_interval() -> None:
    reference = grids(20.0, 600, 1)
    model = grids(20.0, 2000, 2)

    result = res.paired_bootstrap_difference(model, model, reference, 60, np.random.default_rng(0))

    assert result["difference"] == pytest.approx(0.0)
    assert result["interval_low"] <= 0.0 <= result["interval_high"]
    assert result["bootstrap_sd"] == pytest.approx(0.0, abs=1e-12)


def test_a_clearly_wider_model_is_resolved_as_worse_with_an_interval_excluding_zero() -> None:
    reference = grids(20.0, 600, 3)
    good, bad = grids(20.0, 2000, 4), grids(30.0, 2000, 5)

    result = res.paired_bootstrap_difference(good, bad, reference, 80, np.random.default_rng(1))

    assert result["difference"] > 0.5
    assert result["interval_low"] > 0.0


def test_the_reference_only_spread_is_the_positive_noise_floor_of_the_metric() -> None:
    reference = grids(20.0, 400, 6)

    floor = res.reference_only_spread(reference, 40, np.random.default_rng(2))

    assert floor["mean"] > 0.0
    assert floor["sd"] > 0.0


def test_the_study_never_reads_a_sealed_set() -> None:
    text = Path(res.__file__).read_text(encoding="utf-8").lower()

    assert "geant4_electron_sealed" not in text
    assert "open_set" not in text


@needs_samples
def test_the_study_runs_end_to_end_on_a_small_slice() -> None:
    results = res.study((10.0,), n_boot=8, events=300, seeds=2)

    row = results["per_energy"]["10"]
    assert results["sealed_data_read"] is False
    assert set(row) >= {"paired_difference_candidate_minus_base", "reference_half_against_half", "model_seed_spread"}
    assert "mean_difference_over_energies" in results
