"""Slice 1 development check: structure, the comparison floor and the refusal to read sealed data."""

from pathlib import Path

import numpy as np
import pytest

from ams_ecal.electron_studies import em_production_development_check as check

needs_samples = pytest.mark.skipif(
    not (
        (check.BASELINE_DIR / "E10GeV" / "events.npz").exists()
        and (check.EXTENDED_DIR / "E10GeV" / "events.npz").exists()
    ),
    reason="the exposed baseline and extended-depth samples are git-ignored: regenerate them from their configs",
)


def test_the_leakage_summary_describes_the_energy_beyond_the_last_layer() -> None:
    fractions = np.tile(np.full(18, 0.05), (10, 1))  # contained 0.9 in every event

    summary = check.leakage_summary(fractions)

    assert summary["mean"] == pytest.approx(0.1)
    assert summary["sd"] == pytest.approx(0.0, abs=1e-12)
    assert summary["q95"] == pytest.approx(0.1)


def test_the_mean_and_spread_of_distance_tables_over_seeds() -> None:
    tables = [{"a": 1.0, "b": 10.0}, {"a": 3.0, "b": 10.0}]

    out = check._mean_sd(tables)

    assert out["a"]["mean"] == 2.0
    assert out["a"]["sd"] == pytest.approx(np.sqrt(2.0))
    assert out["b"]["sd"] == 0.0


def test_the_check_never_reads_a_sealed_set() -> None:
    text = Path(check.__file__).read_text(encoding="utf-8").lower()

    assert "geant4_electron_sealed" not in text
    assert "open_set" not in text


@needs_samples
def test_the_check_runs_end_to_end_and_compares_against_both_samples_and_dec001(tmp_path) -> None:
    results, artefacts = check.development_check((10.0,), ensemble_events=400, seeds=2)

    block = results["per_energy"]["10"]
    assert results["sealed_data_read"] is False
    assert set(block) >= {"calibration_set", "development_holdout", "sample_to_sample_floor", "leakage", "moments_slice1"}
    for label in ("calibration_set", "development_holdout"):
        assert set(block[label]) == {"slice1", "dec001", "removed_vs_dec001"}
    hold = block["development_holdout"]
    assert hold["slice1"]["layer0_log_ratio"]["mean"] < hold["dec001"]["layer0_log_ratio"]

    from ams_ecal.electron_studies.em_production_development_check_plots import (
        make_plots,
    )

    make_plots(results, artefacts, tmp_path)

    assert (tmp_path / "distances.png").exists()
    assert (tmp_path / "mean_layer_errors.png").exists()
