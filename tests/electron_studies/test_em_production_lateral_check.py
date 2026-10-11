"""Slice 2 lateral development check: the metrics, the floor logic, the proposed signs and the sealed-data rule."""

from pathlib import Path

import numpy as np
import pytest

from ams_ecal.detector.geometry import load_geometry
from ams_ecal.electron_model import em_production_lateral as lat
from ams_ecal.electron_model import em_production_lateral_calibration as cal
from ams_ecal.electron_studies import em_production_lateral_check as check
from ams_ecal.validation.dataset import GEOMETRY_CONFIG

needs_samples = pytest.mark.skipif(
    not (
        (check.BASELINE_DIR / "E10GeV" / "events.npz").exists()
        and (check.EXTENDED_DIR / "E10GeV" / "events.npz").exists()
        and lat.DEFAULT_ARTIFACT.exists()
    ),
    reason="needs the git-ignored exposed samples (regenerate them from their configs) and the lateral artifact",
)


def synthetic_sample(events: int = 300, seed: int = 3) -> check.Sample:
    model = lat.build_model(cal.START)
    return check.slice2_sample(model, 50.0, events, seed)


def test_the_profile_error_of_identical_grids_is_zero_and_in_points() -> None:
    sample = synthetic_sample(200)

    assert check.profile_errors_points(sample.grid, sample.grid) == pytest.approx(np.zeros(18))
    other = synthetic_sample(200, seed=4)
    assert (check.profile_errors_points(sample.grid, other.grid) < 100).all()


def test_a_sample_compared_with_itself_has_zero_distance_and_coupling_difference() -> None:
    geometry = load_geometry(GEOMETRY_CONFIG)
    sample = synthetic_sample(300)

    result = check.comparison(sample, sample, geometry)

    assert max(result["ks"].values()) == pytest.approx(0.0)
    assert result["coupling_spearman"]["difference"] == pytest.approx(0.0)
    assert result["spot_number_rms_log_ratio"] == pytest.approx(0.0)
    assert result["profile_error_points"]["layers_3_17_max"] == pytest.approx(0.0)


def test_the_conditioned_spot_numbers_are_positive_for_every_layer() -> None:
    numbers = check.conditioned_spot_numbers(synthetic_sample(400).grid)

    assert numbers.shape == (18,)
    assert (numbers > 0).all()


def test_the_proposed_signs_are_reported_against_the_floor() -> None:
    good = {
        "profile_error_points": {"layers_3_17_max": 1.5},
        "coupling_spearman": {"difference": 0.03},
        "spot_number_rms_log_ratio": 0.2,
    }
    bad = {
        "profile_error_points": {"layers_3_17_max": 3.0},
        "coupling_spearman": {"difference": 0.2},
        "spot_number_rms_log_ratio": 0.5,
    }
    floor = {"coupling_spearman": {"difference": 0.04}}

    assert all(check.signs(good, floor).values())
    assert not any(check.signs(bad, floor).values())


def test_the_check_never_reads_a_sealed_set() -> None:
    text = Path(check.__file__).read_text(encoding="utf-8").lower()

    assert "geant4_electron_sealed" not in text
    assert "open_set" not in text


@needs_samples
def test_the_check_runs_end_to_end_and_compares_with_the_floor_and_the_control(
    tmp_path,
) -> None:
    results, artefacts = check.development_check((10.0,), ensemble_events=300, control_events=40, seeds=1)

    row = results["per_energy"]["10"]
    assert results["sealed_data_read"] is False
    assert set(row) >= {
        "sample_to_sample_floor",
        "calibration_set",
        "development_holdout",
    }
    for label in ("calibration_set", "development_holdout"):
        assert set(row[label]) == {
            "slice2",
            "slice1_with_dec001_lateral",
            "proposed_signs_met",
        }

    from ams_ecal.electron_studies.em_production_lateral_check_plots import make_plots

    make_plots(results, artefacts, tmp_path)

    assert (tmp_path / "profile_error_by_layer.png").exists()
    assert (tmp_path / "spot_numbers.png").exists()
    assert (tmp_path / "ks_distances.png").exists()
