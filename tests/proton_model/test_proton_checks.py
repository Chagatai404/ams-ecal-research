"""Paired variant checks and the materiality convention, on synthetic events."""

import json
from pathlib import Path

import numpy as np
import pytest

from ams_ecal.detector.geometry import load_geometry
from ams_ecal.geant4_simulation.geant4_backend import OUTPUT_SCHEMA_VERSION
from ams_ecal.proton_model.proton_calibration import validation_mask
from ams_ecal.proton_model.proton_checks import (
    compare_variant,
    is_material,
    load_aligned,
    material_flags,
    paired_ratio,
)

GEOMETRY = Path(__file__).parents[2] / "configs" / "geometry.yaml"
DEPTH_MM = 166.5


@pytest.fixture
def geometry():
    return load_geometry(GEOMETRY)


def synthetic_events(n: int = 240, seed: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    occurred = rng.random(n) < 0.5

    def grid() -> np.ndarray:
        g = rng.gamma(2.0, 1.0, size=(n, 18, 72)) * (rng.random((n, 18, 72)) < 0.05)
        g[:, :, 36] += 0.6  # a track cell in every layer, so no event is empty
        return g

    return {
        "truth_occurred": occurred,
        "truth_z_mm": np.where(occurred, rng.uniform(1.0, 150.0, n), 0.0),
        "entry_x_mm": rng.uniform(0.0, 9.0, n),
        "entry_y_mm": rng.uniform(0.0, 9.0, n),
        "readout_grid_mev": grid(),
        "deposit_grid_mev": grid() * 10.0,
    }


# --- the materiality convention ----------------------------------------


def test_a_small_shift_is_not_material_however_precise() -> None:
    assert not is_material(0.04, standard_error=0.001, baseline_value=1.0)


def test_a_large_shift_is_material_when_it_is_precisely_measured() -> None:
    assert is_material(0.10, standard_error=0.01, baseline_value=1.0)


def test_a_large_shift_is_not_material_when_it_is_within_twice_its_error() -> None:
    assert not is_material(0.10, standard_error=0.06, baseline_value=1.0)


def test_the_convention_is_symmetric_in_sign() -> None:
    assert is_material(-0.10, standard_error=0.01, baseline_value=1.0)


# --- paired ratios ------------------------------------------------------


def test_identical_samples_give_a_ratio_of_one_and_no_flag() -> None:
    values = np.random.default_rng(2).gamma(3.0, 5.0, 400)

    result = paired_ratio(values, values.copy(), np.median)

    assert result["ratio"] == pytest.approx(1.0)
    assert result["standard_error"] == pytest.approx(0.0, abs=1e-12)
    assert not result["material"]


def test_a_twenty_percent_scale_change_is_recovered_and_flagged() -> None:
    values = np.random.default_rng(3).gamma(3.0, 5.0, 400)

    result = paired_ratio(values, 1.2 * values, np.mean)

    assert result["ratio"] == pytest.approx(1.2)
    assert result["material"]


def test_independent_noise_of_the_same_distribution_is_not_flagged() -> None:
    rng = np.random.default_rng(4)

    result = paired_ratio(rng.gamma(3.0, 5.0, 600), rng.gamma(3.0, 5.0, 600), np.median)

    assert not result["material"]


def test_the_bootstrap_is_reproducible() -> None:
    rng = np.random.default_rng(5)
    a, b = rng.gamma(3.0, 5.0, 300), rng.gamma(3.0, 5.0, 300)

    assert paired_ratio(a, b, seed=7) == paired_ratio(a, b, seed=7)


# --- the full comparison -------------------------------------------------


def test_a_variant_identical_to_the_baseline_raises_no_flag(geometry) -> None:
    baseline = synthetic_events()
    variant = {name: value.copy() for name, value in baseline.items()}

    comparison = compare_variant(baseline, variant, geometry, DEPTH_MM)

    assert material_flags(comparison) == []
    assert comparison["interaction"]["status_agreement"] == 1.0
    assert comparison["interacting"]["depth_shift_median_abs_mm"] == 0.0


def test_rescaling_the_fibre_signal_is_flagged_but_not_the_deposit(geometry) -> None:
    baseline = synthetic_events()
    variant = {name: value.copy() for name, value in baseline.items()}
    variant["readout_grid_mev"] = variant["readout_grid_mev"] * 1.3

    flags = material_flags(compare_variant(baseline, variant, geometry, DEPTH_MM))

    assert "interacting.fibre_energy_median" in flags
    assert "crossing.fibre_energy_mean" in flags
    assert "interacting.total_deposit_median" not in flags
    assert "crossing.total_deposit_mean" not in flags


# --- alignment of two samples ---------------------------------------------


def write_batch(root: Path, sample: str, seeds: np.ndarray) -> None:
    directory = root / sample / "E10GeV"
    directory.mkdir(parents=True)
    n = len(seeds)
    np.savez(
        directory / "events.npz",
        event_index=np.arange(n),
        seed=seeds,
        truth_occurred=np.zeros(n, dtype=bool),
    )
    (directory / "metadata.json").write_text(
        json.dumps({"output_schema_version": OUTPUT_SCHEMA_VERSION}), encoding="utf-8"
    )


def test_aligned_arrays_contain_only_calibration_events(tmp_path) -> None:
    seeds = np.arange(100, 140)
    write_batch(tmp_path, "baseline", seeds)
    write_batch(tmp_path, "variant", seeds[:24])

    baseline, variant, _ = load_aligned("baseline", "variant", 10.0, tmp_path)

    assert len(baseline["event_index"]) == len(variant["event_index"]) == 18
    assert not validation_mask(baseline["event_index"]).any()
    assert np.array_equal(baseline["seed"], variant["seed"])


def test_samples_run_at_different_seeds_are_refused(tmp_path) -> None:
    write_batch(tmp_path, "baseline", np.arange(100, 140))
    write_batch(tmp_path, "variant", np.arange(500, 540))

    with pytest.raises(ValueError, match="same seeds"):
        load_aligned("baseline", "variant", 10.0, tmp_path)
