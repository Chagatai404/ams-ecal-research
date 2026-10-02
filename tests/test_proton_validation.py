"""The held-out comparison machinery, on synthetic distributions with known answers."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ams_ecal.geant4_backend import OUTPUT_SCHEMA_VERSION
from ams_ecal.proton_validation import (
    compare_distribution,
    interaction_section,
    split_batch,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260930)


def gamma_sample(rng, n, scale=1.0):
    return rng.gamma(3.0, 2.0, n) * scale


# --- one distribution against one held-out sample ------------------------


def test_the_same_distribution_is_not_a_material_discrepancy(rng) -> None:
    result = compare_distribution(
        gamma_sample(rng, 20_000), gamma_sample(rng, 500), gamma_sample(rng, 1500)
    )

    assert not result["material"]
    assert result["ks"] < 0.10
    assert result["at_floor"]


def test_a_thirty_percent_scale_error_is_a_material_discrepancy(rng) -> None:
    result = compare_distribution(gamma_sample(rng, 20_000, 1.3), gamma_sample(rng, 500))

    assert result["material"]
    assert any(result["material_by_quantile"])
    assert result["material_by_ks"]


def test_a_missing_tail_is_caught_by_the_quantiles_even_if_the_bulk_agrees(rng) -> None:
    heldout = np.concatenate([gamma_sample(rng, 450), gamma_sample(rng, 50, 8.0)])
    model = gamma_sample(rng, 20_000)

    result = compare_distribution(model, heldout)

    assert result["material_by_quantile"][-1]  # the 95% quantile


def test_a_well_measured_shift_below_ten_percent_is_not_material_by_quantile(rng) -> None:
    # with 5000 held-out events the quantiles are known to a few percent
    result = compare_distribution(gamma_sample(rng, 20_000, 1.06), gamma_sample(rng, 5000))

    assert not any(result["material_by_quantile"])


def test_a_shift_inside_the_heldout_interval_is_not_flagged_however_large(rng) -> None:
    # a tiny held-out sample cannot resolve a 12% shift: its interval is wide
    result = compare_distribution(gamma_sample(rng, 20_000, 1.12), gamma_sample(rng, 25))

    lower, upper = result["heldout_quantile_interval"]
    for flagged, model_q, low, high in zip(
        result["material_by_quantile"], result["model_quantiles"], lower, upper, strict=True
    ):
        if low <= model_q <= high:
            assert not flagged


def test_the_result_reports_both_sets_of_quantiles_and_the_distance(rng) -> None:
    result = compare_distribution(gamma_sample(rng, 5000), gamma_sample(rng, 300))

    assert result["quantile_levels"] == [0.05, 0.16, 0.5, 0.84, 0.95]
    assert len(result["model_quantiles"]) == len(result["heldout_quantiles"]) == 5
    assert result["wasserstein"] >= 0
    assert result["n_model"] == 5000 and result["n_heldout"] == 300


def test_the_comparison_is_reproducible(rng) -> None:
    a, b = gamma_sample(rng, 2000), gamma_sample(rng, 300)

    assert compare_distribution(a, b, seed=3) == compare_distribution(a, b, seed=3)


# --- the interaction row ---------------------------------------------------


def stub_model(length_mm: float):
    return SimpleNamespace(
        effective_length_mm=length_mm, calibration=SimpleNamespace(depth_mm=166.5)
    )


def heldout_interactions(rng, n, length_mm=255.0):
    distance = rng.exponential(length_mm, n)
    occurred = distance < 166.5
    return {"truth_occurred": occurred, "truth_z_mm": np.where(occurred, distance, 0.0)}


def test_data_drawn_from_the_modelled_law_rarely_fails_the_interaction_row() -> None:
    # A 95% interval excludes the truth 5% of the time by construction, so a
    # correct model is flagged now and then; the rate is the property to test.
    flagged = [
        interaction_section(
            stub_model(255.0), heldout_interactions(np.random.default_rng(seed), 1000)
        )["material"]
        for seed in range(100)
    ]

    assert np.mean(flagged) < 0.15


def test_the_modelled_survival_probability_is_the_exponential_at_the_ecal_depth() -> None:
    result = interaction_section(
        stub_model(255.0), heldout_interactions(np.random.default_rng(0), 1000)
    )

    assert result["model_no_inelastic_fraction"] == pytest.approx(0.5206, abs=1e-3)


def test_data_from_a_different_length_fails_the_interaction_row(rng) -> None:
    result = interaction_section(stub_model(255.0), heldout_interactions(rng, 4000, length_mm=120.0))

    assert result["material"]


# --- reading a batch -----------------------------------------------------------


def test_a_batch_splits_into_disjoint_calibration_and_validation_events(tmp_path) -> None:
    n = 40
    directory = tmp_path / "baseline" / "E10GeV"
    directory.mkdir(parents=True)
    np.savez(
        directory / "events.npz",
        event_index=np.arange(n),
        truth_occurred=np.zeros(n, dtype=bool),
        first_secondary_offsets=np.zeros(n + 1, dtype=np.int64),
    )
    (directory / "metadata.json").write_text(
        json.dumps({"output_schema_version": OUTPUT_SCHEMA_VERSION}), encoding="utf-8"
    )

    calibration, validation, _ = split_batch("baseline", 10.0, tmp_path)

    assert set(calibration["event_index"]).isdisjoint(validation["event_index"])
    assert len(calibration["event_index"]) + len(validation["event_index"]) == n
    assert all(index % 4 == 3 for index in validation["event_index"])
