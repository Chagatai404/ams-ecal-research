"""The added validation checks: each row must see a planted difference and must not see none."""

from pathlib import Path

import numpy as np
import pytest

from ams_ecal.detector.geometry import load_geometry
from ams_ecal.validation.added_checks import (
    EventSample,
    classifier_material,
    classifier_two_sample_row,
    correlation_flags,
    event_features,
    lag_averaged_correlation,
    layer_correlation_row,
    multiscale_panel,
    occupancy,
    occupancy_material,
    run_cell,
    sparsity_row,
    summarise,
    verdicts,
    wilson_interval,
)

ROOT = Path(__file__).parents[2]
GEOMETRY = load_geometry(ROOT / "configs" / "geometry.yaml")
N_LAYERS, N_CELLS = GEOMETRY.number_of_layers, GEOMETRY.cells_per_layer


def layers(n: int, rho: float, seed: int) -> np.ndarray:
    """``n`` events of 18 layer energies sharing one latent with weight ``rho``."""

    rng = np.random.default_rng(seed)
    common = rng.standard_normal((n, 1))
    return np.exp(np.sqrt(rho) * common + np.sqrt(1 - rho) * rng.standard_normal((n, N_LAYERS)))


def grids(n: int, seed: int, hit_probability: float = 0.05, scale: float = 1.0) -> EventSample:
    rng = np.random.default_rng(seed)
    cells = rng.exponential(scale, (n, N_LAYERS, N_CELLS)) * (
        rng.random((n, N_LAYERS, N_CELLS)) < hit_probability
    )
    cells[:, :, 36] += 5.0  # a crossed cell, as a track would make
    return EventSample(cells, np.full(n, 4.5), np.full(n, 4.5))


def test_lag_averaged_correlation_is_one_for_identical_layers_and_zero_for_independent() -> None:
    same = np.repeat(layers(300, 0.0, 0)[:, :1], N_LAYERS, axis=1)
    independent = layers(3000, 0.0, 1)

    assert lag_averaged_correlation(same) == pytest.approx(1.0)
    assert np.abs(lag_averaged_correlation(independent)).max() < 0.06


def test_the_layer_correlation_row_sees_planted_correlation_and_only_that() -> None:
    reference = layers(1500, 0.5, 2)

    wrong = layer_correlation_row(layers(1500, 0.0, 3), reference, n_boot=40)
    right = layer_correlation_row(layers(1500, 0.5, 4), reference, n_boot=40)

    assert wrong["material"] and all(wrong["material_by_lag"])
    assert not right["material"]
    assert wrong["largest_pairwise_difference"] > 0.3


def test_a_difference_inside_the_bootstrap_interval_is_not_material() -> None:
    reference = layers(80, 0.3, 5)  # small sample: a wide interval
    model = layers(3000, 0.3, 6)

    row = layer_correlation_row(model, reference, n_boot=60)

    assert not row["material"]


def test_the_noise_floor_is_reported_against_the_reference() -> None:
    row = layer_correlation_row(
        layers(500, 0.4, 7), layers(500, 0.4, 8), noise_layers=layers(500, 0.4, 9), n_boot=20
    )

    assert len(row["noise_floor_difference"]) == len(row["lags"])


def test_event_features_have_the_declared_columns() -> None:
    sample = grids(20, 10)

    features, names = event_features(sample, GEOMETRY)

    assert features.shape == (20, 9 + 2 * N_LAYERS) and len(names) == features.shape[1]
    assert names[0] == "energy_mev" and "core_fraction" in names
    assert np.all(features[:, 0] > 0)


def test_the_classifier_sees_a_planted_difference_and_not_identical_samples() -> None:
    rng = np.random.default_rng(11)
    reference = rng.standard_normal((600, 6))
    same = rng.standard_normal((600, 6))
    shifted = rng.standard_normal((600, 6)) + np.array([0.0, 1.5, 0, 0, 0, 0])

    quiet = classifier_two_sample_row(same, reference, n_permutations=10)
    loud = classifier_two_sample_row(shifted, reference, n_permutations=10)

    assert quiet["auc"] < 0.60 and not quiet["material"]
    assert loud["auc"] > 0.75 and loud["material"]
    assert quiet["n_per_class"] == 600


def test_the_classifier_needs_the_effect_size_as_well_as_the_null() -> None:
    rng = np.random.default_rng(12)
    reference = rng.standard_normal((4000, 3))
    barely = rng.standard_normal((4000, 3)) + np.array([0.12, 0, 0])  # significant, but AUC ~ 0.53

    row = classifier_two_sample_row(barely, reference, n_permutations=10)

    assert row["auc"] < 0.60
    assert not row["material"]


def test_samples_of_unequal_size_are_cut_to_the_smaller() -> None:
    rng = np.random.default_rng(13)

    row = classifier_two_sample_row(
        rng.standard_normal((300, 3)), rng.standard_normal((500, 3)), n_permutations=5
    )

    assert row["n_per_class"] == 300


def test_wilson_interval_matches_the_textbook_value() -> None:
    low, high = wilson_interval(50, 100)

    assert (low, high) == pytest.approx((0.4038, 0.5962), abs=2e-4)
    assert wilson_interval(0, 100)[0] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_occupancy_is_the_fraction_of_cells_above_the_threshold() -> None:
    cells = np.zeros((4, N_LAYERS, N_CELLS))
    cells[:, :, :18] = 1.0

    assert occupancy(cells, 0.5) == pytest.approx(np.full(N_LAYERS, 0.25))
    assert occupancy(cells, 1.5) == pytest.approx(np.zeros(N_LAYERS))


def test_the_sparsity_row_needs_both_the_size_and_the_interval() -> None:
    reference = grids(400, 14)

    wrong = sparsity_row(grids(400, 15, hit_probability=0.12).grids, reference.grids)
    right = sparsity_row(grids(400, 16).grids, reference.grids)

    assert wrong["material"] and not right["material"]
    assert len(wrong["thresholds"]) == 3


def test_a_tiny_but_significant_occupancy_difference_is_not_material() -> None:
    reference = grids(2000, 17, hit_probability=0.050)

    row = sparsity_row(grids(2000, 18, hit_probability=0.060).grids, reference.grids)

    assert abs(row["thresholds"][0]["difference"]) < 0.02
    assert not row["material"]


def test_the_multiscale_panel_reports_only_reliable_windows_and_gives_no_verdict() -> None:
    panel = multiscale_panel(grids(30, 19).grids, n_events=30)

    assert panel and all(set(v) == {"n", "median", "q25", "q75"} for v in panel.values())
    assert all("q=" in key for key in panel)


def test_a_cell_runs_every_row_and_flags_a_planted_difference() -> None:
    reference = grids(300, 20)
    model = grids(300, 21, hit_probability=0.12)

    cell = run_cell(model, reference, GEOMETRY, noise=grids(300, 22), n_boot=20, n_permutations=5)
    summary = summarise({"planted": cell})

    assert set(verdicts(cell)) == {"layer_correlation", "classifier", "sparsity"}
    assert verdicts(cell)["sparsity"] and verdicts(cell)["classifier"]
    assert "multiscale" in cell and "verdict" not in cell["multiscale"]
    assert summary["flag_counts"]["sparsity"] == 1 and summary["n_cells"] == 1


def test_a_sample_must_be_consistent() -> None:
    with pytest.raises(ValueError, match="same length"):
        EventSample(np.zeros((3, N_LAYERS, N_CELLS)), np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match="events, layers, cells"):
        EventSample(np.zeros((3, N_CELLS)), np.zeros(3), np.zeros(3))


def test_a_correlation_lag_is_material_only_when_both_conditions_hold() -> None:
    low, high = np.array([0.20, 0.20, 0.20, 0.20]), np.array([0.30, 0.30, 0.30, 0.30])
    model = np.array([0.45, 0.32, 0.45, 0.25])  # outside+big, outside+small, outside+big(-), inside
    difference = np.array([0.15, 0.07, -0.12, 0.00])

    flags = correlation_flags(difference, model, low, high)

    assert flags.tolist() == [True, False, True, False]
    # big difference but a wide interval around the reference: not material
    assert not correlation_flags(np.array([0.15]), np.array([0.40]), np.array([0.0]), np.array([0.6]))[0]
    # exactly the size threshold counts
    assert correlation_flags(np.array([0.10]), np.array([0.6]), np.array([0.2]), np.array([0.3]))[0]


def test_the_classifier_verdict_needs_the_effect_size_and_the_null() -> None:
    assert classifier_material(0.70, 0.52)  # both
    assert not classifier_material(0.58, 0.52)  # above the null, too small an effect
    assert not classifier_material(0.65, 0.66)  # large, but within the null
    assert classifier_material(0.60, 0.52)  # exactly the effect-size threshold counts


def test_the_occupancy_verdict_needs_the_size_and_the_interval() -> None:
    assert occupancy_material(0.03, 0.08, 0.04, 0.06)  # both
    assert not occupancy_material(0.03, 0.08, 0.00, 0.10)  # big, inside the interval
    assert not occupancy_material(0.01, 0.08, 0.04, 0.06)  # outside the interval, too small
    assert occupancy_material(-0.02, 0.03, 0.04, 0.06)  # exactly the threshold, below the interval
