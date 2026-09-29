"""Pilot statistics and observables on synthetic events with known answers."""

from pathlib import Path

import numpy as np
import pytest

from ams_ecal.geometry import load_geometry
from ams_ecal.pilot_analysis import (
    bootstrap_interval,
    censored_exponential_rate,
    conditional_variance_fraction,
    event_observables,
    layer_centres_mm,
    mip_scale,
    pearson,
    spearman,
    track_cells,
    wilson_interval,
)

GEOMETRY = Path(__file__).parents[1] / "configs" / "geometry.yaml"


@pytest.fixture
def geometry():
    return load_geometry(GEOMETRY)


def through_going(n_layers: int, per_cell: float = 1.0, cell: int = 36) -> np.ndarray:
    grid = np.zeros((1, n_layers, 72))
    grid[0, :, cell] = per_cell
    return grid


# --- observables --------------------------------------------------------


def test_a_straight_track_has_textbook_observables(geometry) -> None:
    grid = through_going(18)
    obs = event_observables(
        grid, geometry, np.array([4.5]), np.array([4.5]),
        cell_threshold_mev=0.5, layer_threshold_mev=0.5,
    )
    z = layer_centres_mm(geometry, 18)
    assert obs["energy_mev"][0] == 18.0
    assert obs["long_cog_mm"][0] == pytest.approx(z.mean())
    assert obs["long_rms_mm"][0] == pytest.approx(z.std())
    assert obs["n_active_layers"][0] == 18 and obs["last_active_layer"][0] == 17
    assert obs["width_mm"][0] == pytest.approx(0.0)  # the track sits at the cell centre
    assert obs["core_fraction"][0] == 1.0 and obs["containment_fraction"][0] == 1.0
    assert obs["n_hit_cells"][0] == 18
    assert obs["participation"][0] == pytest.approx(18.0)
    assert obs["max_cell_fraction"][0] == pytest.approx(1 / 18)


def test_a_single_hot_cell_is_maximally_concentrated(geometry) -> None:
    grid = np.zeros((1, 18, 72))
    grid[0, 7, 40] = 50.0  # four cells beside the track cell 36
    obs = event_observables(
        grid, geometry, np.array([4.5]), np.array([4.5]),
        cell_threshold_mev=0.5, layer_threshold_mev=0.5,
    )
    assert obs["max_layer"][0] == 7
    assert obs["long_rms_mm"][0] == pytest.approx(0.0)
    assert obs["width_mm"][0] == pytest.approx(4 * 9.0)
    assert obs["core_fraction"][0] == 0.0 and obs["containment_fraction"][0] == 0.0
    assert obs["participation"][0] == pytest.approx(1.0)


def test_thresholds_decide_activity_counts(geometry) -> None:
    grid = through_going(18, per_cell=1.0)
    grid[0, 3:, 36] = 0.2  # a faint tail
    obs_low = event_observables(grid, geometry, np.array([4.5]), np.array([4.5]),
                                cell_threshold_mev=0.1, layer_threshold_mev=0.1)
    obs_high = event_observables(grid, geometry, np.array([4.5]), np.array([4.5]),
                                 cell_threshold_mev=0.5, layer_threshold_mev=0.5)
    assert obs_low["n_hit_cells"][0] == 18 and obs_high["n_hit_cells"][0] == 3
    assert obs_high["last_active_layer"][0] == 2


def test_the_same_code_reads_an_extended_grid(geometry) -> None:
    grid = through_going(270)
    obs = event_observables(grid, geometry, np.array([4.5]), np.array([4.5]),
                            cell_threshold_mev=0.5, layer_threshold_mev=0.5)
    assert obs["n_active_layers"][0] == 270
    assert obs["long_cog_mm"][0] == pytest.approx(135 * 9.25)


def test_empty_events_give_nan_not_errors(geometry) -> None:
    obs = event_observables(np.zeros((2, 18, 72)), geometry, np.zeros(2), np.zeros(2),
                            cell_threshold_mev=0.5, layer_threshold_mev=0.5)
    assert np.all(np.isnan(obs["long_cog_mm"])) and np.all(obs["max_layer"] == -1)


def test_mip_scale_reads_the_crossed_cell(geometry) -> None:
    grids = np.concatenate([through_going(18, 0.8), through_going(18, 1.2)])
    scale = mip_scale(grids, geometry, np.array([4.5, 4.5]), np.array([4.5, 4.5]))
    assert scale.cell_mev == pytest.approx(1.0)
    assert scale.layer_mev == pytest.approx(1.0)
    assert track_cells(geometry, 18, np.array([4.5]), np.array([-4.5]))[0].tolist() == (
        [35, 35, 36, 36] * 4 + [35, 35]
    )


# --- statistics ---------------------------------------------------------


def test_variance_fraction_recovers_a_known_share() -> None:
    rng = np.random.default_rng(3)
    d = rng.uniform(0, 1, 20_000)
    y = 2.0 * d + rng.normal(0, 0.4, d.size)
    # Var(2D) = 4/12 = 1/3; noise variance 0.16 -> S_D = 0.3333 / 0.4933.
    truth = (1 / 3) / (1 / 3 + 0.16)
    result = conditional_variance_fraction(d, y, n_bins=20)
    # Binning a linear trend loses the within-bin slope: about 1/K^2 of it.
    assert result.epsilon_squared == pytest.approx(truth, abs=0.01)
    assert result.eta_squared >= result.epsilon_squared


def test_variance_fraction_is_near_zero_without_a_relation() -> None:
    rng = np.random.default_rng(4)
    d, y = rng.normal(size=2000), rng.normal(size=2000)
    result = conditional_variance_fraction(d, y, n_bins=10)
    assert result.epsilon_squared == pytest.approx(0.0, abs=0.01)
    # The raw eta^2 carries the expected upward bias of about (K-1)/N.
    assert result.eta_squared == pytest.approx(9 / 2000, abs=0.006)


def test_variance_fraction_refuses_too_few_events() -> None:
    with pytest.raises(ValueError, match="too few"):
        conditional_variance_fraction(np.arange(10.0), np.arange(10.0), n_bins=10)


def test_correlations_and_their_bootstrap_interval() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=4000)
    y = 0.6 * x + 0.8 * rng.normal(size=4000)
    assert pearson(x, y) == pytest.approx(0.6, abs=0.03)
    assert spearman(x, x**3) == pytest.approx(1.0)
    interval = bootstrap_interval(pearson, [x, y], n_boot=300, seed=1)
    assert interval.low < 0.6 < interval.high
    assert interval.high - interval.low < 0.06


def test_wilson_interval_matches_a_hand_calculation() -> None:
    interval = wilson_interval(55, 100)
    assert interval.estimate == 0.55
    assert interval.low == pytest.approx(0.4524, abs=1e-3)
    assert interval.high == pytest.approx(0.6438, abs=1e-3)


def test_censored_exponential_rate_recovers_the_interaction_length() -> None:
    rng = np.random.default_rng(6)
    length, true_lambda = 166.5, 270.0
    depths = rng.exponential(true_lambda, 50_000)
    inside = depths[depths <= length]
    fit = censored_exponential_rate(inside, int((depths > length).sum()), length)
    assert 1 / fit.rate_per_mm == pytest.approx(true_lambda, rel=0.02)
    assert fit.survival(length) == pytest.approx(np.exp(-length / true_lambda), abs=0.005)
