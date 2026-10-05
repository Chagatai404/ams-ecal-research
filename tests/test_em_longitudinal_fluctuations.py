"""The longitudinal-fluctuation check: the fit recovers what is planted, and the control tells the truth."""

import numpy as np
import pytest

from ams_ecal.em_longitudinal_fluctuations import (
    DATA,
    analyse_energy,
    fit_cumulative_fractions,
    fit_events,
    fit_layer_fractions,
    grindhammer_peters_sampling,
    lateral_coupling,
    layer_bounds_x0,
    planted_control,
    profile_fractions,
)

BOUNDS = layer_bounds_x0()


def test_the_layers_tile_the_calorimeter_depth_and_a_profile_never_exceeds_the_primary_energy() -> None:
    assert BOUNDS.shape == (18, 2)
    assert np.all(BOUNDS[:, 1] > BOUNDS[:, 0]) and np.all(BOUNDS[1:, 0] >= BOUNDS[:-1, 1] - 1e-9)

    fractions = profile_fractions(np.log(5.0), np.log(5.0), BOUNDS)

    assert np.all(fractions >= 0.0) and 0.8 < fractions.sum() <= 1.0  # the rest leaks out the back


@pytest.mark.parametrize("fit", [fit_layer_fractions, fit_cumulative_fractions])
def test_a_noiseless_profile_is_recovered_exactly(fit) -> None:
    truth = np.array([np.log(5.5), np.log(5.0)])

    recovered = fit(profile_fractions(*truth, BOUNDS), BOUNDS)

    assert recovered == pytest.approx(truth, abs=1e-5)


def test_the_grindhammer_peters_values_at_10_gev_match_the_hand_computation() -> None:
    values = grindhammer_peters_sampling(10_000.0, 7.6)

    assert values["ln_y"] == pytest.approx(np.log(10_000.0 / 7.6))
    assert values["sigma_ln_t"] == pytest.approx(1.0 / (-2.5 + 1.25 * 7.1796), rel=1e-3)
    assert values["sigma_ln_alpha"] == pytest.approx(1.0 / (-0.82 + 0.79 * 7.1796), rel=1e-3)
    assert values["rho"] == pytest.approx(0.784 - 0.023 * 7.1796, rel=1e-3)
    assert grindhammer_peters_sampling(100_000.0, 7.6)["rho"] < values["rho"]  # falls with energy


def test_the_control_recovers_a_planted_correlation_without_noise_and_zero_when_none_is_planted() -> None:
    mean, spread, quiet = np.array([1.9, 1.6]), np.array([0.12, 0.16]), np.zeros(18)

    weak = planted_control(BOUNDS, mean, spread, quiet, 0.0, fit_layer_fractions, n_events=80)
    strong = planted_control(BOUNDS, mean, spread, quiet, 0.8, fit_layer_fractions, n_events=80)

    assert abs(weak) < 0.2
    assert strong == pytest.approx(0.8, abs=0.1)


def test_a_fit_does_not_manufacture_a_correlation_from_noise() -> None:
    mean, spread = np.array([1.9, 1.6]), np.array([0.12, 0.16])
    noise = np.full(18, 0.1)

    for fit in (fit_layer_fractions, fit_cumulative_fractions):
        assert abs(planted_control(BOUNDS, mean, spread, noise, 0.0, fit, n_events=150)) < 0.25


def test_fitting_many_events_returns_one_row_per_event() -> None:
    fractions = np.array([profile_fractions(np.log(t), np.log(a), BOUNDS) for t, a in ((5, 5), (6, 6))])

    rows = fit_events(fractions, fit_layer_fractions, BOUNDS)

    assert rows.shape == (2, 2)
    assert np.exp(rows[1, 0]) > np.exp(rows[0, 0])  # the deeper maximum is found deeper


@pytest.mark.skipif(not (DATA / "E10GeV").is_dir(), reason="the exploration electron sample is not present")
def test_the_analysis_reports_both_estimators_the_control_and_the_literature_values() -> None:
    result = analyse_energy(10.0, control_events=30)

    assert result["n_events"] == 1000
    for key in ("least_squares", "cumulative"):
        assert len(result[key]["mean"]) == len(result[key]["std"]) == 2
        assert -1.0 <= result[key]["rho"] <= 1.0
    assert set(result["control_recovered_rho"]) == {"planted_rho_0", "planted_rho_0.6"}
    assert result["grindhammer_peters_sampling"]["sigma_ln_t"] > 0
    assert 0.5 < result["contained_fraction_median"] <= 1.0
    assert set(result["lateral_coupling_spearman"]) >= {"width_mm", "n_hit_cells"}


def test_the_lateral_coupling_sees_a_planted_dependence_and_not_independent_variables() -> None:
    rng = np.random.default_rng(3)
    parameters = rng.standard_normal((600, 2))
    observables = {
        name: rng.standard_normal(600) for name in ("width_mm", "n_hit_cells", "core_fraction", "containment_fraction", "max_cell_fraction")
    }
    observables["width_mm"] = 2.0 * parameters[:, 0] + 0.3 * rng.standard_normal(600)

    result = lateral_coupling(parameters, observables)

    assert result["width_mm"]["with_ln_t"] > 0.9
    assert abs(result["width_mm"]["with_ln_alpha"]) < 0.15
    assert abs(result["n_hit_cells"]["with_ln_t"]) < 0.15
