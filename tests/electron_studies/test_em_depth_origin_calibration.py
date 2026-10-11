"""Depth-origin calibration: planted recovery, identifiability machinery, gate and floor logic."""

from types import SimpleNamespace

import numpy as np
import pytest

import ams_ecal.electron_studies.em_depth_origin_calibration as calibration
from ams_ecal.electron_studies.em_depth_origin_calibration import (
    DETERMINISTIC,
    PRIMARY,
    ModelSpec,
    Targets,
    analyse,
    ensemble_mean_fractions,
    entrance_gate,
    fit_model,
    floor_study,
    held_out_score,
    hessian_summary,
    predict,
    stochastic_structure,
    surface_summary,
    x_max_of_energy,
)
from ams_ecal.electron_studies.em_longitudinal_fluctuations import layer_bounds_x0
from ams_ecal.electron_studies.em_longitudinal_structure_analysis import gp_formulae

BOUNDS = layer_bounds_x0()
ENERGIES_GEV = np.array([10.0, 20.0, 50.0, 100.0])
CRITICAL_ENERGY_MEV = 7.6
GATE_CONDITIONS = {
    "layer0_residual_large_at_every_energy",
    "layer0_residual_significant_at_every_energy",
    "same_sign_at_every_energy",
    "reproducible_magnitude_across_energy",
    "not_absorbed_by_alternatives",
    "not_a_leakage_accounting_problem",
}


def planted_targets(
    spec: ModelSpec, params: dict[str, float], *, n_events: int = 400, noise: float = 0.04, seed: int = 3
) -> Targets:
    """Events scattered around the model's own mean, so the calibration has a known answer."""

    rng = np.random.default_rng(seed)
    blank = np.zeros((len(ENERGIES_GEV), 18))
    skeleton = Targets(ENERGIES_GEV, blank, np.ones_like(blank), [], CRITICAL_ENERGY_MEV, BOUNDS)
    model, _ = predict(spec, {**calibration.DEFAULTS, **spec.fixed, **params}, skeleton)
    events = [
        np.clip(row * (1.0 + noise * rng.standard_normal((n_events, 18))), 0.0, None) for row in model
    ]
    mean = np.array([e.mean(axis=0) for e in events])
    se = np.array([e.std(axis=0, ddof=1) / np.sqrt(n_events) for e in events])
    return Targets(ENERGIES_GEV, mean, se, events, CRITICAL_ENERGY_MEV, BOUNDS)


def with_floor(floor_a: float, floor_b: float) -> Targets:
    spec = ModelSpec("planted", (), {"z0": -1.5, "beta": 0.58, "floor_a": floor_a, "floor_b": floor_b})
    return planted_targets(spec, {})


# ------------------------------------------------------------------ model and coordinates


def test_the_depth_law_and_the_origin_sign_convention() -> None:
    assert x_max_of_energy(7.6 * np.e**3, 7.6, -0.5) == pytest.approx(2.5)
    centres = BOUNDS.mean(axis=1)
    for z0 in (0.0, -2.0):
        fractions = ensemble_mean_fractions(
            z0, 0.6, np.array([5.0]), np.array([20000.0]), CRITICAL_ENERGY_MEV, BOUNDS, fluctuate=False
        )[0]
        # the mode in u = x - z0 sits at x_max however far upstream the origin is placed
        assert abs(centres[fractions.argmax()] - 5.0) <= 1.5


def test_a_more_upstream_origin_with_the_same_x_max_changes_the_profile_shape() -> None:
    args = (np.array([5.0]), np.array([20000.0]), CRITICAL_ENERGY_MEV, BOUNDS)

    front = ensemble_mean_fractions(0.0, 0.6, *args, fluctuate=False)
    upstream = ensemble_mean_fractions(-2.0, 0.6, *args, fluctuate=False)

    assert not np.allclose(front, upstream, atol=1e-4)


def test_the_fluctuating_mean_reduces_to_the_deterministic_profile_when_the_widths_vanish(monkeypatch) -> None:
    def no_scatter(ln_y: float) -> dict[str, dict[str, float]]:
        return {"sampling": {"sigma_ln_t": 0.0, "sigma_ln_alpha": 0.0, "rho": 0.0}}

    monkeypatch.setattr(calibration, "gp_formulae", no_scatter)
    args = (-1.0, 0.57, np.array([4.0, 6.0]), np.array([10000.0, 50000.0]), CRITICAL_ENERGY_MEV, BOUNDS)

    stochastic = ensemble_mean_fractions(*args, fluctuate=True)
    deterministic = ensemble_mean_fractions(*args, fluctuate=False)

    assert stochastic == pytest.approx(deterministic, abs=1e-10)


def test_fluctuations_change_the_ensemble_mean_and_the_covariant_width_differs_from_the_naive_width() -> None:
    args = (-2.0, 0.6, np.array([4.0]), np.array([10000.0]), CRITICAL_ENERGY_MEV, BOUNDS)

    smooth = ensemble_mean_fractions(*args, fluctuate=False)
    covariant = ensemble_mean_fractions(*args, covariant=True)
    naive = ensemble_mean_fractions(*args, covariant=False)

    assert not np.allclose(smooth, covariant, atol=1e-5)
    assert not np.allclose(naive, covariant, atol=1e-6)


def test_the_covariant_and_naive_widths_coincide_when_the_origin_is_the_front_face() -> None:
    args = (0.0, 0.6, np.array([4.0]), np.array([10000.0]), CRITICAL_ENERGY_MEV, BOUNDS)

    covariant = ensemble_mean_fractions(*args, covariant=True)
    naive = ensemble_mean_fractions(*args, covariant=False)

    assert covariant == pytest.approx(naive, abs=1e-12)


def test_the_covariant_width_is_the_gp_width_scaled_by_x_max_over_the_origin_depth() -> None:
    z0, delta = -2.0, -0.5
    rng = np.random.default_rng(0)
    fits = {20.0: np.column_stack([rng.normal(2.0, 0.2, 300), rng.normal(1.8, 0.3, 300)])}

    block = stochastic_structure(fits, z0, delta, CRITICAL_ENERGY_MEV)["20"]

    x_max = x_max_of_energy(20000.0, CRITICAL_ENERGY_MEV, delta)
    gp = gp_formulae(float(np.log(20000.0 / CRITICAL_ENERGY_MEV)))["sampling"]
    assert block["gp_sigma_ln_t_covariant"] == pytest.approx(gp["sigma_ln_t"] * x_max / (x_max - z0))
    assert block["ratios"]["sigma_ln_t_over_gp_covariant"] == pytest.approx(
        block["ln_t"]["sd"] / block["gp_sigma_ln_t_covariant"]
    )


# ------------------------------------------------------------------ planted recovery


def test_a_planted_origin_and_beta_are_recovered() -> None:
    targets = planted_targets(PRIMARY, {"z0": -1.5, "beta": 0.58})

    fit = fit_model(PRIMARY, targets)

    assert fit.theta[0] == pytest.approx(-1.5, abs=0.1)
    assert fit.theta[1] == pytest.approx(0.58, abs=0.02)


def test_the_front_face_model_cannot_fit_a_planted_upstream_origin() -> None:
    targets = planted_targets(PRIMARY, {"z0": -2.0, "beta": 0.55})

    origin_fit = fit_model(PRIMARY, targets)
    front_face = fit_model(calibration.FRONT_FACE_FREE_BETA, targets)

    assert origin_fit.chi2 < 0.2 * front_face.chi2


def test_a_planted_amplitude_per_energy_is_absorbed_only_by_the_amplitude_model() -> None:
    targets = planted_targets(PRIMARY, {"z0": -1.0, "beta": 0.6})
    factor = np.array([[0.7], [0.9], [1.1], [1.3]])
    scaled = Targets(
        targets.energies_gev,
        targets.mean * factor,
        targets.se * factor,
        targets.events,
        targets.critical_energy_mev,
        targets.bounds,
    )

    plain = fit_model(PRIMARY, scaled)
    free = fit_model(calibration.AMPLITUDE_FREE, scaled)

    assert free.theta == pytest.approx([-1.0, 0.6], abs=0.1)
    assert free.chi2 < 0.1 * plain.chi2


# ------------------------------------------------------------------ identifiability


def test_the_joint_fit_constrains_the_origin_and_beta_more_tightly_than_one_energy_alone() -> None:
    targets = planted_targets(PRIMARY, {"z0": -1.5, "beta": 0.58})

    joint = hessian_summary(fit_model(PRIMARY, targets))
    single = hessian_summary(fit_model(PRIMARY, targets.subset([1])))

    assert joint["scaled_sd"]["z0"] < single["scaled_sd"]["z0"]
    assert joint["scaled_sd"]["beta"] < single["scaled_sd"]["beta"]
    assert joint["condition_number"] > 1.0  # the two parameters are never exactly independent
    assert -1.0 <= joint["correlation"] <= 1.0


def test_the_surface_summary_reports_a_valley_and_a_grid_edge_when_the_region_is_unbounded() -> None:
    # a long diagonal valley (z0 and beta trade off) that runs off the grid
    z0, beta = np.meshgrid(calibration.Z0_GRID, calibration.BETA_GRID, indexing="ij")
    surface = ((beta - 0.6 - 0.05 * (z0 + 2.0)) / 0.01) ** 2 + 0.01 * (z0 + 2.0) ** 2

    summary = surface_summary(surface, SimpleNamespace(scale=1.0))

    assert summary["region_extent"]["95%"]["touches_grid_edge"] is True
    valley = summary["profile_valley"]["best_beta_given_z0"]
    assert abs(valley[0] - valley[-1]) > 0.05
    low, high = summary["marginal_one_sigma_profile"]["z0"]
    assert high - low > 1.0


def test_a_compact_surface_does_not_touch_the_grid_edge() -> None:
    z0, beta = np.meshgrid(calibration.Z0_GRID, calibration.BETA_GRID, indexing="ij")
    surface = ((z0 + 1.0) / 0.1) ** 2 + ((beta - 0.6) / 0.01) ** 2

    summary = surface_summary(surface, SimpleNamespace(scale=1.0))

    assert summary["region_extent"]["95%"]["touches_grid_edge"] is False
    assert summary["grid_minimum"]["z0"] == pytest.approx(-1.0, abs=0.11)


# ------------------------------------------------------------------ held-out scoring


def test_held_out_scoring_is_better_for_the_true_model_than_for_a_wrong_one() -> None:
    targets = planted_targets(PRIMARY, {"z0": -1.5, "beta": 0.58})
    train, test = targets.halves()

    right = held_out_score(PRIMARY, train, test)
    wrong = held_out_score(calibration.FRONT_FACE_AMS, train, test)

    assert right["chi2_per_point"] < wrong["chi2_per_point"]
    assert right["rms_relative_layers_0_3"] < wrong["rms_relative_layers_0_3"]


# ------------------------------------------------------------------ the entrance gate and the floor


def test_the_gate_stays_closed_when_there_is_no_entrance_excess() -> None:
    targets = planted_targets(PRIMARY, {"z0": -1.5, "beta": 0.58})
    primary = fit_model(PRIMARY, targets)

    gate = entrance_gate(primary, {"amplitude_free": fit_model(calibration.AMPLITUDE_FREE, targets)}, targets, targets)

    assert set(gate["conditions"]) == GATE_CONDITIONS
    assert gate["open"] is False
    assert gate["conditions"]["layer0_residual_large_at_every_energy"] is False


def test_the_gate_opens_only_when_every_condition_holds() -> None:
    targets = with_floor(60.0, 0.0)
    primary = fit_model(PRIMARY, targets)
    alternatives = {"amplitude_free": fit_model(calibration.AMPLITUDE_FREE, targets)}

    gate = entrance_gate(primary, alternatives, targets, targets)

    assert set(gate["conditions"]) == GATE_CONDITIONS
    assert gate["open"] == all(gate["conditions"].values())
    # whatever the verdict, a layer-0 excess that the model lacks must show up as a positive deficit
    assert np.all(np.array(gate["layer0_relative_residual"]) < 0.0)


def test_a_planted_constant_floor_is_found_by_the_floor_study_and_accepted() -> None:
    targets = with_floor(60.0, 0.0)

    floor = floor_study(fit_model(PRIMARY, targets), targets)

    assert floor["constant"]["parameters"]["floor_a"] == pytest.approx(60.0, abs=15.0)
    assert floor["verdicts"]["constant"]["accepted"] is True
    assert floor["verdicts"]["constant"]["held_out_improvement_leave_one_energy_out"] > 0.2


def test_the_floor_study_rejects_a_floor_that_buys_nothing() -> None:
    targets = planted_targets(PRIMARY, {"z0": -1.5, "beta": 0.58})

    floor = floor_study(fit_model(PRIMARY, targets), targets)

    assert floor["verdicts"]["constant"]["accepted"] is False


def test_the_acceptance_needs_every_held_out_criterion(monkeypatch) -> None:
    targets = with_floor(60.0, 0.0)
    primary = fit_model(PRIMARY, targets)
    monkeypatch.setitem(calibration.GATE, "cv_improvement_to_accept", 5.0)  # an impossible requirement

    floor = floor_study(primary, targets)

    assert floor["verdicts"]["constant"]["accepted"] is False


def test_the_stochastic_mean_fits_the_planted_stochastic_data_better_than_the_deterministic_mean() -> None:
    targets = planted_targets(PRIMARY, {"z0": -1.5, "beta": 0.58})

    stochastic = fit_model(PRIMARY, targets)
    deterministic = fit_model(DETERMINISTIC, targets)

    assert stochastic.chi2 < deterministic.chi2


def test_the_estimator_control_does_not_invent_skewness_for_noise_free_symmetric_events() -> None:
    rng = np.random.default_rng(5)
    z0 = -1.0
    ln_t = rng.normal(2.1, 0.12, 300)
    ln_alpha = 1.75 + 1.2 * (ln_t - 2.1) + rng.normal(0.0, 0.1, 300)
    fractions = calibration.profile_fractions_batch(ln_t, ln_alpha, BOUNDS, z0)
    loaded = {20.0: SimpleNamespace(fractions={"deposition": fractions})}
    fits = {20.0: np.column_stack([ln_t, ln_alpha])}

    control = calibration.estimator_skewness_control(loaded, fits, BOUNDS, z0, n=400)["20"]

    assert abs(control["planted_skewness_ln_t"]) < 0.4
    assert abs(control["refitted_skewness_ln_t"] - control["planted_skewness_ln_t"]) < 0.3
    assert control["refitted_sd_ln_t"] == pytest.approx(control["planted_sd_ln_t"], rel=0.15)


def test_a_fixed_ams_beta_with_a_free_origin_is_a_one_parameter_fit() -> None:
    targets = planted_targets(calibration.AMS_BETA_FREE_ORIGIN, {"z0": -1.4})

    fit = fit_model(calibration.AMS_BETA_FREE_ORIGIN, targets)

    assert len(fit.theta) == 1
    assert fit.theta[0] == pytest.approx(-1.4, abs=0.1)
    assert fit.parameters()["beta"] == pytest.approx(0.65)


def test_the_model_family_fits_hold_beta_where_the_form_says_so() -> None:
    targets = planted_targets(PRIMARY, {"z0": -1.5, "beta": 0.58})

    fits = calibration.model_family_fits(
        targets, families=("beta_fixed_common_depth", "beta_free_common_depth")
    )

    assert fits["beta_fixed_common_depth"]["beta"] == pytest.approx(0.65)
    assert fits["beta_free_common_depth"]["beta"] == pytest.approx(0.58, abs=0.02)
    assert fits["beta_free_common_depth"]["chi2"] < fits["beta_fixed_common_depth"]["chi2"]
    assert set(calibration.model_family_specs(4, "none")) == {
        "beta_fixed_common_depth",
        "beta_fixed_free_depth",
        "beta_fixed_free_depth_deterministic",
        "beta_free_common_depth",
        "beta_free_free_depth",
        "beta_free_free_depth_deterministic",
    }


# ------------------------------------------------------------------ end to end on the exposed electrons


def test_the_calibration_runs_end_to_end_and_writes_tables_and_plots(tmp_path) -> None:
    results, artefacts = analyse((10.0, 100.0), n_boot=2, n_boot_secondary=2, n_ensemble=150, skewness=False)

    assert -6.0 <= results["A_primary"]["z0"] <= 1.0
    assert 0.3 <= results["A_primary"]["beta"] <= 1.1
    assert set(results["E_gate"]["conditions"]) == GATE_CONDITIONS
    assert results["A_fits"]["origin_beta"]["chi2"] < results["A_fits"]["front_face_ams"]["chi2"]
    assert set(results["D_stochastic_structure"]["least_squares"]) == {"10", "100"}
    assert set(artefacts["dec001_ensembles"]) == {10.0, 100.0}

    calibration.write_tables(results, tmp_path, artefacts)
    from ams_ecal.electron_studies.em_depth_origin_calibration_plots import make_plots

    make_plots(results, artefacts, tmp_path / "plots")

    assert {p.name for p in tmp_path.glob("*.csv")} >= {
        "depth_origin_calibration_fits.csv",
        "depth_origin_calibration_surface.csv",
        "depth_origin_calibration_structure.csv",
        "depth_origin_calibration_residuals.csv",
    }
    assert (tmp_path / "plots" / "surface_z0_beta.png").exists()
    assert (tmp_path / "plots" / "deposition_vs_readout.png").exists()
