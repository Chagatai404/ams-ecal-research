"""Profile shape study: the literature numbers, the tail family, the skewness controls and the AMS table."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

import ams_ecal.electron_studies.em_depth_origin_calibration as calibration
from ams_ecal.electron_studies import em_profile_shape_study as study
from ams_ecal.electron_studies.em_longitudinal_fluctuations import layer_bounds_x0
from ams_ecal.electron_studies.em_longitudinal_structure_analysis import (
    profile_fractions_batch,
)

BOUNDS = layer_bounds_x0()
ENERGIES_GEV = np.array([10.0, 20.0, 50.0, 100.0])
CRITICAL_ENERGY_MEV = 7.6


def planted_targets(spec: calibration.ModelSpec, params: dict[str, float], *, n_events: int = 300, seed: int = 4):
    rng = np.random.default_rng(seed)
    blank = np.zeros((len(ENERGIES_GEV), 18))
    skeleton = calibration.Targets(ENERGIES_GEV, blank, np.ones_like(blank), [], CRITICAL_ENERGY_MEV, BOUNDS)
    model, _ = calibration.predict(spec, {**calibration.DEFAULTS, **spec.fixed, **params}, skeleton)
    events = [np.clip(row * (1.0 + 0.04 * rng.standard_normal((n_events, 18))), 0.0, None) for row in model]
    mean = np.array([e.mean(axis=0) for e in events])
    se = np.array([e.std(axis=0, ddof=1) / np.sqrt(n_events) for e in events])
    return calibration.Targets(ENERGIES_GEV, mean, se, events, CRITICAL_ENERGY_MEV, BOUNDS)


# ------------------------------------------------------------------ the sources, as numbers


def test_the_grindhammer_peters_homogeneous_values_for_lead_at_10_gev() -> None:
    values = study.gp_homogeneous(10.0, CRITICAL_ENERGY_MEV)

    y = 10000.0 / CRITICAL_ENERGY_MEV
    assert values["T"] == pytest.approx(np.log(y) - 0.858)
    assert values["alpha"] == pytest.approx(0.21 + (0.492 + 2.38 / 82) * np.log(y))
    assert values["beta"] == pytest.approx((values["alpha"] - 1.0) / values["T"])
    assert 0.45 < values["beta"] < 0.50


def test_beta_in_the_grindhammer_peters_set_hardly_moves_with_energy() -> None:
    betas = [study.gp_homogeneous(e, CRITICAL_ENERGY_MEV)["beta"] for e in (10.0, 20.0, 50.0, 100.0)]

    assert max(betas) - min(betas) < 0.03


def test_the_pdg_electron_values() -> None:
    values = study.pdg_electron(50.0, CRITICAL_ENERGY_MEV)

    assert values["beta"] == 0.5
    assert values["T"] == pytest.approx(np.log(50000.0 / CRITICAL_ENERGY_MEV) - 0.5)
    assert values["alpha"] == pytest.approx(1.0 + 0.5 * values["T"])


def test_the_literature_tail_rate_is_the_inverse_of_the_central_attenuation_length() -> None:
    low, high = study.TAIL_ATTENUATION_LEAD_X0

    assert low < 1.0 / study.TAIL_KAPPA_LITERATURE < high


# ------------------------------------------------------------------ the tail family


def test_a_slower_tail_puts_more_energy_behind_the_last_layer() -> None:
    slow = study.tail_fractions(BOUNDS, -1.0, 4.0, 0.15)
    fast = study.tail_fractions(BOUNDS, -1.0, 4.0, 0.6)

    assert (1.0 - slow.sum()) > (1.0 - fast.sum())
    assert slow.sum() < 1.0


def test_the_gamma_plus_tail_fit_recovers_a_planted_weight_and_core() -> None:
    ln_t, ln_alpha, weight = np.log(7.5), np.log(5.2), 0.12
    tail = study.tail_fractions(BOUNDS, -1.0, 4.3, 0.278)
    core = profile_fractions_batch(np.array([ln_t]), np.array([ln_alpha]), BOUNDS, -1.0)[0]
    fractions = (1.0 - weight) * core + weight * tail

    fit = study.fit_gamma_plus_tail(fractions, BOUNDS, -1.0, 4.3, 0.278)

    assert fit == pytest.approx([ln_t, ln_alpha, weight], abs=1e-3)


def test_a_gamma_only_fit_of_a_profile_with_a_tail_leaves_a_structured_residual() -> None:
    tail = study.tail_fractions(BOUNDS, -1.0, 4.3, 0.278)
    core = profile_fractions_batch(np.array([np.log(7.5)]), np.array([np.log(5.2)]), BOUNDS, -1.0)[0]
    fractions = 0.85 * core + 0.15 * tail

    gamma_fit, residual_norm = study.fit_free_with_origin(fractions, BOUNDS, -1.0)
    gamma = profile_fractions_batch(gamma_fit[0:1], gamma_fit[1:2], BOUNDS, -1.0)[0]

    assert residual_norm > 1e-3
    assert gamma.sum() > fractions.sum()  # the gamma-only fit over-contains, as in the Geant4 electrons


def test_the_extension_families_and_the_requirement_on_the_chosen_one() -> None:
    specs = study.extension_specs()

    assert study.CHOSEN_TAIL_FAMILY in specs
    assert specs["ams_beta_fixed_with_tail"].fixed["beta"] == study.BETA_AMS
    with pytest.raises(ValueError, match="must include"):
        study.analyse((10.0,), families=("primary",))


def test_the_tail_model_wins_on_held_out_scoring_only_when_a_tail_is_planted() -> None:
    truth = {"z0": -1.0, "beta": 0.6, "tail_w": 0.15, "tail_alpha": 4.3}
    with_tail = planted_targets(study.extension_specs()[study.CHOSEN_TAIL_FAMILY], truth)
    without = planted_targets(calibration.PRIMARY, {"z0": -1.0, "beta": 0.6})
    families = ("primary", study.CHOSEN_TAIL_FAMILY)

    planted = study.extension_study(with_tail, families)
    null = study.extension_study(without, families)

    assert planted[study.CHOSEN_TAIL_FAMILY]["held_out_improvement_over_primary"]["leave_one_energy_out"] > 0.5
    assert null[study.CHOSEN_TAIL_FAMILY]["held_out_improvement_over_primary"]["leave_one_energy_out"] < 0.2


# ------------------------------------------------------------------ the ln T marginal


def test_the_skew_normal_parameters_reproduce_the_first_three_moments() -> None:
    shape, location, scale = study.skew_normal_parameters(2.0, 0.12, 0.8)

    sample = stats.skewnorm.rvs(shape, loc=location, scale=scale, size=200_000, random_state=1)

    assert sample.mean() == pytest.approx(2.0, abs=0.002)
    assert sample.std() == pytest.approx(0.12, abs=0.002)
    assert stats.skew(sample) == pytest.approx(0.8, abs=0.05)


def test_the_joint_draws_have_the_requested_marginals_and_correlation() -> None:
    rng = np.random.default_rng(0)
    shape, location, scale = study.skew_normal_parameters(2.0, 0.12, 0.7)
    ln_t = stats.skewnorm.rvs(shape, loc=location, scale=scale, size=5000, random_state=2)
    measured = np.column_stack([ln_t, 1.7 + 1.3 * (ln_t - ln_t.mean()) + rng.normal(0.0, 0.1, 5000)])

    gaussian = study.draw_joint(measured, "gaussian", 20_000, np.random.default_rng(1))
    skewed = study.draw_joint(measured, "skew_normal", 20_000, np.random.default_rng(1))
    empirical = study.draw_joint(measured, "empirical", 20_000, np.random.default_rng(1))

    assert abs(stats.skew(gaussian[:, 0])) < 0.1
    assert stats.skew(skewed[:, 0]) == pytest.approx(stats.skew(measured[:, 0]), abs=0.15)
    assert np.corrcoef(skewed.T)[0, 1] == pytest.approx(np.corrcoef(measured.T)[0, 1], abs=0.05)
    assert set(map(tuple, empirical[:50].tolist())) <= set(map(tuple, measured.tolist()))


def test_the_centred_leakage_distance_ignores_a_pure_mean_offset() -> None:
    rng = np.random.default_rng(3)
    base = rng.dirichlet(np.ones(19) * 5.0, size=800)[:, :18]  # the 19th share plays the leakage: it varies
    shifted = base + 0.03 / 18  # every contained fraction moves by exactly 0.03

    assert study.centred_leakage_ks(shifted, base) < 0.01
    assert stats.ks_2samp(shifted.sum(axis=1), base.sum(axis=1), method="asymp").statistic > 0.5


def test_the_tail_artifact_control_does_not_invent_skewness_when_the_truth_is_symmetric() -> None:
    rng = np.random.default_rng(5)
    origin, tail_alpha, tail_kappa = -1.0, 4.3, 0.278
    ln_t = rng.normal(2.0, 0.11, 400)
    ln_alpha = 1.7 + 1.2 * (ln_t - 2.0) + rng.normal(0.0, 0.08, 400)
    core = profile_fractions_batch(ln_t, ln_alpha, BOUNDS, origin)
    tail = study.tail_fractions(BOUNDS, origin, tail_alpha, tail_kappa)
    fractions = 0.88 * core + 0.12 * tail[None, :]
    loaded = {20.0: SimpleNamespace(fractions={"deposition": fractions})}
    fits = {20.0: np.column_stack([ln_t, ln_alpha, np.full(400, 0.12)])}

    control = study.tail_artifact_control(loaded, fits, BOUNDS, origin, tail_alpha, tail_kappa, n=300)["20"]

    assert abs(control["planted_skewness_ln_t"]) < 0.4
    assert abs(control["gamma_only_refit_skewness_ln_t"] - control["planted_skewness_ln_t"]) < 0.3
    assert abs(control["tail_refit_skewness_ln_t"] - control["planted_skewness_ln_t"]) < 0.3


# ------------------------------------------------------------------ the AMS table


def test_the_ams_beta_table_recovers_a_planted_beta_and_the_origin_where_it_is_0_65(monkeypatch) -> None:
    monkeypatch.setattr(study, "ORIGINS", (0.0, -1.0, -2.0))
    true_origin, true_beta = -1.0, 0.65
    loaded = {}
    for energy, t_origin in ((10.0, 7.0), (100.0, 9.5)):
        alpha = 1.0 + true_beta * t_origin
        core = profile_fractions_batch(np.array([np.log(t_origin)]), np.array([np.log(alpha)]), BOUNDS, true_origin)
        deposition = np.repeat(core, 6, axis=0)
        loaded[energy] = SimpleNamespace(fractions={"deposition": deposition, "readout": 0.065 * deposition})

    table = study.ams_beta_versus_origin(loaded, BOUNDS)

    for representation in ("deposition", "readout"):
        medians = np.array(table[representation]["median_beta_by_energy_and_origin"])
        assert medians[:, 1] == pytest.approx(0.65, abs=0.01)  # the true origin recovers the planted beta
        assert medians[:, 0].max() < medians[:, 1].min()  # a front-face origin gives a smaller beta
        crossing = table[representation]["origin_where_median_beta_is_0_65_by_energy"]
        assert crossing == pytest.approx([-1.0, -1.0], abs=0.05)


# ------------------------------------------------------------------ the data, and the whole


def test_the_fine_profile_matches_the_layer_grid_and_has_36_voxels() -> None:
    fine = study.load_fine_profile(10.0)

    assert len(fine["mean_fine_profile_mev"]) == 36
    assert fine["layer_grid_matches_fine_sum_max_abs_mev"] < 1e-2
    assert all(0.04 < f < 0.09 for f in fine["scintillator_fraction_by_layer"])


def test_the_study_runs_end_to_end_and_writes_tables_and_plots(tmp_path) -> None:
    families = ("primary", study.CHOSEN_TAIL_FAMILY, "ams_beta_fixed_with_tail")

    results, artefacts = study.analyse(
        (10.0, 100.0), families=families, ensemble_events=120, control_events=40, origins=False
    )

    rows = results["A_extensions_deposition"]
    assert set(rows) == set(families)
    assert rows[study.CHOSEN_TAIL_FAMILY]["chi2"] < rows["primary"]["chi2"]
    assert set(results["C_skew_consequence_gamma_only"]) == {"profile_family", "10", "100"}
    assert results["A_depth_scale"]["prefix_depth_x0_geant4"] < 17.0
    assert "gamma_fits" in artefacts

    study.write_tables(results, tmp_path)
    from ams_ecal.electron_studies.em_profile_shape_study_plots import make_plots

    make_plots(results, artefacts, tmp_path / "plots")

    assert (tmp_path / "profile_shape_study_extensions.csv").exists()
    assert (tmp_path / "plots" / "extension_scores.png").exists()
    assert (tmp_path / "plots" / "skew_consequence.png").exists()
