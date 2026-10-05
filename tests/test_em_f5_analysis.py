"""F5 analysis: profile forms, diagnostics, controls and the counterfactual machinery."""

import numpy as np
import pytest

from ams_ecal.em_f5_analysis import (
    BETA_AMS,
    DISTANCES,
    VARIANTS,
    EnergyData,
    ams_mean_gp_fluctuation,
    analyse,
    beta_values,
    counterfactual_ensembles,
    dec001_ensemble,
    fit_fixed_beta,
    fit_fixed_beta_with_origin,
    fit_free_with_origin,
    fit_mean_profile,
    gp_formulae,
    joint_diagnostics,
    marginal_diagnostics,
    partial_spearman,
    profile_fixed_beta,
    profile_fractions_batch,
    profile_general,
    removed_fraction,
    write_tables,
)
from ams_ecal.em_f5_plots import make_plots
from ams_ecal.em_longitudinal_fluctuations import (
    DATA,
    layer_bounds_x0,
    profile_fractions,
)

BOUNDS = layer_bounds_x0()


# ---------------------------------------------------------------- profile forms


def test_the_batch_profile_equals_the_single_event_profile() -> None:
    ln_t, ln_alpha = np.array([1.8, 2.1]), np.array([1.5, 1.8])

    batch = profile_fractions_batch(ln_t, ln_alpha, BOUNDS)

    for i in range(2):
        assert batch[i] == pytest.approx(profile_fractions(ln_t[i], ln_alpha[i], BOUNDS))


def test_the_ams_form_is_the_general_gamma_profile_with_alpha_equal_one_plus_beta_t() -> None:
    ln_t = np.array([1.9])
    alpha = 1.0 + BETA_AMS * np.exp(ln_t)

    fixed = profile_fixed_beta(ln_t, BOUNDS)[0]
    general = profile_fractions_batch(ln_t, np.log(alpha), BOUNDS)[0]

    assert fixed == pytest.approx(general, rel=1e-9)
    assert profile_general(BOUNDS, 1.9, beta=BETA_AMS) == pytest.approx(fixed)


def test_origin_shift_amplitude_and_missing_alpha_are_handled() -> None:
    base = profile_general(BOUNDS, 1.9, 1.6)

    assert profile_general(BOUNDS, 1.9, 1.6, amplitude=0.5) == pytest.approx(0.5 * base)
    shifted = profile_general(BOUNDS, 1.9, 1.6, origin=-0.5)
    assert shifted[0] > base[0]  # a shower that starts before the front face puts more energy in layer 0
    with pytest.raises(ValueError, match="ln_alpha"):
        profile_general(BOUNDS, 1.9)


def test_beta_is_alpha_minus_one_over_t() -> None:
    assert beta_values(np.log(5.0), np.log(1.0 + 0.65 * 5.0)) == pytest.approx(0.65)


def test_the_fixed_beta_fit_is_exact_for_an_ams_profile_and_worse_for_another_beta() -> None:
    ams = profile_fixed_beta(np.log(5.5), BOUNDS)[0]
    other_alpha = np.array([np.log(1.0 + 0.5 * 5.5)])
    other = profile_fractions_batch(np.array([np.log(5.5)]), other_alpha, BOUNDS)[0]

    ln_t, residual_ams = fit_fixed_beta(ams, BOUNDS)
    _, residual_other = fit_fixed_beta(other, BOUNDS)

    assert ln_t == pytest.approx(np.log(5.5), abs=1e-4)
    assert residual_ams < 1e-6 < residual_other


# ---------------------------------------------------------------- diagnostics


def test_marginal_diagnostics_separate_a_normal_from_a_skewed_sample() -> None:
    rng = np.random.default_rng(1)
    normal = marginal_diagnostics(rng.standard_normal(4000))
    skewed = marginal_diagnostics(rng.lognormal(0.0, 0.8, 4000))

    assert abs(normal["skewness"]) < 0.15 and normal["max_abs_quantile_deviation"] < 0.2
    assert skewed["skewness"] > 1.0 and skewed["max_abs_quantile_deviation"] > 0.4
    assert normal["robust_sd"] == pytest.approx(normal["sd"], rel=0.1)


def test_joint_diagnostics_recover_a_planted_correlation_and_flag_heavy_tails() -> None:
    rng = np.random.default_rng(2)
    z1, z2 = rng.standard_normal(3000), rng.standard_normal(3000)
    a, b = z1, 0.6 * z1 + 0.8 * z2

    gaussian = joint_diagnostics(a, b)
    heavy = joint_diagnostics(rng.standard_t(2, 3000), rng.standard_t(2, 3000))

    assert gaussian["pearson"] == pytest.approx(0.6, abs=0.05)
    assert gaussian["spearman"] == pytest.approx(0.6, abs=0.05)
    assert gaussian["mahalanobis_ks_vs_chi2_2"] < 0.05 < heavy["mahalanobis_ks_vs_chi2_2"]


def test_the_grindhammer_peters_sets_differ_and_match_the_hand_values() -> None:
    values = gp_formulae(7.1796)

    assert values["sampling"]["sigma_ln_t"] == pytest.approx(1.0 / (-2.5 + 1.25 * 7.1796))
    assert values["homogeneous"]["sigma_ln_t"] == pytest.approx(1.0 / (-1.4 + 1.26 * 7.1796))
    assert values["sampling"]["sigma_ln_t"] > values["homogeneous"]["sigma_ln_t"]
    assert values["sampling"]["rho"] == pytest.approx(0.784 - 0.023 * 7.1796)


def test_a_partial_correlation_removes_a_planted_confounder_and_keeps_a_direct_dependence() -> None:
    rng = np.random.default_rng(3)
    confounder = rng.standard_normal(3000)
    a = confounder + 0.3 * rng.standard_normal(3000)
    only_through_confounder = confounder + 0.3 * rng.standard_normal(3000)
    direct = a + 0.3 * rng.standard_normal(3000)

    assert partial_spearman(a, only_through_confounder, None) > 0.8
    assert abs(partial_spearman(a, only_through_confounder, confounder[:, None])) < 0.1
    assert partial_spearman(a, direct, confounder[:, None]) > 0.6


def test_the_removed_fraction_is_one_for_a_perfect_variant_and_negative_for_a_worse_one() -> None:
    base = {"a": 0.4, "b": 0.0, "c": 0.2}
    variant = {"a": 0.1, "b": 0.3, "c": 0.4}

    result = removed_fraction(base, variant)

    assert result["a"] == pytest.approx(0.75)
    assert result["b"] is None  # nothing to remove
    assert result["c"] == pytest.approx(-1.0)
    assert len(DISTANCES) == 10


def test_a_discrepancy_already_at_its_noise_floor_is_not_given_a_ratio() -> None:
    base = {"ks_cog": 0.03, "ks_rms": 0.9}
    variant = {"ks_cog": 0.09, "ks_rms": 0.1}

    result = removed_fraction(base, variant)

    assert result["ks_cog"] is None  # 0.03 is below the 0.05 floor: worse-by-noise must not read as a failure
    assert result["ks_rms"] == pytest.approx(1.0 - 0.1 / 0.9)


# ---------------------------------------------------------------- mean-profile structure


def test_an_origin_shift_is_found_when_one_is_planted_and_a_fixed_beta_without_it_fits_worse() -> None:
    truth = profile_general(BOUNDS, 1.9, beta=BETA_AMS, origin=-0.6)

    with_origin = fit_mean_profile(truth, BOUNDS, "fixed_beta_origin")
    without = fit_mean_profile(truth, BOUNDS, "fixed_beta")

    assert with_origin["parameters"][1] == pytest.approx(-0.6, abs=0.05)
    assert with_origin["rms_residual"] < 1e-5 < without["rms_residual"]
    assert without["layer0_geant4_over_fit"] > 1.5  # the unshifted form under-predicts layer 0


# ---------------------------------------------------------------- counterfactual machinery


def test_restoring_alpha_widens_the_event_to_event_profile_spread() -> None:
    joint = ams_mean_gp_fluctuation(50.0, "deposition", BOUNDS, 1500, 5)
    fixed_beta = dec001_ensemble(50.0, "deposition", 1500, 5)

    assert joint.shape == fixed_beta.shape == (1500, 18)
    assert np.all(joint >= 0.0)
    assert joint.std(axis=0)[6:12].mean() > fixed_beta.std(axis=0)[6:12].mean()
    # the backbone is kept: the mean contained energy stays close
    assert joint.sum(axis=1).mean() == pytest.approx(fixed_beta.sum(axis=1).mean(), abs=0.03)


def test_every_variant_is_generated_with_the_right_shape() -> None:
    rng = np.random.default_rng(6)
    fits = np.column_stack([rng.normal(2.0, 0.12, 300), rng.normal(1.7, 0.16, 300)])
    data = EnergyData(
        50.0,
        {},
        np.zeros((1, 1, 1)),
        np.zeros(1),
        np.zeros(1),
        {
            "least_squares": fits,
            "origin_mean_profile_fixed_beta": fits,
            "origin_mean_profile_fixed_beta_fixed_beta_ln_t": fits[:, 0],
        },
        {"mean_profile_fixed_beta": -2.0},
    )

    ensembles = counterfactual_ensembles(data, BOUNDS, 40, seed=7)

    assert set(ensembles) == set(VARIANTS)
    assert all(e.shape == (40, 18) for e in ensembles.values())


# ---------------------------------------------------------------- the whole analysis on one energy


@pytest.mark.skipif(
    not (DATA / "E10GeV").is_dir(), reason="the exploration electron sample is not present"
)
def test_the_analysis_runs_end_to_end_and_writes_tables_and_plots(tmp_path) -> None:
    results, loaded, ensembles = analyse((10.0,), n_counterfactual=60, n_boot=5)

    block = results["10"]
    joint = block["A_B_joint_structure"]["estimators"]["least_squares"]
    assert 0.3 < joint["beta"]["median"] < 0.8
    assert set(block["C_counterfactual"]["metrics"]) == set(VARIANTS)
    assert set(block["A_B_joint_structure"]["origin_refit"]) == {
        "origins_x0",
        "mean_profile_fixed_beta",
        "mean_profile_free_gamma",
    }
    assert block["E_confounders"]["min_cells_from_border"] > 0
    assert "fluctuation_effect_only" in block["C_counterfactual"]
    assert len(block["D_mean_backbone"]["energy_mev_layers_0_3_geant4_mean_sd"]) == 4
    assert set(block["D_mean_backbone"]["structural_fits"]) == {
        "gamma_free",
        "gamma_free_origin",
        "gamma_free_amplitude",
        "fixed_beta",
        "fixed_beta_origin",
    }

    write_tables(results, tmp_path)
    make_plots(results, loaded, ensembles, tmp_path / "plots")

    assert {p.name for p in tmp_path.glob("*.csv")} == {
        "f5_joint_structure.csv",
        "f5_counterfactual_distances.csv",
        "f5_confounders.csv",
    }
    assert len(list((tmp_path / "plots").glob("*.png"))) == 9


def test_a_shifted_origin_is_recovered_by_the_fits_that_are_told_it() -> None:
    origin = -2.2
    truth = profile_fractions_batch(np.array([2.1]), np.array([1.7]), BOUNDS, origin)[0]

    free, residual = fit_free_with_origin(truth, BOUNDS, origin)
    fixed_ln_t, _ = fit_fixed_beta_with_origin(
        profile_fixed_beta(np.array([2.3]), BOUNDS, BETA_AMS, origin)[0], BOUNDS, origin
    )

    assert free == pytest.approx([2.1, 1.7], abs=1e-4) and residual < 1e-7
    assert fixed_ln_t == pytest.approx(2.3, abs=1e-4)
    assert profile_fractions_batch(np.array([2.1]), np.array([1.7]), BOUNDS, origin)[0][0] > profile_fractions_batch(
        np.array([2.1]), np.array([1.7]), BOUNDS, 0.0
    )[0][0]  # an upstream origin puts more energy in layer 0
