"""em_production Slice 1: the gamma-family generator, its draws, its refusals and its parameter artifact."""

import json
from dataclasses import replace

import numpy as np
import pytest
from scipy import stats
from scipy.special import gammainc

from ams_ecal.detector.tracking import TrackState
from ams_ecal.electron_model import em_production as ep

BOUNDS = tuple((k * 17.0 / 18.0, (k + 1) * 17.0 / 18.0) for k in range(18))
CRITICAL_ENERGY_MEV = 7.6
PARAMETERS = ep.EMProductionParameters(
    origin_x0=0.0,
    beta=0.53,
    tail_weight=0.05,
    tail_weight_log_energy_slope=-0.02,
    tail_shape=2.0,
    rho_intercept=0.91,
    rho_log_slope=-0.04,
    ln_t_skewness=0.74,
)


def longitudinal(parameters: ep.EMProductionParameters = PARAMETERS) -> ep.EMProductionLongitudinalModel:
    return ep.EMProductionLongitudinalModel(parameters, CRITICAL_ENERGY_MEV, BOUNDS)


# ------------------------------------------------------------------ parameters and the artifact


def test_the_parameters_refuse_impossible_values() -> None:
    for change in (
        {"beta": 0.0},
        {"tail_shape": 1.0},
        {"tail_weight": 0.96},
        {"ln_t_skewness": 1.0},
        {"tail_rate_per_x0": -0.1},
        {"origin_x0": float("nan")},
        {"calibrated_energy_range_gev": (100.0, 10.0)},
    ):
        with pytest.raises(ValueError):
            replace(PARAMETERS, **change)


def test_the_parameters_round_trip_through_a_json_dictionary() -> None:
    again = ep.EMProductionParameters.from_dict(json.loads(json.dumps(PARAMETERS.as_dict())))

    assert again == PARAMETERS
    assert ep.parameters_digest(again) == ep.parameters_digest(PARAMETERS)


def test_an_artifact_round_trips_and_a_tampered_one_is_refused(tmp_path) -> None:
    path = ep.write_parameters_artifact(PARAMETERS, tmp_path / "parameters.json", {"note": "test"}, "tests")

    loaded, document = ep.read_parameters_artifact(path)
    assert loaded == PARAMETERS
    assert document["model"] == ep.MODEL_NAME

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["parameters"]["beta"] = 0.6
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="altered"):
        ep.read_parameters_artifact(path)


def test_an_artifact_of_another_kind_or_schema_is_refused(tmp_path) -> None:
    path = ep.write_parameters_artifact(PARAMETERS, tmp_path / "p.json", {}, "tests")
    document = json.loads(path.read_text(encoding="utf-8"))
    for key, value, message in (
        ("kind", "other", "not an em_production"),
        ("schema_version", 99, "schema"),
        ("model", "x", "model"),
    ):
        path.write_text(json.dumps({**document, key: value}), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            ep.read_parameters_artifact(path)


def test_the_committed_artifact_loads_and_matches_its_recorded_digest() -> None:
    parameters, document = ep.read_parameters_artifact(ep.DEFAULT_ARTIFACT)

    assert ep.parameters_digest(parameters) == document["parameters_sha256"]
    assert parameters.origin_x0 == 0.0  # the front-face convention
    assert "not validated" in document["status"]
    assert document["calibration"]["data"]["sealed_data_read"] is False


# ------------------------------------------------------------------ the draws


def test_the_skew_normal_helper_reproduces_standard_deviation_and_skewness() -> None:
    shape, scale = ep.skew_normal_shape_and_scale(0.12, 0.74)
    sample = stats.skewnorm.rvs(shape, scale=scale, size=300_000, random_state=1)

    assert sample.std() == pytest.approx(0.12, rel=0.01)
    assert stats.skew(sample) == pytest.approx(0.74, abs=0.05)
    assert ep.skew_normal_shape_and_scale(0.12, 0.0) == (0.0, 0.12)


def test_the_helper_agrees_with_the_one_used_by_the_shape_study() -> None:
    from ams_ecal.electron_studies.em_profile_shape_study import skew_normal_parameters

    shape, scale = ep.skew_normal_shape_and_scale(0.11, 0.6)
    study_shape, _, study_scale = skew_normal_parameters(2.0, 0.11, 0.6)

    assert shape == pytest.approx(study_shape)
    assert scale == pytest.approx(study_scale)


def test_without_skewness_ln_t_is_the_gaussian_of_the_covariant_width() -> None:
    model = longitudinal(replace(PARAMETERS, ln_t_skewness=0.0))
    z_t, z_perp = np.random.default_rng(3).standard_normal((2, 100))

    ln_t, _ = model.draw_shape(50_000.0, z_t, z_perp)

    sigma_t, _, _ = model.widths(50_000.0)
    expected = np.log(model.median_depth_from_origin_x0(50_000.0)) + sigma_t * z_t
    assert ln_t == pytest.approx(expected, abs=1e-12)


def test_the_drawn_shape_has_the_requested_median_skewness_widths_and_correlation() -> None:
    model = longitudinal()
    z = np.random.default_rng(4).standard_normal((2, 400_000))
    energy = 50_000.0

    ln_t, ln_alpha = model.draw_shape(energy, z[0], z[1])

    sigma_t, sigma_a, rho = model.widths(energy)
    assert np.median(ln_t) == pytest.approx(np.log(model.median_depth_from_origin_x0(energy)), abs=2e-3)
    assert ln_t.std() == pytest.approx(sigma_t, rel=0.02)
    assert stats.skew(ln_t) == pytest.approx(PARAMETERS.ln_t_skewness, abs=0.08)
    assert ln_alpha.std() == pytest.approx(sigma_a, rel=0.02)
    assert np.corrcoef(ln_t, ln_alpha)[0, 1] == pytest.approx(rho, abs=0.02)


def test_the_widths_follow_the_grindhammer_peters_sampling_set_with_the_covariant_factor() -> None:
    model = longitudinal()
    energy = 20_000.0
    ln_y = np.log(energy / CRITICAL_ENERGY_MEV)
    x_max = ln_y - 0.5

    sigma_t, sigma_a, rho = model.widths(energy)

    assert sigma_t == pytest.approx(1.0 / (-2.5 + 1.25 * ln_y) * x_max / (x_max - PARAMETERS.origin_x0))
    assert sigma_a == pytest.approx(1.0 / (-0.82 + 0.79 * ln_y))
    assert rho == pytest.approx(0.91 - 0.04 * ln_y)


# ------------------------------------------------------------------ the profile


def test_a_profile_without_a_tail_is_the_analytic_gamma_integral() -> None:
    model = longitudinal(replace(PARAMETERS, tail_weight=0.0, tail_weight_log_energy_slope=0.0))
    ln_t, ln_alpha = np.array([np.log(7.5)]), np.array([np.log(5.0)])

    fractions = model.layer_fractions(30_000.0, ln_t, ln_alpha)[0]

    alpha, rate = 5.0, 4.0 / 7.5
    bounds = np.array(BOUNDS)
    expected = gammainc(alpha, rate * bounds[:, 1]) - gammainc(alpha, rate * bounds[:, 0])
    assert fractions == pytest.approx(expected)


def test_the_tail_moves_energy_behind_the_prefix_and_the_weight_is_clipped() -> None:
    plain = longitudinal(replace(PARAMETERS, tail_weight=0.0, tail_weight_log_energy_slope=0.0))
    tailed = longitudinal(replace(PARAMETERS, tail_weight=0.3, tail_weight_log_energy_slope=0.0))
    args = (np.array([np.log(7.5)]), np.array([np.log(5.0)]))

    assert tailed.layer_fractions(30_000.0, *args).sum() != pytest.approx(plain.layer_fractions(30_000.0, *args).sum())
    steep = longitudinal(replace(PARAMETERS, tail_weight=0.5, tail_weight_log_energy_slope=0.1))
    assert steep.tail_weight(100_000.0) <= ep.MAX_TAIL_WEIGHT
    negative = longitudinal(replace(PARAMETERS, tail_weight=0.0, tail_weight_log_energy_slope=-0.1))
    assert negative.tail_weight(100_000.0) == 0.0


def test_the_fractions_are_nonnegative_never_renormalised_and_leave_leakage() -> None:
    fractions = longitudinal().sample_layer_energy_fractions_batch(50_000.0, np.random.default_rng(5), 2000)

    assert fractions.shape == (2000, 18)
    assert fractions.min() >= 0.0
    contained = fractions.sum(axis=1)
    assert contained.max() < 1.0
    assert 0.8 < contained.mean() < 0.97


def test_the_quadrature_mean_matches_the_monte_carlo_mean() -> None:
    model = longitudinal()
    z = np.random.default_rng(6).standard_normal((60_000, 2))

    quadrature = model.mean_layer_fractions(50_000.0, z)
    ln_t, ln_alpha = model.draw_shape(50_000.0, z[:, 0], z[:, 1])
    direct = model.layer_fractions(50_000.0, ln_t, ln_alpha).mean(axis=0)

    assert quadrature == pytest.approx(direct)


def test_a_batch_equals_the_same_draws_taken_one_event_at_a_time() -> None:
    model = longitudinal()
    batch = model.sample_layer_energy_fractions_batch(20_000.0, np.random.default_rng(7), 4)
    rng = np.random.default_rng(7)
    single = np.array([model.sample_layer_energy_fractions(20_000.0, rng) for _ in range(4)])

    assert batch == pytest.approx(single)


# ------------------------------------------------------------------ refusals


def test_the_model_refuses_energies_outside_the_calibrated_range_unless_told_otherwise() -> None:
    model = longitudinal()
    rng = np.random.default_rng(8)

    for energy in (2_000.0, 500_000.0):
        with pytest.raises(ValueError, match="calibrated range"):
            model.sample_layer_energy_fractions(energy, rng)
    assert len(model.sample_layer_energy_fractions(500_000.0, rng, allow_extrapolation=True)) == 18
    with pytest.raises(ValueError):
        model.sample_layer_energy_fractions(-1.0, rng)


def test_the_width_law_domain_and_the_origin_are_checked() -> None:
    with pytest.raises(ValueError, match="width laws"):
        longitudinal().widths(40.0)
    deep = longitudinal(replace(PARAMETERS, origin_x0=1.0))
    with pytest.raises(ValueError, match="in front of the calibrated origin"):
        deep.median_depth_from_origin_x0(10.0)


# ------------------------------------------------------------------ the event generator


def test_an_event_is_reproducible_from_its_seed_and_distributes_the_layer_energies() -> None:
    model = ep.build_model(PARAMETERS)
    track = TrackState(x0_mm=4.5, y0_mm=4.5, z0_mm=0.0, theta_rad=0.0, phi_rad=0.0)
    args = {
        "event_id": "e",
        "primary_energy_mev": 50_000.0,
        "track": track,
        "random_seed": 11,
        "configuration_sha256": "0" * 64,
    }

    first, second = model.generate_event(**args), model.generate_event(**args)
    other = model.generate_event(**{**args, "random_seed": 12})

    assert first.cell_energies_mev == second.cell_energies_mev
    assert first.cell_energies_mev != other.cell_energies_mev
    total = sum(sum(row) for row in first.cell_energies_mev)
    assert 0.5 * 50_000.0 < total < 50_000.0
    assert first.provenance.simulation_backend == "fastmc"
    assert first.provenance.random_seed == 11


def test_the_event_stream_zips_energies_and_tracks_strictly_and_seeds_are_independent() -> None:
    model = ep.build_model(PARAMETERS)
    track = TrackState(x0_mm=4.5, y0_mm=4.5, z0_mm=0.0, theta_rad=0.0, phi_rad=0.0)
    common = {"base_seed": 1, "configuration_sha256": "0" * 64}

    with pytest.raises(ValueError):
        list(model.generate_events(primary_energies_mev=[20_000.0], tracks=[track, track], **common))
    events = list(model.generate_events(primary_energies_mev=[20_000.0] * 3, tracks=[track] * 3, **common))
    assert len({e.provenance.random_seed for e in events}) == 3
    assert ep.spawn_event_seeds(1, 3) == ep.spawn_event_seeds(1, 3)
    with pytest.raises(ValueError):
        ep.spawn_event_seeds(-1, 3)


def test_the_committed_model_loads_and_generates_a_plausible_event() -> None:
    model = ep.load_em_production()
    track = TrackState(x0_mm=4.5, y0_mm=4.5, z0_mm=0.0, theta_rad=0.0, phi_rad=0.0)

    event = model.generate_event(
        event_id="e", primary_energy_mev=50_000.0, track=track, random_seed=1, configuration_sha256="0" * 64
    )

    assert len(event.cell_energies_mev) == 18
    assert model.parameters.calibrated_energy_range_gev == (10.0, 100.0)
