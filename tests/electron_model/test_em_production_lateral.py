"""em_production Slice 2 (lateral spot model): laws, spot draws, conservation, reproducibility, refusals, artifact."""

import json
from dataclasses import replace

import numpy as np
import pytest

from ams_ecal.detector.tracking import TrackState
from ams_ecal.electron_model import em_production_lateral as lat
from ams_ecal.electron_model.em_production import (
    DEFAULT_ARTIFACT as LONGITUDINAL_ARTIFACT,
)

PARAMETERS = lat.LateralParameters(
    core_intercept=0.002,
    core_log_energy_slope=-0.0051,
    core_depth_slope=0.1761,
    tail_scale=0.6845,
    tail_centre=0.1281,
    tail_decay_rate=-14.2926,
    tail_growth_intercept=0.4413,
    tail_growth_log_energy_slope=-0.0633,
    weight_amplitude=2.2903,
    weight_centre=0.7046,
    weight_width_intercept=1.5458,
    weight_width_log_energy_slope=-0.1884,
    spot_amplitude=100.0,
    spot_energy_exponent=0.85,
    spot_depth_scale=0.9,
    spot_shape_scale=0.9,
)


@pytest.fixture(scope="module")
def model() -> lat.EMProductionLateralModel:
    return lat.build_model(PARAMETERS)


def axes(model, n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    entry = rng.uniform(0.0, 9.0, size=(n, 2))
    return model.axis_for_entry(entry[:, 0], entry[:, 1])


def test_strip_probabilities_are_a_distribution_with_a_leakage_category(model) -> None:
    tau = np.linspace(0.1, 2.0, 18)[None, :].repeat(5, axis=0)

    probabilities = model.strip_probabilities(50_000.0, axes(model, 5), tau)

    assert probabilities.shape == (5, 18, 73)
    assert (probabilities >= 0).all()
    assert probabilities.sum(axis=-1) == pytest.approx(1.0)
    assert probabilities[..., -1].max() < 0.01  # a 648 mm layer leaks almost nothing sideways


def test_the_depth_coordinate_is_depth_over_the_maximum_for_the_median_event(
    model,
) -> None:
    energy = 50_000.0
    longitudinal = model.longitudinal
    t_ref = longitudinal.median_depth_from_origin_x0(energy)
    ln_t = np.array([np.log(t_ref)])
    ln_alpha = np.array([np.log(1.0 + longitudinal.parameters.beta * t_ref)])

    tau = model.depth_coordinate(energy, ln_t, ln_alpha)

    assert tau[0] == pytest.approx(model.layer_depth_x0() / t_ref)


def test_a_later_shower_has_a_smaller_depth_coordinate_in_the_same_layer(model) -> None:
    energy = 50_000.0
    t_ref = model.longitudinal.median_depth_from_origin_x0(energy)
    alpha = np.log(1.0 + model.longitudinal.parameters.beta * t_ref)

    early = model.depth_coordinate(energy, np.array([np.log(0.8 * t_ref)]), np.array([alpha]))
    late = model.depth_coordinate(energy, np.array([np.log(1.2 * t_ref)]), np.array([alpha]))

    assert (late < early).all()


def test_the_radial_parameters_follow_the_grindhammer_peters_shapes(model) -> None:
    tau = np.linspace(0.05, 2.5, 60)

    core, tail, weight = model.radial_parameters(50_000.0, tau)

    assert (np.diff(core) >= 0).all()  # R_C grows with depth
    assert tail.argmin() not in (0, len(tau) - 1)  # R_T has a minimum at moderate depth
    assert weight.argmax() not in (0, len(tau) - 1)  # p peaks at moderate depth
    assert ((weight >= 0) & (weight <= 1)).all()


def test_the_radial_law_refuses_a_non_positive_weight_width(model) -> None:
    broken = replace(
        model,
        parameters=replace(PARAMETERS, weight_width_intercept=-0.1, weight_width_log_energy_slope=0.0),
    )

    with pytest.raises(ValueError, match="p3"):
        broken.radial_parameters(50_000.0, np.array([0.5]))


def test_the_spot_counts_rise_with_energy_peak_in_the_middle_and_never_vanish(
    model,
) -> None:
    low, high = model.spot_counts(10_000.0), model.spot_counts(100_000.0)

    assert (low >= 1).all()
    assert high.sum() > 3 * low.sum()
    assert 2 < int(high.argmax()) < 16


def test_the_sequential_binomial_is_a_multinomial() -> None:
    rng = np.random.default_rng(5)
    probabilities = np.array([0.5, 0.3, 0.15, 0.05])
    totals = np.full((20_000,), 40)

    counts = lat.conditional_binomial_counts(rng, totals, np.tile(probabilities, (20_000, 1)))

    assert (counts.sum(axis=1) == 40).all()
    assert counts.mean(axis=0) == pytest.approx(40 * probabilities, rel=0.03)
    assert counts.var(axis=0) == pytest.approx(40 * probabilities * (1 - probabilities), rel=0.08)
    expected = -np.sqrt(probabilities[0] * probabilities[1] / ((1 - probabilities[0]) * (1 - probabilities[1])))
    assert np.corrcoef(counts[:, 0], counts[:, 1])[0, 1] == pytest.approx(expected, abs=0.03)


def test_cell_energy_conserves_the_layer_energy_apart_from_the_leakage(model) -> None:
    rng = np.random.default_rng(2)
    batch = model.sample_batch(50_000.0, axes(model, 400), rng)

    cells = batch.cell_energy_mev.sum(axis=2)
    layer = 50_000.0 * batch.layer_fractions

    assert (cells <= layer * (1 + 1e-9)).all()
    assert cells.sum() / layer.sum() > 0.995  # only the spots beyond the 72 cells are lost


def test_one_event_is_a_batch_of_one_and_is_reproducible(model) -> None:
    track = TrackState(x0_mm=3.3, y0_mm=5.1, z0_mm=0.0, theta_rad=0.0, phi_rad=0.0)
    axis = model.axis_for_track(track)[None, :]

    one = model.sample_cell_energies_mev(50_000.0, track, np.random.default_rng(11))
    batch = model.sample_batch(50_000.0, axis, np.random.default_rng(11))
    again = model.sample_cell_energies_mev(50_000.0, track, np.random.default_rng(11))

    assert np.array(one) == pytest.approx(batch.cell_energy_mev[0])
    assert np.array(one) == pytest.approx(np.array(again))


def test_a_generated_event_is_an_ecal_event_with_provenance(model) -> None:
    track = TrackState(x0_mm=4.0, y0_mm=4.0, z0_mm=0.0, theta_rad=0.0, phi_rad=0.0)

    event = model.generate_event(
        event_id="e1",
        primary_energy_mev=20_000.0,
        track=track,
        random_seed=7,
        configuration_sha256="0" * 64,
    )

    assert event.provenance.simulation_backend == "fastmc"
    assert event.provenance.simulation_version.startswith("em-production-slice2")
    assert sum(sum(layer) for layer in event.cell_energies_mev) < 20_000.0


def test_the_batch_mean_matches_the_quadrature_mean_profile(model) -> None:
    rng = np.random.default_rng(9)
    axis = axes(model, 6000, seed=4)
    batch = model.sample_batch(50_000.0, axis, rng)
    simulated = batch.cell_energy_mev.sum(axis=0) / batch.cell_energy_mev.sum(axis=(0, 2))[:, None]
    z = np.random.default_rng(10).standard_normal((6000, 2))
    quadrature = model.mean_cell_fractions(50_000.0, z, axis)
    quadrature = quadrature / quadrature.sum(axis=1, keepdims=True)

    distance = 0.5 * np.abs(simulated - quadrature).sum(axis=1)
    assert distance[3:].max() < 0.02
    assert distance[:3].max() < 0.12  # a layer with a handful of spots is noisy: one spot is one cell


def test_the_model_refuses_energies_outside_the_calibrated_range(model) -> None:
    rng = np.random.default_rng(1)

    with pytest.raises(ValueError, match="calibrated range"):
        model.sample_batch(200_000.0, axes(model, 3), rng)


def test_a_moliere_radius_that_differs_from_the_geometry_is_refused(model) -> None:
    with pytest.raises(ValueError, match="differs from the geometry"):
        replace(model, parameters=replace(PARAMETERS, moliere_radius_mm=27.6))


def test_the_artifact_round_trips_and_a_changed_parameter_is_refused(tmp_path) -> None:
    longitudinal = json.loads(LONGITUDINAL_ARTIFACT.read_text(encoding="utf-8"))["parameters_sha256"]
    path = lat.write_parameters_artifact(PARAMETERS, tmp_path / "p.json", {"note": "test"}, "tests", longitudinal)

    parameters, document = lat.read_parameters_artifact(path)
    loaded = lat.load_em_production_lateral(path)

    assert parameters == PARAMETERS
    assert document["longitudinal_parameters_sha256"] == longitudinal
    assert loaded.longitudinal_sha256 == longitudinal
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["parameters"]["tail_scale"] = 0.7
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        lat.read_parameters_artifact(path)


def test_a_lateral_artifact_calibrated_on_another_longitudinal_artifact_is_refused() -> None:
    with pytest.raises(
        ValueError,
        match="differs from the one the lateral parameters were calibrated on",
    ):
        lat.build_model(PARAMETERS, expected_longitudinal_sha256="0" * 64)


# ---- extension E1: opt-in, off by default


def test_the_extension_is_off_by_default_and_the_committed_artifact_still_loads() -> None:
    assert PARAMETERS.spot_tail_refinement == 1.0
    assert PARAMETERS.front_spot_limit == 0.0

    parameters, _ = lat.read_parameters_artifact()

    assert parameters.spot_tail_refinement == 1.0
    assert parameters.front_spot_limit == 0.0


def test_the_extension_refuses_a_coarser_tail_or_a_negative_cap() -> None:
    with pytest.raises(ValueError, match="spot_tail_refinement"):
        replace(PARAMETERS, spot_tail_refinement=0.5)
    with pytest.raises(ValueError, match="front_spot_limit"):
        replace(PARAMETERS, front_spot_limit=-1.0)


def test_the_front_cap_acts_in_the_front_layers_only(model) -> None:
    capped = replace(model, parameters=replace(PARAMETERS, front_spot_limit=13.0))

    before, after = model.spot_counts(50_000.0), capped.spot_counts(50_000.0)

    assert (after <= before).all()
    assert after[0] <= 13 and after[1] <= 26 and after[2] <= 52
    assert after[10:] == pytest.approx(before[10:], rel=0.15)


def test_two_populations_conserve_the_layer_energy_apart_from_the_leakage(model) -> None:
    refined = replace(model, parameters=replace(PARAMETERS, spot_tail_refinement=3.0))
    rng = np.random.default_rng(2)

    batch = refined.sample_batch(50_000.0, axes(model, 400), rng)

    cells, layer = batch.cell_energy_mev.sum(axis=2), 50_000.0 * batch.layer_fractions
    assert (cells <= layer * (1 + 1e-9)).all()
    assert cells.sum() / layer.sum() > 0.99


def test_refined_tail_spots_occupy_more_cells_than_equal_spots(model) -> None:
    refined = replace(model, parameters=replace(PARAMETERS, spot_tail_refinement=4.0))
    axis = axes(model, 300, seed=6)

    equal = model.sample_batch(50_000.0, axis, np.random.default_rng(3)).cell_energy_mev
    finer = refined.sample_batch(50_000.0, axis, np.random.default_rng(3)).cell_energy_mev

    assert (finer > 0).sum() > 1.2 * (equal > 0).sum()


def test_the_two_population_mean_profile_equals_the_single_population_one(model) -> None:
    refined = replace(model, parameters=replace(PARAMETERS, spot_tail_refinement=3.0))
    axis = axes(model, 6000, seed=4)

    equal = model.sample_batch(50_000.0, axis, np.random.default_rng(9)).cell_energy_mev
    finer = refined.sample_batch(50_000.0, axis, np.random.default_rng(9)).cell_energy_mev
    equal_profile = equal.sum(axis=0) / equal.sum(axis=(0, 2))[:, None]
    finer_profile = finer.sum(axis=0) / finer.sum(axis=(0, 2))[:, None]

    assert (0.5 * np.abs(equal_profile - finer_profile).sum(axis=1))[3:].max() < 0.02


def test_the_mean_second_moment_matches_the_direct_sum_for_one_event(model) -> None:
    z = np.zeros((1, 2))
    axis = axes(model, 1, seed=8)

    moments = model.mean_second_moments_mm2(50_000.0, z, axis)

    ln_t, ln_alpha = model.longitudinal.draw_shape(50_000.0, z[:, 0], z[:, 1])
    tau = model.depth_coordinate(50_000.0, ln_t, ln_alpha)
    probabilities = model.strip_probabilities(50_000.0, axis, tau)[..., :-1]
    edges = model.cell_edges_mm
    offsets = 0.5 * (edges[:-1] + edges[1:])[None, None, :] - axis[:, :, None]
    assert moments.shape == (18,)
    assert moments == pytest.approx((probabilities * offsets**2).sum(axis=2)[0])
    assert (moments > 0).all()


def test_a_heavier_tail_raises_the_second_moment_far_more_than_the_central_profile(model) -> None:
    heavy = replace(model, parameters=replace(PARAMETERS, tail_scale=PARAMETERS.tail_scale * 1.5))
    z = np.random.default_rng(3).standard_normal((300, 2))
    axis = axes(model, 300, seed=2)

    light_moment, heavy_moment = (m.mean_second_moments_mm2(50_000.0, z, axis) for m in (model, heavy))
    light_profile, heavy_profile = (m.mean_cell_fractions(50_000.0, z, axis) for m in (model, heavy))

    moment_change = np.abs(heavy_moment / light_moment - 1.0)[3:].mean()
    profile_change = 0.5 * np.abs(heavy_profile - light_profile).sum(axis=1)[3:].mean()
    assert moment_change > 5.0 * profile_change


def test_the_tail_exponent_one_is_the_closed_form_and_larger_exponents_match_numerical_integration() -> None:
    from math import gamma, pi, sqrt

    from scipy import integrate

    x, radius = np.array([-80.0, -9.0, 0.0, 4.0, 30.0, 150.0]), 12.0
    assert lat.strip_cdf(x, radius, 1.0) == pytest.approx(0.5 * (1.0 + x / np.sqrt(x * x + radius * radius)))
    assert lat.strip_cdf(x, radius, 1.0 + 1e-7) == pytest.approx(lat.strip_cdf(x, radius, 1.0), abs=1e-6)

    s = 1.7

    def density(t: float) -> float:
        return gamma(s + 0.5) / (sqrt(pi) * gamma(s) * radius) * (1.0 + t * t / radius**2) ** (-(s + 0.5))

    numeric = np.array([integrate.quad(density, -np.inf, xi, limit=400)[0] for xi in x])
    assert lat.strip_cdf(x, radius, s) == pytest.approx(numeric, abs=1e-6)


def test_a_faster_tail_removes_far_energy_and_the_default_is_unchanged(model) -> None:
    faster = replace(model, parameters=replace(PARAMETERS, tail_exponent=1.8))
    axis = axes(model, 50, seed=11)
    tau = np.full((50, 18), 1.0)

    base = model.strip_probabilities(50_000.0, axis, tau)
    fast = faster.strip_probabilities(50_000.0, axis, tau)

    assert PARAMETERS.tail_exponent == 1.0
    assert fast.sum(axis=-1) == pytest.approx(1.0)
    far = slice(0, 36 - 12)  # cells at least 108 mm from the axis cell
    assert fast[..., far].sum() < 0.6 * base[..., far].sum()


def test_a_tail_exponent_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="tail_exponent"):
        replace(PARAMETERS, tail_exponent=0.8)


def test_an_artifact_with_the_extension_keys_round_trips(tmp_path) -> None:
    extended = replace(PARAMETERS, spot_tail_refinement=2.5, front_spot_limit=14.0)
    longitudinal = json.loads(LONGITUDINAL_ARTIFACT.read_text(encoding="utf-8"))["parameters_sha256"]

    path = lat.write_parameters_artifact(extended, tmp_path / "e1.json", {"note": "test"}, "tests", longitudinal)
    loaded, _ = lat.read_parameters_artifact(path)

    assert loaded == extended
