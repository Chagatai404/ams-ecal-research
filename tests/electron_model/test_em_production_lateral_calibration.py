"""em_production lateral calibration: planted recovery of the radial law and the spot structure, and the records."""

from dataclasses import replace

import numpy as np
import pytest

from ams_ecal.electron_model import em_production_lateral as lat
from ams_ecal.electron_model import em_production_lateral_calibration as cal
from ams_ecal.electron_studies.em_lateral_structure_study import total_variation

needs_sample = pytest.mark.skipif(
    not (cal.DATA / "E50GeV" / "events.npz").exists(),
    reason="the exposed baseline electron sample is git-ignored: regenerate it from configs/geant4_electron_sample.yaml",
)
ENERGIES = (20.0, 50.0)


@pytest.fixture(scope="module")
def template() -> lat.EMProductionLateralModel:
    return lat.build_model(cal.START)


def observed_from(truth: lat.EMProductionLateralModel) -> dict[float, dict]:
    z, axis = cal.quadrature(truth)
    return {e: {"profile": cal.predicted_profile(truth, e, z, axis)} for e in ENERGIES}


def test_the_quadrature_set_is_fixed_and_covers_the_entry_cell(template) -> None:
    z, axis = cal.quadrature(template)
    z2, axis2 = cal.quadrature(template)

    assert z.shape == (256, 2)
    assert axis.shape == (256, 18)
    assert z == pytest.approx(z2)
    assert axis == pytest.approx(axis2)
    assert axis.min() >= 0.0
    assert axis.max() <= 9.0
    assert abs(z.mean()) < 0.1
    assert z.std() == pytest.approx(1.0, abs=0.1)


def test_the_predicted_profile_is_normalised_per_layer(template) -> None:
    z, axis = cal.quadrature(template)

    profile = cal.predicted_profile(template, 50.0, z, axis)

    assert profile.shape == (18, 72)
    assert profile.sum(axis=1) == pytest.approx(1.0)


def test_the_radial_fit_recovers_a_planted_profile(template) -> None:
    truth_parameters = replace(
        cal.START,
        tail_scale=cal.START.tail_scale * 1.15,
        core_depth_slope=cal.START.core_depth_slope * 1.25,
        weight_centre=cal.START.weight_centre + 0.06,
    )
    truth = replace(template, parameters=truth_parameters)
    observed = observed_from(truth)
    z, axis = cal.quadrature(template)

    fitted, record = cal.fit_radial_law(template, observed, max_nfev=40)

    before = max(
        100
        * total_variation(
            cal.predicted_profile(template, e, z, axis)[layer],
            observed[e]["profile"][layer],
        )
        for e in ENERGIES
        for layer in range(18)
    )
    after_model = replace(template, parameters=fitted)
    after = max(
        100
        * total_variation(
            cal.predicted_profile(after_model, e, z, axis)[layer],
            observed[e]["profile"][layer],
        )
        for e in ENERGIES
        for layer in range(18)
    )
    assert after < 0.4
    assert after < 0.3 * before
    assert record["free_parameters"] == list(lat.RADIAL_NAMES)
    assert record["quadrature_points"] == 256


def test_the_spot_fit_recovers_a_planted_spot_structure(template, monkeypatch) -> None:
    monkeypatch.setattr(cal, "SPOT_EVENTS", 300)
    truth_parameters = replace(
        cal.START,
        spot_amplitude=150.0,
        spot_energy_exponent=0.9,
        spot_depth_scale=1.05,
        spot_shape_scale=0.95,
    )
    truth = replace(template, parameters=truth_parameters)
    prepared = cal.prepare_spot_batches(truth, ENERGIES, events=300)
    observed = {
        e: {"spot_number_conditioned": n}
        for e, n in cal.simulated_spot_numbers(truth, prepared, cal.SPOT_SEED + 1).items()
    }

    fitted, record = cal.fit_spot_structure(template, observed, max_iterations=150)

    start_objective = np.mean(
        [
            np.mean(
                (
                    np.log(cal.simulated_spot_numbers(template, prepared, cal.SPOT_SEED + 1)[e][2:])
                    - np.log(observed[e]["spot_number_conditioned"][2:])
                )
                ** 2
            )
            for e in ENERGIES
        ]
    )
    assert record["objective_mean_squared_log_ratio"] < 0.25 * start_objective
    assert record["free_parameters"] == list(lat.SPOT_NAMES)
    assert "layers 0-1" in record["front_layers_note"]
    assert fitted.spot_energy_exponent == pytest.approx(0.9, abs=0.15)


@needs_sample
def test_the_observed_loader_reads_the_exposed_baseline_with_its_targets() -> None:
    observed = cal.load_observed((50.0,), max_events=120)

    block = observed[50.0]
    assert block["events"] == 120
    assert block["profile"].shape == (18, 72)
    assert block["profile"].sum(axis=1) == pytest.approx(1.0)
    assert block["spot_number_conditioned"].shape == (18,)
    assert (block["spot_number_conditioned"] > 0).all()


@needs_sample
def test_the_calibration_runs_end_to_end_on_a_small_slice_and_records_its_conventions(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cal, "SPOT_EVENTS", 200)
    observed = cal.load_observed((50.0,), max_events=150)

    parameters, record, model = cal.calibrate(observed, max_nfev=3, max_spot_iterations=3)

    assert record["sealed_data_read"] is False
    assert record["method"]["moliere_radius_mm"] == 18.0
    assert record["data"]["sample"] == "baseline"
    assert record["longitudinal_parameters_sha256"] == model.longitudinal_sha256
    path = lat.write_parameters_artifact(parameters, tmp_path / "p.json", record, "tests", model.longitudinal_sha256)
    assert lat.read_parameters_artifact(path)[0] == parameters
