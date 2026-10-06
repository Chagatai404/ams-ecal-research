"""em_production calibration: planted recovery of the parameters, the constraint and the fixed-origin convention."""

from dataclasses import replace

import numpy as np
import pytest

from ams_ecal.electron_model import em_production as ep
from ams_ecal.electron_model import em_production_calibration as cal

needs_sample = pytest.mark.skipif(
    not (cal.SAMPLE_DIR / "E10GeV" / "events.npz").exists(),
    reason="the exposed baseline electron sample is git-ignored: regenerate it from configs/geant4_electron_sample.yaml",
)
ENERGIES = (10.0, 50.0, 100.0)
TRUTH = replace(
    cal.START_PARAMETERS,
    origin_x0=0.0,
    beta=0.55,
    tail_weight=0.06,
    tail_weight_log_energy_slope=-0.02,
    tail_shape=2.2,
    rho_intercept=0.9,
    rho_log_slope=-0.04,
    ln_t_skewness=0.7,
)


def planted_fractions(
    parameters: ep.EMProductionParameters, n: int = 800
) -> tuple[dict[float, np.ndarray], np.ndarray, float]:
    template = ep.build_model(parameters)
    bounds = np.asarray(template.longitudinal.layer_bounds_x0, dtype=float)
    out = {}
    for energy in ENERGIES:
        rng = np.random.default_rng(int(energy) + 100)
        out[energy] = template.longitudinal.sample_layer_energy_fractions_batch(1000.0 * energy, rng, n)
    return out, bounds, template.longitudinal.critical_energy_mev


def test_the_quadrature_set_is_fixed_and_standard_normal() -> None:
    first, second = cal.quadrature_normals(), cal.quadrature_normals()

    assert first.shape == (2048, 2)
    assert first == pytest.approx(second)
    assert first.mean(axis=0) == pytest.approx([0.0, 0.0], abs=0.03)
    assert first.std(axis=0) == pytest.approx([1.0, 1.0], abs=0.03)
    assert abs(np.corrcoef(first.T)[0, 1]) < 0.05


def test_the_correlation_line_and_skewness_are_read_from_the_measured_moments() -> None:
    critical = 7.6
    moments = {
        f"{e:g}": {"pearson": 0.9 - 0.04 * np.log(1000.0 * e / critical), "skewness_ln_t": 0.6 + 0.01 * k}
        for k, e in enumerate(ENERGIES)
    }

    fit = cal.fit_fluctuation_parameters(moments, critical)

    assert fit["rho_intercept"] == pytest.approx(0.9)
    assert fit["rho_log_slope"] == pytest.approx(-0.04)
    assert fit["ln_t_skewness"] == pytest.approx(0.61)


def test_the_central_fit_recovers_planted_parameters_at_a_fixed_origin() -> None:
    fractions, bounds, critical = planted_fractions(TRUTH)

    fitted, diagnostics = cal.fit_central_parameters(
        fractions, TRUTH, critical, bounds, cal.quadrature_normals(), fixed_origin=0.0
    )

    assert fitted.origin_x0 == 0.0
    assert fitted.beta == pytest.approx(TRUTH.beta, abs=0.03)
    assert fitted.tail_weight == pytest.approx(TRUTH.tail_weight, abs=0.02)
    assert fitted.tail_shape == pytest.approx(TRUTH.tail_shape, abs=0.6)
    assert diagnostics["free_parameters"] == ["beta", "tail_weight", "tail_weight_log_energy_slope", "tail_shape"]
    assert diagnostics["origin_fixed_at_x0"] == 0.0
    assert diagnostics["contained_penalty_active"] is False
    assert diagnostics["chi2_per_dof"] < 3.0


def test_the_contained_fraction_constraint_is_reported() -> None:
    fractions, bounds, critical = planted_fractions(TRUTH, n=300)
    squeezed = {e: f * np.where(np.arange(18) < 4, 1.0, 0.97) for e, f in fractions.items()}

    _, diagnostics = cal.fit_central_parameters(
        squeezed, TRUTH, critical, bounds, cal.quadrature_normals(), fixed_origin=0.0
    )

    assert diagnostics["contained_tolerance"] == cal.CONTAINED_TOLERANCE
    assert len(diagnostics["contained_difference_model_minus_geant4"]) == len(ENERGIES)


def test_a_free_origin_adds_the_origin_to_the_fitted_parameters() -> None:
    fractions, bounds, critical = planted_fractions(TRUTH, n=300)

    _, diagnostics = cal.fit_central_parameters(fractions, TRUTH, critical, bounds, cal.quadrature_normals(), None)

    assert "origin_x0" in diagnostics["free_parameters"]
    assert diagnostics["origin_fixed_at_x0"] is None


@needs_sample
def test_the_sample_loader_reads_the_exposed_baseline_with_its_provenance() -> None:
    sample = cal.load_sample((10.0,), max_events=50)

    assert sample["fractions"][10.0].shape == (50, 18)
    assert 0.5 < sample["fractions"][10.0].sum(axis=1).mean() < 1.0
    batch = sample["provenance"]["10"]
    assert batch["events"] == 50
    assert batch["configuration_sha256"]
    assert batch["sample"] == "baseline"


@needs_sample
def test_the_calibration_runs_end_to_end_on_a_small_slice_and_records_its_conventions(tmp_path) -> None:
    sample = cal.load_sample(max_events=120)

    parameters, record = cal.calibrate(sample)

    assert parameters.origin_x0 == cal.FRONT_FACE_ORIGIN_X0
    assert record["method"]["converged"] is True
    assert record["method"]["origin_convention"] == "front face"
    assert record["data"]["sealed_data_read"] is False
    assert record["central_fit"]["chi2_per_dof"] > 0
    path = ep.write_parameters_artifact(parameters, tmp_path / "p.json", record, "tests")
    assert ep.read_parameters_artifact(path)[0] == parameters
