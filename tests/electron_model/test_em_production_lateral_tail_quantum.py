"""Extension E2 of the lateral slice: the radial tail's spots carry a fixed absolute energy (opt-in, default off)."""

from dataclasses import replace

import numpy as np
import pytest

from ams_ecal.electron_model import em_production_lateral as lat

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


def axes(model: lat.EMProductionLateralModel, n: int, seed: int) -> np.ndarray:
    entry = np.random.default_rng(seed).uniform(0.0, 9.0, size=(n, 2))
    return model.axis_for_entry(entry[:, 0], entry[:, 1])


def with_quantum(model: lat.EMProductionLateralModel, quantum: float) -> lat.EMProductionLateralModel:
    return replace(model, parameters=replace(PARAMETERS, tail_quantum_mev=quantum))


def test_the_quantum_is_off_by_default_and_a_negative_one_is_refused() -> None:
    assert PARAMETERS.tail_quantum_mev == 0.0
    with pytest.raises(ValueError, match="tail_quantum_mev"):
        replace(PARAMETERS, tail_quantum_mev=-1.0)


def test_the_quantum_and_the_relative_refinement_of_e1_are_alternatives() -> None:
    with pytest.raises(ValueError, match="alternatives"):
        replace(PARAMETERS, tail_quantum_mev=1.3, spot_tail_refinement=2.0)


def test_tail_spots_conserve_the_layer_energy_apart_from_the_leakage(model) -> None:
    batch = with_quantum(model, 1.3).sample_batch(50_000.0, axes(model, 300, seed=3), np.random.default_rng(5))

    cells, layer = batch.cell_energy_mev.sum(axis=2), 50_000.0 * batch.layer_fractions
    assert (cells <= layer * (1 + 1e-9)).all()
    assert cells.sum() / layer.sum() > 0.99


def test_a_smaller_quantum_occupies_more_cells_and_the_far_tail_energy_is_unchanged(model) -> None:
    axis = axes(model, 400, seed=8)

    coarse = with_quantum(model, 6.0).sample_batch(50_000.0, axis, np.random.default_rng(7)).cell_energy_mev
    fine = with_quantum(model, 1.0).sample_batch(50_000.0, axis, np.random.default_rng(7)).cell_energy_mev

    assert (fine > 0).sum() > 1.15 * (coarse > 0).sum()
    far = np.s_[:, :, :20]
    assert fine[far].sum() == pytest.approx(coarse[far].sum(), rel=0.15)


def test_the_mean_profile_is_unchanged_by_the_quantum(model) -> None:
    axis = axes(model, 6000, seed=4)

    equal = model.sample_batch(50_000.0, axis, np.random.default_rng(9)).cell_energy_mev
    fine = with_quantum(model, 1.3).sample_batch(50_000.0, axis, np.random.default_rng(9)).cell_energy_mev
    equal_profile = equal.sum(axis=0) / equal.sum(axis=(0, 2))[:, None]
    fine_profile = fine.sum(axis=0) / fine.sum(axis=(0, 2))[:, None]

    assert (0.5 * np.abs(equal_profile - fine_profile).sum(axis=1))[3:].max() < 0.02


def test_an_artifact_with_the_quantum_round_trips_and_the_committed_one_still_verifies(model, tmp_path) -> None:
    path = tmp_path / "parameters.json"
    lat.write_parameters_artifact(
        replace(PARAMETERS, tail_quantum_mev=1.36), path, {"note": "test"}, "test", model.longitudinal_sha256
    )

    parameters, _ = lat.read_parameters_artifact(path)

    assert parameters.tail_quantum_mev == 1.36
    committed, _ = lat.read_parameters_artifact()
    assert committed.tail_quantum_mev == 0.0
