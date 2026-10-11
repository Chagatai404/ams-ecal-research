"""Batch generation: identical to the per-event generator however it is chunked or parallelised."""

from pathlib import Path

import numpy as np
import pytest

from ams_ecal.detector.tracking import TrackState
from ams_ecal.proton_model.proton import ProtonShowerModel, spawn_event_seeds
from ams_ecal.proton_model.proton_batch import (
    BatchResult,
    generate_batch,
    generate_batch_parallel,
)

ROOT = Path(__file__).parents[2]
ARTIFACT = ROOT / "data" / "calibration" / "proton_model" / "ftfp_bert_v2"

pytestmark = pytest.mark.skipif(
    not ARTIFACT.is_dir(), reason="the calibration artifact has not been built"
)


@pytest.fixture(scope="module")
def model() -> ProtonShowerModel:
    return ProtonShowerModel.from_config()


def inputs(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    energies = np.exp(rng.uniform(np.log(12_000.0), np.log(95_000.0), n))  # off-anchor energies
    return energies, rng.uniform(0.0, 9.0, n), rng.uniform(0.0, 9.0, n), spawn_event_seeds(seed + 5, n)


def test_a_batch_is_the_per_event_generator_event_for_event(model) -> None:
    energies, x, y, seeds = inputs(24)

    batch = generate_batch(model, energies, x, y, seeds)

    for i in range(0, 24, 3):
        track = TrackState(x0_mm=x[i], y0_mm=y[i], z0_mm=0.0, theta_rad=0.0, phi_rad=0.0)
        event = model.generate_event(
            event_id="x", primary_energy_mev=energies[i], track=track, random_seed=seeds[i]
        )
        assert np.array_equal(
            batch.grids_mev[i], np.array(event.cell_energies_mev, dtype=np.float32)
        )
        status = dict(event.provenance.model_details)["interaction_status"]
        assert batch.interacting[i] == (status == "interacting")


def test_both_branches_appear_and_only_interacting_events_have_a_depth(model) -> None:
    batch = generate_batch(model, *inputs(60, seed=1))

    assert batch.interacting.any() and (~batch.interacting).any()
    assert np.all(np.isfinite(batch.interaction_depth_mm[batch.interacting]))
    assert np.all(np.isnan(batch.interaction_depth_mm[~batch.interacting]))
    assert np.all(batch.interaction_depth_mm[batch.interacting] >= 0.0)


def test_the_arrays_are_valid_and_float32(model) -> None:
    batch = generate_batch(model, *inputs(30, seed=2))

    assert batch.grids_mev.dtype == np.float32 and batch.grids_mev.shape == (30, 18, 72)
    assert np.all(np.isfinite(batch.grids_mev)) and np.all(batch.grids_mev >= 0.0)
    assert len(batch) == 30
    assert isinstance(batch, BatchResult)


def test_the_result_does_not_depend_on_how_the_batch_is_cut(model) -> None:
    energies, x, y, seeds = inputs(20, seed=3)

    whole = generate_batch(model, energies, x, y, seeds)
    parts = [
        generate_batch(model, energies[a : a + 7], x[a : a + 7], y[a : a + 7], seeds[a : a + 7])
        for a in (0, 7, 14)
    ]

    assert np.array_equal(whole.grids_mev, np.concatenate([p.grids_mev for p in parts]))


def test_a_parallel_batch_equals_the_serial_one() -> None:
    energies, x, y, seeds = inputs(16, seed=4)
    serial = generate_batch(ProtonShowerModel.from_config(), energies, x, y, seeds)

    parallel = generate_batch_parallel("readout", energies, x, y, seeds, workers=2, chunk=5)

    assert np.array_equal(serial.grids_mev, parallel.grids_mev)
    assert np.array_equal(serial.interacting, parallel.interacting)
    assert np.array_equal(serial.seeds, parallel.seeds)


def test_the_deposition_representation_is_available_in_a_batch() -> None:
    energies, x, y, seeds = inputs(8, seed=5)

    readout = generate_batch_parallel("readout", energies, x, y, seeds, workers=1)
    deposition = generate_batch_parallel("deposition", energies, x, y, seeds, workers=1)

    assert deposition.grids_mev.sum() > 8 * readout.grids_mev.sum()
    assert np.array_equal(readout.interacting, deposition.interacting)  # one seed, one draw


def test_inconsistent_arrays_and_repeated_seeds_are_refused(model) -> None:
    energies, x, y, seeds = inputs(5, seed=6)

    with pytest.raises(ValueError, match="same length"):
        generate_batch(model, energies[:4], x, y, seeds)
    with pytest.raises(ValueError, match="own seed"):
        generate_batch(model, energies, x, y, np.array([1, 2, 3, 3, 4]))
    with pytest.raises(ValueError, match="positive"):
        generate_batch_parallel("readout", energies, x, y, seeds, workers=0)


def test_an_energy_outside_the_calibrated_domain_is_refused(model) -> None:
    energies, x, y, seeds = inputs(3, seed=7)
    energies[1] = 400_000.0

    with pytest.raises(ValueError, match="outside the calibrated domain"):
        generate_batch(model, energies, x, y, seeds)
