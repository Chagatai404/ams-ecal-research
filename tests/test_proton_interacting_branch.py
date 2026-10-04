"""The full proton generator: both branches, dispatch, reproducibility, artifact round trip."""

from dataclasses import replace
from math import exp
from pathlib import Path

import numpy as np
import pytest

from ams_ecal.proton import MODEL_VERSION_FULL, ProtonShowerModel, spawn_event_seeds
from ams_ecal.proton_calibration import ProtonCalibration
from ams_ecal.proton_structure import N_LAYERS
from ams_ecal.tracking import TrackState

ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "data" / "calibration" / "proton_model" / "ftfp_bert_v2"
ENERGY_MEV = 40_000.0

pytestmark = pytest.mark.skipif(
    not ARTIFACT.is_dir(), reason="the calibration artifact has not been built"
)


def track(x: float = 4.5, y: float = 4.5) -> TrackState:
    return TrackState(x0_mm=x, y0_mm=y, z0_mm=0.0, theta_rad=0.0, phi_rad=0.0)


@pytest.fixture(scope="module")
def model() -> ProtonShowerModel:
    return ProtonShowerModel.from_config()


@pytest.fixture(scope="module")
def interacting_seeds(model) -> list[int]:
    seeds = spawn_event_seeds(31, 400)
    return [s for s in seeds if model.interaction_for_seed(ENERGY_MEV, track(), s).interacts][:25]


def event(model, seed, representation="readout"):
    return model.as_representation(representation).generate_event(
        event_id=f"b-{seed}", primary_energy_mev=ENERGY_MEV, track=track(), random_seed=seed
    )


def test_the_committed_artifact_carries_every_branch(model) -> None:
    assert model.has_interacting_branch
    assert model.model_version == MODEL_VERSION_FULL
    assert model.calibration.interacting is not None and model.calibration.lateral is not None


def test_an_interacting_event_is_a_valid_event_that_says_where_it_interacted(
    model, interacting_seeds
) -> None:
    result = event(model, interacting_seeds[0])

    grid = np.array(result.cell_energies_mev)
    assert grid.shape == (N_LAYERS, 72)
    assert np.all(np.isfinite(grid)) and np.all(grid >= 0.0)
    details = dict(result.provenance.model_details)
    assert details["interaction_status"] == "interacting"
    assert 0.0 <= float(details["interaction_depth_mm"]) < model.calibration.depth_mm
    assert 0 <= int(details["interaction_layer"]) < N_LAYERS


def test_the_same_seed_gives_the_same_interacting_event(model, interacting_seeds) -> None:
    seed = interacting_seeds[1]

    assert event(model, seed).cell_energies_mev == event(model, seed).cell_energies_mev
    assert event(model, seed).cell_energies_mev != event(model, interacting_seeds[2]).cell_energies_mev


def test_the_readout_event_does_not_depend_on_whether_the_deposition_one_was_generated(
    model, interacting_seeds
) -> None:
    seed = interacting_seeds[3]
    first = event(model, seed, "readout").cell_energies_mev
    event(model, seed, "deposition")  # generating the other representation in between
    again = event(model, seed, "readout").cell_energies_mev

    assert first == again


def test_one_seed_names_one_interaction_depth_in_both_representations(
    model, interacting_seeds
) -> None:
    for seed in interacting_seeds[:10]:
        readout = dict(event(model, seed, "readout").provenance.model_details)
        deposition = dict(event(model, seed, "deposition").provenance.model_details)

        assert readout["interaction_depth_mm"] == deposition["interaction_depth_mm"]


def test_the_deposition_representation_is_the_larger_energy_scale(model, interacting_seeds) -> None:
    readout = [event(model, s).total_ecal_energy_mev for s in interacting_seeds[:15]]
    deposition = [event(model, s, "deposition").total_ecal_energy_mev for s in interacting_seeds[:15]]

    assert np.median(deposition) > 8 * np.median(readout)


def test_generate_event_dispatches_by_the_seeds_own_interaction_draw(model) -> None:
    seeds = spawn_event_seeds(32, 600)
    statuses = [
        dict(event(model, s).provenance.model_details)["interaction_status"] for s in seeds[:200]
    ]

    expected = 1.0 - exp(-model.calibration.depth_mm / model.effective_length_mm)
    assert set(statuses) == {"crossing", "interacting"}
    assert statuses.count("interacting") / len(statuses) == pytest.approx(expected, abs=0.09)


def test_a_seed_that_crosses_is_refused_by_the_interacting_generator(model) -> None:
    crossing_seed = next(
        s
        for s in spawn_event_seeds(33, 50)
        if not model.interaction_for_seed(ENERGY_MEV, track(), s).interacts
    )

    with pytest.raises(ValueError, match="draws no interaction"):
        model.generate_interacting_event(
            event_id="x", primary_energy_mev=ENERGY_MEV, track=track(), random_seed=crossing_seed
        )


def test_an_energy_outside_the_calibrated_domain_is_refused(model, interacting_seeds) -> None:
    with pytest.raises(ValueError, match="outside the calibrated domain"):
        model.generate_event(
            event_id="x", primary_energy_mev=500_000.0, track=track(), random_seed=interacting_seeds[0]
        )


def test_interacting_events_are_more_energetic_than_crossing_ones(model) -> None:
    seeds = spawn_event_seeds(34, 300)
    totals: dict[str, list[float]] = {"crossing": [], "interacting": []}
    for seed in seeds[:120]:
        result = event(model, seed)
        totals[dict(result.provenance.model_details)["interaction_status"]].append(
            result.total_ecal_energy_mev
        )

    assert np.median(totals["interacting"]) > 10 * np.median(totals["crossing"])


def test_a_calibration_without_the_interacting_tables_refuses_an_interacting_seed(
    model, interacting_seeds
) -> None:
    plain = replace(model, calibration=replace(model.calibration, interacting=None, lateral=None))

    assert not plain.has_interacting_branch
    with pytest.raises(NotImplementedError):
        plain.generate_event(
            event_id="x", primary_energy_mev=ENERGY_MEV, track=track(), random_seed=interacting_seeds[0]
        )


def test_the_artifact_round_trips_with_every_table(model, tmp_path) -> None:
    saved = model.calibration.save(tmp_path / "artifact")

    loaded = ProtonCalibration.load(saved)

    assert loaded.content_sha256 == model.calibration.content_sha256
    for name, array in model.calibration.arrays().items():
        assert np.array_equal(array, loaded.arrays()[name]), name
    assert loaded.interacting is not None and loaded.lateral is not None


def test_energy_in_front_of_the_interaction_is_split_into_the_protons_own_deposit_and_backsplash(
    model, interacting_seeds, monkeypatch
) -> None:
    import ams_ecal.proton as proton_module

    def front_hits_and_layer_energy(cap: float):
        monkeypatch.setattr(proton_module, "UPSTREAM_MIP_CAP", cap)
        hits, energy = [], []
        for seed in interacting_seeds[:12]:
            result = event(model, seed, "deposition")
            grid = np.array(result.cell_energies_mev)
            front = int(dict(result.provenance.model_details)["interaction_layer"])
            hits.append(int((grid[:front] > 0.28).sum()))
            energy.append(grid.sum(axis=1))
        return np.array(hits), np.array(energy)

    all_backsplash_hits, all_backsplash_energy = front_hits_and_layer_energy(0.0)
    all_own_hits, all_own_energy = front_hits_and_layer_energy(1e9)

    assert all_backsplash_energy == pytest.approx(all_own_energy)  # the split moves energy between cells only
    assert all_backsplash_hits.sum() > 1.2 * all_own_hits.sum()  # backsplash lights more cells
