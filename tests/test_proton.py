"""Proton model generator: the interaction draw and the crossing branch."""

from dataclasses import replace
from math import exp, log1p
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from ams_ecal.crossing import FibreCrossingGeometry
from ams_ecal.event import ECALEvent
from ams_ecal.geometry import load_geometry
from ams_ecal.proton import ProtonShowerModel, spawn_event_seeds
from ams_ecal.proton_calibration import CrossingTable, ProtonCalibration
from ams_ecal.proton_config import load_proton_config
from ams_ecal.tracking import TrackState

ROOT = Path(__file__).parents[1]
GEOMETRY = ROOT / "configs" / "geometry.yaml"
CONFIG = ROOT / "configs" / "fastmc_proton.yaml"
ARTIFACT = ROOT / "data" / "calibration" / "proton_model" / "ftfp_bert_v2"

LENGTH_MM = 255.0
DEPTH_MM = 166.5
ENERGY_MEV = 30_000.0


def track(x: float = 4.5, y: float = 4.5, theta: float = 0.0) -> TrackState:
    return TrackState(x0_mm=x, y0_mm=y, z0_mm=0.0, theta_rad=theta, phi_rad=0.0)


def make_calibration() -> ProtonCalibration:
    levels = np.array([0.0, 0.25, 0.5, 0.75, 0.95, 1.0])
    # layer energy grows with the chord bin and with energy; deposition = 10 x readout
    bins = np.array([0.4, 0.5, 0.6])[:, None] * np.array([0.2, 0.8, 1.0, 1.3, 2.5, 6.0])[None, :]
    readout = np.stack([bins, 1.2 * bins])
    table = CrossingTable(
        energies_gev=np.array([10.0, 100.0]),
        chord_edges_mm=np.array([2.6, 3.2]),
        levels=levels,
        quantiles_mev={"readout": readout, "deposition": 10.0 * readout},
        counts=np.full((2, 3), 1000),
    )
    manifest = {
        "schema_version": 1,
        "physics_list": "ftfp_bert",
        "content_sha256": "a" * 64,
        "source": {"geant4_version": "test-11.4"},
    }
    return ProtonCalibration(manifest, table, LENGTH_MM, 3.0, DEPTH_MM)


@pytest.fixture(scope="module")
def model() -> ProtonShowerModel:
    geometry = load_geometry(GEOMETRY)
    return ProtonShowerModel(
        load_proton_config(CONFIG),
        make_calibration(),
        FibreCrossingGeometry(geometry),
        "0" * 64,
    )


def generate(model: ProtonShowerModel, seed: int, **kwargs) -> ECALEvent:
    return model.generate_crossing_event(
        event_id=f"p-{seed}",
        primary_energy_mev=kwargs.pop("energy", ENERGY_MEV),
        track=kwargs.pop("track", track()),
        random_seed=seed,
    )


# --- the interaction draw ------------------------------------------------


def test_the_crossing_fraction_is_the_survival_probability(model) -> None:
    seeds = spawn_event_seeds(1, 20_000)
    crosses = [not model.interaction_for_seed(ENERGY_MEV, track(), s).interacts for s in seeds]

    expected = exp(-DEPTH_MM / LENGTH_MM)
    assert np.mean(crosses) == pytest.approx(expected, abs=0.012)


def test_the_interaction_depth_follows_the_truncated_exponential(model) -> None:
    seeds = spawn_event_seeds(2, 20_000)
    depths = np.array(
        [
            d.depth_mm
            for s in seeds
            if (d := model.interaction_for_seed(ENERGY_MEV, track(), s)).interacts
        ]
    )

    def cdf(z: np.ndarray) -> np.ndarray:
        return (1 - np.exp(-z / LENGTH_MM)) / (1 - exp(-DEPTH_MM / LENGTH_MM))

    assert stats.kstest(depths, cdf).pvalue > 0.001


def test_one_draw_decides_status_and_depth_consistently(model) -> None:
    for seed in spawn_event_seeds(3, 500):
        u = np.random.default_rng(seed).random()
        distance = -LENGTH_MM * log1p(-u)
        draw = model.interaction_for_seed(ENERGY_MEV, track(), seed)

        assert draw.interacts == (distance < DEPTH_MM)
        if draw.interacts:
            assert draw.depth_mm == pytest.approx(distance)
            assert 0.0 <= draw.depth_mm < draw.path_mm


def test_a_numeric_interaction_length_overrides_the_calibrated_one(model) -> None:
    config = replace(
        model.config, interaction=replace(model.config.interaction, effective_length_mm=1.0)
    )
    short = replace(model, config=config)

    assert model.effective_length_source == "calibration"
    assert short.effective_length_source == "configuration"
    assert short.effective_length_mm == 1.0
    fraction = np.mean(
        [
            short.interaction_for_seed(ENERGY_MEV, track(), s).interacts
            for s in spawn_event_seeds(4, 500)
        ]
    )
    assert fraction > 0.99


# --- reproducibility --------------------------------------------------------


def crossing_seeds(model: ProtonShowerModel, count: int, base: int = 5) -> list[int]:
    seeds = spawn_event_seeds(base, 20 * count)
    keep = [s for s in seeds if not model.interaction_for_seed(ENERGY_MEV, track(), s).interacts]
    return keep[:count]


def test_the_same_seed_gives_the_same_event(model) -> None:
    seed = crossing_seeds(model, 1)[0]

    assert generate(model, seed) == generate(model, seed)


def test_different_seeds_give_different_events(model) -> None:
    a, b = crossing_seeds(model, 2)

    assert generate(model, a).cell_energies_mev != generate(model, b).cell_energies_mev


def test_an_event_does_not_depend_on_the_events_generated_before_it(model) -> None:
    seeds = crossing_seeds(model, 5)
    alone = generate(model, seeds[3])
    for seed in seeds[:3]:
        generate(model, seed)

    assert generate(model, seeds[3]) == alone


# --- the crossing event ------------------------------------------------------


def test_a_crossing_event_is_a_valid_canonical_event(model) -> None:
    event = generate(model, crossing_seeds(model, 1)[0])

    assert isinstance(event, ECALEvent)
    assert event.particle_type == "proton"
    assert (len(event.cell_energies_mev), len(event.cell_energies_mev[0])) == (18, 72)
    assert all(np.isfinite(v) and v >= 0 for row in event.cell_energies_mev for v in row)
    assert event.total_ecal_energy_mev > 0


def test_energy_only_appears_in_cells_the_track_crosses(model) -> None:
    t = track(-0.3, -0.3)
    event = generate(model, crossing_seeds(model, 1)[0], track=t)
    grid = np.array(event.cell_energies_mev)

    assert set(np.flatnonzero(grid.sum(axis=0)).tolist()) <= {35, 36}


def test_every_layer_receives_energy(model) -> None:
    event = generate(model, crossing_seeds(model, 1)[0])

    assert all(energy > 0 for energy in event.layer_energies_mev)


def test_the_entry_phase_moves_the_mean_signal_in_the_calibrated_direction(model) -> None:
    coordinates = np.linspace(0.0, 1.35, 28, endpoint=False)
    totals = {c: model.crossing.cross(track(c, c)).chord_mm.sum() for c in coordinates}
    low, high = min(totals, key=totals.get), max(totals, key=totals.get)
    seeds = crossing_seeds(model, 300)

    def mean_energy(c: float) -> float:
        return np.mean([generate(model, s, track=track(c, c)).total_ecal_energy_mev for s in seeds])

    assert totals[high] > totals[low]
    assert mean_energy(high) > mean_energy(low)


def test_the_deposition_representation_carries_its_own_larger_scale(model) -> None:
    deposition = model.as_representation("deposition")
    seeds = crossing_seeds(model, 200)

    readout = np.mean([generate(model, s).total_ecal_energy_mev for s in seeds])
    deposited = np.mean([generate(deposition, s).total_ecal_energy_mev for s in seeds])

    assert deposited / readout == pytest.approx(10.0, rel=0.1)


def test_one_seed_shares_the_interaction_draw_between_representations(model) -> None:
    deposition = model.as_representation("deposition")
    for seed in spawn_event_seeds(6, 200):
        assert model.interaction_for_seed(
            ENERGY_MEV, track(), seed
        ) == deposition.interaction_for_seed(ENERGY_MEV, track(), seed)


def test_the_layer_fluctuations_of_the_two_representations_are_independent(model) -> None:
    deposition = model.as_representation("deposition")
    seeds = crossing_seeds(model, 300)
    readout = np.array([generate(model, s).layer_energies_mev for s in seeds]).ravel()
    deposited = np.array([generate(deposition, s).layer_energies_mev for s in seeds]).ravel()

    assert abs(stats.spearmanr(readout, deposited).statistic) < 0.05


# --- provenance ---------------------------------------------------------------


def test_every_event_carries_its_model_and_calibration_provenance(model) -> None:
    seed = crossing_seeds(model, 1)[0]
    provenance = generate(model, seed).provenance

    assert provenance.simulation_backend == "fastmc"
    assert provenance.random_seed == seed
    assert provenance.configuration_sha256 == "0" * 64
    assert provenance.detail("model") == "block6b-proton"
    assert provenance.detail("model_version") == "1-slice2"
    assert provenance.detail("physics_list") == "ftfp_bert"
    assert provenance.detail("representation") == "readout"
    assert provenance.detail("interaction_status") == "crossing"
    assert provenance.detail("calibration_content_sha256") == "a" * 64
    assert provenance.detail("effective_length_source") == "calibration"
    assert float(provenance.detail("effective_length_mm")) == LENGTH_MM
    assert provenance.detail("geant4_version") == "test-11.4"


def test_the_representation_is_recorded_in_the_event(model) -> None:
    event = generate(model.as_representation("deposition"), crossing_seeds(model, 1)[0])

    assert event.provenance.detail("representation") == "deposition"


# --- refusals -----------------------------------------------------------------


@pytest.mark.parametrize("energy_mev", [9_999.0, 100_001.0, 1_000.0])
def test_energies_outside_the_calibrated_domain_are_refused(model, energy_mev) -> None:
    with pytest.raises(ValueError, match="calibrated domain"):
        generate(model, 1, energy=energy_mev)


def test_a_tilted_track_is_refused(model) -> None:
    with pytest.raises(ValueError, match="maximum"):
        generate(model, 1, track=track(theta=0.05))


def test_a_track_outside_the_active_area_is_refused(model) -> None:
    with pytest.raises(ValueError, match="outside"):
        generate(model, 1, track=track(x=400.0))


def test_the_configuration_must_match_the_calibration_scenario(model) -> None:
    config = replace(
        model.config, calibration=replace(model.config.calibration, physics_list="qgsp_bert")
    )

    with pytest.raises(ValueError, match="qgsp_bert"):
        replace(model, config=config)


def test_the_domain_may_not_exceed_the_calibrated_range(model) -> None:
    domain = replace(model.config.domain, energy_max_gev=200.0)

    with pytest.raises(ValueError, match="exceeds the calibrated range"):
        replace(model, config=replace(model.config, domain=domain))


# --- the full generator -------------------------------------------------------


def test_generate_event_returns_crossing_events_and_refuses_interacting_ones(model) -> None:
    seeds = spawn_event_seeds(7, 200)
    crossing = next(
        s for s in seeds if not model.interaction_for_seed(ENERGY_MEV, track(), s).interacts
    )
    interacting = next(
        s for s in seeds if model.interaction_for_seed(ENERGY_MEV, track(), s).interacts
    )

    event = model.generate_event(
        event_id=f"p-{crossing}",
        primary_energy_mev=ENERGY_MEV,
        track=track(),
        random_seed=crossing,
    )
    assert event == generate(model, crossing)
    with pytest.raises(NotImplementedError, match="interacting-proton factorization"):
        model.generate_event(
            event_id="b", primary_energy_mev=ENERGY_MEV, track=track(), random_seed=interacting
        )


# --- seeds --------------------------------------------------------------------


def test_spawned_seeds_are_reproducible_and_distinct() -> None:
    assert spawn_event_seeds(9, 1000) == spawn_event_seeds(9, 1000)
    assert len(set(spawn_event_seeds(9, 1000))) == 1000
    assert spawn_event_seeds(9, 5) == spawn_event_seeds(9, 10)[:5]


@pytest.mark.parametrize("base_seed", [-1, True, 1.5])
def test_an_invalid_base_seed_is_refused(base_seed) -> None:
    with pytest.raises(ValueError, match="base_seed"):
        spawn_event_seeds(base_seed, 3)


# --- against the real artifact ---------------------------------------------------


@pytest.mark.skipif(not ARTIFACT.is_dir(), reason="the calibration artifact has not been built")
def test_the_model_loads_from_the_committed_configuration_and_artifact() -> None:
    real = ProtonShowerModel.from_config()

    assert real.calibration.physics_list == "ftfp_bert"
    assert 240.0 < real.effective_length_mm < 270.0
    assert len(real.configuration_sha256) == 64
    seed = next(
        s
        for s in spawn_event_seeds(8, 50)
        if not real.interaction_for_seed(ENERGY_MEV, track(), s).interacts
    )
    event = real.generate_crossing_event(
        event_id="real", primary_energy_mev=ENERGY_MEV, track=track(), random_seed=seed
    )
    assert 5.0 < event.total_ecal_energy_mev < 40.0  # MeV of fibre energy for a crossing proton
