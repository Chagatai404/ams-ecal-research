"""Geant4 backend: pure helpers always, transport only where Geant4 is installed.

The transport tests run the real command-line entry point in a subprocess,
because Geant4 permits a single run manager per process. They test invariants
and interfaces, never particular random values.
"""

import os
import subprocess
import sys
from math import isnan
from pathlib import Path

import numpy as np
import pytest

from ams_ecal.geant4_backend import (
    EntrySpot,
    configure_geant4_data,
    entry_point_mm,
    event_seeds,
    geant4_available,
    load_batch,
    load_pilot_config,
    to_ecal_events,
)
from ams_ecal.geometry import load_geometry

ROOT = Path(__file__).parents[1]
GEOMETRY = ROOT / "configs" / "geometry.yaml"


# --- pure helpers -------------------------------------------------------


def test_event_seeds_are_reproducible_distinct_and_fit_a_c_long() -> None:
    seeds = event_seeds(20260929, 10.0, 500)
    assert seeds == event_seeds(20260929, 10.0, 500)
    assert len(set(seeds)) == 500
    assert all(0 <= s < 2**31 for s in seeds)
    assert set(seeds).isdisjoint(event_seeds(20260929, 20.0, 500))


def test_entry_points_stay_in_the_spot_and_follow_the_seed() -> None:
    spot = EntrySpot(0.0, 9.0, 0.0, 9.0)
    points = [entry_point_mm(s, spot) for s in event_seeds(1, 10.0, 200)]
    assert all(0.0 <= x < 9.0 and 0.0 <= y < 9.0 for x, y in points)
    assert entry_point_mm(123, spot) == entry_point_mm(123, spot)


def test_pilot_configuration_loads() -> None:
    pilot = load_pilot_config()
    assert pilot.particle == "proton"
    assert pilot.energies_gev == (10.0, 20.0, 50.0, 100.0)
    assert pilot.samples["baseline"].physics_list == "FTFP_BERT"
    assert pilot.samples["alternate"].physics_list == "QGSP_BERT"
    assert pilot.samples["extended"].geometry == "extended"


# --- real transport -----------------------------------------------------


def _datasets_present() -> bool:
    if not geant4_available():
        return False
    found = configure_geant4_data()
    return {"G4LEDATA", "G4PARTICLEXSDATA", "G4ENSDFSTATEDATA"} <= set(found)


needs_geant4 = pytest.mark.skipif(
    not _datasets_present(), reason="geant4_pybind or its datasets not installed"
)


def _run(out: Path, *extra: str) -> None:
    env = dict(os.environ)
    completed = subprocess.run(
        [sys.executable, "-m", "ams_ecal.geant4_backend", "run", "--sample", "baseline",
         "--energy", "10", "--out", str(out), *extra],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=900, check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    if not _datasets_present():
        pytest.skip("geant4_pybind or its datasets not installed")
    out = tmp_path_factory.mktemp("g4") / "batch"
    _run(out, "--events", "8", "--workers", "2")
    return load_batch(out)


@needs_geant4
def test_the_executable_writes_a_complete_batch(batch) -> None:
    arrays, meta = batch.arrays, batch.metadata
    assert len(batch) == 8
    assert arrays["fibre_grid_mev"].shape == (8, 18, 72)
    assert arrays["prefix_grid_mev"].shape == (8, 18, 72)
    for key in ("geant4_version", "physics_list", "configuration_sha256", "git",
                "seed_policy", "geant4_datasets", "materials", "output_schema_version"):
        assert key in meta
    assert meta["physics_list"] == "FTFP_BERT"
    assert meta["geant4_version"].startswith("geant4-11")


@needs_geant4
def test_event_identity_and_seeds_are_preserved(batch) -> None:
    pilot = load_pilot_config()
    assert batch.arrays["event_index"].tolist() == list(range(8))
    assert batch.arrays["seed"].tolist() == list(
        event_seeds(pilot.samples["baseline"].base_seed, 10.0, 8)
    )


@needs_geant4
def test_deposits_are_physical_and_consistent(batch) -> None:
    a = batch.arrays
    for key in ("fibre_grid_mev", "prefix_grid_mev"):
        assert np.all(np.isfinite(a[key])) and np.all(a[key] >= 0)
    # Fibres are part of the prefix, and every prefix voxel lies in the readout.
    assert np.all(a["fibre_total_mev"] <= a["prefix_total_mev"] * (1 + 1e-6))
    assert np.allclose(a["prefix_grid_mev"].sum(axis=(1, 2)), a["prefix_total_mev"], rtol=1e-4)
    assert np.all(a["prefix_total_mev"] < 10_000.0)  # never above the primary
    # Sparse fine deposits agree with the dense grids.
    for i in range(len(batch)):
        _, fine = batch.fine_deposits(i)
        _, fibre = batch.fibre_deposits(i)
        assert fine.sum() == pytest.approx(a["prefix_total_mev"][i], rel=1e-4)
        assert fibre.sum() == pytest.approx(a["fibre_total_mev"][i], rel=1e-4, abs=1e-6)


@needs_geant4
def test_interaction_truth_is_consistent(batch) -> None:
    a = batch.arrays
    for occurred, z, process in zip(
        a["truth_occurred"], a["truth_z_mm"], a["truth_process"], strict=True
    ):
        if occurred:
            # The AMS-only geometry has material only in the ECAL.
            assert 0.0 <= z <= 166.5 and process.endswith("Inelastic")
        else:
            assert isnan(z) and process == ""


@needs_geant4
def test_material_depth_matches_the_configured_detector(batch) -> None:
    materials = batch.metadata["materials"]
    assert materials["composite_density_g_cm3"] == pytest.approx(6.8)
    assert materials["prefix_depth_x0"] == pytest.approx(17.0, rel=0.05)
    assert materials["prefix_depth_lambda_i"] == pytest.approx(0.6, rel=0.08)


@needs_geant4
def test_a_recorded_seed_reproduces_its_event(batch, tmp_path) -> None:
    seeds = batch.arrays["seed"][[1, 5]].tolist()
    out = tmp_path / "again"
    _run(out, "--seeds", ",".join(map(str, seeds)))
    again = load_batch(out)
    for mine, original in enumerate((1, 5)):
        assert np.array_equal(
            again.arrays["prefix_grid_mev"][mine], batch.arrays["prefix_grid_mev"][original]
        )
        assert again.arrays["truth_occurred"][mine] == batch.arrays["truth_occurred"][original]


@needs_geant4
def test_batches_become_canonical_ecal_events(batch) -> None:
    events = to_ecal_events(batch, load_geometry(GEOMETRY))
    assert len(events) == 8
    assert all(e.provenance.simulation_backend == "geant4" for e in events)
    assert all(e.particle_type == "proton" for e in events)
    assert events[0].total_ecal_energy_mev == pytest.approx(
        batch.arrays["prefix_total_mev"][0], rel=1e-4
    )
