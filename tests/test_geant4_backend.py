"""Geant4 backend: pure helpers always, transport only where Geant4 is installed.

The transport tests run the real command-line entry point in subprocesses,
because Geant4 permits a single run manager per process. They test invariants
and interfaces, never particular random values. Four of them are the gates the
researcher set before the baseline production run (2026-09-29):

1. the readout grid holds scintillator energy only, and the energy accounting
   closes;
2. truth comes from the primary's step-defining process and agrees with the
   independent secondary-based finder;
3. the constructed geometry passes Geant4's overlap check and matches its
   analytic fingerprint;
4. an event's output does not depend on which worker, or how many, ran it.
"""

import json
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
    # A smaller sample with the same base seed reuses the leading events.
    assert event_seeds(20260929, 10.0, 50) == seeds[:50]


def test_entry_points_stay_in_the_spot_and_follow_the_seed() -> None:
    spot = EntrySpot(0.0, 9.0, 0.0, 9.0)
    points = [entry_point_mm(s, spot) for s in event_seeds(1, 10.0, 200)]
    assert all(0.0 <= x < 9.0 and 0.0 <= y < 9.0 for x, y in points)
    assert entry_point_mm(123, spot) == entry_point_mm(123, spot)


def test_pilot_configuration_loads() -> None:
    pilot = load_pilot_config()
    assert pilot.particle == "proton"
    assert pilot.energies_gev == (10.0, 20.0, 50.0, 100.0)
    assert pilot.production_cut_mm == 0.7
    assert list(pilot.samples) == ["baseline", "fixed_entry", "alternate", "extended"]
    assert pilot.samples["baseline"].physics_list == "FTFP_BERT"
    assert pilot.samples["alternate"].physics_list == "QBBC"
    assert pilot.samples["extended"].geometry == "extended"


def test_fixed_entry_control_shares_baseline_seeds_at_the_cell_centre() -> None:
    pilot = load_pilot_config()
    fixed, baseline = pilot.samples["fixed_entry"], pilot.samples["baseline"]
    assert fixed.base_seed == baseline.base_seed
    seeds = event_seeds(fixed.base_seed, 50.0, fixed.events_per_energy)
    assert seeds == event_seeds(baseline.base_seed, 50.0, baseline.events_per_energy)[
        : fixed.events_per_energy
    ]
    assert {entry_point_mm(s, fixed.entry_spot) for s in seeds[:20]} == {(4.5, 4.5)}


# --- real transport -----------------------------------------------------


def _datasets_present() -> bool:
    if not geant4_available():
        return False
    found = configure_geant4_data()
    return {"G4LEDATA", "G4PARTICLEXSDATA", "G4ENSDFSTATEDATA"} <= set(found)


needs_geant4 = pytest.mark.skipif(
    not _datasets_present(), reason="geant4_pybind or its datasets not installed"
)


def _backend(*args: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "ams_ecal.geant4_backend", *args],
        cwd=ROOT, capture_output=True, text=True, timeout=1800, check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]


def _run(out: Path, *extra: str) -> None:
    _backend("run", "--sample", "baseline", "--energy", "10", "--out", str(out), *extra)


N_EVENTS = 8


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    if not _datasets_present():
        pytest.skip("geant4_pybind or its datasets not installed")
    out = tmp_path_factory.mktemp("g4") / "batch"
    _run(out, "--events", str(N_EVENTS), "--workers", "2")
    return load_batch(out)


@needs_geant4
def test_the_executable_writes_a_complete_batch(batch) -> None:
    arrays, meta = batch.arrays, batch.metadata
    assert len(batch) == N_EVENTS
    assert arrays["readout_grid_mev"].shape == (N_EVENTS, 18, 72)
    assert arrays["deposit_grid_mev"].shape == (N_EVENTS, 18, 72)
    for key in ("geant4_version", "physics_list", "configuration_sha256", "git",
                "seed_policy", "geant4_datasets", "materials", "truth_definition",
                "geometry_approximations", "output_schema_version"):
        assert key in meta
    assert meta["physics_list"] == "FTFP_BERT"
    assert meta["geant4_version"].startswith("geant4-11")


@needs_geant4
def test_production_cut_and_its_thresholds_are_in_the_provenance(batch) -> None:
    meta = batch.metadata
    assert meta["production_cut_mm"] == 0.7
    assert meta["materials"]["production_cut_mm"] == 0.7
    thresholds = meta["materials"]["production_thresholds_mev"]
    for material in ("matrix", "fibre"):
        for particle in ("gamma", "e-", "e+"):
            assert thresholds[material][particle] > 0.0
    # The same range means a higher energy threshold in the denser matrix.
    assert thresholds["matrix"]["e-"] > thresholds["fibre"]["e-"]


@needs_geant4
def test_event_identity_and_seeds_are_preserved(batch) -> None:
    pilot = load_pilot_config()
    assert batch.arrays["event_index"].tolist() == list(range(N_EVENTS))
    assert batch.arrays["seed"].tolist() == list(
        event_seeds(pilot.samples["baseline"].base_seed, 10.0, N_EVENTS)
    )


# Gate 1 ----------------------------------------------------------------


@needs_geant4
def test_gate1_readout_holds_scintillator_energy_only(batch) -> None:
    a = batch.arrays
    readout = a["readout_grid_mev"].sum(axis=(1, 2))
    assert np.allclose(readout, a["edep_scintillator_mev"], rtol=1e-5, atol=1e-6)
    assert np.all(a["unmapped_scintillator_mev"] == 0.0)
    # Lead and glue deposit most of the energy, and none of it is in the image.
    assert np.all(readout < a["edep_total_mev"])
    assert np.all(a["edep_matrix_mev"] > a["edep_scintillator_mev"])


@needs_geant4
def test_gate1_energy_accounting_closes(batch) -> None:
    a = batch.arrays
    # Fibre and matrix scorers are independent of the all-material mesh.
    assert np.allclose(
        a["edep_scintillator_mev"] + a["edep_matrix_mev"], a["edep_total_mev"], rtol=1e-9
    )
    assert np.allclose(a["deposit_grid_mev"].sum(axis=(1, 2)), a["edep_total_mev"], rtol=1e-4)
    for grid in ("readout_grid_mev", "deposit_grid_mev"):
        assert np.all(np.isfinite(a[grid])) and np.all(a[grid] >= 0)
    assert np.all(a["edep_total_mev"] < 10_000.0)  # never above the primary
    for i in range(len(batch)):
        _, fine = batch.fine_deposits(i)
        _, fibre = batch.fibre_deposits(i)
        assert fine.sum() == pytest.approx(a["edep_total_mev"][i], rel=1e-4)
        assert fibre.sum() == pytest.approx(a["edep_scintillator_mev"][i], rel=1e-4, abs=1e-6)


# Gate 2 ----------------------------------------------------------------


@needs_geant4
def test_gate2_truth_is_the_primarys_hadronic_inelastic_step(batch) -> None:
    a = batch.arrays
    materials = batch.metadata["materials"]
    for i in range(len(batch)):
        if a["truth_occurred"][i]:
            z = a["truth_z_mm"][i]
            assert 0.0 <= z <= 166.5  # the AMS-only geometry has no other material
            assert a["truth_process"][i] == "protonInelastic"
            assert a["truth_process_type"][i] == "fHadronic"
            assert a["truth_material"][i] in {"ECAL_matrix", "G4_POLYSTYRENE"}
            assert 0.0 < a["truth_primary_kinetic_energy_before_mev"][i] <= 10_000.0
            assert a["truth_depth_x0"][i] == pytest.approx(
                z / materials["composite_radiation_length_mm"]
            )
            assert a["truth_depth_lambda_i"][i] == pytest.approx(
                z / materials["composite_nuclear_interaction_length_mm"]
            )
            pdg, _ = batch.first_secondaries(i)
            assert len(pdg) == a["truth_n_secondaries"][i] > 0
        else:
            assert isnan(a["truth_z_mm"][i]) and a["truth_process"][i] == ""
            assert len(batch.first_secondaries(i)[0]) == 0


@needs_geant4
def test_gate2_both_truth_finders_agree(batch) -> None:
    a = batch.arrays
    assert np.array_equal(a["truth_occurred"], a["xcheck_occurred"])
    both = a["truth_occurred"]
    assert np.allclose(a["truth_z_mm"][both], a["xcheck_z_mm"][both], rtol=0, atol=1e-6)
    # The step finder also counts products Geant4 creates at a later time
    # (delayed nuclear de-excitation), which the time-grouped finder misses.
    assert np.all(a["truth_n_secondaries"][both] >= a["xcheck_n_secondaries"][both])
    assert np.all(a["xcheck_n_other_hadronic"] == 0)


# Gate 3 ----------------------------------------------------------------


@needs_geant4
def test_gate3_geometry_passes_the_overlap_check_and_matches_its_fingerprint(tmp_path) -> None:
    out = tmp_path / "audit.json"
    _backend("audit", "--variant", "ams", "--resolution", "50", "--out", str(out))
    report = json.loads(out.read_text(encoding="utf-8"))
    check = report["overlap_check"]
    assert check["volumes_with_overlaps"] == {}
    assert check["volumes_checked"]["Superlayer"] == 9
    assert check["volumes_checked"]["Fibre_x"] == check["volumes_checked"]["Fibre_y"] == 4795
    assert report["fibres_in_ams_prefix"] == 43_155
    assert report["effective_density_g_cm3"] == pytest.approx(6.8, rel=1e-6)
    assert report["geant4_ecal_mass_kg"] == pytest.approx(report["total_mass_kg"], rel=1e-6)
    assert sum(report["volume_fractions"].values()) == pytest.approx(1.0)
    assert report["depth_x0"] == pytest.approx(17.0, rel=0.05)
    assert report["depth_lambda_i"] == pytest.approx(0.6, rel=0.08)
    assert any("98 lead foils" in text for text in report["approximations"])


# Gate 4 ----------------------------------------------------------------


_SPARSE = {"fibre_ids", "fibre_edep_mev", "fine_index", "fine_edep_mev",
           "first_secondary_pdg", "first_secondary_ke_mev"}


@needs_geant4
def test_gate4_output_does_not_depend_on_worker_count_or_order(batch, tmp_path) -> None:
    seeds = batch.arrays["seed"][:6].tolist()
    single = tmp_path / "one_worker"
    _run(single, "--seeds", ",".join(map(str, seeds)), "--workers", "1")
    one = load_batch(single)

    # The fixture ran the same six events among eight, split over two workers.
    for key, values in one.arrays.items():
        if key == "cpu_seconds" or key.endswith("_offsets") or key in _SPARSE:
            continue
        assert np.array_equal(
            values, batch.arrays[key][:6], equal_nan=values.dtype.kind == "f"
        ), key
    for i in range(6):
        for getter in ("fine_deposits", "fibre_deposits", "first_secondaries"):
            for mine, theirs in zip(
                getattr(one, getter)(i), getattr(batch, getter)(i), strict=True
            ):
                assert np.array_equal(mine, theirs), (getter, i)

    # One event on its own reproduces bit for bit.
    alone = tmp_path / "alone"
    _run(alone, "--seeds", str(seeds[4]))
    again = load_batch(alone).arrays
    assert np.array_equal(again["readout_grid_mev"][0], batch.arrays["readout_grid_mev"][4])
    assert np.array_equal(again["deposit_grid_mev"][0], batch.arrays["deposit_grid_mev"][4])
    assert again["truth_occurred"][0] == batch.arrays["truth_occurred"][4]


# Interfaces ------------------------------------------------------------


@needs_geant4
def test_batches_become_canonical_ecal_events_from_the_readout(batch) -> None:
    events = to_ecal_events(batch, load_geometry(GEOMETRY))
    assert len(events) == N_EVENTS
    assert all(e.provenance.simulation_backend == "geant4" for e in events)
    assert all(e.particle_type == "proton" for e in events)
    assert events[0].total_ecal_energy_mev == pytest.approx(
        batch.arrays["edep_scintillator_mev"][0], rel=1e-4
    )
