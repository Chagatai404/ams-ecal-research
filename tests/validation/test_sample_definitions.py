"""The sample definitions: held-out rules, named populations, the sealed refusal and the manifest, on synthetic files only."""

import json

import numpy as np
import pytest

from ams_ecal.validation import sample_definitions as sd


def write_set(root, spec_name, energy=10.0, n=40, interacting_every=2):
    spec = sd.SAMPLES[spec_name]
    directory = root / spec.directory / f"E{energy:g}GeV"
    directory.mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.savez(
        directory / "events.npz",
        event_index=np.arange(n),
        seed=np.arange(n) + 1000,
        truth_occurred=(np.arange(n) % interacting_every == 0),
        deposit_grid_mev=rng.random((n, 18, 72)).astype(np.float32),
        readout_grid_mev=rng.random((n, 18, 72)).astype(np.float32),
        edep_total_mev=rng.random(n) + 1.0,
        edep_scintillator_mev=rng.random(n) * 0.1,
    )
    meta = {
        "n_events": n,
        "base_seed": 7,
        "physics_list": "FTFP_BERT",
        "production_cut_mm": 0.7,
        "git": {"commit": "abc", "tracked_changes": False},
        "configuration_sha256": "0" * 64,
        "geant4_version": "test",
        "created_utc": "2026-10-11T00:00:00Z",
    }
    (directory / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return directory


def test_pilot_protons_hold_out_the_quarter_and_development_sets_hold_out_nothing(tmp_path):
    write_set(tmp_path, "proton_pilot")
    write_set(tmp_path, "proton_development_large")
    pilot = sd.load_events("proton_pilot", 10.0, roots=[tmp_path])
    large = sd.load_events("proton_development_large", 10.0, roots=[tmp_path])
    assert not np.any(pilot.event_index % 4 == 3)
    assert pilot.n_excluded_by_held_out == 10
    assert len(large) == 40 and large.n_excluded_by_held_out == 0


def test_the_quarter_everywhere_option_applies_the_pilot_rule_to_a_development_set(tmp_path):
    write_set(tmp_path, "proton_development_large")
    events = sd.load_events("proton_development_large", 10.0, hold_out_quarter_everywhere=True, roots=[tmp_path])
    assert not np.any(events.event_index % 4 == 3)
    assert events.held_out_rule.endswith("+quarter_everywhere")


def test_populations_are_named_and_disjoint(tmp_path):
    write_set(tmp_path, "proton_development_large")
    kw = {"roots": [tmp_path]}
    interacting = sd.load_events("proton_development_large", 10.0, population="interacting", **kw)
    crossing = sd.load_events("proton_development_large", 10.0, population="crossing", **kw)
    everyone = sd.load_events("proton_development_large", 10.0, population="all", **kw)
    assert interacting.interacting.all() and not crossing.interacting.any()
    assert len(interacting) + len(crossing) == len(everyone)
    assert not set(interacting.event_index) & set(crossing.event_index)


def test_truth_populations_are_a_proton_notion(tmp_path):
    write_set(tmp_path, "electron_development_large")
    with pytest.raises(sd.SampleError, match="protons only"):
        sd.load_events("electron_development_large", 10.0, population="interacting", roots=[tmp_path])


def test_a_sealed_set_cannot_be_loaded_here(tmp_path):
    for name in ("proton_sealed", "electron_sealed"):
        with pytest.raises(sd.SampleError, match="sealed"):
            sd.load_events(name, 10.0, roots=[tmp_path])


def test_unknown_population_and_representation_are_refused(tmp_path):
    write_set(tmp_path, "proton_development_large")
    with pytest.raises(sd.SampleError):
        sd.load_events("proton_development_large", 10.0, population="hard", roots=[tmp_path])
    with pytest.raises(sd.SampleError):
        sd.load_events("proton_development_large", 10.0, representation="fine", roots=[tmp_path])


def test_the_first_data_root_that_holds_the_set_wins(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    write_set(second, "electron_development_large")
    assert sd.locate("electron_development_large", [first, second]).is_relative_to(second)
    write_set(first, "electron_development_large")
    assert sd.locate("electron_development_large", [first, second]).is_relative_to(first)
    with pytest.raises(sd.SampleError, match="not found"):
        sd.locate("electron_pilot", [first, second])


def test_manifest_hashes_exposed_files_and_detects_a_change(tmp_path):
    directory = write_set(tmp_path, "electron_development_large", n=20)
    manifest = sd.build_manifest(["electron_development_large"], roots=[tmp_path])
    entry = manifest["samples"]["electron_development_large"]["energies"]["E10GeV"]
    assert entry["n_events"] == 20 and entry["unique_seeds"] == 20
    assert entry["events_sha256"] == sd.sha256_file(directory / "events.npz")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert sd.verify_manifest(path, roots=[tmp_path]) == []
    with (directory / "events.npz").open("ab") as handle:
        handle.write(b"x")
    assert sd.verify_manifest(path, roots=[tmp_path]) == ["electron_development_large/E10GeV: events.npz hash differs"]


def test_add_extends_the_manifest_and_never_replaces_an_entry(tmp_path, monkeypatch):
    write_set(tmp_path, "electron_emv_cuts_reverted", n=10)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": 1, "samples": {"older_entry": {"x": 1}}}), encoding="utf-8")
    monkeypatch.setattr(sd, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(sd, "data_roots", lambda: [tmp_path])
    sd.main(["add", "electron_emv_cuts_reverted"])
    samples = json.loads(manifest_path.read_text(encoding="utf-8"))["samples"]
    assert samples["older_entry"] == {"x": 1} and "E10GeV" in samples["electron_emv_cuts_reverted"]["energies"]
    with pytest.raises(SystemExit):  # a recorded sample is never replaced
        sd.main(["add", "electron_emv_cuts_reverted"])


def test_the_manifest_refers_to_sealed_sets_without_reading_their_events():
    entry = sd._sample_manifest("proton_sealed", roots=[])  # no data root at all: nothing may be opened
    assert entry["sealed"] is True and "energies" not in entry
    assert entry["committed_hash_manifest"].endswith("research/sealed/proton_manifest.json")
