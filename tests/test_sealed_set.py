"""The sealed-set guard: hashing, the final-only reader and the opening ledger."""

import json

import numpy as np
import pytest

from ams_ecal import sealed_set
from ams_ecal.geant4_backend import OUTPUT_SCHEMA_VERSION
from ams_ecal.sealed_set import (
    SealedSetError,
    is_consumed,
    open_set,
    read_ledger,
    seal,
    verify,
)


@pytest.fixture
def sealed(tmp_path):
    data = tmp_path / "data"
    batch = data / "sealed" / "E10GeV"
    batch.mkdir(parents=True)
    np.savez(batch / "events.npz", entry_x_mm=np.linspace(0, 9, 40), event_index=np.arange(40))
    (batch / "metadata.json").write_text(
        json.dumps({"output_schema_version": OUTPUT_SCHEMA_VERSION}), encoding="utf-8"
    )
    ledger_dir = tmp_path / "research_sealed"
    manifest = seal("proton", data, ledger_dir, repo_root=tmp_path)
    return data, ledger_dir, manifest


def opened(sealed, **kwargs):
    data, ledger_dir, _ = sealed
    return open_set("proton", [10.0], data_root=data, directory=ledger_dir, repo_root=data, **kwargs)


def test_sealing_records_a_hash_of_every_file(sealed) -> None:
    _, _, manifest = sealed

    files = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    assert set(files) == {"sealed/E10GeV/events.npz", "sealed/E10GeV/metadata.json"}
    assert all(len(entry["sha256"]) == 64 for entry in files.values())
    assert manifest.name == "proton_manifest.json"


def test_a_set_is_sealed_only_once(sealed) -> None:
    data, ledger_dir, _ = sealed

    with pytest.raises(SealedSetError, match="sealed once"):
        seal("proton", data, ledger_dir, repo_root=data)


def test_the_reader_refuses_without_the_final_flag(sealed) -> None:
    for flag in (False, None, 1):
        with pytest.raises(SealedSetError, match="final"):
            opened(sealed, final=flag, purpose="x")
    assert read_ledger(sealed[1]) == []


def test_the_reader_refuses_an_unstated_purpose(sealed) -> None:
    with pytest.raises(SealedSetError, match="purpose"):
        opened(sealed, final=True, purpose=" ")


def test_an_opening_is_recorded_and_then_the_set_is_consumed(sealed) -> None:
    batches = opened(sealed, final=True, purpose="final validation")

    assert set(batches) == {10.0}
    entries = read_ledger(sealed[1])
    assert len(entries) == 1 and entries[0]["outcome"] == "opened"
    assert is_consumed("proton", sealed[1])
    with pytest.raises(SealedSetError, match="consumed"):
        opened(sealed, final=True, purpose="again")


def test_a_reopening_must_be_explicit_and_is_recorded_as_such(sealed) -> None:
    opened(sealed, final=True, purpose="first")

    opened(sealed, final=True, purpose="second", reopen_consumed=True)

    assert [e["outcome"] for e in read_ledger(sealed[1])] == [
        "opened",
        "opened (set already consumed)",
    ]


def test_a_changed_file_is_detected_before_anything_is_read(sealed) -> None:
    data, ledger_dir, _ = sealed
    path = data / "sealed" / "E10GeV" / "events.npz"
    np.savez(path, entry_x_mm=np.zeros(40), event_index=np.arange(40))

    with pytest.raises(SealedSetError, match="sealed hash"):
        verify("proton", data, ledger_dir)
    with pytest.raises(SealedSetError, match="sealed hash"):
        opened(sealed, final=True, purpose="x")
    assert read_ledger(ledger_dir) == []  # a refused opening leaves no entry


def test_an_added_file_is_detected(sealed) -> None:
    data, ledger_dir, _ = sealed
    (data / "sealed" / "E10GeV" / "extra.txt").write_text("x", encoding="utf-8")

    with pytest.raises(SealedSetError, match="extra"):
        verify("proton", data, ledger_dir)


def test_the_real_sets_are_known() -> None:
    assert set(sealed_set.SETS) == {"proton", "electron"}
