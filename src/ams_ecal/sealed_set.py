"""Sealed Geant4 validation sets: hash manifest, guarded reader, opening ledger.

A sealed set is generated once, hashed, and never looked at until the final validation. The
rules (``research/plans/2026-10-02_added_validation_checks_preregistration.md``, rule 2) are
enforced here rather than by convention:

* ``seal`` records the SHA-256 and size of every file of a set in a manifest that is
  COMMITTED (``research/sealed/<set>_manifest.json``). Anyone can later check that no file
  changed.
* ``open_set`` is the ONLY reader. It refuses unless ``final=True`` is passed, verifies every
  hash against the committed manifest, and appends one line to the committed opening ledger
  (``research/sealed/opening_ledger.jsonl``) BEFORE returning data. A set that already has an
  opening in the ledger is CONSUMED: a further opening raises unless ``reopen_consumed=True``,
  which is then recorded as such. A threshold changed after a result was seen is a new
  registration and the set counts as consumed.
* Nothing else in the repository may read a sealed directory; the harness dry run uses
  stand-ins (pilot calibration events), never these files.

    uv run python -m ams_ecal.sealed_set seal proton
    uv run python -m ams_ecal.sealed_set status
"""

import argparse
import hashlib
import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ams_ecal.geant4_backend import PROJECT_ROOT, Batch, load_batch

SEALED_DIRECTORY = PROJECT_ROOT / "research" / "sealed"
LEDGER = SEALED_DIRECTORY / "opening_ledger.jsonl"
SETS = {
    "proton": PROJECT_ROOT / "data" / "geant4_proton_sealed",
    "electron": PROJECT_ROOT / "data" / "geant4_electron_sealed",
}
SAMPLE = "sealed"


class SealedSetError(RuntimeError):
    """A sealed set was asked to do something its rules forbid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def manifest_path(name: str, directory: Path = SEALED_DIRECTORY) -> Path:
    return directory / f"{name}_manifest.json"


def seal(
    name: str,
    data_root: Path | None = None,
    directory: Path = SEALED_DIRECTORY,
    repo_root: Path = PROJECT_ROOT,
) -> Path:
    """Hash every file of a generated set and write the manifest. Refuses to overwrite one."""

    data_root = data_root or SETS[name]
    target = manifest_path(name, directory)
    if target.exists():
        raise SealedSetError(f"{target} exists: a sealed set is sealed once")
    files = sorted(p for p in data_root.rglob("*") if p.is_file())
    if not files:
        raise SealedSetError(f"no files under {data_root}")
    entries = {
        p.relative_to(data_root).as_posix(): {"sha256": _sha256(p), "bytes": p.stat().st_size}
        for p in files
    }
    directory.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {"set": name, "created_utc": _utc(), "git_commit": _git_commit(repo_root), "files": entries},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return target


def read_ledger(directory: Path = SEALED_DIRECTORY) -> list[dict[str, Any]]:
    path = directory / LEDGER.name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_consumed(name: str, directory: Path = SEALED_DIRECTORY) -> bool:
    return any(e["set"] == name and e["outcome"].startswith("opened") for e in read_ledger(directory))


def verify(name: str, data_root: Path | None = None, directory: Path = SEALED_DIRECTORY) -> None:
    """Raise unless every file matches the committed manifest, with no file added or missing."""

    data_root = data_root or SETS[name]
    manifest = json.loads(manifest_path(name, directory).read_text(encoding="utf-8"))
    expected = manifest["files"]
    present = {p.relative_to(data_root).as_posix() for p in data_root.rglob("*") if p.is_file()}
    if present != set(expected):
        raise SealedSetError(
            f"{name}: files differ from the manifest (missing {sorted(set(expected) - present)}, "
            f"extra {sorted(present - set(expected))})"
        )
    for relative, entry in expected.items():
        if _sha256(data_root / relative) != entry["sha256"]:
            raise SealedSetError(f"{name}: {relative} does not match its sealed hash")


def open_set(
    name: str,
    energies_gev: Sequence[float],
    *,
    final: bool,
    purpose: str,
    data_root: Path | None = None,
    directory: Path = SEALED_DIRECTORY,
    repo_root: Path = PROJECT_ROOT,
    reopen_consumed: bool = False,
) -> dict[float, Batch]:
    """The only reader of a sealed set. See the module docstring for the rules."""

    if final is not True:
        raise SealedSetError("a sealed set is opened only for the final validation (final=True)")
    if not purpose.strip():
        raise SealedSetError("state the purpose of the opening")
    data_root = data_root or SETS[name]
    verify(name, data_root, directory)
    consumed = is_consumed(name, directory)
    if consumed and not reopen_consumed:
        raise SealedSetError(f"{name} was already opened (see {LEDGER.name}); it is consumed")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / LEDGER.name).open("a", encoding="utf-8") as handle:  # recorded BEFORE any read
        handle.write(
            json.dumps(
                {
                    "utc": _utc(),
                    "set": name,
                    "purpose": purpose,
                    "git_commit": _git_commit(repo_root),
                    "outcome": "opened (set already consumed)" if consumed else "opened",
                },
                sort_keys=True,
            )
            + "\n"
        )
    return {
        float(energy): load_batch(data_root / SAMPLE / f"E{float(energy):g}GeV")
        for energy in energies_gev
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sealer = sub.add_parser("seal", help="hash a generated set and write its manifest")
    sealer.add_argument("name", choices=sorted(SETS))
    sub.add_parser("status", help="show manifests and the opening ledger")
    args = parser.parse_args(argv)
    if args.command == "seal":
        print(seal(args.name))
    else:
        for name in sorted(SETS):
            sealed = manifest_path(name).exists()
            print(f"{name}: sealed={sealed} consumed={is_consumed(name)}")
        for entry in read_ledger():
            print(entry)


if __name__ == "__main__":
    main()
