"""The one place that says which Geant4 events belong to which sample, where the files are, and which events may be read.

Why this module exists (review C8, C9, SV-10): three analyses of the multiscale work each defined their own sample. The
eligibility count excluded the proton events with ``event_index % 4 == 3``; the large development run did not; the "all protons"
rows mixed crossing protons into the multiscale question. Every analysis now asks this module for its events.

Three things are fixed here and nowhere else:

* **Where a set lives.** Sets are looked up in an ordered list of data roots (``data/`` of this repository, then the two
  generation worktrees ``../ams-ecal-qml-gen/data`` and ``../ams-ecal-qml-gen2/data``); the environment variable
  ``AMS_ECAL_DATA_ROOTS`` (path-separated) is searched first. Nothing is copied: a set stays where it was generated and its
  identity is its file hashes (``research/data_manifest.json``), not its path.
* **Which events are held out** (``HeldOutPolicy``). The pilot protons keep the quarter ``event_index % 4 == 3`` for the
  proton-model validation. The development-large sets were never used for any calibration (their own seeds), so no events are held
  out of them by default; ``hold_out_quarter_everywhere=True`` applies the pilot rule to them too, as a recorded sensitivity.
* **Which population an analysis reads.** ``interacting`` (truth first interaction occurred), ``crossing`` (it did not) and
  ``all`` protons are separate names; the multiscale question uses ``interacting``, the realistic-performance endpoint ``all``.
  Truth selection is a property of simulation, not of data, so it is only ever a named population, never a hidden filter.

Sealed sets are listed (so the manifest knows them) but ``load_events`` refuses them: their only reader is
``ams_ecal.validation.sealed_set.open_set``.

    uv run python -m ams_ecal.validation.sample_definitions write     # once; a manifest is a record
    uv run python -m ams_ecal.validation.sample_definitions verify
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from ams_ecal.electron_model.em_production import PROJECT_ROOT

REPRESENTATIONS = ("deposit_grid_mev", "readout_grid_mev")
POPULATIONS = ("all", "interacting", "crossing")
MANIFEST_PATH = PROJECT_ROOT / "research" / "data_manifest.json"
SEALED_MANIFEST_DIRECTORY = PROJECT_ROOT / "research" / "sealed"
HELD_OUT_MODULUS, HELD_OUT_REMAINDER = 4, 3
DATA_ROOTS_ENV = "AMS_ECAL_DATA_ROOTS"


class HeldOutPolicy(Enum):
    """Which events of a set an exploratory analysis must not touch."""

    PILOT_QUARTER = "event_index % 4 == 3 is held out for the proton-model validation"
    NONE = "no events are held out: the set was never used to calibrate or validate a model"
    SEALED = "the whole set is sealed; only validation.sealed_set.open_set may read it"


class Role(Enum):
    CALIBRATION = "calibration"  # the FastMC models were fitted on it
    DEVELOPMENT = "development"  # exposed; exploratory analysis and tool development only
    SYSTEMATIC = "systematic"  # exposed variant for a robustness reading
    SEALED = "sealed"


@dataclass(frozen=True)
class SampleSpec:
    name: str
    particle: str  # "electron" | "proton"; the component sets add "pion" and "muon" (validation particles; populations are proton-only)
    role: Role
    directory: str  # relative to a data root; holds ``E<energy>GeV/events.npz`` and ``metadata.json``
    held_out: HeldOutPolicy
    note: str = ""


SAMPLES: dict[str, SampleSpec] = {
    spec.name: spec
    for spec in (
        SampleSpec(
            "electron_pilot",
            "electron",
            Role.CALIBRATION,
            "geant4_electron_sample/baseline",
            HeldOutPolicy.NONE,
            "4 x 1000 electrons; the EM generator was calibrated on it; generated at a1d68d4 with a dirty tracked tree",
        ),
        SampleSpec(
            "proton_pilot",
            "proton",
            Role.CALIBRATION,
            "geant4_proton_pilot/baseline",
            HeldOutPolicy.PILOT_QUARTER,
            "4 x 4000 protons; the proton model was calibrated on event_index % 4 != 3",
        ),
        SampleSpec(
            "electron_development_large",
            "electron",
            Role.DEVELOPMENT,
            "geant4_electron_train/train",
            HeldOutPolicy.NONE,
            "4 x 10,000 electrons, base seed 20272001, commit 48a4a78 (the name 'train' is historical, DECIDE-9)",
        ),
        SampleSpec(
            "proton_development_large",
            "proton",
            Role.DEVELOPMENT,
            "geant4_proton_train/train",
            HeldOutPolicy.NONE,
            "4 x 20,000 protons, base seed 20272002, commit 48a4a78",
        ),
        SampleSpec(
            "electron_cut_0p1mm",
            "electron",
            Role.SYSTEMATIC,
            "geant4_electron_syst_cut0p1mm/cut0p1mm",
            HeldOutPolicy.NONE,
            "2 x 2000 electrons (20, 100 GeV), production cut 0.1 mm, base seed 20272004",
        ),
        SampleSpec(
            "electron_emz",
            "electron",
            Role.SYSTEMATIC,
            "geant4_electron_syst_emz/emz",
            HeldOutPolicy.NONE,
            "2 x 2000 electrons, FTFP_BERT_EMZ (EM option 4), commit 1984f4f",
        ),
        SampleSpec(
            "electron_emv",
            "electron",
            Role.SYSTEMATIC,
            "geant4_electron_syst_emv/emv",
            HeldOutPolicy.NONE,
            "2 x 2000 electrons, FTFP_BERT_EMV (EM option 1), commit 1984f4f; scintillator share anomaly OPEN (review item 7)",
        ),
        SampleSpec(
            "electron_emv_cuts_reverted",
            "electron",
            Role.SYSTEMATIC,
            "geant4_electron_syst_emv_cuts_reverted/emv_cuts_reverted",
            HeldOutPolicy.NONE,
            "1 x 2000 electrons at 20 GeV, FTFP_BERT_EMV with ApplyCuts reverted; same base seed 20272005 as electron_emv (paired)",
        ),
        SampleSpec(
            "electron_emv_msc_reverted",
            "electron",
            Role.SYSTEMATIC,
            "geant4_electron_syst_emv_msc_reverted/emv_msc_reverted",
            HeldOutPolicy.NONE,
            "1 x 2000 electrons at 20 GeV, FTFP_BERT_EMV with MscRangeFactor reverted; same base seed 20272005 as electron_emv (paired)",
        ),
        SampleSpec(
            "electron_emv_both_reverted",
            "electron",
            Role.SYSTEMATIC,
            "geant4_electron_syst_emv_both_reverted/emv_both_reverted",
            HeldOutPolicy.NONE,
            "1 x 2000 electrons at 20 GeV, FTFP_BERT_EMV with both parameters reverted (closure test); same base seed 20272005",
        ),
        SampleSpec(
            "component_electron",
            "electron",
            Role.DEVELOPMENT,
            "geant4_component_electron/train",
            HeldOutPolicy.NONE,
            "6 x 2000 electrons (5-200 GeV) with component tags (component_*_grid_mev, 7 channels); base seed 20273001; generated "
            "with the component-tagging patch uncommitted at the time (metadata tracked_changes true, committed in 11cba53)",
        ),
        SampleSpec(
            "component_proton",
            "proton",
            Role.DEVELOPMENT,
            "geant4_component_proton/train",
            HeldOutPolicy.NONE,
            "6 x 5000 protons (5-200 GeV) with component tags, base seed 20273002; 6 of 300 events differ from the stored untagged "
            "development file at 10 GeV with the same seeds and another worker count (cause unknown, not caused by tagging); "
            "seed 50181583 is also in the 10 GeV event list of component_proton_large (ledgered)",
        ),
        SampleSpec(
            "component_pion",
            "pion",
            Role.DEVELOPMENT,
            "geant4_component_pion/train",
            HeldOutPolicy.NONE,
            "6 x 2000 pi+ (5-200 GeV) with component tags, base seed 20273003; validation particle only (PA.5)",
        ),
        SampleSpec(
            "component_muon",
            "muon",
            Role.DEVELOPMENT,
            "geant4_component_muon/train",
            HeldOutPolicy.NONE,
            "6 x 2000 mu- (5-200 GeV) with component tags, base seed 20273004; validation particle only (PA.5)",
        ),
        SampleSpec(
            "component_proton_qbbc",
            "proton",
            Role.SYSTEMATIC,
            "geant4_component_proton_qbbc/train",
            HeldOutPolicy.NONE,
            "2 x 5000 protons (10, 100 GeV), physics list QBBC, with component tags, base seed 20273005; the physics-list systematic",
        ),
        SampleSpec(
            "component_proton_large",
            "proton",
            Role.DEVELOPMENT,
            "geant4_component_proton_large/train",
            HeldOutPolicy.NONE,
            "4 x 20,000 protons (10, 20, 50, 100 GeV) with component tags, base seed 20274001, run 2026-10-15; 3 events share a seed "
            "with an existing proton set of the same energy and geometry (50181583 component_proton 10 GeV, 2025251122 "
            "proton_development_large 20 GeV, 879865327 proton_pilot 10 GeV): do not pool without removing them",
        ),
        SampleSpec("electron_sealed", "electron", Role.SEALED, "geant4_electron_sealed/sealed", HeldOutPolicy.SEALED),
        SampleSpec("proton_sealed", "proton", Role.SEALED, "geant4_proton_sealed/sealed", HeldOutPolicy.SEALED),
    )
}


class SampleError(RuntimeError):
    """A sample was asked for in a way its definition forbids, or could not be found."""


def data_roots() -> list[Path]:
    """Ordered data roots: the environment variable first, then this repository, then the two generation worktrees."""

    roots = [Path(p) for p in os.environ.get(DATA_ROOTS_ENV, "").split(os.pathsep) if p]
    roots += [
        PROJECT_ROOT / "data",
        PROJECT_ROOT.parent / "ams-ecal-qml-gen" / "data",
        PROJECT_ROOT.parent / "ams-ecal-qml-gen2" / "data",
    ]
    return roots


def locate(name: str, roots: Iterable[Path] | None = None) -> Path:
    """Directory of a registered sample: the first data root that contains it."""

    spec = SAMPLES[name]
    for root in roots if roots is not None else data_roots():
        candidate = root / spec.directory
        if candidate.is_dir():
            return candidate
    raise SampleError(f"sample {name!r} ({spec.directory}) was not found under any data root: {data_roots()}")


def energies_available(name: str, roots: Iterable[Path] | None = None) -> list[float]:
    directory = locate(name, roots)
    return sorted(float(p.name[1:-3]) for p in directory.glob("E*GeV") if (p / "events.npz").is_file())


def held_out_mask(spec: SampleSpec, event_index: np.ndarray, hold_out_quarter_everywhere: bool = False) -> np.ndarray:
    """True for the events an exploratory analysis must NOT read."""

    if spec.held_out is HeldOutPolicy.SEALED:
        raise SampleError(f"{spec.name} is sealed")
    if spec.held_out is HeldOutPolicy.PILOT_QUARTER or hold_out_quarter_everywhere:
        return event_index % HELD_OUT_MODULUS == HELD_OUT_REMAINDER
    return np.zeros(len(event_index), dtype=bool)


def population_mask(spec: SampleSpec, truth_occurred: np.ndarray, population: str) -> np.ndarray:
    """Events of a named population. Electrons have only ``all``; ``interacting`` and ``crossing`` are truth-defined proton classes."""

    if population not in POPULATIONS:
        raise SampleError(f"unknown population {population!r}; choose from {POPULATIONS}")
    if population == "all":
        return np.ones(len(truth_occurred), dtype=bool)
    if spec.particle != "proton":
        raise SampleError(f"population {population!r} is defined for protons only (truth first interaction)")
    occurred = truth_occurred.astype(bool)
    return occurred if population == "interacting" else ~occurred


@dataclass(frozen=True)
class Events:
    """The events an analysis may read, with the rule that selected them."""

    sample: str
    energy_gev: float
    population: str
    representation: str
    held_out_rule: str
    grids: np.ndarray
    event_index: np.ndarray
    seed: np.ndarray
    interacting: np.ndarray  # truth flag, carried for stratification only; never a feature
    edep_total_mev: np.ndarray
    edep_scintillator_mev: np.ndarray
    n_excluded_by_held_out: int

    def __len__(self) -> int:
        return len(self.grids)


def load_events(
    name: str,
    energy_gev: float,
    representation: str = "deposit_grid_mev",
    population: str = "all",
    hold_out_quarter_everywhere: bool = False,
    roots: Iterable[Path] | None = None,
    dtype: Any = np.float64,
) -> Events:
    """Load one energy of one registered sample under its held-out rule and a named population."""

    spec = SAMPLES[name]
    if spec.role is Role.SEALED:
        raise SampleError(f"{name} is sealed: read it only through ams_ecal.validation.sealed_set.open_set")
    if representation not in REPRESENTATIONS:
        raise SampleError(f"unknown representation {representation!r}; choose from {REPRESENTATIONS}")
    path = locate(name, roots) / f"E{energy_gev:g}GeV" / "events.npz"
    if not path.is_file():
        raise SampleError(f"{name} has no events at {energy_gev:g} GeV ({path})")
    arrays = np.load(path)
    event_index = arrays["event_index"]
    hold = held_out_mask(spec, event_index, hold_out_quarter_everywhere)
    keep = ~hold & population_mask(spec, arrays["truth_occurred"], population)
    rule = spec.held_out.name + ("+quarter_everywhere" if hold_out_quarter_everywhere else "")
    return Events(
        sample=name,
        energy_gev=float(energy_gev),
        population=population,
        representation=representation,
        held_out_rule=rule,
        grids=arrays[representation][keep].astype(dtype),
        event_index=event_index[keep],
        seed=arrays["seed"][keep],
        interacting=arrays["truth_occurred"][keep].astype(bool),
        edep_total_mev=arrays["edep_total_mev"][keep],
        edep_scintillator_mev=arrays["edep_scintillator_mev"][keep],
        n_excluded_by_held_out=int(hold.sum()),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_manifest(name: str, roots: Iterable[Path] | None) -> dict[str, Any]:
    spec = SAMPLES[name]
    entry: dict[str, Any] = {
        "particle": spec.particle,
        "role": spec.role.value,
        "directory": spec.directory,
        "held_out_policy": spec.held_out.name,
        "note": spec.note,
    }
    if spec.role is Role.SEALED:
        sealed = SEALED_MANIFEST_DIRECTORY / f"{spec.particle}_manifest.json"
        entry["sealed"] = True
        entry["committed_hash_manifest"] = str(sealed.relative_to(PROJECT_ROOT)).replace("\\", "/")
        entry["committed_hash_manifest_sha256"] = sha256_file(sealed) if sealed.is_file() else None
        entry["files"] = "not listed: the committed sealed manifest is the record; the event files were not read"
        return entry
    directory = locate(name, roots)
    energies: dict[str, Any] = {}
    for energy_dir in sorted(directory.glob("E*GeV"), key=lambda p: float(p.name[1:-3])):
        meta = json.loads((energy_dir / "metadata.json").read_text(encoding="utf-8"))
        events_path = energy_dir / "events.npz"
        seeds = np.load(events_path)["seed"]
        energies[energy_dir.name] = {
            "events_sha256": sha256_file(events_path),
            "events_bytes": events_path.stat().st_size,
            "metadata_sha256": sha256_file(energy_dir / "metadata.json"),
            "n_events": int(meta["n_events"]),
            "base_seed": int(meta["base_seed"]),
            "unique_seeds": len(np.unique(seeds)),
            "physics_list": meta["physics_list"],
            "production_cut_mm": meta["production_cut_mm"],
            "git_commit": meta["git"]["commit"],
            "git_tracked_changes": bool(meta["git"]["tracked_changes"]),
            "configuration_sha256": meta["configuration_sha256"],
            "geant4_version": meta["geant4_version"],
            "created_utc": meta["created_utc"],
        }
    entry["energies"] = energies
    return entry


def build_manifest(names: Iterable[str] | None = None, roots: Iterable[Path] | None = None) -> dict[str, Any]:
    """Hash every exposed set and describe it; sealed sets are referenced through their committed manifests, never read."""

    return {
        "schema_version": 1,
        "purpose": "identity of every Geant4 set by file hash and generation metadata; paths are not identity",
        "samples": {n: _sample_manifest(n, roots) for n in (names or SAMPLES)},
    }


def verify_manifest(manifest_path: Path = MANIFEST_PATH, roots: Iterable[Path] | None = None) -> list[str]:
    """Problems found re-hashing the exposed files against the committed manifest (empty list = every hash matches)."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    for name, entry in manifest["samples"].items():
        if entry.get("sealed"):
            continue
        directory = locate(name, roots)
        for energy_name, recorded in entry["energies"].items():
            if sha256_file(directory / energy_name / "events.npz") != recorded["events_sha256"]:
                problems.append(f"{name}/{energy_name}: events.npz hash differs")
    return problems


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Write, extend or verify the Geant4 data manifest.")
    parser.add_argument("action", choices=("write", "add", "verify"))
    parser.add_argument("names", nargs="*", help="samples to add (action 'add' only)")
    args = parser.parse_args(argv)
    if args.action == "write":
        if MANIFEST_PATH.exists():
            raise SystemExit(f"{MANIFEST_PATH} exists; delete it deliberately to rewrite (a manifest is a record)")
        MANIFEST_PATH.write_text(json.dumps(build_manifest(), indent=2), encoding="utf-8")
        print(f"wrote {MANIFEST_PATH}")
    elif args.action == "add":
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        already = [n for n in args.names if n in manifest["samples"]]
        if not args.names or already:
            raise SystemExit(f"name the samples to add; already recorded (never replaced): {already}")
        manifest["samples"].update(build_manifest(args.names)["samples"])
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"added {args.names} to {MANIFEST_PATH}")
    else:
        problems = verify_manifest()
        print("all hashes match" if not problems else "\n".join(problems))
        raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
