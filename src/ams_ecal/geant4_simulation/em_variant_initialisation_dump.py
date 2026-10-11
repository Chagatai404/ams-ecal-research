"""Dump what Geant4 sets up for an EM physics list at initialisation, with NO events (slice S0, review item 7 and C15).

Why. With ``FTFP_BERT_EMV`` (EM option 1) the scintillator share of the deposited energy of 20 GeV electrons is 4.13% against
6.53% for FTFP_BERT, while the total deposit is identical. The configurations differ only in the EM constructor, and the rebuilt
standard list reproduces FTFP_BERT event for event, so the suspect is something option 1 sets: an EM parameter (step function,
multiple-scattering range factor, fluctuation or lowest-energy switches) or a different process/model assignment. This tool
records both for one list, so the three lists can be compared without running a single event.

Geant4 allows one run manager per process: run it once per list.

    python -m ams_ecal.geant4_simulation.em_variant_initialisation_dump dump --config configs/variants/geant4_electron_syst_emv.yaml --out results/em_variants/emv.json

It decides nothing. Whether an option-1 setting can move the sampling fraction by 37% is an open physics question (no source in hand);
swapping settings one at a time on events needs its own approval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ams_ecal.detector.geometry import load_geometry
from ams_ecal.geant4_simulation import geant4_backend as backend

PARTICLES = ("e-", "e+", "gamma")
SCALAR_TYPES = (bool, int, float, str)
SETTER_PREFIXES = ("Set", "Dump", "Print", "Add", "Remove", "Clear", "Reset", "Define", "Lock")


def scalar_getters(obj: Any) -> dict[str, Any]:
    """Every zero-argument method of ``obj`` that returns a plain scalar, called once (setters and printers are never called).

    Methods that raise, need arguments or return objects are listed under ``"_skipped"``, so nothing is hidden.
    """

    values: dict[str, Any] = {}
    skipped: list[str] = []
    for name in sorted(dir(obj)):
        if name.startswith("_") or name.startswith(SETTER_PREFIXES):
            continue
        attribute = getattr(obj, name, None)
        if not callable(attribute):
            continue
        try:
            result = attribute()
        except Exception:  # noqa: BLE001 - pybind raises assorted types for methods that need arguments; each is listed as skipped
            skipped.append(name)
            continue
        if isinstance(result, SCALAR_TYPES):
            values[name] = result
        else:
            skipped.append(name)
    values["_skipped"] = skipped
    return values


def process_names(particle_table: Any, particle: str) -> list[str]:
    processes = particle_table.FindParticle(particle).GetProcessManager().GetProcessList()  # a Python list in geant4_pybind
    return sorted(str(process.GetProcessName()) for process in processes)


def dump(config: Path, sample: str | None = None) -> dict[str, Any]:
    """Build the transport of one sample of a pilot-format configuration (zero events) and describe its EM setup."""

    pilot = backend.load_pilot_config(config)
    spec = pilot.samples[sample] if sample else next(iter(pilot.samples.values()))
    geometry = load_geometry(backend.GEOMETRY_CONFIG)
    transport = backend.load_transport_config(backend.TRANSPORT_CONFIG)
    app = backend.ECALTransport(**backend._setup(pilot, spec, geometry, transport))
    g4 = app.g4
    return {
        "physics_list": spec.physics_list,
        "production_cut_mm": pilot.production_cut_mm,
        "geant4_version": g4.G4Version,
        "em_parameters": scalar_getters(g4.G4EmParameters.Instance()),
        "processes": {p: process_names(g4.G4ParticleTable.GetParticleTable(), p) for p in PARTICLES},
        "events_run": 0,
    }


def differences(reference: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """EM parameters and processes that differ between two dumps (a pure function, so it is testable without Geant4)."""

    def changed(a: dict, b: dict) -> dict[str, list]:
        return {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if k != "_skipped" and a.get(k) != b.get(k)}

    return {
        "em_parameters": changed(reference["em_parameters"], other["em_parameters"]),
        "processes": {
            p: {
                "only_in_reference": sorted(set(reference["processes"][p]) - set(other["processes"][p])),
                "only_in_other": sorted(set(other["processes"][p]) - set(reference["processes"][p])),
            }
            for p in reference["processes"]
        },
    }


NOISE_PREFIXES = ("<<< Geant4 Physics List", "Max 2J", "### Birks", "G4_POLYSTYRENE", "====")


def printed_table_lines(stdout_text: str) -> list[str]:
    """The informative lines of the table Geant4 itself prints at initialisation (EM parameters, per-process model settings).

    The ``dump`` action cannot read the msc step-limit type, the energy-loss step function, the fluctuation model or the combined
    transportation switch: the pybind build returns neither the enum nor the tuple, so ``scalar_getters`` lists them as skipped.
    Geant4 prints all of them, so the saved standard output of two ``dump`` runs is the complete record to compare.
    """

    keep = []
    for raw in stdout_text.splitlines():
        line = raw.rstrip()
        if line.strip() and not line.lstrip().startswith(NOISE_PREFIXES):
            keep.append(line)
    return keep


def printed_table_differences(reference_text: str, other_text: str) -> dict[str, list[str]]:
    """Lines only in the reference and lines only in the other (order-insensitive; repeated lines are compared as sets)."""

    a, b = set(printed_table_lines(reference_text)), set(printed_table_lines(other_text))
    return {"only_in_reference": sorted(a - b), "only_in_other": sorted(b - a)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)
    logs = sub.add_parser("compare-logs", help="differences between the printed tables of two saved dump runs (standard output)")
    logs.add_argument("reference", type=Path)
    logs.add_argument("other", type=Path)
    one = sub.add_parser("dump", help="describe one list (needs the geant4 group)")
    one.add_argument("--config", type=Path, required=True)
    one.add_argument("--sample", default=None)
    one.add_argument("--out", type=Path, required=True)
    both = sub.add_parser("compare", help="differences between two dumps")
    both.add_argument("reference", type=Path)
    both.add_argument("other", type=Path)
    args = parser.parse_args(argv)
    if args.action == "dump":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dump(args.config, args.sample), indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    elif args.action == "compare-logs":
        read_text = lambda p: p.read_text(encoding="utf-8", errors="replace")
        print(json.dumps(printed_table_differences(read_text(args.reference), read_text(args.other)), indent=2))
    else:
        read = lambda p: json.loads(p.read_text(encoding="utf-8"))
        print(json.dumps(differences(read(args.reference), read(args.other)), indent=2))


if __name__ == "__main__":
    main()
