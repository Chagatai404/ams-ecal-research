"""Seed-collision check for planned Geant4 productions (EXP-011 pilot; reusable for later ones).

Per-event seeds are a pure function of (base seed, energy, count) in ``geant4_backend.event_seeds`` and are 31 bits wide, so a
chance collision between two sets is possible. A collision matters when it is between sets of the same particle, energy and
geometry (identical events in two files); other collisions are only recorded. No event of any set, sealed ones included, is
read: only the committed configs and the seed function are used.

    python -m ams_ecal.geant4_simulation.seed_collision_check
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ams_ecal.geant4_simulation.geant4_backend import (
    PROJECT_ROOT,
    PilotConfigError,
    event_seeds,
    load_pilot_config,
)

PLANNED_PREFIX = "geant4_highenergy_pilot_"
RESULT_PATH = (
    PROJECT_ROOT / "results" / "decision_checks" / "2026-10-22_exp011_seed_check.json"
)


def configured_sets(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    """Every (config, sample, energy) set with its particle, geometry, base seed and seeds; and the configs that did not parse."""
    sets: list[dict[str, Any]] = []
    skipped: list[str] = []
    for path in sorted(paths):
        try:
            config = load_pilot_config(path)
        except PilotConfigError, KeyError, TypeError, ValueError:
            skipped.append(path.name)
            continue
        for sample in config.samples.values():
            for energy in config.energies_gev:
                sets.append(
                    {
                        "config": path.name,
                        "sample": sample.name,
                        "particle": config.particle,
                        "geometry": sample.geometry,
                        "physics_list": sample.physics_list,
                        "energy_gev": energy,
                        "base_seed": sample.base_seed,
                        "seeds": event_seeds(
                            sample.base_seed, energy, sample.events_per_energy
                        ),
                    }
                )
    return sets, skipped


def find_collisions(
    planned: list[dict[str, Any]], existing: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Seeds shared between a planned set and an existing (or another planned) set."""
    index: dict[int, list[dict[str, Any]]] = {}
    for entry in existing + planned:
        for seed in entry["seeds"]:
            index.setdefault(seed, []).append(entry)
    found = []
    for entry in planned:
        for seed in entry["seeds"]:
            for other in index[seed]:
                if other is entry:
                    continue
                found.append(
                    {
                        "seed": seed,
                        "planned": f"{entry['config']}:{entry['sample']}:{entry['energy_gev']:g}",
                        "other": f"{other['config']}:{other['sample']}:{other['energy_gev']:g}",
                        "same_particle_energy_geometry": bool(
                            entry["particle"] == other["particle"]
                            and entry["energy_gev"] == other["energy_gev"]
                            and entry["geometry"] == other["geometry"]
                        ),
                    }
                )
    return found


def run() -> dict[str, Any]:
    paths = sorted((PROJECT_ROOT / "configs").rglob("geant4_*.yaml"))
    sets, skipped = configured_sets(paths)
    planned = [s for s in sets if s["config"].startswith(PLANNED_PREFIX)]
    existing = [s for s in sets if not s["config"].startswith(PLANNED_PREFIX)]
    collisions = find_collisions(planned, existing)
    base_seeds_in_use = sorted({s["base_seed"] for s in existing})
    reused = sorted({s["base_seed"] for s in planned} & set(base_seeds_in_use))
    result = {
        "date": "2026-10-22",
        "n_planned_seeds": sum(len(s["seeds"]) for s in planned),
        "n_existing_seeds": sum(len(s["seeds"]) for s in existing),
        "n_existing_sets": len(existing),
        "configs_not_parsed_as_pilot_configs": skipped,
        "planned_base_seeds": sorted({s["base_seed"] for s in planned}),
        "planned_base_seeds_already_in_use": reused,
        "collisions": collisions,
        "collisions_same_particle_energy_geometry": [
            c for c in collisions if c["same_particle_energy_geometry"]
        ],
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def main() -> None:
    result = run()
    print(
        f"planned seeds {result['n_planned_seeds']}, existing {result['n_existing_seeds']} in {result['n_existing_sets']} sets"
    )
    print(
        "planned base seeds already in use:",
        result["planned_base_seeds_already_in_use"],
    )
    print(
        f"collisions {len(result['collisions'])}, of the same particle/energy/geometry "
        f"{len(result['collisions_same_particle_energy_geometry'])}"
    )
    for c in result["collisions"]:
        print("  ", c)


if __name__ == "__main__":
    main()
