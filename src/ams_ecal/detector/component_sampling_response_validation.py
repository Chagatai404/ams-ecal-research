"""Criteria 1-4 and 6 of the component-wise R-A, exactly as registered in design section 11 (2026-10-15).

Writes ``results/detector_response/component_ra_criteria.json``. Sufficient statistics are cached in
``component_statistics.pkl`` (accumulation reads about 1 GB of git-ignored component files once).
"""

from __future__ import annotations

import json
import pickle
import re
from typing import Any

import numpy as np

from ams_ecal.detector.component_sampling_response import (
    COMPONENTS,
    RatioTable,
    Statistics,
    actual_layers,
    closure,
)
from ams_ecal.validation.sample_definitions import PROJECT_ROOT

TOLERANCE = 0.03
SAMPLES = ("electron", "proton", "pion", "muon", "proton_qbbc")
FIT_PARTS = (0, 2)
TEST_PARTS = (1, 3)
CHUNK = 500
OUT = PROJECT_ROOT / "results" / "detector_response"
CACHE = OUT / "component_statistics.pkl"


def accumulate() -> tuple[dict[tuple[str, float], Statistics], float]:
    stats: dict[tuple[str, float], Statistics] = {}
    worst_check = 0.0
    for sample in SAMPLES:
        root = PROJECT_ROOT / "data" / f"geant4_component_{sample}" / "train"
        folders = sorted(
            root.glob("E*GeV"), key=lambda p: float(re.findall(r"[\d.]+", p.name)[0])
        )
        for folder in folders:
            energy = float(re.findall(r"[\d.]+", folder.name)[0])
            with np.load(folder / "events.npz") as data:
                deposit = data["component_deposit_grid_mev"]
                fibre = data["component_readout_grid_mev"]
                index = data["event_index"]
                total_fibre = data["readout_grid_mev"]
            check = float(np.abs(fibre.sum(axis=1) - total_fibre).max())
            worst_check = max(worst_check, check)
            s = Statistics.zeros()
            for start in range(0, len(index), CHUNK):
                part = slice(start, start + CHUNK)
                s.add_events(deposit[part], fibre[part], index[part])
            stats[(sample, energy)] = s
            print(
                sample,
                energy,
                int(s.n_events.sum()),
                f"component sum check {check:.2e}",
            )
    return stats, worst_check


def load_stats() -> tuple[dict[tuple[str, float], Statistics], float]:
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    OUT.mkdir(parents=True, exist_ok=True)
    result = accumulate()
    CACHE.write_bytes(pickle.dumps(result))
    return result


def _sum(items: list[Statistics]) -> Statistics:
    total = items[0]
    for item in items[1:]:
        total = total + item
    return total


def _row(table: RatioTable, test: Statistics) -> dict[str, Any]:
    c = closure(table, test)
    halves = [
        closure(table, test.parts((p,))) for p in TEST_PARTS if test.n_events[p] > 0
    ]
    return {
        "n_events": int(test.n_events.sum()),
        "layer_closure": c.tolist(),
        "max_abs_layer_closure": float(np.abs(c).max()),
        "worst_layer": int(np.abs(c).argmax()),
        "passes": bool(np.abs(c).max() <= TOLERANCE),
        "half_split_max_abs": [float(np.abs(h).max()) for h in halves],
        "orphan_fraction": float(test.orphan_fibre.sum() / actual_layers(test).sum()),
        "fallback_share_own_pooled_overall": table.fallback_share(test),
    }


def run() -> dict[str, Any]:
    stats, worst_check = load_stats()
    energies = sorted({e for (_, e) in stats})
    out: dict[str, Any] = {
        "registration": "design section 11 (2026-10-15)",
        "tolerance": TOLERANCE,
        "components": list(COMPONENTS),
        "component_sum_check_max_mev": worst_check,
    }

    # criterion 1: fit even-part electrons + protons, test odd parts per particle and energy
    fit1 = _sum(
        [stats[k].parts(FIT_PARTS) for k in stats if k[0] in ("electron", "proton")]
    )
    table1 = RatioTable.fit(fit1)
    out["criterion_1"] = {
        f"{p}_{e:g}": _row(table1, stats[(p, e)].parts(TEST_PARTS))
        for p in ("electron", "proton")
        for e in energies
        if (p, e) in stats
    }
    # criterion 2: fit electrons + protons (all parts), test pions (and muons, reported)
    fit2 = _sum([stats[k] for k in stats if k[0] in ("electron", "proton")])
    table2 = RatioTable.fit(fit2)
    out["criterion_2"] = {
        f"{p}_{e:g}": _row(table2, stats[(p, e)])
        for p in ("pion", "muon")
        for e in energies
        if (p, e) in stats
    }
    # criterion 3: leave one energy out
    out["criterion_3"] = {}
    for held in (20.0, 50.0, 200.0):
        fit3 = _sum(
            [stats[k] for k in stats if k[0] in ("electron", "proton") and k[1] != held]
        )
        table3 = RatioTable.fit(fit3)
        for p in ("electron", "proton", "pion"):
            if (p, held) in stats:
                out["criterion_3"][f"{p}_{held:g}"] = _row(table3, stats[(p, held)])
    # criterion 4: mixed showers: fit without protons, predict protons
    fit4 = _sum([stats[k] for k in stats if k[0] in ("electron", "pion")])
    table4 = RatioTable.fit(fit4)
    out["criterion_4"] = {
        f"proton_{e:g}": _row(table4, stats[("proton", e)]) for e in energies
    }
    # criterion 6: physics list
    out["criterion_6"] = {
        "closure_of_ftfp_model_on_qbbc": {},
        "mean_ratio_by_component": {},
    }
    for e in (10.0, 100.0):
        if ("proton_qbbc", e) in stats:
            out["criterion_6"]["closure_of_ftfp_model_on_qbbc"][f"{e:g}"] = _row(
                table2, stats[("proton_qbbc", e)]
            )
            for name in ("proton", "proton_qbbc"):
                s = stats[(name, e)]
                out["criterion_6"]["mean_ratio_by_component"][f"{name}_{e:g}"] = (
                    s.sum_fibre.sum(axis=(0, 2, 3)) / s.sum_deposit.sum(axis=(0, 2, 3))
                ).tolist()
    (OUT / "component_ra_criteria.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8"
    )
    return out


def main() -> None:
    result = run()
    print("component sum check (max MeV):", result["component_sum_check_max_mev"])
    for name in ("criterion_1", "criterion_2", "criterion_3", "criterion_4"):
        print(name)
        for key, row in result[name].items():
            halves = ["%.3f" % h for h in row["half_split_max_abs"]]
            print(
                f"  {key:14s} max|closure| {row['max_abs_layer_closure']:.4f} "
                f"(layer {row['worst_layer']:2d}) pass={row['passes']} halves={halves}"
            )


if __name__ == "__main__":
    main()
