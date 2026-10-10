"""Development reading of the registered statistics and the ladder on the LARGE Geant4 development sample (electrons against protons).

NOT the registered test and NOT a pre-registration change. Data: the development-large Geant4 sets (10,000 electrons and 20,000
protons per energy, exposed, never a validation set); the sealed sets are not read. Two harder conventional rungs are EXPLORATORY,
added after the first development ladder found the registered conventional rung already at AUC 1.000 (plan section 12):
``conv_noE`` drops the total deposited energy, ``conv_noE_nocells`` also drops the occupied-cell count. They show how much
headroom a harder rung leaves; whether to amend the registered rule, and which rung, is the researcher's decision, and the choice
is made before the sealed opening. Every variant is reported (forking paths are disclosed, none is selected here).

Command: ``python -m ams_ecal.multiscale_estimators.multiscale_gtrain_development --energies 10 20 50``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ams_ecal.electron_model.em_production import PROJECT_ROOT
from ams_ecal.multiscale_estimators import multiscale_ladder as ml
from ams_ecal.multiscale_estimators import multiscale_test as mt

DATA_ROOT = PROJECT_ROOT.parent / "ams-ecal-qml-gen" / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "multiscale_controls"
SEED = 20261010
BOOTSTRAPS = 100
SIZES = (100, 300, 1000, 3000)
REPRESENTATIONS = ("deposit_grid_mev", "readout_grid_mev")
RUNGS = {
    "conv": ("conv",),
    "conv+ms": ("conv", "ms"),
    "conv_noE": ("conv_noE",),
    "conv_noE+ms": ("conv_noE", "ms"),
    "conv_noE_nocells": ("conv_noE_nocells",),
    "conv_noE_nocells+ms": ("conv_noE_nocells", "ms"),
    "ms": ("ms",),
}
GAIN_PAIRS = (("conv", "conv+ms"), ("conv_noE", "conv_noE+ms"), ("conv_noE_nocells", "conv_noE_nocells+ms"))
EFFICIENCY_RUNGS = ("conv_noE", "conv_noE+ms", "conv_noE_nocells", "conv_noE_nocells+ms")


def derived_blocks(names: list[str], conventional: np.ndarray) -> dict[str, np.ndarray]:
    """The conventional block and its two exploratory reductions (columns dropped by name)."""

    def without(*drop: str) -> np.ndarray:
        return conventional[:, [i for i, n in enumerate(names) if n not in drop]]

    return {
        "conv": conventional,
        "conv_noE": without("log_total_energy"),
        "conv_noE_nocells": without("log_total_energy", "occupied_cells"),
    }


def load_grids(
    particle: str, energy: float, representation: str, data_root: Path
) -> tuple[np.ndarray, np.ndarray | None]:
    arrays = np.load(data_root / f"geant4_{particle}_train" / "train" / f"E{energy:g}GeV" / "events.npz")
    interacting = arrays["truth_occurred"] if particle == "proton" else None
    return arrays[representation].astype(float), interacting


def cached_table(grids: np.ndarray, cache: Path | None) -> np.ndarray:
    if cache is not None and cache.exists():
        return np.load(cache)
    table = mt.multiscale_table(grids)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, table)
    return table


def one_energy(energy: float, data_root: Path, cache_dir: Path | None, bootstraps: int) -> dict:
    rng = np.random.default_rng(SEED + int(energy))
    row: dict = {}
    for representation in REPRESENTATIONS:
        tag = f"{energy:g}_{representation}"
        electrons, _ = load_grids("electron", energy, representation, data_root)
        protons, interacting = load_grids("proton", energy, representation, data_root)
        tables = {
            "electron": cached_table(electrons, cache_dir / f"electron_{tag}.npy" if cache_dir else None),
            "proton": cached_table(protons, cache_dir / f"proton_{tag}.npy" if cache_dir else None),
        }
        cells = {
            p: (g > 0).reshape(len(g), -1).sum(axis=1).astype(float)
            for p, g in (("electron", electrons), ("proton", protons))
        }
        block: dict = {
            "events": {"electron": len(electrons), "proton": len(protons), "interacting_proton": int(interacting.sum())}
        }
        # (1) the registered statistics, electrons against interacting protons, matched on occupied cells
        block["statistics_vs_interacting_protons"] = {
            name: mt.compare_feature(
                tables["electron"][:, j],
                tables["proton"][interacting, j],
                cells["electron"],
                cells["proton"][interacting],
                rng,
                bootstraps,
            )
            for j, name in enumerate(mt.feature_names())
        }
        # (2) the ladder: the registered rung and the two exploratory harder rungs
        conv_names, _ = mt.conventional_features(electrons[:2])
        for label, selector in (
            ("vs_interacting_protons", interacting),
            ("vs_all_protons", np.ones(len(protons), bool)),
        ):
            pooled = np.concatenate([electrons, protons[selector]])
            labels = np.concatenate([np.ones(len(electrons), int), np.zeros(int(selector.sum()), int)])
            _, conventional = mt.conventional_features(pooled)
            blocks = derived_blocks(conv_names, conventional)
            blocks["ms"] = np.concatenate([tables["electron"], tables["proton"][selector]])
            block[f"ladder_{label}"] = ml.ladder_from_blocks(
                blocks,
                labels,
                RUNGS,
                sizes=SIZES if label == "vs_interacting_protons" else (),
                seed=SEED,
                bootstraps=bootstraps,
                gain_pairs=GAIN_PAIRS,
                efficiency_rungs=EFFICIENCY_RUNGS,
            )
        row[representation] = block
    return row


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--energies", type=float, nargs="+", default=[10.0, 20.0, 50.0])
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    suffix = "_".join(f"{e:g}" for e in args.energies)
    out_path = args.out or RESULTS_DIR / f"multiscale_gtrain_development_{suffix}GeV.json"
    results: dict = {
        "status": "DEVELOPMENT on the large exposed Geant4 sample; not the registered test; exploratory rungs disclosed; "
        "sealed sets not read",
        "sealed_data_read": False,
        "per_energy": {},
    }
    for energy in args.energies:
        results["per_energy"][f"{energy:g}"] = one_energy(energy, args.data_root, args.cache_dir, args.bootstraps)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"{energy:g} GeV done", flush=True)


if __name__ == "__main__":
    main()
