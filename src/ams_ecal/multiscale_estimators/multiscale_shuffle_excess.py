"""S3 stage 2: the cell-shuffle excess of the 24 multiscale statistics, electrons against interacting protons.

Registration: research/COMPLETED_WORK.md (was 2026-10-15_s3_stage2_registration.md). Per event and statistic: excess = real statistic minus the mean of
the same statistic over 10 energy-preserving shuffled copies of the image, with two surrogates (``full``: every layer's cells
permuted over the 72 positions, the registered A3 null; ``block9``: cells permuted inside the 8 blocks of 9 adjacent cells of each
layer, which keeps the lateral envelope above 9 cells). The class difference of the excess is tested with the stage-1 engine.
Exploratory; exposed development data only; no sealed set.

    python -m ams_ecal.multiscale_estimators.multiscale_shuffle_excess
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from ams_ecal.inference import permutation_nulls as pn
from ams_ecal.multiscale_estimators import multiscale_test as mt
from ams_ecal.multiscale_estimators import multiscale_verdict_reread as vr
from ams_ecal.validation import sample_definitions as sd

SURROGATES = ("full", "block9")
SHUFFLES = 10
BLOCK = 9
BOOTSTRAP_DRAWS = 2000
CHUNK_EVENTS = 500


def shuffle_full(grids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Every layer's cells permuted over all positions of that layer (``permutation_nulls.shuffle_cells_within_layers``)."""

    return pn.shuffle_cells_within_layers(grids, rng)


def shuffle_blocks(grids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cells permuted inside each block of ``BLOCK`` adjacent cells of every layer; the energy of every block is preserved."""

    n, layers, cells = grids.shape
    if cells % BLOCK:
        raise ValueError(f"{cells} cells per layer is not a multiple of the block size {BLOCK}")
    blocks = grids.reshape(n, layers, cells // BLOCK, BLOCK)
    order = np.argsort(rng.random(blocks.shape), axis=-1)
    return np.take_along_axis(blocks, order, axis=-1).reshape(grids.shape)


SURROGATE_FUNCTIONS = {"full": shuffle_full, "block9": shuffle_blocks}


def excess_table(grids: np.ndarray, surrogate: str, shuffles: int, rng: np.random.Generator) -> np.ndarray:
    """``(events, 24)`` excess = real - mean over ``shuffles`` shuffled copies; NaN where the statistic is undefined (ineligible)."""

    grids = np.asarray(grids, dtype=np.float64)
    real = mt.multiscale_table(grids)
    eligible = np.isfinite(real).any(axis=1)
    mean = np.full_like(real, np.nan)
    if eligible.any():
        shuffle = SURROGATE_FUNCTIONS[surrogate]
        accumulated = np.zeros((int(eligible.sum()), real.shape[1]))
        for _ in range(shuffles):
            accumulated += mt.multiscale_table(shuffle(grids[eligible], rng))
        mean[eligible] = accumulated / shuffles
    return real - mean


def _chunk(args: tuple[np.ndarray, str, int, int]) -> np.ndarray:
    grids, surrogate, shuffles, seed = args
    return excess_table(grids, surrogate, shuffles, np.random.default_rng(seed))


def block_excess(
    energy: float, representation: str, surrogate: str, shuffles: int, pool: ProcessPoolExecutor
) -> dict[str, np.ndarray]:
    """Cached event-level excess tables of one block, with the stratification keys of stage 1."""

    vr.CACHE.mkdir(parents=True, exist_ok=True)
    path = vr.CACHE / f"excess_{surrogate}_{energy:g}_{representation}.npz"
    if path.exists():
        return dict(np.load(path))
    out = dict(vr.load_block(energy, representation, None))
    for name, sample, population in (
        ("e", "electron_development_large", "all"),
        ("p", "proton_development_large", "interacting"),
    ):
        grids = sd.load_events(sample, energy, representation, population, dtype=np.float32).grids
        parts = np.array_split(grids, max(1, len(grids) // CHUNK_EVENTS))
        seeds = vr.SEED + 7919 * (1 + int(energy)) + (0 if name == "e" else 104729)
        jobs = [(part, surrogate, shuffles, seeds + k) for k, part in enumerate(parts)]
        out[f"{name}_excess"] = np.vstack(list(pool.map(_chunk, jobs)))
    np.savez(path, **out)
    return out


def median_with_interval(
    values: np.ndarray, rng: np.random.Generator, draws: int = BOOTSTRAP_DRAWS
) -> tuple[float, float, float]:
    """Median of the finite values and a 95% event-bootstrap interval."""

    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return float("nan"), float("nan"), float("nan")
    boots = np.median(finite[rng.integers(0, len(finite), (draws, len(finite)))], axis=1)
    low, high = np.percentile(boots, [2.5, 97.5])
    return float(np.median(finite)), float(low), float(high)


def run_excess_block(args: tuple[float, str, str, int, int]) -> tuple[float, str, dict[str, Any]]:
    energy, representation, surrogate, permutations, seed = args
    data = dict(np.load(vr.CACHE / f"excess_{surrogate}_{energy:g}_{representation}.npz"))
    rng = np.random.default_rng(seed)
    result = vr.block_test(
        data["e_excess"], data["p_excess"], data["e_occupancy"], data["p_occupancy"], permutations, rng
    )
    columns = data["e_excess"].shape[1]
    result["median_excess"] = {
        cls: [median_with_interval(data[f"{cls}_excess"][:, j], rng) for j in range(columns)] for cls in ("e", "p")
    }
    return energy, representation, result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--permutations", type=int, default=vr.N_PERMUTATIONS)
    parser.add_argument("--shuffles", type=int, default=SHUFFLES)
    parser.add_argument("--energies", type=float, nargs="+", default=list(vr.ENERGIES))
    parser.add_argument("--surrogates", nargs="+", default=list(SURROGATES), choices=SURROGATES)
    args = parser.parse_args(argv)
    t0 = time.time()
    with ProcessPoolExecutor(vr.WORKERS) as pool:
        for surrogate in args.surrogates:
            for energy in args.energies:
                for rep in vr.REPRESENTATIONS:
                    block_excess(energy, rep, surrogate, args.shuffles, pool)
                    print(f"excess {surrogate} {energy:g} {rep} {time.time() - t0:.0f}s", flush=True)
    summary: dict[str, Any] = {}
    names = mt.feature_names()
    for surrogate in args.surrogates:
        jobs = [
            (energy, rep, surrogate, args.permutations, vr.SEED + 31 * i + j + (0 if surrogate == "full" else 1000))
            for i, energy in enumerate(args.energies)
            for j, rep in enumerate(vr.REPRESENTATIONS)
        ]
        blocks: dict[tuple[float, str], dict[str, Any]] = {}
        with ProcessPoolExecutor(vr.WORKERS) as pool:
            for energy, rep, result in pool.map(run_excess_block, jobs):
                blocks[(energy, rep)] = result
                print(f"test {surrogate} {energy:g} {rep} {time.time() - t0:.0f}s", flush=True)
        report = vr.assemble(blocks, args.permutations)
        index = {(t["energy_gev"], t["representation"], t["statistic"]): t for t in report["tests"]}
        for (energy, rep), block in blocks.items():
            for j, feature in enumerate(names):
                test = index[(energy, rep, feature)]
                test["median_excess_electron"] = block["median_excess"]["e"][j]
                test["median_excess_proton"] = block["median_excess"]["p"][j]
                if test["testable"]:  # Bonferroni across the four (surrogate, representation) families
                    test["westfall_young_4_families"] = float(min(1.0, 2 * test["westfall_young_192"]))
                    test["holm_4_families"] = float(min(1.0, 2 * test["holm_192"]))
        report["registration"] = "research/COMPLETED_WORK.md (was 2026-10-15_s3_stage2_registration.md)"
        report["surrogate"] = surrogate
        report["shuffles_per_event"] = args.shuffles
        del report["agreement"]
        usable = [t for t in report["tests"] if t["testable"]]

        def excludes_zero(interval: tuple[float, float, float]) -> bool:
            return not (interval[1] <= 0 <= interval[2])

        report["summary"] = {
            "testable": len(usable),
            "class_difference_significant_holm_4_families": sum(t["holm_4_families"] <= vr.ALPHA for t in usable),
            "class_difference_significant_westfall_young_4_families": sum(
                t["westfall_young_4_families"] <= vr.ALPHA for t in usable
            ),
            "electron_median_excess_excludes_zero": sum(excludes_zero(t["median_excess_electron"]) for t in usable),
            "proton_median_excess_excludes_zero": sum(excludes_zero(t["median_excess_proton"]) for t in usable),
        }
        (vr.RESULTS / f"s3_stage2_shuffle_excess_{surrogate}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        summary[surrogate] = report["summary"]
        print(surrogate, json.dumps(report["summary"]), flush=True)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
