"""EXP-010: sensitivity check of the 3D estimator controls (option b), synthetic only.

Registered before this code was run (2026-10-22, hash 5ec9ca672a04f1c0; text in ``~/.research-os-backups/registrations``).
The registered module ``multiscale_3d.py`` and its NEGATIVE verdict are unchanged and stay primary; this module only adds

* a 3D saturation rule: scale index k = log2(boxes at that scale) / 3, saturation level k* = log2(n_eff / a) / d_f with
  a = 1 and d_f = 3; a window is reliable when the mean k over its scales is at most k*; saturated windows are DROPPED
  and the ordering clause is never waived;
* three baselines (B1 the original separable map, B2 the Poisson-sampled ensemble-mean profile of the even-index 100 GeV
  development electrons, B3 a broader analytic map), a window being usable only if usable under all three;
* two positive controls: the original equal-hits cascade and a cascade whose hit number is chosen by bisection so its
  median occupied voxels equal the baseline's. A window needs power, and ordering, at the registered thresholds under both;
* the ordering diagnostic of step 0 (fractions of replicates with D0 < D1 and D1 < D2), the finer hit grid and a rerun of
  the original configuration with the new seed as noise reference.

    python -m ams_ecal.multiscale_estimators.multiscale_3d_sensitivity --rung 1
    python -m ams_ecal.multiscale_estimators.multiscale_3d_sensitivity --rung 2
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from math import log2
from typing import Any

import numpy as np

from ams_ecal.electron_model.em_production import PROJECT_ROOT
from ams_ecal.multiscale_estimators import multiscale_3d as m3
from ams_ecal.validation import sample_definitions as sd

SEED = 20261023
HIT_COUNTS = (100, 300, 1000, 3000, 10_000, 15_000, 20_000, 30_000)
A_SAT = 1.0
D_F = 3.0
PILOT_REPLICATES = 50
MATCH_HIT_FACTOR = 30  # upper end of the bisection for the occupancy-matched cascade (implementation detail)
B2_SAMPLE = "electron_development_large"
B2_ENERGY = 100.0
BASELINE_IDS = {"B1": 1, "B2": 2, "B3": 3}
RESULT_DIR = PROJECT_ROOT / "results" / "multiscale_controls"
MAP_PATH = RESULT_DIR / "exp010_b2_mean_map.npy"
WORKERS = 6

K_SCALE = np.array(
    [
        log2((m3.MESH[0] // kx) * (m3.MESH[1] // kx) * (m3.MESH[2] // kz)) / 3
        for kx, kz in m3.SCALES
    ]
)


# --- the saturation rule -----------------------------------------------------------------------------------------------------


def saturation_level(n_eff: float) -> float:
    """k* = log2(n_eff / a) / d_f."""
    return log2(n_eff / A_SAT) / D_F


def reliable_windows(n_eff: float) -> list[bool]:
    """Per window of ``multiscale_3d.WINDOWS``: mean scale index at most k*."""
    k_star = saturation_level(n_eff)
    return [bool(np.mean([K_SCALE[i] for i in w]) <= k_star) for w in m3.WINDOWS]


# --- the baselines ------------------------------------------------------------------------------------------------------------


def separable_sampler(lateral_sd: float, depth_tau: float):
    """Hits of a separable map: Gaussian lateral core with a 0.02 floor times r^2 exp(-r / tau) in depth (B1: 9, 6; B3: 18, 9)."""
    c = np.arange(m3.MESH[0], dtype=float)
    lateral = m3._cdf(
        0.02 + np.exp(-0.5 * ((c - (m3.MESH[0] - 1) / 2) / lateral_sd) ** 2)
    )
    r = np.arange(1, m3.MESH[2] + 1, dtype=float)
    depth = m3._cdf(r**2 * np.exp(-r / depth_tau))

    def sample(n_hits: int, rng: np.random.Generator):
        return (
            np.searchsorted(lateral, rng.random(n_hits)),
            np.searchsorted(lateral, rng.random(n_hits)),
            np.searchsorted(depth, rng.random(n_hits)),
        )

    return sample


def build_mean_map() -> np.ndarray:
    """Mean energy per fine voxel over the even ``event_index`` events of the 100 GeV development electrons."""
    path = sd.locate(B2_SAMPLE) / f"E{B2_ENERGY:g}GeV" / "events.npz"
    with np.load(path) as data:
        offsets = data["fine_offsets"]
        index = data["fine_index"]
        edep = data["fine_edep_mev"]
        event_index = data["event_index"]
    even_event = event_index % 2 == 0
    keep = np.repeat(even_event, np.diff(offsets))
    flat = index[keep].astype(np.int64)
    if flat.max() >= int(np.prod(m3.MESH)):
        raise ValueError("fine_index exceeds the 216 x 216 x 36 mesh")
    total = np.bincount(
        flat, weights=edep[keep].astype(np.float64), minlength=int(np.prod(m3.MESH))
    )
    return total / int(even_event.sum())


def mean_map() -> np.ndarray:
    if MAP_PATH.is_file():
        return np.load(MAP_PATH)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result = build_mean_map()
    np.save(MAP_PATH, result)
    return result


def mean_map_sampler(energy_map: np.ndarray):
    """Hits drawn with probability proportional to the ensemble-mean energy of each voxel (equal quanta)."""
    cdf = m3._cdf(energy_map)

    def sample(n_hits: int, rng: np.random.Generator):
        return m3.unravel(np.searchsorted(cdf, rng.random(n_hits)))

    return sample


def baseline_sampler(name: str):
    if name == "B1":
        return m3.sample_smooth
    if name == "B2":
        return mean_map_sampler(mean_map())
    if name == "B3":
        return separable_sampler(18.0, 9.0)
    raise ValueError(f"unknown baseline {name!r}")


# --- the occupancy-matched cascade ---------------------------------------------------------------------------------------------


def match_cascade_hits(
    target_occupancy: float, n_hits: int, seed: np.random.SeedSequence
) -> tuple[int, bool]:
    """Hit number of the cascade whose median occupied voxels equal ``target_occupancy`` (bisection on pilot replicates).

    The search is two-sided: the cascade may be more or less occupied than the baseline at equal hits (at 30,000 hits
    it is more). Returns ``(hits, matched)``; ``matched`` is False when the target lies outside the range from
    ``1 / MATCH_HIT_FACTOR`` to ``MATCH_HIT_FACTOR`` times the baseline hits.
    """
    rng = np.random.default_rng(seed)

    def occupancy(n: int) -> float:
        return float(
            np.median(
                [
                    len(m3._event_from_hits(*m3.sample_cascade(n, rng))[3])
                    for _ in range(PILOT_REPLICATES)
                ]
            )
        )

    low, high = max(10, n_hits // MATCH_HIT_FACTOR), MATCH_HIT_FACTOR * n_hits
    if occupancy(low) > target_occupancy:
        return low, False
    if occupancy(high) < target_occupancy:
        return high, False
    while high - low > max(1, int(0.01 * low)):
        mid = (low + high) // 2
        if occupancy(mid) < target_occupancy:
            low = mid
        else:
            high = mid
    return high, True


# --- one (baseline, hit count) cell -------------------------------------------------------------------------------------------


def _ordered(stats: np.ndarray) -> np.ndarray:
    return np.mean((stats[:, 0] >= stats[:, 1]) & (stats[:, 1] >= stats[:, 2]), axis=0)


def evaluate_cell(
    baseline: str, n_hits: int, replicates: int, seed: int = SEED, sampler=None
) -> dict[str, Any]:
    sampler = sampler or baseline_sampler(baseline)

    def rng(role: int) -> np.random.Generator:
        return np.random.default_rng(
            np.random.SeedSequence([seed, n_hits, BASELINE_IDS[baseline], role])
        )

    timing: dict[str, float] = {}
    t = time.perf_counter()
    base, base_occ = m3._collect(sampler, n_hits, rng(0), replicates)
    fresh, _ = m3._collect(sampler, n_hits, rng(1), replicates)
    timing["baseline_and_fresh_s"] = time.perf_counter() - t
    t = time.perf_counter()
    pos_eq, occ_eq = m3._collect(m3.sample_cascade, n_hits, rng(2), replicates)
    timing["cascade_equal_hits_s"] = time.perf_counter() - t
    t = time.perf_counter()
    target = float(np.median(base_occ))
    n_match, matched = match_cascade_hits(
        target,
        n_hits,
        np.random.SeedSequence([seed, n_hits, BASELINE_IDS[baseline], 3]),
    )
    timing["occupancy_matching_s"] = time.perf_counter() - t
    t = time.perf_counter()
    pos_occ, occ_matched = m3._collect(m3.sample_cascade, n_match, rng(4), replicates)
    timing["cascade_matched_s"] = time.perf_counter() - t

    reliable = reliable_windows(target)
    low, high = np.percentile(base[:, 3], m3.BASELINE_INTERVAL, axis=0)
    fpr = np.mean((fresh[:, 3] < low) | (fresh[:, 3] > high), axis=0)
    power_eq = np.mean(pos_eq[:, 3] > high, axis=0)
    power_occ = np.mean(pos_occ[:, 3] > high, axis=0)
    order_eq, order_occ = _ordered(pos_eq), _ordered(pos_occ)
    d0_below_d1 = np.mean(pos_eq[:, 0] < pos_eq[:, 1], axis=0)
    d1_below_d2 = np.mean(pos_eq[:, 1] < pos_eq[:, 2], axis=0)
    rows = []
    for k, window in enumerate(m3.WINDOWS):
        usable = bool(
            reliable[k]
            and matched
            and fpr[k] <= m3.MAX_FALSE_POSITIVE_RATE
            and power_eq[k] >= m3.MIN_DETECTION_POWER
            and power_occ[k] >= m3.MIN_DETECTION_POWER
            and order_eq[k] >= m3.MIN_ORDERING_RATE
            and order_occ[k] >= m3.MIN_ORDERING_RATE
        )
        rows.append(
            {
                "window": list(window),
                "reliable": reliable[k],
                "false_positive_rate": float(fpr[k]),
                "power_equal_hits": float(power_eq[k]),
                "power_occupancy_matched": float(power_occ[k]),
                "ordering_equal_hits": float(order_eq[k]),
                "ordering_occupancy_matched": float(order_occ[k]),
                "fraction_d0_below_d1": float(d0_below_d1[k]),
                "fraction_d1_below_d2": float(d1_below_d2[k]),
                "usable": usable,
            }
        )
    return {
        "baseline": baseline,
        "n_hits": n_hits,
        "replicates": replicates,
        "median_occupied_voxels": target,
        "saturation_level_k_star": saturation_level(target),
        "n_reliable_windows": int(sum(reliable)),
        "cascade_hits_occupancy_matched": n_match,
        "occupancy_matched": matched,
        "cascade_median_occupied_equal_hits": float(np.median(occ_eq)),
        "cascade_median_occupied_matched": float(np.median(occ_matched)),
        "timing_s": timing,
        "windows": rows,
    }


def summarise(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Windows usable under all three baselines at a hit count, and the eligibility threshold."""
    by_count: dict[int, dict[str, dict[str, Any]]] = {}
    for cell in cells:
        by_count.setdefault(cell["n_hits"], {})[cell["baseline"]] = cell
    usable_windows: dict[str, Any] = {}
    for n_hits, group in sorted(by_count.items()):
        if set(group) != set(BASELINE_IDS):
            continue
        for k, window in enumerate(m3.WINDOWS):
            if all(group[b]["windows"][k]["usable"] for b in BASELINE_IDS):
                entry = usable_windows.setdefault(
                    str(tuple(window)), {"usable_at_hits": [], "threshold": None}
                )
                entry["usable_at_hits"].append(n_hits)
                if entry["threshold"] is None:
                    entry["threshold"] = max(
                        group[b]["median_occupied_voxels"] for b in BASELINE_IDS
                    )
    return {
        "usable_under_all_baselines": usable_windows,
        "n_usable": len(usable_windows),
    }


def _task(args: tuple[str, int, int]) -> dict[str, Any]:
    return evaluate_cell(*args)


def _reference_task(n_hits: int) -> dict[str, Any]:
    """The original configuration (B1, equal-hits cascade, no saturation rule) with the new seed: the noise reference."""
    results = m3.evaluate(n_hits, SEED, m3.REPLICATES)
    return {
        "n_hits": n_hits,
        "d0_d2": [r.__dict__ for r in results if r.statistic == "D0-D2"],
    }


def run_rung1() -> dict[str, Any]:
    t0 = time.perf_counter()
    cell = evaluate_cell("B1", 30_000, 100)
    result = {
        "registration": "EXP-010, 2026-10-22, hash 5ec9ca672a04f1c0",
        "rung": 1,
        "seed": SEED,
        "wall_seconds": time.perf_counter() - t0,
        "cell": cell,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "exp010_rung1.json").write_text(
        json.dumps(result, indent=1), encoding="utf-8"
    )
    return result


def run_rung2(replicates: int = m3.REPLICATES) -> dict[str, Any]:
    t0 = time.perf_counter()
    mean_map()  # build the B2 map once before the workers start
    tasks = [(b, n, replicates) for b in BASELINE_IDS for n in HIT_COUNTS]
    with ProcessPoolExecutor(WORKERS) as pool:
        cells = list(pool.map(_task, tasks))
        reference = list(pool.map(_reference_task, m3.HIT_COUNTS))
    result = {
        "registration": "EXP-010, 2026-10-22, hash 5ec9ca672a04f1c0",
        "rung": 2,
        "seed": SEED,
        "wall_seconds": time.perf_counter() - t0,
        "summary": summarise(cells),
        "cells": cells,
        "noise_reference_original_configuration": reference,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "exp010_full.json").write_text(
        json.dumps(result, indent=1, default=str), encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rung", type=int, choices=(1, 2), required=True)
    args = parser.parse_args(argv)
    result = run_rung1() if args.rung == 1 else run_rung2()
    print(f"wall {result['wall_seconds']:.0f} s")


if __name__ == "__main__":
    main()
