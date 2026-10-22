"""Noise floor of the R-A per-layer closure criterion; writes ``r_a_noise_floor.json``.

Registered before the run (2026-10-22, EXP-012 step 1): no model is fitted. Each class at each energy is split into
two random disjoint halves of floor(N/2) events; per layer the ratio of (fibre / deposit) between the halves minus one
is what a perfect conditional model would still show from finite statistics. The maximum absolute value over the 18
layers is summarised by its median and 95th percentile over 1,000 splits.

* 95th percentile >= 3%: the 3% criterion is structurally unmeetable at that size;
* 1.5% to 3%: observed closure is partly noise;
* below 1.5%: observed closure errors are real model errors.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ams_ecal.validation.sample_definitions import (
    PROJECT_ROOT,
    energies_available,
    load_events,
)

N_SPLITS = 1000
SEED = 20261022
TOLERANCE = 0.03
PARTLY_NOISE = 0.015
RESULT_PATH = PROJECT_ROOT / "results" / "detector_response" / "r_a_noise_floor.json"
CASES = (
    ("electron_development_large", "all"),
    ("proton_development_large", "crossing"),
    ("proton_development_large", "interacting"),
    ("component_proton_large", "crossing"),
    ("component_proton_large", "interacting"),
)


def half_split_floor(
    deposit_layer: np.ndarray,
    fibre_layer: np.ndarray,
    rng: np.random.Generator,
    n_splits: int = N_SPLITS,
) -> np.ndarray:
    """Maximum absolute layer ratio-of-ratios minus one, one value per random half-split.

    ``deposit_layer`` and ``fibre_layer`` are (events, layers) sums over cells.
    """

    n = len(deposit_layer)
    half = n // 2
    values = np.empty(n_splits)
    for i in range(n_splits):
        order = rng.permutation(n)
        a, b = order[:half], order[half : 2 * half]
        ratio_a = fibre_layer[a].sum(axis=0) / deposit_layer[a].sum(axis=0)
        ratio_b = fibre_layer[b].sum(axis=0) / deposit_layer[b].sum(axis=0)
        values[i] = np.nanmax(np.abs(ratio_a / ratio_b - 1.0))
    return values


def classify(p95: float) -> str:
    if p95 >= TOLERANCE:
        return "structurally_unmeetable"
    if p95 >= PARTLY_NOISE:
        return "partly_noise"
    return "real_model_error"


def run() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    rows: dict[str, Any] = {}
    for sample, population in CASES:
        for energy in energies_available(sample):
            deposit = load_events(
                sample, energy, "deposit_grid_mev", population, dtype=np.float32
            )
            fibre = load_events(
                sample, energy, "readout_grid_mev", population, dtype=np.float32
            )
            n = len(deposit)
            if n < 4:
                continue
            floor = half_split_floor(
                deposit.grids.sum(axis=2, dtype=np.float64),
                fibre.grids.sum(axis=2, dtype=np.float64),
                rng,
            )
            p95 = float(np.quantile(floor, 0.95))
            rows[f"{sample}|{population}|{energy:g}"] = {
                "n_events": n,
                "half_size": n // 2,
                "median": float(np.median(floor)),
                "p95": p95,
                "reading": classify(p95),
            }
            print(
                f"{sample:28s} {population:12s} {energy:6g} GeV n={n:6d} "
                f"median {np.median(floor):.4f} p95 {p95:.4f} {classify(p95)}"
            )
    result = {
        "registration": "EXP-012 step 1, 2026-10-22 (hash in research/COMPLETED_WORK.md section 5)",
        "seed": SEED,
        "n_splits": N_SPLITS,
        "rule": {
            "unmeetable_at_or_above": TOLERANCE,
            "partly_noise_at_or_above": PARTLY_NOISE,
        },
        "rows": rows,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
