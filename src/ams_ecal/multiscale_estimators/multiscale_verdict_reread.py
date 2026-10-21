"""S3 stage 1: a valid re-reading of the 192 development KS verdicts (registration: research/COMPLETED_WORK.md (was 2026-10-15_s3_reread_registration.md)).

Electrons against truth-interacting protons, 24 multiscale statistics x 2 representations x 4 energies, tested by label permutation
within occupancy strata (10,000 draws), on the events eligible for each statistic, with Westfall-Young and Holm adjustment
(inside each representation family, Bonferroni across the two). Exploratory; exposed development data only; no sealed set.

    python -m ams_ecal.multiscale_estimators.multiscale_verdict_reread
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ams_ecal.electron_model.em_production import PROJECT_ROOT
from ams_ecal.inference import multiplicity as mp
from ams_ecal.inference import permutation_nulls as pn
from ams_ecal.inference.missingness_controls import MIN_ELIGIBLE_PER_CLASS
from ams_ecal.multiscale_estimators import multiscale_test as mt
from ams_ecal.validation import sample_definitions as sd

ENERGIES = (10.0, 20.0, 50.0, 100.0)
REPRESENTATIONS = ("deposit_grid_mev", "readout_grid_mev")
N_PERMUTATIONS = 10_000
CHUNK = 250
STRATA = 5
ALPHA = 0.05
SEED = 20261015
WORKERS = 6
CACHE = PROJECT_ROOT / "data" / "s3_reread_cache"
RESULTS = PROJECT_ROOT / "results" / "multiscale_controls"
OLD_VERDICTS = ("multiscale_gtrain_development_10_20_50GeV.json", "multiscale_gtrain_development_100GeV.json")


# --- the engine: the stratified KS for many labellings at once ---------------------------------------------------------------


def permuted_labels(labels: np.ndarray, strata: np.ndarray, rows: int, rng: np.random.Generator) -> np.ndarray:
    """``rows`` labellings, each ``labels`` shuffled inside every stratum (``(rows, n)``)."""

    out = np.empty((rows, len(labels)), dtype=np.int8)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        out[:, idx] = rng.permuted(np.tile(labels[idx], (rows, 1)), axis=1)
    return out


def stratified_ks_matrix(
    values: np.ndarray, labels: np.ndarray, strata: np.ndarray, minimum: int = pn.MIN_PER_STRATUM
) -> np.ndarray:
    """Stratum-weighted KS of ``values`` ``(n, m)`` (NaN = undefined) for every labelling ``(rows, n)``: ``(rows, m)``.

    The same definition as ``permutation_nulls.stratum_ks`` (KS per stratum on the finite values, weights min(n1, n0), strata with
    fewer than ``minimum`` finite events in a class left out), for all labellings and statistics at once. NaN when no stratum
    qualifies.
    """

    values = np.asarray(values, dtype=float)
    rows = labels.shape[0]
    numerator = np.zeros((rows, values.shape[1]))
    weight = np.zeros((rows, values.shape[1]))
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        stratum_labels = labels[:, idx].astype(np.float32)
        for j in range(values.shape[1]):
            v = values[idx, j]
            finite = np.isfinite(v)
            count = int(finite.sum())
            if count < 2 * minimum:
                continue
            order = np.argsort(v[finite], kind="stable")
            sorted_values = v[finite][order]
            boundary = np.append(sorted_values[1:] != sorted_values[:-1], True)
            cumulative = np.cumsum(stratum_labels[:, finite][:, order], axis=1)
            n1 = cumulative[:, -1:]
            n0 = count - n1
            valid = (n1[:, 0] >= minimum) & (n0[:, 0] >= minimum)
            if not valid.any():
                continue
            position = np.arange(1, count + 1, dtype=np.float32)[boundary]
            c1 = cumulative[:, boundary]
            with np.errstate(divide="ignore", invalid="ignore"):
                distance = np.abs(c1 / n1 - (position - c1) / n0)
            ks = np.where(np.isfinite(distance), distance, 0.0).max(axis=1)
            w = np.minimum(n1, n0)[:, 0]
            numerator[:, j] += np.where(valid, w * ks, 0.0)
            weight[:, j] += np.where(valid, w, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numerator / weight
    out[weight == 0] = np.nan
    return out


def block_test(
    electron_values: np.ndarray,
    proton_values: np.ndarray,
    electron_key: np.ndarray,
    proton_key: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """One (energy, representation) block: observed statistics and the permutation null of all columns together."""

    values = np.vstack([electron_values, proton_values])
    labels = np.concatenate([np.ones(len(electron_values), np.int8), np.zeros(len(proton_values), np.int8)])
    pooled_key = np.concatenate([electron_key, proton_key])
    if pooled_key.ndim == 2:  # joint strata: every key column in equal-count quintiles, combined
        strata = sum(pn.equal_count_strata(pooled_key[:, k], STRATA) * STRATA**k for k in range(pooled_key.shape[1]))
    else:
        strata = pn.equal_count_strata(pooled_key, STRATA)
    observed = stratified_ks_matrix(values, labels[None, :], strata)[0]
    null = np.empty((permutations, values.shape[1]))
    for start in range(0, permutations, CHUNK):
        rows = min(CHUNK, permutations - start)
        null[start : start + rows] = stratified_ks_matrix(values, permuted_labels(labels, strata, rows, rng), strata)
    reports = [pn.stratum_ks_report(values[:, j], labels, strata) for j in range(values.shape[1])]
    return {
        "observed": observed,
        "null": null,
        "finite_electrons": np.isfinite(electron_values).sum(axis=0),
        "finite_protons": np.isfinite(proton_values).sum(axis=0),
        "strata_used": np.array([r["strata_used"] for r in reports]),
        "strata_skipped": np.array([r["strata_skipped"] for r in reports]),
    }


# --- data and features ---------------------------------------------------------------------------------------------------------


def _table(chunk: np.ndarray) -> np.ndarray:
    return mt.multiscale_table(chunk.astype(np.float64))


def features(grids: np.ndarray, pool: ProcessPoolExecutor | None) -> np.ndarray:
    parts = np.array_split(grids, max(1, len(grids) // 500))
    return np.vstack(list(pool.map(_table, parts))) if pool else np.vstack([_table(p) for p in parts])


def load_block(energy: float, representation: str, pool: ProcessPoolExecutor | None) -> dict[str, np.ndarray]:
    """Cached statistics and stratification keys of one block (electrons, interacting protons)."""

    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"block_{energy:g}_{representation}.npz"
    if path.exists():
        return dict(np.load(path))
    out = {}
    for name, sample, population in (("e", "electron_development_large", "all"), ("p", "proton_development_large", "interacting")):
        grids = sd.load_events(sample, energy, representation, population, dtype=np.float32).grids
        out[f"{name}_values"] = features(grids, pool)
        out[f"{name}_occupancy"] = (grids > 0).sum(axis=(1, 2)).astype(float)
        out[f"{name}_energy"] = grids.sum(axis=(1, 2), dtype=np.float64)
    np.savez(path, **out)
    return out


def run_block(args: tuple[float, str, str, int, int]) -> tuple[float, str, dict[str, Any]]:
    energy, representation, key_name, permutations, seed = args
    data = load_block(energy, representation, None)
    rng = np.random.default_rng(seed)
    if key_name == "joint":
        e_key = np.column_stack([data["e_occupancy"], data["e_energy"]])
        p_key = np.column_stack([data["p_occupancy"], data["p_energy"]])
    else:
        e_key, p_key = data[f"e_{key_name}"], data[f"p_{key_name}"]
    result = block_test(data["e_values"], data["p_values"], e_key, p_key, permutations, rng)
    result["n_electrons"], result["n_protons"] = len(data["e_values"]), len(data["p_values"])
    return energy, representation, result


# --- assembly --------------------------------------------------------------------------------------------------------------------


def old_verdicts() -> dict[tuple[float, str, str], bool]:
    out: dict[tuple[float, str, str], bool] = {}
    for name in OLD_VERDICTS:
        per_energy = json.loads((RESULTS / name).read_text(encoding="utf-8"))["per_energy"]
        for energy, by_rep in per_energy.items():
            for rep, body in by_rep.items():
                for feature, row in body["statistics_vs_interacting_protons"].items():
                    out[(float(energy), rep, feature)] = bool(row["material_by_the_registered_rule"])
    return out


def assemble(blocks: dict[tuple[float, str], dict[str, Any]], permutations: int) -> dict[str, Any]:
    names = mt.feature_names()
    old = old_verdicts()
    tests: list[dict[str, Any]] = []
    families: dict[str, list[int]] = {rep: [] for rep in REPRESENTATIONS}
    for rep in REPRESENTATIONS:
        for energy in ENERGIES:
            if (energy, rep) not in blocks:
                continue
            b = blocks[(energy, rep)]
            for j, feature in enumerate(names):
                enough = b["finite_electrons"][j] >= MIN_ELIGIBLE_PER_CLASS and b["finite_protons"][j] >= MIN_ELIGIBLE_PER_CLASS
                observed = float(b["observed"][j])
                null_column = b["null"][:, j]
                usable = bool(enough and np.isfinite(observed) and np.isfinite(null_column).all())
                test = {
                    "energy_gev": energy,
                    "representation": rep,
                    "statistic": feature,
                    "finite_electrons": int(b["finite_electrons"][j]),
                    "finite_protons": int(b["finite_protons"][j]),
                    "testable": usable,
                    "strata_used": int(b["strata_used"][j]),
                    "strata_skipped": int(b["strata_skipped"][j]),
                    "old_material_verdict": old.get((energy, rep, feature)),
                    "_block": (energy, rep),
                    "_column": j,
                }
                if usable:
                    floor = float(np.percentile(null_column, 95))
                    test.update(
                        observed_ks=observed,
                        floor_95th=floor,
                        excess=observed - floor,
                        p_value=float((1.0 + (null_column >= observed).sum()) / (1.0 + len(null_column))),
                    )
                    families[rep].append(len(tests))
                tests.append(test)
    for members in families.values():
        if not members:
            continue
        observed = np.array([tests[i]["observed_ks"] for i in members])
        null = np.column_stack([blocks[tests[i]["_block"]]["null"][:, tests[i]["_column"]] for i in members])
        raw = np.array([tests[i]["p_value"] for i in members])
        adjusted = {
            "westfall_young": mp.westfall_young_stepdown(observed, null),
            "holm": mp.holm(raw),
            "bh_q": mp.benjamini_hochberg(raw),
        }
        for k, i in enumerate(members):
            for name, values in adjusted.items():
                tests[i][f"{name}_in_family"] = float(values[k])
            tests[i]["westfall_young_192"] = float(min(1.0, 2 * adjusted["westfall_young"][k]))
            tests[i]["holm_192"] = float(min(1.0, 2 * adjusted["holm"][k]))
    for t in tests:
        del t["_block"], t["_column"]
    usable_tests = [t for t in tests if t["testable"]]

    def count(rule) -> int:
        return sum(1 for t in usable_tests if rule(t))

    agreement = {
        "tests_total": len(tests),
        "not_testable": len(tests) - len(usable_tests),
        "old_material": sum(1 for t in tests if t["old_material_verdict"]),
        "old_material_and_testable": count(lambda t: t["old_material_verdict"]),
        "old_material_still_significant_holm": count(lambda t: t["old_material_verdict"] and t["holm_192"] <= ALPHA),
        "old_material_still_significant_westfall_young": count(
            lambda t: t["old_material_verdict"] and t["westfall_young_192"] <= ALPHA
        ),
        "old_not_material_but_significant_holm": count(
            lambda t: t["old_material_verdict"] is False and t["holm_192"] <= ALPHA
        ),
        "old_not_material_but_significant_westfall_young": count(
            lambda t: t["old_material_verdict"] is False and t["westfall_young_192"] <= ALPHA
        ),
        "significant_holm_192": count(lambda t: t["holm_192"] <= ALPHA),
        "significant_westfall_young_192": count(lambda t: t["westfall_young_192"] <= ALPHA),
        "significant_unadjusted": count(lambda t: t["p_value"] <= ALPHA),
    }
    return {
        "registration": "research/COMPLETED_WORK.md (was 2026-10-15_s3_reread_registration.md)",
        "status": "EXPLORATORY; exposed development data; no sealed set read",
        "permutations": permutations,
        "alpha": ALPHA,
        "agreement": agreement,
        "tests": tests,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--key", choices=("occupancy", "energy", "joint"), default="occupancy")
    parser.add_argument("--energies", type=float, nargs="+", default=list(ENERGIES))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as pool:  # features first (cached), then the permutation blocks
        for energy in args.energies:
            for rep in REPRESENTATIONS:
                load_block(energy, rep, pool)
                print(f"features {energy:g} {rep} {time.time() - t0:.0f}s", flush=True)
    jobs = [
        (energy, rep, args.key, args.permutations, SEED + 100 * i + j)
        for i, energy in enumerate(args.energies)
        for j, rep in enumerate(REPRESENTATIONS)
    ]
    blocks: dict[tuple[float, str], dict[str, Any]] = {}
    with ProcessPoolExecutor(WORKERS) as pool:
        for energy, rep, result in pool.map(run_block, jobs):
            blocks[(energy, rep)] = result
            print(f"block {energy:g} {rep} done {time.time() - t0:.0f}s", flush=True)
    result = assemble(blocks, args.permutations)
    result["strata_key"] = args.key
    default = {
        "occupancy": "s3_reread_stage1.json",
        "energy": "s3_reread_stage1_energy_strata.json",
        "joint": "s3_reread_stage1_joint_strata.json",
    }[args.key]
    (args.out or RESULTS / default).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result["agreement"], indent=1))


if __name__ == "__main__":
    main()
