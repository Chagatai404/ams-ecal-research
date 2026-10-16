"""S3 stage 3: electrons against interacting protons matched on occupied cells and deposited energy.

Registration: research/plans/2026-10-16_s3_stage3_matched_pairs_registration.md. Electrons and protons of different primary energies are
paired 1:1 by nearest-neighbour matching in the standardized logarithms of occupied cells and deposited energy within a caliper; the 24
multiscale statistics (raw and the stage-2 excess) are compared on the matched pairs with a pair-swap permutation test. Exploratory;
exposed development data; no sealed set.

    python -m ams_ecal.multiscale_estimators.multiscale_matched_pairs
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from ams_ecal.inference import multiplicity as mp
from ams_ecal.multiscale_estimators import multiscale_test as mt
from ams_ecal.multiscale_estimators import multiscale_verdict_reread as vr

PAIRINGS = (
    (20.0, 50.0),
    (20.0, 100.0),
    (50.0, 100.0),
)  # (electron GeV, proton GeV), registered from the development overlap study
CALIPERS = (0.25, 0.5)
MIN_CELLS = 336
MIN_PAIRS = 100
NEIGHBOURS = 50
BALANCE_LIMIT = 0.10
STAT_TYPES = ("raw", "excess_full", "excess_block9")
SWAP_CHUNK = 500
BOOTSTRAP_DRAWS = 2000
ALPHA = 0.05


# --- matching ---------------------------------------------------------------------------------------------------------------------


def standardized_logs(
    key_e: np.ndarray, key_p: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Logarithms of the two keys standardized with the pooled mean and standard deviation; also returns the pooled sd."""

    log_e, log_p = np.log(key_e), np.log(key_p)
    pooled = np.vstack([log_e, log_p])
    mean, sd = pooled.mean(axis=0), pooled.std(axis=0)
    return (log_e - mean) / sd, (log_p - mean) / sd, sd


def match_pairs(
    key_e: np.ndarray, key_p: np.ndarray, caliper: float
) -> tuple[np.ndarray, np.ndarray]:
    """1:1 matching without replacement: electrons in the order of increasing distance to their nearest proton, each taking its nearest
    still-unmatched proton among the ``NEIGHBOURS`` nearest within the caliper. Returns the electron and proton indices of the pairs."""

    ze, zp, _ = standardized_logs(key_e, key_p)
    k = min(NEIGHBOURS, len(zp))
    distance, index = cKDTree(zp).query(ze, k=k)
    distance, index = distance.reshape(len(ze), k), index.reshape(len(ze), k)
    taken: set[int] = set()
    electrons: list[int] = []
    protons: list[int] = []
    for i in np.argsort(distance[:, 0]):
        for d, j in zip(distance[i], index[i], strict=True):
            if d > caliper:
                break
            if int(j) not in taken:
                taken.add(int(j))
                electrons.append(int(i))
                protons.append(int(j))
                break
    return np.array(electrons, dtype=int), np.array(protons, dtype=int)


def balance(
    key_e: np.ndarray, key_p: np.ndarray, ie: np.ndarray, ip: np.ndarray
) -> list[float]:
    """Standardized mean difference (pooled sd of the eligible classes) of both log keys over the matched pairs."""

    _, _, sd = standardized_logs(key_e, key_p)
    mean_e, mean_p = np.log(key_e[ie]).mean(axis=0), np.log(key_p[ip]).mean(axis=0)
    return [float((mean_e[k] - mean_p[k]) / sd[k]) for k in range(2)]


# --- the test ---------------------------------------------------------------------------------------------------------------------


def pair_swap_test(
    values_e: np.ndarray,
    values_p: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """KS distance between the electron and proton values of the valid pairs, per column, with a null from swapping the classes
    inside pairs (each pair with probability 1/2). Pairs with a non-finite member are removed from that column in both classes."""

    n = len(values_e)
    valid = np.isfinite(values_e) & np.isfinite(values_p)
    values = np.vstack(
        [np.where(valid, values_e, np.nan), np.where(valid, values_p, np.nan)]
    )
    strata = np.zeros(2 * n, dtype=int)
    observed_labels = np.r_[np.ones(n, np.int8), np.zeros(n, np.int8)][None, :]
    observed = vr.stratified_ks_matrix(values, observed_labels, strata, MIN_PAIRS)[0]
    null = np.empty((permutations, values.shape[1]))
    for start in range(0, permutations, SWAP_CHUNK):
        rows = min(SWAP_CHUNK, permutations - start)
        swap = (rng.random((rows, n)) < 0.5).astype(np.int8)
        null[start : start + rows] = vr.stratified_ks_matrix(
            values, np.concatenate([1 - swap, swap], axis=1), strata, MIN_PAIRS
        )
    return {"observed": observed, "null": null, "valid_pairs": valid.sum(axis=0)}


def paired_median_difference(
    values_e: np.ndarray,
    values_p: np.ndarray,
    rng: np.random.Generator,
    draws: int = BOOTSTRAP_DRAWS,
) -> list[tuple[float, float, float]]:
    """Median of electron - proton over the valid pairs of each column with a 95% bootstrap interval over pairs."""

    out = []
    for j in range(values_e.shape[1]):
        d = (values_e[:, j] - values_p[:, j])[
            np.isfinite(values_e[:, j]) & np.isfinite(values_p[:, j])
        ]
        if len(d) < 2:
            out.append((float("nan"),) * 3)
            continue
        boots = np.median(d[rng.integers(0, len(d), (draws, len(d)))], axis=1)
        low, high = np.percentile(boots, [2.5, 97.5])
        out.append((float(np.median(d)), float(low), float(high)))
    return out


def adjust_family(test: dict[str, np.ndarray]) -> dict[str, Any]:
    """Raw p-values and Westfall-Young / Holm / BH adjustment over the testable columns of one family."""

    usable = (
        (test["valid_pairs"] >= MIN_PAIRS)
        & np.isfinite(test["observed"])
        & np.isfinite(test["null"]).all(axis=0)
    )
    out: dict[str, Any] = {"usable": usable}
    if not usable.any():
        return out
    observed, null = test["observed"][usable], test["null"][:, usable]
    raw = (1.0 + (null >= observed).sum(axis=0)) / (1.0 + len(null))
    out.update(
        raw=raw,
        floor=np.percentile(null, 95, axis=0),
        westfall_young=mp.westfall_young_stepdown(observed, null),
        holm=mp.holm(raw),
        bh=mp.benjamini_hochberg(raw),
    )
    return out


# --- one pairing -------------------------------------------------------------------------------------------------------------------


def load_pair_data(
    e_energy: float, p_energy: float, representation: str
) -> dict[str, np.ndarray]:
    """Eligible electrons (energy ``e_energy``) and interacting protons (``p_energy``): keys and the three statistic types."""

    be = vr.load_block(e_energy, representation, None)
    bp = vr.load_block(p_energy, representation, None)
    out: dict[str, np.ndarray] = {}
    eligible_e = np.flatnonzero(be["e_occupancy"] >= MIN_CELLS)
    eligible_p = np.flatnonzero(bp["p_occupancy"] >= MIN_CELLS)
    out["key_e"] = np.column_stack([be["e_occupancy"], be["e_energy"]])[eligible_e]
    out["key_p"] = np.column_stack([bp["p_occupancy"], bp["p_energy"]])[eligible_p]
    out["raw_e"], out["raw_p"] = be["e_values"][eligible_e], bp["p_values"][eligible_p]
    for surrogate in ("full", "block9"):
        excess_e = np.load(
            vr.CACHE / f"excess_{surrogate}_{e_energy:g}_{representation}.npz"
        )["e_excess"]
        excess_p = np.load(
            vr.CACHE / f"excess_{surrogate}_{p_energy:g}_{representation}.npz"
        )["p_excess"]
        out[f"excess_{surrogate}_e"], out[f"excess_{surrogate}_p"] = (
            excess_e[eligible_e],
            excess_p[eligible_p],
        )
    return out


def run_pairing(args: tuple[float, float, str, float, int, int]) -> dict[str, Any]:
    e_energy, p_energy, representation, caliper, permutations, seed = args
    rng = np.random.default_rng(seed)
    data = load_pair_data(e_energy, p_energy, representation)
    result: dict[str, Any] = {
        "electron_gev": e_energy,
        "proton_gev": p_energy,
        "representation": representation,
        "caliper_sd": caliper,
        "eligible_electrons": len(data["key_e"]),
        "eligible_protons": len(data["key_p"]),
    }
    if len(data["key_e"]) < MIN_PAIRS or len(data["key_p"]) < MIN_PAIRS:
        result.update(
            pairs=0, testable=False, reason="fewer than 100 eligible events in a class"
        )
        return result
    ie, ip = match_pairs(data["key_e"], data["key_p"], caliper)
    smd = (
        balance(data["key_e"], data["key_p"], ie, ip) if len(ie) else [float("nan")] * 2
    )
    result.update(
        pairs=len(ie),
        matched_fraction_electrons=float(len(ie) / len(data["key_e"])),
        matched_fraction_protons=float(len(ip) / len(data["key_p"])),
        standardized_mean_difference_log_cells_log_energy=smd,
        balanced=bool(len(ie) and max(abs(v) for v in smd) <= BALANCE_LIMIT),
    )
    if len(ie) < MIN_PAIRS:
        result.update(testable=False, reason="fewer than 100 matched pairs")
        return result
    result["testable"] = True
    names = mt.feature_names()
    families = {}
    for stat_type in STAT_TYPES:
        values_e, values_p = data[f"{stat_type}_e"][ie], data[f"{stat_type}_p"][ip]
        test = pair_swap_test(values_e, values_p, permutations, rng)
        adjusted = adjust_family(test)
        medians = paired_median_difference(values_e, values_p, rng)
        position = {int(c): k for k, c in enumerate(np.flatnonzero(adjusted["usable"]))}
        rows = []
        for j, name in enumerate(names):
            row: dict[str, Any] = {
                "statistic": name,
                "valid_pairs": int(test["valid_pairs"][j]),
                "testable": j in position,
            }
            if j in position:
                k = position[j]
                row.update(
                    observed_ks=float(test["observed"][j]),
                    floor_95th=float(adjusted["floor"][k]),
                    p_value=float(adjusted["raw"][k]),
                    westfall_young_in_family=float(adjusted["westfall_young"][k]),
                    holm_in_family=float(adjusted["holm"][k]),
                    bh_q=float(adjusted["bh"][k]),
                    median_difference_electron_minus_proton=medians[j],
                )
            rows.append(row)
        families[stat_type] = rows
    result["families"] = families
    return result


# --- same-class guard ---------------------------------------------------------------------------------------------------------------


def guard_trial(args: tuple[int, int, int]) -> tuple[float, float, float, int]:
    seed, permutations, cap = args
    data = vr.load_block(20.0, "deposit_grid_mev", None)
    eligible = np.flatnonzero(data["e_occupancy"] >= MIN_CELLS)
    rng = np.random.default_rng(seed)
    order = rng.permutation(eligible)
    half = len(order) // 2
    a, b = order[:half], order[half : 2 * half]
    key_a = np.column_stack([data["e_occupancy"][a], data["e_energy"][a]])
    key_b = np.column_stack([data["e_occupancy"][b], data["e_energy"][b]])
    ia, ib = match_pairs(key_a, key_b, CALIPERS[0])
    ia, ib = ia[:cap], ib[:cap]
    test = pair_swap_test(
        data["e_values"][a][ia], data["e_values"][b][ib], permutations, rng
    )
    adjusted = adjust_family(test)
    if not adjusted["usable"].any():
        return (float("nan"), float("nan"), float("nan"), 0)
    return (
        float(adjusted["westfall_young"].min() <= ALPHA),
        float(adjusted["holm"].min() <= ALPHA),
        float(adjusted["raw"].min() <= ALPHA),
        int(adjusted["usable"].sum()),
    )


def run_guard(trials: int, permutations: int, cap: int = 150) -> dict[str, Any]:
    with ProcessPoolExecutor(vr.WORKERS) as pool:
        rows = list(
            pool.map(
                guard_trial,
                [(40000 + t, permutations, cap) for t in range(trials)],
                chunksize=5,
            )
        )
    wy, holm, raw, n = (np.array(c) for c in zip(*rows, strict=True))
    tolerance = ALPHA + 3 * np.sqrt(ALPHA * (1 - ALPHA) / trials)
    return {
        "trials": trials,
        "permutations": permutations,
        "pair_cap": cap,
        "usable_statistics_median": float(np.nanmedian(n)),
        "westfall_young_fwer": float(np.nanmean(wy)),
        "holm_fwer": float(np.nanmean(holm)),
        "unadjusted_minimum_p_rate": float(np.nanmean(raw)),
        "tolerance": float(tolerance),
        "passes": bool(np.nanmean(wy) <= tolerance and np.nanmean(holm) <= tolerance),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--permutations", type=int, default=vr.N_PERMUTATIONS)
    parser.add_argument(
        "--guard-trials",
        type=int,
        default=0,
        help="run only the same-class guard with this many trials",
    )
    parser.add_argument("--guard-permutations", type=int, default=999)
    args = parser.parse_args(argv)
    t0 = time.time()
    if args.guard_trials:
        result = run_guard(args.guard_trials, args.guard_permutations)
        (vr.RESULTS / "s3_stage3_matched_pairs_guard.json").write_text(
            json.dumps(result, indent=1), encoding="utf-8"
        )
        print(json.dumps(result, indent=1))
        return
    jobs = [
        (e, p, rep, caliper, args.permutations, vr.SEED + 17 * i + 3 * j + k)
        for i, (e, p) in enumerate(PAIRINGS)
        for j, rep in enumerate(vr.REPRESENTATIONS)
        for k, caliper in enumerate(CALIPERS)
    ]
    with ProcessPoolExecutor(vr.WORKERS) as pool:
        results = list(pool.map(run_pairing, jobs))
    for r in results:
        print(
            f"e{r['electron_gev']:g} p{r['proton_gev']:g} {r['representation'][:3]} caliper {r['caliper_sd']}: "
            f"pairs {r['pairs']} testable {r['testable']} balanced {r.get('balanced')} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    for caliper in CALIPERS:  # Bonferroni across the families with at least one testable statistic, per caliper
        families = [
            rows
            for r in results
            if r["testable"] and r["caliper_sd"] == caliper
            for rows in r["families"].values()
            if any(row["testable"] for row in rows)
        ]
        factor = max(len(families), 1)
        for rows in families:
            for row in rows:
                if row["testable"]:
                    row["westfall_young_all_families"] = float(
                        min(1.0, factor * row["westfall_young_in_family"])
                    )
                    row["holm_all_families"] = float(
                        min(1.0, factor * row["holm_in_family"])
                    )
        for r in results:
            if r["caliper_sd"] == caliper:
                r["bonferroni_factor_families"] = factor
    report = {
        "registration": "research/plans/2026-10-16_s3_stage3_matched_pairs_registration.md",
        "status": "EXPLORATORY; exposed development data; no sealed set read",
        "permutations": args.permutations,
        "results": results,
    }
    (vr.RESULTS / "s3_stage3_matched_pairs.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8"
    )
    print("written", round(time.time() - t0))


if __name__ == "__main__":
    main()
