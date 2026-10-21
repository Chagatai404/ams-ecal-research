"""Guard for the S3 re-read (registration section 3): family-wise error of the 24 statistics on same-class data.

Electrons of one energy are split at random into two groups of 2,000 (so every statistic is a true null); the block test of the
re-read (24 statistics, stratified KS, joint label permutations) is run with 999 permutations (Holm over 24 needs a smallest p of
about 0.002, which 199 draws cannot give) and the share of trials with any adjusted p <= 0.05 is the family-wise error of
Westfall-Young and of Holm. Tolerance written before the run: at most 0.05 + 3 Monte Carlo standard errors. Exposed electrons only.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ams_ecal.inference import multiplicity as mp
from ams_ecal.multiscale_estimators import multiscale_verdict_reread as vr

TRIALS, PERMUTATIONS, GROUP, ALPHA = 1000, 999, 2000, 0.05


def one_trial(args):
    energy, representation, seed = args
    data = vr.load_block(energy, representation, None)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data["e_values"]))
    a, b = order[:GROUP], order[GROUP : 2 * GROUP]
    block = vr.block_test(
        data["e_values"][a], data["e_values"][b], data["e_occupancy"][a], data["e_occupancy"][b], PERMUTATIONS, rng
    )
    usable = (block["finite_electrons"] >= 100) & (block["finite_protons"] >= 100) & np.isfinite(block["observed"])
    if not usable.any():  # nothing testable in this block (too few eligible events): not a family to guard
        return (float("nan"), float("nan"), float("nan"), 0)
    observed, null = block["observed"][usable], block["null"][:, usable]
    raw = (1.0 + (null >= observed).sum(axis=0)) / (1.0 + PERMUTATIONS)
    return (
        float(mp.westfall_young_stepdown(observed, null).min() <= ALPHA),
        float(mp.holm(raw).min() <= ALPHA),
        float(raw.min() <= ALPHA),
        int(usable.sum()),
    )


if __name__ == "__main__":
    out = {}
    with ProcessPoolExecutor(6) as pool:
        for energy in (20.0, 100.0):
            for rep in vr.REPRESENTATIONS:
                rows = list(pool.map(one_trial, [(energy, rep, 7000 + t) for t in range(TRIALS)], chunksize=5))
                wy, holm, raw, n = (np.array(c) for c in zip(*rows))
                if not n.any():
                    out[f"{energy:g}_{rep}"] = {"trials": TRIALS, "not_testable": True, "usable_statistics_median": 0}
                    print(energy, rep, "not testable", flush=True)
                    continue
                tol = 3 * np.sqrt(ALPHA * (1 - ALPHA) / TRIALS)
                out[f"{energy:g}_{rep}"] = {
                    "trials": TRIALS, "permutations": PERMUTATIONS, "usable_statistics_median": float(np.median(n)),
                    "westfall_young_fwer": float(wy.mean()), "holm_fwer": float(holm.mean()),
                    "unadjusted_minimum_p_rate": float(raw.mean()), "tolerance": ALPHA + tol,
                    "passes": bool(wy.mean() <= ALPHA + tol and holm.mean() <= ALPHA + tol),
                }
                print(energy, rep, out[f"{energy:g}_{rep}"], flush=True)
    json.dump(out, open("results/decision_checks/2026-10-15_s3_reread_familywise_guard.json", "w"), indent=1)
