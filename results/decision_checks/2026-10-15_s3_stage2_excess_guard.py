"""Guard for the stage-2 excess statistic (stage-2 registration section 3): family-wise error on same-class data.

Electrons of one energy are split at random into two groups of 2,000 (every statistic is a true null); the event-level EXCESS tables of
stage 2 (both surrogates, from the cache) are tested with the stage-1 block test (stratified KS, joint label permutations, occupancy
strata), 999 permutations, 1,000 trials; the share of trials with any adjusted p <= 0.05 is the family-wise error of Westfall-Young and
of Holm. Tolerance written before the run: 0.05 + 3 Monte Carlo standard errors (0.0707). Exposed electrons only; 20 and 100 GeV."""
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ams_ecal.inference import multiplicity as mp
from ams_ecal.multiscale_estimators import multiscale_verdict_reread as vr

TRIALS, PERMUTATIONS, GROUP, ALPHA = 1000, 999, 2000, 0.05


def one_trial(args):
    surrogate, energy, representation, seed = args
    data = np.load(vr.CACHE / f"excess_{surrogate}_{energy:g}_{representation}.npz")
    excess, occupancy = data["e_excess"], data["e_occupancy"]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(excess))
    a, b = order[:GROUP], order[GROUP : 2 * GROUP]
    block = vr.block_test(excess[a], excess[b], occupancy[a], occupancy[b], PERMUTATIONS, rng)
    usable = (block["finite_electrons"] >= 100) & (block["finite_protons"] >= 100) & np.isfinite(block["observed"])
    if not usable.any():
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
        for surrogate in ("full", "block9"):
            for energy in (20.0, 100.0):
                for rep in vr.REPRESENTATIONS:
                    rows = list(pool.map(one_trial, [(surrogate, energy, rep, 9000 + t) for t in range(TRIALS)], chunksize=5))
                    wy, holm, raw, n = (np.array(c) for c in zip(*rows))
                    key = f"{surrogate}_{energy:g}_{rep}"
                    if not n.any():
                        out[key] = {"trials": TRIALS, "not_testable": True}
                        print(key, "not testable", flush=True)
                        continue
                    tol = ALPHA + 3 * np.sqrt(ALPHA * (1 - ALPHA) / TRIALS)
                    out[key] = {
                        "trials": TRIALS, "permutations": PERMUTATIONS, "usable_statistics_median": float(np.median(n)),
                        "westfall_young_fwer": float(wy.mean()), "holm_fwer": float(holm.mean()),
                        "unadjusted_minimum_p_rate": float(raw.mean()), "tolerance": float(tol),
                        "passes": bool(wy.mean() <= tol and holm.mean() <= tol),
                    }
                    print(key, out[key], flush=True)
    json.dump(out, open("results/decision_checks/2026-10-15_s3_stage2_excess_guard.json", "w"), indent=1)
