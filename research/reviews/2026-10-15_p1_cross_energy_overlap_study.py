"""P1 design study, part 2 (2026-10-15): does pairing electrons of primary energy E_e with protons of a HIGHER primary energy E_p restore overlap in
(occupied cells, deposited energy)? Uses the cached stage-1 events, events with >= 336 occupied cells, deposit representation (the true
deposit; the readout is limited by eligibility). Descriptive; the choice of a pairing from this study is a design decision on DEVELOPMENT data."""
import json
import numpy as np
from scipy.spatial import cKDTree
from ams_ecal.inference import permutation_nulls as pn
from ams_ecal.multiscale_estimators import multiscale_verdict_reread as vr

MIN_CELLS, MIN_PER_CLASS = 336, 20
REP = "deposit_grid_mev"

def keys(energy, cls):
    d = vr.load_block(energy, REP, None)
    k = np.column_stack([d[f"{cls}_occupancy"], d[f"{cls}_energy"]])
    return k[k[:, 0] >= MIN_CELLS]

def usable_cells(ke, kp, n=5):
    pooled = np.vstack([ke, kp])
    s = pn.equal_count_strata(pooled[:, 0], n) + n * pn.equal_count_strata(pooled[:, 1], n)
    lab = np.r_[np.ones(len(ke)), np.zeros(len(kp))]
    good = [k for k in np.unique(s) if (lab[s == k] == 1).sum() >= MIN_PER_CLASS and (lab[s == k] == 0).sum() >= MIN_PER_CLASS]
    return len(good), int(sum((s == k).sum() for k in good))

def pairs(ke, kp, caliper):
    z = np.log(np.vstack([ke, kp])); mean, sd = z.mean(0), z.std(0)
    ze, zp = (np.log(ke) - mean) / sd, (np.log(kp) - mean) / sd
    dist, idx = cKDTree(zp).query(ze, k=min(50, len(zp)))
    taken, n = set(), 0
    for i in np.argsort(dist[:, 0]):
        for d, j in zip(dist[i], idx[i]):
            if d > caliper: break
            if j not in taken: taken.add(j); n += 1; break
    return n

out = {}
for e_e in (10.0, 20.0, 50.0, 100.0):
    for e_p in (10.0, 20.0, 50.0, 100.0):
        if e_p < e_e: continue
        ke, kp = keys(e_e, "e"), keys(e_p, "p")
        if len(ke) < 100 or len(kp) < 100: continue
        r = {"electron_events": int(len(ke)), "proton_events": int(len(kp)),
             "median_deposit_gev_electron": float(np.median(ke[:, 1]) / 1000), "median_deposit_gev_proton": float(np.median(kp[:, 1]) / 1000),
             "usable_5x5_cells": usable_cells(ke, kp)[0], "events_in_usable_cells": usable_cells(ke, kp)[1],
             "caliper_0.25_pairs": pairs(ke, kp, 0.25), "caliper_0.5_pairs": pairs(ke, kp, 0.5)}
        out[f"e{e_e:g}_p{e_p:g}"] = r
        print(f"electrons {e_e:>5g} GeV vs protons {e_p:>5g} GeV:", r, flush=True)
json.dump(out, open("research/reviews/2026-10-15_p1_cross_energy_overlap_study.json", "w"), indent=1)
