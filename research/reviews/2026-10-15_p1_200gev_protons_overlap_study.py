"""P1 design study, part 3 (2026-10-15): do the existing 200 GeV protons (component_proton, 5,000 events, tagged development set) overlap
electrons of 50 and 100 GeV in (occupied cells, deposited energy)? Same measures as part 2: events with >= 336 occupied cells, deposit
representation, usable pooled 5 x 5 cells (>= 20 per class) and caliper pairs. Descriptive; development data; the 200 GeV set is small
(about 2,400 interacting protons), so counts are a floor for the production size the researcher decided to add to the new sealed sets."""
import json
import numpy as np
from scipy.spatial import cKDTree
from ams_ecal.inference import permutation_nulls as pn
from ams_ecal.validation import sample_definitions as sd

MIN_CELLS, MIN_PER_CLASS, REP = 336, 20, "deposit_grid_mev"

def keys(sample, energy, population):
    g = sd.load_events(sample, energy, REP, population, dtype=np.float32).grids
    k = np.column_stack([(g > 0).sum(axis=(1, 2)).astype(float), g.sum(axis=(1, 2), dtype=np.float64)])
    return k[k[:, 0] >= MIN_CELLS]

def usable_cells(ke, kp, n=5):
    pooled = np.vstack([ke, kp])
    s = pn.equal_count_strata(pooled[:, 0], n) + n * pn.equal_count_strata(pooled[:, 1], n)
    lab = np.r_[np.ones(len(ke)), np.zeros(len(kp))]
    good = [k for k in np.unique(s) if (lab[s == k] == 1).sum() >= MIN_PER_CLASS and (lab[s == k] == 0).sum() >= MIN_PER_CLASS]
    return len(good), int(sum((s == k).sum() for k in good))

def pairs(ke, kp, caliper):
    z = np.log(np.vstack([ke, kp])); mean, sd_ = z.mean(0), z.std(0)
    ze, zp = (np.log(ke) - mean) / sd_, (np.log(kp) - mean) / sd_
    dist, idx = cKDTree(zp).query(ze, k=min(50, len(zp)))
    taken, n = set(), 0
    for i in np.argsort(dist[:, 0]):
        for d, j in zip(dist[i], idx[i]):
            if d > caliper: break
            if j not in taken: taken.add(j); n += 1; break
    return n

out = {}
kp = keys("component_proton", 200.0, "interacting")
print("interacting 200 GeV protons with >=336 cells:", len(kp), "median deposit GeV", round(float(np.median(kp[:, 1]) / 1000), 1))
for sample, energy in (("electron_development_large", 20.0), ("electron_development_large", 50.0), ("electron_development_large", 100.0), ("component_electron", 100.0), ("component_electron", 200.0)):
    ke = keys(sample, energy, "all")
    r = {"electron_events": int(len(ke)), "proton_events": int(len(kp)), "median_deposit_gev_electron": float(np.median(ke[:, 1]) / 1000),
         "usable_5x5_cells": usable_cells(ke, kp)[0], "events_in_usable_cells": usable_cells(ke, kp)[1],
         "caliper_0.25_pairs": pairs(ke, kp, 0.25), "caliper_0.5_pairs": pairs(ke, kp, 0.5)}
    out[f"{sample}_{energy:g}_vs_proton_200"] = r
    print(f"{sample:28s} {energy:>5g} GeV vs protons 200 GeV:", r, flush=True)
json.dump(out, open("research/reviews/2026-10-15_p1_200gev_protons_overlap_study.json", "w"), indent=1)
