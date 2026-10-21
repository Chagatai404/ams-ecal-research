"""Design study for the P1 strata of the DECIDE-1 amendment (2026-10-15): how many electrons and interacting protons remain comparable
when matched on occupied cells and deposited energy? Uses the cached stage-1 events (exposed development data; electrons against
truth-interacting protons; events with at least 336 occupied cells = eligible for the base window). The rules compared are
descriptive; the choice of a rule from this study is a design decision on DEVELOPMENT data and is stated as such in the amendment."""
import json
import numpy as np
from ams_ecal.inference import permutation_nulls as pn
from ams_ecal.multiscale_estimators import multiscale_verdict_reread as vr

MIN_CELLS, MIN_PER_CLASS = 336, 20
out = {}


def usable_cells(key_e, key_p, n=5):
    """Pooled 5 x 5 equal-count cells: number with at least MIN_PER_CLASS events of each class."""
    pooled = np.vstack([key_e, key_p])
    s = pn.equal_count_strata(pooled[:, 0], n) + n * pn.equal_count_strata(pooled[:, 1], n)
    labels = np.r_[np.ones(len(key_e)), np.zeros(len(key_p))]
    good = [k for k in np.unique(s) if (labels[s == k] == 1).sum() >= MIN_PER_CLASS and (labels[s == k] == 0).sum() >= MIN_PER_CLASS]
    return len(good), int(sum(((s == k)).sum() for k in good))


def support_box(key_e, key_p, lo=5, hi=95):
    """Intersection of the central [lo, hi] percentile ranges of the two classes on every axis."""
    low = np.maximum(np.percentile(key_e, lo, axis=0), np.percentile(key_p, lo, axis=0))
    high = np.minimum(np.percentile(key_e, hi, axis=0), np.percentile(key_p, hi, axis=0))
    inside = lambda k: ((k >= low) & (k <= high)).all(axis=1)
    return inside


def caliper_pairs(key_e, key_p, caliper=0.25):
    """Greedy 1:1 nearest-neighbour matching without replacement on the standardised log keys; number of pairs within the caliper."""
    z = np.log(np.vstack([key_e, key_p]))
    mean, sd = z.mean(axis=0), z.std(axis=0)
    ze, zp = (np.log(key_e) - mean) / sd, (np.log(key_p) - mean) / sd
    from scipy.spatial import cKDTree
    tree = cKDTree(zp)
    dist, idx = tree.query(ze, k=min(50, len(zp)))
    taken, pairs = set(), 0
    order = np.argsort(dist[:, 0])
    for i in order:
        for d, j in zip(dist[i], idx[i]):
            if d > caliper: break
            if j not in taken:
                taken.add(j); pairs += 1; break
    return pairs


for energy in vr.ENERGIES:
    for rep in vr.REPRESENTATIONS:
        d = vr.load_block(energy, rep, None)
        ke = np.column_stack([d["e_occupancy"], d["e_energy"]]); kp = np.column_stack([d["p_occupancy"], d["p_energy"]])
        ke, kp = ke[ke[:, 0] >= MIN_CELLS], kp[kp[:, 0] >= MIN_CELLS]
        row = {"eligible_electrons": int(len(ke)), "eligible_protons": int(len(kp))}
        if len(ke) >= 100 and len(kp) >= 100:
            row["pooled_5x5_usable_cells"], row["pooled_5x5_events_in_usable_cells"] = usable_cells(ke, kp)
            inside = support_box(ke, kp)
            ie, ip = inside(ke), inside(kp)
            row["support_box_events"] = [int(ie.sum()), int(ip.sum())]
            row["support_box_fraction"] = [float(ie.mean()), float(ip.mean())]
            if ie.sum() >= 100 and ip.sum() >= 100:
                row["support_box_5x5_usable_cells"], row["support_box_5x5_events_in_usable_cells"] = usable_cells(ke[ie], kp[ip])
                row["support_box_3x3_usable_cells"] = usable_cells(ke[ie], kp[ip], 3)[0]
            row["caliper_0.25sd_pairs"] = caliper_pairs(ke, kp, 0.25)
            row["caliper_0.5sd_pairs"] = caliper_pairs(ke, kp, 0.5)
        out[f"{energy:g}_{rep[:3]}"] = row
        print(f"{energy:>5g} {rep[:3]}", row, flush=True)
json.dump(out, open("results/decision_checks/2026-10-15_p1_common_support_study.json", "w"), indent=1)
