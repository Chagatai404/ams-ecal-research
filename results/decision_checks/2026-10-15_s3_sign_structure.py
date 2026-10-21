import json, time, numpy as np
from scipy.stats import spearmanr
from ams_ecal.multiscale_estimators import multiscale_test as mt
from ams_ecal.inference import permutation_nulls as pn
from ams_ecal.validation import sample_definitions as sd

N_EVENTS, N_PERM, SEED = 1500, 500, 20261015
out = {}
for energy in (20.0, 100.0):
    t0 = time.time()
    rng = np.random.default_rng(SEED + int(energy))
    cols, names = {}, []
    for sample in ("electron_development_large", "proton_development_large"):
        ev = sd.load_events(sample, energy, "deposit_grid_mev", "all")
        g = ev.grids[:N_EVENTS]
        n_c, conv = mt.conventional_features(g)
        ms = mt.multiscale_table(g)
        cols[sample] = (conv, ms)
    cn = n_c
    mn = [f"ms_{i}" for i in range(cols["electron_development_large"][1].shape[1])]
    X = np.vstack([np.hstack([cols[s][0], cols[s][1]]) for s in cols])
    y = np.concatenate([np.ones(N_EVENTS, int), np.zeros(N_EVENTS, int)])
    names = list(cn) + mn
    m = X.shape[1]
    # features with usable data
    finite = np.isfinite(X).mean(axis=0)
    keep = [j for j in range(m) if finite[j] >= 0.5 and np.nanstd(X[:, j]) > 0]
    # A. pooled feature correlation (Spearman, pairwise complete)
    sub = X[:, keep]
    rho = np.full((len(keep), len(keep)), np.nan)
    for a in range(len(keep)):
        for b in range(a, len(keep)):
            ok = np.isfinite(sub[:, a]) & np.isfinite(sub[:, b])
            if ok.sum() > 100:
                rho[a, b] = rho[b, a] = spearmanr(sub[ok, a], sub[ok, b])[0]
    iu = np.triu_indices(len(keep), 1)
    r = rho[iu]; r = r[np.isfinite(r)]
    # B. null KS correlations
    null = np.empty((N_PERM, len(keep)))
    for k in range(N_PERM):
        perm = pn.permute_labels_within_strata(y, None, rng)
        null[k] = [pn.stratum_ks(sub[:, j], perm, None) for j in range(len(keep))]
    good = np.isfinite(null).all(axis=0) & (null.std(axis=0) > 0)
    C = np.corrcoef(null[:, good].T)
    iu2 = np.triu_indices(C.shape[0], 1)
    c = C[iu2]
    noise = 1 / np.sqrt(N_PERM)
    out[f"{energy:g}"] = {
        "n_features_used": int(len(keep)), "n_features_null": int(good.sum()),
        "feature_spearman": {"pairs": int(r.size), "fraction_negative": float((r < 0).mean()),
            "fraction_below_-0.2": float((r < -0.2).mean()), "min": float(r.min()), "median": float(np.median(r))},
        "null_ks_correlation": {"pairs": int(c.size), "fraction_negative": float((c < 0).mean()),
            "fraction_below_-2se": float((c < -2 * noise).mean()), "fraction_above_+2se": float((c > 2 * noise).mean()),
            "min": float(c.min()), "median": float(np.median(c)), "se_per_pair": float(noise)},
        "seconds": round(time.time() - t0),
    }
    print(energy, json.dumps(out[f"{energy:g}"]))
import pathlib
pathlib.Path("results/detector_response").mkdir(exist_ok=True)
pathlib.Path("results/decision_checks/2026-10-15_s3_sign_structure.json").write_text(json.dumps({"seed": SEED, "n_events_per_class": N_EVENTS, "n_permutations": N_PERM, "data": "exposed development-large electrons and protons; no sealed sets", "results": out}, indent=1))
