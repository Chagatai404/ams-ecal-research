"""Time of one registered cross-fit (HGB, five folds) on exposed electron/proton features, to cost a classifier permutation test.
Measured on the machine in use (idle apart from this script); no sealed data."""
import json, time, numpy as np
from ams_ecal.inference.cross_fitting import out_of_fold_scores
from ams_ecal.multiscale_estimators import multiscale_test as mt
from ams_ecal.validation import sample_definitions as sd

out = {}
for energy in (20.0, 100.0):
    xs, ys = [], []
    for label, sample in ((1, "electron_development_large"), (0, "proton_development_large")):
        g = sd.load_events(sample, energy, "deposit_grid_mev", "all").grids[:2000]
        _, conv = mt.conventional_features(g)
        ms = mt.multiscale_table(g)
        xs.append(np.hstack([conv, ms])); ys.append(np.full(len(g), label))
    x, y = np.vstack(xs), np.concatenate(ys)
    x = np.nan_to_num(x, nan=-1.0)
    rng = np.random.default_rng(1)
    times = []
    for rep in range(3):
        t0 = time.time(); out_of_fold_scores(x, y, seed=rep); times.append(time.time() - t0)
    out[f"{energy:g}"] = {"events": int(len(y)), "features": int(x.shape[1]), "seconds_per_cross_fit": times,
                           "median_seconds": float(np.median(times))}
    print(energy, out[f"{energy:g}"], flush=True)
json.dump(out, open("results/decision_checks/2026-10-15_s3_classifier_timing.json", "w"), indent=1)
