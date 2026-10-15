"""Size and power of Holm, Benjamini-Hochberg and Westfall-Young step-down on real feature dependence (S3 check, 2026-10-15).

Design (fixed before the run):
* pool: 6,000 exposed electrons of ``electron_development_large`` at 20 GeV; features = the conventional features and the
  multiscale table, restricted to columns finite for every pool event (so no NaN handling enters the statistic);
* a trial draws 2 x 1,000 distinct pool events (labels random, so every feature is a true null), then plants a mean shift of
  ``d`` pool standard deviations in the label-1 events of ``k`` randomly chosen features (k = 0: global null);
* statistic: two-sample KS distance per feature; null: ``B`` = 999 label permutations, the same permutation for all features;
* methods applied with the repository implementations: raw p, Holm, Benjamini-Hochberg, Westfall-Young step-down maxT;
* per scenario: family-wise error (any unplanted feature rejected), false discovery proportion (mean V / max(R, 1)) and power
  (mean share of planted features rejected) at alpha = 0.05.
Exposed development data only; no sealed set is read. B = 999 is a simulation economy; production uses 10,000.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ams_ecal.inference import multiplicity as mp

N_PER_CLASS, B, ALPHA, TRIALS, WORKERS, SEED = 1000, 999, 0.05, 300, 6, 20261015
SCENARIOS = [(0, 0.0), (5, 0.2), (12, 0.2)]
POOL_FILE = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None


def build_pool(path: pathlib.Path) -> None:
    from ams_ecal.multiscale_estimators import multiscale_test as mt
    from ams_ecal.validation import sample_definitions as sd

    grids = sd.load_events("electron_development_large", 20.0, "deposit_grid_mev", "all").grids[:6000]
    _, conventional = mt.conventional_features(grids)
    table = mt.multiscale_table(grids)
    x = np.hstack([conventional, table])
    ok = np.isfinite(x).all(axis=0) & (x.std(axis=0) > 0)
    np.save(path, x[:, ok])


def ks_many(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """KS distance per feature for every label row: ``x`` (n, m), ``labels`` (r, n) of 0/1; returns (r, m)."""

    r, n = labels.shape
    n1 = labels.sum(axis=1, keepdims=True)
    n0 = n - n1
    out = np.empty((r, x.shape[1]))
    for j in range(x.shape[1]):
        order = np.argsort(x[:, j], kind="stable")
        v = x[order, j]
        boundary = np.append(v[1:] != v[:-1], True)
        lab = labels[:, order].astype(np.float32)
        c1 = np.cumsum(lab, axis=1)
        pos = np.arange(1, n + 1, dtype=np.float32)
        diff = np.abs(c1 / n1 - (pos - c1) / n0)
        out[:, j] = diff[:, boundary].max(axis=1)
    return out


def trial(args: tuple[int, int, float, int]) -> dict[str, float]:
    k, scenario_index, shift, seed = args
    pool = np.load(POOL_FILE)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), 2 * N_PER_CLASS, replace=False)
    x = pool[idx].copy()
    labels = rng.permutation(np.repeat([1, 0], N_PER_CLASS))
    m = x.shape[1]
    planted = np.zeros(m, bool)
    if k:
        planted[rng.choice(m, k, replace=False)] = True
        sd = pool.std(axis=0)
        x[np.ix_(labels == 1, np.flatnonzero(planted))] += shift * sd[planted]
    observed = ks_many(x, labels[None, :])[0]
    perms = rng.permuted(np.tile(labels, (B, 1)), axis=1)
    null = ks_many(x, perms)
    raw = (1.0 + (null >= observed).sum(axis=0)) / (1.0 + B)
    adjusted = {
        "raw": raw,
        "holm": mp.holm(raw),
        "bh": mp.benjamini_hochberg(raw),
        "westfall_young": mp.westfall_young_stepdown(observed, null),
    }
    row: dict[str, float] = {}
    for name, p in adjusted.items():
        reject = p <= ALPHA
        v = float((reject & ~planted).sum())
        row[f"{name}_fwer"] = float(v > 0)
        row[f"{name}_fdp"] = v / max(int(reject.sum()), 1)
        row[f"{name}_power"] = float((reject & planted).sum() / planted.sum()) if k else float("nan")
    row["m"] = float(m)
    return row


def main() -> None:
    result: dict[str, object] = {
        "design": __doc__,
        "alpha": ALPHA,
        "permutations": B,
        "trials": TRIALS,
        "n_per_class": N_PER_CLASS,
        "seed": SEED,
        "scenarios": {},
    }
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as pool_ex:
        for si, (k, shift) in enumerate(SCENARIOS):
            jobs = [(k, si, shift, SEED + 1000 * si + t) for t in range(TRIALS)]
            rows = list(pool_ex.map(trial, jobs, chunksize=5))
            summary: dict[str, float] = {"k_planted": k, "shift_sd": shift, "n_features": rows[0]["m"]}
            for key in rows[0]:
                vals = np.array([r[key] for r in rows], dtype=float)
                if np.isfinite(vals).any() and key != "m":
                    summary[key] = float(np.nanmean(vals))
                    summary[key + "_se"] = float(np.nanstd(vals) / np.sqrt(np.isfinite(vals).sum()))
            result["scenarios"][f"k={k},shift={shift}"] = summary  # type: ignore[index]
            print(json.dumps(summary), f"{time.time() - t0:.0f}s", flush=True)
    pathlib.Path("research/reviews/2026-10-15_s3_multiplicity_simulation.json").write_text(
        json.dumps(result, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    if POOL_FILE is None:
        raise SystemExit("usage: python script.py POOL.npy")
    if not POOL_FILE.exists():
        build_pool(POOL_FILE)
    main()
