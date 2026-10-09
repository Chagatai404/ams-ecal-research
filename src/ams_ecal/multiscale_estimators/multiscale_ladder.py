"""The information ladder of the registered multiscale test (pre-registration section 6).

Rungs: (1) conventional features, (2) multiscale features, (3) conventional + multiscale, (4) the raw 18 x 72 grid, (5) raw +
multiscale. One registered classifier (``added_checks.CLASSIFIER_SETTINGS``, five folds), out-of-fold scores, so every rung is
scored on the same events. Reported per rung: the AUC, and the proton false-positive rate (and rejection = 1 / FPR) at 90% and
99% electron efficiency, each with a 95% bootstrap interval over events; and the paired AUC gain of rung 3 over rung 1 with its
interval. The AUC is an outcome, never a tuning target. Sample efficiency: the rungs 1-3 trained on n events per class.

Development use here is on exposed events; the registered run is on G-final once.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ams_ecal.multiscale_estimators import multiscale_test as mt
from ams_ecal.validation.added_checks import CLASSIFIER_SETTINGS, N_FOLDS

EFFICIENCIES = (0.90, 0.99)
BOOTSTRAPS = 200
MATERIAL_AUC_GAIN = 0.01
RUNGS = {
    "conventional": ("conv",),
    "multiscale": ("ms",),
    "conventional+multiscale": ("conv", "ms"),
    "raw": ("raw",),
    "raw+multiscale": ("raw", "ms"),
}
SAMPLE_EFFICIENCY_RUNGS = ("conventional", "multiscale", "conventional+multiscale")


def build_blocks(electron_grids: np.ndarray, proton_grids: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Feature blocks for the pooled events (electrons first) and the labels (1 = electron)."""

    grids = np.concatenate([electron_grids, proton_grids])
    labels = np.concatenate([np.ones(len(electron_grids), int), np.zeros(len(proton_grids), int)])
    _, conventional = mt.conventional_features(grids)
    blocks = {"conv": conventional, "ms": mt.multiscale_table(grids), "raw": grids.reshape(len(grids), -1)}
    return blocks, labels


def stack(blocks: dict[str, np.ndarray], parts: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([blocks[p] for p in parts])


def false_positive_rate(scores: np.ndarray, labels: np.ndarray, efficiency: float) -> float:
    """Fraction of protons scoring at or above the threshold that keeps ``efficiency`` of the electrons."""

    threshold = np.quantile(scores[labels == 1], 1.0 - efficiency)
    return float((scores[labels == 0] >= threshold).mean())


def out_of_fold_scores(
    x: np.ndarray, labels: np.ndarray, per_class: int | None, seed: int = 0, folds: int = N_FOLDS
) -> np.ndarray:
    """Out-of-fold scores; the training set of each fold is cut to ``per_class`` events per class (None = all)."""

    rng = np.random.default_rng(seed)
    scores = np.zeros(len(labels))
    for train, test in StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(x, labels):
        if per_class is not None:
            keep = [rng.permutation(train[labels[train] == c])[:per_class] for c in (0, 1)]
            train = np.concatenate(keep)
        usable = ~np.isnan(x[train]).all(
            axis=0
        )  # a column that is NaN in the whole training set carries nothing and breaks the binning
        if not usable.any():
            scores[test] = 0.5
            continue
        model = HistGradientBoostingClassifier(**CLASSIFIER_SETTINGS).fit(x[train][:, usable], labels[train])
        scores[test] = model.predict_proba(x[test][:, usable])[:, 1]
    return scores


def summarise_scores(scores: np.ndarray, labels: np.ndarray, rng: np.random.Generator, bootstraps: int) -> dict:
    e, p = np.where(labels == 1)[0], np.where(labels == 0)[0]
    draws = {"auc": [], **{f"fpr_{int(100 * eff)}": [] for eff in EFFICIENCIES}}
    for _ in range(bootstraps):
        idx = np.concatenate([rng.choice(e, len(e)), rng.choice(p, len(p))])
        draws["auc"].append(roc_auc_score(labels[idx], scores[idx]))
        for eff in EFFICIENCIES:
            draws[f"fpr_{int(100 * eff)}"].append(false_positive_rate(scores[idx], labels[idx], eff))
    out = {
        "auc": float(roc_auc_score(labels, scores)),
        "auc_95": [float(v) for v in np.percentile(draws["auc"], [2.5, 97.5])],
    }
    floor = 1.0 / max(len(p), 1)
    for eff in EFFICIENCIES:
        key = f"fpr_{int(100 * eff)}"
        fpr = false_positive_rate(scores, labels, eff)
        out[key] = fpr
        out[f"rejection_{int(100 * eff)}"] = 1.0 / max(fpr, floor)
        out[f"rejection_{int(100 * eff)}_95"] = [1.0 / max(v, floor) for v in np.percentile(draws[key], [97.5, 2.5])]
    return out


def paired_gain(
    scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray, rng: np.random.Generator, bootstraps: int
) -> dict:
    """AUC of ``scores_b`` minus that of ``scores_a`` on the same events, with a paired bootstrap interval and the registered rule."""

    e, p = np.where(labels == 1)[0], np.where(labels == 0)[0]
    gains = []
    for _ in range(bootstraps):
        idx = np.concatenate([rng.choice(e, len(e)), rng.choice(p, len(p))])
        gains.append(roc_auc_score(labels[idx], scores_b[idx]) - roc_auc_score(labels[idx], scores_a[idx]))
    gain = float(roc_auc_score(labels, scores_b) - roc_auc_score(labels, scores_a))
    low, high = (float(v) for v in np.percentile(gains, [2.5, 97.5]))
    return {
        "gain": gain,
        "gain_95": [low, high],
        "adds_information_by_the_registered_rule": bool(gain >= MATERIAL_AUC_GAIN and low > 0),
    }


def ladder(
    electron_grids: np.ndarray,
    proton_grids: np.ndarray,
    sizes: tuple[int, ...] = (),
    seed: int = 0,
    bootstraps: int = BOOTSTRAPS,
    rungs: dict[str, tuple[str, ...]] = RUNGS,
) -> dict:
    blocks, labels = build_blocks(electron_grids, proton_grids)
    rng = np.random.default_rng(seed)
    scores = {name: out_of_fold_scores(stack(blocks, parts), labels, None, seed) for name, parts in rungs.items()}
    out: dict = {
        "events": {"electron": len(electron_grids), "proton": len(proton_grids)},
        "eligible_fraction": {
            "electron": float(np.isfinite(blocks["ms"][labels == 1]).any(axis=1).mean()),
            "proton": float(np.isfinite(blocks["ms"][labels == 0]).any(axis=1).mean()),
        },
        "rungs": {name: summarise_scores(s, labels, rng, bootstraps) for name, s in scores.items()},
    }
    if "conventional" in scores and "conventional+multiscale" in scores:
        out["gain_conventional_plus_multiscale_over_conventional"] = paired_gain(
            scores["conventional"], scores["conventional+multiscale"], labels, rng, bootstraps
        )
    out["sample_efficiency_auc"] = {}
    for name in SAMPLE_EFFICIENCY_RUNGS:
        if name not in rungs:
            continue
        curve = {}
        for size in sizes:
            s = out_of_fold_scores(stack(blocks, rungs[name]), labels, size, seed)
            curve[str(size)] = float(roc_auc_score(labels, s))
        out["sample_efficiency_auc"][name] = curve
    return out


def development(bootstraps: int = BOOTSTRAPS) -> dict:
    """Development run on EXPOSED events: electrons against truth-interacting protons (all rungs) and against all calibration
    protons (rungs 1-3), per energy and representation. Not the registered test; sealed sets are not read."""

    from ams_ecal.multiscale_estimators.eligibility_count import ENERGIES_GEV, SAMPLES

    out: dict = {
        "status": "DEVELOPMENT on exposed events; not the registered test; sealed sets not read",
        "per_energy": {},
    }
    for energy in ENERGIES_GEV:
        row: dict = {}
        for representation in ("deposit_grid_mev", "readout_grid_mev"):
            electrons = np.load(SAMPLES["electron"] / f"E{energy:g}GeV" / "events.npz")[representation].astype(float)
            arrays = np.load(SAMPLES["proton"] / f"E{energy:g}GeV" / "events.npz")
            calibration = arrays["event_index"] % 4 != 3
            protons_all = arrays[representation][calibration].astype(float)
            protons_int = arrays[representation][calibration & arrays["truth_occurred"]].astype(float)
            row[representation] = {
                "vs_interacting_protons": ladder(electrons, protons_int, sizes=(100, 300), bootstraps=bootstraps),
                "vs_all_protons": ladder(
                    electrons,
                    protons_all,
                    bootstraps=bootstraps,
                    rungs={k: v for k, v in RUNGS.items() if "raw" not in v},
                ),
            }
        out["per_energy"][f"{energy:g}"] = row
        print(f"{energy:g} GeV done", flush=True)
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json
    from pathlib import Path

    from ams_ecal.electron_model.em_production import PROJECT_ROOT

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results" / "multiscale_controls" / "multiscale_ladder_development.json",
    )
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    args = parser.parse_args(argv)
    args.out.write_text(json.dumps(development(args.bootstraps), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
