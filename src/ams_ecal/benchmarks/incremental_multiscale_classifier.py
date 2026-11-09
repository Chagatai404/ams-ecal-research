"""EXP-018: do readout multiscale features (D0, D1, D2) add electron/proton information beyond conventional variables?

EXPLORATORY; development data; pilot size; no claim beyond the stated estimand.

Implements, literally, the registration
``C:/Users/ASUS/.research-os-backups/registrations/2026-11-09_exp018_incremental_classifier_realistic_pilot.txt``
(SHA-256 prefix ``15f2e58a0b3d34d3``; written before any classifier was fitted on these data). Where the registration left a choice to
the implementer the code says ``# REGISTRATION AMBIGUITY:`` and the choice is listed in ``ambiguities()`` and in the result JSON.

    python -m ams_ecal.benchmarks.incremental_multiscale_classifier --out results/realistic_regime/exp018_incremental_classifier

Pipeline: fiducial events of the EXP-015 rung-1 Geant4 batches (electrons; truth-interacting protons) -> hard subset defined only from
the two matching keys (never from a classifier output) -> 24 conventional features C and 18 multiscale features M -> models A (C) and
B (C + M) cross-fitted on the same folds (stratified 5-fold, repeated 20 times) -> per-fold class-balanced log-loss difference
delta = loss(B) - loss(A), corrected repeated cross-validation interval (Nadeau and Bengio) -> decision rule, negative control
(M permuted across events) and same-class null.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from ams_ecal.benchmarks.learning_curves import balanced_log_loss
from ams_ecal.inference import rejection_limits as rl
from ams_ecal.multiscale_estimators import multiscale_test as mt
from ams_ecal.multiscale_estimators import realistic_regime_analysis as rra

EXPLORATORY_LABEL = (
    "EXPLORATORY; development data; pilot size; no claim beyond the stated estimand"
)
REGISTRATION_PATH = Path(
    "C:/Users/ASUS/.research-os-backups/registrations/2026-11-09_exp018_incremental_classifier_realistic_pilot.txt"
)
REGISTRATION_SHA256_PREFIX = "15f2e58a0b3d34d3"

# --- fixed constants of the registration ----------------------------------------------------------------------------------------------

MINIMUM_OCCUPIED_CELLS = 572  # (a) the stricter DEC-013 eligibility threshold
CALIPER = 0.25  # (b) common support, standardised units, both keys
MINIMUM_CLASS_EVENTS = 100  # fewer in either class: "not testable"
N_FOLDS = 5
N_REPEATS = 20
SEED = 20261109  # repeat seeds are SEED + r
SAME_CLASS_REPEATS = 5
ELECTRON_EFFICIENCY = 0.90
CONFIDENCE = 0.95
SENSITIVITY_THETA_MAX_DEG = 12.0
MAX_JOBS = 6
HGB_SETTINGS: dict[str, Any] = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": False,
}
LOGISTIC_SETTINGS: dict[str, Any] = {"C": 1.0, "max_iter": 2000}
N_CONVENTIONAL = 24
N_MULTISCALE = 18

DECISION_ADDS = "multiscale features add information (exploratory)"
DECISION_NONE = "no addition detected"
DECISION_WORSE = "multiscale features make the model worse"
NEGATIVE_CONTROL_PASSED = "passed"
NEGATIVE_CONTROL_INVALID = "invalid (pipeline leaks: upper limit below 0)"
NEGATIVE_CONTROL_WORSE = "not passed (interval excludes 0 on the worse side)"


def ambiguities() -> list[str]:
    """Every point where the registration left a choice to the implementer (see the ``# REGISTRATION AMBIGUITY`` comments)."""

    return list(REGISTRATION_AMBIGUITIES)


REGISTRATION_AMBIGUITIES = [
    (
        "Matching keys: k2 is the sum of readout_grid_mev as the registration says (not the file value edep_scintillator_mev used as the "
        "energy key in EXP-015); k1 is the count of readout cells with a positive deposit."
    ),
    (
        "Standardisation: logs of the two keys, then mean and population standard deviation (ddof = 0) over ALL fiducial events of both "
        "classes with a positive cell count and energy, before cut (a) is applied (the registration says 'pooled fiducial events')."
    ),
    (
        "Common support: 'within 0.25 standardised units in BOTH keys' is read as Chebyshev distance <= 0.25 (inclusive) to at least one "
        "event of the other class that also passes (a); it is evaluated once, not iterated."
    ),
    (
        "Fold seed: the stratified split of repeat r uses random_state = 20261109 + r; the HistGradientBoosting random_state of fold f "
        "of repeat r is 20261109 + 5 r + f (immaterial: no early stopping and fewer than 2e5 rows, so it is not used)."
    ),
    (
        "Class-balanced sample weights are computed on the training fold only (sklearn 'balanced') and are also given to the "
        "logistic regression (the registration is silent on its weights); the standardiser is fitted on the training fold only."
    ),
    (
        "Nadeau-Bengio ratio n_test/n_train: the ratio of the mean actual test size to the mean actual training size over the 100 "
        "folds (about 0.25); s^2 is the sample variance (ddof = 1) of the 100 deltas; the interval is two-sided 95 % with t on 99 "
        "degrees of freedom, and the decision rule uses that interval's limits."
    ),
    (
        "Negative control: all 18 multiscale columns are permuted jointly (one row permutation per repeat, generator "
        "default_rng([20261109, 18, r])) over the events of the hard subset and reused by all five folds of the repeat. The "
        "registration names 'invalid' only for an upper limit below 0; an interval that excludes 0 on the worse side is reported as "
        "'not passed' (the control 'must contain 0'), and the primary result is then not to be read."
    ),
    (
        "Same-class null: uses the hard-subset electrons (not all fiducial electrons); one stratified 5-fold cross-fit per repeat "
        "(fold seed 20261109 + 1000 + r, random balanced split generator default_rng([20261109, 19, r])); the AUC is that of the pooled "
        "out-of-fold scores; the standard error is the Hanley-McNeil null value sqrt((n1 + n2 + 1) / (12 n1 n2)); the check passes when "
        "every repeat of both models lies within 0.5 +- 3 SE (the mean over the repeats is also reported)."
    ),
    (
        "AUC difference and the protons passing at 90 % electron efficiency use, for every repeat, the pooled out-of-fold scores of "
        "that repeat; the threshold is the order statistic of rejection_limits.electron_threshold on those electron scores (protons "
        "never enter it). Summaries are the mean and range over the 20 repeats (the repeats reuse the same events, so they are not "
        "independent); the Clopper-Pearson one-sided 95 % limit is reported per repeat for k = 0."
    ),
    (
        "theta <= 12 deg: the stratum is taken within the hard subset as defined (standardisation and common support are not "
        "recomputed within the stratum); the test is run only if both classes keep >= 100 events."
    ),
    (
        "Full fiducial sample: all fiducial electrons and truth-interacting protons, with the registered per-window eligibility "
        "masking of the multiscale features (NaN below 336 / 572 occupied cells), handled by the native NaN support of "
        "HistGradientBoosting; missingness can leak class information (found in the S0 audit), so this secondary view is expected to "
        "be trivially separable and says nothing about the multiscale features. The logistic model is not run on it (it needs "
        "imputation)."
    ),
    (
        "Secondary logistic model: same hard subset, same folds and weights as the primary HistGradientBoosting comparison."
    ),
    (
        "Ablations (C + D0 only, C + D1 and D2 only): same folds as the primary; descriptive, interval reported with the same "
        "Nadeau-Bengio formula but no decision."
    ),
    (
        "Figure 2 shows the out-of-fold scores of repeat 0 only (log-odds axis), with that repeat's 90 % electron-efficiency threshold."
    ),
]


# --- registration check ---------------------------------------------------------------------------------------------------------------


def registration_info(path: Path = REGISTRATION_PATH) -> dict[str, str]:
    """SHA-256 of the registration file; refuses to run if the prefix is not the registered one."""

    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if not digest.startswith(REGISTRATION_SHA256_PREFIX):
        raise RuntimeError(
            f"registration file {path} has SHA-256 {digest}, expected prefix {REGISTRATION_SHA256_PREFIX}: refusing to run"
        )
    return {
        "path": str(path),
        "sha256": digest,
        "sha256_prefix_registered": REGISTRATION_SHA256_PREFIX,
    }


def git_commit(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "working_tree_dirty": bool(
            run("status", "--porcelain", "--untracked-files=no")
        ),
    }


# --- 2. hard subset -------------------------------------------------------------------------------------------------------------------


def hard_subset_masks(
    cells_electron,
    energy_electron,
    cells_proton,
    energy_proton,
    minimum_cells: int = MINIMUM_OCCUPIED_CELLS,
    caliper: float = CALIPER,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks over the (fiducial) electrons and protons given: (a) at least ``minimum_cells`` occupied cells, and (b) some event
    of the other class that also passes (a) within ``caliper`` standardised units in BOTH keys (Chebyshev distance).

    Keys are k1 = log(occupied cells) and k2 = log(energy), each standardised over the pooled events of both classes. Only the two
    keys enter; no classifier output is used.
    """

    ce, ee = np.asarray(cells_electron, float), np.asarray(energy_electron, float)
    cp, ep = np.asarray(cells_proton, float), np.asarray(energy_proton, float)
    usable_e, usable_p = (ce > 0) & (ee > 0), (cp > 0) & (ep > 0)

    def keys(c, e, usable):
        out = np.zeros((len(c), 2))
        out[usable] = np.column_stack([np.log(c[usable]), np.log(e[usable])])
        return out

    ke, kp = keys(ce, ee, usable_e), keys(cp, ep, usable_p)
    pooled = np.vstack([ke[usable_e], kp[usable_p]])
    mean, std = (
        pooled.mean(axis=0),
        pooled.std(axis=0),
    )  # REGISTRATION AMBIGUITY: population SD (ddof = 0)
    ze, zp = (ke - mean) / std, (kp - mean) / std
    pass_e, pass_p = usable_e & (ce >= minimum_cells), usable_p & (cp >= minimum_cells)
    keep_e, keep_p = np.zeros(len(ce), bool), np.zeros(len(cp), bool)
    if pass_e.any() and pass_p.any():
        index_e, index_p = np.flatnonzero(pass_e), np.flatnonzero(pass_p)
        # REGISTRATION AMBIGUITY: "within 0.25 in BOTH keys" read as Chebyshev distance <= 0.25 (inclusive)
        distance_to_proton, _ = cKDTree(zp[index_p]).query(ze[index_e], p=np.inf)
        distance_to_electron, _ = cKDTree(ze[index_e]).query(zp[index_p], p=np.inf)
        keep_e[index_e[distance_to_proton <= caliper]] = True
        keep_p[index_p[distance_to_electron <= caliper]] = True
    return keep_e, keep_p


# --- data -----------------------------------------------------------------------------------------------------------------------------


@dataclass
class FiducialClass:
    """Fiducial events of one class with the quantities the hard subset and the features need."""

    name: str
    keep: np.ndarray  # row indices into the ClassEvents arrays
    images: rra.ClassImages
    cells: np.ndarray
    energy_mev: np.ndarray  # sum of readout_grid_mev (NOT the file value)
    theta_deg: np.ndarray
    band_lower_gev: np.ndarray


def gather_class(
    events: rra.ClassEvents, selected: np.ndarray, name: str, class_code: int
) -> FiducialClass:
    images = rra.build_class_images(events, selected, name, class_code)
    keep = np.flatnonzero(selected & rra.fiducial_of(events))
    assert len(keep) == len(images.grids)
    # REGISTRATION AMBIGUITY: k2 = sum of readout_grid_mev (images.energy_mev holds the file value, so it is not used)
    return FiducialClass(
        name=name,
        keep=keep,
        images=images,
        cells=np.asarray(images.cells, float),
        energy_mev=images.grids.sum(axis=(1, 2)),
        theta_deg=np.degrees(events.primary_theta_rad[keep]),
        band_lower_gev=events.band_lower_gev[keep],
    )


def conventional_matrix(grids: np.ndarray) -> np.ndarray:
    names, table = mt.conventional_features(np.asarray(grids, float))
    assert len(names) == N_CONVENTIONAL
    return table


def multiscale_matrix(
    images: rra.ClassImages, index: np.ndarray, eligibility: bool
) -> np.ndarray:
    return images.dims(np.asarray(index), eligibility)


def band_table(
    electrons: FiducialClass,
    protons: FiducialClass,
    keep_e: np.ndarray,
    keep_p: np.ndarray,
) -> list[dict[str, Any]]:
    """Fiducial events, events passing (a) and hard-subset events per class and generation band (lower edge, GeV)."""

    rows = []
    for fc, keep in ((electrons, keep_e), (protons, keep_p)):
        passing = fc.cells >= MINIMUM_OCCUPIED_CELLS
        for band in sorted(set(fc.band_lower_gev.tolist())):
            in_band = fc.band_lower_gev == band
            rows.append(
                {
                    "class": fc.name,
                    "band_lower_gev": float(band),
                    "fiducial": int(in_band.sum()),
                    "passing_572_cells": int((in_band & passing).sum()),
                    "hard_subset": int((in_band & keep).sum()),
                }
            )
    return rows


# --- 4. models ------------------------------------------------------------------------------------------------------------------------


def fit_predict_fold(
    kind: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    random_state: int = 0,
) -> np.ndarray:
    """Electron probability of the test rows from a model fitted on the training rows only, with class-balanced sample weights
    computed on the training rows."""

    weights = compute_sample_weight("balanced", y_train)
    if kind == "hgb":
        model = HistGradientBoostingClassifier(
            **HGB_SETTINGS, random_state=random_state
        )
        model.fit(x_train, y_train, sample_weight=weights)
    elif kind == "logistic":
        if np.isnan(x_train).any() or np.isnan(x_test).any():
            raise ValueError("the logistic model needs finite features")
        model = make_pipeline(StandardScaler(), LogisticRegression(**LOGISTIC_SETTINGS))
        model.fit(x_train, y_train, logisticregression__sample_weight=weights)
    else:
        raise ValueError(f"unknown model {kind!r}")
    return model.predict_proba(x_test)[:, 1]


def fold_log_loss(p_electron: np.ndarray, y: np.ndarray) -> float:
    """Class-balanced log-loss of the electron probabilities of the rows with labels ``y`` (1 = electron)."""

    return balanced_log_loss(p_electron[y == 1], p_electron[y == 0])


def run_repeat(
    sets: dict[str, np.ndarray],
    y: np.ndarray,
    kind: str,
    fold_seed: int,
    repeat: int,
    n_splits: int = N_FOLDS,
) -> dict[str, Any]:
    """One repeat of stratified cross-fitting: in every fold every feature set is fitted on the same training rows and scored on the
    same test rows. Returns per-fold log-losses and the pooled out-of-fold scores of each set."""

    with threadpool_limits(limits=1):
        oof = {name: np.full(len(y), np.nan) for name in sets}
        rows = []
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=fold_seed
        )
        for fold, (train, test) in enumerate(
            splitter.split(next(iter(sets.values())), y)
        ):
            row: dict[str, Any] = {
                "repeat": repeat,
                "fold": fold,
                "n_train": len(train),
                "n_test": len(test),
            }
            for name, x in sets.items():
                # REGISTRATION AMBIGUITY: "random_state = fold seed" read as SEED + n_splits * repeat + fold
                p = fit_predict_fold(
                    kind, x[train], y[train], x[test], SEED + n_splits * repeat + fold
                )
                oof[name][test] = p
                row[f"logloss_{name}"] = fold_log_loss(p, y[test])
            rows.append(row)
    return {"rows": rows, "oof": oof}


def cross_fit(
    sets_per_repeat: list[dict[str, np.ndarray]],
    y: np.ndarray,
    kind: str,
    n_jobs: int,
    fold_seed_offset: int = 0,
) -> list[dict[str, Any]]:
    jobs = min(int(n_jobs), MAX_JOBS)
    tasks = [
        (sets, y, kind, SEED + fold_seed_offset + r, r)
        for r, sets in enumerate(sets_per_repeat)
    ]
    if jobs <= 1:
        return [run_repeat(*task) for task in tasks]
    return Parallel(n_jobs=jobs)(delayed(run_repeat)(*task) for task in tasks)


# --- 5. statistic and decision rule ---------------------------------------------------------------------------------------------------


def nadeau_bengio_interval(
    deltas: np.ndarray,
    n_test: float,
    n_train: float,
    folds: int = N_FOLDS,
    repeats: int = N_REPEATS,
    confidence: float = CONFIDENCE,
) -> dict[str, float]:
    """Mean of the per-fold differences with the corrected repeated cross-validation variance (Nadeau and Bengio 2003):
    var = (1 / (k r) + n_test / n_train) s^2, interval mean +- t_{1 - alpha/2, k r - 1} sqrt(var)."""

    d = np.asarray(deltas, float)
    kr = folds * repeats
    if len(d) != kr:
        raise ValueError(f"expected k r = {kr} fold differences, got {len(d)}")
    s2 = float(np.var(d, ddof=1))
    variance = (1.0 / kr + n_test / n_train) * s2
    t = float(stats.t.ppf(0.5 + confidence / 2.0, kr - 1))
    half = t * math.sqrt(variance)
    mean = float(d.mean())
    return {
        "mean": mean,
        "s2": s2,
        "ratio_n_test_over_n_train": float(n_test / n_train),
        "variance_of_mean": variance,
        "standard_error": math.sqrt(variance),
        "degrees_of_freedom": kr - 1,
        "t_quantile": t,
        "lower": mean - half,
        "upper": mean + half,
        "half_width": half,
    }


def decide(lower: float, upper: float) -> str:
    """Section 5 decision rule on the 95 % interval of delta = loss(B) - loss(A)."""

    if upper < 0:
        return DECISION_ADDS
    if lower > 0:
        return DECISION_WORSE
    return DECISION_NONE


def negative_control_status(lower: float, upper: float) -> str:
    """Section 6: the interval must contain 0; an upper limit below 0 means the pipeline leaks (run invalid)."""

    if upper < 0:
        return NEGATIVE_CONTROL_INVALID
    if lower > 0:
        # REGISTRATION AMBIGUITY: not 'invalid' by the registration's wording, but the control does not contain 0
        return NEGATIVE_CONTROL_WORSE
    return NEGATIVE_CONTROL_PASSED


def overall_status(decision: str, control: str, same_class_passed: bool | None) -> str:
    if control == NEGATIVE_CONTROL_INVALID:
        return "invalid"
    if control != NEGATIVE_CONTROL_PASSED:
        return "negative control not passed: the primary result is not to be read"
    if same_class_passed is False:
        return "same-class null not passed: the primary result is not to be read"
    return decision


def compare(
    results: list[dict[str, Any]], base: str, other: str, folds: int = N_FOLDS
) -> dict[str, Any]:
    """Per-fold differences loss(other) - loss(base) over all repeats and their Nadeau-Bengio interval."""

    rows = [row for result in results for row in result["rows"]]
    deltas = np.array(
        [row[f"logloss_{other}"] - row[f"logloss_{base}"] for row in rows]
    )
    interval = nadeau_bengio_interval(
        deltas,
        float(np.mean([row["n_test"] for row in rows])),
        float(np.mean([row["n_train"] for row in rows])),
        folds,
        len(results),
    )
    return {
        "base": base,
        "other": other,
        "mean_logloss_base": float(np.mean([row[f"logloss_{base}"] for row in rows])),
        "mean_logloss_other": float(np.mean([row[f"logloss_{other}"] for row in rows])),
        "deltas": deltas.tolist(),
        "interval": interval,
        "decision": decide(interval["lower"], interval["upper"]),
    }


# --- 7. secondary quantities ----------------------------------------------------------------------------------------------------------


def hanley_mcneil_null_se(n_positive: int, n_negative: int) -> float:
    """Standard error of an AUC under the null AUC = 0.5."""

    return math.sqrt((n_positive + n_negative + 1) / (12.0 * n_positive * n_negative))


def score_summary(
    results: list[dict[str, Any]], y: np.ndarray, names: list[str]
) -> dict[str, Any]:
    """AUC and the protons passing at 90 % electron efficiency of the pooled out-of-fold scores of every repeat, per feature set."""

    out: dict[str, Any] = {}
    n_protons = int((y == 0).sum())
    for name in names:
        aucs, passing, thresholds, limits = [], [], [], []
        for result in results:
            scores = result["oof"][name]
            aucs.append(float(roc_auc_score(y, scores)))
            threshold = rl.electron_threshold(scores[y == 1], ELECTRON_EFFICIENCY)
            k = int((scores[y == 0] >= threshold).sum())
            thresholds.append(float(threshold))
            passing.append(k)
            limits.append(rl.rejection_from_counts(k, n_protons, CONFIDENCE).fpr_upper)
        out[name] = {
            "auc_per_repeat": aucs,
            "auc_mean": float(np.mean(aucs)),
            "auc_sd_over_repeats": float(np.std(aucs, ddof=1))
            if len(aucs) > 1
            else float("nan"),
            "threshold_per_repeat": thresholds,
            "protons_passing_per_repeat": passing,
            "protons": n_protons,
            "protons_passing_mean": float(np.mean(passing)),
            "protons_passing_min": int(min(passing)),
            "protons_passing_max": int(max(passing)),
            "repeats_with_zero_passing": int(sum(k == 0 for k in passing)),
            "false_positive_rate_clopper_pearson_upper_95_per_repeat": limits,
            "censored_upper_limit_fpr_when_zero_pass": float(
                rl.clopper_pearson_upper(0, n_protons, CONFIDENCE)
            ),
        }
    return out


def auc_difference(summary: dict[str, Any], base: str, other: str) -> dict[str, float]:
    diff = np.array(summary[other]["auc_per_repeat"]) - np.array(
        summary[base]["auc_per_repeat"]
    )
    return {
        "mean": float(diff.mean()),
        "sd_over_repeats": float(diff.std(ddof=1)) if len(diff) > 1 else float("nan"),
        "min": float(diff.min()),
        "max": float(diff.max()),
    }


def permute_multiscale(multiscale: np.ndarray, repeat: int) -> np.ndarray:
    """The 18 multiscale columns permuted JOINTLY across events (one fixed row permutation per repeat), breaking their link to the
    label while keeping their mutual structure."""

    order = np.random.default_rng([SEED, 18, repeat]).permutation(len(multiscale))
    return multiscale[order]


def same_class_null(
    conventional: np.ndarray, multiscale: np.ndarray, repeats: int = SAME_CLASS_REPEATS
) -> dict[str, Any]:
    """Electrons split at random into two pseudo-classes; models A and B must give AUC 0.5 within 3 standard errors."""

    n = len(conventional)
    n1 = n // 2
    se = hanley_mcneil_null_se(n1, n - n1)
    both = np.column_stack([conventional, multiscale])
    per_model: dict[str, list[float]] = {"A": [], "B": []}
    for r in range(repeats):
        y = np.zeros(n, int)
        y[np.random.default_rng([SEED, 19, r]).permutation(n)[:n1]] = 1
        result = run_repeat(
            {"A": conventional, "B": both}, y, "hgb", SEED + 1000 + r, r
        )
        for name, values in per_model.items():
            values.append(float(roc_auc_score(y, result["oof"][name])))
    passed = all(
        abs(a - 0.5) <= 3 * se for values in per_model.values() for a in values
    )
    return {
        "n_electrons": n,
        "pseudo_class_sizes": [n1, n - n1],
        "standard_error_null": se,
        "auc_per_repeat": per_model,
        "auc_mean": {k: float(np.mean(v)) for k, v in per_model.items()},
        "max_abs_deviation_in_standard_errors": float(
            max(abs(a - 0.5) for values in per_model.values() for a in values) / se
        ),
        "passed": bool(passed),
    }


# --- analysis ------------------------------------------------------------------------------------------------------------------------


def primary_sets(
    conventional: np.ndarray, multiscale: np.ndarray
) -> dict[str, np.ndarray]:
    """A (C), B (C + M) and the two ablations; ``multiscale`` columns are ordered window by window as D0, D1, D2."""

    d0 = multiscale[:, 0::3]
    d12 = np.column_stack([multiscale[:, 1::3], multiscale[:, 2::3]])
    return {
        "A": conventional,
        "B": np.column_stack([conventional, multiscale]),
        "ablation_D0": np.column_stack([conventional, d0]),
        "ablation_D1_D2": np.column_stack([conventional, d12]),
    }


def _fold_csv_rows(
    analysis: str, results: list[dict[str, Any]], base: str, other: str
) -> list[dict[str, Any]]:
    return [
        {
            "analysis": analysis,
            "repeat": row["repeat"],
            "fold": row["fold"],
            "n_train": row["n_train"],
            "n_test": row["n_test"],
            "logloss_base": row[f"logloss_{base}"],
            "logloss_comparison": row[f"logloss_{other}"],
            "delta": row[f"logloss_{other}"] - row[f"logloss_{base}"],
        }
        for result in results
        for row in result["rows"]
    ]


def _compare_summary(c: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in c.items() if k != "deltas"}


def run_hard_subset_analysis(
    conventional: np.ndarray,
    multiscale: np.ndarray,
    y: np.ndarray,
    n_jobs: int,
    repeats: int = N_REPEATS,
    with_controls: bool = True,
) -> dict[str, Any]:
    """Primary comparison (HistGradientBoosting) with, optionally, the negative control and the ablations."""

    sets_per_repeat = []
    for r in range(repeats):
        sets = (
            primary_sets(conventional, multiscale)
            if with_controls
            else {
                k: v
                for k, v in primary_sets(conventional, multiscale).items()
                if k in ("A", "B")
            }
        )
        if with_controls:
            sets["B_permuted"] = np.column_stack(
                [conventional, permute_multiscale(multiscale, r)]
            )
        sets_per_repeat.append(sets)
    results = cross_fit(sets_per_repeat, y, "hgb", n_jobs)
    out: dict[str, Any] = {"results": results, "primary": compare(results, "A", "B")}
    out["scores"] = score_summary(results, y, list(sets_per_repeat[0]))
    out["auc_difference_B_minus_A"] = auc_difference(out["scores"], "A", "B")
    if with_controls:
        out["negative_control"] = compare(results, "A", "B_permuted")
        out["ablation_D0"] = compare(results, "A", "ablation_D0")
        out["ablation_D1_D2"] = compare(results, "A", "ablation_D1_D2")
    return out


def run_analysis(
    events_electron: rra.ClassEvents,
    events_proton: rra.ClassEvents,
    out: Path,
    n_jobs: int = MAX_JOBS,
    repeats: int = N_REPEATS,
    registration_path: Path = REGISTRATION_PATH,
    project_root: Path | None = None,
) -> dict[str, Any]:
    t_start = time.time()
    timings: dict[str, float] = {}

    def tick(name: str, since: float) -> float:
        timings[name] = round(time.time() - since, 2)
        return time.time()

    registration = registration_info(registration_path)
    out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "label": EXPLORATORY_LABEL,
        "experiment": "EXP-018",
        "registration": registration,
        "git": git_commit(project_root or Path(__file__).resolve().parents[3]),
        "settings": {
            "hist_gradient_boosting": HGB_SETTINGS,
            "logistic": LOGISTIC_SETTINGS,
            "folds": N_FOLDS,
            "repeats": repeats,
            "seed": SEED,
            "minimum_occupied_cells": MINIMUM_OCCUPIED_CELLS,
            "caliper_standardised_units": CALIPER,
            "n_jobs": min(n_jobs, MAX_JOBS),
        },
        "ambiguities": ambiguities(),
    }

    # --- data and hard subset
    t = time.time()
    electrons = gather_class(
        events_electron, np.ones(len(events_electron), bool), "electron", 1
    )
    protons = gather_class(
        events_proton,
        events_proton.truth_occurred.astype(bool),
        "proton_interacting",
        2,
    )
    keep_e, keep_p = hard_subset_masks(
        electrons.cells, electrons.energy_mev, protons.cells, protons.energy_mev
    )
    sizes = {"electron": int(keep_e.sum()), "proton_interacting": int(keep_p.sum())}
    result["fiducial_events"] = {
        "electron": len(electrons.keep),
        "proton_interacting": len(protons.keep),
    }
    result["hard_subset"] = {
        "sizes": sizes,
        "by_band": band_table(electrons, protons, keep_e, keep_p),
        "note": (
            "keys: k1 = log(occupied readout cells), k2 = log(sum of readout_grid_mev), standardised over the pooled fiducial "
            "events; (a) >= 572 occupied cells, (b) an event of the other class within 0.25 in both keys"
        ),
    }
    t = tick("data_and_hard_subset", t)
    testable = min(sizes.values()) >= MINIMUM_CLASS_EVENTS
    if not testable:
        result["primary"] = {
            "decision": "not testable",
            "reason": f"a class has fewer than {MINIMUM_CLASS_EVENTS} hard-subset events",
        }
        result["overall_status"] = "not testable"
        _write_outputs(out, result, [], None)
        return result

    index_e, index_p = np.flatnonzero(keep_e), np.flatnonzero(keep_p)
    conventional = np.vstack(
        [
            conventional_matrix(electrons.images.grids[index_e]),
            conventional_matrix(protons.images.grids[index_p]),
        ]
    )
    multiscale = np.vstack(
        [
            multiscale_matrix(electrons.images, index_e, eligibility=True),
            multiscale_matrix(protons.images, index_p, eligibility=True),
        ]
    )
    if not np.isfinite(multiscale).all():
        raise RuntimeError(
            "non-finite multiscale feature after cut (a): the registration expects none"
        )
    y = np.r_[np.ones(len(index_e), int), np.zeros(len(index_p), int)]
    theta = np.r_[electrons.theta_deg[index_e], protons.theta_deg[index_p]]
    t = tick("features_hard_subset", t)

    # --- primary, negative control, ablations, secondaries
    main = run_hard_subset_analysis(conventional, multiscale, y, n_jobs, repeats)
    t = tick("hist_gradient_boosting_cross_fit_hard_subset", t)
    primary, control = main["primary"], main["negative_control"]
    null = same_class_null(conventional[y == 1], multiscale[y == 1])
    t = tick("same_class_null", t)
    control_status = negative_control_status(
        control["interval"]["lower"], control["interval"]["upper"]
    )
    result["primary"] = {
        **_compare_summary(primary),
        "estimand": "mean over folds of class-balanced log-loss(B) - log-loss(A), B = C + M, A = C",
        "interval_half_width_size_that_could_have_been_seen": primary["interval"][
            "half_width"
        ],
        "n_folds_total": len(primary["deltas"]),
    }
    result["negative_control"] = {**_compare_summary(control), "status": control_status}
    result["same_class_null"] = null
    result["overall_status"] = overall_status(
        primary["decision"], control_status, null["passed"]
    )
    result["secondary"] = {
        "label": "descriptive; no decision",
        "scores": main["scores"],
        "auc_difference_B_minus_A": main["auc_difference_B_minus_A"],
        "ablation_D0_only": _compare_summary(main["ablation_D0"]),
        "ablation_D1_D2_only": _compare_summary(main["ablation_D1_D2"]),
    }
    fold_rows = (
        _fold_csv_rows("primary_hgb_B_vs_A", main["results"], "A", "B")
        + _fold_csv_rows(
            "negative_control_hgb_Bpermuted_vs_A", main["results"], "A", "B_permuted"
        )
        + _fold_csv_rows("ablation_D0_hgb", main["results"], "A", "ablation_D0")
        + _fold_csv_rows("ablation_D1_D2_hgb", main["results"], "A", "ablation_D1_D2")
    )

    # --- logistic model on the same hard subset
    logistic_sets = [
        {"A": conventional, "B": np.column_stack([conventional, multiscale])}
        for _ in range(repeats)
    ]
    logistic = cross_fit(logistic_sets, y, "logistic", n_jobs)
    logistic_compare = compare(logistic, "A", "B")
    result["secondary"]["logistic_B_vs_A"] = {
        **_compare_summary(logistic_compare),
        "scores": score_summary(logistic, y, ["A", "B"]),
    }
    fold_rows += _fold_csv_rows("logistic_B_vs_A", logistic, "A", "B")
    t = tick("logistic_cross_fit", t)

    # --- theta <= 12 deg part of the hard subset
    in_stratum = theta <= SENSITIVITY_THETA_MAX_DEG
    stratum_sizes = {
        "electron": int((in_stratum & (y == 1)).sum()),
        "proton_interacting": int((in_stratum & (y == 0)).sum()),
    }
    if min(stratum_sizes.values()) >= MINIMUM_CLASS_EVENTS:
        s = run_hard_subset_analysis(
            conventional[in_stratum],
            multiscale[in_stratum],
            y[in_stratum],
            n_jobs,
            repeats,
            with_controls=False,
        )
        result["secondary"]["theta_le_12_deg"] = {
            "sizes": stratum_sizes,
            **_compare_summary(s["primary"]),
            "scores": s["scores"],
            "auc_difference_B_minus_A": s["auc_difference_B_minus_A"],
        }
        fold_rows += _fold_csv_rows(
            "theta_le_12_deg_hgb_B_vs_A", s["results"], "A", "B"
        )
    else:
        result["secondary"]["theta_le_12_deg"] = {
            "sizes": stratum_sizes,
            "decision": "not testable",
            "reason": f"a class keeps fewer than {MINIMUM_CLASS_EVENTS} events with theta <= {SENSITIVITY_THETA_MAX_DEG:g} deg",
        }
    t = tick("theta_le_12_deg", t)

    # --- full fiducial sample, no hard-subset restriction
    all_e, all_p = np.arange(len(electrons.keep)), np.arange(len(protons.keep))
    full_c = np.vstack(
        [
            conventional_matrix(electrons.images.grids),
            conventional_matrix(protons.images.grids),
        ]
    )
    full_m = np.vstack(
        [
            multiscale_matrix(electrons.images, all_e, eligibility=True),
            multiscale_matrix(protons.images, all_p, eligibility=True),
        ]
    )
    full_y = np.r_[np.ones(len(all_e), int), np.zeros(len(all_p), int)]
    full = run_hard_subset_analysis(
        full_c, full_m, full_y, n_jobs, repeats, with_controls=False
    )
    result["secondary"]["full_fiducial_sample"] = {
        "sizes": {"electron": len(all_e), "proton_interacting": len(all_p)},
        "multiscale_features_with_missing_values_fraction": float(
            np.isnan(full_m).any(axis=1).mean()
        ),
        **_compare_summary(full["primary"]),
        "scores": full["scores"],
        "auc_difference_B_minus_A": full["auc_difference_B_minus_A"],
        "note": "no hard-subset restriction; expected trivially separable (conventional A alone separates); not about the multiscale features",
    }
    fold_rows += _fold_csv_rows(
        "full_fiducial_sample_hgb_B_vs_A", full["results"], "A", "B"
    )
    t = tick("full_fiducial_sample", t)

    result["timings_seconds"] = {**timings, "total": round(time.time() - t_start, 2)}
    _write_outputs(
        out, result, fold_rows, {"main": main, "y": y, "hard_subset_events": len(y)}
    )
    result["timings_seconds"]["total"] = round(time.time() - t_start, 2)
    rra.write_json(out / "exp018_result.json", result)
    return result


# --- outputs --------------------------------------------------------------------------------------------------------------------------


def _write_outputs(
    out: Path,
    result: dict[str, Any],
    fold_rows: list[dict[str, Any]],
    figure_data: dict[str, Any] | None,
) -> None:
    rra.write_json(out / "exp018_result.json", result)
    rra.write_csv(out / "exp018_per_fold.csv", fold_rows)
    if figure_data is not None:
        figure_per_fold_delta(
            figure_data["main"], out / "exp018_fig1_per_fold_delta.png", result
        )
        figure_scores(
            figure_data["main"],
            figure_data["y"],
            out / "exp018_fig2_out_of_fold_scores.png",
            result,
        )


def figure_per_fold_delta(
    main: dict[str, Any], path: Path, result: dict[str, Any]
) -> None:
    plt = rra._pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    panels = (
        (
            main["primary"],
            "B (C + M) minus A (C): per-fold change in class-balanced log-loss",
            "#d62728",
        ),
        (
            main["negative_control"],
            "negative control: B with the 18 M columns permuted across events, minus A",
            "#7f7f7f",
        ),
    )
    for ax, (c, title, colour) in zip(axes, panels, strict=True):
        d, iv = np.array(c["deltas"]), c["interval"]
        ax.hist(d, bins=30, color=colour, alpha=0.6)
        ax.axvspan(
            iv["lower"],
            iv["upper"],
            color="k",
            alpha=0.12,
            label="95% interval (Nadeau-Bengio)",
        )
        ax.axvline(0, color="k", lw=1)
        ax.axvline(
            iv["mean"], color="k", ls="--", lw=1, label=f"mean {iv['mean']:+.4f}"
        )
        ax.set_title(
            f"{title}\nmean {iv['mean']:+.4f}, 95% interval [{iv['lower']:+.4f}, {iv['upper']:+.4f}]: {c['decision']}",
            fontsize=9,
        )
        ax.set_ylabel("folds (100)")
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel(
        "delta = log-loss(B) - log-loss(A) on the test fold (negative: B better)"
    )
    fig.suptitle(f"EXP-018 per-fold deltas\n{EXPLORATORY_LABEL}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_scores(
    main: dict[str, Any], y: np.ndarray, path: Path, result: dict[str, Any]
) -> None:
    plt = rra._pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    edges = np.linspace(-14, 14, 57)
    for ax, name, label in zip(
        axes,
        ("A", "B"),
        ("A: conventional (24)", "B: conventional + multiscale (42)"),
        strict=True,
    ):
        scores = main["results"][0]["oof"][name]
        logit = np.log(np.clip(scores, 1e-6, 1 - 1e-6) / np.clip(1 - scores, 1e-6, 1))
        ax.hist(
            np.clip(logit[y == 1], -13.9, 13.9),
            bins=edges,
            histtype="stepfilled",
            alpha=0.5,
            color="#1f77b4",
            label=f"electrons ({(y == 1).sum()})",
        )
        ax.hist(
            np.clip(logit[y == 0], -13.9, 13.9),
            bins=edges,
            histtype="stepfilled",
            alpha=0.5,
            color="#d62728",
            label=f"protons ({(y == 0).sum()})",
        )
        threshold = rl.electron_threshold(scores[y == 1], ELECTRON_EFFICIENCY)
        k = int((scores[y == 0] >= threshold).sum())
        ax.axvline(
            np.log(
                np.clip(threshold, 1e-6, 1 - 1e-6) / np.clip(1 - threshold, 1e-6, 1)
            ),
            color="k",
            ls="--",
            label=f"90% electron efficiency; {k} protons pass",
        )
        ax.set_title(
            f"{label}\nAUC {main['scores'][name]['auc_per_repeat'][0]:.3f}", fontsize=9
        )
        ax.set_xlabel("out-of-fold log-odds of 'electron' (repeat 0)")
        ax.set_ylim(
            0, ax.get_ylim()[1] * 1.4
        )  # head-room so the legend does not cover the threshold line
        ax.legend(fontsize=7, loc="upper center")
    axes[0].set_ylabel("events")
    fig.suptitle(
        f"EXP-018 out-of-fold scores in the hard subset (repeat 0 of 20)\n{EXPLORATORY_LABEL}",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--registration", type=Path, default=REGISTRATION_PATH)
    parser.add_argument("--n-jobs", type=int, default=MAX_JOBS)
    args = parser.parse_args(argv)
    registration_info(
        args.registration
    )  # abort before loading anything if the registration changed
    events = {}
    for name in ("electron", "proton"):
        directory = rra.find_class_directory(args.data_root, name, "rung1")
        if directory is None:
            raise SystemExit(f"no rung-1 {name} batches below {args.data_root}")
        events[name] = rra.load_class_events(directory, name)
        print(f"{name}: {directory}")
    result = run_analysis(
        events["electron"],
        events["proton"],
        args.out,
        args.n_jobs,
        registration_path=args.registration,
    )
    print(
        "overall status:",
        result["overall_status"],
        "| hard subset:",
        result["hard_subset"]["sizes"],
    )


if __name__ == "__main__":
    main()
