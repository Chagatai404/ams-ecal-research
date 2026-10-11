"""Cross-fitting with training variance, role-separated fitting and the standard baselines (gates SV-2, SV-3, SV-7; review C2, C7).

* ``repeated_cross_fit``: the registered classifier scored out of fold under several fold seeds, reporting the spread. A single
  fold seed hides training and splitting variance (C7).
* ``fit_calibrate_evaluate``: ONE model trained on the training events only, scoring the calibration electrons and the evaluation
  electrons and protons it has never seen; the roles are checked disjoint and handed to ``rejection_limits.evaluate_at_efficiency``.
  This replaces the out-of-fold pool of five models with a threshold set on the evaluated events (C2).
* ``repeated_role_splits``: ``fit_calibrate_evaluate`` over several random splits, so the interval of a claim carries the split
  variance, not only the counting error.
* ``standard_baselines``: energy-only (rank AUC and the classifier) and a standardised logistic regression beside every ladder
  (SV-7), and ``same_class_null`` (SV-7): two samples of the SAME class must score AUC 0.5.

The classifier is the registered one (``added_checks.CLASSIFIER_SETTINGS``). Nothing here tunes it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ams_ecal.inference import rejection_limits as rl
from ams_ecal.validation.added_checks import CLASSIFIER_SETTINGS, N_FOLDS


def _model(kind: str):
    if kind == "hgb":
        return HistGradientBoostingClassifier(**CLASSIFIER_SETTINGS)
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    raise ValueError(f"unknown model {kind!r}")


def out_of_fold_scores(x: np.ndarray, y: np.ndarray, seed: int, kind: str = "hgb", folds: int = N_FOLDS) -> np.ndarray:
    """Scores of every event from a model that never saw it (stratified folds, fold seed ``seed``)."""

    x, y = np.asarray(x, dtype=float), np.asarray(y)
    scores = np.zeros(len(y))
    for train, test in StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(x, y):
        usable = ~np.isnan(x[train]).all(axis=0)  # a column that is NaN in the whole training fold carries nothing
        scores[test] = _model(kind).fit(x[train][:, usable], y[train]).predict_proba(x[test][:, usable])[:, 1]
    return scores


@dataclass(frozen=True)
class CrossFitSpread:
    seeds: tuple[int, ...]
    auc: tuple[float, ...]
    mean: float
    sd: float
    minimum: float
    maximum: float


def repeated_cross_fit(x: np.ndarray, y: np.ndarray, seeds: Sequence[int] = range(5), kind: str = "hgb") -> CrossFitSpread:
    """Out-of-fold AUC for each fold seed, with the spread to be reported beside any single number."""

    aucs = tuple(float(roc_auc_score(y, out_of_fold_scores(x, y, s, kind))) for s in seeds)
    return CrossFitSpread(
        seeds=tuple(seeds),
        auc=aucs,
        mean=float(np.mean(aucs)),
        sd=float(np.std(aucs, ddof=1)) if len(aucs) > 1 else float("nan"),
        minimum=min(aucs),
        maximum=max(aucs),
    )


def fit_calibrate_evaluate(
    train_x: np.ndarray,
    train_y: np.ndarray,
    calibration_electron_x: np.ndarray,
    evaluation_electron_x: np.ndarray,
    evaluation_proton_x: np.ndarray,
    efficiency: float,
    role_keys: dict[str, np.ndarray] | None = None,
    confidence: float = rl.CONFIDENCE,
    conservative: bool = False,
    bootstrap: int = 0,
    rng: np.random.Generator | None = None,
    kind: str = "hgb",
) -> rl.EvaluationResult:
    """Train on ``train_x, train_y`` only (label 1 = electron); calibrate on electrons; evaluate on fresh electrons and protons.

    ``role_keys`` (optional) maps role names to event keys (``rejection_limits.event_key``); overlap raises before any fitting.
    """

    if role_keys is not None:
        rl.assert_disjoint_roles(role_keys)
    usable = ~np.isnan(np.asarray(train_x, dtype=float)).all(axis=0)
    model = _model(kind).fit(np.asarray(train_x, dtype=float)[:, usable], train_y)

    def score(x: np.ndarray) -> np.ndarray:
        return model.predict_proba(np.asarray(x, dtype=float)[:, usable])[:, 1]

    return rl.evaluate_at_efficiency(
        score(calibration_electron_x),
        score(evaluation_electron_x),
        score(evaluation_proton_x),
        efficiency,
        confidence=confidence,
        conservative=conservative,
        bootstrap=bootstrap,
        rng=rng,
    )


def repeated_role_splits(
    electron_x: np.ndarray,
    proton_x: np.ndarray,
    efficiency: float,
    n_repeats: int,
    seed: int = 0,
    train_fraction: float = 0.4,
    calibration_fraction: float = 0.2,
    protons_in_training: float = 0.4,
    kind: str = "hgb",
) -> list[rl.EvaluationResult]:
    """``fit_calibrate_evaluate`` on ``n_repeats`` random splits.

    Electrons: ``train_fraction`` train, ``calibration_fraction`` calibrate, the rest evaluate. Protons: ``protons_in_training``
    train, the rest evaluate. The spread of the results over repeats is the training and splitting variance.
    """

    results = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng([seed, repeat])
        e_order, p_order = rng.permutation(len(electron_x)), rng.permutation(len(proton_x))
        n_train, n_cal = int(train_fraction * len(e_order)), int(calibration_fraction * len(e_order))
        e_train, e_cal, e_eval = np.split(e_order, [n_train, n_train + n_cal])
        p_train, p_eval = np.split(p_order, [int(protons_in_training * len(p_order))])
        x_train = np.concatenate([np.asarray(electron_x)[e_train], np.asarray(proton_x)[p_train]])
        y_train = np.concatenate([np.ones(len(e_train), int), np.zeros(len(p_train), int)])
        results.append(
            fit_calibrate_evaluate(
                x_train,
                y_train,
                np.asarray(electron_x)[e_cal],
                np.asarray(electron_x)[e_eval],
                np.asarray(proton_x)[p_eval],
                efficiency,
                role_keys={"train_electrons": e_train, "calibration_electrons": e_cal, "evaluation_electrons": e_eval},
                kind=kind,
            )
        )
    return results


def standard_baselines(
    features: np.ndarray,
    names: Sequence[str],
    y: np.ndarray,
    energy_column: str = "log_total_energy",
    seeds: Sequence[int] = range(3),
) -> dict[str, float]:
    """AUC of the baselines every ladder must be shown beside: total energy alone and a standardised logistic regression."""

    features = np.asarray(features, dtype=float)
    column = list(names).index(energy_column)
    return {
        "energy_rank_auc": float(roc_auc_score(y, features[:, column])),
        "energy_only_classifier": repeated_cross_fit(features[:, [column]], y, seeds, "hgb").mean,
        "logistic_all_features": repeated_cross_fit(features, y, seeds, "logistic").mean,
    }


def same_class_null(x_a: np.ndarray, x_b: np.ndarray, seeds: Sequence[int] = range(3), tolerance: float = 0.02) -> dict:
    """Two samples of the same class: the cross-fitted AUC must be 0.5 within ``tolerance`` or the pipeline is not sound."""

    x = np.concatenate([np.asarray(x_a, dtype=float), np.asarray(x_b, dtype=float)])
    y = np.concatenate([np.ones(len(x_a), int), np.zeros(len(x_b), int)])
    spread = repeated_cross_fit(x, y, seeds)
    return {"mean_auc": spread.mean, "auc": spread.auc, "within_tolerance": bool(abs(spread.mean - 0.5) <= tolerance)}
