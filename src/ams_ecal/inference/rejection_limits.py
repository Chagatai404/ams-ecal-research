"""Proton rejection at a fixed electron efficiency, reported as counts with exact limits (validity gates SV-1 and SV-2; review C1, C2).

The problem this replaces. ``multiscale_ladder.summarise_scores`` computed the proton false-positive rate on the same events that
set the threshold and, when no proton passed, reported ``rejection = number of protons`` with a degenerate bootstrap interval
[N, N]. "0 of 9,696 protons passed" does not mean the rejection is 9,696: it means the rejection is above about 3,237 at 95%
confidence. Everything here reports k of N and an exact bound.

* **Counts and Clopper-Pearson limits** (``clopper_pearson_*``, ``rejection_from_counts``). With k protons passing out of N
  independent protons at a threshold fixed independently of them, the one-sided 95% upper bound on the false-positive rate is
  the Beta quantile ``Beta(k + 1, N - k).ppf(0.95)`` (``1 - 0.05**(1/N)``, about 3/N, for k = 0). The rejection is its inverse
  and is a **lower limit**; for k = 0 it is *censored*: no point estimate exists and none is returned.
* **Sample sizes** (``protons_needed``): how many independent protons a claimed rejection needs, for k = 0, 1, 2, ... failures.
* **Four data roles** (SV-2): train, calibrate the operating point (electrons only), evaluate (fresh electrons for the achieved
  efficiency, fresh protons for the false-positive rate). ``assert_disjoint_roles`` refuses overlapping events, and
  ``evaluate_at_efficiency`` takes the three score arrays as separate arguments, so an evaluation event cannot reach the threshold.
* **Threshold uncertainty.** A threshold taken as an order statistic of m calibration electrons gives, on fresh electrons, an
  efficiency that is Beta distributed (``threshold_efficiency_interval``). ``conservative_threshold`` picks the order statistic
  whose one-sided lower bound still reaches the target efficiency; ``bootstrap`` in ``evaluate_at_efficiency`` propagates the
  calibration uncertainty into the rejection limit.

Assumptions that remain with the caller: the evaluation protons are independent, are the population the claim is about (all
protons and truth-interacting protons give different rejections), and are not used for anything before the count.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import log

import numpy as np
from scipy import stats

CONFIDENCE = 0.95


class RolesOverlapError(ValueError):
    """Two data roles share events: the operating point or the training would be evaluated on its own data."""


# --- exact binomial limits -------------------------------------------------------------------------------------------------


def clopper_pearson_upper(k: int, n: int, confidence: float = CONFIDENCE) -> float:
    """One-sided upper confidence bound on a binomial probability from k successes in n trials (1.0 if k == n)."""

    _check_counts(k, n)
    return 1.0 if k >= n else float(stats.beta.ppf(confidence, k + 1, n - k))


def clopper_pearson_lower(k: int, n: int, confidence: float = CONFIDENCE) -> float:
    """One-sided lower confidence bound (0.0 if k == 0)."""

    _check_counts(k, n)
    return 0.0 if k <= 0 else float(stats.beta.ppf(1.0 - confidence, k, n - k + 1))


def clopper_pearson_interval(k: int, n: int, confidence: float = CONFIDENCE) -> tuple[float, float]:
    """Two-sided exact interval with the given total confidence."""

    _check_counts(k, n)
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if k <= 0 else float(stats.beta.ppf(tail, k, n - k + 1))
    upper = 1.0 if k >= n else float(stats.beta.ppf(1.0 - tail, k + 1, n - k))
    return lower, upper


def _check_counts(k: int, n: int) -> None:
    if n <= 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n and n > 0, got k={k}, n={n}")


@dataclass(frozen=True)
class RejectionResult:
    """Proton rejection from counts. ``rejection_lower_limit`` is always defined; ``rejection_estimate`` only if k > 0."""

    k: int
    n: int
    confidence: float
    fpr_estimate: float
    fpr_upper: float  # one-sided
    rejection_lower_limit: float  # 1 / fpr_upper, one-sided at ``confidence``
    rejection_estimate: float | None  # n / k, None when k == 0 (censored)
    rejection_two_sided: tuple[float, float] | None  # (lower, upper) at ``confidence``; None when k == 0

    @property
    def censored(self) -> bool:
        return self.k == 0

    def statement(self) -> str:
        """The sentence a result may use: a bound when censored, an estimate with an interval otherwise."""

        if self.censored:
            return f"rejection > {self.rejection_lower_limit:,.0f} ({self.confidence:.0%} CL; 0 of {self.n:,} protons passed)"
        lo, hi = self.rejection_two_sided or (float("nan"), float("nan"))
        return (
            f"rejection {self.rejection_estimate:,.0f} ({self.confidence:.0%} CI {lo:,.0f} to {hi:,.0f}; "
            f"{self.k} of {self.n:,} protons passed)"
        )


def rejection_from_counts(k: int, n: int, confidence: float = CONFIDENCE) -> RejectionResult:
    """Rejection of the protons from k passing out of n, with exact limits. Never returns ``n`` for k == 0."""

    upper = clopper_pearson_upper(k, n, confidence)
    two_sided = None
    if k > 0:
        lo, hi = clopper_pearson_interval(k, n, confidence)
        two_sided = (1.0 / hi, np.inf if lo == 0.0 else 1.0 / lo)
    return RejectionResult(
        k=k,
        n=n,
        confidence=confidence,
        fpr_estimate=k / n,
        fpr_upper=upper,
        rejection_lower_limit=1.0 / upper,
        rejection_estimate=None if k == 0 else n / k,
        rejection_two_sided=two_sided,
    )


def protons_needed(target_rejection: float, k: int = 0, confidence: float = CONFIDENCE) -> int:
    """Smallest number of independent protons for which k passing still proves ``rejection > target`` at ``confidence``."""

    if target_rejection <= 1.0:
        raise ValueError("target rejection must exceed 1")
    if k == 0:  # closed form: 1 - (1 - c)**(1/n) <= 1/target
        return int(np.ceil(log(1.0 - confidence) / log(1.0 - 1.0 / target_rejection)))
    low, high = k + 1, 2 * k + 2
    while rejection_from_counts(k, high, confidence).rejection_lower_limit < target_rejection:
        low, high = high, high * 2
    while low < high:
        mid = (low + high) // 2
        if rejection_from_counts(k, mid, confidence).rejection_lower_limit >= target_rejection:
            high = mid
        else:
            low = mid + 1
    return int(low)


# --- the operating point ---------------------------------------------------------------------------------------------------


def event_key(particle_code: int, energy_gev: float, event_index: np.ndarray | int) -> np.ndarray:
    """One integer identity per event (particle, energy, index), to test that roles are disjoint. Index must stay below 10**7."""

    index = np.asarray(event_index, dtype=np.int64)
    if np.any(index < 0) or np.any(index >= 10**7):
        raise ValueError("event_index must be in [0, 10**7)")
    return np.int64(particle_code) * 10**14 + np.int64(round(energy_gev * 1000)) * 10**7 + index


def assert_disjoint_roles(roles: Mapping[str, np.ndarray]) -> None:
    """Raise ``RolesOverlapError`` if any two named roles (train, calibration, evaluation, ...) share an event key."""

    names = list(roles)
    problems = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = np.intersect1d(np.asarray(roles[a]), np.asarray(roles[b]))
            if len(shared):
                problems.append(f"{a} and {b} share {len(shared)} events")
    if problems:
        raise RolesOverlapError("; ".join(problems))


def electron_threshold(calibration_electron_scores: np.ndarray, efficiency: float) -> float:
    """Score at or above which at least ``efficiency`` of the calibration electrons lie (an order statistic, never interpolated)."""

    scores = np.sort(np.asarray(calibration_electron_scores, dtype=float))
    m = len(scores)
    if m == 0 or not 0.0 < efficiency < 1.0:
        raise ValueError("need calibration electrons and 0 < efficiency < 1")
    return float(scores[int(np.floor((1.0 - efficiency) * m))])


def threshold_efficiency_interval(
    n_calibration: int, efficiency: float, confidence: float = CONFIDENCE
) -> tuple[float, float]:
    """Two-sided interval for the efficiency that the order-statistic threshold of ``electron_threshold`` gives on FRESH electrons.

    The r-th smallest of m calibration scores (r = floor((1 - eff) m) + 1) has a fresh-electron efficiency distributed as
    Beta(m - r + 1, r), whatever the score distribution (continuous scores, no ties).
    """

    m = n_calibration
    r = int(np.floor((1.0 - efficiency) * m)) + 1
    tail = (1.0 - confidence) / 2.0
    return (float(stats.beta.ppf(tail, m - r + 1, r)), float(stats.beta.ppf(1.0 - tail, m - r + 1, r)))


def conservative_threshold(
    calibration_electron_scores: np.ndarray, efficiency: float, confidence: float = CONFIDENCE
) -> float:
    """The largest order-statistic threshold whose fresh-electron efficiency is at least ``efficiency`` with ``confidence``."""

    scores = np.sort(np.asarray(calibration_electron_scores, dtype=float))
    m = len(scores)
    r = int(np.floor((1.0 - efficiency) * m)) + 1
    while r > 1 and stats.beta.ppf(1.0 - confidence, m - r + 1, r) < efficiency:
        r -= 1
    return float(scores[r - 1])


@dataclass(frozen=True)
class EvaluationResult:
    threshold: float
    n_calibration_electrons: int
    achieved_efficiency: float  # on the fresh evaluation electrons
    achieved_efficiency_interval: tuple[float, float]  # exact two-sided
    rejection: RejectionResult
    rejection_lower_limit_bootstrap: tuple[float, float] | None  # 5th, 95th percentile over calibration resamples


def _threshold(scores: np.ndarray, efficiency: float, confidence: float, conservative: bool) -> float:
    return conservative_threshold(scores, efficiency, confidence) if conservative else electron_threshold(scores, efficiency)


def evaluate_at_efficiency(
    calibration_electron_scores: np.ndarray,
    evaluation_electron_scores: np.ndarray,
    evaluation_proton_scores: np.ndarray,
    efficiency: float,
    confidence: float = CONFIDENCE,
    conservative: bool = False,
    bootstrap: int = 0,
    rng: np.random.Generator | None = None,
) -> EvaluationResult:
    """Threshold from calibration electrons only; efficiency from fresh electrons; k of N from fresh protons.

    The three score arrays are distinct arguments and must come from models that never saw the evaluation events (the caller's
    duty, checkable with ``assert_disjoint_roles``). ``bootstrap`` > 0 resamples the calibration electrons, recomputes the
    threshold and the rejection lower limit each time, and returns the 5th-95th percentile range of that limit.
    """

    cal = np.asarray(calibration_electron_scores, dtype=float)
    protons = np.asarray(evaluation_proton_scores, dtype=float)
    electrons = np.asarray(evaluation_electron_scores, dtype=float)
    threshold = _threshold(cal, efficiency, confidence, conservative)
    kept = int((electrons >= threshold).sum())
    lower, upper = clopper_pearson_interval(kept, len(electrons), confidence)
    result = rejection_from_counts(int((protons >= threshold).sum()), len(protons), confidence)
    spread = None
    if bootstrap > 0:
        rng = rng or np.random.default_rng(0)
        limits = []
        for _ in range(bootstrap):
            t = _threshold(rng.choice(cal, len(cal)), efficiency, confidence, conservative)
            limits.append(rejection_from_counts(int((protons >= t).sum()), len(protons), confidence).rejection_lower_limit)
        spread = (float(np.percentile(limits, 5)), float(np.percentile(limits, 95)))
    return EvaluationResult(
        threshold=threshold,
        n_calibration_electrons=len(cal),
        achieved_efficiency=kept / len(electrons),
        achieved_efficiency_interval=(lower, upper),
        rejection=result,
        rejection_lower_limit_bootstrap=spread,
    )
