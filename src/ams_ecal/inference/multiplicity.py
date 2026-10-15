"""Multiple comparisons: one named primary per endpoint family, adjusted secondaries, control across families (gate SV-6; review C13).

The development run reported 192 KS verdicts (79 "material") with no adjustment. The rule from the redesign is:

* every endpoint family (P1 physics difference, P2 readout survival, P3 simulator fidelity, P4 realistic performance) has ONE
  named primary, tested at the family's alpha;
* the primaries are corrected against each other (``holm``: family-wise error across families);
* secondaries are decided inside their family: ``westfall_young_stepdown`` when the joint permutation draws of all the
  family's statistics are available (researcher's preference 2026-10-15: no extra compute over Holm, more power; its
  validity needs a joint-exchangeability argument, Westfall and Troendle 2008 Assumption C, and a family-wise-error check
  on same-class data for the family's statistics before a claim), otherwise ``holm`` (strong family-wise control for any
  dependence). Holm is always reported beside it as the sensitivity; ``benjamini_hochberg`` q-values are an exploratory view
  and never decide (see research/plans/2026-10-15_s3_decisions_literature_record.md);
* a secondary is **exploratory** whatever its q-value unless its family's primary was confirmed (``decide``).

Inputs are p-values or permutation draws; the tests, the strata and the statistic belong to ``permutation_nulls``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values), valid for independent or positively dependent tests."""

    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    scaled = p[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def holm(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values: strong control of the family-wise error rate for any dependence."""

    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    scaled = p[order] * (m - np.arange(m))
    adjusted = np.maximum.accumulate(scaled)
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def westfall_young_stepdown(observed: np.ndarray, null_draws: np.ndarray) -> np.ndarray:
    """Westfall-Young step-down maxT adjusted p-values.

    ``observed`` has shape (m,) and ``null_draws`` shape (B, m): the m statistics recomputed on B label permutations, the same
    permutation for all m in one row. Larger means more extreme. Strong FWER control under subset pivotality.
    """

    observed = np.asarray(observed, dtype=float)
    null = np.asarray(null_draws, dtype=float)
    b, m = null.shape
    order = np.argsort(-observed)  # most extreme first
    adjusted = np.empty(m)
    # running maxima from the least extreme upward: u_i = max over hypotheses ranked i..m
    running = np.full(b, -np.inf)
    for rank in range(m - 1, -1, -1):
        j = order[rank]
        running = np.maximum(running, null[:, j])
        adjusted[j] = (1.0 + np.sum(running >= observed[j])) / (1.0 + b)
    # enforce monotonicity along the ranking
    ordered = adjusted[order]
    adjusted[order] = np.maximum.accumulate(ordered)
    return np.minimum(adjusted, 1.0)


@dataclass(frozen=True)
class FamilyDecision:
    family: str
    primary: str
    primary_p: float
    primary_adjusted_p: float  # Holm across the families' primaries
    confirmed: bool
    secondary_q: dict[str, float]  # Benjamini-Hochberg inside the family: reported as an exploratory view, never the decision
    secondary_holm_p: dict[str, float]  # Holm inside the family (strong family-wise control, any dependence): the sensitivity
    secondary_adjusted_p: dict[str, float]  # the decision: Westfall-Young step-down when joint draws were given, otherwise Holm
    secondary_method: str  # "westfall_young" or "holm"
    secondary_status: dict[str, str]  # "exploratory" unless the family primary is confirmed and the adjusted p <= alpha


def decide(
    primaries: Mapping[str, tuple[str, float]],
    secondaries: Mapping[str, Mapping[str, float]] | None = None,
    alpha: float = 0.05,
    joint_draws: Mapping[str, tuple[Mapping[str, float], np.ndarray]] | None = None,
) -> dict[str, FamilyDecision]:
    """``primaries``: family -> (named primary statistic, its p-value). ``secondaries``: family -> {statistic: p-value}.

    ``joint_draws``: family -> ({statistic: observed value}, null draws of shape (B, m) in the same statistic order, one
    label permutation per row for all m statistics). A family with joint draws is adjusted by ``westfall_young_stepdown``
    (researcher's preference 2026-10-15; its validity needs the joint-exchangeability argument and a family-wise-error
    check on same-class data for that family's statistics before a claim); every other family, and always the
    ``secondary_holm_p`` sensitivity column, uses ``holm``.
    """

    families = list(primaries)
    raw = np.array([primaries[f][1] for f in families])
    adjusted = holm(raw)
    out: dict[str, FamilyDecision] = {}
    for family, adj in zip(families, adjusted, strict=True):
        name, p = primaries[family]
        sec = dict((secondaries or {}).get(family, {}))
        q = dict(zip(sec, benjamini_hochberg(np.array(list(sec.values()))), strict=True)) if sec else {}
        holm_p = dict(zip(sec, holm(np.array(list(sec.values()))), strict=True)) if sec else {}
        method, adjusted_p = "holm", holm_p
        if sec and joint_draws and family in joint_draws:
            observed_by_name, null = joint_draws[family]
            if list(observed_by_name) != list(sec):
                raise ValueError(f"family {family}: joint draws must list the same statistics in the same order as the secondaries")
            adjusted_p = dict(
                zip(sec, westfall_young_stepdown(np.array(list(observed_by_name.values())), np.asarray(null)), strict=True)
            )
            method = "westfall_young"
        confirmed = bool(adj <= alpha)
        out[family] = FamilyDecision(
            family=family,
            primary=name,
            primary_p=float(p),
            primary_adjusted_p=float(adj),
            confirmed=confirmed,
            secondary_q={k: float(v) for k, v in q.items()},
            secondary_holm_p={k: float(v) for k, v in holm_p.items()},
            secondary_adjusted_p={k: float(v) for k, v in adjusted_p.items()},
            secondary_method=method,
            secondary_status={k: ("supported" if confirmed and v <= alpha else "exploratory") for k, v in adjusted_p.items()},
        )
    return out
