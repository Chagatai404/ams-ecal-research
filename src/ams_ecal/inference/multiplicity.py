"""Multiple comparisons: one named primary per endpoint family, adjusted secondaries, control across families (gate SV-6; review C13).

The development run reported 192 KS verdicts (79 "material") with no adjustment. The rule from the redesign is:

* every endpoint family (P1 physics difference, P2 readout survival, P3 simulator fidelity, P4 realistic performance) has ONE
  named primary, tested at the family's alpha;
* the primaries are corrected against each other (``holm``: family-wise error across families);
* secondaries are reported with ``benjamini_hochberg`` q-values (false-discovery rate) inside their family, or with the
  Westfall-Young step-down maxT adjusted p-values when the permutation draws of all statistics are available, which respects
  their dependence (``westfall_young_stepdown``);
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
    secondary_q: dict[str, float]  # Benjamini-Hochberg inside the family
    secondary_status: dict[str, str]  # "exploratory" unless the family primary is confirmed and q <= alpha


def decide(
    primaries: Mapping[str, tuple[str, float]],
    secondaries: Mapping[str, Mapping[str, float]] | None = None,
    alpha: float = 0.05,
) -> dict[str, FamilyDecision]:
    """``primaries``: family -> (named primary statistic, its p-value). ``secondaries``: family -> {statistic: p-value}."""

    families = list(primaries)
    raw = np.array([primaries[f][1] for f in families])
    adjusted = holm(raw)
    out: dict[str, FamilyDecision] = {}
    for family, adj in zip(families, adjusted, strict=True):
        name, p = primaries[family]
        sec = dict((secondaries or {}).get(family, {}))
        q = dict(zip(sec, benjamini_hochberg(np.array(list(sec.values()))), strict=True)) if sec else {}
        confirmed = bool(adj <= alpha)
        out[family] = FamilyDecision(
            family=family,
            primary=name,
            primary_p=float(p),
            primary_adjusted_p=float(adj),
            confirmed=confirmed,
            secondary_q={k: float(v) for k, v in q.items()},
            secondary_status={k: ("supported" if confirmed and v <= alpha else "exploratory") for k, v in q.items()},
        )
    return out
