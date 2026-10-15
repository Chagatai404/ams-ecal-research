"""Coverage and type-I-error simulations that validate the S1 tools (plan S1 validation: Clopper-Pearson coverage; permutation test and
family-wise control on electron-against-electron splits).

Stop rule (plan S1): if any coverage or error rate fails, no interval or p-value from the failing tool is reported until it is fixed.
Every check returns a ``passed`` flag against a tolerance written here, before the run, from Monte Carlo error alone.

* ``clopper_pearson_coverage``: the one-sided upper bound on a binomial rate covers the true rate at least ``confidence`` of the time
  (CP is conservative: coverage >= confidence, not equal), including when most draws are k = 0.
* ``rejection_limit_coverage``: the same statement for the rejection lower limit.
* ``permutation_type_one_error``: two samples from ONE distribution (or two halves of one real class): the permutation test rejects
  at about alpha, never above alpha plus three Monte Carlo standard errors.
* ``familywise_type_one_error``: m statistics on null data; Westfall-Young step-down and Holm keep the family-wise error at most
  alpha (plus Monte Carlo error); the unadjusted minimum p-value does not (shown, as the reason for the adjustment).
* ``shuffle_excess_interval_coverage``: the event-bootstrap interval of the median cell-shuffle excess covers the true excess (0 in an
  exchangeable world) at least at its confidence.
* ``main``: runs all of it, including real development-large electron halves, and writes ``results/inference/coverage.json``.

    python -m ams_ecal.inference.coverage_simulations --energies 20 100
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ams_ecal.electron_model.em_production import PROJECT_ROOT
from ams_ecal.inference import multiplicity as mp
from ams_ecal.inference import permutation_nulls as pn
from ams_ecal.inference import rejection_limits as rl

RESULTS_PATH = PROJECT_ROOT / "results" / "inference" / "coverage.json"
SEED = 20261012
FEATURES = ("log_total_energy", "occupied_cells", "layer_cog", "width_cells", "core_fraction")


def _mc_tolerance(p: float, trials: int, sigmas: float = 3.0) -> float:
    return sigmas * float(np.sqrt(max(p * (1 - p), 1e-12) / trials))


def clopper_pearson_coverage(
    true_rate: float, n: int, confidence: float, trials: int, rng: np.random.Generator
) -> dict[str, float | bool]:
    k = rng.binomial(n, true_rate, size=trials)
    uppers = {int(v): rl.clopper_pearson_upper(int(v), n, confidence) for v in np.unique(k)}
    covered = np.mean([uppers[int(v)] >= true_rate for v in k])
    return {
        "true_rate": true_rate,
        "n": n,
        "trials": trials,
        "coverage": float(covered),
        "required": confidence,
        "share_of_draws_with_k_equal_0": float(np.mean(k == 0)),
        "passed": bool(covered >= confidence - _mc_tolerance(confidence, trials)),
    }


def rejection_limit_coverage(
    true_rejection: float, n: int, confidence: float, trials: int, rng: np.random.Generator
) -> dict[str, float | bool]:
    """Does ``rejection > lower limit`` hold at the stated confidence? (the limit must lie below the true rejection)"""

    k = rng.binomial(n, 1.0 / true_rejection, size=trials)
    limits = {int(v): rl.rejection_from_counts(int(v), n, confidence).rejection_lower_limit for v in np.unique(k)}
    covered = np.mean([limits[int(v)] <= true_rejection for v in k])
    return {
        "true_rejection": true_rejection,
        "n": n,
        "trials": trials,
        "coverage": float(covered),
        "required": confidence,
        "share_of_draws_censored": float(np.mean(k == 0)),
        "passed": bool(covered >= confidence - _mc_tolerance(confidence, trials)),
    }


def permutation_type_one_error(
    draw_pair: Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray]],
    trials: int,
    alpha: float,
    n_permutations: int,
    rng: np.random.Generator,
    strata_of: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, float | bool]:
    p_values = []
    for _ in range(trials):
        a, b = draw_pair(rng)
        values = np.concatenate([a, b])
        labels = np.concatenate([np.ones(len(a), int), np.zeros(len(b), int)])
        strata = strata_of(values) if strata_of else None
        p_values.append(pn.permutation_pvalue(pn.stratum_ks, values, labels, strata, rng, n_permutations).p_value)
    p_arr = np.array(p_values)
    rate = float(np.mean(p_arr <= alpha))
    return {
        "alpha": alpha,
        "trials": trials,
        "rejection_rate": rate,
        "mean_p_value": float(p_arr.mean()),
        "passed": bool(rate <= alpha + _mc_tolerance(alpha, trials)),
    }


def familywise_type_one_error(
    draw_pair: Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray]],
    trials: int,
    alpha: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, float | bool]:
    """``draw_pair`` returns two (events, m) arrays of the SAME class; the m KS statistics are tested together."""

    unadjusted = holm_hits = wy_hits = 0
    for _ in range(trials):
        a, b = draw_pair(rng)
        values = np.vstack([a, b])
        labels = np.concatenate([np.ones(len(a), int), np.zeros(len(b), int)])
        m = values.shape[1]
        observed = np.array([pn.stratum_ks(values[:, j], labels, None) for j in range(m)])
        null = np.empty((n_permutations, m))
        for r in range(n_permutations):
            permuted = pn.permute_labels_within_strata(labels, None, rng)
            null[r] = [pn.stratum_ks(values[:, j], permuted, None) for j in range(m)]
        raw = np.array([(1.0 + np.sum(null[:, j] >= observed[j])) / (1.0 + n_permutations) for j in range(m)])
        unadjusted += raw.min() <= alpha
        holm_hits += mp.holm(raw).min() <= alpha
        wy_hits += mp.westfall_young_stepdown(observed, null).min() <= alpha
    tol = _mc_tolerance(alpha, trials)
    return {
        "alpha": alpha,
        "trials": trials,
        "unadjusted_minimum_p_rate": unadjusted / trials,
        "holm_rate": holm_hits / trials,
        "westfall_young_rate": wy_hits / trials,
        "passed": bool(holm_hits / trials <= alpha + tol and wy_hits / trials <= alpha + tol),
    }


def shuffle_excess_interval_coverage(
    draw_grids: Callable[[np.random.Generator], np.ndarray],
    statistic: Callable[[np.ndarray], float],
    trials: int,
    n_shuffles: int,
    n_bootstrap: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> dict[str, float | bool]:
    """Coverage of the true value 0 by the event-bootstrap interval of the median cell-shuffle excess (``paired_shuffle_excess``).

    ``draw_grids`` must draw a world whose cells are exchangeable within a layer, so the true excess is exactly 0; the interval
    must then contain 0 at least ``confidence`` of the time (minus Monte Carlo error). The same check on a world with spatial
    structure must NOT cover 0 (shown in the tests: a check that cannot fail checks nothing).
    """

    covered = 0
    for _ in range(trials):
        low, high = pn.paired_shuffle_excess(draw_grids(rng), statistic, n_shuffles, rng, n_bootstrap).median_excess_95
        covered += bool(low <= 0.0 <= high)
    rate = covered / trials
    return {
        "confidence": confidence,
        "trials": trials,
        "coverage_of_zero": float(rate),
        "passed": bool(rate >= confidence - _mc_tolerance(confidence, trials)),
    }


def _real_electron_features(energy: float) -> tuple[list[str], np.ndarray]:
    from ams_ecal.multiscale_estimators import multiscale_test as mt
    from ams_ecal.validation import sample_definitions as sd

    events = sd.load_events("electron_development_large", energy, "deposit_grid_mev", "all")
    return mt.conventional_features(events.grids)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--energies", type=float, nargs="+", default=[20.0, 100.0])
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(SEED)
    result: dict = {
        "status": "coverage and type-I-error simulations of the S1 tools; exposed development electrons only; sealed sets not read",
        "clopper_pearson": [
            clopper_pearson_coverage(p, n, 0.95, 4000, rng) for n in (1000, 9696, 20_000) for p in (0.0, 1e-5, 1e-4, 1e-3, 1e-2)
        ],
        "rejection_limit": [rejection_limit_coverage(r, n, 0.95, 4000, rng) for n in (9696, 20_000) for r in (1e3, 1e4, 1e5)],
        "real_electron_halves": {},
    }
    n_half = 1000
    for energy in args.energies:
        names, table = _real_electron_features(energy)
        use = [names.index(c) for c in FEATURES]
        occupancy = names.index("occupied_cells")

        def draw_one(g: np.random.Generator, t=table, col=occupancy):
            order = g.permutation(len(t))
            return t[order[:n_half], col], t[order[n_half : 2 * n_half], col]

        def draw_many(g: np.random.Generator, t=table, columns=use):
            order = g.permutation(len(t))
            return t[order[:n_half]][:, columns], t[order[n_half : 2 * n_half]][:, columns]

        result["real_electron_halves"][f"{energy:g}"] = {
            "permutation_ks_occupied_cells": permutation_type_one_error(draw_one, args.trials, 0.05, args.permutations, rng),
            "permutation_ks_occupied_cells_stratified": permutation_type_one_error(
                draw_one, args.trials, 0.05, args.permutations, rng, strata_of=lambda v: pn.equal_count_strata(v, 5)
            ),
            "familywise_five_conventional_features": familywise_type_one_error(
                draw_many, max(args.trials // 2, 50), 0.05, args.permutations, rng
            ),
        }
    def exchangeable_grids(g: np.random.Generator) -> np.ndarray:
        return g.exponential(1.0, size=(60, 18, 72)) * (g.random((60, 18, 72)) < 0.3)

    def neighbour_product(grid: np.ndarray) -> float:
        return float(np.mean(grid[:, :-1] * grid[:, 1:]) / (np.mean(grid) ** 2 + 1e-12))

    result["cell_shuffle_excess_interval"] = [
        shuffle_excess_interval_coverage(exchangeable_grids, neighbour_product, args.trials, 20, 200, rng)
    ]
    checks = [*result["clopper_pearson"], *result["rejection_limit"]]
    checks += result["cell_shuffle_excess_interval"]
    for per_energy in result["real_electron_halves"].values():
        checks += list(per_energy.values())
    result["all_checks_pass"] = bool(all(c["passed"] for c in checks))
    result["n_checks"] = len(checks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{len(checks)} checks, all pass: {result['all_checks_pass']}")
    for c in checks:  # a failing tool must not be used: say which
        if not c["passed"]:
            print("FAILED:", c)


if __name__ == "__main__":
    main()
