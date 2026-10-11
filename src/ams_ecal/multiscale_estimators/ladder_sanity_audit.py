"""Sanity audit of the development ladder (slice S0, review items 1, 3 and C8-C9): is the near-perfect separation an implementation error?

Question. Conventional features separate Geant4 electrons from protons at AUC 0.9999-1.0000. The review found no implementation
error by reading code; this audit tests the pipeline with controls whose answer is known in advance.

Expected outcomes, FIXED BEFORE THE FIRST RUN (a result outside them is a finding to explain, never a threshold to move):

* same-class null (electrons against electrons, same configuration, different seeds or commit; random halves of one file):
  AUC 0.5 +- 0.02 when both groups hold at least 1000 events;
* label permutation of the electron/proton problem: mean AUC 0.5 +- 0.02, every draw within 0.5 +- 0.03;
* fold-seed spread of the ladder AUC: sd below 0.002 (the problem is saturated, so it must be stable);
* duplicates: none within a set; none between two sets of different seed; zero-energy events counted and reported;
* the univariate AUC of the total deposit is a property of one population. A KS of 0.998 means one threshold separates 99.8% of the
  two samples; for a score with a concave ROC curve (monotone likelihood ratio) that forces AUC >= (1 + KS) / 2 = 0.999, so a
  reported 0.991 beside 0.998 means a different population (or a non-monotone feature). The audit computes both on every
  population and prints the pairs (the inequality is NOT general: a two-sided feature can have a large KS and a small AUC).

Everything here is exploratory, on exposed sets through ``validation.sample_definitions``; no sealed set is read, no registered
number is changed, and the ladder code (``multiscale_ladder``, ``multiscale_test``) is used as it is, including its known
problems (zero-failure floor, same-event threshold): hence AUC and counts k of N are reported and rejections are not.

Command: ``python -m ams_ecal.multiscale_estimators.ladder_sanity_audit --energies 10 20 50 100``
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ams_ecal.electron_model.em_production import PROJECT_ROOT
from ams_ecal.multiscale_estimators import multiscale_ladder as ml
from ams_ecal.multiscale_estimators import multiscale_test as mt
from ams_ecal.validation import sample_definitions as sd
from ams_ecal.validation.added_checks import N_FOLDS

RESULTS_PATH = PROJECT_ROOT / "results" / "sanity_audit" / "ladder_sanity_audit.json"
SEED = 20261011
PERMUTATIONS = 10
FOLD_SEEDS = 5
INJECTED_COPIES = (1, 3)
CELL_THRESHOLDS = (336, 572)  # the registered eligibility thresholds (multiscale_test.REGISTERED_WINDOWS)
REPRESENTATIONS = sd.REPRESENTATIONS

# Feature families by the names of ``multiscale_test.conventional_features`` (layer_fraction_i for the 18 layers).
FAMILIES = {
    "energy": ("log_total_energy",),
    "occupancy": ("occupied_cells",),
    "longitudinal": ("layer_cog", *[f"layer_fraction_{i}" for i in range(18)]),
    "lateral": ("width_cells", "core_fraction", "max_cell_fraction"),
}
LAYER_BLOCKS = {
    "layers_0_5": tuple(f"layer_fraction_{i}" for i in range(6)),
    "layers_6_11": tuple(f"layer_fraction_{i}" for i in range(6, 12)),
    "layers_12_17": tuple(f"layer_fraction_{i}" for i in range(12, 18)),
}


# --- small tools -----------------------------------------------------------------------------------------------------------


def columns(names: list[str], wanted: tuple[str, ...]) -> list[int]:
    return [names.index(n) for n in wanted]


def pooled(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Features of two groups stacked, labels 1 for ``a`` (electrons or the first set) and 0 for ``b``."""

    return np.concatenate([a, b]), np.concatenate([np.ones(len(a), int), np.zeros(len(b), int)])


def hgb_auc(x: np.ndarray, labels: np.ndarray, seed: int = 0) -> float:
    """Out-of-fold AUC of the registered classifier (the ladder's own function)."""

    return float(roc_auc_score(labels, ml.out_of_fold_scores(x, labels, None, seed)))


def logistic_auc(x: np.ndarray, labels: np.ndarray, seed: int = 0) -> float:
    scores = np.zeros(len(labels))
    for train, test in StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(x, labels):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(x[train], labels[train])
        scores[test] = model.predict_proba(x[test])[:, 1]
    return float(roc_auc_score(labels, scores))


def univariate(values_a: np.ndarray, values_b: np.ndarray) -> dict[str, float]:
    """Rank AUC (probability a value of ``a`` exceeds one of ``b``) and KS distance of two samples of one variable."""

    x, y = pooled(values_a, values_b)
    return {
        "auc_a_above_b": float(roc_auc_score(y, x)),
        "ks": float(stats.ks_2samp(values_a, values_b).statistic),
        "n_a": len(values_a),
        "n_b": len(values_b),
    }


def counts_above_electron_threshold(scores: np.ndarray, labels: np.ndarray, efficiency: float = 0.99) -> dict[str, float]:
    """Protons at or above the score keeping ``efficiency`` of the electrons: k of N. INHERITS the ladder's same-event threshold."""

    threshold = np.quantile(scores[labels == 1], 1.0 - efficiency)
    protons = scores[labels == 0]
    return {"k_passing": int((protons >= threshold).sum()), "n_protons": len(protons), "efficiency": efficiency}


def row_hashes(grids: np.ndarray) -> list[bytes]:
    """One 16-byte digest per event grid (float32 bytes), to find identical events."""

    flat = np.ascontiguousarray(grids.astype(np.float32).reshape(len(grids), -1))
    return [hashlib.blake2b(row.tobytes(), digest_size=16).digest() for row in flat]


# --- the checks ------------------------------------------------------------------------------------------------------------


def population_consistency(energy: float) -> dict:
    """Univariate AUC and KS of the log total of each grid, for every pairing of electron set and proton population.

    Resolves which population stands behind each reported number (0.991 against 0.998). Pilot protons: the held-out quarter is
    excluded (their rule); development-large protons: nothing excluded (their rule) and, as a sensitivity, the quarter excluded.
    """

    out: dict = {}
    pairs = (
        ("pilot", "electron_pilot", "proton_pilot", False),
        ("development_large", "electron_development_large", "proton_development_large", False),
        ("development_large_quarter_excluded", "electron_development_large", "proton_development_large", True),
    )
    for label, e_name, p_name, quarter in pairs:
        row: dict = {}
        for representation in REPRESENTATIONS:
            electrons = sd.load_events(e_name, energy, representation, "all", dtype=np.float32)
            e_total = np.log(electrons.grids.reshape(len(electrons), -1).sum(axis=1, dtype=np.float64))
            cell = {}
            for population in sd.POPULATIONS:
                protons = sd.load_events(
                    p_name, energy, representation, population, hold_out_quarter_everywhere=quarter, dtype=np.float32
                )
                total = protons.grids.reshape(len(protons), -1).sum(axis=1, dtype=np.float64)
                positive = total > 0
                cell[population] = {
                    **univariate(e_total, np.log(total[positive])),
                    "zero_energy_protons": int((~positive).sum()),
                    "n_excluded_by_held_out": protons.n_excluded_by_held_out,
                }
            row[representation] = cell
        out[label] = row
    return out


def same_class_nulls(energy: float, rng: np.random.Generator) -> dict:
    """Same particle, same configuration, different seeds (and commit): the AUC must be 0.5. Within-set halves are the reference."""

    out: dict = {}
    for representation in REPRESENTATIONS:
        row: dict = {}
        pilot_e = sd.load_events("electron_pilot", energy, representation, "all")
        large_e = sd.load_events("electron_development_large", energy, representation, "all")
        _, fe_p = mt.conventional_features(pilot_e.grids)
        _, fe_l = mt.conventional_features(large_e.grids)
        x, y = pooled(fe_p, fe_l)
        row["electron_pilot_vs_development_large"] = {
            "n": [len(fe_p), len(fe_l)],
            "auc_conventional": hgb_auc(x, y),
            "total_energy": univariate(fe_p[:, 0], fe_l[:, 0]),
        }
        half = rng.permutation(len(fe_l))
        a, b = fe_l[half[: len(half) // 2]], fe_l[half[len(half) // 2 :]]
        x, y = pooled(a, b)
        row["electron_development_large_random_halves"] = {"n": [len(a), len(b)], "auc_conventional": hgb_auc(x, y)}
        for population in ("all", "interacting"):
            pilot_p = sd.load_events("proton_pilot", energy, representation, population)
            large_p = sd.load_events("proton_development_large", energy, representation, population)
            _, fp_p = mt.conventional_features(pilot_p.grids)
            _, fp_l = mt.conventional_features(large_p.grids)
            x, y = pooled(fp_p, fp_l)
            row[f"proton_{population}_pilot_vs_development_large"] = {
                "n": [len(fp_p), len(fp_l)],
                "auc_conventional": hgb_auc(x, y),
                "total_energy": univariate(fp_p[:, 0], fp_l[:, 0]),
            }
        out[representation] = row
    return out


def duplicates_and_seeds(energy: float) -> dict:
    """Identical events within a set and between sets; zero-energy events; seed values shared between sets."""

    sets = {
        "electron_pilot": sd.load_events("electron_pilot", energy, "deposit_grid_mev", "all", dtype=np.float32),
        "electron_development_large": sd.load_events(
            "electron_development_large", energy, "deposit_grid_mev", "all", dtype=np.float32
        ),
        "proton_pilot_calibration_events": sd.load_events("proton_pilot", energy, "deposit_grid_mev", "all", dtype=np.float32),
        "proton_development_large": sd.load_events(
            "proton_development_large", energy, "deposit_grid_mev", "all", dtype=np.float32
        ),
    }
    hashes = {name: row_hashes(e.grids) for name, e in sets.items()}
    out: dict = {"within_set": {}, "between_sets": {}, "seed_overlap": {}}
    for name, h in hashes.items():
        out["within_set"][name] = {
            "n": len(h),
            "identical_to_an_earlier_event": int(len(h) - len(set(h))),
            "zero_energy_events": int((sets[name].grids.reshape(len(h), -1).sum(axis=1) == 0).sum()),
        }
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            out["between_sets"][f"{a} | {b}"] = len(set(hashes[a]) & set(hashes[b]))
            out["seed_overlap"][f"{a} | {b}"] = len(np.intersect1d(sets[a].seed, sets[b].seed))
    return out


def injection_positive_control(rng: np.random.Generator, energy: float) -> dict:
    """Does the harness move when events leak across folds? Null task (random labels on one electron file), then every event repeated.

    The repeats keep their label and land in other folds, which is exactly the leak a test of the harness should catch.
    """

    large = sd.load_events("electron_development_large", energy, "deposit_grid_mev", "all")
    _, features = mt.conventional_features(large.grids)
    features = features[rng.permutation(len(features))[:4000]]
    labels = (rng.permutation(len(features)) % 2).astype(int)  # random labels: no class structure by construction
    out = {"null_no_duplicates": hgb_auc(features, labels, 0)}
    for copies in INJECTED_COPIES:
        x = np.concatenate([features] * (copies + 1))
        y = np.concatenate([labels] * (copies + 1))
        out[f"null_each_event_present_{copies + 1}_times"] = hgb_auc(x, y, 0)
    return out


def ladder_controls(electrons: sd.Events, protons: sd.Events, rng: np.random.Generator) -> dict:
    """Baselines, ablations, permutation and fold-seed controls for one electron set against one proton population."""

    names, fe = mt.conventional_features(electrons.grids)
    _, fp = mt.conventional_features(protons.grids)
    x, y = pooled(fe, fp)
    scores = ml.out_of_fold_scores(x, y, None, 0)
    out: dict = {
        "events": {"electron": len(fe), "proton": len(fp)},
        "conventional_hgb": {"auc": float(roc_auc_score(y, scores)), **counts_above_electron_threshold(scores, y)},
    }
    energy_col = columns(names, FAMILIES["energy"])
    out["energy_only"] = {
        "univariate_auc_electron_above_proton": float(roc_auc_score(y, x[:, energy_col[0]])),
        "hgb_auc": hgb_auc(x[:, energy_col], y),
        "logistic_auc": logistic_auc(x[:, energy_col], y),
    }
    out["logistic_all_conventional_auc"] = logistic_auc(x, y)
    # the eligibility indicator alone (the missingness the multiscale rung is accused of measuring)
    cells = x[:, columns(names, ("occupied_cells",))[0]]
    out["indicator_only"] = {f"cells_ge_{t}": {"auc": float(roc_auc_score(y, (cells >= t).astype(float)))} for t in CELL_THRESHOLDS}
    ms_table = mt.multiscale_table(np.concatenate([electrons.grids, protons.grids]))
    out["multiscale_rung_auc_as_registered"] = hgb_auc(ms_table, y)
    out["multiscale_eligible_fraction"] = {
        "electron": float(np.isfinite(ms_table[y == 1]).any(axis=1).mean()),
        "proton": float(np.isfinite(ms_table[y == 0]).any(axis=1).mean()),
    }
    # feature-family and layer-block ablations (single block / leave one block out)
    ablation: dict = {}
    for label, wanted in {**FAMILIES, **LAYER_BLOCKS}.items():
        cols = columns(names, wanted)
        rest = [i for i in range(len(names)) if i not in cols]
        ablation[label] = {"only_auc": hgb_auc(x[:, cols], y), "without_auc": hgb_auc(x[:, rest], y)}
    out["ablations"] = ablation
    # label permutation of the real problem
    perm = [hgb_auc(x, rng.permutation(y)) for _ in range(PERMUTATIONS)]
    out["label_permutation"] = {
        "draws": PERMUTATIONS,
        "mean": float(np.mean(perm)),
        "min": float(min(perm)),
        "max": float(max(perm)),
    }
    # fold-seed spread, registered rung and the harder rung without energy and occupancy
    harder = [i for i, n in enumerate(names) if n not in ("log_total_energy", "occupied_cells")]
    spread = {
        "conventional": [hgb_auc(x, y, seed) for seed in range(FOLD_SEEDS)],
        "without_energy_and_occupancy": [hgb_auc(x[:, harder], y, seed) for seed in range(FOLD_SEEDS)],
    }
    out["fold_seed_spread"] = {
        k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "min": float(min(v)), "max": float(max(v))}
        for k, v in spread.items()
    }
    return out


def energy_blind_pooled(energies: list[float], representation: str) -> dict:
    """Energies pooled, total deposit and occupancy dropped, the rest normalised by construction (fractions, widths): does the
    problem stay easy when the classifier is not told the beam energy and the event size?"""

    fe_list, fp_list, per_energy = [], [], {}
    keep_names: list[str] = []
    for energy in energies:
        electrons = sd.load_events("electron_development_large", energy, representation, "all")
        protons = sd.load_events("proton_development_large", energy, representation, "interacting")
        names, fe = mt.conventional_features(electrons.grids)
        _, fp = mt.conventional_features(protons.grids)
        keep_names = [n for n in names if n not in ("log_total_energy", "occupied_cells")]
        cols = columns(names, tuple(keep_names))
        x, y = pooled(fe[:, cols], fp[:, cols])
        per_energy[f"{energy:g}"] = hgb_auc(x, y)
        fe_list.append(fe[:, cols])
        fp_list.append(fp[:, cols])
    x, y = pooled(np.concatenate(fe_list), np.concatenate(fp_list))
    return {
        "features": keep_names,
        "per_energy_auc": per_energy,
        "pooled_across_energies_auc": hgb_auc(x, y),
        "events": {"electron": int((y == 1).sum()), "proton": int((y == 0).sum())},
    }


# --- the run ---------------------------------------------------------------------------------------------------------------


def audit_one_energy(energy: float) -> dict:
    rng = np.random.default_rng(SEED + int(energy))
    row: dict = {
        "population_consistency": population_consistency(energy),
        "same_class_nulls": same_class_nulls(energy, rng),
        "duplicates_and_seeds": duplicates_and_seeds(energy),
        "duplicate_injection_positive_control": injection_positive_control(rng, energy),
        "ladder_controls": {},
    }
    for representation in REPRESENTATIONS:
        electrons = sd.load_events("electron_development_large", energy, representation, "all")
        block: dict = {}
        for label, population, quarter in (
            ("vs_interacting_protons", "interacting", False),
            ("vs_all_protons", "all", False),
            ("vs_interacting_protons_quarter_excluded", "interacting", True),
        ):
            protons = sd.load_events(
                "proton_development_large", energy, representation, population, hold_out_quarter_everywhere=quarter
            )
            block[label] = ladder_controls(electrons, protons, rng)
        row["ladder_controls"][representation] = block
    return row


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--energies", type=float, nargs="+", default=[10.0, 20.0, 50.0, 100.0])
    parser.add_argument("--pooled", action="store_true", help="run the energy-blind pooled test over --energies instead")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.pooled:
        out_path = args.out or RESULTS_PATH.with_name("ladder_sanity_audit_energy_blind_pooled.json")
        result: dict = {r: energy_blind_pooled(args.energies, r) for r in REPRESENTATIONS}
    else:
        suffix = "_".join(f"{e:g}" for e in args.energies)
        out_path = args.out or RESULTS_PATH.with_name(f"ladder_sanity_audit_{suffix}GeV.json")
        result = {
            "status": "EXPLORATORY sanity audit on exposed sets; sealed sets not read; ladder code used unchanged",
            "sealed_data_read": False,
            "per_energy": {},
        }
        for energy in args.energies:
            result["per_energy"][f"{energy:g}"] = audit_one_energy(energy)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"{energy:g} GeV done", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
