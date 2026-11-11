"""EXP-019: powered repeat of the incremental-classifier test (registration 4efe4d15d845e6aa, approved 2026-11-11).

Do readout D0/D1/D2 add electron/proton information beyond 24 conventional variables, and if any, is it smaller than
delta_min = 0.010 in class-balanced log-loss? Primary sample: the fresh powered realistic-regime events only (researcher's
DECIDE-A); pilot + powered pooled is a secondary. The hard subset uses the EXP-018 standardisation FROZEN as constants.
Detector-response sensitivities R1/R2 (Poisson photoelectrons at 10 and 5 per MIP) and R3 (effective fibre saturation,
e_s = 260 MeV per fibre, behind a fibre-sum gate) re-select the hard subset under each response.

Features, models, the Nadeau-Bengio interval and the score summaries are imported unchanged from the EXP-018 module
(``incremental_multiscale_classifier``), which is not modified; this module re-implements only what carries the new seed
or the new rules. EXPLORATORY; development data; no claim beyond the stated estimand.

    PYTHONPATH=src python -m ams_ecal.benchmarks.powered_incremental_classifier --out results/realistic_regime/exp019_powered_incremental_classifier
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from ams_ecal.benchmarks import incremental_multiscale_classifier as imc
from ams_ecal.multiscale_estimators import realistic_regime_analysis as rra

EXPLORATORY_LABEL = "EXPLORATORY; development data; no claim beyond the stated estimand"
REGISTRATION_PATH = Path(
    "C:/Users/ASUS/.research-os-backups/registrations/2026-11-11_exp019_powered_incremental_classifier.txt"
)
REGISTRATION_SHA256_PREFIX = "4efe4d15d845e6aa"

SEED = 20261111
DELTA_MIN = 0.010
N_FOLDS, N_REPEATS = 5, 20
SAME_CLASS_REPEATS = 5
MINIMUM_CLASS_EVENTS = 100
THETA_MAX_DEG = 12.0
PE_LEVELS = (10.0, 5.0)  # R1, R2
SATURATION_SCALE_MEV = 260.0  # R3 (EXP-017)
# Section 2: the EXP-018 standardisation (mean, population SD) of k1 = log occupied cells, k2 = log sum readout_grid_mev.
FROZEN_MEAN = np.array([6.030194926529844, 8.38842274841936])
FROZEN_SD = np.array([0.6404516481075088, 1.8563055413530718])
# Grid-offset keys: the pilot keeps the EXP-018 keys (row + class * 1e7) so its features are identical; powered events
# get an extra offset so no two events share a key in the pooled sample (implementation choice, not in the registration).
POWERED_KEY_OFFSET = 5_000_000
# Amendment 1 (2026-11-11, hash 08f86018f9ec1a6a): the fibre-sum gate reads "to float precision" as
# |re-summed - stored| <= GATE_ATOL_MEV + GATE_RTOL * |stored| in every cell. Written after the other EXP-019 results.
AMENDMENT1_PATH = Path("C:/Users/ASUS/.research-os-backups/registrations/2026-11-11_exp019_amendment1_fibre_sum_tolerance.txt")
AMENDMENT1_SHA256_PREFIX = "08f86018f9ec1a6a"
GATE_ATOL_MEV = 1e-3
GATE_RTOL = 1e-5

OUTCOME_ADDS = "multiscale features add information"
OUTCOME_NO_GAIN = "no gain as large as delta_min"
OUTCOME_INCONCLUSIVE = "inconclusive at delta_min"
OUTCOME_WORSE = "multiscale features make the model worse"


# --- registration ---------------------------------------------------------------------------------------------------------------------


def registration_info(path: Path = REGISTRATION_PATH) -> dict[str, str]:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if not digest.startswith(REGISTRATION_SHA256_PREFIX):
        raise SystemExit(f"registration changed: sha256 {digest[:16]} != {REGISTRATION_SHA256_PREFIX}")
    return {"path": str(path), "sha256": digest, "sha256_prefix_registered": REGISTRATION_SHA256_PREFIX}


# --- section 5: decision rule ---------------------------------------------------------------------------------------------------------


def decide(lower: float, upper: float, delta_min: float = DELTA_MIN) -> dict[str, Any]:
    """Outcome on the 95 % interval of delta = loss(B) - loss(A) (negative = B better; the gain is -delta)."""

    if upper < 0:
        return {"outcome": OUTCOME_ADDS, "smaller_than_delta_min": bool(lower > -delta_min)}
    if lower > 0:
        return {"outcome": OUTCOME_WORSE}
    if lower > -delta_min:
        return {"outcome": OUTCOME_NO_GAIN}
    return {"outcome": OUTCOME_INCONCLUSIVE}


# --- section 2: hard subset with the frozen standardisation ---------------------------------------------------------------------------


def standardised_keys(cells: np.ndarray, energy_mev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c, e = np.asarray(cells, float), np.asarray(energy_mev, float)
    usable = (c > 0) & (e > 0)
    k = np.zeros((len(c), 2))
    k[usable] = np.column_stack([np.log(c[usable]), np.log(e[usable])])
    return (k - FROZEN_MEAN) / FROZEN_SD, usable


def frozen_hard_subset(cells_e, energy_e, cells_p, energy_p) -> tuple[np.ndarray, np.ndarray]:
    """(a) >= 572 occupied cells; (b) an event of the other class passing (a) within 0.25 frozen-standardised units in both keys
    (Chebyshev distance, inclusive), evaluated once."""

    ze, ue = standardised_keys(cells_e, energy_e)
    zp, up = standardised_keys(cells_p, energy_p)
    pe = ue & (np.asarray(cells_e) >= imc.MINIMUM_OCCUPIED_CELLS)
    pp = up & (np.asarray(cells_p) >= imc.MINIMUM_OCCUPIED_CELLS)
    keep_e, keep_p = np.zeros(len(ze), bool), np.zeros(len(zp), bool)
    if pe.any() and pp.any():
        ie, ip = np.flatnonzero(pe), np.flatnonzero(pp)
        de, _ = cKDTree(zp[ip]).query(ze[ie], p=np.inf)
        dp, _ = cKDTree(ze[ie]).query(zp[ip], p=np.inf)
        keep_e[ie[de <= imc.CALIPER]] = True
        keep_p[ip[dp <= imc.CALIPER]] = True
    return keep_e, keep_p


def matched_energy_ratio(images_e, images_p, idx_e, idx_p) -> float:
    """Median over hard-subset electrons of (primary energy of the nearest hard-subset proton in key space) / (electron energy)."""

    ze, _ = standardised_keys(images_e.cells[idx_e], images_e.grids[idx_e].sum(axis=(1, 2)))
    zp, _ = standardised_keys(images_p.cells[idx_p], images_p.grids[idx_p].sum(axis=(1, 2)))
    _, nearest = cKDTree(zp).query(ze, p=np.inf)
    return float(np.median(images_p.primary_energy_gev[idx_p][nearest] / images_e.primary_energy_gev[idx_e]))


# --- data: one class of one sample under one detector response ------------------------------------------------------------------------


class Part:
    """Fiducial events of one class from one sample: ClassImages (grids, cells, keys) plus theta and the source label."""

    def __init__(self, images: rra.ClassImages, theta_deg: np.ndarray, source: np.ndarray):
        self.images, self.theta_deg, self.source = images, theta_deg, source

    @property
    def energy_mev(self) -> np.ndarray:
        return self.images.grids.sum(axis=(1, 2))


def fibre_grids(class_dir: Path, events: rra.ClassEvents, keep: np.ndarray, scale_mev: float | None) -> tuple[np.ndarray, dict[str, Any]]:
    """Readout grids re-summed from the per-fibre deposits (optionally saturated), and the gate statistics of the UNSATURATED
    re-sum against readout_grid_mev: the largest absolute difference (MeV), the largest relative difference, and the number of
    cells outside the amendment-1 tolerance |d| <= GATE_ATOL_MEV + GATE_RTOL * |stored|."""

    from ams_ecal.detector.geometry import load_geometry
    from ams_ecal.detector.transport_geometry import FibreLayout
    from ams_ecal.geant4_simulation.geant4_backend import GEOMETRY_CONFIG, load_batch

    geometry = load_geometry(GEOMETRY_CONFIG)
    fibre_layer, fibre_cell = FibreLayout(geometry).readout_lookup(geometry.number_of_superlayers)
    directories = sorted({p.parent for p in Path(class_dir).rglob("events.npz")})
    batches: dict[int, Any] = {}
    layers, cells = events.readout_grid_mev.shape[1:]
    out = np.zeros((len(keep), layers, cells))
    worst, worst_relative, violations = 0.0, 0.0, 0
    for k, row in enumerate(keep):
        b, r = (int(v) for v in events.event_band_position[row])
        if b not in batches:
            batches[b] = load_batch(directories[b])
        ids, edep = batches[b].fibre_deposits(r)
        layer, cell = fibre_layer[ids], fibre_cell[ids]
        mapped = (layer >= 0) & (cell >= 0) & (layer < layers)
        flat = layer[mapped] * cells + cell[mapped]
        e = edep[mapped].astype(np.float64)
        plain = np.bincount(flat, weights=e, minlength=layers * cells).reshape(layers, cells)
        stored = events.readout_grid_mev[row].astype(np.float64)
        difference = np.abs(plain - stored)
        worst = max(worst, float(difference.max()))
        nonzero = stored != 0
        if nonzero.any():
            worst_relative = max(worst_relative, float((difference[nonzero] / np.abs(stored[nonzero])).max()))
        violations += int((difference > GATE_ATOL_MEV + GATE_RTOL * np.abs(stored)).sum())
        if scale_mev is None:
            out[k] = plain
        else:
            out[k] = np.bincount(flat, weights=e / (1.0 + e / scale_mev), minlength=layers * cells).reshape(layers, cells)
    return out, {"max_abs_difference_mev": worst, "max_relative_difference": worst_relative, "cells_outside_tolerance": violations}


def build_part(
    events: rra.ClassEvents,
    selected: np.ndarray,
    name: str,
    class_code: int,
    key_offset: int,
    source: str,
    response: tuple[str, float] | None = None,
    class_dir: Path | None = None,
) -> tuple[Part, dict[str, Any]]:
    keep = np.flatnonzero(selected & rra.fiducial_of(events))
    grids = events.readout_grid_mev[keep].astype(float)
    info: dict[str, Any] = {}
    if response is not None and response[0] == "pe":
        rng = np.random.default_rng([SEED, int(response[1]), class_code])
        grids = rra.photoelectron_smearing(grids, response[1], rng)
    elif response is not None and response[0] == "saturation":
        assert class_dir is not None
        grids, gate = fibre_grids(class_dir, events, keep, response[1])
        info["fibre_sum_gate"] = gate
    images = rra.ClassImages(
        name=name,
        grids=grids,
        cells=rra.occupied_cells(grids),
        energy_mev=grids.sum(axis=(1, 2)),
        band_lower_gev=events.band_lower_gev[keep],
        event_keys=keep + class_code * 10_000_000 + key_offset,
        primary_energy_gev=events.primary_energy_gev[keep].astype(float),
    )
    theta = np.degrees(events.primary_theta_rad[keep])
    return Part(images, theta, np.full(len(keep), source)), info


def concatenate(parts: list[Part], name: str) -> Part:
    images = rra.ClassImages(
        name=name,
        grids=np.concatenate([p.images.grids for p in parts]),
        cells=np.concatenate([p.images.cells for p in parts]),
        energy_mev=np.concatenate([p.images.energy_mev for p in parts]),
        band_lower_gev=np.concatenate([p.images.band_lower_gev for p in parts]),
        event_keys=np.concatenate([p.images.event_keys for p in parts]),
        primary_energy_gev=np.concatenate([p.images.primary_energy_gev for p in parts]),
    )
    return Part(images, np.concatenate([p.theta_deg for p in parts]), np.concatenate([p.source for p in parts]))


# --- section 4: cross-fitting with this registration's seed ---------------------------------------------------------------------------


def run_repeat(sets: dict[str, np.ndarray], y: np.ndarray, kind: str, fold_seed: int, repeat: int) -> dict[str, Any]:
    """As EXP-018's run_repeat, with model random_state SEED + 5 r + f of THIS registration."""

    with threadpool_limits(limits=1):
        oof = {name: np.full(len(y), np.nan) for name in sets}
        rows = []
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=fold_seed)
        for fold, (train, test) in enumerate(splitter.split(next(iter(sets.values())), y)):
            row: dict[str, Any] = {"repeat": repeat, "fold": fold, "n_train": len(train), "n_test": len(test)}
            for name, x in sets.items():
                p = imc.fit_predict_fold(kind, x[train], y[train], x[test], SEED + N_FOLDS * repeat + fold)
                oof[name][test] = p
                row[f"logloss_{name}"] = imc.fold_log_loss(p, y[test])
            rows.append(row)
    return {"rows": rows, "oof": oof}


def cross_fit(sets_per_repeat: list[dict[str, np.ndarray]], y: np.ndarray, kind: str, n_jobs: int, offset: int = 0) -> list[dict[str, Any]]:
    tasks = [(sets, y, kind, SEED + offset + r, r) for r, sets in enumerate(sets_per_repeat)]
    if n_jobs <= 1:
        return [run_repeat(*t) for t in tasks]
    return Parallel(n_jobs=n_jobs)(delayed(run_repeat)(*t) for t in tasks)


def permute_multiscale(multiscale: np.ndarray, repeat: int) -> np.ndarray:
    return multiscale[np.random.default_rng([SEED, 18, repeat]).permutation(len(multiscale))]


def same_class_null(conventional: np.ndarray, multiscale: np.ndarray) -> dict[str, Any]:
    n = len(conventional)
    n1 = n // 2
    se = imc.hanley_mcneil_null_se(n1, n - n1)
    both = np.column_stack([conventional, multiscale])
    per_model: dict[str, list[float]] = {"A": [], "B": []}
    for r in range(SAME_CLASS_REPEATS):
        y = np.zeros(n, int)
        y[np.random.default_rng([SEED, 19, r]).permutation(n)[:n1]] = 1
        result = run_repeat({"A": conventional, "B": both}, y, "hgb", SEED + 1000 + r, r)
        for name, values in per_model.items():
            values.append(float(roc_auc_score(y, result["oof"][name])))
    deviation = max(abs(a - 0.5) for v in per_model.values() for a in v) / se
    return {
        "n_electrons": n,
        "standard_error_null": se,
        "auc_per_repeat": per_model,
        "max_abs_deviation_in_standard_errors": float(deviation),
        "passed": bool(deviation <= 3.0),
    }


def compared(results: list[dict[str, Any]], base: str, other: str) -> dict[str, Any]:
    c = imc.compare(results, base, other)
    c.pop("decision", None)  # EXP-018's three-way rule; replaced by section 5
    c["reading"] = decide(c["interval"]["lower"], c["interval"]["upper"])
    return c


def _summary(c: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in c.items() if k != "deltas"}


# --- one sample -----------------------------------------------------------------------------------------------------------------------


def analyse_sample(
    label: str,
    electrons: Part,
    protons: Part,
    n_jobs: int,
    controls: bool,
    logistic: bool,
    theta_stratum: bool,
    fold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    keep_e, keep_p = frozen_hard_subset(electrons.images.cells, electrons.energy_mev, protons.images.cells, protons.energy_mev)
    idx_e, idx_p = np.flatnonzero(keep_e), np.flatnonzero(keep_p)
    sizes = {"electron": len(idx_e), "proton_interacting": len(idx_p)}
    out: dict[str, Any] = {
        "fiducial": {"electron": len(electrons.theta_deg), "proton_interacting": len(protons.theta_deg)},
        "hard_subset": sizes,
        "hard_subset_by_source": {
            "electron": {s: int((electrons.source[idx_e] == s).sum()) for s in np.unique(electrons.source)},
            "proton_interacting": {s: int((protons.source[idx_p] == s).sum()) for s in np.unique(protons.source)},
        },
    }
    if min(sizes.values()) < MINIMUM_CLASS_EVENTS:
        out["primary"] = {"reading": {"outcome": "not testable"}}
        return out
    out["median_matched_proton_over_electron_primary_energy"] = matched_energy_ratio(electrons.images, protons.images, idx_e, idx_p)
    conventional = np.vstack([imc.conventional_matrix(electrons.images.grids[idx_e]), imc.conventional_matrix(protons.images.grids[idx_p])])
    multiscale = np.vstack([electrons.images.dims(idx_e, True), protons.images.dims(idx_p, True)])
    if not np.isfinite(multiscale).all():
        raise RuntimeError(f"{label}: non-finite multiscale feature after cut (a)")
    y = np.r_[np.ones(len(idx_e), int), np.zeros(len(idx_p), int)]
    theta = np.r_[electrons.theta_deg[idx_e], protons.theta_deg[idx_p]]

    def run(c: np.ndarray, m: np.ndarray, yy: np.ndarray, with_controls: bool) -> tuple[dict[str, Any], list]:
        sets_per_repeat = []
        for r in range(N_REPEATS):
            sets = imc.primary_sets(c, m) if with_controls else {"A": c, "B": np.column_stack([c, m])}
            if with_controls:
                sets["B_permuted"] = np.column_stack([c, permute_multiscale(m, r)])
            sets_per_repeat.append(sets)
        results = cross_fit(sets_per_repeat, yy, "hgb", n_jobs)
        res: dict[str, Any] = {"primary": compared(results, "A", "B")}
        res["scores"] = imc.score_summary(results, yy, list(sets_per_repeat[0]))
        res["auc_difference_B_minus_A"] = imc.auc_difference(res["scores"], "A", "B")
        if with_controls:
            res["negative_control"] = compared(results, "A", "B_permuted")
            res["ablation_D0"] = compared(results, "A", "ablation_D0")
            res["ablation_D1_D2"] = compared(results, "A", "ablation_D1_D2")
        return res, results

    main, results = run(conventional, multiscale, y, controls)
    out["primary"] = _summary(main["primary"])
    out["scores"] = main["scores"]
    out["auc_difference_B_minus_A"] = main["auc_difference_B_minus_A"]
    fold_rows += imc._fold_csv_rows(f"{label}_hgb_B_vs_A", results, "A", "B")
    if controls:
        nc = main["negative_control"]
        out["negative_control"] = {**_summary(nc), "status": imc.negative_control_status(nc["interval"]["lower"], nc["interval"]["upper"])}
        out["ablation_D0"] = _summary(main["ablation_D0"])
        out["ablation_D1_D2"] = _summary(main["ablation_D1_D2"])
        out["same_class_null"] = same_class_null(conventional[y == 1], multiscale[y == 1])
        control_ok = out["negative_control"]["status"] == imc.NEGATIVE_CONTROL_PASSED and out["same_class_null"]["passed"]
        out["status"] = main["primary"]["reading"]["outcome"] if control_ok else "controls not passed: the result is not to be read"
        fold_rows += imc._fold_csv_rows(f"{label}_negative_control", results, "A", "B_permuted")
    if logistic:
        log_results = cross_fit([{"A": conventional, "B": np.column_stack([conventional, multiscale])} for _ in range(N_REPEATS)], y, "logistic", n_jobs)
        out["logistic_B_vs_A"] = _summary(compared(log_results, "A", "B"))
    if theta_stratum:
        s = theta <= THETA_MAX_DEG
        s_sizes = {"electron": int((s & (y == 1)).sum()), "proton_interacting": int((s & (y == 0)).sum())}
        if min(s_sizes.values()) >= MINIMUM_CLASS_EVENTS:
            res, s_results = run(conventional[s], multiscale[s], y[s], False)
            out["theta_le_12_deg"] = {"sizes": s_sizes, **_summary(res["primary"]), "auc_difference_B_minus_A": res["auc_difference_B_minus_A"]}
            fold_rows += imc._fold_csv_rows(f"{label}_theta_le_12_hgb_B_vs_A", s_results, "A", "B")
        else:
            out["theta_le_12_deg"] = {"sizes": s_sizes, "reading": {"outcome": "not testable"}}
    return out


# --- run ------------------------------------------------------------------------------------------------------------------------------


def figure_intervals(result: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for key, title in (("primary_fresh", "fresh (primary)"), ("secondary_pooled", "pooled"), ("R1_pe10", "fresh, 10 p.e./MIP"), ("R2_pe5", "fresh, 5 p.e./MIP"), ("R3_saturation", "fresh, saturation 260 MeV")):
        block = result.get(key, {})
        interval = block.get("primary", {}).get("interval")
        if interval:
            rows.append((title, interval))
        for sub, sub_title in (("negative_control", "  permuted M (control)"), ("theta_le_12_deg", "  theta <= 12 deg")):
            iv = block.get(sub, {}).get("interval") if isinstance(block.get(sub), dict) else None
            if iv:
                rows.append((title + sub_title, iv))
    fig, ax = plt.subplots(figsize=(9.0, 0.45 * len(rows) + 1.8))
    ax.axvspan(-DELTA_MIN, DELTA_MIN, color="0.92", zorder=0, label=f"|change| < delta_min = {DELTA_MIN}")
    ax.axvline(0, color="0.4", lw=0.8)
    for i, (title, iv) in enumerate(reversed(rows)):
        ax.errorbar(iv["mean"], i, xerr=[[iv["mean"] - iv["lower"]], [iv["upper"] - iv["mean"]]], fmt="o", color="#1b3a6b", capsize=3)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.set_xlabel("change in class-balanced log-loss, C + M minus C\n(negative = multiscale features help)")
    ax.set_title("EXP-019: 95% intervals (Nadeau-Bengio), EXPLORATORY", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def r3_gate(info: dict[str, Any]) -> dict[str, Any]:
    """Amendment-1 fibre-sum gate over both classes, with the amendment's hash and label."""

    digest = hashlib.sha256(AMENDMENT1_PATH.read_bytes()).hexdigest()
    if not digest.startswith(AMENDMENT1_SHA256_PREFIX):
        raise SystemExit(f"amendment 1 changed: sha256 {digest[:16]} != {AMENDMENT1_SHA256_PREFIX}")
    gates = [info[c]["fibre_sum_gate"] for c in ("electron", "proton_interacting")]
    return {
        "amendment": {"path": str(AMENDMENT1_PATH), "sha256": digest},
        "label": "run after amendment 1, written after the other EXP-019 results were known; descriptive only",
        "fibre_sum_gate": {
            "max_abs_difference_mev": max(g["max_abs_difference_mev"] for g in gates),
            "max_relative_difference": max(g["max_relative_difference"] for g in gates),
            "cells_outside_tolerance": sum(g["cells_outside_tolerance"] for g in gates),
            "tolerance": f"|d| <= {GATE_ATOL_MEV} MeV + {GATE_RTOL} * |stored|",
            "passed": all(g["cells_outside_tolerance"] == 0 for g in gates),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--skip-secondaries", action="store_true", help="primary only (for a quick check)")
    parser.add_argument("--only-r3", action="store_true", help="amendment 1: rerun only R3 into the existing result file")
    args = parser.parse_args(argv)
    if args.only_r3:
        run_r3_only(args)
        return
    t0 = time.time()
    registration = registration_info()
    args.out.mkdir(parents=True, exist_ok=True)
    dirs = {
        ("pilot", "electron"): args.data_root / "geant4_realistic_regime_pilot_electron",
        ("pilot", "proton"): args.data_root / "geant4_realistic_regime_pilot_proton",
        ("powered", "electron"): args.data_root / "geant4_realistic_regime_powered_electron",
        ("powered", "proton"): args.data_root / "geant4_realistic_regime_powered_proton",
    }
    events = {k: rra.load_class_events(d, k[1]) for k, d in dirs.items()}

    def parts(sample: str, response=None) -> tuple[Part, Part, dict[str, Any]]:
        offset = POWERED_KEY_OFFSET if sample == "powered" else 0
        ev_e, ev_p = events[(sample, "electron")], events[(sample, "proton")]
        pe, ie = build_part(ev_e, np.ones(len(ev_e), bool), "electron", 1, offset, sample, response, dirs[(sample, "electron")])
        pp, ip = build_part(ev_p, ev_p.truth_occurred.astype(bool), "proton_interacting", 2, offset, sample, response, dirs[(sample, "proton")])
        return pe, pp, {"electron": ie, "proton_interacting": ip}

    from ams_ecal.benchmarks.incremental_multiscale_classifier import git_commit

    result: dict[str, Any] = {
        "label": EXPLORATORY_LABEL,
        "experiment": "EXP-019",
        "registration": registration,
        "git": git_commit(Path(__file__).resolve().parents[3]),
        "settings": {"seed": SEED, "delta_min": DELTA_MIN, "frozen_mean": FROZEN_MEAN.tolist(), "frozen_sd": FROZEN_SD.tolist(), "n_jobs": args.n_jobs, "hist_gradient_boosting": imc.HGB_SETTINGS},
        "implementation_notes": [
            "Powered events get grid-offset keys row + class * 1e7 + 5e6 so pooled events never share a key; pilot keys are the EXP-018 ones.",
            "The matching energy key is the sum of the (response-modified) readout grid, as in EXP-018 and the EXP-015 smearing analysis.",
            "Response sensitivities run A and B only (no permuted control, no ablations) on the fresh sample; read with the same delta_min rule.",
            "matched energy ratio: nearest hard-subset proton in frozen key space (Chebyshev) for each hard-subset electron.",
        ],
    }
    fold_rows: list[dict[str, Any]] = []
    timings: dict[str, float] = {}

    fresh_e, fresh_p, _ = parts("powered")
    t = time.time()
    result["primary_fresh"] = analyse_sample("primary_fresh", fresh_e, fresh_p, args.n_jobs, True, True, True, fold_rows)
    timings["primary_fresh"] = round(time.time() - t, 1)
    result["overall_status"] = result["primary_fresh"].get("status", result["primary_fresh"]["primary"]["reading"]["outcome"])
    rra.write_json(args.out / "exp019_result.json", result)
    print("primary:", result["overall_status"], result["primary_fresh"]["primary"].get("interval"), flush=True)

    if not args.skip_secondaries:
        pilot_e, pilot_p, _ = parts("pilot")
        t = time.time()
        result["secondary_pooled"] = analyse_sample(
            "secondary_pooled", concatenate([pilot_e, fresh_e], "electron"), concatenate([pilot_p, fresh_p], "proton_interacting"), args.n_jobs, True, True, True, fold_rows
        )
        timings["secondary_pooled"] = round(time.time() - t, 1)
        rra.write_json(args.out / "exp019_result.json", result)
        for key, response in (("R1_pe10", ("pe", PE_LEVELS[0])), ("R2_pe5", ("pe", PE_LEVELS[1])), ("R3_saturation", ("saturation", SATURATION_SCALE_MEV))):
            t = time.time()
            e_part, p_part, info = parts("powered", response)
            block: dict[str, Any] = {"response": list(response)}
            if response[0] == "saturation":
                block.update(r3_gate(info))
                if not block["fibre_sum_gate"]["passed"]:
                    block["reading"] = {"outcome": "not run: fibre-sum gate failed"}
                    result[key] = block
                    continue
            block.update(analyse_sample(key, e_part, p_part, args.n_jobs, False, False, False, fold_rows))
            result[key] = block
            timings[key] = round(time.time() - t, 1)
            rra.write_json(args.out / "exp019_result.json", result)
            print(key, block.get("primary", {}).get("reading"), flush=True)

    result["timings_seconds"] = {**timings, "total": round(time.time() - t0, 1)}
    rra.write_json(args.out / "exp019_result.json", result)
    rra.write_csv(args.out / "exp019_per_fold.csv", fold_rows)
    figure_intervals(result, args.out / "exp019_fig1_intervals.png")
    print("overall:", result["overall_status"], "| total s", result["timings_seconds"]["total"], flush=True)


def run_r3_only(args: argparse.Namespace) -> None:
    """Amendment 1: run R3 alone on the fresh powered events and write it into the existing exp019_result.json; every other
    block of the file is left as it was."""

    import json

    t0 = time.time()
    registration_info()
    path = args.out / "exp019_result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result["R3_saturation_original_run"] = result.get("R3_saturation")
    dirs = {name: args.data_root / f"geant4_realistic_regime_powered_{name}" for name in ("electron", "proton")}
    ev_e, ev_p = rra.load_class_events(dirs["electron"], "electron"), rra.load_class_events(dirs["proton"], "proton")
    response = ("saturation", SATURATION_SCALE_MEV)
    pe, ie = build_part(ev_e, np.ones(len(ev_e), bool), "electron", 1, POWERED_KEY_OFFSET, "powered", response, dirs["electron"])
    pp, ip = build_part(ev_p, ev_p.truth_occurred.astype(bool), "proton_interacting", 2, POWERED_KEY_OFFSET, "powered", response, dirs["proton"])
    block: dict[str, Any] = {"response": list(response), **r3_gate({"electron": ie, "proton_interacting": ip})}
    fold_rows: list[dict[str, Any]] = []
    if block["fibre_sum_gate"]["passed"]:
        block.update(analyse_sample("R3_saturation", pe, pp, args.n_jobs, False, False, False, fold_rows))
    else:
        block["reading"] = {"outcome": "not run: fibre-sum gate failed (amendment 1)"}
    block["seconds"] = round(time.time() - t0, 1)
    result["R3_saturation"] = block
    rra.write_json(path, result)
    if fold_rows:
        rra.write_csv(args.out / "exp019_r3_amendment1_per_fold.csv", fold_rows)
    figure_intervals(result, args.out / "exp019_fig1_intervals.png")
    print("R3:", block["fibre_sum_gate"], block.get("hard_subset"), block.get("primary", {}).get("interval"), block.get("primary", block).get("reading"), flush=True)


if __name__ == "__main__":
    main()
