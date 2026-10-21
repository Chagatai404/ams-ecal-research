"""Benchmark harness: learning curves of electron/proton classifiers with role separation, log-loss, rejection and a GPU network.

Registration: research/COMPLETED_WORK.md (was 2026-10-16_benchmark_harness_dry_run_registration.md). EXPLORATORY dry run on existing ideal-regime
development data; it exercises the machinery and measures its cost, it draws no inference about multiscale features or efficiency.

Roles per block (energy, representation), disjoint: calibration electrons (threshold only), evaluation electrons and protons, and
training pools. A learning-curve point is (training size per class, repeat): the same training subsets feed every model of that repeat,
so model differences are paired. Models are never tuned.

    python -m ams_ecal.benchmarks.learning_curves
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ams_ecal.electron_model.em_production import PROJECT_ROOT
from ams_ecal.inference import rejection_limits as rl
from ams_ecal.multiscale_estimators import multiscale_gtrain_development as msgd
from ams_ecal.multiscale_estimators import multiscale_test as mt
from ams_ecal.multiscale_estimators import multiscale_verdict_reread as vr
from ams_ecal.validation import sample_definitions as sd
from ams_ecal.validation.added_checks import CLASSIFIER_SETTINGS

SIZES = (100, 300, 1000, 3000)
REPEATS = 5
N_CALIBRATION, N_EVAL_ELECTRONS, N_EVAL_PROTONS = 2000, 2000, 4000
EFFICIENCIES = (0.90, 0.99)
MIP_MEV = (
    0.5596007704734802  # fibre-only MIP of a cell (readout_response.MIP_FIBRE_MEV)
)
EPOCHS, BATCH, LEARNING_RATE = 30, 128, 1e-3
SEED = 20261016
RESULT_PATH = PROJECT_ROOT / "results" / "benchmarks" / "harness_dry_run.json"
RUNGS = ("energy_only", "conv", "conv_noE", "conv_noE_nocells")
BASE_MODELS = (
    tuple(RUNGS) + tuple(f"{r}+ms" for r in RUNGS[1:]) + ("conv_logistic", "cnn_raw")
)


# --- metrics --------------------------------------------------------------------------------------------------------------------


def balanced_log_loss(p_electron: np.ndarray, p_proton: np.ndarray) -> float:
    """Class-balanced log-loss for the electron probability: the mean over the two classes of the mean -log of the right-class probability."""
    eps = 1e-12
    return float(
        0.5 * np.mean(-np.log(np.clip(p_electron, eps, 1)))
        + 0.5 * np.mean(-np.log(np.clip(1 - p_proton, eps, 1)))
    )


def metrics(
    p_cal: np.ndarray, p_eval_e: np.ndarray, p_eval_p: np.ndarray
) -> dict[str, Any]:
    """Log-loss, AUC (diagnostic) and the rejection at each electron efficiency (threshold from the calibration electrons only)."""
    y = np.r_[np.ones(len(p_eval_e)), np.zeros(len(p_eval_p))]
    out: dict[str, Any] = {
        "log_loss": balanced_log_loss(p_eval_e, p_eval_p),
        "auc": float(roc_auc_score(y, np.r_[p_eval_e, p_eval_p])),
    }
    for eff in EFFICIENCIES:
        r = rl.evaluate_at_efficiency(p_cal, p_eval_e, p_eval_p, eff)
        out[f"efficiency_{eff:g}"] = {
            "achieved_efficiency": r.achieved_efficiency,
            "protons_passing": r.rejection.k,
            "protons": r.rejection.n,
            "rejection_lower_limit": r.rejection.rejection_lower_limit,
            "censored": r.rejection.censored,
        }
    return out


# --- models ---------------------------------------------------------------------------------------------------------------------


def fit_predict_tabular(
    kind: str, x_train: np.ndarray, y_train: np.ndarray, x_list: list[np.ndarray]
) -> list[np.ndarray]:
    """Fit on the training rows, return the electron probability of every array in ``x_list``; columns all-NaN in training are dropped."""
    x_train = np.asarray(x_train, dtype=float)
    usable = ~np.isnan(x_train).all(axis=0)
    if kind == "logistic":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    else:
        model = HistGradientBoostingClassifier(**CLASSIFIER_SETTINGS)
    model.fit(x_train[:, usable], y_train)
    return [
        model.predict_proba(np.asarray(x, dtype=float)[:, usable])[:, 1] for x in x_list
    ]


def fit_predict_cnn(
    images_train: np.ndarray,
    y_train: np.ndarray,
    images_list: list[np.ndarray],
    seed: int,
    device: str | None = None,
) -> tuple[list[np.ndarray], dict[str, float]]:
    """The fixed small network of the registration on log(1 + E / MIP) images; returns probabilities and cost (seconds, peak GPU MB)."""
    import torch
    from torch import nn

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    def prepare(images: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(
            np.log1p(np.asarray(images, dtype=np.float32) / MIP_MEV)
        )[:, None]

    net = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 9 * 36, 64), nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)  # fmt: skip
    x = prepare(images_train).to(device)
    y = torch.from_numpy(np.asarray(y_train, dtype=np.float32))[:, None].to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(EPOCHS):
        order = torch.randperm(len(x), generator=gen).to(device)
        for start in range(0, len(x), BATCH):
            idx = order[start : start + BATCH]
            optimiser.zero_grad()
            loss_fn(net(x[idx]), y[idx]).backward()
            optimiser.step()
    net.eval()
    outputs = []
    with torch.no_grad():
        for images in images_list:
            chunks = [
                torch.sigmoid(net(prepare(images[i : i + 2048]).to(device)))
                .cpu()
                .numpy()[:, 0]
                for i in range(0, len(images), 2048)
            ]
            outputs.append(np.concatenate(chunks))
    peak = float(torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else 0.0
    return outputs, {"seconds": time.time() - t0, "peak_gpu_mb": peak}


# --- data and roles -------------------------------------------------------------------------------------------------------------


def load_block(
    energy: float, representation: str, pool: ProcessPoolExecutor | None
) -> dict[str, Any]:
    """Electrons and ALL protons of one energy and representation: grids, conventional features and the 2D multiscale table."""
    blocks: dict[str, Any] = {}
    for name, sample in (
        ("e", "electron_development_large"),
        ("p", "proton_development_large"),
    ):
        grids = sd.load_events(
            sample, energy, representation, "all", dtype=np.float32
        ).grids
        names, conventional = mt.conventional_features(grids)
        blocks[name] = {
            "grids": grids,
            "names": names,
            "conv": conventional,
            "ms": vr.features(grids, pool),
        }
    return blocks


def rung_matrices(side: dict[str, Any]) -> dict[str, np.ndarray]:
    names, conv = side["names"], side["conv"]
    derived = msgd.derived_blocks(names, conv)
    out = {
        "energy_only": conv[:, [names.index("log_total_energy")]],
        "conv_logistic": conv,
    }
    for rung in RUNGS[1:]:
        out[rung] = derived[rung]
        out[f"{rung}+ms"] = np.hstack([derived[rung], side["ms"]])
    return out


def split_roles(n_electrons: int, n_protons: int, seed: int) -> dict[str, np.ndarray]:
    """Disjoint role index sets; protons are offset by ``n_electrons`` so that one key space serves the overlap check."""
    rng = np.random.default_rng(seed)
    e, p = rng.permutation(n_electrons), rng.permutation(n_protons)
    roles = {
        "calibration_electrons": e[:N_CALIBRATION],
        "evaluation_electrons": e[N_CALIBRATION : N_CALIBRATION + N_EVAL_ELECTRONS],
        "electron_training_pool": e[N_CALIBRATION + N_EVAL_ELECTRONS :],
        "evaluation_protons": p[:N_EVAL_PROTONS] + n_electrons,
        "proton_training_pool": p[N_EVAL_PROTONS:] + n_electrons,
    }
    rl.assert_disjoint_roles(roles)
    return roles


def run_block(args: tuple[float, str, int, tuple[int, ...], bool]) -> dict[str, Any]:
    energy, representation, repeats, sizes, with_cnn = args
    with ProcessPoolExecutor(vr.WORKERS) as pool:
        data = load_block(energy, representation, pool)
    n_e, n_p = len(data["e"]["grids"]), len(data["p"]["grids"])
    roles = split_roles(n_e, n_p, SEED + int(energy))
    matrices = {side: rung_matrices(data[side]) for side in ("e", "p")}
    e_cal, e_eval = roles["calibration_electrons"], roles["evaluation_electrons"]
    p_eval = roles["evaluation_protons"] - n_e
    points: list[dict[str, Any]] = []
    for n in sizes:
        for r in range(repeats):
            rng = np.random.default_rng([SEED, int(energy), n, r])
            e_idx = rng.choice(roles["electron_training_pool"], n, replace=False)
            p_idx = rng.choice(roles["proton_training_pool"] - n_e, n, replace=False)
            y = np.r_[np.ones(n), np.zeros(n)]
            row: dict[str, Any] = {"size_per_class": n, "repeat": r, "models": {}}
            for model in BASE_MODELS:
                if model == "cnn_raw":
                    if not with_cnn:
                        continue
                    x_train = np.concatenate(
                        [data["e"]["grids"][e_idx], data["p"]["grids"][p_idx]]
                    )
                    images = [
                        data["e"]["grids"][e_cal],
                        data["e"]["grids"][e_eval],
                        data["p"]["grids"][p_eval],
                    ]
                    (pc, pe, pp), cost = fit_predict_cnn(x_train, y, images, SEED + r)
                    row["models"][model] = {**metrics(pc, pe, pp), **cost}
                    continue
                t0 = time.time()
                x_e, x_p = matrices["e"][model], matrices["p"][model]
                pc, pe, pp = fit_predict_tabular(
                    "logistic" if model == "conv_logistic" else "hgb",
                    np.vstack([x_e[e_idx], x_p[p_idx]]),
                    y,
                    [x_e[e_cal], x_e[e_eval], x_p[p_eval]],
                )
                row["models"][model] = {
                    **metrics(pc, pe, pp),
                    "seconds": time.time() - t0,
                }
            points.append(row)
    return {
        "energy_gev": energy,
        "representation": representation,
        "roles": {k: len(v) for k, v in roles.items()},
        "points": points,
    }


def paired_differences(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Per size: mean and standard deviation over repeats of (log-loss with +ms) - (log-loss without), for every rung (spread, not an interval)."""
    out: dict[str, Any] = {}
    for rung in RUNGS[1:]:
        for n in sorted({p["size_per_class"] for p in points}):
            d = [
                p["models"][f"{rung}+ms"]["log_loss"] - p["models"][rung]["log_loss"]
                for p in points
                if p["size_per_class"] == n
            ]
            out[f"{rung}|{n}"] = {
                "mean": float(np.mean(d)),
                "sd_over_repeats": float(np.std(d, ddof=1))
                if len(d) > 1
                else float("nan"),
                "repeats": len(d),
            }
    return out


def same_class_null(
    energy: float,
    representation: str,
    with_cnn: bool = True,
    n: int = 1000,
    seed: int = SEED,
) -> dict[str, float]:
    """Electron-against-electron halves for the model classes: the held-out AUC must be 0.5 within 0.02."""
    grids = sd.load_events(
        "electron_development_large", energy, representation, "all", dtype=np.float32
    ).grids
    _, conv = mt.conventional_features(grids)
    order = np.random.default_rng(seed).permutation(len(grids))
    a, b = order[:n], order[n : 2 * n]
    test_a, test_b = order[2 * n : 3 * n], order[3 * n : 4 * n]
    y = np.r_[np.ones(n), np.zeros(n)]
    out = {}
    for kind, label in (("hgb", "conv_hgb"), ("logistic", "conv_logistic")):
        pa, pb = fit_predict_tabular(
            kind, np.vstack([conv[a], conv[b]]), y, [conv[test_a], conv[test_b]]
        )
        out[label] = float(roc_auc_score(y, np.r_[pa, pb]))
    if with_cnn:
        (pa, pb), _ = fit_predict_cnn(
            np.concatenate([grids[a], grids[b]]),
            y,
            [grids[test_a], grids[test_b]],
            seed,
        )
        out["cnn_raw"] = float(roc_auc_score(y, np.r_[pa, pb]))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--no-cnn", action="store_true")
    parser.add_argument(
        "--blocks",
        nargs="+",
        default=["20:deposit_grid_mev", "100:deposit_grid_mev", "100:readout_grid_mev"],
    )
    args = parser.parse_args(argv)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "registration": "research/COMPLETED_WORK.md (was 2026-10-16_benchmark_harness_dry_run_registration.md)",
        "status": "DRY RUN; exploratory",
        "blocks": [],
        "same_class_null": {},
    }
    t0 = time.time()
    for spec in args.blocks:
        energy_text, representation = spec.split(":")
        energy = float(energy_text)
        block = run_block(
            (energy, representation, args.repeats, tuple(args.sizes), not args.no_cnn)
        )
        block["paired_log_loss_difference_ms"] = paired_differences(block["points"])
        report["blocks"].append(block)
        report["same_class_null"][spec] = same_class_null(
            energy, representation, not args.no_cnn
        )
        print(f"{spec} done {time.time() - t0:.0f}s", flush=True)
        RESULT_PATH.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("written", round(time.time() - t0))


if __name__ == "__main__":
    main()
