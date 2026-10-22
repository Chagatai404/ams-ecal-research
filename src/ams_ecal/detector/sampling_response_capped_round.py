"""R-A capped round, Step A (selection on the old odd events); writes ``r_a_capped_round_selection.json``.

Registered before any fit (2026-10-22, EXP-012, hash 417e7a22681d944d; text in ``~/.research-os-backups/registrations``).
Three candidate versions of the unsplit proton class table, each conditioned on a truth tag from the component tags:

* V1: crossing protons as their own class; interacting protons by electromagnetic share with edges
  0.1, 0.2, 0.3, 0.5, 0.65, 0.8, 0.9;
* V2: electromagnetic share (edges 0.3, 0.5, 0.65, 0.8, 0.9) jointly with the primary energy: one table per
  (share bin, energy), interpolated linearly in log energy for a held-out energy;
* V3: slow-hadron share (had_low + ion + neutron over the event total), edges 0.05, 0.1, 0.2, 0.35.

Fit on the even ``event_index`` events of ``component_proton_large``, validate on the odd ones. A candidate is eligible
if its closure is no larger than the baseline's at every energy and at least one percentage point smaller at two or
more energies; the selected candidate is the eligible one with the smallest maximum-over-energies closure.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ams_ecal.detector.sampling_response import (
    MIN_CLASS_CELLS,
    N_LAYERS,
    SamplingResponse,
    event_class,
    size_class,
)
from ams_ecal.detector.sampling_response_noise_floor import (
    N_SPLITS,
    half_split_floor,
)
from ams_ecal.detector.sampling_response_validation import (
    CLOSURE_TOLERANCE,
    SEED,
    layer_closure,
)
from ams_ecal.validation.sample_definitions import (
    PROJECT_ROOT,
    energies_available,
    locate,
)

SAMPLE = "component_proton_large"
FLOOR_SEED = 20261022
RESULT_PATH = (
    PROJECT_ROOT / "results" / "detector_response" / "r_a_capped_round_selection.json"
)
V1_SHARE_EDGES = (0.1, 0.2, 0.3, 0.5, 0.65, 0.8, 0.9)
V2_SHARE_EDGES = (0.3, 0.5, 0.65, 0.8, 0.9)
V3_SLOW_EDGES = (0.05, 0.1, 0.2, 0.35)
SLOW_COMPONENTS = (1, 4, 5)  # had_low, ion, neutron in COMPONENTS
THIN_FALLBACK_FRACTION = 0.20
MIN_IMPROVEMENT = 0.01
MIN_ENERGIES_IMPROVED = 2


def classes_v1(share: np.ndarray, interacting: np.ndarray) -> np.ndarray:
    """Class 0: crossing; classes 1..8: interacting protons by electromagnetic share."""

    return np.where(interacting, 1 + np.digitize(share, V1_SHARE_EDGES), 0)


def classes_v2(share: np.ndarray) -> np.ndarray:
    return np.digitize(share, V2_SHARE_EDGES)


def classes_v3(slow_share: np.ndarray) -> np.ndarray:
    return np.digitize(slow_share, V3_SLOW_EDGES)


def interpolate_in_log_energy(
    low: SamplingResponse,
    high: SamplingResponse,
    energy_low: float,
    energy_high: float,
    energy: float,
    label: str,
) -> SamplingResponse:
    """Linear interpolation of the quantile tables and class mean ratios in log(energy)."""

    weight = (np.log(energy) - np.log(energy_low)) / (
        np.log(energy_high) - np.log(energy_low)
    )
    return SamplingResponse(
        particle=label,
        tables=(1 - weight) * low.tables + weight * high.tables,
        mean_ratio=(1 - weight) * low.mean_ratio + weight * high.mean_ratio,
        n_fitted_cells=np.minimum(low.n_fitted_cells, high.n_fitted_cells),
        fitted_from=f"log-energy interpolation of {low.fitted_from} and {high.fitted_from}",
    )


def eligible(candidate: dict[float, float], baseline: dict[float, float]) -> bool:
    """No larger than the baseline at any energy and at least 1 point smaller at two or more energies."""

    never_worse = all(candidate[e] <= baseline[e] for e in baseline)
    improved = sum(baseline[e] - candidate[e] >= MIN_IMPROVEMENT for e in baseline)
    return never_worse and improved >= MIN_ENERGIES_IMPROVED


def fallback_fraction(model: SamplingResponse, deposit: np.ndarray) -> float:
    """Fraction of positive-deposit cells whose (layer, size, event) class used a pooled table."""

    sizes = size_class(deposit)
    events = np.broadcast_to(
        event_class(deposit.sum(axis=(1, 2), dtype=np.float64))[:, None, None],
        deposit.shape,
    )
    layers = np.broadcast_to(np.arange(N_LAYERS)[None, :, None], deposit.shape)
    positive = deposit > 0
    fitted = model.n_fitted_cells[layers, sizes, events] >= MIN_CLASS_CELLS
    return float(1.0 - fitted[positive].mean())


def _load(energy: float) -> dict[str, np.ndarray]:
    with np.load(locate(SAMPLE) / f"E{energy:g}GeV" / "events.npz") as data:
        components = data["component_deposit_grid_mev"]
        total = np.maximum(components.sum(axis=(1, 2, 3), dtype=np.float64), 1e-12)
        em = components[:, 0].sum(axis=(1, 2), dtype=np.float64)
        slow = components[:, list(SLOW_COMPONENTS)].sum(
            axis=(1, 2, 3), dtype=np.float64
        )
        return {
            "deposit": data["deposit_grid_mev"],
            "fibre": data["readout_grid_mev"],
            "index": data["event_index"],
            "interacting": data["truth_occurred"].astype(bool),
            "share": em / total,
            "slow": slow / total,
        }


def _fit(
    parts: dict[float, dict[str, np.ndarray]],
    classes: dict[float, np.ndarray],
    n_classes: int,
    label: str,
) -> list[SamplingResponse]:
    """One table per class, pooled over energies, fitted on the even events."""

    models = []
    for g in range(n_classes):
        deposit, fibre = [], []
        for energy, part in parts.items():
            keep = (part["index"] % 2 == 0) & (classes[energy] == g)
            deposit.append(part["deposit"][keep])
            fibre.append(part["fibre"][keep])
        models.append(
            SamplingResponse.fit(
                np.concatenate(deposit),
                np.concatenate(fibre),
                particle=f"{label}_{g}",
                fitted_from=f"{SAMPLE} even events, class {g} of {label}",
            )
        )
    return models


def _fit_per_energy(
    parts: dict[float, dict[str, np.ndarray]],
    classes: dict[float, np.ndarray],
    n_classes: int,
    energies: list[float],
    label: str,
) -> dict[tuple[int, float], SamplingResponse | None]:
    """One table per (class, energy), fitted on the even events of that energy; None if the cell is empty."""

    out: dict[tuple[int, float], SamplingResponse | None] = {}
    for g in range(n_classes):
        for energy in energies:
            part = parts[energy]
            keep = (part["index"] % 2 == 0) & (classes[energy] == g)
            out[(g, energy)] = (
                SamplingResponse.fit(
                    part["deposit"][keep],
                    part["fibre"][keep],
                    particle=f"{label}_{g}_{energy:g}",
                    fitted_from=f"{SAMPLE} even events, class {g}, {energy:g} GeV",
                )
                if keep.sum() > 0
                else None
            )
    return out


def _predict(
    models: list[SamplingResponse | None],
    deposit: np.ndarray,
    class_of_event: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[int, float]]:
    out = np.zeros(deposit.shape, np.float32)
    fallback: dict[int, float] = {}
    for g, model in enumerate(models):
        pick = class_of_event == g
        if model is not None and pick.any():
            out[pick] = model.apply(deposit[pick], rng)
            fallback[g] = fallback_fraction(model, deposit[pick])
    return out, fallback


def _row(
    predicted: np.ndarray,
    actual: np.ndarray,
    class_of_event: np.ndarray,
    n_classes: int,
    fallback: dict[int, float],
) -> dict[str, Any]:
    closure = layer_closure(predicted, actual)
    per_class = {}
    for g in range(n_classes):
        pick = class_of_event == g
        if pick.sum() >= 30:
            c = layer_closure(predicted[pick], actual[pick])
            fraction = fallback.get(g)
            per_class[str(g)] = {
                "n": int(pick.sum()),
                "max_abs": float(np.abs(c).max()),
                "fallback_fraction": fraction,
                "thin": bool(
                    fraction is not None and fraction > THIN_FALLBACK_FRACTION
                ),
            }
    return {
        "n_validation_events": len(actual),
        "layer_closure": closure.tolist(),
        "max_abs_layer_closure": float(np.abs(closure).max()),
        "worst_layer": int(np.abs(closure).argmax()),
        "per_class": per_class,
    }


def run() -> dict[str, Any]:
    energies = [float(e) for e in energies_available(SAMPLE)]
    parts = {e: _load(e) for e in energies}
    odd = {e: parts[e]["index"] % 2 == 1 for e in energies}

    # Pooled-proton noise floor, computed before any fit.
    floor_rng = np.random.default_rng(FLOOR_SEED)
    floors = {}
    for e in energies:
        floor = half_split_floor(
            parts[e]["deposit"].sum(axis=2, dtype=np.float64),
            parts[e]["fibre"].sum(axis=2, dtype=np.float64),
            floor_rng,
        )
        floors[e] = {
            "median": float(np.median(floor)),
            "p95": float(np.quantile(floor, 0.95)),
        }
        print(
            f"floor pooled protons {e:g} GeV median {floors[e]['median']:.4f} p95 {floors[e]['p95']:.4f}"
        )

    rng = np.random.default_rng(SEED)
    rows: dict[str, dict[str, Any]] = {
        "baseline": {},
        "V1_crossing_fine_share": {},
        "V2_share_energy": {},
        "V3_slow_hadron": {},
    }

    def evaluate(name: str, models, class_fn, n_classes: int) -> None:
        for e in energies:
            cls = class_fn(parts[e])
            predicted, fallback = _predict(
                models, parts[e]["deposit"][odd[e]], cls[odd[e]], rng
            )
            rows[name][f"{e:g}"] = _row(
                predicted, parts[e]["fibre"][odd[e]], cls[odd[e]], n_classes, fallback
            )

    zero = {e: np.zeros(len(parts[e]["index"]), dtype=int) for e in energies}
    baseline = _fit(parts, zero, 1, "baseline")
    evaluate("baseline", baseline, lambda p: np.zeros(len(p["index"]), dtype=int), 1)

    n1 = len(V1_SHARE_EDGES) + 2
    v1_class = {
        e: classes_v1(parts[e]["share"], parts[e]["interacting"]) for e in energies
    }
    v1 = _fit(parts, v1_class, n1, "V1")
    evaluate(
        "V1_crossing_fine_share",
        v1,
        lambda p: classes_v1(p["share"], p["interacting"]),
        n1,
    )

    n3 = len(V3_SLOW_EDGES) + 1
    v3_class = {e: classes_v3(parts[e]["slow"]) for e in energies}
    v3 = _fit(parts, v3_class, n3, "V3")
    evaluate("V3_slow_hadron", v3, lambda p: classes_v3(p["slow"]), n3)

    # V2: a table per (share bin, energy); a bin empty at an energy falls back to the bin pooled over energies.
    n2 = len(V2_SHARE_EDGES) + 1
    v2_class = {e: classes_v2(parts[e]["share"]) for e in energies}
    per_energy = _fit_per_energy(parts, v2_class, n2, energies, "V2")
    pooled_bins = _fit(parts, v2_class, n2, "V2pooled")
    empty_cells = [f"{g}@{e:g}" for (g, e), m in per_energy.items() if m is None]
    for e in energies:
        models = [per_energy[(g, e)] or pooled_bins[g] for g in range(n2)]
        cls = v2_class[e]
        predicted, fallback = _predict(
            models, parts[e]["deposit"][odd[e]], cls[odd[e]], rng
        )
        rows["V2_share_energy"][f"{e:g}"] = _row(
            predicted, parts[e]["fibre"][odd[e]], cls[odd[e]], n2, fallback
        )

    # V2 leave-one-energy-out at 20 and 50 GeV (interpolation test, no threshold).
    loeo: dict[str, Any] = {}
    for held in (20.0, 50.0):
        kept = [e for e in energies if e != held]
        low = max(e for e in kept if e < held)
        high = min(e for e in kept if e > held)
        sub = {e: parts[e] for e in kept}
        sub_class = {e: v2_class[e] for e in kept}
        tables = _fit_per_energy(sub, sub_class, n2, kept, f"V2_without_{held:g}")
        pooled = _fit(sub, sub_class, n2, f"V2pooled_without_{held:g}")
        models = []
        for g in range(n2):
            lo_m, hi_m = tables[(g, low)], tables[(g, high)]
            models.append(
                interpolate_in_log_energy(
                    lo_m, hi_m, low, high, held, f"V2_{g}_{held:g}"
                )
                if lo_m is not None and hi_m is not None
                else pooled[g]
            )
        cls = v2_class[held]
        predicted, fallback = _predict(
            models, parts[held]["deposit"][odd[held]], cls[odd[held]], rng
        )
        loeo[f"{held:g}"] = _row(
            predicted, parts[held]["fibre"][odd[held]], cls[odd[held]], n2, fallback
        )

    base_closure = {
        e: rows["baseline"][f"{e:g}"]["max_abs_layer_closure"] for e in energies
    }
    verdict: dict[str, Any] = {}
    for name in ("V1_crossing_fine_share", "V2_share_energy", "V3_slow_hadron"):
        closure = {e: rows[name][f"{e:g}"]["max_abs_layer_closure"] for e in energies}
        verdict[name] = {
            "closure_by_energy": {f"{e:g}": c for e, c in closure.items()},
            "baseline_by_energy": {f"{e:g}": c for e, c in base_closure.items()},
            "eligible": eligible(closure, base_closure),
            "max_over_energies": max(closure.values()),
        }
    pool = {n: v for n, v in verdict.items() if v["eligible"]}
    selected = min(pool, key=lambda n: pool[n]["max_over_energies"]) if pool else None
    result = {
        "registration": "EXP-012 steps 2-5, 2026-10-22, hash 417e7a22681d944d",
        "step": "A (selection on the old odd events)",
        "tolerance": CLOSURE_TOLERANCE,
        "seed": SEED,
        "floor_seed": FLOOR_SEED,
        "floor_n_splits": N_SPLITS,
        "pooled_proton_floor": {f"{e:g}": f for e, f in floors.items()},
        "v2_cells_empty_replaced_by_pooled_bin": empty_cells,
        "rows": rows,
        "v2_leave_one_energy_out": loeo,
        "verdict": verdict,
        "selected": selected,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def main() -> None:
    result = run()
    for name, row in result["rows"].items():
        for energy, detail in row.items():
            print(
                f"{name:24s} {energy:>5s} GeV max {detail['max_abs_layer_closure']:.4f} (layer {detail['worst_layer']:2d})"
            )
    for held, detail in result["v2_leave_one_energy_out"].items():
        print(
            f"V2 leave-one-out {held:>5s} GeV max {detail['max_abs_layer_closure']:.4f}"
        )
    for name, v in result["verdict"].items():
        print(f"{name}: eligible={v['eligible']} max {v['max_over_energies']:.4f}")
    print("selected:", result["selected"])


if __name__ == "__main__":
    main()
