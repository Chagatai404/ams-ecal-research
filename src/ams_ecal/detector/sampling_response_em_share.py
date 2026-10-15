"""Electromagnetic-share-conditioned proton class tables, exactly as registered in design section 17 (2026-10-15).

One ``SamplingResponse`` per bin of the event's electromagnetic deposit share (truth, from the component tags); a validation event
is predicted with the table of its own bin. Baseline: one table without bins on the same events. Results are conditional on the
share and are not a standalone R-A.

    python -m ams_ecal.detector.sampling_response_em_share
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from ams_ecal.detector.sampling_response import N_LAYERS, SamplingResponse
from ams_ecal.detector.sampling_response_validation import (
    CLOSURE_TOLERANCE,
    RESULT_PATH,
    SEED,
    layer_closure,
)
from ams_ecal.validation.sample_definitions import PROJECT_ROOT

SHARE_EDGES = (
    0.3,
    0.5,
    0.65,
    0.8,
    0.9,
)  # bins [0, .3), [.3, .5), [.5, .65), [.65, .8), [.8, .9), [.9, 1]
N_BINS = len(SHARE_EDGES) + 1
ROOT = PROJECT_ROOT / "data" / "geant4_component_proton" / "train"
LARGE_ROOT = PROJECT_ROOT / "data" / "geant4_component_proton_large" / "train"  # 20,000 per energy at 10, 20, 50, 100 GeV


def load_energy(energy: float, root=ROOT) -> dict[str, np.ndarray]:
    with np.load(root / f"E{energy:g}GeV" / "events.npz") as data:
        components = data["component_deposit_grid_mev"]
        em = components[:, 0].sum(axis=(1, 2), dtype=np.float64)
        total = np.maximum(components.sum(axis=(1, 2, 3), dtype=np.float64), 1e-12)
        return {
            "deposit": data["deposit_grid_mev"],
            "fibre": data["readout_grid_mev"],
            "index": data["event_index"],
            "share_bin": np.digitize(em / total, SHARE_EDGES),
        }


def _fit(
    parts: list[dict[str, np.ndarray]],
    select: list[np.ndarray],
    by_bin: bool,
    label: str,
) -> list[SamplingResponse]:
    groups = range(N_BINS) if by_bin else [None]
    models = []
    for g in groups:
        deposit, fibre = [], []
        for part, mask in zip(parts, select, strict=True):
            keep = mask if g is None else mask & (part["share_bin"] == g)
            deposit.append(part["deposit"][keep])
            fibre.append(part["fibre"][keep])
        models.append(
            SamplingResponse.fit(
                np.concatenate(deposit),
                np.concatenate(fibre),
                particle=f"proton_{label}_{g}",
                fitted_from=f"{label} bin {g}",
            )
        )
    return models


def _predict(
    models: list[SamplingResponse],
    part: dict[str, np.ndarray],
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.zeros((int(mask.sum()), N_LAYERS, part["deposit"].shape[2]), np.float32)
    bins = part["share_bin"][mask]
    deposit = part["deposit"][mask]
    for g, model in enumerate(models):
        pick = bins == g if len(models) > 1 else np.ones(len(bins), bool)
        if pick.any():
            out[pick] = model.apply(deposit[pick], rng)
    return out


def _row(
    models: list[SamplingResponse],
    part: dict[str, np.ndarray],
    mask: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    predicted = _predict(models, part, mask, rng)
    actual = part["fibre"][mask]
    closure = layer_closure(predicted, actual)
    index = part["index"][mask]
    halves = [
        float(
            np.abs(
                layer_closure(predicted[index % 4 == r], actual[index % 4 == r])
            ).max()
        )
        for r in (1, 3)
    ]
    per_bin = {}
    bins = part["share_bin"][mask]
    for g in range(N_BINS):
        pick = bins == g
        if pick.sum() >= 30:
            c = layer_closure(predicted[pick], actual[pick])
            per_bin[str(g)] = {"n": int(pick.sum()), "max_abs": float(np.abs(c).max())}
    total_p, total_a = predicted.sum(axis=(1, 2)), actual.sum(axis=(1, 2))
    return {
        "n_validation_events": int(mask.sum()),
        "layer_closure": closure.tolist(),
        "max_abs_layer_closure": float(np.abs(closure).max()),
        "worst_layer": int(np.abs(closure).argmax()),
        "passes_3_percent": bool(np.abs(closure).max() <= CLOSURE_TOLERANCE),
        "half_split_max_abs": halves,
        "per_share_bin": per_bin,
        "event_total_relative_width_model": float(total_p.std() / total_p.mean()),
        "event_total_relative_width_geant4": float(total_a.std() / total_a.mean()),
    }


def run(dataset: str = "small") -> dict[str, Any]:
    """``small``: the 5,000-per-energy set of design section 17; ``large``: the 20,000-per-energy repeat of section 19 (same
    rules, bin edges and criteria; leave-one-energy-out at 20 and 50 GeV, the 5 and 200 GeV energies are not in that set)."""

    root = LARGE_ROOT if dataset == "large" else ROOT
    held_energies = (20.0, 50.0) if dataset == "large" else (5.0, 20.0, 50.0, 200.0)
    energies = sorted(
        float(re.findall(r"[\d.]+", p.name)[0]) for p in root.glob("E*GeV")
    )
    parts = [load_energy(e, root) for e in energies]
    rng = np.random.default_rng(SEED)
    even = [p["index"] % 2 == 0 for p in parts]
    odd = [~m for m in even]
    result: dict[str, Any] = {
        "registration": "research/plans/2026-10-12_particle_agnostic_RA_design_DRAFT.md, section 17",
        "tolerance": CLOSURE_TOLERANCE,
        "share_edges": list(SHARE_EDGES),
        "share_bin_fractions_all_events": {
            f"{e:g}": [float(np.mean(p["share_bin"] == g)) for g in range(N_BINS)]
            for e, p in zip(energies, parts, strict=True)
        },
        "in_sample_energies": {"share_conditioned": {}, "baseline": {}},
        "leave_one_energy_out": {"share_conditioned": {}, "baseline": {}},
    }
    for label, by_bin in (("share_conditioned", True), ("baseline", False)):
        models = _fit(parts, even, by_bin, label)
        for e, p, m in zip(energies, parts, odd, strict=True):
            result["in_sample_energies"][label][f"{e:g}"] = _row(models, p, m, rng)
        for held in held_energies:
            keep = [i for i, e in enumerate(energies) if e != held]
            models = _fit(
                [parts[i] for i in keep],
                [even[i] for i in keep],
                by_bin,
                f"{label}_without_{held:g}",
            )
            i = energies.index(held)
            result["leave_one_energy_out"][label][f"{held:g}"] = _row(
                models, parts[i], odd[i], rng
            )
    name = "r_a_proton_em_share_large_closure.json" if dataset == "large" else "r_a_proton_em_share_closure.json"
    result["dataset"] = dataset
    (RESULT_PATH.parent / name).write_text(
        json.dumps(result, indent=1), encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("small", "large"), default="small")
    result = run(parser.parse_args(argv).dataset)
    for section in ("in_sample_energies", "leave_one_energy_out"):
        print(section)
        for label in ("share_conditioned", "baseline"):
            for e, row in result[section][label].items():
                print(
                    f"  {label:18s} {e:>5s} GeV max {row['max_abs_layer_closure']:.4f} (layer {row['worst_layer']:2d}) "
                    f"pass={row['passes_3_percent']} halves={[f'{h:.3f}' for h in row['half_split_max_abs']]}"
                )


if __name__ == "__main__":
    main()
