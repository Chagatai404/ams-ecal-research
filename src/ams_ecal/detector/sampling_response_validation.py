"""Validation of R-A (``sampling_response``) on the development sets; writes ``r_a_closure.json``.

Registered before the run (proposal D6.8, 2026-10-12):

* split: events with an even ``event_index`` fit the model, odd ``event_index`` validate it; no sealed set is read;
* pass criterion: at every energy the mean fibre energy per layer, predicted from the held-out deposits, is within
  3% of the Geant4 value, for all 18 layers;
* descriptive (no threshold, reported as measured): cell-value quantiles and the per-event relative width of the
  total fibre energy, model versus Geant4;
* transfer check: the electron-fitted model applied to protons (expected to fail; documents the scope limit).
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from ams_ecal.detector.sampling_response import N_LAYERS, SamplingResponse
from ams_ecal.validation.sample_definitions import (
    PROJECT_ROOT,
    energies_available,
    load_events,
)

CLOSURE_TOLERANCE = 0.03
SEED = 20261012
QUANTILES = (0.5, 0.9, 0.99, 0.999)
RESULT_PATH = PROJECT_ROOT / "results" / "detector_response" / "r_a_closure.json"
MODEL_PATH = PROJECT_ROOT / "results" / "detector_response" / "r_a_electron_model.json"


def _split(
    sample: str, energy: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    deposit = load_events(sample, energy, "deposit_grid_mev", dtype=np.float32)
    fibre = load_events(sample, energy, "readout_grid_mev", dtype=np.float32)
    even = deposit.event_index % 2 == 0
    return (
        deposit.grids[even],
        fibre.grids[even],
        deposit.grids[~even],
        fibre.grids[~even],
    )


def layer_closure(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Relative difference of the mean fibre energy per layer, predicted over Geant4 minus one."""

    return (
        predicted.sum(axis=(0, 2), dtype=np.float64)
        / actual.sum(axis=(0, 2), dtype=np.float64)
        - 1.0
    )


def _descriptive(predicted: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    positive = actual > 0
    total_p = predicted.sum(axis=(1, 2), dtype=np.float64)
    total_a = actual.sum(axis=(1, 2), dtype=np.float64)
    return {
        "cell_quantiles": list(QUANTILES),
        "cell_quantile_model_mev": np.quantile(predicted[positive], QUANTILES).tolist(),
        "cell_quantile_geant4_mev": np.quantile(actual[positive], QUANTILES).tolist(),
        "fraction_zero_geant4": float(np.mean(actual == 0)),
        "fraction_zero_model": float(np.mean(predicted == 0)),
        "event_total_relative_width_model": float(total_p.std() / total_p.mean()),
        "event_total_relative_width_geant4": float(total_a.std() / total_a.mean()),
    }


def run(
    electron_sample: str = "electron_development_large",
    proton_sample: str = "proton_development_large",
) -> dict[str, Any]:
    energies = energies_available(electron_sample)
    rng = np.random.default_rng(SEED)
    held: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    fit_deposit, fit_fibre = [], []
    for energy in energies:
        deposit_fit, fibre_fit, deposit_val, fibre_val = _split(electron_sample, energy)
        fit_deposit.append(deposit_fit)
        fit_fibre.append(fibre_fit)
        held[energy] = (deposit_val, fibre_val)
    model = SamplingResponse.fit(
        np.concatenate(fit_deposit),
        np.concatenate(fit_fibre),
        particle="electron",
        fitted_from=f"{electron_sample}, even event_index, energies {energies} GeV",
    )
    del fit_deposit, fit_fibre
    electron_rows = {}
    for energy, (deposit_val, fibre_val) in held.items():
        predicted = model.apply(deposit_val, rng)
        closure = layer_closure(predicted, fibre_val)
        electron_rows[f"{energy:g}"] = {
            "n_validation_events": len(deposit_val),
            "layer_closure": closure.tolist(),
            "max_abs_layer_closure": float(np.abs(closure).max()),
            "passes_3_percent": bool(np.abs(closure).max() <= CLOSURE_TOLERANCE),
            **_descriptive(predicted, fibre_val),
        }
    proton_rows = {}
    for energy in energies_available(proton_sample):
        deposit = load_events(
            proton_sample, energy, "deposit_grid_mev", dtype=np.float32
        ).grids[:3000]
        fibre = load_events(
            proton_sample, energy, "readout_grid_mev", dtype=np.float32
        ).grids[:3000]
        closure = layer_closure(model.apply(deposit, rng), fibre)
        proton_rows[f"{energy:g}"] = {
            "max_abs_layer_closure": float(np.abs(closure).max())
        }
    result = {
        "registration": "research/plans/2026-10-12_detector_response_model_proposal_DRAFT.md, D6.8",
        "tolerance": CLOSURE_TOLERANCE,
        "seed": SEED,
        "fit_rule": "even event_index fit, odd event_index validate; development sets only",
        "electron_validation": electron_rows,
        "electron_model_on_protons_transfer_check": proton_rows,
        "layers": N_LAYERS,
    }
    model.save(MODEL_PATH)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def run_proton(
    proton_sample: str = "proton_development_large",
    electron_sample: str = "electron_development_large",
) -> dict[str, Any]:
    """Same registered rule as ``run`` for the per-class table fitted on protons (2026-10-15, researcher: proton table).

    Protons are fitted and validated on all of their interactions together (crossing and interacting); the
    proton-fitted model applied to electrons documents the scope limit, as the electron model on protons does.
    """

    energies = energies_available(proton_sample)
    rng = np.random.default_rng(SEED)
    held: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    fit_deposit, fit_fibre = [], []
    for energy in energies:
        deposit_fit, fibre_fit, deposit_val, fibre_val = _split(proton_sample, energy)
        fit_deposit.append(deposit_fit)
        fit_fibre.append(fibre_fit)
        held[energy] = (deposit_val, fibre_val)
    model = SamplingResponse.fit(
        np.concatenate(fit_deposit),
        np.concatenate(fit_fibre),
        particle="proton",
        fitted_from=f"{proton_sample}, even event_index, energies {energies} GeV",
    )
    del fit_deposit, fit_fibre
    proton_rows = {}
    for energy, (deposit_val, fibre_val) in held.items():
        predicted = model.apply(deposit_val, rng)
        closure = layer_closure(predicted, fibre_val)
        proton_rows[f"{energy:g}"] = {
            "n_validation_events": len(deposit_val),
            "layer_closure": closure.tolist(),
            "max_abs_layer_closure": float(np.abs(closure).max()),
            "passes_3_percent": bool(np.abs(closure).max() <= CLOSURE_TOLERANCE),
            **_descriptive(predicted, fibre_val),
        }
    electron_rows = {}
    for energy in energies_available(electron_sample):
        deposit = load_events(
            electron_sample, energy, "deposit_grid_mev", dtype=np.float32
        ).grids[:3000]
        fibre = load_events(
            electron_sample, energy, "readout_grid_mev", dtype=np.float32
        ).grids[:3000]
        closure = layer_closure(model.apply(deposit, rng), fibre)
        electron_rows[f"{energy:g}"] = {
            "max_abs_layer_closure": float(np.abs(closure).max())
        }
    result = {
        "registration": "research/plans/2026-10-12_detector_response_model_proposal_DRAFT.md, D6.8; proton table approved 2026-10-15",
        "tolerance": CLOSURE_TOLERANCE,
        "seed": SEED,
        "fit_rule": "even event_index fit, odd event_index validate; development sets only",
        "proton_validation": proton_rows,
        "proton_model_on_electrons_transfer_check": electron_rows,
        "layers": N_LAYERS,
    }
    model.save(RESULT_PATH.parent / "r_a_proton_model.json")
    (RESULT_PATH.parent / "r_a_proton_closure.json").write_text(
        json.dumps(result, indent=1), encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--particle", choices=("electron", "proton"), default="electron"
    )
    if parser.parse_args(argv).particle == "proton":
        proton = run_proton()
        for energy, row in proton["proton_validation"].items():
            print(
                f"protons {energy} GeV: max |layer closure| {row['max_abs_layer_closure']:.4f}  pass={row['passes_3_percent']}"
            )
        for energy, row in proton["proton_model_on_electrons_transfer_check"].items():
            print(
                f"proton model on electrons {energy} GeV: max |layer closure| {row['max_abs_layer_closure']:.3f}"
            )
        return
    result = run()
    for energy, row in result["electron_validation"].items():
        print(
            f"electrons {energy} GeV: max |layer closure| {row['max_abs_layer_closure']:.4f}  pass={row['passes_3_percent']}"
        )
    for energy, row in result["electron_model_on_protons_transfer_check"].items():
        print(
            f"electron model on protons {energy} GeV: max |layer closure| {row['max_abs_layer_closure']:.3f}"
        )


if __name__ == "__main__":
    main()
