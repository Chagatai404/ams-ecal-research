"""Analysis of the FTFP_BERT_EMV one-at-a-time swap test (S0 follow-up, approved 2026-10-11): which EM parameter lowers the scintillator share?

Background. At 20 GeV the median per-event scintillator share of the deposited energy is about 6.5% for FTFP_BERT, 6.6% for the
0.1 mm cut and EMZ sets, and 4.1% for FTFP_BERT_EMV, with the same total deposit. The initialisation dump found that EMV differs
from FTFP_BERT in two EM parameters only, ``ApplyCuts`` (false -> true) and ``MscRangeFactor`` (0.04 -> 0.2). Three lists restore them
one at a time and together (``geant4_backend.EM_PARAMETER_OVERRIDES``), 2000 electrons at 20 GeV each, with the SAME per-event seeds as
the EMV set, so event i of every swap set is the same primary as event i of the EMV set (paired).

THE RULE, FIXED BEFORE ANY SWAP EVENT EXISTED (never relaxed after the numbers are read):

* share(set) = median over events of ``edep_scintillator / edep_total``; reference values: ``electron_development_large`` (FTFP_BERT,
  20 GeV, 10,000 events) and ``electron_emv`` (20 GeV, 2000 events).
* restored fraction f(swap) = (share(swap) - share(EMV)) / (share(FTFP_BERT) - share(EMV)); 0 = no effect, 1 = fully restored.
  95% interval: bootstrap of events (swap and EMV resampled jointly by paired index; FTFP_BERT independently), 1000 draws.
* closure test (both parameters reverted): PASSES if |share(both) / share(FTFP_BERT) - 1| <= 0.03 (the spread of the FTFP_BERT, cut and
  EMZ medians is 2.3%, rounded up); if it fails, these two parameters are not the whole explanation and the dump or the swap is
  incomplete.
* a single parameter is "mostly responsible" if the lower end of its 95% interval for f is >= 0.5, "not responsible" if the upper end is
  <= 0.2, and "partly responsible / unresolved" otherwise.
* secondary readings, reported not gated: the ratio by layer (readout layer sum / deposit layer sum), and the total deposit ratio to
  EMV (energy conservation: median within 0.5%).

It decides nothing about which list is physically right: restoring the share only identifies the setting that causes it.

    python -m ams_ecal.geant4_simulation.em_variant_swap_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ams_ecal.electron_model.em_production import PROJECT_ROOT
from ams_ecal.validation import sample_definitions as sd

RESULTS_PATH = PROJECT_ROOT / "results" / "em_variants" / "swap_analysis.json"
ENERGY = 20.0
SWAPS = ("electron_emv_cuts_reverted", "electron_emv_msc_reverted", "electron_emv_both_reverted")
CLOSURE_TOLERANCE = 0.03
MOSTLY_LOWER, NOT_UPPER = 0.5, 0.2
BOOTSTRAPS = 1000
SEED = 20261013


def share(edep_scintillator: np.ndarray, edep_total: np.ndarray) -> float:
    """Median over events of the scintillator share of the deposited energy."""

    return float(np.median(np.asarray(edep_scintillator, dtype=float) / np.asarray(edep_total, dtype=float)))


def restored_fraction(share_swap: float, share_emv: float, share_reference: float) -> float:
    return (share_swap - share_emv) / (share_reference - share_emv)


def verdict(interval: tuple[float, float]) -> str:
    low, high = interval
    if low >= MOSTLY_LOWER:
        return "mostly responsible"
    if high <= NOT_UPPER:
        return "not responsible"
    return "partly responsible / unresolved"


def closure_passes(share_both: float, share_reference: float) -> bool:
    return abs(share_both / share_reference - 1.0) <= CLOSURE_TOLERANCE


def bootstrap_restored_fraction(
    swap: tuple[np.ndarray, np.ndarray],
    emv: tuple[np.ndarray, np.ndarray],
    reference: tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
    draws: int = BOOTSTRAPS,
) -> tuple[float, float]:
    """95% interval of f; ``swap`` and ``emv`` are (scintillator, total) arrays in the same event order (paired)."""

    n, m = len(swap[0]), len(reference[0])
    values = []
    for _ in range(draws):
        i, j = rng.integers(0, n, n), rng.integers(0, m, m)
        values.append(
            restored_fraction(
                share(swap[0][i], swap[1][i]), share(emv[0][i], emv[1][i]), share(reference[0][j], reference[1][j])
            )
        )
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def layer_ratio(readout: np.ndarray, deposit: np.ndarray) -> np.ndarray:
    """Median over events of readout layer energy / deposit layer energy, for each of the 18 layers."""

    r, d = readout.sum(axis=2), deposit.sum(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(d > 0, r / d, np.nan)
    return np.nanmedian(ratio, axis=0)


def _arrays(name: str) -> dict[str, np.ndarray]:
    deposit = sd.load_events(name, ENERGY, "deposit_grid_mev", "all", dtype=np.float32)
    readout = sd.load_events(name, ENERGY, "readout_grid_mev", "all", dtype=np.float32)
    return {
        "scint": deposit.edep_scintillator_mev,
        "total": deposit.edep_total_mev,
        "seed": deposit.seed,
        "layer_ratio": layer_ratio(readout.grids, deposit.grids),
    }


def analyse() -> dict:
    rng = np.random.default_rng(SEED)
    reference, emv = _arrays("electron_development_large"), _arrays("electron_emv")
    s_ref, s_emv = share(reference["scint"], reference["total"]), share(emv["scint"], emv["total"])
    out: dict = {
        "status": "EMV swap test, exposed development data, 20 GeV; rule fixed in the module docstring before any swap event existed",
        "share_FTFP_BERT": s_ref,
        "share_EMV": s_emv,
        "ratio_EMV_to_FTFP_BERT": s_emv / s_ref,
        "layer_ratio_FTFP_BERT": reference["layer_ratio"].tolist(),
        "layer_ratio_EMV": emv["layer_ratio"].tolist(),
        "swaps": {},
    }
    for name in SWAPS:
        try:
            swap = _arrays(name)
        except sd.SampleError as error:
            out["swaps"][name] = {"available": False, "reason": str(error)}
            continue
        paired = bool(np.array_equal(swap["seed"], emv["seed"]))
        f = restored_fraction(share(swap["scint"], swap["total"]), s_emv, s_ref)
        interval = bootstrap_restored_fraction(
            (swap["scint"], swap["total"]), (emv["scint"], emv["total"]), (reference["scint"], reference["total"]), rng
        )
        out["swaps"][name] = {
            "available": True,
            "events_paired_with_emv_by_seed": paired,
            "share": share(swap["scint"], swap["total"]),
            "ratio_to_FTFP_BERT": share(swap["scint"], swap["total"]) / s_ref,
            "restored_fraction": f,
            "restored_fraction_95": list(interval),
            "verdict": verdict(interval),
            "total_deposit_median_ratio_to_EMV": float(np.median(swap["total"]) / np.median(emv["total"])),
            "layer_ratio": swap["layer_ratio"].tolist(),
        }
    both = out["swaps"].get("electron_emv_both_reverted", {})
    if both.get("available"):
        out["closure_test_passed"] = closure_passes(both["share"], s_ref)
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = parser.parse_args(argv)
    result = analyse()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"FTFP_BERT {result['share_FTFP_BERT']:.4f}  EMV {result['share_EMV']:.4f}")
    for name, row in result["swaps"].items():
        if row.get("available"):
            print(
                f"{name}: share {row['share']:.4f}, restored {row['restored_fraction']:.2f} "
                f"{row['restored_fraction_95']} -> {row['verdict']}"
            )
        else:
            print(f"{name}: not available yet")
    if "closure_test_passed" in result:
        print("closure test (both reverted) passed:", result["closure_test_passed"])


if __name__ == "__main__":
    main()
