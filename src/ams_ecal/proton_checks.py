"""Block 6B Slice 0: paired configuration-variant checks (plan sections 25.1, 25.2).

Two compact Geant4 samples were run at the SAME base seed as the baseline, so
event ``i`` of a variant starts from the same seed and the same entry point as
event ``i`` of the baseline:

* ``cut_0p1mm``                 production cut 0.1 mm instead of 0.7 mm;
* ``material_relative_volume``  lead+glue matrix built from AMS's published
                                lead:fibre:glue volume ratio instead of being
                                density-matched to 6.8 g/cm^3.

The question for each is narrow: is the change to the quantities Block 6B will
calibrate bigger than what is already accepted as stochastic or model spread?
It is not to tune either setting.

MATERIALITY CONVENTION (written before the comparison was run, 2026-09-29): a
shift in a median or a mean is MATERIAL only if it exceeds 5% of the baseline
value AND twice its bootstrap standard error. The accepted model-dependence
scale for comparison is the FTFP_BERT / QGSP_BERT visible-energy difference of
about 13-28% found in the pilot.

Only CALIBRATION events (``ams_ecal.proton_calibration``) with an index present
in both samples are used, so no held-out event enters the comparison.

    uv run python -m ams_ecal.proton_checks

writes ``results/block6b/slice0/variant_checks.json``.
"""

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from ams_ecal.geant4_backend import PROJECT_ROOT, load_batch
from ams_ecal.geometry import ECALGeometry, load_geometry
from ams_ecal.pilot_analysis import (
    bootstrap_interval,
    censored_exponential_rate,
    event_observables,
    wilson_interval,
)
from ams_ecal.proton_calibration import calibration_mask

DATA = PROJECT_ROOT / "data" / "geant4_proton_pilot"
RESULTS = PROJECT_ROOT / "results" / "block6b" / "slice0"

MATERIAL_FRACTION = 0.05
MATERIAL_SIGMAS = 2.0
PILOT_MIP_CELL_MEV = 0.5596007704734802
PILOT_CELL_THRESHOLD_MEV = 0.2798003852367401
PILOT_LAYER_THRESHOLD_MEV = 0.2843949943780899
TINY_CELL_FRACTION_OF_MIP = 0.1


def is_material(estimate: float, standard_error: float, baseline_value: float) -> bool:
    """Apply the materiality convention to one shift."""

    shift = abs(estimate)
    return bool(
        shift > MATERIAL_FRACTION * abs(baseline_value)
        and shift > MATERIAL_SIGMAS * standard_error
    )


def _statistic_ratio(reducer: Callable[[np.ndarray], float]) -> Callable[..., float]:
    def ratio(baseline: np.ndarray, variant: np.ndarray) -> float:
        return float(reducer(variant) / reducer(baseline))

    return ratio


def paired_ratio(
    baseline: np.ndarray,
    variant: np.ndarray,
    reducer: Callable[[np.ndarray], float] = np.median,
    n_boot: int = 400,
    seed: int = 0,
) -> dict[str, Any]:
    """Return ``reducer(variant) / reducer(baseline)`` with a paired bootstrap.

    Rows are resampled jointly, so the pairing (same seed, same primary) is
    respected. ``material`` applies the convention to ``ratio - 1``.
    """

    base, var = np.asarray(baseline, float), np.asarray(variant, float)
    interval = bootstrap_interval(
        _statistic_ratio(reducer), [base, var], n_boot=n_boot, seed=seed
    )
    standard_error = (interval.high - interval.low) / (2 * 1.96)
    return {
        "ratio": interval.estimate,
        "interval": [interval.low, interval.high],
        "standard_error": standard_error,
        "material": is_material(interval.estimate - 1.0, standard_error, 1.0),
    }


def load_aligned(
    baseline_sample: str, variant_sample: str, energy_gev: float, data: Path = DATA
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Return event-aligned CALIBRATION arrays of a baseline and a variant."""

    base = load_batch(data / baseline_sample / f"E{energy_gev:g}GeV")
    var = load_batch(data / variant_sample / f"E{energy_gev:g}GeV")
    n = min(len(base), len(var))
    index = base.arrays["event_index"][:n]
    if not np.array_equal(index, var.arrays["event_index"][:n]):
        raise ValueError("event indices of the two samples do not line up")
    if not np.array_equal(base.arrays["seed"][:n], var.arrays["seed"][:n]):
        raise ValueError("the two samples were not run at the same seeds")
    keep = calibration_mask(index)

    def take(batch: Any) -> dict[str, np.ndarray]:
        return {
            name: value[:n][keep]
            for name, value in batch.arrays.items()
            if getattr(value, "shape", None) and value.shape[:1] == (len(batch),)
        }

    return take(base), take(var), var.metadata


def _observables(
    arrays: dict[str, np.ndarray], key: str, geometry: ECALGeometry
) -> dict[str, np.ndarray]:
    return event_observables(
        arrays[key],
        geometry,
        arrays["entry_x_mm"],
        arrays["entry_y_mm"],
        cell_threshold_mev=PILOT_CELL_THRESHOLD_MEV,
        layer_threshold_mev=PILOT_LAYER_THRESHOLD_MEV,
    )


def _tiny_cells(grid: np.ndarray) -> np.ndarray:
    return (np.asarray(grid) > TINY_CELL_FRACTION_OF_MIP * PILOT_MIP_CELL_MEV).sum(axis=(1, 2))


def compare_variant(
    baseline: dict[str, np.ndarray],
    variant: dict[str, np.ndarray],
    geometry: ECALGeometry,
    depth_mm: float,
) -> dict[str, Any]:
    """Compare a variant with the baseline on the quantities 6B calibrates."""

    base_occ, var_occ = baseline["truth_occurred"], variant["truth_occurred"]
    out: dict[str, Any] = {"n_events": len(base_occ)}

    # --- interaction ---------------------------------------------------
    def interaction(occurred: np.ndarray, depth: np.ndarray) -> dict[str, Any]:
        crossing = int((~occurred).sum())
        rate = censored_exponential_rate(depth[occurred], crossing, depth_mm)
        fraction = wilson_interval(crossing, len(occurred))
        return {
            "no_inelastic_fraction": [fraction.estimate, fraction.low, fraction.high],
            "lambda_eff_mm": 1.0 / rate.rate_per_mm,
            "lambda_eff_relative_error": 1.0 / np.sqrt(max(rate.n_interacting, 1)),
        }

    out["interaction"] = {
        "baseline": interaction(base_occ, baseline["truth_z_mm"]),
        "variant": interaction(var_occ, variant["truth_z_mm"]),
        "status_agreement": float((base_occ == var_occ).mean()),
    }

    # --- crossing protons (non-interacting in both) --------------------
    both_crossing = ~base_occ & ~var_occ
    b_read = _observables(baseline, "readout_grid_mev", geometry)
    v_read = _observables(variant, "readout_grid_mev", geometry)
    b_dep = _observables(baseline, "deposit_grid_mev", geometry)
    v_dep = _observables(variant, "deposit_grid_mev", geometry)
    crossing: dict[str, Any] = {"n": int(both_crossing.sum())}
    for name, base_values, var_values, reducer in (
        ("fibre_energy_mean", b_read["energy_mev"], v_read["energy_mev"], np.mean),
        ("fibre_energy_median", b_read["energy_mev"], v_read["energy_mev"], np.median),
        ("fibre_energy_sd", b_read["energy_mev"], v_read["energy_mev"], np.std),
        ("total_deposit_mean", b_dep["energy_mev"], v_dep["energy_mev"], np.mean),
        (
            "cells_above_tenth_mip",
            _tiny_cells(baseline["readout_grid_mev"]),
            _tiny_cells(variant["readout_grid_mev"]),
            np.mean,
        ),
        (
            "track_cell_containment",
            b_read["containment_fraction"],
            v_read["containment_fraction"],
            np.mean,
        ),
    ):
        crossing[name] = paired_ratio(
            base_values[both_crossing], var_values[both_crossing], reducer
        )
    ks = stats.ks_2samp(b_read["energy_mev"][both_crossing], v_read["energy_mev"][both_crossing])
    crossing["fibre_energy_ks"] = {"statistic": float(ks.statistic), "p_value": float(ks.pvalue)}
    out["crossing"] = crossing

    # --- interacting protons (interacting in both) ---------------------
    both = base_occ & var_occ
    layer_mm = geometry.mean_readout_slice_thickness_mm
    first = np.floor(baseline["truth_z_mm"][both] / layer_mm).astype(int)
    b_layers = baseline["readout_grid_mev"][both].sum(axis=2)
    v_layers = variant["readout_grid_mev"][both].sum(axis=2)
    rows = np.flatnonzero(first <= 14)
    after = np.arange(1, 4)[None, :]
    b_amp = np.take_along_axis(b_layers[rows], first[rows][:, None] + after, axis=1).sum(axis=1)
    v_amp = np.take_along_axis(v_layers[rows], first[rows][:, None] + after, axis=1).sum(axis=1)
    interacting: dict[str, Any] = {
        "n": int(both.sum()),
        "depth_shift_median_abs_mm": float(
            np.median(np.abs(baseline["truth_z_mm"][both] - variant["truth_z_mm"][both]))
        ),
    }
    metrics = {
        "fibre_energy_median": (b_read["energy_mev"][both], v_read["energy_mev"][both], np.median),
        "fibre_energy_mean": (b_read["energy_mev"][both], v_read["energy_mev"][both], np.mean),
        "total_deposit_median": (b_dep["energy_mev"][both], v_dep["energy_mev"][both], np.median),
        "fibre_over_deposit_median": (
            b_read["energy_mev"][both] / b_dep["energy_mev"][both],
            v_read["energy_mev"][both] / v_dep["energy_mev"][both],
            np.median,
        ),
        "amplitude_A3_median": (b_amp, v_amp, np.median),
        "hit_cells_mean": (b_read["n_hit_cells"][both], v_read["n_hit_cells"][both], np.mean),
        "cells_above_tenth_mip_mean": (
            _tiny_cells(baseline["readout_grid_mev"][both]),
            _tiny_cells(variant["readout_grid_mev"][both]),
            np.mean,
        ),
        "lateral_width_median": (b_read["width_mm"][both], v_read["width_mm"][both], np.median),
        "core_fraction_mean": (
            b_read["core_fraction"][both],
            v_read["core_fraction"][both],
            np.mean,
        ),
        "max_cell_fraction_mean": (
            b_read["max_cell_fraction"][both],
            v_read["max_cell_fraction"][both],
            np.mean,
        ),
        "participation_median": (
            b_read["participation"][both],
            v_read["participation"][both],
            np.median,
        ),
        "long_cog_median": (b_read["long_cog_mm"][both], v_read["long_cog_mm"][both], np.median),
        "long_rms_median": (b_read["long_rms_mm"][both], v_read["long_rms_mm"][both], np.median),
    }
    for name, (base_values, var_values, reducer) in metrics.items():
        interacting[name] = paired_ratio(base_values, var_values, reducer)
    for name, key in (
        ("fibre_energy", "energy_mev"),
        ("hit_cells", "n_hit_cells"),
        ("lateral_width", "width_mm"),
    ):
        ks = stats.ks_2samp(b_read[key][both], v_read[key][both])
        interacting[f"{name}_ks"] = {"statistic": float(ks.statistic), "p_value": float(ks.pvalue)}
    out["interacting"] = interacting
    return out


def material_flags(comparison: dict[str, Any]) -> list[str]:
    """Return the names of every compared quantity flagged material."""

    flagged = []
    for section in ("crossing", "interacting"):
        for name, value in comparison[section].items():
            if isinstance(value, dict) and value.get("material"):
                flagged.append(f"{section}.{name}")
    return flagged


def build_checks(data: Path = DATA, energies: Sequence[float] = (10.0, 100.0)) -> dict[str, Any]:
    geometry = load_geometry(PROJECT_ROOT / "configs" / "geometry.yaml")
    depth_mm = geometry.number_of_layers * geometry.mean_readout_slice_thickness_mm
    result: dict[str, Any] = {
        "convention": {
            "material_if_shift_exceeds_fraction": MATERIAL_FRACTION,
            "and_exceeds_sigmas": MATERIAL_SIGMAS,
            "model_scale_for_comparison": "FTFP_BERT vs QGSP_BERT visible energy, about 13-28%",
        }
    }
    for variant in ("cut_0p1mm", "material_relative_volume"):
        if not (data / variant).is_dir():
            continue
        result[variant] = {}
        for energy in energies:
            if not (data / variant / f"E{energy:g}GeV" / "events.npz").is_file():
                continue
            baseline, varied, metadata = load_aligned("baseline", variant, energy, data)
            comparison = compare_variant(baseline, varied, geometry, depth_mm)
            comparison["material_flags"] = material_flags(comparison)
            comparison["variant_metadata"] = {
                "production_cut_mm": metadata["production_cut_mm"],
                "matrix_constraint": metadata["materials"]["matrix_constraint"],
                "composite_density_g_cm3": metadata["materials"]["composite_density_g_cm3"],
                "prefix_depth_lambda_i": metadata["materials"]["prefix_depth_lambda_i"],
                "prefix_depth_x0": metadata["materials"]["prefix_depth_x0"],
                "configuration_sha256": metadata["configuration_sha256"],
                "git_commit": metadata["git"]["commit"],
            }
            result[variant][f"{energy:g}"] = comparison
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    checks = build_checks(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "variant_checks.json"
    path.write_text(json.dumps(checks, indent=2, default=float), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
