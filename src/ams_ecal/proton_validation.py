"""Block 6B validation against HELD-OUT Geant4 events.

This is the only place the validation events (``event_index % 4 == 3``) are
read, and they are read here only to COMPARE: nothing in the model was fitted
to them. The comparison follows the contract written before the generator
existed (``research/plans/2026-09-29_block6b_slice0_dependency_analysis.md``,
section 8):

* every aspect passes or fails SEPARATELY - there is no single "realism score";
* a distribution is a MATERIAL DISCREPANCY when the two-sample KS distance is at
  least 0.10 with p < 0.01, or when one of the 5/16/50/84/95% quantiles is off
  by more than 10% AND lies outside the bootstrap 95% interval of the held-out
  value;
* every comparison also reports the NOISE FLOOR: the same statistic between the
  CALIBRATION events and the held-out events, which no model built from the
  calibration events could be expected to beat.

Slice 2 covers the rows that exist: the interaction draw and the crossing
proton. Rows for interacting events are added with their slices.

    uv run python -m ams_ecal.proton_validation

writes ``results/block6b/slice2/crossing_validation.json`` and a figure.
"""

import argparse
import json
import time
from collections.abc import Sequence
from math import exp
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from ams_ecal.geant4_backend import PROJECT_ROOT, load_batch
from ams_ecal.pilot_analysis import event_observables, wilson_interval
from ams_ecal.proton import ProtonShowerModel, spawn_event_seeds
from ams_ecal.proton_calibration import calibration_mask, validation_mask
from ams_ecal.tracking import TrackState

DATA = PROJECT_ROOT / "data" / "geant4_proton_pilot"
RESULTS = PROJECT_ROOT / "results" / "block6b" / "slice2"

ENERGIES_GEV = (10.0, 20.0, 50.0, 100.0)
QUANTILES = (0.05, 0.16, 0.50, 0.84, 0.95)
KS_MATERIAL = 0.10
KS_P_MATERIAL = 0.01
QUANTILE_MATERIAL = 0.10
# the pilot's fixed analysis thresholds (half a measured MIP)
CELL_THRESHOLD_MEV = 0.2798003852367401
LAYER_THRESHOLD_MEV = 0.2843949943780899
N_MODEL_EVENTS = 20_000
ENTRY_SPOT_MM = (0.0, 9.0)


# ----------------------------------------------------------------------
# One distribution against one held-out sample
# ----------------------------------------------------------------------


def compare_distribution(
    model: np.ndarray,
    heldout: np.ndarray,
    calibration: np.ndarray | None = None,
    *,
    n_boot: int = 400,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare a model sample with a held-out sample under the contract.

    ``calibration`` (optional) is the noise-floor sample. Returns the KS
    distance and p-value, the quantiles of both samples, the bootstrap interval
    of each held-out quantile, and the material-discrepancy verdict.
    """

    model, heldout = np.asarray(model, float), np.asarray(heldout, float)
    ks = stats.ks_2samp(model, heldout)
    model_q = np.quantile(model, QUANTILES)
    heldout_q = np.quantile(heldout, QUANTILES)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, len(QUANTILES)))
    for b in range(n_boot):
        boot[b] = np.quantile(heldout[rng.integers(0, len(heldout), len(heldout))], QUANTILES)
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    relative = (model_q - heldout_q) / np.where(heldout_q == 0, np.nan, np.abs(heldout_q))
    outside = (model_q < low) | (model_q > high)
    flagged = (np.abs(relative) > QUANTILE_MATERIAL) & outside
    median = float(np.median(heldout))
    wasserstein = float(stats.wasserstein_distance(model, heldout))
    result: dict[str, Any] = {
        "n_model": len(model),
        "n_heldout": len(heldout),
        "ks": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "quantile_levels": list(QUANTILES),
        "model_quantiles": model_q.tolist(),
        "heldout_quantiles": heldout_q.tolist(),
        "heldout_quantile_interval": [low.tolist(), high.tolist()],
        "relative_quantile_difference": relative.tolist(),
        "wasserstein": wasserstein,
        "wasserstein_over_heldout_median": wasserstein / abs(median) if median != 0 else None,
        "material_by_ks": bool(ks.statistic >= KS_MATERIAL and ks.pvalue < KS_P_MATERIAL),
        "material_by_quantile": [bool(f) for f in flagged],
    }
    result["material"] = bool(result["material_by_ks"] or flagged.any())
    if calibration is not None:
        floor = stats.ks_2samp(np.asarray(calibration, float), heldout)
        result["noise_floor_ks"] = float(floor.statistic)
        result["noise_floor_ks_p"] = float(floor.pvalue)
        result["at_floor"] = bool(ks.statistic <= floor.statistic + 0.02)
    return result


# ----------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------


def split_batch(
    sample: str, energy_gev: float, data: Path = DATA
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Return (calibration, validation) arrays and the metadata of one batch."""

    batch = load_batch(data / sample / f"E{energy_gev:g}GeV")
    n = len(batch)
    index = batch.arrays["event_index"]

    def take(mask: np.ndarray) -> dict[str, np.ndarray]:
        return {
            name: value[mask]
            for name, value in batch.arrays.items()
            if getattr(value, "shape", None) and value.shape[:1] == (n,)
        }

    return take(calibration_mask(index)), take(validation_mask(index)), batch.metadata


def crossing_observables(
    grids: np.ndarray, entry_x: np.ndarray, entry_y: np.ndarray, geometry
) -> dict[str, np.ndarray]:
    """Observables of crossing events, on fibre grids, as in the pilot."""

    observables = event_observables(
        grids,
        geometry,
        entry_x,
        entry_y,
        cell_threshold_mev=CELL_THRESHOLD_MEV,
        layer_threshold_mev=LAYER_THRESHOLD_MEV,
    )
    observables["layer_energy_mev"] = np.asarray(grids, float).sum(axis=2)
    return observables


def generate_crossing_sample(
    model: ProtonShowerModel, energy_gev: float, n_events: int, base_seed: int
) -> dict[str, Any]:
    """Generate ``n_events`` crossing events with the pilot's entry rule.

    Entry points are uniform over the illuminated cell, drawn from a stream
    separate from the generator's own, so they are independent of the draws
    inside the model. Seeds that draw an interaction are skipped (that branch
    is not implemented); the survivors ARE the model's crossing sample.
    """

    energy_mev = 1000.0 * energy_gev
    grids = np.empty((n_events, model.geometry.number_of_layers, model.geometry.cells_per_layer))
    entry = np.empty((n_events, 2))
    total_paths = np.empty(n_events)
    kept = 0
    started = time.perf_counter()
    for seed in spawn_event_seeds(base_seed, 4 * n_events):
        if kept == n_events:
            break
        rng = np.random.default_rng([seed, 1])
        x, y = rng.uniform(*ENTRY_SPOT_MM), rng.uniform(*ENTRY_SPOT_MM)
        track = TrackState(x0_mm=x, y0_mm=y, z0_mm=0.0, theta_rad=0.0, phi_rad=0.0)
        if model.interaction_for_seed(energy_mev, track, seed).interacts:
            continue
        event = model.generate_crossing_event(
            event_id=f"val-{energy_gev:g}-{kept}",
            primary_energy_mev=energy_mev,
            track=track,
            random_seed=seed,
        )
        grids[kept] = np.array(event.cell_energies_mev)
        entry[kept] = (x, y)
        total_paths[kept] = model.crossing.cross(track).chord_mm.sum()
        kept += 1
    if kept < n_events:
        raise RuntimeError("could not draw enough crossing events")
    return {
        "grids": grids,
        "entry_x": entry[:, 0],
        "entry_y": entry[:, 1],
        "chord_total_mm": total_paths,
        "seconds": time.perf_counter() - started,
    }


def chord_totals(model: ProtonShowerModel, entry_x: np.ndarray, entry_y: np.ndarray) -> np.ndarray:
    return np.array(
        [
            model.crossing.cross(
                TrackState(x0_mm=float(x), y0_mm=float(y), z0_mm=0.0, theta_rad=0.0, phi_rad=0.0)
            ).chord_mm.sum()
            for x, y in zip(entry_x, entry_y, strict=True)
        ]
    )


# ----------------------------------------------------------------------
# The two sections
# ----------------------------------------------------------------------


def interaction_section(model: ProtonShowerModel, heldout: dict[str, np.ndarray]) -> dict[str, Any]:
    """Held-out crossing fraction and depth law against the model's exponential."""

    length = model.effective_length_mm
    depth = model.calibration.depth_mm
    occurred = heldout["truth_occurred"]
    crossing = int((~occurred).sum())
    fraction = wilson_interval(crossing, len(occurred))
    expected = exp(-depth / length)
    depths = heldout["truth_z_mm"][occurred]

    def cdf(z: np.ndarray) -> np.ndarray:
        return (1 - np.exp(-z / length)) / (1 - expected)

    ks = stats.kstest(depths, cdf)
    inside = bool(fraction.low <= expected <= fraction.high)
    return {
        "effective_length_mm": length,
        "model_no_inelastic_fraction": expected,
        "heldout_no_inelastic_fraction": [fraction.estimate, fraction.low, fraction.high],
        "fraction_inside_heldout_interval": inside,
        "depth_ks": float(ks.statistic),
        "depth_ks_p": float(ks.pvalue),
        "n_heldout_interacting": int(occurred.sum()),
        "material": bool(not inside or (ks.statistic >= KS_MATERIAL and ks.pvalue < KS_P_MATERIAL)),
    }


def crossing_section(
    model: ProtonShowerModel,
    representation: str,
    energy_gev: float,
    data: Path = DATA,
    n_events: int = N_MODEL_EVENTS,
    base_seed: int = 20260930,
) -> dict[str, Any]:
    """Compare generated crossing protons with held-out Geant4 crossing protons."""

    geometry = model.geometry
    calibration, heldout, _ = split_batch("baseline", energy_gev, data)
    key = {"readout": "readout_grid_mev", "deposition": "deposit_grid_mev"}[representation]
    held = ~heldout["truth_occurred"]
    cal = ~calibration["truth_occurred"]
    generated = generate_crossing_sample(
        model.as_representation(representation), energy_gev, n_events, base_seed
    )
    model_obs = crossing_observables(
        generated["grids"], generated["entry_x"], generated["entry_y"], geometry
    )
    held_obs = crossing_observables(
        heldout[key][held], heldout["entry_x_mm"][held], heldout["entry_y_mm"][held], geometry
    )
    cal_obs = crossing_observables(
        calibration[key][cal],
        calibration["entry_x_mm"][cal],
        calibration["entry_y_mm"][cal],
        geometry,
    )

    out: dict[str, Any] = {
        "representation": representation,
        "n_model": n_events,
        "n_heldout": int(held.sum()),
        "seconds_per_1000_events": 1000 * generated["seconds"] / n_events,
    }
    for name in ("energy_mev", "n_hit_cells", "max_cell_fraction", "containment_fraction"):
        if representation == "deposition" and name != "energy_mev":
            continue  # cell-level activity is defined on the fibre image
        out[name] = compare_distribution(model_obs[name], held_obs[name], cal_obs[name])
    out["layer_energy_pooled"] = compare_distribution(
        model_obs["layer_energy_mev"].ravel(),
        held_obs["layer_energy_mev"].ravel(),
        cal_obs["layer_energy_mev"].ravel(),
    )
    out["mean_layer_energy_mev"] = {
        "model": model_obs["layer_energy_mev"].mean(axis=0).tolist(),
        "heldout": held_obs["layer_energy_mev"].mean(axis=0).tolist(),
        "heldout_standard_error": (
            held_obs["layer_energy_mev"].std(axis=0) / np.sqrt(held.sum())
        ).tolist(),
    }
    out["event_total_variance_over_layer_variance_sum"] = {
        "model": float(
            model_obs["energy_mev"].var() / model_obs["layer_energy_mev"].var(axis=0).sum()
        ),
        "heldout": float(
            held_obs["energy_mev"].var() / held_obs["layer_energy_mev"].var(axis=0).sum()
        ),
    }
    # entry-phase dependence: mean event energy by quartile of the total chord path
    held_chord = chord_totals(model, heldout["entry_x_mm"][held], heldout["entry_y_mm"][held])
    edges = np.quantile(held_chord, [0.25, 0.5, 0.75])
    held_bin = np.searchsorted(edges, held_chord, side="right")
    model_bin = np.searchsorted(edges, generated["chord_total_mm"], side="right")
    rows = []
    for q in range(4):
        h = held_obs["energy_mev"][held_bin == q]
        m = model_obs["energy_mev"][model_bin == q]
        rows.append(
            {
                "quartile": q,
                "model_mean": float(m.mean()),
                "heldout_mean": float(h.mean()),
                "heldout_standard_error": float(h.std() / np.sqrt(len(h))),
                "model_median": float(np.median(m)),
                "heldout_median": float(np.median(h)),
            }
        )
    out["energy_by_chord_quartile"] = rows
    out["material_aspects"] = sorted(
        name
        for name, value in out.items()
        if isinstance(value, dict) and value.get("material") is True
    )
    return out


def build_validation(
    model: ProtonShowerModel,
    data: Path = DATA,
    energies: Sequence[float] = ENERGIES_GEV,
    n_events: int = N_MODEL_EVENTS,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": (
            "Block 6B Slice 2 validation: interaction draw and crossing branch only, against "
            "HELD-OUT Geant4 events (event_index % 4 == 3). Interacting events are not yet generated."
        ),
        "contract": "research/plans/2026-09-29_block6b_slice0_dependency_analysis.md section 8",
        "model": {
            "calibration_content_sha256": model.calibration.content_sha256,
            "physics_list": model.calibration.physics_list,
            "effective_length_mm": model.effective_length_mm,
        },
        "convention": {
            "material_if": f"KS >= {KS_MATERIAL} with p < {KS_P_MATERIAL}, or a quantile off by "
            f"> {QUANTILE_MATERIAL:.0%} and outside the held-out bootstrap 95% interval",
        },
    }
    for energy in energies:
        _, heldout, _ = split_batch("baseline", energy, data)
        entry: dict[str, Any] = {"interaction": interaction_section(model, heldout)}
        for representation in ("readout", "deposition"):
            entry[representation] = crossing_section(model, representation, energy, data, n_events)
        result[f"{energy:g}"] = entry
    return result


def figure(model: ProtonShowerModel, data: Path, out: Path, n_events: int = 5000) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(ENERGIES_GEV), 2, figsize=(10, 3 * len(ENERGIES_GEV)))
    geometry = model.geometry
    for row, energy in enumerate(ENERGIES_GEV):
        calibration, heldout, _ = split_batch("baseline", energy, data)
        held = ~heldout["truth_occurred"]
        cal = ~calibration["truth_occurred"]
        generated = generate_crossing_sample(model, energy, n_events, 20260930)
        model_obs = crossing_observables(
            generated["grids"], generated["entry_x"], generated["entry_y"], geometry
        )
        held_obs = crossing_observables(
            heldout["readout_grid_mev"][held],
            heldout["entry_x_mm"][held],
            heldout["entry_y_mm"][held],
            geometry,
        )
        cal_obs = crossing_observables(
            calibration["readout_grid_mev"][cal],
            calibration["entry_x_mm"][cal],
            calibration["entry_y_mm"][cal],
            geometry,
        )
        ax = axes[row, 0]
        for label, values, style in (
            ("FastMC", model_obs["energy_mev"], "-"),
            ("held-out Geant4", held_obs["energy_mev"], "--"),
            ("calibration Geant4", cal_obs["energy_mev"], ":"),
        ):
            x = np.sort(values)
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), style, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("event fibre energy of a crossing proton (MeV)")
        ax.set_ylabel(f"ECDF, {energy:g} GeV")
        ax.legend(fontsize=7)
        ax = axes[row, 1]
        ax.plot(model_obs["layer_energy_mev"].mean(axis=0), label="FastMC")
        ax.plot(held_obs["layer_energy_mev"].mean(axis=0), "--", label="held-out Geant4")
        ax.set_xlabel("readout layer")
        ax.set_ylabel("mean layer energy (MeV)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--events", type=int, default=N_MODEL_EVENTS)
    args = parser.parse_args(argv)
    model = ProtonShowerModel.from_config()
    args.out.mkdir(parents=True, exist_ok=True)
    result = build_validation(model, args.data, ENERGIES_GEV, args.events)
    (args.out / "crossing_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    figure(model, args.data, args.out / "crossing_validation.png")
    print(args.out / "crossing_validation.json")


if __name__ == "__main__":
    main()
