"""Extended-depth electrons: the longitudinal tail measured, not inferred (analysis only).

Decision record ``research/DECISIONS.md`` (DEC-014, 2026-10-06 rows), note
``research/plans/2026-10-06_em_extended_tail_note.md``. Reads the DEVELOPMENT sample
``data/geant4_electron_extended`` (``configs/geant4_electron_extended.yaml``: electrons at 10, 20, 50, 100 GeV
followed through 270 layers, the 18-layer prefix plus 126 superlayers of compatible material). It is exposed
development data, not a sealed set; no sealed data is read and no generator is changed or written.

The profile-shape study found that a tail component of the longitudinal profile is the largest missing piece,
but that an 18-layer prefix cannot identify its rate, shape or weight. With the energy behind the prefix now
measured this module answers, in order:

1. Is the new sample the same physics as the baseline sample (prefix layers, same energies, other seeds)?
2. Did the prefix-only fits predict the energy behind layer 17 and its distribution? (the test of the tail)
3. What do the 270-layer profiles say about the tail: its decay length, its energy dependence, its weight and
   shape in a gamma-core-plus-tail model fitted over the whole measured depth?
4. At the event level: does the tail weight fluctuate, does it track the shape parameters or the ln T skewness,
   and how much of the event-to-event leakage does a gamma core fitted on the prefix explain?

Layers are 17/18 X0 thick, as in every earlier fit, so results stay comparable; the composite is 0.927 X0 per
layer in Geant4's own radiation length (2% shorter), which scales a decay rate by that factor.
Nothing here changes DEC-001. Words: calibrated, candidate, development; never validated.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from ams_ecal.electron_studies import em_depth_origin_calibration as calibration
from ams_ecal.electron_studies import em_profile_shape_study as shape
from ams_ecal.electron_studies.em_depth_origin_calibration import (
    ModelSpec,
    Targets,
    fit_model,
)
from ams_ecal.electron_studies.em_longitudinal_structure_analysis import (
    _clean,
    _model,
    beta_values,
    fit_free_with_origin,
    joint_diagnostics,
    load_energy,
    marginal_diagnostics,
    profile_fractions_batch,
)
from ams_ecal.geant4_simulation.geant4_backend import PROJECT_ROOT

ENERGIES_GEV = (10.0, 20.0, 50.0, 100.0)
EXTENDED_DIR = PROJECT_ROOT / "data" / "geant4_electron_extended" / "extended"
RESULTS_DIR = PROJECT_ROOT / "results" / "em_generator"
STUDY_JSON = RESULTS_DIR / "extended_tail_study.json"
STUDY_PLOTS = RESULTS_DIR / "extended_tail_study"

LAYER_X0 = 17.0 / 18.0  # X0 per layer, the convention of every earlier fit
PREFIX_LAYERS = 18
FIT_LAYERS = 80  # beyond this the measured energy is below 3e-5 of E at every energy
SE_FLOOR = 3e-6  # layer fractions with no scatter (empty layers) get this standard error
TAIL_WINDOW = (22, 60)  # layers used for the model-free decay length
EVENT_WEIGHT_FLOOR = 1e-5
BOOTSTRAP = 200


def layer_bounds(n_layers: int) -> np.ndarray:
    """``(n_layers, 2)`` front and back depth of each layer in X0 from the front face."""

    k = np.arange(n_layers)
    return np.column_stack([k * LAYER_X0, (k + 1) * LAYER_X0])


@dataclass
class Extended:
    """One energy of the extended-depth sample, as fractions of the primary energy."""

    energy_gev: float
    deposit: np.ndarray  # (n, 270) all-material layer energy / E
    readout: np.ndarray  # (n, 270) scintillator layer energy / E
    entry_x_mm: np.ndarray
    entry_y_mm: np.ndarray

    @property
    def behind_prefix(self) -> np.ndarray:
        return self.deposit[:, PREFIX_LAYERS:].sum(axis=1)


def load_extended(energy_gev: float, directory: Path = EXTENDED_DIR, max_events: int | None = None) -> Extended:
    arrays = np.load(directory / f"E{energy_gev:g}GeV" / "events.npz")
    keep = slice(None, max_events)
    scale = 1000.0 * energy_gev
    return Extended(
        energy_gev,
        arrays["extended_deposit_grid_mev"][keep].sum(axis=2).astype(float) / scale,
        arrays["extended_readout_grid_mev"][keep].sum(axis=2).astype(float) / scale,
        arrays["entry_x_mm"][keep].astype(float),
        arrays["entry_y_mm"][keep].astype(float),
    )


def make_targets(events: dict[float, Extended], n_layers: int, critical_energy_mev: float) -> Targets:
    """Mean layer fractions of the first ``n_layers`` layers with their standard errors (floored)."""

    fractions = [e.deposit[:, :n_layers] for e in events.values()]
    mean = np.array([f.mean(axis=0) for f in fractions])
    se = np.array([np.maximum(f.std(axis=0, ddof=1) / np.sqrt(len(f)), SE_FLOOR) for f in fractions])
    return Targets(
        np.array(list(events), dtype=float), mean, se, fractions, critical_energy_mev, layer_bounds(n_layers)
    )


# ------------------------------------------------------------------ 1. the same physics as the baseline sample


def prefix_consistency(events: dict[float, Extended], baseline: dict[float, Any]) -> dict[str, Any]:
    """Prefix layers of the new sample against the baseline sample (different seeds, same configuration)."""

    out: dict[str, Any] = {}
    for energy, ext in events.items():
        a = ext.deposit[:, :PREFIX_LAYERS]
        b = baseline[energy].fractions["deposition"]
        se = np.sqrt(a.var(axis=0, ddof=1) / len(a) + b.var(axis=0, ddof=1) / len(b))
        z = (a.mean(axis=0) - b.mean(axis=0)) / se
        out[f"{energy:g}"] = {
            "z_by_layer": z.tolist(),
            "max_abs_z": float(np.abs(z).max()),
            "rms_z": float(np.sqrt((z**2).mean())),
            "contained_prefix_extended": float(a.sum(axis=1).mean()),
            "contained_prefix_baseline": float(b.sum(axis=1).mean()),
            "ks_contained_prefix": float(stats.ks_2samp(a.sum(axis=1), b.sum(axis=1), method="asymp").statistic),
            "layer0_mev_extended": float(a[:, 0].mean() * 1000.0 * energy),
            "layer0_mev_baseline": float(b[:, 0].mean() * 1000.0 * energy),
        }
    return out


# ------------------------------------------------------------------ 2. the prefix-only prediction of the tail


def predicted_layer_fractions(
    spec: ModelSpec, params: dict[str, float], energies_gev: np.ndarray, ec_mev: float, n_layers: int
) -> np.ndarray:
    skeleton_mean = np.zeros((len(energies_gev), n_layers))
    skeleton = Targets(energies_gev, skeleton_mean, np.ones_like(skeleton_mean), [], ec_mev, layer_bounds(n_layers))
    model, _ = calibration.predict(spec, {**calibration.DEFAULTS, **spec.fixed, **params}, skeleton)
    return model


def prefix_specs() -> dict[str, ModelSpec]:
    """The shape-study families plus the exponential-onset tail (an exponential behind the profile maximum)."""

    kappa = {"tail_kappa": shape.TAIL_KAPPA_LITERATURE}
    return {
        **shape.extension_specs(),
        "exponential_tail_literature_rate": ModelSpec("exp_lit", ("z0", "beta", "tail_w", "tail_onset"), kappa),
        "exponential_tail_free_rate": ModelSpec("exp_free", ("z0", "beta", "tail_w", "tail_onset", "tail_kappa")),
    }


def prefix_prediction(events: dict[float, Extended], ec_mev: float, families: Sequence[str]) -> dict[str, Any]:
    """Fit each family on the 18 prefix layers only, then compare its energy behind layer 17 with the measurement."""

    specs = prefix_specs()
    prefix = make_targets(events, PREFIX_LAYERS, ec_mev)
    energies = prefix.energies_gev
    measured = np.array([e.behind_prefix.mean() for e in events.values()])
    out: dict[str, Any] = {
        "measured_behind_prefix_mean": measured.tolist(),
        "measured_behind_prefix_sd": [float(e.behind_prefix.std(ddof=1)) for e in events.values()],
        "measured_contained_270_layers": [float(e.deposit.sum(axis=1).mean()) for e in events.values()],
        "measured_contained_prefix": [float(e.deposit[:, :PREFIX_LAYERS].sum(axis=1).mean()) for e in events.values()],
    }
    for name in families:
        spec = specs[name]
        fit = fit_model(spec, prefix)
        params = {k: float(v) for k, v in fit.parameters().items()}
        model = predicted_layer_fractions(spec, params, energies, ec_mev, 270)
        predicted = model[:, PREFIX_LAYERS:].sum(axis=1)
        out[name] = {
            "parameters": {k: params[k] for k in (*spec.free, *spec.fixed)},
            "chi2_per_dof_on_prefix": fit.scale,
            "predicted_contained_prefix": model[:, :PREFIX_LAYERS].sum(axis=1).tolist(),
            "predicted_behind_prefix": predicted.tolist(),
            "difference_predicted_minus_measured": (predicted - measured).tolist(),
            "ratio_predicted_over_measured": (predicted / measured).tolist(),
            "predicted_layers_18_29": model[:, PREFIX_LAYERS:30].sum(axis=1).tolist(),
            "measured_layers_18_29": [
                float(e.deposit[:, PREFIX_LAYERS:30].sum(axis=1).mean()) for e in events.values()
            ],
            "predicted_layers_30_plus": model[:, 30:].sum(axis=1).tolist(),
            "measured_layers_30_plus": [float(e.deposit[:, 30:].sum(axis=1).mean()) for e in events.values()],
        }
    return out


# ------------------------------------------------------------------ 3. the tail in the full measured depth


def decay_length(
    events: dict[float, Extended], window: tuple[int, int] = TAIL_WINDOW, n_boot: int = BOOTSTRAP
) -> dict[str, Any]:
    """Model-free attenuation length: the slope of ln(mean layer energy) against depth in a window of layers."""

    first, last = window
    depth = (np.arange(first, last) + 0.5) * LAYER_X0
    out: dict[str, Any] = {"window_layers": list(window), "window_depth_x0": [first * LAYER_X0, last * LAYER_X0]}
    rng = np.random.default_rng(7)
    for energy, ext in events.items():
        mean = ext.deposit[:, first:last].mean(axis=0)
        positive = mean > 0
        slope = np.polyfit(depth[positive], np.log(mean[positive]), 1)[0]
        boot = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(ext.deposit), len(ext.deposit))
            m = ext.deposit[pick][:, first:last].mean(axis=0)
            ok = m > 0
            boot.append(np.polyfit(depth[ok], np.log(m[ok]), 1)[0])
        out[f"{energy:g}"] = {
            "rate_per_x0": float(-slope),
            "attenuation_length_x0": float(-1.0 / slope),
            "rate_sd_bootstrap": float(np.std(boot, ddof=1)),
            "attenuation_length_x0_central_68": [float(-1.0 / np.percentile(boot, q)) for q in (84, 16)],
        }
    rates = np.array([out[f"{e:g}"]["rate_per_x0"] for e in events])
    out["rate_slope_per_ln_energy"] = (
        float(np.polyfit(np.log(list(events)), rates, 1)[0]) if len(events) >= 2 else None
    )
    return out


def local_decay_rates(events: dict[float, Extended]) -> dict[str, Any]:
    """The decay rate of the mean layer energy in consecutive windows of layers (per X0), by energy."""

    windows = [(14, 20), (20, 26), (26, 32), (32, 40), (40, 50)]
    out: dict[str, Any] = {"windows_layers": windows}
    for energy, ext in events.items():
        mean = ext.deposit.mean(axis=0)
        rates = []
        for first, last in windows:
            depth = (np.arange(first, last) + 0.5) * LAYER_X0
            ok = mean[first:last] > 0
            rates.append(float(-np.polyfit(depth[ok], np.log(mean[first:last][ok]), 1)[0]))
        out[f"{energy:g}"] = rates
    return out


def full_depth_specs() -> dict[str, ModelSpec]:
    kappa = {"tail_kappa": shape.TAIL_KAPPA_LITERATURE}
    return {
        "gamma_only": ModelSpec("g", ("z0", "beta"), amplitude="free"),
        "tail_literature_rate": ModelSpec("tl", ("z0", "beta", "tail_w", "tail_alpha"), kappa, amplitude="free"),
        "tail_free_rate": ModelSpec("tf", ("z0", "beta", "tail_w", "tail_alpha", "tail_kappa"), amplitude="free"),
        "exponential_tail_literature_rate": ModelSpec(
            "el", ("z0", "beta", "tail_w", "tail_onset"), kappa, amplitude="free"
        ),
        "exponential_tail_free_rate": ModelSpec(
            "ef", ("z0", "beta", "tail_w", "tail_onset", "tail_kappa"), amplitude="free"
        ),
    }


def _group_rms(relative: np.ndarray, first: int, last: int) -> list[float]:
    block = relative[:, first:last]
    return [float(np.sqrt(np.mean(row[np.isfinite(row)] ** 2))) for row in block]


def full_depth_fits(events: dict[float, Extended], ec_mev: float) -> dict[str, Any]:
    """Mean-profile fits over the first ``FIT_LAYERS`` layers (the normalisation per energy is free)."""

    targets = make_targets(events, FIT_LAYERS, ec_mev)
    out: dict[str, Any] = {"layers_fitted": FIT_LAYERS, "se_floor": SE_FLOOR}
    for name, spec in full_depth_specs().items():
        fit = fit_model(spec, targets)
        params = fit.parameters()
        model, amplitude = calibration.predict(spec, params, targets)
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = (model - targets.mean) / np.where(targets.mean > 0, targets.mean, np.nan)
        has_rate = "tail_kappa" in spec.free or "tail_kappa" in spec.fixed
        out[name] = {
            "free": list(spec.free),
            "parameters": {k: float(params[k]) for k in (*spec.free, *spec.fixed)},
            "chi2": fit.chi2,
            "dof": fit.dof,
            "chi2_per_dof": fit.scale,
            "amplitude": amplitude.tolist(),
            "rms_relative_layers_0_3": _group_rms(relative, 0, 4),
            "rms_relative_layers_4_17": _group_rms(relative, 4, 18),
            "rms_relative_layers_18_39": _group_rms(relative, 18, 40),
            "relative_residual_by_layer": np.nan_to_num(relative).tolist(),
            "tail_attenuation_length_x0": float(1.0 / params["tail_kappa"]) if has_rate else None,
        }
    return out


# ------------------------------------------------------------------ 4. event level


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.spearmanr(a, b).statistic)


def event_level(events: dict[float, Extended], calibration_origin: float) -> dict[str, Any]:
    """Model-free event statistics of the energy behind the prefix and what a prefix-only gamma core says about it."""

    prefix_bounds = layer_bounds(PREFIX_LAYERS)
    out: dict[str, Any] = {"calibration_origin_x0": calibration_origin}
    for energy, ext in events.items():
        prefix = ext.deposit[:, :PREFIX_LAYERS]
        prefix_fits = np.array([fit_free_with_origin(f, prefix_bounds, calibration_origin)[0] for f in prefix])
        gamma_prefix = profile_fractions_batch(prefix_fits[:, 0], prefix_fits[:, 1], prefix_bounds, calibration_origin)
        gamma_behind = 1.0 - gamma_prefix.sum(axis=1)
        measured = ext.behind_prefix
        residual = measured - gamma_behind
        deep = ext.deposit[:, 30:].sum(axis=1)
        out[f"{energy:g}"] = {
            "behind_prefix_measured": marginal_diagnostics(measured),
            "behind_prefix_from_prefix_gamma": marginal_diagnostics(gamma_behind),
            "behind_prefix_residual": marginal_diagnostics(residual),
            "layers_30_plus_measured": marginal_diagnostics(deep),
            "mean_behind_prefix_residual": float(residual.mean()),
            "sd_behind_prefix_measured": float(measured.std(ddof=1)),
            "sd_behind_prefix_from_prefix_gamma": float(gamma_behind.std(ddof=1)),
            "spearman_measured_vs_prefix_gamma_behind": _spearman(measured, gamma_behind),
            "r2_of_prefix_gamma_for_behind": float(np.corrcoef(measured, gamma_behind)[0, 1] ** 2),
            "spearman_residual_ln_t_prefix": _spearman(residual, prefix_fits[:, 0]),
            "spearman_residual_ln_alpha_prefix": _spearman(residual, prefix_fits[:, 1]),
            "spearman_residual_layer0": _spearman(residual, prefix[:, 0]),
            "spearman_residual_entry_x": _spearman(residual, ext.entry_x_mm),
            "spearman_residual_entry_y": _spearman(residual, ext.entry_y_mm),
            "spearman_layers_30_plus_vs_behind": _spearman(deep, measured),
            "spearman_layers_30_plus_ln_t_prefix": _spearman(deep, prefix_fits[:, 0]),
            "ln_t_prefix_skewness": float(stats.skew(prefix_fits[:, 0])),
            "ln_t_prefix": marginal_diagnostics(prefix_fits[:, 0]),
            "ln_alpha_prefix": marginal_diagnostics(prefix_fits[:, 1]),
            "joint_prefix": joint_diagnostics(prefix_fits[:, 0], prefix_fits[:, 1]),
            "beta_median_prefix": float(np.median(beta_values(prefix_fits[:, 0], prefix_fits[:, 1]))),
        }
    return out


# ------------------------------------------------------------------ orchestration


def analyse(
    energies: Sequence[float] = ENERGIES_GEV,
    *,
    directory: Path = EXTENDED_DIR,
    max_events: int | None = None,
    families: Sequence[str] = (
        "primary",
        shape.CHOSEN_TAIL_FAMILY,
        "tail_free_rate",
        "tail_literature_rate_energy_weight",
        "exponential_tail_literature_rate",
        "exponential_tail_free_rate",
    ),
    event_fits: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    events = {float(e): load_extended(e, directory, max_events) for e in energies}
    baseline = {float(e): load_energy(e) for e in energies}
    ec_mev = _model("deposition").longitudinal.critical_energy_mev
    results: dict[str, Any] = {
        "sample": {
            "events_per_energy": {f"{e:g}": len(v.deposit) for e, v in events.items()},
            "layers": int(next(iter(events.values())).deposit.shape[1]),
            "contained_fraction_all_layers_mean": {
                f"{e:g}": float(v.deposit.sum(axis=1).mean()) for e, v in events.items()
            },
        }
    }
    artefacts: dict[str, Any] = {"events": events}
    results["A_prefix_consistency"] = prefix_consistency(events, baseline)
    results["B_prefix_only_prediction"] = prefix_prediction(events, ec_mev, families)
    results["C_decay_length"] = decay_length(events)
    results["C_local_decay_rates"] = local_decay_rates(events)
    deep = full_depth_fits(events, ec_mev)
    results["C_full_depth_fits"] = deep
    artefacts["deep"] = deep
    if event_fits:
        calibration_origin = float(results["B_prefix_only_prediction"]["primary"]["parameters"]["z0"])
        results["D_event_level"] = event_level(events, calibration_origin)
    return results, artefacts


def write_tables(results: dict[str, Any], directory: Path) -> None:
    prediction = results["B_prefix_only_prediction"]
    energies = list(results["sample"]["events_per_energy"])
    with (directory / "extended_tail_study_prediction.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["energy_gev", "model", "predicted_behind_prefix", "measured_behind_prefix", "ratio"])
        for k, energy in enumerate(energies):
            measured = prediction["measured_behind_prefix_mean"][k]
            writer.writerow([energy, "measured", measured, measured, 1.0])
        for name, row in prediction.items():
            if not isinstance(row, dict) or "predicted_behind_prefix" not in row:
                continue
            for k, energy in enumerate(energies):
                writer.writerow(
                    [energy, name, row["predicted_behind_prefix"][k], prediction["measured_behind_prefix_mean"][k],
                     row["ratio_predicted_over_measured"][k]]
                )
    with (directory / "extended_tail_study_decay_length.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["energy_gev", "rate_per_x0", "rate_sd", "attenuation_length_x0"])
        for energy in energies:
            row = results["C_decay_length"][energy]
            writer.writerow([energy, row["rate_per_x0"], row["rate_sd_bootstrap"], row["attenuation_length_x0"]])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=STUDY_JSON)
    parser.add_argument("--plots", type=Path, default=STUDY_PLOTS)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    results, artefacts = analyse()
    document = {
        "status": (
            "ANALYSIS ONLY on the EXPOSED extended-depth Geant4 electrons (development sample, "
            "configs/geant4_electron_extended.yaml). No generator was written or changed; no sealed data was read."
        ),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **_clean(results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2), encoding="utf-8")
    write_tables(results, args.out.parent)
    if not args.no_plots:
        from ams_ecal.electron_studies.em_extended_tail_study_plots import make_plots

        make_plots(results, artefacts, args.plots)
    print(args.out)


if __name__ == "__main__":
    main()
