"""Event-by-event longitudinal fluctuations of Geant4 electrons against the Grindhammer-Peters forms.

A reproducible check (research/plans/2026-10-05_em_literature_verification_record.md, adversarial
pass). For each exploration electron (data/geant4_electron_sample, unsealed; the sealed set is not
touched) a gamma-shaped longitudinal profile is fitted to the 18 layer energies of the
``deposition`` grid, divided by the primary energy so that the energy beyond the last layer is the
longitudinal leakage. The fit gives ``ln T`` and ``ln alpha`` per event (``T = (alpha - 1) / beta``
is the depth of the maximum in X0). The report gives their means, standard deviations and
correlation, and the Grindhammer-Peters sampling-calorimeter values for the same energy
(hep-ex/0001020, Appendix A.2.2 as verified from the full text: sigma(ln T) = 1/(-2.5 + 1.25 ln y),
sigma(ln alpha) = 1/(-0.82 + 0.79 ln y), rho = 0.784 - 0.023 ln y, y = E / E_c).

Two fit estimators are compared (least squares on the layer fractions, and on their cumulative
sums) and each is run on a CONTROL: events with a planted correlation of (ln T, ln alpha) and layer
noise, so that the correlation the estimator returns can be told from the correlation that exists.
An earlier unchecked comparison reported a correlation of 0.87-0.90 and called the literature
value refuted; the control exists to say how much of a measured correlation the fit itself makes.

    uv run python -m ams_ecal.electron_studies.em_longitudinal_fluctuations
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from scipy.optimize import least_squares
from scipy.special import gammainc

from ams_ecal.detector.geometry import load_geometry
from ams_ecal.geant4_simulation.geant4_backend import PROJECT_ROOT, load_batch
from ams_ecal.proton_model.proton_validation import crossing_observables
from ams_ecal.validation.dataset import GEOMETRY_CONFIG, load_electron_model

DATA = PROJECT_ROOT / "data" / "geant4_electron_sample" / "baseline"
RESULTS = PROJECT_ROOT / "results" / "em_generator" / "longitudinal_fluctuations.json"
ENERGIES_GEV = (10.0, 20.0, 50.0, 100.0)
START = (1.9, 1.6)  # (ln T, ln alpha) starting point of every fit
LOWER, UPPER = (-1.0, 0.05), (4.0, 4.0)
NOISE_CAP = 0.25  # relative layer noise is the Geant4 fit residual, capped (the gamma form misfits the first layers)
CONTROL_EVENTS = 400
CONTROL_SPREAD = 0.8  # planted spreads are this fraction of the measured ones (the measured include fit noise)


def layer_bounds_x0() -> np.ndarray:
    """``(18, 2)`` lower and upper depth of every readout layer, in X0 from the entry face."""

    return np.array(load_geometry(GEOMETRY_CONFIG).uniform_layer_bounds_x0, dtype=float)


def profile_fractions(ln_t: float, ln_alpha: float, bounds: np.ndarray) -> np.ndarray:
    """Energy fraction of every layer for a gamma profile with maximum depth ``T`` and shape ``alpha``."""

    alpha, depth_of_maximum = np.exp(ln_alpha), np.exp(ln_t)
    rate = (alpha - 1.0) / depth_of_maximum
    return gammainc(alpha, rate * bounds[:, 1]) - gammainc(alpha, rate * bounds[:, 0])


def fit_layer_fractions(fractions: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """``(ln T, ln alpha)`` by least squares on the layer fractions."""

    return least_squares(
        lambda p: profile_fractions(p[0], p[1], bounds) - fractions, START, bounds=(LOWER, UPPER)
    ).x


def fit_cumulative_fractions(fractions: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """``(ln T, ln alpha)`` by least squares on the cumulative sums of the layer fractions."""

    target = np.cumsum(fractions)
    return least_squares(
        lambda p: np.cumsum(profile_fractions(p[0], p[1], bounds)) - target,
        START,
        bounds=(LOWER, UPPER),
    ).x


def grindhammer_peters_sampling(energy_mev: float, critical_energy_mev: float) -> dict[str, float]:
    """The sampling-calorimeter fluctuation values of hep-ex/0001020, Appendix A.2.2 (p. 14)."""

    ln_y = float(np.log(energy_mev / critical_energy_mev))
    return {
        "ln_y": ln_y,
        "sigma_ln_t": 1.0 / (-2.5 + 1.25 * ln_y),
        "sigma_ln_alpha": 1.0 / (-0.82 + 0.79 * ln_y),
        "rho": 0.784 - 0.023 * ln_y,
    }


LATERAL_OBSERVABLES = ("width_mm", "n_hit_cells", "core_fraction", "containment_fraction", "max_cell_fraction")


def lateral_coupling(
    parameters: np.ndarray, observables: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """Spearman correlation of each event's lateral observable with its fitted ln T and ln alpha.

    Grindhammer-Peters couple the radial generation to the longitudinal fluctuation (their Eq. 34);
    this measures how much lateral variation per event follows the longitudinal parameters, in
    Geant4 electrons, with no model assumption.
    """

    return {
        name: {
            "with_ln_t": float(stats.spearmanr(parameters[:, 0], observables[name]).statistic),
            "with_ln_alpha": float(stats.spearmanr(parameters[:, 1], observables[name]).statistic),
        }
        for name in LATERAL_OBSERVABLES
    }


def correlation(parameters: np.ndarray) -> float:
    return float(np.corrcoef(parameters[:, 0], parameters[:, 1])[0, 1])


def fit_events(
    fractions: np.ndarray, fit: Callable[[np.ndarray, np.ndarray], np.ndarray], bounds: np.ndarray
) -> np.ndarray:
    return np.array([fit(f, bounds) for f in fractions])


def planted_control(
    bounds: np.ndarray,
    mean: np.ndarray,
    spread: np.ndarray,
    layer_noise: np.ndarray,
    planted_rho: float,
    fit: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    n_events: int = CONTROL_EVENTS,
    seed: int = 2,
) -> float:
    """The correlation ``fit`` returns for events whose (ln T, ln alpha) have correlation ``planted_rho``."""

    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n_events)
    z2 = planted_rho * z1 + np.sqrt(1.0 - planted_rho**2) * rng.standard_normal(n_events)
    ln_t = mean[0] + spread[0] * z1
    ln_alpha = mean[1] + spread[1] * z2
    recovered = np.array(
        [
            fit(
                np.maximum(
                    profile_fractions(ln_t[i], ln_alpha[i], bounds)
                    * (1.0 + layer_noise * rng.standard_normal(len(bounds))),
                    1e-9,
                ),
                bounds,
            )
            for i in range(n_events)
        ]
    )
    return correlation(recovered)


def analyse_energy(energy_gev: float, *, control_events: int = CONTROL_EVENTS) -> dict[str, Any]:
    bounds = layer_bounds_x0()
    critical = load_electron_model("deposition").longitudinal.critical_energy_mev
    arrays = load_batch(DATA / f"E{energy_gev:g}GeV").arrays
    grids = arrays["deposit_grid_mev"].astype(float)
    fractions = grids.sum(axis=2) / (1000.0 * energy_gev)
    by_least_squares = fit_events(fractions, fit_layer_fractions, bounds)
    by_cumulative = fit_events(fractions, fit_cumulative_fractions, bounds)
    residual = np.array(
        [
            (f - profile_fractions(*p, bounds)) / np.maximum(profile_fractions(*p, bounds), 1e-9)
            for f, p in zip(fractions, by_least_squares, strict=True)
        ]
    )
    layer_noise = np.minimum(np.sqrt((residual**2).mean(axis=0)), NOISE_CAP)
    mean = by_least_squares.mean(axis=0)
    spread = CONTROL_SPREAD * by_least_squares.std(axis=0)
    control = {
        f"planted_rho_{rho:g}": {
            "least_squares": planted_control(
                bounds, mean, spread, layer_noise, rho, fit_layer_fractions, n_events=control_events
            ),
            "cumulative": planted_control(
                bounds, mean, spread, layer_noise, rho, fit_cumulative_fractions, n_events=control_events
            ),
        }
        for rho in (0.0, 0.6)
    }
    return {
        "n_events": len(fractions),
        "critical_energy_mev": critical,
        "grindhammer_peters_sampling": grindhammer_peters_sampling(1000.0 * energy_gev, critical),
        "least_squares": {
            "mean": mean.tolist(),
            "std": by_least_squares.std(axis=0).tolist(),
            "rho": correlation(by_least_squares),
        },
        "cumulative": {
            "mean": by_cumulative.mean(axis=0).tolist(),
            "std": by_cumulative.std(axis=0).tolist(),
            "rho": correlation(by_cumulative),
        },
        "control_recovered_rho": control,
        "lateral_coupling_spearman": lateral_coupling(
            by_least_squares,
            crossing_observables(
                grids, arrays["entry_x_mm"], arrays["entry_y_mm"], load_geometry(GEOMETRY_CONFIG)
            ),
        ),
        "contained_fraction_median": float(np.median(fractions.sum(axis=1))),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--control-events", type=int, default=CONTROL_EVENTS)
    args = parser.parse_args(argv)
    result: dict[str, Any] = {
        "status": (
            "EXPLORATION analysis of the unsealed Geant4 electrons. Estimator comparison with a "
            "planted-correlation control. Not a validation."
        ),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for energy in ENERGIES_GEV:
        result[f"{energy:g}"] = analyse_energy(energy, control_events=args.control_events)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
