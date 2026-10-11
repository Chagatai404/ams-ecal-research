"""Lateral structure of Geant4 electron showers against the candidate radial laws (analysis only).

Question (the lateral slice, before any generator code): which radial law describes the mean energy profile on the
18 x 72 cell grid of the exposed Geant4 electrons, and what fluctuation structure does a spot model have to reproduce?

Literature structure (read in the library PDFs): Grindhammer and Peters, hep-ex/0001020, Eqs. 22-41 and Appendix A; the
AMS reconstruction, Kounine et al., NIM A 869 (2017) 110, Eqs. 2-5; the AMS side-leakage paper, Chin. Phys. C 40 (2016)
096204, Eq. 2. All three use the same two-component radial law

    f(r) = p 2 r R_C^2 / (r^2 + R_C^2)^2 + (1 - p) 2 r R_T^2 / (r^2 + R_T^2)^2 ,

whose strip marginal (the quantity a one-coordinate layer measures) has the closed form
F(x) = (1 + x / sqrt(x^2 + R^2)) / 2 for each component.

What is measured here, on exposed electrons only (the sealed sets are never read):

1. the mean cell-fraction profile per layer and energy against DEC-001's law, a one-component law and the two-component
   law (fitted, and with the Grindhammer-Peters homogeneous constants);
2. the fitted radial parameters against depth (units of the shower maximum) and energy;
3. the radial containment implied by the fitted law against the PDG statement (10% outside R_M, 99% inside 3.5 R_M);
4. the relative fluctuation sigma^2(eps) / (eps (1 - eps)) of the central three-cell fraction, raw and after removing its
   dependence on the event's model-free centre of gravity (the Grindhammer-Peters tau_i mechanism): the inverse is the
   effective spot number N_S per layer.

Results: ``results/em_generator/lateral_structure_study*.{json,csv}``.
Command: ``python -m ams_ecal.electron_studies.em_lateral_structure_study``.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from ams_ecal.detector.geometry import ECALGeometry, load_geometry
from ams_ecal.electron_model.fastmc_config import load_fastmc_config
from ams_ecal.electron_model.lateral import AMSLateralShowerModel

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "geant4_electron_sample" / "baseline"
RESULTS_DIR = ROOT / "results" / "em_generator"
GEOMETRY_CONFIG = ROOT / "configs" / "geometry.yaml"
FASTMC_CONFIG = ROOT / "configs" / "fastmc.yaml"

ENERGIES_GEV = (10.0, 20.0, 50.0, 100.0)
AXIS_CELL = 36  # the entry spot lies in [0, 9] mm, which is cell 36 of 72
CRITICAL_ENERGY_MEV = 7.6
COMPOSITE_X0_MM = 9.98
LEAD_Z = 82.0
MAXIMUM_OFFSET_X0 = 0.5  # deposition-regime shower maximum: ln(E / E_c) - 0.5 (Slice 1)
SUB_CELL_POSITIONS = 24  # the entry position is uniform inside the axis cell
FIT_STARTS = (
    (2.0, 15.0, 0.5),
    (3.0, 25.0, 0.7),
    (1.5, 8.0, 0.3),
    (5.0, 40.0, 0.8),
    (1.2, 6.0, 0.9),
)
STATUS = (
    "analysis only; exposed baseline Geant4 electrons; the sealed sets are not read; no generator changed; "
    "radii in mm; two-component fits use the closed-form strip marginal"
)


def strip_cdf(x: np.ndarray, radius: np.ndarray | float) -> np.ndarray:
    """CDF of the strip marginal of the radial density ``2 r R^2 / (r^2 + R^2)^2``."""

    return 0.5 * (1.0 + x / np.sqrt(x * x + radius * radius))


class StripGeometry:
    """Cell edges and the sub-cell axis positions that the strip fractions are averaged over."""

    def __init__(self, geometry: ECALGeometry) -> None:
        pitch, cells = geometry.cell_pitch_mm, geometry.cells_per_layer
        self.edges_mm = -cells * pitch / 2.0 + np.arange(cells + 1) * pitch
        axis = self.edges_mm[AXIS_CELL] + (np.arange(SUB_CELL_POSITIONS) + 0.5) / SUB_CELL_POSITIONS * pitch
        self.axis_mm = axis
        self.offsets = self.edges_mm[None, :] - axis[:, None]
        self.layer_thickness_mm = geometry.mean_readout_slice_thickness_mm
        self.moliere_mm = geometry.nominal_moliere_radius_mm
        self.cells = cells

    def fractions(self, core_mm: float, tail_mm: float, core_weight: float) -> np.ndarray:
        """Mean strip fractions per cell for a two-component law (not renormalised: the rest is leakage)."""

        cdf = core_weight * strip_cdf(self.offsets, core_mm) + (1.0 - core_weight) * strip_cdf(self.offsets, tail_mm)
        return np.diff(cdf, axis=1).mean(axis=0)


def fit_radial_law(strip: StripGeometry, observed: np.ndarray, single: bool = False) -> tuple[float, float, float]:
    """Ordered ``(R_C <= R_T, p)`` (or one radius) fitted to the cumulative profile with equal weights.

    The cumulative residual balances the core against the tail, which an error-weighted fit does not: with tiny
    statistical errors it fits only the central cells. The core radius is bounded at 0.5 mm: a 9 mm cell does not
    resolve a core below about 1 mm.
    """

    target = np.cumsum(observed)
    best = None
    for core0, tail0, weight0 in FIT_STARTS:
        if single:
            result = least_squares(
                lambda q: np.cumsum(strip.fractions(q[0], q[0], 1.0)) - target,
                [2.0 * core0],
                bounds=([0.3], [200.0]),
            )
        else:
            result = least_squares(
                lambda q: np.cumsum(strip.fractions(q[0], q[0] + q[1], q[2])) - target,
                [core0, tail0 - core0, weight0],
                bounds=([0.5, 0.5, 0.0], [40.0, 400.0, 1.0]),
            )
        if best is None or result.cost < best.cost:
            best = result
    q = best.x
    return (float(q[0]), float(q[0]), 1.0) if single else (float(q[0]), float(q[0] + q[1]), float(q[2]))


def total_variation(model: np.ndarray, observed: np.ndarray) -> float:
    """Half the L1 distance between two cell-fraction profiles, in fractions of the layer energy."""

    return float(0.5 * np.abs(observed - model).sum())


def radial_fraction_outside(radius_mm: float, core_mm: float, tail_mm: float, core_weight: float) -> float:
    """Energy fraction outside a circle of radius ``radius_mm`` for the two-component law."""

    return float(
        core_weight * core_mm**2 / (radius_mm**2 + core_mm**2)
        + (1.0 - core_weight) * tail_mm**2 / (radius_mm**2 + tail_mm**2)
    )


def grindhammer_peters_radial(energy_gev: float, depth_x0: float, moliere_mm: float) -> dict[str, float]:
    """Homogeneous-medium radial parameters (Appendix A.1.3) for lead, in mm.

    Assumptions that are NOT verified: ``ln E`` with E in GeV, depth in units of ``T_hom = ln y - 0.858``, lead Z = 82
    for the lead and fibre composite, and the composite Moliere radius as the unit.
    """

    y = 1000.0 * energy_gev / CRITICAL_ENERGY_MEV
    tau = depth_x0 / (np.log(y) - 0.858)
    ln_e = np.log(energy_gev)
    z1, z2 = 0.0251 + 0.00319 * ln_e, 0.1162 - 0.000381 * LEAD_Z
    k1, k2, k3, k4 = 0.659 - 0.00309 * LEAD_Z, 0.645, -2.59, 0.3585 + 0.0421 * ln_e
    p1, p2, p3 = (
        2.632 - 0.00094 * LEAD_Z,
        0.401 + 0.00187 * LEAD_Z,
        1.313 - 0.0686 * ln_e,
    )
    argument = (p2 - tau) / p3
    return {
        "tau_hom": float(tau),
        "core_mm": float((z1 + z2 * tau) * moliere_mm),
        "tail_mm": float(k1 * (np.exp(k3 * (tau - k2)) + np.exp(k4 * (tau - k2))) * moliere_mm),
        "core_weight": float(np.clip(p1 * np.exp(argument - np.exp(argument)), 0.0, 1.0)),
    }


def spot_number(fraction: np.ndarray, depth_coordinate: np.ndarray) -> dict[str, float]:
    """Inverse relative variance of a layer fraction, raw and after removing its dependence on event depth.

    For ``N`` equal spots drawn independently from the radial law the fraction in any radial interval has variance
    ``eps (1 - eps) / N`` (Grindhammer and Peters Eq. 29). The depth coordinate is the event's model-free energy centre
    of gravity; a quadratic is removed so that the shower-to-shower variation of the radial law does not count as spots.
    """

    mean = float(fraction.mean())
    scale = mean * (1.0 - mean)
    raw = float(fraction.var(ddof=1) / scale)
    coefficients = np.polyfit(depth_coordinate, fraction, 2)
    residual = fraction - np.polyval(coefficients, depth_coordinate)
    conditioned = float(residual.var(ddof=3) / scale)
    return {
        "relative_variance_raw": raw,
        "relative_variance_conditioned": conditioned,
        "spot_number_raw": 1.0 / raw,
        "spot_number_conditioned": 1.0 / conditioned,
        "explained_variance_fraction": float(1.0 - conditioned / raw),
    }


def load_deposit_grid(energy_gev: float, max_events: int | None = None) -> np.ndarray:
    arrays = np.load(DATA / f"E{energy_gev:g}GeV" / "events.npz")
    grid = arrays["deposit_grid_mev"].astype(float)
    return grid if max_events is None else grid[:max_events]


def energy_study(
    energy_gev: float,
    strip: StripGeometry,
    dec001: AMSLateralShowerModel,
    max_events: int | None = None,
) -> dict[str, Any]:
    grid = load_deposit_grid(energy_gev, max_events)
    layer_energy = grid.sum(axis=2)
    layers = grid.shape[1]
    centre_of_gravity = (layer_energy * (np.arange(layers) + 0.5)).sum(axis=1) / layer_energy.sum(axis=1)
    maximum_x0 = float(np.log(1000.0 * energy_gev / CRITICAL_ENERGY_MEV) - MAXIMUM_OFFSET_X0)
    rows = []
    for layer in range(layers):
        observed = grid[:, layer, :].sum(axis=0) / layer_energy[:, layer].sum()
        core, tail, weight = fit_radial_law(strip, observed)
        one = strip.fractions(*fit_radial_law(strip, observed, single=True))
        two = strip.fractions(core, tail, weight)
        depth_x0 = (layer + 0.5) * strip.layer_thickness_mm / COMPOSITE_X0_MM
        published = grindhammer_peters_radial(energy_gev, depth_x0, strip.moliere_mm)
        published_fractions = strip.fractions(published["core_mm"], published["tail_mm"], published["core_weight"])
        dec001_fractions = np.mean(
            [dec001.cell_energy_fractions(float(a), layer, 1000.0 * energy_gev) for a in strip.axis_mm[::4]],
            axis=0,
        )
        central = slice(AXIS_CELL - 1, AXIS_CELL + 2)
        usable = layer_energy[:, layer] > 0
        central_fraction = grid[usable, layer, central].sum(axis=1) / layer_energy[usable, layer]
        rows.append(
            {
                "layer": layer,
                "depth_x0": depth_x0,
                "tau": depth_x0 / maximum_x0,
                "core_mm": core,
                "tail_mm": tail,
                "core_weight": weight,
                "tv_points": {
                    "two_component_fit": 100 * total_variation(two, observed),
                    "one_component_fit": 100 * total_variation(one, observed),
                    "dec001": 100 * total_variation(dec001_fractions, observed),
                    "grindhammer_peters_published": 100 * total_variation(published_fractions, observed),
                },
                "central3_fraction": {
                    "geant4": float(observed[central].sum()),
                    "two_component_fit": float(two[central].sum()),
                    "dec001": float(dec001_fractions[central].sum()),
                },
                "grindhammer_peters_published": published,
                "outside_moliere_radius": radial_fraction_outside(strip.moliere_mm, core, tail, weight),
                "outside_3p5_moliere_radii": radial_fraction_outside(3.5 * strip.moliere_mm, core, tail, weight),
                "spots": spot_number(central_fraction, centre_of_gravity[usable]),
                "layer_energy_share": float(layer_energy[:, layer].sum() / layer_energy.sum()),
            }
        )
    share = np.array([r["layer_energy_share"] for r in rows])
    return {
        "events": len(grid),
        "shower_maximum_x0": maximum_x0,
        "centre_of_gravity_layers": {
            "mean": float(centre_of_gravity.mean()),
            "sd": float(centre_of_gravity.std()),
        },
        "layers": rows,
        "energy_weighted_outside_moliere_radius": float(
            sum(s * r["outside_moliere_radius"] for s, r in zip(share, rows, strict=True)) / share.sum()
        ),
        "energy_weighted_outside_3p5_moliere_radii": float(
            sum(s * r["outside_3p5_moliere_radii"] for s, r in zip(share, rows, strict=True)) / share.sum()
        ),
    }


def study(energies: tuple[float, ...] = ENERGIES_GEV, max_events: int | None = None) -> dict[str, Any]:
    geometry = load_geometry(GEOMETRY_CONFIG)
    strip = StripGeometry(geometry)
    dec001 = AMSLateralShowerModel(config=load_fastmc_config(FASTMC_CONFIG).lateral_em, geometry=geometry)
    per_energy = {f"{e:g}": energy_study(e, strip, dec001, max_events) for e in energies}
    return {
        "status": STATUS,
        "sealed_data_read": False,
        "composite_moliere_radius_mm": strip.moliere_mm,
        "per_energy": per_energy,
        "conventions": {
            "axis": "the entry point; the energy centroid of the prefix is within 0.2-0.65 mm (sd) of it",
            "observable": "deposit_grid_mev normalised by the layer total, mean over events (ratio of sums)",
            "sub_cell_positions": SUB_CELL_POSITIONS,
            "tau": f"layer-centre depth over ln(E / E_c) - {MAXIMUM_OFFSET_X0}",
        },
    }


def write_outputs(results: dict[str, Any], directory: Path = RESULTS_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "lateral_structure_study.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (directory / "lateral_structure_study_layers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "energy_gev",
                "layer",
                "depth_x0",
                "tau",
                "core_mm",
                "tail_mm",
                "core_weight",
                "tv_two",
                "tv_one",
                "tv_dec001",
                "tv_gp_published",
                "spot_number_raw",
                "spot_number_conditioned",
                "explained_variance",
            ]
        )
        for energy, block in results["per_energy"].items():
            for r in block["layers"]:
                t, s = r["tv_points"], r["spots"]
                writer.writerow(
                    [
                        energy,
                        r["layer"],
                        r["depth_x0"],
                        r["tau"],
                        r["core_mm"],
                        r["tail_mm"],
                        r["core_weight"],
                        t["two_component_fit"],
                        t["one_component_fit"],
                        t["dec001"],
                        t["grindhammer_peters_published"],
                        s["spot_number_raw"],
                        s["spot_number_conditioned"],
                        s["explained_variance_fraction"],
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args()
    results = study(max_events=args.max_events)
    write_outputs(results)
    for energy, block in results["per_energy"].items():
        mid = block["layers"][8]
        print(
            f"{energy} GeV: layer 8 tv two/one/dec001/gp = "
            + " / ".join(
                f"{mid['tv_points'][k]:.2f}"
                for k in (
                    "two_component_fit",
                    "one_component_fit",
                    "dec001",
                    "grindhammer_peters_published",
                )
            )
            + f"; outside R_M {block['energy_weighted_outside_moliere_radius']:.3f}, outside 3.5 R_M {block['energy_weighted_outside_3p5_moliere_radii']:.4f}"
        )


if __name__ == "__main__":
    main()
