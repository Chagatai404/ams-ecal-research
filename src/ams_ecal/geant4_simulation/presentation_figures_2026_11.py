"""Physics and simulation-quality figures for the group presentation of November 2026 (descriptive; no new claim).

Every figure states its sample, incidence, energy and simulation level. Simulation level everywhere: Geant4 11.4.1, FTFP_BERT,
scintillator energy deposited in the fibres ("visible energy"), no light attenuation, no photoelectrons (unless stated), no
saturation, no PMT/electronics effects, no material above the ECAL.

    PYTHONPATH=src python -m ams_ecal.geant4_simulation.presentation_figures_2026_11 [--theme ilimni]
writes results/presentation_2026_11/[ilimni/]*.png and numbers.json.

P1-P7 are physics and simulation-quality figures. P8 (pooled matched D_q, post hoc, exploratory) re-computes the EXP-015
pooled matching exactly as the original analysis did; P9 re-plots the EXP-019 intervals from the saved result file
(no refit). Labels use matplotlib mathtext so symbols render as mathematics.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from ams_ecal.benchmarks.powered_incremental_classifier import DELTA_MIN
from ams_ecal.geant4_simulation.geant4_backend import load_batch
from ams_ecal.multiscale_estimators import multiscale as ms
from ams_ecal.multiscale_estimators import (
    realistic_regime_analysis as rra,
)

OUT = Path("results/presentation_2026_11")
DATA = Path("data")
LAYERS, CELLS = 18, 72
X0_PER_LAYER = 16.684629744658373 / LAYERS  # our geometry's material table (proton calibration manifest)
PREFIX_LAMBDA_I = 0.6182108671949843  # same source
PREFIX_MM = 166.5
MIP_FIBRE_MEV = 0.5596
NAVY, AMBER, GREY = "#1b3a6b", "#b07000", "#6b7785"  # electrons, protons, references
ENERGY_RAMP = ("#9fb3d1", "#5f7fae", "#2f548c", NAVY)  # 10, 20, 50, 100 GeV: darker = higher energy
WINDOW_BAND = "0.92"
EVENT_CMAP = "magma"
EVENT_FACE = "white"  # empty cells in the event displays
FIXED_E = (10, 20, 50, 100)
EXP019_RESULT = Path("results/realistic_regime/exp019_powered_incremental_classifier/exp019_result.json")
numbers: dict = {}
plt.rcParams["mathtext.fontset"] = "dejavusans"  # mathematics in the same face as the labels

# Ilimni palette on white, for the slides of the group talk (ilimni-web/site/style.css: ink #101223, indigo #3B3BFF,
# ember #F2542D, muted #62677A, soft panel #EFF1FF). Only colours change; every computed number is identical to the
# default run (numbers.json is written to the ilimni/ subfolder for comparison).
ILIMNI = {
    "NAVY": "#3B3BFF",  # electrons
    "AMBER": "#E0451F",  # protons: ember, darkened slightly for thin lines on white
    "GREY": "#62677A",  # references
    "ENERGY_RAMP": ("#AEB0FF", "#7476FF", "#3B3BFF", "#1C1C8F"),  # darker = higher energy
    "WINDOW_BAND": "#EFF1FF",
    "rc": {
        "axes.edgecolor": "#62677A",
        "axes.labelcolor": "#101223",
        "axes.titlecolor": "#101223",
        "text.color": "#101223",
        "xtick.color": "#3E4357",
        "ytick.color": "#3E4357",
    },
}


def use_ilimni_theme() -> None:
    global NAVY, AMBER, GREY, ENERGY_RAMP, WINDOW_BAND, OUT
    NAVY, AMBER, GREY = ILIMNI["NAVY"], ILIMNI["AMBER"], ILIMNI["GREY"]
    ENERGY_RAMP, WINDOW_BAND = ILIMNI["ENERGY_RAMP"], ILIMNI["WINDOW_BAND"]
    OUT = OUT / "ilimni"
    plt.rcParams.update(ILIMNI["rc"])


def style(ax, title=None):
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, loc="left")


def x_view_rows() -> np.ndarray:
    rows = np.arange(LAYERS)
    return ((rows // 2) % 2) == 0  # superlayers alternate starting with x (geometry.py)


def fixed_electrons(energy: int) -> dict:
    return load_batch(DATA / "geant4_electron_sample" / "baseline" / f"E{energy}GeV").arrays


def fig_longitudinal():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    depth = (np.arange(LAYERS) + 0.5) * X0_PER_LAYER
    tmax = []
    for E, c in zip(FIXED_E, ENERGY_RAMP, strict=True):
        g = fixed_electrons(E)["readout_grid_mev"].astype(float)
        prof = g.sum(axis=2)
        frac = prof / prof.sum(axis=1, keepdims=True)
        mean = frac.mean(axis=0)
        a1.plot(depth, mean, "-o", ms=3, color=c, label=f"{E} GeV")
        i = int(np.argmax(mean))
        if 0 < i < LAYERS - 1:
            y0, y1, y2 = mean[i - 1 : i + 2]
            shift = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        else:
            shift = 0.0
        tmax.append((i + 0.5 + shift) * X0_PER_LAYER)
    a1.set_xlabel(r"depth $t$ in radiation lengths $X_0$ (our geometry: 16.68 $X_0$ in 18 layers)")
    a1.set_ylabel("mean fraction of visible energy per layer")
    a1.legend(frameon=False, fontsize=8)
    style(a1, "Electrons, normal incidence, centre of the face")
    lnE = np.log(np.array(FIXED_E, float))
    slope, icpt = np.polyfit(lnE, tmax, 1)
    a2.plot(lnE, tmax, "o", color=NAVY)
    xx = np.linspace(lnE.min() - 0.2, lnE.max() + 0.2, 10)
    a2.plot(xx, slope * xx + icpt, "-", color=NAVY, lw=1, label=rf"fit: slope {slope:.2f} $X_0$ per unit $\ln E$")
    a2.plot(xx, 1.0 * (xx - lnE[0]) + tmax[0], "--", color=GREY, lw=1, label=r"slope 1 (shower theory: $t_{\max} = \ln(E/E_c) + C$)")
    a2.set_xticks(lnE, [f"{E}" for E in FIXED_E])
    a2.set_xlabel(r"electron energy $E$ (GeV, log scale)")
    a2.set_ylabel(r"depth of shower maximum $t_{\max}$ ($X_0$)")
    a2.legend(frameon=False, fontsize=8)
    style(a2, r"Shower maximum $t_{\max}$ against $\ln E$")
    fig.tight_layout()
    fig.savefig(OUT / "P1_longitudinal_profile_and_shower_max.png", dpi=160)
    plt.close(fig)
    numbers["shower_max_x0"] = dict(zip([str(E) for E in FIXED_E], [round(t, 2) for t in tmax], strict=True))
    numbers["shower_max_slope_x0_per_ln_e"] = round(float(slope), 3)


def fig_resolution():
    rng = np.random.default_rng(20261112)
    rows = []
    for E in FIXED_E:
        g = fixed_electrons(E)["readout_grid_mev"].astype(float)
        vis = g.sum(axis=(1, 2)) / 1000.0
        mean_pe = 10.0 * g / MIP_FIBRE_MEV
        smeared = (rng.poisson(mean_pe) * (MIP_FIBRE_MEV / 10.0)).sum(axis=(1, 2)) / 1000.0

        def core(v):
            lo, hi = np.percentile(v, [5, 95])
            w = v[(v >= lo) & (v <= hi)]
            return float(w.mean()), float(np.std(v, ddof=1) / v.mean()), float(np.std(w, ddof=1) / w.mean())

        m, r_all, r_core = core(vis)
        _, r_all_pe, _ = core(smeared)
        rows.append((E, m, r_all, r_core, r_all_pe))
    E = np.array([r[0] for r in rows], float)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(E, 100 * np.array([r[2] for r in rows]), "o-", color=NAVY, label="our Geant4, visible energy (scintillator)")
    a1.plot(E, 100 * np.array([r[4] for r in rows]), "s--", color=AMBER, label="same + Poisson photoelectrons, 10 per MIP")
    ee = np.linspace(8, 120, 100)
    # Kounine 2017 Fig. 14: sigma = sqrt(10.5^2 / E + 1.4^2) in %, E in GeV: the two terms add in quadrature.
    a1.plot(ee, np.hypot(10.5 / np.sqrt(ee), 1.4), "-", color=GREY,
            label=r"AMS reconstructed energy: $10.5\%/\sqrt{E/\mathrm{GeV}} \oplus 1.4\%$ (Kounine 2017, Fig. 14)")
    a1.set_xscale("log")
    a1.set_xlabel(r"electron energy $E$ (GeV)")
    a1.set_ylabel(r"relative spread $\sigma_E / \langle E \rangle$ (%)")
    a1.legend(frameon=False, fontsize=7)
    style(a1, "Energy resolution (not like-for-like: AMS is leakage-corrected)")
    vf = np.array([r[1] for r in rows]) / E
    a2.plot(E, 100 * vf, "o-", color=NAVY)
    a2.set_xscale("log")
    a2.set_xlabel(r"electron energy $E$ (GeV)")
    a2.set_ylabel(r"visible fraction $E_{\mathrm{vis}} / E$ (%)")
    style(a2, "Visible fraction falls slowly with energy (rear leakage)")
    fig.tight_layout()
    fig.savefig(OUT / "P2_resolution_and_visible_fraction_fixed_energy.png", dpi=160)
    plt.close(fig)
    numbers["fixed_energy_electrons"] = [
        {"E_gev": r[0], "mean_visible_gev_core": round(r[1], 3), "rel_sigma_all": round(r[2], 4), "rel_sigma_core90": round(r[3], 4), "rel_sigma_all_pe10": round(r[4], 4)}
        for r in rows
    ]


def fig_proton_interaction():
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    out = {}
    zz = np.linspace(0, PREFIX_MM, 60)
    for E, c in zip(FIXED_E, ENERGY_RAMP, strict=True):
        a = load_batch(DATA / "geant4_proton_pilot" / "baseline" / f"E{E}GeV").arrays
        occ = np.asarray(a["truth_occurred"], bool)
        z = np.asarray(a["truth_z_mm"], float)
        surv = [(~occ).mean() + (occ & (z > t)).mean() for t in zz]
        ax.plot(zz, surv, "-", color=c, label=f"{E} GeV protons (n={len(occ)})")
        out[str(E)] = {"n": len(occ), "p_no_interaction": round(float((~occ).mean()), 4)}
    ax.plot(zz, np.exp(-zz / PREFIX_MM * PREFIX_LAMBDA_I), "--", color=GREY,
            label=rf"$\exp(-z/\lambda_I)$; our material: {PREFIX_LAMBDA_I:.3f} $\lambda_I$ in total")
    ax.set_xlabel(r"depth $z$ into the ECAL (mm)")
    ax.set_ylabel("fraction with no inelastic interaction yet")
    ax.set_ylim(0.4, 1.02)
    ax.legend(frameon=False, fontsize=7)
    style(ax, "Protons, normal incidence: first inelastic interaction")
    fig.tight_layout()
    fig.savefig(OUT / "P3_proton_first_interaction_depth.png", dpi=160)
    plt.close(fig)
    numbers["proton_no_interaction"] = out
    numbers["proton_no_interaction_material_expectation"] = round(float(np.exp(-PREFIX_LAMBDA_I)), 4)


def realistic(name: str):
    parts = []
    for kind in ("pilot", "powered"):
        ev = rra.load_class_events(DATA / f"geant4_realistic_regime_{kind}_{name}", name)
        sel = rra.fiducial_of(ev)
        if name == "proton":
            sel &= ev.truth_occurred.astype(bool)
        parts.append((ev, sel))
    return parts


def fig_visible_fraction_bands(electrons, protons):
    def gather(parts):
        r, band = [], []
        for ev, sel in parts:
            r.append(ev.edep_scintillator_mev[sel] / 1000.0 / ev.primary_energy_gev[sel])
            band.append(ev.band_lower_gev[sel])
        return np.concatenate(r), np.concatenate(band)

    re, be = gather(electrons)
    rp, bp = gather(protons)
    bands = [b for b in sorted(set(bp.tolist()) | set(be.tolist())) if b >= 200]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bins = np.logspace(-3.3, -1.1, 70)
    for k, b in enumerate(bands):
        off = k * 1.0
        for r, bb, col in ((re, be, NAVY), (rp, bp, AMBER)):
            v = r[bb == b]
            if len(v) < 20:
                continue
            h, edges = np.histogram(v, bins=bins)
            h = h / h.max() * 0.85
            ax.fill_between(np.sqrt(edges[:-1] * edges[1:]), off, off + h, step="mid", color=col, alpha=0.55, lw=0)
        upper = bands[k + 1] if k + 1 < len(bands) else 8000
        ax.text(10 ** -3.28, off + 0.35, f"{b / 1000:g}-{upper / 1000:g} TeV", fontsize=8)
    ax.set_xscale("log")
    ax.set_yticks([])
    ax.set_xlabel(r"visible fraction $E_{\mathrm{vis}} / E$ (scintillator energy over primary energy)")
    ax.plot([], [], color=NAVY, lw=6, alpha=0.55, label="electrons")
    ax.plot([], [], color=AMBER, lw=6, alpha=0.55, label="interacting protons")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style(ax, r"Realistic regime ($\cos\theta$-weighted to 25$^\circ$, full face, fiducial): the e/p visible-energy gap by band")
    fig.tight_layout()
    fig.savefig(OUT / "P4_visible_fraction_by_band.png", dpi=160)
    plt.close(fig)
    numbers["visible_fraction_median"] = {"electron": round(float(np.median(re)), 4), "proton_interacting": round(float(np.median(rp)), 4)}


def fig_event_displays(electrons, protons):
    ev_e, sel_e = electrons[1]
    ev_p, sel_p = protons[1]
    ie = np.flatnonzero(sel_e & (ev_e.band_lower_gev == 800))
    ip = np.flatnonzero(sel_p & (ev_p.band_lower_gev >= 2000))
    vis_e = ev_e.readout_grid_mev[ie].sum(axis=(1, 2))
    target = np.median(vis_e)
    e_row = ie[np.argmin(np.abs(vis_e - target))]
    vis_p = ev_p.readout_grid_mev[ip].sum(axis=(1, 2))
    p_row = ip[np.argmin(np.abs(vis_p - target))]
    xv = x_view_rows()
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for r, (ev, row, lab) in enumerate(((ev_e, e_row, "electron"), (ev_p, p_row, "interacting proton"))):
        g = ev.readout_grid_mev[row].astype(float)
        E = ev.primary_energy_gev[row]
        th = np.degrees(ev.primary_theta_rad[row])
        vmax = g.max()
        kw = {"cmap": EVENT_CMAP, "norm": matplotlib.colors.LogNorm(vmin=max(vmax * 1e-4, 1e-3), vmax=vmax), "aspect": "auto", "origin": "upper"}
        for a in axes[r]:
            a.set_facecolor(EVENT_FACE)
        axes[r, 0].imshow(np.where(g[xv] > 0, g[xv], np.nan), **kw)
        axes[r, 1].imshow(np.where(g[~xv] > 0, g[~xv], np.nan), **kw)
        im = axes[r, 2].imshow(np.where(g > 0, g, np.nan), **kw)
        axes[r, 0].set_title(rf"{lab}, {E:.0f} GeV, $\theta$ = {th:.0f}$^\circ$: x view (10 layers)", fontsize=8, loc="left")
        axes[r, 1].set_title(f"y view (8 layers); visible {g.sum() / 1000:.1f} GeV", fontsize=8, loc="left")
        axes[r, 2].set_title(r"readout image used by the 2D $D_q$ (views interleaved)", fontsize=8, loc="left")
        for a in axes[r]:
            a.set_xlabel("cell (9 mm)", fontsize=8)
            a.set_ylabel("layer (depth)", fontsize=8)
        fig.colorbar(im, ax=axes[r, 2], label="MeV per cell", fraction=0.04)
    fig.suptitle("Same visible energy, different particles (realistic regime, scintillator level). Each view sees its own coordinate.", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "P5_event_displays_views.png", dpi=160)
    plt.close(fig)
    numbers["event_display"] = {
        "electron": {"E_gev": float(ev_e.primary_energy_gev[e_row]), "visible_gev": float(ev_e.readout_grid_mev[e_row].sum() / 1000)},
        "proton": {"E_gev": float(ev_p.primary_energy_gev[p_row]), "visible_gev": float(ev_p.readout_grid_mev[p_row].sum() / 1000)},
    }


def fig_view_offset(electrons, protons):
    xv = x_view_rows()
    cols = np.arange(CELLS)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for parts, col, lab in ((electrons, NAVY, "electrons"), (protons, AMBER, "interacting protons")):
        d = []
        for ev, sel in parts:
            g = ev.readout_grid_mev[sel].astype(float)
            ex, ey = g[:, xv, :].sum(1), g[:, ~xv, :].sum(1)
            ok = (ex.sum(1) > 0) & (ey.sum(1) > 0)
            cx = (ex * cols).sum(1) / np.where(ex.sum(1) > 0, ex.sum(1), 1)
            cy = (ey * cols).sum(1) / np.where(ey.sum(1) > 0, ey.sum(1), 1)
            d.append(np.abs(cx - cy)[ok])
        d = np.concatenate(d)
        ax.hist(d, bins=np.arange(0, 66, 2), histtype="step", lw=1.5, color=col, density=True, label=f"{lab} (median {np.median(d):.0f} cells)")
        numbers[f"view_centroid_offset_median_cells_{lab.split()[-1]}"] = round(float(np.median(d)), 1)
    ax.set_xlabel("|x-view centre - y-view centre| in the interleaved image (cells)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
    style(ax, "Full-face entry separates the two views in the readout image")
    fig.tight_layout()
    fig.savefig(OUT / "P6_view_offset_in_readout_image.png", dpi=160)
    plt.close(fig)


def fig_box_counting(electrons):
    ev, sel = electrons[1]
    ie = np.flatnonzero(sel & (ev.band_lower_gev == 800))
    g = ev.readout_grid_mev[ie[len(ie) // 2]].astype(float)
    sides = [1, 2, 3, 6, 9, 18]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for q, c in ((0, GREY), (1, AMBER), (2, NAVY)):
        y = ms.log_measure(g, sides, q)
        label = {
            0: r"$q=0$: $\ln N(s)$, occupied boxes",
            1: r"$q=1$: entropy $S(s) = -\sum_i p_i \ln p_i$",
            2: r"$q=2$: $\ln \sum_i p_i^2$",
        }[q]
        ax.plot(np.log(sides), y, "o-", color=c, label=label)
    ax.axvspan(np.log(2) - 0.05, np.log(6) + 0.05, color=WINDOW_BAND, zorder=0, label="example fit window: sides 2 to 6 cells")
    ax.set_xticks(np.log(sides), [str(s) for s in sides])
    ax.set_xlabel(r"box side $s$ (cells of 9 mm; log scale)")
    ax.set_ylabel(r"$\ln Z_q(s)$, or $S(s)$ at $q=1$")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    style(ax, r"One electron: $D_q$ is the slope over a finite window of box sides")
    fig.tight_layout()
    fig.savefig(OUT / "P7_box_counting_one_event.png", dpi=160)
    plt.close(fig)


def fig_pooled_dimensions():
    """EXP-015 pooled matching (post hoc, exploratory): D_0, D_1, D_2 of matched electrons and interacting protons.

    Same calls and seed as the original pilot analysis: pilot events, all electrons, interacting protons, pooled caliper
    matching over every proton band (label "P-all"), window of box sides 1-2-3-6-9-18.
    """
    from scipy.stats import ks_2samp

    e = rra.load_class_events(DATA / "geant4_realistic_regime_pilot_electron" / "pilot", "electron")
    p = rra.load_class_events(DATA / "geant4_realistic_regime_pilot_proton" / "pilot", "proton")
    ei = rra.build_class_images(e, np.ones(len(e), bool), "electron", 1)
    pi = rra.build_class_images(p, p.truth_occurred.astype(bool), "proton_interacting", 2)
    pool = rra.analyse_pooled(ei, pi, 200, np.random.default_rng([rra.ANALYSIS_SEED, 7]), with_excess=False, per_band=False)
    pa = next(x for x in pool if x.label == "P-all")
    names = rra.dimension_names()
    de, dp = ei.dims(pa.electron_index), pi.dims(pa.proton_index)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    out = {"pairs": int(pa.result.pairs)}
    for k, q in enumerate((0, 1, 2)):
        j = names.index(f"D{q}@1-2-3-6-9-18")
        ok = np.isfinite(de[:, j]) & np.isfinite(dp[:, j])
        a, b = de[ok, j], dp[ok, j]
        ks = float(ks_2samp(a, b).statistic)
        out[f"D{q}"] = {"valid_pairs": int(ok.sum()), "ks": round(ks, 4)}
        bins = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 25)
        ax = axs[k]
        ax.hist(a, bins, histtype="step", color=NAVY, lw=2, label="matched electrons")
        ax.hist(b, bins, histtype="step", color=AMBER, lw=2, ls="--", label="matched interacting protons")
        ax.text(0.97, 0.95, f"KS = {ks:.3f}\nvalid pairs = {ok.sum()}", transform=ax.transAxes, ha="right", va="top", fontsize=9)
        ax.set_xlabel(rf"$D_{q}$, window of box sides 1-2-3-6-9-18 cells")
        ax.set_ylabel("events per bin" if k == 0 else "")
        style(ax, rf"$D_{q}$")
    fig.suptitle(f"Generalized dimensions of {out['pairs']} matched electron-proton pairs (pooled matching, post hoc, "
                 "exploratory; pilot, scintillator level)", fontsize=10)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "P8_pooled_matched_dimensions.png", dpi=160)
    plt.close(fig)
    numbers["pooled_matched_dimensions"] = out


def fig_intervals():
    """EXP-019 95% intervals (Nadeau-Bengio), re-plotted from the saved result; same rows as figure_intervals()."""
    result = json.loads(EXP019_RESULT.read_text(encoding="utf-8"))
    rows = []
    blocks = (("primary_fresh", "fresh events (primary test)"), ("secondary_pooled", "pilot + fresh pooled"),
              ("R1_pe10", "fresh, 10 p.e. per MIP"), ("R2_pe5", "fresh, 5 p.e. per MIP"), ("R3_saturation", r"fresh, saturation $e_s$ = 260 MeV"))
    subs = (("negative_control", ": shuffled multiscale columns (control)"), ("theta_le_12_deg", r": only $\theta \leq 12^\circ$"))
    for key, title in blocks:
        block = result.get(key, {})
        interval = block.get("primary", {}).get("interval")
        if interval:
            rows.append((title, interval))
        for sub, sub_title in subs:
            iv = block.get(sub, {}).get("interval") if isinstance(block.get(sub), dict) else None
            if iv:
                rows.append((title + sub_title, iv))
    fig, ax = plt.subplots(figsize=(10.5, 0.45 * len(rows) + 2.2))
    ax.axvspan(-DELTA_MIN, DELTA_MIN, color=WINDOW_BAND, zorder=0, label=rf"$|\Delta| < \delta$ = {DELTA_MIN}: smaller than the gain set in advance")
    ax.axvline(0, color=GREY, lw=0.8)
    for i, (_, iv) in enumerate(reversed(rows)):
        ax.errorbar(iv["mean"], i, xerr=[[iv["mean"] - iv["lower"]], [iv["upper"] - iv["mean"]]], fmt="o", color=NAVY, capsize=3)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.set_xticks(np.arange(-0.02, 0.0201, 0.01))
    ax.set_xlabel(r"$\Delta = \mathcal{L}_{C+M} - \mathcal{L}_{C}$, change in class-balanced log-loss" "\n"
                  "(negative = the multiscale features help)")
    fig.suptitle("EXP-019: 95% intervals (Nadeau-Bengio) for adding multiscale features M to conventional variables C",
                 fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "P9_exp019_intervals.png", dpi=160)
    plt.close(fig)
    numbers["exp019_intervals"] = {title: {k: round(v, 5) for k, v in iv.items() if k in ("mean", "lower", "upper")} for title, iv in rows}


def main():
    import sys

    if "--theme" in sys.argv and sys.argv[sys.argv.index("--theme") + 1] == "ilimni":
        use_ilimni_theme()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_longitudinal()
    fig_resolution()
    fig_proton_interaction()
    electrons, protons = realistic("electron"), realistic("proton")
    fig_visible_fraction_bands(electrons, protons)
    fig_event_displays(electrons, protons)
    fig_view_offset(electrons, protons)
    fig_box_counting(electrons)
    fig_pooled_dimensions()
    fig_intervals()
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1), encoding="utf-8")
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
