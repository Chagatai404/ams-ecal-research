"""R-A proton diagnostic (2026-10-15): where does the class-table closure error sit? No new fit.

Models: the unsplit proton table and the crossing / interacting tables already fitted on proton_development_large (even
event_index). Test events: the tagged proton files data/geant4_component_proton (independent events, never used in a fit).
Closure per layer = predicted / Geant4 - 1 for the summed fibre energy of a group of events. Groups: crossing; interacting
in bins of first-interaction depth (lambda_I, equal-count terciles... six bins); interacting in quartiles of the
electromagnetic share of the deposit (component 'em' / total, truth). Reported: the max over layers, and layers 0-2 alone.
"""
import json, pathlib, re
import numpy as np
from ams_ecal.detector.sampling_response import SamplingResponse

ROOT = pathlib.Path("results/detector_response")
unsplit = SamplingResponse.load(ROOT / "r_a_proton_model.json")
crossing = SamplingResponse.load(ROOT / "r_a_proton_crossing_model.json")
interacting = SamplingResponse.load(ROOT / "r_a_proton_interacting_model.json")
rng = np.random.default_rng(20261015)


def closure(pred, act):
    return pred.sum(axis=(0, 2), dtype=np.float64) / act.sum(axis=(0, 2), dtype=np.float64) - 1.0


def summarise(name, pred, act, out):
    c = closure(pred, act)
    out[name] = {"n": int(len(pred)), "max_abs": float(np.abs(c).max()), "worst_layer": int(np.abs(c).argmax()),
                 "mean_layers_0_2": float(c[:3].mean()), "mean_layers_3_17": float(c[3:].mean())}


result = {}
for energy in (5, 10, 20, 50, 100, 200):
    with np.load(f"data/geant4_component_proton/train/E{energy}GeV/events.npz") as d:
        dep, fib = d["deposit_grid_mev"], d["readout_grid_mev"]
        occurred, depth = d["truth_occurred"], d["truth_depth_lambda_i"]
        comp = d["component_deposit_grid_mev"]
    em_share = comp[:, 0].sum(axis=(1, 2)) / np.maximum(comp.sum(axis=(1, 2, 3)), 1e-9)
    out = {}
    p_unsplit = unsplit.apply(dep, rng)
    p_split = np.where(occurred[:, None, None], interacting.apply(dep, rng), crossing.apply(dep, rng))
    summarise("all_unsplit", p_unsplit, fib, out)
    summarise("all_split", p_split, fib, out)
    summarise("crossing_split", p_split[~occurred], fib[~occurred], out)
    summarise("interacting_split", p_split[occurred], fib[occurred], out)
    idx = np.flatnonzero(occurred)
    edges = np.quantile(depth[idx], np.linspace(0, 1, 7))
    which = np.clip(np.digitize(depth[idx], edges[1:-1]), 0, 5)
    for b in range(6):
        sel = idx[which == b]
        summarise(f"interacting_depth_bin{b}[{edges[b]:.2f}-{edges[b+1]:.2f} lambda]", p_split[sel], fib[sel], out)
    qs = np.quantile(em_share[idx], [0, .25, .5, .75, 1])
    wq = np.clip(np.digitize(em_share[idx], qs[1:-1]), 0, 3)
    for b in range(4):
        sel = idx[wq == b]
        summarise(f"interacting_em_share_q{b}[{qs[b]:.2f}-{qs[b+1]:.2f}]", p_split[sel], fib[sel], out)
    result[str(energy)] = out
    print(energy, "GeV", {k: round(v["max_abs"], 3) for k, v in out.items() if k.startswith(("all", "cross", "interacting_split"))}, flush=True)
pathlib.Path("results/decision_checks/2026-10-15_ra_proton_diagnostic.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
