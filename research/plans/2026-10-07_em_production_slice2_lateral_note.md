# em_production Slice 2: the Grindhammer-Peters lateral spot model (candidate, development; not validated)

Date 2026-10-07, branch `fastmc-validated-dataset`. Decisions: DEC-014 (F2) and the 2026-10-07 rows of `DECISIONS.md`. Formulation: `2026-10-07_em_lateral_structure_note.md` section 5.
Code: `src/ams_ecal/electron_model/em_production_lateral.py` (the generator and its artifact), `em_production_lateral_calibration.py`;
`src/ams_ecal/electron_studies/em_production_lateral_check.py` (+ `_plots.py`); tests `tests/electron_model/test_em_production_lateral*.py`,
`tests/electron_studies/test_em_production_lateral_check.py`; artifact `data/calibration/em_production/gp_lateral_v1/parameters.json`;
results `results/em_generator/em_production_lateral_calibration.json`, `em_production_lateral_check.json` and three plots in `em_production_lateral_check/`.
Commands: `python -m ams_ecal.electron_model.em_production_lateral_calibration`, `python -m ams_ecal.electron_studies.em_production_lateral_check`.

**Status.** Implementation of the lateral slice, approved by the researcher on 2026-10-07 ("Use the geometry's 18 mm, go ahead with the formulation, Grindhammer-Peters functional structure should work").
A **calibrated candidate for development**, checked on exposed Geant4 electrons only: the sealed electron set was **not opened**, Slice 1, DEC-001 and the proton model are **unchanged**, nothing here is called validated.
The unit is the geometry's R_M = 18 mm (the researcher's choice); the hold-out is the extended-depth prefix (the default: question 3 of the earlier note was not answered).

## 1. What the generator is

Slice 1 draws the event's `(T_i, alpha_i)` and layer energies. Slice 2 adds, per layer: the radial law of Grindhammer and Peters (core + tail, weight `p`) with `R_C, R_T, p` as functions of the event's own depth coordinate `tau_i` (their Eq. 34, from `T_i, alpha_i`) and `ln E`; and `N_l` equal-energy spots per layer, a multinomial over the closed-form strip fractions of the 72 cells. Energy: the layer energy of Slice 1, never renormalised; spots beyond the 72 cells are lateral leakage (0.02% of a layer). One event is a batch of one (same draw order).
Refusals: outside 10-100 GeV (Slice 1's longitudinal check), a non-positive weight width, a Moliere radius different from the geometry's, a longitudinal artifact different from the one the lateral artifact was calibrated on, an altered artifact.

Calibrated constants (R_M units, E in GeV): core `-0.0026 - 0.0046 ln E + 0.1849 tau`; tail `0.696 {exp(-13.50 (tau - 0.130)) + exp(k4 (tau - 0.130))}`, `k4 = 0.479 - 0.0684 ln E`; weight `2.334 exp{(0.717 - tau)/p3 - exp(.)}`, `p3 = 1.476 - 0.173 ln E`; spots `N_spot = 180.6 E^0.911`, depth density a gamma of scale 0.906 and shape 0.900 times the reference. None at a bound. Provenance: the functional structure is transferred from the literature (Grindhammer and Peters Eqs. 24-26, 29-34, 38-40; AMS Eqs. 2-3), the values are calibrated.

## 2. Calibration (exposed baseline electrons, 1000 per energy)

- **Radial law**, 12 constants jointly over 4 energies x 18 layers, ensemble mean over a 256-point Sobol quadrature of Slice 1's draws, **energy-weighted** (ratio of sums, as the Geant4 profile; the unweighted mean of my first version was biased because the layer energy and the radial shape are coupled through tau_i, found by a test): mean cell-profile error, layers 3-17, mean / worst: 1.3 / 3.1, 1.3 / 2.4, 1.7 / 3.0, 2.3 / 4.2 points of layer energy at 10 / 20 / 50 / 100 GeV; layers 0-2 worst 2.9, 5.1, 7.5, 11.0.
- **Spot structure**, 4 constants, to the conditioned relative variance of the central three cells with the same estimator on generated events: rms log ratio **0.19 on layers 2-17**, 0.59 over all layers. **Layers 0-1 are not fitted**: the measured conditioned spot number there (about 13 and 24) is the same at all four energies, which a gamma depth density scaled by `E^b` cannot reach (the model has 1-2 spots in layer 0). This is a limitation, with an extension proposed in section 5.

## 3. Development check (calibration set, hold-out = extended prefix, control = Slice 1 with DEC-001's lateral law, floor = Geant4 baseline against extended prefix)

Hold-out, 10 / 20 / 50 / 100 GeV; Slice 2 against control against floor:

| metric | Slice 2 | control | floor |
|---|---|---|---|
| mean cell-profile error, layers 3-17, mean (points) | 1.4 / 1.2 / 1.8 / 2.2 | 3.0 / 2.3 / 2.5 / 2.7 | 0.9 / 0.7 / 0.4 / 0.6 |
| same, worst layer | 3.3 / 2.8 / 3.4 / 4.0 | 4.3 / 3.1 / 3.9 / 4.8 | 1.3 / 1.4 / 0.7 / 1.2 |
| layers 0-2 worst | 3.8 / 4.9 / 7.9 / 11.2 | 18 / 25 / 34 / 43 | 1.4 / 1.2 / 1.0 / 1.6 |
| core-fraction KS | 0.17 / 0.10 / 0.19 / 0.20 | 0.60 / 0.60 / 0.59 / 0.64 | 0.04-0.10 |
| containment KS | 0.19 / 0.16 / 0.40 / 0.51 | 0.67 / 0.67 / 0.70 / 0.79 | 0.07-0.13 |
| largest-cell-fraction KS | 0.34 / 0.25 / 0.16 / 0.09 | 0.29 / 0.26 / 0.23 / 0.22 | 0.03-0.06 |
| **width KS** | **0.68 / 0.68 / 0.70 / 0.71** | 0.46 / 0.59 / 0.71 / 0.69 | 0.08-0.16 |
| **hit-cells KS** | **1.0 / 1.0 / 1.0 / 1.0** | 0.88 / 0.98 / 1.0 / 1.0 | 0.06-0.13 |
| Spearman(width, energy centre of gravity), Geant4 | -0.23 / -0.41 / -0.56 / -0.62 | | |
| same, Slice 2 | -0.34 / -0.35 / -0.43 / -0.51 | | |
| same, control | +0.98 / +0.98 / +0.99 / +0.99 | | |
| spot number rms log ratio, layers 2-17 | 0.17 / 0.10 / 0.16 / 0.34 | | floor 0.08-0.17 (all layers) |

The calibration set gives the same picture (the hold-out agrees with it: no sign of overfitting).

**What improved.** The core fraction, the containment and the profile are far closer than the control; above all **the lateral-longitudinal coupling has the right sign and size** (-0.34 to -0.51 against Geant4 -0.23 to -0.62), where DEC-001's deterministic law gives +0.98 (the opposite sign): the coupling follows from the tau_i mechanism and was not hard-coded.
**What did not.** The width distribution (KS 0.7, not better than the control) and the hit-cell count (KS 1.0): the model has 182 hit cells against 334 at 20 GeV and 278 against 483 at 100 GeV, and a median width 16.1 against 13.9 mm at 20 GeV with a broader spread (13.5-18.6 against 12.9-15.0).

**Proposed signs (mine, reported not gated): mostly not met.** Profile error at most 2 points in layers 3-17: the worst layer is 2.8-4.0 (the mean is 1.2-2.2, so about half of the layers pass); coupling within twice the floor or 0.05: met at 10 and 20 GeV on the calibration set and at 100 GeV on the hold-out only (differences -0.11 to +0.19 elsewhere, the floor 0.02-0.08); spot number within a factor 1.28 (rms log 0.25): met on layers 2-17 at 10-50 GeV, not over all layers (0.44-0.75, layers 0-1) and not at 100 GeV (0.34 on layers 2-17).

## 4. Diagnosis (measured, not yet acted on)

Equal-energy spots calibrated to the central-cell variance leave the outskirts with too few, too coarse spots: real showers deposit the wide tail as many small low-energy deposits. Too few outer spots means too few occupied cells (hit cells) and, because a single far coarse spot moves the rms strongly, a wider and broader width distribution. The AMS reconstruction's own Poisson law (Kounine et al. Eq. 5) lets the quantum size depend on the distance R from the shower axis, which supports a radius-dependent spot energy; the equal-energy spot of Grindhammer and Peters does not have that freedom.

## 5. Named limitations and a proposed extension (needs the researcher's approval; nothing done)

1. Layers 0-1: an energy-independent spot number (about 13, 24, 62 in layers 0-2) that the approved structure cannot reach; layers 0-2 profile error 3-11 points.
2. Layer-to-layer residual noise is not modelled (plan stage 4).
3. Radial law: layers 3-17 worst-layer error 2.4-4.2 points against the proposed 2: the Grindhammer-Peters `ln E` dependence sits only in `z1`, `k4`, `p3`; the error grows with energy.
4. Hit cells and width (sections 3-4).
5. One coordinate per layer; the 2D structure of the fibre direction is not modelled; R_M is the geometry's 18 mm (the Rossi value 27.6 mm was not used).
6. Calibrated range 10-100 GeV; the hold-out is not a fresh validation; the sealed electron set stays unopened.

**Proposed extension (E1), for approval:** a spot energy that depends on the radius, `E_S(r) = E_layer / N_l x w(r)` with a normalised weight (so the energy is conserved), calibrated to the cell occupancy and the central-cell variance together (two parameters), and a front-limited spot count `1/N = 1/N_gamma + 1/N_front(t)` (two parameters) for layers 0-2. Each has a measured need (sections 2 and 3) and the first has AMS-specific support; both go beyond the formulation approved on 2026-10-07 and are therefore not implemented.

## 6. Extension E1 was tried under the researcher's condition and NOT adopted (2026-10-08)

The researcher's reply to section 5: "Do it only if it doesn't cause an overfitting and it doesn't complicate the parameterization". The condition was made operational **in code before any E1 number existed**
(`src/ams_ecal/electron_studies/em_production_lateral_e1_decision.py`; thresholds in its docstring and in the result file) and applied to the hold-out (the extended-depth prefix): the extension had to improve
BOTH the width and the hit-cell count at every energy by at least 30% on average (A); the improvement had to transfer from the calibration set to the hold-out (at least 70%) with the spot-number error gap
at most 0.05 and no collapse to a switched-off value (B: no overfitting); and it had to add at most two parameters while nothing that Slice 2 did well got worse (C: parsimony, no regression).

**What was built (opt-in, off by default, so the Slice 2 artifact and behaviour are untouched).** The parameterization was kept to **two** added numbers instead of the four proposed: `kappa` (the tail
component's spots carry 1 / kappa of a core spot's energy, one parameter in place of a free radius law) and `n0` (a cap n0 x 2^layer on the spot number of layer l, the cascade doubling per layer fixed, one parameter
in place of two). Calibration: the radial law fixed, the six spot constants fitted to the conditioned fluctuation of the central three cells AND of the outskirts, every layer; the width and the hit-cell
count were not targets. Result: `kappa = 2.66`, `n0 = 48.7`, converged, not at a bound; the objective on the two statistics fell from 0.51 (Slice 2) to 0.24.
Code: `em_production_lateral.py` (the two optional parameters), `em_production_lateral_calibration.py --extension`; artifact `data/calibration/em_production/gp_lateral_e1_v1/parameters.json`; results `em_production_lateral_e1_calibration.json`, `em_production_lateral_e1_check.json`, `em_production_lateral_e1_decision.json`.

**Verdict: E1 NOT ADOPTED; Slice 2 stays the candidate.** Hold-out, 10 / 20 / 50 / 100 GeV, Slice 2 against E1 (floor):

| metric | Slice 2 | E1 | floor |
|---|---|---|---|
| hit-cells KS | 1.0 / 1.0 / 1.0 / 1.0 | 0.57 / 0.62 / 0.60 / 0.48 | 0.06-0.13 |
| width KS | 0.68 / 0.68 / 0.70 / 0.71 | 0.81 / 0.81 / 0.81 / 0.79 | 0.08-0.16 |
| core fraction, containment, largest cell KS | unchanged within 0.02 | | |
| profile error layers 3-17 mean | 1.4 / 1.2 / 1.8 / 2.2 | 1.4 / 1.2 / 1.8 / 2.2 | |
| Spearman(width, centre of gravity); Geant4 -0.23 / -0.41 / -0.56 / -0.62 | -0.34 / -0.35 / -0.43 / -0.51 | -0.53 / -0.59 / -0.67 / -0.73 | |
| spot-number rms log ratio, all layers | 0.44 / 0.45 / 0.63 / 0.75 | 1.19 / 1.15 / 1.08 / 1.04 | |

- **No overfitting** (B1 for the hit cells: the hold-out improvement 0.43 against 0.49 on the calibration set; B2 gap 0.010; B3 not collapsed) **and no complication** (C1: two added parameters).
- **But:** the hit-cell count improves materially (KS 1.0 to 0.5-0.6, still 4-8 times the floor) and the **width gets worse** (KS 0.68-0.71 to 0.79-0.81), so A fails; the lateral-longitudinal coupling overshoots Geant4 at 10-50 GeV
  (-0.53 to -0.67 against -0.23 to -0.56), so C2 fails (its change +0.07 against 0.05 allowed); B1 fails for the width (the change is the same, negative, on both sets: a systematic effect, not overfitting).
- **The front-limited count was mis-specified.** It is a cap, and a cap can only lower the spot number, while layer 0 needs more spots (the model has 3 against 13-17 measured, unchanged by E1); it acted at layers 2-5 only. A floor, not a cap, is what the data ask for. The all-layer spot-number error rose because of layers 0-1 and because the energy exponent drifted to `b = 1.05`.
- **Interpretation (not tested).** A finer tail population cures the cell occupancy but does not cure the width, so the width excess is probably not a spot-granularity effect alone. The width is the rms of the lateral distribution and is dominated by the far tail of the radial law; the fitted law leaves 0.15 of the energy outside 18 mm and 0.02 outside 3.5 R_M against the PDG's 0.10 and 0.01 (indicative only, one coordinate), so the far tail of the Cauchy-like radial law is the next suspect.

## 7. Width diagnostic (2026-10-08): the excess is the far tail of the mean radial law

Requested by the researcher after section 6 ("Yes, run the width diagnostic, then commit"). Code `src/ams_ecal/electron_studies/em_lateral_width_diagnostic.py`, test `tests/electron_studies/test_em_lateral_width_diagnostic.py`,
results `results/em_generator/em_production_lateral_width_diagnostic.json` and `.png`; command `python -m ams_ecal.electron_studies.em_lateral_width_diagnostic`. Exposed electrons only, the sealed sets unread, no model changed.
The width is `sqrt(sum E_cell d^2 / E)` over all 72 cells, so it is a far-tail statistic. Three hypotheses were separated by comparing the Geant4 hold-out with Slice 2 WITH its spots and with its EXPECTATION (infinitely many spots, `E_layer x strip fraction`
of the same events): H1 the far tail of the mean radial law, H2 spot granularity, H3 the depth coupling.

| (hold-out) 10 / 20 / 50 / 100 GeV | Geant4 | Slice 2, spots | Slice 2, expectation |
|---|---|---|---|
| median width (mm) | 14.2 / 14.0 / 13.8 / 13.7 | 16.9 / 16.1 / 15.3 / 14.9 | 17.2 / 16.3 / 15.4 / 15.0 |
| mean width^2 (mm^2) | 204 / 198 / 191 / 187 | 289 / 262 / 235 / 222 | 292 / 262 / 235 / 222 |
| width^2 beyond 108 mm (bins 108-216 and 216-700), mm^2 | 33.5 / 32.6 / 31.7 / 31.7 | 115.6 / 102.3 / 88.2 / 81.5 | 117.7 / 101.8 / 88.0 / 81.5 |
| width^2 at 27-108 mm | 119 / 115 / 110 / 107 | 122 / 109 / 97 / 92 | 123 / 110 / 97 / 92 |
| energy beyond 108 mm | 0.0017 / 0.0016 / 0.0016 / 0.0016 | | 0.0041 / 0.0035 / 0.0030 / 0.0028 |
| energy beyond 216 mm | 0.0001 | | 0.0006 / 0.0006 / 0.0005 / 0.0004 |

- **H1 holds; H2 does not.** The expectation already carries the whole excess (its median and its mean width^2 equal those with spots: the spots only broaden the distribution, 5-95% range 13.5-20.4 against 13.0-15.7 mm at 10 GeV); the excess sits entirely beyond 108 mm (about 85 mm^2 of the 85-90 mm^2 total at 10 GeV), while the bins inside 108 mm are equal (0-27 mm) or slightly LOW (27-108 mm: -0.8 to -7.9 mm^2 per bin at 20-100 GeV, -1.6 and +4.7 at 10 GeV).
  The model has 1.8-2.4 times Geant4's energy beyond 108 mm and 4-6 times beyond 216 mm. It is concentrated in layers 8-13 (width^2 20-26 mm^2 against 16 in Geant4 at 10 GeV).
- **Why the calibration did not see it.** The radial law was fitted to the equal-weight CUMULATIVE cell profile, to which an energy fraction of 0.1% beyond 108 mm contributes nothing visible, while the width weights it by d^2 (about 10^4-10^5 mm^2). The Cauchy-like radial density (about 1/d^3) runs on to the edge of the calorimeter; the real shower falls off faster.
  This also explains why E1 (finer spots, same mean profile) could not help the width.
- **H3 (the depth coupling) is not the cause of the width excess**, but it is not excluded as a cause of the width's spread; not tested separately.
- **What this suggests (not done, needs approval).** The radial fit needs a target that sees the tail, for example the second moment of the mean profile (the mean width^2 per layer and energy) added to the cumulative residual, with the same twelve constants. That adds no parameter; whether the Grindhammer-Peters structure can satisfy both is the open question. A truncation of the tail would add a parameter and is NOT proposed first.

## 8. Tail-sensitive radial calibration (2026-10-08): the width is curable without a new parameter, but not without breaking the core

The researcher: "Yes, run the tail-sensitive radial calibration as well, again if it doesn't cause overfitting and doesn't complicate parameterization". The rule was written in code before the first run
(`src/ams_ecal/electron_studies/em_production_lateral_tail_decision.py`): the width KS must improve at every energy by 30% on average on the hold-out (A); the improvement must transfer from the calibration set, the
profile-error gap must stay within 0.2 points and no constant may end at a bound (B: no overfitting); no parameter may be added and nothing that worked may get materially worse (C).
**What was changed:** the SAME twelve radial constants (no parameter added); the radial residual gets a second target, the relative error of each (energy, layer) second moment of the mean profile (`mean_second_moments_mm2`, the quantity the
width is built from; `observed_second_moments` from the exposed electrons), added to the cumulative residual with a fixed weight; the spot constants are then refitted as in Slice 2. Command `python -m ams_ecal.electron_model.em_production_lateral_calibration --tail-sensitive [--moment-weight W --tag NAME]`.

**Two variants were run, and this is stated openly: the first weight was a poor choice.**
1. **Weight 0.02** (fixed before running): the moment term has 18 residuals per energy against 1296 cumulative ones, so it carried about a tenth of the squared weight and the fit barely moved (the constants changed by 1-3%); the layer second moments stayed 14-37% too large on average (worst 35-52%).
   Hold-out width KS improved by 0.03 (4% on average), the containment KS drifted by 0.032 (above the 0.03 allowance). Not adopted (A fails; C2 fails narrowly). Artifact `gp_lateral_tail_v1`, results `em_production_lateral_tail_*`.
2. **Weight 0.4** (the "balanced" weight, defined BEFORE its run from the principle that a 5% moment error should count like the whole fitted cumulative residual of the layer, `0.05 x w = sqrt(72) x 0.0025`): the layer second moments are matched
   (layers 3-17 mean error -0.0% to +3%, worst 4-10%). Artifact `gp_lateral_tail_balanced_v1`, results `em_production_lateral_tail_balanced_*`.

| hold-out, 10 / 20 / 50 / 100 GeV | Slice 2 | balanced (0.4) | floor |
|---|---|---|---|
| **width KS** | 0.68 / 0.68 / 0.70 / 0.71 | **0.23 / 0.19 / 0.20 / 0.20** | 0.08-0.16 |
| core-fraction KS | 0.17 / 0.10 / 0.19 / 0.20 | **0.56 / 0.72 / 0.77 / 0.77** | 0.04-0.10 |
| containment KS | 0.19 / 0.16 / 0.40 / 0.51 | **0.72 / 0.87 / 0.93 / 0.97** | 0.07-0.13 |
| largest-cell-fraction KS | 0.34 / 0.25 / 0.16 / 0.09 | 0.33 / 0.27 / 0.24 / 0.22 | 0.03-0.06 |
| hit-cells KS | 1.0 | 1.0 | 0.06-0.13 |
| profile error layers 3-17, mean (points) | 1.4 / 1.2 / 1.8 / 2.2 | 3.8 / 3.6 / 3.3 / 4.0 | 0.4-0.9 |
| profile error layers 3-17, worst | 3.3 / 2.8 / 3.4 / 4.0 | 7.9 / 6.6 / 5.7 / 5.6 | |
| Spearman(width, centre of gravity); Geant4 -0.23 / -0.41 / -0.56 / -0.62 | -0.34 / -0.35 / -0.43 / -0.51 | -0.31 / -0.33 / -0.43 / -0.54 | |

**Verdict: NOT ADOPTED (both variants); Slice 2 stays the candidate.** For the balanced variant: A holds (width KS reduced by 70% on average, at every energy), B holds (the hold-out improvement 0.49 against 0.51 on the calibration set, the profile-error gap 0.08, no constant at a bound),
C1 holds (no parameter added), **C2 fails**: the profile error rises by 2.0 points, the core-fraction KS by 0.54 and the containment KS by 0.56.
- **No overfitting and no added complexity; the cost is structural.** The width is a far-tail statistic and the width excess is curable with the same twelve constants, but only by trading the core: the Grindhammer-Peters radial law with a Cauchy-like tail couples the
  tail to the body, so lowering the far tail lowers the energy at 5-50 mm and the central mass moves. The twelve-constant structure cannot hold the near and the far profile together.
- **Two tries, one forking path.** A weight between 0.02 and 0.4 would trade less but was not run: with two variants already tried, a third would be tuning the weight to the verdict. The trade-off curve itself is the result.
- **Hit cells are unchanged by the radial law** (KS 1.0 in both), as the E1 trial found for the spots from the other side: the cell occupancy is neither a pure radial-law nor a pure spot-granularity effect.

## 9. A faster-falling tail (2026-10-08): one added parameter; it passes every criterion but one, by 0.11 points

The researcher approved it ("try it") after section 8 asked whether to add a parameter. The rule is the same as in section 8, fixed in code before the run, with ONE approved change: C1 allows exactly one added parameter
(`em_production_lateral_tail_decision.py --max-added-parameters 1`).
**What was added:** `tail_exponent` `s`: the tail component's survival function is `(R_T^2 / (r^2 + R_T^2))^s`; `s = 1` is the Grindhammer-Peters Cauchy-like tail (Slice 2), `s > 1` falls faster. The strip marginal stays closed form
(a scaled Student-t, CDF `1/2 + sign(x) I_z(1/2, s) / 2`, `z = x^2/(x^2 + R^2)`, exactly the old formula at `s = 1`, checked against numerical integration to 3e-12). Opt-in, default 1: Slice 2 is unchanged.
Calibration: thirteen radial constants (the twelve plus `s`, bounded to 1-4, started at 1.5), cumulative profile plus the second-moment target at the weight 0.4 fixed in advance (section 8: no new weight fork), then the spot refit.
Command `python -m ams_ecal.electron_model.em_production_lateral_calibration --tail-sensitive --faster-tail --moment-weight 0.4 --tag faster_tail`; artifact `gp_lateral_faster_tail_v1`; results `em_production_lateral_faster_tail_*`.
Result: `s = 1.81`, the tail scale 1.58 R_M, nothing at a bound; the layer second moments are matched (layers 3-17 mean -1% to +2%, worst 6-9%).

| hold-out, 10 / 20 / 50 / 100 GeV | Slice 2 | faster tail (s = 1.81) | floor |
|---|---|---|---|
| **width KS** | 0.68 / 0.68 / 0.70 / 0.71 | **0.19 / 0.15 / 0.19 / 0.20** | 0.08-0.16 |
| core-fraction KS | 0.17 / 0.10 / 0.19 / 0.20 | 0.06 / 0.11 / 0.11 / 0.28 | 0.04-0.10 |
| containment KS | 0.19 / 0.16 / 0.40 / 0.51 | **0.13 / 0.10 / 0.12 / 0.27** | 0.07-0.13 |
| largest-cell-fraction KS | 0.34 / 0.25 / 0.16 / 0.09 | 0.31 / 0.25 / 0.20 / 0.16 | 0.03-0.06 |
| hit-cells KS | 1.0 | 1.0 | 0.06-0.13 |
| profile error layers 3-17, mean (points) | 1.4 / 1.2 / 1.8 / 2.2 | **2.5 / 1.8 / 1.7 / 2.3** | 0.4-0.9 |
| profile error layers 3-17, worst | 3.3 / 2.8 / 3.4 / 4.0 | 5.2 / 3.6 / 4.3 / 6.4 | |
| profile error layers 0-2, worst | 3.8 / 4.9 / 7.9 / 11.2 | 4.2 / 6.6 / 9.3 / 10.3 | |
| Spearman(width, centre of gravity); Geant4 -0.23 / -0.41 / -0.56 / -0.62 | -0.34 / -0.35 / -0.43 / -0.51 | -0.35 / -0.40 / -0.50 / -0.62 | |

**Verdict by the pre-stated rule: NOT ADOPTED, because one sub-criterion fails.** A holds (width KS reduced by 74% on average, at every energy); B holds (the improvement transfers: 0.51 on the hold-out against 0.56 on the calibration set; the profile-error gap 0.036; nothing at a bound: no overfitting);
C1 holds (one added parameter); C2 fails on one item: **the hold-out profile error (layers 3-17 mean) rises by 0.41 points against the 0.3 allowed**. Every other C2 item holds: the core-fraction KS changes by -0.026, the containment KS by -0.16, the largest-cell KS by +0.019, the coupling difference by -0.056, the spot number by -0.001.
- **It is the closest candidate so far**, and the only one that cures the width without breaking the core: the Cauchy-like tail forced a trade (section 8); a faster tail does not (containment and coupling improve).
- **Where the profile error comes from:** the rise is at 10 GeV (1.4 to 2.5, worst layer 3.3 to 5.2) and at 100 GeV (worst 4.0 to 6.4); at 20 and 50 GeV the mean profile error is 1.8 / 1.7 against 1.2 / 1.8. The thirteen-constant fit with the moment target costs profile accuracy in the layers near the shower maximum at the extreme energies.
- **The threshold was not relaxed after the fact.** Whether 0.41 against 0.3 is acceptable given the gains is the researcher's decision (adopting it would be an explicit override of the rule written before the run); it is not made here. Hit cells are unchanged (KS 1.0): a radial-law change does not move them either.

## 10. Next steps (not started)

The researcher's decision on the faster-tail candidate (override the 0.3-point threshold, or refine within the rule, for example by letting the exponent depend on energy, a second added parameter that needs approval), a combined radial-and-spot study of the hit-cell count, layer-to-layer residual noise (stage 4),
the pre-stated gamma-inadequacy thresholds (still pending), then the single opening of the sealed electron set. Each needs approval; E1 stays opt-in, off; the tail-sensitive and faster-tail artifacts are recorded results, not candidates.

## 11. The 0.3-point threshold: how it was chosen, and what the metric resolves (2026-10-08)

The researcher asked whether 0.41 against 0.3 is a meaningful difference and how 0.3 was chosen.
- **How it was chosen (stated plainly): by me, by analogy, not derived.** The project's registered materiality conventions are KS 0.10 (proton validation) and 0.05 (dataset), classifier AUC 0.60 above a permutation null, lag 0.10, occupancy 0.02 and a 0.5-point floor for the mean LEAKAGE
  (`em_longitudinal_structure_analysis.NOISE_FLOORS`). There is none for the cell-profile error. I set the profile tolerance a little below the leakage floor. The other tolerances of the rules (30% benefit, 70% transfer, 0.2-point gap, KS 0.03, coupling 0.05, spot number 0.1) were chosen the same way.
- **What the metric resolves** (`src/ams_ecal/electron_studies/em_lateral_metric_resolution.py`, paired bootstrap over the Geant4 hold-out events, both models scored against the same resampled reference): the faster tail minus Slice 2, layers 3-17 mean profile error, at 10 / 20 / 50 / 100 GeV:
  **+1.19 [+0.91, +1.25], +0.50 [+0.24, +0.63], -0.14 [-0.19, -0.02], +0.06 [-0.10, +0.20] points**; mean **+0.40 +- 0.04**. The Geant4 half-against-half floor of the metric is 0.8-1.2 points, the model seed spread 0.02-0.08.
  So a difference of 0.1 point IS statistically resolved by the paired design (the reference noise, a positive bias, cancels), and the +0.41 is real: it is not noise.
- **But "resolved" is not "meaningful".** The aggregate hides the structure: the regression is concentrated at 10 and 20 GeV (+1.2, +0.5), the faster tail is better at 50 GeV and neutral at 100 GeV. A single mean threshold judged the candidate by its worst energies. Both 0.3 and 0.41 are below the Geant4 sample-to-sample floor of this very metric (0.4-0.9), and the metric
  does not map onto a physical consequence. A defensible threshold would be per energy and tied to a consequence (for example the classifier or the observables that matter), not to a mean. **The decision on the faster tail therefore rests on the researcher, not on the 0.3 line; the rule is kept as recorded (not adopted), and nothing is adopted by a rule changed after the result.**

## 12. Is the model distinguishable from Geant4? Yes, completely (2026-10-08)

`src/ams_ecal/electron_studies/em_lateral_distinguishability.py` runs the project's registered classifier two-sample test (fixed gradient boosting, cross-validated AUC, permutation null, `ams_ecal.validation.added_checks`) on EXPOSED data only, as a diagnostic and never as a tuning target (the researcher's standing rule: the generator is not trained against a classifier).
- **Slice 2 and the faster tail: AUC 1.000 at 10, 20, 50 and 100 GeV** (null 99th percentile 0.53-0.54): perfectly separable. Removing the hit-cell and width features does not change it (1.00): other features separate completely.
- **The floor is not 0.5.** Two exposed Geant4 samples (the baseline against the extended-depth prefix) give AUC 0.58 / 0.61 / 0.70 / 0.75; the separating feature is the rear layer's occupancy (layer 17, univariate 0.60-0.69): the extended geometry has material behind the 18 layers. The hold-out is therefore a slightly different Geant4 population, which also sets how far "indistinguishable" can be pursued with this reference.
- **Single-feature separation** (`--single-features`, folded rank AUC): for both candidates the top separators at every energy are `n_hit_cells` and the per-layer occupied-cell counts `occupied_cells_i` (1.00 each, alone). The width alone separates at 0.88-0.89 for Slice 2 and 0.64-0.65 for the faster tail (the faster tail nearly removes the width signature).
  The profile, coupling, core and containment are not what separates the samples.

## 13. What carries the missing occupancy, and a proposed mechanism (analysis done; implementation awaits approval)

`src/ams_ecal/electron_studies/em_lateral_occupancy_diagnostic.py` (exposed electrons; the models use 2000 events): cells above the cell threshold (0.28 MeV), by distance from the track.
- **Occupied cells per event** (hold-out Geant4 against Slice 2 against faster tail): 278 / 149 / 147 at 10 GeV, 335 / 182 / 181 at 20, 419 / 231 / 230 at 50, 488 / 278 / 272 at 100: the models have about 55% of Geant4's occupied cells, and the faster tail does not change this.
- **By distance** (20 GeV; occupied fraction, Geant4 against Slice 2): 0-9 mm 0.997 / 0.967, 9-27 mm 0.981 / 0.870, **27-54 mm 0.910 / 0.478, 54-108 mm 0.487 / 0.113**, 108-216 mm 0.055 / 0.017. At 100 GeV: 54-108 mm 0.839 / 0.305. The missing cells are at 27-216 mm, in every layer group (front, middle and rear all low).
- **The energy carried by an occupied cell in the outskirts is a constant MeV-scale quantum in Geant4:** 1.2-1.5 MeV beyond 108 mm and 1.1-1.2 MeV beyond 216 mm, the same at 20 and 100 GeV. In the model it is 7.7-9.1 MeV there, which is the model's spot energy (the layer energy over its spot number): **the model's outer quantum is about 7 times too coarse**,
  and an outer cell is occupied only if a spot lands in it. (Close to the track the occupied cell energy is hundreds of MeV and matches: 410 against 428 MeV at 20 GeV.)
- The 108+ mm energy of the model is also about twice Geant4's (56 against 29 MeV per event at 20 GeV), the tail excess of section 7 that the faster tail removes.

**Proposed extension E2 (NOT implemented; needs the researcher's explicit approval, as the v0.8 research-session skill and CLAUDE.md require before a new mechanism is built).**
1. **Mechanism:** the radial TAIL component's spots carry a fixed absolute energy `eps_t` (a few tenths of an MeV to about 1 MeV), independent of the primary energy and of the layer, in place of the layer energy over the spot number; the core component is unchanged. The tail energy `(1 - p) E_layer` is then carried by `(1 - p) E_layer / eps_t` spots.
   Physically: low-energy photon and electron quanta (Compton and photoelectric deposits, the MeV scale of the critical energy) rather than equal shares of a layer's energy. E1's `kappa` was the relative form of this (tail spots at 1/2.66 of the core spot energy, still about 3 MeV); the measurement says the quantum is absolute.
   One added parameter.
2. **Calibration without tuning to a discriminator:** `eps_t` is the MEASURED mean energy per occupied cell at distances of at least 108 mm in the exposed baseline electrons (a single model-free number, about 1.2-1.5 MeV), fixed there and evaluated on the hold-out; it is not fitted to the hit-cell count, the width or the classifier. A free fit of `eps_t` would be a second variant to report, with the weight discipline of section 8.
3. **Bases:** E2 on Slice 2 (one added parameter) and on the faster tail (two added parameters in total), both reported.
4. **Adoption rule to be fixed in code before the run, proposed:** (A) occupied cells per event within 10% of the Geant4 hold-out at every energy and the hit-cell KS reduced by at least 50% against Slice 2; (B) no overfitting: the improvement transfers from the calibration set to the hold-out (at least 70%) and `eps_t` is not at a bound; (C) at most the parameter budget stated for the base, and no regression of the profile, core, containment, width and coupling beyond the tolerances of section 8.
   The classifier AUC and the single-feature separation are reported as outcomes, never as criteria. Given section 11, the researcher may prefer per-energy tolerances; that is for the researcher to set before the run.
5. **Expected limits (not claimed):** the rear-layer occupancy that already separates the two Geant4 samples (AUC 0.60-0.69 at layer 17) cannot be matched better than the hold-out agrees with the baseline; the longitudinal part (Slice 1) was never at the floor, so a perfectly occupied lateral model may still be distinguishable. The classifier AUC of 1.000 is therefore not expected to reach 0.5 from this step alone.

## 14. Research OS v0.8 and the choices I introduced that the researcher has not reviewed (ownership gaps)

The Research OS was updated to v0.8 (2026-10-08; installed: the `SKILL.md` files match the repository, `defense.py`, `tasks.py`, `literature_check.py` and `paper_access.py` are present). Changes that bear on this work: the literature scout and source verifier now distinguish reading depth from verification (applied retroactively in
`2026-10-07_em_lateral_structure_note.md`, the reading-depth table); the research-session skill asks for the defend-research workflow when consequential agent work is modified, with selective ownership records, and for human approval of a plan before building (E2 above waits for it). No defense target (the structured claim list with bound evidence and learning ledgers) was built: it needs the researcher's claims and ledgers,
and no presentation is pending; the list below is the agent-introduced input such a workflow would take.
Choices introduced by the agent that the researcher has not reviewed, and where they live:
1. the adoption-rule tolerances (30%, 70%, 0.2 / 0.3 points, KS 0.03 / 0.05, coupling 0.05, spot number 0.1): `em_production_lateral_e1_decision.py`, `em_production_lateral_tail_decision.py` (`THRESHOLDS`); none is derived (section 11);
2. the second-moment weights 0.02 and 0.4 (`MOMENT_WEIGHT`, `--moment-weight`, `em_production_lateral_calibration.py`);
3. the equal-weight cumulative residual of the radial fit, the 0.5 mm core bound, the 256-point Sobol quadrature, the energy-weighted ensemble mean;
4. the spot structure: the gamma depth density with two scale constants, the spot fit on layers 2-17 only, the central-three-cell statistic and the quadratic centre-of-gravity conditioning, the doubling constant 2 per layer of E1;
5. the generalised tail exponent (a Student-t strip marginal) of the faster tail;
6. the GP constants' assumptions: `E` in GeV, `Z = 82` for the composite (not verified in the paper).
A viva on the lateral slice, or a defense target for its claims, is available on request.
