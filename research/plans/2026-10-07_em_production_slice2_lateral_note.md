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

## 8. Next steps (not started)

A tail-sensitive radial calibration (the second moment as a second target, no new parameter), layer-to-layer residual noise (stage 4), the pre-stated gamma-inadequacy thresholds (still pending), then the single opening of the sealed electron set. Each needs approval; E1 is recorded here and stays opt-in, off.
