# Depth-origin and mean-beta calibration of Geant4 electrons (calibration analysis only)

Date 2026-10-06, branch `fastmc-validated-dataset`. Decisions: `DECISIONS.md` (DEC-014, F1-F5 and the 2026-10-06 rows). Code:
`src/ams_ecal/electron_studies/em_depth_origin_calibration.py` (+ `em_depth_origin_calibration_plots.py`); tests
`tests/electron_studies/test_em_depth_origin_calibration.py`; results `results/em_generator/depth_origin_calibration.json`,
`depth_origin_calibration_fits.csv`, `_surface.csv`, `_structure.csv`, `_skewness_versus_origin.csv`, `_residuals.csv`,
`depth_origin_calibration_model_families.json` and twelve plots in `results/em_generator/depth_origin_calibration/`. Command:
`python -m ams_ecal.electron_studies.em_depth_origin_calibration` (add `--model-families` for section 2). This is the slice called "F6" in the
decision record; see `NAMES.md`.

**Status.** Calibration and analysis only. No generator was written or changed (DEC-001 and the proton generator are untouched), no
sealed data was read (the sealed electron set stays eligible as the final fresh electron validation set), nothing is validated. All
numbers are on the EXPOSED Geant4 electrons (1000 events at each of 10, 20, 50, 100 GeV): development / calibration data. The fitted
origin is a **calibrated coordinate convention**, not a physical shower-start point; beta = 0.65 at the front face is not asserted.

## 1. Answers

| question | answer |
|---|---|
| Best-fit z0 and beta (deposition, one common pair, DEC-001 depth law, stochastic ensemble mean) | **z0 = -0.98 X0** (upstream of the front face), **beta = 0.571**. Bootstrap (200 event resamples) sd 0.021 and 0.003, correlation -0.89: statistical errors only. |
| Identifiability | **Identified within this model family, with one strong valley, and the statistical error is not the limiting uncertainty.** The objective has a single compact minimum (68% / 95% regions are 3 / 7 grid points, no grid-edge contact). The valley runs along beta = 0.571 - 0.14 (z0 + 0.98): moving z0 by -0.5 X0 costs +58 scaled delta chi-square (at beta 0.64, z0 -1.5), +199 at -2.0. The model-form spread is much larger than the bootstrap: z0 -0.98 (primary), -1.28 (no fluctuations), -0.96 (naive width), -1.06 (free depth offset), -1.12 (free amplitude); readout -1.08. So **z0 = -1.0 +/- 0.3 and beta = 0.57 +/- 0.02 is the honest statement**, and across fixed-beta families z0 runs from -0.98 to -2.1 (section 2). |
| Does one common pair serve all energies? | Mostly. Independent per-energy fits: z0 -0.86, -0.83, -1.06, -1.20 and beta 0.579, 0.559, 0.578, 0.583 (scaled sd 0.09-0.17 and 0.012-0.025). Leaving one energy out, the chi-square per point is 28, 21, 31, 78 against 36 in sample: **100 GeV is predicted about twice as badly**. The limiting factor is not commonality: even the per-energy fits leave chi-square per degree of freedom 17, 14, 32, 64. |
| Does the AMS mean constraint (beta = 0.65) survive? | **Not for the deposition profile; marginally for the readout profile.** With beta fixed at 0.65 the best origin is -1.43 (common depth): chi-square 4621 against 2494 (scaled delta 59), layers 4-17 rms 4-5% against 2-3.6%. With free depth per energy: 3792 against 2392 (scaled delta 39). On the **readout** profile (amplitude free, as AMS fits observed signals): 2349 against 1788 (scaled delta 22) at common depth, **1721 against 1574 (scaled delta 6) with free depth**: beta = 0.65 is nearly acceptable there. Early layers with beta = 0.65 are not worse (layers 0-3 rms 17.5 / 15.5 / 11.3 / 17.4% against 11 / 11 / 15 / 22%). |
| Does the GP structure survive at the calibrated coordinate? | **Yes, with a modest correlation deficit.** Covariant widths (sigma_GP x_max / T_o): ln T measured / GP 0.97, 1.07, 1.04, 1.10; ln alpha 0.96, 1.05, 0.98, 1.02 (all inside 0.75-1.25). Pearson rho(ln T, ln alpha) 0.48, 0.48, 0.49, 0.40 (least squares) or 0.58, 0.58, 0.56, 0.47 (cumulative) against GP 0.62, 0.60, 0.58, 0.57: a deficit of 0.09-0.17 (least squares) or 0.02-0.10 (cumulative). A refit control shows the estimator itself lowers rho by only about 0.03. Event-level beta median 0.59-0.60 (least squares) or 0.55-0.56 (cumulative), close to the calibrated 0.571. No compensating-fit warning: the origin that fixes the mean does not break the fluctuation relations. |
| Does ln T still need non-Gaussian treatment? | **Yes; not settled.** Skewness of ln T at the calibrated origin +0.75, +0.85, +0.96, +0.79 (least squares), +0.43 to +0.81 (cumulative); excess kurtosis 1.4-2.4. It grows slightly as the origin moves upstream (10 GeV: +0.61 at 0 to +0.90 at -3 X0), so **no origin convention removes it**. Planted-null control (a bivariate NORMAL with the measured mean and covariance, plus residuals borrowed from real events, refitted): refitted skewness -0.05 to +0.03, so the per-event fit does not create it (caveat: borrowed residuals do not reproduce each event's own shape-dependent noise). The original GP bivariate Gaussian is therefore **not supported as it stands for ln T**; a marginal model or copula is **recommended for evaluation, not implemented**. Its consequence has not been measured: even with the measured Gaussian covariance, the contained-fraction distribution keeps KS 0.21-0.39. |
| Is a first-layer floor needed? | **No: the gate is closed and the floor is rejected as not applicable.** Layer-0 residual (model - Geant4)/Geant4 is +14.5%, +6.6%, -18.9%, -37.1% at 10, 20, 50, 100 GeV: it **changes sign with energy**, fails "same sign", "reproducible magnitude" and the leakage-accounting condition (contained fraction off by 1.5-2.5%); it is significant (|pull| 4-27) and not absorbed by free amplitude, delta or free depth. In MeV the model layer 0 is nearly flat (42.7 to 47.1) while Geant4 grows (37.3 to 74.9). A floor would add at most 28 MeV at 100 GeV, but layers 0-3 are over-predicted in total by +50 to +380 MeV (+5% to +13%): the dominant misfit is in layers 1-3, not an entrance deficit. No floor parameter was fitted. |
| Deposition against readout (interpretation only) | Readout (amplitude free): z0 -1.08, beta 0.597 (bootstrap sd 0.029, 0.005), a readout-minus-deposition difference of -0.10 X0 and +0.026 in beta: 3.5 and 5.8 bootstrap sd, but small in absolute size. The deposition optimum is outside the readout 95% region (scaled delta 11.9). The fitted amplitude is constant, 0.064 at every energy. This is consistent with a representation (sampling-response) difference; there is **no evidence** that AMS defined its origin nearer a readout convention, and none against it. Not a second tuning target. |
| Is implementation now justified? | **No** (section 4). |

## 2. Why the origin depends on the model form (reconciling the earlier longitudinal-structure analysis)

The earlier analysis (`2026-10-05_em_longitudinal_structure_analysis_note.md`) reported an origin of about -2.3 to -2.9 X0 with beta fixed at 0.65
and a free depth. This calibration reproduces the order of magnitude and shows where it comes from. Same data, deposition (readout in brackets);
`--model-families` regenerates this table:

| family | z0 | beta | chi-square / dof |
|---|---|---|---|
| beta fixed 0.65, free depth per energy, **deterministic** profile | -2.10 (-2.04) | 0.65 | 63.9 (25.7) |
| beta fixed 0.65, free depth, ensemble over (ln T, ln alpha) | -1.56 (-1.52) | 0.65 | 56.6 (25.7) |
| beta fixed 0.65, common depth law | -1.43 (-1.39) | 0.65 | 65.1 (33.1) |
| beta free, free depth, deterministic | -1.38 (-1.67) | 0.565 (0.607) | 37.0 (22.7) |
| beta free, free depth, ensemble | -1.05 (-1.27) | 0.576 (0.615) | 36.2 (23.8) |
| beta free, common depth law (primary) | **-0.98** (-1.08) | **0.571** (0.597) | 35.6 (25.5) |

The origin moves by about 1 X0 with whether beta is fixed and whether the fluctuation average is taken. It is a coordinate convention tied to the
model form; it should be carried with the form that produced it and never quoted alone. The earlier -2.3 to -2.9 is not reproduced exactly (the
nearest case here, deterministic with beta fixed, gives -2.1 and -2.0); the remaining gap is not reconciled.

## 3. What the calibration improves, and what it does not

| quantity | front-face AMS (beta 0.65, z0 0) | front-face free beta | calibrated (z0 -0.98, beta 0.571) |
|---|---|---|---|
| total chi-square (72 / 71 / 70 dof) | 38176 | 12872 | **2494** |
| layers 0-3 rms relative residual, 10 / 20 / 50 / 100 GeV | 50 / 55 / 62 / 66% | 31 / 34 / 41 / 45% | **11 / 11 / 15 / 22%** |
| layers 4-17 rms | 9-11% | 4-10% | **2-3.6%** |
| contained fraction, model vs Geant4 (0.943 / 0.927 / 0.905 / 0.877) | 0.972 / 0.962 / 0.942 / 0.923 | 0.942 / 0.927 / 0.901 / 0.876 | 0.957 / 0.944 / 0.921 / 0.899 |

Ensembles at the calibrated coordinate against DEC-001 (distance removed, with the measured spreads / the GP-prescribed spreads): layer-0 log ratio
83-99% / 89-97%; early four layers 83-97% / 65-74%; layer-0-3 profile error 74-89% / 68-77%; mean leakage 53-74% / 55-61%; leakage spread 93-98% /
66-88%; rms width 90-93% / 85-88%; lag correlations only 25-35% / 21-33%. Remaining distances: mean leakage still 1.3-2.2 points too small, layers 0-3
11-22%, contained-fraction KS 0.21-0.39.

This is a **partial success** on the brief's criterion: the mean profile, early-layer energy and mean leakage improve together and the T/alpha spreads
stay plausible, but the profile is not statistically adequate (chi-square per dof 36) and the 100 GeV profile is the worst. The residual is structured
(model layer-0 energy too flat in E, layers 1-3 too large, containment too high), so it is a shape problem of the gamma form with this depth law, not a
single missing parameter.

## 4. Implementation gate (the seven conditions)

| condition | verdict | evidence |
|---|---|---|
| 1. a defensible common origin | **partly** | -0.98 X0 for the primary form, spread -1.0 to -2.1 across forms, per-energy -0.83 to -1.20; defensible only as a convention carried with its model form |
| 2. beta identifiable or the degeneracy manageable | **yes** | one valley, beta = 0.571 - 0.14 (z0 + 0.98); bootstrap sd 0.003; model-form range 0.558-0.597 |
| 3. mean profile substantially improved | **yes, not adequate** | chi-square 38176 to 2494; layers 0-3 rms 50-66% to 11-22%; chi-square per dof still 36 |
| 4. calibrated-coordinate GP widths supported | **yes** | covariant ratios 0.97-1.10 and 0.96-1.05 |
| 5. T-alpha structure defensible | **yes, with a deficit** | rho 0.40-0.49 against 0.57-0.62 (cumulative 0.47-0.58) |
| 6. ln T marginal settled | **no** | skewness +0.75 to +0.96 persists at every origin; not a fit artifact in the planted-null control |
| 7. floor rejected or justified | **rejected (not applicable)** | gate closed: sign changes with energy; not a constant or linear floor |

**Recommendation: do not start the implementation slice yet.** Conditions 1, 3 and 6 are not met. What I would do next, all analysis only:

1. **Find why the gamma form with this depth law cannot reach the 100 GeV profile** (layer 0 flat in E, layers 1-3 high, containment high): try an energy-dependent
   beta or origin, then the depth law itself, and a non-gamma entrance shape, each scored by leave-one-energy-out chi-square and held-out resampling. The aim is
   to know whether one common pair is sufficient or whether the model needs a named energy dependence.
2. **Measure whether the skewed ln T marginal matters**: simulate the Gaussian and a skewed marginal (the empirical marginal as the upper bound) and compare
   the observables it should change (leakage tail, contained-fraction KS, longitudinal width).
3. **State the AMS comparison on the readout representation** with the readout detector response in view, since the AMS constraint is nearly compatible
   there and clearly not on deposition.

If you accept the residual as a documented limitation instead, the first implementation slice would be the smallest consistent one: the mean profile at (z0, beta)
of the chosen form with the sampling-set covariant GP widths and the original bivariate Gaussian, `deposition` only, DEC-001 kept as the control, no floor, no lateral
change, calibration on the exposed electrons, and the sealed electron set opened once at the end. I do not recommend that before items 1 and 2.

## 5. Requirement carried to the lateral slice (not implemented)

The longitudinal-lateral coupling survives entry-phase and edge controls, and rear leakage mediates only part of it (longitudinal-structure analysis).
When the Grindhammer-Peters spot model is implemented, the coupling must survive the same confounder checks; the measured correlations are not to be encoded by hand.

## 6. Limits

- Exposed development data only; the SEs rescale by chi-square per dof (about 36), so formal intervals are model-conditional.
- The ensemble mean uses the GP sampling-set widths with the covariant factor x_max / T_o; the naive width changes z0 by +0.03 and beta by +0.006.
- The skewness control borrows residuals from other events; it does not reproduce an event's own shape-dependent noise.
- The estimator choice matters for rho and skewness (least squares against cumulative); both are reported.
- E_c = 7.6 MeV and x_max = ln(E / E_c) - 0.5 are taken from DEC-001; a free depth per energy (xmax_k = 6.7, 7.4, 8.3, 9.0) did not improve the fit materially.
