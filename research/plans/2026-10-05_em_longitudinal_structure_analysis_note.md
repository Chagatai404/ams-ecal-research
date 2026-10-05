# F5 analysis note: the longitudinal structure of Geant4 electrons (analysis only)

Date 2026-10-05, branch `fastmc-validated-dataset`. Decision record: `DECISIONS.md` (DEC-014 and F1-F5). Code: `src/ams_ecal/electron_studies/em_longitudinal_structure_analysis.py` (+ `em_longitudinal_structure_plots.py`, `em_longitudinal_fluctuations.py`); results: `results/em_generator/longitudinal_structure_analysis.json`, `longitudinal_structure_joint.csv`, `longitudinal_structure_counterfactual_distances.csv`, `longitudinal_structure_confounders.csv`, plots in `results/em_generator/longitudinal_structure/`.

**Status.** Analysis only. No generator was changed (DEC-001 and the proton generator are untouched), no sealed data was read, nothing is validated. All numbers are on the EXPOSED Geant4 electrons (`data/geant4_electron_sample`, 1000 events at each of 10, 20, 50, 100 GeV): development / calibration data, so every comparison below is in-sample and can never be validation evidence. Sources used for decisions: the Grindhammer-Peters paper (verified from the full text) and the repository's own AMS references; the unresolved literature items are unchanged.

## 1. Answers, in the order of the brief

| question | answer |
|---|---|
| A. joint structure | (ln T, ln alpha) fluctuate as a correlated pair with a spread that agrees with the Grindhammer-Peters SAMPLING set within about 11% (not with the homogeneous set); rho 0.51-0.62 at a front-face depth origin. ln T is right-skewed and heavy-tailed (skewness +0.6 to +0.9, excess kurtosis +1.1 to +2.1, tail quantiles up to 0.7 sigma off a normal, least-squares fits; smaller with the cumulative estimator); ln alpha is close to normal (skewness 0.1-0.8). |
| B. event-level beta | At a front-face origin beta has median 0.52-0.53 (cumulative estimator 0.48-0.51), spread 0.08-0.11, no energy dependence (slope 0.003 per ln E), and only 48-55% of events within 20% of 0.65. **But beta, and rho, are not identified independently of the depth-origin convention**: refitted with the origin at -0.45 X0 the median is 0.56, with the origin that makes beta = 0.65 fit the mean profile (-2.3 to -2.85 X0) it is 0.70-0.72 with 60-73% of events within 20% of 0.65, and rho falls to 0.2-0.3. |
| C. does restoring alpha fix the early layers and leakage? | Partly, and for different reasons than expected (section 4). Restoring alpha FLUCTUATIONS fixes variance-type mismatches (longitudinal width, leakage spread, part of the layer correlations); it does almost nothing for the early-layer LEVEL or the mean leakage. Those depend on the MEAN beta and the depth origin. Neither correlated T/alpha alone nor the AMS backbone alone fixes the early layers. |
| D. AMS mean backbone | The AMS-form mean profile (alpha = 1 + 0.65 T) with the origin at the front face does NOT describe the Geant4 mean deposition profile (fit residual 6x a free fit; layer 0 under-predicted by 18-170x). With an origin offset of about -2.3 to -2.9 X0 it does, to within about 1.6x of a free fit. On the READOUT profile (the object the AMS statement is about) the AMS form with a free origin (about -1.9 to -2.2 X0) and free amplitude fits at the 1e-4 level and layer 0 is reproduced to 0.85-1.21. So the tension is the origin convention, not the value 0.65 as such. |
| E. lateral coupling | Survives the confounder controls. Entry-cell phase is not a confounder (partial = raw); rear leakage mediates part of it; ln alpha adds nothing beyond ln T. The coupling is real in Geant4 at the level of 0.3-0.8 (section 6). |
| Is correlated T/alpha enough to proceed? | **No, not on its own.** It is necessary (it fixes the variance structure) but a mean-profile structural issue remains: the depth origin (degenerate with beta), a first-layer energy floor, and a skewed ln T. Details and the recommended next slice in sections 7-8. |

## 2. A. Joint longitudinal structure

Fits of a gamma profile to the 18 layer fractions of each event (energy per layer over the primary energy, so the missing tail is the leakage), depth from the front face. Two estimators; `least_squares` is the layer-fraction fit, `cumulative` the fit to cumulative sums. Planted-correlation controls (`em_longitudinal_fluctuations.py`) show the fits do not inflate rho.

| E (GeV) | sigma(ln T) data / GP-sampling / GP-homog. / DEC-001 deposition | sigma(ln alpha) data / GP-sampling | rho LS / cumulative / GP-sampling | mean ln T data / DEC-001 deposition | skew ln T, ln alpha (LS) |
|---|---|---|---|---|---|
| 10 | 0.154 / 0.154 / 0.131 / 0.131 | 0.209 / 0.206 | 0.62 / 0.69 / 0.62 | 1.876 / 1.891 | +0.61, +0.41 |
| 20 | 0.149 / 0.136 / 0.117 / 0.117 | 0.205 / 0.185 | 0.60 / 0.68 / 0.60 | 1.992 / 1.991 | +0.75, +0.79 |
| 50 | 0.124 / 0.118 / 0.103 / 0.103 | 0.168 / 0.163 | 0.59 / 0.65 / 0.58 | 2.115 / 2.110 | +0.88, +0.61 |
| 100 | 0.119 / 0.107 / 0.095 / 0.095 | 0.160 / 0.150 | 0.51 / 0.56 / 0.57 | 2.206 / 2.191 | +0.72, +0.12 |

- The measured spread follows the **sampling** set, not the homogeneous set that the DEC-001 `deposition` regime uses, so DEC-001's width is 15-22% too narrow against these events, while its MEAN depth is right (within 0.015 in ln T). The data therefore pair the deposition-regime mean with the sampling-regime width, which the configuration comment calls incoherent. The Geant4 `deposition` grid is the energy deposited in layers of a sampling structure; whether the sampling width is legitimately inherited is a question for the researcher, not settled here.
- The fitted spreads include fit noise, so the intrinsic spreads are somewhat smaller; the 0-11% excess over the formula is of that size.
- Non-Gaussianity (effect sizes, not p-values; `ln_t.max_abs_quantile_deviation` 0.27-0.71 sigma, Mahalanobis KS against chi-square(2) 0.07-0.11): ln T is materially skewed and heavy-tailed at every energy; the cumulative estimator reduces but does not remove it (skewness +0.3 to +0.7). This is the F1 criterion for considering a copula or a skewed marginal; the estimator dependence means it should be re-checked at the calibrated origin before a decision (at the shifted origin ln T skewness is +0.9 to +1.1).
- Grindhammer-Peters appendix numbering: the verifier recorded the sampling fluctuation formulae at App. A.2.2 (p. 14), the configuration comments cite A.2.3. Not reconciled; the numerical values agree between the two.

## 3. B. Event-level beta and the depth origin

| E (GeV) | beta median, sd (LS, front face) | within 20% of 0.65 | beta median at origin -0.45 X0 | beta median, within 20%, rho at the mean-profile origin (-2.29 / -2.54 / -2.63 / -2.85 X0) | readout beta median (diagnostic) |
|---|---|---|---|---|---|
| 10 | 0.520, 0.108 | 0.48 | 0.559 | 0.709, 0.63, rho 0.32 | 0.540 |
| 20 | 0.524, 0.105 | 0.51 | 0.559 | 0.721, 0.60, rho 0.32 | 0.549 |
| 50 | 0.522, 0.083 | 0.50 | 0.557 | 0.702, 0.73, rho 0.32 | 0.544 |
| 100 | 0.528, 0.081 | 0.55 | 0.557 | 0.708, 0.72, rho 0.21 | 0.550 |

Fixing beta at 0.65 costs a median factor 1.35-2.02 in fit residual per event at the front-face origin, 1.14-1.38 at the mean-profile origin. So beta = 0.65 is not at the centre of the front-face distribution, and is near the centre once the origin is moved upstream by 2-3 X0. The central value of beta is therefore a statement about a convention (where depth zero is), not a convention-free property of the showers. The AMS statement is about a fit of observed signals with its own convention, which this repository does not have; the readout fits above are the closest available test.

## 4. C. Counterfactual ensembles (4000 events per variant and energy)

Variants, all against the Geant4 deposition fractions: DEC-001 in each of its regimes; `fixed_beta_data_T` (data ln T distribution, alpha = 1 + 0.65 T); `fixed_beta_matched_mean` (same, beta = the data median); `joint_data` (data-calibrated bivariate Gaussian); `joint_ams_mean_gp_fluct` (DEC-001's T distribution with alpha restored using the GP sampling sigma and rho and the AMS backbone mean); the same with DEC-001's sampling-regime T; and two variants at the shifted origin. Layer-0 ratio is the model's mean layer-0 fraction over Geant4's; leakage is the energy beyond the last layer.

| E | variant | layer 0 | layers 0-3 | leakage mean (Geant4) | leakage sd (Geant4) | long. rms KS | max lag-correlation difference |
|---|---|---|---|---|---|---|---|
| 10 | DEC-001 deposition | 0.11 | 0.81 | 0.023 (0.057) | 0.013 (0.029) | 0.89 | 0.44 |
| 10 | fixed beta, matched mean | 0.30 | 0.97 | 0.040 | 0.021 | 0.47 | 0.44 |
| 10 | joint, data calibrated | 0.37 | 0.96 | 0.046 | 0.031 | 0.10 | 0.29 |
| 10 | joint, AMS mean + GP fluctuations | 0.16 | 0.82 | 0.027 | 0.019 | 0.46 | 0.29 |
| 100 | DEC-001 deposition | 0.02 | 0.64 | 0.072 (0.123) | 0.028 (0.052) | 0.91 | 0.31 |
| 100 | fixed beta, matched mean | 0.07 | 0.86 | 0.108 | 0.043 | 0.36 | 0.31 |
| 100 | joint, data calibrated | 0.10 | 0.88 | 0.114 | 0.054 | 0.12 | 0.23 |
| 100 | joint, AMS mean + GP fluctuations | 0.03 | 0.66 | 0.074 | 0.033 | 0.56 | 0.18 |

(The 20 and 50 GeV rows are in `longitudinal_structure_counterfactual_distances.csv`.) The decomposition, as the fraction of the discrepancy removed, with the same ln T throughout:

- **Mean beta only** (0.65 to the measured median; `fixed_beta_data_T` to `fixed_beta_matched_mean`): removes 65-75% of the layers 0-3 discrepancy, 64-80% of the profile error in layers 0-3, 52-69% of the mean-leakage discrepancy, about 50-69% of the contained-energy KS, 32-38% of the layer-0 discrepancy; it does not touch the layer correlations.
- **Fluctuation only** (matched mean beta to the joint): removes 66-85% of the leakage-spread mismatch, 68-80% of the longitudinal-width KS, 24-41% of the layer-correlation mismatch, 11-18% of the layer-0 discrepancy, and nothing (or a little harm) for the layers 0-3 level.
- **The F1 structure as literally stated** (DEC-001's T and AMS backbone mean, alpha restored with the GP sigma and rho): removes only 13-18% of the layer-0 discrepancy, 4-8% of the layers 0-3 level, 5-12% of the mean leakage, but 29-49% of the leakage spread, 39-49% of the width KS and 34-43% of the layer-correlation mismatch. The ensemble mean of this structure is 1.5-4 times the deterministic backbone in layers 0-1 (the nonlinearity of the gamma profile), still far from Geant4.
- **At the shifted origin** (-2.3 to -2.85 X0, AMS beta = 0.65 kept): the layer-0 level overshoots by a factor 1.6-2.2 (joint 1.6-1.8) and layers 0-3 by 1.09-1.35, the profile error in layers 4-17 is good (0.02-0.036), the mean leakage is 0.036-0.105 against 0.057-0.123. The truth for layer 0 lies between the front-face fits (under by 3-50x with fixed beta, 2.7-10x with joint) and the shifted-origin fits (over by 1.6-2.2x).
- Residual after all of it: layer-to-layer correlations (largest lag-averaged difference 0.18-0.29 for the best variants), contained-energy KS 0.15-0.48, leakage spread within 10% only at the data-calibrated variants.

## 5. D. The mean profile, the first layers and the AMS backbone

- The first two layers of Geant4: layer 0 holds 37.3, 42.7, 58.3, 74.9 MeV and layer 1 holds 134, 164, 218, 267 MeV at 10, 20, 50, 100 GeV, against 2.0 to 0.6 MeV and 47 to 38 MeV for the AMS backbone. Layer 0 grows as E^0.31 (linear: about 35 MeV + 0.42 MeV per GeV), layer 1 as E^0.30; a shower fraction would scale as E. A slowly growing energy floor of tens of MeV points to leading-particle ionisation and very early development, which a gamma profile started at the front face does not contain. This is an inference from the energy dependence, not a measured mechanism.
- Structural fits of the Geant4 mean deposition profile (rms residual; layer-0 Geant4/fit): free gamma 0.0011-0.0015 (3.5-15); free gamma with origin 0.0010-0.0013, origin -0.45 to -0.5 X0 (1.7-5.1); free gamma with amplitude 0.0008-0.0011, amplitude consistent with 1 (4.0-19; normalisation is not the problem); fixed beta 0.65 at the front face 0.0072-0.0087 (18-170); fixed beta with origin 0.0018-0.0024, origin -2.3 to -2.85 X0 (0.58-0.68).
- On the readout mean profile (amplitude free): free gamma 6e-5 to 8e-5 (layer 0 under by 4-19x); fixed beta 0.65 at the front face 2.9e-4 to 3.9e-4 (18-158); **fixed beta with origin 6e-5 to 1.2e-4 (0.85-1.21)** at origins -1.9 to -2.2 X0. The AMS form with an upstream origin reproduces the observed-signal mean profile as well as a free gamma, and reproduces layer 0, which the free gamma without an origin does not.
- Normalisation, finite-layer integration and the leakage treatment (F4 items 4-6) are not the cause (amplitude fits at 1, the gamma tail is the leakage); the "definition of deposition" item (7) is the grid sum over all material in the layer. Items 1-3: the second degree of freedom and event-level beta explain the width and spread, the mean beta and the origin explain most of the layers 0-3 level, and a residual layer-0 energy floor remains.

## 6. E. Lateral-longitudinal coupling and confounders

Spearman of the per-event lateral observable with ln T (ranks of all variables), at 10 / 20 / 50 / 100 GeV: raw, after entry-cell phase, after phase and rear leakage.

| observable | raw | controlling entry phase | phase and rear leakage |
|---|---|---|---|
| width | -0.27 / -0.37 / -0.56 / -0.70 | -0.28 / -0.37 / -0.57 / -0.71 | -0.19 / -0.17 / -0.25 / -0.41 |
| hit cells | -0.56 / -0.61 / -0.64 / -0.68 | -0.56 / -0.61 / -0.64 / -0.68 | -0.39 / -0.37 / -0.32 / -0.28 |
| core fraction | +0.38 / +0.50 / +0.66 / +0.78 | +0.40 / +0.53 / +0.72 / +0.84 | +0.23 / +0.26 / +0.33 / +0.56 |
| containment | +0.40 / +0.49 / +0.69 / +0.83 | +0.41 / +0.50 / +0.70 / +0.84 | +0.26 / +0.28 / +0.33 / +0.58 |

- Entry phase (offset of the shower axis from its cell centre, within +-4.5 mm) does not confound: partial equals raw within the bootstrap interval (95% half-widths 0.04-0.06); stratified by phase tertile the values stay the same. All events are at least 35 cells from the detector border: edge effects are absent in this sample. Controlling ln alpha changes ln T's coupling little (for example width at 100 GeV -0.70 to -0.67).
- Rear leakage is a plausible mediator (ln T with rear leakage: Spearman 0.65-0.84; deeper showers leak more): controlling it removes about 30-60% of the coupling, leaving 0.17-0.58 in magnitude. The coupling is carried by ln T; ln alpha's own association is mostly explained by its correlation with ln T.
- Reading: the coupling is not a phase or edge artefact and is not only leakage. Whether the Grindhammer-Peters spot model, with its depth coupling through tau_i, reproduces it quantitatively can be tested only with an implementation. The numbers above are the target, not a parameter to hard-code.

## 7. Supported, falsified, unresolved

**Supported.** A correlated (ln T, ln alpha) structure with the Grindhammer-Peters sampling spreads and a correlation of 0.5-0.6 at a front-face origin; the DEC-001 mean depth of the maximum (deposition regime); the AMS functional form with an origin offset as a description of the OBSERVED (readout) mean profile; a real lateral-longitudinal coupling that survives phase and edge controls; the second degree of freedom as the cure for the longitudinal width and leakage spread.

**Falsified (for this data and this origin convention).** That the AMS backbone with beta = 0.65 and a front-face origin describes the Geant4 deposition mean profile; that DEC-001's `deposition`-regime width is adequate (15-22% too narrow); that restoring alpha with the AMS mean fixed removes the early-layer deficit or the mean leakage mismatch; that a Gaussian ln T is adequate; the adversarial report's "rho 0.87-0.90, formula refuted" (earlier, superseded).

**Unresolved.** (i) The depth-origin convention: beta, rho and the early-layer level all depend on it, and AMS's convention is not documented in the repository; it is a parameter, not a settled number. (ii) The layer-0/1 energy floor (the mechanism is inferred, not measured). (iii) Whether the apparent non-Gaussianity of ln T survives at the calibrated origin and with a better estimator. (iv) Layer-to-layer correlations (about 0.2 left). (v) Whether the Grindhammer-Peters spot model reproduces the coupling. (vi) The three literature items already listed (broader counterevidence search, Kotwal-Hays leakage reading, Longo & Sestili details) and the appendix numbering A.2.2/A.2.3. (vii) Everything here is one sample of 1000 events per energy; estimator dependence is visible in the cumulative-versus-least-squares differences.

## 8. Recommendation for the next slice (waits for approval; no generator physics is changed before it)

Correlated T/alpha is necessary and not sufficient. The outcomes of the brief map to: A (alpha alone fixes most of the longitudinal mismatch): **no**, it fixes the variance structure; B (a remaining structural mean-profile problem needs investigation): **yes**, the depth origin together with beta, and a first-layer floor; C (a smooth mean correction is necessary): **not yet justified**: the origin is a named, physical convention that must be calibrated first, and only a residual after that would justify a smooth low-dimensional c(depth, log E) of the kind F4 allows; D (the coupling survives): **yes**.

Proposed next slice, still ANALYSIS ONLY, small:
1. A joint mean-profile calibration study with at most three named parameters, smooth in ln E: the depth origin t0(E), the mean beta (or alpha) and, only if the residual demands it, one first-layer floor term; fitted with relative weights so that layers 0-2 count; compare with the AMS-fixed-beta alternative. This answers whether the AMS backbone (beta = 0.65, origin as a calibrated parameter) is adequate for the deposition profile, as F1 intends, and whether a floor is needed.
2. Re-estimate the (ln T, ln alpha) fluctuations at the calibrated origin with both estimators, and decide the ln T marginal (Gaussian, or a copula/skewed marginal) from those effect sizes, per F1.
3. Then, on approval, the first implementation slice: a new longitudinal module for `em_production` (stage 1-2, development model, tests, no lateral yet), leaving DEC-001 untouched as `em_smooth_null`.

Decisions for the researcher: (a) is the depth origin an acceptable calibrated parameter of the production generator (it changes the meaning of the AMS 0.65 constraint to "beta at a calibrated origin")? (b) may the width follow the sampling set (as measured) in the deposition regime? (c) approval of the F6 analysis above.

## 9. How to reproduce

`uv run python -m ams_ecal.electron_studies.em_longitudinal_structure_analysis` (about 15 minutes; writes the JSON, the three CSVs and the nine plots); tests in `tests/electron_studies/test_em_longitudinal_structure_analysis.py`, `tests/electron_studies/test_em_longitudinal_fluctuations.py`. The exact test count of the whole suite at this commit is recorded in `research/STATE.md`.
