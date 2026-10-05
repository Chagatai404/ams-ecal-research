# EM production generator — PLAN (no implementation before approval)

_Drafted 2026-10-05 on branch `fastmc-validated-dataset`, after the first out-of-sample comparison of the DEC-001 EM generator with Geant4 electrons. Direction accepted by the researcher on 2026-10-05 (`research/DECISIONS.md`); the model's formulation is NOT yet approved, and by `CLAUDE.md` nothing is implemented until it is._

## 1. Why

`results/em_generator/exploration_comparison.json` (module `ams_ecal.em_exploration_check`, unsealed exploration electrons, `deposition` representation): KS 0.57-0.75 on event energy, 0.90-0.94 on longitudinal RMS, 0.64-0.73 on core fraction, 0.76-0.80 on containment, 0.72-1.0 on hit cells, 0.29-0.68 on width; classifier AUC 1.00 at every energy; layer-correlation row material at every energy. The generator was never fitted to these events, so this is an honest out-of-sample falsification of its adequacy as a PRODUCTION generator. It is not a defect in its other role.

## 2. Two models with two roles

| name | what | role |
|---|---|---|
| `em_smooth_null` | the DEC-001 generator, unchanged (`ams_ecal.stochastic`) | permanent smooth analytic reference: control for the multiscale estimators, interpretable baseline. Never mixed into the e/p dataset. |
| `em_production` | NEW, calibrated to Geant4 electrons | the electron half of datasets. Supersedes DEC-001 for dataset generation only. |

The AMS-inspired mean longitudinal profile may stay inside `em_production` as a prior (physics constraint). Scientific upside: smooth null -> calibrated FastMC -> full Geant4 is a controlled hierarchy for the multiscale project; a statistic that moves monotonically along it is measuring shower structure that the simple parameterisation erased.

## 3. What the comparison localises (targets for the hierarchy)

1. Visible energy and leakage: model energy 3-5% high at 100 GeV with a missing low tail (Geant4 5% quantile 78.0 GeV against the model's 88.1 GeV): rear leakage and fluctuation of the contained fraction.
2. Longitudinal: mean layer-energy ratio model/Geant4 is 0.02-0.07 in layer 0 and 0.22-0.41 in layer 1, 1.15 in the middle, 0.76-0.89 at the end; the longitudinal RMS has almost no event-to-event spread (30.9-31.8 mm against 31.4-36.9 mm).
3. Lateral: width 14.9 against 13.9 mm, core fraction 0.89 against 0.87, 387 against 334 hit cells at 20 GeV and 621 against 483 at 100 GeV: too wide and too smooth.
4. Correlations: layer-to-layer and longitudinal-lateral, flagged by the layer-correlation row.

## 4. Calibration order (hierarchical; each stage frozen before the next)

Stage 1 visible energy and leakage. Stage 2 longitudinal fluctuations and shower maximum. Stage 3 lateral width, tails and hit sparsity. Stage 4 layer-to-layer and longitudinal-lateral correlations. Never optimise classifier AUC or any single discriminator: AUC stays an omnibus diagnostic, otherwise the generator is trained to fool one classifier. No stage adds parameters merely to remove a KS value; each parameter needs a named physical or statistical meaning and a measured need.

## 5. Procedure required by CLAUDE.md for stochastic generation (status)

| step | status |
|---|---|
| probe the researcher's current understanding | not done: part of the planned tutoring session |
| missing prerequisites identified | not done |
| independent external review of shower-fluctuation literature | DONE as discovery, thin (2026-10-05, abstract depth, air-shower papers wrongly used and struck); see `plans/2026-10-05_em_literature_verification_record.md` |
| independent verification of implementation-affecting claims | DONE for what exists: Grindhammer-Peters verified from the full text; AMS-02 ECAL geometry verified; no published fluctuation numbers for THIS detector (the Geant4 electrons supply them: `results/em_generator/longitudinal_fluctuations.json`); adversarial pass done (its headline refuted by a control, see the record, section 4b); a fuller counterevidence search is still open |
| reconcile external evidence with the deterministic generator | partly: the comparison above is the data side of it |
| formulate the stochastic physical model | not done |
| researcher approval | not given |
| implementation | not started |

## 6. Data and protocol

- Calibration and development data: `data/geant4_electron_sample` (4 energies x 1000 electrons, exploration). All of it has now been looked at, so its events can never provide validation evidence (amendment of 2026-10-05 to the registrations).
- **The sealed electron set (`data/geant4_electron_sealed`, 7 energies x 1000, base seed 20271002) was not exposed and remains closed.** A "fresh untouched validation sample" may be unnecessary: the existing sealed set can serve, unless the researcher wants a new one with different seeds. Decision needed.
- Tolerances: from a split-Geant4 self-comparison noise floor plus physics-motivated tolerance per observable class (energy/leakage, longitudinal, lateral, sparsity, correlations); registered BEFORE any sealed opening.
- Geant4 EM sensitivity: a different EM option (for example `FTFP_BERT_EMZ`; availability in the installed 11.4.1 build to be verified) on electrons. It needs its own approval under the Geant4 workflow. It is not the QGSP_BERT comparison, which changes only hadronic physics.

## 7. Questions for the researcher (answered 2026-10-05: Q10 yes; Q11 use the existing sealed electron set; Q12 literature first, tutoring after)

- **Q10.** Approve the two-role naming and keeping the DEC-001 code untouched as `em_smooth_null`?
- **Q11.** Use the existing sealed electron set as the untouched validation sample, or generate a new one?
- **Q12.** When should the tutoring on the generator's physics (probe, prerequisites) take place relative to the literature verification?

## 8. Reconciliation and PROPOSED formulation (2026-10-05, for the researcher's approval; nothing implemented)

### 8.1 What the DEC-001 code does and what the evidence says (reconciliation)

Read in `src/ams_ecal/stochastic.py` (lines 169-263) and `longitudinal.py` (lines 311-340): one standard normal per event gives `T0 = exp(mu(E) + s(E) z)` with `s(E) = 1/(intercept + slope ln(E/E_c))` (the width law has the Grindhammer-Peters form; the shower-maximum width therefore already agrees with the measured sigma(ln T), `results/em_generator/longitudinal_fluctuations.json`), then `alpha = 1 + beta T0` with `beta` FIXED, a gamma profile integrated over the 18 layers, and a deterministic lateral grid (`lateral.track_centered_cell_fractions`).

| element | DEC-001 | measured in Geant4 electrons (exploration) | literature (verified) |
|---|---|---|---|
| sigma(ln T) | G-P width law | agrees with the formula within about 11% | G-P App. A.2.2 |
| second longitudinal variable | none: alpha locked to T0 through fixed beta (rho = 1) | sigma(ln alpha) 0.16-0.21, rho(ln T, ln alpha) 0.5-0.7 | G-P: bivariate Gaussian, rho 0.57-0.62 |
| mean profile | gamma, fixed beta | layer 0 holds 2-7% and layer 1 22-41% of Geant4's energy; middle layers +15%, last layers -11 to -24% | G-P has no thin-calorimeter regime |
| leakage / contained fraction | follows from the profile | median 0.951 / 0.936 / 0.915 / 0.890 at 10 / 20 / 50 / 100 GeV; model 3-5% high at 100 GeV with no low tail | not found for this detector |
| lateral | deterministic grid | width, hit cells, core fraction, containment correlate with ln T (Spearman up to 0.7-0.8 at 100 GeV); too wide and too smooth in the model | G-P: lateral parameters deterministic in tau, fluctuations from spot generation coupled through tau_i |
| layer residuals around the profile | none | relative rms about 4-10% in the middle layers at 100 GeV, up to 25% or more at the ends | not found |

### 8.2 Proposed model `em_production` (hierarchical; each stage frozen before the next; nothing here is fitted to classifier AUC)

1. **Mean profile and leakage (stage 1).** Mean `<ln T>(E)` and `<ln alpha>(E)` as smooth functions of `ln y` with free coefficients fitted to the Geant4 profile fits, replacing the fixed beta; Grindhammer-Peters forms as the starting structure. First test, before any model code: how much of the first-layer deficit and of the leakage misfit disappears when beta is free; if a per-layer mean correction is still needed it is added as a named, interpolated table (a stated number of parameters), not hidden.
2. **Longitudinal fluctuations (stage 2).** `(ln T, ln alpha)` bivariate with `sigma(ln T)`, `sigma(ln alpha)` and `rho` as functions of `ln y`, starting from the Grindhammer-Peters sampling forms, coefficients calibrated. If the bivariate distribution is measurably non-Gaussian (skewness of ln alpha 0.3-0.6 was reported, unchecked), use a Gaussian copula with empirical marginals, as in the proton model. Plus a layer-level multiplicative residual around the profile, with its measured scale per layer.
3. **Lateral structure (stage 3).** One event-level lateral latent coupled to the longitudinal variables with the measured coupling (not assumed), a depth-dependent core-plus-halo radial law, and cell-level granularity through the same quanta mechanism as the proton lateral model (`proton_lateral.py`), with its own calibration for electrons. Sharing the mechanism is a choice with a consequence: the two classes then differ by calibrated parameters, not by one being smooth and the other not.
4. **Correlations (stage 4).** Layer-to-layer correlation of the residuals through the same Gaussian copula with a common factor and an AR chain used for the protons; checked against the layer-correlation row.

Representation: `deposition` only. The fibre-energy scale is the separate detector-response step R-A.

### 8.3 Stop rule and discipline

A stage is accepted when its contract rows reach the registered tolerances on the calibration data; it is then frozen. A stage that cannot reach them with a smooth model is recorded as a residual, not tuned further. No parameter is added without a named meaning and a measured need. The sealed electron set is opened once, at the end, through `sealed_set.open_set`.

### 8.4 Decisions needed to proceed

- **F1.** Approve the structural prior (Grindhammer-Peters forms with coefficients calibrated to Geant4, copula if the marginals are non-Gaussian) for stages 1-2.
- **F2.** Reuse the proton quanta mechanism for the electron lateral structure (stage 3), or implement the Grindhammer-Peters spot model.
- **F3.** `deposition` only for `em_production` (the detector response later), as proposed.
- **F4.** Whether a per-layer mean correction table (named parameters) is acceptable if the free-beta gamma profile does not remove the first-layer deficit.
- **F5.** Approve the first implementation step as ANALYSIS only: the stage-1 test (free beta against the first-layer deficit and the leakage), with no generator code.
