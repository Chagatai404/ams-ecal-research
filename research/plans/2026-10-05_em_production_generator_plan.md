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
| independent verification of implementation-affecting claims | PARTLY DONE: Grindhammer-Peters verified from the full text; AMS-02 ECAL geometry verified; fluctuation numbers for THIS detector and rear-leakage fluctuations NOT found; adversarial counterevidence pass not done |
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
