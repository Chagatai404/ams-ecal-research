# Pre-registration of the first multiscale test (numbers CONFIRMED 2026-10-09 by the researcher, "for now"; to be reviewed after first use)

_Drafted 2026-10-09 (phase 2 of `2026-10-09_multiscale_test_dataset_plan.md`). Status: **CONFIRMED FOR NOW.** The researcher approved the plan ("Yes to all five") and, after the numbers were explained, confirmed them on 2026-10-09: "These seem reasonable I confirm all for now, we will see how useful they prove after". The confirmation is provisional by their words: how useful the numbers prove is reviewed after the first use, and a change is made only as a recorded amendment BEFORE the final opening, never after it. Nothing below has been run on the registered data. Every number is fixed in code before G-train is analysed and none is changed afterwards._

Reads with the literature record (`2026-10-09_multiscale_literature_verification_record.md`), the estimator pre-registration and controls (`2026-10-02_*`), and the eligibility result (plan section 10).

## 1. Questions, in the order they are answered

* **Q-A (measurement):** On which events and scale windows are the estimators interpretable? (answered by the eligibility rule and the controls, not by showers)
* **Q-B (difference):** Within eligible, occupancy-matched events, do electrons and interacting protons differ in D0, D1, D2 or D0 - D2, beyond the Geant4 split-sample floor?
* **Q-C (survival):** Is the difference present at the fine scale and at the 18 x 72 deposition and readout grids?
* **Q-D (information):** Do the multiscale features add classification information beyond the conventional features, and does it change the sample efficiency?

Negative outcomes are reported in the same detail as positive ones (the roadmap's list of publishable outcomes). H-001 is a hypothesis; the Anchordoqui et al. result (D_q about equal for Geant4 EM showers) is a counter-indication that the design must be able to confirm.

## 2. Data roles

Development: the exposed sets and G-train (never reported as validation). The registered test: G-final = the sealed e/p sets, **opened once** through `sealed_set.open_set`. G-syst and the existing proton variants: systematics. FastMC (F-v0) is a separate arm and is not part of Q-A to Q-D.

## 3. Eligibility (Q-A), from the controls

* An event is **eligible for the six-scale window set** if its occupied cells (any energy above zero) are at least **336** (the median occupied cells of the 1000-hit control images) and for the **four 3000-hit windows** if at least **572**. The windows are the ones the controls found USABLE at those hit counts (plan section 10).
* Events below 336 cells are **reported as ineligible, never dropped silently**; the eligible fraction per class, energy and representation is a result.
* Proposed addendum before any fine-scale value is read: **re-run the estimator controls on the fibre / fine-voxel geometry** (the same rules as DEC-013: false positives at most 10%, detection power at least 90%, ordering fraction 0.90 as the interpretation awaiting confirmation) and set the fine-scale eligibility from that run.

## 4. Matching (Q-B)

* Classes: electrons against **truth-interacting** protons (truth is used only to define the stratum; the crossing half, with about 20 cells, is reported separately and is never mixed in).
* Same energy grid; within each energy, five equal-count strata of occupied cells and of total deposited energy (matched by reweighting, reported with and without matching).
* Statistic: the KS distance between the class distributions of each D (D0, D1, D2, D0 - D2) per window, with a 95% bootstrap interval over events; the floor is the KS distance between two random halves of the same class (the Geant4 split-sample floor), and the null control is the energy-preserving shuffle of cells within layers.
* **Materiality (proposed):** KS at least **0.10** (the project's proton-validation materiality) AND above the split-sample floor by more than its bootstrap interval AND the sign unchanged between the baseline sample and the extended / systematics samples.

## 5. Survival (Q-C)

The same statistic and rule on (i) the fine-scale deposits, (ii) the 18 x 72 deposition grid, (iii) the 18 x 72 readout grid, for the same events. "Survives" means the materiality rule holds in (iii); "destroyed by segmentation" means it holds in (i) and not in (iii); "absent" means it holds in none. The fine-scale rows are gated by the addendum of section 3.

## 6. Information (Q-D)

* Classifier: the registered fixed gradient-boosting classifier (`ams_ecal.validation.added_checks`), five-fold cross-validated AUC, electrons against all protons AND against interacting protons; the multiscale features are computed on eligible events only, with a missing-value indicator otherwise.
* Ladder: (1) conventional features only: total deposited energy, layer profile and centre of gravity, width, containment, occupied cells, largest cell fraction; (2) multiscale features only; (3) conventional + multiscale; (4) raw 18 x 72 grid with the same classifier; (5) raw + multiscale.
* **Added information:** the AUC gain (3) minus (1) is at least **0.01** with the lower bound of its 95% paired-bootstrap interval above 0, at the same training size and the same events. The classifier AUC is reported as an outcome; it is never a tuning target.
* **Proton rejection at fixed electron efficiency (added at the researcher's request, 2026-10-09):** for every rung of the ladder, the proton rejection (1 / false-positive rate) at **90% and 99% electron efficiency**, with 95% paired-bootstrap intervals, because the AUC may leave no headroom if the conventional features already separate the classes; the same 0.01-AUC rule stays, and a rejection gain is reported next to it (no separate threshold is set: it is reported, not used to adopt).
* Sample efficiency: AUC against training size per class (100, 300, 1000, 3000, 10000), curves for (1), (3), (4), (5).

## 7. Controls and systematics

Smooth null (`em_smooth_null` through the same pipeline and the controls' smooth baseline); the cell shuffle; G-syst (alternative EM lists `FTFP_BERT_EMZ` and `FTFP_BERT_EMV`, reached through the physics-list factory; 0.1 mm cut). The earlier `FTFP_BERT_TRV` choice was withdrawn: TRV and ATL change the hadronic handover only and the existing proton variants (QBBC, QGSP_BERT, 0.1 mm cut): per observable, the sign and the KS change; a result that changes sign under a variant is reported as systematics-dominated. Normal incidence only, stated with every result.

## 8. Opening discipline

G-final is opened once; the thresholds above, the windows, the strata and the code are frozen in a tagged commit before; the opening ledger records the purpose; no number is changed after.

## 9. Numbers (confirmed for now, 2026-10-09)

336 and 572 cells (eligibility); KS 0.10; five strata; ladder AUC gain 0.01; bootstrap 95%; the training sizes; the 0.90 ordering fraction for the fine-scale controls (still an interpretation).
