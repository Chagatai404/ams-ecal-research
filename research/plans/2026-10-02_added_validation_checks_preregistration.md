# Added validation checks — PRE-REGISTRATION (numbers CONFIRMED 2026-10-02)

_Drafted 2026-10-02. Status: **registered**; the researcher confirmed all numbers as drafted and approved scikit-learn ("Yes to all", 2026-10-02). DEC-007 accepted the concept (a layer-correlation row, a classifier two-sample test, CaloChallenge-style sparsity, an ungated multiscale panel) and required it to be registered **before any sealed Geant4 event exists**. The numbers below are my proposal; until the researcher confirms or amends them (Q2 of `2026-10-02_fastmc_validated_dataset_and_detector_response_plan.md`) no sealed set is generated._

These rows are **added to** the original contract (`2026-09-29_proton_dependency_analysis.md` §8), which is unchanged: interaction, crossing, energy, longitudinal, lateral, activity and correlation rows keep their stated thresholds (KS >= 0.10 with p < 0.01, or a 5/16/50/84/95% quantile off by > 10% and outside the bootstrap 95% interval, correlations differ by >= 0.15 and outside the bootstrap interval).

## 0. Rules that apply to every row below

1. **Population.** The sealed sets of the plan (section 4, B2) only. Per energy, the FastMC sample is drawn with the same entry-point rule and has at least 20,000 events (the classifier test uses a size-matched design, section 2).
2. **One opening.** The sealed files are opened once, through the guarded reader, with the harness frozen at a committed hash. A threshold changed after a result has been seen is a **new dated registration**, and the set it was changed on counts as consumed.
3. **Harness dry run first.** The whole harness is run beforehand on **stand-ins**: the pilot's calibration events (never the held-out events, never the sealed events) against FastMC. Its purpose is to catch bugs, not to tune.
4. **Noise floor.** Every comparison also reports the same statistic between the pilot calibration events and the sealed events (a sample from the same Geant4 configuration), which no model could be expected to beat.
5. **No score.** Each row passes or fails separately, per energy, per class (crossing or interacting, by the Geant4 truth flag and the model's own status), per representation. About 5-8% of rows are expected to flag by chance under a correct model (measured on the interaction row in 2026-09-30); a verdict is read collectively, never from one flag.
6. **Effect sizes, not only p-values.** With 20,000 model events a trivial difference is significant; each row needs an effect-size criterion *and* an interval criterion.

## 1. Layer-correlation row

Statistic: for each class, representation and energy, the Spearman matrix of the 18 layer energies; summarised by the **lag-averaged correlation** rho_bar(d) = mean over layer pairs with |i - j| = d, for d = 1, 2, 3, 5, 8, 12.
Material iff, for any listed lag, |rho_bar_model - rho_bar_Geant4| >= **0.10** and the model value lies outside the 95% bootstrap interval (400 resamples of Geant4 events) of the Geant4 value. The single largest |delta rho| over all pairs is reported, not gated.
Why: the crossing branch failed because layers were drawn independently; this row measures that directly and is also the row the burst latent and the interacting model's residual correlation must pass.

## 2. Classifier two-sample test

Question: can a classifier tell FastMC events from Geant4 events of the same class, energy and representation?
- **Design.** Equal numbers of FastMC and Geant4 events per energy and class (all sealed events of that cell, FastMC matched in number and entry rule); 5-fold stratified cross-validation; area under the ROC curve (AUC) from the held-out folds; a null distribution from 200 label permutations.
- **Features (declared now, fixed).** Event energy; hit-cell count, max-cell fraction, containment (as in the pilot); longitudinal centre, RMS, maximum layer; the 18 layer energies (log1p); lateral width; core fraction; number of occupied cells per layer (18). The 18 x 72 image itself is **not** a feature in this registration (a convolutional discriminator is a later, separately registered check).
- **Classifier.** Primary: gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`, fixed hyper-parameters, no tuning). **scikit-learn is not currently a dependency**; adding it to a `validation` dependency group needs the researcher's approval (Q2b). If refused, the fallback registered here is L2-regularised logistic regression on the standardised features plus all pairwise products, fitted with `scipy.optimize` (weaker, deterministic, no new dependency).
- **Material iff** cross-validated AUC >= **0.60** and above the 99th percentile of the permutation null. AUC 0.5 is indistinguishable; the 0.60 is an effect size chosen so that a classifier that needs a real structural difference to reach it counts, and a tiny statistical wobble does not.
- **Reading.** A high AUC identifies *that* the generators differ, not *what* differs; the feature importances and the other rows say what. It is also the row that tests the plan's section 5 warning (electron/proton simulator asymmetry): run on electrons and on protons separately, each against its own Geant4, so a pass means each class is close to its own transport, not that the classes are fairly matched to one another.

## 3. CaloChallenge-style standardised sparsity

Statistic: for each layer l and threshold t in {0.1, 0.5, 1.0} x the pilot MIP cell scale (0.56 MeV, so t = 0.056, 0.28, 0.56 MeV; the 0.28 MeV value is the pilot's own cell threshold), the occupancy f_l(t) = P(cell energy > t), per class, energy and representation; and the occupancy against layer-energy decile.
Material iff the layer-averaged occupancy at any t differs by >= **0.02** absolute and the model value lies outside the Wilson 95% interval of the Geant4 value. Per-layer occupancy curves are reported, not gated.

## 4. Multiscale panel — UNGATED, report only

D0, D1, D2 on the combined 18 x 72 image (DEC-012: combined is primary), box sides 1, 2, 3, 6, 9, 18 (DEC-013), only over windows that `ams_ecal.multiscale.reliable_windows` marks reliable for the event's hit count. FastMC versus Geant4 distributions per energy and class. **No verdict and no threshold.** It is labelled exploratory: the estimator-validity gate found the estimators usable on 2 of 13 windows at 1000 hits and 4 of 14 at 3000, and FastMC is a smooth control that is not required to reproduce multiscale structure (DEC-010). A difference here is a measurement for the control experiment, not a failure of the dataset.

## 5. What is not in this registration

- Dataset-level checks (energy/geometry matching between classes, seed isolation, split integrity, label leakage) get their own registration when the dataset is assembled.
- Detector-response rows: after the evidence pass.
- A learned image discriminator.

## 6. Questions for the researcher

- **Q2.** Confirm or amend: the lag set and 0.10 (section 1); AUC 0.60 and the 99th-percentile null (section 2); occupancy 0.02 and the three thresholds (section 3).
- **Q2b.** May `scikit-learn` be added as a `validation` dependency group, or should the scipy logistic-regression fallback be primary?
