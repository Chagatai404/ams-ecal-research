# Dataset-level checks — PRE-REGISTRATION (numbers PROPOSED, awaiting the researcher)

_Drafted 2026-10-04 on branch `fastmc-validated-dataset`, before any dataset has been assembled. The added generator-validation checks have their own registration (`2026-10-02_added_validation_checks_preregistration.md`, section 5 deferred these). Every number below is my proposal; the first dataset assembled before the researcher confirms them is labelled DEVELOPMENT and is not called validated._

## 0. What is being checked

The dataset is a labelled mixture of electron and proton events from the two FastMC generators, in one representation (`deposition` for version v0, the perfect-event dataset; DEC-006). These checks ask whether the DATASET, not each generator, can mislead a downstream user: through the sampling design, through leaked seeds or splits, or through a label that is predictable from something other than the physics.

## 1. Design that the checks assume (fixed in `ams_ecal.validation.dataset`, recorded in the dataset provenance)

- Energy: log-uniform over 10-100 GeV, **the same distribution for both classes**, drawn from a stream that does not depend on the class.
- Entry point: uniform over the illuminated cell (0-9 mm in x and y), normal incidence, the same for both classes.
- Seeds: one stream of seeds per dataset; **no seed appears twice anywhere**, in particular not in two splits or in both classes. (Two classes sharing a seed would share their first random variates, a hidden correlation between labels.)
- Splits: train / validation / test of 70 / 15 / 15 per class, assigned from a seeded permutation of the events of each class, fixed before generation. Exact class balance in every split.
- Classes are interacting and crossing protons as the proton model draws them; the interacting/crossing status is recorded as a provenance column and is NOT a label.

## 2. Rows

| row | statistic | material iff |
|---|---|---|
| D1 energy identity | two-sample KS of the primary energies, electron vs proton | KS >= 0.05 |
| D2 entry identity | the same, for entry x and for entry y | KS >= 0.05 in either |
| D3 seed isolation | counts of repeated seeds within a class, across splits, across classes | any repeat |
| D4 split integrity | each event in exactly one split; split sizes equal the design; exact class balance per split; KS of energies between every pair of splits | any violation, or KS >= 0.05 |
| D5 nuisance leakage | cross-validated AUC of the classifier of section 2 of the generator registration (same settings, 200 permutations) on **[energy, entry x, entry y] only**, electron vs proton | AUC >= 0.55 and above the permutation null's 99th percentile |
| D6 validity | every value finite and non-negative; array shapes; events with zero total energy counted | any non-finite or negative value; zero-energy events reported, not gated |
| D7 provenance completeness | every event has class, seed, energy, entry, split, model version, artifact hash; protons have interaction status and depth | any missing field |
| D8 regeneration | 100 events chosen at random are regenerated from their own provenance row alone | any grid not bit-identical |

## 3. Reported, NOT gated: generator-native separability

D9. The classifier of the generator registration (section 2 features, fixed settings) trained on electron versus proton events of the dataset. **No threshold.** Reason: the electron generator is smooth by design (DEC-001) and the proton generator is not (bursts, spill, quanta, sparsity). A classifier is expected to separate them, perhaps perfectly. That is the asymmetry flagged in the plan (section 5, decision Q4), a statement about the generators, not a defect to be fixed by tuning this dataset. It is reported and the dataset README repeats it: a researcher who trains a model on this dataset must not read its accuracy as physics.

## 4. Rules

1. The rows are computed on the assembled dataset exactly once per assembly; a threshold changed after a result has been seen is a new dated registration.
2. The dataset is generated from a clean commit with the artifact hash recorded; D8 re-runs from that commit.
3. No sealed Geant4 file is read by any dataset step.
4. A flag is reported as such. D1-D4 and D6-D8 are consistency checks of code: a flag is a bug to fix before the dataset is used, and the fixed dataset is a new assembly.

## 5. Questions for the researcher

- **Q8.** Confirm or amend the thresholds KS 0.05 (D1, D2, D4) and AUC 0.55 (D5).
- **Q9.** Confirm that v0 is `deposition` only for both classes, 10-100 GeV log-uniform, normal incidence, 70/15/15, and that D9 is reported rather than gated.

## Amendment 2026-10-05 (Q8, Q9 answered in part)

- **AUC 0.55 (D5) is confirmed**, to be reported with a bootstrap or seed-to-seed uncertainty, not as a single point estimate.
- **KS 0.05 (D1, D2, D4) is not confirmed as a universal threshold.** D1, D2 and D4 compare samples drawn by the same code from the same distribution, so there KS 0.05 is a consistency check of the sampling code at the planned sample sizes. A KS threshold used to compare a generator with Geant4 must instead come from a split-Geant4 self-comparison noise floor plus physics-motivated tolerances per observable class, registered before any sealed opening and not adjusted to what the generators are observed to give.
- Q9 (v0 is `deposition` only, 10-100 GeV log-uniform, normal incidence, 70/15/15, D9 reported rather than gated) is not yet answered.
- The electron generator in the dataset is the new production EM generator once it exists (decision of 2026-10-05); the smooth DEC-001 generator is a separate null and is not mixed into the e/p dataset.
