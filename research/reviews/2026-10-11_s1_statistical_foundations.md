# Slice S1 — statistical foundations as tested code (2026-10-11)

_Evidence record, not a plan. The single plan is `research/RESEARCH_PLAN.md`; section 18.4 defines S1 and section 18.3 the gates SV-1..SV-8. Code only: no registered number, no existing ladder or test module and no sealed set was touched. Nothing here is an accepted decision._

## 1. What was built (package `src/ams_ecal/inference/`, tests in `tests/inference/`, 56 tests)

| gate | module | what it does | replaces / answers |
|---|---|---|---|
| SV-1, SV-2 | `rejection_limits.py` | exact Clopper-Pearson limits; rejection as a censored lower limit when k = 0 (never "N"); `protons_needed`; four data roles with `assert_disjoint_roles`; operating point from calibration electrons as an order statistic with its exact Beta uncertainty, a conservative variant, and `evaluate_at_efficiency` that takes calibration electrons, fresh electrons and fresh protons as separate arguments | C1 (zero-failure floor 1/N), C2 (threshold on the evaluated events) |
| SV-3, SV-7 | `cross_fitting.py` | repeated cross-fits with the seed spread; one model trained on training events only and scored on events it never saw; repeated random role splits; energy-only and logistic baselines; the same-class null | C7 (single fold seed), plan SV-7 |
| SV-4 | `missingness_controls.py` | indicator-only separation, explicit missing-indicator columns, eligible-in-both matching on occupied cells, "not testable" cells | C3, C4, C11 (NaN as a feature) |
| SV-5 | `permutation_nulls.py` | label permutation within strata with p = (1 + count) / (1 + B); stratified KS that reports strata and events used; matched-size split floors; the cell-shuffle null with a paired per-event excess | C4, C5, C6, plan item A3 (never built before) |
| SV-6 | `multiplicity.py` | Benjamini-Hochberg, Holm, Westfall-Young step-down, and `decide`: one named primary per family, Holm across families, BH inside, secondaries exploratory unless their family's primary is confirmed | C13 (192 unadjusted tests) |
| SV-8 | `claim_specification.py` | the estimand -> null -> statistic -> uncertainty -> assumptions -> acceptance -> failure template; refuses empty fields, an unknown family and an unnamed primary; a digest to make later edits visible | the review's claim template |
| validation | `coverage_simulations.py` | coverage and type-I-error simulations; `python -m ams_ecal.inference.coverage_simulations` writes `results/inference/coverage.json` | the plan's S1 validation test |

SV-9 (independent review of the frozen analysis code) and SV-10 (63-bit seeds for new sets, seed intersection checked before an opening) are not S1 deliverables; SV-10's sample module and manifest were S0.

## 2. Validation results (27 checks, all pass; exposed development electrons and simulated draws; sealed sets not read)

Tolerances were written in the module before the run: coverage must not fall below the stated confidence by more than three Monte Carlo standard errors; a type-I rate must not exceed alpha by more than three.

| check | setting | result |
|---|---|---|
| Clopper-Pearson upper bound covers the true rate | 15 cells: N in {1,000; 9,696; 20,000} x true rate in {0, 1e-5, 1e-4, 1e-3, 1e-2}, 4,000 draws each | lowest coverage 0.955 (required 0.95); includes cells where most draws are k = 0 |
| rejection lower limit is below the true rejection | N in {9,696; 20,000} x rejection in {1e3, 1e4, 1e5}, 4,000 draws | lowest coverage 0.960 |
| permutation test, two random halves of 1,000 development electrons, KS of occupied cells, alpha 0.05, 200 trials x 199 permutations | 20 GeV unstratified / stratified; 100 GeV unstratified / stratified | rejection rate 0.060 / 0.045; 0.045 / 0.030 |
| family-wise error, five conventional features tested together, 100 trials x 199 permutations | 20 GeV; 100 GeV | unadjusted minimum p rejects 0.20 and 0.13 (it should reject 0.05); Holm and Westfall-Young 0.08 and 0.06 |

The same simulations, with a planted difference, make the check **fail** (unit test `test_the_permutation_test_holds_its_level_on_null_data_and_the_check_fails_when_a_difference_is_planted`), so the harness can detect a broken tool.

**Honest reading.** The 3-standard-error tolerance is loose at 100-200 trials: the family-wise rates 0.08 and 0.06 pass but sit above the nominal 0.05, and one of two cannot be told from 0.05 with 100 trials (standard error 0.022). The rates are consistent with control, not a proof of it. The registered confirmatory analysis (S8) should repeat this with at least 1,000 trials per cell before any family-wise claim is relied on. The halves are taken from one finite file without replacement, where a permutation test is exact in principle, so these runs test the implementation (labels, strata, +1 rule, Holm and Westfall-Young code), not the exchangeability assumption between independent Geant4 sets; that assumption is tested by the cross-set same-class nulls of the S0 audit (0.483 to 0.515).

## 3. What S1 does not do yet

- The old modules (`multiscale_ladder`, `multiscale_test`, `multiscale_gtrain_development`, `added_checks`) are unchanged and keep their known problems; they are migrated to these tools when S3 re-reads the existing Geant4 evidence, so the earlier development numbers are neither restated nor invalidated here beyond the S0 record (zero-failure floors are lower limits of about N/3).
- No estimator on the 3D voxels (A4) and no multiscale control images on realistic amplitudes: S4.
- The independence assumption (each Geant4 event has its own seed) holds within a set; sets that share base seeds by design (the paired physics variants) are paired samples and need paired tests: not covered by `permutation_pvalue`, which treats the two groups as independent.
- The sample-size table assumes independent protons from one population and a threshold fixed independently of them.

## 4. For the researcher
Nothing in S1 needs an approval to exist; using it needs two choices before S3 starts: (1) the number of permutations and cross-fit seeds to register (the code defaults are 1,000 permutations and 5 seeds); (2) whether secondary statistics are tested with Westfall-Young (needs the permutation draws of all statistics at once, more compute) or Benjamini-Hochberg (cheaper, controls the false-discovery share).
