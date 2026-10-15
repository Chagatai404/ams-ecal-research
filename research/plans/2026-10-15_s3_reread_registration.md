# S3 re-read of the existing Geant4 evidence: registration of stage 1 (2026-10-15, written BEFORE any statistic was computed)

Status: exploratory development reading on exposed data, with valid tools. Nothing here is confirmatory; no sealed set is read; the pre-registered multiscale test and the DECIDE-1 amendment are not changed. The researcher started S3 on 2026-10-15 ("start housekeeping and the S3 re-read"); the statistics, strata, number of permutations and multiplicity procedure below are the agent's concrete choices inside the researcher's decisions of 2026-10-15 (B = 10,000; Westfall-Young where joint draws exist, Holm always beside it, BH exploratory), and the researcher may change them before the results are read as anything more than exploratory.

## 1. Question of stage 1

The development run reported 192 KS verdicts (24 multiscale statistics x 2 representations x 4 energies, electrons against truth-interacting protons), 79 "material" in the large run, with no multiplicity control, bootstrap intervals with ties, a floor and a verdict that were different statistics, and some verdicts resting on about 30 electrons (review items C4, C5, C6, C13). **Which of those 192 differences survive a permutation test within occupancy strata, restricted to events eligible for the statistic, with family-wise control?** The old verdicts are the comparison, not the target.

## 2. Data and statistics (fixed here)

- Electrons (`electron_development_large`, 10,000 per energy) against the named population `interacting` of `proton_development_large` (truth first interaction occurred, about 9,700 per energy), at 10, 20, 50 and 100 GeV, both `deposit_grid_mev` and `readout_grid_mev`. No event is held out (policy NONE for these sets).
- 24 statistics per event: D0, D1, D2 and D0 - D2 over the six registered windows (`multiscale_test.multiscale_table`); NaN (ineligible) where the event has fewer occupied cells than the window's registered minimum (336 or 572). NaN events are not dropped silently: they are counted and the test is run on the finite events ("eligible in both" is the finite events of each class within the statistic).
- A test is **not testable** (listed, not analysed) when fewer than 100 finite events exist in either class (`missingness_controls.MIN_ELIGIBLE_PER_CLASS`).
- Strata: five equal-count quintiles of the number of occupied cells of the event's own grid (pooled electrons and interacting protons, all events, computed once per energy and representation, independent of the labels), as in the development reading. Strata with fewer than 20 finite events in either class are left out of the statistic (`MIN_PER_STRATUM`), the rule applied identically to the observed and to every permuted labelling.
- Statistic: stratum-weighted Kolmogorov-Smirnov distance (`permutation_nulls.stratum_ks`: KS per stratum, weights min(n1, n0)).
- Null: 10,000 label permutations within the strata (labels shuffled among all events of a stratum, then the statistic computed on the finite events), p = (1 + count) / (1 + 10,000). The 95th percentile of the null statistics is the floor of that test; "excess" = observed - floor.

## 3. Multiplicity (the researcher's decision applied)

- Joint permutation draws: one labelling per row for all 24 statistics of one (energy, representation) block; blocks are independent samples (different energies) or different representations of the same events. To keep the joint-draw argument simple, the two representations are two **families** (96 tests each: 24 statistics x 4 energies), each with its own permutations; rows of different energies are paired arbitrarily (independent blocks).
- Within each family: Westfall-Young step-down maxT adjusted p-values (`westfall_young_stepdown`) and Holm adjusted p-values, both reported. Across the two families: the adjusted p-value is multiplied by 2 (Bonferroni), a conservative statement about "any of the 192". Benjamini-Hochberg q-values are reported as an exploratory view.
- **Guard for a Westfall-Young claim:** the family-wise error of these statistics must have been checked on same-class data (electron halves, at least 1,000 trials) before any statement beyond "exploratory" rests on the Westfall-Young column; until then the Holm column decides. The check is part of stage 1 and is run after the main re-read.
- Every result of this stage is labelled **exploratory**: no primary is named here (the primaries belong to the DECIDE-1 amendment), so no secondary is promoted.

## 4. Reported

Per test: eligible counts per class, strata used and skipped, observed statistic, floor, excess, p, Westfall-Young and Holm adjusted p, BH q, the old verdict (`material_by_the_registered_rule`), and the agreement table: of the old "material" verdicts how many are significant at adjusted p <= 0.05, and of the old "not material" how many are; the number of tests not testable; sensitivity with the strata taken from the total deposited energy instead of the occupancy (same statistics, same B), reported beside the main result.

## 5. What stage 1 does not do (deferred, with the reason)

- **View-aware readout statistics** need physical box sizes for the alternating x and y views (DECIDE-4, S4): not defined yet.
- **Cut and EMZ variant comparison, hard-subset ladder, cell-shuffle excess of the interacting protons against a smooth baseline (plan item C2)**: stage 2 and 3, each registered before it is computed; the cell-shuffle excess needs the coverage-checked interval already built (`coverage_simulations.shuffle_excess_interval_coverage`).
- Anything on actual AMS performance; all sealed data.

## 6. What would make the result unreadable

A tool bug (the engine is checked against the existing `stratum_ks` on observed and permuted labellings before the run); many tests not testable at 10 GeV and on the readout (expected: eligibility is missingness there, S0 record); the family-wise guard failing (then only Holm is reported).

## 7. Addendum written 2026-10-15 after the main run and before the sensitivity runs

The main run (occupancy strata) finished first: 64 of 192 tests are testable, all 64 significant, and all of them were already "material" in the development run (results `results/multiscale_controls/s3_reread_stage1.json`). Before the registered energy-strata sensitivity is computed, one further sensitivity is added because the DECIDE-1 draft's P1 primary matches on occupancy AND energy jointly: strata = the 25 cells of five equal-count quintiles of occupied cells x five equal-count quintiles of total deposited energy (each from the pooled events of the block), same statistics, same 10,000 permutations, same rules. The question it answers: do the testable differences persist when electrons and interacting protons are matched on both. Reported beside the main result; nothing in the main result is changed by it.
