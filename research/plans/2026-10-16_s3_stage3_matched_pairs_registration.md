# S3 stage 3: electrons against interacting protons matched on occupied cells and deposited energy — registration (2026-10-16, written BEFORE any matched statistic was computed)

Status: exploratory, exposed development data, no sealed set. The researcher asked for this stage on 2026-10-15/16 ("yes, run the guard and register the matched-pair stage") and decided that the matching of the amendment draft (`plans/2026-10-12_s2_hypotheses_and_amendment_DRAFT.md` section 7.2) is kept. Stage 1 and 2 matched on occupancy only (stage 1: also on energy and jointly, with 1-2 populated cells); this stage uses the matching that the design studies of 2026-10-15 (`reviews/2026-10-15_p1_*_study.*`) found workable: electrons and protons of **different primary energies**, paired so that their deposited energy and occupancy overlap.

## 1. Question

Among electrons and interacting protons that have (nearly) the same number of occupied cells and (nearly) the same deposited energy, do the multiscale statistics, and their departure from the cell-shuffled images (stage 2), still differ? This is the comparison "at equal occupancy and equal deposited energy" that P1 of the amendment draft names. A difference here is not explained by occupancy or deposited energy; it is also a statement about a thin matched subsample of each class, not about whole showers.

## 2. Pairings (fixed here, from the development overlap study)

Only the pairings that gave at least 100 caliper pairs (0.25 sd) in `2026-10-15_p1_cross_energy_overlap_study.json` (deposit representation, events with at least 336 occupied cells) are used: **electrons 20 GeV with protons 50 GeV (100 pairs), electrons 20 GeV with protons 100 GeV (136), electrons 50 GeV with protons 100 GeV (110)**. The other pairings are not run (fewer than 100 pairs). **Electrons 50 GeV with protons 200 GeV gave 48 pairs with the existing 2,119 eligible protons and is not testable by the rule**; it waits for the larger 200 GeV production (decided 2026-10-15). The choice of these three pairings was made on exposed development data and is a design choice, never re-made on confirmatory data.

## 3. Matching (fixed here)

- Population: electrons (all) and truth-interacting protons (`interacting`) of the pairing, events with at least 336 occupied cells **in the representation analysed** (deposit or readout; the base-window eligibility), so the matching is done separately for `deposit_grid_mev` and `readout_grid_mev`.
- Keys: the natural logarithms of the number of occupied cells and of the total deposited energy of the representation, each standardized with the pooled mean and standard deviation of the two eligible classes.
- Rule: electrons in the order of increasing distance to their nearest proton (among the 50 nearest); each electron takes its nearest still-unmatched proton within the caliper; 1:1 without replacement. Caliper 0.25 sd (main) and 0.5 sd (sensitivity, reported beside it).
- Number of pairs and covariate balance are computed **before** any multiscale statistic: standardized mean difference (pooled sd) of both log keys after matching must be at most 0.10, otherwise the pairing is reported as "imbalanced". A pairing/representation with fewer than 100 pairs is **not testable**.

## 4. Statistics, test, multiplicity (fixed here)

- Statistics: the 24 multiscale statistics of `multiscale_test.multiscale_table` ("raw"), and their stage-2 excess for both surrogates (`full` and `block9`; read from the stage-2 cache of the same events). For a statistic only the pairs in which **both** members have a finite value are used; a statistic is testable with at least 100 such pairs.
- Test: KS distance between the electron values and the proton values of the valid pairs; null by exchanging the class labels inside pairs (each pair swapped with probability 1/2; 10,000 draws; p = (1 + count) / (1 + 10,000)), exact under exchangeability inside pairs. Estimate: the median of the paired difference (electron - proton) with a 95% bootstrap interval over pairs (2,000 draws).
- Multiplicity: a family is (pairing, representation, statistic type); inside a family Westfall-Young step-down and Holm over its testable statistics with the same pair swaps for all statistics; across families Bonferroni with the number of families that have at least one testable statistic. Benjamini-Hochberg exploratory. Everything is labelled exploratory.
- **Guard (same-class, before any Westfall-Young statement):** electrons of 20 GeV (deposit) are split into two random halves, the halves are matched with the same rule and caliper (pairs capped at 150 to mimic the available number), the 24 raw statistics are tested with the same pair swaps; 1,000 trials, 999 permutations, family-wise error of Westfall-Young and Holm within 0.05 + 3 Monte Carlo standard errors (0.0707). Until it passes, Holm decides.

## 5. Reading rules written in advance

- Only a difference that holds in the matched pairs can be called "beyond occupancy and deposited energy" in these data; whether it holds is a statement about the matched subsample (the electrons that deposit little for their class and the protons that deposit a lot), and the matched fraction of each class is reported.
- A raw-statistic difference that vanishes after matching (and an excess difference that vanishes) says that stage 1 and 2 differences were explained by occupancy and energy. A raw difference that persists but an excess difference that does not (or the reverse) is reported as such; nothing is interpreted as scaling or fractal structure (estimator controls of S4 are not met).
- Results across the three pairings are not independent (electrons 20 GeV appear twice, protons 100 GeV twice); agreement among pairings is reported but not counted as replication.

## 6. What this stage does not do

The 200 GeV pairing, the variants, hard subsets, view-aware statistics, 3D voxels, anything on actual AMS performance, all sealed data.
