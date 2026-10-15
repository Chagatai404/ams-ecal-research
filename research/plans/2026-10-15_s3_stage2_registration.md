# S3 stage 2: cell-shuffle excess of the multiscale statistics, electrons against interacting protons — registration (2026-10-15, written BEFORE any excess was computed)

Status: exploratory, exposed development data, no sealed set. The researcher approved this stage on 2026-10-15 ("yes" to the cell-shuffle excess as the next S3 stage). It follows stage 1 (`research/reviews/2026-10-15_s3_stage1_reread.md`), which showed that electrons and interacting protons differ in 64 testable multiscale statistics, and asks whether those differences reflect structure beyond what a scrambled image of the same event would show.

## 1. Question

For each event, how far does each multiscale statistic of the real image lie from the same statistic of images with the same cells scrambled while energy is preserved? And does that departure differ between electrons and interacting protons? The first part is plan item A3 (the energy-preserving cell-shuffle null); the second is plan item C2 (the proton look against a matched smooth baseline).

## 2. Surrogates (fixed here)

- **Full-layer shuffle (the registered A3 null):** the cells of every layer are permuted among all 72 positions of that layer, independently per layer. Preserved: per-layer energy, per-layer multiset of cell values, per-layer number of occupied cells (so the window eligibility of the event is unchanged). Destroyed: where in the layer the energy sits, hence the lateral envelope of the shower and any fine structure.
- **9-cell-block shuffle (added because the full shuffle also destroys the lateral envelope, which the conventional width features carry):** each layer is cut into 8 blocks of 9 adjacent cells and the cells are permuted inside each block. Preserved: everything above, plus the energy of every 9-cell block, so the envelope at scales of 9 cells and more and the box counts at box sides 9 and 18. Destroyed: the arrangement below 9 cells. Boxes of sides 1, 2, 3 and 6 change; so a difference seen only with the block shuffle is a difference in sub-9-cell organisation.
- 10 shuffled copies per event and surrogate; the shuffled statistic is the mean over the 10 copies.

## 3. Quantities and tests (fixed here)

- Data, populations, strata, statistics and eligibility exactly as stage 1: electrons (all) against truth-interacting protons, 10, 20, 50, 100 GeV, deposit and readout, the 24 statistics of `multiscale_test.multiscale_table`; events with a NaN statistic are not used for that statistic; a test needs at least 100 finite events per class (the testable set is the 64 of stage 1, since eligibility depends on occupied cells, which the shuffles preserve).
- **Event-level excess** = (real statistic) - (mean over the 10 copies of the same statistic of the shuffled image). One value per event, statistic and surrogate.
- **Estimate (per class, statistic, surrogate):** the median excess with a 95% interval from 2,000 event-bootstrap draws (the interval whose coverage is checked by `shuffle_excess_interval_coverage`; reported, not a test).
- **Test of the class difference of excess:** the stage-1 engine applied to the event-level excess values: stratum-weighted KS between electron and proton excess distributions inside five occupancy strata, 10,000 label permutations within strata, p = (1 + count) / (1 + 10,000). Multiplicity: Westfall-Young and Holm inside each (surrogate, representation) family, then Bonferroni across the four families; Benjamini-Hochberg exploratory. Everything is labelled exploratory.
- **Guard:** the family-wise error of this excess statistic must be checked on same-class data (electron halves, at least 1,000 trials, 999 permutations, tolerance 0.05 + 3 Monte Carlo standard errors) before any Westfall-Young statement; the stage-1 guard covers the raw statistics, not the excess.

## 4. Reported

Per test: finite counts, median excess per class with its interval, the observed KS of the excess distributions, the null floor, p, adjusted p. Plus: the share of statistics for which the median excess of a class is distinguishable from zero (interval excludes 0) for each surrogate; the share for which the electron and proton excess differ; and whether the answer changes between the two surrogates (full against block).

## 5. Reading rules written in advance

- A nonzero median excess says the real image departs from its scrambled version; it is not evidence of fractal or multifractal structure, only of non-random arrangement. Compact shower geometry produces a large excess under the full shuffle for both classes.
- A class difference under the full shuffle but not under the block shuffle means the difference lies in the lateral envelope (width), which conventional features carry; a difference that persists under the block shuffle is a difference in sub-9-cell organisation and is the candidate for finite-scale heterogeneity. Neither is a claim about scaling laws: the estimator controls (S4) are not met.
- If the excess distributions of the two classes are separated only in strata with thin common support, the test is reported as limited by overlap.

## 6. What this stage does not do

Variants, hard-subset ladder, view-aware statistics, anything on the 3D fine voxels (S4), anything on actual AMS performance, all sealed data.
