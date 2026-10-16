# S3 stage 2: cell-shuffle excess of the multiscale statistics, electrons against interacting protons (2026-10-15) — EXPLORATORY

Registration (written before any excess was computed): `research/plans/2026-10-15_s3_stage2_registration.md`. Code `src/ams_ecal/multiscale_estimators/multiscale_shuffle_excess.py` (surrogate invariants and excess behaviour tested, `tests/multiscale_estimators/test_multiscale_shuffle_excess.py`). Results `results/multiscale_controls/s3_stage2_shuffle_excess_full.json` and `..._block9.json`. Data and tests as stage 1 (electrons against truth-interacting protons, 10, 20, 50, 100 GeV, deposit and readout, the 64 testable statistics, occupancy strata, 10,000 permutations); event excess = real statistic minus the mean over 10 shuffled copies; two surrogates: `full` (cells permuted over the whole layer, the A3 null) and `block9` (cells permuted inside 9-cell blocks, which keeps the lateral envelope above 9 cells). Multiplicity: Holm and Westfall-Young inside each (surrogate, representation) family, Bonferroni across the four families. Exposed development data; no sealed set read.

## 1. Result

| surrogate | testable | class difference significant, Holm (4 families) | Westfall-Young (4 families) | electron median excess excludes 0 | proton median excess excludes 0 | KS of the excess distributions (median) | p at the floor |
|---|---|---|---|---|---|---|---|
| full-layer shuffle | 64 | 64 | 64 | 63 | 63 | 0.34 | 97% |
| 9-cell-block shuffle | 64 | 61 | 62 | 64 | 63 | 0.31 | 94% |

Established (data):
1. **Both classes depart from their scrambled images in essentially every statistic** (the median excess excludes zero for 63 or 64 of 64 in each class and surrogate). Real showers are not scrambled images; this is expected and says nothing about scaling.
2. **The two classes depart differently.** Under the full shuffle all 64 class differences of the excess distribution are significant; under the 9-cell-block shuffle 61 (Holm) or 62 (Westfall-Young). The three that are not significant under the block shuffle are all 100 GeV deposit statistics (`D2@1-2-3-6-9`, p adjusted 0.065; `D2@2-3`; `D0-D2@2-3`), with observed KS 0.035 to 0.068 against a floor of 0.053.
3. **Most of the excess is the lateral envelope, and the class difference is not.** The median absolute excess under the block shuffle is about one third of that under the full shuffle (electrons 0.33, protons 0.30), but the KS between the two classes' excess distributions does not shrink (median ratio block / full 1.15; median KS 0.31 against 0.34). For some statistics the sign of the excess reverses between the surrogates (for example D2 of the widest window at 100 GeV deposit: electrons +0.31 under the full shuffle, -0.10 under the block shuffle; protons +0.26 and -0.10).
4. The proton excess is smaller in size than the electron excess (median ratio of absolute medians 0.83 full, 0.77 block).

Reading, by the rules written in advance (registration section 5): a class difference that persists under the block shuffle is a difference in the organisation below 9 cells, between electrons and interacting protons, in 61 or 62 of the 64 testable statistics. It is a candidate for finite-scale heterogeneity, **not** evidence of fractal or multifractal scaling: the estimator controls (S4) are not met, and the statistics are measured at a handful of box sides on a 72-cell layer.

Not established, and why:
- **Occupancy is the only matching.** The strata are the registered occupancy quintiles; electrons and interacting protons of the same primary energy differ strongly in deposited energy (stage 1, and the P1 overlap studies), so part of the class difference of the excess can still reflect energy. The matched design of the amendment draft (equal occupancy and deposited energy, electrons and protons of different primary energies) is the test that can answer this and is not yet run on these statistics.
- **Why the sign changes with the surrogate** is not examined; the excess depends on the window and the statistic.
- **The family-wise guard for the excess statistic has passed** (section 3 below, run 2026-10-16), so the Westfall-Young column (62 of 64 under the block shuffle, 64 under the full shuffle) may be read beside Holm (61 and 64).
- The three non-significant block-shuffle tests are at 100 GeV deposit with small effects; with the 10 shuffles per event their significance is within the resolution of the design, not a finding of "no difference".

## 2. Next steps this result points to (proposals; each registered before it is computed)

1. The matched-pair comparison of the amendment (electrons 20 GeV against protons 100 GeV, 136 to 167 pairs; electrons 50 GeV against protons 200 GeV needs the larger proton production) on the raw statistics and on the excess, to separate structure from energy.
2. The guard for the excess statistic (electron halves, 1,000 trials).
3. The variants (cut, EMZ) and the hard-subset stages, then view-aware statistics after DECIDE-4.

## 3. Guard for the excess statistic (stage-2 registration section 3): PASSED (2026-10-16)

`research/reviews/2026-10-15_s3_stage2_excess_guard.py` and `.json`: electrons of one energy split into two random groups of 2,000 (all statistics true nulls), the event-level excess tables of both surrogates tested with the stage-1 block test (occupancy strata, joint label permutations), 1,000 trials and 999 permutations per block; tolerance written before the run 0.05 + 3 Monte Carlo standard errors = 0.0707.

| surrogate | energy, representation | usable statistics | Westfall-Young family-wise error | Holm |
|---|---|---|---|---|
| full | 20 GeV deposit | 8 | 0.054 | 0.040 |
| full | 20 GeV readout | 0 | not testable | - |
| full | 100 GeV deposit | 24 | 0.042 | 0.029 |
| full | 100 GeV readout | 8 | 0.047 | 0.035 |
| block9 | 20 GeV deposit | 8 | 0.046 | 0.016 |
| block9 | 20 GeV readout | 0 | not testable | - |
| block9 | 100 GeV deposit | 24 | 0.049 | 0.027 |
| block9 | 100 GeV readout | 8 | 0.059 | 0.021 |

All six testable blocks are inside the tolerance (Westfall-Young 0.042 to 0.059, Holm 0.016 to 0.040). One standard error of a rate near 0.05 is 0.007, so 0.059 is not distinguishable from 0.05. The guard covers electron features and occupancy strata only (a same-class null); it does not cover other statistics, populations or the matched-pair design (guarded separately in stage 3).

