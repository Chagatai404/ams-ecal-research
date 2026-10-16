# S3 stage 3: electrons against interacting protons matched on occupied cells and deposited energy (2026-10-16) — EXPLORATORY

Registration (before any matched statistic): `research/plans/2026-10-16_s3_stage3_matched_pairs_registration.md`. Code `src/ams_ecal/multiscale_estimators/multiscale_matched_pairs.py` (tests: `tests/multiscale_estimators/test_multiscale_matched_pairs.py`: matching is 1:1 and inside the caliper, balances the keys, the pair-swap test holds its level under the null and detects a shift). Results `results/multiscale_controls/s3_stage3_matched_pairs.json` and `..._guard.json`. Electrons and truth-interacting protons of different primary energies, paired 1:1 within a caliper on the standardized logarithms of occupied cells and deposited energy; 24 multiscale statistics (raw, and the stage-2 excess for the full-layer and the 9-cell-block shuffle); KS between the matched samples; null by exchanging the classes inside pairs (10,000 draws); Westfall-Young and Holm inside each family, Bonferroni across the 9 testable families. Exposed development data; no sealed set read.

## 1. The matching (computed before any statistic)

| pairing (electron / proton GeV) | representation | pairs, caliper 0.25 | share of eligible electrons / protons | balance (standardized mean difference of log cells, log energy) | pairs, caliper 0.5 (balanced?) |
|---|---|---|---|---|---|
| 20 / 50 | deposit | 100 | 1.1% / 1.3% | -0.035, 0.030 (balanced) | 149 (yes) |
| 20 / 100 | deposit | 136 | 1.5% / 1.7% | -0.060, 0.014 (balanced) | 167 (no) |
| 50 / 100 | deposit | 110 | 1.1% / 1.4% | -0.044, 0.071 (balanced) | 175 (no) |
| any | readout | 0, 0, 58 | - | - | 0, 0, 91 |

Established: the three deposit pairings give at least 100 balanced pairs at the main caliper (the numbers equal the design study's). **On the readout no pairing is testable**: electrons of 20 GeV have only 12 eligible events (336 occupied cells), and 50 / 100 GeV gives 58 pairs (91 at the wide caliper), below the rule of 100. The matched pairs are about 1-2% of each class: electrons that deposit little for their class matched to protons that deposit a lot.

## 2. Result (deposit representation, caliper 0.25; Holm across the 9 families, a statistic is "significant" at adjusted p <= 0.05)

Only 8 of the 24 statistics are testable (the two windows with the 336-cell minimum x 4 statistics): the matched events have at least 336 occupied cells but too few have 572.

| pairing | raw statistics (8) | excess, full shuffle (8) | excess, 9-cell-block shuffle (8) |
|---|---|---|---|
| 20 / 50 GeV (100 pairs) | 4 significant (D1 and D2 and D0-D2 of the window 2-3-6-9, D1 of the widest) | 1 (Westfall-Young 3) | 0 |
| 20 / 100 GeV (136 pairs) | 6 | 5 | 0 |
| 50 / 100 GeV (110 pairs) | 5 | 0 | 0 |

KS between the matched classes for the raw statistics 0.25 to 0.67 (medians 0.22-0.33) against a permutation floor of about 0.17-0.19; paired median differences are small (0.01 to 0.13 in the dimension). D0 (box-counting) is never different; D1, D2 and D0 - D2 are, and their sign changes between the windows (for the widest window D1 and D2 of the electrons are larger than the protons' in two pairings, for the 2-3-6-9 window smaller). The wide-caliper sensitivity gives the same pattern (raw 4-7, full-shuffle excess 5, 5, 0, block-shuffle excess 0), with two of its three pairings not balanced.

Established (data):
1. **Matched on occupied cells and deposited energy, electrons and interacting protons still differ in about half of the testable raw statistics** (4, 6 and 5 of 8), in D1, D2 and D0 - D2 and not in D0.
2. **The sub-9-cell organisation does not differ detectably after matching.** The class difference of the excess under the 9-cell-block shuffle, significant in 61 of 64 occupancy-matched tests in stage 2, is significant in **none** of the 24 matched tests (all three pairings, both calipers). By the rule written in advance (registration section 5) the stage-2 block-shuffle difference is explained by occupancy and deposited energy, as far as a matched subsample of 100-136 pairs can say.
3. **The full-shuffle excess difference is not consistent across pairings** (5 of 8 for 20 / 100 GeV, 1 or 3 for 20 / 50, 0 for 50 / 100): no stable statement.

Not established, and why:
- **Power.** With 100-136 pairs a KS distance is detectable only above a floor of about 0.16-0.19. The block-shuffle KS of the matched pairs is 0.11-0.14 (median), against 0.31 in stage 2: the effect shrinks by more than half, but a smaller true effect cannot be excluded.
- **The matched subsample is a thin tail** (1-2% of each class); a statement about it is not a statement about whole showers. Matching is on deposit; the same pairs were not examined on the detector level.
- **Only 8 of 24 statistics, deposit only.** The 572-cell windows and the whole readout representation are not testable with these samples and the 336/572-cell eligibility; **detector-level survival of any matched difference is untested**. A 3D eligibility rule (S4) and larger matched samples are needed.
- The three pairings share events (electrons 20 GeV twice, protons 100 GeV twice) and were chosen on the same development data: agreement is not replication.
- Nothing here is evidence of scaling or fractal structure.

## 3. Guards

- **Matched-pair design (registration section 4):** electrons of 20 GeV (deposit), two random halves matched with the same rule, pairs capped at 150, 1,000 trials, 999 permutations: Westfall-Young family-wise error **0.039**, Holm **0.024**, unadjusted minimum p 0.201; tolerance 0.0707; **passed**.
- **Excess statistic of stage 2 (stage-2 registration section 3):** 8 blocks of 1,000 trials, **passed** on all six testable blocks (Westfall-Young 0.042 to 0.059, Holm 0.016 to 0.040; tolerance 0.0707): see the stage-2 record, section 3.

## 4. Consequences

- For the amendment: P1's matched comparison, at the sizes available, has power only for KS >= about 0.17-0.19; the 200 GeV protons and a larger production, and a 3D eligibility rule, are what make a test at the readout possible. The finding that the block-shuffle (sub-9-cell) difference disappears after matching is the HEGRA-type outcome (no difference beyond occupancy and energy) for that statistic; it should be stated in the amendment as the expected null and its power.
- For the benchmarks: nothing changes; this stage concerns P1/P2-type differences.
