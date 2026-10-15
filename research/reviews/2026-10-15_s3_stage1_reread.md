# S3 stage 1: valid re-reading of the 192 development KS verdicts (2026-10-15) — EXPLORATORY

Registration (written before any statistic): `research/plans/2026-10-15_s3_reread_registration.md` (section 7 is an addendum written after the main run and before the sensitivities). Code `src/ams_ecal/multiscale_estimators/multiscale_verdict_reread.py` (the vectorized engine reproduces `permutation_nulls.stratum_ks` on observed and permuted labellings, tested, `tests/multiscale_estimators/test_multiscale_verdict_reread.py`). Results: `results/multiscale_controls/s3_reread_stage1.json` (occupancy strata, the main run), `..._energy_strata.json`, `..._joint_strata.json`. Data: electron and proton development-large sets, electrons against truth-interacting protons, 10, 20, 50, 100 GeV, deposit and readout; exposed data, no sealed set read. Statistic: stratum-weighted KS; 10,000 label permutations within strata; Westfall-Young and Holm inside each representation family, multiplied by 2 across the two families.

## 1. Result

| strata | testable of 192 | significant, Holm (x2) | significant, Westfall-Young (x2) | unadjusted p <= 0.05 | p-values at the permutation floor |
|---|---|---|---|---|---|
| occupied cells (main) | 64 | 64 | 64 | 64 | all |
| total deposited energy | 64 | 64 | 64 | 64 | all |
| occupied cells x energy (25 joint cells) | 64 | 54 | 55 | 62 | 78% |

Established (data):
1. **Only 64 of the 192 tests can be run.** In the other 128 the median number of eligible electrons is 2 (the registered windows need 336 or 572 occupied cells; electrons below about 50 GeV on the readout rarely have them). Testable per block: 8 (the two 336-cell windows x 4 statistics) except 100 GeV deposit with 24. At 10 and 20 GeV on the readout none of the 24 is testable.
2. **All 64 testable tests were "material" in the development run, and all are confirmed** with occupancy strata and with energy strata (observed KS 0.145 to 0.842 against a permutation floor of 0.046 to 0.057 with occupancy strata; every p-value is at the floor 1/10,001, so Westfall-Young and Holm cannot be told apart here).
3. **The 15 old "material" verdicts that are not testable rested on about 31 electrons** (review item C4 confirmed); **all 113 old "not material" verdicts are untestable**: that label meant "too few eligible events", not "no difference".
4. **Matching on occupancy and energy together removes part of the effect.** 54 (Holm) or 55 (Westfall-Young) of the 64 stay significant; the median KS falls from 0.67-0.86 (energy strata) to 0.34-0.60 (joint strata), most at 50 and 100 GeV. The 10 tests that lose significance are D0, D2 and D0 - D2 statistics at 50 and 100 GeV, with KS 0.11 to 0.20 against a joint floor of 0.14 to 0.16.

Not established, and why:
- **The joint-strata result rests on a thin region of common support.** In every joint-strata test only 1 or 2 of the 25 cells contain at least 20 finite events of both classes (the others are skipped by the registered rule): electrons and interacting protons of the same primary energy barely overlap in (occupied cells, deposited energy). The 10 lost tests are the ones with 1 or 2 usable cells and the lowest effect; their loss is as much loss of power as evidence against a difference. Whether a difference exists "beyond occupancy and energy" can therefore be assessed only on a small overlap region.
- **The tests show that the classes differ in these statistics, not why.** No claim of multiscale (fractal) structure follows: occupancy and energy matching are the only controls so far, the cell-shuffle surrogate (plan item A3/C2) is stage 2.
- Westfall-Young and Holm agree here only because every p-value is at its minimum; the comparison says nothing about their relative power in a harder setting (that is the simulation of 2026-10-15).

## 2. Consequences for other records

- **DECIDE-1 amendment (P1 primary):** it specifies "five equal-count strata of each, jointly" on occupied voxels and energy. This stage shows pooled-quantile strata leave 1 to 2 populated cells per test at the same primary energy, so the planned stratification is likely underpowered by construction; the amendment should specify strata on the region of common support (for example electron-quantile cells restricted to where both classes have at least 20 events) and report the overlap, before the freeze. Candidate wording is the researcher's decision.
- **Eligibility rule:** the 336 / 572 thresholds produce the missingness that makes 128 tests untestable; the 3D eligibility rule planned for S4 replaces them.
- **P5 and the benchmarks:** unchanged; this stage concerns P1/P2-type differences.

## 3. Guard for a Westfall-Young claim (registration section 3): PASSED, with a note

Electron-against-electron halves (2,000 + 2,000 per trial), the block test of the re-read (stratified KS, joint label permutations), 1,000 trials, 999 permutations, `research/reviews/2026-10-15_s3_reread_familywise_guard.py` and `.json` (a first run crashed on the block with no testable statistic and was fixed; the design and tolerance did not change). Tolerance written before the run: 0.05 + 3 Monte Carlo standard errors = 0.0707.

| block | usable statistics | Westfall-Young family-wise error | Holm | smallest unadjusted p <= 0.05 |
|---|---|---|---|---|
| 20 GeV deposit | 8 | 0.051 | 0.038 | 0.252 |
| 20 GeV readout | 0 | not testable | - | - |
| 100 GeV deposit | 24 | 0.058 | 0.041 | 0.505 |
| 100 GeV readout | 8 | 0.057 | 0.041 | 0.251 |

All Westfall-Young rates are below the registered tolerance, so the guard passes and the Westfall-Young column may be used beside Holm for these statistics. Note: all three rates lie slightly above the nominal 0.05 (0.051, 0.058, 0.057; one standard error is 0.007 per block), while Holm is below it in all three; the excess is within noise and no sign of a failure, but the direction is the same in the three blocks, so a repeat with more trials is advisable before any confirmatory Westfall-Young claim. The guard covers the statistic definition and the electron features only (same-class null), not other statistics or other populations. The 20 GeV readout block could not be guarded because it has no testable statistic.
