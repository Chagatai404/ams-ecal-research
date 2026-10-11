# Slice S0 — sanity audit of the development ladder, data manifest and EMV initialisation dump (2026-10-11)

_Evidence record, not a plan. The single plan is `research/RESEARCH_PLAN.md`; its section 18.4 defines S0. Exploratory, on EXPOSED sets only; no sealed set was read; no registered number, no ladder code and no earlier result was changed. Nothing here is an accepted decision._

## 0. What was done, and with which code

| item | code / record | state |
|---|---|---|
| data manifest | `research/data_manifest.json` (SHA-256 and generation metadata of 22 exposed files; the two sealed sets by reference to their committed manifests, event files not read); `python -m ams_ecal.validation.sample_definitions write\|verify` | DONE; the 5.4 GB stays in `../ams-ecal-qml-gen*/data` (nothing moved, nothing untracked under `data/`) |
| one sample-definition module | `src/ams_ecal/validation/sample_definitions.py` + 9 tests: data roots, held-out rule per set, named populations, sealed refusal | DONE; the old modules (`eligibility_count`, `multiscale_development`, `multiscale_ladder`, `multiscale_gtrain_development`) still define their own samples and are migrated in S1 (their behaviour is unchanged) |
| sanity audit | `src/ams_ecal/multiscale_estimators/ladder_sanity_audit.py` + 6 tests; results `results/sanity_audit/ladder_sanity_audit_{10,20,50,100}GeV.json`, `..._energy_blind_pooled.json` | DONE (this record) |
| EMV initialisation dump | `src/ams_ecal/geant4_simulation/em_variant_initialisation_dump.py` + 3 tests; `results/em_variants/{baseline,emz,emv}.json`; no events run | DONE (section 7) |
| C15 comment fix | `configs/variants/geant4_electron_syst_{emv,emz}.yaml` first-line comments | DONE; changes their file hashes (the comment says so) |

## 1. Expectations written in the module docstring before the first run, and what happened

**Disclosure of one wording change.** After the four audit processes had been launched, and before any output had been read, the last expectation below was reworded in the docstring. The original read "a KS of 0.998 cannot sit beside an AUC of 0.991 for the same two samples: AUC >= (1 + KS) / 2", which is not a general inequality (it needs a concave ROC curve; a two-sided feature can have a large KS and an AUC of one half, see `test_a_two_sided_feature_has_large_ks_but_auc_one_half`). The corrected wording is weaker as a theorem and equal as a check: all 16 rows of the table in section 2 satisfy the strict original form AUC >= (1 + KS) / 2 anyway. After the runs, `ruff check --fix` made lint-only edits to the audit module (`range(0, 6)` to `range(6)`, redundant `int()` casts removed); no logic changed and the tests were re-run.

**Scope of the duplicate and seed checks.** They cover the baseline pilot sets and the development-large sets (four sets), one energy at a time, and hash the deposition grid only. They do not cover the pilot variants (which share base seeds with the baseline by design), cross-energy seed coincidences, or the sealed sets, so they do not contradict review C14 (4,009 duplicated seed values in the whole exposed pool).

| expectation | outcome |
|---|---|
| same-class null AUC 0.5 +- 0.02 (both groups >= 1000) | HELD in all 32 pairs: 0.483 to 0.515 (largest deviation 0.0144) |
| label permutation mean 0.5 +- 0.02, every draw within 0.5 +- 0.03 | HELD: 8 cells x 3 populations x 10 draws, range 0.488 to 0.513 |
| fold-seed sd of the ladder AUC < 0.002 | HELD: largest sd 8.4e-05 |
| no duplicates within or between sets; zero-energy events reported | HELD: 0 identical events within or between the four sets at every energy; 0 zero-energy events; 0 shared seed values among the four sets compared, one energy at a time (scope above) |
| a KS of 0.998 must not sit beside a univariate AUC of 0.991 for the same samples | RESOLVED: different representations (section 2); (1+KS)/2 <= AUC in all 16 rows |

The audit therefore found **no implementation error**. The S0 stop rule did not trigger.

## 2. The 0.991 and the 0.998

The reported "univariate AUC 0.991 at 10 GeV" is the **readout** (scintillator-only) total against truth-interacting protons (0.9907; pilot 0.9885); the "KS 0.998" is the **deposition** total (0.9976 interacting, 0.9981 all). They describe two representations of the same events, not one population. Within a representation every AUC is above (1 + KS)/2 (a monotone feature). In the deposition representation the total energy alone separates the classes at AUC 1.0000.

#### Population behind each single-feature number

| GeV | representation | protons | rank AUC | KS | implied floor (1+KS)/2 |
|---|---|---|---|---|---|
| 10 | deposit | interacting | 1.0000 | 0.9976 | 0.9988 |
| 10 | deposit | all | 1.0000 | 0.9981 | 0.9990 |
| 10 | readout | interacting | 0.9907 | 0.9508 | 0.9754 |
| 10 | readout | all | 0.9955 | 0.9698 | 0.9849 |
| 20 | deposit | interacting | 0.9999 | 0.9963 | 0.9981 |
| 20 | deposit | all | 1.0000 | 0.9971 | 0.9985 |
| 20 | readout | interacting | 0.9990 | 0.9789 | 0.9895 |
| 20 | readout | all | 0.9995 | 0.9861 | 0.9930 |
| 50 | deposit | interacting | 0.9999 | 0.9920 | 0.9960 |
| 50 | deposit | all | 1.0000 | 0.9940 | 0.9970 |
| 50 | readout | interacting | 0.9996 | 0.9847 | 0.9923 |
| 50 | readout | all | 0.9998 | 0.9880 | 0.9940 |
| 100 | deposit | interacting | 0.9997 | 0.9925 | 0.9963 |
| 100 | deposit | all | 0.9999 | 0.9942 | 0.9971 |
| 100 | readout | interacting | 0.9995 | 0.9863 | 0.9931 |
| 100 | readout | all | 0.9998 | 0.9906 | 0.9953 |

## 3. Controls

#### Development-large electrons against truth-interacting protons

| GeV | repr. | conventional (HGB) | k of N | total energy alone (rank AUC) | logistic, all conventional | label permutation (10 draws) min..max | same-class null range (4 pairs) | fold-seed sd |
|---|---|---|---|---|---|---|---|---|
| 10 | deposit | 1.00000 | 0 of 9696 | 1.0000 | 1.00000 | 0.490..0.506 | 0.486..0.511 | 1.4e-06 |
| 10 | readout | 0.99991 | 1 of 9696 | 0.9907 | 1.00000 | 0.494..0.507 | 0.483..0.506 | 4.5e-05 |
| 20 | deposit | 1.00000 | 0 of 9647 | 0.9999 | 1.00000 | 0.490..0.512 | 0.491..0.513 | 7.9e-07 |
| 20 | readout | 0.99998 | 1 of 9647 | 0.9990 | 1.00000 | 0.495..0.508 | 0.492..0.504 | 7.4e-05 |
| 50 | deposit | 0.99989 | 0 of 9709 | 0.9999 | 0.99999 | 0.492..0.513 | 0.490..0.515 | 5.5e-05 |
| 50 | readout | 0.99999 | 0 of 9709 | 0.9996 | 0.99999 | 0.492..0.505 | 0.483..0.502 | 2.8e-06 |
| 100 | deposit | 0.99989 | 2 of 9766 | 0.9997 | 1.00000 | 0.492..0.506 | 0.486..0.510 | 4.5e-05 |
| 100 | readout | 0.99999 | 0 of 9766 | 0.9995 | 1.00000 | 0.493..0.510 | 0.496..0.512 | 4.2e-06 |

Reading: the registered classifier is stable (seed sd), blind to a random label (permutation), and blind to a different seed or commit of the same physics (same-class nulls). The pilot electrons made at `a1d68d4` with a dirty tracked tree are not distinguishable from the clean `48a4a78` development-large electrons **at the resolution of this test**: with 1000 against 10,000 events the standard error of an AUC is about 0.01, so the statement is "no difference above roughly 0.02 AUC", not "identical". k of N shows 0 to 2 protons above the electron 99% threshold among 9.6-9.8 thousand: the zero-failure floor of the review (C1), not a measurement.

#### Duplicate-injection positive control

| GeV | no duplicates | 2 copies | 4 copies |
|---|---|---|---|
| 10 | 0.504 | 0.805 | 0.907 |
| 20 | 0.499 | 0.812 | 0.916 |
| 50 | 0.498 | 0.792 | 0.906 |
| 100 | 0.504 | 0.800 | 0.905 |

The cross-validation harness is **sensitive to leakage**: when the same events sit in training and test folds the AUC of an impossible task rises from 0.50 to 0.80 (2 copies) and 0.91 (4 copies). That makes the zero duplicates and the null results informative.

## 4. Why the problem is easy: the classes differ in every family (overdetermined)

| GeV | energy | occupancy | longitudinal | lateral | layers 0-5 | layers 6-11 | layers 12-17 |
|---|---|---|---|---|---|---|---|
| 10 only | 0.9999 | 0.9638 | 0.9982 | 0.9991 | 0.9965 | 0.9858 | 0.9644 |
| 10 without | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 20 only | 0.9997 | 0.9599 | 0.9979 | 0.9995 | 0.9970 | 0.9832 | 0.9667 |
| 20 without | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 50 only | 0.9997 | 0.9559 | 0.9977 | 0.9990 | 0.9974 | 0.9789 | 0.9657 |
| 50 without | 0.9999 | 1.0000 | 1.0000 | 0.9999 | 0.9999 | 1.0000 | 1.0000 |
| 100 only | 0.9994 | 0.9527 | 0.9979 | 0.9994 | 0.9974 | 0.9749 | 0.9667 |
| 100 without | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Each family or layer block alone gives AUC >= 0.95; removing any one leaves 0.9999-1.0000; a logistic regression on the conventional features gives 0.99999-1.00000 (so no tree interaction is needed). The **energy-blind pooled test** (all four energies pooled; total deposit and occupied-cell count removed; only shapes and fractions left): AUC **0.99995 in both representations** (per energy 0.99988-0.99999). So "the beam energy is known" (review item 1, and the model answer of Q6 in the tutor session) is **only part of the explanation**: the shapes alone separate electrons from truth-interacting protons at 10-100 GeV, normal incidence, one central cell, zero threshold. The problem is easy because electrons and interacting protons differ in every property at once in this regime; whether that survives AMS-like conditions is what the redesign (S5b/S7) tests.

## 5. The multiscale rung against the eligibility indicator alone

| GeV | repr. | electrons eligible (>=336 cells) | protons eligible | indicator-only >=336 | indicator-only >=572 | multiscale rung as registered | rung minus best indicator |
|---|---|---|---|---|---|---|---|
| 10 | deposit | 0.030 | 0.680 | 0.825 | 0.615 | 0.838 | +0.013 |
| 10 | readout | 0.000 | 0.411 | 0.706 | 0.503 | 0.702 | -0.003 |
| 20 | deposit | 0.910 | 0.745 | 0.582 | 0.691 | 0.986 | +0.295 |
| 20 | readout | 0.001 | 0.531 | 0.765 | 0.532 | 0.763 | -0.001 |
| 50 | deposit | 1.000 | 0.805 | 0.598 | 0.770 | 0.999 | +0.229 |
| 50 | readout | 0.030 | 0.649 | 0.809 | 0.619 | 0.823 | +0.014 |
| 100 | deposit | 1.000 | 0.833 | 0.584 | 0.783 | 0.999 | +0.217 |
| 100 | readout | 0.747 | 0.708 | 0.520 | 0.687 | 0.961 | +0.273 |

Reading: at **10 GeV (both representations) and on the readout at 20 and 50 GeV** the registered multiscale-only rung is within 0.014 of the AUC of "had at least 336 lit cells", so it **measures missingness** (the review's item 3, now measured in all eight cells, not three). In **deposition at 20-100 GeV and on the readout at 100 GeV** the rung exceeds the best indicator by 0.22-0.30: it carries real within-eligible information, which is redundant with the conventional features (conventional AUC 0.9999-1.0000 beside it). Nothing here says the multiscale features are useless in harder regimes.

## 6. The held-out-quarter rule is immaterial to every reported number

Applying the pilot rule (`event_index % 4 == 3` removed) to the development-large protons changes no conventional, energy-only or multiscale AUC by more than **0.0038**. The default recorded in `sample_definitions` (pilot protons: quarter held out; development-large: none) is therefore a bookkeeping choice; it is flagged to the researcher, not decided.

## 7. EMV: what the initialisation dump shows (no events)

> **CORRECTION, 2026-10-11 (later, after the swap closure test failed).** The text of this section, "EMV differs from FTFP_BERT in two parameters only", is **wrong as stated**. It came from the scalar getters of `G4EmParameters`, and the pybind build cannot return the msc step-limit enum, the energy-loss step-function tuple, the fluctuation-model name or the combined-transportation switch (they were listed under `_skipped` and I did not look at that list). Geant4's own printed table (saved in `results/em_variants/printed_table_*.txt`, differences in `printed_table_differences_FTFP_BERT_vs_EMV.json`) shows that EMV differs in at least: `ApplyCuts` false -> true; `MscRangeFactor` 0.04 -> 0.2; **msc step limit `UseSafety` -> `Minimal`; e+- energy-loss step function (0.2, 1 mm) -> (0.8, 1 mm); fluctuation model `Urban` -> `Universal`; combined TransportationWithMsc Disabled -> Enabled; Rayleigh scattering (Livermore) absent**. The pre-stated closure test (both swapped parameters reverted) FAILED: the scintillator share returns only to 0.0512 against 0.0653 for FTFP_BERT (restored fraction 0.41). Section 7 below is kept as originally written; read the correction above first.
>
> **Swap results (approved test, rule fixed before the events existed; `results/em_variants/swap_analysis.json`, 2,000 electrons at 20 GeV each, seeds paired with the EMV set, total deposit within 0.2% of EMV in all three):**
>
> | list | scintillator share | restored fraction f (95% interval) | verdict by the rule |
> |---|---|---|---|
> | FTFP_BERT (10,000 events) | 0.0653 | 1 by definition | reference |
> | FTFP_BERT_EMV | 0.0413 | 0 by definition | |
> | EMV with `ApplyCuts` reverted | 0.0420 | 0.03 (0.03 to 0.03) | not responsible |
> | EMV with `MscRangeFactor` reverted | 0.0504 | 0.38 (0.38 to 0.38) | partly responsible / unresolved |
> | EMV with both reverted | 0.0512 | 0.41 (0.41 to 0.42) | **closure test failed** (needs within 3% of 0.0653; it is 22% below) |
>
> The two effects add (0.03 + 0.38 = 0.41); about 59% of the deficit sits in settings that were not swapped (msc step limit, e+- step function, fluctuation model, combined transportation, Rayleigh). The bootstrap intervals are narrow because they contain only the event-sampling error of medians over 2,000 and 10,000 events; they do not contain the difference between the commits and seeds of the reference set (the Geant4 spread of the FTFP_BERT, cut and EMZ medians is 2.3%). The swap sets were generated at commit `77e876c` with uncommitted source changes (the backend edit; the manifest records `git_tracked_changes: true`). Each further swap needs the researcher's approval; none was run.

`results/em_variants/`, compared with `em_variant_initialisation_dump compare`:

| list | EM parameters that differ from FTFP_BERT | process names |
|---|---|---|
| FTFP_BERT_EMV (option 1) | `ApplyCuts` false -> **true**; `MscRangeFactor` 0.04 -> **0.2** | identical |
| FTFP_BERT_EMZ (option 4) | `Fluo` on, `LowestElectronEnergy` 1 keV -> 0.1 keV, `MscRangeFactor` 0.04 -> 0.08, `MscSkin` 1 -> 3, `NumberOfBins` 84 -> 240, bins per decade 7 -> 20, ICRU90 data, Mott correction, ... | `ePairProd` added |

**Interpretation, not a finding.** With `ApplyCuts` true the production cut (here 0.7 mm) is also applied to the secondaries of gamma processes and of the low-energy EM spectrum: soft photo-electrons below the cut are not tracked and deposit locally in the lead instead of reaching a fibre. A lower scintillator share that grows with depth (soft photons dominate deeper in the shower) is what that mechanism would give. The larger `MscRangeFactor` lengthens steps through the thin fibres. Whether either setting, or both, produces the 37% lower scintillator share is **untested**: the test is a swap of option-1 settings one at a time on events (about 2,000 each), which needs the researcher's approval and was not run. Until then the EMV set is not used as a systematic.

## 8. Limits of this audit

- It inherits the ladder's same-event threshold and zero-failure floor: only AUC and k of N are reported, never a rejection.
- One classifier (the registered HGB) and one logistic regression; the injection control tests the HGB harness only.
- The pooled test pools four fixed energies; it says nothing about a continuous spectrum.
- Truth-interacting protons are a simulation-only population; the "all protons" rows contain crossing protons, which separate trivially (AUC 1.0000, see the JSON).
- The multiscale rung is the registered rung with its known problems (NaN as a feature, thresholds 336/572 from equal-quanta controls); S1 repairs the tools, S4 the estimators on 3D voxels.

## 9. For the researcher (nothing here is decided)

1. Keep the recorded held-out defaults (pilot protons: quarter held out; development-large: none)? Immaterial to every number, but it must be one recorded rule (C8).
2. Approve the EMV one-at-a-time swap on about 2,000 events (`ApplyCuts`, then `MscRangeFactor`)? Until then EMV stays unused.
3. The Q6 model answer in the tutor session is qualified by section 4 (corrected in the session note).
4. Still open from the program decisions: the DECIDE-1 amendment text (S2), the DECIDE-2 exit date, regenerate-or-ledger for the 2 sealed-proton seed collisions (C14; the exposed sets compared here share no seed, and the sealed-versus-exposed intersection was not touched).
