# Reviewer agents, round 1 — 2026-10-05

Branch `fastmc-validated-dataset`, HEAD 4701165 (frozen artifact `cee305f1…`). Released at the researcher's request ("release the review agents to review every bit"), each in a fresh context with a neutral brief, no repository edits, and no access to the sealed sets. This file keeps what each reviewer reported and **my assessment of it**, including where I think a reviewer is wrong or read stale text. The assessments are mine, not the researcher's. Full reports are in the session transcript.

The reproducibility auditor and the literature scout were interrupted or blocked on the first attempt and were rerun; their results are at the end of this file.

## Physics reviewer — no scientific blocker

- DEC-005 realisation (per-offset tables + copula, not literal amplitude x profile) is mathematically sound but **awaits the researcher's acceptance**. Agree; already recorded in DECISIONS.md.
- `UPSTREAM_MIP_CAP` is a phenomenological device, not physics. Agree; labelled as such.
- Lateral fixed-point match validates energy-weighted shares, not cell-by-cell structure. Agree.
- Containment residual 0.18-0.20 (the number quoted is the pre-repair one; after the repairs it is 0.10-0.15). Agree it is unexplained.
- Absolute visible-energy scale is FTFP_BERT-dependent; an alternative-physics-list calibration is needed before any claim about hadronic shower physics. Agree; known.

## Simulation reviewer — separation intact

- Sealed configs match the pilot (geometry, FTFP_BERT, cut, entry rule); seeds disjoint; E70/E100 rerun identical.
- The guard cannot be bypassed from repository code; no other reader of the sealed directories exists.
- The held-out half was used once (2026-09-29, failed); the repairs were calibrated on calibration events only; sealed data were generated 2026-10-02, before the 2026-10-04 repairs.
- Not checked: tamper-resistance against git history rewriting.

## Statistics reviewer — one issue labelled "invalidating", which I partly dispute

- **A. `UPSTREAM_MIP_CAP` chosen by grid search on calibration events.** The reviewer calls this overfitting that violates the pre-registration. My assessment: tuning a model parameter on calibration events is what calibration events are for, no held-out or sealed event was used, and the pre-registration governs the sealed run. But the reviewer's underlying question deserves a straight answer: **yes, the cap (and the other repairs) were chosen in response to the harness dry run's flags**, whereas the registration says the dry run is "to catch bugs, not to tune". The numbers measured on the calibration events after the repair are in-sample and are labelled so. Whether the registration's wording should be amended to allow a repair triggered by a dry-run finding is the researcher's call.
- B. Burst and lateral models are fitted jointly on the same calibration events, so neither can be validated alone in-sample. Agree; the sealed run is the test, and the residuals are to be reported, not tuned away.
- C. The dry run on pilot calibration events cannot estimate performance. Agree; it is labelled a stand-in.
- Checked and found sound: the permutation test design, the Wilson interval with n = number of events, the 4:1 split, the lag-averaged-correlation bootstrap.
- Not checked: electron-versus-proton AUC magnitude (reported, not gated, as D9); production-cut sensitivity on the full sealed set.

## ML reviewer — no blocker; one disclosure item that I will act on

- No seed repeats, no seed/energy/split correlation, 45 features exactly as registered, no preprocessing leakage, permutation null correct.
- **`interacting` and `interaction_depth_mm` are stored next to the grids in `events.npz`, and `interacting` is a perfect proxy for the label (every electron is False).** The reviewer judged it safe by convention; I judge that a downstream user loading one file can use them by accident. Action: write them (and the seeds) to a separate provenance file, keep only grids, label and split in the training file.
- Small-n datasets flag D1/D2/D4/D5 by chance; the thresholds assume the full-size dataset. Agree; the tiny-dataset flags in test runs are expected.

## Adversarial reviewer — several threats are stale, three are real

Stale or misread: threat 1 (the 64% failure) is the 2026-09-29 held-out result of the old crossing model, which is exactly what the repair addressed; the repaired model has not been validated on held-out data and nothing here claims it. Threats 4 and 5 describe checks and models that were "not built" in the plan text of 2026-10-02 and are built now. Real:

1. **The EM generator has never been compared with Geant4 transport.** Correct, and the most important open item. Action: run the EM contract on the exploration electron sample (`data/geant4_electron_sample`, not sealed) before anything else is said about the electron half.
2. **All validation is against one physics code (Geant4 11.4.1, FTFP_BERT, AMS-only geometry).** Correct and stated in the module docstrings ("Geant4-derived phenomenology"), but "validated" can mislead. The pilot holds 1000 QGSP_BERT events; the researcher may want a robustness comparison on them. Not done.
3. **The word "validated".** Agree that the dataset must be called a development dataset until the sealed verdict. The manifest status already says DEVELOPMENT.
- The reviewer's QGSP_BERT proposal ("500 events in 5 minutes") underestimates the Geant4 cost; the existing pilot QGSP_BERT events are the cheap route.
- Detector-response evidence is incomplete; no detector-level dataset is claimed.

## Decisions waiting for the researcher

1. Accept or amend the DEC-005 formulation (per-offset tables + copula).
2. Whether a model repair triggered by a harness dry run is acceptable under the registration, or whether the registration should say so explicitly.
3. Confirm the dataset-level thresholds (Q8, Q9 of the dataset registration).
4. Whether to add a QGSP_BERT robustness comparison using the existing 1000 pilot events before the sealed opening.

## Reproducibility auditor (rerun) — partly confirmed, partly not reconciled

- **Confirmed exactly:** the calibration artifact rebuilt in a clean worktree reproduces the content hash `cee305f1de265bac96e519c7c57270240737c305fab1f86cc6e506ae2e61f7cb`; a dataset assembled twice from the same seed (40 events per class, base seed 5) is byte-identical; the environment is pinned by `uv.lock` (Python 3.14.4).
- **Not reconciled, so I do not count it as confirmed:** the auditor reports 474 passed, 10 skipped, no warnings, in 118 s; my own full run on the same commit is 676 passed in about 12 minutes with 3 warnings. The slow lateral and calibration tests are the likely difference, but the report does not say which tests were run. The requested cross-process and chunk-size determinism experiment was replaced by pointing at unit tests that cover it. The in-sample check was only run at `--events 300` and its numbers were not compared in detail.

## Literature scout (rerun) — a source map at abstract and snippet depth, nothing verified

20 sources, almost all at abstract, snippet or title depth; the scout flags this itself and six arXiv full texts could not be parsed. Per Research OS v0.7 nothing is promoted; each source still needs a per-paper outcome. My notes on what is usable and what is not:

- **Consistent with the repository, at snippet depth:** AMS-02 ECAL as 648 x 648 mm, 166 mm, 18 layers, 9 superlayers, about 17 X0 (arXiv:1009.5349); the CaloChallenge classifier-AUC protocol (arXiv:2410.21611): AUC 0.5 means indistinguishable, ensembles of 10 classifiers, low-level (voxel) and high-level (shower observables) classifiers. The repository's registration uses one classifier on high-level features, not an ensemble of voxel classifiers; whether that is enough is not settled by this map.
- **Not usable as stated:** the "30-40% non-interacting" and "0.33 lambda_I" figures come from cosmic-ray and collider snippets for other detectors; the AMS-like ECAL is about 0.6 lambda_I, so they are not comparable. The line "known pitfall: classifiers separate smooth parametrisations from Geant4" is the scout's own inference attached to CaloFlow and AtlFast3, not something the retrieved text says. "5 x-layers, 4 y-layers" was checked on 2026-10-05 and is correct as a count of superlayers per view (Li et al., arXiv:1308.5564, first page).
- **Gaps the scout reports and that matter here:** no source found on layer-to-layer correlation matrices of proton showers in this detector class; no quantification of dataset leakage in electron/proton studies; no sample-size or power guidance for two-sample classifier tests.
- Overall: the literature pass has not yet answered the sub-questions the repository's decisions depend on. A source-verification pass on the specific claims is still needed.
