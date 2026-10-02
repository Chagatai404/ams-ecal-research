# Validated FastMC dataset and the start of the detector response — PLAN

_Drafted 2026-10-02 on branch `fastmc-validated-dataset` (from `geant4-proton-pilot` at `ca448eb`), at the researcher's request: "plan and implement the remaining part of the stochastic event generation and start detector response. We need a validated FastMC dataset asap."_

Status of this document: **a plan.** Section 7 separates what the researcher's message of 2026-10-02 authorises from what still needs a decision. Nothing in sections 3-6 is an accepted conclusion.

Reads with: `research/DECISIONS.md` (DEC-003 to DEC-013), `research/STATE.md`, the proton dependency analysis (§3 factorization, §8 contract, §11 crossing result) and the weekend plan of the same date.

## 1. What "validated FastMC dataset" has to mean

A dataset is *validated* when every row below has been run **once, on Geant4 events no model choice ever saw**, with the verdict reported row by row (no single score, `RESEARCH_PROTOCOL.md` §5):

| layer | what is checked | status today |
|---|---|---|
| EM generator vs Geant4 electrons | contract rows (energy, longitudinal, lateral, activity, correlations) | **never done.** The generator is parameterised from literature and unit-tested; it has never been compared with transport. A 4000-electron Geant4 sample now exists (`data/geant4_electron_sample/`, unread). |
| Proton crossing branch | the §8 contract | run once, **failed** on event total, hits, max-cell, containment (§11 of the dependency analysis) |
| Proton interacting branch | the §8 contract + added checks | **not implemented** |
| Dataset level | energy/geometry matching between classes, seed isolation, split integrity, containment/leakage, no accidental label leakage | planned in `STATE.md`, not built |
| Detector response | per-effect evidence, then validation | **blocked** on the evidence pass (section 5) |

## 2. The shortest honest path, and why it does not wait for the detector response

DEC-006 fixed `deposition` as the common electron/proton representation. Both generators can emit it **without any detector response**. So the first validated dataset is a **perfect-event (true-deposition) dataset**, labelled as such; the detector response then adds a second, detector-level version. This is a recommendation for the researcher to accept or reject (decision Q1).

```text
Track A  proton generator         A1 crossing repair (burst + spill)  ->  A2 interacting model  ->  A3 batch path  ->  freeze
Track B  validation data          B1 register added checks (needs researcher's numbers)  ->  B2 sealed Geant4 sets (background)  ->  B3 one opening
Track C  electron side            C1 EM generator contract (pre-registered)  ->  C2 one opening on sealed electrons
Track D  detector response        D1 evidence pass (running)  ->  D2 sampling response R-A (Geant4-calibrated)  ->  D3 instrumental effects R-B (evidence-gated)
Track E  dataset assembly         E1 energy/angle sampling, seed policy, split, dataset-level checks  ->  v0 perfect-event dataset  ->  v1 detector-level
```

A, B and C run in parallel. The long wall-clock item is **A2** (the interacting model). Compute is not the constraint: the pilot cost 710 s for 4000 protons at 100 GeV on 12 workers, and the 4000-electron batch took about 51 minutes.

## 3. Track A — proton generator (steps 4-8 of the accepted sequence)

### A1. Crossing repair: burst latent and lateral spill (DEC-008, DEC-009, EXP-005)

Design evidence, from **calibration crossing events only** (`event_index % 4 != 3`, about 1550-1590 per energy; the held-out events were not read):

1. **Bursts are cascades, not multipliers.** At 50-100 GeV the aligned deposition excess rises over one to three layers after onset and decays slowly; only 9-12% of a burst's in-ECAL energy is in its onset layer. At 10 GeV bursts are mostly one layer (about 89% in the onset layer). Late-onset bursts carry about half the in-ECAL energy of early-onset bursts (truncation by the back). Onset is roughly uniform over depth.
2. **Fibre and deposition bursts are the same bursts.** Spearman of the two excess sums within the largest 10% of deposition bursts: 0.51 / 0.76 / 0.84 / 0.88 at 10 / 20 / 50 / 100 GeV; fibre excess is about 6% of deposition excess (median 0.056-0.062). The amendment of DEC-006 (independent layer draws per representation) was based on the per-layer correlation of ordinary layers, 0.15-0.24; for bursts a shared latent is supported. **Design: one burst latent per seed shared by both representations**, with a Gaussian copula on the two amplitudes at the measured correlation. This is a refinement of "deposition and readout fluctuations may differ", not a reversal.
3. **Spill is a function of the layer's own excess, not an independent process.** P(layer has fibre energy outside the crossed cells > 0.1 MeV) = 2% for ordinary layers, 16-20% for 1.5-3x the median, about 50% for 3-8x, about 90% above 8x. In events without a deposition burst the layer-count clustering is close to binomial at 50-100 GeV (variance 0.96 vs 0.58). Spill is mostly one cell per layer (94.4% / 92% of layers have none; 5-6.5% have one), the energy of a spill cell is nearly independent of its distance (median 0.25-0.35 MeV out to far offsets) while the **rate** falls with distance. **Design: spill is a conditional split of the layer's total**, given its ratio to the chord-conditioned median, so layer and event totals are unchanged.
4. **The bulk table must lose the bursts.** The existing crossing table's tail already contains the burst layers; adding a burst on top would double count. The bulk table is rebuilt from burst-free layers and the burst is added on top.
5. *Interpretation, not a result:* these look like knock-on-electron cascades in the lead (about 9% of events with a > 100 MeV knock-on is the order of magnitude kinematics gives). It is recorded as a consistency check to run, not as the model. The model stays empirical.

Deliverables: calibration artifact **schema 2** (new directory; `ftfp_bert_v1` is kept untouched as the record of the first crossing validation), builder, generator, tests, and an **in-sample** diagnostic against the calibration events (explicitly not a validation).

Integrity: the repaired crossing branch is **not** re-validated on the 2026-09-29 held-out events a second time. It is validated once, on the sealed set (B2).

### A2. Interacting model (DEC-005, EXP-006)

Amplitude x universal profile + albedo + lateral-scale latent, with the three amendments (back-edge factor; correlated residual in log-energy space with its normality tested first; shared albedo latent). It reuses the A1 machinery: the *spill* kernel is the same kernel (dependency analysis §3, "the same kernel the interacting events need"), and the *component abstraction* (onset layer, amplitude, profile in offset, lateral kernel) is shared between a knock-on burst and a hadronic interaction. Risks named in the dependency analysis and kept: cell-level sparsity from a smooth kernel (fall-back: explicit `N_hit` table), shape residual, albedo. The §35 stop rule stands.

**First A2 evidence (calibration events only, 2026-10-02): the residual is NOT log-normal.** DEC-005(b) required the distribution of the log-energy residual to be tested before it is called log-normal. For interacting events with the eight layers after the interaction layer inside the ECAL (n = 884 / 910 / 931 / 908 at 10 / 20 / 50 / 100 GeV; fibre energy; `ln e_k - ln A3 - mean profile`): skewness -1.77 / -0.38 / -1.49 / -1.51, excess kurtosis +26.9 / +4.5 / +12.5 / +9.7, Shapiro-Wilk p < 1e-38 at every energy, standardised 1% quantile -2.7 / -2.8 / -3.4 / -3.7 against -2.33 for a normal, while the quartiles sit at about +-0.45 (normal +-0.67): peaked with a heavy lower tail (occasional strongly suppressed layers). Adjacent-layer residual correlation +0.35 / +0.44 / +0.47 / +0.53 (the analysis had +0.3 to +0.4). Zero-energy layers are 0.4% at 10 GeV and fewer above. Consequence for the design: the family is an **empirical marginal quantile table** of the residual (energy-dependent scale) joined by a correlated Gaussian copula, the same hybrid as elsewhere; a log-normal residual would be wrong. The back-edge factor and the shared albedo latent are still to be measured. Script: not yet in the repository (scratch); it becomes part of the A2 builder.

### A3. Batch path (step 7)

11-14 ms/event is too slow (10^6 events about 3 h). The cost is per-event Python (`cross` twice, tuple-of-tuples event with 1296-value validation). The A1 sampling code is written vectorised from the start; the per-event `ECALEvent` API stays for provenance, the dataset path returns arrays plus a provenance table.

## 4. Track B — validation data and registration

**B1. Register the added checks before any sealed event exists (DEC-007).** Draft: `research/plans/2026-10-02_added_validation_checks_preregistration.md`. Numbers are **proposed**; the researcher confirms them (Q2). Generation is held until then, as decided.

**B2. Sealed sets (approved item 6 of the weekend plan), proposed content:**

| set | species | energies (GeV) | events | purpose |
|---|---|---|---|---|
| proton sealed | p, FTFP_BERT, AMS-only, normal incidence, entry spot as the pilot | 10, 20, 50, 100 (3000 each) and **14, 30, 70 (3000 each)** | 21,000 | anchors plus **off-anchor** energies: the model interpolates in ln E, and a validation only at anchors would not test that |
| electron sealed | e-, same geometry and cut | 10, 20, 50, 100 and 14, 30, 70 (1000 each) | 7,000 | the EM contract; the existing 4000-electron sample stays the exploration/calibration set |

Seeds disjoint from every earlier batch; generated from a clean commit; per-file SHA-256 recorded; a code guard in the reader that refuses to open a sealed file without an explicit `--final` flag and appends to an **opening ledger** that is committed. Estimated wall time from the measured costs: about 1-1.5 h per species on 12 workers. Off-anchor energies are an addition to what you approved (item 6): decision Q3.

## 5. Tracks C and D — electrons and the detector response

**C. EM generator contract.** The EM generator is smooth by design (DEC-001: deterministic lateral profile, no cell-level fluctuation). The proton generator will not be smooth (bursts, spill, quanta, sparsity). **A classifier trained on the dataset could separate electrons from protons by that asymmetry alone.** That is the "accidental label leakage" the dataset check exists for, and the classifier two-sample test would flag it. DEC-001 excluded a stochastic lateral model from the EM generator, so this is not changed here: it is flagged as the most likely dataset-level finding, with the decision needed once measured against sealed electrons (Q4). Until then the dataset README must say which observables are not comparable between classes.

**D1. Evidence pass (started 2026-10-02).** A fresh-context discovery scout ran with the neutral question and no model details. **Its result is thin and unverified**: abstract/snippet depth only, it did not find the two AMS papers already verified in the vault (Zhang 2016, Li 2013), and its per-effect "corrected before the fit" table is inference, not evidence. It is recorded as a discovery pass that did **not** answer the sub-question. Next for D1: a source-verification pass on the primary AMS ECAL references (full text), then the adversarial pass, then the per-effect inversion table. Nothing from the scout is promoted.

**D2. A split of the detector response that the evidence map did not make (proposal, Q5).**
- **R-A, sampling response** (true deposition -> fibre energy: depth-dependent fibre fraction, sampling fluctuation). Geant4 already provides both `deposit_grid_mev` and `readout_grid_mev` for protons, and the electron sample provides them for electrons. This is exactly the "electron fibre-energy scale from a small Geant4 electron calibration" that DEC-006 said was missing, and it can be **calibrated, then validated on sealed electrons**. It is built against the `deposition` regime, as the 2026-09-28 amendment requires, so it does not re-apply the sampling shift. It is also the experiment that converts the double-counting inference (open question 5 of the evidence map) into a measurement: fit gamma profiles to both deposition and readout in Geant4 electrons and compare.
- **R-B, instrumental effects** (gain and PMT position effect, noise, thresholds, ADC range and saturation, dead cells, cross-talk). No Geant4 equivalent; AMS-specific; stays blocked on the evidence pass and its per-effect inversion table.

## 5b. Research OS v0.7.0 (checked 2026-10-02, after the researcher's update)

v0.7.0 was already installed in `~/.claude` (all 15 skills and the 10 agent definitions match the repository HEAD apart from line endings), so no reinstall was needed. What changes for this work:

- **Literature sessions** are complete only when every selected useful paper has an outcome: saved PDF and metadata, an existing-library match, an unavailable PDF with its citation, or a reviewed rejection (`skills/research-session`). **Applied retroactively to D1:** the evidence pass above is **incomplete** under this rule. The scout's candidate papers have **no recorded outcome**; none was reviewed, none was ingested (downloading a PDF needs explicit permission), and several are snippet-level only. They stay "candidate, unreviewed" until a verification pass assigns each an outcome.
- **The tutor** is concept-first with separate conceptual, transfer and application evidence (`skills/tutor`). It applies to the learn-back after the interacting model (section 6), not to the code work.
- Visual selection is intent-based (`skills/visualize`); no new visual is needed for this slice.

## 5c. Progress, 2026-10-02 (branch `fastmc-validated-dataset`, uncommitted)

**A1, crossing repair: implemented and tested; in-sample fit checked; NOT validated.**

| item | state |
|---|---|
| `src/ams_ecal/proton_structure.py` (new) | burst latent with onset, amplitude, **downstream extent** and fibre share; lateral spill as a conditional split; bulk coupling; builder; vectorised samplers |
| `src/ams_ecal/proton_calibration.py` | artifact schema 2 (loads schema 1 too); `build_calibration(structure=True)`; bulk table rebuilt from burst-free events |
| `src/ams_ecal/crossing.py` | `place_layer_energies` (exact energy conservation per layer) |
| `src/ams_ecal/proton.py` | structured crossing path, fixed draw order, burst latent in the provenance; schema-1 path kept |
| artifact | `data/calibration/proton_model/ftfp_bert_v2` (development build from an unclean tree, `tracked_changes: true`; the **frozen** artifact is rebuilt from a clean commit before the sealed set is generated). `ftfp_bert_v1` is untouched. |
| tests | 529 passing (484 before), ruff clean; three mutation checks confirmed the new tests catch a broken bulk selection, a spill that ignores the layer ratio, and an untruncated burst |
| in-sample check | `uv run python -m ams_ecal.proton_structure_check` (calibration events only) -> `results/proton_model/crossing_structure_in_sample.json` |

**Beyond the two accepted repairs, for the researcher (Q6 and Q7):**

1. *Shared burst latent* across readout and deposition (Q6), supported by the calibration correlation 0.5-0.9 between the two excess sums.
2. *Downstream extent* as a latent. This is the fourth element of the structure you specified (probability, onset, energy, extent). It was needed because a fixed-extent burst under-dispersed the event total.
3. **Bulk coupling (Q7), NOT one of the accepted repairs.** In burst-free events the two readout layers of one superlayer correlate (fibre lag-1 within a superlayer 0.13-0.16, across superlayers about 0.02; deposition 0.15-0.19 and 0.11-0.12). Independent bulk draws miss it. It is a three-parameter Gaussian copula on the bulk layer draws (within-superlayer, neighbouring-superlayer, lag-2), calibrated on burst-free events. It is the smallest change that reproduces a measured layer-correlation row, but it is a third element and DEC-005's "no generic covariance engine" spirit asks you to decide it. Proposal: keep it; it is one row of the added checks.
4. The bulk per-layer table is now built from burst-free **events**, not from all layers: pre-onset layers of burst events are measurably heavier than the layers of burst-free events, and pooling them made burst-free events 2% too energetic.

**Findings from the calibration events that are not yet explained** (recorded, not acted on): the extreme burst tail rests on about three events per energy. In `results/proton_model/crossing_structure_in_sample.json` (4000 generated events against about 1550 calibration events per energy; KS floor about 0.03-0.04) the readout rows are at the floor for the event total (KS 0.016-0.031) and close for hit cells (0.047-0.058), maximum cell (0.035-0.055) and containment (0.060-0.077, inflated by a spike at 1); lag-1 layer correlation matches within 0.03. What is still off: the event-total variance ratio is 19-24% low in deposition at 50-100 GeV (4.42 vs 5.80, 5.45 vs 6.69) and 6-10% low in readout, and lag-5 correlation is 10-25% low in deposition at 100 GeV (0.10 vs 0.13). Generation costs 10-18 ms per event, unchanged from before (Python-bound; the batch path is A3).

## 6. Order of work in this session and the gates

1. Done: branch; baseline suite (484 passed, 3m21s); evidence pass launched (thin, section 5); calibration-only structure study; **A1 implemented and tested** (section 5c).
2. Next in code: A2, the interacting model, then A3, the batch path. Notebooks only after the model is final (DEC, 2026-10-02).
3. Blocked on the researcher: Q2 (numbers) -> B2 generation. Q1, Q3-Q7 shape tracks A, C-E.
4. Before the sealed set: commit, rebuild the frozen artifact from a clean tree, register the added checks (B1).

## 7. What is authorised, what is not

| item | authority |
|---|---|
| New branch `fastmc-validated-dataset` | researcher, chat, 2026-10-02 |
| Proton steps 4-8 as in the weekend plan (burst + spill, interacting model, batch path, freeze) | **item 7 of the weekend plan was "approval to start Monday"; the 2026-10-02 message ("plan and implement the remaining part of the stochastic event generation") is treated as that approval.** Please correct me if it was not. |
| Start the detector response | the evidence pass only (D1). No detector-response code. |
| Sealed proton set generated early | approved 2026-10-02 (item 6), **after** B1 is registered |
| Burst latent structural, lateral spill | accepted 2026-09-30 (DEC-008, DEC-009) |
| Shared burst latent across both representations | **new, proposed here**, supported by the calibration evidence in A1; needs the researcher's nod (Q6) because it refines the DEC-006 amendment |
| Off-anchor sealed energies; electron sealed set; using the electron sample for EM contract; R-A | **proposed, not authorised** |

## 8. Decisions needed, in the order they block work

| # | question | recommendation |
|---|---|---|
| Q2 | Confirm or amend the numbers in the added-checks pre-registration | confirm as drafted; this unblocks sealed generation |
| Q1 | First validated dataset = perfect-event (`deposition`) version, detector-level version second | yes |
| Q6 | Shared burst latent across readout and deposition | yes, per the A1 evidence |
| Q3 | Add off-anchor energies (14, 30, 70 GeV) to the sealed proton set; add an electron sealed set | yes |
| Q4 | If the EM generator separates from Geant4 electrons at cell level, extend it with a calibrated cell-level fluctuation (reversing DEC-001's exclusion) | decide after the electron contract, not now |
| Q5 | Split the detector response into R-A (Geant4-calibrated sampling response, may start after the EM contract) and R-B (evidence-gated) | yes |
| Q7 | Keep the three-parameter bulk coupling (section 5c) as a third element of the crossing repair | yes; it is the smallest change that reproduces a measured layer-correlation row |

## 9. Risks

- The interacting model may hit the §35 stop rule (cell-level sparsity from a smooth kernel). Then the report states what failed and the minimal added variable.
- Everything proton-side is Geant4 11.4.1 with one physics list; the dataset inherits that limitation (QGSP_BERT calibration, EXP-007, is still required before any claim about hadronic physics).
- Normal incidence only. A dataset with an angular distribution needs an angular calibration that does not exist; the first dataset is normal-incidence and says so.
- The detector-level version depends on R-B evidence that may be partly non-public (AMS internal notes).
