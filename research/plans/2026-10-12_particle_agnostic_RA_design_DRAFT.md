# Particle-agnostic sampling response (R-A): design, DRAFT for approval (2026-10-12)

_Status: **DESIGN, not a decision, nothing implemented beyond the exploratory checks quoted here.** Follows section 9 of `2026-10-12_detector_response_model_proposal_DRAFT.md` (the per-class electron R-A and its two open problems: it fails on protons, and it does not interpolate between energies). Experiments below used development sets only; scratch scripts, no sealed data._

## 1. The problem stated precisely

R-A maps the all-material deposit per cell `D` to the scintillating-fibre energy `R` (what the detector sees). A model fitted on electrons is wrong by 39-47% (maximum per-layer error) on protons. We want one mapping that can be applied to *any* shower, including mixed ones (a proton shower contains electromagnetic sub-showers from pi0 decay), and that is smooth in energy.

## 2. What the data say (exploratory, development sets)

**E1. Deposit-only features cannot make the map transfer.** A gradient-boosted model of `R/D` (cell-weighted) with four feature sets, trained on electrons and tested on protons at 10 and 100 GeV:

| features | electrons -> protons, max layer error | protons -> electrons | e10+p10+e100 -> p100 |
|---|---|---|---|
| layer, log D, event total | 41.7% | 75.1% | 43.4% |
| + fine structure (occupied voxels in the cell, share of the largest voxel) | 41.6% | 91.3% | 44.0% |
| + neighbouring cells (same layer, adjacent layers) | 42.5% | 80.1% | 44.5% |
| without event total | 42.0% | 99.0% | 34.3% |

Even when protons at the *other* energy and electrons are in the training set (last column), a held-out proton energy is off by 34-45%. So the information that sets `R/D` is **not in the deposit field at 3 mm voxel resolution**; adding geometry of the deposit does not help. (Voxel mesh 216 x 216 x 36; a cell is 3 x 216 x 2 voxels, mapping verified exactly against the grid for the first layers. R and D are consistent to better than 0.6% of the fibre energy, so geometry mismatch is not the cause.)

**E2. The ratio depends on the kind of energy deposition.** Event-level `sum R / sum D`:

| class | 10 GeV | 20 GeV | 50 GeV | 100 GeV |
|---|---|---|---|---|
| electrons | 0.0653 | - | - | about 0.0655 |
| protons that cross without interacting (MIP-like) | 0.0741 | 0.0725 | 0.0713 | 0.0703 |
| protons that interact | 0.0987 | 0.0902 | 0.0817 | 0.0774 |

Three different values and a strong energy dependence for interacting protons (0.099 falling to 0.077). A linear regression of the interacting-proton ratio on the pi0 energy fraction (a truth quantity, used for diagnosis only) explains only R2 = 0.24-0.27, with zero-pi0 intercepts 0.109, 0.099, 0.088, 0.082 at 10, 20, 50, 100 GeV. Extrapolating the fit to a pure pi0 shower gives 0.039 at 10 GeV and 0.060 at 100 GeV, against the electron value 0.065, so a one-parameter electromagnetic/hadronic mixture with a *fixed* hadronic ratio does not close either.

**Reading, kept separate from fact.** What is measured: three sampling ratios (electron-shower, MIP, interacting-hadron) and an energy trend. What is **hypothesis, not tested**: that the higher interacting-hadron ratio comes from hydrogen-rich polystyrene fibres receiving recoil protons from neutron scattering (lead has almost none), and that the energy dependence reflects a changing mix of slow hadronic products. The existing Geant4 files contain no per-particle breakdown, so they cannot test it.

## 3. Design: a component-wise sampling response

**Principle.** The sampling fraction is a property of *how* energy is deposited (by what kind of particle and at what energy), not of the primary particle's name. Define R-A on **deposit components**, and apply it to any shower that supplies components. A proton shower then uses the electromagnetic ratio for its pi0 sub-showers and the hadronic ratios for the rest, with no per-primary model.

**Components (proposed, to be confirmed by the tagging run):**
1. electromagnetic: energy deposited while a step's particle is e-, e+ or gamma;
2. minimum-ionising / fast charged hadrons (muons, pions and protons above a kinetic-energy cut, to be fixed from the data before any fit);
3. slow charged hadrons and nuclear fragments (below the cut);
4. neutron-induced (energy deposited by recoil protons and ions set in motion by neutrons), tagged through the parent chain if feasible.

**Model per component `c`.** `R_c = D_c * rho_c`, with `rho_c` drawn from a conditional distribution that depends on the cell's component deposit size and layer, and on **energy only through a smooth function of the local deposit scale** (a low-order polynomial in `ln D_c`, not tables per energy class). Cell fibre energy is the sum over components. The scatter model of the present R-A (empirical quantile tables) is kept as the stochastic part, but indexed by `(c, layer, log D_c)` and *without* the event-total class, which is what prevented interpolation.

**Inputs the downstream chain needs:** per-component deposit grids. Geant4 provides them after the tagging run (section 4). Fast simulations must produce component grids (the electron production generator is electromagnetic by construction; the proton model has hadronic and pi0 parts to be mapped to components; to be checked against `proton_model` before any claim).

## 4. What has to be produced (Geant4 tagging run; needs approval before launch)

A small calibration sample, generated in the usual Geant4 11.4.1 FTFP_BERT configuration, with a stepping-level scorer that adds, per cell, the all-material deposit and the fibre deposit **split by component** (the four classes above, with the kinetic-energy cut for class 2/3 recorded in the output metadata).

| item | proposal |
|---|---|
| particles | electrons, protons (interacting and crossing kept), plus **muons and charged pions** as extra checks that the components are particle-independent (a pion shower must be explained by the same component ratios as a proton shower) |
| energies | 5, 10, 20, 50, 100, 200 GeV (two more than the present four so interpolation can be tested on genuinely held-out energies) |
| events | 5,000 per particle and energy for electrons and protons; 2,000 for muons and pions |
| seeds | new base seeds registered in the manifest; sealed seeds untouched (DECIDE-2) |
| physics list | FTFP_BERT as in the main sets; the hadronic component is the physics-list-dependent part, so a QBBC copy at two energies is added to measure how much the hadronic ratio depends on the list (the systematic) |
| cost | to be estimated from the existing wall times and quoted before launching |

## 5. Acceptance criteria (written before any fit)

1. **Component closure (in-sample energies):** per-layer mean fibre energy within 3% (the existing tolerance, D6.8), per particle and energy, on held-out events.
2. **Leave-one-particle-out:** the model fitted on electrons + protons (+ muons) predicts **pions** within 3% per layer at every energy. This is the test of "particle-agnostic". If the criterion fails, the components are not sufficient and the design is rejected, not tuned.
3. **Leave-one-energy-out:** fitting without 20 GeV, without 50 GeV and without 200 GeV, the held-out energy is predicted within 3%. This is the test of smoothness in energy. (The present class-table model gives 8-9% and fails.)
4. **Mixed-shower check:** protons, with pi0 energy known from truth, are reproduced with the same component ratios used for electrons and charged hadrons (no proton-specific parameter).
5. Descriptive, no threshold: cell-value quantiles, per-event relative width of the total fibre energy, the tail; the present model's known weaknesses (median 7.5-13% low, width 15% high at 10 GeV) are compared.
6. Physics-list sensitivity: difference of the hadronic ratio between FTFP_BERT and QBBC reported as a systematic.

A result that fails (2), (3) or (4) is reported as a negative finding and the fallback (below) is used.

## 6. Alternatives considered

| option | verdict |
|---|---|
| one table per particle class (the current electron R-A, plus a proton one) | works where the class is known; fails for mixed showers and does not give one model; keep as the **fallback** and for the electron-only analyses |
| deposit-only features (voxel structure, neighbours) | **tested and rejected** (E1) |
| learn `R` directly with a neural network from `D` | would hide the physics and fail the same tests (E1); rejected |
| model fibre energy from fibre positions (microscopic) | accurate but needs the Geant4 geometry inside a fast model; defeats the purpose of a fast simulation |
| abandon R-A for hadrons: run Geant4 for all events | always valid but costs the very simulation budget FastMC is meant to save; kept as the validation reference |

## 7. Risks and limits

- The component definition may not be sufficient (E2 residual R2 shows the pi0 fraction alone is not). Criteria 2-4 are designed to find out cheaply.
- The hadronic components depend on the Geant4 physics list; absolute values carry a physics-model systematic that no AMS data constrain here.
- Fibre-level microphysics that Geant4 gets wrong (Birks saturation in polystyrene, not modelled in these files) will propagate: the sampling ratio of slow, densely ionising hadrons is the most sensitive; flagged, not corrected.
- Tagging changes the stepping action: it must not alter the physics. A control run with and without tagging at one energy (same seeds) must reproduce the existing grids exactly.

## 8. Decisions requested

| # | question | proposal |
|---|---|---|
| PA.1 | Accept the component-wise design and acceptance criteria of section 5 | yes |
| PA.2 | Approve a Geant4 tagging run as specified in section 4 (compute budget to be quoted before launch) | yes, after the cost quote |
| PA.3 | Components 1-4 as listed, with the class 2/3 cut set from the data before fitting | yes |
| PA.4 | Keep the per-class tables as the fallback and for electron-only studies meanwhile | yes |
| PA.5 | Include pions and muons only as validation particles | yes |

## 9. PA.1-PA.5 approved by the researcher on 2026-10-12; cost quote for PA.2 (not yet launched)

**Basis (measured):** mean CPU seconds per event of the existing development sets (`cpu_seconds` in the event files; 12-14 workers on a 16-logical-CPU machine, wall time = total CPU / workers within 1%): electrons 2.51 s (10 GeV), 6.18 (20), 9.74 (50), 30.0 (100); protons 0.39, 0.64, 1.42, 2.31 s. **Extrapolated, not measured:** 5 GeV and 200 GeV (about 0.25-0.30 s/GeV for electrons, 0.023 s/GeV for protons); muons and pions (muons negligible; charged pions assumed 1.2 x protons); the **tagging overhead is unknown and assumed +30%** until the control run measures it.

| item | events | CPU time |
|---|---|---|
| electrons, 6 energies | 2,000 each (proposed; 5,000 each would be 2.5 x) | about 220,000 s (61 h) |
| protons, 6 energies | 5,000 each | about 48,000 s (13 h) |
| charged pions, 6 energies | 2,000 each | about 23,000 s (6 h) |
| muons, 6 energies | 2,000 each | about 2,000 s |
| protons, QBBC, 10 and 100 GeV | 5,000 each | about 13,500 s (4 h) |
| **subtotal** | | about 307,000 s (85 h) |
| **with +30% tagging overhead** | | **about 400,000 s = 111 CPU-hours, about 8 hours wall at 14 workers** |

With 5,000 electrons per energy instead of 2,000: about 830,000 s = 230 CPU-hours, about 16.5 hours wall. 2,000 is proposed because the electromagnetic component needs only a sampling ratio per layer and size class, not a rare-event tail.

**Other costs:** the 100 and 200 GeV electron points are 70% of the electron time. Storage: only component deposit and fibre grids are stored (no fibre or fine-voxel lists), about 40 KB per event, roughly 3 GB in total. My effort before launch: implement the component scorer in `geant4_backend.py` (about a day with tests), plus a control run (tagged versus untagged, same seeds, one energy) that must reproduce the existing grids exactly before anything else is generated (about 10 minutes of compute).

**Proposal to cut the cost if wanted:** drop 5 GeV and 200 GeV for electrons (leave-one-energy-out then uses 20 and 50 only): saves about 40% of the electron time (about 35 h CPU, 2.5 h wall). Not recommended, because 200 GeV is the interpolation test furthest from the fitted energies.

## 10. Scorer built, control passed, run launched (2026-10-12, 20:48 UTC)

**Implementation** (`geant4_backend.py`, opt-in `component_tagging: true` in the pilot config; off by default so every existing configuration is unchanged): extra Geant4 primitive scorers with particle and kinetic-energy filters on the fibre detector and on the prefix voxel mesh, scored in C++ (no Python in the stepping hot path). Stored arrays `component_deposit_grid_mev` and `component_readout_grid_mev`, shape `(events, 7, 18, 72)`, channels `em, had_low, had_mid, had_high, ion, neutron, other`.

**Deviations from section 3 (disclosed).**
1. The "neutron-induced" component of section 3 cannot be tagged by a particle filter (a recoil proton is a proton). It is replaced by a `neutron` channel (energy booked to neutron steps, mostly local deposits of recoils below the production threshold) plus the species and energy classes; whether neutron recoils explain the high hadronic ratio is still untested and is now testable only through the slow-proton channel.
2. Charged hadrons are split into three kinetic-energy bins, [0, 20), [20, 200), [200, inf) MeV, instead of a single cut; the cut is chosen by merging bins before fitting (PA.3 spirit, not letter).
3. `ion` matches only the generic ion and light species (alpha, deuteron, triton, He3). Heavy fragments carry their own particle names and fall into `other`, which is the unfiltered grid minus the six scored channels (floored at zero). In 10 GeV protons `other` holds 3.0% of the deposit and 5.0% of the fibre energy, so it is a physical channel and will be fitted, not ignored.

**Control (registered before the run: tagged output must reproduce the existing grids exactly).**
- Electrons, 10 GeV, first 100 existing events by recorded seed: deposit and readout grids **identical** to the stored development file.
- Protons, 10 GeV, first 300 recorded seeds: tagged and untagged reruns are **identical (300 of 300)**; both differ from the stored file in 6 of 300 events (2%). Reason not found; same seeds, different worker count (4 against 14). **Reproducibility caveat for the existing proton development set, not caused by tagging; to be recorded in the data manifest notes.** Electron events reproduce exactly.
- Overhead: about 0% (tagged 1.90 s/event electrons, 0.33 s protons against 2.32 and 0.35 for the same events untagged; differences are within run-to-run noise). The +30% budgeted in section 9 is therefore not expected; the wall time of the full run should be below 8 hours.
- Early observation, not a result: the electromagnetic channel of 10 GeV proton showers has R/D = 0.066, equal to the electron shower value 0.065.

**Run.** `bash configs/run_component_tagging.sh 14` (resumable; muons, protons, pions, protons with QBBC, electrons last), configs `configs/geant4_component_{electron,proton,pion,muon,proton_qbbc}.yaml`, base seeds 20273001-20273005 (not used elsewhere), output `data/geant4_component_*` (git-ignored), logs in `data/geant4_component_logs/`. Nothing here is sealed data. The machine is held awake by a PowerShell process for up to 12 hours or until the run ends.

## 11. Fit rules registered BEFORE the component fit (2026-10-15)

Written after the integrity table of the tagged run (per-component R/D by particle) was seen and before any fit or closure was computed. One fit as registered; a failure is reported and a new registered version is proposed, nothing is retuned on the same data.

**Model (mean model, the object criteria 1-4 test).** For component `c` in {em, had_low, had_mid, had_high, ion, neutron, other}, layer `l`, deposit-size class `b` (the 25 classes of `SIZE_EDGES_MEV`): `ratio[c,l,b] = sum R / sum D` over fit cells with `D > 0`. Predicted fibre energy of a cell = `sum_c D_c * ratio[c,l,b(D_c)]`. No energy and no particle label enters. Fallbacks, in order, when a class has fewer than 400 cells: `(c,b)` pooled over layers, then `(c)`. The share of predicted energy served by each fallback level is reported. Fibre energy in a cell where a component has `R > 0` and `D = 0` cannot be predicted and is reported as orphan energy.

**Charged-hadron cut.** The three kinetic-energy bins stay separate, because their pooled ratios differ by far more than 3% (about 0.23, 0.09, 0.074 in the integrity table). Merging is not used.

**Fit and test sets.** Split by `event_index % 4`: fit parts 0 and 2, validate parts 1 and 3 (parts 1 and 3 are also reported separately as a noise indicator, not used for pass/fail).
1. Criterion 1: fit = electrons + protons (FTFP_BERT, all six energies, even event_index); test = their odd events, per particle and energy.
2. Criterion 2: fit = electrons + protons (all events); test = pions, per energy. Muons are predicted and reported, outside pass/fail.
3. Criterion 3: for E in {20, 50, 200} GeV, fit = electrons + protons at all other energies; test = electrons and protons at E (pions reported).
4. Criterion 4 (mixed shower): fit = electrons + pions (all events); test = protons, per energy. No proton-specific parameter.
5. Criterion 6: the FTFP_BERT model applied to QBBC protons at 10 and 100 GeV, plus per-component mean ratios of the two lists.

**Pass rule.** Maximum over the 18 layers of |predicted / Geant4 - 1| <= 0.03 at every energy tested; any layer above 0.03 fails that test for that energy. Tolerance and class edges are fixed here.

**Not part of this step.** The stochastic scatter (quantile tables) and the descriptive criterion 5 are a second step after criteria 1-4, using the same classes.

## 12. Result of the registered component fit (2026-10-15): the criteria are NOT met

One fit, as registered in section 11 (`src/ams_ecal/detector/component_sampling_response_validation.py`, results `results/detector_response/component_ra_criteria.json`; component sums equal the total fibre energy to 2e-4 MeV). Maximum over 18 layers of |predicted / Geant4 - 1|, tolerance 3%:

| criterion | rows passing | range of max closure (electrons, protons) |
|---|---|---|
| 1. in-sample energies, odd events | 5 of 12 | 2.3% to 7.3% |
| 2. leave-one-particle-out, pions (muons reported) | pions 0 of 6; muons 1 of 6 | 3.0% to 5.3% (pions) |
| 3. leave-one-energy-out (20, 50, 200 GeV) | 1 of 9 | electrons 3.9% to 5.9%, protons 2.2% to 3.6% |
| 4. mixed shower, fit without protons, predict protons | 2 of 6 | 2.0% to 8.9% |
| 6. FTFP_BERT model on QBBC protons | 10 GeV 3.8%, 100 GeV 4.9% (both above 3%) | |

**Reading, with what is established and what is not.**
- Established (data): the model is better than the earlier tables by a wide margin where they are comparable (the electron table on protons was 39-47% off; leave-one-energy-out for the table model was 8-9%; here 2-6%), but it does **not** meet the registered 3% in most rows. By the registered rule, criteria 1-4 are reported as failed and the per-class tables stay the fallback (section 5).
- Established (diagnostic, exploratory): the error is almost entirely in the **electromagnetic component** (5 GeV electrons: -4.4% of the total energy, all of it in the EM channel). The EM ratio R/D at large cell deposits is not constant: electrons 0.068 (5 GeV) to 0.061 (200 GeV), protons 0.067 to 0.063. The pooled table, dominated by higher-energy showers, is therefore too low at low energy and too high at high energy. The two half-samples of electrons and protons agree with each other (e.g. 0.072 and 0.074), so these failures are systematic, not noise.
- Hypothesis (untested): the EM ratio depends on the stage or energy scale of the shower (the energy spectrum of the electrons that cross the cell), which the local deposit size does not carry. The earlier event-total class absorbed exactly this dependence, which is consistent with the hypothesis but does not prove it. Note that the electron and proton EM ratios agree with each other at matched energy (about 0.067-0.068 at 5 GeV; 0.061-0.063 at 200 GeV): the energy-scale dependence looks shared across particles, which keeps the particle-agnostic idea open.
- Limits of this test: pions and muons have only 2000 events per energy, and their two half-sample closures differ by 3-10 percentage points, so a 3% tolerance cannot be resolved with them; criterion 2 is under-powered, not decisive. Fallback classes (pooled over layers or components) serve up to 19-28% of the predicted energy in some rows, which may add to the failures (sparse high-deposit classes); this was not controlled in this step.
- Physics list: the mean ratio of the slow-proton/hadron, ion and neutron channels differs between FTFP_BERT and QBBC (had_low 0.224 against 0.213, ion 0.125 against 0.152, neutron 0.487 against 0.458, other 0.150 against 0.128 at 10 GeV; EM, had_mid, had_high equal to about 0.5%). This is the hadronic-model systematic, a measured difference between two Geant4 lists, not a validated uncertainty.

**Not done / next candidate (needs the researcher's decision before any run).** A version 2 with one extra context variable that is smooth in energy (a continuous function of the shower's total deposit, instead of the discrete event classes), registered together with its criteria before the fit; or accept the per-class tables as the working R-A and treat the component result as the documented negative/partial finding. Nothing was retuned after seeing these numbers.

## 13. Proton class table (researcher's go 2026-10-15): fitted and validated under the same registered rule; 3 of 4 energies fail

`python -m ams_ecal.detector.sampling_response_validation --particle proton`: per-class quantile tables (same model as the electron R-A) fitted on even-`event_index` protons of the development-large set (10, 20, 50, 100 GeV), validated on odd events, 3% per layer at every energy (D6.8; nothing retuned).

| energy | max per-layer closure | verdict |
|---|---|---|
| 10 GeV | 8.3% (layer 0, over-predicted) | fail |
| 20 GeV | 3.0% (layer 0; 0.0300 against the 0.03 limit) | pass at the limit |
| 50 GeV | 5.2% (layer 2) | fail |
| 100 GeV | 5.3% (layer 2) | fail |

The relative width of the event-total fibre energy is reproduced (model 1.21 / 1.30 / 1.39 / 1.44 against Geant4 1.22 / 1.31 / 1.39 / 1.45). The proton table applied to electrons is 105-148% off, as expected (scope limit). **Reading (hypothesis, untested):** the proton sample mixes crossing protons (minimum-ionising, essentially one cell per layer) with interacting showers; the event-total class does not separate them. A proton table fitted separately for crossing and for interacting events (the truth flag exists in the files) is the natural next version; it would be registered before its fit, and needs the researcher's go. Until then the proton table is **development only, not meeting the registered closure**.
