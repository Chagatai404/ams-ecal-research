# Proton model implementation plan — researcher's specification, 2026-09-29

_Received in chat on 2026-09-29, after the Geant4 proton pilot. Recorded below in full so it
survives across sessions: the researcher's wording is kept, but many of the original's
one-item-per-line code blocks are flattened into prose lists; the pseudocode and the graph are
kept as blocks. The researcher's instruction: implement it **after the tutoring**
(the codebase-and-Geant4 probe in Obsidian,
`01 Projects/AMS ECAL QML/Tutor Sessions/2026-09-29 Codebase and Geant4 Probe.md`).
It is the researcher's decision on the proton model structure proposed in `research/STATE.md`._

## Researcher amendment, 2026-09-29 (during the tutoring)

**Two representations for protons, as for electrons.** the EM event generator offers a `deposition` regime
(true all-material deposit, the "perfect event") and a signal-level regime. The researcher
decided the proton model must likewise provide **true deposition** and **readout (fibre /
scintillator energy)** representations, so electron and proton FastMC events can always be
compared in one common representation. The pilot stores both per event (`deposit_grid_mev`,
`readout_grid_mev`), so the calibration builder produces both from the same events; choosing a
representation is explicit configuration, and provenance records it. Dependency analysis (step 0) must also settle
how this pairs with the EM event generator's two regimes (the EM event generator `sampling` regime moves the longitudinal
*shape* toward signal level but not the energy scale).

---

You are working on my AMS-02 ECAL research repository:
https://github.com/Chagatai404/ams-ecal-research
Your task is to implement the proton model — proton phenomenology using the results of the completed Geant4 proton pilot.
Do not reopen the EM event generator physics.
Do not repeat the broad hadronic literature search.
Do not expand this into the full RQ-001 Geant4 program.
This task is specifically:
Convert the measured thin-ECAL proton behavior from the Geant4 pilot into the simplest transparent, fast, configurable proton FastMC model that reproduces the detector-level distributions relevant to this project.

## 1. Read first

Before editing anything, inspect at minimum: README.md, RESEARCH_PROTOCOL.md, AGENTS.md,
research/STATE.md, research/PUBLICATION_ROADMAP.md,
research/questions/RQ-001_multiscale_shower_information.md, results/geant4_proton_pilot/,
notebooks/08_geant4_proton_pilot.ipynb. Then inspect: src/ams_ecal/, tests/, configs/, existing
FastMC code, the EM event generator implementation, geometry/readout code, ECALEvent, provenance/config
machinery. Use the actual repository state as authoritative. Do not assume filenames or APIs
from this prompt if the repo already contains a better existing abstraction.

## 2. Scientific status entering the proton model

EM event generator is complete. The Geant4 proton pilot is complete using Geant4 11.4.1 and the exact
AMS-like geometry. Approximately 32,000 proton events were generated across 10, 20, 50 and
100 GeV with: FTFP_BERT nominal, QBBC comparison, QGSP_BERT alternate high-energy model,
fixed-entry control, extended calorimeter control. The following pilot results now define the
scientific basis of proton model.

## 3. Interaction probability

The nominal AMS ECAL depth is 166.5 mm. The Geant4 proton samples give approximately no
inelastic interaction: 0.51–0.53, compared with exp(-0.6) ≈ 0.549. The fitted effective proton
interaction length is approximately lambda_eff ≈ 249–262 mm, with lambda_eff ≈ 255 mm as a
useful nominal calibration. The effective proton depth of the ECAL is therefore approximately
0.64–0.67 lambda rather than exactly 0.6 for these Geant4 proton samples. No significant energy
dependence of the interaction-depth distribution was resolved across the pilot energy range.

For the proton model v1: model the first inelastic interaction distance with one exponential
interaction-length parameter. Do not separately sample "interaction yes/no" and then
"interaction depth" unless the code architecture genuinely requires that representation.
Prefer one latent draw:

```
S_int ~ Exponential(lambda_eff)
if S_int >= traversed ECAL material depth:
    no inelastic interaction
else:
    first interaction occurs at S_int
```

This automatically makes interaction probability and interaction depth self-consistent.

## 4. Important provenance of lambda_eff

Do not present ~255 mm as a universal AMS proton interaction length. It is an effective proton
interaction length calibrated from the Geant4 11.4.1 pilot in the current AMS-like
material/geometry model over the tested energy range. Keep it configurable. Its provenance
should include at least: Geant4 version, physics list, geometry/config hash, energy range,
particle, incidence assumptions.

## 5. Current corrected interpretation of the depth-dominance hypothesis

The original proposition "first-interaction depth dominates proton event-to-event variation" is
only partly supported. The pilot found approximately: longitudinal centre S_D ~ 0.72;
hit/cell multiplicity S_D ~ 0.36–0.57; visible/deposited energy S_D ~ 0.24–0.50; lateral
width S_D ~ 0. Therefore the accepted interpretation is: first-interaction depth is the
dominant observed driver of longitudinal position, an important but incomplete driver of
visible energy and multiplicity, and not a meaningful direct driver of lateral width in the
tested normal-incidence thin-ECAL regime. The proton model factorization should reflect this.

## 6. Truncation result

The same-event thin-prefix vs extended-calorimeter experiment showed that correlations from
deep calorimeters do not transfer directly into the AMS ECAL. Interaction depth vs
deposited/visible energy was approximately -0.44 to -0.71 in the AMS-like thin ECAL and
+0.18 to +0.28 in the extended calorimeter; the correlation can reverse sign under truncation.
Consequences: do not import deep-calorimeter covariance matrices, whole-shower longitudinal
parameterizations, or generic hadronic correlation structures into proton model. Use the thin-geometry
pilot directly.

## 7. Detector-level mixture result

Truth non-interacting fraction ~51–53%; detector-level MIP-like fraction almost identical to
truth non-interacting; interacting events that remain MIP-like only ~1.4–3.6%, mostly very late
interactions, roughly within the final ~2 cm. The visible-energy distribution is broad,
continuous, with a heavy low-energy tail, not lognormal. There is no convincing evidence for a
separate low-visible interacting mode. Therefore: do not build an explicit three-class latent
model (non-interacting / low-visible interacting / showering). The low-visible region should
emerge continuously from interaction depth + remaining detector depth + stochastic interacting
response.

## 8. Models rejected by the Geant4 pilot

Do not implement as the proton model baseline: fixed visible-energy fraction; flat per-layer proton
shower; generic whole-hadronic-shower profile truncated at the ECAL edge; transferred
deep-calorimeter correlation matrix; independent low-visible interaction class; lognormal
visible-energy distribution; interaction-depth-controlled lateral width.

## 9. Important pre-implementation synthesis check

Before writing the final generator, use the existing Geant4 events to perform one small
dependency analysis. Do not generate another large production sample for this. The purpose is
to determine which conditional dependencies need to remain explicit. At minimum examine
relationships among: remaining depth R, visible/deposited energy E_vis, longitudinal centre,
longitudinal RMS, hit/cell multiplicity, lateral width, where R = L - S_int for interacting
events. Specifically determine whether, after conditioning on R and/or E_vis, there remains
substantial dependence between: lateral width and E_vis; hit multiplicity and longitudinal
morphology; longitudinal shape and E_vis. Use simple interpretable diagnostics (binned
conditional correlations, partial correlations where justified, conditional
medians/quantiles, simple 2D plots). Do not fit complex ML dependency models. The purpose is
merely to decide which arrows belong in the proton model generative graph. Record this result before
implementing the final factorization.

## 10. Proposed high-level proton model generative graph

```
primary energy E, track state / geometry
  -> interaction distance S_int
       -> S_int >= detector path -> crossing-track event
  -> remaining depth R
  -> visible interacting response E_vis
       -> longitudinal morphology
       -> hit/cell activity
       -> lateral development
```

Do not treat this diagram as immutable. Use the dependency analysis from §9 to finalize the
minimal conditioning structure.

## 11. Crossing-track / non-interacting component

Do NOT implement the pre-interaction component as "each layer receives an IID random MIP
deposit" without accounting for geometry. The fixed-entry Geant4 control showed the signal of a
crossing proton depends on where the track lies relative to the fibre pattern. Interacting
shower events were much less affected, but crossing tracks were sensitive to entry/fibre phase.
Therefore the baseline should conceptually be: known TrackState → exact geometry / fibre
crossing → expected sensitive path / cell placement → stochastic crossing-track deposition. The
track determines which cells are crossed, sensitive path length, fibre/view geometry. Then
stochastic fluctuations are applied around that geometrically defined response. Reuse the
existing geometry/readout machinery wherever possible. Do not introduce a second approximate
detector mapping.

## 12. Pre-interaction track before a shower

For an interacting event (S_int < L), generate the crossing-track component only up to the
interaction point. The event should contain pre-interaction MIP-like track + post-interaction
hadronic activity. Do not generate a full detector-crossing MIP track and then add a shower on
top (double-counting). Keep energy components conceptually clear.

## 13. Visible interacting energy

For interacting events R = detector path remaining after first interaction. Visible/deposited
energy depends substantially on primary energy E and remaining depth R and has broad stochastic
spread. Model conceptually E_vis ~ P(E_vis | E, R). Do not use a fixed fraction of E and do not
assume a named probability family unless the pilot data support it.

## 14. Tabulated versus fitted distributions — use a hybrid

Fit the physics that has a clear justified analytic form; use empirical conditional
distributions for the rest. Fit analytically: interaction distance exponential(lambda_eff).
Potentially fit simple smooth trends (conditional median / scale vs E and R) only if a simple
fit describes the Geant4 distributions well. For broad stochastic distributions such as visible
energy, prefer initially empirical conditional CDFs or conditional quantile tables, e.g.
u ~ Uniform(0,1), E_vis = Q_vis(u | E, R). This avoids imposing an unsupported lognormal /
gamma / Gaussian / mixture on a distribution already shown to be broad and non-lognormal.

## 15. Interpolation

Anchor energies 10, 20, 50, 100 GeV. Do not simply snap every event to the nearest energy table
if interpolation is straightforward. Likely: interpolation in log(E) for energy-dependent
quantities, and remaining-depth bins or simple interpolation in R. Keep interpolation
deterministic and testable. Do not extrapolate beyond the validated range unless explicitly
configured and visibly labelled. For proton model v1, restricting the validated domain to 10–100 GeV is
acceptable. Outside-domain use should raise, warn clearly, or require explicit extrapolation
configuration, following existing project conventions.

## 16. Longitudinal morphology

Strong dependence on first-interaction/remaining depth. Model the shower portion relative to
the interaction point rather than detector entry when appropriate: longitudinal_shape ~
P(shape | E, R, E_vis) depending on the dependency analysis. The profile should begin at the
interaction location, respect finite detector boundaries, preserve back leakage, and sum
consistently to the generated interacting visible energy. Do not use a whole-shower profile
whose natural maximum lies beyond the detector and merely truncate it.

## 17. Longitudinal representation

Choose the simplest representation that reproduces the Geant4 thin-ECAL layer distributions:
empirical normalized layer-fraction templates conditioned on R/E; low-dimensional basis of
normalized profiles; or simple shifted profile with stochastic width. Priority:
interpretability, energy conservation, few parameters, reproducibility, conditional agreement
with Geant4. If empirical conditional layer-fraction distributions work well enough, use them
as the v1 reference; a future simplified analytic fit can replace them if validation shows no
meaningful loss.

## 18. Hit/cell multiplicity

Partly controlled by interaction depth, strongly associated with visible energy. Do not draw it
independently unless the dependency analysis supports independence. Likely N_hit ~
P(N_hit | E_vis, R, ...) or allow multiplicity to emerge naturally from the spatial
energy-deposition model. Avoid a separate random variable if already determined by E_vis,
longitudinal morphology, lateral deposition, threshold definition. Prefer emergent behaviour.

## 19. Lateral width

Depth-associated variance ~ 0. Do not directly condition width on first-interaction depth.
Zero dependence on depth does not imply full independence: check whether width depends on E,
E_vis, hit multiplicity, longitudinal morphology; choose the minimal conditioning. A plausible
form is w_lat ~ P(w_lat | E, E_vis), not automatically accepted — use the §9 analysis.

## 20. Spatial deposition

The generated event must populate the canonical 18 × 72 representation, reusing TrackState,
geometry, readout orientation, cell mapping. The generator must not become only a list of
summary observables; it must generate physically plausible event-level cell energies suitable
for classical ML, multiscale analysis, later QML, FastMC-vs-Geant4 comparison, while remaining
phenomenological rather than microscopic.

## 21. Energy accounting

Preserve the EM event generator's explicit energy accounting. Distinguish conceptually: sensitive/readout
deposit; passive-material deposit if represented internally; longitudinal leakage; lateral
leakage; energy carried beyond the detector. Do not pretend FastMC can reconstruct the
microscopic nuclear invisible-energy decomposition unless explicitly modeled. Do not introduce
a constant "42% invisible-energy" factor.

## 22. Physics-list model dependence

Qualitative structure robust across FTFP_BERT, QBBC, QGSP_BERT, but the absolute visible-energy
scale showed model dependence (QGSP_BERT ~13–28% lower at ~20–100 GeV). This is epistemic
model uncertainty, not event-level variance. Do not average physics lists, randomly choose a
list per event, or inflate per-event variance. Implement separate calibration variants: nominal
FTFP_BERT-derived; systematic QGSP_BERT-derived; QBBC-derived comparison where useful (e.g.
`proton_model_calibration = "ftfp_bert" | "qgsp_bert"`). QBBC need not become a production
calibration if it adds no distinct high-energy structure, but preserve its pilot result.

## 23. Why QGSP_BERT remains

QGSP_BERT was added because QBBC and FTFP_BERT use closely related high-energy proton treatment
in the pilot regime. It provided a genuinely different high-energy model and exposed the
visible-energy-scale systematic. Keep the result.

## 24. Do not treat Geant4 as truth

Documentation and code should say "Geant4-calibrated" / "Geant4-derived phenomenology", not
"true proton distribution". Later AMS/test-beam evidence may alter the calibration.

## 25. Small remaining checks before freeze

25.1 Production-cut sensitivity: compact sample with a reasonable alternate production cut;
focus on hit/cell multiplicity, lateral width, small-deposit structure. If negligible relative
to stochastic/model uncertainty, record and stop. Do not optimize cuts.
25.2 Material-composition systematic: compact comparison for the 98 Pb + 1 Al foil
approximation and any relevant material-composition simplification; check lambda_eff,
visible-energy scale, key morphology. If negligible, keep as documented approximation; if
material, expose as calibration uncertainty or correct the geometry if the correct description
is established.
25.3 No broad incidence-angle campaign yet: architect the API so the model can later condition
on track angle and path length.

## 26. Backsplash

Extended geometry: 26–45 MeV returning into the ECAL region vs 11–13 MeV without. Do not use
extended-geometry ECAL lateral width as a proton model calibration source; use the AMS-only simulation
for width. Backsplash is not a proton model blocker; record as a caveat.

## 27. Event-level algorithm target

```
INPUT: primary energy E, TrackState, ECAL geometry, RNG, proton calibration config
COMPUTE: path through ECAL L
DRAW: S_int ~ Exponential(lambda_eff)
IF S_int >= L:
    crossing-track response using exact track/fibre geometry + stochastic per-crossing response
ELSE:
    crossing-track deposits only before S_int
    R = L - S_int
    E_vis ~ P(E_vis | E, R)
    longitudinal morphology conditioned on the validated minimal dependencies
    lateral morphology using validated minimal dependencies, no unjustified depth dependence
    project/distribute into canonical cells
    preserve finite detector leakage/accounting
RETURN: ECALEvent with provenance
```

The implementation should reuse existing abstractions and may not literally mirror this.

## 28. Correlation preservation

Do not independently sample every observable. Reproduce important Geant4 correlations; at
minimum validate depth↔visible energy, depth↔longitudinal centre, depth↔longitudinal RMS,
visible energy↔hit multiplicity, visible energy↔lateral width if present. Only preserve
correlations that are material. No generic covariance engine.

## 29. Empirical calibration artifacts

Do not hard-code large tables into Python. Store calibration data in an explicit versioned
form (configs/, data/calibration/, results/.../calibration/ per conventions) with provenance:
source Geant4 run, physics list, energy anchors, geometry hash, sample counts, generation date,
schema version. The runtime model loads a compact finalized artifact, not raw Geant4 events.

## 30. Calibration construction must be reproducible

Add a script/module that rebuilds the proton model calibration artifact from the accepted pilot outputs
(pilot events → calibration builder → versioned compact proton model calibration). Deterministic where
possible.

## 31. Train/calibration versus validation split

Deterministic split of the existing pilot data into calibration and validation subsets (event
IDs or another reproducible method). The validation subset stays untouched while choosing /
interpolating the model where practical. Not a full ML train/val/test system.

## 32. Proton model validation

Compare FastMC protons with held-out Geant4 AMS projections at matched energy, normal
incidence, entry conditions. At minimum: interaction (no-interaction fraction, depth
distribution); crossing tracks (total signal, layer distribution, entry/fibre-phase
dependence); energy (distribution, median, quantiles, tail); longitudinal (layer fractions,
centre, RMS, observed maximum, active layers); lateral (width, core concentration, cell
distribution); activity (hit multiplicity, maximum cell, sparsity/concentration); correlations
(depth↔energy, depth↔centre, depth↔RMS, energy↔multiplicity, relevant lateral). Do not
validate only means.

## 33. Statistical comparison

Quantile tables, ECDF comparisons, Wasserstein distance, KS where appropriate, correlation
difference, conditional distribution plots. No single arbitrary "proton model realism score"; aspects
pass or fail separately.

## 34. Acceptance philosophy

Acceptable when fast, reproducible, transparent, physically defensible, free of obvious
generator artifacts, reasonably faithful to the important thin-ECAL proton distributions, and
preserving structures relevant to downstream controlled studies. Do not endlessly tune
secondary statistics. Apply Research OS triage A (blocker) / B (testable uncertainty) /
C (reversible choice) / D (future refinement).

## 35. Important failure criterion

If the simplest conditional/tabulated model cannot reproduce the held-out joint distributions
without a large complicated latent architecture: STOP. Report what fails, why, which
observable/correlation cannot be represented, what minimal additional variable would fix it.
Do not silently grow proton model into a large generative ML model, normalizing flow, diffusion model or
high-dimensional copula.

## 36. Tests

Reproducibility (same seed → same event; different seed → variation). Interaction (S_int
exponential; no-interaction ⇔ S_int >= L; interaction point inside the detector). Geometry
(track deposits in valid cells; pre-interaction deposits stop appropriately; entry/fibre phase
affects crossing response as expected). Energy (nonnegative, finite, explicit accounting, no
accidental renormalization). Empirical distributions (sampling within calibration support;
monotonic quantile/CDF interpolation; boundary energies). Configuration (nominal vs systematic
selectable; provenance preserved). Event model (valid ECALEvent; 18×72). Ensemble sanity
within expected tolerances at representative energies; avoid brittle exact-moment tests.

## 37. Performance

Calibration loads once; sampling cheap; no per-event optimization or large data loading.
Benchmark enough for large downstream datasets. Do not micro-optimize prematurely.

## 38. Provenance

Every proton event identifies: FastMC version, the proton model model version, calibration version,
nominal/systematic physics-list calibration, geometry/config hash, seed, primary energy,
TrackState. If EventProvenance cannot represent the calibration version cleanly, extend it
minimally.

## 39. Documentation wording

Use: Geant4-calibrated proton phenomenology; effective interaction length; empirical
conditional distribution; physics-list systematic; validated over 10–100 GeV at normal
incidence. Avoid: true proton shower model; exact AMS proton response; physical interaction
length of AMS.

## 40. Teaching notebook

After code and validation: teach (1) why a thin hadronic calorimeter needs a different model
than a contained shower; (2) exponential interaction depth; (3) why one exponential draw
determines both status and position; (4) truth non-interacting vs detector MIP-like;
(5) remaining depth R; (6) the depth-dominance hypothesis variance-decomposition result; (7) why longitudinal and
lateral variables need different conditioning; (8) why empirical conditional CDFs instead of
forced analytic laws; (9) why physics-list differences are epistemic systematics, not event
noise; (10) how the generator maps these ideas into 18×72 events; (11) where the model is
validated and where uncertain. The notebook calls tested production code.

## 41. Research-state update

When proton model passes validation, update research/STATE.md. Accepted: exponential effective
interaction depth; geometry-aware crossing-track response; continuous interacting
visible-energy tail; remaining-depth-conditioned longitudinal model; no direct
interaction-depth control of lateral width unless new analysis supports it; FTFP_BERT nominal
calibration; QGSP_BERT energy-scale systematic. Rejected: fixed visible fraction; flat shower
profile; whole-shower truncation model; transferred deep-calorimeter correlation matrix;
explicit separate low-visible interacting class; lognormal visible-energy law. Still open:
angular dependence; real AMS backsplash/environment; production-cut sensitivity if not fully
resolved; material-composition systematic; exact Pb/Al construction approximation; external
validation against AMS/test-beam data. Keep Geant4-model dependence explicit.

## 42. Scope boundary

Not in this task: the detector response; full RQ-001 multifractal analysis; e-vs-p ML
benchmarks; QML; full angular proton calibration; Geant4 production redesign; learned
generative shower model.

## 43. Implementation slices

- Dependency analysis (step 0) — reconcile and dependency analysis: inspect pilot artifacts and FastMC
  architecture; compute remaining conditional dependencies; run the small production-cut /
  material checks; write the final proton model dependency graph. Do not implement the generator until
  this is clear.
- Interaction draw (step 1) — calibration builder: pilot events → deterministic builder → compact versioned
  artifact; nominal FTFP_BERT first, then QGSP_BERT systematic; tests.
- Crossing branch (step 2) — interaction + crossing-track component: S_int ~ Exp(lambda_eff);
  geometry-aware crossing-track deposits; non-interacting branch; pre-interaction track branch.
  Validate before adding shower deposition.
- Step 3 — interacting visible energy: P(E_vis | E, R) by empirical conditional
  sampling/interpolation; validate on held-out events.
- Step 4 — longitudinal morphology: simplest accepted conditional model; interaction-origin
  consistency, finite-depth behaviour, energy accounting; validate.
- Step 5 — lateral/activity model: only supported dependencies; width not driven by depth;
  multiplicity emergent where possible; validate.
- Final validation (step 6) — joint validation: matched FastMC vs held-out Geant4; marginals, conditionals,
  correlations, tails; resolve only material discrepancies.
- Alternative physics-list calibration (step 7) — systematic calibration: validate the QGSP_BERT path as an explicit epistemic
  scenario.
- Step 8 — notebook and documentation; reproduction commands and validation results.

## 44. Final report before declaring the proton model complete

Model (exact factorization); calibration (tabulated vs fitted); interaction (final lambda_eff,
uncertainty/provenance); crossing tracks (geometry dependence); visible energy
(sampling/interpolation); longitudinal (how D/R enters); lateral (dependencies and why);
correlations reproduced; systematics (FTFP_BERT vs QGSP_BERT); validation (held-out Geant4);
domain (energy range, angle range, geometry assumptions); performance; remaining material
limitations; files/commits; exact test results.

## 45. Success criteria

1. one exponential draw determines interaction status and depth; 2. crossing tracks use
detector geometry, not IID layer noise; 3. interacting visible energy follows the measured broad
conditional distribution; 4. low-visible events emerge without a forced class; 5. longitudinal
morphology depends strongly on interaction/remaining depth; 6. lateral width not driven by
depth; 7. important correlations reproduced; 8. empirical distributions built reproducibly;
9. calibration and validation separated; 10. FTFP_BERT nominal; 11. QGSP_BERT as a model
systematic, not event variance; 12. valid canonical 18×72 ECALEvents; 13. explicit
energy/leakage accounting; 14. fast enough for dataset production; 15. documented as
Geant4-calibrated phenomenology; 16. simple enough to replace/refine later.

Central design rule: use analytic physics where the pilot established it, and empirical
conditional distributions where the physics remains phenomenological. Do not invent a prettier
probability law than the evidence supports.

Begin with the dependency analysis (step 0).
