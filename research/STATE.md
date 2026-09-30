# AMS ECAL Research — Current Research State

_Last human review: 2026-09-21_
_Last agent update: 2026-09-30 (Block 6A complete; Geant4 proton pilot complete; Block 6B Slice 0-2 recorded; D1/D2/D4/D5/D6 accepted by the researcher, conditional on a literature check)_

## Central research question

What physically meaningful multiscale information in electromagnetic and
proton-induced showers survives AMS-02 ECAL-like readout, how much does it add
beyond conventional calorimeter observables, and can that validated structure
later motivate useful classical or quantum inductive biases?

The accepted program order is:

```text
establish shower structure
→ test detector survival
→ quantify information content
→ classical ML benchmarks
→ QML on validated structure
```

## Immediate research question

**RQ-001**

> To what extent do electromagnetic and hadronic particle showers exhibit
> discriminative multiscale spatial structure, how much of this structure
> survives AMS-02 ECAL-like segmentation, and what information does it
> contribute to electron/positron-versus-proton classification beyond
> conventional calorimetric observables and standard ML representations?

See:

`research/questions/RQ-001_multiscale_shower_information.md`

## Current hypothesis

**H-001 — E0**

Electron/positron and proton-induced showers may produce different
scale-dependent spatial energy distributions that remain measurable after
finite detector readout.

Candidate observables include:

- generalized dimensions `D_q`;
- partition-function scaling `tau(q)`;
- multifractal-spectrum summaries where justified;
- lacunarity;
- entropy/concentration across scale;
- occupancy and energy moments under coarse-graining.

This remains a hypothesis.

The project does **not** currently claim that AMS showers are mathematical
fractals, that a stable multifractal spectrum exists, or that AMS granularity is
sufficient to resolve one.

## Null / competing explanations

Apparent multiscale differences may instead arise from:

- energy mismatch;
- incidence-angle mismatch;
- containment or boundary effects;
- ordinary shower width/depth differences;
- longitudinal leakage;
- finite segmentation;
- thresholds/noise;
- preprocessing choices;
- unstable finite-resolution estimators;
- simulator artifacts.

## Current evidence level

**E0 — literature-motivated AMS-specific research hypothesis**

Real highly granular calorimeter studies motivate fractal-sensitive shower
analysis, but the hypothesis has not yet been established for AMS-02 ECAL-like
readout in this project.

---

# Accepted publication direction — 2026-09-21

The first intended publication is now a **multiscale shower-information study**,
not a QML benchmark.

Paper 1 will ask whether scale-dependent/fractal-sensitive shower structure:

- exists robustly in detailed electron/proton transport;
- survives AMS-like segmentation;
- contains information beyond conventional shower variables;
- improves or changes classical ML behavior, especially in low-data regimes.

FastMC is a smooth phenomenological simulator and control. Because its
longitudinal/lateral parameterizations intentionally replace microscopic
branching with smooth distributions, FastMC alone cannot establish or falsify
microscopic fractal/multifractal shower structure.

The decisive simulation chain for RQ-001 is:

```text
fine-grained Geant4
→ same events projected to AMS-like 18 × 72
→ multiscale-preservation analysis
→ FastMC smooth-control comparison
→ classical information / ML study
```

QML is deliberately downstream.

Paper 2+ will test quantum architectures only after Paper 1 establishes what
multiscale structure exists, survives readout, and is useful.

See:

`research/PUBLICATION_ROADMAP.md`

---

# Conceptual research framing

The project distinguishes:

- **multiscale/fractal-sensitive shower structure:** active physics question;
- **deterministic chaos:** adjacent mathematical direction, not currently an
  AMS shower claim;
- **QML:** a later computational method that may exploit validated structure,
  not evidence that the physical structure exists.

The project must not infer deterministic chaos from branching, irregularity, or
visual complexity.

See:

`research/CHAOS_FRACTALS_QML.md`

---

# Current development state

Stage II FastMC Blocks 0–5 are complete.

The repository currently contains:

- structured AMS-02 ECAL geometry;
- tracker projection;
- alternating ECAL readout mapping;
- canonical `ECALEvent`;
- deterministic longitudinal EM gamma profile;
- deterministic lateral EM profile;
- explicit finite-depth longitudinal leakage;
- explicit finite-width lateral leakage;
- deterministic track-centered `18 × 72` lateral fractions;
- **stochastic electromagnetic event generation (Block 6A)**, with per-event seed and
  configuration provenance.

Current FastMC notebooks:

- `00_ecal_calorimetry_and_geometry.ipynb`
- `01_tracker_state_and_projection.ipynb`
- `02_readout_orientation_and_cell_mapping.ipynb`
- `03_canonical_event_model.ipynb`
- `04_ecal_geometry_fidelity.ipynb`
- `05_longitudinal_em_shower.ipynb`
- `06_lateral_em_shower.ipynb`
- `07_stochastic_em_events.ipynb`

## Block 6A — IMPLEMENTED 2026-09-21, amended and committed 2026-09-28 (`fe0e699`)

Implemented from the accepted model without reopening the fluctuation literature.

New files:

- `src/ams_ecal/stochastic.py` — `StochasticEMShowerModel`
- `tests/test_stochastic.py` — 25 tests
- `notebooks/07_stochastic_em_events.ipynb`

Changed:

- `configs/fastmc.yaml` — new `stochastic_em` section, schema_version 2 -> 3
- `src/ams_ecal/fastmc_config.py` — `StochasticEMConfig`, reusable `config_digest`
- `src/ams_ecal/longitudinal.py` — gamma integration exposed at an explicit shape
  parameter so the deterministic and stochastic paths share one implementation

Full suite: 226 passing, ruff clean. Notebook executes end to end.

Verified numbers at 100 GeV: `s = 0.1069`; `T_bar = 8.985 X_0`; sampled `E[T0] = 8.987 X_0`,
so the centring identity holds; ensemble containment sits `-0.0049` below the deterministic
value, the sign Jensen's inequality predicts; width-law validity floor at 56.2 MeV.

Committed in `fe0e699` (amended documentation `8ad8cae`, `934d295`).

### Block 6A — COMPLETE 2026-09-29 (`d8c10fc`)

Closing slice of the 2026-09-29 plan. No physics reopened: the implemented model is the
2026-09-28 regime amendment, which supersedes the 2026-09-21 equations quoted in that plan
(offset -0.5 paired with the sampling width is the mismatched pair the amendment removed).

Added the three checks the plan required and the suite lacked:

- **energy accounting closes exactly** - primary = in-grid + out-of-back + out-of-sides, to
  1e-12, replaying each event from its own seed;
- **longitudinal leakage is the event's own gamma tail** beyond 17 X_0 at its *sampled* alpha;
- **lateral leakage grows toward the side** at an unchanged longitudinal draw.

Notebook 07 gained a cell showing the same three-way split for its demonstration event
(100 GeV: 94,355.0 + 5,609.7 + 35.2 = 100,000.0 MeV). Suite: 233 passing, ruff clean.

## Next engineering target

Block 7 is **blocked** pending the evidence pass below. Block 6B (protons) has no accepted
model. The next engineering step is therefore a human decision, not code.

---

# Accepted evidence and decisions — Block 6

> [!warning] Superseded in part on 2026-09-28
> The equations in this section are the model as **accepted on 2026-09-21**. Three things have
> changed since, and the amendments below the Block 7 section are authoritative where they
> differ:
>
> 1. `T = ln(E/E_c) - 0.5` was believed to be the AMS-consistent mean. Verification showed AMS
>    publishes **no** mean-depth formula, so this value is PDG and is now the `deposition`
>    regime only.
> 2. The mean and the fluctuation width are a **matched pair** selected by a `regime`. The
>    default `sampling` regime uses offset `-0.812` plus a geometry-derived `-0.353` shift, and
>    width coefficients `(-2.5, 1.25)`; `deposition` uses `-0.5` and `(-1.4, 1.26)`.
> 3. Excluding a fluctuating beta is not a simplification away from AMS. AMS holds `b = 0.65`
>    fixed for all showers and all energies, so Block 6A matches AMS.
>
> Retained unchanged: the single stochastic variable `T0`, the lognormal with centring
> `mu = ln(T_bar) - s^2/2`, `alpha = 1 + beta*T0`, the entry-referenced origin, and the
> deterministic lateral profile.

## Mean longitudinal model

The mean longitudinal profile is a gamma distribution with:

```text
T = (alpha - 1) / beta
```

so:

```text
alpha = 1 + beta*T
```

For the AMS ECAL implementation:

```text
beta = 0.65
T(E) = ln(E / E_c) - 0.5
```

The `-0.5` offset corresponds to the electron **energy-deposition** maximum,
not the older Rossi Approximation-B electron-number maximum.

The published AMS longitudinal form is algebraically consistent with the
current implementation.

## Beta provenance

`beta = 0.65` has peer-reviewed AMS provenance.

AMS fits shower parameters on observed showers while keeping the detector-
specific scale parameter fixed at `b = 0.65`.

The project therefore keeps beta fixed in the first stochastic FastMC model.

## Origin convention

The accepted Block 6A baseline is **detector-entry referenced**.

Do not sample an explicit shower-start depth in the first model.

A first-bremsstrahlung-relative alternative may remain architecturally possible
for later Geant4 comparison, but it is not part of the baseline.

## Accepted Block 6A stochastic model

For each event:

```text
T_bar(E) = ln(E / E_c) - 0.5

s(E) =
    1 / (-2.5 + 1.25 * ln(E / E_c))

mu(E) =
    ln(T_bar(E)) - 0.5 * s(E)^2

ln(T0) ~ Normal(mu(E), s(E)^2)

alpha_event =
    1 + 0.65 * T0
```

Then:

1. integrate the gamma profile over the 18 finite readout intervals;
2. convert fractions to stochastic layer energies;
3. distribute each layer energy using the existing deterministic lateral
   profile;
4. preserve longitudinal and lateral leakage;
5. produce an `ECALEvent` with explicit RNG provenance.

### Provenance of the fluctuation width

The `ln(T)` width law is a **transferred approximation** from Grindhammer &
Peters' sampling-calorimeter parameterization.

It is not an AMS-specific fitted fluctuation law.

This is accepted because FastMC is deliberately a simple phenomenological
generator whose adequacy will later be tested against Geant4.

### Explicitly excluded from Block 6A

- fluctuating beta;
- explicit shower-start sampling;
- independent per-layer random draws;
- a two-variable `(T, alpha)` fluctuation model;
- a new stochastic lateral model;
- microscopic transport;
- attempts to uniquely decompose every latent variance source.

## Acceptance philosophy

FastMC only needs to be:

- fast;
- reproducible;
- physically defensible;
- free of obvious simulator artifacts;
- realistic enough for controlled downstream benchmarks;
- transparent about transferred approximations.

The definitive test of whether a simplification matters is later comparison
against detailed transport and, where possible, real detector/test-beam
information.

---

# Block 6A amendments — 2026-09-28

Three changes, all from reading Grindhammer & Peters and the AMS sources directly.

## 1. AMS publishes no mean-depth formula. VERIFIED.

Kounine et al., NIM A 869 (2017) 110-117, p. 113, and Aguilar et al., Physics Reports 894 (2021)
section 1.7.1, p. 19: the AMS form is

```text
dE/dt = E0 (b t)^(b T0) b exp(-b t) / Gamma(b T0 + 1)
```

i.e. `alpha = 1 + b*T0` exactly, with **b = 0.65 fixed for all showers and all energies** and
`T0` obtained per shower by fitting observed ECAL cell deposits. There is **no** published AMS
`T_bar(E)`. Two consequences:

- `T = ln(E/E_c) - 0.5` was never AMS; it is PDG, and the choice of mean is ours to make.
- Changing `T_bar` does **not** break agreement with the published AMS functional form, which
  holds for any `T0`. The Block 4 consistency claim survives.

## 2. The rho = 1 concern is resolved in favour of the current design.

Because AMS itself holds `b` fixed and fits `T0` per shower, Block 6A is structurally the same
model AMS uses. The Grindhammer & Peters two-variable correlated `(ln T, ln alpha)` model is the
outlier, and its `alpha` parameterization has no AMS provenance. The lost degree of freedom is
real - about 12% conditional spread in `alpha` at fixed depth, worth roughly +/- 2% in contained
energy - and remains worth a Geant4 sensitivity test, but it is **not** a departure from AMS.

## 3. Regime switch: sampling versus true deposition.

The mean depth and the fluctuation width are a **matched pair**; mixing them is incoherent, and
the previous configuration did mix them. `configs/fastmc.yaml` now carries a top-level `regime`:

| regime | describes | offset | s(E) at 100 GeV | T_bar at 100 GeV |
|---|---|---|---|---|
| `deposition` | true deposition, the **perfect event** | PDG -0.5 | 0.0948 | 8.985 X_0 |
| `sampling` (default) | signal-level longitudinal shape | G&P -0.812 plus a geometry shift of -0.353 | 0.1069 | 8.319 X_0 |

The sampling depth shift is computed from `configs/geometry.yaml` (`F_S = 4.897`,
`e/mip = 0.651`), so it tracks the detector description rather than being hard-coded.

**The perfect event is recoverable from the seed.** The generator draws one standard normal
variate from the seed *first* and applies the regime transformation after, so a single seed names
a corresponding pair of events. `model.true_deposition().generate_event(..., random_seed=s)`
returns the true-deposition event behind the sampled event that seed `s` produced. Verified in
the notebook: the quantile `z` agrees to twelve decimal places across regimes.

**Block 7 boundary, now explicit.** Under `regime: sampling` the sampling distortion of the
longitudinal shape is already applied, so Block 7 must not re-apply the depth shift or the extra
shape fluctuation. Under `regime: deposition` nothing detector-related has been applied and
Block 7 owns all of it. Building Block 7 against `deposition` shrinks the double-counting surface
to the fitted-parameter question alone.

Schema: `configs/fastmc.yaml` is now version 5. Suite: 230 passing, ruff clean.

## Stage 18 ("Learn again") — COMPLETE for Block 6A, 2026-09-28

The post-build teach-back skipped on 2026-09-21 has run. Recorded in Obsidian at:

```text
01 Projects/AMS ECAL QML/Tutor Sessions/
  2026-09-28 Block 6A Teach-Back and Post-Build Probe.md
```

21 questions, 7 teaching nodes, teach-back written by the researcher. Two corrections to earlier
material, both documentation or teaching, neither a code defect:

- The 2026-09-20 tutor session claimed the ensemble mean must reproduce the deterministic profile.
  That holds only for `T0`, the centred variable. Layer fractions and containment differ by
  Jensen's inequality (100 GeV, 4000 events: containment -0.0039, about 10 standard errors;
  layer 1 +130%; layer 10 -2.7%). Corrected by banner in that note. The code and tests were
  already right: they assert centring on `T0` only.
- Notebook 07 cell 21 and the 2026-09-21 research note describe the deterministic-limit ratio 26.9
  as an amplification constant. It is layer 1's sensitivity (13.2) times `|z| = 2.041` for the
  worst of five seeds. The proportionality assertion remains valid. **Fixed 2026-09-28** at the
  researcher's request: the notebook now prints the decomposition (26.9 = 13.2 x 2.041), and the
  research note carries a correction banner.

## Open questions raised by the teach-back — NOT decided

All category B: recorded, configurable, not blocking. None reopens DEC-001.

1. **What `T_bar` is the mean of.** Grindhammer and Peters define `-0.812` through `<ln T>` and
   generate `ln T = <ln T> + sigma*z` with no centring (arXiv:hep-ex/0001020 section 3.1, Eq. 11;
   appendix A.1.2), so `exp(<ln T>)` is their **median**. The `sampling` regime centres it as a
   mean and lands `exp(-s^2/2)` shallower than the source: 0.048 X_0 at 100 GeV, 0.15 X_0 at
   1 GeV. The PDG `-0.5` in `deposition` is the maximum of the average profile, neither mean nor
   median of per-shower `T`. Interpretation of verified source text.
2. **The regime gap is half physics, half convention.** At 100 GeV the regimes differ by 0.665 X_0:
   0.353 is the G&P sampling shift, 0.312 is PDG `-0.5` versus G&P `-0.812` for the same
   homogeneous physics. Applying G&P's own shift to `deposition` gives 8.63 X_0, not the
   `sampling` regime's 8.32. Relevant to Block 7, which is to be built against `deposition`.
3. The ×27 misreading above (documentation only).
4. The scope of "AMS publishes no mean-depth formula": verified in **two** AMS publications, and
   should be quoted with that scope.

Documentation fixes applied 2026-09-28 at the researcher's request: the stale
`TWO KNOWN INCONSISTENCIES` comment in `configs/fastmc.yaml` rewritten (comment only, no values
changed, but the configuration SHA-256 is now `78db5fc2...`); notebook 07 re-executed; garbled
LaTeX repaired in notebook 07 and in DEC-001.

---

# Geant4 proton calibration pilot — COMPLETE 2026-09-29, awaiting the 6B model decision

Branch `geant4-proton-pilot`. Slices 2-10 of the 2026-09-29 plan are done; the plan now stops
at **STOP FOR MODEL DECISION**. Block 6B is **not** implemented. Geant4 is a model, not
detector truth: every number below is Geant4 11.4.1 in our implemented material model.

Record: `results/geant4_proton_pilot/` (summary.json, CSV tables, figures 1-10, geometry
audits), `notebooks/08_geant4_proton_pilot.ipynb`. Raw batches in `data/geant4_proton_pilot/`
(ignored by git; regenerate with `uv run --group geant4 python -m ams_ecal.geant4_backend all`).
Report: `uv run python -m ams_ecal.pilot_report`.

## Infrastructure (reusable for RQ-001)

- `geant4_pybind` 0.1.3 (Geant4 **11.4.1**), optional group `geant4`; datasets in
  `~/.geant4_pybind`. Real C++ transport; energy scoring in C++; Python per event, per new
  track, and per step of the primary only.
- Geometry derived from `configs/geometry.yaml` (single source of dimensions): 43,155
  explicit fibres in a density-matched lead+glue matrix; optional extension of 126 further
  AMS-like superlayers. Geant4 overlap check: **no overlaps** (1000 points/volume); Geant4
  mass 475.4153 kg equals the analytic composition; 16.68 X_0, 0.618 lambda_I (geometric).
- Per event: `readout_grid_mev` (**detector image: scintillator only**), `deposit_grid_mev`
  (truth accounting), `edep_scintillator/matrix/total` (close to 1e-9), fine fibre and
  3 x 3 x 4.625 mm voxel deposits, first-interaction truth from the primary's
  **step-defining process** (position, depth in mm/X_0/lambda_I, energy before, material,
  every product), cross-checked by a secondary-based finder (agree exactly on occurrence and
  depth). Metadata: commit, Geant4 + dataset versions, physics list, **production cut 0.7 mm
  and its energy thresholds**, config SHA-256, approximations.
- Researcher's four pre-baseline gates passed and are tests: scintillator-only readout;
  step-based truth; geometry audit; **worker-independent, bit-for-bit reproducibility**.

## Samples (10 / 20 / 50 / 100 GeV protons, normal incidence)

| sample | list | geometry | events/energy | commit |
|---|---|---|---|---|
| baseline | FTFP_BERT | AMS-only | 4000 | `00d70fd` clean |
| fixed_entry | FTFP_BERT | AMS-only, entry at cell centre | 1000 | `00d70fd` clean |
| alternate | QBBC | AMS-only | 1000 | `00d70fd` clean |
| extended | FTFP_BERT | prefix + 126 superlayers | 800 | `00d70fd` clean |
| high_energy_model | QGSP_BERT | AMS-only | 1000 | `cb7b7dc` clean |

QGSP_BERT was **added after** the QBBC run, beyond the researcher's plan: Geant4's own model
table shows FTFP_BERT and QBBC both use FTFP for protons above 3 GeV, so QBBC could not test
the model of the first interaction. QGSP_BERT uses QGS above 12 GeV with the same
cross-sections as FTFP_BERT.

## Answers to the plan's §32 questions — PROPOSED, not accepted

Analysis choices fixed before reading production data: MIP scale measured from truth
non-interacting events; thresholds 0.5 x MIP (varied 0.25-1); MIP band = 99% quantile of
non-interacting scintillator energy (95%, 99.9% varied); S_D = bias-adjusted epsilon^2 over
10 equal-count depth bins (5, 20 varied); 95% bootstrap / Wilson intervals.

1. **Interaction.** P(no inelastic) = 0.530 / 0.512 / 0.516 / 0.526 (CI half-width ~0.016)
   vs exp(-0.6) = 0.549; lambda_eff = 262 / 249 / 251 / 260 mm, so the prefix is
   0.64-0.67 lambda for protons; no energy trend resolved. Depth exponential (KS p 0.25-0.91).
2. **Detector level.** Crossing protons give 11.4-13.1 MeV scintillator (0.56 MeV per crossed
   cell; 7.4% of the ~150-190 MeV they deposit). Detector MIP-like fraction = truth
   non-interacting fraction within ~1 point. Only 1.4-3.6% of inelastic events are MIP-like,
   concentrated in the last ~2 cm. 7-10% of first interactions are quasi-elastic-like, yet
   still light the fibres.
3. **Energy distribution.** Interacting visible energy is broad and left-tailed: the
   log-energy residual at fixed D has sd 0.76-0.91 and skewness -1.5 to -2.2 - not
   lognormal; a continuous low-visible tail, not a separate mode.
4. **P7 (S_D).** Longitudinal centre ~0.70-0.74 (D dominates *where*); hits 0.36-0.57;
   visible energy 0.24-0.50; longitudinal RMS 0.33-0.48; width 0.00-0.03. **P7 is partly
   supported**: D is necessary, not sufficient, and irrelevant for width.
5. **Truncation.** Same events: r(D, E) = -0.44 to -0.71 in the prefix, **+0.18 to +0.28**
   in the full shower; hits flip likewise; S_D(full) < 0.15 for everything. Event ordering by
   prefix energy is anti-correlated with full-shower energy (-0.20 to -0.37). Deep-calorimeter
   correlations **do not transfer**.
6. **Physics list.** Robust across FTFP_BERT / QBBC / QGSP_BERT: interaction probability,
   depth law, crossing protons, the qualitative P7 pattern. **Model-dependent**: the
   visible-energy scale at 20-100 GeV (QGSP_BERT medians ~13-28% lower; KS p 1e-6 at 20 and
   50 GeV) and shower width (~8% narrower).
7. **Controls.** Fixed entry changes the crossing-proton signal significantly (fibre-lattice
   phase is part of the MIP fluctuation) but not interacting events.

## Block 6B structure proposed from the pilot — FOR THE RESEARCHER'S DECISION

I (Bernoulli, lambda_eff ~255 mm) → D (truncated exponential) → independent per-layer
crossing-track deposits before D (measured distribution; adjacent layers rho ~0.1-0.18) →
at D: log visible energy = mu(E, remaining depth) + eps, with a heavy-tailed eps of
(exploratory) nearly energy-independent shape; longitudinal shape starting at D with its own
residual latent; hit multiplicity following visible energy (rho ~0.8-0.9); lateral width with
its own latent, ~independent of D. Rejected by the data: fixed visible fraction, flat
individual profile, truncated whole-shower profile, transferred correlation matrix, separate
low-visible class. **Open for the decision:** tabulated (empirical quantile) vs parametric
families; how to carry the FTFP_BERT/QGSP_BERT energy-scale systematic.

## New open questions (recorded, not decided)

- **Backsplash**: material behind the ECAL raises the prefix signal of protons that interact
  downstream from 11-13 to 26-45 MeV (34-47% above the MIP band) and makes prefix width
  D-dependent. What lies behind the real AMS ECAL, and does it matter for AMS protons?
- **Hadronic-model systematic** on the visible-energy scale (see 6).
- **Entry-phase effect** on crossing protons: should FastMC marginalize it or model it?
- **Incidence angle**: only normal incidence studied.
- **Production-cut sensitivity**: required before any sub-cell / multiscale claim (researcher,
  gate point 5).
- **Material systematic**: composition-matched alternative (`relative_volume`, or AMS's later
  ~58/33% description quoting 0.7 lambda_I) not yet run.
- 98 lead + 1 aluminium foil drawn as a homogeneous matrix (systematic to revisit).

---

# Block 6B Slice 0 — dependency analysis COMPLETE 2026-09-29; decisions ACCEPTED 2026-09-30 (conditional)

Full record: `research/plans/2026-09-29_block6b_slice0_dependency_analysis.md`. Reproduce:
`uv run python -m ams_ecal.proton_dependency` and `uv run python -m ams_ecal.proton_checks`.
Everything is Geant4 11.4.1 in this project's material model (E3), on **calibration events only**:
`event_index % 4 == 3` is held out for validation (`src/ams_ecal/proton_calibration.py`).

**Observed (calibration events).**

- The post-interaction amplitude (fibre energy in the three layers after the interaction layer) is
  nearly independent of interaction depth (rho -0.09 to +0.03) with an energy-independent
  distribution shape; its scale grows as about E^0.54.
- The layer profile after the interaction is a universal function of the layer offset from the
  interaction layer (first principal component 79-91% of the log-profile variance).
- Upstream of the interaction the layers carry real shower activity (excess 10-18 MeV in the layer
  just before it), rank-correlated 0.5-0.7 with the amplitude: **not a clean MIP track**.
- Exact fibre geometry is validated (0.195-0.204 MeV/mm in crossed fibres vs 0.205 for a MIP) but
  explains only 3-5% of crossing-proton variance; delta rays and small cascades carry 17-24%.
- Production cut 0.7 -> 0.1 mm: no material change to hits, width or small-deposit structure at the
  resolution of a 750-event paired check. Material `relative_volume`: a predictable -2.6% in
  P(no inelastic); readout representation unaffected, deposition representation +5-7% for MIPs.

**Accepted by the researcher, 2026-09-30, with one condition.** In the researcher's words: "I
accept the proposals you make but you should check the literature if our model makes sense. You
should especially check studies about preserving the spatial structure and the fractal connection
we are looking for. Afterwards tutor me on it again in the learn-back stage." So D1, D2, D4, D5
and D6 below are ACCEPTED, and the model's plausibility against the literature (independent
discovery, then verification and an adversarial pass, per `RESEARCH_PROTOCOL.md` section 2) is an
open condition: a mismatch is a finding to reconcile, not something to smooth over. The
learn-back tutoring follows the literature check.

**Sequence set by the researcher, 2026-09-30** (steps marked done):

1. commit the current evidence - DONE (`da8268f` code, `4f194c0` evidence);
2. rebuild the calibration artifact from that clean commit - DONE (`a5e7be3`; content hash
   unchanged, `539eaab0...`, built from `4f194c0` with no tracked changes);
3. freeze D1/D2/D4/D5/D6 in this file - DONE (this section);
4. implement the D5/D6 crossing repair;
5. run a second-look diagnostic;
6. build the interacting model under D1;
7. optimize the batch path;
8. freeze everything;
9. generate a fresh Geant4 test set;
10. final Slice 6 validation on that fresh set.

The fresh test set (steps 9-10) removes the integrity problem of re-using the held-out events
after the crossing repair: the 2026-09-29 crossing validation was the only look those events get.

Decisions (numbers refer to the Slice 0 record, section 9 and 11):

- **D1** replace the plan's `P(E_vis | E, R)` table and R-conditioned templates by *amplitude x
  universal profile + upstream albedo + one lateral-scale latent*, with hit multiplicity emergent.
- **D2** couple the deposition and readout representations through shared latent draws; until an
  electron fibre-energy scale exists, the common e/p representation is `deposition`, because
  Block 6A's `sampling` regime moves the shape to signal level but not the energy scale.
- **D4** pre-registered validation contract for Slice 6.

**Slices 1-2 implemented (crossing branch) and validated ONCE on held-out events** - result: the
interaction draw and the per-layer response pass at the noise floor; the crossing branch **fails**
the event-total (readout at 50-100 GeV, deposition at all energies), hit-multiplicity, max-cell
and containment rows (KS 0.10-0.37). Both were predicted from calibration data: independent layers
and cells restricted to crossed fibres. Accepted repairs, not yet implemented: **D5** an event-level
burst latent; **D6** a per-layer lateral spill. Interacting events (Slices 3-5) are not
implemented (D1 accepted; they follow the crossing repair). Speed is 11-14 ms/event, too slow for
large datasets (step 7 of the sequence). Whole suite: 442 tests. Record: section 11 of the Slice 0
file. The calibration artifact was rebuilt from a clean tree at `a5e7be3`.

# Block 7 status — BLOCKED pending evidence, 2026-09-21

Detector response remains planned, and is now explicitly **blocked** rather than merely
unstarted.

An independent-discovery and source-verification pass ran on 2026-09-21. It did **not** reach
specialist validation, adversarial review, or a plan. Recorded in Obsidian at:

```text
01 Projects/AMS ECAL QML/Evidence Maps/
  2026-09-21 Detector Response and Double Counting in Fitted Shower Parameters.md
```

## What the verification established

**Verified, AMS-specific.** AMS corrects dead-cell and side-leakage energy by integrating a
parameterized shower profile over the missing region, and applies the correction to measured cell
energies before the reconstructed total is formed (Zhang et al., Chinese Physics C 40 (2016)
096204, eqs. 1-2). Its longitudinal parameters were fitted to 180 GeV electron test-beam data;
its lateral parameters came from Geant4 simulation. AMS also corrects a photomultiplier
position effect at **layer** level (Li et al., Chinese Physics C 37 (2013) 026201, eq. 6).

**Not established.** That reusing such parameters as if they described true deposition introduces
quantifiable bias. No source states this. It remains a **reasonable inference**, supported
indirectly by Grindhammer & Peters section 3.5, and it must not be promoted further.

## Consequent decision, PROPOSED not accepted

Block 7 implementation does not begin until the adversarial pass has run and, for each candidate
Block 7 effect, it is stated whether that effect was already inverted out of the deposits AMS
fitted its shower parameters to. Adding an effect AMS never removed is safe; adding one AMS
corrected away is a double-counting candidate.

## Two findings that rebound onto Block 6A

Both surfaced from reading Grindhammer & Peters directly, both category **B** under the Research
OS triage, neither blocking:

1. **Mean-depth convention.** Appendix A.2.3 confirms `sigma(ln T) = (-2.5 + 1.25 ln y)^-1` is the
   **sampling-calorimeter** coefficient set, so the FastMC configuration comment is correct. But
   the same appendix corrects the *mean* depth for sampling geometry, and FastMC does not apply
   that correction. For this geometry the gap is an energy-independent **0.665 X_0**, about 0.70
   of a readout layer, against a 1-sigma T0 spread of 0.96 X_0 at 100 GeV. Block 6A therefore
   pairs a homogeneous-convention mean with a sampling-convention width.
   **This is a discrepancy, not yet an error** - AMS fitted `b = 0.65` with its own convention.
   Deciding which mean is right for AMS is a human decision and a one-line config change.

2. **Imposed correlation.** Grindhammer & Peters fluctuate `ln T` and `ln alpha` as a correlated
   pair with `rho = 0.784 - 0.023 ln y`, deriving beta per event. Fixing beta and setting
   `alpha = 1 + beta*T` is the special case `rho = 1`. At 100 GeV the source value is **0.57**, so
   Block 6A over-correlates shower depth and profile shape.

Both are recorded in `configs/fastmc.yaml` next to the affected constants.

## Remaining potential effects

- sampling / visible-energy response;
- noise;
- thresholds;
- gains;
- saturation;
- dead channels;
- calibration effects.

Because some AMS shower parameters were fitted to observed deposits, overlap
with later response modeling must be watched for possible double counting.

This is a Block 7 concern, not a Block 6A blocker.

---

# Block 8 status

FastMC dataset generation and validation remains planned.

Dataset validation must include:

- energy/geometry matching between classes;
- seed isolation;
- split integrity;
- containment/leakage checks;
- response distributions;
- absence of accidental label leakage.

---

# RQ-001 / multiscale direction

## First decisive experiment

**EXP-001 — detailed-transport multiscale preservation study**

FastMC is not the physical evidence source for RQ-001.

Minimum comparison:

1. fine-grained Geant4 shower deposits;
2. the same events projected to the canonical `18 × 72` representation;
3. FastMC as a smooth phenomenological control.

Interpretation:

- signal in fine Geant4 but not AMS projection:
  detector segmentation removes the relevant scales;
- signal in fine Geant4 and AMS projection but not FastMC:
  FastMC masks physically accessible multiscale structure;
- comparable relevant observables in projected Geant4 and FastMC:
  FastMC preserves enough structure for that tested purpose.

Ordinary agreement in mean profile, containment, or energy resolution does not
answer this question.

## Candidate multiscale observables

Only after estimator validation:

- `D_0`, `D_1`, `D_2`, selected `D_q`;
- `Z_q(epsilon)`;
- `tau(q)`;
- multifractal-spectrum summaries where scale support is adequate;
- lacunarity;
- entropy/concentration;
- occupancy/energy moments under controlled coarse-graining.

## Open questions

- Is the accessible AMS-like scale range large enough for stable estimation?
- Which `q` values are numerically meaningful?
- How should alternating X/Y views be handled?
- Should scaling be measured globally, per view, per depth range, or all three?
- How large is finite-resolution bias?
- Do multiscale observables add information beyond shower depth, width,
  containment, and energy?
- Can raw ML learn the same structure implicitly?
- Does explicit multiscale information improve low-data/sample efficiency?

---

# QML status

QML remains in scope as a later publication-level stage.

Accepted order:

```text
detailed transport
→ validate multiscale observables
→ test AMS-readout survival
→ conventional + multiscale classical analysis
→ raw classical ML controls
→ QML/QCNN on a justified representation
```

Candidate future directions:

1. compact validated multiscale inputs for VQCs or quantum kernels;
2. QCNNs as candidate hierarchical/multiscale inductive biases;
3. scale-structured Hamiltonian embeddings;
4. detector-topology-aware quantum connectivity.

No quantum advantage is assumed.

---

# Established project conventions

- AMS-specific detector facts should default to official AMS/collaboration
  sources or peer-reviewed AMS publications.
- Scientific constants and model parameters remain configurable.
- Reusable physics/numerical logic belongs in tested source code.
- Notebooks are for derivation, teaching, visualization, and validation.
- Hypothesis-sensitive literature discovery should be independent of existing
  implementation assumptions when practical.
- Cross-validation means independent reconstruction/reproduction rather than
  agreement checking.
- The human researcher is the final authority on accepted assumptions,
  hypotheses, and conclusions.
- Fractal/multifractal structure may be tested directly; deterministic chaos
  may not be claimed without a justified dynamical-system formulation.

---

# Next session — start here

_Updated 2026-09-29._ Block 6A is committed (`fe0e699`) and pushed to `stochastic-events`; the
mean-depth convention was settled by the regime amendment; stage 18 is complete.

**Researcher decision, 2026-09-28: Block 6B before Block 7.** In the researcher's words: "Let's
start 6B first, block 7 is meaningless before that." Rationale: every e/p comparison in RQ-001 and
Block 8 needs a proton population; detector response is refinement on top of it.

0. **Block 6B structure DECIDED by the researcher, 2026-09-29**: the implementation plan is
   recorded in `research/plans/2026-09-29_block6b_implementation_plan.md` (one exponential draw
   for interaction status and depth; geometry-aware crossing tracks; empirical conditional
   distributions for visible energy and morphology; FTFP_BERT nominal and QGSP_BERT systematic
   calibrations; held-out validation). **Amended 2026-09-29:** protons get both a **true
   deposition** and a **readout (fibre)** representation, like 6A electrons, so e/p events share
   one representation. **Begin its Slice 0 only after the tutoring below.**
   The Geant4 proton pilot is complete.
   Read the pilot section above, `results/geant4_proton_pilot/` and notebook 08. Before the
   physics, the researcher asked to be probed and tutored on the codebase and Geant4:
   `01 Projects/AMS ECAL QML/Tutor Sessions/2026-09-29 Codebase and Geant4 Probe.md`
   (Q1 posted). Then: decide the 6B structure (Slice 11), implement (Slice 12), validate
   against Geant4 distributions and correlations (Slice 13), teach and record (Slice 14).
1. **Block 6B, stage 1 (learning probe)** - paused. Tutor session
   `01 Projects/AMS ECAL QML/Tutor Sessions/2026-09-28 FastMC Block 6B Proton Showers.md`.
   The 2026-09-28 literature pass ended at a blocker: the thin-calorimeter regime is not
   settled by literature, so the Geant4 pilot now precedes any 6B model.
2. Parked: Block 7 tutor session (Q1 posted and waiting); the Block 7 adversarial pass.
3. Queued, not started: open question 9 of the Block 7 evidence map, the depth origin of AMS's
   fitted `T0`, as a full Research OS literature task.

## Next human decision

0. ~~Block 6B Slice 0 outcome~~ - D1, D2, D4, D5, D6 accepted 2026-09-30 by the researcher,
   conditional on the literature check described above; the next steps are the researcher's
   sequence, steps 4-10.
1. ~~Block 6B structure~~ - decided 2026-09-29 by the researcher's implementation plan
   (`research/plans/2026-09-29_block6b_implementation_plan.md`): hybrid of analytic
   exponential interaction depth and empirical conditional distributions; QGSP_BERT as a
   separate calibration scenario, never event noise.
2. Whether backsplash, incidence angle and the material systematic must be answered before
   6B or after it.
3. Carried over: whether to act on open questions 1 and 2 above (which statistic `T_bar` represents; whether
   `deposition` should use G&P homogeneous constants so the regimes differ only by the sampling
   shift). Both are configuration-level.
4. ~~Documentation fixes~~ - approved and applied 2026-09-28.
5. ~~Sequencing: Block 6B versus Block 7~~ - decided 2026-09-28: 6B first.
6. ~~Pre-baseline gates~~ - set by the researcher 2026-09-29; all four passed.
