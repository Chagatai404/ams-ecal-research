# Block 6B — literature check, adversarial pass and validity of decisions D1–D6

_Written 2026-09-30 by Claude as primary worker. Agent report: the researcher decides what is
accepted. Evidence labels follow `RESEARCH_PROTOCOL.md` §3 and §4; every statement about the
literature carries the level at which it was checked (§1)._

The researcher's request (2026-09-30): "I accept the proposals you make but you should check the
literature if our model makes sense. You should especially check studies about preserving the
spatial structure and the fractal connection we are looking for. Afterwards tutor me on it again
in the learn-back stage." and, later, "Continue with the adversarial pass meanwhile and finalize
the decision validities afterwards".

---

## 1. Method and contamination control

| stage | who | what they were given |
|---|---|---|
| independent discovery | two `literature-scout` agents in fresh contexts | only neutral questions and the minimum detector regime (lead / scintillating fibre, ~17 X0, ~0.6 lambda_I, 9 mm cells, alternating views, 10–100 GeV). **Not** given: our factorization, equations, citations or results. Forbidden to open the repository or the paper library. |
| source verification | four `source-verifier` agents, plus my own reads | the claim, the candidate pointer and the regime only, not the scouts' reasoning |
| reading of primary text | me | Ruan et al. PRL 112, 012001 (library copy, pp. 1–3); CALICE Si-W ECAL hadron interactions, arXiv:1203.1240 (pp. 1–4); the two other `Fractals/` papers in the library (pp. 1–2) |
| adversarial | `adversarial-reviewer` in a fresh context | the claim under test, measured facts and the literature status, not our reasoning |
| runnable adversarial checks | me, calibration events only | the reviewer's runnable tests that were sound (§3) |
| reconciliation | me, with the researcher's library | the inherited citations, used only now |

Every arXiv identifier that appears below was confirmed to exist, with a matching title, against
arXiv's own metadata (48 of 48). That is a statement about existence and titles only.

**Verification levels used below.** *Full text*: a verifier or I read the passage. *Abstract*: only
the abstract / metadata was accessible. *Existence*: identifier and title confirmed, content not
checked. *Not verified*: taken from a scout only.

---

## 2. What was established, by question

### 2.1 Preserving spatial structure in fast simulation

| claim | source | level | note |
|---|---|---|---|
| Community metrics for fast calorimeter simulation include **sparsity** (defined as one minus the fraction of voxels per layer above threshold), **layer correlations** (Pearson), high-level-feature histograms, **classifier-based separation** (binary AUC, multiclass log-posterior), FID-like and manifold metrics | CaloChallenge 2022, arXiv:2410.21611, Sec. 8.1–8.5 | full text (verifier, library copy) | the paper does **not** rank the metrics by sensitivity |
| Generative models mis-model sparsity (CaloDiffusion, CaloINN), specific-layer patterns (CaloScore) and shower widths (CaloINN) | arXiv:2406.12898 | full text (HTML) | qualitative; three models, one family of datasets |
| Fast-simulation validation is dominated by 1-D histograms of high-level features | arXiv:2406.12898, arXiv:2410.21611 | full text | consistent with our contract |
| Numbers such as "30–50% overpopulation of cells" or "hit occupancy mismatch ≤ 10%" | attributed to CaloClouds, AtlFast3 | **not verified** — not found in the accessible text | do not cite |
| "Classifiers trained on generated showers lose < 5%" | attributed to arXiv:1807.01954 | **contradicted** — that paper generates showers and does not test classifiers; true source unknown | do not cite |
| Parametrised fast simulation lacks explicit cell/layer correlation modelling | arXiv:2303.18150 (review), arXiv:2109.02551 (AtlFast3) | existence only | plausible, consistent with our own crossing result, unverified |

**Reading.** The observables on which our crossing branch failed — sparsity / hit multiplicity,
width and halo, layer correlations — are the same ones the field reports as failure points. That
supports D5 and D6 as repairs aimed at the right structure. It does not show that a repaired
phenomenological model will pass.

### 2.2 Hadronic showers in thin calorimeters

| claim | source | level |
|---|---|---|
| Geant4 lists differ from CALICE data by a few percent to ~10%; Bertini → FTFP transition at 4–5 GeV; QGSP_BERT transitions at ≈ 9.5–9.9 GeV (BERT→LEP) and ≈ 12–25 GeV (LEP→QGS) | arXiv:1510.04063, arXiv:1306.3037 | full text (verifier) |
| Hadron shower = compact electromagnetic core + hadronic halo, from two-component fits to profiles (10–80 GeV, steel) | arXiv:1602.08578 | full text |
| Invisible energy can reach ~40% of the non-EM energy with large event-to-event fluctuations; when e/h ≠ 1 these fluctuations tend to dominate hadronic resolution | arXiv:1704.00661 (Wigmans), Sec. 3.4 | full text; general, **not thin-specific** |
| Events in a thin silicon-tungsten ECAL fall into MIP and elastic (non-interacting) and two interacting topologies ("pointlike", "fireball"); all Geant4 lists reproduce the inelastic frequencies; non-interacting cross sections are well modelled; the transverse profile has two maxima (MIP events near 5 mm, fireball events near 20 mm) | CALICE, arXiv:1203.1240 (LCWS11 proceedings, 2–10 GeV, 1 × 1 cm² pads) | full text, pp. 1–4; **preliminary** |
| Protons vs pions differ in response and shower geometry | arXiv:1412.2653 | existence and general content only; the search assistant's specific numbers were **not** checked |
| Proton–pion separation power and its dependence on segmentation (Deep Sets, 93.8 % at 10 GeV → 67.2 % at 100 GeV) | arXiv:2608.19064 | web-abstract level (verifier) |

**Not found (search scope stated, not a claim of absence).** No external measurement was found of
(i) the non-interacting fraction, (ii) backward-going energy (albedo), or (iii) the visible-energy
distribution for a lead / fibre calorimeter of about 0.6 lambda_I. Searches: two independent
scouts, one hadronic verifier, and my own queries on "thin calorimeter", "albedo", "backsplash",
"non-interacting proton", "minimum ionizing". One scout's statement that fewer than 5% of protons
cross without interacting is wrong: `exp(-0.6)` = 55%, and our Geant4 pilot gives 52%.

**Reading.** The two-population picture (a crossing track and interacting events), the core-plus-halo
lateral shape, a broad left-skewed visible-energy fluctuation and Geant4-list model dependence are
all consistent with what was read. The elements that are new — an amplitude independent of
interaction depth, a profile universal in the offset from the interaction layer, and upstream
activity tied to the shower — have **no external support and no external contradiction** in this
regime. They rest on Geant4 alone (E3) and on one physics list.

### 2.3 The fractal / multifractal connection

| claim | source | level |
|---|---|---|
| A shower "fractal dimension" from **box counting of binary hits** in transverse layers, `FD = <log(N_beta/N_alpha)/log(alpha)> + 1`, box sizes 2–150 mm, separates e, hadron and muon showers; `FD_em = 1.41 + 0.21 log10(E/GeV)`, `FD_had = 1.24 + 0.15 log10(E/GeV)` | Ruan et al., PRL 112, 012001 (2014) = arXiv:1312.7662 | full text (library copy) |
| That work is **Monte Carlo only** ("using Monte Carlo simulation" in the abstract) of a CALICE-type **iron / RPC digital hadron calorimeter** at 1–10 mm cells, 1–80 GeV; no test-beam data | same | full text |
| Its identification cut uses `log10(N_hits) − 2.8·FD`; it shows no fractal-dimension-only versus hit-count-only comparison | same | full text |
| Finite-size and finite-scale artefacts of multifractal estimates are documented; restrict to q > 0; use finite-size scaling; compare with monofractal and shuffled surrogates | arXiv:2603.04609 (2D Ising / MFDFA, abstract), arXiv:1307.2014, arXiv:0807.4854, arXiv:1204.3755, arXiv:1606.02957 | abstract level; **specific to their systems** |
| Cascade / branching arguments give self-similar particle production; intermittency and multifractality have been studied in emulsion and heavy-ion data | arXiv:hep-ph/9605384, arXiv:hep-ph/0006108 (Fractal Electromagnetic Showers), arXiv:1801.03256 | existence only |
| Deep networks on raw cells beat engineered shower shapes | arXiv:1806.05667 (and others) | existence only; not verified |
| A published test of whether a fast or generative simulation preserves multiscale / fractal / intermittency measures | — | **not found** in ≈ 10 differently worded searches by four agents and me. This is "not found in these searches", **not** proof that none exists |

**Two corrections to what the scouts first reported.** The fractal-dimension paper is not a
silicon-tungsten test-beam measurement, and the "fractal dimension" it uses is not energy-weighted.

**Reading.**
- The only calorimeter fractal-dimension result found is a simulation of a different detector, on
  binary hits, with no demonstration of information beyond conventional variables. It shows that the
  approach can work somewhere, not that it works here.
- Our readout (9 mm cells, 72 cells per layer, 18 layers, alternating single-projection views) has
  perhaps two to three usable doublings of scale between a cell and the extent of a shower. The
  estimator literature says that regime demands q > 0, a stated scale range, finite-size checks and
  surrogates.
- RQ-001 stays at **E0**: nothing here promotes it. Nothing refutes it either.
- **FastMC preserving multiscale structure is untested and is not a design requirement.**
  `PUBLICATION_ROADMAP.md` already calls FastMC a smooth control. The literature confirms that
  claim: the FastMC-versus-Geant4 multiscale comparison is a genuinely open question, which is what
  makes it worth measuring.

### 2.4 Scout errors caught by verification (kept so they are not re-imported)

1. The fractal-dimension paper was described as silicon-tungsten test-beam; it is iron/RPC Monte Carlo.
2. "< 5% of protons cross a 0.6 lambda_I calorimeter without interacting": false by a factor of ten.
3. arXiv:1807.01954 was cited for a downstream-classifier result it does not contain.
4. The same identifier, arXiv:2608.19064, was described in two incompatible ways by two scouts;
   verification shows one description was correct (Deep Sets, Pb-W).
5. Several specific percentages (overpopulation, occupancy mismatch, "80 %" shower-start accuracy)
   could not be found in accessible text.

---

## 3. Adversarial pass

The reviewer's headline: the multiscale claim is untested and the interacting branch is unbuilt.
Both are true and already recorded. Triage of its objections:

| objection | verdict | reason |
|---|---|---|
| A1 pooling the interaction length hides an energy dependence | **tested, passes** | likelihood-ratio test for one common rate: LR = 1.81 on 3 dof, **p = 0.61**; per-energy 262 / 253 / 251 / 257 mm, spread 4.7% |
| A2 the 0.7 mm production cut suppresses shallow interactions | rejected | a production cut acts on secondary production, not on the primary's hadronic interaction; the paired 0.1 mm run gave 249 vs 250 mm and 261 vs 261 mm |
| A3–A5 amplitude "independence" is a weak effect and a truncation artefact | **tested, partly right** | see §3.1 |
| A6 the log-profile PCA hides shape dependence | **tested, right in substance** | see §3.2 |
| A8 upstream–amplitude correlation is confounded by primary energy | rejected | every correlation was computed within one energy |
| A9 layer-1 energy is unstable under a 1 mm track offset | not tested | the threshold (30%) is invented; low priority |
| A10 upstream scatter may be correlated between layers | **tested, right** | see §3.3 |
| A11 information lost in binning (< 0.60 threshold) | not tested | invented threshold; the readout representation is fixed by the detector |
| A13 "FATAL: the quanta mechanism is falsified" | rejected | the quanta mechanism is **not implemented anywhere**. The crossing failure came from putting energy only in crossed fibres; D6 addresses that. The report itself later lists no fatal item. |
| M1 the "gate" test: `D_box` at 9/18/36/72/166 mm on held-out events; "flat `D_box` voids the claim"; fractional-Brownian-motion control | **rejected as designed, accepted in spirit** | it mixes the longitudinal (166 mm) and lateral axes, treats a flat `D_box` as meaning something it does not, uses fBm where a multiplicative cascade is the right positive control, uses ad-hoc thresholds (RMSE < 0.2, ±0.15), and uses the held-out events. Its core point stands: **estimator validity on the actual grid has to be shown before any multiscale statement** → D8. |
| "run the multiscale test before Slices 3–5" | rejected | the interacting model is needed for conventional-observable fidelity and for an e/p population regardless of the multiscale outcome |

### 3.1 Amplitude versus depth (calibration events)

| energy (GeV) | Spearman(A3, depth), all | only depth < 60 mm | median A3, late (≥ 100 mm) / early (< 42 mm) | Spearman(A over k = 4..6, depth) |
|---|---|---|---|---|
| 10 | −0.064 | −0.019 | 0.94 [0.84, 1.05] | −0.048 |
| 20 | −0.076 | −0.006 | 0.86 [0.78, 0.98] | −0.056 |
| 50 | +0.029 | +0.008 | 1.01 [0.93, 1.13] | +0.023 |
| 100 | −0.085 | +0.074 | 0.87 [0.80, 0.99] | −0.057 |

Where the back edge cannot act (depth < 60 mm) the amplitude is independent of depth
(|ρ| ≤ 0.07). Near the back it is 13–14% lower at 20 and 100 GeV (lower bounds of the intervals
touch 1) and about 6% and 1% at 10 and 50 GeV. So independence is **conditional on geometry**, as
the reviewer said; the edge factor is part of the design.

### 3.2 What the amplitude × profile factorization leaves unexplained

Residual standard deviation of ln(layer energy) about `A · g(k)`, events with ≥ 8 layers after the
interaction: **0.59 / 0.42 / 0.42 / 0.38** at 10 / 20 / 50 / 100 GeV, i.e. about 80 / 52 / 52 /
46% per layer, with lag-1 residual correlation +0.27 / +0.26 / +0.39 / +0.39. The negative lag-3
values (−0.22 to −0.24) are partly an artefact of removing row and column means from eight
layers (expected ≈ −0.14). The first principal component still carries 79–91% of the variance,
but the residual is not small and is correlated between adjacent layers.

### 3.3 Upstream activity

Spearman between upstream layers `j = 1` and `j = 3`: raw +0.60 / +0.59 / +0.67 / +0.68; **given the
amplitude (10 equal-count bins) +0.43 / +0.39 / +0.45 / +0.39**. The upstream scatter shares a
latent that the amplitude does not explain.

### 3.4 Crossing protons (layers two or more apart)

Median Pearson correlation of fibre energy between layers 0.02 / 0.06 / 0.12 / 0.16 at 10 / 20 /
50 / 100 GeV; the 90th percentile of the pair correlations is 0.06 / 0.19 / 0.37 / 0.48. Median
Spearman is at most 0.09. The coupling is a tail effect that grows with energy — the signature of
the bursts D5 targets.

---

## 4. Validity of decisions D1–D6

"Valid" means: consistent with the literature that could be checked, not contradicted by the
adversarial checks, and safe to build on. "Amended" lists what the evidence says the decision must
additionally contain. None is invalid. Two amendments change the accepted wording of D1, so they
are for the researcher to confirm.

| decision | verdict | evidence for | evidence against / limits | amendments |
|---|---|---|---|---|
| **D1** amplitude × universal profile + upstream albedo + one lateral scale, hits emergent | **VALID WITH THREE AMENDMENTS** | independence of A from depth where truncation is absent (|ρ| ≤ 0.07); pooled λ consistent (p = 0.61); PC1 79–91%; core-plus-halo lateral shape matches CALICE decomposition; two-population picture matches the thin Si-W ECAL study; visible-energy fluctuation width plausible against invisible-energy literature | no external data in this thin regime; Geant4 with one physics list; QGSP_BERT transition zone (≈ 9.5–25 GeV) overlaps the 10 and 20 GeV anchors | **(a)** a back-edge factor on the amplitude window (−13% for the last depth tercile at 20 and 100 GeV); **(b)** a correlated log-normal residual around `A·g(k)` (46–80% per layer, lag-1 ≈ +0.3–0.4); **(c)** an albedo-strength latent shared by the upstream layers (ρ ≈ +0.4 given A), not independent per-layer noise |
| **D2** shared interaction draw only; `deposition` as the common e/p representation | **VALID** | measured fibre–deposit correlation for crossing protons 0.45–0.52 (event total), 0.15–0.24 (layer) | 6A `sampling` regime is not in fibre units, so the readout-level e/p comparison still needs an electron scale | none |
| **D4** pre-registered validation contract | **VALID BUT INCOMPLETE** | rows chosen match the field's failure points | lacks the field's omnibus and correlation metrics, and any scale-dependent row | **D7** below |
| **D5** event-level burst latent (crossing) | **VALID, STRENGTHENED** | event-total variance 1.4–5.5× the layer sum; layer coupling grows with energy (§3.4); layer correlations are a reported failure point of generators | a single-latent burst may not capture 10 GeV single-layer bursts and 100 GeV cascades together | none; validate the two burst types separately |
| **D6** per-layer lateral spill | **VALID** | 17–24% of fibre energy outside crossed fibres; hits 17–22 vs ≤ 19; sparsity and widths are reported failure points | needs the same core-plus-halo kernel as D1's lateral model, calibrated once | none |

**Ordering.** The researcher's sequence is unchanged. D1's amendments enter at step 6 (interacting
model); D7 must be registered before step 9 (the fresh test set) is generated.

---

## 5. Two new proposals (both accepted by the researcher on 2026-09-30, see §7)

**D7 — amend the validation contract (§8 of the Slice 0 record) before the fresh test set exists.**
Additions only; nothing is removed.
1. A **layer-to-layer correlation matrix** row (Pearson, and a tail-sensitive statistic), as in
   CaloChallenge Sec. 8.2. Gated.
2. A **classifier two-sample test** (FastMC versus Geant4 events), reported with the same
   calibration-versus-fresh noise floor, as in CaloChallenge Sec. 8.3. An omnibus measurement of
   how well *joint* spatial structure is preserved; reported, and gated only if you choose.
3. **Sparsity defined as in CaloChallenge** (one minus the fraction of cells per layer above
   threshold), alongside the existing hit-cell row, so the numbers are comparable with the literature.
4. An **ungated multiscale panel**: coarse-graining moments `Σ p^q` (q > 0 only, q ∈ {0.5, 1, 2, 3})
   at lateral aggregation of 1, 2, 4, 8 cells and at layer-pair aggregation, FastMC versus Geant4.
   Reported as a *measurement of difference*, not a pass/fail, because FastMC is a smooth control
   and its difference from Geant4 is a finding for RQ-001. Its interpretation depends on D8.

**D8 — estimator-validity study on the actual readout grid, before any multiscale claim.** This is
step 3 of the multifractal direction in `CLAUDE.md` ("validate estimators on controlled cases").
On synthetic data only (plus calibration events for the smooth null): multiplicative cascades with
known `τ(q)` projected onto the 18 × 72 alternating-view grid as positive controls; a smooth
kernel with Poisson quanta and a within-layer cell-shuffle as negative controls; bias and variance
of the moment slopes against the scale window, for q > 0. Output: the usable scale window, and
whether *any* difference between cascade-like and smooth patterns is detectable at this
resolution. If it is not, no multiscale statement about FastMC or about e/p is meaningful and RQ-001's
multiscale arm needs a finer readout (fibre-level Geant4) than the AMS grid. About one to two days.
It gates multiscale **claims**, not Slices 3–5.

---

## 6. What remains unverified or unknown

- Content of most `Abstract` / `Existence` sources above beyond their abstracts and titles.
- Whether a test of multiscale preservation by fast simulation exists (not found in these searches).
- Any external constraint on albedo, non-interacting fractions and visible-energy distributions for a
  lead / fibre calorimeter of about 0.6 lambda_I.
- Whether the amplitude / profile universality survives a different Geant4 physics list; the
  QGSP_BERT slice (Slice 7) is the test, and its 10 and 20 GeV points sit in a model-transition zone.
- Whether the adversarial reviewer's remaining objections (A9, A11, A12) matter; they are cheap and
  can be run when convenient.

Papers found in this check were added to the researcher's library at
`C:\Users\ASUS\Desktop\QML Research\papers` (`Fractals/`, `FastMC/`), with an index there. They are
inherited citations from now on.

---

## 7. Researcher's decision, 2026-09-30

The researcher answered §4 and §5 in chat. The decision of record is in `research/STATE.md`
(Block 6B, "Researcher decisions on D1-D8"). Summary: D1 accepted with amendments (b) modified;
D2, D5, D6 accepted; D4 and D7 accepted together; D8 accepted and does not block Slices 3-5.

**Citation check on the decision text (agent, abstracts read on 2026-09-30).** The decision text
cited four papers. Three match what it says. One claim was attached to the wrong paper and is
stated more strongly than its source:

| claim in the decision text | paper cited | what the abstract says |
|---|---|---|
| CaloChallenge uses classifier and aggregate metrics beyond 1-D histograms | arXiv:2410.21611 | matches (already verified at full text, §2.1) |
| two-component longitudinal and radial descriptions of hadron showers | arXiv:1602.08578 | matches (10-80 GeV pions and protons, scintillator-steel; full text read) |
| the fractal-dimension result is Monte-Carlo-only and from a different detector | arXiv:1312.7662 | matches ("using Monte-Carlo simulation"; iron/RPC) |
| Geant4/data discrepancies are largest in longitudinal and transverse structure | cited as 1602.08578 in the decision text; the statement is in **arXiv:1411.7215** | 1411.7215 (CALICE Si-W ECAL, about 350,000 pions at 2-10 GeV): "reasonable overall description... Monte Carlo predictions are within 20% of the data, and for many observables much closer. The largest quantitative discrepancies are found in the longitudinal and transverse distributions of reconstructed energy." |

So the supported statement is: model-versus-data disagreement is *up to about 20%* with the longitudinal
and transverse energy distributions the most sensitive; it is not shown to be "substantial" beyond
that. Arguably 1411.7215 matters more to this project than 1602.08578: it is a thin silicon-tungsten
ECAL tested against Geant4 hadronic models, though with pions at 2-10 GeV (below our 10-100 GeV
range). It was **not in the source set of this pass**; only its abstract has been read; it is not yet
in the paper library. Evidence class for using it: *transferred approximation*.
