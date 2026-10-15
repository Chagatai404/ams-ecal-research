# S2 — independent literature discovery and verification record (2026-10-12)

_Evidence record, not a plan and not a decision. Slice S2 of `research/RESEARCH_PLAN.md` section 18.4. Follows `RESEARCH_PROTOCOL.md` section 2: neutral question -> independent discovery -> source verification -> counterevidence -> reconciliation (the reconciliation is in `2026-10-12_s2_hypotheses_and_amendment_DRAFT.md`). It supersedes nothing in `2026-10-09_multiscale_literature_verification_record.md`; it adds to it._

## 1. Provenance and independence

| step | who | inputs | what they were told | output |
|---|---|---|---|---|
| discovery round 1 | four literature-scout subagents, fresh contexts | neutral questions only | no repository, no results, no implementation names, no local library; no PDF downloads; "be adversarial" | source maps with reading depth per source |
| discovery round 2 | two scouts, fresh contexts | neutral questions on the gaps of round 1 (citations of Ruan 2014, null results, occupancy and grid artefacts) | same | source maps |
| verification | the author (Claude) | the PDFs | the claim to check, read in `pdftotext` output | the "verified" column below |
| human decisions | the researcher | this record | | pending |

Independence limits, stated: the scouts' search tools returned snippets and tool-side summaries, not page text; the scouts themselves flagged that **no source reached full text** and that most numbers are `UNVERIFIED`. Two scout claims were **wrong or unsupported** when checked: (1) a snippet put the "superposition of sub-showers smooths self-similarity" caveat in the HEGRA abstract; it is in the introduction of the full text (and is true there); (2) a snippet said a CEPC network used fractal dimension as an input; its abstract does not mention it (still unchecked in full text). The rule stands: **a scout's number is a lead; only a number read in the PDF with a locator is used.**

Reading depth key: **FULL TEXT** (read by the author in this pass or an earlier recorded pass), **ABSTRACT** (abstract page only), **SNIPPET/TITLE** (search result only), **CITATION** (seen in a reference list).

## 2. Sources verified in full text (locators are section, equation, figure or table of the PDF)

| id | source | what was verified | relevance | provenance label |
|---|---|---|---|---|
| V1 | Ruan, Jeans, Boudry, Brient, Videau, PRL 112, 012001 (2014); arXiv:1312.7662; library `Fractals/Fractal Dimension on Highly Granular Calorimeter.pdf` | **Monte Carlo only** (abstract). Detector: 48 layers, 20 mm Fe, 6.5 mm RPC, **binary readout**, 10 x 10 mm^2 cells (Method), simulated at 1 x 1 mm^2. Observable: **number of hits** N_alpha after grouping alpha x alpha cells; FD = log R_{alpha,beta} / log(alpha/beta) - 1 (Eq. 1), i.e. a **support-based, transverse box count** with the layer count subtracted; energy is not a weight. Results: FD_e(E) = 1.41 + 0.21 log10(E/GeV) at 1 mm (Eq. 2), FD_hadron = 1.24 + 0.15 log10 E, muon 1.2. At 40 GeV, 10 mm cells, Table I: e+ 100%, pi+ 99.5%, mu 96.9% correct. **FD is linear in log N_hits** (Fig. 5 left) and **the identification cut uses N_hits as well as FD** (FD >= 0.68 and N_hits > 200 at fixed energy; global cut on log10(N_hits) - 2.8 FD_10mm: 98% EM and 96% hadronic over 1-80 GeV, 99.5% / 98.7% above 10 GeV). The paper states FD is "rigorously defined at infinitesimally small sizes" but accepts any shower with more than 10 hits. **No test-beam data; no comparison with conventional shower variables; no test that FD adds information beyond hit count** (none in the text). | The only primary calorimeter source for a fractal PID claim; it is **not** a D_q or energy-weighted analysis, and **its own data show FD tied to hit count and energy**. | externally established **for simulation of an RPC-Fe HCAL**; transferred approximation for any other detector |
| V2 | Schaefer, Hofmann, Lampeitl, Hemberger, NIM A 465 (2001) 394; arXiv:astro-ph/0101318 (downloaded 2026-10-12 into the library) | Multifractal + wavelet parameters on **HEGRA Cherenkov images**, Mkn 501 data. Sec. 1: sub-shower light superposition may give a smooth image, so the light distribution "does not necessarily have to have fractal properties". Sec. 3.4 / Fig. 5: the multifractal parameter is **strongly correlated with the Hillas scaled width and concentration**. Sec. 3.5, Table 2: separation similar to concentration, **worse than scaled width**. Sec. 4, Table 3: a network on multifractal + wavelet inputs alone **falls behind** the Hillas network; combining gives **a slight improvement, "limited by the strong correlation"**. Sec. 5: **the pixel size of the HEGRA cameras is "much too coarse" to characterise the fractal nature of shower images**. | The closest precedent for "multifractal vs conventional shape variables": a **small gain limited by redundancy, and a coarse-pixel warning**. Counterevidence for the strong form of H-B; direct support for the redundancy check. | externally established for air-Cherenkov images at HEGRA pixel size; transferred approximation here |
| V3 | Anchordoqui, Kirasirova, McCauley, Paul, Reucroft, Swain, "Fractal Electromagnetic Showers", hep-ph/0006108 (library `Fractals/Fractal EM Showers.pdf`) | Sec. 3: the energy summed within radius R is a straight line in log-log only at **small radii**; at large radii boundaries break scaling; "anomalous structure" near the core. D_q is defined for **q > 0**. For small q in the scaling region **the D_q are "all approximately equal within the errors ... and approximately unity, suggesting a good degree of homogeneity"**. Full realistic Geant4 simulation (aluminium), not a pure cascade. | Prior that EM showers are **nearly monofractal** (D_q equal); the radial scaling window is a fraction of the shower. Counterindicative to a strong EM multifractal claim (recorded 2026-10-09). | externally established for Geant4 EM showers in aluminium; transferred here |
| V4 | CaloChallenge 2022, arXiv:2410.21611 (library `FastMC/2410.21611v2.pdf`) | Sec. 8: metric families = 1D **separation power** of high-level histograms (energy per layer, centre-of-energy widths, **layer sparsity** = 1 - fraction of voxels above threshold), **Pearson correlation** of layer energies (Sec. 8.2), **classifier AUC** on low- and high-level inputs (Sec. 8.3), FPD/KPD, multiclass log-posterior. The paper says the classifier test "is sensitive to the full distribution, including correlations"; that **its result depends on preprocessing** (physical-space inputs focus on the brightest voxels; log/logit increases sensitivity); and that a difference may lie "only in features and correlations that are irrelevant for the down-stream physics analysis". **No fractal, multifractal or box-counting metric occurs** (the only "multi-scale" hit is the architecture of flow networks). | Defines the **comparator set for a fidelity (P3) claim**: a multiscale statistic is worth adding only if it flags a defect the standard set misses. | externally established (community benchmark) |
| V5 | "A Comprehensive Evaluation of Generative Models in Calorimeter Shower Simulation", arXiv:2406.12898 (library) | Abstract and Sec. 3.1: a standardised set of metrics "derived from physics, computer vision, and statistics"; models "fell short in accurately replicating Geant4 data". **No fractal/multiscale metric** (text search). | Same as V4. | externally established |
| V6 | L2LFlows, arXiv:2302.11594 (downloaded 2026-10-12) | Results: both models describe most pairwise correlations of observables but **"struggle with specific correlations, involving the second moments in the x- and y-direction"**. | An example of **correlation-level failure** that marginal metrics miss; supports H-E as plausible, not proven. | externally established for the stated flow models and datasets |
| V7 | CALICE, "Hadron shower decomposition in the highly granular CALICE analogue hadron calorimeter", arXiv:1602.08578 (library) | Fe-AHCAL, 10-80 GeV, Geant4 9.6.p1. **FTFP_BERT** reproduces proton profiles at all energies and pion profiles within 5% at 15 and 30 GeV; overestimates near the maximum by up to 12% at 80 GeV; **on-axis radial deposition overestimated by up to 20% at 80 GeV** for pions. **QGSP_BERT** underestimates pions at 10 GeV, **overestimates the maximum by more than 20% at 80 GeV** (pions and protons) and underestimates the tail above 30 GeV. Both profiles fitted with two-component (core + halo) functions. | The size of Geant4 hadronic-list disagreement and of Geant4-versus-data disagreement in a steel-scintillator calorimeter. **Sets the scale of any simulation-only multiscale claim** (H-F). | externally established for Fe-AHCAL, Geant4 9.6; transferred to Pb-fibre and Geant4 11.4 |
| V8 | CALICE preliminary, "Comparison of hadron shower data with simulations", arXiv:1007.2947 (downloaded 2026-10-12) | Sec. 4.2: mean shower radius from a Gaussian fit in the region with 90% of events, 15-80 GeV pions: **the selected Geant4 models give about 15% smaller shower radius than the data** (the approximation sign is lost in text extraction; magnitude read as 15%). | Direction consistent with V7 (Geant4 deposits too much on axis). Preliminary note, old Geant4. | externally established, preliminary, old |
| V9 | Kantelhardt et al., MFDFA, physics/0202070 (library) | Abstract and Sec. IV: comparing original and **shuffled** series separates multifractality due to long-range correlations from that due to a broad probability density. | The shuffle null is a legitimate construct **for time series**; it does not test occupancy/sampling in point-set images. | externally established for time series |
| V10 | Koo and Ju, arXiv:2605.27925 (library; full text read 2026-10-02) | Apparent box-counting exponents of finite trajectories follow an occupancy crossover; local-slope stability alone is not a reliable diagnostic. | Basis of the estimator-validity gate (DEC-013). Preprint, not peer reviewed; trajectories, not energy-weighted images. | transferred approximation |
| V11 | Gneiting, Sevcikova, Percival, Stat. Sci. 27 (2012) 247; arXiv:1101.1444 (downloaded 2026-10-12) | Compares box-count, Hall-Wood, semi-periodogram, DCT, wavelet and variation (madogram) estimators for time series and surfaces; **recommends the madogram on efficiency and robustness**; the **box-count estimator generally shows a bias** in the simulations. | Supports a robust second estimator as a cross-check; **roughness of a function, not energy-measure dimensions**. | transferred approximation |

## 3. Sources at ABSTRACT, SNIPPET or CITATION depth (leads, NOT evidence)

| source | depth | what is claimed (unverified) | next step |
|---|---|---|---|
| Faleiro, Gomez, Relano, Astropart. Phys. 19 (2003) 617 | TITLE | gamma-hadron discrimination by the multifractal spectrum of density fluctuations in EAS; the most direct independent-group EAS multifractal result | link to the researcher |
| Kempa, J. Phys. G (1994); Haungs, Kempa et al. (KASCADE) | SNIPPET | f(alpha) may separate iron from proton primaries | link |
| Pagliaro et al. (ARGO-YBJ) | SNIPPET | multifractal + lacunarity + NN, simulated 1-10 TeV showers | link |
| Razdan (TACTIC), astro-ph/0205015, 0109476, Bhat et al. 1008.2255 | SNIPPET | binomial-cascade origin of Cherenkov multifractality; "about 90% gamma acceptance" and "almost 100%" in simulation (suspect) | read if the air-shower branch is kept |
| Ruan et al., J. Phys. Conf. Ser. 490 (2014) 012227 | SNIPPET | proceedings with 40 GeV numbers; two snippets disagree on the pi+/K_L order | read; link |
| Hashmani, Akbas, Demirkoz, arXiv:2402.16285 (downloaded 2026-10-12) | ABSTRACT; AMS rejection facts were read in an earlier pass (review section 4) | DL models on AMS ECAL images: MC 0.2-2 TeV, ISS 50-70 GeV | no new claim used |
| Song et al., arXiv:2310.09489 (CEPC ResNet PID) | ABSTRACT | one snippet said FD is an input; the abstract does not | open the full text before anyone cites it |
| Grassberger (1988, 2003), Schurmann (2004) entropy bias | CITATION/SNIPPET | small-sample bias of entropy and dimension estimates | cite only the principle; Grassberger 1988 paywalled: link |
| Eckmann and Ruelle, Physica D 56 (1992) 185 | SNIPPET | a correlation-dimension estimate from N points cannot exceed about 2 log10 N | **not read**; the "N >= 10^(D/2)" form was inferred by a scout, not stated in any source read. Do not use as a rule; link |
| Hall and Wood (1993); Theiler (1990); Halsey et al. (1986); Chhabra and Jensen (1989) | CITATION | foundational estimator and f(alpha) references | not read; paywalled: links |
| Bialas and Peschanski (1986); Hwa (1994) | SNIPPET | scaled factorial moments / intermittency; Bose-Einstein correlation artefact | only if a factorial-moment branch is opened |
| Ruan citation index (Semantic Scholar tool summary: 10 citing items incl. 2310.09489, 1806.04879, 1701.07542, 1510.05224) | SNIPPET | none confirmed to test FD on data or against multiplicity | an INSPIRE `refersto:arxiv:1312.7662` pass by the researcher or an agent with browser access |

## 4. What is **not found** (search scope stated; absence is not established)

Rounds 1 and 2 searched: fractal/multifractal/lacunarity/intermittency of showers; D_q finite-sample bias and scaling windows; surrogate and Poisson nulls; anisotropic and composite grids; calorimeter fast-simulation fidelity metrics; hadronic substructure; e/p rejection; citations of Ruan 2014. **Not found**, with the caveat that the search tools returned snippets:

1. any test-beam analysis of the shower fractal dimension, any independent reproduction, any test of FD against hit multiplicity, shower radius or start depth, any particle-flow or ML paper confirmed to use FD as an input, any finite-statistics critique of FD in calorimeters;
2. any study that finds EM showers monofractal other than V3, and any energy-weighted D_q spectrum for calorimeter showers;
3. any multiscale, fractal or lacunarity statistic used as a **fidelity metric** for calorimeter simulation (V4, V5 and the scouts agree);
4. any published case where a reported D_q of a **pixelated image or sparse point set** was shown to be an occupancy artefact (analogues exist in time series: V9, the SST/MFDFA and Ising preprints; and the Bose-Einstein artefact of intermittency);
5. any treatment of box counting on **grids that alternate measured coordinates** (the 18 x 72 readout) or on non-square boxes; grid-origin averaging is standard practice but only educational pages were found;
6. a **quantitative production-cut or EM-option sensitivity** of EM shower shape in a sampling calorimeter from a recent paper;
7. any recent (Geant4 11) hadronic shower-shape validation; the only Geant4 numbers verified are from Geant4 9.6 and older.

Items 1-5 are where the project could add something, and where any claim of novelty needs the explicit scope "not found in these searches".

## 5. Classification of implementation-affecting assumptions (protocol labels)

| assumption | label | evidence |
|---|---|---|
| a transverse box-count statistic of shower hits separates showers of different type | externally established **in simulation, for a binary-readout RPC HCAL** | V1; not for energy-weighted D_q, not for a fibre ECAL, not on data |
| fractal-type parameters are redundant with conventional shape variables to a large degree | externally established **in air-Cherenkov images at HEGRA pixel size** | V2 (correlation, slight combined gain) |
| EM showers are nearly monofractal | externally established for one Geant4 aluminium study, small q | V3 |
| coarse pixels limit the fractal characterisation | externally stated (V2, Sec. 5); applies to the AMS 9 mm cell by analogy: **transferred approximation** | V2 |
| standard fidelity metrics (separation power, layer correlations, sparsity, classifier AUC) are the comparator set for a simulation-fidelity claim | externally established | V4, V5, V6 |
| Geant4 hadronic lists differ from each other and from data by 10-20% in radial and longitudinal profiles | externally established for Fe-AHCAL at Geant4 9.6; **transferred** to Geant4 11.4 and Pb-fibre | V7, V8 |
| box-count slopes of finite data are biased by occupancy | transferred approximation (trajectories) | V10, V11 |
| a shuffle null separates correlation structure from a broad value distribution | externally established for time series; **project phenomenological assumption** for within-layer cell shuffles | V9 |
| the usable scaling window on the readout is about 1 decade | **project assumption** from grid geometry and the DEC-013 controls | S0/S1 records |
| D0, D1, D2 are the stable orders; negative q is excluded | project decision (2026-10-02) with the agent's reasoning, not yet supported by a source read | pending |

## 6. Per-paper outcomes (Research OS rule)

- **Saved to the library on 2026-10-12** (`C:/Users/ASUS/Desktop/QML Research/papers`, by arXiv identifier, open access, on the researcher's go): `Fractals/1101.1444.pdf`, `Fractals/astro-ph_0101318.pdf`, `FastMC/2302.11594.pdf`, `FastMC/2402.16285.pdf`, `FastMC/1007.2947.pdf`, `FastMC/1803.04165.pdf` (HSF white paper, **not read**), `FastMC/2106.05285.pdf` (CaloFlow, **not read**).
- **Already in the library, verified in this pass:** V1, V3, V4, V5, V7, V9, V10.
- **Not on disk, paywalled or not on arXiv (links to the researcher):** Faleiro et al. 2003; Kempa 1994; Grassberger 1988; Hall and Wood 1993; Halsey et al. 1986; Chhabra and Jensen 1989; Eckmann and Ruelle 1992; Pagliaro et al.; the Ruan conference proceedings (the CERN indico file 4327-FD_V6.pdf is public).
- **Rejected as evidence:** every quantitative statement taken from a search snippet or a tool summary.

## 7. What this changes in the working hypotheses (pointers; the reasoning is in the DRAFT amendment)

1. H-B (hadronic heterogeneity beyond conventional morphology) has **no independent external confirmation** and one **close counter-precedent** (V2). A null result is the prior-plausible outcome and must be publishable.
2. H-A (EM universality) has a **monofractal prior** (V3); "no multiscale difference" is an expected, not a failing, result.
3. The "occupancy-conditional" estimand is supported by the primary source itself (V1: FD tied to log N_hits).
4. H-E needs the standard fidelity set as comparator (V4); no external case exists for a multiscale fidelity metric.
5. H-F's scale is set by V7 and V8 (10-20% list disagreement): a physics-list-stable sign is a requirement, and "simulation-only" is a standing limitation of every P1/P2 statement.
