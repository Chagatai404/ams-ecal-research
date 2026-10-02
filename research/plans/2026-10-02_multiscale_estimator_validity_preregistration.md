# Multiscale estimator validity: pre-registration of the synthetic controls (acceptance rules CONFIRMED, 2026-10-02)

_Written 2026-10-02, **before any estimator code existed and before any estimator output was read**. It belongs to the
estimator-controls experiment (record EXP-008) of `research/plans/2026-10-02_weekend_multiscale_check_and_proton_model_completion_plan.md`,
which the researcher approved on 2026-10-02. **The acceptance rules in section 4 were confirmed by the researcher on 2026-10-02 as written**
(record DEC-013). Control output may be read only after the estimator code and its tests exist._

## 1. Scope

- **Grids.** Combined 18 x 72 image as the primary (researcher's decision, 2026-10-02); per-view images as secondary.
  On 18 x 72, square index boxes divide evenly at side s = 1, 2, 3, 6, 9, 18 (six scales). A per-view image is 9 x 72 and
  allows only s = 1, 3, 9 (three scales), so per-view results are reported with that limit and cannot support a slope claim.
- **Estimators.** Partition sums Z_q(s) = sum_i p_i^q, p_i the energy fraction of box i, for q = 0, 1, 2: D_0 from the
  occupied-box count N(s), D_1 from the entropy form, D_2 from Z_2. Local slopes between adjacent scales are always reported
  next to any fitted slope. Lacunarity, entropy-across-scale and multifractal spectra are out of scope.
- **Why only q = 0, 1, 2.** Literature (section 5) advises against q < 0 and against large q for finite or sparse data.

## 2. Controls (all on 18 x 72 and on a finer grid for comparison)

| Control | Purpose |
|---|---|
| Smooth source sampled by the same number of hits (Poisson quanta), 1000 replicates per hit count | Negative: the counting-limit baseline |
| Same images with cells shuffled within each layer | Negative: spatial structure removed, layer energies kept |
| Multiplicative binomial cascade on a 2^k grid, projected to 18 x 72 | Positive: known D_q on the fine grid, passed through the lossy readout |
| Hit-count sweep, 30 to 3000 quanta | Finite-sample bias of each estimator |

## 3. Scale-window rule (from the literature, transferred)

Following Koo and Ju (2026): the box-count saturates at the number of distinct occupied cells N_eff, and the reliability of
a fitted window is set by `x = k_mid - k*`, with k* = log2(N_eff / A) / D_f, k = log2(1/box size) and A of order 1.
**Rule:** a fitted slope is interpretable only for a window with x <= 0 (coarser than saturation). Local-slope
stability is **not** used as a reliability test (the paper shows the saturated plateau is maximally stable and maximally
biased). A = 1 and D_f = 2 (conservative) are the starting values and are **project assumptions**, checked on the controls.
Any estimate from a window with x > 0 is reported as "saturated, not interpretable" and cannot support any claim.

## 4. Acceptance rules (confirmed by the researcher on 2026-10-02)

| Rule | What it requires | Basis |
|---|---|---|
| False-positive limit | For fresh smooth and shuffled controls, the estimate falls outside the baseline's central 95% interval (same N_eff, same scales) in **no more than 10%** of replicates | surrogate-data practice (literature); the 95% and 10% are conventions, **project assumption** |
| Detection power | On the projected cascade, D_0 - D_2 exceeds the baseline's 97.5th percentile in **at least 90%** of replicates | theory (Hentschel-Procaccia monotonicity) plus **project assumption** for the 90% |
| Ordering | D_0 >= D_1 >= D_2 holds in the cascade replicates. The fraction required is not stated here; see the addendum (section 7) | theory; the fraction is an **interpretation awaiting the researcher** |
| Saturation-window rule | Only windows with x <= 0 count (section 3) | literature, transferred |
| Recovery reference | The error between the known D_q and the estimate is reported per scale window and per hit count, with a **15% reference line** (not a pass/fail gate) | **project assumption.** No standard numeric tolerance was found. The one benchmark found (5% convergence) needs a number of points that grows exponentially with D (reported as log10 N_5 about 2.54 D(q) - 0.11, snippet level), which 18 x 72 images with hundreds of quanta cannot meet, so 5% is **not** adopted |
| Verdict | USABLE at AMS granularity only if the false-positive limit, detection power, ordering and the saturation-window rule all hold; otherwise NOT USABLE here. "Not usable" is a valid, reportable outcome | |

Primary evidence for usability is **separation from a calibrated baseline** (false-positive limit and detection power), not closeness to
a theoretical number (recovery reference), because the literature supports calibrated surrogate comparison as the standard and offers
no recovery threshold.

 and how deeply each item was read

| Claim | Source | Read to | Evidence class |
|---|---|---|---|
| D_q = lim (1/(q-1)) log sum p_i^q / log l | Hentschel and Procaccia, Physica D 8, 435 (1983) | search-result summaries quoting the definition (MathWorld, arXiv listing); **primary paper not read** | established mathematics, secondary confirmation |
| Box-count saturates at the sampled-point count; E[N] = b[1-(1-1/b)^N_eff]; k*; x is the reliability indicator; slope-stability diagnostic fails | Koo and Ju, arXiv:2605.27925 (2026) | **full text, pp. 1-4** (abstract, introduction, theory, occupancy model) | **transferred approximation**: point-sampled stochastic trajectories, not energy-weighted deposit images. In the researcher's library; first read for this purpose on 2026-10-02 |
| Their occupancy expectation is the one used in the tutoring visual VIS-005 | same | independently reproduced: exact expectation agrees with 400 simulated patterns within 0.2% | verified computation |
| Avoid q < 0 and large q for finite/sparse data; q >= 0, about 0 to 5 used in practice | several arXiv results (e.g. 1808.02851, 0808.3068, 1305.7384) | **search snippets only** | unverified, snippet level |
| Shuffled or surrogate data show finite-size artefacts; a real signal must exceed the shuffled baseline | arXiv:2603.04609 and related (in the researcher's library) | **search snippets and abstract only** | unverified, snippet level; also listed in the 2026-09-30 evidence map at abstract level |
| Required sample size grows exponentially with D (5% convergence) | Pattern Recognition 209 (2026), "Convergence of numerical box-counting ..." | **search snippet only** | unverified, snippet level |

Counter-evidence sought: the search found no source recommending q < 0 or negative-q box-counting for sparse images, and
no source giving a numeric recovery tolerance for sparse point-sampled images. This is **not** a claim that none exists:
the search was a few queries. An independent source-verification pass on the snippet-level rows is recommended before any
result is published, and is cheap.

## 6. Status

The values 95%, 10%, 90% and 15% are confirmed project assumptions (DEC-013), not literature values. Estimator code, tests on
exact patterns and the controls may now be built. Nothing here is an experimental result.

## 7. Addendum, 2026-10-02 (after the first run of the controls): a deviation and an interpretation

- **Deviation.** The first run applied only the detection-power part of the rule for detecting the cascade and merely reported the
  ordering D_0 >= D_1 >= D_2, omitting the ordering rule written in section 4. Found after seeing output; corrected in the code to match this document. The first run's verdict ("USABLE" at 300, 1000, 3000 hits) is kept in the results record.
- **Interpretation, awaiting the researcher.** Section 4 does not say what fraction of cascade replicates must satisfy the ordering.
  The code uses **0.90** (`MIN_ORDERING_RATE`). The verdict is the same for 0.5, 0.9 and 1.0 at every hit count except 3000
  (4 windows at 0.5 and 0.9, 3 at 1.0), so the choice does not change any conclusion.
- **Result (synthetic controls only).** NOT USABLE at 30, 100 and 300 hits; USABLE on 2 of 13 windows at 1000 and 4 of 14 at 3000.
  See `research/plans/2026-10-02_multiscale_estimator_controls_results.md` and `results/multiscale_controls/`.
