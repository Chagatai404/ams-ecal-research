# Weekend multiscale check, proton model completion and the road to the detector response — PROPOSED PLAN

_Drafted 2026-10-02 at the researcher's request: finish the proton model, then move to the detector response after learning again, and plan the first multiscale check for the weekend. **Status when drafted: proposed. Nothing in this file was approved or executed.** Approval is per item, in the table at the end, which records each approval with its date._

Role of this document: it is a plan in the sense of `RESEARCH_PROTOCOL.md`. No simulation, build or experiment starts
until the researcher approves the specific item. It follows the proton model sequence already accepted on 2026-09-30
(`research/STATE.md`, steps 4-10) and adds the multiscale check (EXP-001 stage 1, gated by the estimator-validity gate / EXP-008).

## 0. What is known, and what is not

Measured or recorded facts this plan relies on:

- The proton pilot batch at 100 GeV (4000 events, FTFP_BERT, AMS-only geometry) took **710 s on 12 workers**
  (`data/geant4_proton_pilot/baseline/E100GeV/metadata.json`), about 0.18 s per event wall time. The machine has 16 cores.
  The proton data on disk is 1.1 GB across all samples.
- Proton events are cheap because the thin ECAL absorbs only about 150 MeV per proton. **An electron deposits about
  95 GeV per event**, so electron cost per event is expected to be much larger. It has **not been measured**. This is
  the main schedule risk and is why a short timing probe comes first.
- The Geant4 backend takes the primary particle as a configuration string (`particle` in
  `configs/geant4_proton_pilot.yaml`), so an electron sample needs a configuration, not new transport code.
  `e-` thresholds are already recorded per material. This is an inference from the code, to be confirmed by the probe.
- The pilot records fine fibre and 3 x 3 x 4.625 mm voxel deposits, and the readout grid, per event.
- On the 18 x 72 grid, square boxes divide it evenly only at sizes **1, 2, 3, 6, 9, 18**: six scales and no more.
  This is a hard limit on any scaling claim at AMS granularity.
- Researcher's knowledge state for the estimators (Research OS knowledge graph, 2026-10-02): box counting is
  defined and the prediction landed; *what a slope means* has **not** landed (Q10 partial, Q11 wrong);
  marginal versus joint landed same-day only; Jensen is `fragile`. A delayed check is scheduled for 2026-10-04 09:00.

Not known: electron cost, the best handling of the alternating X/Y views, and whether any multiscale estimator is
usable on 18 x 72 at all. The last is exactly what the check is for.

## 1. Order of work, with the reason for each

```text
Fri 2-Oct  learn the math just enough  ->  pre-register  ->  electron timing probe
Sat 3-Oct  estimator code + synthetic controls on 18 x 72  (estimator-validity gate)
Sun 4-Oct  first look at existing proton Geant4 data, exploratory;  electron production if affordable
Mon 5-Oct onward   the proton model steps 4-8, then sealed test set, final validation; learn again; the detector response plan
```

Principle: do the **gate first** (the estimator-validity gate: synthetic controls). A multiscale number computed before the estimators are
validated cannot be interpreted, and the estimator-validity gate is accepted as the gate for every multiscale or fractal claim. The first
check on real Geant4 events is therefore *after* the controls on the same grid, on the same day at the earliest.

## 2. Estimator controls: the estimator-validity gate on the 18 x 72 grid (EXP-008)

**Before any code (learn, define, pre-register).** The research workflow says: learn and verify the mathematics,
define estimators, validate on controlled cases. So:

1. **Learn, about 30 min.** A tutoring block (Obsidian, with a verified visual) on the partition sum
   Z_q(s) = sum p_i^q, and on D_q as the slope of its log against log(1/s), starting from the q = 2 toy already
   used on 2026-09-20. Box counting is Z_0. No more is needed for the first check. Multifractal spectra stay out of scope.
2. **Verify the definitions** against a standard source (bounded source-verifier pass; the definitions are
   textbook mathematics, not hypothesis-sensitive). Record the page and equation in the pre-registration.
3. **Pre-register**, in a new `research/plans/` file before any estimator output is read: the estimators, the six box
   sizes, the fit rule and scale range, the controls, and the acceptance numbers below.

**Estimators (first check).** On a non-negative deposit image: occupied-box count N(s), D_0; and Z_q(s) for q = 1 and 2
(D_1 via the entropy form, D_2). Local slopes between adjacent scales are reported alongside any fitted slope, so the
scale dependence shown in the 2026-10-02 tutoring is visible, not averaged away. Lacunarity and entropy-across-scale
are listed as candidate extras, not part of the first check.

**Controls, all on 18 x 72 and on a finer grid for comparison:**

| Control | Purpose | Known answer |
|---|---|---|
| Smooth source sampled by the same number of hits (Poisson quanta) | Negative: the counting-limit baseline | slope set by hits and scale range, not structure |
| Same image with cells shuffled within each layer | Negative: kills spatial structure, keeps layer energies | baseline-like |
| Multiplicative binomial cascade on 2^k grid, then **projected** to 18 x 72 | Positive: structure with known D_q, passed through the lossy readout | known D_q on the fine grid |
| Hit-count sweep (30 to 3000 quanta) | Finite-sample bias of each estimator | none; the bias curve is the output |

**Proposed acceptance (to be fixed by you before output is read; these numbers are a proposal):** an estimator is
"usable at AMS granularity" for a given q only if (a) on the cascade control it recovers the known D_q within 15%
after projection, and (b) on the negative controls its value stays within the baseline's own spread, over the same
hit counts and scale range. Otherwise the result is "not usable here", which is a valid, reportable outcome.

**Output:** `src/ams_ecal/multiscale.py` plus tests; a results folder with the control tables; a verified teaching
visual of the controls. Cost: one day of build, minutes of compute. No Geant4.

## 3. Proton first look — first look at existing proton Geant4 events (EXP-001, stage 1, exploratory)

Only after the estimator controls pass for the estimators it will use, and only if the researcher confirms.

- **Data:** the existing FTFP_BERT baseline proton events at 10/20/50/100 GeV. **Calibration events only**
  (`event_index % 4 != 3`); the held-out events stay untouched, as already decided.
- **Do:** compute the validated estimators on (i) the fine voxel deposits and (ii) the same events projected to
  18 x 72, and compare each with a **matched smooth baseline** (same hit count, same layer-energy profile).
- **What it can say:** whether proton showers differ from the smooth baseline at fine scales, and whether any
  difference survives projection to 18 x 72. This is part of H-002 and H-003 for protons only.
- **What it cannot say, and the write-up must state:** anything about electron versus proton (no electrons yet),
  anything about particle ID, anything about fractals as a physical claim, or anything about AMS data. It is Geant4
  11.4.1 with one physics list. The production cut was only checked at readout level, and the pilot's own gate says
  no sub-cell claim without a cut-sensitivity study; this check is therefore labelled exploratory and the cut study
  is listed as a follow-up.
- **Pre-registered:** the comparison and the decision rule, written before the first number is read.

## 4. Electron data — electrons, so the real e/p comparison exists (EXP-001, stage 2)

1. **Timing probe, about 20 events at 10 and 100 GeV**, same geometry, cut, entry spot and seed policy as the proton
   pilot. This is a measurement, not a result, and is the only electron Geant4 run proposed before approval of more.
2. **Go/no-go** on a full electron sample using the probe: events per energy, energies, expected wall time, disk size.
   If the cost is hours, run it in the background while Tracks A and B proceed (the machine has 16 cores).
   If the cost is days, reduce energies or events, or subsample, and say so. The decision is the researcher's.
3. **Do not compare** electrons with the existing proton set until both are projected and processed by the same code.

## 5. Proton model completion (the accepted sequence, steps 4-10)

| Step | Work | Notes |
|---|---|---|
| 4 | Burst latent and lateral spill on the crossing branch (EXP-005) | structural burst (probability or type, onset, energy, downstream extent), as specified |
| 5 | Second-look diagnostic | |
| 6 | Interacting model under the interacting-proton factorization with its three amendments (EXP-006) | residual tested for normality before being called log-normal |
| 7 | Speed: 11-14 ms per event is too slow for large datasets | batch path |
| 8 | Freeze; register the added validation checks rows | registered before the test set is generated |
| 9 | Fresh Geant4 proton test set | **proposed change below** |
| 10 | Final the final validation (step 6) validation on that set | pre-registered the validation contract and the added validation checks contract |
| 7' | Alternative physics-list calibration (step 7): QGSP_BERT calibration (EXP-007) | needed only before claims about hadronic physics |

**Proposed speed-up, needs explicit approval because it reorders the accepted sequence.** Step 9 does not need a
frozen model, only a frozen, registered contract: the fresh proton set can be generated **early and sealed** (never
opened, file hashes recorded, disjoint seeds), then opened once at step 10. Integrity is preserved because the set has
not been seen by anyone, and the generation runs in the background while the build proceeds. Added validation checks must be registered
**before** generation, as already decided, which is a half-day task. If the original order is preferred, step 9 stays
after the freeze.

## 6. Learning gates

- **Weekend:** the partition-function block above, before estimator definitions.
- **2026-10-04 09:00:** the scheduled delayed check (histograms versus joint, Jensen zero condition and conserved sum,
  the slope-and-hit-count idea). It already covers the idea the weekend's controls depend on.
- **After step 10:** the post-build teach-back, a teach-back on the interacting model (amplitude x profile, albedo, burst), before any
  Detector response work.
- **Detector response** stays blocked until its evidence and adversarial pass runs: for each candidate detector effect, state
  whether AMS already inverted it out of the deposits its shower parameters were fitted to (the double-counting
  question in `research/STATE.md`). That pass is independent literature work and starts after step 10, then a plan,
  then explicit approval.

## 7. Risks, stated plainly

- Electron cost may make the weekend electron sample impossible. Then the weekend delivers the validated estimator
  controls and a proton-only first look, and the e/p comparison follows later. That is still a real result.
- Six scales at AMS granularity may be too few for any estimator. The controls will show it, and the answer
  "not usable here" is acceptable and publishable under the roadmap's outcome categories.
- The weekend is tight if the learning block and the delayed check both run. If time slips, cut the proton first look, not the estimator controls.
- Everything here is Geant4 with one physics list and no AMS data.

## 8. Approval table (items approved on 2026-10-02 are marked with the date; blank means still open)

| # | Item | Needs | Approved? |
|---|---|---|---|
| 1 | Estimator controls: learn block, pre-registration, estimator code and synthetic controls (EXP-008) | your approval | **2026-10-02** |
| 2 | Acceptance rules for the controls (95% interval, 10% false positives, 90% power, 15% recovery reference) | your numbers | **2026-10-02**, confirmed as written after a literature check (record DEC-013); the ordering fraction 0.90 still awaits confirmation |
| 3 | Proton first look on calibration events, exploratory (EXP-001 stage 1) | your approval, after the estimator controls | open |
| 4 | Electron timing probe: 20 events at 10 and 100 GeV | your approval | **2026-10-02** |
| 5 | Full electron sample (10, 20, 50, 100 GeV x 1000) | decision after the probe | **2026-10-02** ("go"); finished the same day |
| 6 | Early sealed fresh proton test set (changes the accepted step order) | explicit approval | **2026-10-02** (the added validation checks must be registered before generation) |
| 7 | Proton model steps 4-8 as in section 5 | your approval to start Monday | open |
| 8 | Handling of the alternating X/Y views | your decision | **2026-10-02**: combined 18 x 72 primary, per-view secondary |

## 9. Result of the electron timing probe (run 2026-10-02, approved item 4)

Config `configs/geant4_electron_probe.yaml`; data in `data/geant4_electron_probe/` (not for analysis). A cost measurement
only: 20 events per energy, FTFP_BERT, AMS-only geometry, 8 workers.

| energy | events | wall time (8 workers) | size on disk |
|---|---|---|---|
| 10 GeV | 20 | 54.5 s | 0.6 MB |
| 100 GeV | 20 | 65 s | 2.5 MB |

**Reading, with its limits.** Both wall times include worker start-up and geometry construction, which is unknown here, so
the per-event cost cannot be read directly. The 10.5 s difference between the two energies times 8 workers gives an
estimate of about 8 CPU-seconds per 100 GeV event if start-up is near 45 s, and the hard upper bound (no start-up at all)
is 26 CPU-seconds. Either way electron events are **far cheaper than feared**: a 1000-event batch at 100 GeV is roughly
10 to 35 minutes on 12 workers, and about 125 kB per event on disk. The first full batch will give the real number.
Protons for comparison: 4000 events at 100 GeV took 710 s on 12 workers.

**Recommendation for item 5 (decision still the researcher's):** electrons at 10, 20, 50 and 100 GeV, 1000 events each,
matched to the proton pilot (same geometry, cut, entry spot, FTFP_BERT), disjoint seeds. Estimated total wall time under
two hours and under 0.5 GB. Nothing from the probe is analysed for shower structure.
