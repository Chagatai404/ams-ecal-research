# Detector-response model proposal (R-A and a reduced R-B), DRAFT for approval (2026-10-12)

_Status: **D6.1-D6.8 approved by the researcher on 2026-10-12; R-A implemented and validated on electrons (section 9); B1-B4 not yet implemented.** Original status line: PROPOSAL, nothing implemented. Basis: the per-effect table in `2026-10-12_s5a_detector_response_evidence_record.md` (sources E1-E9). Nothing below is to be coded until the researcher approves, changes or rejects each numbered choice in section 7. Ties to plan slices S4, S5b, S7 and decision DECIDE-6._

## 1. The question the model has to serve

Not "simulate the AMS electronics". The question is: **does the multiscale feature (and any PID gain from it) survive the readout?** So the model is a *degradation ladder* with declared, separable steps, each switchable, and the result is a sensitivity curve per step. No step claims to reproduce AMS data; no AMS flight data are in hand to validate against.

## 2. What the evidence allows (and does not)

| class | effects | what we do |
|---|---|---|
| **Already inverted by AMS before shower variables exist** (E1, E6, E9) | fibre attenuation, cell-to-cell gain (raw 15% spread), HG/LG ratio, temperature, PMT position/edge, rear leakage | **do not simulate as distortions.** The target is the *corrected* cell energy. |
| **Stochastic, cannot be inverted, value documented** | photoelectron statistics, about 8 p.e. per MIP cell (E2) | **simulate** (step B1) |
| **Residual after correction, value documented** | per-cell gain error: 2.4% (daily, Kalman), 7.2% (daily, no filter), 1.5% (helium, low-response cells) (E7, E8) | **scan** (step B2) |
| **Exists, value NOT FOUND** | readout zero-suppression threshold, cross-talk, residual attenuation error, saturation, Birks constant | **declared sensitivity scans only**, labelled "no source", never quoted as AMS values (steps B3, B4) |

## 3. Chain and ordering

```
Geant4 deposit (3D truth, mm voxels, DECIDE-4 reference)
  -> R-A  sampling response: deposit -> fibre energy per 9 mm cell (calibrated on Geant4 `deposit_grid_mev` vs `readout_grid_mev`)
  -> B1   photoelectron (Poisson) smearing, per cell
  -> B2   per-cell multiplicative gain residual (one draw per detector realisation)
  -> B4   cross-talk to neighbours (scan only)
  -> B3   readout threshold (scan only)
  -> one view per layer, 18 layers x 72 cells (view-aware readout, DECIDE-4)
```
The ordering of B2/B3 relative to each other is a **modelling choice, not a documented fact** (the order of the gain switch against zero-suppression is NOT FOUND in the evidence record). I propose gain error before threshold (the threshold acts on the digitised, equalized value); one setting is run in both orders so the choice is measured, not asserted.

## 4. Steps in detail

**R-A (sampling response).** Fibre energy per cell from the Geant4 files that already hold both grids. Fit the mean response (depth-dependent fibre fraction) and the sampling fluctuation on the *development* electron/proton sets; validate on a held-out split of those, not on sealed sets (DECIDE-2). Already accepted in principle 2026-10-02 (plan G1).

**B1 photoelectrons.** For a cell with fibre energy E (MeV): mean p.e. `mu = 8 * E / 7` (8 p.e. per MIP, 7 MeV per MIP cell: E2, E6), then draw `N ~ Poisson(mu)` and return `7 * N / 8`. The mean is preserved; the relative width is `1/sqrt(mu)`, which is 35% at one MIP. *Caveat to approve:* the 8 p.e. figure is quoted per cell for a MIP; whether it applies at the cell centre, averaged along the fibre, or at the far end is **NOT FOUND**, and attenuation lowers p.e. far from the PMT. Because the attenuation is later *inverted*, the noise it creates is not. I therefore propose to scan `mu` at 0.5x, 1x and 2x of the 8 p.e./MIP value as well as the nominal.

**B2 gain residual.** One factor `g_c = 1 + s * z_c`, `z_c ~ N(0,1)`, per cell, **drawn once per detector realisation and kept fixed for all events of that realisation** (a gain error is a persistent calibration error, not event noise). Scan `s` = 0, 1.5%, 2.4%, 7.2% (all documented); use several independent realisations per setting so the spread over detector realisations is reported, not just one draw.

**B3 threshold (no source).** Cell value zeroed below `t` x (one MIP, 7 MeV), scan `t` = 0, 0.25, 0.5, 1. The 0.5 value is the one already used in the S0 audit and the S4 design, so the ladder stays comparable with earlier results.

**B4 cross-talk (no source).** A fraction `x` of each cell's value is moved to its two neighbours in the same view and layer; scan `x` = 0, 1%, 3%, 10%. Energy-conserving, so it changes shape, not total.

**Not modelled, by decision of the evidence:** attenuation, raw 15% gain spread, HG/LG range and gain switching, temperature, ADC saturation (negligible below 100 GeV, the project's range), orbital/SAA effects (nothing found).

## 5. How the output is used

- Paired degradation curves (plan S4): the same event through ladder steps, feature value and PID performance recorded after each, with estimator controls and the cell-shuffle null applied at every step.
- Each B-step is reported one-factor-at-a-time, plus one combined "nominal" (B1 at 1x, B2 at 2.4%, B3 at 0.5, B4 at 0) that is *labelled a scan point, not an AMS reproduction*.
- Statements permitted: "feature X survives / does not survive step S at setting v". Not permitted: "AMS sees feature X".

## 6. Acceptance checks (written before any code)

1. All switches off: output equals the R-A output exactly (bitwise for a fixed seed).
2. B1: over 10^6 draws, sample mean equals input mean within statistical error, and variance equals `E * 7/8` in energy units (Poisson), per energy bin.
3. B2: with s = 0 identity; the realisation is the same for every event of one realisation and differs between realisations.
4. B4: total energy conserved to floating-point precision; edge cells handled without leakage out of the grid (documented rule).
5. Determinism: the same seed gives the same output; seeds recorded in the sample manifest.
6. R-A: closure on held-out development events (mean fibre energy per layer within a tolerance fixed in the registration **before** it is evaluated; proposed 3%, the same tolerance as the EMV swap closure).
7. Every scan value carries a label "documented" (with locator) or "no source", written into the output JSON.

## 7. Decisions requested (each can be answered yes / no / change)

| # | choice | my proposal |
|---|---|---|
| D6.1 | Start R-A now on the development sets | yes |
| D6.2 | R-B limited to B1-B4 above, all else "already corrected" | yes |
| D6.3 | B1 mean `mu = 8 E / 7` plus scan of 0.5x/1x/2x | yes |
| D6.4 | B2 as a per-realisation per-cell factor, s in {0, 1.5, 2.4, 7.2}% | yes |
| D6.5 | B3 and B4 as undocumented scans, labelled "no source" | yes |
| D6.6 | Order gain error before threshold, with one setting run both ways | yes |
| D6.7 | New module `src/ams_ecal/detector/readout_response.py` with a config dataclass; no change to existing estimators | yes |
| D6.8 | Closure tolerance for R-A (3%) | yes / choose another |

## 8. Known weak points (adversarial)

- B1 treats the p.e. count as proportional to cell energy; real light collection depends on where along the cell the energy is deposited (E1 attenuation). Whether this matters is exactly what the 0.5x/2x scan tests; it is **a transferred approximation**, not AMS-specific evidence.
- The documented 2.4% / 7.2% are *equalization precisions of the corrected response*, not necessarily the residual gain error entering each event; treating them as the standard deviation of a Gaussian per-cell factor is an assumption.
- The AMS reconstruction applies corrections in a chain whose own residuals are not published; independent residual terms may combine differently than my single factor `s`.
- No flight data in hand to test the whole ladder. The model can therefore show *robustness or fragility of the feature*, not fidelity.
- Birks quenching is not in the ladder; it would act before R-A and is absorbed into the R-A calibration to Geant4, which is itself a Geant4 statement (Geant4 hits `readout_grid_mev` depend on its own quenching setting; to be checked when R-A starts).

## 9. R-A result (2026-10-12, development sets only; module `src/ams_ecal/detector/sampling_response.py`, validation `sampling_response_validation.py`, output `results/detector_response/r_a_closure.json`)

**Measured facts that shaped the model**
- `deposit_grid_mev` is energy in all material, `readout_grid_mev` is fibre energy only (Geant4 metadata); the electron sampling fraction is about 6.5% and nearly flat in depth.
- **The cell-level ratio R/D depends on particle type**: an electron-fitted map is wrong by 39-47% on protons (maximum per-layer error, four energies). R-A is therefore fitted **per particle class**; the electron model must not be applied to protons. (The proton FastMC already carries its own readout representation.)
- Cell-level scatter is large (R/D has a standard deviation comparable to its mean over most deposit sizes), so a deterministic map would not be a detector image.

**Model:** empirical conditional resampling of the ratio R/D in classes of (layer, cell deposit size, event total deposit); mean-preserving per class; fallback pools shape over layers but rescales the level by the layer's mean ratio.

**Registered criterion (D6.8): all 18 layer means within 3% on held-out events (odd `event_index`; fit on even).** Result: **pass at all four energies** (maximum layer error 1.05%, 0.72%, 1.6%, 1.3% at 10, 20, 50, 100 GeV).

**Disclosures (iteration on development data, tolerance unchanged)**
1. The first version (class = layer x deposit size only) **failed** at 10 and 20 GeV (5.5%, 4.1%), because the ratio at fixed cell deposit depends on the shower energy. The event total deposit was added as a context class and the criterion then passed. This is a development-set model choice, not a sealed-set result.
2. A bug-like weakness found by a synthetic test: the pooled fallback mixed layer levels; fixed by the layer rescale. The real-data result above was re-run after the fix.
3. **Limitation, measured:** leaving a whole energy out of the fit (20 GeV or 50 GeV) gives 8-9% layer errors. The model is valid at the fitted energies, **not between them**. A spectrum-weighted production (S5b) needs either more fitted energies or a model with a smooth energy dependence; this is open.
4. Descriptive, no threshold: the model's per-event relative width of the total fibre energy is 4.3, 4.2, 5.0, 5.8% against Geant4's 3.8, 4.0, 4.8, 5.8% at 10-100 GeV (about 15% too wide at 10 GeV, converging at 100 GeV): independent cell draws slightly overstate event-level sampling fluctuation. Cell-value quantiles (among cells with fibre energy) agree within about 1% at the 90th, 99th and 99.9th percentiles, but the **median is 7.5-13% too low** (the low-energy end of the cell distribution is not reproduced as well; it matters for threshold scans).

**Correction to section 4 (B1 and B3 units).** The "7 MeV per MIP" figure (E6) is the energy a MIP loses in a cell in all material; the readout grid holds fibre energy only, whose MIP value in this repository is `MIP_CELL_MEV = 0.5596 MeV` (`validation/added_checks.py`). The photoelectron mean for B1 must therefore be `mu = 8 * E_fibre / 0.5596` (8 p.e. per fibre-MIP), and B3's threshold unit is the same 0.5596 MeV (consistent with the 0.5 x MIP threshold already used in the S0 audit). Using 7 MeV on fibre energy would have overstated the photoelectron noise by a factor of about 12.

## 10. B1-B4 implemented (2026-10-12; `src/ams_ecal/detector/readout_response.py`, 17 tests in `tests/detector/test_readout_response.py`, whole detector test folder 192 passed)

Implemented as proposed (D6.2-D6.7): `ReadoutResponseConfig` (all defaults off = identity), `draw_gain_map` (one draw per detector realisation), `apply_readout_response` (order B1, B2, B4, B3; `threshold_before_gain` runs B1, B3, B2, B4). Every parameter reports a provenance label ("documented" or "no source"). B1 uses the fibre MIP 0.5596 MeV (the section 9 correction); a test asserts it equals `MIP_CELL_MEV`. Acceptance checks 1-5 and 7 are tests; check 6 is the R-A closure of section 9.

**Sanity run on real Geant4 fibre energy (1500 development events per row, one factor at a time; an unregistered look at the size of each step, not a result).** Hits are cells above 0.5 MIP.

| step | electrons 10 GeV hits | electrons 100 GeV hits | protons 10 GeV hits | protons 10 GeV total energy |
|---|---|---|---|---|
| none | 122.6 | 264.6 | 62.7 | - |
| B1 8 p.e./MIP | -5.2% | -4.1% | -7.0% | 0.0% |
| B1 4 / 16 p.e. | -9.6% / -2.8% | -7.6% / -2.2% | -12.4% / -3.8% | 0.0% |
| B2 2.4% / 7.2% gain | 0.0% / -0.2% | 0.0% / -0.1% | 0.0% / -0.2% | -0.06% / -0.18% |
| B4 3% / 10% cross-talk | +1.5% / +3.8% | +0.5% / +1.3% | +0.9% / +3.5% | 0.0% |
| B3 0.5 MIP | 0.0% (equals the hit definition) | 0.0% | 0.0% | -2.7% |
| B3 1.0 MIP | -24.0% | -18.6% | -30.3% | -8.2% |
| nominal (8 p.e., 2.4%, 0.5 MIP) | -2.0% | -1.5% | -2.7% | -2.6% |

Readings that follow from the table (and nothing more): total energy is unchanged by B1 and B4 (mean preservation and conservation confirmed on real data); the **hit count is most sensitive to the undocumented threshold and to the photoelectron scale**, and barely to the gain error; the protons lose more of their energy to the threshold than electrons (their energy sits in many low cells). Because the hit count drives the box-counting dimension D0 (Block F), the threshold and p.e. scans are the ones to run first in the degradation ladder.
