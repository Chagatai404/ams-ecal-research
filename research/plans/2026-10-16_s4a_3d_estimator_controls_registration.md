# S4 part 1: 3D multiscale estimator controls and eligibility on the fine voxels — registration (2026-10-16, written BEFORE any 3D control was run)

Status: methods development on synthetic maps and exposed development events; no confirmatory claim, no sealed set. The researcher approved starting S4 part 1 on 2026-10-16 ("Start those"). It implements plan item A4 (estimator controls on a finer grid) and the 3D eligibility rule of DECIDE-4 (accepted 2026-10-11): eligibility is derived from 3D controls, not from the 2D thresholds (336 / 572 occupied cells) that left 128 of 192 stage-1 tests untestable.

## 1. Object

The truth-level fine reference: sparse energy deposits on the prefix mesh of the Geant4 files, 216 x 216 x 36 voxels of 3 x 3 x 4.625 mm (flat index `(iz * 216 + iy) * 216 + ix`, `fine_index` and `fine_edep_mev`, stored in `electron_train`, `proton_train` and the sample and pilot sets). Energy-weighted: the box probability is the fraction of the event's total fine-voxel energy in the box.

## 2. Box scales (fixed here)

Boxes are aligned to the mesh and described by (kx, kz) voxels (kx in x and y, kz in z), nine scales: (1,1), (2,1), (3,2), (4,3), (6,4), (9,6), (18,12), (27,18), (54,36). Each divides the mesh exactly (216 divisible by kx; 36 divisible by kz). Physical box size 3kx x 3kx x 4.625kz mm; the scale used in the fit is the **effective side** s = (3kx x 3kx x 4.625kz)^(1/3) in mm (3.5, 5.5, 9.1, 11.5, 18.2, 27.3, 54.6, 82.0, 164 mm). Boxes are not exact cubes (aspect within 1.3 of unity except the finest, 3 x 3 x 4.625 mm); the aspect is recorded, not corrected. Dimensions D_q are the least-squares slope of ln Z_q (q = 0, 2) or of the entropy (q = 1) against ln s over a window, exactly as in the 2D estimator (`multiscale.py`); the statistics are D0, D1, D2 and D0 - D2 over every contiguous window of at least two of the nine scales (36 windows).

**Detector-aligned boxes are used in this part.** DECIDE-4 also asks for an axis-free principal-axis variant (truth axis only as a control); that variant is a later registered step because it changes the boxes, and no result of this part is read as "axis-free".

## 3. Controls and acceptance rules (the DEC-013 rules, unchanged numbers)

The numbers 95 percent, 10 percent, 90 percent, 90 percent and 15 percent are the project assumptions of DEC-013, not new choices; they are applied as in `multiscale_controls.py` with the hit counts 100, 300, 1000, 3000, 10000, 30000 (the real events have a median of about 6,000 occupied voxels at 20 GeV, so the range brackets them):
- **Smooth baseline (negative control):** a structure-free probability map on the mesh (gamma-like depth profile over the 36 slices x a Gaussian lateral core of sigma 9 voxels with a 2% floor): illustrative, not a calibrated shower. Hits are multinomial draws (equal quanta); the baseline's central 95% interval of each statistic and window defines "no structure".
- **False-positive limit:** fresh smooth replicates outside the baseline interval must be at most 10%.
- **Detection power and ordering (positive control):** a deterministic multiplicative cascade on the mesh: a 4 x 4 x 1 root split of the (54,36) boxes, then 2 x 2 x 2, 3 x 3 x 3, 3 x 3 x 3 and 3 x 3 x 2 splits down to the voxels, with fixed separable weights (0.7 / 0.3 style as in 2D); its D0 - D2 must exceed the baseline's 97.5th percentile in at least 90% of replicates and D0 >= D1 >= D2 must hold in at least 90%.
- **Recovery (reference):** the estimate of D_q across each aligned isotropic cascade level ((27,18) to (54,36), (9,6) to (27,18), (3,2) to (9,6)) against the exact value must be within 15% (reference, as in 2D); the anisotropic last level ((1,1) to (3,2)) is reported but not used for the rule.
- 1,000 replicates per hit count and map (200 for the recovery table); fixed seed.
- **Window usable at a hit count** = passes false-positive limit, power and ordering. **Eligibility threshold of a window** = the median number of occupied voxels of the control images at the lowest hit count at which the window is usable (the 2D rule: median occupied cells). An event is eligible for a window if its number of occupied fine voxels is at least that threshold. No saturation-index rule with assumed constants (A, D_f) is used: the controls decide.

## 4. Reported

The usable windows per hit count, their thresholds, the recovery table, and then the **eligible fraction of real events** for each window: electrons and truth-interacting protons of the development-large sets at 10, 20, 50 and 100 GeV (fine voxels only; the occupied-voxel count of an event is the only quantity read, no D_q of real events in this part), against the 2D eligibility (336 / 572 cells, stage-1 record). Plus the count of the 2D untestable tests that become eligible in 3D.

## 5. Reading rules written in advance

- A window is "usable" only on the controls; usable windows say that the estimator can tell the cascade from the smooth map at that number of occupied voxels, not that real showers have scaling.
- If no window is usable at any hit count the result is "no 3D window under these rules" and the 3D route is reported as not available (a result).
- The stage-1 and 3D eligibilities are not compared as a validation of either: the 3D rule replaces the 2D thresholds only if the controls pass.

## 6. What this part does not do

D_q of real events, class comparisons, degradation curves to the readout (they need the R-A choice), the principal-axis variant, anything on actual AMS performance, all sealed data.
