<!-- Derived by scripts/notebook_coverage.py from the repository. Regenerate it; do not edit it by hand. It never creates or changes a notebook. -->

# Notebook coverage

Scanned 2026-10-17: **86 source modules** under `src/ams_ecal/`, **10 covered** (a notebook imports them), **2 mentioned only**, **74 with no notebook**. 9 notebooks read.

Notebooks here document completed, tested implementations. This list says where none exists yet; whether a unit is *complete*, or worth a notebook, is the researcher's decision. "Status" is the module map's own word (`unknown` if the map does not list the module); "tested" says whether the test file the map names exists.

## No notebook covers these (74)

| Unit | Status | Tested | Lines | First seen |
|---|---|---|---|---|
| `benchmarks/learning_curves.py` | infrastructure | yes | 315 | 2026-10-17 |
| `detector/component_sampling_response.py` | analysis | yes | 132 | 2026-10-17 |
| `detector/component_sampling_response_validation.py` | analysis | yes | 158 | 2026-10-17 |
| `detector/sampling_response_em_share.py` | analysis | yes | 174 | 2026-10-17 |
| `detector/sampling_response.py` | calibration | yes | 184 | 2026-10-17 |
| `detector/sampling_response_validation.py` | control | yes | 315 | 2026-10-17 |
| `detector/crossing.py` | infrastructure | yes | 199 | 2026-10-17 |
| `detector/projection.py` | infrastructure | yes | 86 | 2026-10-17 |
| `detector/readout_response.py` | infrastructure | yes | 112 | 2026-10-17 |
| `electron_model/em_production_calibration.py` | calibration | yes | 285 | 2026-10-17 |
| `electron_model/em_production_lateral_calibration.py` | calibration | yes | 586 | 2026-10-17 |
| `electron_model/em_production.py` | generator | yes | 381 | 2026-10-17 |
| `electron_model/em_production_lateral.py` | generator | yes | 570 | 2026-10-17 |
| `electron_studies/em_exploration_check.py` | analysis | yes | 117 | 2026-10-17 |
| `electron_studies/em_extended_tail_study.py` | analysis | yes | 373 | 2026-10-17 |
| `electron_studies/em_extended_tail_study_plots.py` | analysis | yes | 161 | 2026-10-17 |
| `electron_studies/em_lateral_distinguishability.py` | analysis | yes | 189 | 2026-10-17 |
| `electron_studies/em_lateral_metric_resolution.py` | analysis | yes | 139 | 2026-10-17 |
| `electron_studies/em_lateral_occupancy_diagnostic.py` | analysis | yes | 118 | 2026-10-17 |
| `electron_studies/em_lateral_structure_study.py` | analysis | yes | 303 | 2026-10-17 |
| `electron_studies/em_lateral_width_diagnostic.py` | analysis | yes | 164 | 2026-10-17 |
| `electron_studies/em_longitudinal_fluctuations.py` | analysis | yes | 193 | 2026-10-17 |
| `electron_studies/em_longitudinal_structure_analysis.py` | analysis | yes | 864 | 2026-10-17 |
| `electron_studies/em_longitudinal_structure_plots.py` | analysis | yes | 284 | 2026-10-17 |
| `electron_studies/em_production_development_check.py` | analysis | yes | 183 | 2026-10-17 |
| `electron_studies/em_production_development_check_plots.py` | analysis | yes | 172 | 2026-10-17 |
| `electron_studies/em_production_lateral_check.py` | analysis | yes | 288 | 2026-10-17 |
| `electron_studies/em_production_lateral_check_plots.py` | analysis | yes | 90 | 2026-10-17 |
| `electron_studies/em_profile_shape_study.py` | analysis | yes | 607 | 2026-10-17 |
| `electron_studies/em_profile_shape_study_plots.py` | analysis | yes | 318 | 2026-10-17 |
| `electron_studies/em_depth_origin_calibration.py` | calibration | yes | 1112 | 2026-10-17 |
| `electron_studies/em_depth_origin_calibration_plots.py` | calibration | yes | 369 | 2026-10-17 |
| `electron_studies/em_production_lateral_e1_decision.py` | control | yes | 147 | 2026-10-17 |
| `electron_studies/em_production_lateral_tail_decision.py` | control | yes | 137 | 2026-10-17 |
| `electron_studies/em_lateral_tail_quantum.py` | unknown | yes | 128 | 2026-10-17 |
| `electron_studies/em_production_lateral_e2_decision.py` | unknown | yes | 148 | 2026-10-17 |
| `geant4_simulation/em_variant_initialisation_dump.py` | analysis | yes | 120 | 2026-10-17 |
| `geant4_simulation/geant4_truth.py` | infrastructure | yes | 117 | 2026-10-17 |
| `geant4_simulation/em_variant_swap_analysis.py` | unknown | yes | 142 | 2026-10-17 |
| `inference/cross_fitting.py` | analysis | yes | 148 | 2026-10-17 |
| `inference/missingness_controls.py` | analysis | yes | 92 | 2026-10-17 |
| `inference/multiplicity.py` | analysis | yes | 115 | 2026-10-17 |
| `inference/permutation_nulls.py` | analysis | yes | 174 | 2026-10-17 |
| `inference/rejection_limits.py` | analysis | yes | 198 | 2026-10-17 |
| `inference/coverage_simulations.py` | control | yes | 205 | 2026-10-17 |
| `inference/claim_specification.py` | process | yes | 45 | 2026-10-17 |
| `multiscale_estimators/ladder_sanity_audit.py` | analysis | yes | 316 | 2026-10-17 |
| `multiscale_estimators/multiscale.py` | analysis | yes | 111 | 2026-10-17 |
| `multiscale_estimators/multiscale_matched_pairs.py` | analysis | yes | 334 | 2026-10-17 |
| `multiscale_estimators/multiscale_shuffle_excess.py` | analysis | yes | 153 | 2026-10-17 |
| `multiscale_estimators/multiscale_verdict_reread.py` | analysis | yes | 273 | 2026-10-17 |
| `multiscale_estimators/multiscale_3d.py` | control | yes | 378 | 2026-10-17 |
| `multiscale_estimators/multiscale_controls.py` | control | yes | 250 | 2026-10-17 |
| `multiscale_estimators/eligibility_count.py` | unknown | yes | 122 | 2026-10-17 |
| `multiscale_estimators/multiscale_development.py` | unknown | no test named | 66 | 2026-10-17 |
| `multiscale_estimators/multiscale_gtrain_development.py` | unknown | yes | 132 | 2026-10-17 |
| `multiscale_estimators/multiscale_ladder.py` | unknown | yes | 195 | 2026-10-17 |
| `multiscale_estimators/multiscale_test.py` | unknown | yes | 131 | 2026-10-17 |
| `proton_model/physics_list_systematic.py` | analysis | yes | 111 | 2026-10-17 |
| `proton_model/proton_checks.py` | analysis | yes | 279 | 2026-10-17 |
| `proton_model/proton_dependency.py` | analysis | yes | 484 | 2026-10-17 |
| `proton_model/proton_structure_check.py` | analysis | no | 273 | 2026-10-17 |
| `proton_model/proton_calibration.py` | calibration | yes | 695 | 2026-10-17 |
| `proton_model/proton.py` | generator | yes | 461 | 2026-10-17 |
| `proton_model/proton_interacting.py` | generator | yes | 566 | 2026-10-17 |
| `proton_model/proton_lateral.py` | generator | yes | 692 | 2026-10-17 |
| `proton_model/proton_structure.py` | generator | yes | 932 | 2026-10-17 |
| `proton_model/proton_batch.py` | infrastructure | yes | 108 | 2026-10-17 |
| `proton_model/proton_config.py` | infrastructure | yes | 134 | 2026-10-17 |
| `proton_model/proton_validation.py` | validation | yes | 380 | 2026-10-17 |
| `validation/added_checks.py` | control | yes | 428 | 2026-10-17 |
| `validation/dataset.py` | infrastructure | yes | 516 | 2026-10-17 |
| `validation/sample_definitions.py` | infrastructure | yes | 377 | 2026-10-17 |
| `validation/sealed_set.py` | infrastructure | yes | 157 | 2026-10-17 |

## Named in a notebook, never imported (2)

| Unit | Status | Tested | Lines | Notebooks |
|---|---|---|---|---|
| `geant4_simulation/pilot_analysis.py` | analysis | yes | 244 | 08_geant4_proton_pilot.ipynb |
| `geant4_simulation/pilot_report.py` | analysis | yes | 700 | 08_geant4_proton_pilot.ipynb |

## Covered (10)

| Unit | Status | Tested | Lines | Notebooks |
|---|---|---|---|---|
| `detector/event.py` | infrastructure | yes | 202 | 03_canonical_event_model.ipynb |
| `detector/geometry.py` | infrastructure | yes | 697 | 00_ecal_calorimetry_and_geometry.ipynb, 01_tracker_state_and_projection.ipynb, 02_readout_orientation_and_cell_mapping.ipynb, 03_canonical_event_model.ipynb, 04_ecal_geometry_fidelity.ipynb, 05_longitudinal_em_shower.ipynb, 06_lateral_em_shower.ipynb, 07_stochastic_em_events.ipynb, 08_geant4_proton_pilot.ipynb |
| `detector/readout.py` | infrastructure | yes | 96 | 02_readout_orientation_and_cell_mapping.ipynb, 03_canonical_event_model.ipynb |
| `detector/tracking.py` | infrastructure | yes | 74 | 01_tracker_state_and_projection.ipynb, 02_readout_orientation_and_cell_mapping.ipynb, 03_canonical_event_model.ipynb, 06_lateral_em_shower.ipynb, 07_stochastic_em_events.ipynb |
| `detector/transport_geometry.py` | infrastructure | yes | 515 | 08_geant4_proton_pilot.ipynb |
| `electron_model/lateral.py` | generator | yes | 285 | 06_lateral_em_shower.ipynb, 07_stochastic_em_events.ipynb |
| `electron_model/longitudinal.py` | generator | yes | 291 | 05_longitudinal_em_shower.ipynb, 07_stochastic_em_events.ipynb |
| `electron_model/stochastic.py` | generator | yes | 355 | 07_stochastic_em_events.ipynb |
| `electron_model/fastmc_config.py` | infrastructure | yes | 346 | 05_longitudinal_em_shower.ipynb, 06_lateral_em_shower.ipynb, 07_stochastic_em_events.ipynb |
| `geant4_simulation/geant4_backend.py` | infrastructure | yes | 1494 | 08_geant4_proton_pilot.ipynb |

```notebook-coverage-state
{
 "closed": {},
 "first_seen": {
  "benchmarks/learning_curves": "2026-10-17",
  "detector/component_sampling_response": "2026-10-17",
  "detector/component_sampling_response_validation": "2026-10-17",
  "detector/crossing": "2026-10-17",
  "detector/projection": "2026-10-17",
  "detector/readout_response": "2026-10-17",
  "detector/sampling_response": "2026-10-17",
  "detector/sampling_response_em_share": "2026-10-17",
  "detector/sampling_response_validation": "2026-10-17",
  "electron_model/em_production": "2026-10-17",
  "electron_model/em_production_calibration": "2026-10-17",
  "electron_model/em_production_lateral": "2026-10-17",
  "electron_model/em_production_lateral_calibration": "2026-10-17",
  "electron_studies/em_depth_origin_calibration": "2026-10-17",
  "electron_studies/em_depth_origin_calibration_plots": "2026-10-17",
  "electron_studies/em_exploration_check": "2026-10-17",
  "electron_studies/em_extended_tail_study": "2026-10-17",
  "electron_studies/em_extended_tail_study_plots": "2026-10-17",
  "electron_studies/em_lateral_distinguishability": "2026-10-17",
  "electron_studies/em_lateral_metric_resolution": "2026-10-17",
  "electron_studies/em_lateral_occupancy_diagnostic": "2026-10-17",
  "electron_studies/em_lateral_structure_study": "2026-10-17",
  "electron_studies/em_lateral_tail_quantum": "2026-10-17",
  "electron_studies/em_lateral_width_diagnostic": "2026-10-17",
  "electron_studies/em_longitudinal_fluctuations": "2026-10-17",
  "electron_studies/em_longitudinal_structure_analysis": "2026-10-17",
  "electron_studies/em_longitudinal_structure_plots": "2026-10-17",
  "electron_studies/em_production_development_check": "2026-10-17",
  "electron_studies/em_production_development_check_plots": "2026-10-17",
  "electron_studies/em_production_lateral_check": "2026-10-17",
  "electron_studies/em_production_lateral_check_plots": "2026-10-17",
  "electron_studies/em_production_lateral_e1_decision": "2026-10-17",
  "electron_studies/em_production_lateral_e2_decision": "2026-10-17",
  "electron_studies/em_production_lateral_tail_decision": "2026-10-17",
  "electron_studies/em_profile_shape_study": "2026-10-17",
  "electron_studies/em_profile_shape_study_plots": "2026-10-17",
  "geant4_simulation/em_variant_initialisation_dump": "2026-10-17",
  "geant4_simulation/em_variant_swap_analysis": "2026-10-17",
  "geant4_simulation/geant4_truth": "2026-10-17",
  "inference/claim_specification": "2026-10-17",
  "inference/coverage_simulations": "2026-10-17",
  "inference/cross_fitting": "2026-10-17",
  "inference/missingness_controls": "2026-10-17",
  "inference/multiplicity": "2026-10-17",
  "inference/permutation_nulls": "2026-10-17",
  "inference/rejection_limits": "2026-10-17",
  "multiscale_estimators/eligibility_count": "2026-10-17",
  "multiscale_estimators/ladder_sanity_audit": "2026-10-17",
  "multiscale_estimators/multiscale": "2026-10-17",
  "multiscale_estimators/multiscale_3d": "2026-10-17",
  "multiscale_estimators/multiscale_controls": "2026-10-17",
  "multiscale_estimators/multiscale_development": "2026-10-17",
  "multiscale_estimators/multiscale_gtrain_development": "2026-10-17",
  "multiscale_estimators/multiscale_ladder": "2026-10-17",
  "multiscale_estimators/multiscale_matched_pairs": "2026-10-17",
  "multiscale_estimators/multiscale_shuffle_excess": "2026-10-17",
  "multiscale_estimators/multiscale_test": "2026-10-17",
  "multiscale_estimators/multiscale_verdict_reread": "2026-10-17",
  "proton_model/physics_list_systematic": "2026-10-17",
  "proton_model/proton": "2026-10-17",
  "proton_model/proton_batch": "2026-10-17",
  "proton_model/proton_calibration": "2026-10-17",
  "proton_model/proton_checks": "2026-10-17",
  "proton_model/proton_config": "2026-10-17",
  "proton_model/proton_dependency": "2026-10-17",
  "proton_model/proton_interacting": "2026-10-17",
  "proton_model/proton_lateral": "2026-10-17",
  "proton_model/proton_structure": "2026-10-17",
  "proton_model/proton_structure_check": "2026-10-17",
  "proton_model/proton_validation": "2026-10-17",
  "validation/added_checks": "2026-10-17",
  "validation/dataset": "2026-10-17",
  "validation/sample_definitions": "2026-10-17",
  "validation/sealed_set": "2026-10-17"
 },
 "last_scan": "2026-10-17"
}
```
