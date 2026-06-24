# Release Manifest

## Code

- `analysis/build_mimic_derived_feature_cache.py`: builds the restricted local MIMIC derived cache from credentialed MIMIC-IV files.
- `analysis/run_external_validation.py`: fits MIMIC development models, applies unchanged coefficients to eICU, and writes aggregate validation outputs.
- `analysis/make_tables_and_figures.py`: recreates public figures from aggregate output CSV files.
- `analysis/run_mimic_*.py`: helper scripts used by the MIMIC cache builder for first-day laboratory, OASIS-like, SOFA-like, and sensitivity components.

## Public Aggregate Results

- `results/eicu_external_validation/cohort_summary.csv`
- `results/eicu_external_validation/external_validation_model_metrics.csv`
- `results/eicu_external_validation/external_validation_delta_auc.csv`
- `results/eicu_external_validation/external_validation_dca.csv`
- `results/eicu_external_validation/eicu_calibration_deciles.csv`
- `results/eicu_external_validation/eicu_lab_coverage.csv`
- `results/eicu_external_validation/lab_distribution_mimic_vs_eicu.csv`
- `results/eicu_external_validation/mimic_development_rcs_knots.csv`
- `results/eicu_external_validation/eicu_apache_benchmarks.csv`
- `results/eicu_external_validation/eicu_apache_incremental_cv.csv`
- `results/manuscript_tables/*.csv`

## Public Figures

- `figures/Figure1_study_design_flow.png`
- `figures/Figure2_auc_validation_layers.png`
- `figures/Figure3_eicu_calibration_deciles.png`
- `figures/Figure4_eicu_decision_curve.png`

## Excluded From Public Release

- raw MIMIC-IV and eICU database files;
- local derived caches in `derived_cache/`;
- individual prediction files such as `eicu_external_predictions_restricted.csv.gz`;
- submission DOCX/PDF files containing author-specific formatting.

