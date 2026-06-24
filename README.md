# Transportability of Nonlinear First-Day Routine Laboratory Models

This repository contains analysis code and shareable aggregate outputs for the manuscript:

**Transportability of Nonlinear First-Day Routine Laboratory Models for Post-Landmark In-Hospital Mortality After ICU Admission**

The study evaluates whether nonlinear continuous modelling of four routine first-day ICU laboratory values transports from MIMIC-IV to eICU for post-landmark in-hospital mortality. The code compares age/sex, binary high-threshold laboratory coding, restricted cubic spline laboratory coding, and a secondary BUN+WBC spline model.

## Data Access

Raw MIMIC-IV and eICU data are not included. They must be obtained directly from PhysioNet by credentialed users who complete the required training and data use agreements.

- MIMIC-IV v3.1: https://physionet.org/content/mimiciv/
- eICU Collaborative Research Database v2.0: https://physionet.org/content/eicu-crd/

Do not upload raw tables, derived patient-level caches, or individual prediction files to this repository.

## Expected Local Data Layout

The scripts can be run with explicit command-line paths. The examples below assume this local layout:

```text
data/
  mimic-iv-3.1/
    hosp/
    icu/
  eicu-crd-2.0/
    patient.csv.gz
    lab.csv.gz
    apachePatientResult.csv.gz
derived_cache/
results/
figures/
```

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the Main Analysis

Build the local MIMIC derived cache:

```bash
python analysis/build_mimic_derived_feature_cache.py \
  --data-dir data/mimic-iv-3.1 \
  --cache-file derived_cache/mimic_first_icu_labstress_cache_v1.csv.gz
```

Run the MIMIC-to-eICU external validation:

```bash
python analysis/run_external_validation.py \
  --mimic-cache derived_cache/mimic_first_icu_labstress_cache_v1.csv.gz \
  --eicu-dir data/eicu-crd-2.0 \
  --eicu-cache derived_cache/eicu_first_icu_labstress_cache_v1.csv.gz \
  --out-dir results/eicu_external_validation \
  --bootstrap-reps 300 \
  --rebuild-cache
```

Recreate public figures from aggregate outputs:

```bash
python analysis/make_tables_and_figures.py \
  --results-dir results/eicu_external_validation \
  --figure-dir figures
```

## Shareable Outputs Included

This release includes aggregate CSV files and figure images. These files contain cohort counts, model performance summaries, delta AUC estimates, decision-curve summaries, laboratory distributions, and calibration-by-decile summaries.

The release deliberately excludes:

- raw MIMIC-IV and eICU tables;
- local derived patient-level caches;
- patient-level prediction files;
- any files containing `subject_id`, `hadm_id`, `stay_id`, or `patientunitstayid` as redistributable outputs.

## Main Model Definitions

- Outcome: in-hospital death after the 24-hour ICU landmark among patients alive at 24 hours.
- Predictor window: ICU admission to 24 hours.
- Core laboratory values: peak first-day glucose, creatinine, BUN, and WBC.
- Binary high-threshold comparator: glucose >=180 mg/dL, creatinine >=1.5 mg/dL, BUN >=30 mg/dL, and WBC >12 x 10^9/L.
- Nonlinear model: restricted cubic splines for continuous peak values.
- Penalisation: fixed L2 penalty lambda=1.0 for model-fitting numerical stability; not tuned using validation data.

## Suggested Data Availability Wording

MIMIC-IV v3.1 and eICU v2.0 are available through PhysioNet to credentialed users who complete required training and data use agreements. Analysis code used to define cohorts, fit models, and reproduce the main tables and figures is available in this repository. Derived patient-level datasets and individual prediction files cannot be redistributed because of database use agreements.

