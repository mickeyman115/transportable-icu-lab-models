# Reproducibility Notes

## Cohort Definition

The target population is first ICU/unit stays with survival to 24 hours after ICU admission and complete first-day values for glucose, creatinine, BUN, and WBC. Patients dying at or before 24 hours are excluded because their outcome occurs before completion of the predictor window.

## Outcome

The outcome is in-hospital mortality after the 24-hour landmark. Patients discharged alive are treated as non-events.

## Predictor Encoding

The binary comparator uses four separate high-threshold indicators based on the same peak first-day laboratory values used in the nonlinear spline model. This comparator is intended to isolate threshold coding versus nonlinear continuous coding, not aggregate-score construction.

## Public Versus Restricted Outputs

Aggregate CSV files in `results/` can be shared. Restricted local outputs, including row-level derived caches and prediction files, must remain outside the public repository.

## Known Boundary

The primary encoding comparison uses peak values and high-threshold binary indicators. Prognostic information from low glucose or leukopenia was not evaluated in the primary comparator and would require models incorporating both minimum and maximum values.

