#!/usr/bin/env python3
"""Build a local derived-feature cache for the MIMIC lab-stress project.

The cache is a restricted local working artifact. It contains one row per first
ICU stay with deidentified row identifiers, derived laboratory/severity scores,
and landmark outcomes. It avoids raw event-level rows and free-text chart data.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from run_mimic_0_6h_sensitivity import scan_labs_window
from run_mimic_acute_on_chronic_kill_test import (
    DATA_DIR_DEFAULT,
    add_collapsed_categories,
    build_first_icu,
    scan_labs,
    write_csv,
)
from run_mimic_oasis_charlson_benchmark import (
    add_charlson,
    scan_first_day_urine,
    scan_oasis_features,
    score_oasis_like,
)
from run_mimic_sofa_benchmark import (
    scan_fio2_chartevents,
    scan_pafi,
    scan_sofa_labs,
    scan_vasopressors,
)


CACHE_DIR_DEFAULT = Path("derived_cache")
CACHE_FILE_DEFAULT = CACHE_DIR_DEFAULT / "mimic_first_icu_labstress_cache_v1.csv.gz"

warnings.filterwarnings("ignore", category=FutureWarning, message="DataFrameGroupBy.apply operated")


def add_anchor_year_fields(data_dir: Path, first: pd.DataFrame) -> pd.DataFrame:
    patients_path = data_dir / "hosp/patients.csv.gz"
    patients = pd.read_csv(patients_path, nrows=1)
    available = set(patients.columns)
    wanted = ["subject_id"]
    for col in ["anchor_year", "anchor_year_group"]:
        if col in available:
            wanted.append(col)
    if len(wanted) == 1:
        out = first.copy()
        out["anchor_year"] = np.nan
        out["anchor_year_group"] = "missing"
        return out
    years = pd.read_csv(patients_path, usecols=wanted)
    return first.merge(years, on="subject_id", how="left")


def rename_labs_by_window(labs: pd.DataFrame, window_hours: int) -> pd.DataFrame:
    rename = {
        "metabolic_stress": f"metabolic_stress_{window_hours}h",
        "renal_stress": f"renal_stress_{window_hours}h",
        "inflammatory_stress": f"inflammatory_stress_{window_hours}h",
        "acute_stress_score": f"acute_stress_score_{window_hours}h",
    }
    return labs.rename(columns=rename)


def score_sofa_like_from_components(
    first: pd.DataFrame,
    sofa_labs: pd.DataFrame,
    urine: pd.DataFrame,
    oasis_features: pd.DataFrame,
    pafi: pd.DataFrame,
    vaso: pd.DataFrame,
) -> pd.DataFrame:
    df = first[["stay_id"]].merge(sofa_labs, on="stay_id", how="left")
    df = df.merge(urine, on="stay_id", how="left")
    df = df.merge(oasis_features[["stay_id", "mbp_min", "gcs_min", "mechvent"]], on="stay_id", how="left")
    df = df.merge(pafi, on="stay_id", how="left")
    df = df.merge(vaso, on="stay_id", how="left")

    df["respiration"] = np.select(
        [
            df["pao2fio2ratio_vent_min"] < 100,
            df["pao2fio2ratio_vent_min"] < 200,
            df["pao2fio2ratio_novent_min"] < 300,
            df["pao2fio2ratio_novent_min"] < 400,
            df[["pao2fio2ratio_vent_min", "pao2fio2ratio_novent_min"]].notna().any(axis=1),
        ],
        [4, 3, 2, 1, 0],
        default=np.nan,
    )
    df["coagulation"] = np.select(
        [
            df["platelet_min"] < 20,
            df["platelet_min"] < 50,
            df["platelet_min"] < 100,
            df["platelet_min"] < 150,
            df["platelet_min"].notna(),
        ],
        [4, 3, 2, 1, 0],
        default=np.nan,
    )
    df["liver"] = np.select(
        [
            df["bilirubin_total_max"] >= 12,
            df["bilirubin_total_max"] >= 6,
            df["bilirubin_total_max"] >= 2,
            df["bilirubin_total_max"] >= 1.2,
            df["bilirubin_total_max"].notna(),
        ],
        [4, 3, 2, 1, 0],
        default=np.nan,
    )
    df["cardiovascular"] = np.select(
        [
            (df["rate_dopamine"] > 15) | (df["rate_epinephrine"] > 0.1) | (df["rate_norepinephrine"] > 0.1),
            (df["rate_dopamine"] > 5) | (df["rate_epinephrine"] <= 0.1) | (df["rate_norepinephrine"] <= 0.1),
            (df["rate_dopamine"] > 0) | (df["rate_dobutamine"] > 0),
            df["mbp_min"] < 70,
            df[["mbp_min", "rate_dopamine", "rate_dobutamine", "rate_epinephrine", "rate_norepinephrine"]]
            .notna()
            .any(axis=1),
        ],
        [4, 3, 2, 1, 0],
        default=np.nan,
    )
    df["cns"] = np.select(
        [
            df["gcs_min"].between(13, 14, inclusive="both"),
            df["gcs_min"].between(10, 12, inclusive="both"),
            df["gcs_min"].between(6, 9, inclusive="both"),
            df["gcs_min"] < 6,
            df["gcs_min"].notna(),
        ],
        [1, 2, 3, 4, 0],
        default=np.nan,
    )
    df["renal"] = np.select(
        [
            df["creatinine_max"] >= 5,
            df["urineoutput"] < 200,
            df["creatinine_max"].between(3.5, 5, inclusive="left"),
            df["urineoutput"] < 500,
            df["creatinine_max"].between(2, 3.5, inclusive="left"),
            df["creatinine_max"].between(1.2, 2, inclusive="left"),
            df[["urineoutput", "creatinine_max"]].notna().any(axis=1),
        ],
        [4, 4, 3, 3, 2, 1, 0],
        default=np.nan,
    )
    components = ["respiration", "coagulation", "liver", "cardiovascular", "cns", "renal"]
    df["sofa_like"] = df[components].fillna(0).sum(axis=1)
    df["sofa_like_nonresp"] = df[["coagulation", "liver", "cardiovascular", "cns", "renal"]].fillna(0).sum(axis=1)
    df = df.rename(
        columns={
            "respiration": "sofa_respiration",
            "coagulation": "sofa_coagulation",
            "liver": "sofa_liver",
            "cardiovascular": "sofa_cardiovascular",
            "cns": "sofa_cns",
            "renal": "sofa_renal",
        }
    )
    return df


def component_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = [
        "glucose_max_24h",
        "creatinine_max_24h",
        "bun_max_24h",
        "wbc_max_24h",
        "core4_complete_24h",
        "glucose_max_6h",
        "creatinine_max_6h",
        "bun_max_6h",
        "wbc_max_6h",
        "core4_complete_6h",
        "sofa_respiration",
        "sofa_coagulation",
        "sofa_liver",
        "sofa_cardiovascular",
        "sofa_cns",
        "sofa_renal",
        "oasis_like",
        "charlson_comorbidity_no_age",
    ]
    for col in cols:
        if col not in df.columns:
            continue
        if df[col].dtype == bool:
            non_missing = int(df[col].sum())
            coverage = float(df[col].mean() * 100)
        else:
            non_missing = int(df[col].notna().sum())
            coverage = float(df[col].notna().mean() * 100)
        rows.append({"component": col, "n": int(len(df)), "non_missing": non_missing, "coverage_percent": coverage})
    return pd.DataFrame(rows)


def write_restricted_readme(cache_dir: Path, cache_file: Path) -> None:
    readme = cache_dir / "README_RESTRICTED.md"
    readme.write_text(
        "\n".join(
            [
                "# Restricted derived cache",
                "",
                "This folder contains local derived working data from MIMIC-IV.",
                "Do not share, commit, upload, or email these files.",
                "",
                f"Main cache: `{cache_file.name}`",
                "",
                "The cache has one row per first ICU stay and contains derived variables,",
                "deidentified row identifiers, shifted time/order fields, and landmark outcomes.",
                "It does not contain raw event-level rows or free-text chart values.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
    parser.add_argument("--cache-file", type=Path, default=CACHE_FILE_DEFAULT)
    parser.add_argument("--lab-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--chart-chunk-size", type=int, default=2_000_000)
    parser.add_argument(
        "--skip-respiration",
        action="store_true",
        help="Skip PaO2/FiO2 respiratory SOFA reconstruction and cache non-respiratory SOFA-like only.",
    )
    args = parser.parse_args()
    args.cache_file.parent.mkdir(parents=True, exist_ok=True)

    first = build_first_icu(args.data_dir)
    first = add_anchor_year_fields(args.data_dir, first)
    first = add_collapsed_categories(first)

    lab24 = rename_labs_by_window(
        scan_labs(args.data_dir, first, args.lab_chunk_size, args.chart_chunk_size),
        24,
    )
    lab6 = rename_labs_by_window(
        scan_labs_window(args.data_dir, first, 6, args.lab_chunk_size, args.chart_chunk_size),
        6,
    )

    oasis_features = scan_oasis_features(args.data_dir, first, args.chart_chunk_size)
    oasis_scored = score_oasis_like(
        first.merge(oasis_features, on="stay_id", how="left"),
        args.data_dir,
    )

    charlson = add_charlson(args.data_dir, first)
    sofa_labs = scan_sofa_labs(args.data_dir, first, args.lab_chunk_size)
    urine = scan_first_day_urine(args.data_dir, first)
    if args.skip_respiration:
        pafi = first[["stay_id"]].copy()
        pafi["pao2fio2ratio_novent_min"] = np.nan
        pafi["pao2fio2ratio_vent_min"] = np.nan
    else:
        fio2_chart = scan_fio2_chartevents(args.data_dir, first, args.chart_chunk_size)
        pafi = scan_pafi(args.data_dir, first, fio2_chart, args.lab_chunk_size)
    vaso = scan_vasopressors(args.data_dir, first)
    sofa = score_sofa_like_from_components(first, sofa_labs, urine, oasis_features, pafi, vaso)
    if args.skip_respiration:
        sofa["sofa_like"] = np.nan

    base_cols = [
        "stay_id",
        "hadm_id",
        "anchor_age",
        "gender",
        "anchor_year",
        "anchor_year_group",
        "intime",
        "landmark_24h",
        "admission_type",
        "admission_group",
        "first_careunit",
        "careunit_group",
        "race",
        "hospital_expire_flag",
        "alive_at_24h",
        "still_in_icu_at_24h",
        "death_30d_after_landmark",
        "death_365d_after_landmark",
    ]
    base_cols = [c for c in base_cols if c in first.columns]
    df = first[base_cols].copy()
    df = df.merge(lab24, on="stay_id", how="left")
    df = df.merge(lab6, on="stay_id", how="left")
    oasis_keep = [
        "stay_id",
        "oasis_like",
        "preiculos",
        "gcs_min",
        "heart_rate_min",
        "heart_rate_max",
        "mbp_min",
        "mbp_max",
        "resp_rate_min",
        "resp_rate_max",
        "temperature_min",
        "temperature_max",
        "urineoutput",
        "mechvent",
        "electivesurgery",
    ]
    df = df.merge(oasis_scored[[c for c in oasis_keep if c in oasis_scored.columns]], on="stay_id", how="left")
    df = df.merge(charlson, on="hadm_id", how="left")
    df = df.merge(sofa, on="stay_id", how="left")

    df["acute_stress_score"] = df["acute_stress_score_24h"]
    df["core4_complete"] = df["core4_complete_24h"]
    df["cache_version"] = "v1_2026-06-05"
    df["cache_sofa_respiration_reconstructed"] = not args.skip_respiration

    df.to_csv(args.cache_file, index=False, compression="gzip")
    write_restricted_readme(args.cache_file.parent, args.cache_file)

    analytic = df[df["alive_at_24h"] & df["core4_complete_24h"]].copy()
    summary = pd.DataFrame(
        [
            {
                "cohort": "all_first_icu",
                "n": int(len(df)),
                "alive_at_24h": int(df["alive_at_24h"].sum()),
                "core4_complete_24h": int(df["core4_complete_24h"].sum()),
                "death_30d_after_landmark": int(df["death_30d_after_landmark"].sum()),
                "death_365d_after_landmark": int(df["death_365d_after_landmark"].sum()),
            },
            {
                "cohort": "alive24_core4_complete",
                "n": int(len(analytic)),
                "alive_at_24h": int(analytic["alive_at_24h"].sum()),
                "core4_complete_24h": int(analytic["core4_complete_24h"].sum()),
                "death_30d_after_landmark": int(analytic["death_30d_after_landmark"].sum()),
                "death_365d_after_landmark": int(analytic["death_365d_after_landmark"].sum()),
            },
        ]
    )
    write_csv(summary, args.cache_file.parent / "cache_cohort_summary.csv")
    write_csv(component_coverage(analytic), args.cache_file.parent / "cache_component_coverage_alive24_core4.csv")

    print(
        json.dumps(
            {
                "cache_file": str(args.cache_file),
                "n_all_first_icu": int(len(df)),
                "n_alive24_core4_complete": int(len(analytic)),
                "columns": int(len(df.columns)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
