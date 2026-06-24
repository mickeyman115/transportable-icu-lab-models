#!/usr/bin/env python3
"""Aggregate/model-only 0-6h acute profile sensitivity analysis."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from run_mimic_acute_on_chronic_kill_test import (
    CHARTEVENTS_ITEMS,
    CONCEPTS,
    DATA_DIR_DEFAULT,
    LABEVENTS_ITEMS,
    add_collapsed_categories,
    build_first_icu,
    update_arrays,
    valid_value,
    write_csv,
)
from run_mimic_oasis_charlson_benchmark import (
    add_charlson,
    correlation_table,
    delta_auc,
    run_model,
    scan_oasis_features,
    score_oasis_like,
)


OUT_DIR_DEFAULT = Path(
    "results/early_0_6h_sensitivity"
)


def scan_labs_window(
    data_dir: Path,
    first: pd.DataFrame,
    window_hours: int,
    lab_chunk_size: int,
    chart_chunk_size: int,
) -> pd.DataFrame:
    stay_ids = first["stay_id"].astype(int).tolist()
    stay_to_pos = {sid: i for i, sid in enumerate(stay_ids)}
    mins = {c: np.full(len(stay_ids), np.nan) for c in CONCEPTS}
    maxs = {c: np.full(len(stay_ids), np.nan) for c in CONCEPTS}

    lab_concept_by_item = {
        itemid: concept for concept, itemids in LABEVENTS_ITEMS.items() for itemid in itemids
    }
    chart_concept_by_item = {
        itemid: concept for concept, itemids in CHARTEVENTS_ITEMS.items() for itemid in itemids
    }

    key_lab = first[["subject_id", "hadm_id", "stay_id", "intime"]].copy()
    key_lab["subject_id"] = key_lab["subject_id"].astype("int64")
    key_lab["hadm_id"] = key_lab["hadm_id"].astype("int64")
    reader = pd.read_csv(
        data_dir / "hosp/labevents.csv.gz",
        usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
        chunksize=lab_chunk_size,
        dtype={"subject_id": "int64", "hadm_id": "float64", "itemid": "int64"},
    )
    for chunk in reader:
        chunk = chunk[chunk["itemid"].isin(lab_concept_by_item) & chunk["hadm_id"].notna()]
        if chunk.empty:
            continue
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk = chunk.merge(key_lab, on=["subject_id", "hadm_id"], how="inner")
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["hours_from_icu"] = (chunk["charttime"] - chunk["intime"]).dt.total_seconds() / 3600
        chunk = chunk[chunk["hours_from_icu"].between(0, window_hours, inclusive="both")]
        chunk["concept"] = chunk["itemid"].map(lab_concept_by_item)
        chunk = chunk[chunk["valuenum"].notna()]
        keep_parts = []
        for concept, g in chunk.groupby("concept"):
            keep_parts.append(g[valid_value(concept, g["valuenum"])])
        if not keep_parts:
            continue
        chunk = pd.concat(keep_parts, ignore_index=True)
        grouped = chunk.groupby(["stay_id", "concept"], as_index=False)["valuenum"].agg(["min", "max"])
        update_arrays(grouped, stay_to_pos, mins, maxs)

    key_chart = first[["stay_id", "intime"]].copy()
    key_chart["stay_id"] = key_chart["stay_id"].astype("int64")
    reader = pd.read_csv(
        data_dir / "icu/chartevents.csv.gz",
        usecols=["stay_id", "itemid", "charttime", "valuenum"],
        chunksize=chart_chunk_size,
        dtype={"stay_id": "float64", "itemid": "int64"},
    )
    for chunk in reader:
        chunk = chunk[chunk["itemid"].isin(chart_concept_by_item) & chunk["stay_id"].notna()]
        if chunk.empty:
            continue
        chunk["stay_id"] = chunk["stay_id"].astype("int64")
        chunk = chunk.merge(key_chart, on="stay_id", how="inner")
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["hours_from_icu"] = (chunk["charttime"] - chunk["intime"]).dt.total_seconds() / 3600
        chunk = chunk[chunk["hours_from_icu"].between(0, window_hours, inclusive="both")]
        chunk["concept"] = chunk["itemid"].map(chart_concept_by_item)
        chunk = chunk[chunk["valuenum"].notna()]
        keep_parts = []
        for concept, g in chunk.groupby("concept"):
            keep_parts.append(g[valid_value(concept, g["valuenum"])])
        if not keep_parts:
            continue
        chunk = pd.concat(keep_parts, ignore_index=True)
        grouped = chunk.groupby(["stay_id", "concept"], as_index=False)["valuenum"].agg(["min", "max"])
        update_arrays(grouped, stay_to_pos, mins, maxs)

    labdf = pd.DataFrame({"stay_id": stay_ids})
    for concept in CONCEPTS:
        labdf[f"{concept}_min_{window_hours}h"] = mins[concept]
        labdf[f"{concept}_max_{window_hours}h"] = maxs[concept]
    labdf[f"core4_complete_{window_hours}h"] = labdf[
        [f"{c}_max_{window_hours}h" for c in CONCEPTS]
    ].notna().all(axis=1)
    labdf["metabolic_stress"] = (
        (labdf[f"glucose_min_{window_hours}h"] < 70) | (labdf[f"glucose_max_{window_hours}h"] >= 180)
    ).astype(int)
    labdf["renal_stress"] = (
        (labdf[f"creatinine_max_{window_hours}h"] >= 1.5) | (labdf[f"bun_max_{window_hours}h"] >= 30)
    ).astype(int)
    labdf["inflammatory_stress"] = (
        (labdf[f"wbc_min_{window_hours}h"] < 4) | (labdf[f"wbc_max_{window_hours}h"] > 12)
    ).astype(int)
    labdf["acute_stress_score"] = labdf[
        ["metabolic_stress", "renal_stress", "inflammatory_stress"]
    ].sum(axis=1)
    labdf.loc[~labdf[f"core4_complete_{window_hours}h"], "acute_stress_score"] = np.nan
    return labdf


def event_table(df: pd.DataFrame, group_col: str, outcome: str) -> pd.DataFrame:
    out = (
        df.groupby(group_col, dropna=False)
        .agg(n=("subject_id", "size"), events=(outcome, "sum"), age_median=("anchor_age", "median"))
        .reset_index()
    )
    out["event_rate_percent"] = out["events"] / out["n"] * 100
    return out


def component_coverage(df: pd.DataFrame, window_hours: int) -> pd.DataFrame:
    rows = []
    for concept in CONCEPTS:
        col = f"{concept}_max_{window_hours}h"
        rows.append(
            {
                "component": concept,
                "n": int(len(df)),
                "non_missing": int(df[col].notna().sum()),
                "coverage_percent": float(df[col].notna().mean() * 100),
            }
        )
    rows.append(
        {
            "component": "core4_complete",
            "n": int(len(df)),
            "non_missing": int(df[f"core4_complete_{window_hours}h"].sum()),
            "coverage_percent": float(df[f"core4_complete_{window_hours}h"].mean() * 100),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--window-hours", type=int, default=6)
    parser.add_argument("--lab-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--chart-chunk-size", type=int, default=2_000_000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    first = build_first_icu(args.data_dir)
    labdf = scan_labs_window(
        args.data_dir,
        first,
        args.window_hours,
        args.lab_chunk_size,
        args.chart_chunk_size,
    )
    oasis_features = scan_oasis_features(args.data_dir, first, args.chart_chunk_size)
    charlson = add_charlson(args.data_dir, first)

    df = first.merge(labdf, on="stay_id", how="left")
    df = df.merge(oasis_features, on="stay_id", how="left")
    df = df.merge(charlson, on="hadm_id", how="left")
    df = add_collapsed_categories(df)
    df = score_oasis_like(df, args.data_dir)

    core_col = f"core4_complete_{args.window_hours}h"
    main = df[df["alive_at_24h"] & df[core_col]].copy()
    icu24 = main[main["still_in_icu_at_24h"]].copy()

    write_csv(component_coverage(df[df["alive_at_24h"]].copy(), args.window_hours), args.out_dir / "component_coverage_alive24.csv")
    write_csv(event_table(main, "acute_stress_score", "death_30d_after_landmark"), args.out_dir / "event_rates_by_acute_score_30d.csv")
    write_csv(event_table(main, "acute_stress_score", "death_365d_after_landmark"), args.out_dir / "event_rates_by_acute_score_365d.csv")
    write_csv(correlation_table(main), args.out_dir / "profile_oasis_correlations.csv")
    write_csv(
        pd.DataFrame(
            [
                {
                    "cohort": "alive24_core4_complete_0_6h",
                    "n": len(main),
                    "death_30d_after_landmark": int(main["death_30d_after_landmark"].sum()),
                    "death_365d_after_landmark": int(main["death_365d_after_landmark"].sum()),
                    "oasis_like_median": float(main["oasis_like"].median()),
                    "charlson_no_age_median": float(main["charlson_comorbidity_no_age"].median()),
                },
                {
                    "cohort": "alive24_still_icu_core4_complete_0_6h",
                    "n": len(icu24),
                    "death_30d_after_landmark": int(icu24["death_30d_after_landmark"].sum()),
                    "death_365d_after_landmark": int(icu24["death_365d_after_landmark"].sum()),
                    "oasis_like_median": float(icu24["oasis_like"].median()),
                    "charlson_no_age_median": float(icu24["charlson_comorbidity_no_age"].median()),
                },
            ]
        ),
        args.out_dir / "cohort_summary.csv",
    )

    model_specs = [
        ("M0_base", ["base"]),
        ("M1_base_charlson", ["base", "charlson"]),
        ("M2_base_charlson_acute", ["base", "charlson", "acute"]),
        ("M3_base_charlson_oasis", ["base", "charlson", "oasis"]),
        ("M4_base_charlson_oasis_acute", ["base", "charlson", "oasis", "acute"]),
    ]
    rows = []
    for cohort_name, cohort_df in [
        ("alive24_core4_complete_0_6h", main),
        ("alive24_still_icu_core4_complete_0_6h", icu24),
    ]:
        for outcome in ["death_30d_after_landmark", "death_365d_after_landmark"]:
            for model_name, terms in model_specs:
                rows.append(run_model(cohort_df, outcome, terms, model_name, cohort_name))
    models = pd.DataFrame(rows)
    write_csv(models, args.out_dir / "benchmark_model_summary.csv")
    write_csv(delta_auc(models), args.out_dir / "model_delta_auc.csv")

    print(
        json.dumps(
            {
                "window_hours": args.window_hours,
                "n_alive24_core4_complete": int(len(main)),
                "n_alive24_still_icu_core4_complete": int(len(icu24)),
                "out_dir": str(args.out_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
