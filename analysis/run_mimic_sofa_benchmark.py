#!/usr/bin/env python3
"""Aggregate/model-only SOFA-like benchmark kill test.

The script reconstructs a first-day SOFA-like score from local MIMIC-IV raw
tables and writes only aggregate/model outputs. It is intended as a stop/go
benchmark before investing in a full official MIMIC-code derived-table build.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score

from run_mimic_acute_on_chronic_kill_test import (
    DATA_DIR_DEFAULT,
    add_collapsed_categories,
    build_first_icu,
    scan_labs,
    write_csv,
)
from run_mimic_oasis_charlson_benchmark import (
    add_charlson,
    calibration_metrics,
    scan_first_day_urine,
    scan_oasis_features,
)


OUT_DIR_DEFAULT = Path("results/sofa_benchmark_kill_test")

SOFA_LAB_ITEMS = {
    "creatinine": [50912, 52024, 52546],
    "bilirubin_total": [50885, 53089],
    "platelet": [51265, 53189],
}
BG_ITEMS = [52033, 50816, 50821]
FIO2_CHART_ITEM = 223835
VASO_ITEMS = {
    221289: "epinephrine",
    221653: "dobutamine",
    221662: "dopamine",
    221906: "norepinephrine",
}

warnings.filterwarnings("ignore", category=FutureWarning, message="DataFrameGroupBy.apply operated")


def update_minmax(grouped: pd.DataFrame, stay_to_pos: dict[int, int], arrays: dict[str, np.ndarray]) -> None:
    for row in grouped.itertuples(index=False):
        pos = stay_to_pos.get(int(row.stay_id))
        if pos is None:
            continue
        concept = row.concept
        mn = float(row.min)
        mx = float(row.max)
        min_name = f"{concept}_min"
        max_name = f"{concept}_max"
        if min_name in arrays and (np.isnan(arrays[min_name][pos]) or mn < arrays[min_name][pos]):
            arrays[min_name][pos] = mn
        if max_name in arrays and (np.isnan(arrays[max_name][pos]) or mx > arrays[max_name][pos]):
            arrays[max_name][pos] = mx


def scan_sofa_labs(data_dir: Path, first: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    stay_ids = first["stay_id"].astype(int).tolist()
    stay_to_pos = {sid: i for i, sid in enumerate(stay_ids)}
    concept_by_item = {itemid: concept for concept, itemids in SOFA_LAB_ITEMS.items() for itemid in itemids}
    arrays = {
        "creatinine_max": np.full(len(stay_ids), np.nan),
        "bilirubin_total_max": np.full(len(stay_ids), np.nan),
        "platelet_min": np.full(len(stay_ids), np.nan),
    }
    key = first[["subject_id", "stay_id", "intime"]].copy()
    key["subject_id"] = key["subject_id"].astype("int64")
    reader = pd.read_csv(
        data_dir / "hosp/labevents.csv.gz",
        usecols=["subject_id", "itemid", "charttime", "valuenum"],
        chunksize=chunk_size,
        dtype={"subject_id": "int64", "itemid": "int64"},
    )
    for chunk in reader:
        chunk = chunk[chunk["itemid"].isin(concept_by_item) & chunk["valuenum"].notna()]
        if chunk.empty:
            continue
        chunk = chunk.merge(key, on="subject_id", how="inner")
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["hours_from_icu"] = (chunk["charttime"] - chunk["intime"]).dt.total_seconds() / 3600
        chunk = chunk[chunk["hours_from_icu"].between(-6, 24, inclusive="both")]
        if chunk.empty:
            continue
        chunk["concept"] = chunk["itemid"].map(concept_by_item)
        chunk = chunk[
            (
                (chunk["concept"].eq("creatinine") & chunk["valuenum"].between(0.1, 30))
                | (chunk["concept"].eq("bilirubin_total") & chunk["valuenum"].between(0.01, 80))
                | (chunk["concept"].eq("platelet") & chunk["valuenum"].between(1, 3000))
            )
        ]
        grouped = chunk.groupby(["stay_id", "concept"], as_index=False)["valuenum"].agg(["min", "max"])
        update_minmax(grouped, stay_to_pos, arrays)

    out = pd.DataFrame({"stay_id": stay_ids})
    for name, arr in arrays.items():
        out[name] = arr
    return out


def scan_fio2_chartevents(data_dir: Path, first: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    key = first[["subject_id", "intime"]].copy()
    key["subject_id"] = key["subject_id"].astype("int64")
    parts = []
    reader = pd.read_csv(
        data_dir / "icu/chartevents.csv.gz",
        usecols=["subject_id", "charttime", "itemid", "valuenum"],
        chunksize=chunk_size,
        dtype={"subject_id": "int64", "itemid": "int64"},
    )
    for chunk in reader:
        chunk = chunk[chunk["itemid"].eq(FIO2_CHART_ITEM) & chunk["valuenum"].notna()]
        if chunk.empty:
            continue
        chunk = chunk.merge(key, on="subject_id", how="inner")
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["hours_from_icu"] = (chunk["charttime"] - chunk["intime"]).dt.total_seconds() / 3600
        chunk = chunk[chunk["hours_from_icu"].between(-10, 24, inclusive="both")]
        if chunk.empty:
            continue
        fio2 = np.where(
            chunk["valuenum"].between(0.2, 1.0, inclusive="right"),
            chunk["valuenum"] * 100,
            np.where(chunk["valuenum"].between(20, 100, inclusive="both"), chunk["valuenum"], np.nan),
        )
        chunk = chunk.assign(fio2_chartevents=fio2)
        chunk = chunk[chunk["fio2_chartevents"].notna()]
        parts.append(chunk[["subject_id", "charttime", "fio2_chartevents"]])
    if not parts:
        return pd.DataFrame(columns=["subject_id", "charttime", "fio2_chartevents"])
    return pd.concat(parts, ignore_index=True).sort_values(["subject_id", "charttime"])


def scan_pafi(data_dir: Path, first: pd.DataFrame, fio2_chart: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    key = first[["subject_id", "stay_id", "intime"]].copy()
    key["subject_id"] = key["subject_id"].astype("int64")
    bg_parts = []
    reader = pd.read_csv(
        data_dir / "hosp/labevents.csv.gz",
        usecols=["subject_id", "specimen_id", "itemid", "charttime", "value", "valuenum"],
        chunksize=chunk_size,
        dtype={"subject_id": "int64", "specimen_id": "int64", "itemid": "int64"},
    )
    for chunk in reader:
        chunk = chunk[chunk["itemid"].isin(BG_ITEMS)]
        if chunk.empty:
            continue
        chunk = chunk.merge(key, on="subject_id", how="inner")
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["hours_from_icu"] = (chunk["charttime"] - chunk["intime"]).dt.total_seconds() / 3600
        chunk = chunk[chunk["hours_from_icu"].between(-6, 24, inclusive="both")]
        if chunk.empty:
            continue
        bg_parts.append(chunk[["subject_id", "stay_id", "specimen_id", "charttime", "itemid", "value", "valuenum"]])
    if not bg_parts:
        return pd.DataFrame(columns=["stay_id", "pao2fio2ratio_novent_min", "pao2fio2ratio_vent_min"])
    raw = pd.concat(bg_parts, ignore_index=True)
    keys = ["subject_id", "stay_id", "specimen_id", "charttime"]
    bg = raw[keys].drop_duplicates()
    specimen = (
        raw.loc[raw["itemid"].eq(52033), keys + ["value"]]
        .dropna(subset=["value"])
        .groupby(keys, as_index=False)["value"]
        .first()
        .rename(columns={"value": "specimen"})
    )
    fio2_lab = (
        raw.loc[raw["itemid"].eq(50816), keys + ["valuenum"]]
        .groupby(keys, as_index=False)["valuenum"]
        .max()
        .rename(columns={"valuenum": "fio2_lab"})
    )
    po2 = (
        raw.loc[raw["itemid"].eq(50821), keys + ["valuenum"]]
        .groupby(keys, as_index=False)["valuenum"]
        .max()
        .rename(columns={"valuenum": "po2"})
    )
    bg = bg.merge(specimen, on=keys, how="left").merge(fio2_lab, on=keys, how="left").merge(po2, on=keys, how="left")
    bg = bg[bg["po2"].notna()]
    bg["specimen"] = bg["specimen"].astype(str)
    bg = bg[bg["specimen"].eq("ART.")]
    if bg.empty:
        return pd.DataFrame(columns=["stay_id", "pao2fio2ratio_novent_min", "pao2fio2ratio_vent_min"])
    bg["fio2_lab"] = np.where(
        bg["fio2_lab"].between(20, 100, inclusive="both"),
        bg["fio2_lab"],
        np.where(bg["fio2_lab"].between(0.2, 1.0, inclusive="right"), bg["fio2_lab"] * 100, np.nan),
    )
    if not fio2_chart.empty:
        bg = bg.sort_values(["charttime", "subject_id"]).reset_index(drop=True)
        fio2_chart = fio2_chart.sort_values(["charttime", "subject_id"]).reset_index(drop=True)
        bg = pd.merge_asof(
            bg,
            fio2_chart,
            by="subject_id",
            on="charttime",
            direction="backward",
            tolerance=pd.Timedelta(hours=4),
        )
    else:
        bg["fio2_chartevents"] = np.nan
    bg["fio2_used"] = bg["fio2_lab"].fillna(bg["fio2_chartevents"])
    bg = bg[bg["fio2_used"].notna() & bg["fio2_used"].gt(0)]
    bg["pao2fio2ratio"] = 100 * bg["po2"] / bg["fio2_used"]
    bg["bg_row_id"] = np.arange(len(bg))
    vent = build_invasive_vent_events(data_dir, first)
    if vent.empty:
        bg["isvent"] = 0
    else:
        bg = bg.merge(vent, on="stay_id", how="left")
        bg["isvent_row"] = (
            bg["starttime"].notna()
            & (bg["charttime"] >= bg["starttime"])
            & (bg["charttime"] <= bg["endtime"])
        ).astype(int)
        isvent = bg.groupby("bg_row_id", as_index=False)["isvent_row"].max().rename(columns={"isvent_row": "isvent"})
        bg = (
            bg.drop(columns=["starttime", "endtime", "isvent_row"], errors="ignore")
            .drop_duplicates("bg_row_id")
            .merge(isvent, on="bg_row_id", how="left")
        )
    out = (
        bg.groupby("stay_id")
        .agg(
            pao2fio2ratio_novent_min=("pao2fio2ratio", lambda s: s[bg.loc[s.index, "isvent"].eq(0)].min()),
            pao2fio2ratio_vent_min=("pao2fio2ratio", lambda s: s[bg.loc[s.index, "isvent"].eq(1)].min()),
        )
        .reset_index()
    )
    return out


def build_invasive_vent_events(data_dir: Path, first: pd.DataFrame) -> pd.DataFrame:
    parts = []
    proc = pd.read_csv(
        data_dir / "icu/procedureevents.csv.gz",
        usecols=["stay_id", "itemid", "starttime", "endtime"],
        parse_dates=["starttime", "endtime"],
    )
    proc = proc[proc["itemid"].eq(225792)]
    if not proc.empty:
        parts.append(proc[["stay_id", "starttime", "endtime"]])
    if not parts:
        return pd.DataFrame(columns=["stay_id", "starttime", "endtime"])
    key = first[["stay_id", "intime"]].copy()
    key["start_window"] = key["intime"] - pd.Timedelta(hours=6)
    key["end_window"] = key["intime"] + pd.Timedelta(hours=24)
    vent = pd.concat(parts, ignore_index=True).merge(key, on="stay_id", how="inner")
    vent = vent[(vent["starttime"] <= vent["end_window"]) & (vent["endtime"] >= vent["start_window"])]
    return vent[["stay_id", "starttime", "endtime"]]


def scan_vasopressors(data_dir: Path, first: pd.DataFrame) -> pd.DataFrame:
    key = first[["stay_id", "intime"]].copy()
    key["start_window"] = key["intime"] - pd.Timedelta(hours=6)
    key["end_window"] = key["intime"] + pd.Timedelta(hours=24)
    inp = pd.read_csv(
        data_dir / "icu/inputevents.csv.gz",
        usecols=["stay_id", "itemid", "starttime", "endtime", "rate", "rateuom", "patientweight"],
        parse_dates=["starttime", "endtime"],
    )
    inp = inp[inp["itemid"].isin(VASO_ITEMS) & inp["rate"].notna()]
    inp = inp.merge(key, on="stay_id", how="inner")
    inp = inp[(inp["starttime"] >= inp["start_window"]) & (inp["starttime"] <= inp["end_window"])]
    inp["treatment"] = inp["itemid"].map(VASO_ITEMS)
    inp["vaso_rate"] = inp["rate"]
    nor_mg = inp["treatment"].eq("norepinephrine") & inp["rateuom"].eq("mg/kg/min") & ~inp["patientweight"].eq(1)
    inp.loc[nor_mg, "vaso_rate"] = inp.loc[nor_mg, "rate"] * 1000.0
    out = (
        inp.pivot_table(index="stay_id", columns="treatment", values="vaso_rate", aggfunc="max")
        .rename(
            columns={
                "epinephrine": "rate_epinephrine",
                "norepinephrine": "rate_norepinephrine",
                "dopamine": "rate_dopamine",
                "dobutamine": "rate_dobutamine",
            }
        )
        .reset_index()
    )
    for col in ["rate_epinephrine", "rate_norepinephrine", "rate_dopamine", "rate_dobutamine"]:
        if col not in out.columns:
            out[col] = np.nan
    return out


def build_sofa_like(data_dir: Path, first: pd.DataFrame, lab_chunk_size: int, chart_chunk_size: int) -> pd.DataFrame:
    labs = scan_sofa_labs(data_dir, first, lab_chunk_size)
    urine = scan_first_day_urine(data_dir, first)
    oasis_features = scan_oasis_features(data_dir, first, chart_chunk_size)
    fio2_chart = scan_fio2_chartevents(data_dir, first, chart_chunk_size)
    pafi = scan_pafi(data_dir, first, fio2_chart, lab_chunk_size)
    vaso = scan_vasopressors(data_dir, first)

    df = first[["stay_id"]].merge(labs, on="stay_id", how="left")
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
            df[["mbp_min", "rate_dopamine", "rate_dobutamine", "rate_epinephrine", "rate_norepinephrine"]].notna().any(axis=1),
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
    return df


def model_matrix(df: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    if "base" in terms:
        x["age"] = pd.to_numeric(df["anchor_age"], errors="coerce")
        x["male"] = df["gender"].eq("M").astype(int)
        dummies = pd.get_dummies(
            df[["admission_group", "careunit_group"]].fillna("missing"),
            drop_first=True,
            dtype=float,
        )
        x = pd.concat([x, dummies], axis=1)
    if "charlson" in terms:
        x["charlson_comorbidity_no_age"] = df["charlson_comorbidity_no_age"].astype(float)
    if "sofa" in terms:
        x["sofa_like"] = df["sofa_like"].astype(float)
    if "acute" in terms:
        x["acute_stress_score"] = df["acute_stress_score"].astype(float)
    return sm.add_constant(x, has_constant="add")


def run_model(df: pd.DataFrame, outcome: str, terms: list[str], model_name: str, cohort_name: str) -> dict[str, object]:
    cols = [
        outcome,
        "anchor_age",
        "gender",
        "admission_group",
        "careunit_group",
        "charlson_comorbidity_no_age",
        "sofa_like",
        "acute_stress_score",
    ]
    use = df[cols].dropna()
    y = use[outcome].astype(int)
    x = model_matrix(use, terms)
    res = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=200)
    pred = res.predict(x)
    cal_intercept, cal_slope = calibration_metrics(y, pred)
    row: dict[str, object] = {
        "cohort": cohort_name,
        "outcome": outcome,
        "model": model_name,
        "n": int(len(use)),
        "events": int(y.sum()),
        "event_rate_percent": float(y.mean() * 100),
        "auc": float(roc_auc_score(y, pred)),
        "brier": float(brier_score_loss(y, pred)),
        "calibration_intercept": cal_intercept,
        "calibration_slope": cal_slope,
    }
    for term in ["charlson_comorbidity_no_age", "sofa_like", "acute_stress_score"]:
        if term in res.params.index:
            beta = res.params[term]
            se = res.bse[term]
            row[f"{term}_or"] = float(np.exp(beta))
            row[f"{term}_ci_low"] = float(np.exp(beta - 1.96 * se))
            row[f"{term}_ci_high"] = float(np.exp(beta + 1.96 * se))
            row[f"{term}_p"] = float(res.pvalues[term])
    return row


def component_missingness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["respiration", "coagulation", "liver", "cardiovascular", "cns", "renal", "sofa_like", "acute_stress_score"]:
        rows.append(
            {
                "component": col,
                "n": int(len(df)),
                "non_missing": int(df[col].notna().sum()),
                "coverage_percent": float(df[col].notna().mean() * 100),
            }
        )
    return pd.DataFrame(rows)


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    variables = ["sofa_like", "respiration", "coagulation", "liver", "cardiovascular", "cns", "renal", "charlson_comorbidity_no_age"]
    rows = []
    for var in variables:
        use = df[["acute_stress_score", var]].dropna()
        rows.append(
            {
                "variable": var,
                "n": int(len(use)),
                "pearson_r_with_acute_score": float(use["acute_stress_score"].corr(use[var], method="pearson")),
                "spearman_r_with_acute_score": float(use["acute_stress_score"].corr(use[var], method="spearman")),
            }
        )
    return pd.DataFrame(rows)


def delta_auc(models: pd.DataFrame) -> pd.DataFrame:
    piv = models.pivot_table(index=["cohort", "outcome"], columns="model", values="auc", aggfunc="first").reset_index()
    rows = []
    for row in piv.itertuples(index=False):
        data = row._asdict()
        rows.append(
            {
                "cohort": data["cohort"],
                "outcome": data["outcome"],
                "delta_auc_acute_after_base_charlson": data.get("M2_base_charlson_acute", np.nan)
                - data.get("M1_base_charlson", np.nan),
                "delta_auc_sofa_after_base_charlson": data.get("M3_base_charlson_sofa", np.nan)
                - data.get("M1_base_charlson", np.nan),
                "delta_auc_acute_after_sofa_charlson": data.get("M4_base_charlson_sofa_acute", np.nan)
                - data.get("M3_base_charlson_sofa", np.nan),
            }
        )
    return pd.DataFrame(rows)


def event_table(df: pd.DataFrame, group_col: str, outcome: str) -> pd.DataFrame:
    tmp = df.copy()
    if group_col == "sofa_quartile":
        tmp[group_col] = pd.qcut(tmp["sofa_like"], q=4, duplicates="drop")
    out = (
        tmp.groupby(group_col, observed=True)
        .agg(n=("subject_id", "size"), events=(outcome, "sum"), age_median=("anchor_age", "median"))
        .reset_index()
    )
    out[group_col] = out[group_col].astype(str)
    out["event_rate_percent"] = out["events"] / out["n"] * 100
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--lab-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--chart-chunk-size", type=int, default=2_000_000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    first = build_first_icu(args.data_dir)
    labdf = scan_labs(args.data_dir, first, args.lab_chunk_size, args.chart_chunk_size)
    sofa = build_sofa_like(args.data_dir, first, args.lab_chunk_size, args.chart_chunk_size)
    charlson = add_charlson(args.data_dir, first)

    df = first.merge(labdf, on="stay_id", how="left")
    df = df.merge(sofa, on="stay_id", how="left")
    df = df.merge(charlson, on="hadm_id", how="left")
    df = add_collapsed_categories(df)

    main_cohort = df[df["alive_at_24h"] & df["core4_complete_24h"]].copy()
    icu24 = main_cohort[main_cohort["still_in_icu_at_24h"]].copy()

    write_csv(
        pd.DataFrame(
            [
                {
                    "cohort": "alive24_core4_complete",
                    "n": len(main_cohort),
                    "death_30d_after_landmark": int(main_cohort["death_30d_after_landmark"].sum()),
                    "death_365d_after_landmark": int(main_cohort["death_365d_after_landmark"].sum()),
                    "sofa_like_median": float(main_cohort["sofa_like"].median()),
                    "charlson_no_age_median": float(main_cohort["charlson_comorbidity_no_age"].median()),
                },
                {
                    "cohort": "alive24_still_icu_core4_complete",
                    "n": len(icu24),
                    "death_30d_after_landmark": int(icu24["death_30d_after_landmark"].sum()),
                    "death_365d_after_landmark": int(icu24["death_365d_after_landmark"].sum()),
                    "sofa_like_median": float(icu24["sofa_like"].median()),
                    "charlson_no_age_median": float(icu24["charlson_comorbidity_no_age"].median()),
                },
            ]
        ),
        args.out_dir / "cohort_summary.csv",
    )
    write_csv(component_missingness(main_cohort), args.out_dir / "component_missingness.csv")
    write_csv(correlation_table(main_cohort), args.out_dir / "profile_sofa_correlations.csv")
    write_csv(event_table(main_cohort, "sofa_quartile", "death_30d_after_landmark"), args.out_dir / "event_rates_by_sofa_quartile_30d.csv")
    write_csv(event_table(main_cohort, "sofa_quartile", "death_365d_after_landmark"), args.out_dir / "event_rates_by_sofa_quartile_365d.csv")

    model_specs = [
        ("M0_base", ["base"]),
        ("M1_base_charlson", ["base", "charlson"]),
        ("M2_base_charlson_acute", ["base", "charlson", "acute"]),
        ("M3_base_charlson_sofa", ["base", "charlson", "sofa"]),
        ("M4_base_charlson_sofa_acute", ["base", "charlson", "sofa", "acute"]),
    ]
    rows = []
    for cohort_name, cohort_df in [
        ("alive24_core4_complete", main_cohort),
        ("alive24_still_icu_core4_complete", icu24),
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
                "n_alive24_core4_complete": int(len(main_cohort)),
                "n_alive24_still_icu_core4_complete": int(len(icu24)),
                "out_dir": str(args.out_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
