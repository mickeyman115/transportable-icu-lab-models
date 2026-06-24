#!/usr/bin/env python3
"""Aggregate/model-only MIMIC acute-on-chronic preliminary kill test.

The script processes patient-level MIMIC data locally but writes only cohort-level
counts, event-rate tables, and model summaries. It does not export analytic rows.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score


DATA_DIR_DEFAULT = Path("data/mimic-iv-3.1")
OUT_DIR_DEFAULT = Path("results/acute_on_chronic_kill_test")

LABEVENTS_ITEMS: dict[str, list[int]] = {
    "glucose": [50809, 50931, 52027, 52569],
    "creatinine": [50912, 52024, 52546],
    "bun": [51006, 52647],
    "wbc": [51300, 51301, 51755, 51756],
}

CHARTEVENTS_ITEMS: dict[str, list[int]] = {
    "glucose": [220621, 225664, 226537, 228388],
    "creatinine": [220615, 229761],
    "bun": [225624],
    "wbc": [220546],
}

CONCEPTS = ["glucose", "creatinine", "bun", "wbc"]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def between_code(code: str, lo: int, hi: int) -> bool:
    digits = re.match(r"^(\d+)", str(code))
    if not digits:
        return False
    try:
        value = int(digits.group(1)[:3])
    except ValueError:
        return False
    return lo <= value <= hi


def diagnosis_category(code: str, version: int) -> list[str]:
    c = str(code).strip().upper().replace(".", "")
    v = int(version)
    cats: list[str] = []
    if v == 9:
        if c.startswith("250"):
            cats.append("diabetes")
        if between_code(c, 401, 405):
            cats.append("hypertension")
        if c.startswith("585"):
            cats.append("ckd")
        if c.startswith("428"):
            cats.append("heart_failure")
        if between_code(c, 410, 414):
            cats.append("ischemic_heart_disease")
        if c.startswith("410"):
            cats.append("myocardial_infarction")
        if between_code(c, 430, 438):
            cats.append("stroke_cerebrovascular")
        if c.startswith("2780"):
            cats.append("obesity")
        if c.startswith("272"):
            cats.append("dyslipidemia")
    elif v == 10:
        if c.startswith(("E08", "E09", "E10", "E11", "E12", "E13", "E14")):
            cats.append("diabetes")
        if c.startswith(("I10", "I11", "I12", "I13", "I14", "I15")):
            cats.append("hypertension")
        if c.startswith("N18"):
            cats.append("ckd")
        if c.startswith("I50"):
            cats.append("heart_failure")
        if c.startswith(("I20", "I21", "I22", "I23", "I24", "I25")):
            cats.append("ischemic_heart_disease")
        if c.startswith(("I21", "I22")):
            cats.append("myocardial_infarction")
        if c.startswith(("I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69", "G45")):
            cats.append("stroke_cerebrovascular")
        if c.startswith("E66"):
            cats.append("obesity")
        if c.startswith("E78"):
            cats.append("dyslipidemia")
    return cats


def build_first_icu(data_dir: Path) -> pd.DataFrame:
    patients = pd.read_csv(data_dir / "hosp/patients.csv.gz")
    patients["dod"] = pd.to_datetime(patients["dod"], errors="coerce")
    admissions = pd.read_csv(
        data_dir / "hosp/admissions.csv.gz",
        parse_dates=["admittime", "dischtime", "deathtime"],
    )
    icu = pd.read_csv(data_dir / "icu/icustays.csv.gz", parse_dates=["intime", "outtime"])
    first = (
        icu.sort_values(["subject_id", "intime", "stay_id"])
        .groupby("subject_id", as_index=False)
        .first()
    )
    first = first.merge(
        patients[["subject_id", "gender", "anchor_age", "dod"]],
        on="subject_id",
        how="left",
    )
    first = first.merge(
        admissions[
            [
                "subject_id",
                "hadm_id",
                "admittime",
                "dischtime",
                "deathtime",
                "admission_type",
                "race",
                "hospital_expire_flag",
            ]
        ],
        on=["subject_id", "hadm_id"],
        how="left",
    )
    first["landmark_24h"] = first["intime"] + pd.Timedelta(hours=24)
    early_hosp_death = (
        first["deathtime"].notna()
        & (first["deathtime"] >= first["intime"])
        & (first["deathtime"] <= first["landmark_24h"])
    )
    first["alive_at_24h"] = ~early_hosp_death
    first["still_in_icu_at_24h"] = first["outtime"].isna() | (first["outtime"] > first["landmark_24h"])
    first["death_30d_after_landmark"] = (
        first["alive_at_24h"]
        & first["dod"].notna()
        & (first["dod"] <= first["landmark_24h"] + pd.Timedelta(days=30))
    )
    first["death_365d_after_landmark"] = (
        first["alive_at_24h"]
        & first["dod"].notna()
        & (first["dod"] <= first["landmark_24h"] + pd.Timedelta(days=365))
    )
    return first


def add_chronic_burden(data_dir: Path, first: pd.DataFrame) -> pd.DataFrame:
    admissions = pd.read_csv(
        data_dir / "hosp/admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
        parse_dates=["admittime", "dischtime"],
    )
    dx = pd.read_csv(
        data_dir / "hosp/diagnoses_icd.csv.gz",
        dtype={"icd_code": str, "icd_version": int},
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
    )
    keep = first[["subject_id", "hadm_id", "admittime"]].rename(
        columns={"hadm_id": "first_hadm_id", "admittime": "first_admittime"}
    )
    dx = dx.merge(keep, on="subject_id", how="inner")
    dx = dx.merge(
        admissions.rename(columns={"admittime": "dx_admittime", "dischtime": "dx_dischtime"}),
        on=["subject_id", "hadm_id"],
        how="left",
    )
    dx["scope"] = np.where(
        dx["hadm_id"].eq(dx["first_hadm_id"]),
        "index_hadm",
        np.where(dx["dx_admittime"] < dx["first_admittime"], "prior_hadm", "ignore"),
    )
    dx = dx[dx["scope"].isin(["index_hadm", "prior_hadm"])]

    rows = []
    for row in dx.itertuples(index=False):
        for cat in diagnosis_category(row.icd_code, int(row.icd_version)):
            rows.append((int(row.subject_id), row.scope, cat))
    flags = pd.DataFrame(rows, columns=["subject_id", "scope", "category"])
    out = first.copy()
    marker_cols = [
        "diabetes",
        "hypertension",
        "dyslipidemia",
        "obesity",
        "ckd",
        "heart_failure",
        "ischemic_heart_disease",
        "myocardial_infarction",
        "stroke_cerebrovascular",
    ]
    if flags.empty:
        for scope in ["index_hadm", "prior_hadm"]:
            for col in marker_cols + [
                "metabolic_domain",
                "kidney_domain",
                "cardiovascular_domain",
                "ckm_domain_count",
            ]:
                out[f"{scope}_{col}"] = 0
        return out

    for scope in ["index_hadm", "prior_hadm"]:
        wide = pd.crosstab(
            flags.loc[flags["scope"].eq(scope), "subject_id"],
            flags.loc[flags["scope"].eq(scope), "category"],
        ).gt(0).astype(int)
        for col in marker_cols:
            if col not in wide.columns:
                wide[col] = 0
        wide = wide[marker_cols].reset_index()
        wide[f"{scope}_metabolic_domain"] = (
            wide[["diabetes", "hypertension", "dyslipidemia", "obesity"]].sum(axis=1) > 0
        ).astype(int)
        wide[f"{scope}_kidney_domain"] = wide["ckd"].astype(int)
        wide[f"{scope}_cardiovascular_domain"] = (
            wide[
                [
                    "heart_failure",
                    "ischemic_heart_disease",
                    "myocardial_infarction",
                    "stroke_cerebrovascular",
                ]
            ].sum(axis=1)
            > 0
        ).astype(int)
        wide[f"{scope}_ckm_domain_count"] = wide[
            [
                f"{scope}_metabolic_domain",
                f"{scope}_kidney_domain",
                f"{scope}_cardiovascular_domain",
            ]
        ].sum(axis=1)
        rename = {col: f"{scope}_{col}" for col in marker_cols}
        wide = wide.rename(columns=rename)
        out = out.merge(wide, on="subject_id", how="left")

    fill_cols = [c for c in out.columns if c.startswith(("index_hadm_", "prior_hadm_"))]
    out[fill_cols] = out[fill_cols].fillna(0).astype(int)
    return out


def valid_value(concept: str, value: pd.Series) -> pd.Series:
    if concept == "glucose":
        return value.between(10, 1000)
    if concept == "creatinine":
        return value.between(0.1, 30)
    if concept == "bun":
        return value.between(1, 300)
    if concept == "wbc":
        return value.between(0.1, 500)
    return value.notna()


def update_arrays(
    grouped: pd.DataFrame,
    stay_to_pos: dict[int, int],
    mins: dict[str, np.ndarray],
    maxs: dict[str, np.ndarray],
) -> None:
    for row in grouped.itertuples(index=False):
        pos = stay_to_pos.get(int(row.stay_id))
        if pos is None:
            continue
        concept = row.concept
        mn = float(row.min)
        mx = float(row.max)
        if math.isnan(mins[concept][pos]) or mn < mins[concept][pos]:
            mins[concept][pos] = mn
        if math.isnan(maxs[concept][pos]) or mx > maxs[concept][pos]:
            maxs[concept][pos] = mx


def scan_labs(data_dir: Path, first: pd.DataFrame, lab_chunk_size: int, chart_chunk_size: int) -> pd.DataFrame:
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
        chunk = chunk[chunk["hours_from_icu"].between(0, 24, inclusive="both")]
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
        chunk = chunk[chunk["hours_from_icu"].between(0, 24, inclusive="both")]
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
        labdf[f"{concept}_min_24h"] = mins[concept]
        labdf[f"{concept}_max_24h"] = maxs[concept]
    labdf["core4_complete_24h"] = labdf[
        [f"{c}_max_24h" for c in CONCEPTS]
    ].notna().all(axis=1)
    labdf["metabolic_stress"] = (
        (labdf["glucose_min_24h"] < 70) | (labdf["glucose_max_24h"] >= 180)
    ).astype(int)
    labdf["renal_stress"] = (
        (labdf["creatinine_max_24h"] >= 1.5) | (labdf["bun_max_24h"] >= 30)
    ).astype(int)
    labdf["inflammatory_stress"] = (
        (labdf["wbc_min_24h"] < 4) | (labdf["wbc_max_24h"] > 12)
    ).astype(int)
    labdf["acute_stress_score"] = labdf[
        ["metabolic_stress", "renal_stress", "inflammatory_stress"]
    ].sum(axis=1)
    labdf.loc[~labdf["core4_complete_24h"], "acute_stress_score"] = np.nan
    labdf["acute_high_ge2"] = (labdf["acute_stress_score"] >= 2).astype(int)
    labdf.loc[~labdf["core4_complete_24h"], "acute_high_ge2"] = np.nan
    return labdf


def event_table(df: pd.DataFrame, group_cols: list[str], outcome: str) -> pd.DataFrame:
    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("subject_id", "size"),
            events=(outcome, "sum"),
            age_median=("anchor_age", "median"),
        )
        .reset_index()
    )
    out["event_rate_percent"] = out["events"] / out["n"] * 100
    return out


def add_collapsed_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    adm = out["admission_type"].fillna("missing").astype(str).str.upper()
    out["admission_group"] = np.select(
        [
            adm.str.contains("ELECTIVE|SURGICAL SAME DAY", regex=True),
            adm.str.contains("URGENT", regex=True),
            adm.str.contains("OBSERVATION", regex=True),
            adm.str.contains("EMER", regex=True),
        ],
        ["elective", "urgent", "observation", "emergency"],
        default="other",
    )
    care = out["first_careunit"].fillna("missing").astype(str)
    care_map = {
        "Medical Intensive Care Unit (MICU)": "MICU",
        "Medical/Surgical Intensive Care Unit (MICU/SICU)": "MICU_SICU",
        "Surgical Intensive Care Unit (SICU)": "SICU",
        "Cardiac Vascular Intensive Care Unit (CVICU)": "CVICU",
        "Coronary Care Unit (CCU)": "CCU",
        "Trauma SICU (TSICU)": "TSICU",
    }
    out["careunit_group"] = care.map(care_map).fillna("OTHER")
    return out


def model_matrix(df: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    base = pd.DataFrame(index=df.index)
    if "base" in terms:
        base["age"] = pd.to_numeric(df["anchor_age"], errors="coerce")
        base["male"] = df["gender"].eq("M").astype(int)
        dummies = pd.get_dummies(
            df[["admission_group", "careunit_group"]].fillna("missing"),
            drop_first=True,
            dtype=float,
        )
        base = pd.concat([base, dummies], axis=1)
    if "chronic" in terms:
        base["chronic_domain_count"] = df["index_hadm_ckm_domain_count"].astype(float)
    if "acute" in terms:
        base["acute_stress_score"] = df["acute_stress_score"].astype(float)
    if "interaction" in terms:
        base["acute_x_chronic"] = (
            df["acute_stress_score"].astype(float) * df["index_hadm_ckm_domain_count"].astype(float)
        )
    return sm.add_constant(base, has_constant="add")


def run_logit(df: pd.DataFrame, outcome: str, terms: list[str], model_name: str, cohort_name: str) -> dict[str, object]:
    use = df[
        [
            outcome,
            "anchor_age",
            "gender",
            "admission_group",
            "careunit_group",
            "index_hadm_ckm_domain_count",
            "acute_stress_score",
        ]
    ].dropna()
    y = use[outcome].astype(int)
    X = model_matrix(use, terms)
    result = sm.GLM(y, X, family=sm.families.Binomial()).fit(maxiter=200)
    pred = result.predict(X)
    auc = roc_auc_score(y, pred)
    row: dict[str, object] = {
        "cohort": cohort_name,
        "outcome": outcome,
        "model": model_name,
        "n": int(len(use)),
        "events": int(y.sum()),
        "event_rate_percent": float(y.mean() * 100),
        "auc": float(auc),
    }
    for term in ["chronic_domain_count", "acute_stress_score", "acute_x_chronic"]:
        if term in result.params.index:
            beta = result.params[term]
            se = result.bse[term]
            row[f"{term}_or"] = float(np.exp(beta))
            row[f"{term}_ci_low"] = float(np.exp(beta - 1.96 * se))
            row[f"{term}_ci_high"] = float(np.exp(beta + 1.96 * se))
            row[f"{term}_p"] = float(result.pvalues[term])
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--lab-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--chart-chunk-size", type=int, default=2_000_000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    first = build_first_icu(args.data_dir)
    first = add_chronic_burden(args.data_dir, first)
    labdf = scan_labs(args.data_dir, first, args.lab_chunk_size, args.chart_chunk_size)
    df = first.merge(labdf, on="stay_id", how="left")
    df = add_collapsed_categories(df)

    main = df[df["alive_at_24h"] & df["core4_complete_24h"]].copy()
    icu24 = main[main["still_in_icu_at_24h"]].copy()

    cohort_summary = pd.DataFrame(
        [
            {
                "cohort": "all_first_icu",
                "n": len(df),
                "death_30d_after_landmark": int(df["death_30d_after_landmark"].sum()),
                "death_365d_after_landmark": int(df["death_365d_after_landmark"].sum()),
                "core4_complete_24h": int(df["core4_complete_24h"].sum()),
                "alive_at_24h": int(df["alive_at_24h"].sum()),
            },
            {
                "cohort": "alive24_core4_complete",
                "n": len(main),
                "death_30d_after_landmark": int(main["death_30d_after_landmark"].sum()),
                "death_365d_after_landmark": int(main["death_365d_after_landmark"].sum()),
                "core4_complete_24h": int(main["core4_complete_24h"].sum()),
                "alive_at_24h": int(main["alive_at_24h"].sum()),
            },
            {
                "cohort": "alive24_still_icu_core4_complete",
                "n": len(icu24),
                "death_30d_after_landmark": int(icu24["death_30d_after_landmark"].sum()),
                "death_365d_after_landmark": int(icu24["death_365d_after_landmark"].sum()),
                "core4_complete_24h": int(icu24["core4_complete_24h"].sum()),
                "alive_at_24h": int(icu24["alive_at_24h"].sum()),
            },
        ]
    )
    write_csv(cohort_summary, args.out_dir / "cohort_summary.csv")

    tables = {
        "event_rates_by_acute_score_30d.csv": event_table(main, ["acute_stress_score"], "death_30d_after_landmark"),
        "event_rates_by_acute_score_365d.csv": event_table(main, ["acute_stress_score"], "death_365d_after_landmark"),
        "event_rates_by_chronic_count_30d.csv": event_table(main, ["index_hadm_ckm_domain_count"], "death_30d_after_landmark"),
        "event_rates_by_chronic_count_365d.csv": event_table(main, ["index_hadm_ckm_domain_count"], "death_365d_after_landmark"),
        "event_rates_by_prior_chronic_count_30d.csv": event_table(main, ["prior_hadm_ckm_domain_count"], "death_30d_after_landmark"),
        "event_rates_by_prior_chronic_count_365d.csv": event_table(main, ["prior_hadm_ckm_domain_count"], "death_365d_after_landmark"),
        "event_rates_by_acute_chronic_30d.csv": event_table(main, ["index_hadm_ckm_domain_count", "acute_stress_score"], "death_30d_after_landmark"),
        "event_rates_by_acute_chronic_365d.csv": event_table(main, ["index_hadm_ckm_domain_count", "acute_stress_score"], "death_365d_after_landmark"),
    }
    for name, table in tables.items():
        write_csv(table, args.out_dir / name)

    synergy = main.copy()
    synergy["chronic_high_ge2"] = (synergy["index_hadm_ckm_domain_count"] >= 2).astype(int)
    synergy["acute_high_ge2"] = (synergy["acute_stress_score"] >= 2).astype(int)
    write_csv(
        event_table(synergy, ["chronic_high_ge2", "acute_high_ge2"], "death_30d_after_landmark"),
        args.out_dir / "event_rates_high_chronic_high_acute_30d.csv",
    )
    write_csv(
        event_table(synergy, ["chronic_high_ge2", "acute_high_ge2"], "death_365d_after_landmark"),
        args.out_dir / "event_rates_high_chronic_high_acute_365d.csv",
    )

    model_rows = []
    model_specs = [
        ("M0_base", ["base"]),
        ("M1_base_chronic", ["base", "chronic"]),
        ("M2_base_acute", ["base", "acute"]),
        ("M3_base_chronic_acute", ["base", "chronic", "acute"]),
        ("M4_base_chronic_acute_interaction", ["base", "chronic", "acute", "interaction"]),
    ]
    for cohort_name, cohort_df in [
        ("alive24_core4_complete", main),
        ("alive24_still_icu_core4_complete", icu24),
    ]:
        for outcome in ["death_30d_after_landmark", "death_365d_after_landmark"]:
            for model_name, terms in model_specs:
                try:
                    model_rows.append(run_logit(cohort_df, outcome, terms, model_name, cohort_name))
                except Exception as exc:
                    model_rows.append(
                        {
                            "cohort": cohort_name,
                            "outcome": outcome,
                            "model": model_name,
                            "error": str(exc),
                        }
                    )
    models = pd.DataFrame(model_rows)
    write_csv(models, args.out_dir / "logistic_model_summary.csv")

    summary = {
        "n_all_first_icu": int(len(df)),
        "n_alive24_core4_complete": int(len(main)),
        "n_alive24_still_icu_core4_complete": int(len(icu24)),
        "out_dir": str(args.out_dir),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
