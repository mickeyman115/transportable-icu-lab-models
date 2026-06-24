#!/usr/bin/env python3
"""Aggregate/model-only OASIS-like and Charlson benchmark kill test.

This script follows the MIMIC-code OASIS/Charlson logic closely enough for a
first-pass stop/go decision, while avoiding patient-level output. The OASIS
ventilation component is reconstructed from available local chart/procedure
records and should be treated as OASIS-like until validated against an official
derived-table build.
"""

from __future__ import annotations

import argparse
import json
import math
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


OUT_DIR_DEFAULT = Path(
    "results/severity_benchmark_kill_test"
)

VITAL_ITEMS = {
    "heart_rate": [220045],
    "mbp": [220052, 220181, 225312],
    "resp_rate": [220210, 224690],
    "temp_f": [223761],
    "temp_c": [223762],
}
GCS_ITEMS = [223900, 223901, 220739]
URINE_ITEMS = [226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558, 227488, 227489]
VENT_MODE_ITEMS = [223849, 229314]
O2_DEVICE_ITEMS = [226732]
ETT_ITEMS = [225307, 225308]

INVASIVE_VENT_MODES = {
    "(S) CMV",
    "APRV",
    "APRV/Biphasic+ApnPress",
    "APRV/Biphasic+ApnVol",
    "APV (cmv)",
    "Ambient",
    "Apnea Ventilation",
    "CMV",
    "CMV/ASSIST",
    "CMV/ASSIST/AutoFlow",
    "CMV/AutoFlow",
    "CPAP/PPS",
    "CPAP/PSV",
    "CPAP/PSV+Apn TCPL",
    "CPAP/PSV+ApnPres",
    "CPAP/PSV+ApnVol",
    "MMV",
    "MMV/AutoFlow",
    "MMV/PSV",
    "MMV/PSV/AutoFlow",
    "P-CMV",
    "PCV+",
    "PCV+/PSV",
    "PCV+Assist",
    "PRES/AC",
    "PRVC/AC",
    "PRVC/SIMV",
    "PSV/SBT",
    "SIMV",
    "SIMV/AutoFlow",
    "SIMV/PRES",
    "SIMV/PSV",
    "SIMV/PSV/AutoFlow",
    "SIMV/VOL",
    "SYNCHRON MASTER",
    "SYNCHRON SLAVE",
    "VOL/AC",
    "APV (simv)",
    "P-SIMV",
    "VS",
    "ASV",
}


def update_minmax(
    grouped: pd.DataFrame,
    stay_to_pos: dict[int, int],
    min_arr: np.ndarray,
    max_arr: np.ndarray,
) -> None:
    for row in grouped.itertuples(index=False):
        pos = stay_to_pos.get(int(row.stay_id))
        if pos is None:
            continue
        mn = float(row.min)
        mx = float(row.max)
        if math.isnan(min_arr[pos]) or mn < min_arr[pos]:
            min_arr[pos] = mn
        if math.isnan(max_arr[pos]) or mx > max_arr[pos]:
            max_arr[pos] = mx


def update_flag(grouped: pd.DataFrame, stay_to_pos: dict[int, int], flag_arr: np.ndarray) -> None:
    for row in grouped.itertuples(index=False):
        pos = stay_to_pos.get(int(row.stay_id))
        if pos is not None:
            flag_arr[pos] = max(flag_arr[pos], int(row.flag))


def scan_oasis_features(data_dir: Path, first: pd.DataFrame, chart_chunk_size: int) -> pd.DataFrame:
    stay_ids = first["stay_id"].astype(int).tolist()
    stay_to_pos = {sid: i for i, sid in enumerate(stay_ids)}
    key = first[["stay_id", "intime"]].copy()
    key["stay_id"] = key["stay_id"].astype("int64")

    arrays = {
        "heart_rate_min": np.full(len(stay_ids), np.nan),
        "heart_rate_max": np.full(len(stay_ids), np.nan),
        "mbp_min": np.full(len(stay_ids), np.nan),
        "mbp_max": np.full(len(stay_ids), np.nan),
        "resp_rate_min": np.full(len(stay_ids), np.nan),
        "resp_rate_max": np.full(len(stay_ids), np.nan),
        "temperature_min": np.full(len(stay_ids), np.nan),
        "temperature_max": np.full(len(stay_ids), np.nan),
    }
    mechvent = np.zeros(len(stay_ids), dtype=int)
    gcs_parts: list[pd.DataFrame] = []

    wanted = set(GCS_ITEMS + VENT_MODE_ITEMS + O2_DEVICE_ITEMS + ETT_ITEMS)
    for ids in VITAL_ITEMS.values():
        wanted.update(ids)

    reader = pd.read_csv(
        data_dir / "icu/chartevents.csv.gz",
        usecols=["subject_id", "stay_id", "charttime", "itemid", "value", "valuenum"],
        chunksize=chart_chunk_size,
        dtype={"subject_id": "int64", "stay_id": "float64", "itemid": "int64"},
    )
    for chunk in reader:
        chunk = chunk[chunk["itemid"].isin(wanted) & chunk["stay_id"].notna()]
        if chunk.empty:
            continue
        chunk["stay_id"] = chunk["stay_id"].astype("int64")
        chunk = chunk.merge(key, on="stay_id", how="inner")
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["hours_from_icu"] = (chunk["charttime"] - chunk["intime"]).dt.total_seconds() / 3600
        chunk = chunk[chunk["hours_from_icu"].between(-6, 24, inclusive="both")]
        if chunk.empty:
            continue

        for concept, itemids in VITAL_ITEMS.items():
            sub = chunk[chunk["itemid"].isin(itemids) & chunk["valuenum"].notna()].copy()
            if sub.empty:
                continue
            if concept == "heart_rate":
                sub = sub[sub["valuenum"].between(0, 300, inclusive="neither")]
                sub["measure"] = sub["valuenum"]
                grouped = sub.groupby("stay_id", as_index=False)["measure"].agg(["min", "max"])
                update_minmax(grouped, stay_to_pos, arrays["heart_rate_min"], arrays["heart_rate_max"])
            elif concept == "mbp":
                sub = sub[sub["valuenum"].between(0, 300, inclusive="neither")]
                sub["measure"] = sub["valuenum"]
                grouped = sub.groupby("stay_id", as_index=False)["measure"].agg(["min", "max"])
                update_minmax(grouped, stay_to_pos, arrays["mbp_min"], arrays["mbp_max"])
            elif concept == "resp_rate":
                sub = sub[sub["valuenum"].between(0, 70, inclusive="neither")]
                sub["measure"] = sub["valuenum"]
                grouped = sub.groupby("stay_id", as_index=False)["measure"].agg(["min", "max"])
                update_minmax(grouped, stay_to_pos, arrays["resp_rate_min"], arrays["resp_rate_max"])
            elif concept == "temp_f":
                sub = sub[sub["valuenum"].between(70, 120, inclusive="neither")]
                sub["measure"] = (sub["valuenum"] - 32) / 1.8
                grouped = sub.groupby("stay_id", as_index=False)["measure"].agg(["min", "max"])
                update_minmax(grouped, stay_to_pos, arrays["temperature_min"], arrays["temperature_max"])
            elif concept == "temp_c":
                sub = sub[sub["valuenum"].between(10, 50, inclusive="neither")]
                sub["measure"] = sub["valuenum"]
                grouped = sub.groupby("stay_id", as_index=False)["measure"].agg(["min", "max"])
                update_minmax(grouped, stay_to_pos, arrays["temperature_min"], arrays["temperature_max"])

        gcs = chunk[chunk["itemid"].isin(GCS_ITEMS)][
            ["subject_id", "stay_id", "charttime", "itemid", "value", "valuenum"]
        ].copy()
        if not gcs.empty:
            gcs_parts.append(gcs)

        vent = chunk[chunk["itemid"].isin(VENT_MODE_ITEMS + O2_DEVICE_ITEMS + ETT_ITEMS)].copy()
        if not vent.empty:
            value = vent["value"].fillna("").astype(str).str.strip()
            invasive = (
                (vent["itemid"].isin(VENT_MODE_ITEMS) & value.isin(INVASIVE_VENT_MODES))
                | (vent["itemid"].isin(O2_DEVICE_ITEMS) & value.str.contains("Endotracheal tube", case=False, regex=False))
                | vent["itemid"].isin(ETT_ITEMS)
            )
            grouped = vent.loc[invasive].assign(flag=1).groupby("stay_id", as_index=False)["flag"].max()
            update_flag(grouped, stay_to_pos, mechvent)

    features = pd.DataFrame({"stay_id": stay_ids})
    for name, arr in arrays.items():
        features[name] = arr
    features["mechvent"] = mechvent
    features = add_procedure_vent(data_dir, first, features)
    features = features.merge(build_first_day_gcs(gcs_parts), on="stay_id", how="left")
    features = features.merge(scan_first_day_urine(data_dir, first), on="stay_id", how="left")
    return features


def add_procedure_vent(data_dir: Path, first: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    path = data_dir / "icu/procedureevents.csv.gz"
    proc = pd.read_csv(
        path,
        usecols=["stay_id", "itemid", "starttime", "endtime"],
        parse_dates=["starttime", "endtime"],
    )
    proc = proc[proc["itemid"].eq(225792)]
    if proc.empty:
        return features
    key = first[["stay_id", "intime"]].copy()
    key["end_24h"] = key["intime"] + pd.Timedelta(hours=24)
    proc = proc.merge(key, on="stay_id", how="inner")
    proc = proc[(proc["starttime"] <= proc["end_24h"]) & (proc["endtime"] >= proc["intime"])]
    proc_flag = pd.DataFrame({"stay_id": proc["stay_id"].unique(), "proc_mechvent": 1})
    out = features.merge(proc_flag, on="stay_id", how="left")
    out["proc_mechvent"] = out["proc_mechvent"].fillna(0).astype(int)
    out["mechvent"] = out[["mechvent", "proc_mechvent"]].max(axis=1)
    return out.drop(columns=["proc_mechvent"])


def build_first_day_gcs(gcs_parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not gcs_parts:
        return pd.DataFrame(columns=["stay_id", "gcs_min"])
    gcs_long = pd.concat(gcs_parts, ignore_index=True)
    gcs_long["value"] = gcs_long["value"].fillna("").astype(str)
    gcs_long["gcs_motor"] = np.where(gcs_long["itemid"].eq(223901), gcs_long["valuenum"], np.nan)
    gcs_long["gcs_verbal"] = np.where(gcs_long["itemid"].eq(223900), gcs_long["valuenum"], np.nan)
    gcs_long.loc[gcs_long["itemid"].eq(223900) & gcs_long["value"].eq("No Response-ETT"), "gcs_verbal"] = 0
    gcs_long["gcs_eyes"] = np.where(gcs_long["itemid"].eq(220739), gcs_long["valuenum"], np.nan)
    gcs_long["endotrachflag"] = (
        gcs_long["itemid"].eq(223900) & gcs_long["value"].eq("No Response-ETT")
    ).astype(int)
    base = (
        gcs_long.groupby(["subject_id", "stay_id", "charttime"], as_index=False)
        .agg(
            gcsmotor=("gcs_motor", "max"),
            gcsverbal=("gcs_verbal", "max"),
            gcseyes=("gcs_eyes", "max"),
            endotrachflag=("endotrachflag", "max"),
        )
        .sort_values(["stay_id", "charttime"])
        .reset_index(drop=True)
    )

    rows = []
    for _, group in base.groupby("stay_id", sort=False):
        prev = None
        for row in group.itertuples(index=False):
            prev_recent = prev is not None and (row.charttime - prev.charttime).total_seconds() <= 6 * 3600
            pv = prev.gcsverbal if prev_recent else np.nan
            pm = prev.gcsmotor if prev_recent else np.nan
            pe = prev.gcseyes if prev_recent else np.nan
            if row.gcsverbal == 0 or (pd.isna(row.gcsverbal) and pv == 0):
                total = 15
            elif pv == 0:
                total = (
                    (row.gcsmotor if not pd.isna(row.gcsmotor) else 6)
                    + (row.gcsverbal if not pd.isna(row.gcsverbal) else 5)
                    + (row.gcseyes if not pd.isna(row.gcseyes) else 4)
                )
            else:
                motor = row.gcsmotor if not pd.isna(row.gcsmotor) else (pm if not pd.isna(pm) else 6)
                verbal = row.gcsverbal if not pd.isna(row.gcsverbal) else (pv if not pd.isna(pv) else 5)
                eyes = row.gcseyes if not pd.isna(row.gcseyes) else (pe if not pd.isna(pe) else 4)
                total = motor + verbal + eyes
            rows.append((int(row.stay_id), float(total)))
            prev = row
    score = pd.DataFrame(rows, columns=["stay_id", "gcs"])
    return score.groupby("stay_id", as_index=False)["gcs"].min().rename(columns={"gcs": "gcs_min"})


def scan_first_day_urine(data_dir: Path, first: pd.DataFrame) -> pd.DataFrame:
    key = first[["stay_id", "intime"]].copy()
    key["end_24h"] = key["intime"] + pd.Timedelta(hours=24)
    urine = pd.read_csv(
        data_dir / "icu/outputevents.csv.gz",
        usecols=["stay_id", "charttime", "itemid", "value"],
        parse_dates=["charttime"],
    )
    urine = urine[urine["itemid"].isin(URINE_ITEMS)]
    urine = urine.merge(key, on="stay_id", how="inner")
    urine = urine[(urine["charttime"] >= urine["intime"]) & (urine["charttime"] <= urine["end_24h"])]
    urine["urineoutput"] = np.where(urine["itemid"].eq(227488) & (urine["value"] > 0), -urine["value"], urine["value"])
    return urine.groupby("stay_id", as_index=False)["urineoutput"].sum()


def score_oasis_like(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    out = df.copy()
    out["preiculos"] = (out["intime"] - out["admittime"]).dt.total_seconds() / 60

    out["preiculos_score"] = np.select(
        [
            out["preiculos"].isna(),
            out["preiculos"] < 10.2,
            out["preiculos"] < 297,
            out["preiculos"] < 1440,
            out["preiculos"] < 18708,
        ],
        [np.nan, 5, 3, 0, 2],
        default=1,
    )
    out["age_score_oasis"] = np.select(
        [
            out["anchor_age"].isna(),
            out["anchor_age"] < 24,
            out["anchor_age"] <= 53,
            out["anchor_age"] <= 77,
            out["anchor_age"] <= 89,
            out["anchor_age"] >= 90,
        ],
        [np.nan, 0, 3, 6, 9, 7],
        default=0,
    )
    out["gcs_score"] = np.select(
        [
            out["gcs_min"].isna(),
            out["gcs_min"] <= 7,
            out["gcs_min"] < 14,
            out["gcs_min"] == 14,
        ],
        [np.nan, 10, 4, 3],
        default=0,
    )
    out["heart_rate_score"] = np.select(
        [
            out["heart_rate_max"].isna(),
            out["heart_rate_max"] > 125,
            out["heart_rate_min"] < 33,
            out["heart_rate_max"].between(107, 125, inclusive="both"),
            out["heart_rate_max"].between(89, 106, inclusive="both"),
        ],
        [np.nan, 6, 4, 3, 1],
        default=0,
    )
    out["mbp_score"] = np.select(
        [
            out["mbp_min"].isna(),
            out["mbp_min"] < 20.65,
            out["mbp_min"] < 51,
            out["mbp_max"] > 143.44,
            (out["mbp_min"] >= 51) & (out["mbp_min"] < 61.33),
        ],
        [np.nan, 4, 3, 3, 2],
        default=0,
    )
    out["resp_rate_score"] = np.select(
        [
            out["resp_rate_min"].isna(),
            out["resp_rate_min"] < 6,
            out["resp_rate_max"] > 44,
            out["resp_rate_max"] > 30,
            out["resp_rate_max"] > 22,
            out["resp_rate_min"] < 13,
        ],
        [np.nan, 10, 9, 6, 1, 1],
        default=0,
    )
    out["temp_score"] = np.select(
        [
            out["temperature_max"].isna(),
            out["temperature_max"] > 39.88,
            out["temperature_min"].between(33.22, 35.93, inclusive="both"),
            out["temperature_max"].between(33.22, 35.93, inclusive="both"),
            out["temperature_min"] < 33.22,
            (out["temperature_min"] > 35.93) & (out["temperature_min"] <= 36.39),
            out["temperature_max"].between(36.89, 39.88, inclusive="both"),
        ],
        [np.nan, 6, 4, 4, 3, 2, 2],
        default=0,
    )
    out["urineoutput_score"] = np.select(
        [
            out["urineoutput"].isna(),
            out["urineoutput"] < 671.09,
            out["urineoutput"] > 6896.80,
            out["urineoutput"].between(671.09, 1426.99, inclusive="both"),
            out["urineoutput"].between(1427.00, 2544.14, inclusive="both"),
        ],
        [np.nan, 10, 8, 5, 1],
        default=0,
    )
    out["mechvent_score"] = np.where(out["mechvent"].eq(1), 9, 0)

    surgical = build_surgical_flag(data_dir, out)
    out = out.merge(surgical, on="stay_id", how="left")
    out["surgical"] = out["surgical"].fillna(0).astype(int)
    out["electivesurgery"] = np.where(
        out["admission_type"].eq("ELECTIVE") & out["surgical"].eq(1),
        1,
        0,
    )
    out["electivesurgery_score"] = np.where(out["electivesurgery"].eq(1), 0, 6)

    score_cols = [
        "age_score_oasis",
        "preiculos_score",
        "gcs_score",
        "heart_rate_score",
        "mbp_score",
        "resp_rate_score",
        "temp_score",
        "urineoutput_score",
        "mechvent_score",
        "electivesurgery_score",
    ]
    out["oasis_like"] = out[score_cols].fillna(0).sum(axis=1)
    return out


def build_surgical_flag(data_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    services = pd.read_csv(data_dir / "hosp/services.csv.gz", parse_dates=["transfertime"])
    key = df[["hadm_id", "stay_id", "intime"]].copy()
    key["end_24h"] = key["intime"] + pd.Timedelta(hours=24)
    services = services.merge(key, on="hadm_id", how="inner")
    services = services[services["transfertime"] < services["end_24h"]]
    curr = services["curr_service"].fillna("").astype(str)
    services["surgical"] = (curr.str.lower().str.contains("surg") | curr.eq("ORTHO")).astype(int)
    return services.groupby("stay_id", as_index=False)["surgical"].max()


def charlson_flags_vectorized(dx: pd.DataFrame) -> pd.DataFrame:
    c = dx["icd_code"].fillna("").astype(str).str.upper().str.replace(".", "", regex=False)
    v9 = dx["icd_version"].eq(9)
    v10 = dx["icd_version"].eq(10)

    def p(n: int) -> pd.Series:
        return c.str[:n]

    flags = pd.DataFrame({"hadm_id": dx["hadm_id"]})
    flags["myocardial_infarct"] = (
        (v9 & p(3).isin(["410", "412"]))
        | (v10 & (p(3).isin(["I21", "I22"]) | p(4).eq("I252")))
    ).astype(int)
    flags["congestive_heart_failure"] = (
        (v9 & (p(3).eq("428") | p(5).isin(["39891", "40201", "40211", "40291", "40401", "40403", "40411", "40413", "40491", "40493"]) | p(4).between("4254", "4259")))
        | (v10 & (p(3).isin(["I43", "I50"]) | p(4).isin(["I099", "I110", "I130", "I132", "I255", "I420", "I425", "I426", "I427", "I428", "I429", "P290"])))
    ).astype(int)
    flags["peripheral_vascular_disease"] = (
        (v9 & (p(3).isin(["440", "441"]) | p(4).isin(["0930", "4373", "4471", "5571", "5579", "V434"]) | p(4).between("4431", "4439")))
        | (v10 & (p(3).isin(["I70", "I71"]) | p(4).isin(["I731", "I738", "I739", "I771", "I790", "I792", "K551", "K558", "K559", "Z958", "Z959"])))
    ).astype(int)
    flags["cerebrovascular_disease"] = (
        (v9 & (p(3).between("430", "438") | p(5).eq("36234")))
        | (v10 & (p(3).isin(["G45", "G46"]) | p(3).between("I60", "I69") | p(4).eq("H340")))
    ).astype(int)
    flags["dementia"] = (
        (v9 & (p(3).eq("290") | p(4).isin(["2941", "3312"])))
        | (v10 & (p(3).isin(["F00", "F01", "F02", "F03", "G30"]) | p(4).isin(["F051", "G311"])))
    ).astype(int)
    flags["chronic_pulmonary_disease"] = (
        (v9 & (p(3).between("490", "505") | p(4).isin(["4168", "4169", "5064", "5081", "5088"])))
        | (v10 & (p(3).between("J40", "J47") | p(3).between("J60", "J67") | p(4).isin(["I278", "I279", "J684", "J701", "J703"])))
    ).astype(int)
    flags["rheumatic_disease"] = (
        (v9 & (p(3).eq("725") | p(4).isin(["4465", "7100", "7101", "7102", "7103", "7104", "7140", "7141", "7142", "7148"])))
        | (v10 & (p(3).isin(["M05", "M06", "M32", "M33", "M34"]) | p(4).isin(["M315", "M351", "M353", "M360"])))
    ).astype(int)
    flags["peptic_ulcer_disease"] = (
        (v9 & p(3).isin(["531", "532", "533", "534"]))
        | (v10 & p(3).isin(["K25", "K26", "K27", "K28"]))
    ).astype(int)
    flags["mild_liver_disease"] = (
        (v9 & (p(3).isin(["570", "571"]) | p(4).isin(["0706", "0709", "5733", "5734", "5738", "5739", "V427"]) | p(5).isin(["07022", "07023", "07032", "07033", "07044", "07054"])))
        | (v10 & (p(3).isin(["B18", "K73", "K74"]) | p(4).isin(["K700", "K701", "K702", "K703", "K709", "K713", "K714", "K715", "K717", "K760", "K762", "K763", "K764", "K768", "K769", "Z944"])))
    ).astype(int)
    flags["diabetes_without_cc"] = (
        (v9 & p(4).isin(["2500", "2501", "2502", "2503", "2508", "2509"]))
        | (v10 & p(4).isin(["E100", "E101", "E106", "E108", "E109", "E110", "E111", "E116", "E118", "E119", "E120", "E121", "E126", "E128", "E129", "E130", "E131", "E136", "E138", "E139", "E140", "E141", "E146", "E148", "E149"]))
    ).astype(int)
    flags["diabetes_with_cc"] = (
        (v9 & p(4).isin(["2504", "2505", "2506", "2507"]))
        | (v10 & p(4).isin(["E102", "E103", "E104", "E105", "E107", "E112", "E113", "E114", "E115", "E117", "E122", "E123", "E124", "E125", "E127", "E132", "E133", "E134", "E135", "E137", "E142", "E143", "E144", "E145", "E147"]))
    ).astype(int)
    flags["paraplegia"] = (
        (v9 & (p(3).isin(["342", "343"]) | p(4).isin(["3341", "3440", "3441", "3442", "3443", "3444", "3445", "3446", "3449"])))
        | (v10 & (p(3).isin(["G81", "G82"]) | p(4).isin(["G041", "G114", "G801", "G802", "G830", "G831", "G832", "G833", "G834", "G839"])))
    ).astype(int)
    flags["renal_disease"] = (
        (v9 & (p(3).isin(["582", "585", "586", "V56"]) | p(4).isin(["5880", "V420", "V451"]) | p(4).between("5830", "5837") | p(5).isin(["40301", "40311", "40391", "40402", "40403", "40412", "40413", "40492", "40493"])))
        | (v10 & (p(3).isin(["N18", "N19"]) | p(4).isin(["I120", "I131", "N032", "N033", "N034", "N035", "N036", "N037", "N052", "N053", "N054", "N055", "N056", "N057", "N250", "Z490", "Z491", "Z492", "Z940", "Z992"])))
    ).astype(int)
    flags["malignant_cancer"] = (
        (v9 & (p(3).between("140", "172") | p(4).between("1740", "1958") | p(3).between("200", "208") | p(4).eq("2386")))
        | (v10 & (p(3).isin(["C43", "C88"]) | p(3).between("C00", "C26") | p(3).between("C30", "C34") | p(3).between("C37", "C41") | p(3).between("C45", "C58") | p(3).between("C60", "C76") | p(3).between("C81", "C85") | p(3).between("C90", "C97")))
    ).astype(int)
    flags["severe_liver_disease"] = (
        (v9 & (p(4).isin(["4560", "4561", "4562"]) | p(4).between("5722", "5728")))
        | (v10 & p(4).isin(["I850", "I859", "I864", "I982", "K704", "K711", "K721", "K729", "K765", "K766", "K767"]))
    ).astype(int)
    flags["metastatic_solid_tumor"] = (
        (v9 & p(3).isin(["196", "197", "198", "199"]))
        | (v10 & p(3).isin(["C77", "C78", "C79", "C80"]))
    ).astype(int)
    flags["aids"] = (
        (v9 & p(3).isin(["042", "043", "044"]))
        | (v10 & p(3).isin(["B20", "B21", "B22", "B24"]))
    ).astype(int)
    return flags


def add_charlson(data_dir: Path, first: pd.DataFrame) -> pd.DataFrame:
    dx = pd.read_csv(
        data_dir / "hosp/diagnoses_icd.csv.gz",
        dtype={"icd_code": str, "icd_version": int, "hadm_id": int},
        usecols=["hadm_id", "icd_code", "icd_version"],
    )
    dx = dx[dx["hadm_id"].isin(first["hadm_id"])]
    flags = charlson_flags_vectorized(dx)
    com = flags.groupby("hadm_id", as_index=False).max()
    out = first[["hadm_id", "anchor_age"]].merge(com, on="hadm_id", how="left")
    flag_cols = [c for c in out.columns if c not in ["hadm_id", "anchor_age"]]
    out[flag_cols] = out[flag_cols].fillna(0).astype(int)
    out["charlson_age_score"] = np.select(
        [
            out["anchor_age"] <= 50,
            out["anchor_age"] <= 60,
            out["anchor_age"] <= 70,
            out["anchor_age"] <= 80,
        ],
        [0, 1, 2, 3],
        default=4,
    )
    out["charlson_comorbidity_no_age"] = (
        out["myocardial_infarct"]
        + out["congestive_heart_failure"]
        + out["peripheral_vascular_disease"]
        + out["cerebrovascular_disease"]
        + out["dementia"]
        + out["chronic_pulmonary_disease"]
        + out["rheumatic_disease"]
        + out["peptic_ulcer_disease"]
        + np.maximum(out["mild_liver_disease"], 3 * out["severe_liver_disease"])
        + np.maximum(2 * out["diabetes_with_cc"], out["diabetes_without_cc"])
        + np.maximum(2 * out["malignant_cancer"], 6 * out["metastatic_solid_tumor"])
        + 2 * out["paraplegia"]
        + 2 * out["renal_disease"]
        + 6 * out["aids"]
    )
    out["charlson_comorbidity_index"] = out["charlson_age_score"] + out["charlson_comorbidity_no_age"]
    return out.drop(columns=["anchor_age"])
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
    if "oasis" in terms:
        x["oasis_like"] = df["oasis_like"].astype(float)
    if "acute" in terms:
        x["acute_stress_score"] = df["acute_stress_score"].astype(float)
    return sm.add_constant(x, has_constant="add")


def calibration_metrics(y: pd.Series, pred: np.ndarray) -> tuple[float, float]:
    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    logit_pred = np.log(pred / (1 - pred))
    x = sm.add_constant(pd.DataFrame({"logit_pred": logit_pred}), has_constant="add")
    try:
        res = sm.GLM(y.astype(int), x, family=sm.families.Binomial()).fit(maxiter=100)
        return float(res.params["const"]), float(res.params["logit_pred"])
    except Exception:
        return np.nan, np.nan


def run_model(df: pd.DataFrame, outcome: str, terms: list[str], model_name: str, cohort_name: str) -> dict[str, object]:
    cols = [
        outcome,
        "anchor_age",
        "gender",
        "admission_group",
        "careunit_group",
        "charlson_comorbidity_no_age",
        "oasis_like",
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
    for term in ["charlson_comorbidity_no_age", "oasis_like", "acute_stress_score"]:
        if term in res.params.index:
            beta = res.params[term]
            se = res.bse[term]
            row[f"{term}_or"] = float(np.exp(beta))
            row[f"{term}_ci_low"] = float(np.exp(beta - 1.96 * se))
            row[f"{term}_ci_high"] = float(np.exp(beta + 1.96 * se))
            row[f"{term}_p"] = float(res.pvalues[term])
    return row


def component_missingness(df: pd.DataFrame) -> pd.DataFrame:
    components = [
        "heart_rate_max",
        "mbp_min",
        "resp_rate_max",
        "temperature_max",
        "gcs_min",
        "urineoutput",
        "mechvent",
        "electivesurgery",
        "charlson_comorbidity_no_age",
        "oasis_like",
        "acute_stress_score",
    ]
    rows = []
    for col in components:
        rows.append(
            {
                "component": col,
                "n": int(len(df)),
                "non_missing": int(df[col].notna().sum()),
                "coverage_percent": float(df[col].notna().mean() * 100),
            }
        )
    return pd.DataFrame(rows)


def event_table(df: pd.DataFrame, group_col: str, outcome: str) -> pd.DataFrame:
    tmp = df.copy()
    if group_col == "oasis_quartile":
        tmp[group_col] = pd.qcut(tmp["oasis_like"], q=4, duplicates="drop")
    out = (
        tmp.groupby(group_col, observed=True)
        .agg(n=("subject_id", "size"), events=(outcome, "sum"), age_median=("anchor_age", "median"))
        .reset_index()
    )
    out[group_col] = out[group_col].astype(str)
    out["event_rate_percent"] = out["events"] / out["n"] * 100
    return out


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "acute_stress_score",
        "oasis_like",
        "charlson_comorbidity_no_age",
        "gcs_score",
        "heart_rate_score",
        "mbp_score",
        "resp_rate_score",
        "temp_score",
        "urineoutput_score",
        "mechvent_score",
    ]
    rows = []
    for var in variables[1:]:
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
                "delta_auc_oasis_after_base_charlson": data.get("M3_base_charlson_oasis", np.nan)
                - data.get("M1_base_charlson", np.nan),
                "delta_auc_acute_after_oasis_charlson": data.get("M4_base_charlson_oasis_acute", np.nan)
                - data.get("M3_base_charlson_oasis", np.nan),
            }
        )
    return pd.DataFrame(rows)


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
    oasis_features = scan_oasis_features(args.data_dir, first, args.chart_chunk_size)
    charlson = add_charlson(args.data_dir, first)

    df = first.merge(labdf, on="stay_id", how="left")
    df = df.merge(oasis_features, on="stay_id", how="left")
    df = df.merge(charlson, on="hadm_id", how="left")
    df = add_collapsed_categories(df)
    df = score_oasis_like(df, args.data_dir)

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
                    "oasis_like_median": float(main_cohort["oasis_like"].median()),
                    "charlson_no_age_median": float(main_cohort["charlson_comorbidity_no_age"].median()),
                },
                {
                    "cohort": "alive24_still_icu_core4_complete",
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
    write_csv(component_missingness(main_cohort), args.out_dir / "component_missingness.csv")
    write_csv(correlation_table(main_cohort), args.out_dir / "profile_oasis_correlations.csv")
    write_csv(event_table(main_cohort, "oasis_quartile", "death_30d_after_landmark"), args.out_dir / "event_rates_by_oasis_quartile_30d.csv")
    write_csv(event_table(main_cohort, "oasis_quartile", "death_365d_after_landmark"), args.out_dir / "event_rates_by_oasis_quartile_365d.csv")

    model_specs = [
        ("M0_base", ["base"]),
        ("M1_base_charlson", ["base", "charlson"]),
        ("M2_base_charlson_acute", ["base", "charlson", "acute"]),
        ("M3_base_charlson_oasis", ["base", "charlson", "oasis"]),
        ("M4_base_charlson_oasis_acute", ["base", "charlson", "oasis", "acute"]),
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

    summary = {
        "n_alive24_core4_complete": int(len(main_cohort)),
        "n_alive24_still_icu_core4_complete": int(len(icu24)),
        "out_dir": str(args.out_dir),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
