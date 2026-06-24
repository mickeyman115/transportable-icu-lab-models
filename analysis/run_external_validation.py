#!/usr/bin/env python3
"""External validation of the MIMIC lab-stress model in eICU.

This script uses only variables that can be defined consistently in both
databases: age, sex, peak first-24h glucose/creatinine/BUN/WBC, and
in-hospital mortality after the 24h ICU landmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
MIMIC_CACHE_DEFAULT = PROJECT / "derived_cache/mimic_first_icu_labstress_cache_v1.csv.gz"
EICU_DIR_DEFAULT = Path("data/eicu-crd-2.0")
OUT_DIR_DEFAULT = PROJECT / "results/eicu_external_validation"
EICU_CACHE_DEFAULT = PROJECT / "derived_cache/eicu_first_icu_labstress_cache_v1.csv.gz"

LAB_VARS = ["glucose_max_24h", "creatinine_max_24h", "bun_max_24h", "wbc_max_24h"]
CONCEPTS = ["glucose", "creatinine", "bun", "wbc"]
LAB_NAME_TO_CONCEPT = {
    "glucose": "glucose",
    "bedside glucose": "glucose",
    "creatinine": "creatinine",
    "bun": "bun",
    "wbc x 1000": "wbc",
}
THRESHOLDS = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30]


def parse_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -35, 35)))


def rcs_basis(x: np.ndarray, knots: list[float]) -> np.ndarray:
    if len(knots) < 3:
        return np.empty((len(x), 0))
    last = knots[-1]
    penultimate = knots[-2]
    out = np.zeros((len(x), len(knots) - 2))
    for j, knot in enumerate(knots[:-2]):
        out[:, j] = (
            np.maximum(x - knot, 0) ** 3
            - np.maximum(x - penultimate, 0) ** 3 * (last - knot) / (last - penultimate)
            + np.maximum(x - last, 0) ** 3 * (penultimate - knot) / (last - penultimate)
        )
    return out


def standardize_apply(x: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    sd = sd.copy()
    sd[sd == 0] = 1.0
    return (x - mean) / sd


def fit_logistic_irls(x: np.ndarray, y: np.ndarray, l2: float = 1.0, max_iter: int = 120) -> np.ndarray:
    x_i = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(x_i.shape[1])
    penalty = np.ones_like(beta)
    penalty[0] = 0.0
    for _ in range(max_iter):
        p = sigmoid(x_i @ beta)
        w = np.clip(p * (1 - p), 1e-6, None)
        grad = x_i.T @ (p - y) + l2 * penalty * beta
        hess = (x_i.T * w) @ x_i + np.diag(l2 * penalty)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
        beta -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return beta


def predict_logistic(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return sigmoid(np.column_stack([np.ones(len(x)), x]) @ beta)


def roc_auc_score_local(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=float)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(pred).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def brier_score_local(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def calibration_metrics(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    p = np.clip(pred, 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    try:
        beta = fit_logistic_irls(logit_p, y, l2=0.0, max_iter=100)
        return float(beta[0]), float(beta[1])
    except Exception:
        return np.nan, np.nan


def net_benefit(y: np.ndarray, pred: np.ndarray, threshold: float) -> float:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=float)
    treated = pred >= threshold
    tp = int(((treated) & (y == 1)).sum())
    fp = int(((treated) & (y == 0)).sum())
    n = len(y)
    return (tp / n) - (fp / n) * threshold / (1 - threshold)


def calibration_deciles(y: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for model in ["binary_core4", "rcs_core4"]:
        pred = np.asarray(predictions[model], dtype=float)
        tmp = pd.DataFrame({"y": np.asarray(y, dtype=int), "pred": pred}).dropna()
        tmp["decile"] = pd.qcut(tmp["pred"], 10, labels=False, duplicates="drop") + 1
        for decile, g in tmp.groupby("decile", observed=True):
            rows.append(
                {
                    "sample": "eicu_external_validation",
                    "model": model,
                    "decile": int(decile),
                    "n": int(len(g)),
                    "events": int(g["y"].sum()),
                    "mean_predicted_risk_percent": float(g["pred"].mean() * 100),
                    "observed_risk_percent": float(g["y"].mean() * 100),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_delta_auc(y: np.ndarray, p_base: np.ndarray, p_new: np.ndarray, reps: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    apparent = roc_auc_score_local(y, p_new) - roc_auc_score_local(y, p_base)
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(roc_auc_score_local(yy, p_new[idx]) - roc_auc_score_local(yy, p_base[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return {
        "delta_auc": float(apparent),
        "delta_auc_ci_low": float(lo),
        "delta_auc_ci_high": float(hi),
        "bootstrap_reps_used": int(len(vals)),
    }


def parse_age(value: object) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if s == "> 89":
        return 90.0
    return float(pd.to_numeric(s, errors="coerce"))


def build_eicu_first_stay(data_dir: Path) -> pd.DataFrame:
    cols = [
        "patientunitstayid",
        "patienthealthsystemstayid",
        "uniquepid",
        "gender",
        "age",
        "ethnicity",
        "hospitalid",
        "hospitaldischargeyear",
        "hospitaldischargeoffset",
        "hospitaldischargestatus",
        "unitvisitnumber",
        "unitstaytype",
        "unittype",
        "unitadmitsource",
        "unitdischargeoffset",
        "unitdischargestatus",
    ]
    patient = pd.read_csv(data_dir / "patient.csv.gz", usecols=cols, low_memory=False)
    patient["age_numeric"] = patient["age"].map(parse_age)
    patient["gender_male"] = patient["gender"].astype(str).str.lower().eq("male").astype(int)
    patient["hospital_death"] = patient["hospitaldischargestatus"].astype(str).str.lower().eq("expired")
    patient["unit_death"] = patient["unitdischargestatus"].astype(str).str.lower().eq("expired")
    patient["early_death_before_24h"] = (
        (patient["hospital_death"] & (pd.to_numeric(patient["hospitaldischargeoffset"], errors="coerce") <= 1440))
        | (patient["unit_death"] & (pd.to_numeric(patient["unitdischargeoffset"], errors="coerce") <= 1440))
    )
    patient["alive_at_24h"] = ~patient["early_death_before_24h"]
    patient["hospital_death_after_landmark"] = (patient["hospital_death"] & patient["alive_at_24h"]).astype(int)
    patient["_unitvisit_sort"] = pd.to_numeric(patient["unitvisitnumber"], errors="coerce").fillna(9999)
    patient["_year_sort"] = pd.to_numeric(patient["hospitaldischargeyear"], errors="coerce").fillna(9999)
    first = (
        patient.sort_values(["uniquepid", "_year_sort", "patienthealthsystemstayid", "_unitvisit_sort", "patientunitstayid"])
        .groupby("uniquepid", as_index=False)
        .first()
    )
    first = first.drop(columns=["_unitvisit_sort", "_year_sort"])
    return first


def scan_eicu_labs(data_dir: Path, first: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    stay_ids = first["patientunitstayid"].astype(int).tolist()
    stay_to_pos = {sid: i for i, sid in enumerate(stay_ids)}
    mins = {c: np.full(len(stay_ids), np.nan) for c in CONCEPTS}
    maxs = {c: np.full(len(stay_ids), np.nan) for c in CONCEPTS}
    target_names = set(LAB_NAME_TO_CONCEPT)

    reader = pd.read_csv(
        data_dir / "lab.csv.gz",
        usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"],
        chunksize=chunk_size,
        low_memory=False,
    )
    for chunk in reader:
        chunk["labname_norm"] = chunk["labname"].astype(str).str.strip().str.lower()
        chunk = chunk[
            chunk["labname_norm"].isin(target_names)
            & chunk["labresultoffset"].between(0, 1440, inclusive="both")
        ].copy()
        if chunk.empty:
            continue
        chunk["concept"] = chunk["labname_norm"].map(LAB_NAME_TO_CONCEPT)
        chunk["labresult"] = pd.to_numeric(chunk["labresult"], errors="coerce")
        chunk = chunk[chunk["patientunitstayid"].isin(stay_to_pos) & chunk["labresult"].notna()]
        if chunk.empty:
            continue
        parts = []
        for concept, g in chunk.groupby("concept", observed=True):
            parts.append(g[valid_value(concept, g["labresult"])])
        if not parts:
            continue
        chunk = pd.concat(parts, ignore_index=True)
        grouped = chunk.groupby(["patientunitstayid", "concept"], as_index=False)["labresult"].agg(["min", "max"])
        for row in grouped.itertuples(index=False):
            pos = stay_to_pos.get(int(row.patientunitstayid))
            if pos is None:
                continue
            concept = row.concept
            mn = float(row.min)
            mx = float(row.max)
            if np.isnan(mins[concept][pos]) or mn < mins[concept][pos]:
                mins[concept][pos] = mn
            if np.isnan(maxs[concept][pos]) or mx > maxs[concept][pos]:
                maxs[concept][pos] = mx

    out = pd.DataFrame({"patientunitstayid": stay_ids})
    for concept in CONCEPTS:
        out[f"{concept}_min_24h"] = mins[concept]
        out[f"{concept}_max_24h"] = maxs[concept]
    out["core4_complete_24h"] = out[[f"{c}_max_24h" for c in CONCEPTS]].notna().all(axis=1)
    out["metabolic_stress_24h"] = ((out["glucose_min_24h"] < 70) | (out["glucose_max_24h"] >= 180)).astype(int)
    out["renal_stress_24h"] = ((out["creatinine_max_24h"] >= 1.5) | (out["bun_max_24h"] >= 30)).astype(int)
    out["inflammatory_stress_24h"] = ((out["wbc_min_24h"] < 4) | (out["wbc_max_24h"] > 12)).astype(int)
    out["acute_stress_score_24h"] = out[["metabolic_stress_24h", "renal_stress_24h", "inflammatory_stress_24h"]].sum(axis=1)
    out.loc[~out["core4_complete_24h"], "acute_stress_score_24h"] = np.nan
    return out


def load_or_build_eicu_cache(data_dir: Path, cache_file: Path, chunk_size: int, rebuild: bool) -> pd.DataFrame:
    if cache_file.exists() and not rebuild:
        return pd.read_csv(cache_file, low_memory=False)
    first = build_eicu_first_stay(data_dir)
    labs = scan_eicu_labs(data_dir, first, chunk_size)
    df = first.merge(labs, on="patientunitstayid", how="left")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False, compression="gzip")
    return df


def add_apache(data_dir: Path, eicu: pd.DataFrame) -> pd.DataFrame:
    path = data_dir / "apachePatientResult.csv.gz"
    if not path.exists():
        return eicu
    apache = pd.read_csv(
        path,
        usecols=[
            "patientunitstayid",
            "apacheversion",
            "apachescore",
            "acutephysiologyscore",
            "predictedhospitalmortality",
            "predictedicumortality",
        ],
        low_memory=False,
    )
    apache["_version_rank"] = apache["apacheversion"].astype(str).map({"IVa": 0, "IV": 1}).fillna(9)
    apache = (
        apache.sort_values(["patientunitstayid", "_version_rank"])
        .groupby("patientunitstayid", as_index=False)
        .first()
        .drop(columns=["_version_rank"])
    )
    return eicu.merge(apache, on="patientunitstayid", how="left")


def load_mimic_common(cache_file: Path) -> pd.DataFrame:
    df = pd.read_csv(cache_file, low_memory=False)
    for col in ["alive_at_24h", "core4_complete_24h"]:
        df[col] = parse_bool(df[col])
    df["age_numeric"] = pd.to_numeric(df["anchor_age"], errors="coerce")
    df["gender_male"] = df["gender"].astype(str).str.upper().eq("M").astype(int)
    df["hospital_death_after_landmark"] = pd.to_numeric(df["hospital_expire_flag"], errors="coerce").fillna(0).astype(int)
    for col in LAB_VARS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["alive_at_24h"] & df["core4_complete_24h"]].copy()
    return df.dropna(subset=["age_numeric", "gender_male", "hospital_death_after_landmark"] + LAB_VARS)


def temporal_split_mimic(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    groups = sorted(df["anchor_year_group"].astype(str).unique())
    val_group = groups[-1]
    dev = df["anchor_year_group"].astype(str).ne(val_group)
    val = df["anchor_year_group"].astype(str).eq(val_group)
    return dev, val, val_group


def make_knots(mimic: pd.DataFrame, dev: pd.Series) -> dict[str, list[float]]:
    return {
        var: sorted(set(float(np.percentile(mimic.loc[dev, var].dropna().to_numpy(), p)) for p in [5, 35, 65, 95]))
        for var in LAB_VARS
    }


def build_peak_binary_indicators(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Peak-aligned binary comparator for the continuous peak-value RCS model."""
    cols = [
        (df["glucose_max_24h"].to_numpy(dtype=float) >= 180).astype(float).reshape(-1, 1),
        (df["creatinine_max_24h"].to_numpy(dtype=float) >= 1.5).astype(float).reshape(-1, 1),
        (df["bun_max_24h"].to_numpy(dtype=float) >= 30).astype(float).reshape(-1, 1),
        (df["wbc_max_24h"].to_numpy(dtype=float) > 12).astype(float).reshape(-1, 1),
    ]
    names = [
        "glucose_high_24h",
        "creatinine_high_24h",
        "bun_high_24h",
        "wbc_high_24h",
    ]
    return np.hstack(cols), names


def build_x(df: pd.DataFrame, model: str, knots: dict[str, list[float]]) -> tuple[np.ndarray, list[str]]:
    cols = [df["age_numeric"].to_numpy(dtype=float).reshape(-1, 1), df["gender_male"].to_numpy(dtype=float).reshape(-1, 1)]
    names = ["age_numeric", "gender_male"]
    if model == "age_sex":
        return np.hstack(cols), names
    if model == "binary_core4":
        binary_x, binary_names = build_peak_binary_indicators(df)
        cols.append(binary_x)
        names.extend(binary_names)
    elif model in {"rcs_core4", "rcs_bun_wbc"}:
        use_labs = LAB_VARS if model == "rcs_core4" else ["bun_max_24h", "wbc_max_24h"]
        for var in use_labs:
            x = df[var].to_numpy(dtype=float)
            cols.append(x.reshape(-1, 1))
            names.append(f"{var}_linear")
            basis = rcs_basis(x, knots[var])
            for j in range(basis.shape[1]):
                cols.append(basis[:, j].reshape(-1, 1))
                names.append(f"{var}_rcs{j + 1}")
    else:
        raise ValueError(f"Unknown model: {model}")
    return np.hstack(cols), names


def fit_mimic_apply(
    mimic: pd.DataFrame,
    eicu: pd.DataFrame,
    dev: pd.Series,
    val: pd.Series,
    model: str,
    knots: dict[str, list[float]],
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    x_all, names = build_x(mimic, model, knots)
    x_eicu, _ = build_x(eicu, model, knots)
    y_all = mimic["hospital_death_after_landmark"].to_numpy(dtype=int)
    y_eicu = eicu["hospital_death_after_landmark"].to_numpy(dtype=int)
    x_dev_raw = x_all[dev.to_numpy()]
    mean = np.nanmean(x_dev_raw, axis=0)
    sd = np.nanstd(x_dev_raw, axis=0)
    beta = fit_logistic_irls(standardize_apply(x_dev_raw, mean, sd), y_all[dev.to_numpy()], l2=1.0)

    samples = {
        "mimic_development": (y_all[dev.to_numpy()], predict_logistic(standardize_apply(x_all[dev.to_numpy()], mean, sd), beta)),
        "mimic_temporal_validation": (y_all[val.to_numpy()], predict_logistic(standardize_apply(x_all[val.to_numpy()], mean, sd), beta)),
        "eicu_external_validation": (y_eicu, predict_logistic(standardize_apply(x_eicu, mean, sd), beta)),
    }
    rows = []
    preds = {}
    for sample, (y, pred) in samples.items():
        cal_i, cal_s = calibration_metrics(y, pred)
        rows.append(
            {
                "model": model,
                "features": "|".join(names),
                "sample": sample,
                "n": int(len(y)),
                "events": int(y.sum()),
                "event_rate_percent": float(y.mean() * 100),
                "auc": roc_auc_score_local(y, pred),
                "brier": brier_score_local(y, pred),
                "calibration_intercept": cal_i,
                "calibration_slope": cal_s,
                "mean_predicted_risk_percent": float(pred.mean() * 100),
            }
        )
        preds[sample] = pred
    return rows, preds


def cohort_summary(mimic: pd.DataFrame, eicu_all: pd.DataFrame, eicu_analytic: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cohort": "mimic_alive24_core4_common",
                "n": int(len(mimic)),
                "events": int(mimic["hospital_death_after_landmark"].sum()),
                "event_rate_percent": float(mimic["hospital_death_after_landmark"].mean() * 100),
                "core4_complete_percent": 100.0,
            },
            {
                "cohort": "eicu_first_stay_all",
                "n": int(len(eicu_all)),
                "events": int(eicu_all["hospital_death_after_landmark"].sum()),
                "event_rate_percent": float(eicu_all["hospital_death_after_landmark"].mean() * 100),
                "core4_complete_percent": float(eicu_all["core4_complete_24h"].mean() * 100),
            },
            {
                "cohort": "eicu_alive24_core4_common",
                "n": int(len(eicu_analytic)),
                "events": int(eicu_analytic["hospital_death_after_landmark"].sum()),
                "event_rate_percent": float(eicu_analytic["hospital_death_after_landmark"].mean() * 100),
                "core4_complete_percent": 100.0,
            },
        ]
    )


def lab_distribution(df: pd.DataFrame, source: str) -> pd.DataFrame:
    rows = []
    for col in LAB_VARS:
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "source": source,
                "variable": col,
                "n": int(s.notna().sum()),
                "median": float(s.median()),
                "p05": float(s.quantile(0.05)),
                "p95": float(s.quantile(0.95)),
                "max": float(s.max()),
            }
        )
    return pd.DataFrame(rows)


def lab_coverage(df: pd.DataFrame, source: str) -> pd.DataFrame:
    rows = []
    denom = int(len(df))
    for col in LAB_VARS:
        n = int(pd.to_numeric(df[col], errors="coerce").notna().sum())
        rows.append({"source": source, "variable": col, "n": denom, "non_missing": n, "coverage_percent": n / denom * 100})
    if "core4_complete_24h" in df:
        core = parse_bool(df["core4_complete_24h"]) if df["core4_complete_24h"].dtype == object else df["core4_complete_24h"].astype(bool)
        rows.append({"source": source, "variable": "core4_complete_24h", "n": denom, "non_missing": int(core.sum()), "coverage_percent": float(core.mean() * 100)})
    return pd.DataFrame(rows)


def apache_metrics(eicu: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = eicu["hospital_death_after_landmark"].to_numpy(dtype=int)
    for col in ["predictedhospitalmortality", "predictedicumortality", "apachescore", "acutephysiologyscore"]:
        if col not in eicu:
            continue
        pred = pd.to_numeric(eicu[col], errors="coerce")
        keep = pred.notna()
        if keep.sum() == 0:
            continue
        p = pred.loc[keep].to_numpy(dtype=float)
        yy = y[keep.to_numpy()]
        if "mortality" in col:
            p = np.clip(p, 1e-6, 1 - 1e-6)
        cal_i, cal_s = calibration_metrics(yy, p if "mortality" in col else (p - np.nanmin(p) + 1e-6) / (np.nanmax(p) - np.nanmin(p) + 2e-6))
        rows.append(
            {
                "benchmark": col,
                "n": int(keep.sum()),
                "events": int(yy.sum()),
                "event_rate_percent": float(yy.mean() * 100),
                "auc": roc_auc_score_local(yy, p),
                "brier": brier_score_local(yy, p) if "mortality" in col else np.nan,
                "calibration_intercept": cal_i if "mortality" in col else np.nan,
                "calibration_slope": cal_s if "mortality" in col else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_decision(out_dir: Path, cohort: pd.DataFrame, metrics: pd.DataFrame, deltas: pd.DataFrame, apache: pd.DataFrame) -> None:
    def cell(model: str, sample: str, col: str) -> float:
        return float(metrics.loc[metrics["model"].eq(model) & metrics["sample"].eq(sample), col].iloc[0])

    eicu_n = int(cohort.loc[cohort["cohort"].eq("eicu_alive24_core4_common"), "n"].iloc[0])
    eicu_events = int(cohort.loc[cohort["cohort"].eq("eicu_alive24_core4_common"), "events"].iloc[0])
    apache_text = "not available"
    if not apache.empty:
        row = apache.sort_values("auc", ascending=False).iloc[0]
        apache_text = f"{row['benchmark']} AUC {row['auc']:.4f}"
    lines = [
        "# eICU External Validation Decision",
        "",
        "Date: 2026-06-15",
        "",
        "## Data Status",
        "",
        "- eICU required tables are present locally.",
        f"- eICU analytic cohort: n = {eicu_n:,}, events = {eicu_events:,}.",
        "",
        "## External Validation AUCs",
        "",
        f"- MIMIC-trained age/sex model in eICU: {cell('age_sex', 'eicu_external_validation', 'auc'):.4f}.",
        f"- MIMIC-trained binary core-four model in eICU: {cell('binary_core4', 'eicu_external_validation', 'auc'):.4f}.",
        f"- MIMIC-trained continuous RCS core-four model in eICU: {cell('rcs_core4', 'eicu_external_validation', 'auc'):.4f}.",
        f"- MIMIC-trained BUN + WBC RCS model in eICU: {cell('rcs_bun_wbc', 'eicu_external_validation', 'auc'):.4f}.",
        f"- Best native eICU APACHE benchmark: {apache_text}.",
        "",
        "## Delta AUCs in eICU",
        "",
    ]
    for row in deltas.itertuples(index=False):
        lines.append(
            f"- {row.comparison}: delta AUC = {row.delta_auc:.4f} "
            f"(95% CI {row.delta_auc_ci_low:.4f} to {row.delta_auc_ci_high:.4f})."
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Use these results to decide whether the manuscript should be upgraded to a two-database external-validation paper. The primary defensible claim is transport of a first-day routine laboratory risk signal for in-hospital mortality; any claim beyond APACHE/SOFA replacement remains prohibited.",
            "",
        ]
    )
    (out_dir / "DECISION_2026-06-15.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eicu-dir", type=Path, default=EICU_DIR_DEFAULT)
    parser.add_argument("--mimic-cache", type=Path, default=MIMIC_CACHE_DEFAULT)
    parser.add_argument("--eicu-cache", type=Path, default=EICU_CACHE_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mimic = load_mimic_common(args.mimic_cache)
    dev, val, val_group = temporal_split_mimic(mimic)
    knots = make_knots(mimic, dev)

    eicu_all = load_or_build_eicu_cache(args.eicu_dir, args.eicu_cache, args.chunk_size, args.rebuild_cache)
    eicu_all = add_apache(args.eicu_dir, eicu_all)
    eicu_all["alive_at_24h"] = parse_bool(eicu_all["alive_at_24h"])
    eicu_all["core4_complete_24h"] = parse_bool(eicu_all["core4_complete_24h"])
    eicu_all["hospital_death_after_landmark"] = pd.to_numeric(eicu_all["hospital_death_after_landmark"], errors="coerce").fillna(0).astype(int)
    eicu_all["gender_male"] = pd.to_numeric(eicu_all["gender_male"], errors="coerce")
    eicu_all["age_numeric"] = pd.to_numeric(eicu_all["age_numeric"], errors="coerce")
    for col in LAB_VARS:
        eicu_all[col] = pd.to_numeric(eicu_all[col], errors="coerce")
    eicu = eicu_all[eicu_all["alive_at_24h"] & eicu_all["core4_complete_24h"]].copy()
    eicu = eicu.dropna(subset=["age_numeric", "gender_male", "hospital_death_after_landmark"] + LAB_VARS)

    model_rows = []
    predictions = {"eicu_external_validation": {"y": eicu["hospital_death_after_landmark"].to_numpy(dtype=int)}}
    for model in ["age_sex", "binary_core4", "rcs_core4", "rcs_bun_wbc"]:
        rows, preds = fit_mimic_apply(mimic, eicu, dev, val, model, knots)
        model_rows.extend(rows)
        predictions["eicu_external_validation"][model] = preds["eicu_external_validation"]
    metrics = pd.DataFrame(model_rows)

    y = predictions["eicu_external_validation"]["y"]
    comparisons = [
        ("binary_core4 vs age_sex", "age_sex", "binary_core4"),
        ("rcs_core4 vs age_sex", "age_sex", "rcs_core4"),
        ("rcs_core4 vs binary_core4", "binary_core4", "rcs_core4"),
        ("rcs_bun_wbc vs age_sex", "age_sex", "rcs_bun_wbc"),
    ]
    deltas = []
    for i, (label, base, new) in enumerate(comparisons):
        d = bootstrap_delta_auc(y, predictions["eicu_external_validation"][base], predictions["eicu_external_validation"][new], args.bootstrap_reps, args.seed + i)
        d.update({"comparison": label, "sample": "eicu_external_validation", "base_model": base, "new_model": new, "n": int(len(y)), "events": int(y.sum())})
        deltas.append(d)
    deltas = pd.DataFrame(deltas)

    dca_rows = []
    for model in ["age_sex", "binary_core4", "rcs_core4", "rcs_bun_wbc"]:
        pred = predictions["eicu_external_validation"][model]
        for threshold in THRESHOLDS:
            dca_rows.append(
                {
                    "sample": "eicu_external_validation",
                    "model": model,
                    "threshold": threshold,
                    "net_benefit_per_100": net_benefit(y, pred, threshold) * 100,
                }
            )
    dca = pd.DataFrame(dca_rows)
    cal_deciles = calibration_deciles(y, predictions["eicu_external_validation"])
    apache = apache_metrics(eicu)
    cohort = cohort_summary(mimic, eicu_all, eicu)
    labs = pd.concat([lab_distribution(mimic, "mimic"), lab_distribution(eicu, "eicu")], ignore_index=True)
    coverage = pd.concat(
        [
            lab_coverage(eicu_all[eicu_all["alive_at_24h"]].copy(), "eicu_alive24_all"),
            lab_coverage(eicu, "eicu_alive24_analytic"),
        ],
        ignore_index=True,
    )
    knot_rows = pd.DataFrame([{"variable": var, "knots": "|".join(f"{x:.6g}" for x in vals)} for var, vals in knots.items()])
    pred_out = eicu[["patientunitstayid", "hospital_death_after_landmark"]].copy()
    for model in ["age_sex", "binary_core4", "rcs_core4", "rcs_bun_wbc"]:
        pred_out[f"pred_{model}"] = predictions["eicu_external_validation"][model]
    for col in ["predictedhospitalmortality", "predictedicumortality", "apachescore", "acutephysiologyscore"]:
        if col in eicu:
            pred_out[col] = eicu[col].values

    cohort.to_csv(args.out_dir / "cohort_summary.csv", index=False)
    metrics.to_csv(args.out_dir / "external_validation_model_metrics.csv", index=False)
    deltas.to_csv(args.out_dir / "external_validation_delta_auc.csv", index=False)
    dca.to_csv(args.out_dir / "external_validation_dca.csv", index=False)
    cal_deciles.to_csv(args.out_dir / "eicu_calibration_deciles.csv", index=False)
    apache.to_csv(args.out_dir / "eicu_apache_benchmarks.csv", index=False)
    labs.to_csv(args.out_dir / "lab_distribution_mimic_vs_eicu.csv", index=False)
    coverage.to_csv(args.out_dir / "eicu_lab_coverage.csv", index=False)
    knot_rows.to_csv(args.out_dir / "mimic_development_rcs_knots.csv", index=False)
    pred_out.to_csv(args.out_dir / "eicu_external_predictions_restricted.csv.gz", index=False, compression="gzip")
    write_decision(args.out_dir, cohort, metrics, deltas, apache)

    print(
        json.dumps(
            {
                "mimic_n": int(len(mimic)),
                "mimic_temporal_validation_group": val_group,
                "eicu_first_stays": int(len(eicu_all)),
                "eicu_analytic_n": int(len(eicu)),
                "eicu_events": int(eicu["hospital_death_after_landmark"].sum()),
                "out_dir": str(args.out_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
