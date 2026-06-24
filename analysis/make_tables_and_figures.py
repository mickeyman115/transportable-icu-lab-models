#!/usr/bin/env python3
"""Recreate public tables and figures from aggregate output files.

This script intentionally reads only aggregate CSV files that can be shared
publicly. Patient-level MIMIC/eICU rows, derived caches, and individual
prediction files are not required for this public figure-generation step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "age_sex": "Age + sex",
    "binary_core4": "Binary high-threshold",
    "rcs_core4": "RCS core-four",
    "rcs_bun_wbc": "BUN + WBC RCS",
}
MODEL_COLORS = {
    "age_sex": "#737373",
    "binary_core4": "#2E74B5",
    "rcs_core4": "#C23B22",
    "rcs_bun_wbc": "#2E8B57",
}
SAMPLES = ["mimic_development", "mimic_temporal_validation", "eicu_external_validation"]
SAMPLE_LABELS = ["MIMIC dev", "MIMIC temporal", "eICU external"]


def load_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "cohort": pd.read_csv(results_dir / "cohort_summary.csv"),
        "metrics": pd.read_csv(results_dir / "external_validation_model_metrics.csv"),
        "dca": pd.read_csv(results_dir / "external_validation_dca.csv"),
        "calibration": pd.read_csv(results_dir / "eicu_calibration_deciles.csv"),
    }


def make_figure1(cohort: pd.DataFrame, figure_dir: Path) -> Path:
    mimic_n = int(cohort.loc[cohort["cohort"].eq("mimic_alive24_core4_common"), "n"].iloc[0])
    eicu_n = int(cohort.loc[cohort["cohort"].eq("eicu_alive24_core4_common"), "n"].iloc[0])

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.axis("off")
    boxes = [
        (0.03, 0.58, 0.27, 0.28, f"MIMIC-IV v3.1\nfirst ICU admissions\nalive at 24h + core-four labs\nn = {mimic_n:,}"),
        (0.37, 0.58, 0.24, 0.28, "Temporal split\n2008-2019 development\n2020-2022 validation"),
        (0.70, 0.58, 0.27, 0.28, "MIMIC-trained models\nage/sex + binary or RCS labs\nOutcome: post-landmark\nin-hospital mortality"),
        (0.20, 0.13, 0.27, 0.28, f"eICU v2.0\nfirst unit stay per patient\nalive at 24h + core-four labs\nn = {eicu_n:,}"),
        (0.57, 0.13, 0.27, 0.28, "External validation\nunchanged MIMIC model\nAPACHE used as boundary\nbenchmark"),
    ]
    for x0, y0, w, h, text in boxes:
        rect = plt.Rectangle((x0, y0), w, h, facecolor="#F4F6F9", edgecolor="#2E74B5", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=9)
    for start, end in [((0.30, 0.72), (0.37, 0.72)), ((0.61, 0.72), (0.70, 0.72)), ((0.47, 0.27), (0.57, 0.27))]:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2))
    fig.tight_layout()
    out = figure_dir / "Figure1_study_design_flow.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out


def make_figure2(metrics: pd.DataFrame, figure_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(SAMPLES))
    width = 0.18
    for i, model in enumerate(["age_sex", "binary_core4", "rcs_core4", "rcs_bun_wbc"]):
        vals = [
            metrics.loc[metrics["model"].eq(model) & metrics["sample"].eq(sample), "auc"].iloc[0]
            for sample in SAMPLES
        ]
        ax.bar(x + (i - 1.5) * width, vals, width, label=MODEL_LABELS[model], color=MODEL_COLORS[model])
    ax.set_ylim(0.55, 0.80)
    ax.set_ylabel("AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(SAMPLE_LABELS)
    ax.legend(loc="upper left", ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = figure_dir / "Figure2_auc_validation_layers.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out


def make_figure3(calibration: pd.DataFrame, figure_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for model in ["binary_core4", "rcs_core4"]:
        sub = calibration[calibration["model"].eq(model)].sort_values("decile")
        ax.plot(
            sub["mean_predicted_risk_percent"],
            sub["observed_risk_percent"],
            marker="o",
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
    lim = [0, 35]
    ax.plot(lim, lim, linestyle="--", color="#555555", linewidth=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Mean predicted risk (%)")
    ax.set_ylabel("Observed risk (%)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = figure_dir / "Figure3_eicu_calibration_deciles.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out


def make_figure4(dca: pd.DataFrame, figure_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for model in ["age_sex", "binary_core4", "rcs_core4", "rcs_bun_wbc"]:
        sub = dca[dca["model"].eq(model)].sort_values("threshold")
        ax.plot(
            sub["threshold"] * 100,
            sub["net_benefit_per_100"],
            marker="o",
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xlabel("Threshold probability (%)")
    ax.set_ylabel("Net benefit per 100 patients")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = figure_dir / "Figure4_eicu_decision_curve.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/eicu_external_validation"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    data = load_results(args.results_dir)
    outputs = [
        make_figure1(data["cohort"], args.figure_dir),
        make_figure2(data["metrics"], args.figure_dir),
        make_figure3(data["calibration"], args.figure_dir),
        make_figure4(data["dca"], args.figure_dir),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
