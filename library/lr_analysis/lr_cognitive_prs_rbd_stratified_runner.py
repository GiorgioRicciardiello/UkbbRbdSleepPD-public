"""
Phase 2: LR Profile Analysis with RBD Stratification and Age Visualization.

For each cognitive, TMT, and PRS variable:
1. Compute LR+/LR- across z-score threshold grid
2. Stratify by age groups (5-year bands) — Figure 1
3. Stratify by RBD tertiles — Figure 2
4. Visual inspection aids threshold selection
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.config import RBD_RISK_COLORS, config as project_config
from library.lr_analysis.config import (
    AGE_COL,
    COGNITIVE_VARS,
    GENETIC_VARS,
    TMT_VARS,
    ZSCORE_THRESHOLD_GRID,
)
from library.lr_analysis.data_prep import build_analysis_frame, filter_cohort
from library.lr_analysis.lr_metrics import (
    compute_lr_profile_by_age,
    compute_lr_profile_by_rbd_strata,
)

# ── Output paths ───────────────────────────────────────────────────────────────

RESULTS_SUBDIR = "lr_analysis"
OUTPUT_DIR = project_config["results"]["root"] / RESULTS_SUBDIR
FIGURES_DIR = OUTPUT_DIR / "lr_profile_figures"


def _zscore_series(series: pd.Series, col_name: str) -> pd.Series:
    """Z-score a series (mean=0, SD=1)."""
    valid = series.notna()
    if valid.sum() == 0:
        return series
    mu = float(series[valid].mean())
    sigma = float(series[valid].std())
    if sigma == 0:
        return pd.Series(0, index=series.index)
    return (series - mu) / sigma


def plot_lr_by_age(
    df_lr: pd.DataFrame,
    variable_label: str,
    output_path: Path,
) -> None:
    """Plot LR+ and LR- across thresholds, hued by age group."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    age_groups = sorted(df_lr["age_group"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(age_groups)))

    # LR+
    for age_group, color in zip(age_groups, colors):
        subset = df_lr[df_lr["age_group"] == age_group].sort_values("threshold")
        ax1.plot(
            subset["threshold"],
            subset["lr_pos"],
            marker="o",
            label=age_group,
            color=color,
            alpha=0.7,
        )
        ax1.fill_between(
            subset["threshold"],
            subset["lr_pos_lci"],
            subset["lr_pos_uci"],
            color=color,
            alpha=0.2,
        )

    ax1.axhline(y=1, color="red", linestyle="--", linewidth=1, label="LR=1 (no effect)")
    ax1.set_xlabel("Z-score Threshold", fontsize=11)
    ax1.set_ylabel("LR+", fontsize=11)
    ax1.set_title(f"{variable_label}\nPositive Likelihood Ratio by Age Group", fontsize=12)
    ax1.legend(fontsize=9, loc="best")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # LR-
    for age_group, color in zip(age_groups, colors):
        subset = df_lr[df_lr["age_group"] == age_group].sort_values("threshold")
        ax2.plot(
            subset["threshold"],
            subset["lr_neg"],
            marker="s",
            label=age_group,
            color=color,
            alpha=0.7,
        )
        ax2.fill_between(
            subset["threshold"],
            subset["lr_neg_lci"],
            subset["lr_neg_uci"],
            color=color,
            alpha=0.2,
        )

    ax2.axhline(y=1, color="red", linestyle="--", linewidth=1, label="LR=1")
    ax2.set_xlabel("Z-score Threshold", fontsize=11)
    ax2.set_ylabel("LR-", fontsize=11)
    ax2.set_title(f"{variable_label}\nNegative Likelihood Ratio by Age Group", fontsize=12)
    ax2.legend(fontsize=9, loc="best")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_lr_by_rbd(
    df_lr: pd.DataFrame,
    variable_label: str,
    output_path: Path,
) -> None:
    """Plot LR+ and LR- across thresholds, hued by RBD tertile."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    rbd_strata = ["Low", "Mid", "High"]
    colors_rbd = {"Low": RBD_RISK_COLORS["Low"], "Mid": RBD_RISK_COLORS["Mid"], "High": RBD_RISK_COLORS["High"]}

    # LR+
    for stratum in rbd_strata:
        subset = df_lr[df_lr["rbd_stratum"] == stratum].sort_values("threshold")
        if len(subset) == 0:
            continue
        ax1.plot(
            subset["threshold"],
            subset["lr_pos"],
            marker="o",
            label=f"RBD {stratum}",
            color=colors_rbd[stratum],
            linewidth=2,
            alpha=0.8,
        )
        ax1.fill_between(
            subset["threshold"],
            subset["lr_pos_lci"],
            subset["lr_pos_uci"],
            color=colors_rbd[stratum],
            alpha=0.2,
        )

    ax1.axhline(y=1, color="black", linestyle="--", linewidth=1, label="LR=1 (no effect)")
    ax1.set_xlabel("Z-score Threshold", fontsize=11)
    ax1.set_ylabel("LR+", fontsize=11)
    ax1.set_title(f"{variable_label}\nPositive LR by RBD Risk Group", fontsize=12)
    ax1.legend(fontsize=10, loc="best")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # LR-
    for stratum in rbd_strata:
        subset = df_lr[df_lr["rbd_stratum"] == stratum].sort_values("threshold")
        if len(subset) == 0:
            continue
        ax2.plot(
            subset["threshold"],
            subset["lr_neg"],
            marker="s",
            label=f"RBD {stratum}",
            color=colors_rbd[stratum],
            linewidth=2,
            alpha=0.8,
        )
        ax2.fill_between(
            subset["threshold"],
            subset["lr_neg_lci"],
            subset["lr_neg_uci"],
            color=colors_rbd[stratum],
            alpha=0.2,
        )

    ax2.axhline(y=1, color="black", linestyle="--", linewidth=1, label="LR=1")
    ax2.set_xlabel("Z-score Threshold", fontsize=11)
    ax2.set_ylabel("LR-", fontsize=11)
    ax2.set_title(f"{variable_label}\nNegative LR by RBD Risk Group", fontsize=12)
    ax2.legend(fontsize=10, loc="best")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def analyze_variable(
    frame: object,
    variable_col: str,
    variable_label: str,
    cohort_name: str,
    variable_type: str,  # "cognitive", "tmt", "genetic"
) -> dict:
    """Analyze single variable: z-score, compute LR profiles, generate plots."""
    print(f"\n  {variable_label} ({variable_col})...")

    # Z-score the variable
    z_series = _zscore_series(frame.df[variable_col], variable_col)
    frame_working = frame.df.copy()
    frame_working[f"{variable_col}_zscore"] = z_series

    # LR profile by age
    df_by_age = compute_lr_profile_by_age(
        frame_working,
        frame.is_case,
        f"{variable_col}_zscore",
        age_col=AGE_COL,
        thresholds=ZSCORE_THRESHOLD_GRID,
    )

    # LR profile by RBD
    df_by_rbd = compute_lr_profile_by_rbd_strata(
        frame_working,
        frame.is_case,
        f"{variable_col}_zscore",
        rbd_strata_col="rg_pctl3",
        thresholds=ZSCORE_THRESHOLD_GRID,
    )

    # Save CSV
    safe_col_name = variable_col.replace("_", "-")
    csv_age = OUTPUT_DIR / f"lr_profile_{cohort_name}_{safe_col_name}_by_age.csv"
    csv_rbd = OUTPUT_DIR / f"lr_profile_{cohort_name}_{safe_col_name}_by_rbd.csv"

    df_by_age.to_csv(csv_age, index=False)
    df_by_rbd.to_csv(csv_rbd, index=False)

    # Plots
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_age = FIGURES_DIR / f"lr_profile_{cohort_name}_{safe_col_name}_by_age.png"
    fig_rbd = FIGURES_DIR / f"lr_profile_{cohort_name}_{safe_col_name}_by_rbd.png"

    plot_lr_by_age(df_by_age, variable_label, fig_age)
    plot_lr_by_rbd(df_by_rbd, variable_label, fig_rbd)

    # Summary stats
    return {
        "variable": variable_col,
        "label": variable_label,
        "n_records": len(df_by_age),
        "n_by_rbd": len(df_by_rbd),
        "csv_age": str(csv_age),
        "csv_rbd": str(csv_rbd),
        "fig_age": str(fig_age),
        "fig_rbd": str(fig_rbd),
    }


def run_phase2_analysis(file_name: str = "ehr_diag_pd_rbd_only_all") -> None:
    """Phase 2: LR profile analysis with age and RBD stratification."""
    print("\n" + "=" * 70)
    print("PHASE 2: LR Profile Analysis (Age & RBD Stratification)")
    print("=" * 70)

    # Load base frame
    print(f"\nLoading data from {file_name}...")
    frame_base = build_analysis_frame(file_name)

    # Verify RBD risk group exists; if not, create from z-score tertiles
    if "rg_pctl3" not in frame_base.df.columns:
        print("[INFO] Creating rg_pctl3 from z-score tertiles...")
        rbd_z = frame_base.df["rbd_zscore"]
        p33 = rbd_z.quantile(1/3)
        p67 = rbd_z.quantile(2/3)

        rbd_groups = pd.cut(
            rbd_z,
            bins=[-np.inf, p33, p67, np.inf],
            labels=["Low", "Mid", "High"],
            include_lowest=True,
        )
        frame_base.df["rg_pctl3"] = rbd_groups

    results_summary = {
        "analysis": "phase2_lr_profile_age_rbd_stratified",
        "file_name": file_name,
        "cohorts": {},
    }

    # ── Cohort B: Cognitive ────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("COHORT B: Cognitive Variables (LR Profile)")
    print("-" * 70)
    frame_cog, cog_stats = filter_cohort(frame_base, "cognitive")
    print(f"N={cog_stats['n_total']:,}, cases={cog_stats['n_cases']}, "
          f"{cog_stats['pct_complete']:.1f}% complete")

    cognitive_results = []
    for col, label in COGNITIVE_VARS.items():
        result = analyze_variable(frame_cog, col, label, "cognitive", "cognitive")
        cognitive_results.append(result)

    results_summary["cohorts"]["cognitive"] = {
        "n": cog_stats["n_total"],
        "n_cases": cog_stats["n_cases"],
        "variables": cognitive_results,
    }

    # ── Cohort D: TMT ──────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("COHORT D: Trail Making Test (LR Profile)")
    print("-" * 70)
    frame_tmt, tmt_stats = filter_cohort(frame_base, "tmt")
    print(f"N={tmt_stats['n_total']:,}, cases={tmt_stats['n_cases']}, "
          f"{tmt_stats['pct_complete']:.1f}% complete")

    tmt_results = []
    for col, label in TMT_VARS.items():
        result = analyze_variable(frame_tmt, col, label, "tmt", "tmt")
        tmt_results.append(result)

    results_summary["cohorts"]["tmt"] = {
        "n": tmt_stats["n_total"],
        "n_cases": tmt_stats["n_cases"],
        "variables": tmt_results,
    }

    # ── Cohort C: Genetic ──────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("COHORT C: Polygenic Risk Score (LR Profile)")
    print("-" * 70)
    frame_gen, gen_stats = filter_cohort(frame_base, "genetic")
    print(f"N={gen_stats['n_total']:,}, cases={gen_stats['n_cases']}, "
          f"{gen_stats['pct_complete']:.1f}% complete")

    genetic_results = []
    for col, label in GENETIC_VARS.items():
        result = analyze_variable(frame_gen, col, label, "genetic", "genetic")
        genetic_results.append(result)

    results_summary["cohorts"]["genetic"] = {
        "n": gen_stats["n_total"],
        "n_cases": gen_stats["n_cases"],
        "variables": genetic_results,
    }

    # Save summary
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "phase2_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("Phase 2 complete!")
    print(f"Results: {OUTPUT_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    run_phase2_analysis()
