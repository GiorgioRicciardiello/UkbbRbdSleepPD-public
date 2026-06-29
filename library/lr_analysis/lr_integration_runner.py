"""
Phase 2 + Phase 3 Integration: Combine LR profiles with interaction test results.

Merges:
- Phase 2: LR+ / LR- profiles stratified by RBD tertile at multiple thresholds
- Phase 3: RBD × predictor interaction LRT results

Outputs:
1. summary_phase2_phase3.csv — per-variable summary with LR+ at key thresholds + LRT results
2. crossval_phase2_phase3.csv — per-variable-per-threshold cross-validation (LR+ spread vs LRT)
3. 9 annotated figures — Phase 2 by-RBD plots with LRT p-value overlaid
"""
from __future__ import annotations

import sys
from pathlib import Path

# Set up sys.path BEFORE importing project modules
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.config import RBD_RISK_COLORS, config as project_config
from library.lr_analysis.config import (
    COGNITIVE_VARS,
    GENETIC_VARS,
    TMT_VARS,
)

# ── Constants ──────────────────────────────────────────────────────────────────

RESULTS_DIR = project_config["results"]["root"] / "lr_analysis"
FIGURES_DIR = RESULTS_DIR / "lr_profile_figures"
ANNOTATED_DIR = FIGURES_DIR / "annotated"
KEY_THRESHOLDS = [0.0, 0.5, 1.0]


# ── Data loaders ───────────────────────────────────────────────────────────────


def load_phase2_by_rbd() -> pd.DataFrame:
    """Load all Phase 2 by-RBD CSV files, inject variable/label/cohort columns."""
    rows = []

    cohort_defs = {
        "cognitive": COGNITIVE_VARS,
        "tmt": TMT_VARS,
        "genetic": GENETIC_VARS,
    }

    for cohort, var_dict in cohort_defs.items():
        for col, label in var_dict.items():
            safe_col_name = col.replace("_", "-")
            csv_path = RESULTS_DIR / f"lr_profile_{cohort}_{safe_col_name}_by_rbd.csv"

            if not csv_path.exists():
                print(f"[WARN] Missing: {csv_path.name}")
                continue

            df = pd.read_csv(csv_path)
            df["variable"] = col
            df["label"] = label
            df["cohort"] = cohort
            rows.append(df)

    return pd.concat(rows, ignore_index=True)


def load_phase3_results() -> pd.DataFrame:
    """Load all Phase 3 interaction CSV files."""
    rows = []

    for cohort in ["cognitive", "tmt", "genetic"]:
        csv_path = RESULTS_DIR / f"rbd_interaction_{cohort}.csv"
        if csv_path.exists():
            rows.append(pd.read_csv(csv_path))

    return pd.concat(rows, ignore_index=True)


# ── Integration functions ──────────────────────────────────────────────────────


def make_summary_table(
    phase2_df: pd.DataFrame,
    phase3_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Merge Phase 2 and Phase 3 results into a per-variable summary table."""
    summary_rows = []

    for _, phase3_row in phase3_df.iterrows():
        var = phase3_row["variable"]
        label = phase3_row["label"]
        cohort = phase3_row["cohort"]

        # Extract Phase 2 data for this variable
        phase2_var = phase2_df[
            (phase2_df["variable"] == var) & (phase2_df["threshold"].isin(KEY_THRESHOLDS))
        ]

        if len(phase2_var) == 0:
            print(f"[WARN] No Phase 2 data for {var}")
            continue

        # Pivot strata to columns at each threshold
        pivot_cols = {}
        for threshold in KEY_THRESHOLDS:
            thresh_data = phase2_var[phase2_var["threshold"] == threshold]
            for stratum in ["Low", "Mid", "High"]:
                stratum_row = thresh_data[thresh_data["rbd_stratum"] == stratum]
                if len(stratum_row) > 0:
                    lr_pos = float(stratum_row["lr_pos"].iloc[0])
                    z_str = str(threshold).replace(".", "-")
                    pivot_cols[f"lr_pos_{stratum}_z{z_str}"] = lr_pos

        # Build summary row
        summary_row = {
            "variable": var,
            "label": label,
            "cohort": cohort,
            "n_total": phase3_row["n_total"],
            "n_cases": phase3_row["n_cases"],
            "lrt_stat": round(phase3_row["lrt_stat"], 4),
            "lrt_df": phase3_row["lrt_df"],
            "lrt_p": round(phase3_row["lrt_p"], 6),
            "lrt_significant": phase3_row["lrt_p"] < 0.05,
            "interaction_or_Mid": round(phase3_row["interaction_or_Mid"], 4),
            "interaction_or_High": round(phase3_row["interaction_or_High"], 4),
            "main_g_or": round(phase3_row["main_g_or"], 4),
            "main_g_p": round(phase3_row["main_g_p"], 6),
        }
        summary_row.update(pivot_cols)
        summary_rows.append(summary_row)

    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(output_path, index=False)
    print(f"[OK] Saved: {output_path.name}")
    return summary_table


def make_cross_validation_report(
    phase2_df: pd.DataFrame,
    phase3_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Cross-validate Phase 2 LR+ spread with Phase 3 LRT significance."""
    cv_rows = []

    for _, phase3_row in phase3_df.iterrows():
        var = phase3_row["variable"]
        label = phase3_row["label"]
        cohort = phase3_row["cohort"]
        lrt_stat = phase3_row["lrt_stat"]
        lrt_p = phase3_row["lrt_p"]
        lrt_significant = lrt_p < 0.05

        # Extract Phase 2 for this variable
        phase2_var = phase2_df[phase2_df["variable"] == var]

        for threshold in KEY_THRESHOLDS:
            thresh_data = phase2_var[phase2_var["threshold"] == threshold]

            # Get LR+ for each stratum
            lr_low = None
            lr_mid = None
            lr_high = None

            for _, row in thresh_data.iterrows():
                if row["rbd_stratum"] == "Low":
                    lr_low = row["lr_pos"]
                elif row["rbd_stratum"] == "Mid":
                    lr_mid = row["lr_pos"]
                elif row["rbd_stratum"] == "High":
                    lr_high = row["lr_pos"]

            if lr_low is None or lr_mid is None or lr_high is None:
                continue

            # Compute LR+ spread (as ratio, max / min)
            lrs = [lr_low, lr_mid, lr_high]
            lr_max = max(lrs)
            lr_min = min(lrs)
            lr_spread_ratio = lr_max / lr_min if lr_min > 0 else float("nan")

            # Flag mismatch: large visual difference but non-significant LRT
            mismatch = False
            if (lr_spread_ratio > 2.0) and (not lrt_significant):
                mismatch = True
            elif (lr_spread_ratio < 1.2) and lrt_significant:
                mismatch = True

            cv_rows.append({
                "variable": var,
                "label": label,
                "cohort": cohort,
                "threshold": threshold,
                "lr_pos_Low": round(lr_low, 4),
                "lr_pos_Mid": round(lr_mid, 4),
                "lr_pos_High": round(lr_high, 4),
                "lr_pos_spread_ratio": round(lr_spread_ratio, 4),
                "lrt_stat": round(lrt_stat, 4),
                "lrt_p": round(lrt_p, 6),
                "lrt_significant": lrt_significant,
                "mismatch_flag": mismatch,
            })

    cv_table = pd.DataFrame(cv_rows)
    cv_table.to_csv(output_path, index=False)
    print(f"[OK] Saved: {output_path.name}")
    return cv_table


def plot_annotated_by_rbd(
    phase2_df: pd.DataFrame,
    phase3_row: pd.Series,
    var: str,
    label: str,
    cohort: str,
) -> None:
    """Plot annotated by-RBD figure with LRT p-value overlay."""
    phase2_var = phase2_df[(phase2_df["variable"] == var)]

    if len(phase2_var) == 0:
        print(f"[SKIP] No Phase 2 data for {var}")
        return

    lrt_stat = phase3_row["lrt_stat"]
    lrt_p = phase3_row["lrt_p"]
    lrt_significant = lrt_p < 0.05

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    rbd_strata = ["Low", "Mid", "High"]
    colors_rbd = {"Low": RBD_RISK_COLORS["Low"], "Mid": RBD_RISK_COLORS["Mid"], "High": RBD_RISK_COLORS["High"]}

    # LR+ panel
    for stratum in rbd_strata:
        subset = phase2_var[phase2_var["rbd_stratum"] == stratum].sort_values("threshold")
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
    ax1.set_title(f"{label}\nPositive LR by RBD Risk Group", fontsize=12)
    ax1.legend(fontsize=10, loc="best")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Add LRT annotation
    p_str = f"{lrt_p:.4f}" if not np.isnan(lrt_p) else "NaN"
    stat_str = f"LRT χ²(df=2) = {lrt_stat:.2f}\np = {p_str}"
    box_color = "lightgreen" if lrt_significant else "lightgray"
    ax1.text(
        0.98, 0.05,
        stat_str,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.7),
    )

    # LR- panel
    for stratum in rbd_strata:
        subset = phase2_var[phase2_var["rbd_stratum"] == stratum].sort_values("threshold")
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
    ax2.set_title(f"{label}\nNegative LR by RBD Risk Group", fontsize=12)
    ax2.legend(fontsize=10, loc="best")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    safe_var_name = var.replace("_", "-")
    fig_path = ANNOTATED_DIR / f"lr_{cohort}_{safe_var_name}_by_rbd_annotated.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path.name}")


def run_integration() -> None:
    """Main integration runner."""
    print("\n" + "=" * 70)
    print("INTEGRATION: Phase 2 (LR Profiles) + Phase 3 (Interaction LRT)")
    print("=" * 70)

    # Load data
    print("\nLoading Phase 2 and Phase 3 results...")
    phase2_df = load_phase2_by_rbd()
    phase3_df = load_phase3_results()
    print(f"  Phase 2: {len(phase2_df)} rows ({len(phase2_df['variable'].unique())} variables)")
    print(f"  Phase 3: {len(phase3_df)} rows ({len(phase3_df['variable'].unique())} variables)")

    # Make summary table
    print("\nGenerating summary table...")
    summary_path = RESULTS_DIR / "summary_phase2_phase3.csv"
    summary_table = make_summary_table(phase2_df, phase3_df, summary_path)

    # Make cross-validation report
    print("\nGenerating cross-validation report...")
    cv_path = RESULTS_DIR / "crossval_phase2_phase3.csv"
    cv_table = make_cross_validation_report(phase2_df, phase3_df, cv_path)

    # Generate annotated figures
    print("\nGenerating annotated figures...")
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    for _, phase3_row in phase3_df.iterrows():
        var = phase3_row["variable"]
        label = phase3_row["label"]
        cohort = phase3_row["cohort"]
        print(f"  {label} ({cohort})...")
        plot_annotated_by_rbd(phase2_df, phase3_row, var, label, cohort)

    # Summary statistics
    print("\n" + "=" * 70)
    n_sig = (summary_table["lrt_significant"] == True).sum()
    n_mismatch = (cv_table["mismatch_flag"] == True).sum()
    print(f"Significant interactions (LRT p < 0.05): {n_sig} / {len(summary_table)}")
    print(f"Cross-validation mismatches: {n_mismatch} / {len(cv_table)}")
    print("=" * 70)


if __name__ == "__main__":
    run_integration()
