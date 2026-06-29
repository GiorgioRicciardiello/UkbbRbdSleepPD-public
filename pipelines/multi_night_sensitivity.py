"""
Actigraphy night-count sensitivity analysis.
=======================================

Characterizes the distribution of actigraphy recording nights across the cohort,
and evaluates how different minimum-nights thresholds affect:
  - Cohort size & composition
  - Risk stratification (Low / Intermediate / High RBD)
  - Incident PD case counts

Dataset
-------
Reads from canonical production paths (run_merge_ukbb_rbd.py).
Night-level data from ehr_diag_pd_rbd_only_all.parquet; night count computed
at runtime from groupby("eid").size() on the raw frame BEFORE make_subject_level().

Outputs
-------
  results/table_one/night_count_summary_by_group.xlsx
    — mean / SD / median / IQR nights by overall cohort + RBD risk strata
  results/table_one/night_threshold_sensitivity.xlsx
    — cohort size at each min-nights cutoff (≥3, ≥7, ≥10, all)
  results/table_one/nights_distribution.pdf / .png
    — two-panel figure: strip plot (nights by RBD group) + bar chart (threshold sensitivity)
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config.config import config
from library.column_registry import col_incident, col_surv_time, col_risk_group_agnostic
from library.risk.risk_helpers import get_clean_risk_data, make_subject_level

# ============================================================================
# CONFIGURATION
# ============================================================================

STRATIFICATION_OUTCOME: str = "outcome_1a_pd_only"
NIGHT_THRESHOLDS: List[int] = [3, 7, 10]
THRESHOLD_LABELS: Dict[int, str] = {3: "≥3 nights", 7: "≥7 nights", 10: "≥10 nights"}

from config.config import RBD_RISK_COLORS as _RBD_COLORS  # noqa: E402
RISK_PALETTE = {
    "Low (0,90%)":            _RBD_COLORS["Low"],
    "Intermediate (90,99%)":  _RBD_COLORS["Intermediate"],
    "High (99,100%)":         _RBD_COLORS["High"],
}

# ============================================================================
# HELPERS
# ============================================================================


def _cont_summary(series: pd.Series) -> Dict[str, float]:
    """Compute n, mean, SD, median, Q1, Q3 for a continuous variable."""
    x = pd.to_numeric(series, errors="coerce").dropna()
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "median": np.nan, "q1": np.nan, "q3": np.nan}
    return {
        "n": n,
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "median": float(x.median()),
        "q1": float(x.quantile(0.25)),
        "q3": float(x.quantile(0.75)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


# ============================================================================
# NIGHT COUNT SUMMARY TABLE
# ============================================================================


def build_night_summary_table(
    df: pd.DataFrame,
    group_col: str,
    groups: List[str],
) -> pd.DataFrame:
    """Build night count summary by RBD risk group.

    Parameters
    ----------
    df : subject-level DataFrame with n_nights column
    group_col : risk group column name
    groups : ordered group labels

    Returns
    -------
    pd.DataFrame with columns: Group | N | Mean ± SD | Median [IQR] | Min | Max
    """
    rows: List[Dict] = []

    # Overall
    s_all = _cont_summary(df["n_nights"])
    rows.append({
        "Group": f"Overall (N={s_all['n']:,})",
        "N": s_all["n"],
        "Mean ± SD": f"{s_all['mean']:.2f} ± {s_all['sd']:.2f}",
        "Median [IQR]": f"{s_all['median']:.1f} [{s_all['q1']:.1f}–{s_all['q3']:.1f}]",
        "Min": int(s_all["min"]),
        "Max": int(s_all["max"]),
    })

    # By group
    for g in groups:
        mask = df[group_col] == g
        s_grp = _cont_summary(df.loc[mask, "n_nights"])
        rows.append({
            "Group": g,
            "N": s_grp["n"],
            "Mean ± SD": f"{s_grp['mean']:.2f} ± {s_grp['sd']:.2f}",
            "Median [IQR]": f"{s_grp['median']:.1f} [{s_grp['q1']:.1f}–{s_grp['q3']:.1f}]",
            "Min": int(s_grp["min"]),
            "Max": int(s_grp["max"]),
        })

    return pd.DataFrame(rows)


# ============================================================================
# THRESHOLD SENSITIVITY TABLE
# ============================================================================


def build_threshold_sensitivity_table(
    df_full: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """Build cohort size and composition at each night threshold.

    Parameters
    ----------
    df_full : subject-level DataFrame (unfiltered)
    group_col : risk group column name

    Returns
    -------
    pd.DataFrame with rows for each threshold + overall unfiltered
    """
    rows: List[Dict] = []
    n_total = len(df_full)
    incident_col = col_incident(STRATIFICATION_OUTCOME)

    # Unfiltered (all nights)
    n_pd_total = (
        pd.to_numeric(df_full[incident_col], errors="coerce").fillna(0).sum()
        if incident_col in df_full.columns
        else 0
    )
    rows.append({
        "Threshold": "All nights (no filter)",
        "N subjects": n_total,
        "N subjects retained": n_total,
        "N subjects dropped": 0,
        "% retained": 100.0,
        "Incident PD retained": int(n_pd_total),
        "% PD retained": 100.0,
        "Mean ± SD nights": f"{df_full['n_nights'].mean():.2f} ± {df_full['n_nights'].std(ddof=1):.2f}",
    })

    # Filtered by threshold
    for threshold in NIGHT_THRESHOLDS:
        df_filt = df_full[df_full["n_nights"] >= threshold].copy()
        n_retained = len(df_filt)
        n_dropped = n_total - n_retained
        pct_retained = 100.0 * n_retained / n_total if n_total > 0 else 0.0

        n_pd_retained = (
            pd.to_numeric(df_filt[incident_col], errors="coerce").fillna(0).sum()
            if incident_col in df_filt.columns
            else 0
        )
        pct_pd_retained = (
            100.0 * n_pd_retained / n_pd_total if n_pd_total > 0 else 0.0
        )

        mean_nights = df_filt["n_nights"].mean()
        sd_nights = df_filt["n_nights"].std(ddof=1)

        rows.append({
            "Threshold": THRESHOLD_LABELS[threshold],
            "N subjects": n_total,
            "N subjects retained": n_retained,
            "N subjects dropped": n_dropped,
            "% retained": pct_retained,
            "Incident PD retained": int(n_pd_retained),
            "% PD retained": pct_pd_retained,
            "Mean ± SD nights": f"{mean_nights:.2f} ± {sd_nights:.2f}",
        })

    return pd.DataFrame(rows)


# ============================================================================
# PLOTTING
# ============================================================================


def plot_nights_distribution(
    df: pd.DataFrame,
    group_col: str,
    groups: List[str],
    output_dir: Path,
) -> None:
    """Create two-panel figure: nights by RBD group + threshold sensitivity.

    Parameters
    ----------
    df : subject-level DataFrame with n_nights column
    group_col : risk group column
    groups : ordered group labels
    output_dir : directory for PDF/PNG output
    """
    # Set Nature-style parameters
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "lines.linewidth": 0.8,
        "patch.linewidth": 0.5,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))

    # ── Panel A: Strip plot of nights by RBD group ──────────────────────────
    ax = axes[0]

    # Prepare data for plotting
    plot_data = []
    for g in groups:
        mask = df[group_col] == g
        nights = df.loc[mask, "n_nights"].values
        plot_data.extend([(g, n) for n in nights])
    df_plot = pd.DataFrame(plot_data, columns=[group_col, "n_nights"])

    # Strip plot with jitter
    x_positions = {g: i for i, g in enumerate(groups)}
    for i, g in enumerate(groups):
        mask = df_plot[group_col] == g
        y = df_plot.loc[mask, "n_nights"].values
        x = np.random.normal(i, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.15, s=10, color=RISK_PALETTE[g], edgecolors="none")

    # Overlay mean ± SD as large markers with error bars
    for i, g in enumerate(groups):
        mask = df[group_col] == g
        nights = df.loc[mask, "n_nights"]
        mean_n = nights.mean()
        sd_n = nights.std(ddof=1)
        ax.errorbar(i, mean_n, yerr=sd_n, fmt="o", markersize=8,
                    color=RISK_PALETTE[g], ecolor=RISK_PALETTE[g],
                    elinewidth=2, capsize=3, label=f"Mean ± SD")

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel("Number of nights", fontsize=8)
    ax.set_xlabel("")
    ax.set_title("A. Actigraphy nights by RBD group", fontsize=8, fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim([0, 8])
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # ── Panel B: Threshold sensitivity bar chart ───────────────────────────
    ax = axes[1]

    threshold_labels_list = ["All nights"] + [THRESHOLD_LABELS[t] for t in NIGHT_THRESHOLDS]
    all_thresholds = [None] + NIGHT_THRESHOLDS  # None represents "all nights"

    n_subjects_list = []
    pct_retained_list = []

    for threshold in all_thresholds:
        if threshold is None:
            df_filt = df
        else:
            df_filt = df[df["n_nights"] >= threshold]
        n_subjects_list.append(len(df_filt))
        pct_retained_list.append(100.0 * len(df_filt) / len(df))

    x = np.arange(len(threshold_labels_list))
    bars = ax.bar(x, n_subjects_list, color="#4472C4", alpha=0.8, edgecolor="black", linewidth=0.5)

    # Annotate bars with N and %
    for i, (bar, n, pct) in enumerate(zip(bars, n_subjects_list, pct_retained_list)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f"N={n:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(threshold_labels_list, rotation=30, ha="right")
    ax.set_ylabel("N subjects", fontsize=8)
    ax.set_xlabel("")
    ax.set_title("B. Cohort size at minimum nights thresholds", fontsize=8, fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim([0, max(n_subjects_list) * 1.15])
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()

    # Save
    pdf_path = output_dir / "nights_distribution.pdf"
    png_path = output_dir / "nights_distribution.png"
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {pdf_path.name}")
    print(f"  Saved: {png_path.name}")
    plt.close()


# ============================================================================
# EXCEL WRITER
# ============================================================================


def _write_excel(path: Path, sheets: List[Tuple[pd.DataFrame, str]]) -> None:
    """Write DataFrames to Excel workbook."""
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for df, sheet in sheets:
                df.to_excel(writer, sheet_name=sheet, index=False)
        print(f"  Saved: {path.name}")
    except PermissionError:
        warnings.warn(f"Cannot write '{path.name}' — file is open in Excel.")


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """Run night-count sensitivity analysis."""

    # ── 1. Paths ─────────────────────────────────────────────────────────────
    dir_final = config["pp"]["final_dir"]
    dir_thresh = config["pp"]["thresholds"]["root"]
    path_results = config["results"]["root"] / "table_one"
    path_results.mkdir(parents=True, exist_ok=True)

    col_irbd = "abk_rbd_score_mean"

    # ── 2. Load night-level data ──────────────────────────────────────────────
    print("[1/5] Loading night-level data …")
    thresholds, df_risk = get_clean_risk_data(
        file_name="ehr_diag_pd_rbd_only_all",
        thresholds_root=dir_thresh,
        final_dir=dir_final,
    )
    print(f"  Night-level rows (before subject collapse): {len(df_risk):,}")

    # ── 3. CRITICAL: Compute n_nights BEFORE make_subject_level() ────────────
    print("[2/5] Computing night counts (before subject-level collapse) …")
    n_nights_series = df_risk.groupby("eid").size().rename("n_nights")
    print(f"  Subjects with night count: {len(n_nights_series):,}")

    # Collapse to subject level
    df_subj = make_subject_level(df_risk, id_col="eid", prob_col=col_irbd)
    print(f"  Subject-level rows (after collapse): {len(df_subj):,}")

    # Re-attach n_nights (computed BEFORE collapse)
    df_subj = df_subj.merge(n_nights_series, on="eid", how="left")
    print(f"  Subjects with n_nights merged: {df_subj['n_nights'].notna().sum():,}")

    # ── 4. Apply prevalent PD exclusion (mirror Table 1) ──────────────────────
    surv_col = col_surv_time(STRATIFICATION_OUTCOME)
    n_before = len(df_subj)
    df_subj = df_subj[df_subj[surv_col].notna()].copy()
    n_excluded = n_before - len(df_subj)
    print(f"  Incident cohort (after prevalent exclusion): {len(df_subj):,} "
          f"(excluded {n_excluded:,})")

    # ── 5. Get risk group and prepare for stratification ──────────────────────
    group_col = col_risk_group_agnostic("percentile_3g")
    if group_col not in df_subj.columns:
        raise ValueError(f"Risk group column '{group_col}' not found in dataset")

    df_subj[group_col] = df_subj[group_col].astype(str)
    groups = sorted([g for g in df_subj[group_col].unique() if g not in ("nan", "None", "")])
    print(f"  Risk groups identified: {groups}")

    # ── 6. Build night summary table ──────────────────────────────────────────
    print("\n[3/5] Building night summary by RBD group …")
    tbl_summary = build_night_summary_table(df_subj, group_col, groups)
    print(tbl_summary.to_string(index=False))

    # ── 7. Build threshold sensitivity table ──────────────────────────────────
    print("\n[4/5] Building threshold sensitivity table …")
    tbl_thresh = build_threshold_sensitivity_table(df_subj, group_col)
    print(tbl_thresh.to_string(index=False))

    # ── 8. Create visualization ──────────────────────────────────────────────
    print("\n[5/5] Creating figures …")
    plot_nights_distribution(df_subj, group_col, groups, path_results)

    # ── 9. Write Excel ───────────────────────────────────────────────────────
    print("\nWriting outputs …")
    summary_path = path_results / "night_count_summary_by_group.xlsx"
    _write_excel(summary_path, [(tbl_summary, "Night Summary")])

    thresh_path = path_results / "night_threshold_sensitivity.xlsx"
    _write_excel(thresh_path, [(tbl_thresh, "Threshold Sensitivity")])

    print(f"\nOutputs written to: {path_results}")
    print("Done.")


# ============================================================================
if __name__ == "__main__":
    main()
