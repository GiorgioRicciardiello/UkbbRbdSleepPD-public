"""
Survival Analysis Module
========================

This module performs survival analysis stratified by risk groups across
different risk-scoring methods (percentile, ROC, PR, F1, survival-based, quartiles).

Outputs:
- Kaplan?Meier curves for each method ? (validation / non-validation).
- Log-rank tests comparing lowest vs highest risk groups.
- Cox PH hazard ratios treating risk group as an ordinal predictor.
- Median survival times per group.
- Number-at-risk tables at regular time intervals.
- A structured DataFrame summarizing all survival statistics.

Dependencies:
    pandas, numpy, matplotlib, lifelines, scipy, library.risk.risk_helpers
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

from library.risk.risk_helpers import get_group_colors
from library.column_registry import (
    col_dx, col_prevalent, col_incident,
    col_surv_time, col_surv_event,
    col_risk_group_agnostic, METHOD_TO_RISK_SUFFIX,
)


# ============================================================
# ATTRITION SUMMARY
# ============================================================

def survival_attrition_summary(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """
    Produces a clear epidemiological summary of case exclusion for survival analysis.

    Returns:
        DataFrame with counts for:
            total subjects
            diagnosed
            prevalent cases
            incident cases
            controls
            survival data completeness
    """
    out = outcome.lower()

    diagnosed_col = col_dx(out)
    prevalent_col = col_prevalent(out)
    incident_col = col_incident(out)
    time_col     = col_surv_time(out)
    event_col    = col_surv_event(out)

    N_total = len(df)
    N_diag = df[diagnosed_col].sum()
    N_prev = df[prevalent_col].sum()
    N_inc  = df[incident_col].sum()
    N_ctrl = (~df[diagnosed_col]).sum()

    # Missing survival time = censored prevalent cases, mostly
    mask_surv_valid = df[event_col].notna() & df[time_col].notna()
    N_surv_valid   = mask_surv_valid.sum()
    N_surv_missing = N_total - N_surv_valid

    summary_df = pd.DataFrame({
        "Outcome": [
            outcome, outcome, outcome,
            outcome, outcome, outcome,
            outcome, outcome
        ],
        "Metric": [
            "Total subjects",
            "Diagnosed cases",
            "   ?- Prevalent cases (excluded)",
            "   ?- Incident cases (included)",
            "Controls (censored)",
            "",
            "Survival data available",
            "Survival data missing",
        ],
        "Count": [
            N_total,
            N_diag,
            N_prev,
            N_inc,
            N_ctrl,
            "",
            N_surv_valid,
            N_surv_missing
        ]
    })

    return summary_df



# ============================================================
# NUMBER-AT-RISK TABLE
# ============================================================

def compute_number_at_risk(
    df: pd.DataFrame,
    groups: List[str],
    time_col: str,
    event_col: str,
    time_points: np.ndarray
) -> pd.DataFrame:
    """
    Computes the number-at-risk per timepoint for each risk group.
    """
    result = {g: [] for g in groups}

    for g in groups:
        sub = df[df["risk_group"] == g]

        for t in time_points:
            # At risk if event has not yet occurred before time t
            at_risk = np.sum(sub[time_col] >= t)
            result[g].append(at_risk)

    return pd.DataFrame(result, index=time_points)



# ============================================================
# KM PANEL CELL (SINGLE METHOD ? SPLIT)
# ============================================================

def km_panel_cell(
    ax: plt.Axes,
    df: pd.DataFrame,
    outcome: str,
    method: str,
    time_col: str,
    event_col: str
) -> Optional[Dict[str, Any]]:
    """
    Draws a Kaplan?Meier survival curve for a given method & split (validation / non-validation).

    Returns:
        {
            "logrank_p": float,
            "HR": float,
            "LCI": float,
            "UCI": float,
            "medians": Dict[group -> median_survival],
            "groups": list of groups,
            "nar_table": pd.DataFrame (# at risk),
            "time_points": np.ndarray
        }
    """

    df = df.copy()

    # Exclude prevalent cases (missing survival vars)
    mask = df[time_col].notna() & df[event_col].notna()
    df = df[mask]

    if df.empty:
        ax.text(0.5, 0.5, "No valid survival data", ha="center")
        ax.axis("off")
        return None

    # Determine risk-group column based on method
    risk_col = col_risk_group_agnostic(method)
    df["risk_group"] = df[risk_col].astype(str)

    # Convert days -> years
    df["time_years"] = df[time_col] #  / 365.25

    groups = sorted(df["risk_group"].unique())
    colors = get_group_colors(len(groups))

    kmf = KaplanMeierFitter()

    # -------------------------------------------------------
    # KM Curves
    # -------------------------------------------------------
    for g, clr in zip(groups, colors):
        sub = df[df["risk_group"] == g]
        if sub.empty:
            continue

        kmf.fit(
            durations=sub["time_years"],
            event_observed=sub[event_col],
            label=g
        )
        # KM curve
        kmf.plot(ax=ax, ci_show=False, color=clr, linewidth=2)

        # --- Add CI band ---
        ci = kmf.confidence_interval_
        lower = ci.iloc[:, 0]
        upper = ci.iloc[:, 1]

        ax.fill_between(
            ci.index.values,
            lower.values,
            upper.values,
            color=clr,
            alpha=0.25,
            step="post"
        )
    # Style
    ax.set_xlabel("Time (years)", fontsize=9)
    ax.set_ylabel("Survival probability", fontsize=9)
    ax.grid(alpha=0.4)

    # -------------------------------------------------------
    # Log-rank Test (lowest vs highest)
    # -------------------------------------------------------
    try:
        g_low, g_high = groups[0], groups[-1]
        df_low  = df[df["risk_group"] == g_low]
        df_high = df[df["risk_group"] == g_high]

        lr_p = logrank_test(
            df_low["time_years"], df_high["time_years"],
            event_observed_A=df_low[event_col],
            event_observed_B=df_high[event_col]
        ).p_value
    except Exception:
        lr_p = np.nan

    # -------------------------------------------------------
    # Cox PH (risk as ordinal)
    # -------------------------------------------------------
    tmp = pd.DataFrame({
        "time_years": df["time_years"].astype(float),
        event_col: df[event_col].astype(int),
        "risk_code": df["risk_group"].astype("category").cat.codes + 1
    })

    cph = CoxPHFitter()
    try:
        cph.fit(tmp, duration_col="time_years", event_col=event_col)
        HR  = float(np.exp(cph.params_["risk_code"]))
        LCI, UCI = np.exp(cph.confidence_intervals_.loc["risk_code"])
        LCI, UCI = float(LCI), float(UCI)
    except Exception:
        HR, LCI, UCI = np.nan, np.nan, np.nan

    # -------------------------------------------------------
    # Group medians
    # -------------------------------------------------------
    medians: Dict[str, float] = {}
    for g in groups:
        sub = df[df["risk_group"] == g]
        if sub.empty:
            medians[g] = np.nan
        else:
            kmf.fit(sub["time_years"], event_observed=sub[event_col])
            medians[g] = float(kmf.median_survival_time_)

    # Annotation (only valid medians)
    median_lines = [
        f"{g}: {m:.1f}y" for g, m in medians.items()
        if pd.notna(m) and np.isfinite(m)
    ]

    if median_lines:
        ax.text(
            0.02, 0.02,
            "Median survival:\n" + "\n".join(median_lines),
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="gray")
        )

    # -------------------------------------------------------
    # Number-at-risk table (every 2 years)
    # -------------------------------------------------------
    max_time = float(np.nanmax(df["time_years"]))
    time_points = np.arange(0, max_time + 0.1, 2)

    nar_table = compute_number_at_risk(
        df=df,
        groups=groups,
        time_col="time_years",
        event_col=event_col,
        time_points=time_points
    )

    return {
        "logrank_p": lr_p,
        "HR": HR,
        "LCI": LCI,
        "UCI": UCI,
        "medians": medians,
        "groups": groups,
        "nar_table": nar_table,
        "time_points": time_points,
    }



# ============================================================
# MAIN FUNCTION: SURVIVAL PANELS PER OUTCOME
# ============================================================

def survival_panels_per_outcome(
    df: pd.DataFrame,
    outcomes: List[str],
    methods: List[str],
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (12, 3)
) -> pd.DataFrame:
    """
    Creates full KM panel figures for each outcome and returns all statistics
    in a single structured DataFrame.

    Returns:
        DataFrame with columns:
            outcome, method, split, HR, LCI, UCI, logrank_p, groups, medians
    """

    all_stats: List[Dict[str, Any]] = []

    for outcome in outcomes:

        time_col  = col_surv_time(outcome)
        event_col = col_surv_event(outcome)

        fig, axes = plt.subplots(
            nrows=len(methods), ncols=2,
            figsize=(figsize[0], figsize[1] * len(methods))
        )
        axes = np.atleast_2d(axes)

        fig.suptitle(f"Survival Panel ? {outcome}", fontsize=15)

        for i, method in enumerate(methods):

            # =============================
            # VALIDATION SET
            # =============================
            ax_val = axes[i, 0]
            df_val = df[df["val"] == True].copy()

            stats_val = km_panel_cell(ax_val, df_val, outcome, method, time_col, event_col)

            if stats_val is not None:
                all_stats.append({
                    "outcome": outcome,
                    "method": method,
                    "split": "validation",
                    "HR": stats_val["HR"],
                    "LCI": stats_val["LCI"],
                    "UCI": stats_val["UCI"],
                    "logrank_p": stats_val["logrank_p"],
                    "groups": stats_val["groups"],
                    "medians": stats_val["medians"]
                })

                ax_val.set_title(
                    f"{method.upper()} ? Validation\n"
                    f"HR={stats_val['HR']:.2f} ({stats_val['LCI']:.2f}, {stats_val['UCI']:.2f}) "
                    f"p={stats_val['logrank_p']:.3g}",
                    fontsize=10
                )
            else:
                ax_val.set_title(f"{method.upper()} ? Validation\nNo survival data", fontsize=10)

            # =============================
            # NON-VALIDATION SET
            # =============================
            ax_non = axes[i, 1]
            df_non = df[df["val"] == False].copy()

            stats_non = km_panel_cell(ax_non, df_non, outcome, method, time_col, event_col)

            if stats_non is not None:
                all_stats.append({
                    "outcome": outcome,
                    "method": method,
                    "split": "non-validation",
                    "HR": stats_non["HR"],
                    "LCI": stats_non["LCI"],
                    "UCI": stats_non["UCI"],
                    "logrank_p": stats_non["logrank_p"],
                    "groups": stats_non["groups"],
                    "medians": stats_non["medians"]
                })

                ax_non.set_title(
                    f"{method.upper()} ? Non-validation\n"
                    f"HR={stats_non['HR']:.2f} ({stats_non['LCI']:.2f}, {stats_non['UCI']:.2f}) "
                    f"p={stats_non['logrank_p']:.3g}",
                    fontsize=10
                )
            else:
                ax_non.set_title(f"{method.upper()} ? Non-validation\nNo survival data", fontsize=10)

        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if save_path:
            out_file = save_path / f"surv_panel_{outcome}.png"
            fig.savefig(out_file, dpi=300)
            print(f"Saved: {out_file}")

        plt.show()

    return pd.DataFrame(all_stats)
