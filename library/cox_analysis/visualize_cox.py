
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from tabulate import tabulate

from config.config import config, outcomes
from library.risk.risk_helpers import get_clean_risk_data, make_subject_level
from library.risk.survival_analysis import METHOD_TO_RISK_SUFFIX
from library.cox_analysis.cox_fit import fit_cox_with_ph_handling_ngroups
from library.cox_analysis.select_risk_groups import make_high_vs_low
from library.cox_analysis.helper import consort_counts, validate_covariates_exist, log
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(
    context="paper",
    style="whitegrid",
    font_scale=1.1
)

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


# %% Primary forest plot (Model A, High vs Low)
# - Purpose: Main inferential figure
def forest_plot_modelA(
        results_df,
        outcome,
        age_group="True",
        figsize=(6, 4)
):
    df = results_df[
        (results_df["outcome"] == outcome) &
        (results_df["age_group"] == str(age_group))
        ].copy()

    df = df.sort_values("HR_A")
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=figsize)

    ax.errorbar(
        df["HR_A"],
        y,
        xerr=[
            df["HR_A"] - df["LCI_A"],
            df["UCI_A"] - df["HR_A"]
        ],
        fmt="o",
        color="black",
        ecolor="black",
        capsize=3
    )

    ax.axvline(1.0, color="red", linestyle="--", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(df["method"])
    ax.set_xscale("log")

    ax.set_xlabel("Hazard Ratio (High vs Low)")
    ax.set_title(f"{outcome.replace('_', ' ')}")

    plt.tight_layout()
    return fig


def forest_plot_by_age_modelA(
        results_df,
        outcome,
        method,
        figsize=(7.5, 4.5),
        digits=2
):
    """
    Forest plot (Model A) across age groups for ONE outcome and ONE method,
    including HR, CI, number of events, and total N in y-axis labels.
    """

    df = results_df[
        (results_df["outcome"] == outcome) &
        (results_df["method"] == method)
        ].copy()

    if df.empty:
        raise ValueError("No data for this outcome/method combination.")

    # Sort age groups for readability
    df["age_group_label"] = df["age_group"].astype(str)
    df = df.sort_values("age_group_label")

    y = np.arange(len(df))

    # Build rich y-axis labels
    y_labels = []
    for _, r in df.iterrows():
        label = (
            f"{r['age_group_label']}  \n  "
            f"HR {r['HR_A']:.{digits}f} "
            f"({r['LCI_A']:.{digits}f}?{r['UCI_A']:.{digits}f})  \n  "
            f"Events {int(r['N_events'])} / N {int(r['N_analysis'])}"
        )
        y_labels.append(label)

    fig, ax = plt.subplots(figsize=figsize)

    ax.errorbar(
        df["HR_A"],
        y,
        xerr=[
            df["HR_A"] - df["LCI_A"],
            df["UCI_A"] - df["HR_A"]
        ],
        fmt="o",
        color="black",
        capsize=3
    )

    ax.axvline(1.0, color="red", linestyle="--", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(y_labels)
    ax.set_xscale("log")

    ax.set_xlabel("Hazard Ratio (High vs Low)")
    ax.set_ylabel("Age group")

    ax.set_title(
        f"{outcome.replace('_', ' ')}\nMethod: {method}",
        fontsize=11
    )

    plt.tight_layout()
    return fig


# %% Side-by-side forest (Model A vs Model B)
# - Purpose: Show attenuation after prodromal adjustment
def forest_plot_A_vs_B(results_df, outcome, figsize=(7, 4)):
    df = results_df[results_df["outcome"] == outcome].copy()
    df = df.sort_values("HR_A")
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=figsize)

    ax.errorbar(
        df["HR_A"], y + 0.1,
        xerr=[df["HR_A"] - df["LCI_A"], df["UCI_A"] - df["HR_A"]],
        fmt="o", label="Model A", capsize=3
    )

    ax.errorbar(
        df["HR_B"], y - 0.1,
        xerr=[df["HR_B"] - df["LCI_B"], df["UCI_B"] - df["HR_B"]],
        fmt="s", label="Model B", capsize=3
    )

    ax.axvline(1.0, color="red", linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(df["method"])
    ax.set_xscale("log")
    ax.set_xlabel("Hazard Ratio")
    ax.legend()

    plt.tight_layout()
    return fig


def forest_plot_A_vs_B_by_age(
        results_df,
        outcome,
        method,
        age_group_col="age_group",
        figsize=(7, 4),
        sort_by="HR_A"
):
    """
    Forest plot for ONE outcome and ONE method.
    Y-axis shows age groups.
    Plots Model A and Model B hazard ratios.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output results table.
    outcome : str
        Outcome name (e.g. 'outcome_1a_pd_only').
    method : str
        Risk stratification method (e.g. 'percentile_2g').
    age_group_col : str
        Column defining age groups.
    figsize : tuple
        Figure size.
    sort_by : str
        Column to sort by ('HR_A' or 'age_group').
    """

    # ----------------------------------
    # Filter data
    # ----------------------------------
    df = results_df[
        (results_df["outcome"] == outcome) &
        (results_df["method"] == method)
        ].copy()

    if df.empty:
        raise ValueError("No rows found for this outcome/method combination.")

    # ----------------------------------
    # Sort rows (optional)
    # ----------------------------------
    if sort_by in df.columns:
        df = df.sort_values(sort_by)

    y = np.arange(len(df))

    # ----------------------------------
    # Build plot
    # ----------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # Model A
    ax.errorbar(
        df["HR_A"],
        y + 0.12,
        xerr=[
            df["HR_A"] - df["LCI_A"],
            df["UCI_A"] - df["HR_A"]
        ],
        fmt="o",
        color="black",
        capsize=3,
        label="Model A (confounder-adjusted)"
    )

    # Model B (if present)
    if df["HR_B"].notna().any():
        ax.errorbar(
            df["HR_B"],
            y - 0.12,
            xerr=[
                df["HR_B"] - df["LCI_B"],
                df["UCI_B"] - df["HR_B"]
            ],
            fmt="s",
            color="gray",
            capsize=3,
            label="Model B (mediator-adjusted)"
        )

    # ----------------------------------
    # Reference line and axes
    # ----------------------------------
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(df[age_group_col])

    ax.set_xscale("log")
    ax.set_xlabel("Hazard Ratio (log scale)")
    ax.set_title(f"{outcome} ? {method}")

    ax.legend(frameon=False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    plt.tight_layout()
    return fig


# %% PH violation heatmap
# - Purpose: Transparency on assumptions (very strong reviewer signal)
def ph_violation_heatmap(ph_df, figsize=(8, 5)):
    df = ph_df.copy()

    ph_cols = [c for c in df.columns if c.startswith("ph_p_")]
    df["min_p"] = df[ph_cols].min(axis=1)

    pivot = df.pivot_table(
        index=["outcome", "method"],
        columns="model",
        values="min_p"
    )

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        -np.log10(pivot),
        cmap="viridis",
        linewidths=0.5,
        cbar_kws={"label": "-log10(PH p-value)"},
        ax=ax
    )

    ax.set_title("Proportional Hazards Diagnostics")
    plt.tight_layout()
    return fig


# %% Events vs HR scatter (small-sample stability check)
# - Purpose: Show estimates are not driven by tiny strata
def events_vs_hr_plot(results_df, model="A", figsize=(6, 4)):
    hr = f"HR_{model}"

    fig, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(
        data=results_df,
        x="N_events",
        y=hr,
        hue="method",
        size="N_analysis",
        sizes=(40, 200),
        ax=ax
    )

    ax.axhline(1.0, color="red", linestyle="--")
    ax.set_yscale("log")

    ax.set_xlabel("Number of Events")
    ax.set_ylabel("Hazard Ratio")

    plt.tight_layout()
    return fig


# %% Attenuation plot (Model A -> Model B)
# - Purpose: Visualize mediation / confounding
def attenuation_plot(results_df, figsize=(5, 5)):
    fig, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(
        data=results_df,
        x="HR_A",
        y="HR_B",
        hue="method",
        ax=ax
    )

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])
    ]

    ax.plot(lims, lims, "--", color="gray")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("HR Model A")
    ax.set_ylabel("HR Model B")

    plt.tight_layout()
    return fig


# %% CONSORT-style event bar plot
# - Purpose: Transparent reporting of contributing data
def consort_barplot(results_df, figsize=(7, 4)):
    df = results_df.copy()

    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=df,
        x="method",
        y="N_events",
        hue="outcome",
        ax=ax
    )

    ax.set_ylabel("Number of Events")
    ax.set_xlabel("Method")

    plt.tight_layout()
    return fig


import seaborn as sns
import matplotlib.pyplot as plt


def consort_barplot_by_age(
        results_df,
        outcome,
        method,
        age_group_col="age_group",
        figsize=(8, 4)
):
    """
    CONSORT-style bar plot showing contributing data
    for ONE outcome and ONE method.

    Bars = number of events
    Text = N analyzed, N censored, N prevalent excluded
    """

    # ----------------------------------
    # Filter
    # ----------------------------------
    df = results_df[
        (results_df["outcome"] == outcome) &
        (results_df["method"] == method)
        ].copy()

    if df.empty:
        raise ValueError("No data for this outcome/method.")

    # Sort age groups for readability
    df = df.sort_values(age_group_col)

    # ----------------------------------
    # Plot
    # ----------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=df,
        x=age_group_col,
        y="N_events",
        color="steelblue",
        ax=ax
    )

    # ----------------------------------
    # Annotate bars with denominators
    # ----------------------------------
    for i, row in df.iterrows():
        ax.text(
            x=list(df[age_group_col]).index(row[age_group_col]),
            y=row["N_events"] + 0.5,
            s=(
                f"Events: {int(row['N_events'])}\n"
                f"N: {int(row['N_analysis'])}\n"
                f"Censored: {int(row['N_censored'])}\n"
                f"Prev excl: {int(row['N_prevalent_excluded'])}"
            ),
            ha="center",
            va="bottom",
            fontsize=9
        )

    # ----------------------------------
    # Labels and title
    # ----------------------------------
    ax.set_ylabel("Number of Incident Events")
    ax.set_xlabel("Age Group")
    ax.set_title(f"CONSORT Contribution ? {outcome} | {method}")

    ax.grid(axis="y", linestyle=":", alpha=0.4)

    plt.tight_layout()
    return fig


# %%
def format_results_table(results_df):
    cols = [
        "outcome", "method", "HR_A", "LCI_A", "UCI_A", "p_A",
        "HR_B", "LCI_B", "UCI_B", "p_B",
        "N_analysis", "N_events"
    ]

    table = results_df[cols].copy()
    table = table.round(3)

    return table


# How to use them:
#     # %% caller plots
#     outcome = 'outcome_1a_pd_only' # 'outcome_1b_pd_ad'
#     method = 'percentile_2g'  # 'percentile_3g'
#     # Main figures
#     fig = forest_plot_modelA(
#         results_df=results_df,
#         outcome="outcome_1a_pd_only",
#         age_group="True"  # age_group_none
#     )
#
#     # fig.savefig("fig1_forest_modelA_pd_only.png", bbox_inches="tight")
#     plt.show()
#
#
#     fig = forest_plot_by_age_modelA(
#         results_df=results_df,
#         outcome=outcome,
#         method=method
#     )
#     plt.show()
#
#
#     # Main figures
#     fig = forest_plot_A_vs_B(
#         results_df=results_df,
#         outcome="outcome_1a_pd_only"
#     )
#
#     # fig.savefig("fig2_forest_A_vs_B_pd_only.png", bbox_inches="tight")
#     plt.show()
#
#     fig = forest_plot_A_vs_B_by_age(
#         results_df=results_df,
#         outcome="outcome_1a_pd_only",
#         method="percentile_3g",
#         age_group_col="age_group"
#     )
#
#     plt.show()
#
#     # Supplementary
#     # assumptions check
#     fig = ph_violation_heatmap(
#         ph_df=ph_df
#     )
#
#     # fig.savefig("figS1_ph_diagnostics_heatmap.png", bbox_inches="tight")
#     plt.show()
#
#     # Use to show estimates are not driven by tiny strata.
#     fig = events_vs_hr_plot(
#         results_df=results_df,
#         model="A"  # or "B"
#     )
#
#     # fig.savefig("figS2_events_vs_hr.png", bbox_inches="tight")
#     plt.show()
#
#     # ?How much does adjustment attenuate the association??
#     fig = attenuation_plot(
#         results_df=results_df
#     )
#
#     # fig.savefig("figS3_attenuation_plot.png", bbox_inches="tight")
#     plt.show()
#
#     fig = consort_barplot(
#         results_df=results_df
#     )
#
#     # fig.savefig("figS4_consort_events.png", bbox_inches="tight")
#     plt.show()
#
#     fig = consort_barplot_by_age(
#         results_df=results_df,
#         outcome="outcome_1a_pd_only",
#         method="percentile_2g"
#     )
#
#     plt.show()