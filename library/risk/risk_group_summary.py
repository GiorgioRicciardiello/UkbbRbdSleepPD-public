import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import os
from typing import Union, List, Optional, Tuple
from pathlib import Path

import os
from lifelines import KaplanMeierFitter
from tqdm import tqdm
from library.column_registry import (
    col_incident, col_tte_days, col_risk_group_agnostic,
    OUTCOMES,
)


def summarize_rbd_risk_groups(
        df: pd.DataFrame,
        output_dir: Union[Path, str] = "./risk_summary"
):
    """


    Requirements:
      df must contain, for each outcome:
         <outcome>_risk_group_mean
         <outcome>_incident
         <outcome>_tte_days

    Automatically:
      - detects outcomes
      - computes:
          * incident counts
          * incident risk ratios
          * KM curves
          * incident bar plots

    Parameters
    ----------
    df : DataFrame
        Full dataframe with incident outcomes and risk groups.
    output_dir : str
        Directory to store results.

    Returns
    -------
    results : dict
        Stores summary tables + plot paths per outcome.
    """


    # ------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ------------------------------------------------------------
    # Detect outcomes automatically
    # outcome = prefix before "_risk_group_mean"
    # ------------------------------------------------------------
    outcomes = [oc for oc in OUTCOMES if col_incident(oc) in df.columns]

    print(f"Detected outcomes: {outcomes}")

    # ------------------------------------------------------------
    # Process each outcome
    # ------------------------------------------------------------
    for outcome in outcomes:

        print(f"\n========== Processing OUTCOME: {outcome} ==========")

        rg_col = col_risk_group_agnostic("percentile_3g")
        incident_col = col_incident(outcome)
        tte_col = col_tte_days(outcome)

        # Validate
        for col in [rg_col, incident_col, tte_col]:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        # Subject-level table
        subj_df = (
            df.groupby("eid")
            .agg(
                risk_group=(rg_col, "first"),
                event=(incident_col, "max"),
                time=(tte_col, "max"),
                control=('control', 'max')
            )
            .reset_index()
        )

        # ============================
        # TABLE 1 ? counts
        # ============================
        count_table = subj_df.groupby("risk_group")["event"].agg(
            positives="sum",
            total="count"
        )
        count_table["proportion_incident"] = (
            count_table["positives"] / count_table["total"]
        )

        # ============================
        # TABLE 2 ? RISK RATIOS
        # ============================
        low_risk = (
            count_table.loc["Low (0?90%)", "positives"]
            / count_table.loc["Low (0?90%)", "total"]
        )

        rr_list = []
        for grp in count_table.index:
            grp_risk = (
                count_table.loc[grp, "positives"]
                / count_table.loc[grp, "total"]
            )
            rr = grp_risk / low_risk if low_risk > 0 else np.nan
            rr_list.append((grp, rr))

        rr_table = pd.DataFrame(rr_list, columns=["risk_group", "risk_ratio"]).set_index("risk_group")

        # ============================
        # KM CURVES
        # ============================
        km = KaplanMeierFitter()
        plt.figure(figsize=(7, 5))

        for grp in subj_df["risk_group"].unique():
            mask = subj_df["risk_group"] == grp
            km.fit(
                durations=subj_df.loc[mask, "time"],
                event_observed=subj_df.loc[mask, "event"],
                label=str(grp)
            )
            km.plot(ci_show=False)

        plt.title(f"KM Curve ? INCIDENT {outcome}")
        plt.xlabel("Time to event (days)")
        plt.ylabel("Survival Probability")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        km_path = os.path.join(output_dir, f"KM_incident_{outcome}.png")
        plt.savefig(km_path, dpi=200)
        plt.close()

        # ============================
        # BAR PLOT
        # ============================
        plt.figure(figsize=(6, 4))
        plt.bar(
            count_table.index,
            count_table["proportion_incident"],
            color=["#A3D5FF", "#FFDFA3", "#FFA3A3"]
        )
        plt.ylabel("Incident Event Proportion")
        plt.title(f"{outcome} ? Incident Proportion by RBD Risk Group")
        plt.tight_layout()

        bar_path = os.path.join(output_dir, f"Incident_Proportion_{outcome}.png")
        plt.savefig(bar_path, dpi=200)
        plt.close()

        # ============================
        # SAVE TABLES
        # ============================
        count_path = os.path.join(output_dir, f"incident_counts_{outcome}.csv")
        rr_path = os.path.join(output_dir, f"incident_risk_ratios_{outcome}.csv")

        count_table.to_csv(count_path)
        rr_table.to_csv(rr_path)

        # ============================
        # STORE RESULTS
        # ============================
        results[outcome] = {
            "incident_counts": count_table,
            "incident_risk_ratios": rr_table,
            "km_plot": km_path,
            "incident_proportion_plot": bar_path,
        }

    return results


def compute_incidence_and_rr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes:
        - Incident rates per 1,000 participants
        - Risk ratios (RR) comparing High and Intermediate vs Low
        - One combined results table for all outcomes

    Assumptions:
        df contains outcome-specific columns in the format:
            <outcome>_incident
            <outcome>_risk_group_mean

    Returns:
        DataFrame: rows = each outcome ? risk group
                   columns = cases, subjects, rate_per_1000, RR_vs_low
    """

    # ------------------------------------------------------------
    # 1. Identify outcomes automatically
    # ------------------------------------------------------------
    outcomes = [oc for oc in OUTCOMES if col_incident(oc) in df.columns]

    # Output table
    rows = []

    # ------------------------------------------------------------
    # 2. Process each outcome
    # ------------------------------------------------------------
    for outcome in tqdm(outcomes, total=len(outcomes), desc="Generating RR tables"):
        rg_col = col_risk_group_agnostic("percentile_3g")
        incident_col = col_incident(outcome)

        # Subject-level view
        subj_df = (
            df.groupby("eid")
            .agg(
                risk_group=(rg_col, "first"),
                incident=(incident_col, "max"),
            )
            .reset_index()
        )

        # Ensure groups are ordered Low -> Intermediate -> High
        group_sizes = subj_df.groupby("risk_group").size()
        groups = group_sizes.index.tolist()

        if not all(g in groups for g in [
            "Low (0?90%)", "Intermediate (90?99%)", "High (99?100%)"
        ]):
            raise ValueError("Risk groups missing or misspelled.")

        # Compute case counts and denominators
        counts = subj_df.groupby("risk_group")["incident"].agg(
            cases="sum", subjects="count"
        )

        # Compute incident rates per 1000
        counts["rate_per_1000"] = (counts["cases"] / counts["subjects"]) * 1000

        # Compute RR vs Low
        low_rate = counts.loc["Low (0?90%)", "rate_per_1000"]

        def compute_rr(rate):
            return rate / low_rate if low_rate > 0 else np.nan

        counts["RR_vs_low"] = counts["rate_per_1000"].apply(compute_rr)

        # Add outcome column
        counts["Outcome"] = outcome

        # Collect rows
        rows.append(counts.reset_index())

    # ------------------------------------------------------------
    # 3. Return combined table
    # ------------------------------------------------------------
    return pd.concat(rows, axis=0).reset_index(drop=True)[[
        "Outcome",
        "risk_group",
        "cases",
        "subjects",
        "rate_per_1000",
        "RR_vs_low"
    ]]



from lifelines import KaplanMeierFitter

def compute_km_per_outcome(df:pd.DataFrame):

    def plot_km(subj_df,
                outcome_name:str='',
                figsize=(10,6)):
        """
        subj_df must have columns:
            - 'time'   : tte_days
            - 'event'  : 1 = incident, 0 = censored
            - 'risk_group': Low / Intermediate / High
        """

        kmf = KaplanMeierFitter()

        risk_groups = subj_df['risk_group'].unique()

        plt.figure(figsize=figsize)

        for group in sorted(risk_groups):
            mask = subj_df['risk_group'] == group
            kmf.fit(
                subj_df.loc[mask, 'time'],
                subj_df.loc[mask, 'event'],
                label=group
            )
            kmf.plot_survival_function(ci_show=False)

        plt.title(f"KM Survival Curve: {outcome_name}")
        plt.xlabel("Time (days)")
        plt.ylabel("Survival probability (event-free)")
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.show()

    outcomes = [oc for oc in OUTCOMES if col_incident(oc) in df.columns]

    print(f"Detected outcomes: {outcomes}")

    # ------------------------------------------------------------
    # Process each outcome
    # ------------------------------------------------------------
    for outcome in outcomes:
        # Subject-level table
        subj_df = get_subjects_per_outcome(df, outcome=outcome)


    plot_km(subj_df, outcome_name='PD')


def get_subjects_per_outcome(df: pd.DataFrame,
                             outcome: str) -> pd.DataFrame:
    """
    Get subjects per outcome and organize them in a subject-level view.

    This function processes a given dataframe to compute a subject-level
    view aggregation based on specific outcome-related columns. It aggregates
    data by subject ID ('eid') and ensures that the computed risk groups are
    ordered as "Low -> Intermediate -> High".

    :param df: A pandas DataFrame containing the data for analysis.
               It must include columns for the specified outcome as well
               as control information.
    :param outcome: The outcome name used to identify relevant columns
                    for risk group, incident events, and time-to-event.
    :raises ValueError: If the specified outcome column is missing
                        from the input dataframe.
    :return: A pandas DataFrame representing the subject-level view
             with aggregated data per subject 'eid'.
    """
    if not outcome in df.columns:
        raise ValueError(f"Missing column: {outcome}")

    rg_col = col_risk_group_agnostic("percentile_3g")
    incident_col = col_incident(outcome)
    tte_col = col_tte_days(outcome)

    # Subject-level view
    subj_df = (
        df.groupby("eid")
        .agg(
            risk_group=(rg_col, "first"),
            event=(incident_col, "max"),
            time=(tte_col, "max"),
            control=('control', 'max')
        )
        .reset_index()
    )
    # Ensure groups are ordered Low -> Intermediate -> High
    # group_sizes = subj_df.groupby("risk_group").size()
    # groups = group_sizes.index.tolist()

    return subj_df

def plot_incidence_rates(meta: dict, figsize=(12, 6)):
    """
    Plot incident rates per 1000 participants for each outcome in the metadata.

    Parameters
    ----------
    meta : dict
        The metadata dictionary you provided.
    figsize : tuple
        Matplotlib figure size.
    """

    # Extract outcomes and rates
    outcomes = list(meta.keys())
    incident_rates = [meta[o]['incident_rate_percent'] * 10 for o in outcomes]
    # percent -> per 1000: multiply by 10

    # Create figure
    plt.figure(figsize=figsize)
    x = np.arange(len(outcomes))

    bars = plt.bar(x, incident_rates)

    # Style
    plt.xticks(x, outcomes, rotation=45, ha='right')
    plt.ylabel("Incident rate per 1,000 participants")
    plt.title("Incident Rates Across Outcomes")
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Labels on top of bars
    for bar, rate in zip(bars, incident_rates):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{rate:.2f}",
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.tight_layout()
    plt.show()
