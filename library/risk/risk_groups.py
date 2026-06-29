"""
Compute various forms of RBD risk grouping, standardize metadata structure, and
handle probability-based subject-level computations.

This module is part used in 'src/build_final_dataset.py` and the functions are
essential for defining the different risk groups from the RBD probabilities based
on the validation set.  It compute outcome-based risk groups using percentile or
quartile thresholds, and generate metadata in a structured format.

It primarily deals with RBD probabilities, facilitating research on outcomes
like neurodegenerative diseases, by splitting subjects into categorized risk
groups based on thresholds derived from validation subsets.

Functions
---------
- _rbd_probability_mean_within_subject: Computes average RBD probability per subject.
- _subject_level_frame: Aggregates nightly data and handles subject-level transformations.
- standardize_metadata: Enforces a common structure for metadata used in risk grouping.
- compute_risk_groups_percentile: Creates risk group categories using percentile thresholds.

"""
from __future__ import annotations
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from pandas import DataFrame
from tabulate import tabulate
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve
)
from scipy.stats import ks_2samp

from sklearn.metrics import roc_curve
from library.risk.risk_helpers import (load_and_normalize_thresholds,
                                       make_subject_level,
                                       plot_rbd_thresholds_methods_separately_per_outcome,
                                       plot_rbd_thresholds_methods_separately_per_outcome_all_data,
                                       plot_rbd_thresholds_methods_separately,
                                       plot_rbd_thresholds_publication,
                                       get_threshold)
from library.column_registry import col_risk_group, col_risk_group_agnostic, col_incident, col_surv_time, col_surv_event


# %% Helper functions
def _rbd_probability_mean_within_subject(df: pd.DataFrame,
                                         rbd_col:str='rbd_prob_class1') -> pd.DataFrame:
    """
    Compute mean RBD probability per subject
    Uses the RBD probabilities (night level) rbd_prob_class1, computes averages across the nights within the subjects
    and assigns the average to the subject level.
    :param df:
    :return:
    """
    if rbd_col not in df.columns:
        raise ValueError(f"Missing required column {rbd_col}.")
    if not 'prob_mean' in df.columns:
        df[rbd_col] = df[rbd_col].astype(float)

        # compute the mean probability within subject and creat the prob mean col
        subj_prob_mean = df.groupby("eid")[rbd_col].mean()
        df["prob_mean"] = df["eid"].map(subj_prob_mean).astype(float)

    return df


def _subject_level_frame(df:pd.DataFrame,
                         outcome_name:str,
                         val_flag:str) -> tuple[DataFrame, DataFrame]:
    """subject-level dataframe creation"""
    incident_col = col_incident(outcome_name)

    subj = (
        df.groupby("eid")
        .agg(
            prob_mean=("prob_mean", "first"),  # all prob are the same within subject
            outcome=(incident_col, "max"),  # outcome column
            flagged=(val_flag, "max"),
        )
        .reset_index()
    )

    val_df = subj[subj["flagged"] == True]

    return subj, val_df



def standardize_metadata(
        outcome_name: str,
        method: str,
        threshold: float | dict | None,
        n_val: int,
        incident_cases: int,
        risk_group_counts: dict,
        extra: dict | None = None
):
    """
    Enforces a unified metadata structure across all risk-group methods.

    Parameters
    ----------
    outcome_name : str
        Name of the outcome.
    method : str
        One of: 'percentile', 'quartile', 'roc', 'pr', 'f1', 'survival'.
    threshold : float or dict
        Single threshold or multiple (e.g., percentile cuts).
    n_val : int
        Number of validation subjects.
    incident_cases : int
        Count of incident cases.
    risk_group_counts : dict
        Distribution of assigned risk groups.
    extra : dict
        Method-specific additional fields (AUC, precision, recall, logrank stat...).

    Returns
    -------
    dict
        Fully standardized metadata entry.
    """
    return {
        "outcome": outcome_name,
        "method": method,
        "thresholds": threshold,
        "n_validation": n_val,
        "incident_cases": incident_cases,
        "risk_group_counts": risk_group_counts,
        "extra": extra or {}
    }



# %% Method 1 - 5 - All in one call
def run_compute_risk_groups(df: pd.DataFrame,
                            outcomes: List[str],
                            out_thresholds_dir: Path,
                            config: dict,
                            rbd_col="rbd_prob_class1",
                            out_frame_dir: Path = None,
                            val_flags: Dict[str, str] = None,
                            file_name: str = 'ehr_diag_pd_rbd',
                            ):
    """
    DEPRECATED: Use run_compute_risk_group_rbd_only instead.

    Processes the input DataFrame to compute multiple risk groups and their related
    metrics, saving the outputs to specified directories. The function applies
    various computations for percentile, ROC thresholds, PR thresholds, F1 scores,
    survival rates, and quartiles, evaluating the separability metrics in the end.

    :param df: The input DataFrame to be processed.
    :param outcomes: List of column names representing specific outcomes to compute
        metrics for.
    :param out_thresholds_dir: The directory where threshold-related outputs will
        be saved.
    :param out_frame_dir: The directory where the final processed DataFrame will
        be saved.
    :param val_flags: Dictionary containing validation flags used during
        computations.
    :param config: Dictionary containing configuration details such as file
        naming and settings.
    :param file_name: Name of the output parquet file (without extension).
    :return: A pandas DataFrame with the computed risk groups and metrics.
    """

    df = df.copy()
    warnings.warn(
        "run_compute_risk_groups is deprecated. Use run_compute_risk_group_rbd_only instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    out_thresholds_dir.mkdir(parents=True, exist_ok=True)
    out_frame_dir.mkdir(parents=True, exist_ok=True)

    # Use a subdirectory for this specific run's thresholds
    path_out = out_thresholds_dir.joinpath(file_name)
    path_out.mkdir(parents=True, exist_ok=True)

    # ======================================================
    # 1. Compute mean RBD probability per subject
    # ======================================================
    df = _rbd_probability_mean_within_subject(df=df,
                                              rbd_col=rbd_col)

    df_risk_2g, meta_pctl_2g = _compute_risk_groups_percentile_2groups(
        df=df,
        outcome_cols=outcomes,
        val_flags=val_flags,
        output_json=path_out.joinpath(
            config["pp"]['thresholds']['percentile_2g'])
    )

    df_risk_3g, meta_pctl_3g = _compute_risk_groups_percentile_3groups(
        df=df_risk_2g,
        outcome_cols=outcomes,
        val_flags=val_flags,
        output_json=path_out.joinpath(
            config["pp"]['thresholds']['percentile_3g'])
    )

    # not using the following for production
    # df_risk, meta_roc = _compute_risk_groups_roc(df=df_risk_3g,
    #                                              outcome_cols=outcomes,
    #                                              out_dir=path_out,
    #                                              verbose=True,
    #                                              val_flags=val_flags,
    #                                              output_json=path_out.joinpath(
    #                                                 config["pp"]['thresholds']['roc'])
    #                                              )
    #
    # df_risk, meta_pr = _compute_risk_groups_pr(df=df_risk,
    #                                            outcome_cols=outcomes,
    #                                            out_dir=path_out,
    #                                            verbose=True,
    #                                            val_flags=val_flags,
    #                                            output_json=path_out.joinpath(
    #                                               config["pp"]["thresholds"]['pr'])
    #                                            )
    #
    # df_risk, meta_f1 = _compute_risk_groups_f1(df=df_risk,
    #                                            outcome_cols=outcomes,
    #                                            out_dir=path_out,
    #                                            verbose=True,
    #                                            val_flags=val_flags,
    #                                            output_json=path_out.joinpath(
    #                                               config["pp"]["thresholds"]['f1'])
    #                                            )
    #
    # df_risk, meta_surv = _compute_risk_groups_survival(df=df_risk,
    #                                                    outcome_cols=outcomes,
    #                                                    val_flags=val_flags,
    #                                                    output_json=path_out.joinpath(
    #                                                       config["pp"]["thresholds"]['surv'])
    #                                                    )
    #
    df_risk, meta_q = _compute_risk_groups_quartiles(df=df_risk_3g,
                                                     outcome_cols=outcomes,
                                                     val_flags=val_flags,
                                                     output_json=path_out.joinpath(
                                                        config["pp"]["thresholds"]['quartile'])
                                                     )

    # compute_separability_metrics(df=df_risk,
    #                              outcome_cols=outcomes,
    #                              out_dir=path_out,
    #                              verbose=True)

    # Step 3 - get the thresholds for the ae group
    threshold_dict_paths = get_threshold(dir_thresholds=path_out)

    thresholds = load_and_normalize_thresholds(threshold_dict_paths=threshold_dict_paths, file_name=file_name)

    with open(path_out.joinpath(config["pp"]["thresholds"]['collection']), "w") as f:
        json.dump(thresholds, f, indent=4)

    # Step 4 - make subject-level
    df_subj = make_subject_level(df_risk,
                                 id_col="eid",
                                 prob_col="prob_mean")

    # remove part of the datasets used for training
    # df_subj = df_subj[~df_subj['train_sleep']]  # katarina's model is deprecated so no more training

    # Step 5 - Inspection of the RBD thresholds
    # Detect if we have a validation split column named 'val'
    has_val_col = 'val' in df_subj.columns

    if has_val_col:
        plot_rbd_thresholds_methods_separately_per_outcome(
            df=df_subj,
            outcomes=outcomes,
            thresholds=thresholds,
            prob_col="rbd_prob",
            save_path=path_out
        )
    else:
        plot_rbd_thresholds_methods_separately_per_outcome_all_data(
            df=df_subj,
            outcomes=outcomes,
            thresholds=thresholds,
            prob_col="rbd_prob",
            save_path=path_out
        )

    if out_frame_dir:
        df_risk.to_parquet(out_frame_dir.joinpath(f'{file_name}.parquet'),
                           index=False)

    return df_risk


# %% Outcome independent thresholds

def run_compute_risk_group_rbd_only(df: pd.DataFrame,
                                    out_thresholds_dir: Path,
                                    config: dict,
                                    outcomes: List[str],
                                    val_col: str | None = None,
                                    out_frame_dir: Path | None = None,
                                    file_name: str = 'rbd_only_risk_groups',
                                    rbd_col="rbd_prob_class1",
                                    ) -> pd.DataFrame:
    """
    Computes risk groups based SOLELY on the distribution of RBD probabilities (night-averaged),
    ignoring outcome labels.

    This is deterministic and outcome-agnostic.

    It computes:
      1. Percentile-based 2 groups (High >= 90th percentile)
      2. Percentile-based 3 groups (Low < 90, 90 <= Intermediate < 99, High >= 99)
      3. Quartile-based groups (Q1, Q2, Q3, Q4)

    Thresholds are derived from:
      - The subset where df[val_col] == True (if val_col is provided and exists).
      - The entire dataset otherwise.

    :param df: Input DataFrame (must have 'eid', 'rbd_prob_class1').
    :param out_thresholds_dir: Directory to save thresholds JSONs.
    :param config: Configuration dict for file naming (uses 'percentile_2g', etc. keys).
    :param val_col: Optional column name for validation split.
    :param out_frame_dir: Optional directory to save the result parquet.
    :param file_name: Name of the output parquet file (without extension).
    :return: DataFrame with new risk group columns.
    """
    df = df.copy()
    out_thresholds_dir = out_thresholds_dir.joinpath(file_name)

    out_thresholds_dir.mkdir(parents=True, exist_ok=True)
    if out_frame_dir:
        out_frame_dir.mkdir(parents=True, exist_ok=True)

    # 1. Compute subject-level means
    df = _rbd_probability_mean_within_subject(df, rbd_col=rbd_col)


    # 2. Identify the "Reference" set for thresholds
    #    Aggregate to subject level first to avoid night-level bias
    subj_df = df.groupby("eid")["prob_mean"].first().reset_index()

    if val_col and val_col in df.columns:
        # Get subject-level val flag
        # (Assuming flag is constant per subject, take max/first)
        subj_val_flags = df.groupby("eid")[val_col].max().astype(bool)
        subj_df = subj_df.merge(subj_val_flags, on="eid", how="left")
        
        # Filter for thresholds
        ref_df = subj_df[subj_df[val_col] == True]
        if ref_df.empty:
            print(f"WARNING: val_col '{val_col}' provided but no True values found. Using ALL data for thresholds.")
            ref_df = subj_df
    else:
        # Use all data
        ref_df = subj_df

    prob_values = ref_df["prob_mean"].dropna().values
    n_total_subj = len(subj_df)
    n_ref_subj = len(ref_df)

    # =========================================================
    # A. Percentile 2-Groups (90th)
    # =========================================================
    p90 = np.percentile(prob_values, 90)
    
    col_2g = col_risk_group_agnostic("percentile_2g")
    # Logic: High if >= p90, else Low
    # We apply this to the NIGHT-level dataframe `df`
    # (Use map for speed and consistency)
    
    # Calculate for all subjects
    subj_df[col_2g] = np.where(subj_df["prob_mean"] >= p90, "High", "Low")
    
    # Map back
    df[col_2g] = df["eid"].map(dict(zip(subj_df["eid"], subj_df[col_2g])))

    # Metadata
    meta_2g = standardize_metadata(
        outcome_name="rbd_only_distribution",
        method="percentile_2g",
        threshold={"p90": float(p90)},
        n_val=n_ref_subj,
        incident_cases=0, # No outcome involved
        risk_group_counts=subj_df[col_2g].value_counts().to_dict(),
        extra={"note": "Thresholds derived from RBD probability distribution only."}
    )
    
    # Save JSON
    name_2g = config.get("pp")["thresholds"]["percentile_2g"]
    with open(out_thresholds_dir.joinpath(name_2g), "w") as f:
        json.dump(meta_2g, f, indent=4)


    # =========================================================
    # B. Percentile 3-Groups (90th, 99th)
    # =========================================================
    p99 = np.percentile(prob_values, 99)
    # p90 already computed
    
    col_3g = col_risk_group_agnostic("percentile_3g")
    
    subj_df[col_3g] = pd.cut(
        subj_df["prob_mean"],
        bins=[-np.inf, p90, p99, np.inf],
        labels=["Low", "Mid", "High"],
        include_lowest=True
    )
    df[col_3g] = df["eid"].map(dict(zip(subj_df["eid"], subj_df[col_3g])))
    
    meta_3g = standardize_metadata(
        outcome_name="rbd_only_distribution",
        method="percentile_3g",
        threshold={"p90": float(p90), "p99": float(p99)},
        n_val=n_ref_subj,
        incident_cases=0,
        risk_group_counts=subj_df[col_3g].value_counts().to_dict(),
        extra={"note": "Thresholds derived from RBD probability distribution only."}
    )

    name_3g = config.get("pp")["thresholds"]["percentile_3g"]
    with open(out_thresholds_dir.joinpath(name_3g), "w") as f:
        json.dump(meta_3g, f, indent=4)

    # =========================================================
    # C. Quartiles
    # =========================================================
    q1, q2, q3 = np.percentile(prob_values, [25, 50, 75])
    
    col_q = col_risk_group_agnostic("quartile")
    subj_df[col_q] = pd.cut(
        subj_df["prob_mean"],
        bins=[-np.inf, q1, q2, q3, np.inf],
        labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"]
    )
    df[col_q] = df["eid"].map(dict(zip(subj_df["eid"], subj_df[col_q])))
    
    meta_q = standardize_metadata(
        outcome_name="rbd_only_distribution",
        method="quartiles",
        threshold={"q1": float(q1), "q2": float(q2), "q3": float(q3)},
        n_val=n_ref_subj,
        incident_cases=0,
        risk_group_counts=subj_df[col_q].value_counts().to_dict(),
        extra={"note": "Thresholds derived from RBD probability distribution only."}
    )
    
    name_q = config.get('pp')['thresholds']['quartile']
    with open(out_thresholds_dir.joinpath(name_q), "w") as f:
        json.dump(meta_q, f, indent=4)

    # ========================================
    # Collection
    # ========================================
    threshold_dict_paths = get_threshold(dir_thresholds=out_thresholds_dir)

    thresholds = load_and_normalize_thresholds(threshold_dict_paths=threshold_dict_paths, file_name=file_name)

    with open(out_thresholds_dir.joinpath(config["pp"]["thresholds"]['collection']), "w") as f:
        json.dump(thresholds, f, indent=4)

        # Step 4 - make subject-level
    df_subj = make_subject_level(df,
                                 id_col="eid",
                                 prob_col="prob_mean")

    # remove part of the datasets used for training
    # df_subj = df_subj[~df_subj['train_sleep']]

    # Step 5 - Inspection of the RBD thresholds
    # Per-outcome figures: thresholds are distribution-based (same cuts for
    # all outcomes) but N / events / controls / risk / RR are outcome-specific.
    plot_rbd_thresholds_methods_separately(
        df=df_subj,
        thresholds=thresholds,
        outcomes=outcomes,
        prob_col="rbd_prob",
        save_path=out_thresholds_dir,
    )

    # Step 5b - Publication-quality figures (3-panel histograms per outcome)
    # plot_rbd_thresholds_publication(
    #     df=df_subj,
    #     figsize = (7.5, 2.8),
    #     thresholds=thresholds,
    #     outcomes=outcomes,
    #     prob_col="rbd_prob",
    #     save_path=out_thresholds_dir,
    #     file_format="png",
    # )

    # Optional: Save Frame
    if out_frame_dir:
        df.to_parquet(out_frame_dir.joinpath(f'{file_name}.parquet'), index=False)

    return df

# %% Method 1 - Percentile-based grouping
# DEPRECATED: outcome-agnostic stratification replaces per-outcome computation
def _compute_risk_groups_percentile_3groups(
    df: pd.DataFrame,
    outcome_cols: List[str],
    val_flags: Dict[str, str],
    output_json: Path,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Compute outcome-specific RBD risk groups using nighttime-averaged RBD probabilities.
    Thresholds are derived from *incident cases only* inside the validation subset.

    For each outcome X, the function will create:
        - X_risk_group_mean (categorical: Low, Intermediate, High)
        - metadata entry storing thresholds, case counts, etc.

    Thresholds are based on percentiles of prob_mean in the validation set:
        - Low risk     : ? 90th percentile
        - Intermediate : 90?99th percentile
        - High risk    : ? 99th percentile

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
            - 'eid'
            - 'rbd_prob_class1' (night-level probability)
            - For each outcome:
                <outcome>_incident     (boolean)
                Validation flag column (from val_flags dict)
    outcome_cols : List[str]
        List of outcome columns (base names), e.g.:
            ["outcome_1a_pd_only", "outcome_2c_pd_otherdementia", ...]
    val_flags : Dict[str,str]
        Mapping outcome -> validation flag column, e.g.:
            {
                "outcome_1a_pd_only": "val_pd",
                "outcome_2a_otherdementia": "val_pd"
            }
    output_json : Path
        Where the metadata JSON will be saved.

    Returns
    -------
    df : pd.DataFrame
        Same df with new columns <outcome>_risk_group_mean for each outcome.
    metadata : Dict[str,dict]
        Dictionary storing thresholds and counts for all outcomes.
    """


    # df = df.copy()

    # # ======================================================
    # # 1. Compute mean RBD probability per subject
    # # ======================================================
    # df = _rbd_probability_mean_within_subject(df)

    metadata:Dict[str, dict] = {}

    # ======================================================
    # Internal per-outcome function
    # ======================================================
    def _compute_for_outcome(outcome_name: str, val_flag: str) -> dict:
        """
        Compute thresholds, risk groups, and summary statistics for a single outcome.
        """

        incident_col = col_incident(outcome_name)
        if incident_col not in df.columns:
            raise ValueError(f"Missing expected incident column: {incident_col}")

        # Subject-level data
        subj = (
            df.groupby("eid")
            .agg(
                prob_mean=("prob_mean", "mean"), # -> mean prob RBD
                outcome=(incident_col, "max"),
                flagged=(val_flag, "max"),
            )
            .reset_index()
        )

        # Validation subset, used only to compute the thresholds
        val_df = subj[subj["flagged"] == True]
        if val_df.empty:
            raise ValueError(f"No validation subjects available for outcome {outcome_name}")

        # Thresholds
        low_cut = np.percentile(val_df["prob_mean"].dropna(), 90)
        high_cut = np.percentile(val_df["prob_mean"].dropna(), 99)

        # Assign risk groups
        rg_col = col_risk_group(outcome_name, "percentile_3g")
        subj[rg_col] = pd.cut(
            subj["prob_mean"].dropna(),
            bins=[-np.inf, low_cut, high_cut, np.inf],
            labels=["Low (0?90%)", "Intermediate (90?99%)", "High (99?100%)"],
            include_lowest=True,
        )

        # Map to night-level frame
        df[rg_col] = df["eid"].map(dict(zip(subj["eid"], subj[rg_col])))

        # Build metadata
        meta = {
            "thresholds": {
                "p90": float(low_cut),
                "p99": float(high_cut),
            },
            "n_subjects_total": int(len(subj)),
            "n_subjects_validation": int(len(val_df)),
            "risk_group_counts": subj[rg_col].value_counts().to_dict(),
            "incident_cases": int(subj["outcome"].sum()),
            "incident_rate_percent": float(subj["outcome"].mean() * 100),
        }

        return meta

    # ======================================================
    # Process each outcome
    # ======================================================
    for outcome_name in outcome_cols:
        if outcome_name not in val_flags:
            raise ValueError(f"No validation flag provided for outcome '{outcome_name}'")

        metadata[outcome_name] = _compute_for_outcome(
            outcome_name=outcome_name, val_flag=val_flags[outcome_name]
        )

    # ======================================================
    # Save metadata
    # ======================================================
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w") as f:
        json.dump(metadata, f, indent=4)

    return df, metadata


# DEPRECATED: outcome-agnostic stratification replaces per-outcome computation
def _compute_risk_groups_percentile_2groups(
    df: pd.DataFrame,
    outcome_cols: List[str],
    val_flags: Dict[str, str],
    output_json: Path,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Compute outcome-specific RBD risk groups using nighttime-averaged RBD probabilities.

    Binary stratification:
        - Reference (0?90%)
        - High (90?100%)

    Thresholds are derived from the validation subset only.

    Returns
    -------
    df : pd.DataFrame
        Input df with added <outcome>_risk_group_mean_2g columns.
    metadata : Dict[str,dict]
        Thresholds and summary statistics per outcome.
    """


    metadata: Dict[str, dict] = {}

    # ======================================================
    # Internal per-outcome function
    # ======================================================
    def _compute_for_outcome(outcome_name: str, val_flag: str) -> dict:

        incident_col = col_incident(outcome_name)
        if incident_col not in df.columns:
            raise ValueError(f"Missing expected incident column: {incident_col}")

        subj = (
            df.groupby("eid")
            .agg(
                prob_mean=("prob_mean", "mean"),
                outcome=(incident_col, "max"),
                flagged=(val_flag, "max"),
            )
            .reset_index()
        )

        # Validation subset (only used to define threshold)
        val_df = subj[subj["flagged"] == True]
        if val_df.empty:
            raise ValueError(f"No validation subjects for outcome {outcome_name}")

        # Single threshold: top 10%
        p90 = np.percentile(val_df["prob_mean"], 90)

        rg_col = col_risk_group(outcome_name, "percentile_2g")
        subj[rg_col] = np.where(
            subj["prob_mean"] >= p90,
            "High (90?100%)",
            "Low (0?90%)",
        )

        # Map back to night-level dataframe
        df[rg_col] = df["eid"].map(dict(zip(subj["eid"], subj[rg_col])))

        meta = {
            "thresholds": {
                "p90": float(p90),
            },
            "n_subjects_total": int(len(subj)),
            "n_subjects_validation": int(len(val_df)),
            "risk_group_counts": subj[rg_col].value_counts().to_dict(),
            "incident_cases": int(subj["outcome"].sum()),
            "incident_rate_percent": float(subj["outcome"].mean() * 100),
        }

        return meta

    # ======================================================
    # Process each outcome
    # ======================================================
    for outcome_name in outcome_cols:
        if outcome_name not in val_flags:
            raise ValueError(f"No validation flag provided for outcome '{outcome_name}'")

        metadata[outcome_name] = _compute_for_outcome(
            outcome_name=outcome_name,
            val_flag=val_flags[outcome_name],
        )

    # ======================================================
    # Save metadata
    # ======================================================
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w") as f:
        json.dump(metadata, f, indent=4)

    return df, metadata


# %% Method 2 ? Quartile-based grouping
# DEPRECATED: outcome-agnostic stratification replaces per-outcome computation
def _compute_risk_groups_quartiles(df:pd.DataFrame,
                                   outcome_cols:List[str],
                                   val_flags:Dict[str,str],
                                   output_json: Path = ""):
    """
    Compute risk groups quartiles for given outcomes based on probability means within subjects.

    This function processes a DataFrame to calculate risk group quartiles for specified outcomes,
    grouping subjects into quartiles based on their "prob_mean" values. The quartile thresholds
    and risk group counts are saved as metadata, optionally written to a JSON file.

    :param df: Input DataFrame containing subject-level probability mean data.
    :type df: pd.DataFrame
    :param outcome_cols: List of outcome column names for which quartiles will be computed.
    :type outcome_cols: List[str]
    :param val_flags: A dictionary mapping outcome names to their corresponding validation flags.
    :type val_flags: Dict[str, str]
    :param output_json: Path to a JSON file where quartile thresholds and group counts
        metadata will be stored. Defaults to an empty string, implying no file is written.
    :type output_json: Path
    :return: The updated DataFrame with new columns for each outcome's risk group quartiles
        and a metadata dictionary containing quartile thresholds, risk group counts, and
        validation subject counts for each outcome.
    :rtype: Tuple[pd.DataFrame, Dict[str, Any]]
    """
    df = df.copy()
    # df = _rbd_probability_mean_within_subject(df=df)

    metadata = {}

    def _compute_for_outcome(outcome_name, val_flag):
        subj, val_df = _subject_level_frame(df, outcome_name, val_flag)
        if val_df.empty:
            raise ValueError(f"No validation subjects for {outcome_name}")

        q1, q2, q3 = np.percentile(val_df["prob_mean"], [25, 50, 75])

        rg_col = col_risk_group(outcome_name, "quartile")
        subj[rg_col] = pd.cut(
            subj["prob_mean"],
            bins=[-np.inf, q1, q2, q3, np.inf],
            labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"]
        )

        df[rg_col] = df["eid"].map(dict(zip(subj["eid"], subj[rg_col])))

        return {
            "thresholds": {"q1": float(q1), "q2": float(q2), "q3": float(q3)},
            "risk_group_counts": subj[rg_col].value_counts().to_dict(),
            "n_val": int(len(val_df)),
        }

    for outcome_name in outcome_cols:
        metadata[outcome_name] = _compute_for_outcome(outcome_name, val_flags[outcome_name])

    with open(output_json, "w") as f:
        json.dump(metadata, f, indent=4)


    return df, metadata


# %% Method 4 ? ROC-optimized threshold (Youden index)

def _compute_risk_groups_roc(df:pd.DataFrame,
                             outcome_cols:List[str],
                             val_flags:Dict[str,str],
                             out_dir: Path | None = None,
                             verbose:bool=False,
                             output_json:Path = "risk_groups_roc.json"):
    """
    Computes risk groups based on ROC analysis for specified outcomes in the provided DataFrame.

    The function operates by calculating the ROC curve for each outcome variable, identifying the
    optimal threshold using Youden's index, and then assigning a binary risk group ("High" or "Low")
    to individuals based on the threshold. The metadata containing details such as the threshold,
    sensitivity, specificity, and other metrics for each outcome is also returned.

    :param df: DataFrame containing the data used to compute risk groups
    :param outcome_cols: List of outcome column names for which to perform ROC analysis
    :param val_flags: Dictionary mapping each outcome to its validation flag indicating the subset
                      to be used for validation
    :return: A tuple containing the modified DataFrame with assigned risk groups and a dictionary
             with metadata for each outcome, including thresholds and associated metrics
    """

    def _plot_roc_for_outcome(outcome_name: str,
                              fpr: np.ndarray,
                              tpr: np.ndarray,
                              thresholds: np.ndarray,
                              best_idx: int,
                              y_true: np.ndarray,
                              verbose: bool = False,
                              save_path: Path | None = None):
        """
        Plot ROC curve for a given outcome, including:
            - ROC curve
            - Youden's index point
            - AUC
            - Sample size info (cases/controls)
            - Threshold + metrics in title
        """

        # --- Compute metrics ---
        roc_auc = auc(fpr, tpr)
        best_thresh = thresholds[best_idx]
        sensitivity = tpr[best_idx]
        specificity = 1 - fpr[best_idx]

        # Compute sample size info
        y_true_arr = np.asarray(y_true).astype(int)
        n_cases = int(y_true_arr.sum())
        n_controls = int((1 - y_true_arr).sum())
        n_total = len(y_true_arr)

        # --- Plot ---
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Chance")

        # Youden point
        plt.scatter(
            fpr[best_idx], tpr[best_idx],
            color="red", s=80,
            label=f"Youden (thr={best_thresh:.4f})"
        )

        # Title including sample size + metrics
        plt_title = (
            f"ROC ? {outcome_name.replace('_', ' ').upper()}\n"
            f"n={n_total} (cases={n_cases}, controls={n_controls}) * "
            f"AUC={roc_auc:.3f} * Sens={sensitivity:.3f} * Spec={specificity:.3f}"
        )
        plt.title(plt_title, fontsize=12)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")

        # Optional text box
        textstr = (
            f"AUC: {roc_auc:.3f}\n"
            f"Best threshold: {best_thresh:.4f}\n"
            f"Sensitivity: {sensitivity:.3f}\n"
            f"Specificity: {specificity:.3f}"
        )
        plt.gcf().text(
            0.63, 0.25, textstr,
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7)
        )

        # Save or show
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if verbose:
            plt.show()

        plt.close()

    def _compute_for_outcome(outcome_name:str,
                             val_flag:str,
                             out_dir:Path,
                             verbose:bool = False):
        """
        Performs outcome-specific computations including threshold determination based
        on ROC analysis, risk group assignments, and the generation of ROC plots.

        This function extracts subject-level data frames filtered by the outcome name
        and validation flag, computes ROC metrics and establishes an optimal
        threshold using the Youden index. It also generates risk group assignments
        and updates the dataframe accordingly. Finally, it provides a summary
        report containing threshold details, ROC metrics for the best threshold,
        case counts, and distribution of risk groups.

        :param outcome_name: The name of the outcome to compute metrics and
            thresholds for.
        :type outcome_name: str
        :param val_flag: A validation flag used to filter the data frame to
            identify validation subjects.
        :type val_flag: str
        :param out_dir: Path to the directory for saving generated ROC plot images.
        :type out_dir: Path
        :param verbose: Whether detailed logs and progress information should be
            displayed during computation.
        :type verbose: bool
        :return: Dictionary containing the computed threshold, TPR and FPR at the
            threshold, number of validation subjects, incident case count, and
            counts of risk groups.
        :rtype: dict
        """
        subj, val_df = _subject_level_frame(df, outcome_name, val_flag)
        if val_df.empty:
            raise ValueError(f"No validation subjects for {outcome_name}")

        # ROC-based threshold from the validation set
        fpr, tpr, thresholds = roc_curve(y_true=val_df["outcome"],
                                         y_score=val_df["prob_mean"])
        youden = tpr - fpr
        best_idx = np.argmax(youden)
        best_thresh = thresholds[best_idx]

        _plot_roc_for_outcome(
            outcome_name=outcome_name,
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
            best_idx=best_idx,
            y_true=val_df["outcome"].values,  # NEW
            verbose=verbose,
            save_path=out_dir.joinpath(f"roc_risk_group_{outcome_name}_roc.png")
        )

        # Assign binary risk (you may refine to 3 groups)
        rg_col = col_risk_group(outcome_name, "roc")
        subj[rg_col] = np.where(subj["prob_mean"] >= best_thresh, "High", "Low")

        df[rg_col] = df["eid"].map(dict(zip(subj["eid"], subj[rg_col])))

        return {
            "threshold": float(best_thresh),
            "roc_tpr": float(tpr[best_idx]),
            "roc_fpr": float(fpr[best_idx]),
            "n_val": int(len(val_df)),
            "incident_cases": int(subj["outcome"].sum()),
            "risk_group_counts": subj[rg_col].value_counts().to_dict(),
        }

    df = df.copy()
    # df = _rbd_probability_mean_within_subject(df=df)
    metadata = {}
    for outcome_name in outcome_cols:
        # outcome_name = outcome_cols[0 ]
        metadata[outcome_name] = _compute_for_outcome(outcome_name=outcome_name,
                                                      val_flag=val_flags[outcome_name],
                                                      out_dir=out_dir,
                                                      verbose=verbose)

    with open(output_json, "w") as f:
        json.dump(metadata, f, indent=4)

    return df, metadata

# %% Method 4b ? PR-optimized threshold (F score) - Class Imbalance

def _compute_risk_groups_pr(df: pd.DataFrame,
                            outcome_cols: List[str],
                            val_flags: Dict[str, str],
                            out_dir: Path | None = None,
                            verbose: bool = False,
                            output_json:Path='risk_groups_pr.json'):

    """
    Computes risk groups using precision-recall (PR) curves for multiple
    outcomes and appends the risk groups as new columns to the input DataFrame.

    This function computes precision-recall curves for specified outcome variables,
    determines the optimal threshold for assigning high/low risk categories based
    on the F1 score, and updates the input DataFrame with risk group information.
    Additionally, it generates metadata containing thresholds and performance
    metrics for each outcome, and optionally saves PR curve plots.

    :param df: The input DataFrame that contains subject-level data, including
        columns for predicted probabilities and outcome labels.
    :type df: pd.DataFrame

    :param outcome_cols: A list of outcome variable names for which risk groups
        need to be computed.
    :type outcome_cols: List[str]

    :param val_flags: A dictionary where keys are outcome variable names and
        values are validation flag filters to be used in computations.
    :type val_flags: Dict[str, str]

    :param out_dir: The directory where PR curve plots will be saved. If None,
        plots will not be saved.
    :type out_dir: Path | None

    :param verbose: Specifies whether plots should be displayed interactively
        and additional information printed.
    :type verbose: bool

    :return: A tuple consisting of:
        - The updated DataFrame with appended risk group columns.
        - Metadata for each outcome containing details such as
          thresholds, precision, recall, F1 score, and risk group counts.
    :rtype: Tuple[pd.DataFrame, Dict[str, Dict[str, Union[float, int, Dict[str, int]]]]]
    """
    df = df.copy()
    # df = _rbd_probability_mean_within_subject(df=df)

    def _plot_pr_for_outcome(outcome_name: str,
                             precision: np.ndarray,
                             recall: np.ndarray,
                             thresholds: np.ndarray,
                             best_idx: int,
                             y_true: np.ndarray,
                             y_scores: np.ndarray,
                             verbose: bool = False,
                             save_path: Path | None = None):

        # Correct AP calculation
        ap = average_precision_score(y_true, y_scores)

        best_thresh = thresholds[best_idx]
        best_prec = precision[best_idx]
        best_rec = recall[best_idx]
        best_f1 = 2 * (best_prec * best_rec) / (best_prec + best_rec + 1e-12)

        y_true_arr = np.asarray(y_true).astype(int)
        n_cases = int(y_true_arr.sum())
        n_controls = int((1 - y_true_arr).sum())
        n_total = len(y_true_arr)

        plt.figure(figsize=(7, 6))
        plt.plot(recall, precision, lw=2, color="blue", label=f"AP = {ap:.3f}")
        plt.scatter(recall[best_idx], precision[best_idx], color="red", s=80,
                    label=f"Best thr={best_thresh:.4f}")

        plt.title(
            f"PR Curve ? {outcome_name.replace('_', ' ').upper()}\n"
            f"n={n_total} (cases={n_cases}, controls={n_controls}) * "
            f"F1={best_f1:.3f} * Prec={best_prec:.3f} * Rec={best_rec:.3f}"
        )

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if verbose:
            plt.show()

        plt.close()

    def _compute_for_outcome(outcome_name: str,
                             val_flag: str,
                             out_dir: Path,
                             verbose: bool = False):
        subj, val_df = _subject_level_frame(df, outcome_name, val_flag)
        if val_df.empty:
            raise ValueError(f"No validation subjects for {outcome_name}")

        precision, recall, thresholds = precision_recall_curve(
            val_df["outcome"].astype(int),
            val_df["prob_mean"].astype(float)
        )

        # Compute F1 for each threshold
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (
            precision[:-1] + recall[:-1] + 1e-12
        )

        best_idx = int(np.argmax(f1_scores))
        best_thresh = thresholds[best_idx]

        _plot_pr_for_outcome(
            outcome_name=outcome_name,
            precision=precision,
            recall=recall,
            thresholds=thresholds,
            best_idx=best_idx,
            y_true=val_df["outcome"].values,
            y_scores=val_df["prob_mean"].values,
            verbose=verbose,
            save_path=out_dir / f"roc_risk_group_{outcome_name}_pr.png"
        )

        # Assign High/Low risk
        rg_col = col_risk_group(outcome_name, "pr")
        subj[rg_col] = np.where(subj["prob_mean"] >= best_thresh, "High", "Low")

        df[rg_col] = df["eid"].map(dict(zip(subj["eid"], subj[rg_col])))

        return {
            "threshold": float(best_thresh),
            "precision": float(precision[best_idx]),
            "recall": float(recall[best_idx]),
            "f1": float(f1_scores[best_idx]),
            "n_val": int(len(val_df)),
            "incident_cases": int(subj["outcome"].sum()),
            "risk_group_counts": subj[rg_col].value_counts().to_dict(),
        }

    metadata = {}
    for outcome_name in outcome_cols:
        metadata[outcome_name] = _compute_for_outcome(
            outcome_name=outcome_name,
            val_flag=val_flags[outcome_name],
            out_dir=out_dir,
            verbose=verbose
        )

    with open(output_json, "w") as f:
        json.dump(metadata, f, indent=4)

    return df, metadata


# %% Method 4c - Maximum F1 Score
def _compute_risk_groups_f1(df: pd.DataFrame,
                            outcome_cols: List[str],
                            val_flags: Dict[str, str],
                            out_dir: Path | None = None,
                            verbose: bool = False,
                            output_json:Path='risk_groups_f1.json'):

    from sklearn.metrics import precision_recall_curve

    df = df.copy()
    # df = _rbd_probability_mean_within_subject(df=df)

    def _compute_for_outcome(outcome_name: str,
                             val_flag: str,
                             verbose: bool = False):
        subj, val_df = _subject_level_frame(df, outcome_name, val_flag)

        if val_df.empty:
            raise ValueError(f"No validation subjects for {outcome_name}")

        # Precision/Recall
        precision, recall, thresholds = precision_recall_curve(
            val_df["outcome"].astype(int),
            val_df["prob_mean"].astype(float)
        )

        # Compute F1 scores
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (
            precision[:-1] + recall[:-1] + 1e-12
        )

        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx]

        # Assign High/Low
        rg_col = col_risk_group(outcome_name, "f1")
        subj[rg_col] = np.where(subj["prob_mean"] >= best_thresh, "High", "Low")
        df[rg_col] = df["eid"].map(dict(zip(subj["eid"], subj[rg_col])))

        return {
            "threshold": float(best_thresh),
            "precision": float(precision[best_idx]),
            "recall": float(recall[best_idx]),
            "f1": float(f1_scores[best_idx]),
            "n_val": int(len(val_df)),
            "incident_cases": int(subj["outcome"].sum()),
            "risk_group_counts": subj[rg_col].value_counts().to_dict(),
        }

    metadata = {
        outcome: _compute_for_outcome(outcome, val_flags[outcome], verbose)
        for outcome in outcome_cols
    }

    with open(output_json, "w") as f:
        json.dump(metadata, f, indent=4)

    return df, metadata



# %% Method 5 ? Survival-optimized threshold (maximizing C-index or log-rank)

from lifelines.statistics import logrank_test

def _compute_risk_groups_survival(df: pd.DataFrame,
                                  outcome_cols: List[str],
                                  val_flags: Dict[str, str],
                                  output_json:Path=''):
    df = df.copy()
    metadata = {}

    def _compute_for_outcome(outcome_name, val_flag):
        subj, val_df = _subject_level_frame(df, outcome_name, val_flag)
        if val_df.empty:
            raise ValueError(f"No validation subjects for {outcome_name}")

        # Valid survival columns
        time_col  = col_surv_time(outcome_name)
        event_col = col_surv_event(outcome_name)

        thresh_candidates = np.percentile(val_df["prob_mean"],
                                          np.linspace(10, 90, 40))
        best_thresh = None
        best_stat = -np.inf

        for t in thresh_candidates:
            g1 = val_df["prob_mean"] >= t
            g2 = val_df["prob_mean"] < t

            if g1.sum() < 10 or g2.sum() < 10:
                continue

            lr = logrank_test(
                df.loc[val_df.index, time_col][g1],
                df.loc[val_df.index, time_col][g2],
                event_observed_A=df.loc[val_df.index, event_col][g1],
                event_observed_B=df.loc[val_df.index, event_col][g2],
            )

            if lr.test_statistic > best_stat:
                best_stat = lr.test_statistic
                best_thresh = t

        # Assign groups
        rg_col = col_risk_group(outcome_name, "surv")
        subj[rg_col] = np.where(subj["prob_mean"] >= best_thresh, "High-risk", "Low-risk")

        df[rg_col] = df["eid"].map(dict(zip(subj["eid"], subj[rg_col])))

        return {
            "threshold": float(best_thresh),
            "logrank_statistic": float(best_stat),
            "risk_group_counts": subj[rg_col].value_counts().to_dict(),
            "n_val": int(len(val_df)),
        }

    for outcome_name in outcome_cols:
        metadata[outcome_name] = _compute_for_outcome(outcome_name, val_flags[outcome_name])

    with open(output_json, "w") as f:
        json.dump(metadata, f, indent=4)


    return df, metadata



def compute_separability_metrics(
        df: pd.DataFrame,
        outcome_cols: list[str],
        out_dir: Path,
        verbose: bool = False
) -> pd.DataFrame:
    """
    Computes separability metrics for each outcome using subject-level prob_mean.
    Saves:
        - separability_table.csv
        - separability_heatmap.png

    Metrics computed:
        - AUC-ROC
        - Average Precision (AP)
        - Best F1 score (optimal threshold)
        - KS statistic (distribution separation)
        - Cohen's d (effect size)
        - Risk Ratio (top 1% vs bottom 99%)
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # ---- subject-level data ----
    subj = df.groupby("eid").agg(
        prob_mean=("prob_mean", "mean"),
        **{f"{oc}_incident": (f"{oc}_incident", "max") for oc in outcome_cols}
    ).reset_index()

    for oc in outcome_cols:
        y = subj[f"{oc}_incident"].values.astype(int)
        p = subj["prob_mean"].values

        cases = subj[y == 1]
        controls = subj[y == 0]

        if y.sum() < 5:
            print(f"[WARNING] Very small number of cases for {oc}")

        # ---- 1. AUC ----
        auc_val = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan

        # ---- 2. Average Precision ----
        ap_val = average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan

        # ---- 3. Best F1 score ----
        precision, recall, thresholds = precision_recall_curve(y, p)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        best_f1 = np.nanmax(f1_scores)

        # ---- 4. KS statistic ----
        ks_val = ks_2samp(cases["prob_mean"], controls["prob_mean"]).statistic

        # ---- 5. Cohen's d ----
        mean1, mean0 = cases["prob_mean"].mean(), controls["prob_mean"].mean()
        s1, s0 = cases["prob_mean"].std(), controls["prob_mean"].std()
        pooled_sd = np.sqrt((s1 ** 2 + s0 ** 2) / 2)
        d_val = (mean1 - mean0) / pooled_sd if pooled_sd > 0 else np.nan

        # ---- 6. Risk Ratio (top 1% vs rest) ----
        high_cut = np.percentile(subj["prob_mean"], 99)
        top = subj[subj["prob_mean"] >= high_cut]
        bottom = subj[subj["prob_mean"] < high_cut]

        risk_top = (top[f"{oc}_incident"].mean() + 1e-6)
        risk_bottom = (bottom[f"{oc}_incident"].mean() + 1e-6)
        rr = risk_top / risk_bottom

        results.append({
            "Outcome": oc,
            "Cases": int(y.sum()),
            "Controls": int((1 - y).sum()),
            "AUC": auc_val,
            "Average Precision": ap_val,
            "Best F1": best_f1,
            "KS Statistic": ks_val,
            "Cohen_d": d_val,
            "Risk Ratio (top1%)": rr,
        })

    table = pd.DataFrame(results)

    # ---- save table ----
    table_path = out_dir / "separability_table.csv"
    table.to_csv(table_path, index=False)

    if verbose:
        print("\nSeparability Table:")
        print(tabulate(table, headers="keys", tablefmt="github", showindex=False))

    # ---- Heatmap ----
    heatmap_df = table.set_index("Outcome")[
        ["AUC", "Average Precision", "Best F1", "KS Statistic", "Cohen_d", "Risk Ratio (top1%)"]
    ]

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap_df,
        annot=True,
        cmap="viridis",
        fmt=".3f",
        linewidths=0.5
    )
    plt.title("Heatmap of Signal Separability Across Outcomes", fontsize=16)
    if out_dir:
        heatmap_path = out_dir / "separability_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        table.to_csv(heatmap_path.with_suffix(".csv"), index=False)
    plt.close()

    return table
