"""
Table saving and summary generation for mediation analysis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def save_table(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """
    Save DataFrame to disk. Uses .xlsx for Excel, .csv otherwise.

    Parameters
    ----------
    df : pd.DataFrame
        Table to save.
    path : Path
        Output file path.
    """
    if df is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        df.to_excel(path, index=False, engine="openpyxl")
    else:
        df.to_csv(path, index=False)
    print(f"    Saved: {path.name} ({len(df)} rows)")


def save_interpretation_tables(
    tables: Dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """
    Save all tables for a single interpretation (A or C).

    Parameters
    ----------
    tables : dict
        Mapping of table name -> DataFrame.
    out_dir : Path
        Interpretation output directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    file_map = {
        "assoc_linear": "assoc_linear.xlsx",
        "assoc_logistic": "assoc_logistic.xlsx",
        "assoc_logistic_3g": "assoc_logistic_3g.xlsx",
        "mediation_steps": "mediation_steps.xlsx",
        "mediation_indirect": "mediation_indirect.xlsx",
        "mediation_model_perf": "mediation_model_perf.xlsx",
        "mediation_feasibility": "mediation_feasibility.xlsx",
        "supplementary_3g": "supplementary_3g_bpath.xlsx",
        "temporal_filter_log": "temporal_filter_log.xlsx",
    }
    for key, filename in file_map.items():
        if key in tables and tables[key] is not None:
            save_table(tables[key], out_dir / filename)


def build_summary_table(
    df_steps: pd.DataFrame,
    df_boot: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a one-row-per-variable mediation summary.

    Combines point estimates (c-path HR, PM%) with bootstrap CIs.

    Parameters
    ----------
    df_steps : pd.DataFrame
        Mediation step results.
    df_boot : pd.DataFrame
        Bootstrap confidence intervals.

    Returns
    -------
    pd.DataFrame
        Summary with columns: variable, label, hr_c, hr_c_ci, p_c,
        hr_indirect, hr_indirect_ci, pm_pct, pm_pct_ci, inconsistent.
    """
    if df_steps is None or df_steps.empty:
        return pd.DataFrame()

    rows = []
    for _, step in df_steps.iterrows():
        row = {
            "variable": step["variable"],
            "label": step["label"],
            "hr_c": step["hr_c"],
            "hr_c_ci": f"{step['hr_c_lower']:.3f}-{step['hr_c_upper']:.3f}",
            "p_c": step["p_c"],
            "p_c_fdr": step.get("p_c_fdr", None),
            "beta_a": step["beta_a"],
            "p_a": step["p_a"],
            "p_a_fdr": step.get("p_a_fdr", None),
            "hr_b": step["hr_b"],
            "p_b": step["p_b"],
            "hr_cprime": step["hr_cprime"],
            "p_cprime": step["p_cprime"],
            "hr_indirect_point": step["hr_indirect"],
            "pm_pct_point": step["pm_pct"],
            "inconsistent_mediation": step["inconsistent_mediation"],
            "c_index_c": step["c_index_c"],
            "c_index_joint": step["c_index_joint"],
            "delta_c": round(step["c_index_joint"] - step["c_index_c"], 4),
            "n": step["n"],
            "events": step["events"],
        }

        # Merge bootstrap CIs
        if df_boot is not None and not df_boot.empty:
            boot_match = df_boot[df_boot["variable"] == step["variable"]]
            if not boot_match.empty:
                brow = boot_match.iloc[0]
                row["hr_indirect"] = brow["hr_indirect"]
                row["hr_indirect_ci"] = (
                    f"{brow['hr_indirect_lci']:.3f}-{brow['hr_indirect_uci']:.3f}"
                )
                row["pm_pct"] = brow["pm_pct"]
                row["pm_pct_ci"] = f"{brow['pm_pct_lci']:.1f}-{brow['pm_pct_uci']:.1f}"
                row["n_converged"] = brow["n_converged"]
            else:
                row["hr_indirect"] = step["hr_indirect"]
                row["hr_indirect_ci"] = "N/A"
                row["pm_pct"] = step["pm_pct"]
                row["pm_pct_ci"] = "N/A"
                row["n_converged"] = 0
        else:
            row["hr_indirect"] = step["hr_indirect"]
            row["hr_indirect_ci"] = "N/A"
            row["pm_pct"] = step["pm_pct"]
            row["pm_pct_ci"] = "N/A"
            row["n_converged"] = 0

        rows.append(row)

    return pd.DataFrame(rows)


def build_model_performance_summary(
    df_perf: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reshape model performance into a comparison table.

    Parameters
    ----------
    df_perf : pd.DataFrame
        Raw model performance rows.

    Returns
    -------
    pd.DataFrame
        Pivoted: one row per variable, columns for c-path vs joint metrics.
    """
    if df_perf is None or df_perf.empty:
        return pd.DataFrame()

    rows = []
    for var in df_perf["variable"].unique():
        var_df = df_perf[df_perf["variable"] == var]
        row = {"variable": var}
        label_row = var_df.iloc[0]
        row["label"] = label_row["label"]

        for _, r in var_df.iterrows():
            prefix = r["model"]
            row[f"{prefix}_c_index"] = r["c_index"]
            row[f"{prefix}_aic"] = r["aic"]
            row[f"{prefix}_lrt_stat"] = r["lrt_stat"]
            row[f"{prefix}_lrt_p"] = r["lrt_p"]

        # Delta-C
        c_path_c = row.get("c_path_total_c_index", None)
        joint_c = row.get("joint_cprime_b_c_index", None)
        if c_path_c is not None and joint_c is not None:
            row["delta_c"] = round(joint_c - c_path_c, 4)

        rows.append(row)

    return pd.DataFrame(rows)
