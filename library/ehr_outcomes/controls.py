"""
Universal control definition for outcome-based epidemiological analysis.

Controls must:
    - Have NO outcomes for ANY of the defined outcome categories
    - Have NO PD / AD / Dementia flags
    - Have NO neurological exclusion codes
"""

import pandas as pd
from config.config import outcomes
from pathlib import Path
from tabulate import tabulate
from library.ehr_outcomes.utils import report_outcomes_by_flags

def add_controls(
    df: pd.DataFrame,
    save_dir: Path | None = None,
    verbose: bool = True,
    overwrite: bool = True,
) -> pd.DataFrame:
    """
    Adds a boolean 'control' column.

    Control definition:
    - No PD, AD, DEM, or DLB diagnosis (ever)
    - No baseline neurological exclusion
    """

    path_file = save_dir.joinpath("3_controls.parquet") if save_dir else None
    if save_dir and not overwrite and path_file.exists():
        return pd.read_parquet(path_file)

    print("Processing control flags...")
    df = df.copy()

    # ---------------------------------------------------------
    # 1. No base neurodegenerative diseases (EVER)
    # DLB excluded from active outcomes (2026-03-29); flag no longer generated.
    # ---------------------------------------------------------
    no_neuro_disease = (
        (~df["pd_flag"]) &
        (~df["ad_flag"]) &
        (~df["dem_flag"])
    )

    # ---------------------------------------------------------
    # 2. No neurological exclusion at baseline
    # ---------------------------------------------------------
    clean_neuro = ~df["neuro_exclude"]

    # ---------------------------------------------------------
    # FINAL CONTROL FLAG
    # ---------------------------------------------------------
    df["control"] = no_neuro_disease & clean_neuro

    # ---------------------------------------------------------
    # Reporting
    # ---------------------------------------------------------
    report_controls_overall(
        df=df,
        output_path=save_dir / "control_summary.csv",
        verbose=True,
    )

    if save_dir:
        df.to_parquet(path_file, index=False)

    return df


def report_controls_overall(
    df: pd.DataFrame,
    control_col: str = "control",
    verbose: bool = True,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Report overall control counts (NOT stratified by outcomes).
    """

    n_total = len(df)
    n_controls = int(df[control_col].sum())
    n_cases = n_total - n_controls

    report = pd.DataFrame([{
        "N_total": n_total,
        "N_controls": n_controls,
        "pct_controls": round(100 * n_controls / n_total, 2),
        "N_cases": n_cases,
        "pct_cases": round(100 * n_cases / n_total, 2),
    }])

    if verbose:
        print("\n" + "=" * 80)
        print("CONTROL SUMMARY")
        print("=" * 80)
        print(tabulate(report, headers="keys", tablefmt="github", showindex=False))
        print("=" * 80)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)

    return report
