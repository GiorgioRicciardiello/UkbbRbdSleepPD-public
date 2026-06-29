"""
Utility functions for Cox prodromal analysis pipeline.
"""

from pathlib import Path

import pandas as pd


def save_table(df: pd.DataFrame, path: Path) -> None:
    """
    Save DataFrame as XLSX.

    Automatically converts .csv extension to .xlsx.

    Parameters
    ----------
    df : pd.DataFrame
        Data to save.
    path : Path or str
        Target path (.xlsx or .csv extension).
        If .csv extension provided, saves as .xlsx instead.
    """
    path = Path(path)
    # Convert .csv to .xlsx
    if path.suffix == ".csv":
        path = path.with_suffix(".xlsx")
    df.to_excel(path, index=False, engine="openpyxl")
