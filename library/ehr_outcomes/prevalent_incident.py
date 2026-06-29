"""
Prevalent / Incident classification for outcome diagnosis timing.

This module classifies each subject into:
    - diagnosed
    - prevalent (diagnosed before wear_time_start)
    - incident  (diagnosed after wear_time_start)
    - time-to-event (TTEdays) from wear_time_start

Required input columns:
    - wear_time_start (datetime-convertible)
    - For each outcome "X":
          outcome flag:       X
          diagnosis date:     X_date

Output columns created for each outcome "X":
    - X_diagnosed
    - X_prevalent
    - X_incident
    - X_TTE_days
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List


def add_prevalent_incident(
    df: pd.DataFrame,
    outcome_cols: List[str],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Computes prevalent vs incident status and time-to-event for each outcome.
    Also prints diagnostic block explaining why 'PD/AD/DEM > ANY outcome'.
    """

    df = df.copy()
    n = df.shape[0]

    # -------------------------------------------------------------
    # Ensure wear_time_start is datetime
    # -------------------------------------------------------------
    df["wear_time_start"] = pd.to_datetime(df["wear_time_start"], errors="coerce")

    # -------------------------------------------------------------
    # Iterate over each outcome
    # -------------------------------------------------------------
    for outc in outcome_cols:

        # Resolve date column
        date_col = f"{outc}_date"
        if date_col not in df.columns:
            raise ValueError(f"Missing required diagnosis date column '{date_col}'")

        # Convert diagnosis date safely
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        # Diagnosed = outcome flag + non-null date
        diagnosed_col = f"{outc}_diagnosed"
        df[diagnosed_col] = df[outc] & df[date_col].notna()

        # Prevalent: diagnosis BEFORE accelerometer start
        prev_col = f"{outc}_prevalent"
        df[prev_col] = df[diagnosed_col] & (df[date_col] < df["wear_time_start"])

        # Incident: diagnosis AFTER accelerometer start
        inc_col = f"{outc}_incident"
        df[inc_col] = df[diagnosed_col] & (df[date_col] >= df["wear_time_start"])

        # TTE
        tte_col = f"{outc}_TTE_days"
        df[tte_col] = (df[date_col] - df["wear_time_start"]).dt.days
        df.loc[df[prev_col], tte_col] = np.nan

    # -------------------------------------------------------------
    # PRINT SUMMARY
    # -------------------------------------------------------------
    if verbose:

        print("\n" + "=" * 72)
        print(" PREVALENT / INCIDENT CLASSIFICATION SUMMARY")
        print("=" * 72)
        print(f"Total subjects: {n:,}\n")
        print(f"{'Outcome':<35} {'Diagnosed':<12} {'Prevalent':<12} {'Incident':<12}")
        print("-" * 72)

        for outc in outcome_cols:
            diagnosed = df[f"{outc}_diagnosed"].sum()
            prevalent = df[f"{outc}_prevalent"].sum()
            incident  = df[f"{outc}_incident"].sum()

            print(
                f"{outc:<35}"
                f"{diagnosed:<12}({diagnosed/n:.2%})  "
                f"{prevalent:<12}({prevalent/n:.2%})  "
                f"{incident:<12}({incident/n:.2%})"
            )

        print("=" * 72 + "\n")

        # =============================================================
        #             ? UNMAPPED CASES DIAGNOSTICS
        # =============================================================
        print("\n" + "=" * 72)
        print(" UNMAPPED DIAGNOSTICS (PD/AD/DEM not matching ANY composite outcome)")
        print("=" * 72)

        # Composite ANY outcome mask
        composite_any = df[outcome_cols].sum(axis=1) > 0

        # Basic ICD flags
        PD = df["PD_flag"]
        AD = df["AD_flag"]
        DEM = df["DEM_flag"]

        # Unmapped = PD/AD/DEM but NOT any composite outcome
        unmapped = (PD | AD | DEM) & (~composite_any)
        n_unmapped = unmapped.sum()

        print(f"Subjects with PD/AD/DEM ICD codes: { (PD | AD | DEM).sum():,}")
        print(f"Subjects in ANY composite outcome: { composite_any.sum():,}")
        print(f"UNMAPPED subjects (raw ICD signal but no composite): {n_unmapped:,}")
        print("-" * 72)

        # Breakdown of unmapped categories
        categories = {
            "PD only (no composite)":        PD & ~AD & ~DEM & unmapped,
            "AD only (no composite)":        AD & ~PD & ~DEM & unmapped,
            "DEM only (no composite)":       DEM & ~PD & ~AD & unmapped,
            "PD + AD only (no composite)":   PD & AD & ~DEM & unmapped,
            "PD + DEM only (no composite)":  PD & DEM & ~AD & unmapped,
            "AD + DEM only (no composite)":  AD & DEM & ~PD & unmapped,
            "PD+AD+DEM (no composite)":      PD & AD & DEM & unmapped,
        }

        for label, mask in categories.items():
            count = mask.sum()
            print(f"{label:<35} : {count:<6} ({count/n:.2%})")

        print("=" * 72 + "\n")

    return df
