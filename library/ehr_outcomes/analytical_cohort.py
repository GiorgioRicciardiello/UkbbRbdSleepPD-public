import pandas as pd
from library.column_registry import col_incident, col_prevalent

def get_analytic_cohort(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """
    Returns the analytic cohort for a given outcome:
        - incident cases for the outcome
        - eligible controls
        - exclusion filters applied: neuro_exclude, val_pd, val_dlb, train_sleep
    """

    incident_col = col_incident(outcome)
    prevalent_col = col_prevalent(outcome)

    # ---------------------------------------------------------
    # 1) CASES = incident for this outcome
    #    (prevalent automatically excluded)
    # ---------------------------------------------------------
    mask_cases = df[incident_col] == True

    # ---------------------------------------------------------
    # 2) CONTROLS = pre-defined clean controls
    # ---------------------------------------------------------
    mask_controls = df["control"] == True

    # ---------------------------------------------------------
    # 3) SPLIT FILTERS (avoid leakage)
    # ---------------------------------------------------------
    mask_splits = (
        (df["val_pd"] == False) &
        (df["val_dlb"] == False) &
        (df["train_sleep"] == False)
    )

    # ---------------------------------------------------------
    # 4) NEUROLOGICAL EXCLUSIONS
    # ---------------------------------------------------------
    mask_neuro = (df["neuro_exclude"] == False)

    # ---------------------------------------------------------
    # 5) FINAL analytic mask
    # ---------------------------------------------------------
    mask_final = (
        (mask_cases | mask_controls) &
        mask_splits &
        mask_neuro
    )

    return df.loc[mask_final].copy()

