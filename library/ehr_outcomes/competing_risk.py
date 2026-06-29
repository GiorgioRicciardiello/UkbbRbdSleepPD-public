"""
Creates competing-risk survival structure for PD / AD / Dementia.
"""

import pandas as pd


def define_time_zero(df, enrollment_date_col="enroll_date"):
    """Defines baseline time zero."""
    df["time0"] = pd.to_datetime(df[enrollment_date_col])
    return df


def build_competing_risk_events(df):
    """
    Create unified event variables:
    - event_date = earliest of PD, AD, DEM
    - event_type = PD / AD / DEM
    - event_indicator = 1/0
    """

    df = df.copy()

    # Find earliest event among all neuro outcomes
    df["event_date"] = pd.concat(
        [df["PD_dx_date"], df["AD_dx_date"], df["DEM_dx_date"]],
        axis=1
    ).min(axis=1)

    def get_event_type(row):
        if pd.isna(row["event_date"]):
            return "none"
        if row["event_date"] == row["PD_dx_date"]:
            return "PD"
        if row["event_date"] == row["AD_dx_date"]:
            return "AD"
        if row["event_date"] == row["DEM_dx_date"]:
            return "DEM"
        return "none"

    df["event_type"] = df.apply(get_event_type, axis=1)
    df["event_indicator"] = df["event_type"].apply(lambda x: 1 if x != "none" else 0)

    df["time_to_event"] = (df["event_date"] - df["time0"]).dt.days

    return df
