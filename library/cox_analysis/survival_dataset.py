import pandas as pd

from library.column_registry import col_surv_time, col_surv_event


def select_survival_dataset(
    df: pd.DataFrame,
    outcome: str,
    time_unit: str = "years",
    drop_prevalent: bool = True,
    incident_col: str | None = None,
) -> pd.DataFrame:
    """
    Select a survival-ready dataset for ONE outcome, using precomputed columns.

    Parameters
    ----------
    df : pd.DataFrame
        Cohort DataFrame (prevalent cases should already be absent).
    outcome : str
        Outcome identifier (e.g. 'outcome_1a_pd_only').
    time_unit : str
        'years' (default) or 'days'.
    drop_prevalent : bool
        Drop rows with NaN surv_time as a secondary guard against prevalent leakage.
    incident_col : str or None
        If provided, overwrite the 'event' column with this binary flag (0/1).
        Use the outcome__incident column to ensure strictly binary event encoding
        and exclude competing deaths (surv_event value=2) from the primary event.
        If None, the __surv_event column (0/1/2) is used as-is.
    """
    surv_time_col = col_surv_time(outcome)
    surv_event_col = col_surv_event(outcome)

    if surv_time_col not in df.columns:
        raise KeyError(f"Missing column: {surv_time_col}")
    if surv_event_col not in df.columns:
        raise KeyError(f"Missing column: {surv_event_col}")
    if incident_col is not None and incident_col not in df.columns:
        raise KeyError(f"Missing incident column: {incident_col}")

    out = df.copy()
    if drop_prevalent:
        out = out[out[surv_time_col].notna()].copy()

    out = out.rename(columns={surv_time_col: "time", surv_event_col: "event"})

    if time_unit == "years":
        out["time"] = out["time"] / 365.25
    elif time_unit != "days":
        raise ValueError("time_unit must be 'days' or 'years'")

    if incident_col is not None:
        # Use the binary incident flag as the event indicator.
        # __surv_event encodes competing deaths as 2, which lifelines treats as
        # a primary event.  __incident is strictly 0/1: True only for subjects
        # diagnosed during follow-up.
        out["event"] = out[incident_col].fillna(False).astype(int)
    else:
        out["event"] = out["event"].astype(int)

    if out["event"].sum() == 0:
        raise ValueError(f"No events observed for outcome '{outcome}'")

    return out
