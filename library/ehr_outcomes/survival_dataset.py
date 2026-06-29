"""
Construct survival-ready datasets (long-format & wide-format).
"""

import pandas as pd


def build_wide_survival(df):
    """
    Output for e.g. Cox model:
    id, time0, time_to_event, event_indicator, event_type
    """
    cols = ["eid", "time0", "time_to_event", "event_indicator", "event_type"]
    return df[cols].copy()


def build_long_survival(df):
    """
    Long format for multi-state or cause-specific hazard models.
    One row per (id, event_type) pair.
    """

    long_rows = []
    for _, r in df.iterrows():
        for outcome in ["PD", "AD", "DEM"]:
            long_rows.append({
                "eid": r["eid"],
                "time0": r["time0"],
                "event_type": outcome,
                "event_indicator": 1 if r["event_type"] == outcome else 0,
                "time_to_event": r["time_to_event"]
            })

    return pd.DataFrame(long_rows)
