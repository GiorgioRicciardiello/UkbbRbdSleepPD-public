import pandas as pd
from typing import Any, Tuple, Dict

# --- Field dictionaries ---
# OUTCOME 1: PD specific
parkinsonism_fields = {
    # Parkinsonism and PD-related
    # 42030: "Date of all cause parkinsonism report",
    # 42031: "Source of all cause parkinsonism report",

    42032: "Date of Parkinson's disease report",    # CONFIRMED
    42033: "Source of Parkinson's disease report",  # CONFIRMED

    # 42034: "Date of progressive supranuclear palsy report",
    # 42035: "Source of progressive supranuclear palsy report",
    # 42036: "Date of multiple system atrophy report",
    # 42037: "Source of multiple system atrophy report",

    # Alzheimer?s and dementia subtypes
    42018: "Date of all cause dementia",           # CONFIRMED
    42019: "Source of all cause dementia report",  # CONFIRMED, all cause dementia, then the date of this report

    42020: "Date of alzheimer's disease report",    # CONFIRMED
    42021: "Source of alzheimer's disease report",  # CONFIRMED


    # 42022: "	Date of vascular dementia report",
    # 42023: "	Source of vascular dementia report",
    # 42024: "	Date of frontotemporal dementia report",
    # 42025: "Source of frontotemporal dementia report",
}

# Outcome 2: other dementia: PD AND AD


# since we are only using sleep feature, should we re-define outcome 2?


def _flag_from_fields(df: pd.DataFrame, field_dict: Dict[int, str], flag_name: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Generic function to flag cases from a dictionary of UKB fields.
    """
    # Map field IDs -> actual dataframe column names (baseline instance assumed: -0.0)
    col_map = {fid: f"{fid}-0.0" for fid in field_dict if f"{fid}-0.0" in df.columns}

    # Boolean flags per field (non-null entries)
    field_flags = {fid: df[col].notna() for fid, col in col_map.items()}

    # Combine all fields to a single flag
    if field_flags:
        df[flag_name] = pd.DataFrame(field_flags).any(axis=1)
        df[flag_name] = df[flag_name].astype(int)
    else:
        df[flag_name] = False

    # Count per field + total
    counts = {field_dict[fid]: field_flags[fid].sum() for fid in field_flags}
    counts[f"Total flagged {flag_name}"] = df[flag_name].sum()

    return df, counts


def flag_parkinsons(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Flag Parkinson?s / parkinsonism cases and return counts per field."""
    return _flag_from_fields(df, parkinsonism_fields, "flag_pd")


def flag_lewybody(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Flag Lewy body dementia cases and return counts per field."""
    return _flag_from_fields(df, lewybody_fields, "flag_lbd")
