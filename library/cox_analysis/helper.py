from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from tabulate import tabulate

from config.config import config, outcomes
from library.risk.risk_helpers import get_clean_risk_data, make_subject_level
from library.risk.survival_analysis import METHOD_TO_RISK_SUFFIX
from library.cox_analysis.select_risk_groups import make_high_vs_low
import matplotlib.pyplot as plt
import seaborn as sns


def validate_covariates_exist(
    df: pd.DataFrame,
    covs: List[str],
    min_unique: int = 2,
    min_variance: float | None = None
) -> List[str]:
    """
    Validate covariates for Cox modeling.

    Drops covariates that:
    - do not exist in df
    - are constant (n_unique < min_unique)
    - have near-zero variance (optional)

    Parameters
    ----------
    df : pd.DataFrame
        Analysis dataframe.
    covs : list[str]
        Candidate covariates.
    min_unique : int
        Minimum number of unique non-null values required.
        Default = 2 (drop constants).
    min_variance : float | None
        If provided, drop covariates with variance < min_variance.

    Returns
    -------
    list[str]
        Valid covariates to include in the model.
    """

    valid_covs = []
    dropped = {}

    for c in covs:
        if c not in df.columns:
            dropped[c] = "missing"
            continue

        s = df[c].dropna()

        # Empty after NA drop
        if s.empty:
            dropped[c] = "all missing"
            continue

        # Constant or near-constant
        n_unique = s.nunique()
        if n_unique < min_unique:
            dropped[c] = f"constant (n_unique={n_unique})"
            continue

        # Near-zero variance (optional, numeric only)
        if min_variance is not None and pd.api.types.is_numeric_dtype(s):
            var = float(np.var(s))
            if var < min_variance:
                dropped[c] = f"near-zero variance (var={var:.2e})"
                continue

        valid_covs.append(c)

    if dropped:
        msg = "; ".join([f"{k}: {v}" for k, v in dropped.items()])
        warnings.warn(f"Dropped covariates: {msg}")

    return valid_covs



def consort_counts(df: pd.DataFrame) -> dict:
    """
    CONSORT-style counts for the final analysis set.
    """
    return {
        "N_analysis": df.shape[0],
        "N_events": int(df["event"].sum()),
        "N_censored": int((df["event"] == 0).sum()),
        "median_followup_years": float(df["time"].median()),
    }


def log(msg: str, level: str = "INFO"):
    print(f"[{level}] {msg}")
