
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

import matplotlib.pyplot as plt

# -----------------------------
# Modeling with PH diagnostics
# -----------------------------
@dataclass
class CoxResult:
    hr: float
    lci: float
    uci: float
    p: float
    ph_violation: bool
    ph_p_min: Optional[float]
    model_type: str  # "standard" or "time_varying_exposure"


def fit_cox_with_ph_handling_ngroups(
    df: pd.DataFrame,
    exposure_col: str,
    covariates: List[str],
    ph_p_threshold: float = 0.05,
    ref_group: int = 0
) -> Tuple[CoxPHFitter, Dict[str, CoxResult], Dict[str, float]]:
    """
    Fit Cox PH model with robust SEs for an exposure with N categorical groups.

    - ref_group is the reference (default = 0)
    - Estimates HRs for all other groups vs reference
    - Checks PH per contrast
    - Adds time-varying effects per violating contrast

    Returns
    -------
    cph : CoxPHFitter
    results : dict[str, CoxResult]
        Keys like: 'group_2_vs_0'
    ph_stats : dict[str, float]
        PH p-values per term
    """

    # -----------------------------
    # Prepare data
    # -----------------------------
    base_cols = ["time", "event", exposure_col] + covariates
    df_fit = df[base_cols].dropna().copy()

    groups = sorted(df_fit[exposure_col].unique())
    if ref_group not in groups:
        raise ValueError(f"Reference group {ref_group} not present in exposure.")

    non_ref_groups = [g for g in groups if g != ref_group]

    if len(non_ref_groups) == 0:
        raise ValueError("Exposure has only the reference group.")

    # -----------------------------
    # Create dummy variables
    # -----------------------------
    dummy_terms = []
    for g in non_ref_groups:
        col = f"exp_grp_{g}"
        df_fit[col] = (df_fit[exposure_col] == g).astype(float)
        dummy_terms.append(col)

    model_terms = dummy_terms + covariates

    # -----------------------------
    # Fit base Cox
    # -----------------------------
    cph = CoxPHFitter()
    cph.fit(
        df_fit[["time", "event"] + model_terms],
        duration_col="time",
        event_col="event",
        robust=True
    )

    # -----------------------------
    # PH testing
    # -----------------------------
    from lifelines.statistics import proportional_hazard_test

    ph_test = proportional_hazard_test(
        cph,
        df_fit[["time", "event"] + model_terms],
        time_transform="rank"
    )

    pvals = ph_test.summary["p"].to_dict()
    ph_stats = {f"ph_p_{k}": float(v) for k, v in pvals.items()}

    violating_terms = [
        term for term in dummy_terms
        if pvals.get(term, 1.0) < ph_p_threshold
    ]

    # -----------------------------
    # Time-varying refit (if needed)
    # -----------------------------
    model_type = "standard"

    if violating_terms:
        model_type = "time_varying"

        df_fit["log_time"] = np.log(df_fit["time"].clip(lower=1e-6))
        tv_terms = []

        for term in violating_terms:
            tv = f"{term}__x__log_time"
            df_fit[tv] = df_fit[term] * df_fit["log_time"]
            tv_terms.append(tv)

        cph = CoxPHFitter()
        cph.fit(
            df_fit[["time", "event"] + model_terms + tv_terms],
            duration_col="time",
            event_col="event",
            robust=True
        )

    # -----------------------------
    # Extract HRs
    # -----------------------------
    results: Dict[str, CoxResult] = {}

    t_ref = float(df_fit["time"].median())
    log_t_ref = float(np.log(max(t_ref, 1e-6)))

    for g in non_ref_groups:
        term = f"exp_grp_{g}"
        label = f"group_{g}_vs_{ref_group}"

        if model_type == "standard" or f"{term}__x__log_time" not in cph.params_:
            s = cph.summary.loc[term]
            results[label] = CoxResult(
                hr=float(s["exp(coef)"]),
                lci=float(s["exp(coef) lower 95%"]),
                uci=float(s["exp(coef) upper 95%"]),
                p=float(s["p"]),
                ph_violation=(term in violating_terms),
                ph_p_min=min(pvals.values()),
                model_type=model_type
            )
        else:
            # Time-varying HR at reference time
            b1 = float(cph.params_.loc[term])
            b2 = float(cph.params_.loc[f"{term}__x__log_time"])
            log_hr = b1 + b2 * log_t_ref
            hr = float(np.exp(log_hr))

            cov = cph.variance_matrix_.loc[
                [term, f"{term}__x__log_time"],
                [term, f"{term}__x__log_time"]
            ].values

            grad = np.array([1.0, log_t_ref])
            var = grad.T @ cov @ grad
            se = np.sqrt(max(var, 0.0))

            z = 1.96
            results[label] = CoxResult(
                hr=hr,
                lci=float(np.exp(log_hr - z * se)),
                uci=float(np.exp(log_hr + z * se)),
                p=float(cph.summary.loc[term, "p"]),
                ph_violation=True,
                ph_p_min=min(pvals.values()),
                model_type=model_type
            )

    return cph, results, ph_stats
