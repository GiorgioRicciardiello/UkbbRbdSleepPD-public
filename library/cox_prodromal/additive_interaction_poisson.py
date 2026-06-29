"""
Poisson regression-based RERI sensitivity analysis.

Poisson regression with log(time) offset provides more stable estimates
of RERI in sparse cells compared to the Cox PH 4-group approach.  The
incidence rate ratio (IRR) from Poisson regression approximates the HR
when event rates are low (rare disease assumption holds for PD at ~0.4%).

Reference: Zou (2004), Am J Epidemiol. Modified Poisson for binary outcomes.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson

from library.cox_prodromal.cox_config import MIN_EVENTS_FOR_MODEL, RIDGE_PENALIZER


def compute_poisson_reri(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_binary_col: str,
    prod_binary_col: str,
    covariates: List[str],
) -> Optional[Dict[str, Any]]:
    """Compute RERI via Poisson regression with log(time) offset.

    Fits: E[events] = exp(b0 + b10*grp10 + b01*grp01 + b11*grp11 + bX*X + log(time))

    IRR_ij = exp(b_ij).  RERI = IRR_11 - IRR_10 - IRR_01 + 1.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset.
    time_col : str
        Follow-up time (years).  Used as offset: log(time).
    event_col : str
        Binary event indicator (0/1).
    rbd_binary_col : str
        Binary RBD (0/1).
    prod_binary_col : str
        Binary prodromal (0/1).
    covariates : list[str]
        Adjustment covariates.

    Returns
    -------
    dict or None
        Keys: irr_10, irr_01, irr_11, reri, ap, si, plus SEs and CIs,
        cell counts, and sparse flag.
    """
    cols = [time_col, event_col, rbd_binary_col, prod_binary_col] + covariates
    df_mod = df[cols].dropna().copy()

    if df_mod[event_col].sum() < MIN_EVENTS_FOR_MODEL:
        return None

    # Require positive follow-up for log offset
    df_mod = df_mod[df_mod[time_col] > 0].copy()
    if df_mod.empty:
        return None

    r = df_mod[rbd_binary_col].astype(int)
    p = df_mod[prod_binary_col].astype(int)

    df_mod["grp_10"] = ((r == 1) & (p == 0)).astype(int)
    df_mod["grp_01"] = ((r == 0) & (p == 1)).astype(int)
    df_mod["grp_11"] = ((r == 1) & (p == 1)).astype(int)
    df_mod["log_time"] = np.log(df_mod[time_col])

    # Cell counts
    ev = df_mod[event_col]
    grp_00_mask = (r == 0) & (p == 0)
    grp_10_mask = (r == 1) & (p == 0)
    grp_01_mask = (r == 0) & (p == 1)
    grp_11_mask = (r == 1) & (p == 1)

    cell_counts = {
        "n_00": int(grp_00_mask.sum()), "events_00": int(ev[grp_00_mask].sum()),
        "n_10": int(grp_10_mask.sum()), "events_10": int(ev[grp_10_mask].sum()),
        "n_01": int(grp_01_mask.sum()), "events_01": int(ev[grp_01_mask].sum()),
        "n_11": int(grp_11_mask.sum()), "events_11": int(ev[grp_11_mask].sum()),
    }
    sparse_cell = any(
        cell_counts[f"events_{g}"] < 10 for g in ("00", "10", "01", "11")
    )

    # Design matrix — cast to float64 explicitly to prevent GLM receiving
    # object-dtype columns when covariates have mixed types (e.g. categorical
    # codes stored as object).
    exog_cols = ["grp_10", "grp_01", "grp_11"] + covariates
    X = df_mod[exog_cols].copy().astype(np.float64)
    X.insert(0, "const", 1.0)

    try:
        model = GLM(
            df_mod[event_col],
            X,
            family=Poisson(),
            offset=df_mod["log_time"],
        )
        result = model.fit(disp=False)
    except Exception as exc:
        warnings.warn(f"Poisson RERI failed: {exc}")
        return None

    # IRRs = exp(coefficients)
    irr_10 = np.exp(result.params["grp_10"])
    irr_01 = np.exp(result.params["grp_01"])
    irr_11 = np.exp(result.params["grp_11"])

    # CIs via delta method (Wald)
    se_10 = result.bse["grp_10"]
    se_01 = result.bse["grp_01"]
    se_11 = result.bse["grp_11"]

    def _irr_ci(coef: float, se: float) -> tuple:
        lo = np.exp(coef - 1.96 * se)
        hi = np.exp(coef + 1.96 * se)
        return round(float(lo), 4), round(float(hi), 4)

    irr_10_ci = _irr_ci(result.params["grp_10"], se_10)
    irr_01_ci = _irr_ci(result.params["grp_01"], se_01)
    irr_11_ci = _irr_ci(result.params["grp_11"], se_11)

    # RERI from IRRs
    reri = float(irr_11 - irr_10 - irr_01 + 1.0)
    ap = reri / float(irr_11) if irr_11 != 0 else np.nan
    denom = (float(irr_10) - 1.0) + (float(irr_01) - 1.0)
    si = (float(irr_11) - 1.0) / denom if abs(denom) > 1e-10 else np.nan

    return {
        "method": "poisson",
        "irr_10": round(float(irr_10), 4),
        "irr_10_lci": irr_10_ci[0], "irr_10_uci": irr_10_ci[1],
        "irr_01": round(float(irr_01), 4),
        "irr_01_lci": irr_01_ci[0], "irr_01_uci": irr_01_ci[1],
        "irr_11": round(float(irr_11), 4),
        "irr_11_lci": irr_11_ci[0], "irr_11_uci": irr_11_ci[1],
        "reri": round(reri, 4),
        "ap": round(ap, 4),
        "si": round(si, 4),
        "N": len(df_mod),
        "events": int(df_mod[event_col].sum()),
        **cell_counts,
        "sparse_cell_warning": sparse_cell,
    }
