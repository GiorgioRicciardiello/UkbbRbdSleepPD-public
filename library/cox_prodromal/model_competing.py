"""
Model 4 -- Competing Risk Analysis.

Provides:
1. Aalen-Johansen cumulative incidence functions (CIF) per group
2. CIF vs 1-KM comparison to quantify competing risk bias
3. Cause-specific Cox (standard Cox treating competing events as censored)

Competing events include cross-outcome neurological diagnoses and
all-cause death (subjects who die before diagnosis can no longer
develop the primary condition).
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter, CoxPHFitter, KaplanMeierFitter

from library.cox_prodromal.categorical_ref import pick_reference_category
from library.cox_prodromal.cox_config import (
    ABSOLUTE_RISK_TIMEPOINTS,
    MIN_EVENTS_FOR_MODEL,
    RIDGE_PENALIZER,
)


def encode_competing_events(
    df: pd.DataFrame,
    primary_outcome: str,
    competing_outcomes: List[str],
) -> Tuple[pd.Series, pd.Series]:
    """
    Construct multi-state event indicator for competing risk analysis.

    Uses the earliest event across primary and competing outcomes.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``{outcome}_surv_time`` and ``{outcome}_surv_event``
        columns for primary and each competing outcome.
    primary_outcome : str
        Primary outcome identifier.
    competing_outcomes : list[str]
        Competing outcome identifiers.

    Returns
    -------
    durations : pd.Series
        Follow-up time (minimum across all event/censor times).
    event_indicator : pd.Series
        0 = censored, 1 = primary event, 2 = competing event.
    """
    time_col = "time"  # already renamed by select_survival_dataset
    event_col = "event"

    # Start with the primary outcome's time and event
    durations = df[time_col].copy()
    event_indicator = df[event_col].copy().astype(int)

    # For each competing outcome, check if it happened earlier
    for i, comp_out in enumerate(competing_outcomes):
        comp_time_col = f"{comp_out}_surv_time"
        comp_event_col = f"{comp_out}_surv_event"

        if comp_time_col not in df.columns or comp_event_col not in df.columns:
            continue

        comp_time = pd.to_numeric(df[comp_time_col], errors="coerce") / 365.25
        comp_event = pd.to_numeric(df[comp_event_col], errors="coerce")

        for idx in df.index:
            ct = comp_time.get(idx, np.nan)
            ce = comp_event.get(idx, 0)
            dt = durations.get(idx, np.nan)
            de = event_indicator.get(idx, 0)

            if pd.isna(ct) or ce != 1:
                continue

            # Competing event happened
            if de == 1:
                # Both events: use whichever happened first
                if ct < dt:
                    durations.at[idx] = ct
                    event_indicator.at[idx] = 2
            else:
                # Only competing event (primary was censored)
                if ct < dt:
                    durations.at[idx] = ct
                    event_indicator.at[idx] = 2

    return durations, event_indicator


def fit_aalen_johansen_cif(
    df: pd.DataFrame,
    time_col: str,
    event_indicator_col: str,
    group_col: str,
    event_of_interest: int = 1,
    timepoints: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Aalen-Johansen CIF per group, accounting for competing events.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``time_col``, ``event_indicator_col``, ``group_col``.
    event_indicator_col : str
        Column with 0=censored, 1=primary, 2=competing.
    group_col : str
        Grouping variable (e.g. RBD risk group).
    event_of_interest : int
        Which event to compute CIF for (default 1 = primary).
    timepoints : list[float], optional
        Fixed timepoints for CIF extraction.

    Returns
    -------
    dict
        Keys: cif_by_group (dict of DataFrames), cif_at_t0 (dict of dicts),
        n_primary, n_competing, n_censored.
    """
    timepoints = timepoints or ABSOLUTE_RISK_TIMEPOINTS

    df_clean = df.dropna(subset=[time_col, event_indicator_col, group_col])
    groups = sorted(df_clean[group_col].astype(str).unique())

    cif_by_group: Dict[str, pd.DataFrame] = {}
    cif_at_t0: Dict[str, Dict[str, float]] = {}

    n_primary = int((df_clean[event_indicator_col] == 1).sum())
    n_competing = int((df_clean[event_indicator_col] == 2).sum())
    n_censored = int((df_clean[event_indicator_col] == 0).sum())

    for grp in groups:
        mask = df_clean[group_col].astype(str) == grp
        sub = df_clean[mask]
        if len(sub) < 5:
            continue

        ajf = AalenJohansenFitter()
        try:
            ajf.fit(
                sub[time_col],
                sub[event_indicator_col].astype(int),
                event_of_interest=event_of_interest,
            )
        except Exception as exc:
            warnings.warn(f"AJ fit failed for group {grp}: {exc}")
            continue

        cif_df = ajf.cumulative_density_.copy()
        cif_by_group[grp] = cif_df

        # Extract CIF at fixed timepoints
        tp_vals: Dict[str, float] = {}
        for t in timepoints:
            cif_at_t = cif_df[cif_df.index <= t]
            if cif_at_t.empty:
                tp_vals[f"CIF_{t:.0f}y_pct"] = np.nan
            else:
                tp_vals[f"CIF_{t:.0f}y_pct"] = round(
                    float(cif_at_t.iloc[-1].values[0]) * 100, 2
                )
        cif_at_t0[grp] = tp_vals

    return {
        "cif_by_group": cif_by_group,
        "cif_at_t0": cif_at_t0,
        "n_primary": n_primary,
        "n_competing": n_competing,
        "n_censored": n_censored,
    }


def compare_cif_vs_km(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    event_indicator_col: str,
    group_col: str,
    timepoints: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Compare CIF (Aalen-Johansen) vs 1-KM to quantify competing risk bias.

    The difference CIF_AJ - CIF_KM shows how much standard KM
    overestimates the cumulative incidence when competing events exist.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain both binary event (for KM) and multi-state event
        (for AJ) columns.
    event_col : str
        Binary event indicator (0/1) for standard KM.
    event_indicator_col : str
        Multi-state indicator (0/1/2) for AJ.
    group_col : str
        Grouping variable.
    timepoints : list[float]
        Fixed timepoints for comparison.

    Returns
    -------
    pd.DataFrame
        Columns: group, timepoint, CIF_AJ_pct, CIF_KM_pct, difference_pct.
    """
    timepoints = timepoints or ABSOLUTE_RISK_TIMEPOINTS
    df_clean = df.dropna(
        subset=[time_col, event_col, event_indicator_col, group_col]
    )
    groups = sorted(df_clean[group_col].astype(str).unique())

    rows = []
    for grp in groups:
        mask = df_clean[group_col].astype(str) == grp
        sub = df_clean[mask]
        if len(sub) < 5:
            continue

        # KM cumulative incidence (1 - S(t))
        kmf = KaplanMeierFitter()
        kmf.fit(sub[time_col], sub[event_col])
        sf_km = kmf.survival_function_

        # AJ cumulative incidence
        ajf = AalenJohansenFitter()
        try:
            ajf.fit(
                sub[time_col],
                sub[event_indicator_col].astype(int),
                event_of_interest=1,
            )
            cif_aj = ajf.cumulative_density_
        except Exception:
            continue

        for t in timepoints:
            # KM
            sf_at_t = sf_km[sf_km.index <= t]
            cif_km = (
                (1 - float(sf_at_t.iloc[-1].values[0])) * 100
                if not sf_at_t.empty else np.nan
            )
            # AJ
            cif_at_t = cif_aj[cif_aj.index <= t]
            cif_aj_val = (
                float(cif_at_t.iloc[-1].values[0]) * 100
                if not cif_at_t.empty else np.nan
            )

            rows.append({
                "group": grp,
                "timepoint": t,
                "CIF_AJ_pct": round(cif_aj_val, 2) if pd.notna(cif_aj_val) else np.nan,
                "CIF_KM_pct": round(cif_km, 2) if pd.notna(cif_km) else np.nan,
                "difference_pct": (
                    round(cif_km - cif_aj_val, 3)
                    if pd.notna(cif_km) and pd.notna(cif_aj_val) else np.nan
                ),
            })

    return pd.DataFrame(rows)


def fit_cause_specific_cox(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_col: str,
    covariates: List[str],
    penalizer: float = RIDGE_PENALIZER,
) -> Optional[Dict[str, Any]]:
    """
    Cause-specific Cox: treat competing events as censored.

    This is methodologically equivalent to the standard Cox in Models 0-3
    but makes the competing-risk framing explicit.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset.
    event_col : str
        Binary event indicator (primary event = 1, else 0).
    rbd_col : str
        Categorical RBD risk group.
    covariates : list[str]
        Adjustment covariates.
    penalizer : float
        Ridge penalizer.

    Returns
    -------
    dict or None
        Keys: summary, c_index, N, events.
    """
    cols = [time_col, event_col, rbd_col] + covariates
    df_mod = df[cols].dropna().copy()
    if df_mod[event_col].sum() < MIN_EVENTS_FOR_MODEL:
        return None

    dum = pd.get_dummies(df_mod[rbd_col], prefix="rbd", drop_first=False)
    rbd_ref = pick_reference_category(dum.columns.tolist())
    dum = dum.drop(columns=[rbd_ref])
    X = pd.concat(
        [df_mod[[time_col, event_col]].reset_index(drop=True),
         dum.reset_index(drop=True),
         df_mod[covariates].reset_index(drop=True)],
        axis=1,
    )

    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(X, duration_col=time_col, event_col=event_col, robust=False)
    except Exception as exc:
        warnings.warn(f"Cause-specific Cox failed: {exc}")
        return None

    summary = cph.summary.reset_index().copy()
    summary.rename(columns={"index": "covariate"}, inplace=True)

    return {
        "summary": summary,
        "c_index": cph.concordance_index_,
        "N": len(X),
        "events": int(X[event_col].sum()),
    }
