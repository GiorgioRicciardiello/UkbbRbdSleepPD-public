"""
Secondary Analysis — RBD Score Predicts Incident Prodromal Symptoms
====================================================================

Research question
-----------------
Does the actigraphy-derived RBD probability score predict the *future onset*
of prodromal synucleinopathy symptoms (constipation, depression, orthostatic
hypotension, erectile dysfunction)?

Temporal structure
------------------
RBD score is measured at the actigraphy baseline (``wear_time_start``).
Prodromal symptoms are ascertained from post-baseline HES ICD-10 records
and self-reported medication prescriptions.  The analysis excludes subjects
who already had the symptom before baseline (prevalent prodromal cases),
ensuring correct temporal ordering: exposure (RBD) precedes outcome
(incident prodromal symptom).

Model specification
-------------------
Per-marker Cox PH model:

    h_k(t) = h_{0k}(t) * exp(beta_1 * rbd_prob_z + beta_2 * age
                              + beta_3 * sex + beta_4 * BMI)

where:
- k indexes the prodromal marker (constipation, depression, etc.)
- rbd_prob_z = (rbd_prob - mean) / SD  (standardised)
- t = time from actigraphy baseline to first post-baseline evidence
- h_{0k}(t) is the marker-specific baseline hazard

Additionally, a prodromal burden (count) model:

    E[Y_i] = exp(beta_1 * rbd_prob_z + beta_2 * age
                 + beta_3 * sex + beta_4 * BMI)

where Y_i = number of incident prodromal markers for subject i.
Fitted via Poisson GLM (or negative-binomial if overdispersed).

At-risk population (per marker)
-------------------------------
- Exclude: ``prodromal_{marker} == 1`` (pre-baseline evidence)
- Event = 1: first HES or medication date for marker >= ``wear_time_start``
  AND <= ``censor_date``
- Event time: (event_date - wear_time_start) / 365.25  [years]
- Censored: (censor_date - wear_time_start) / 365.25

Markers analysed
----------------
Only markers with >= MIN_EVENTS_FOR_MODEL incident events are modelled.
Markers with insufficient events are reported as such in the output.

References
----------
- Schrag A et al. "Identifying prodromal symptoms at high specificity for
  Parkinson's disease." Parkinsonism Relat Disord. 2023;115:105834.
  PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10556459/
- NAPS Consortium. "Baseline characteristics of the North American
  prodromal Synucleinopathy cohort." Mov Disord. 2023;38(6):1030.
  PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10109527/
- Berg D et al. "MDS research criteria for prodromal Parkinson's disease."
  Mov Disord. 2015;30(12):1600-1611.
  PubMed: https://pubmed.ncbi.nlm.nih.gov/26474317/

Output
------
Tables:
- ``secondary_per_marker_cox.csv`` — HR per SD, 95% CI, p, C-index per marker
- ``secondary_prodromal_burden.csv`` — Poisson/NB IRR per SD
- ``secondary_feasibility.csv`` — at-risk N, incident events per marker
- ``secondary_ph_diagnostics.csv`` — Schoenfeld PH tests
- ``secondary_c_index.csv`` — discrimination metrics
- ``secondary_fdr_corrected.csv`` — FDR-corrected p-values across markers

Figures:
- ``KM_{marker}_2g.png`` — KM by RBD 2-group (Low/High)
- ``KM_{marker}_3g.png`` — KM by RBD 3-group (Low/Mid/High)
- ``forest_rbd_predicts_prodromal.png`` — Forest plot of HRs

Author: Giorgio Ricciardello
Date: 2026-03-09
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
import statsmodels.api as sm

from config.config import config
from library.risk.risk_helpers import get_clean_risk_data, make_subject_level
from library.column_registry import col_risk_group_agnostic, METHOD_TO_RISK_SUFFIX
from library.cox_prodromal.cox_config import (
    BASE_COVARIATES,
    MIN_EVENTS_FOR_MODEL,
    METHODS,
    PRIMARY_METHOD,
    RIDGE_PENALIZER,
    RISK_PALETTE,
)
from library.cox_prodromal.data_prep import (
    build_extended_covariates,
    load_prodromal_dataset,
)
from library.cox_prodromal.diagnostics import apply_fdr, run_ph_test
from library.cox_prodromal.plotting import get_rbd_group_color
from library.cox_prodromal.utils import save_table


# ============================================================================
# CONFIGURATION
# ============================================================================

# Marker → (prodromal_flag_col, hes_date_col, med_date_col_or_None, display_label)
# Only markers with feasible event counts are included.
# Dream enactment, hyposmia, anosmia, anxiety excluded (insufficient events).
MARKER_SPEC: Dict[str, Tuple[str, str, Optional[str], str]] = {
    "constipation": (
        "prodromal_constipation_bl",
        "constipation_hes_date",
        "med_laxatives_date",
        "Constipation",
    ),
    "depression": (
        "prodromal_depression_bl",
        "depression_hes_date",
        "med_depression_date",
        "Depression",
    ),
    "orthostatic": (
        "prodromal_orthostatic_bl",
        "Orthostatic_hes_date",
        "med_orthostatic_hypotension_date",
        "Orthostatic Hypotension",
    ),
    "erectile_dysfunction": (
        "prodromal_erectile_dysfunction_bl",
        "erectile_dysfunction_hes_date",
        "med_pde5_inhibitors_date",
        "Erectile Dysfunction",
    ),
}

# Markers excluded due to insufficient incident events (documented)
EXCLUDED_MARKERS: Dict[str, str] = {
    "dream_enactment": "0 incident HES events",
    "hyposmia": "0 incident HES events",
    "anosmia": "39 incident events (insufficient for stable estimation)",
    "anxiety": "315 incident events (marginal; excluded for robustness)",
}

STRATIFICATION_OUTCOME: str = "outcome_1a_pd_only"
SEED: int = 42


# ============================================================================
# DATA CONSTRUCTION
# ============================================================================

@dataclass(frozen=True)
class MarkerSurvivalData:
    """Survival dataset for a single prodromal marker outcome."""
    marker: str
    label: str
    df: pd.DataFrame        # columns: time, event, rbd_prob, covariates, risk_group cols
    n_at_risk: int
    n_events: int
    n_prevalent_excluded: int


def build_marker_survival_dataset(
    df: pd.DataFrame,
    marker: str,
    prodromal_col: str,
    hes_date_col: str,
    med_date_col: Optional[str],
    label: str,
    baseline_col: str = "wear_time_start",
    censor_col: str = "censor_date",
) -> Optional[MarkerSurvivalData]:
    """
    Construct a survival dataset where the event is incident prodromal onset.

    Parameters
    ----------
    df : pd.DataFrame
        Full cohort with prodromal flags, HES dates, medication dates.
    marker : str
        Short marker name (e.g. 'constipation').
    prodromal_col : str
        Pre-baseline binary flag column (1 = prevalent, excluded).
    hes_date_col : str
        Column with earliest HES diagnosis date for this condition.
    med_date_col : str or None
        Column with earliest medication date, or None if no med mapping.
    label : str
        Human-readable label for reports.
    baseline_col : str
        Actigraphy baseline date column.
    censor_col : str
        Administrative censor date column.

    Returns
    -------
    MarkerSurvivalData or None
        None if insufficient events for modelling.
    """
    baseline = pd.to_datetime(df[baseline_col], errors="coerce")
    censor = pd.to_datetime(df[censor_col], errors="coerce")
    prevalent = pd.to_numeric(df[prodromal_col], errors="coerce").fillna(0).astype(int)

    # At-risk: no pre-baseline evidence
    at_risk_mask = prevalent == 0
    n_prevalent = int((~at_risk_mask).sum())

    # Post-baseline HES event
    hes_date = pd.to_datetime(df[hes_date_col], errors="coerce")
    hes_post = hes_date.where(at_risk_mask & (hes_date >= baseline) & (hes_date <= censor))

    # Post-baseline medication event
    if med_date_col is not None and med_date_col in df.columns:
        med_date = pd.to_datetime(df[med_date_col], errors="coerce")
        med_post = med_date.where(at_risk_mask & (med_date >= baseline) & (med_date <= censor))
    else:
        med_post = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    # Earliest post-baseline evidence (HES or medication)
    event_date = pd.concat([hes_post, med_post], axis=1).min(axis=1)
    event = (at_risk_mask & event_date.notna()).astype(int)

    # Time in years
    time_event = (event_date - baseline).dt.days / 365.25
    time_censor = (censor - baseline).dt.days / 365.25
    time = time_event.where(event == 1, time_censor)

    # Build output DataFrame
    df_surv = df.loc[at_risk_mask].copy()
    df_surv["time"] = time.loc[at_risk_mask].values
    df_surv["event"] = event.loc[at_risk_mask].values

    # Drop rows with invalid time
    df_surv = df_surv.dropna(subset=["time"])
    df_surv = df_surv[df_surv["time"] > 0].copy()

    n_events = int(df_surv["event"].sum())
    n_at_risk = len(df_surv)

    if n_events < MIN_EVENTS_FOR_MODEL:
        warnings.warn(
            f"Marker '{marker}': {n_events} events < {MIN_EVENTS_FOR_MODEL} "
            f"minimum — skipping."
        )
        return None

    return MarkerSurvivalData(
        marker=marker,
        label=label,
        df=df_surv,
        n_at_risk=n_at_risk,
        n_events=n_events,
        n_prevalent_excluded=n_prevalent,
    )


# ============================================================================
# MODEL FITTING
# ============================================================================

def fit_rbd_continuous_cox(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_col: str,
    covariates: List[str],
    penalizer: float = RIDGE_PENALIZER,
) -> Optional[Dict[str, Any]]:
    """
    Cox PH with continuous RBD score (standardised per-SD) as sole exposure.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset with time, event, rbd_prob, covariates.
    time_col : str
        Duration column (years).
    event_col : str
        Event indicator (0/1).
    rbd_col : str
        Continuous RBD probability column.
    covariates : list[str]
        Adjustment covariates.
    penalizer : float
        Ridge penalizer for CoxPHFitter.

    Returns
    -------
    dict or None
        Keys: hr_per_sd, hr_lci, hr_uci, p, c_index, c_index_null,
        c_index_incremental, ph_df, N, events, rbd_mean, rbd_sd, summary.
    """
    cols = [time_col, event_col, rbd_col] + covariates
    df_mod = df[cols].dropna().copy()

    if df_mod[event_col].sum() < MIN_EVENTS_FOR_MODEL:
        return None

    # Standardise RBD score
    rbd_mean = float(df_mod[rbd_col].mean())
    rbd_sd = float(df_mod[rbd_col].std())
    if rbd_sd == 0:
        return None
    df_mod["rbd_z"] = (df_mod[rbd_col] - rbd_mean) / rbd_sd

    # Null model (covariates only)
    cph_null = CoxPHFitter(penalizer=penalizer)
    X_null = df_mod[[time_col, event_col] + covariates]
    try:
        cph_null.fit(X_null, duration_col=time_col, event_col=event_col)
        c_null = float(cph_null.concordance_index_)
    except Exception:
        c_null = np.nan

    # Full model (rbd_z + covariates)
    cph = CoxPHFitter(penalizer=penalizer)
    X_full = df_mod[[time_col, event_col, "rbd_z"] + covariates]
    try:
        cph.fit(X_full, duration_col=time_col, event_col=event_col)
    except Exception as exc:
        warnings.warn(f"Cox fit failed: {exc}")
        return None

    summary = cph.summary.copy()
    summary["covariate"] = summary.index

    rbd_row = summary.loc["rbd_z"]
    hr = float(rbd_row["exp(coef)"])
    hr_lci = float(rbd_row["exp(coef) lower 95%"])
    hr_uci = float(rbd_row["exp(coef) upper 95%"])
    p_val = float(rbd_row["p"])
    c_full = float(cph.concordance_index_)

    # PH test
    try:
        ph_df = run_ph_test(cph, X_full)
    except Exception:
        ph_df = pd.DataFrame()

    return {
        "hr_per_sd": round(hr, 4),
        "hr_lci": round(hr_lci, 4),
        "hr_uci": round(hr_uci, 4),
        "p": p_val,
        "c_index": round(c_full, 4),
        "c_index_null": round(c_null, 4),
        "c_index_incremental": round(c_full - c_null, 4) if np.isfinite(c_null) else np.nan,
        "ph_df": ph_df,
        "N": len(df_mod),
        "events": int(df_mod[event_col].sum()),
        "rbd_mean": round(rbd_mean, 6),
        "rbd_sd": round(rbd_sd, 6),
        "summary": summary,
    }


def fit_prodromal_burden_model(
    df: pd.DataFrame,
    count_col: str,
    rbd_col: str,
    covariates: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Poisson / negative-binomial regression of prodromal burden on RBD score.

    Parameters
    ----------
    df : pd.DataFrame
        Subject-level data with count of incident prodromal events.
    count_col : str
        Column with integer count of incident prodromal markers.
    rbd_col : str
        Continuous RBD probability column.
    covariates : list[str]
        Adjustment covariates.

    Returns
    -------
    dict or None
        Keys: irr_per_sd, irr_lci, irr_uci, p, model_type, overdispersion,
        N, mean_count, summary.
    """
    cols = [count_col, rbd_col] + covariates
    df_mod = df[cols].dropna().copy()

    if len(df_mod) < 50:
        return None

    # Standardise RBD
    rbd_mean = float(df_mod[rbd_col].mean())
    rbd_sd = float(df_mod[rbd_col].std())
    if rbd_sd == 0:
        return None
    df_mod["rbd_z"] = (df_mod[rbd_col] - rbd_mean) / rbd_sd

    y = df_mod[count_col].astype(int)
    X = sm.add_constant(df_mod[["rbd_z"] + covariates])

    # Fit Poisson first
    try:
        poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    except Exception as exc:
        warnings.warn(f"Poisson GLM failed: {exc}")
        return None

    # Check overdispersion (deviance / df_resid > 1.5 → use NB)
    overdispersion = poisson_model.deviance / poisson_model.df_resid
    model_type = "poisson"
    final_model = poisson_model

    if overdispersion > 1.5:
        try:
            nb_model = sm.GLM(
                y, X, family=sm.families.NegativeBinomial()
            ).fit()
            final_model = nb_model
            model_type = "negative_binomial"
        except Exception:
            pass  # fall back to Poisson

    # Extract RBD coefficient
    rbd_idx = list(X.columns).index("rbd_z")
    coef = final_model.params.iloc[rbd_idx]
    ci = final_model.conf_int().iloc[rbd_idx]
    p_val = float(final_model.pvalues.iloc[rbd_idx])

    return {
        "irr_per_sd": round(float(np.exp(coef)), 4),
        "irr_lci": round(float(np.exp(ci[0])), 4),
        "irr_uci": round(float(np.exp(ci[1])), 4),
        "p": p_val,
        "model_type": model_type,
        "overdispersion": round(overdispersion, 3),
        "N": len(df_mod),
        "mean_count": round(float(y.mean()), 3),
        "rbd_mean": round(rbd_mean, 6),
        "rbd_sd": round(rbd_sd, 6),
        "summary": final_model.summary2().tables[1],
    }


# ============================================================================
# KM PLOTTING
# ============================================================================

def plot_km_by_rbd_group(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    rbd_col: str,
    marker_label: str,
    method_label: str,
    save_path: str,
) -> Dict[str, Any]:
    """
    Plot Kaplan-Meier curves for incident prodromal marker by RBD risk group.

    Parameters
    ----------
    df : pd.DataFrame
        Survival dataset with time, event, and RBD risk group column.
    time_col : str
        Duration column (years).
    event_col : str
        Event indicator (0/1).
    rbd_col : str
        Categorical RBD risk group column.
    marker_label : str
        Human-readable marker name for title.
    method_label : str
        Method label (e.g. 'PERCENTILE_2G').
    save_path : str
        Output file path.

    Returns
    -------
    dict
        Keys: logrank_p, N, events, groups.
    """
    df_plot = df.dropna(subset=[time_col, event_col, rbd_col]).copy()
    df_plot[rbd_col] = df_plot[rbd_col].astype(str)
    groups = sorted(
        [g for g in df_plot[rbd_col].unique() if g not in ("nan", "None", "")],
        key=lambda x: 0 if "low" in x.lower() else (1 if "mid" in x.lower() else 2),
    )

    if len(groups) < 2:
        return {"logrank_p": np.nan, "N": len(df_plot), "events": 0, "groups": groups}

    fig, ax = plt.subplots(figsize=(8, 5))

    kmf_dict: Dict[str, KaplanMeierFitter] = {}
    for grp in groups:
        mask = df_plot[rbd_col] == grp
        kmf = KaplanMeierFitter()
        kmf.fit(
            df_plot.loc[mask, time_col],
            event_observed=df_plot.loc[mask, event_col],
            label=grp,
        )
        kmf_dict[grp] = kmf

        color = get_rbd_group_color(grp)
        n_grp = int(mask.sum())
        n_ev = int(df_plot.loc[mask, event_col].sum())

        # Plot 1 - S(t) as cumulative incidence
        sf = kmf.survival_function_
        cif = 1 - sf
        ax.step(
            cif.index, cif.values.ravel(),
            where="post", color=color, linewidth=2,
            label=f"{grp} (N={n_grp:,}, events={n_ev:,})",
        )

    # Log-rank test
    try:
        lr = multivariate_logrank_test(
            df_plot[time_col], df_plot[rbd_col], df_plot[event_col],
        )
        logrank_p = float(lr.p_value)
    except Exception:
        logrank_p = np.nan

    p_str = f"p < 0.001" if logrank_p < 0.001 else f"p = {logrank_p:.3f}"
    ax.set_title(
        f"Incident {marker_label} by RBD Risk Group ({method_label})\n"
        f"Log-rank {p_str}",
        fontsize=12,
    )
    ax.set_xlabel("Time from actigraphy baseline (years)", fontsize=11)
    ax.set_ylabel("Cumulative incidence", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "logrank_p": logrank_p,
        "N": len(df_plot),
        "events": int(df_plot[event_col].sum()),
        "groups": groups,
    }


# ============================================================================
# FOREST PLOT
# ============================================================================

def plot_forest(
    results: List[Dict[str, Any]],
    save_path: str,
) -> None:
    """
    Forest plot of HR per SD of RBD score across prodromal markers.

    Parameters
    ----------
    results : list[dict]
        Each dict has: marker_label, hr_per_sd, hr_lci, hr_uci, p, p_fdr, events.
    save_path : str
        Output file path.
    """
    if not results:
        return

    n = len(results)
    fig, ax = plt.subplots(figsize=(8, max(3, 1.5 + 0.6 * n)))

    y_pos = list(range(n))
    labels = []

    for i, r in enumerate(results):
        hr = r["hr_per_sd"]
        lci = r["hr_lci"]
        uci = r["hr_uci"]
        p_fdr = r.get("p_fdr", r["p"])
        events = r["events"]

        color = "#B2182B" if lci > 1.0 else ("#2166AC" if uci < 1.0 else "#666666")

        ax.plot([lci, uci], [i, i], color=color, linewidth=2, solid_capstyle="round")
        ax.plot(hr, i, "o", color=color, markersize=8, zorder=5)

        p_str = "< 0.001" if p_fdr < 0.001 else f"{p_fdr:.3f}"
        labels.append(f"{r['marker_label']}  (n={events:,}, p_FDR={p_str})")

    ax.axvline(1.0, color="grey", linestyle="--", linewidth=1, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Hazard Ratio per SD of RBD Score (95% CI)", fontsize=11)
    ax.set_title(
        "RBD Score Predicts Incident Prodromal Symptoms\n"
        "(secondary analysis, FDR-corrected)",
        fontsize=12,
    )
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_rbd_predicts_prodromal(
    mode: str = "abk",
    methods: Optional[List[str]] = None,
    seed: int = SEED,
) -> None:
    """
    Run the full secondary analysis: RBD score predicts incident prodromal markers.

    Orchestration sequence:
    1. Load data, prepare covariates
    2. Build per-marker survival datasets
    3. Per-marker Cox PH (continuous RBD per-SD)
    4. KM plots by RBD risk group (2g and 3g)
    5. Prodromal burden (Poisson/NB) model
    6. FDR correction, save tables, forest plot

    Parameters
    ----------
    mode : str
        Data mode ('abk' or 'katarina').
    methods : list[str], optional
        RBD stratification methods (default: percentile_2g, percentile_3g).
    seed : int
        Random seed for reproducibility.
    """
    methods = methods or METHODS
    rng = np.random.default_rng(seed)

    # ── 1. Paths ────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    path_results = config["results"]["root"] / f"secondary_rbd_prodromal_{timestamp}"
    path_figures = path_results / "figures"
    path_results.mkdir(parents=True, exist_ok=True)
    path_figures.mkdir(parents=True, exist_ok=True)

    # ── 2. Load data ────────────────────────────────────────────────────
    print("[1/6] Loading data ...")
    thresholds, df_risk = load_prodromal_dataset(mode=mode)
    df_risk, extended_covariates = build_extended_covariates(df_risk, BASE_COVARIATES)

    # Ensure rbd_prob exists (make_subject_level renames it)
    col_irbd = "abk_rbd_score_mean"
    if "rbd_prob" not in df_risk.columns and col_irbd in df_risk.columns:
        df_risk["rbd_prob"] = pd.to_numeric(df_risk[col_irbd], errors="coerce")

    print(f"  Cohort N = {len(df_risk):,}")

    # ── 3. Build survival datasets per marker ───────────────────────────
    print("\n[2/6] Building per-marker survival datasets ...")
    marker_datasets: Dict[str, MarkerSurvivalData] = {}
    feasibility_rows: List[Dict] = []

    for marker, (prod_col, hes_date_col, med_date_col, label) in MARKER_SPEC.items():
        msd = build_marker_survival_dataset(
            df_risk, marker, prod_col, hes_date_col, med_date_col, label,
        )

        feasibility_rows.append({
            "marker": marker,
            "label": label,
            "n_total": len(df_risk),
            "n_prevalent_excluded": msd.n_prevalent_excluded if msd else 0,
            "n_at_risk": msd.n_at_risk if msd else 0,
            "n_events": msd.n_events if msd else 0,
            "viable": msd is not None,
        })

        if msd is not None:
            marker_datasets[marker] = msd
            print(
                f"  {label:<25s} at-risk={msd.n_at_risk:>8,}  "
                f"events={msd.n_events:>6,}  "
                f"prevalent_excluded={msd.n_prevalent_excluded:>6,}"
            )

    # Report excluded markers
    for marker, reason in EXCLUDED_MARKERS.items():
        feasibility_rows.append({
            "marker": marker,
            "label": marker.replace("_", " ").title(),
            "n_total": len(df_risk),
            "n_prevalent_excluded": 0,
            "n_at_risk": 0,
            "n_events": 0,
            "viable": False,
            "exclusion_reason": reason,
        })

    df_feasibility = pd.DataFrame(feasibility_rows)
    save_table(df_feasibility, path_results / "secondary_feasibility.csv")

    if not marker_datasets:
        warnings.warn("No viable markers — aborting secondary analysis.")
        return

    # ── 4. Per-marker Cox PH ────────────────────────────────────────────
    print("\n[3/6] Fitting per-marker Cox models ...")
    cox_rows: List[Dict] = []
    ph_rows: List[Dict] = []

    for marker, msd in marker_datasets.items():
        result = fit_rbd_continuous_cox(
            msd.df, "time", "event", "rbd_prob", extended_covariates,
        )
        if result is None:
            continue

        cox_rows.append({
            "marker": marker,
            "marker_label": msd.label,
            "hr_per_sd": result["hr_per_sd"],
            "hr_lci": result["hr_lci"],
            "hr_uci": result["hr_uci"],
            "p": result["p"],
            "c_index": result["c_index"],
            "c_index_null": result["c_index_null"],
            "c_index_incremental": result["c_index_incremental"],
            "N": result["N"],
            "events": result["events"],
            "rbd_mean": result["rbd_mean"],
            "rbd_sd": result["rbd_sd"],
        })

        # PH diagnostics
        if not result["ph_df"].empty:
            for cov_name, ph_row in result["ph_df"].iterrows():
                ph_rows.append({
                    "marker": marker,
                    "marker_label": msd.label,
                    "covariate": cov_name,
                    "ph_stat": round(ph_row.get("ph_stat", np.nan), 4),
                    "ph_p": round(ph_row.get("ph_p", np.nan), 4),
                    "ph_violation": bool(ph_row.get("ph_violation", False)),
                })

        print(
            f"  {msd.label:<25s} HR/SD={result['hr_per_sd']:.3f} "
            f"({result['hr_lci']:.3f}–{result['hr_uci']:.3f}) "
            f"p={result['p']:.4f}  C={result['c_index']:.3f}"
        )

    # FDR correction
    df_cox = pd.DataFrame(cox_rows)
    if not df_cox.empty:
        df_cox["p_fdr"] = apply_fdr(df_cox["p"]).values
        save_table(df_cox, path_results / "secondary_per_marker_cox.csv")

    df_ph = pd.DataFrame(ph_rows)
    if not df_ph.empty:
        save_table(df_ph, path_results / "secondary_ph_diagnostics.csv")

    # C-index table
    if not df_cox.empty:
        c_cols = [
            "marker", "marker_label", "c_index", "c_index_null",
            "c_index_incremental", "N", "events",
        ]
        df_cox[c_cols].to_csv(path_results / "secondary_c_index.csv", index=False)
        save_table(df_cox, path_results / "secondary_fdr_corrected.csv")

    # ── 5. KM plots ─────────────────────────────────────────────────────
    print("\n[4/6] Generating KM plots ...")
    km_rows: List[Dict] = []

    for marker, msd in marker_datasets.items():
        for method in methods:
            rbd_col = col_risk_group_agnostic(method)

            if rbd_col not in msd.df.columns:
                continue

            save_path = str(path_figures / f"KM_{marker}_{method}.png")
            km_result = plot_km_by_rbd_group(
                msd.df, "time", "event", rbd_col,
                msd.label, method.upper(), save_path,
            )

            km_rows.append({
                "marker": marker,
                "marker_label": msd.label,
                "method": method,
                "logrank_p": km_result["logrank_p"],
                "N": km_result["N"],
                "events": km_result["events"],
            })

    df_km = pd.DataFrame(km_rows)
    if not df_km.empty:
        save_table(df_km, path_results / "secondary_km_logrank.csv")

    # ── 6. Prodromal burden model ───────────────────────────────────────
    print("\n[5/6] Fitting prodromal burden model ...")

    # Count incident prodromal events per subject across all viable markers
    # Use the full cohort (subjects at risk for >= 1 marker)
    burden_cols = []
    for marker, msd in marker_datasets.items():
        incident_col = f"incident_{marker}"
        df_risk[incident_col] = 0
        at_risk_idx = msd.df.index
        event_idx = msd.df.loc[msd.df["event"] == 1].index
        df_risk.loc[event_idx, incident_col] = 1
        burden_cols.append(incident_col)

    df_risk["prodromal_incident_count"] = df_risk[burden_cols].sum(axis=1)

    burden_result = fit_prodromal_burden_model(
        df_risk, "prodromal_incident_count", "rbd_prob", extended_covariates,
    )
    if burden_result is not None:
        df_burden = pd.DataFrame([{
            "irr_per_sd": burden_result["irr_per_sd"],
            "irr_lci": burden_result["irr_lci"],
            "irr_uci": burden_result["irr_uci"],
            "p": burden_result["p"],
            "model_type": burden_result["model_type"],
            "overdispersion": burden_result["overdispersion"],
            "N": burden_result["N"],
            "mean_count": burden_result["mean_count"],
        }])
        save_table(df_burden, path_results / "secondary_prodromal_burden.csv")

        print(
            f"  Burden model ({burden_result['model_type']}): "
            f"IRR/SD={burden_result['irr_per_sd']:.3f} "
            f"({burden_result['irr_lci']:.3f}–{burden_result['irr_uci']:.3f}) "
            f"p={burden_result['p']:.4f}  "
            f"overdispersion={burden_result['overdispersion']:.2f}"
        )

    # ── 7. Forest plot ──────────────────────────────────────────────────
    print("\n[6/6] Generating forest plot ...")
    if not df_cox.empty:
        forest_data = df_cox.to_dict("records")
        plot_forest(forest_data, str(path_figures / "forest_rbd_predicts_prodromal.png"))

    # ── Summary ─────────────────────────────────────────────────────────
    sep = "=" * 72
    print(f"\n{sep}")
    print("  SECONDARY ANALYSIS — RBD PREDICTS PRODROMAL SYMPTOMS")
    print(sep)
    if not df_cox.empty:
        for _, r in df_cox.iterrows():
            p_str = "< 0.001" if r["p_fdr"] < 0.001 else f"{r['p_fdr']:.3f}"
            print(
                f"  {r['marker_label']:<25s} HR/SD={r['hr_per_sd']:.3f} "
                f"({r['hr_lci']:.3f}–{r['hr_uci']:.3f}) "
                f"p_FDR={p_str}  deltaC={r['c_index_incremental']:.4f}"
            )
    if burden_result:
        print(
            f"\n  Burden: IRR/SD={burden_result['irr_per_sd']:.3f} "
            f"({burden_result['irr_lci']:.3f}–{burden_result['irr_uci']:.3f}) "
            f"p={burden_result['p']:.4f}"
        )
    print(f"\n  Output: {path_results}")
    print(sep)
    print("\n[DONE]")


# ============================================================================
if __name__ == "__main__":
    run_rbd_predicts_prodromal()
