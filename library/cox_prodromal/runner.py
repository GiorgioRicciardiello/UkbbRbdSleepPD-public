"""
Pipeline orchestration for Cox prodromal analysis.

Thin orchestrator: no statistical logic. Calls module functions
and collects results into tables for report generation.

Parallelism
-----------
The outer loop over outcomes is embarrassingly parallel — each outcome
operates on an independent survival dataset with no cross-outcome dependencies
until the post-processing step. ``_process_one_outcome`` is a top-level
picklable worker that can be dispatched to a ``ProcessPoolExecutor``.

Worker count: min(n_outcomes, cpu_count - 1, MAX_WORKERS) so one core is
always left for the main process and OS scheduler.
"""
from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.config import config
from library.column_registry import (
    AGNOSTIC_RISK_COLS,
    col_incident,
    col_risk_group_agnostic,
    col_surv_event,
    col_surv_time,
    METHOD_TO_RISK_SUFFIX,
)

from library.cox_prodromal.cox_config import (
    ABSOLUTE_RISK_TIMEPOINTS,
    AGE_COL,
    AGE_STRATA,
    BASE_COVARIATES,
    BOOTSTRAP_JOBS,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    COMPETING_OUTCOMES,
    GBA_COL,
    HES_GAP_COL,
    HES_GAP_THRESHOLD_YEARS,
    MAX_WORKERS,
    MODEL_A_COVARIATES,
    PRS_COLS,
    PRODROMAL_BINARY_VARS,
    LAG_YEARS,
    METHODS,
    MIN_EVENTS_FOR_MODEL,
    OUTCOMES,
    PRIMARY_METHOD,
    PRIMARY_OUTCOME,
    PRODROMAL_TMT_VARS,
    PRODROMAL_VARS,
    RIDGE_PENALIZER,
    RUN_MEDIATION,
    SCREENING_PERCENTILES,
    SCREENING_TIME_HORIZONS,
)
from library.cox_prodromal.data_prep import (
    apply_lag_filter,
    build_availability_table,
    build_extended_covariates,
    build_survival_dataset_for_outcome,
    categorize_continuous,
    discretize_prodromal,
    filter_active_variables,
    insert_after_data,
    load_prodromal_dataset,
)
from library.cox_prodromal.diagnostics import apply_fdr, summarize_ph_violations
from library.cox_prodromal.model_a_rbd_prs import (
    fit_model_a_rbd_prs_continuous,
    fit_model_a_rbd_prs_categorical,
    fit_model_a_rbd_prs_interaction,
    fit_model_a_discrimination_comparison,
)
from library.cox_prodromal.model_f_rbd_prs_strata_interaction import (
    fit_rbd_prs_interaction_cox,
    bootstrap_rbd_prs_interaction,
)
from library.cox_prodromal.model_g_rbd_gba import (
    fit_model_g_rbd_gba_continuous,
    fit_model_g_rbd_gba_categorical,
)
from library.cox_prodromal.model_baseline import fit_baseline_cox
from library.cox_prodromal.model_rbd import (
    fit_rbd_only_cox,
    fit_rbd_continuous_per_sd,
    fit_rbd_threshold_stability,
)
from library.cox_prodromal.model_additive import fit_additive_cox
from library.cox_prodromal.model_interaction import fit_interaction_cox
from library.cox_prodromal.model_competing import (
    compare_cif_vs_km,
    encode_competing_events,
    fit_aalen_johansen_cif,
    fit_cause_specific_cox,
)
from library.cox_prodromal.additive_interaction import bootstrap_additive_interaction
from library.cox_prodromal.additive_interaction_poisson import compute_poisson_reri
from library.cox_prodromal.model_time_varying import fit_time_interaction_sensitivity
from library.cox_prodromal.screening_metrics import compute_screening_metrics
from library.cox_prodromal.calibration import (
    calibration_in_the_large,
    calibration_slope,
)
from library.cox_prodromal.discrimination import (
    bootstrap_delta_c_test,
    compute_idi,
    compute_nri,
    extract_predicted_risks,
)
from library.cox_prodromal.splines import fit_spline_cox_prodromal, fit_spline_cox_rbd
from library.cox_prodromal.rbd_spline_analysis import run_rbd_spline_analysis
from library.cox_prodromal.plotting import (
    compute_absolute_risks_km,
    plot_cumulative_incidence_rbd,
    plot_rbd_distribution_single,
    plot_rbd_only_km_full,
    plot_three_panel_km,
)
from library.cox_prodromal.report_builder import generate_scientific_report
from library.cox_prodromal.utils import save_table


# MAX_WORKERS and BOOTSTRAP_JOBS imported from cox_config


# ── Row collectors ─────────────────────────────────────────────────────────

def _flatten_summary(
    result: Dict[str, Any],
    extra: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Flatten a Cox summary DataFrame into a list of row dicts.

    Includes per-covariate: coef (log-HR), se(coef), SE of HR via delta
    method (HR * se(coef)), and Wald z-statistic alongside the existing
    HR, CI, and p-value columns.
    """
    rows = []
    for _, srow in result["summary"].iterrows():
        # Extract values safely, converting Series to scalars if needed
        def _safe_get(s, key, default=np.nan):
            """Safely extract a scalar value from Series."""
            val = s.get(key, default)
            # If we get a Series instead of scalar, take its first value
            if isinstance(val, pd.Series):
                val = val.iloc[0] if len(val) > 0 else default
            return val

        hr = _safe_get(srow, "exp(coef)", np.nan)
        se_coef = _safe_get(srow, "se(coef)", np.nan)
        coef_val = _safe_get(srow, "coef", np.nan)
        z_val = _safe_get(srow, "z", np.nan)
        hr_lower = _safe_get(srow, "exp(coef) lower 95%", np.nan)
        hr_upper = _safe_get(srow, "exp(coef) upper 95%", np.nan)
        p_val = _safe_get(srow, "p", np.nan)

        row = {
            **extra,
            "covariate": _safe_get(srow, "covariate", srow.name),
            "coef": round(float(coef_val), 6) if pd.notna(coef_val) else np.nan,
            "se_coef": round(float(se_coef), 6) if pd.notna(se_coef) else np.nan,
            "se_hr": round(float(hr) * float(se_coef), 6) if pd.notna(hr) and pd.notna(se_coef) else np.nan,
            "z": round(float(z_val), 4) if pd.notna(z_val) else np.nan,
            "HR": round(float(hr), 4) if pd.notna(hr) else np.nan,
            "HR_lower": round(float(hr_lower), 4) if pd.notna(hr_lower) else np.nan,
            "HR_upper": round(float(hr_upper), 4) if pd.notna(hr_upper) else np.nan,
            "p": p_val,
            "N": result["N"],
            "events": result["events"],
        }
        rows.append(row)
    return rows


def _extract_fit_row(
    result: Dict[str, Any],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract model-level fit metrics from a model result dict.

    Returns a single row dict with AIC, BIC, log-likelihood, LRT,
    and null-model comparisons for the model fit summary table.
    """
    return {
        **extra,
        "AIC": result.get("AIC", np.nan),
        "BIC": result.get("BIC", np.nan),
        "log_likelihood": result.get("log_likelihood", np.nan),
        "n_params": result.get("n_params", np.nan),
        "AIC_null": result.get("AIC_null", np.nan),
        "BIC_null": result.get("BIC_null", np.nan),
        "log_likelihood_null": result.get("log_likelihood_null", np.nan),
        "delta_AIC": (
            round(result.get("AIC", np.nan) - result.get("AIC_null", np.nan), 2)
            if pd.notna(result.get("AIC")) and pd.notna(result.get("AIC_null"))
            else np.nan
        ),
        "delta_BIC": (
            round(result.get("BIC", np.nan) - result.get("BIC_null", np.nan), 2)
            if pd.notna(result.get("BIC")) and pd.notna(result.get("BIC_null"))
            else np.nan
        ),
        "LRT_stat": result.get("LRT_stat", np.nan),
        "LRT_p": result.get("LRT_p", np.nan),
        "LRT_df": result.get("LRT_df", np.nan),
        "c_index": round(result.get("c_index", np.nan), 4),
        "c_index_null": (
            round(result.get("c_index_null", np.nan), 4)
            if pd.notna(result.get("c_index_null")) else np.nan
        ),
        "N": result.get("N", np.nan),
        "events": result.get("events", np.nan),
    }


# ── Per-outcome worker ─────────────────────────────────────────────────────

def _process_one_outcome(
    outcome: str,
    df_risk: pd.DataFrame,
    active_vars: Dict[str, str],
    extended_covariates: List[str],
    methods: List[str],
    path_results: Path,
    run_competing: bool,
    run_additive_interaction: bool,
    n_bootstrap: int,
    seed: int,
    n_bootstrap_jobs: int = BOOTSTRAP_JOBS,
) -> Dict[str, Any]:
    """
    Run all models for a single outcome and return collected row lists.

    This function is a top-level, picklable worker for ``ProcessPoolExecutor``.
    Each call is independent of all other outcomes — there are no shared writes.

    Parameters
    ----------
    outcome : str
        Outcome key (e.g. 'outcome_1a_pd_only').
    df_risk : pd.DataFrame
        Full subject-level dataset (read-only; each worker receives its own copy).
    active_vars : dict[str, str]
        {column: display_label} for all active prodromal variables.
    extended_covariates : list[str]
        Adjustment covariate column names.
    methods : list[str]
        RBD stratification methods to iterate over.
    path_results : Path
        Root output directory (already created by caller).
    run_competing : bool
        Whether to run Model 4 competing risk analysis.
    run_additive_interaction : bool
        Whether to run RERI/AP/SI bootstrap analysis.
    n_bootstrap : int
        Bootstrap resamples for RERI / delta-C.
    seed : int
        Random seed. Derived per-outcome from the caller's base seed.

    Returns
    -------
    dict
        Keys matching the result-list names in ``run_prodromal_pipeline``.
    """
    # Use non-interactive backend inside worker processes to avoid
    # display-connection errors on headless / spawned processes.
    import matplotlib
    matplotlib.use("Agg")

    import time as _time
    _t_outcome_start = _time.time()

    def _log_timing(label: str, t0: float) -> float:
        """Log elapsed time for a pipeline section and return new t0."""
        elapsed = _time.time() - t0
        import sys
        sys.stdout.write(f"    [{outcome}] {label}: {elapsed:.1f}s\n")
        sys.stdout.flush()
        return _time.time()

    # ── Local result collectors ─────────────────────────────────────────
    cohort_rows: List[Dict] = []
    model_a_rows: List[Dict] = []
    model_a_interaction_rows: List[Dict] = []
    model_a_disc_comp_rows: List[Dict] = []
    model_f_rows: List[Dict] = []  # Model F: RBD strata × PRS_PD interaction
    model_f_strat_rows: List[Dict] = []  # Stratified PRS effects
    model_g_rows: List[Dict] = []       # Model G: RBD × GBA carrier interaction
    model_g_cell_rows: List[Dict] = []  # Model G cell counts
    baseline_rows: List[Dict] = []
    ph_rows: List[Dict] = []
    c_index_rows: List[Dict] = []
    rbd_only_rows: List[Dict] = []
    rbd_cont_rows: List[Dict] = []
    rbd_thresh_all: List[pd.DataFrame] = []
    additive_rows: List[Dict] = []
    interaction_rows: List[Dict] = []
    km_lr_rows: List[Dict] = []
    abs_risk_rows: List[pd.DataFrame] = []
    spline_rows: List[Dict] = []
    rbd_spline_rows: List[Dict] = []
    lag_rows: List[Dict] = []
    reri_rows: List[Dict] = []
    sensitivity_rows: List[Dict] = []
    comp_cif_rows: List[pd.DataFrame] = []
    comp_cox_rows: List[Dict] = []
    model_fit_rows: List[Dict] = []
    screening_rows: List[Dict] = []
    age_strat_rows: List[Dict] = []
    ph_time_rows: List[pd.DataFrame] = []
    poisson_reri_rows: List[Dict] = []
    disc_rows: List[Dict] = []
    cal_rows: List[Dict] = []

    # ── Survival dataset ────────────────────────────────────────────────
    _t0 = _time.time()
    path_outcome = path_results / outcome
    path_outcome.mkdir(parents=True, exist_ok=True)

    df_surv = build_survival_dataset_for_outcome(
        df=df_risk,
        outcome=outcome,
        active_vars=active_vars,
        extended_covariates=extended_covariates
    )
    if df_surv is None:
        return _empty_result()

    n_ev = int(df_surv["event"].sum())
    fu_med = df_surv.loc[df_surv["event"] == 0, "time"].median()
    cohort_rows.append({
        "outcome": outcome,
        "N_cohort": len(df_surv),
        "n_events": n_ev,
        "n_controls": int((df_surv["event"] == 0).sum()),
        "median_follow_up_years": round(fu_med, 2) if pd.notna(fu_med) else np.nan,
    })

    # ── Pre-compute boolean masks (avoid repeated dropna scans) ────────
    time_event_mask = df_surv[["time", "event"]].notna().all(axis=1)
    cov_mask = df_surv[extended_covariates].notna().all(axis=1)
    rbd_masks: Dict[str, pd.Series] = {}
    for _method in methods:
        _rbd_col = col_risk_group_agnostic(_method)
        if _rbd_col in df_surv.columns:
            rbd_masks[_rbd_col] = df_surv[_rbd_col].notna()

    _t0 = _log_timing("Survival dataset build", _t0)

    # ── MODEL 0: RBD-only (per method) ─────────────────────────────────
    path_km_full = path_outcome / "km_full"
    path_km_full.mkdir(exist_ok=True)

    for method in methods:
        rbd_col = col_risk_group_agnostic(method)
        if rbd_col not in df_surv.columns:
            continue

        df_full_rbd = df_surv[time_event_mask & rbd_masks[rbd_col]].copy()
        df_full_rbd[rbd_col] = df_full_rbd[rbd_col].astype(str)
        plot_rbd_only_km_full(
            df_full_rbd, "time", "event", rbd_col,
            outcome.replace("_", " ").title(),
            method.upper(),
            str(path_km_full / f"KM_full_{method}.png"),
        )

        # Cumulative incidence (1−KM) companion figure for primary outcome ×
        # primary method.  Low event rates make survival plots hard to read;
        # CI plots expose absolute risk differences directly.
        if outcome == PRIMARY_OUTCOME and method == PRIMARY_METHOD:
            plot_cumulative_incidence_rbd(
                df_full_rbd, "time", "event", rbd_col,
                outcome.replace("_", " ").title(),
                method.upper(),
                str(path_km_full / f"CI_full_{method}.png"),
            )

        df_rbd = df_surv[time_event_mask & rbd_masks[rbd_col] & cov_mask].copy()
        df_rbd[rbd_col] = df_rbd[rbd_col].astype(str)

        rbd_result = fit_rbd_only_cox(
            df_rbd, "time", "event", rbd_col, extended_covariates
        )
        if rbd_result is not None:
            extra = {"outcome": outcome, "method": method, "model": "M0_rbd_only"}
            rbd_only_rows.extend(_flatten_summary(rbd_result, extra))
            model_fit_rows.append(_extract_fit_row(
                rbd_result,
                {"outcome": outcome, "model": "M0_rbd_categorical",
                 "method": method, "prodromal_var": "", "prodromal_label": ""},
            ))
            for cov_name, ph_row in rbd_result["ph_df"].iterrows():
                ph_rows.append({
                    "outcome": outcome, "model": "M0",
                    "prodromal_var": "rbd", "covariate": cov_name,
                    "ph_stat": round(ph_row.get("ph_stat", np.nan), 4),
                    "ph_p": round(ph_row.get("ph_p", np.nan), 4),
                    "ph_violation": bool(ph_row.get("ph_violation", False)),
                })

    # RBD continuous per-SD (primary method only)
    if "abk_rbd_score_mean" in df_surv.columns:
        rbd_cont = fit_rbd_continuous_per_sd(
            df_surv, "time", "event", "abk_rbd_score_mean", extended_covariates
        )
        if rbd_cont is not None:
            rbd_cont_rows.append({
                "outcome": outcome,
                "coef": round(rbd_cont["coef"], 6),
                "se_coef": round(rbd_cont["se_coef"], 6),
                "se_hr": round(rbd_cont["se_hr"], 6),
                "z": round(rbd_cont["z"], 4),
                "hr_per_sd": round(rbd_cont["hr_per_sd"], 4),
                "hr_lci": round(rbd_cont["hr_lci"], 4),
                "hr_uci": round(rbd_cont["hr_uci"], 4),
                "p": rbd_cont["p"],
                "c_index": round(rbd_cont["c_index"], 4),
                "rbd_mean": round(rbd_cont["rbd_mean"], 6),
                "rbd_sd": round(rbd_cont["rbd_sd"], 6),
                "N": rbd_cont["N"],
                "events": rbd_cont["events"],
            })
            model_fit_rows.append(_extract_fit_row(
                rbd_cont,
                {"outcome": outcome, "model": "M0_rbd_continuous",
                 "method": "continuous", "prodromal_var": "", "prodromal_label": ""},
            ))

        thresh_df = fit_rbd_threshold_stability(
            df_surv, "time", "event", "abk_rbd_score_mean", extended_covariates
        )
        if not thresh_df.empty:
            thresh_df["outcome"] = outcome
            rbd_thresh_all.append(thresh_df)

        if outcome == PRIMARY_OUTCOME:
            rbd_sp = fit_spline_cox_rbd(
                df_surv, "time", "event", "abk_rbd_score_mean", extended_covariates
            )
            if rbd_sp is not None:
                rbd_spline_rows.append({
                    "outcome": outcome,
                    "c_index_spline": round(rbd_sp["c_index_spline"], 4),
                    "c_index_linear": round(rbd_sp.get("c_index_linear", np.nan), 4),
                    "lr_stat": round(rbd_sp.get("lr_stat", np.nan), 3),
                    "lr_p": round(rbd_sp.get("lr_p", np.nan), 4),
                    "N": rbd_sp["N"],
                    "events": rbd_sp["events"],
                })

    _t0 = _log_timing("Model 0 (all RBD-only fits + KM)", _t0)

    # ── MODEL A: RBD + PRS + ancestry PCs ────────────────────────────────
    # Restricted to PRIMARY_OUTCOME (outcome_1a_pd_only) only.
    # Rationale: PRS was constructed for PD; applying it to secondary outcomes
    # (AD, vascular dementia) is scientifically inappropriate and causes
    # model instability — secondary outcomes have EPV < 5 with 18 parameters,
    # making all coefficient SEs explode and RBD appearing spuriously non-significant.
    has_prs = all(c in df_surv.columns for c in PRS_COLS)
    prs_non_null = 0
    if has_prs:
        prs_non_null = df_surv[PRS_COLS].notna().all(axis=1).sum()

    if has_prs and prs_non_null > 0 and outcome == PRIMARY_OUTCOME:
        # Build covariate list with PRS + PCs
        model_a_covs = extended_covariates + MODEL_A_COVARIATES

        for method in methods:
            rbd_cat_col = col_risk_group_agnostic(method)
            if rbd_cat_col not in df_surv.columns:
                continue

            if "abk_rbd_score_mean" not in df_surv.columns:
                continue

            # Filter to subjects with complete PRS + time/event + RBD + covariates
            model_a_mask = (
                df_surv[PRS_COLS].notna().all(axis=1) &
                time_event_mask &
                df_surv[[rbd_cat_col, "abk_rbd_score_mean"]].notna().all(axis=1) &
                df_surv[model_a_covs].notna().all(axis=1)
            )
            df_model_a = df_surv[model_a_mask].copy()
            df_model_a[rbd_cat_col] = df_model_a[rbd_cat_col].astype(str)

            if df_model_a["event"].sum() < MIN_EVENTS_FOR_MODEL:
                continue

            # Continuous (z-scored RBD)
            cont_result = fit_model_a_rbd_prs_continuous(
                df_model_a, "time", "event", "abk_rbd_score_mean", model_a_covs
            )
            if cont_result is not None:
                extra = {
                    "outcome": outcome,
                    "method": method,
                    "model": "MA_rbd_continuous",
                }
                model_a_rows.extend(_flatten_summary(cont_result, extra))
                model_fit_rows.append(_extract_fit_row(
                    cont_result,
                    {"outcome": outcome, "model": "MA_rbd_continuous",
                     "method": method, "prodromal_var": "", "prodromal_label": ""},
                ))
                for cov_name, ph_row in cont_result["ph_df"].iterrows():
                    ph_rows.append({
                        "outcome": outcome, "model": "MA_continuous",
                        "prodromal_var": "rbd_prs", "covariate": cov_name,
                        "ph_stat": round(ph_row.get("ph_stat", np.nan), 4),
                        "ph_p": round(ph_row.get("ph_p", np.nan), 4),
                        "ph_violation": bool(ph_row.get("ph_violation", False)),
                    })

            # Categorical (RBD group)
            cat_result = fit_model_a_rbd_prs_categorical(
                df_model_a, "time", "event", rbd_cat_col, model_a_covs
            )
            if cat_result is not None:
                extra = {
                    "outcome": outcome,
                    "method": method,
                    "model": "MA_rbd_categorical",
                }
                model_a_rows.extend(_flatten_summary(cat_result, extra))
                model_fit_rows.append(_extract_fit_row(
                    cat_result,
                    {"outcome": outcome, "model": "MA_rbd_categorical",
                     "method": method, "prodromal_var": "", "prodromal_label": ""},
                ))
                for cov_name, ph_row in cat_result["ph_df"].iterrows():
                    ph_rows.append({
                        "outcome": outcome, "model": "MA_categorical",
                        "prodromal_var": "rbd_prs", "covariate": cov_name,
                        "ph_stat": round(ph_row.get("ph_stat", np.nan), 4),
                        "ph_p": round(ph_row.get("ph_p", np.nan), 4),
                        "ph_violation": bool(ph_row.get("ph_violation", False)),
                    })

            # Interaction: rbd_z × prs_score_pd
            int_result = fit_model_a_rbd_prs_interaction(
                df_model_a, "time", "event", "abk_rbd_score_mean", model_a_covs
            )
            if int_result is not None:
                extra_int = {
                    "outcome": outcome,
                    "method": method,
                    "model": "MA_rbd_x_prs_pd",
                }
                model_a_interaction_rows.extend(_flatten_summary(int_result, extra_int))
                # Append interaction-specific fit metrics as extra columns
                fit_row = _extract_fit_row(
                    int_result,
                    {"outcome": outcome, "model": "MA_rbd_x_prs_pd",
                     "method": method, "prodromal_var": "", "prodromal_label": ""},
                )
                fit_row["lrt_interaction_stat"] = int_result.get("lrt_interaction_stat", np.nan)
                fit_row["lrt_interaction_p"] = int_result.get("lrt_interaction_p", np.nan)
                fit_row["AIC_additive"] = int_result.get("AIC_additive", np.nan)
                fit_row["BIC_additive"] = int_result.get("BIC_additive", np.nan)
                fit_row["delta_AIC_vs_additive"] = int_result.get("delta_AIC_vs_additive", np.nan)
                fit_row["c_index_additive"] = int_result.get("c_index_additive", np.nan)
                fit_row["delta_c_vs_additive"] = int_result.get("c_index_incremental_vs_additive", np.nan)
                model_a_interaction_rows.append({**extra_int, **fit_row, "_row_type": "fit_metrics"})

            # Discrimination comparison: M0 (RBD+base) vs M1 (additive) vs M2 (interaction)
            disc_comp = fit_model_a_discrimination_comparison(
                df_model_a, "time", "event", "abk_rbd_score_mean",
                extended_covariates, model_a_covs
            )
            if disc_comp is not None:
                model_a_disc_comp_rows.append({
                    "outcome": outcome,
                    "method": method,
                    **disc_comp,
                })

    _t0 = _log_timing("Model A (RBD + PRS + PCs)", _t0)

    # ── MODEL F: RBD Strata × PRS_PD Interaction ────────────────────────
    # Restricted to PRIMARY_OUTCOME (outcome_1a_pd_only) only.
    # Tests multiplicative interaction between RBD risk groups and genetic PD risk.
    # Reports stratified PRS effects per RBD stratum + interaction p-value.
    if has_prs and prs_non_null > 0 and outcome == PRIMARY_OUTCOME:
        print(f"\n  [Model F] RBD × PRS interaction analysis ...")

        # Use percentile_3g (Low/Mid/High) for primary analysis
        method_f = PRIMARY_METHOD  # "percentile_3g"
        rbd_cat_col_f = col_risk_group_agnostic(method_f)

        if rbd_cat_col_f in df_surv.columns:
            # Filter to subjects with complete PRS + RBD + covariates
            model_f_covs = extended_covariates + MODEL_A_COVARIATES
            model_f_mask = (
                df_surv[PRS_COLS].notna().all(axis=1) &
                time_event_mask &
                df_surv[[rbd_cat_col_f, "abk_rbd_score_mean"]].notna().all(axis=1) &
                df_surv[model_f_covs].notna().all(axis=1)
            )
            df_model_f = df_surv[model_f_mask].copy()
            df_model_f[rbd_cat_col_f] = df_model_f[rbd_cat_col_f].astype(str)

            if df_model_f["event"].sum() >= MIN_EVENTS_FOR_MODEL:
                # Fit full interaction Cox
                f_result = fit_rbd_prs_interaction_cox(
                    df_model_f, "time", "event", rbd_cat_col_f,
                    "prs_score_pd", model_f_covs
                )

                if f_result is not None:
                    # Extract full model summary
                    extra_f = {
                        "outcome": outcome,
                        "method": method_f,
                        "model": "MF_rbd_prs_strata_interaction",
                    }
                    model_f_rows.extend(_flatten_summary(f_result, extra_f))

                    # Model fit metrics
                    model_fit_rows.append(_extract_fit_row(
                        f_result,
                        {"outcome": outcome, "model": "MF_rbd_prs_interaction",
                         "method": method_f, "prodromal_var": "", "prodromal_label": ""},
                    ))

                    # PH test
                    for cov_name, ph_row in f_result["ph_df"].iterrows():
                        ph_rows.append({
                            "outcome": outcome, "model": "MF_interaction",
                            "prodromal_var": "rbd_prs_strata", "covariate": cov_name,
                            "ph_stat": round(ph_row.get("ph_stat", np.nan), 4),
                            "ph_p": round(ph_row.get("ph_p", np.nan), 4),
                            "ph_violation": bool(ph_row.get("ph_violation", False)),
                        })

                    # Bootstrap stratified PRS effects
                    print(f"    Bootstrapping stratified PRS effects ({BOOTSTRAP_N} resamples)...")
                    stratification_groups = sorted(df_model_f[rbd_cat_col_f].unique())

                    boot_result = bootstrap_rbd_prs_interaction(
                        df_model_f, "time", "event", rbd_cat_col_f,
                        "prs_score_pd", model_f_covs,
                        stratification_groups=stratification_groups,
                        n_bootstrap=BOOTSTRAP_N,
                        seed=BOOTSTRAP_SEED,
                        penalizer=RIDGE_PENALIZER,
                        n_jobs=BOOTSTRAP_JOBS,
                    )

                    if boot_result is not None:
                        for grp in stratification_groups:
                            model_f_strat_rows.append({
                                "outcome": outcome,
                                "method": method_f,
                                "rbd_stratum": str(grp),
                                "prs_hr": boot_result.get(f"prs_hr_{grp}", np.nan),
                                "prs_hr_lci": boot_result.get(f"prs_hr_{grp}_lci", np.nan),
                                "prs_hr_uci": boot_result.get(f"prs_hr_{grp}_uci", np.nan),
                                "N": f_result["N"],
                                "events": f_result["events"],
                            })

    _t0 = _log_timing("Model F (RBD strata × PRS_PD interaction)", _t0)

    # ── MODEL G: RBD × GBA carrier interaction ───────────────────────────
    # Restricted to PRIMARY_OUTCOME only.
    # GBA variants are PD-specific; secondary outcomes (AD, vascular dementia)
    # have no established GBA association.
    has_gba = GBA_COL in df_surv.columns
    gba_non_null = 0
    if has_gba:
        gba_non_null = int(df_surv[GBA_COL].notna().sum())

    if has_gba and gba_non_null > 0 and outcome == PRIMARY_OUTCOME:
        print(f"\n  [Model G] RBD × GBA carrier interaction ...")

        method_g = PRIMARY_METHOD  # "percentile_3g"
        rbd_cat_col_g = col_risk_group_agnostic(method_g)

        if rbd_cat_col_g in df_surv.columns:
            model_g_mask = (
                df_surv[GBA_COL].notna() &
                time_event_mask &
                df_surv[[rbd_cat_col_g, "abk_rbd_score_mean"]].notna().all(axis=1) &
                df_surv[extended_covariates].notna().all(axis=1)
            )
            df_model_g = df_surv[model_g_mask].copy()
            df_model_g[rbd_cat_col_g] = df_model_g[rbd_cat_col_g].astype(str)

            if df_model_g["event"].sum() >= MIN_EVENTS_FOR_MODEL:
                extra_g_cont = {
                    "outcome": outcome,
                    "method": method_g,
                    "model": "MG_rbd_continuous_x_gba",
                }
                extra_g_cat = {
                    "outcome": outcome,
                    "method": method_g,
                    "model": "MG_rbd_categorical_x_gba",
                }

                # Continuous: z-scored RBD × GBA
                g_cont = fit_model_g_rbd_gba_continuous(
                    df_model_g, "time", "event",
                    "abk_rbd_score_mean", GBA_COL, extended_covariates,
                )
                if g_cont is not None:
                    model_g_rows.extend(_flatten_summary(g_cont, extra_g_cont))
                    model_g_rows.append({
                        **extra_g_cont,
                        "_row_type": "fit_metrics",
                        "lrt_interaction_stat": g_cont.get("lrt_interaction_stat", np.nan),
                        "lrt_interaction_p": g_cont.get("lrt_interaction_p", np.nan),
                        "n_gba_carriers": g_cont.get("n_gba_carriers", np.nan),
                        **_extract_fit_row(g_cont, extra_g_cont),
                    })
                    for cov_name, ph_row in g_cont["ph_df"].iterrows():
                        ph_rows.append({
                            "outcome": outcome, "model": "MG_continuous",
                            "prodromal_var": "rbd_gba", "covariate": cov_name,
                            "ph_stat": round(ph_row.get("ph_stat", np.nan), 4),
                            "ph_p": round(ph_row.get("ph_p", np.nan), 4),
                            "ph_violation": bool(ph_row.get("ph_violation", False)),
                        })

                # Categorical: RBD groups × GBA
                g_cat = fit_model_g_rbd_gba_categorical(
                    df_model_g, "time", "event",
                    rbd_cat_col_g, GBA_COL, extended_covariates,
                )
                if g_cat is not None:
                    model_g_rows.extend(_flatten_summary(g_cat, extra_g_cat))
                    model_g_rows.append({
                        **extra_g_cat,
                        "_row_type": "fit_metrics",
                        "lrt_interaction_stat": g_cat.get("lrt_interaction_stat", np.nan),
                        "lrt_interaction_p": g_cat.get("lrt_interaction_p", np.nan),
                        "reri": g_cat.get("reri", np.nan),
                        "n_gba_carriers": g_cat.get("n_gba_carriers", np.nan),
                        **_extract_fit_row(g_cat, extra_g_cat),
                    })
                    for cc in g_cat.get("cell_counts", []):
                        model_g_cell_rows.append({
                            "outcome": outcome,
                            "method": method_g,
                            **cc,
                        })
                    for cov_name, ph_row in g_cat["ph_df"].iterrows():
                        ph_rows.append({
                            "outcome": outcome, "model": "MG_categorical",
                            "prodromal_var": "rbd_gba", "covariate": cov_name,
                            "ph_stat": round(ph_row.get("ph_stat", np.nan), 4),
                            "ph_p": round(ph_row.get("ph_p", np.nan), 4),
                            "ph_violation": bool(ph_row.get("ph_violation", False)),
                        })

                # Categorical (2-group): RBD Low/High × GBA
                # Added for exploratory analysis with reduced sparsity (n_events >= 1 in all cells)
                method_g_2g = "percentile_2g"
                rbd_cat_col_2g = col_risk_group_agnostic(method_g_2g)
                if rbd_cat_col_2g in df_surv.columns:
                    extra_g_cat_2g = {
                        "outcome": outcome,
                        "method": method_g_2g,
                        "model": "MG_rbd_categorical_2g_x_gba",
                    }
                    g_cat_2g = fit_model_g_rbd_gba_categorical(
                        df_model_g, "time", "event",
                        rbd_cat_col_2g, GBA_COL, extended_covariates,
                    )
                    if g_cat_2g is not None:
                        model_g_rows.extend(_flatten_summary(g_cat_2g, extra_g_cat_2g))
                        model_g_rows.append({
                            **extra_g_cat_2g,
                            "_row_type": "fit_metrics",
                            "lrt_interaction_stat": g_cat_2g.get("lrt_interaction_stat", np.nan),
                            "lrt_interaction_p": g_cat_2g.get("lrt_interaction_p", np.nan),
                            "reri": g_cat_2g.get("reri", np.nan),
                            "n_gba_carriers": g_cat_2g.get("n_gba_carriers", np.nan),
                            **_extract_fit_row(g_cat_2g, extra_g_cat_2g),
                        })
                        for cc in g_cat_2g.get("cell_counts", []):
                            model_g_cell_rows.append({
                                "outcome": outcome,
                                "method": method_g_2g,
                                **cc,
                            })
                        for cov_name, ph_row in g_cat_2g["ph_df"].iterrows():
                            ph_rows.append({
                                "outcome": outcome, "model": "MG_categorical_2g",
                                "prodromal_var": "rbd_gba", "covariate": cov_name,
                                "ph_stat": round(ph_row.get("ph_stat", np.nan), 4),
                                "ph_p": round(ph_row.get("ph_p", np.nan), 4),
                                "ph_violation": bool(ph_row.get("ph_violation", False)),
                            })

    _t0 = _log_timing("Model G (RBD × GBA carrier interaction)", _t0)

    # ── SCREENING METRICS (PPV / NPV at percentile thresholds) ─────────
    if "abk_rbd_score_mean" in df_surv.columns:
        scr_df = compute_screening_metrics(
            df_surv, "abk_rbd_score_mean", "time", "event",
        )
        if not scr_df.empty:
            scr_df["outcome"] = outcome
            for _, srow in scr_df.iterrows():
                screening_rows.append(srow.to_dict())

    # ── AGE-STRATIFIED SENSITIVITY (Model A per stratum) ─────────────
    if AGE_COL in df_surv.columns and outcome == PRIMARY_OUTCOME:
        for age_lo, age_hi in AGE_STRATA:
            label = f"{age_lo}-{age_hi}" if age_hi < 200 else f">{age_lo}"
            df_age = df_surv[
                (df_surv[AGE_COL] >= age_lo) & (df_surv[AGE_COL] < age_hi)
            ].copy()
            if df_age.empty or df_age["event"].sum() < MIN_EVENTS_FOR_MODEL:
                continue

            for method in methods:
                rbd_col = col_risk_group_agnostic(method)
                if rbd_col not in df_age.columns:
                    continue
                df_age_m = df_age[df_age[rbd_col].notna()].copy()
                df_age_m[rbd_col] = df_age_m[rbd_col].astype(str)
                # Covariates without age (avoid collinearity within stratum)
                covs_no_age = [c for c in extended_covariates if c != AGE_COL]
                age_result = fit_rbd_only_cox(
                    df_age_m, "time", "event", rbd_col, covs_no_age,
                )
                if age_result is not None:
                    for _, srow in age_result["summary"].iterrows():
                        hr_val = srow.get("exp(coef)", np.nan)
                        se_val = srow.get("se(coef)", np.nan)
                        age_strat_rows.append({
                            "outcome": outcome,
                            "model": "M0_rbd_only",
                            "method": method,
                            "age_stratum": label,
                            "covariate": srow.get("covariate", srow.name),
                            "coef": round(srow.get("coef", np.nan), 6),
                            "se_coef": round(se_val, 6),
                            "se_hr": round(hr_val * se_val, 6) if pd.notna(hr_val) and pd.notna(se_val) else np.nan,
                            "z": round(srow.get("z", np.nan), 4),
                            "HR": round(hr_val, 4),
                            "HR_lower": round(srow.get("exp(coef) lower 95%", np.nan), 4),
                            "HR_upper": round(srow.get("exp(coef) upper 95%", np.nan), 4),
                            "p": srow.get("p", np.nan),
                            "N": age_result["N"],
                            "events": age_result["events"],
                        })

    _t0 = _log_timing("Screening + Age strata", _t0)

    # ── MODELS 1–3 per prodromal variable ──────────────────────────────
    # Reuse cov_mask computed in pre-compute block above.
    df_surv_cov = df_surv.loc[cov_mask]

    for prod_var, prod_label in active_vars.items():
        if prod_var not in df_surv.columns:
            continue

        # Complete-case dataset for this prodromal variable
        df_cc = df_surv_cov.loc[df_surv_cov[prod_var].notna()].copy()
        if df_cc.empty or df_cc["event"].sum() < MIN_EVENTS_FOR_MODEL:
            continue

        prod_grp_col = f"{prod_var}_grp"
        df_cc[prod_grp_col] = discretize_prodromal(df_cc, prod_var)

        # ── Model 1: Baseline Cox ────────────────────────────────────
        result = fit_baseline_cox(
            df_cc, "time", "event", prod_grp_col, extended_covariates
        )
        if result is not None:
            extra_bl = {
                "outcome": outcome,
                "prodromal_var": prod_var,
                "prodromal_label": prod_label,
            }
            baseline_rows.extend(_flatten_summary(result, extra_bl))
            model_fit_rows.append(_extract_fit_row(
                result,
                {"outcome": outcome, "model": "M1_baseline",
                 "method": "", "prodromal_var": prod_var,
                 "prodromal_label": prod_label},
            ))

            c_index_rows.append({
                "outcome": outcome,
                "prodromal_var": prod_var,
                "prodromal_label": prod_label,
                "c_index_full": round(result["c_index"], 4),
                "c_index_null": (
                    round(result["c_index_null"], 4)
                    if pd.notna(result["c_index_null"]) else np.nan
                ),
                "c_index_incremental": (
                    round(result["c_index_incremental"], 4)
                    if pd.notna(result["c_index_incremental"]) else np.nan
                ),
                "N": result["N"],
                "events": result["events"],
            })

            for cov_name, ph_row in result["ph_df"].iterrows():
                ph_rows.append({
                    "outcome": outcome, "model": "M1",
                    "prodromal_var": prod_var,
                    "prodromal_label": prod_label,
                    "covariate": cov_name,
                    "ph_stat": round(ph_row.get("ph_stat", np.nan), 4),
                    "ph_p": round(ph_row.get("ph_p", np.nan), 4),
                    "ph_violation": bool(ph_row.get("ph_violation", False)),
                })

        # ── Sensitivity: HES-active subcohort ───────────────────────
        if prod_var in PRODROMAL_BINARY_VARS and HES_GAP_COL in df_cc.columns:
            df_cc_hes = df_cc[df_cc[HES_GAP_COL] <= HES_GAP_THRESHOLD_YEARS].copy()
            n_excluded_gap = len(df_cc) - len(df_cc_hes)
            if (
                not df_cc_hes.empty
                and df_cc_hes["event"].sum() >= MIN_EVENTS_FOR_MODEL
            ):
                sens_result = fit_baseline_cox(
                    df_cc_hes, "time", "event",
                    prod_grp_col, extended_covariates,
                )
                if sens_result is not None:
                    extra_sens = {
                        "outcome": outcome,
                        "prodromal_var": prod_var,
                        "prodromal_label": prod_label,
                        "analysis": f"sensitivity_hes_active_{HES_GAP_THRESHOLD_YEARS:.0f}y",
                        "N_sensitivity": len(df_cc_hes),
                        "N_excluded_gap": n_excluded_gap,
                        "events_sensitivity": int(df_cc_hes["event"].sum()),
                    }
                    sensitivity_rows.extend(_flatten_summary(sens_result, extra_sens))

        # ── Spline: continuous vars, primary outcome only ────────────
        if outcome == PRIMARY_OUTCOME and prod_var not in PRODROMAL_BINARY_VARS:
            sp_result = fit_spline_cox_prodromal(
                df_surv, "time", "event", prod_var, extended_covariates
            )
            if sp_result is not None:
                spline_rows.append({
                    "outcome": outcome,
                    "prodromal_var": prod_var,
                    "prodromal_label": prod_label,
                    "c_index_spline": round(sp_result["c_index_spline"], 4),
                    "c_index_linear": round(sp_result.get("c_index_linear", np.nan), 4),
                    "lr_stat": round(sp_result.get("lr_stat", np.nan), 3),
                    "lr_p": round(sp_result.get("lr_p", np.nan), 4),
                    "N": sp_result["N"],
                    "events": sp_result["events"],
                })

        # ── Lag sensitivity: primary outcome only ────────────────────
        if outcome == PRIMARY_OUTCOME:
            df_lag = apply_lag_filter(df_cc, "time", "event", LAG_YEARS)
            lag_result = fit_baseline_cox(
                df_lag, "time", "event", prod_grp_col, extended_covariates
            )
            if lag_result is not None:
                for _, srow in lag_result["summary"].iterrows():
                    cov = srow.get("covariate", srow.name)
                    if not str(cov).startswith("prod_"):
                        continue
                    primary_hr = next(
                        (r["HR"] for r in baseline_rows
                         if r["outcome"] == outcome
                         and r["prodromal_var"] == prod_var
                         and str(r.get("covariate", "")).startswith("prod_")),
                        np.nan,
                    )
                    lag_rows.append({
                        "outcome": outcome,
                        "prodromal_var": prod_var,
                        "prodromal_label": prod_label,
                        "covariate": cov,
                        "HR_primary": primary_hr,
                        "HR_lag2y": round(srow.get("exp(coef)", np.nan), 4),
                        "HR_lag2y_lower": round(srow.get("exp(coef) lower 95%", np.nan), 4),
                        "HR_lag2y_upper": round(srow.get("exp(coef) upper 95%", np.nan), 4),
                        "p_lag2y": srow.get("p", np.nan),
                        "N_lag": lag_result["N"],
                        "events_lag": lag_result["events"],
                    })

        # ── Models 2–3 per method ────────────────────────────────────
        for method in methods:
            rbd_col = col_risk_group_agnostic(method)
            if rbd_col not in df_cc.columns:
                if rbd_col in df_surv.columns:
                    df_cc[rbd_col] = df_surv.loc[df_cc.index, rbd_col].values
                else:
                    continue

            df_cc_m = df_cc.loc[df_cc[rbd_col].notna()].copy()
            df_cc_m[rbd_col] = df_cc_m[rbd_col].astype(str)

            if df_cc_m.empty or df_cc_m["event"].sum() < MIN_EVENTS_FOR_MODEL:
                continue

            df_cc_m["combined_grp"] = (
                df_cc_m[rbd_col] + " / " + df_cc_m[prod_grp_col].astype(str)
            )

            # Inline KM plot rendering — no deferred queue
            df_full_for_km = df_surv[time_event_mask & rbd_masks[rbd_col]].copy()
            df_full_for_km[rbd_col] = df_full_for_km[rbd_col].astype(str)

            km_pvals = plot_three_panel_km(
                df_cc_m, df_full_for_km, "time", "event",
                rbd_col, prod_grp_col, "combined_grp",
                outcome.replace("_", " ").title(),
                method.upper(), prod_label,
                str(path_outcome / f"KM_{method}_{prod_var}.png"),
            )
            km_lr_rows.append({
                "outcome": outcome, "method": method,
                "prodromal_var": prod_var,
                "prodromal_label": prod_label,
                "N_cc": len(df_cc_m),
                "events_cc": int(df_cc_m["event"].sum()),
                "N_full": len(df_full_for_km),
                "events_full": int(df_full_for_km["event"].sum()),
                **km_pvals,
            })

            # Model 2: Additive
            add_result = fit_additive_cox(
                df_cc_m, "time", "event", rbd_col,
                prod_grp_col, extended_covariates,
            )
            if add_result is not None:
                extra_add = {
                    "outcome": outcome, "method": method,
                    "prodromal_var": prod_var,
                    "prodromal_label": prod_label,
                    "model": "M2_additive",
                }
                additive_rows.extend(_flatten_summary(add_result, extra_add))
                model_fit_rows.append(_extract_fit_row(
                    add_result,
                    {"outcome": outcome, "model": "M2_additive",
                     "method": method, "prodromal_var": prod_var,
                     "prodromal_label": prod_label},
                ))

            # Model 3: Interaction
            int_result = fit_interaction_cox(
                df_cc_m, "time", "event", rbd_col,
                prod_grp_col, extended_covariates,
            )
            if int_result is not None:
                extra_int = {
                    "outcome": outcome, "method": method,
                    "prodromal_var": prod_var,
                    "prodromal_label": prod_label,
                    "model": "M3_interaction",
                }
                interaction_rows.extend(_flatten_summary(int_result, extra_int))
                model_fit_rows.append(_extract_fit_row(
                    int_result,
                    {"outcome": outcome, "model": "M3_interaction",
                     "method": method, "prodromal_var": prod_var,
                     "prodromal_label": prod_label},
                ))

            # Absolute risks (primary method only)
            if method == PRIMARY_METHOD:
                ar_df = compute_absolute_risks_km(
                    df_cc_m, "time", "event", "combined_grp",
                    ABSOLUTE_RISK_TIMEPOINTS,
                )
                if ar_df is not None and not ar_df.empty:
                    ar_df["outcome"] = outcome
                    ar_df["prodromal_var"] = prod_var
                    ar_df["prodromal_label"] = prod_label
                    abs_risk_rows.append(ar_df)

            # Additive interaction (primary outcome + method, binary vars only).
            # RBD is binarised below (High=1, all else=0), so this works for
            # both 2g and 3g stratifications.
            if (
                run_additive_interaction
                and outcome == PRIMARY_OUTCOME
                and method == PRIMARY_METHOD
                and prod_var in PRODROMAL_BINARY_VARS
            ):
                rbd_binary = (
                    df_cc_m[rbd_col]
                    .map(lambda x: 1 if "high" in str(x).lower() else 0)
                )
                prod_binary = (
                    df_cc_m[prod_grp_col]
                    .map(lambda x: 1 if str(x).lower() in ("yes", "high") else 0)
                )
                df_ai = df_cc_m.copy()
                df_ai["rbd_bin"] = rbd_binary
                df_ai["prod_bin"] = prod_binary

                ai_result = bootstrap_additive_interaction(
                    df_ai, "time", "event",
                    "rbd_bin", "prod_bin", extended_covariates,
                    n_bootstrap=n_bootstrap, seed=seed,
                    n_jobs=n_bootstrap_jobs,
                )
                if ai_result is not None:
                    reri_rows.append({
                        "outcome": outcome,
                        "prodromal_var": prod_var,
                        "prodromal_label": prod_label,
                        "reri": round(ai_result.reri, 4),
                        "reri_lci": round(ai_result.reri_lci, 4),
                        "reri_uci": round(ai_result.reri_uci, 4),
                        "ap": round(ai_result.ap, 4),
                        "ap_lci": round(ai_result.ap_lci, 4),
                        "ap_uci": round(ai_result.ap_uci, 4),
                        "synergy_index": round(ai_result.synergy_index, 4),
                        "si_lci": round(ai_result.si_lci, 4),
                        "si_uci": round(ai_result.si_uci, 4),
                        "hr_11": round(ai_result.hr_11, 4),
                        "hr_10": round(ai_result.hr_10, 4),
                        "hr_01": round(ai_result.hr_01, 4),
                        "N": ai_result.N,
                        "events": ai_result.events,
                        "n_bootstrap": ai_result.n_bootstrap,
                        "n_00": ai_result.n_00,
                        "events_00": ai_result.events_00,
                        "n_10": ai_result.n_10,
                        "events_10": ai_result.events_10,
                        "n_01": ai_result.n_01,
                        "events_01": ai_result.events_01,
                        "n_11": ai_result.n_11,
                        "events_11": ai_result.events_11,
                        "sparse_cell_warning": ai_result.sparse_cell,
                    })

                # Poisson RERI sensitivity (same binary vars)
                poisson_result = compute_poisson_reri(
                    df_ai, "time", "event",
                    "rbd_bin", "prod_bin", extended_covariates,
                )
                if poisson_result is not None:
                    poisson_reri_rows.append({
                        "outcome": outcome,
                        "prodromal_var": prod_var,
                        "prodromal_label": prod_label,
                        **poisson_result,
                    })

    _t0 = _log_timing("Models 1-3 inner loop (all prodromal vars)", _t0)

    # ── TMT SENSITIVITY: M1–M3 on TMT-assessed subcohort (N≈46k) ─────
    # Kept separate from the primary loop to preserve full-cohort N for
    # all other prodromal markers. Results tagged analysis="TMT_sensitivity"
    # and written to sensitivity_rows (supplementary table).
    tmt_active = filter_active_variables(df_surv_cov, PRODROMAL_TMT_VARS)
    for prod_var, prod_label in tmt_active.items():
        if prod_var not in df_surv_cov.columns:
            continue

        df_tmt = df_surv_cov.loc[df_surv_cov[prod_var].notna()].copy()
        if df_tmt.empty or df_tmt["event"].sum() < MIN_EVENTS_FOR_MODEL:
            continue

        prod_grp_col = f"{prod_var}_grp"
        df_tmt[prod_grp_col] = discretize_prodromal(df_tmt, prod_var)

        # M1: prodromal-only baseline
        tmt_m1 = fit_baseline_cox(
            df_tmt, "time", "event", prod_grp_col, extended_covariates
        )
        if tmt_m1 is not None:
            sensitivity_rows.extend(_flatten_summary(tmt_m1, {
                "outcome": outcome,
                "prodromal_var": prod_var,
                "prodromal_label": prod_label,
                "analysis": "TMT_sensitivity_M1",
                "N_sensitivity": tmt_m1["N"],
                "events_sensitivity": tmt_m1["events"],
            }))

        # M2 + M3: additive and interaction with RBD, per method
        for method in methods:
            rbd_col = col_risk_group_agnostic(method)
            if rbd_col not in df_tmt.columns:
                if rbd_col in df_surv.columns:
                    df_tmt[rbd_col] = df_surv.loc[df_tmt.index, rbd_col].values
                else:
                    continue

            df_tmt_m = df_tmt.loc[df_tmt[rbd_col].notna()].copy()
            df_tmt_m[rbd_col] = df_tmt_m[rbd_col].astype(str)
            if df_tmt_m.empty or df_tmt_m["event"].sum() < MIN_EVENTS_FOR_MODEL:
                continue

            tmt_m2 = fit_additive_cox(
                df_tmt_m, "time", "event",
                rbd_col, prod_grp_col, extended_covariates,
            )
            if tmt_m2 is not None:
                sensitivity_rows.extend(_flatten_summary(tmt_m2, {
                    "outcome": outcome,
                    "prodromal_var": prod_var,
                    "prodromal_label": prod_label,
                    "method": method,
                    "analysis": "TMT_sensitivity_M2",
                    "N_sensitivity": tmt_m2["N"],
                    "events_sensitivity": tmt_m2["events"],
                }))

            tmt_m3 = fit_interaction_cox(
                df_tmt_m, "time", "event",
                rbd_col, prod_grp_col, extended_covariates,
            )
            if tmt_m3 is not None:
                sensitivity_rows.extend(_flatten_summary(tmt_m3, {
                    "outcome": outcome,
                    "prodromal_var": prod_var,
                    "prodromal_label": prod_label,
                    "method": method,
                    "analysis": "TMT_sensitivity_M3",
                    "N_sensitivity": tmt_m3["N"],
                    "events_sensitivity": tmt_m3["events"],
                }))

    _t0 = _log_timing("TMT sensitivity loop", _t0)

    # ── PH TIME-VARYING SENSITIVITY (primary outcome only) ────────────
    if outcome == PRIMARY_OUTCOME and ph_rows:
        ph_violator_names = list({
            r["covariate"] for r in ph_rows
            if r.get("ph_violation") and r["outcome"] == outcome
        })
        if ph_violator_names:
            ph_tv_df = fit_time_interaction_sensitivity(
                df_surv, "time", "event", ph_violator_names, extended_covariates,
            )
            if not ph_tv_df.empty:
                ph_tv_df["outcome"] = outcome
                ph_time_rows.append(ph_tv_df)

    _t0 = _log_timing("PH time-varying sensitivity", _t0)

    # ── DISCRIMINATION & CALIBRATION ─────────────────────────────────
    if outcome == PRIMARY_OUTCOME:
        rbd_col_disc = col_risk_group_agnostic(PRIMARY_METHOD)
        if rbd_col_disc in df_surv.columns and "abk_rbd_score_mean" in df_surv.columns:
            # Age + sex only (Model X-base)
            base_only_covs = [c for c in ["cov_age_recruitment_21022", "cov_sex_31"]
                              if c in df_surv.columns]
            # Full covariates + RBD (Model A columns)
            model_a_cols = extended_covariates + ["abk_rbd_score_mean"]
            model_a_cols = [c for c in model_a_cols if c in df_surv.columns]

            # Delta-C: Model A vs covariates-only (null)
            try:
                dc_result = bootstrap_delta_c_test(
                    df_surv, "time", "event",
                    model_full_cols=model_a_cols,
                    model_null_cols=extended_covariates,
                    n_bootstrap=n_bootstrap, seed=seed,
                    n_jobs=n_bootstrap_jobs,
                )
                if dc_result is not None:
                    disc_rows.append({
                        "outcome": outcome,
                        "comparison": "Model_A_vs_covariates",
                        "model_full": "RBD + covariates",
                        "model_null": "covariates only",
                        **dc_result,
                    })
            except Exception as _dc_exc:
                warnings.warn(f"Delta-C (A vs cov) failed: {_dc_exc}")

            # Delta-C: age+sex only vs age+sex+RBD
            if len(base_only_covs) >= 2:
                try:
                    dc_base = bootstrap_delta_c_test(
                        df_surv, "time", "event",
                        model_full_cols=base_only_covs + ["abk_rbd_score_mean"],
                        model_null_cols=base_only_covs,
                        n_bootstrap=n_bootstrap, seed=seed,
                        n_jobs=n_bootstrap_jobs,
                    )
                    if dc_base is not None:
                        disc_rows.append({
                            "outcome": outcome,
                            "comparison": "RBD_over_age_sex",
                            "model_full": "age + sex + RBD",
                            "model_null": "age + sex only",
                            **dc_base,
                        })
                except Exception as _dc_exc:
                    warnings.warn(f"Delta-C (RBD over age+sex) failed: {_dc_exc}")

            # ── Delta-C, NRI, IDI: RBD + prodromal vs RBD alone ──────────────────
            # PRIMARY DISCRIMINATION QUESTION: Do prodromal variables meaningfully
            # improve prediction of {outcome} beyond the RBD score alone?
            #
            # This comparison evaluates the incremental discriminative contribution
            # of prodromal markers (cognitive + binary behavioral) when added to an
            # RBD + demographics baseline model. Uses bootstrap-estimated delta-C,
            # NRI (category-based reclassification), and IDI (continuous discrimination
            # slope improvement) to assess clinical utility of the expanded model.
            #
            # Models compared:
            #   - Full model: RBD + all active prodromal vars + demographics
            #   - Null model: RBD + demographics only (no prodromal)
            #
            prodromal_cols = [c for c in active_vars.keys() if c in df_surv.columns]
            if prodromal_cols:
                model_rbd_only_cols = ["abk_rbd_score_mean"] + extended_covariates
                model_rbd_only_cols = [c for c in model_rbd_only_cols if c in df_surv.columns]

                model_rbd_prodromal_cols = model_rbd_only_cols + prodromal_cols

                # Delta-C bootstrap test
                try:
                    dc_prod = bootstrap_delta_c_test(
                        df_surv, "time", "event",
                        model_full_cols=model_rbd_prodromal_cols,
                        model_null_cols=model_rbd_only_cols,
                        n_bootstrap=n_bootstrap, seed=seed,
                        n_jobs=n_bootstrap_jobs,
                    )
                    if dc_prod is not None:
                        disc_rows.append({
                            "outcome": outcome,
                            "comparison": "RBD_prodromal_vs_RBD_alone",
                            "model_full": "RBD + prodromal + demographics",
                            "model_null": "RBD + demographics only",
                            **dc_prod,
                        })
                except Exception as _dc_prod_exc:
                    warnings.warn(f"Delta-C (RBD+prodromal vs RBD) failed: {_dc_prod_exc}")

                # NRI & IDI: RBD + prodromal vs RBD alone
                try:
                    from lifelines import CoxPHFitter as _CoxPH
                    _all_cols_prod = list(set(model_rbd_prodromal_cols + model_rbd_only_cols +
                                             ["time", "event"]))
                    _df_prod_nri = df_surv[_all_cols_prod].dropna().copy()
                    if _df_prod_nri["event"].sum() >= MIN_EVENTS_FOR_MODEL:
                        # Fit both models
                        _cph_prod_full = _CoxPH(penalizer=RIDGE_PENALIZER)
                        _cph_prod_full.fit(
                            _df_prod_nri[["time", "event"] + model_rbd_prodromal_cols],
                            duration_col="time", event_col="event", robust=False,
                        )
                        _cph_prod_null = _CoxPH(penalizer=RIDGE_PENALIZER)
                        _cph_prod_null.fit(
                            _df_prod_nri[["time", "event"] + model_rbd_only_cols],
                            duration_col="time", event_col="event", robust=False,
                        )
                        _risk_prod_full = extract_predicted_risks(_cph_prod_full, _df_prod_nri, "time", "event")
                        _risk_prod_null = extract_predicted_risks(_cph_prod_null, _df_prod_nri, "time", "event")
                        _events_prod = _df_prod_nri["event"].values

                        # NRI at median predicted risk (full model)
                        _median_risk_prod = float(np.median(_risk_prod_full))
                        _nri_prod = compute_nri(_risk_prod_null, _risk_prod_full, _events_prod, _median_risk_prod)
                        disc_rows.append({
                            "outcome": outcome,
                            "comparison": "NRI_RBD_prodromal_vs_RBD_alone",
                            "model_full": "RBD + prodromal + demographics",
                            "model_null": "RBD + demographics only",
                            "nri_threshold": round(_median_risk_prod, 6),
                            "nri_threshold_method": "median predicted risk (full model)",
                            **_nri_prod,
                        })

                        # IDI (continuous discrimination improvement)
                        _idi_prod = compute_idi(_risk_prod_null, _risk_prod_full, _events_prod)
                        disc_rows.append({
                            "outcome": outcome,
                            "comparison": "IDI_RBD_prodromal_vs_RBD_alone",
                            "model_full": "RBD + prodromal + demographics",
                            "model_null": "RBD + demographics only",
                            **_idi_prod,
                        })
                except Exception as _prod_nri_exc:
                    warnings.warn(f"NRI/IDI (RBD+prodromal vs RBD) failed: {_prod_nri_exc}")

            # NRI & IDI: Model A vs covariates-only
            try:
                from lifelines import CoxPHFitter as _CoxPH
                _all_cols = list(set(model_a_cols + extended_covariates +
                                     ["time", "event"]))
                _df_nri = df_surv[_all_cols].dropna().copy()
                if _df_nri["event"].sum() >= MIN_EVENTS_FOR_MODEL:
                    # Fit both models
                    _cph_full = _CoxPH(penalizer=RIDGE_PENALIZER)
                    _cph_full.fit(
                        _df_nri[["time", "event"] + model_a_cols],
                        duration_col="time", event_col="event", robust=False,
                    )
                    _cph_null = _CoxPH(penalizer=RIDGE_PENALIZER)
                    _cph_null.fit(
                        _df_nri[["time", "event"] + extended_covariates],
                        duration_col="time", event_col="event", robust=False,
                    )
                    _risk_full = extract_predicted_risks(_cph_full, _df_nri, "time", "event")
                    _risk_null = extract_predicted_risks(_cph_null, _df_nri, "time", "event")
                    _events = _df_nri["event"].values

                    # NRI at median predicted risk
                    _median_risk = float(np.median(_risk_full))
                    _nri = compute_nri(_risk_null, _risk_full, _events, _median_risk)
                    disc_rows.append({
                        "outcome": outcome,
                        "comparison": "NRI_Model_A_vs_covariates",
                        "model_full": "RBD + covariates",
                        "model_null": "covariates only",
                        "nri_threshold": round(_median_risk, 6),
                        "nri_threshold_method": "median predicted risk (full model)",
                        **_nri,
                    })

                    # IDI
                    _idi = compute_idi(_risk_null, _risk_full, _events)
                    disc_rows.append({
                        "outcome": outcome,
                        "comparison": "IDI_Model_A_vs_covariates",
                        "model_full": "RBD + covariates",
                        "model_null": "covariates only",
                        **_idi,
                    })

                    # Calibration
                    _cal_slope = calibration_slope(_cph_full, _df_nri, "time", "event")
                    if _cal_slope is not None:
                        cal_rows.append({
                            "outcome": outcome,
                            "model": "Model_A",
                            "metric": "calibration_slope",
                            **_cal_slope,
                        })
                    _cal_large = calibration_in_the_large(_cph_full, _df_nri, "time", "event")
                    if _cal_large is not None:
                        cal_rows.append({
                            "outcome": outcome,
                            "model": "Model_A",
                            "metric": "calibration_in_the_large",
                            **_cal_large,
                        })
            except Exception as _disc_exc:
                warnings.warn(f"NRI/IDI/calibration failed for {outcome}: {_disc_exc}")

    _t0 = _log_timing("Discrimination & Calibration (bootstrap)", _t0)

    # ── MODEL 4: Competing risks (primary outcome only) ─────────────────
    if (
        run_competing
        and outcome in COMPETING_OUTCOMES
        and outcome == PRIMARY_OUTCOME
    ):
        comp_outcomes = COMPETING_OUTCOMES[outcome]
        try:
            durations, event_ind = encode_competing_events(
                df_surv, outcome, comp_outcomes
            )
            df_surv["comp_event"] = event_ind
            df_surv["comp_time"] = durations

            rbd_col_comp = col_risk_group_agnostic(PRIMARY_METHOD)
            if rbd_col_comp in df_surv.columns:
                cif_km_df = compare_cif_vs_km(
                    df_surv, "comp_time", "event",
                    "comp_event", rbd_col_comp,
                )
                if not cif_km_df.empty:
                    cif_km_df["outcome"] = outcome
                    comp_cif_rows.append(cif_km_df)

                cs_result = fit_cause_specific_cox(
                    df_surv, "comp_time", "event",
                    rbd_col_comp, extended_covariates,
                )
                if cs_result is not None:
                    for _, srow in cs_result["summary"].iterrows():
                        comp_cox_rows.append({
                            "outcome": outcome,
                            "covariate": srow.get("covariate", srow.name),
                            "HR": round(srow.get("exp(coef)", np.nan), 4),
                            "HR_lower": round(srow.get("exp(coef) lower 95%", np.nan), 4),
                            "HR_upper": round(srow.get("exp(coef) upper 95%", np.nan), 4),
                            "p": srow.get("p", np.nan),
                            "c_index": round(cs_result["c_index"], 4),
                            "N": cs_result["N"],
                            "events": cs_result["events"],
                        })
        except Exception as exc:
            warnings.warn(f"Competing risk analysis failed for {outcome}: {exc}")

    _log_timing("Competing risks (Model 4)", _t0)
    _log_timing(f"TOTAL {outcome}", _t_outcome_start)

    return {
        "cohort_rows": cohort_rows,
        "model_a_rows": model_a_rows,
        "model_a_interaction_rows": model_a_interaction_rows,
        "model_a_disc_comp_rows": model_a_disc_comp_rows,
        "model_f_rows": model_f_rows,
        "model_f_strat_rows": model_f_strat_rows,
        "model_g_rows": model_g_rows,
        "model_g_cell_rows": model_g_cell_rows,
        "baseline_rows": baseline_rows,
        "ph_rows": ph_rows,
        "c_index_rows": c_index_rows,
        "rbd_only_rows": rbd_only_rows,
        "rbd_cont_rows": rbd_cont_rows,
        "rbd_thresh_all": rbd_thresh_all,
        "additive_rows": additive_rows,
        "interaction_rows": interaction_rows,
        "km_lr_rows": km_lr_rows,
        "abs_risk_rows": abs_risk_rows,
        "spline_rows": spline_rows,
        "rbd_spline_rows": rbd_spline_rows,
        "lag_rows": lag_rows,
        "reri_rows": reri_rows,
        "sensitivity_rows": sensitivity_rows,
        "comp_cif_rows": comp_cif_rows,
        "comp_cox_rows": comp_cox_rows,
        "model_fit_rows": model_fit_rows,
        "screening_rows": screening_rows,
        "age_strat_rows": age_strat_rows,
        "ph_time_rows": ph_time_rows,
        "poisson_reri_rows": poisson_reri_rows,
        "disc_rows": disc_rows,
        "cal_rows": cal_rows,
    }


def _empty_result() -> Dict[str, Any]:
    """Return an empty result dict with the correct keys."""
    return {
        "cohort_rows": [], "model_a_rows": [],
        "model_a_interaction_rows": [], "model_a_disc_comp_rows": [],
        "model_f_rows": [], "model_f_strat_rows": [],
        "model_g_rows": [], "model_g_cell_rows": [],
        "baseline_rows": [], "ph_rows": [],
        "c_index_rows": [], "rbd_only_rows": [], "rbd_cont_rows": [],
        "rbd_thresh_all": [], "additive_rows": [], "interaction_rows": [],
        "km_lr_rows": [], "abs_risk_rows": [], "spline_rows": [],
        "rbd_spline_rows": [], "lag_rows": [], "reri_rows": [],
        "sensitivity_rows": [], "comp_cif_rows": [], "comp_cox_rows": [],
        "model_fit_rows": [], "screening_rows": [], "age_strat_rows": [],
        "ph_time_rows": [], "poisson_reri_rows": [],
        "disc_rows": [], "cal_rows": [],
    }


def _merge_result(
    acc: Dict[str, Any],
    incoming: Dict[str, Any],
) -> None:
    """In-place merge of an outcome worker result into the accumulator."""
    for key, val in incoming.items():
        acc[key].extend(val)


# ── Column pruning for IPC overhead reduction ─────────────────────────────

def _compute_needed_columns(
    df: pd.DataFrame,
    outcomes: List[str],
    active_vars: Dict[str, str],
    extended_covariates: List[str],
    methods: List[str],
) -> List[str]:
    """
    Identify the minimal set of columns needed by ``_process_one_outcome``.

    Reduces pickle/IPC overhead by stripping columns that no worker touches.

    These are the columns:
    ['ad_flag__surv_event',
     'cov_bmi',
     'control',
     'cov_age_recruitment_21022',
     'cov_alcohol',
     'cov_fi_questions_attempted_20128_bl',
     'cov_fluid_intelligence_20016_bl',
     'cov_numeric_memory_max_20240_bl',
     'cov_pairs_status_20244_bl',
     'cov_react_time_mean_20023_bl',
     'cog_fluid_intelligence_bl', 'cog_react_time_bl',
     'cog_numeric_memory_bl', 'cog_pairs_matching_bl',  -> semantic baseline aliases
     'cog_fluid_intelligence_fu', 'cog_react_time_fu',  -> follow-up
     'cog_fluid_intelligence_delta', 'cog_react_time_delta',  -> change
     'cov_sex_31',
     'cov_smoking',
     'death_date',
     'death_flag',
     'dem_flag__surv_event',
     'eid',
     'hes_gap_pre_baseline_years',
     'id',
     'outcome_1a_pd_only', -> PD columns all diagnosis
     'outcome_1a_pd_only__incident',
     'outcome_1a_pd_only__prevalent',
     'outcome_1a_pd_only__surv_days',
     'outcome_1a_pd_only__surv_event',
     'outcome_1b_pd_ad__incident',
     'outcome_1b_pd_ad__surv_days',
     'outcome_1b_pd_ad__surv_event',
     'outcome_2a_vasculardementia__incident',
     'outcome_2a_vasculardementia__surv_days',
     'outcome_2a_vasculardementia__surv_event',
     'outcome_2b_pd_vasculardementia__incident',
     'outcome_2b_pd_vasculardementia__surv_days',
     'outcome_2b_pd_vasculardementia__surv_event',
     'outcome_4a_ad_only__incident',
     'outcome_4a_ad_only__surv_days',
     'outcome_4a_ad_only__surv_event',
     'pd_flag__surv_event',
     'prodromal_anxiety_bl',
     'prodromal_constipation_bl',
     'prodromal_depression_bl',
     'prodromal_erectile_dysfunction_bl',
     'prodromal_orthostatic_bl',
     'prodromal_constipation_post', 'prodromal_depression_post',  -> incident post-baseline
     'prodromal_burden_post',  -> incident post-baseline burden
     'abk_rbd_score_mean', -> RBD score
     'rg_pctl2',
     'rg_pctl3', -> RBD groups
     'rg_q4',
     'cog_tmt_ratio_log_bl',
     'wear_time_start']


    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (used to check which columns actually exist).
    outcomes : list[str]
        Outcome keys being analysed.
    active_vars : dict[str, str]
        Active prodromal variable mapping.
    extended_covariates : list[str]
        Adjustment covariate column names.
    methods : list[str]
        RBD stratification method names.

    Returns
    -------
    list[str]
        Deduplicated column names present in ``df``.
    """
    needed: set = set()

    # Identity + control flag
    needed.update(["eid", "id", "control"])

    # Covariates
    needed.update(extended_covariates)

    # Model A: PRS + ancestry PCs (for RBD + genetic liability models)
    needed.update(PRS_COLS)
    needed.update(MODEL_A_COVARIATES)

    # Model G: GBA carrier
    needed.add(GBA_COL)

    # Prodromal variables (primary + TMT sensitivity)
    needed.update(active_vars.keys())
    needed.update(PRODROMAL_TMT_VARS.keys())

    # RBD probability + risk group columns
    needed.add("abk_rbd_score_mean")
    for method in methods:
        rbd_col = col_risk_group_agnostic(method)
        needed.add(rbd_col)

    # Agnostic risk columns
    needed.update(AGNOSTIC_RISK_COLS)

    # Per-outcome survival + incident columns
    for outcome in outcomes:
        needed.add(col_surv_time(outcome))
        needed.add(col_surv_event(outcome))
        needed.add(col_incident(outcome))

    # Competing risk: death columns + all outcome surv columns
    needed.update(["death_flag", "death_date", "wear_time_start"])
    for col in df.columns:
        if col.endswith("_surv_event") or col.endswith("_surv_time"):
            needed.add(col)

    # HES gap column for sensitivity analysis
    needed.add(HES_GAP_COL)

    # Keep only columns that actually exist
    return [c for c in needed if c in df.columns]


# ── Main pipeline ──────────────────────────────────────────────────────────

def run_prodromal_pipeline(
    outcomes: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    run_competing: bool = True,
    run_additive_interaction: bool = True,
    run_nri_idi: bool = True,
    run_calibration: bool = True,
    run_mediation_analysis: bool = RUN_MEDIATION,
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    n_bootstrap_jobs: int = BOOTSTRAP_JOBS,
) -> None:
    """
    Full Cox prodromal analysis pipeline.

    Orchestration sequence per outcome:
    1. Load & prepare data
    2. Model 0 (RBD-only: categorical + continuous + stability + spline)
    3. Per prodromal variable:
       a. Model 1 (baseline)
       b. Per method: Model 2 (additive), Model 3 (interaction), KM plots
       c. Additive interaction (RERI/AP/SI)
       d. Spline analysis (continuous vars, primary outcome only)
       e. Lag sensitivity (primary outcome only)
    4. Model 4 (competing risks, primary outcome only)
    5. Discrimination (delta-C, NRI, IDI) and calibration
    6. FDR correction, save tables, generate report

    Outcomes are processed in parallel using ``ProcessPoolExecutor``
    (max workers = min(n_outcomes, cpu_count - 1)).

    Parameters
    ----------
    outcomes : list[str], optional
        Outcomes to analyse (default: all from OUTCOMES).
    methods : list[str], optional
        RBD stratification methods (default: percentile_2g, percentile_3g).
    run_competing : bool
        Whether to run Model 4 competing risk analysis.
    run_additive_interaction : bool
        Whether to run RERI/AP/SI bootstrap analysis.
    run_nri_idi : bool
        Whether to compute NRI and IDI.
    run_calibration : bool
        Whether to compute calibration metrics.
    run_mediation_analysis : bool
        Whether to run the RBD-prodromal mediation analysis (Interpretations A
        and C) after the main Cox pipeline. Default follows RUN_MEDIATION in
        cox_config.
    n_bootstrap : int
        Number of bootstrap resamples for delta-C, RERI, etc.
    seed : int
        Random seed for all bootstrap procedures.
    """
    outcomes = outcomes or OUTCOMES
    methods = methods or METHODS

    # ── 1. Paths ───────────────────────────────────────────────────────
    from datetime import datetime
    timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    path_results = config["results"]["root"] / f"cox_prodromal_abk_{timestamp}"
    path_report = path_results / "report"
    path_results.mkdir(parents=True, exist_ok=True)
    path_report.mkdir(parents=True, exist_ok=True)

    # ── 2. Load data ───────────────────────────────────────────────────
    print("[1/7] Loading data ...")
    thresholds, df_risk = load_prodromal_dataset()
    print(f'\t\t Risk Data Dim: {df_risk.shape} | Unique subjects: {df_risk['id'].nunique()}')
    print("[2/7] Preparing covariates ...")
    df_risk, extended_covariates = build_extended_covariates(df_risk, BASE_COVARIATES)

    # ── Impute missing lifestyle covariates (smoking, alcohol) with median ────
    # Same method as used for BMI. Prevents loss of cases in downstream analyses
    # (e.g., spline model .dropna() call).
    for cov in ["cov_smoking", "cov_alcohol"]:
        if cov in df_risk.columns:
            n_missing_before = df_risk[cov].isna().sum()
            if n_missing_before > 0:
                median_val = df_risk[cov].median()
                df_risk[cov] = df_risk[cov].fillna(median_val)
                n_missing_after = df_risk[cov].isna().sum()
                print(f"  Imputed {cov}: {n_missing_before:,} to {n_missing_after:,} missing (median={median_val:.2f})")

    # ── 2.5 RBD score distribution ────────────────────────────────────
    print("[2.5/7] Saving RBD score distribution ...")
    try:
        plot_rbd_distribution_single(
            df=df_risk,
            prob_col="rbd_prob",
            group_col="rg_pctl3",
            incident_col="outcome_1a_pd_only__incident",
            group_order=["Low", "Mid", "High"],
            save_path=path_results,
            filename_stem="rbd_score_distribution",
        )
    except Exception as _exc:
        print(f"  [WARN] RBD distribution plot skipped: {_exc}")

    # ── 3. Data availability ───────────────────────────────────────────
    print("[3/7] Data availability ...")
    all_vars = {**PRODROMAL_VARS, **PRODROMAL_BINARY_VARS}
    avail_df = build_availability_table(df_risk, all_vars)
    save_table(avail_df, path_results / "data_availability_report.csv")
    save_table(avail_df, path_report / "table_2_availability.csv")

    active_vars = filter_active_variables(df_risk, all_vars)

    # ── 4. Parallel analysis loop ──────────────────────────────────────
    n_workers = min(len(outcomes), MAX_WORKERS)
    print(f"[4/7] Running models ({len(outcomes)} outcomes, {n_workers} workers, "
          f"{n_bootstrap_jobs} bootstrap jobs/worker) ...\n")

    # Column-prune df_risk to reduce IPC pickle overhead (~600 MB → ~100 MB)
    needed_cols = _compute_needed_columns(
        df_risk, outcomes, active_vars, extended_covariates, methods,
    )
    needed_cols = needed_cols + ['outcome_1a_pd_only__incident', 'outcome_1a_pd_only__prevalent', 'outcome_1a_pd_only']
    needed_cols = sorted(list(set(needed_cols)))
    df_risk_slim = df_risk[needed_cols].copy()

    print(f"  Column pruning: {df_risk.shape[1]} -> {df_risk_slim.shape[1]} columns "
          f"(~{df_risk_slim.memory_usage(deep=True).sum() / 1e6:.0f} MB)")

    # Accumulator with correct keys, pre-initialised to empty lists
    accumulated = _empty_result()

    if n_workers <= 1:
        # Serial fallback: avoids spawn overhead for single-outcome runs
        for i, outcome in enumerate(tqdm(outcomes, desc="Outcomes")):
            outcome_seed = seed + i * 1000
            result = _process_one_outcome(
                outcome=outcome,
                df_risk=df_risk_slim,
                active_vars=active_vars,
                extended_covariates=extended_covariates,
                methods=methods,
                path_results=path_results,
                run_competing=run_competing,
                run_additive_interaction=run_additive_interaction,
                n_bootstrap=n_bootstrap,
                seed=outcome_seed,
                n_bootstrap_jobs=n_bootstrap_jobs,
            )

            _merge_result(accumulated, result)
    else:
        futures_map = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for i, outcome in enumerate(outcomes):
                outcome_seed = seed + i * 1000
                future = executor.submit(
                    _process_one_outcome,
                    outcome, df_risk_slim, active_vars, extended_covariates,
                    methods, path_results, run_competing,
                    run_additive_interaction, n_bootstrap, outcome_seed,
                    n_bootstrap_jobs,
                )
                futures_map[future] = outcome

            with tqdm(total=len(outcomes), desc="Outcomes") as pbar:
                for future in as_completed(futures_map):
                    outcome = futures_map[future]
                    try:
                        result = future.result()
                        _merge_result(accumulated, result)
                        pbar.set_postfix({"done": outcome})
                    except Exception as exc:
                        warnings.warn(f"Outcome {outcome} failed: {exc}")
                    pbar.update(1)

    # Unpack accumulators
    cohort_rows       = accumulated["cohort_rows"]
    model_a_rows      = accumulated["model_a_rows"]
    model_f_rows      = accumulated["model_f_rows"]
    model_f_strat_rows = accumulated["model_f_strat_rows"]
    model_g_rows      = accumulated["model_g_rows"]
    model_g_cell_rows = accumulated["model_g_cell_rows"]
    baseline_rows     = accumulated["baseline_rows"]
    ph_rows           = accumulated["ph_rows"]
    c_index_rows      = accumulated["c_index_rows"]
    rbd_only_rows     = accumulated["rbd_only_rows"]
    rbd_cont_rows     = accumulated["rbd_cont_rows"]
    rbd_thresh_all    = accumulated["rbd_thresh_all"]
    additive_rows     = accumulated["additive_rows"]
    interaction_rows  = accumulated["interaction_rows"]
    km_lr_rows        = accumulated["km_lr_rows"]
    abs_risk_rows     = accumulated["abs_risk_rows"]
    spline_rows       = accumulated["spline_rows"]
    rbd_spline_rows   = accumulated["rbd_spline_rows"]
    lag_rows          = accumulated["lag_rows"]
    reri_rows         = accumulated["reri_rows"]
    sensitivity_rows  = accumulated["sensitivity_rows"]
    comp_cif_rows     = accumulated["comp_cif_rows"]
    comp_cox_rows     = accumulated["comp_cox_rows"]
    model_a_interaction_rows = accumulated["model_a_interaction_rows"]
    model_a_disc_comp_rows   = accumulated["model_a_disc_comp_rows"]
    model_fit_rows    = accumulated["model_fit_rows"]
    screening_rows    = accumulated["screening_rows"]
    age_strat_rows    = accumulated["age_strat_rows"]
    ph_time_rows      = accumulated["ph_time_rows"]
    poisson_reri_rows = accumulated["poisson_reri_rows"]
    disc_rows         = accumulated["disc_rows"]
    cal_rows          = accumulated["cal_rows"]

    # ── 5. Post-processing ─────────────────────────────────────────────
    print("\n[5/7] Saving tables ...")

    df_cohort_tbl = pd.DataFrame(cohort_rows)
    save_table(df_cohort_tbl, path_report / "table_1_cohort.csv")

    # ── Model A: RBD + PRS + PCs ────────────────────────────────────────
    df_ma = pd.DataFrame(model_a_rows)
    df_ma_pd = pd.DataFrame()
    if not df_ma.empty:
        save_table(df_ma, path_results / "model_a_rbd_prs_cox.csv")
        df_ma_pd = df_ma[df_ma["outcome"] == PRIMARY_OUTCOME].copy()
        if not df_ma_pd.empty:
            save_table(df_ma_pd, path_report / "table_model_a_rbd_pd.csv")

    # ── Model A interaction + discrimination comparison (separate table) ────
    # Kept separate from model_a_rbd_prs_cox to avoid mixing additive and
    # interaction rows.  N is reported per row because the PRS-complete
    # subset is smaller than the full analytical cohort.
    df_int = pd.DataFrame([
        r for r in model_a_interaction_rows if r.get("_row_type") != "fit_metrics"
    ])
    df_int_fit = pd.DataFrame([
        r for r in model_a_interaction_rows if r.get("_row_type") == "fit_metrics"
    ])
    df_disc = pd.DataFrame(model_a_disc_comp_rows)

    if not df_int.empty or not df_disc.empty:
        xlsx_path = path_results / "model_a_rbd_prs_interaction_cox.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            if not df_int.empty:
                df_int_clean = df_int.drop(columns=["_row_type"], errors="ignore")
                df_int_clean.to_excel(writer, sheet_name="coefficients", index=False)
            if not df_int_fit.empty:
                df_int_fit_clean = df_int_fit.drop(columns=["_row_type"], errors="ignore")
                df_int_fit_clean.to_excel(writer, sheet_name="model_fit", index=False)
            if not df_disc.empty:
                df_disc.to_excel(writer, sheet_name="discrimination", index=False)
        print(f"  Saved interaction + discrimination table -> {xlsx_path}")

    # ── Model F: RBD strata × PRS_PD interaction ───────────────────────────
    df_mf = pd.DataFrame(model_f_rows)
    df_mf_strat = pd.DataFrame(model_f_strat_rows)

    if not df_mf.empty or not df_mf_strat.empty:
        xlsx_path = path_results / "model_f_rbd_prs_stratified_interaction.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            if not df_mf.empty:
                df_mf.to_excel(writer, sheet_name="full_model_coefficients", index=False)
            if not df_mf_strat.empty:
                df_mf_strat.to_excel(writer, sheet_name="stratified_prs_effects", index=False)
        print(f"  Saved Model F (RBD x PRS strata interaction) table -> {xlsx_path}")
        # Also save to report directory
        if not df_mf_strat.empty:
            save_table(df_mf_strat, path_report / "table_F_rbd_prs_stratified_interaction.csv")

    # ── Model G: RBD × GBA carrier interaction ───────────────────────────────
    df_mg_coef = pd.DataFrame([
        r for r in model_g_rows if r.get("_row_type") != "fit_metrics"
    ])
    df_mg_fit = pd.DataFrame([
        r for r in model_g_rows if r.get("_row_type") == "fit_metrics"
    ])
    df_mg_cells = pd.DataFrame(model_g_cell_rows)

    if not df_mg_coef.empty or not df_mg_cells.empty:
        xlsx_path = path_results / "model_g_rbd_gba_interaction.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            if not df_mg_coef.empty:
                df_mg_coef_clean = df_mg_coef.drop(columns=["_row_type"], errors="ignore")
                df_mg_coef_clean.to_excel(writer, sheet_name="coefficients", index=False)
            if not df_mg_fit.empty:
                df_mg_fit_clean = df_mg_fit.drop(columns=["_row_type"], errors="ignore")
                df_mg_fit_clean.to_excel(writer, sheet_name="model_fit", index=False)
            if not df_mg_cells.empty:
                df_mg_cells.to_excel(writer, sheet_name="cell_counts", index=False)
        print(f"  Saved Model G (RBD x GBA interaction) table -> {xlsx_path}")
        if not df_mg_cells.empty:
            save_table(df_mg_cells, path_report / "table_G_rbd_gba_cell_counts.csv")

    df_bl = pd.DataFrame(baseline_rows)
    df_bl_pd = pd.DataFrame()
    if not df_bl.empty:
        df_bl["is_prod_row"] = df_bl["covariate"].str.startswith("prod_")
        for out in df_bl["outcome"].unique():
            mask = (df_bl["outcome"] == out) & df_bl["is_prod_row"]
            df_bl.loc[mask, "p_fdr"] = apply_fdr(df_bl.loc[mask, "p"]).values
        df_bl.drop(columns=["is_prod_row"], inplace=True)
        save_table(df_bl, path_results / "baseline_cox_HRs.csv")
        df_bl_pd = df_bl[df_bl["outcome"] == PRIMARY_OUTCOME].copy()
        save_table(df_bl_pd, path_report / "table_5_baseline_cox_pd.csv")

    df_ph = pd.DataFrame(ph_rows)
    df_ph_summary = pd.DataFrame()
    if not df_ph.empty:
        save_table(df_ph, path_results / "ph_diagnostics.csv")
        save_table(df_ph, path_report / "table_3_ph_diagnostics.csv")
        df_ph_summary = summarize_ph_violations(df_ph)
        if not df_ph_summary.empty:
            save_table(df_ph_summary, path_results / "ph_violation_summary.csv")
            save_table(df_ph_summary, path_report / "table_3b_ph_violation_summary.csv")

    df_ci = pd.DataFrame(c_index_rows)
    df_ci_pd = pd.DataFrame()
    if not df_ci.empty:
        df_ci_pd = df_ci[df_ci["outcome"] == PRIMARY_OUTCOME].copy()
        save_table(df_ci, path_results / "c_index.csv")
        save_table(df_ci_pd, path_report / "table_11a_c_index.csv")

    df_rbd = pd.DataFrame(rbd_only_rows)
    df_rbd_pd = pd.DataFrame()
    if not df_rbd.empty:
        save_table(df_rbd, path_results / "rbd_only_cox.csv")
        df_rbd_pd = df_rbd[df_rbd["outcome"] == PRIMARY_OUTCOME].copy()
        save_table(df_rbd_pd, path_report / "table_4a_rbd_only_pd.csv")

    df_rbd_cont = pd.DataFrame(rbd_cont_rows)
    if not df_rbd_cont.empty:
        save_table(df_rbd_cont, path_results / "rbd_continuous.csv")
        save_table(df_rbd_cont, path_report / "table_4b_rbd_continuous.csv")

    df_thresh = pd.DataFrame()
    if rbd_thresh_all:
        df_thresh = pd.concat(rbd_thresh_all, ignore_index=True)
        save_table(df_thresh, path_results / "rbd_threshold_stability.csv")
        save_table(df_thresh, path_report / "table_4c_threshold_stability.csv")

    df_add = pd.DataFrame(additive_rows)
    df_add_pd = pd.DataFrame()
    if not df_add.empty:
        save_table(df_add, path_results / "additive_cox.csv")
        df_add_pd = df_add[
            (df_add["outcome"] == PRIMARY_OUTCOME)
            & (df_add["method"] == PRIMARY_METHOD)
        ].copy()
        save_table(df_add_pd, path_report / "table_6a_additive_pd.csv")

    df_int = pd.DataFrame(interaction_rows)
    df_int_pd = pd.DataFrame()
    if not df_int.empty:
        save_table(df_int, path_results / "interaction_cox.csv")
        df_int_pd = df_int[
            (df_int["outcome"] == PRIMARY_OUTCOME)
            & (df_int["method"] == PRIMARY_METHOD)
        ].copy()
        save_table(df_int_pd, path_report / "table_6b_interaction_pd.csv")

    df_reri = pd.DataFrame(reri_rows)
    if not df_reri.empty:
        save_table(df_reri, path_results / "additive_interaction.csv")
        save_table(df_reri, path_report / "table_7_additive_interaction.csv")

    df_lr = pd.DataFrame(km_lr_rows)
    if not df_lr.empty:
        save_table(df_lr, path_results / "km_logrank_summary.csv")

    df_ar_pd = pd.DataFrame()
    if abs_risk_rows:
        df_ar = pd.concat(abs_risk_rows, ignore_index=True)
        save_table(df_ar, path_results / "absolute_risks.csv")
        df_ar_pd = df_ar[df_ar["outcome"] == PRIMARY_OUTCOME].copy()
        save_table(df_ar_pd, path_report / "table_8_absolute_risks.csv")

    df_sp = pd.DataFrame(spline_rows)
    if not df_sp.empty:
        save_table(df_sp, path_results / "spline_cox.csv")
        save_table(df_sp, path_report / "table_9a_spline_cox.csv")

    df_rbd_sp = pd.DataFrame(rbd_spline_rows)
    if not df_rbd_sp.empty:
        save_table(df_rbd_sp, path_results / "rbd_spline.csv")
        save_table(df_rbd_sp, path_report / "table_9b_rbd_spline.csv")

    df_lag = pd.DataFrame(lag_rows)
    if not df_lag.empty:
        save_table(df_lag, path_results / "lag_sensitivity.csv")
        save_table(df_lag, path_report / "table_10_lag_sensitivity.csv")

    df_sens = pd.DataFrame(sensitivity_rows)
    if not df_sens.empty:
        save_table(df_sens, path_results / "sensitivity_hes_active.csv")
        save_table(df_sens, path_report / "table_10b_sensitivity_hes_active.csv")

    df_comp = pd.DataFrame()
    if comp_cif_rows:
        df_comp = pd.concat(comp_cif_rows, ignore_index=True)
        save_table(df_comp, path_results / "competing_risk_cif_vs_km.csv")
        save_table(df_comp, path_report / "table_12a_cif_vs_km.csv")

    df_comp_cox = pd.DataFrame(comp_cox_rows)
    if not df_comp_cox.empty:
        save_table(df_comp_cox, path_results / "competing_risk_cox.csv")
        save_table(df_comp_cox, path_report / "table_12b_competing_cox.csv")

    df_model_fit = pd.DataFrame(model_fit_rows)
    if not df_model_fit.empty:
        save_table(df_model_fit, path_results / "model_fit_summary.csv")
        save_table(df_model_fit, path_report / "table_13_model_fit.csv")

    df_screening = pd.DataFrame(screening_rows)
    if not df_screening.empty:
        save_table(df_screening, path_results / "screening_metrics.csv")
        save_table(df_screening, path_report / "table_15_screening_metrics.csv")

    df_age_strat = pd.DataFrame(age_strat_rows)
    if not df_age_strat.empty:
        save_table(df_age_strat, path_results / "age_stratified_sensitivity.csv")
        save_table(df_age_strat, path_report / "table_14_age_stratified.csv")

    df_ph_time = pd.DataFrame()
    if ph_time_rows:
        df_ph_time = pd.concat(ph_time_rows, ignore_index=True)
        save_table(df_ph_time, path_results / "ph_time_interaction_sensitivity.csv")
        save_table(df_ph_time, path_report / "table_3c_ph_time_interaction.csv")

    df_poisson_reri = pd.DataFrame(poisson_reri_rows)
    if not df_poisson_reri.empty:
        save_table(df_poisson_reri, path_results / "poisson_reri_sensitivity.csv")
        save_table(df_poisson_reri, path_report / "table_7b_reri_poisson.csv")

    df_disc = pd.DataFrame(disc_rows)
    if not df_disc.empty:
        save_table(df_disc, path_results / "discrimination_summary.csv")
        save_table(df_disc, path_report / "table_11b_discrimination.csv")

    df_cal = pd.DataFrame(cal_rows)
    if not df_cal.empty:
        save_table(df_cal, path_results / "calibration_summary.csv")
        save_table(df_cal, path_report / "table_11c_calibration.csv")

    # ── 5b. RBD spline dose-response (PRIMARY_OUTCOME only) ───────────────
    print("\n[5b/7] RBD spline dose-response analysis ...")
    try:
        run_rbd_spline_analysis(
            df_risk=df_risk,
            thresholds=thresholds,
            extended_covariates=extended_covariates,
            rbd_only_rows=rbd_only_rows,
            path_report=path_report,
            path_results=path_results,
            outcome=PRIMARY_OUTCOME,
        )
    except Exception as _spline_exc:
        warnings.warn(f"RBD spline dose-response analysis failed: {_spline_exc}")

    # ── 6. Terminal summary ────────────────────────────────────────────
    print("\n[6/7] Summary ...")
    sep = "-" * 72
    print(f"\n{sep}")
    print("  ANALYSIS SUMMARY")
    print(sep)

    if not df_bl.empty:
        prod_rows_summary = df_bl[df_bl["covariate"].str.startswith("prod_")].copy()
        sig_fdr = prod_rows_summary[
            prod_rows_summary.get("p_fdr", pd.Series(dtype=float)) < 0.05
        ]
        print(f"\n  Baseline Cox significant after FDR (all outcomes): {len(sig_fdr)}")
        if not sig_fdr.empty:
            for _, r in sig_fdr.sort_values("p_fdr").head(10).iterrows():
                print(
                    f"    {r['outcome']:35s} {r['prodromal_label']:30s} "
                    f"HR={r['HR']:.2f} p_FDR={r['p_fdr']:.3f}"
                )

    if not df_int.empty:
        int_terms = df_int[df_int["covariate"].str.contains("__x__", na=False)]
        sig_int = int_terms[int_terms["p"] < 0.05]
        print(f"\n  Significant interactions (p<0.05): {len(sig_int)}")
        for _, r in sig_int.sort_values("p").head(10).iterrows():
            print(
                f"    {r['outcome']:35s} {r['method']:18s} "
                f"{r['prodromal_label']:30s} HR={r['HR']:.2f} p={r['p']:.3f}"
            )

    if not df_reri.empty:
        print(f"\n  Additive interaction results: {len(df_reri)}")
        for _, r in df_reri.iterrows():
            print(
                f"    {r['outcome']:35s} {r['prodromal_label']:30s} "
                f"RERI={r['reri']:.3f} AP={r['ap']:.3f}"
            )

    print(f"\n  Report tables -> {path_report}")

    # ── 7. Generate report ─────────────────────────────────────────────
    print("\n[7/7] Generating scientific report ...")
    report_tables = {
        "cohort": df_cohort_tbl,
        "availability": avail_df,
        "ph_diagnostics": df_ph,
        "rbd_only_pd": df_rbd_pd,
        "baseline_cox_pd": df_bl_pd,
        "additive_cox_pd": df_add_pd,
        "interaction_pd": df_int_pd,
        "additive_interaction": df_reri,
        "absolute_risks": df_ar_pd,
        "spline_cox": df_sp,
        "rbd_spline": df_rbd_sp,
        "lag_sensitivity": df_lag,
        "c_index": df_ci_pd,
        "discrimination": df_disc,
        "calibration": df_cal,
        "competing_risk_cif": df_comp,
        "competing_risk_cox": df_comp_cox,
        "threshold_stability": df_thresh,
        "rbd_continuous": df_rbd_cont,
        "sensitivity_hes_active": df_sens,
        "km_logrank": df_lr,
        "model_fit": df_model_fit,
        "ph_violation_summary": df_ph_summary,
        "screening_metrics": df_screening,
        "age_stratified": df_age_strat,
        "ph_time_interaction": df_ph_time,
        "poisson_reri": df_poisson_reri,
    }
    generate_scientific_report(report_tables, path_report, active_vars)

    # ── 7b. Optional mediation analysis ───────────────────────────────
    if run_mediation_analysis:
        print("\n[7b/7] Running mediation analysis ...")
        try:
            from library.rbd_prodromal_mediation.runner import run_mediation
            run_mediation(
                results_dir=path_results,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        except Exception as _med_exc:
            warnings.warn(f"Mediation analysis failed: {_med_exc}")

    print(f"\n{sep}")
    print(f"  Output directory : {path_results}")
    print(f"  Report directory : {path_report}")
    print(sep)
    print("\n[DONE]")
