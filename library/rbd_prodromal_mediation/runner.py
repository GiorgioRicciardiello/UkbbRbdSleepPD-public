"""
Runner for the RBD-Prodromal Association & Mediation Analysis.

Orchestrates:
  - Interpretation A (binary prodromal markers -> RBD -> PD)
  - Interpretation C (cognitive markers -> RBD -> PD)
  - Combined summary reports

Can be invoked standalone or as an optional stage from the main
cox_prodromal runner.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from library.cox_prodromal.cox_config import BOOTSTRAP_JOBS
from library.rbd_prodromal_mediation.config import (
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    INTERPRETATION_A_VARS,
    INTERPRETATION_C_VARS,
    PRIMARY_OUTCOME,
    PRODROMAL_BINARY_VARS,
)
from library.rbd_prodromal_mediation.data_prep import (
    MediationCohort,
    load_mediation_cohort,
)
from library.rbd_prodromal_mediation.model_association import run_association_models
from library.rbd_prodromal_mediation.model_mediation import run_mediation_analysis
from library.rbd_prodromal_mediation.reporting import (
    build_model_performance_summary,
    build_summary_table,
    save_interpretation_tables,
    save_table,
)


def _split_active_vars(
    active_vars: Dict[str, str],
) -> tuple[Dict[str, str], Dict[str, str]]:
    """
    Split active variables into Interpretation A (binary) and C (cognitive).

    Parameters
    ----------
    active_vars : dict
        Full set of active variables.

    Returns
    -------
    tuple[dict, dict]
        (interp_a_vars, interp_c_vars)
    """
    interp_a = {k: v for k, v in active_vars.items() if k in INTERPRETATION_A_VARS}
    interp_c = {k: v for k, v in active_vars.items() if k in INTERPRETATION_C_VARS}
    return interp_a, interp_c


def _run_interpretation(
    df: pd.DataFrame,
    vars_dict: Dict[str, str],
    covariates: list[str],
    interp_name: str,
    out_dir: Path,
    temporal_log: Optional[pd.DataFrame],
    n_bootstrap: int,
    seed: int,
    n_jobs: int = 1,
) -> Dict[str, pd.DataFrame]:
    """
    Run association + mediation for one interpretation (A or C).

    Parameters
    ----------
    df : pd.DataFrame
        Analytic survival dataset.
    vars_dict : dict
        Active variables for this interpretation.
    covariates : list[str]
        Adjustment covariates.
    interp_name : str
        "A" or "C".
    out_dir : Path
        Output directory for this interpretation.
    temporal_log : pd.DataFrame or None
        Temporal filter log (only for Interpretation C).
    n_bootstrap : int
        Bootstrap resamples.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, pd.DataFrame]
        All output tables for this interpretation.
    """
    if not vars_dict:
        print(f"\n  Interpretation {interp_name}: no active variables, skipping")
        return {}

    print(f"\n{'=' * 60}")
    print(f"  INTERPRETATION {interp_name}: {len(vars_dict)} variables")
    print(f"{'=' * 60}")

    t0 = time.time()

    # Association models (1a, 1b, 1b-3g)
    print(f"\n  [1/2] Association models ...")
    df_ols, df_logistic, df_multinomial = run_association_models(
        df, vars_dict, covariates,
    )

    elapsed_assoc = time.time() - t0
    print(f"  Association models complete ({elapsed_assoc:.1f}s)")

    # Mediation (Baron & Kenny)
    print(f"\n  [2/2] Mediation analysis (B={n_bootstrap}) ...")
    t1 = time.time()
    df_steps, df_boot, df_perf, df_supp, df_feas = run_mediation_analysis(
        df, vars_dict, covariates,
        n_bootstrap=n_bootstrap, seed=seed, n_jobs=n_jobs,
    )

    elapsed_med = time.time() - t1
    print(f"  Mediation complete ({elapsed_med:.1f}s)")

    # Collect tables
    tables: Dict[str, pd.DataFrame] = {
        "assoc_linear": df_ols,
        "assoc_logistic": df_logistic,
        "assoc_logistic_3g": df_multinomial,
        "mediation_steps": df_steps,
        "mediation_indirect": df_boot,
        "mediation_model_perf": df_perf,
        "mediation_feasibility": df_feas,
        "supplementary_3g": df_supp,
    }
    if temporal_log is not None:
        tables["temporal_filter_log"] = temporal_log

    # Save
    save_interpretation_tables(tables, out_dir)

    return tables


def run_mediation(
    results_dir: Optional[Path] = None,
    outcome: str = PRIMARY_OUTCOME,
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    n_jobs: int = BOOTSTRAP_JOBS,
) -> None:
    """
    Full mediation analysis pipeline.

    1. Load and prepare mediation cohort
    2. Run Interpretation A (binary prodromals)
    3. Run Interpretation C (cognitive markers)
    4. Build combined summary reports

    Parameters
    ----------
    results_dir : Path, optional
        Parent results directory. If None, creates timestamped dir.
    outcome : str
        Target outcome (default: outcome_1a_pd_only).
    n_bootstrap : int
        Bootstrap resamples (default: 1000).
    seed : int
        Random seed (default: 42).
    n_jobs : int
        Parallel workers for bootstrap (default: BOOTSTRAP_JOBS from config).
    """
    t_start = time.time()

    # Output directories
    if results_dir is None:
        from datetime import datetime
        from config.config import config
        timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        results_dir = config["results"]["root"] / f"cox_prodromal_abk_{timestamp}"

    med_dir = results_dir / "mediation"
    dir_a = med_dir / "interpretation_A"
    dir_c = med_dir / "interpretation_C"
    dir_report = med_dir / "report"

    for d in [dir_a, dir_c, dir_report]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Load cohort ────────────────────────────────────────────────
    cohort = load_mediation_cohort(outcome=outcome)

    # Split variables by interpretation
    interp_a_vars, interp_c_vars = _split_active_vars(cohort.active_vars)
    print(f"\n  Interpretation A vars: {len(interp_a_vars)}")
    print(f"  Interpretation C vars: {len(interp_c_vars)}")

    # ── 2. Interpretation A ───────────────────────────────────────────
    tables_a = _run_interpretation(
        df=cohort.df,
        vars_dict=interp_a_vars,
        covariates=cohort.extended_covariates,
        interp_name="A",
        out_dir=dir_a,
        temporal_log=None,
        n_bootstrap=n_bootstrap,
        seed=seed,
        n_jobs=n_jobs,
    )

    # ── 3. Interpretation C ───────────────────────────────────────────
    tables_c = _run_interpretation(
        df=cohort.df,
        vars_dict=interp_c_vars,
        covariates=cohort.extended_covariates,
        interp_name="C",
        out_dir=dir_c,
        temporal_log=cohort.temporal_filter_log,
        n_bootstrap=n_bootstrap,
        seed=seed + 10000,
        n_jobs=n_jobs,
    )

    # ── 4. Combined summaries ─────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  COMBINED SUMMARIES")
    print(f"{'=' * 60}")

    for label, tables in [("A", tables_a), ("C", tables_c)]:
        if not tables:
            continue
        df_steps = tables.get("mediation_steps")
        df_boot = tables.get("mediation_indirect")
        summary = build_summary_table(df_steps, df_boot)
        if not summary.empty:
            save_table(summary, dir_report / f"mediation_summary_{label}.xlsx")

    # Model performance summary (combined)
    perf_frames = []
    for tables in [tables_a, tables_c]:
        if tables and "mediation_model_perf" in tables:
            perf_frames.append(tables["mediation_model_perf"])
    if perf_frames:
        df_all_perf = pd.concat(perf_frames, ignore_index=True)
        perf_summary = build_model_performance_summary(df_all_perf)
        if not perf_summary.empty:
            save_table(perf_summary, dir_report / "model_performance_summary.xlsx")

    # ── Terminal summary ──────────────────────────────────────────────
    elapsed = time.time() - t_start
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  MEDIATION ANALYSIS COMPLETE ({elapsed:.0f}s)")
    print(sep)
    print(f"  Outcome: {outcome}")
    print(f"  Bootstrap: B={n_bootstrap}, seed={seed}")
    print(f"  RBD z-score: mean={cohort.rbd_mean:.4f}, std={cohort.rbd_std:.4f}")
    print(f"  Cohort: N={cohort.n_total:,}, events={cohort.n_events:,}")

    for label, tables in [("A", tables_a), ("C", tables_c)]:
        if not tables:
            continue
        steps = tables.get("mediation_steps")
        if steps is not None and not steps.empty:
            n_sig = int((steps["p_c"] < 0.05).sum())
            n_inconsistent = int(steps["inconsistent_mediation"].sum())
            print(f"\n  Interpretation {label}: "
                  f"{len(steps)} variables tested, "
                  f"{n_sig} with significant c-path (p<0.05), "
                  f"{n_inconsistent} inconsistent mediation")

    print(f"\n  Output: {med_dir}")
    print(sep)
    print("[MEDIATION DONE]")


# ── Standalone entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    run_mediation()
