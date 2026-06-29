"""
Table serialisation and terminal reporting for the RBD–PRS association analysis.

Outputs
───────
- table_descriptive.csv   : mean ± SD by stratum (risk group × case/control)
- table_correlation.csv   : Spearman ρ, CI, analytical p, permutation p
- table_ols.csv           : OLS β (std + raw), SE, t, p, partial R², model R²,
                            residual diagnostics
- table_gam.csv           : GAM edf, pseudo-R², F-test vs. OLS, best lambda
- Terminal summary of key statistics for each analysis block
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

from library.rbd_prs_association.analysis import AnalysisResults

logger = logging.getLogger(__name__)

_FMT = "{:.4f}"


# ── Serialisation helpers ────────────────────────────────────────────────────

def save_descriptive(descriptive_df: pd.DataFrame, tables_dir: Path) -> Path:
    """Save descriptive statistics table."""
    path = tables_dir / "table_descriptive.csv"
    descriptive_df.to_csv(path, index=False, float_format="%.4f")
    logger.info("Saved: %s", path)
    return path


def save_correlation_table(results: AnalysisResults, tables_dir: Path) -> pd.DataFrame:
    """Build and save Spearman correlation table."""
    rows = [
        {
            "prs": r.prs_col,
            "stratum": r.stratum,
            "n": r.n,
            "rho": r.rho,
            "ci_lower": r.ci_lower,
            "ci_upper": r.ci_upper,
            "p_analytical": r.p_value,
            "p_permutation": r.p_permutation,
        }
        for r in results.spearman
    ]
    df = pd.DataFrame(rows)
    path = tables_dir / "table_correlation.csv"
    df.to_csv(path, index=False, float_format="%.4f")
    logger.info("Saved: %s", path)
    return df


def save_ols_table(results: AnalysisResults, tables_dir: Path) -> pd.DataFrame:
    """Build and save OLS regression table."""
    rows = [
        {
            "prs": r.prs_col,
            "stratum": r.stratum,
            "n": r.n,
            "beta_standardised": r.beta_std,
            "beta_raw": r.beta_raw,
            "se": r.se,
            "t_stat": r.t_stat,
            "p_value": r.p_value,
            "ci_lower": r.ci_lower,
            "ci_upper": r.ci_upper,
            "partial_r2": r.partial_r2,
            "model_r2": r.model_r2,
            "adj_r2": r.adj_r2,
            "shapiro_wilk_p": r.residual_normality_p,
            "breusch_pagan_p": r.breusch_pagan_p,
        }
        for r in results.ols
    ]
    df = pd.DataFrame(rows)
    path = tables_dir / "table_ols.csv"
    df.to_csv(path, index=False, float_format="%.4f")
    logger.info("Saved: %s", path)
    return df


def save_gam_table(results: AnalysisResults, tables_dir: Path) -> pd.DataFrame:
    """Build and save GAM results table."""
    rows = [
        {
            "prs": r.prs_col,
            "stratum": r.stratum,
            "n": r.n,
            "edf": r.edf,
            "pseudo_r2": r.pseudo_r2,
            "ols_r2": r.ols_r2,
            "nonlinearity_F": r.nonlinearity_f,
            "nonlinearity_p": r.nonlinearity_p,
            "gam_deviance": r.gam_deviance,
            "ols_deviance": r.ols_deviance,
            "best_lambda": r.best_lambda,
        }
        for r in results.gam
    ]
    df = pd.DataFrame(rows)
    path = tables_dir / "table_gam.csv"
    df.to_csv(path, index=False, float_format="%.4f")
    logger.info("Saved: %s", path)
    return df


# ── Terminal console summary ──────────────────────────────────────────────────

def print_console_summary(
    results: AnalysisResults,
    descriptive_df: pd.DataFrame,
    tables_dir: Path,
) -> None:
    """Print key metrics to terminal in a structured, readable format."""

    sep = "─" * 70

    print(f"\n{sep}")
    print("  RBD–PRS ASSOCIATION ANALYSIS  |  Key Results Summary")
    print(sep)

    # ── Descriptive: overall N and case counts ─────────────────────────────
    overall = descriptive_df.loc[
        (descriptive_df["stratum"] == "Overall") &
        (descriptive_df["variable"] == "abk_rbd_score_mean")
    ]
    if not overall.empty:
        row = overall.iloc[0]
        print(f"\nCohort (PRS-complete): N = {int(row['n'])}")
        print(f"  RBD score  mean={row['mean']:.4f}  SD={row['sd']:.4f}  "
              f"median={row['median']:.4f}  IQR=[{row['q25']:.4f}, {row['q75']:.4f}]")

    # ── Spearman: full cohort ──────────────────────────────────────────────
    print(f"\n{sep}")
    print("  SPEARMAN ρ  (RBD score × PRS)")
    print(sep)
    full_spear = [r for r in results.spearman if r.stratum == "Full cohort"]
    for r in full_spear:
        sig = "***" if r.p_permutation < 0.001 else "**" if r.p_permutation < 0.01 else "*" if r.p_permutation < 0.05 else "ns"
        print(
            f"  {r.prs_col:<20}  ρ={r.rho:+.4f}  "
            f"95%CI=[{r.ci_lower:+.4f}, {r.ci_upper:+.4f}]  "
            f"p={r.p_value:.2e}  p_perm={r.p_permutation:.4f}  {sig}  N={r.n}"
        )

    # ── Spearman: by risk group ────────────────────────────────────────────
    print(f"\n  By risk group:")
    rg_strata = ["Low", "Intermediate", "High"]
    for rg in rg_strata:
        rg_results = [r for r in results.spearman if r.stratum == rg]
        if not rg_results:
            continue
        print(f"    [{rg}]")
        for r in rg_results:
            sig = "***" if r.p_permutation < 0.001 else "**" if r.p_permutation < 0.01 else "*" if r.p_permutation < 0.05 else "ns"
            print(
                f"      {r.prs_col:<20}  ρ={r.rho:+.4f}  "
                f"95%CI=[{r.ci_lower:+.4f}, {r.ci_upper:+.4f}]  "
                f"p_perm={r.p_permutation:.4f}  {sig}  N={r.n}"
            )

    # ── OLS ───────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  OLS REGRESSION  (adjusted: age, sex, BMI, PC1–PC10)")
    print(sep)
    for r in results.ols:
        sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else "ns"
        print(
            f"  [{r.stratum}]  {r.prs_col:<20}  "
            f"β_std={r.beta_std:+.4f}  β_raw={r.beta_raw:+.4f}  "
            f"SE={r.se:.4f}  p={r.p_value:.2e}  {sig}  "
            f"partial-R²={r.partial_r2:.4f}  model-R²={r.model_r2:.4f}"
        )
        diag = ""
        if r.residual_normality_p < 0.05:
            diag += "  [WARN: non-normal residuals]"
        if r.breusch_pagan_p < 0.05:
            diag += "  [WARN: heteroscedastic]"
        if diag:
            print(f"    Diagnostics:{diag}")

    # ── GAM ───────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  GAM  (non-linearity test: edf > 1 → non-linear relationship)")
    print(sep)
    for r in results.gam:
        sig = "***" if r.nonlinearity_p < 0.001 else "**" if r.nonlinearity_p < 0.01 else "*" if r.nonlinearity_p < 0.05 else "ns"
        linear_verdict = "LINEAR" if r.edf < 1.5 else "NON-LINEAR"
        print(
            f"  [{r.stratum}]  {r.prs_col:<20}  "
            f"edf={r.edf:.2f}  pseudo-R²={r.pseudo_r2:.4f}  "
            f"F={r.nonlinearity_f:.2f}  p_nonlin={r.nonlinearity_p:.4f}  {sig}  "
            f"→ {linear_verdict}"
        )

    print(f"\n{sep}\n")


def save_all_tables(
    results: AnalysisResults,
    descriptive_df: pd.DataFrame,
    tables_dir: Path,
) -> None:
    """Persist all tables and print console summary.

    Parameters
    ----------
    tables_dir : Path
        Directory where CSV tables are written.  Must already exist.
    """
    save_descriptive(descriptive_df, tables_dir)
    save_correlation_table(results, tables_dir)
    save_ols_table(results, tables_dir)
    save_gam_table(results, tables_dir)
    print_console_summary(results, descriptive_df, tables_dir)
