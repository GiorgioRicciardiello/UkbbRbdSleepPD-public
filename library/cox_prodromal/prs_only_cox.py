"""
Standalone PRS-only Cox PH analysis — outcome_1a_pd_only.

Model
-----
h(t) = h0(t) exp(b_prs * PRS_std
              + b_pc1..10 * PC1..10
              + b_age * age + b_sex * sex + b_bmi * bmi
              + b_smk * smoking + b_alc * alcohol)

PRS is z-score standardised on the analytic sample so HR is per 1-SD increment.
RBD score is intentionally excluded — this quantifies PRS predictive value alone.

Run once; does not modify any pipeline file.
Output: results/prs_only_cox/
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from library.cox_prodromal.cox_config import (
    LAG_YEARS,
    PC_COLS,
    PRIMARY_OUTCOME,
    PRS_COLS,
    RIDGE_PENALIZER,
)
from library.cox_prodromal.data_prep import (
    apply_lag_filter,
    build_extended_covariates,
    build_survival_dataset_for_outcome,
    load_prodromal_dataset,
)

# ── Constants ─────────────────────────────────────────────────────────────────

OUTCOME: str = PRIMARY_OUTCOME          # "outcome_1a_pd_only"
PRS_COL: str = PRS_COLS[0]             # "prs_score_pd"
PRS_STD_COL: str = "prs_score_pd_std"  # z-scored version used in model

OUTPUT_DIR: Path = Path(__file__).parents[2] / "results" / "prs_only_cox"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _standardize_prs(
    df: pd.DataFrame,
    prs_col: str,
) -> Tuple[pd.DataFrame, float, float]:
    """Z-score standardise PRS column in-place on the analytic sample.

    Returns
    -------
    df : pd.DataFrame
        Copy with new ``prs_score_pd_std`` column.
    mu : float
        Sample mean used for standardisation.
    sigma : float
        Sample SD (ddof=1) used for standardisation.
    """
    df = df.copy()
    mu: float = float(df[prs_col].mean())
    sigma: float = float(df[prs_col].std(ddof=1))
    if sigma == 0.0 or np.isnan(sigma):
        raise ValueError(
            f"PRS column '{prs_col}' has zero or NaN SD — cannot standardise."
        )
    df[PRS_STD_COL] = (df[prs_col] - mu) / sigma
    print(
        f"  PRS standardised: mean={mu:.4f}, SD={sigma:.4f} "
        f"(n={int(df[prs_col].notna().sum()):,})"
    )
    return df, mu, sigma


def _impute_median(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Impute missing values with column median; mirrors runner.py behaviour."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        n_miss = int(df[col].isna().sum())
        if n_miss > 0:
            med = float(df[col].median())
            df[col] = df[col].fillna(med)
            print(
                f"  Imputed {n_miss} missing in '{col}' with median={med:.3f}"
            )
    return df


def _build_hr_table(
    cph: CoxPHFitter,
    prs_mean: float,
    prs_sd: float,
) -> pd.DataFrame:
    """Extract tidy HR table from a fitted CoxPHFitter instance.

    Columns: variable, HR, CI_lower_95, CI_upper_95, p_value, significant, note.
    """
    s = cph.summary.copy()
    hr_table = pd.DataFrame({
        "variable":    s.index.tolist(),
        "HR":          s["exp(coef)"].round(3).tolist(),
        "CI_lower_95": s["exp(coef) lower 95%"].round(3).tolist(),
        "CI_upper_95": s["exp(coef) upper 95%"].round(3).tolist(),
        "p_value":     s["p"].round(4).tolist(),
        "significant": (s["p"] < 0.05).tolist(),
        "note":        [""] * len(s),
    })
    prs_mask = hr_table["variable"] == PRS_STD_COL
    hr_table.loc[prs_mask, "note"] = (
        f"per 1 SD; raw mean={prs_mean:.4f}, SD={prs_sd:.4f}"
    )
    return hr_table


def _build_summary_text(
    cph: CoxPHFitter,
    n_subjects: int,
    n_events: int,
    prs_mean: float,
    prs_sd: float,
    available_pcs: List[str],
) -> str:
    """Compose a plain-text model summary for the .txt output file."""
    concordance: float = cph.concordance_index_
    ll_ratio_p: float = cph.log_likelihood_ratio_test().p_value
    lines = [
        "=" * 64,
        "PRS-Only Cox PH  |  outcome_1a_pd_only",
        "=" * 64,
        f"  Outcome          : {OUTCOME}",
        f"  N subjects       : {n_subjects:,}",
        f"  N events (PD)    : {n_events:,}",
        f"  Lag filter       : {LAG_YEARS:.0f} years",
        f"  PRS (z-scored)   : mean={prs_mean:.4f}, SD={prs_sd:.4f}",
        f"  Ancestry PCs     : {len(available_pcs)} included",
        f"  Concordance (C)  : {concordance:.4f}",
        f"  Log-LR test p    : {ll_ratio_p:.4e}",
        f"  Penalizer        : {RIDGE_PENALIZER}",
        "",
        "Predictors in model:",
        f"  - {PRS_STD_COL}  (PRS per 1 SD)",
        "  - " + ", ".join(available_pcs) if available_pcs else "  (no PCs available)",
        "  - cov_age_recruitment_21022",
        "  - cov_sex_31",
        "  - cov_bmi",
        "  - cov_smoking",
        "  - cov_alcohol",
        "=" * 64,
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_prs_only_cox() -> None:
    """Fit PRS-only Cox PH model for PD outcome and save results."""
    print("\n[prs_only_cox] Loading dataset...")
    _, df = load_prodromal_dataset()
    print(f"  Loaded N={len(df):,} subjects")

    print("\n[prs_only_cox] Building covariates...")
    df, extended_covariates = build_extended_covariates(df)

    # Impute lifestyle covariates exactly as the main runner does
    lifestyle_cols = [c for c in ("cov_smoking", "cov_alcohol") if c in extended_covariates]
    df = _impute_median(df, lifestyle_cols)

    # PC columns must be in the carry list so build_survival_dataset_for_outcome
    # includes them in the returned DataFrame (PRS_COLS carries prs_score_pd
    # automatically; PCs are not in PRS_COLS and need explicit inclusion).
    all_carry_covariates = extended_covariates + [c for c in PC_COLS if c in df.columns]

    print(f"\n[prs_only_cox] Building survival dataset for {OUTCOME}...")
    df_surv: Optional[pd.DataFrame] = build_survival_dataset_for_outcome(
        df=df,
        outcome=OUTCOME,
        active_vars={},
        extended_covariates=all_carry_covariates,
    )
    if df_surv is None or df_surv.empty:
        raise RuntimeError(f"Empty survival dataset for {OUTCOME}")
    print(
        f"  Pre-lag: N={len(df_surv):,}, events={int(df_surv['event'].sum())}"
    )

    print(f"\n[prs_only_cox] Applying {LAG_YEARS:.0f}-year lag filter...")
    df_surv = apply_lag_filter(
        df_surv, time_col="time", event_col="event", lag_years=LAG_YEARS
    )

    # Guard: PRS column must be present
    if PRS_COL not in df_surv.columns:
        raise KeyError(
            f"PRS column '{PRS_COL}' absent from survival dataset. "
            "Verify the source parquet includes PRS columns."
        )

    available_pcs = [c for c in PC_COLS if c in df_surv.columns]
    missing_pcs = [c for c in PC_COLS if c not in df_surv.columns]
    if missing_pcs:
        print(f"  WARNING: missing PC columns (excluded from model): {missing_pcs}")

    print("\n[prs_only_cox] Standardising PRS...")
    df_surv, prs_mean, prs_sd = _standardize_prs(df_surv, prs_col=PRS_COL)

    # Full covariate list: PRS_std + PCs + age/sex/bmi/smoking/alcohol
    model_covariates: List[str] = [PRS_STD_COL] + available_pcs + extended_covariates
    model_cols: List[str] = ["time", "event"] + model_covariates

    df_cc = df_surv[model_cols].dropna()
    n_subjects = len(df_cc)
    n_events = int(df_cc["event"].sum())
    print(f"\n[prs_only_cox] Analytic sample: N={n_subjects:,}, events={n_events}")

    if n_events < 5:
        raise RuntimeError(
            f"Too few events ({n_events}) to fit a Cox model (minimum 5)."
        )

    print("\n[prs_only_cox] Fitting Cox PH model...")
    cph = CoxPHFitter(penalizer=RIDGE_PENALIZER)
    cph.fit(df_cc, duration_col="time", event_col="event")

    hr_table = _build_hr_table(cph, prs_mean, prs_sd)
    summary_text = _build_summary_text(
        cph, n_subjects, n_events, prs_mean, prs_sd, available_pcs
    )

    print("\n" + summary_text)
    print("\nHR Table:")
    print(hr_table.to_string(index=False))

    # Persist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hr_path = OUTPUT_DIR / "prs_only_cox_results.csv"
    txt_path = OUTPUT_DIR / "prs_only_cox_summary.txt"
    hr_table.to_csv(hr_path, index=False)
    txt_path.write_text(summary_text, encoding="utf-8")

    print(f"\n[prs_only_cox] HR table  -> {hr_path}")
    print(f"[prs_only_cox] Summary   -> {txt_path}")


if __name__ == "__main__":
    run_prs_only_cox()
