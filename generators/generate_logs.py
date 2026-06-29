"""
Generate CONSORT log files from the current merged EHR+RBD dataset.

Reads:  data/pp/res_build_final_dataset/ehr_diag_pd_rbd_only_all.parquet
Writes: results/logs/{final_consort,consort_rbd,outcome_flags,neuro_exclusion_summary}/

These files feed generate_consort_table.py (Supplementary Table S2).

Stages captured
---------------
EHR/outcome staging (final_consort/staging_consort_long.csv):
  1. Initial cohort (actigraphy + RBD merge)
  2. After neurological exclusions
  3. After prevalent-case exclusion (analytical cohort)

RBD/actigraphy staging (consort_rbd/consort_rbd_risk_long.csv):
  1. After night QC (all subjects with RBD scores)
  2. Eligible for risk analysis (non-training, non-prevalent)

Night-QC logs (Panel B / Panel D) are NOT regenerated here because
they require raw actigraphy night-level data from the ABK pipeline.
generate_consort_table.py handles missing Panel B/D gracefully.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import config, outcomes as _cfg_outcomes

# ── Paths ──────────────────────────────────────────────────────────────────
PARQUET = Path("data/pp/res_build_final_dataset/ehr_diag_pd_rbd_only_all.parquet")
LOGS    = Path("results/logs")

# Sourced from config.config (single source of truth).
OUTCOMES = list(_cfg_outcomes)


# ── Helpers ────────────────────────────────────────────────────────────────

def _pct(lost: int, before: int) -> float:
    """Percent lost, guarded against zero-division."""
    return 100.0 * lost / before if before > 0 else float("nan")


def _consort_row(
    metric: str,
    before: int,
    after: int,
    transition: str,
) -> dict:
    lost = max(before - after, 0)
    return {
        "Metric":           metric,
        "Before":           before,
        "After":            after,
        "Lost":             lost,
        "% Lost":           _pct(lost, before),
        "Stage transition": transition,
    }


def _load_subject_level() -> pd.DataFrame:
    """Load parquet and collapse to one row per subject (first night)."""
    df = pd.read_parquet(PARQUET)
    return df.groupby("eid", as_index=False).first()


# ── Panel A source 1: EHR staging CONSORT ─────────────────────────────────

def build_staging_consort(df_subj: pd.DataFrame) -> pd.DataFrame:
    """
    Subject-level CONSORT flow for staging_consort_long.csv.

    Stages:
      (a) Actigraphy cohort (after EHR+RBD merge) →
          After neurological exclusions
      (b) After neurological exclusions →
          Analytical cohort (prevalent PD excluded via surv_days NaN)
    """
    surv_col = "outcome_1a_pd_only__surv_days"

    n_total  = len(df_subj)
    n_no_neuro = int((df_subj["neuro_exclude"] == 0).sum())
    # analytical cohort: neuro-excluded removed AND prevalent PD removed
    mask_analytic = (df_subj["neuro_exclude"] == 0) & df_subj[surv_col].notna()
    n_analytic = int(mask_analytic.sum())

    rows = [
        _consort_row(
            "Subjects", n_total, n_no_neuro,
            "Actigraphy cohort (after RBD merge) -> After neuro exclusions",
        ),
        _consort_row(
            "Subjects", n_no_neuro, n_analytic,
            "After neuro exclusions -> Analytical cohort (prevalent excl.)",
        ),
    ]

    # Outcome-specific counts at the analytical cohort stage
    df_analytic = df_subj[mask_analytic]
    for out in OUTCOMES:
        label = out.upper().replace("_", " ")
        dx_col   = f"{out}__dx"
        prev_col = f"{out}__prevalent"
        inc_col  = f"{out}__incident"

        n_dx   = int(df_analytic[dx_col].sum())   if dx_col   in df_analytic.columns else 0
        n_prev = int(df_analytic[prev_col].sum())  if prev_col in df_analytic.columns else 0
        n_inc  = int(df_analytic[inc_col].sum())   if inc_col  in df_analytic.columns else 0

        rows.append({
            "Metric":           f"{label} DIAGNOSED",
            "Before":           n_dx,
            "After":            n_dx,
            "Lost":             0,
            "% Lost":           0.0,
            "Stage transition": "Final analytical cohort",
        })
        rows.append({
            "Metric":           f"{label} PREVALENT",
            "Before":           n_prev,
            "After":            n_prev,
            "Lost":             0,
            "% Lost":           0.0,
            "Stage transition": "Final analytical cohort",
        })
        rows.append({
            "Metric":           f"{label} INCIDENT",
            "Before":           n_inc,
            "After":            n_inc,
            "Lost":             0,
            "% Lost":           0.0,
            "Stage transition": "Final analytical cohort",
        })

    return pd.DataFrame(rows)


# ── Panel A source 2: RBD/actigraphy staging CONSORT ──────────────────────

def build_rbd_consort(df_night: pd.DataFrame, df_subj: pd.DataFrame) -> pd.DataFrame:
    """
    Subject-level CONSORT flow for consort_rbd_risk_long.csv.

    Stages:
      (a) After night QC (all night rows with RBD score) →
          Eligible for risk analysis (neuro excluded and prevalent excl.)
    """
    # Night level: all rows with an RBD score
    n_nights_total = len(df_night)
    n_subj_total   = int(df_night["eid"].nunique())

    # Subject-level eligible cohort (same definition as get_clean_risk_data)
    surv_col = "outcome_1a_pd_only__surv_days"
    mask_eligible = (df_subj["neuro_exclude"] == 0) & df_subj[surv_col].notna()
    eligible_eids = set(df_subj.loc[mask_eligible, "eid"])

    n_subj_eligible   = len(eligible_eids)
    n_nights_eligible = int(df_night["eid"].isin(eligible_eids).sum())

    rows = [
        # Subjects
        _consort_row(
            "Subjects", n_subj_total, n_subj_eligible,
            "After night QC -> Eligible for risk analysis",
        ),
        # Nights
        _consort_row(
            "Nights", n_nights_total, n_nights_eligible,
            "After night QC -> Eligible for risk analysis",
        ),
    ]

    # Outcome-specific eligible and incident counts
    df_elig_subj = df_subj[mask_eligible]
    for out in OUTCOMES:
        inc_col = f"{out}__incident"
        label   = out.upper()

        n_with_inc = int(df_elig_subj[inc_col].notna().sum()) if inc_col in df_elig_subj.columns else n_subj_eligible
        n_incident = int(df_elig_subj[inc_col].sum())         if inc_col in df_elig_subj.columns else 0

        rows.append({
            "Metric":           f"{label} eligible subjects",
            "Before":           n_subj_eligible,
            "After":            n_with_inc,
            "Lost":             max(n_subj_eligible - n_with_inc, 0),
            "% Lost":           _pct(max(n_subj_eligible - n_with_inc, 0), n_subj_eligible),
            "Stage transition": "Risk-eligible -> Outcome-eligible",
        })
        rows.append({
            "Metric":           f"{label} incident cases",
            "Before":           n_with_inc,
            "After":            n_incident,
            "Lost":             float("nan"),
            "% Lost":           float("nan"),
            "Stage transition": "Outcome-eligible",
        })

    return pd.DataFrame(rows)


# ── Outcome summary ────────────────────────────────────────────────────────

def build_outcome_summary(df_subj: pd.DataFrame) -> pd.DataFrame:
    """
    Per-outcome counts for outcome_flags/outcome_summary.csv.

    Columns expected by generate_consort_table.py:
        outcome, diagnosed, prevalent, incident, incident_%, median_tte
    """
    rows = []
    for out in OUTCOMES:
        dx_col   = f"{out}__dx"
        prev_col = f"{out}__prevalent"
        inc_col  = f"{out}__incident"
        tte_col  = f"{out}__tte_days"

        n_dx   = int(df_subj[dx_col].sum())   if dx_col   in df_subj.columns else 0
        n_prev = int(df_subj[prev_col].sum())  if prev_col in df_subj.columns else 0
        n_inc  = int(df_subj[inc_col].sum())   if inc_col  in df_subj.columns else 0
        n_total = len(df_subj)

        inc_pct = 100.0 * n_inc / n_total if n_total > 0 else float("nan")

        if tte_col in df_subj.columns:
            median_tte = float(df_subj.loc[df_subj[inc_col] == True, tte_col].median())
            median_tte = 0 if math.isnan(median_tte) else int(median_tte)
        else:
            median_tte = 0

        rows.append({
            "outcome":     out,
            "diagnosed":   n_dx,
            "prevalent":   n_prev,
            "incident":    n_inc,
            "incident_%":  round(inc_pct, 3),
            "median_tte":  median_tte,
        })

    return pd.DataFrame(rows)


# ── Neuro-exclusion outcome impact ─────────────────────────────────────────

def build_neuro_impact(df_subj: pd.DataFrame) -> pd.DataFrame:
    """
    Per-outcome neurological exclusion impact for
    neuro_exclusion_summary/exclusion_outcome_impact.csv.

    Columns expected by generate_consort_table.py:
        Outcome, Cases before exclusion, Cases excluded, Cases after exclusion
    """
    rows = []
    for out in OUTCOMES:
        dx_col = f"{out}__dx"
        if dx_col not in df_subj.columns:
            continue

        cases_before = int(df_subj[dx_col].sum())
        cases_excluded = int(
            df_subj.loc[df_subj["neuro_exclude"] != 0, dx_col].sum()
        )
        cases_after = cases_before - cases_excluded

        rows.append({
            "Outcome":                out,
            "Cases before exclusion": cases_before,
            "Cases excluded":         cases_excluded,
            "Cases after exclusion":  cases_after,
        })

    return pd.DataFrame(rows)


# ── Write all logs ─────────────────────────────────────────────────────────

def main() -> None:
    """Build and save all CONSORT log files to results/logs/."""
    print("Loading parquet...")
    df_night = pd.read_parquet(PARQUET)
    df_subj  = df_night.groupby("eid", as_index=False).first()
    print(f"  Night rows: {len(df_night):,} | Subjects: {len(df_subj):,}")

    # ── staging_consort_long.csv ──────────────────────────────────────────
    out_dir = LOGS / "final_consort"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_staging = build_staging_consort(df_subj)
    path_out = out_dir / "staging_consort_long.csv"
    df_staging.to_csv(path_out, index=False)
    print(f"Written: {path_out}")

    # ── consort_rbd_risk_long.csv ─────────────────────────────────────────
    out_dir = LOGS / "consort_rbd"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_rbd = build_rbd_consort(df_night, df_subj)
    path_out = out_dir / "consort_rbd_risk_long.csv"
    df_rbd.to_csv(path_out, index=False)
    print(f"Written: {path_out}")

    # ── outcome_summary.csv ───────────────────────────────────────────────
    out_dir = LOGS / "outcome_flags"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_outcomes = build_outcome_summary(df_subj)
    path_out = out_dir / "outcome_summary.csv"
    df_outcomes.to_csv(path_out, index=False)
    print(f"Written: {path_out}")

    # ── exclusion_outcome_impact.csv ──────────────────────────────────────
    out_dir = LOGS / "neuro_exclusion_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_neuro = build_neuro_impact(df_subj)
    path_out = out_dir / "exclusion_outcome_impact.csv"
    df_neuro.to_csv(path_out, index=False)
    print(f"Written: {path_out}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n=== Cohort summary ===")
    n_total    = len(df_subj)
    n_neuro_ex = int((df_subj["neuro_exclude"] != 0).sum())
    surv_col   = "outcome_1a_pd_only__surv_days"
    n_analytic = int(
        ((df_subj["neuro_exclude"] == 0) & df_subj[surv_col].notna()).sum()
    )
    print(f"  Total subjects (after RBD merge): {n_total:,}")
    print(f"  Neuro-excluded:                   {n_neuro_ex:,}")
    print(f"  Analytical cohort:                {n_analytic:,}")
    print("\nOutcome counts (analytical cohort):")
    print(df_outcomes.to_string(index=False))


if __name__ == "__main__":
    main()
