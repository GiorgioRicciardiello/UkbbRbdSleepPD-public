"""
LR-analysis data audit.

Checks data readiness for computing likelihood ratios (LR+/LR−) of the
actigraphy-based RBD score for incident PD prediction.

Reports:
1. Case/control counts (incident PD vs controls).
2. Missingness for all key variables.
3. RBD z-score distribution (controls-only normalization — no leakage).
4. Sex-stratified event counts (power check for sex-stratified LR).
5. Prodromal marker availability (binary).
6. Field 30557 ("acting out dreams") scan — how many subjects answered it.

Run as:
    C:\\Users\\riccig01\\anaconda3\\envs\\stats_env\\python.exe -m src.lr_analysis.audit
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from repo root regardless of cwd.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from library.cox_prodromal.data_prep import load_prodromal_dataset
from library.column_registry import col_incident

# ── Constants ─────────────────────────────────────────────────────────────────

OUTCOME = "outcome_1a_pd_only"
RBD_COL = "abk_rbd_score_mean"
SEX_COL = "cov_sex_31"          # 0 = female, 1 = male (UKBB encoding)
CONTROL_COL = "control"
FIELD_30557_PATTERNS = ["30557", "dream_enactment", "acting_out", "act_dream"]

PRODROMAL_COLS = [
    "prodromal_anosmia_bl",
    "prodromal_anxiety_bl",
    "prodromal_constipation_bl",
    "prodromal_depression_bl",
    "prodromal_dream_enactment_bl",
    "prodromal_erectile_dysfunction_bl",
    "prodromal_hyposmia_bl",
    "prodromal_orthostatic_bl",
]

# Minimum incident events per sex stratum for a stable LR estimate.
MIN_EVENTS_SEX_STRATUM = 20

_SEP = "=" * 72


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(n: int, total: int) -> str:
    """Format count and percentage."""
    return f"{n:>7,}  ({n / total * 100:5.1f}%)" if total > 0 else f"{n:>7,}  (  N/A)"


def _zscore_controls_only(
    df: pd.DataFrame,
    col: str,
    control_mask: pd.Series,
) -> pd.Series:
    """
    Compute z-score of *col* using mean/SD from controls only.

    No-leakage approach: cases are NOT included in the normalization.
    NaN propagates for missing values.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
        Column to standardize.
    control_mask : pd.Series[bool]
        True for subjects who are controls (not cases).

    Returns
    -------
    pd.Series
        Z-scored values aligned to df.index.
    """
    ctrl_vals = df.loc[control_mask, col].dropna()
    mu = float(ctrl_vals.mean())
    sigma = float(ctrl_vals.std(ddof=1))
    if sigma == 0.0:
        raise ValueError(f"Zero variance in controls for column '{col}'")
    return (df[col] - mu) / sigma


def _field30557_cols(df: pd.DataFrame) -> list[str]:
    """Return column names that likely correspond to UKBB field 30557."""
    hits = []
    for col in df.columns:
        col_lower = col.lower()
        for pat in FIELD_30557_PATTERNS:
            if pat in col_lower:
                hits.append(col)
                break
    return hits


# ── Audit sections ────────────────────────────────────────────────────────────

def _section_case_control(df_cohort: pd.DataFrame) -> None:
    """Report case / control counts."""
    incident_col = col_incident(OUTCOME)
    is_case = df_cohort[incident_col].fillna(False).astype(bool)
    is_ctrl = df_cohort[CONTROL_COL].fillna(False).astype(bool)
    n_total = len(df_cohort)

    print(_SEP)
    print("SECTION 1 — Case / Control Counts")
    print(_SEP)
    print(f"  Total subjects in cohort:   {n_total:>7,}")
    print(f"  Incident PD cases:          {_pct(int(is_case.sum()), n_total)}")
    print(f"  Controls:                   {_pct(int(is_ctrl.sum()), n_total)}")
    n_both = int((is_case & is_ctrl).sum())
    if n_both:
        print(f"  WARNING: {n_both} subjects flagged as both case AND control.")
    n_neither = int((~is_case & ~is_ctrl).sum())
    if n_neither:
        print(f"  Subjects with neither flag: {n_neither} (not used in LR analysis)")
    print()


def _section_missingness(df_cohort: pd.DataFrame) -> None:
    """Report missingness for RBD score, sex, and prodromal markers."""
    incident_col = col_incident(OUTCOME)
    is_case = df_cohort[incident_col].fillna(False).astype(bool)
    is_ctrl = df_cohort[CONTROL_COL].fillna(False).astype(bool)
    df_lr = df_cohort[is_case | is_ctrl].copy()
    n = len(df_lr)

    print(_SEP)
    print("SECTION 2 — Missingness (analysis set: cases + controls)")
    print(_SEP)
    print(f"  Analysis-set N = {n:,}")
    print()

    key_cols = [RBD_COL, SEX_COL] + PRODROMAL_COLS
    for col in key_cols:
        if col not in df_lr.columns:
            print(f"  {'MISSING COLUMN':<40s}  {col}")
            continue
        n_missing = int(df_lr[col].isna().sum())
        n_avail = n - n_missing
        print(
            f"  {col:<40s}  available: {_pct(n_avail, n)}  "
            f"missing: {_pct(n_missing, n)}"
        )
    print()


def _section_rbd_distribution(df_cohort: pd.DataFrame) -> None:
    """Report RBD score distribution in cases vs controls + z-score preview."""
    incident_col = col_incident(OUTCOME)
    is_case = df_cohort[incident_col].fillna(False).astype(bool)
    is_ctrl = df_cohort[CONTROL_COL].fillna(False).astype(bool)
    df_lr = df_cohort[is_case | is_ctrl].copy()

    if RBD_COL not in df_lr.columns:
        print(f"  SKIP: column '{RBD_COL}' not found.")
        return

    print(_SEP)
    print("SECTION 3 — RBD Score Distribution (cases vs controls)")
    print(_SEP)

    for label, mask in [("Cases (incident PD)", is_case), ("Controls", is_ctrl)]:
        vals = df_lr.loc[mask[df_lr.index], RBD_COL].dropna()
        if vals.empty:
            print(f"  {label}: no data")
            continue
        print(
            f"  {label:<25s}  N={len(vals):>6,}  "
            f"mean={vals.mean():.4f}  SD={vals.std():.4f}  "
            f"median={vals.median():.4f}  "
            f"[{vals.quantile(0.25):.4f}, {vals.quantile(0.75):.4f}] IQR"
        )

    # Demonstrate no-leakage z-score using control distribution only.
    ctrl_mask_lr = is_ctrl[df_lr.index]
    try:
        df_lr["rbd_zscore_no_leak"] = _zscore_controls_only(df_lr, RBD_COL, ctrl_mask_lr)
        ctrl_z = df_lr.loc[ctrl_mask_lr, "rbd_zscore_no_leak"].dropna()
        case_z = df_lr.loc[is_case[df_lr.index], "rbd_zscore_no_leak"].dropna()
        print()
        print(f"  Z-score (ctrl-normalized, no leakage):")
        print(
            f"    Controls:  mean={ctrl_z.mean():.3f}  SD={ctrl_z.std():.3f}  "
            f"(should be ~0 and ~1)"
        )
        print(
            f"    Cases:     mean={case_z.mean():.3f}  SD={case_z.std():.3f}  "
            f"(shift indicates signal)"
        )
    except Exception as exc:
        print(f"  Z-score computation failed: {exc}")
    print()


def _section_sex_stratified(df_cohort: pd.DataFrame) -> None:
    """Report event counts by sex for power assessment."""
    incident_col = col_incident(OUTCOME)
    is_case = df_cohort[incident_col].fillna(False).astype(bool)
    is_ctrl = df_cohort[CONTROL_COL].fillna(False).astype(bool)
    df_lr = df_cohort[is_case | is_ctrl].copy()

    if SEX_COL not in df_lr.columns:
        print(f"  SKIP: sex column '{SEX_COL}' not found.")
        return

    print(_SEP)
    print("SECTION 4 — Sex-Stratified Power Check")
    print(_SEP)
    print(f"  Minimum events per stratum required: {MIN_EVENTS_SEX_STRATUM}")
    print()

    sex_labels = {0: "Female (sex=0)", 1: "Male (sex=1)"}
    all_ok = True
    for sex_val, sex_label in sex_labels.items():
        mask_sex = df_lr[SEX_COL] == sex_val
        n_sex = int(mask_sex.sum())
        n_cases_sex = int((is_case[df_lr.index] & mask_sex).sum())
        n_ctrl_sex = int((is_ctrl[df_lr.index] & mask_sex).sum())
        ok = n_cases_sex >= MIN_EVENTS_SEX_STRATUM
        flag = "OK" if ok else "WARN: TOO FEW EVENTS"
        if not ok:
            all_ok = False
        print(
            f"  {sex_label:<25s}  N={n_sex:>6,}  "
            f"cases={n_cases_sex:>4,}  controls={n_ctrl_sex:>6,}  [{flag}]"
        )
    print()
    if all_ok:
        print("  => Sex-stratified LR analysis is feasible in both strata.")
    else:
        print("  => One or more strata have <20 events — sex-stratified LR may be unstable.")
    print()


def _section_prodromal(df_cohort: pd.DataFrame) -> None:
    """Report prevalence of binary prodromal markers in cases vs controls."""
    incident_col = col_incident(OUTCOME)
    is_case = df_cohort[incident_col].fillna(False).astype(bool)
    is_ctrl = df_cohort[CONTROL_COL].fillna(False).astype(bool)
    df_lr = df_cohort[is_case | is_ctrl].copy()

    print(_SEP)
    print("SECTION 5 — Prodromal Marker Prevalence (cases vs controls)")
    print(_SEP)
    print(f"  {'Marker':<40s}  {'Cases %':>8s}  {'Controls %':>10s}  {'Available?':>10s}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*10}  {'-'*10}")

    for col in PRODROMAL_COLS:
        if col not in df_lr.columns:
            print(f"  {col:<40s}  {'':>8s}  {'':>10s}  NOT IN DATA")
            continue
        c_mask = is_case[df_lr.index]
        ctrl_mask = is_ctrl[df_lr.index]
        case_prev = df_lr.loc[c_mask, col].mean()
        ctrl_prev = df_lr.loc[ctrl_mask, col].mean()
        print(
            f"  {col:<40s}  {case_prev*100:>7.1f}%  {ctrl_prev*100:>9.1f}%  OK"
        )
    print()


def _section_field30557(df_cohort: pd.DataFrame) -> None:
    """Scan for field 30557 ('acting out dreams') and report coverage."""
    incident_col = col_incident(OUTCOME)
    is_case = df_cohort[incident_col].fillna(False).astype(bool)
    is_ctrl = df_cohort[CONTROL_COL].fillna(False).astype(bool)
    df_lr = df_cohort[is_case | is_ctrl].copy()
    n = len(df_lr)

    print(_SEP)
    print("SECTION 6 — Field 30557 ('Frequency of acting out dreams') Coverage")
    print(_SEP)

    hits = _field30557_cols(df_cohort)
    if not hits:
        print("  No columns matching field 30557 / dream_enactment patterns found.")
        print(
            "  Note: 'prodromal_dream_enactment_bl' is ICD10-derived (G4752), "
            "not the touchscreen questionnaire."
        )
    else:
        for col in hits:
            if col not in df_lr.columns:
                print(f"  {col}: not in analysis set")
                continue
            n_avail = int(df_lr[col].notna().sum())
            n_cases_avail = int(df_lr.loc[is_case[df_lr.index], col].notna().sum())
            n_ctrl_avail = int(df_lr.loc[is_ctrl[df_lr.index], col].notna().sum())
            print(
                f"  {col:<45s}  total={_pct(n_avail, n)}  "
                f"cases={n_cases_avail}  controls={n_ctrl_avail}"
            )
            # Value counts
            if df_lr[col].dtype in [object, "category"] or df_lr[col].nunique() <= 10:
                vc = df_lr[col].value_counts(dropna=False).head(8)
                for val, cnt in vc.items():
                    print(f"      {str(val):>20s}: {cnt:,}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def run_audit() -> None:
    """Load data and run all audit sections."""
    print(_SEP)
    print("LR-MDS DATA AUDIT")
    print(f"Outcome: {OUTCOME}")
    print(_SEP)

    print("Loading dataset...")
    _, df = load_prodromal_dataset()
    print(f"  Loaded {len(df):,} subjects, {len(df.columns)} columns.")
    print()

    incident_col = col_incident(OUTCOME)
    if incident_col not in df.columns:
        print(f"ERROR: incident column '{incident_col}' not found in dataset.")
        print(f"Available columns with 'incident': {[c for c in df.columns if 'incident' in c]}")
        return

    if CONTROL_COL not in df.columns:
        print(f"ERROR: '{CONTROL_COL}' column not found in dataset.")
        print(f"Columns containing 'control': {[c for c in df.columns if 'control' in c.lower()]}")
        return

    is_case = df[incident_col].fillna(False).astype(bool)
    is_ctrl = df[CONTROL_COL].fillna(False).astype(bool)
    df_cohort = df[is_case | is_ctrl].copy()

    _section_case_control(df_cohort)
    _section_missingness(df_cohort)
    _section_rbd_distribution(df_cohort)
    _section_sex_stratified(df_cohort)
    _section_prodromal(df_cohort)
    _section_field30557(df_cohort)

    print(_SEP)
    print("AUDIT COMPLETE")
    print(_SEP)


if __name__ == "__main__":
    run_audit()
