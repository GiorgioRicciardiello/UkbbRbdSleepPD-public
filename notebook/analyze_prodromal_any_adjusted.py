"""
Adjusted Prodromal Comparison by RBD Risk Group — ANY (baseline OR follow-up)
==============================================================================
Logistic regression per prodromal marker (binary, any = bl | post) across
three RBD risk strata (Low / Mid / High) in non-converters (controls only).

Model per marker:
    logistic(marker_any ~ C(rg_pctl3, Treatment('Low')) + age + sex + follow_up_years)

Outputs:
  - N (%) per group (raw counts, same as Table 2)
  - OR [95% CI] for Mid vs Low and High vs Low (adjusted)
  - Global 3-group chi-square p (unadjusted, for reference)
  - BH-FDR correction across the 5 testable markers

Cohort: controls only (non-converters), N=86,973
Source: ehr_diag_pd_rbd_only_all.parquet
Output: notebook/results/prodromal_any_adjusted/
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import logit
from statsmodels.stats.multitest import multipletests

from config.config import config
from library.risk.risk_helpers import get_clean_risk_data, make_subject_level

# ── Constants ──────────────────────────────────────────────────────────────────

FILE_NAME: str = "ehr_diag_pd_rbd_only_all"
RG_COL: str = "rg_pctl3"
AGE_COL: str = "cov_age_recruitment_21022"
SEX_COL: str = "cov_sex_31"
FU_COL: str = "follow_up_years"

GROUP_ORDER: List[str] = ["Low", "Mid", "High"]

# dream_enactment, hyposmia, anosmia excluded — structural zeros in HES.
PRODROMAL_MARKERS: Dict[str, str] = {
    "constipation":        "Constipation",
    "depression":          "Depression",
    "anxiety":             "Anxiety",
    "orthostatic":         "Orthostatic Hypotension",
    "erectile_dysfunction": "Erectile Dysfunction",
}

OUT_DIR: Path = Path("notebook/results/prodromal_any_adjusted")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_controls() -> pd.DataFrame:
    """
    Load parquet, apply standard exclusions, collapse to subject level,
    filter to non-converters (control == True).

    Returns
    -------
    pd.DataFrame
        One row per subject, N=86,973.
    """
    _, df_night = get_clean_risk_data(file_name=FILE_NAME)
    df = make_subject_level(df_night, id_col="eid", prob_col="abk_rbd_score_mean")
    df = df[df["control"] == True].reset_index(drop=True)
    print(f"  Controls N = {len(df):,}")
    counts = df[RG_COL].value_counts().reindex(GROUP_ORDER)
    print(f"  RBD group distribution:\n{counts.to_string()}\n")
    return df


# ── Feature construction ───────────────────────────────────────────────────────

def add_any_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute prodromal_<marker>_any = bl | post for each marker.
    NaN in either source treated as 0 (not recorded = not present).

    Returns a copy with new *_any columns added.
    """
    df = df.copy()
    for marker in PRODROMAL_MARKERS:
        col_bl = f"prodromal_{marker}_bl"
        col_post = f"prodromal_{marker}_post"
        col_any = f"prodromal_{marker}_any"
        bl = pd.to_numeric(df.get(col_bl, pd.Series(0, index=df.index)),
                           errors="coerce").fillna(0).astype(bool)
        post = pd.to_numeric(df.get(col_post, pd.Series(0, index=df.index)),
                             errors="coerce").fillna(0).astype(bool)
        df[col_any] = (bl | post).astype(int)
    return df


# ── Statistics ─────────────────────────────────────────────────────────────────

def group_counts(df: pd.DataFrame, col: str) -> Dict[str, Tuple[int, float]]:
    """
    Return {group: (n_positive, pct)} for each group in GROUP_ORDER.

    Parameters
    ----------
    df  : subject-level DataFrame with RG_COL and col columns.
    col : binary (0/1) outcome column.

    Returns
    -------
    Dict mapping group label → (n_positive, percent_of_group_N).
    """
    result: Dict[str, Tuple[int, float]] = {}
    for g in GROUP_ORDER:
        sub = df.loc[df[RG_COL] == g, col]
        n_pos = int(sub.sum())
        pct = 100.0 * n_pos / len(sub) if len(sub) > 0 else np.nan
        result[g] = (n_pos, pct)
    return result


def global_chisq(df: pd.DataFrame, col: str) -> Tuple[float, float]:
    """
    3-group chi-square test (unadjusted) for a binary marker across RBD groups.

    Returns (chi2_statistic, p_value).
    """
    ct = pd.crosstab(df[RG_COL], df[col])
    if ct.shape[1] < 2 or ct.shape[0] < 2:
        return np.nan, np.nan
    chi2, p, *_ = stats.chi2_contingency(ct, correction=False)
    return float(chi2), float(p)


def fit_logistic(
    df: pd.DataFrame,
    col_any: str,
) -> Dict[str, Dict[str, float]]:
    """
    Fit logistic regression for one binary marker, adjusted for age, sex,
    and follow-up time. Low is the reference group.

    Model:
        logit(col_any) ~ C(rg_pctl3, Treatment('Low'))
                         + age + sex + follow_up_years

    Parameters
    ----------
    df      : controls DataFrame with all required columns.
    col_any : binary outcome column (0/1).

    Returns
    -------
    Dict with keys 'Mid' and 'High', each a dict:
        {'or': float, 'ci_lo': float, 'ci_hi': float, 'p': float}
    Raises warnings and returns NaN dicts on convergence failure.
    """
    required = [col_any, RG_COL, AGE_COL, SEX_COL, FU_COL]
    sub = df[required].copy()
    sub[col_any] = pd.to_numeric(sub[col_any], errors="coerce")
    sub[AGE_COL] = pd.to_numeric(sub[AGE_COL], errors="coerce")
    sub[SEX_COL] = pd.to_numeric(sub[SEX_COL], errors="coerce")
    sub[FU_COL] = pd.to_numeric(sub[FU_COL], errors="coerce")
    sub = sub.dropna()

    # Rename for formula safety
    sub = sub.rename(columns={
        col_any: "y",
        RG_COL: "rg",
        AGE_COL: "age",
        SEX_COL: "sex",
        FU_COL: "fu_years",
    })
    sub["rg"] = pd.Categorical(sub["rg"], categories=GROUP_ORDER, ordered=False)

    nan_result = {"or": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan}

    if sub["y"].sum() < 5:
        warnings.warn(f"{col_any}: fewer than 5 events — logistic skipped.")
        return {"Mid": nan_result, "High": nan_result}

    try:
        formula = "y ~ C(rg, Treatment(reference='Low')) + age + sex + fu_years"
        model = logit(formula, data=sub).fit(disp=False, maxiter=200)
    except Exception as exc:
        warnings.warn(f"{col_any}: logistic failed — {exc}")
        return {"Mid": nan_result, "High": nan_result}

    out: Dict[str, Dict[str, float]] = {}
    conf = model.conf_int(alpha=0.05)

    for group in ("Mid", "High"):
        param_name = f"C(rg, Treatment(reference='Low'))[T.{group}]"
        if param_name not in model.params:
            out[group] = nan_result
            continue
        log_or = model.params[param_name]
        lo, hi = conf.loc[param_name]
        p_val = model.pvalues[param_name]
        out[group] = {
            "or":    float(np.exp(log_or)),
            "ci_lo": float(np.exp(lo)),
            "ci_hi": float(np.exp(hi)),
            "p":     float(p_val),
        }

    return out


# ── Table assembly ─────────────────────────────────────────────────────────────

def _fmt_or(or_: float, lo: float, hi: float) -> str:
    """Format OR [95% CI] as string."""
    if np.isnan(or_):
        return "—"
    return f"{or_:.2f} [{lo:.2f}-{hi:.2f}]"


def _fmt_p(p: float) -> str:
    """Format p-value."""
    if np.isnan(p):
        return "—"
    if p < 0.001:
        return f"{p:.3e}"
    return f"{p:.4f}"


def _fmt_pct(n: int, pct: float) -> str:
    return f"{n:,} ({pct:.1f}%)"


def build_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all per-marker statistics and assemble the results table.

    Columns:
        Marker | N_Overall | N_Low | N_Mid | N_High
        | OR_Mid [95%CI] | p_Mid | OR_High [95%CI] | p_High
        | p_global | p_FDR

    FDR: BH correction applied to global chi-square p-values across markers.

    Parameters
    ----------
    df : controls DataFrame with *_any columns added.

    Returns
    -------
    pd.DataFrame one row per marker.
    """
    rows: List[Dict] = []
    global_ps: List[float] = []

    for marker, label in PRODROMAL_MARKERS.items():
        col_any = f"prodromal_{marker}_any"

        if col_any not in df.columns:
            warnings.warn(f"{col_any} not found — skipping.")
            continue

        # Raw counts
        overall_n = int(df[col_any].sum())
        overall_pct = 100.0 * overall_n / len(df)
        gc = group_counts(df, col_any)

        # Global chi-square
        chi2, p_global = global_chisq(df, col_any)

        # Logistic regression (adjusted)
        lr = fit_logistic(df, col_any)

        rows.append({
            "Marker":           label,
            "N_Overall":        _fmt_pct(overall_n, overall_pct),
            "N_Low":            _fmt_pct(*gc["Low"]),
            "N_Mid":            _fmt_pct(*gc["Mid"]),
            "N_High":           _fmt_pct(*gc["High"]),
            "OR_Mid [95%CI]":   _fmt_or(lr["Mid"]["or"], lr["Mid"]["ci_lo"], lr["Mid"]["ci_hi"]),
            "p_Mid":            _fmt_p(lr["Mid"]["p"]),
            "OR_High [95%CI]":  _fmt_or(lr["High"]["or"], lr["High"]["ci_lo"], lr["High"]["ci_hi"]),
            "p_High":           _fmt_p(lr["High"]["p"]),
            "p_global":         _fmt_p(p_global),
            "_p_global_raw":    p_global,  # kept for FDR, dropped before export
        })
        global_ps.append(p_global)

    tbl = pd.DataFrame(rows)

    # BH-FDR across markers on global chi-square p
    valid_mask = ~np.isnan(tbl["_p_global_raw"].values.astype(float))
    fdr_vals = np.full(len(tbl), np.nan)
    if valid_mask.sum() > 0:
        _, fdr_corrected, _, _ = multipletests(
            tbl.loc[valid_mask, "_p_global_raw"].values.astype(float),
            method="fdr_bh",
        )
        fdr_vals[valid_mask] = fdr_corrected

    tbl["p_FDR"] = [_fmt_p(v) for v in fdr_vals]
    tbl = tbl.drop(columns=["_p_global_raw"])

    return tbl


# ── Output ─────────────────────────────────────────────────────────────────────

def save_outputs(tbl: pd.DataFrame, out_dir: Path) -> None:
    """
    Save results table as CSV and Excel.

    Parameters
    ----------
    tbl     : assembled results DataFrame.
    out_dir : output directory (created if absent).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "results_prodromal_any_adjusted_logistic.csv"
    tbl.to_csv(csv_path, index=False)
    print(f"  CSV  -> {csv_path}")

    xlsx_path = out_dir / "results_prodromal_any_adjusted_logistic.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        tbl.to_excel(writer, index=False, sheet_name="Prodromal_Any_Adjusted")
        ws = writer.sheets["Prodromal_Any_Adjusted"]
        # Auto-width columns
        for col_idx, col in enumerate(tbl.columns, start=1):
            max_len = max(len(str(col)), tbl[col].astype(str).str.len().max())
            ws.column_dimensions[
                ws.cell(row=1, column=col_idx).column_letter
            ].width = min(max_len + 2, 40)
    print(f"  XLSX -> {xlsx_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run adjusted prodromal any-visit comparison and save results."""
    print("=" * 70)
    print("Adjusted Prodromal Comparison — ANY (baseline OR follow-up)")
    print("Cohort: controls only | Model: logit + age + sex + follow_up_years")
    print("=" * 70)

    print("\n[1/3] Loading cohort …")
    df = load_controls()

    print("[2/3] Computing _any flags and running models …")
    df = add_any_flags(df)
    tbl = build_results_table(df)

    print("\nResults preview:")
    print(tbl.to_string(index=False))

    print("\n[3/3] Saving outputs …")
    save_outputs(tbl, OUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
