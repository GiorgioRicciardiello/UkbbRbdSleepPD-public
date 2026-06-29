"""
LR+/LR- for the pre-defined RBD risk groups (percentile 2g and 3g).

Uses the SAME thresholds as the survival analysis (derived from the
validation-set RBD score distribution, stored in risk_percentile_2g.json
and risk_percentile_3g.json).  Applied to the raw ``abk_rbd_score_mean``
so risk-group membership is identical to the Cox analysis.

For ordinal/multi-category tests the correct metric is the
*stratum-specific likelihood ratio*:

    LR(k) = P(category = k | PD)  /  P(category = k | no PD)
           = [n(k, case) / n_cases] / [n(k, ctrl) / n_ctrls]

A monotone ordering (Low < Mid < High) is expected; tested explicitly.
CIs are log-normal (same method as the continuous-threshold analysis).
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import config as project_config
from library.lr_analysis.config import MIN_CELL_COUNT, RBD_COL, RESULTS_SUBDIR
from library.lr_analysis.data_prep import build_analysis_frame
from library.lr_analysis.lr_metrics import _wilson_ci


# ── Threshold loading ─────────────────────────────────────────────────────────

def _load_thresholds(file_name: str = "ehr_diag_pd_rbd_only_all") -> dict:
    """Load 2g and 3g percentile threshold JSON files.

    Returns
    -------
    dict with keys 'p90', 'p99'.
    """
    base = project_config["pp"]["thresholds"]["root"] / file_name
    with open(base / "risk_percentile_2g.json") as f:
        d2g = json.load(f)
    with open(base / "risk_percentile_3g.json") as f:
        d3g = json.load(f)

    p90 = float(d2g["thresholds"]["p90"])
    p99 = float(d3g["thresholds"]["p99"])
    return {"p90": p90, "p99": p99}


# ── Risk group assignment ─────────────────────────────────────────────────────

def _assign_2g(score: pd.Series, p90: float) -> pd.Series:
    """Assign 2-group labels identical to the survival analysis."""
    return pd.Categorical(
        np.where(score >= p90, "High", "Low"),
        categories=["Low", "High"],
        ordered=True,
    )


def _assign_3g(score: pd.Series, p90: float, p99: float) -> pd.Series:
    """Assign 3-group labels identical to the survival analysis."""
    def _label(x: float) -> str:
        if x >= p99:
            return "High"
        if x >= p90:
            return "Intermediate"
        return "Low"

    return pd.Categorical(
        score.map(_label),
        categories=["Low", "Intermediate", "High"],
        ordered=True,
    )


# ── Stratum-specific LR ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class StratumLR:
    """Stratum-specific LR for one risk category."""

    scheme: str            # "2g" or "3g"
    category: str          # "Low", "Intermediate", "High"
    n_total: int
    n_cases: int
    n_controls: int
    pct_cases: float       # n_cases / n_cases_total
    pct_controls: float    # n_controls / n_controls_total
    event_rate: float      # n_cases / n_total (within stratum)
    lr: float              # stratum-specific LR
    lr_lci: float
    lr_uci: float
    stable: bool


def _stratum_lr(
    n_k_case: int,
    n_k_ctrl: int,
    n_case_total: int,
    n_ctrl_total: int,
    scheme: str,
    category: str,
) -> StratumLR:
    """Compute stratum-specific LR with log-normal 95% CI.

    LR(k) = [n_k_case / n_case_total] / [n_k_ctrl / n_ctrl_total]

    SE(ln LR) = sqrt(1/n_k_case - 1/n_case_total + 1/n_k_ctrl - 1/n_ctrl_total)
    (exact Woolf log-transform SE for stratum-specific LR)
    """
    stable = (
        n_k_case >= MIN_CELL_COUNT
        and n_k_ctrl >= MIN_CELL_COUNT
    )
    if not stable:
        warnings.warn(
            f"[{scheme}] {category}: sparse cells "
            f"(cases={n_k_case}, controls={n_k_ctrl}).",
            UserWarning,
            stacklevel=3,
        )

    p_case = n_k_case / n_case_total if n_case_total > 0 else float("nan")
    p_ctrl = n_k_ctrl / n_ctrl_total if n_ctrl_total > 0 else float("nan")
    lr = p_case / p_ctrl if p_ctrl > 0 else float("nan")
    n_total = n_k_case + n_k_ctrl
    event_rate = n_k_case / n_total if n_total > 0 else float("nan")

    # Log-normal CI (Woolf method for stratum LR)
    if n_k_case > 0 and n_k_ctrl > 0 and n_case_total > 0 and n_ctrl_total > 0:
        se_ln = np.sqrt(
            1 / n_k_case - 1 / n_case_total
            + 1 / n_k_ctrl - 1 / n_ctrl_total
        )
        ln_lr = np.log(lr)
        lr_lci = float(np.exp(ln_lr - 1.96 * se_ln))
        lr_uci = float(np.exp(ln_lr + 1.96 * se_ln))
    else:
        lr_lci = lr_uci = float("nan")

    return StratumLR(
        scheme=scheme,
        category=category,
        n_total=n_total,
        n_cases=n_k_case,
        n_controls=n_k_ctrl,
        pct_cases=round(p_case * 100, 2),
        pct_controls=round(p_ctrl * 100, 2),
        event_rate=round(event_rate * 100, 3),
        lr=round(lr, 4),
        lr_lci=round(lr_lci, 4),
        lr_uci=round(lr_uci, 4),
        stable=stable,
    )


def compute_risk_group_lrs(
    df: pd.DataFrame,
    is_case: pd.Series,
    p90: float,
    p99: float,
) -> tuple[list[StratumLR], list[StratumLR]]:
    """Compute stratum-specific LRs for 2g and 3g risk groups.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis-set rows (incident PD + controls).
    is_case : pd.Series[bool]
    p90 : float
        90th-percentile threshold (boundary between Low and High/Intermediate).
    p99 : float
        99th-percentile threshold (boundary between Intermediate and High).

    Returns
    -------
    results_2g : list[StratumLR]
        [Low, High]
    results_3g : list[StratumLR]
        [Low, Intermediate, High]
    """
    score = df[RBD_COL]
    n_case_total = int(is_case.sum())
    n_ctrl_total = int((~is_case).sum())

    # ── 2-group ──────────────────────────────────────────────────────────────
    groups_2g = _assign_2g(score, p90)
    results_2g: list[StratumLR] = []
    for cat in ["Low", "High"]:
        mask = groups_2g == cat
        results_2g.append(_stratum_lr(
            n_k_case=int((mask & is_case).sum()),
            n_k_ctrl=int((mask & ~is_case).sum()),
            n_case_total=n_case_total,
            n_ctrl_total=n_ctrl_total,
            scheme="2g",
            category=cat,
        ))

    # ── 3-group ──────────────────────────────────────────────────────────────
    groups_3g = _assign_3g(score, p90, p99)
    results_3g: list[StratumLR] = []
    for cat in ["Low", "Intermediate", "High"]:
        mask = groups_3g == cat
        results_3g.append(_stratum_lr(
            n_k_case=int((mask & is_case).sum()),
            n_k_ctrl=int((mask & ~is_case).sum()),
            n_case_total=n_case_total,
            n_ctrl_total=n_ctrl_total,
            scheme="3g",
            category=cat,
        ))

    # Monotonicity check: LR should increase from Low → High
    lrs_3g = [r.lr for r in results_3g]
    if lrs_3g[0] > lrs_3g[1] or lrs_3g[1] > lrs_3g[2]:
        warnings.warn(
            f"3g LRs are not monotone: {lrs_3g}. "
            "Consider whether tertiles are ordered correctly.",
            UserWarning,
            stacklevel=2,
        )

    return results_2g, results_3g


def compute_risk_group_lrs_from_columns(
    df: pd.DataFrame,
    is_case: pd.Series,
    col_3g: str = "rg_pctl3",
    col_2g: str = "rg_pctl2",
) -> tuple[list[StratumLR], list[StratumLR]]:
    """Compute stratum-specific LRs using pre-built risk-group columns.

    Uses the same group assignments as the survival analysis rather than
    re-deriving from JSON thresholds.  The 3g column uses the label "Mid"
    internally; output categories are normalised to "Intermediate".

    Parameters
    ----------
    df : pd.DataFrame
    is_case : pd.Series[bool]
    col_3g : str
        Column with Low / Mid / High labels (3-group scheme).
    col_2g : str
        Column with Low / High labels (2-group scheme).

    Returns
    -------
    results_2g : list[StratumLR]
    results_3g : list[StratumLR]
    """
    n_case_total = int(is_case.sum())
    n_ctrl_total = int((~is_case).sum())

    # ── 2-group ──────────────────────────────────────────────────────────────
    grp2 = df[col_2g]
    results_2g: list[StratumLR] = []
    for cat in ["Low", "High"]:
        mask = grp2 == cat
        results_2g.append(_stratum_lr(
            n_k_case=int((mask & is_case).sum()),
            n_k_ctrl=int((mask & ~is_case).sum()),
            n_case_total=n_case_total,
            n_ctrl_total=n_ctrl_total,
            scheme="2g",
            category=cat,
        ))

    # ── 3-group ── rg_pctl3 uses "Mid"; output normalised to "Intermediate" ──
    grp3 = df[col_3g]
    col_to_out = [("Low", "Low"), ("Mid", "Intermediate"), ("High", "High")]
    results_3g: list[StratumLR] = []
    for col_cat, out_cat in col_to_out:
        mask = grp3 == col_cat
        results_3g.append(_stratum_lr(
            n_k_case=int((mask & is_case).sum()),
            n_k_ctrl=int((mask & ~is_case).sum()),
            n_case_total=n_case_total,
            n_ctrl_total=n_ctrl_total,
            scheme="3g",
            category=out_cat,
        ))

    lrs_3g = [r.lr for r in results_3g]
    if lrs_3g[0] > lrs_3g[1] or lrs_3g[1] > lrs_3g[2]:
        warnings.warn(
            f"3g LRs are not monotone: {lrs_3g}.",
            UserWarning,
            stacklevel=2,
        )

    return results_2g, results_3g


# ── Table formatting ──────────────────────────────────────────────────────────

def _stratum_lrs_to_df(results: list[StratumLR]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "scheme": r.scheme,
            "category": r.category,
            "N": r.n_total,
            "n_cases": r.n_cases,
            "n_controls": r.n_controls,
            "pct_of_cases_%": r.pct_cases,
            "pct_of_controls_%": r.pct_controls,
            "event_rate_%": r.event_rate,
            "LR": r.lr,
            "LR_lci": r.lr_lci,
            "LR_uci": r.lr_uci,
            "stable": r.stable,
        })
    return pd.DataFrame(rows)


def _print_table(results: list[StratumLR], scheme: str, p90: float, p99: float) -> None:
    print(f"\n{'-'*72}")
    print(f"  Risk-group LR - {scheme} (p90={p90:.3f}" +
          (f", p99={p99:.3f}" if scheme == "3g" else "") + ")")
    print(f"{'-'*72}")
    hdr = f"  {'Category':<15s}  {'N':>7s}  {'Cases':>6s}  {'% cases':>7s}  "
    hdr += f"{'% ctrls':>7s}  {'EvtRate%':>8s}  {'LR [95% CI]':>25s}  Stable"
    print(hdr)
    print(f"  {'-'*15}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*25}  {'-'*6}")
    for r in results:
        ci_str = f"{r.lr:.2f} [{r.lr_lci:.2f}–{r.lr_uci:.2f}]"
        print(
            f"  {r.category:<15s}  {r.n_total:>7,}  {r.n_cases:>6,}  "
            f"{r.pct_cases:>6.1f}%  {r.pct_controls:>6.1f}%  "
            f"{r.event_rate:>7.2f}%  {ci_str:>25s}  {'ok' if r.stable else '!!'}"
        )


# ── Runner ────────────────────────────────────────────────────────────────────

def run_risk_group_lr(file_name: str = "ehr_diag_pd_rbd_only_all") -> None:
    """Compute and save risk-group LR tables."""
    print("=" * 60)
    print("Risk-group LR analysis (survival analysis risk categories)")
    print("=" * 60)

    # Load data
    frame = build_analysis_frame(file_name)

    # Prefer pre-built columns (same boundaries as survival analysis)
    if "rg_pctl3" in frame.df.columns and "rg_pctl2" in frame.df.columns:
        print("  Using pre-built rg_pctl3/rg_pctl2 columns (consistent with Cox model).")
        p90 = float(frame.df.loc[frame.df["rg_pctl3"].isin(["Mid", "High"]), RBD_COL].min())
        p99 = float(frame.df.loc[frame.df["rg_pctl3"] == "High", RBD_COL].min())
        print(f"  Effective thresholds from column: p90~{p90:.4f}, p99~{p99:.4f}")
        res_2g, res_3g = compute_risk_group_lrs_from_columns(
            df=frame.df, is_case=frame.is_case,
        )
    else:
        print("  Columns rg_pctl3/rg_pctl2 not found — falling back to JSON thresholds.")
        thresholds = _load_thresholds(file_name)
        p90, p99 = thresholds["p90"], thresholds["p99"]
        print(f"  Thresholds: p90 = {p90:.4f}, p99 = {p99:.4f}")
        res_2g, res_3g = compute_risk_group_lrs(
            df=frame.df, is_case=frame.is_case, p90=p90, p99=p99,
        )

    # Print tables
    _print_table(res_2g, "2g", p90, p99)
    _print_table(res_3g, "3g", p90, p99)

    # Save
    out_dir = project_config["results"]["root"] / RESULTS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df_2g = _stratum_lrs_to_df(res_2g)
    df_3g = _stratum_lrs_to_df(res_3g)
    combined = pd.concat([df_2g, df_3g], ignore_index=True)
    combined.to_csv(out_dir / "risk_group_lr.csv", index=False)

    # Append to Excel workbook
    excel_path = out_dir / "lr_analysis_tables.xlsx"
    if excel_path.exists():
        from openpyxl import load_workbook
        with pd.ExcelWriter(
            excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            df_2g.to_excel(writer, sheet_name="RiskGroup_LR_2g", index=False)
            df_3g.to_excel(writer, sheet_name="RiskGroup_LR_3g", index=False)
        print(f"\n  Appended to {excel_path}")
    else:
        combined.to_csv(out_dir / "risk_group_lr.csv", index=False)

    print(f"  CSV: {out_dir / 'risk_group_lr.csv'}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    run_risk_group_lr()
