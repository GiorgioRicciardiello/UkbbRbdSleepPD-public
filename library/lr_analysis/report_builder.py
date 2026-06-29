"""
Assemble output tables for the LR-MDS analysis.

Writes to CSV and a consolidated multi-sheet Excel workbook.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from library.lr_analysis.lr_metrics import (
    EmpiricalMarkerLR,
    LogisticORResult,
    LRResult,
)
from library.lr_analysis.mds_bayesian import PosteriorSummary


def _lr_results_to_df(results: list[LRResult]) -> pd.DataFrame:
    """Flatten a list of LRResult to a DataFrame with expanded CI columns."""
    return pd.DataFrame([r.to_dict() for r in results])


def _empirical_lrs_to_df(results: list[EmpiricalMarkerLR]) -> pd.DataFrame:
    """Flatten EmpiricalMarkerLR results to DataFrame with expanded CIs."""
    rows = []
    for e in results:
        row = {
            "col": e.col,
            "label": e.label,
            "lr_pos": e.lr_pos,
            "lr_pos_lci": e.lr_pos_ci[0],
            "lr_pos_uci": e.lr_pos_ci[1],
            "lr_neg": e.lr_neg,
            "lr_neg_lci": e.lr_neg_ci[0],
            "lr_neg_uci": e.lr_neg_ci[1],
            "tp": e.tp, "fp": e.fp, "fn": e.fn, "tn": e.tn,
            "stable": e.stable,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _or_results_to_df(results: list[LogisticORResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results])


def _posterior_summaries_to_df(summaries: list[PosteriorSummary]) -> pd.DataFrame:
    return pd.DataFrame([asdict(s) for s in summaries])


def write_tables(
    out_dir: Path,
    lr_profile: pd.DataFrame,
    lr_at_youden: list[LRResult],
    sex_stratified_lr: list[LRResult],
    or_results: list[LogisticORResult],
    empirical_marker_lrs: list[EmpiricalMarkerLR],
    posterior_summaries: list[PosteriorSummary],
    zscore_params: dict,
) -> None:
    """Write all analysis tables to CSV and Excel.

    Parameters
    ----------
    out_dir : Path
        Output directory.
    lr_profile : pd.DataFrame
        LR profile over threshold grid (from compute_lr_profile).
    lr_at_youden : list[LRResult]
        Overall LR at Youden threshold.
    sex_stratified_lr : list[LRResult]
        [female_result, male_result] at Youden threshold.
    or_results : list[LogisticORResult]
        [unadjusted_or, adjusted_or].
    empirical_marker_lrs : list[EmpiricalMarkerLR]
        Empirical LRs for viable prodromal markers (C1).
    posterior_summaries : list[PosteriorSummary]
        Summary stats for C1 and C2 posteriors.
    zscore_params : dict
        {'mu': float, 'sigma': float, 'youden_threshold': float}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Individual CSVs
    lr_profile.to_csv(out_dir / "lr_profile.csv", index=False)

    youden_df = _lr_results_to_df(lr_at_youden + sex_stratified_lr)
    youden_df.to_csv(out_dir / "lr_at_youden.csv", index=False)

    or_df = _or_results_to_df(or_results)
    or_df.to_csv(out_dir / "logistic_or.csv", index=False)

    emp_df = _empirical_lrs_to_df(empirical_marker_lrs)
    emp_df.to_csv(out_dir / "empirical_prodromal_lrs.csv", index=False)

    post_df = _posterior_summaries_to_df(posterior_summaries)
    post_df.to_csv(out_dir / "posterior_summaries.csv", index=False)

    params_df = pd.DataFrame([zscore_params])
    params_df.to_csv(out_dir / "zscore_params.csv", index=False)

    # Consolidated Excel workbook
    excel_path = out_dir / "lr_analysis_tables.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        lr_profile.to_excel(writer, sheet_name="LR_Profile", index=False)
        youden_df.to_excel(writer, sheet_name="LR_Youden_Sex", index=False)
        or_df.to_excel(writer, sheet_name="Logistic_OR", index=False)
        emp_df.to_excel(writer, sheet_name="Empirical_Prodromal_LRs", index=False)
        post_df.to_excel(writer, sheet_name="Posterior_Summaries", index=False)
        params_df.to_excel(writer, sheet_name="ZScore_Params", index=False)

    print(f"[report_builder] Tables written to {out_dir}")
    print(f"  Excel workbook: {excel_path}")
