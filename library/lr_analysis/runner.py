"""
Runner — orchestrates the full LR-MDS analysis.

Execution order:
1. Load and prepare data (data_prep).
2. Option A — LR+/LR- for actigraphy RBD z-score.
   a. Youden-optimal threshold.
   b. LR at Youden: overall + sex-stratified.
   c. LR profile across threshold grid.
   d. Logistic OR: unadjusted + adjusted (continuous z-score).
3. Option C1 — Empirical Bayesian.
   a. Empirical LRs for 4 viable prodromal markers.
   b. Per-subject posterior (UKBB prior + empirical LRs + actigraphy LR).
4. Option C2 — Hybrid Bayesian.
   a. Per-subject posterior (Berg prior + Heinzel LRs + actigraphy LR).
5. Summary statistics for posteriors.
6. Write tables and figures.

Run as:
    C:\\Users\\riccig01\\anaconda3\\envs\\stats_env\\python.exe -m src.lr_analysis.runner
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.config import config as project_config

from library.lr_analysis.config import (
    MDS_AGE_PRIOR,
    RBD_ZSCORE_COL,
    RESULTS_SUBDIR,
)
from library.lr_analysis.data_prep import build_analysis_frame
from library.lr_analysis.lr_metrics import (
    compute_empirical_marker_lrs,
    compute_logistic_or,
    compute_lr_at_threshold,
    compute_lr_profile,
    compute_sex_stratified_lr,
    compute_youden_threshold,
)
from library.lr_analysis.mds_bayesian import (
    compute_posterior_c1_empirical,
    compute_posterior_c2_hybrid,
    summarise_posteriors,
)
from library.lr_analysis.plotting import (
    plot_combined_lr_forest,
    plot_empirical_vs_published,
    plot_fagan_nomogram,
    plot_lr_profile,
    plot_posteriors,
    plot_sex_stratified_lr,
)
from library.lr_analysis.report_builder import write_tables
from library.lr_analysis.risk_group_lr import (
    compute_risk_group_lrs,
    compute_risk_group_lrs_from_columns,
    _load_thresholds,
)


def run() -> None:
    """Execute the full LR-MDS analysis pipeline."""
    out_dir = project_config["results"]["root"] / RESULTS_SUBDIR
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Data preparation ──────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 — Data preparation")
    print("=" * 60)
    frame = build_analysis_frame()
    df = frame.df
    is_case = frame.is_case
    is_ctrl = frame.is_ctrl

    # ── 2. Option A — Actigraphy RBD z-score LR ─────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — Option A: Actigraphy RBD z-score LR")
    print("=" * 60)

    zs = df[RBD_ZSCORE_COL].dropna().values
    ic = is_case[df[RBD_ZSCORE_COL].notna()].values
    youden_thr = compute_youden_threshold(zs, ic)
    print(f"  Youden-optimal z-score threshold: {youden_thr:.4f}")

    # 2a. Overall LR at Youden threshold
    lr_overall = compute_lr_at_threshold(df, is_case, threshold=youden_thr)
    print(
        f"  Overall: LR+ = {lr_overall.lr_pos:.2f} "
        f"[{lr_overall.lr_pos_ci[0]:.2f}–{lr_overall.lr_pos_ci[1]:.2f}]  "
        f"LR- = {lr_overall.lr_neg:.3f} "
        f"[{lr_overall.lr_neg_ci[0]:.3f}–{lr_overall.lr_neg_ci[1]:.3f}]"
    )

    # 2b. Sex-stratified LR at Youden threshold
    sex_results = compute_sex_stratified_lr(df, is_case, threshold=youden_thr)
    for r in sex_results:
        print(
            f"  {r.stratum.capitalize()}: LR+ = {r.lr_pos:.2f} "
            f"[{r.lr_pos_ci[0]:.2f}–{r.lr_pos_ci[1]:.2f}]  "
            f"LR- = {r.lr_neg:.3f} "
            f"[{r.lr_neg_ci[0]:.3f}–{r.lr_neg_ci[1]:.3f}]"
        )

    # 2c. LR profile
    lr_profile_df = compute_lr_profile(df, is_case)

    # 2d. Logistic OR (unadjusted + adjusted)
    or_unadj = compute_logistic_or(df, is_case, adjusted=False)
    or_adj = compute_logistic_or(df, is_case, adjusted=True)
    print(
        f"  Unadjusted OR per 1 SD: {or_unadj.or_estimate:.2f} "
        f"[{or_unadj.or_lci:.2f}–{or_unadj.or_uci:.2f}]  p={or_unadj.p_value:.4f}"
    )
    print(
        f"  Adjusted OR per 1 SD:   {or_adj.or_estimate:.2f} "
        f"[{or_adj.or_lci:.2f}–{or_adj.or_uci:.2f}]  p={or_adj.p_value:.4f}"
    )

    # 2e. Risk group LRs (3-group scheme for combined forest plot)
    # Prefer pre-built columns for consistency with Cox model boundaries.
    if "rg_pctl3" in df.columns and "rg_pctl2" in df.columns:
        res_2g, res_3g = compute_risk_group_lrs_from_columns(df=df, is_case=is_case)
    else:
        thresholds = _load_thresholds(file_name="ehr_diag_pd_rbd_only_all")
        p90, p99 = thresholds["p90"], thresholds["p99"]
        res_2g, res_3g = compute_risk_group_lrs(df=df, is_case=is_case, p90=p90, p99=p99)
    print("  Risk-group LRs (3g):")
    for r in res_3g:
        print(
            f"    {r.category:<12s}  LR = {r.lr:.2f} "
            f"[{r.lr_lci:.2f}–{r.lr_uci:.2f}]"
        )

    # ── 3. Option C1 — Empirical Bayesian ────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — Option C1: Empirical Bayesian")
    print("=" * 60)

    empirical_lrs = compute_empirical_marker_lrs(df, is_case)
    for e in empirical_lrs:
        print(
            f"  {e.label:<45s}  LR+ = {e.lr_pos:.2f} "
            f"[{e.lr_pos_ci[0]:.2f}–{e.lr_pos_ci[1]:.2f}]  "
            f"LR- = {e.lr_neg:.3f}"
        )

    posterior_c1 = compute_posterior_c1_empirical(
        df=df,
        ukbb_age_prior=frame.ukbb_age_prior,
        empirical_marker_lrs=empirical_lrs,
        actigraphy_lr_result=lr_overall,
    )

    # ── 4. Option C2 — Hybrid Bayesian ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — Option C2: Hybrid Bayesian (Berg prior + Heinzel LRs)")
    print("=" * 60)

    posterior_c2 = compute_posterior_c2_hybrid(
        df=df,
        actigraphy_lr_result=lr_overall,
    )

    # ── 5. Posterior summaries ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 — Posterior summaries")
    print("=" * 60)

    summaries_c1 = summarise_posteriors(posterior_c1, is_case, "C1_empirical")
    summaries_c2 = summarise_posteriors(posterior_c2, is_case, "C2_hybrid")
    all_summaries = summaries_c1 + summaries_c2

    for s in all_summaries:
        print(
            f"  [{s.framework}] {s.stratum:<10s}  N={s.n:>6,}  "
            f"median={s.median*100:.3f}%  "
            f">1%: {s.pct_above_1pct:.1f}%  >5%: {s.pct_above_5pct:.1f}%"
        )

    # ── 6. Figures ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6 — Figures")
    print("=" * 60)

    plot_lr_profile(
        lr_profile=lr_profile_df,
        youden_threshold=youden_thr,
        out_path=fig_dir / "lr_profile.png",
    )

    female_r, male_r = sex_results[0], sex_results[1]
    plot_sex_stratified_lr(
        overall=lr_overall,
        female=female_r,
        male=male_r,
        out_path=fig_dir / "sex_stratified_lr.png",
    )

    # Combined LR+ and LR- forest plot
    plot_combined_lr_forest(
        risk_group_lrs_3g=res_3g,
        or_unadj=or_unadj,
        or_adj=or_adj,
        empirical_lrs=empirical_lrs,
        out_path=fig_dir / "combined_lr_forest.png",
    )

    # Representative priors for Fagan nomogram: 50-year-old and 70-year-old
    plot_fagan_nomogram(
        lr_pos=lr_overall.lr_pos,
        lr_neg=lr_overall.lr_neg,
        prior_probs=[MDS_AGE_PRIOR[50], MDS_AGE_PRIOR[70]],
        out_path=fig_dir / "fagan_nomogram.png",
    )

    df_posterior = df.assign(
        posterior_c1=posterior_c1.values,
        posterior_c2=posterior_c2.values,
    )
    plot_posteriors(
        posterior_c2=df_posterior["posterior_c2"],
        posterior_c1=df_posterior["posterior_c1"],
        is_case=is_case,
        out_path=fig_dir / "posteriors.png",
    )

    plot_empirical_vs_published(
        empirical_lrs=empirical_lrs,
        out_path=fig_dir / "empirical_vs_published.png",
    )
    print(f"  Figures saved to {fig_dir}")

    # ── 7. Tables ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7 — Tables")
    print("=" * 60)

    zscore_params = {
        "mu_ctrl": frame.zscore_mu,
        "sigma_ctrl": frame.zscore_sigma,
        "youden_threshold_zscore": youden_thr,
        "n_cases": frame.n_cases,
        "n_controls": frame.n_controls,
    }

    write_tables(
        out_dir=out_dir,
        lr_profile=lr_profile_df,
        lr_at_youden=[lr_overall],
        sex_stratified_lr=sex_results,
        or_results=[or_unadj, or_adj],
        empirical_marker_lrs=empirical_lrs,
        posterior_summaries=all_summaries,
        zscore_params=zscore_params,
    )

    # Save per-subject posteriors
    (
        df_posterior[["posterior_c1", "posterior_c2"]]
        .to_parquet(out_dir / "posteriors_per_subject.parquet")
    )

    # Save audit log
    audit = {
        "zscore_params": zscore_params,
        "ukbb_age_prior": frame.ukbb_age_prior,
        "youden_threshold": youden_thr,
        "lr_overall": {
            "lr_pos": lr_overall.lr_pos,
            "lr_pos_ci": list(lr_overall.lr_pos_ci),
            "lr_neg": lr_overall.lr_neg,
            "lr_neg_ci": list(lr_overall.lr_neg_ci),
        },
    }
    with open(out_dir / "audit_log.json", "w") as f:
        json.dump(audit, f, indent=2, default=str)

    print("\nANALYSIS COMPLETE")
    print(f"  Results: {out_dir}")


if __name__ == "__main__":
    run()
