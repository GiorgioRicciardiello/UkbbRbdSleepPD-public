"""
Runner for secondary LR analyses: Cognitive variables, PRS, and TMT.

Phase 1: Univariate and adjusted logistic OR for each variable, by cohort:
  - Cohort B (Cognitive): N≈600 complete case, all 5 cognitive variables
  - Cohort C (Genetic): N≈1,100, PRS with ancestry PC adjustment
  - Cohort D (TMT): N≈650, three TMT measures (A, B, B/A ratio)

Generates Table 2B, Table 2C, Table 2D with sample completeness reporting.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

from config.config import config as project_config
from library.lr_analysis.config import (
    COGNITIVE_VARS,
    GENETIC_VARS,
    TMT_VARS,
)
from library.lr_analysis.data_prep import build_analysis_frame, filter_cohort
from library.lr_analysis.lr_metrics import compute_logistic_or_cohort

# ── Output paths ───────────────────────────────────────────────────────────────

RESULTS_SUBDIR = "lr_analysis"
OUTPUT_DIR = project_config["results"]["root"] / RESULTS_SUBDIR


# ── Result collection ──────────────────────────────────────────────────────────

def _build_result_row(
    variable_col: str,
    variable_label: str,
    result_crude,
    result_adjusted,
    cohort_stats,
) -> dict:
    """Build a single row for results table."""
    return {
        "variable": variable_col,
        "label": variable_label,
        "crude_or": result_crude.or_estimate,
        "crude_lci": result_crude.or_lci,
        "crude_uci": result_crude.or_uci,
        "crude_p": result_crude.p_value,
        "adjusted_or": result_adjusted.or_estimate,
        "adjusted_lci": result_adjusted.or_lci,
        "adjusted_uci": result_adjusted.or_uci,
        "adjusted_p": result_adjusted.p_value,
        "n_total": cohort_stats["n_total"],
        "n_cases": cohort_stats["n_cases"],
        "n_controls": cohort_stats["n_controls"],
        "pct_complete": cohort_stats["pct_complete"],
    }


# ── Cohort B: Cognitive Variables ──────────────────────────────────────────────

def run_cognitive_analysis(frame_base) -> pd.DataFrame:
    """Analyze 5 cognitive variables in complete-case cohort (N≈600)."""
    print("\n" + "=" * 70)
    print("COHORT B: Cognitive Variables")
    print("=" * 70)

    # Filter to cognitive cohort
    frame_cog, cog_stats = filter_cohort(frame_base, "cognitive")
    print(f"Cohort B: N={cog_stats['n_total']:,} "
          f"(cases={cog_stats['n_cases']}, controls={cog_stats['n_controls']}, "
          f"{cog_stats['pct_complete']:.1f}% complete)")

    results = []
    for col, label in COGNITIVE_VARS.items():
        print(f"\n  {label} ({col})...")

        # Crude
        res_crude, stats = compute_logistic_or_cohort(
            frame_cog.df, frame_cog.is_case, col, "cognitive", adjusted=False
        )

        # Adjusted
        res_adj, _ = compute_logistic_or_cohort(
            frame_cog.df, frame_cog.is_case, col, "cognitive", adjusted=True
        )

        row = _build_result_row(col, label, res_crude, res_adj, stats)
        results.append(row)

        print(f"    Crude: OR={res_crude.or_estimate:.3f} "
              f"({res_crude.or_lci:.3f}–{res_crude.or_uci:.3f}) p={res_crude.p_value:.4f}")
        print(f"    Adjusted: OR={res_adj.or_estimate:.3f} "
              f"({res_adj.or_lci:.3f}–{res_adj.or_uci:.3f}) p={res_adj.p_value:.4f}")

    return pd.DataFrame(results)


# ── Cohort D: Trail Making Test ────────────────────────────────────────────────

def run_tmt_analysis(frame_base) -> pd.DataFrame:
    """Analyze 3 TMT measures (A, B, B/A ratio) in complete-case cohort (N≈650)."""
    print("\n" + "=" * 70)
    print("COHORT D: Trail Making Test")
    print("=" * 70)

    # Filter to TMT cohort
    frame_tmt, tmt_stats = filter_cohort(frame_base, "tmt")
    print(f"Cohort D: N={tmt_stats['n_total']:,} "
          f"(cases={tmt_stats['n_cases']}, controls={tmt_stats['n_controls']}, "
          f"{tmt_stats['pct_complete']:.1f}% complete)")

    results = []
    for col, label in TMT_VARS.items():
        print(f"\n  {label} ({col})...")

        # Crude
        res_crude, stats = compute_logistic_or_cohort(
            frame_tmt.df, frame_tmt.is_case, col, "tmt", adjusted=False
        )

        # Adjusted
        res_adj, _ = compute_logistic_or_cohort(
            frame_tmt.df, frame_tmt.is_case, col, "tmt", adjusted=True
        )

        row = _build_result_row(col, label, res_crude, res_adj, stats)
        results.append(row)

        print(f"    Crude: OR={res_crude.or_estimate:.3f} "
              f"({res_crude.or_lci:.3f}–{res_crude.or_uci:.3f}) p={res_crude.p_value:.4f}")
        print(f"    Adjusted: OR={res_adj.or_estimate:.3f} "
              f"({res_adj.or_lci:.3f}–{res_adj.or_uci:.3f}) p={res_adj.p_value:.4f}")

    return pd.DataFrame(results)


# ── Cohort C: Genetic Risk Score (PRS) ─────────────────────────────────────────

def run_genetic_analysis(frame_base) -> pd.DataFrame:
    """Analyze PRS in genetic cohort (N≈1,100) with ancestry PC adjustment."""
    print("\n" + "=" * 70)
    print("COHORT C: Polygenic Risk Score (with Ancestry Adjustment)")
    print("=" * 70)

    # Filter to genetic cohort
    frame_gen, gen_stats = filter_cohort(frame_base, "genetic")
    print(f"Cohort C: N={gen_stats['n_total']:,} "
          f"(cases={gen_stats['n_cases']}, controls={gen_stats['n_controls']}, "
          f"{gen_stats['pct_complete']:.1f}% complete)")

    results = []
    for col, label in GENETIC_VARS.items():
        print(f"\n  {label} ({col})...")

        # Crude
        res_crude, stats = compute_logistic_or_cohort(
            frame_gen.df, frame_gen.is_case, col, "genetic", adjusted=False
        )

        # Demographic adjustment (age, sex)
        # Note: genetic cohort doesn't include BMI in current definition
        # We'll do age+sex only for this model
        res_demographic, _ = compute_logistic_or_cohort(
            frame_gen.df, frame_gen.is_case, col, "genetic", adjusted=True
        )
        # This will adjust for age + sex + PC1 + PC2 + PC3 (full ancestry adjustment)
        # For "demographic" only (age+sex), we'd need a separate function.
        # For now, use the full ancestry-adjusted model.

        row = _build_result_row(col, label, res_crude, res_demographic, stats)
        results.append(row)

        print(f"    Crude: OR={res_crude.or_estimate:.3f} "
              f"({res_crude.or_lci:.3f}–{res_crude.or_uci:.3f}) p={res_crude.p_value:.4f}")
        print(f"    Ancestry-adjusted: OR={res_demographic.or_estimate:.3f} "
              f"({res_demographic.or_lci:.3f}–{res_demographic.or_uci:.3f}) "
              f"p={res_demographic.p_value:.4f}")

    return pd.DataFrame(results)


# ── Main runner ────────────────────────────────────────────────────────────────

def run_cognitive_prs_analysis(file_name: str = "ehr_diag_pd_rbd_only_all") -> None:
    """Orchestrate Phase 1 analyses (cognitive, TMT, PRS)."""
    print("\n" + "=" * 70)
    print("LR ANALYSIS: Cognitive, TMT, and Genetic Risk Factors")
    print("=" * 70)

    # Load base analysis frame
    print(f"\nLoading data from {file_name}...")
    frame_base = build_analysis_frame(file_name)

    # Run cohort-specific analyses
    df_cognitive = run_cognitive_analysis(frame_base)
    df_tmt = run_tmt_analysis(frame_base)
    df_genetic = run_genetic_analysis(frame_base)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_cognitive.to_csv(OUTPUT_DIR / "cognitive_or_results.csv", index=False)
    print(f"\n[OK] Cognitive results saved: cognitive_or_results.csv")

    df_tmt.to_csv(OUTPUT_DIR / "tmt_or_results.csv", index=False)
    print(f"[OK] TMT results saved: tmt_or_results.csv")

    df_genetic.to_csv(OUTPUT_DIR / "genetic_or_results.csv", index=False)
    print(f"[OK] Genetic results saved: genetic_or_results.csv")

    # Save combined results JSON
    results_json = {
        "analysis": "cognitive_prs_lr_phase1",
        "file_name": file_name,
        "cohort_b_cognitive": {
            "n": int(df_cognitive["n_total"].iloc[0]),
            "n_cases": int(df_cognitive["n_cases"].iloc[0]),
            "n_controls": int(df_cognitive["n_controls"].iloc[0]),
            "pct_complete": float(df_cognitive["pct_complete"].iloc[0]),
        },
        "cohort_d_tmt": {
            "n": int(df_tmt["n_total"].iloc[0]),
            "n_cases": int(df_tmt["n_cases"].iloc[0]),
            "n_controls": int(df_tmt["n_controls"].iloc[0]),
            "pct_complete": float(df_tmt["pct_complete"].iloc[0]),
        },
        "cohort_c_genetic": {
            "n": int(df_genetic["n_total"].iloc[0]),
            "n_cases": int(df_genetic["n_cases"].iloc[0]),
            "n_controls": int(df_genetic["n_controls"].iloc[0]),
            "pct_complete": float(df_genetic["pct_complete"].iloc[0]),
        },
    }

    with open(OUTPUT_DIR / "cognitive_prs_summary.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"✓ Summary saved: cognitive_prs_summary.json")

    print("\n" + "=" * 70)
    print("Phase 1 complete. Ready for Phase 2 (RBD stratification analysis).")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    run_cognitive_prs_analysis()
