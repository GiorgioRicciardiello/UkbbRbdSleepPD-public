"""
Phase 3: RBD × Cognitive/TMT/PRS Interaction Analysis via LRT.

For each variable in cognitive, TMT, and genetic cohorts:
1. Fit M1 (reduced): PD ~ C(RBD) + G + age + sex + BMI + alcohol + smoking
2. Fit M2 (full): PD ~ C(RBD) + G + C(RBD):G + age + sex + BMI + alcohol + smoking
3. Compute LRT = -2(llf_M1 - llf_M2), test df=2 (two interaction terms)
4. Extract interaction ORs and main effect of G in full model
5. Save results to CSV with LRT statistics and interaction effects
"""
from __future__ import annotations

import sys
from pathlib import Path

# Set up sys.path BEFORE importing project modules
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import warnings

import numpy as np
import pandas as pd

from config.config import config as project_config
from library.lr_analysis.config import (
    COGNITIVE_VARS,
    GENETIC_VARS,
    INTERACTION_CONFOUNDERS,
    INTERACTION_CONFOUNDERS_GENETIC,
    RBD_TERTILE_COL,
    TMT_VARS,
)
from library.lr_analysis.data_prep import build_analysis_frame, filter_cohort
from library.lr_analysis.lr_metrics import compute_rbd_interaction_test

# ── Output paths ───────────────────────────────────────────────────────────────

RESULTS_SUBDIR = "lr_analysis"
OUTPUT_DIR = project_config["results"]["root"] / RESULTS_SUBDIR


def analyze_variable_interaction(
    frame: object,
    variable_col: str,
    variable_label: str,
    cohort_name: str,
    confounders: list[str],
) -> dict:
    """Test interaction for single variable."""
    print(f"  {variable_label} ({variable_col})...")

    result = compute_rbd_interaction_test(
        frame.df,
        frame.is_case,
        variable_col,
        RBD_TERTILE_COL,
        confounders,
        variable_label,
        cohort_name,
    )
    return result.to_dict()


def run_rbd_interaction_analysis(
    file_name: str = "ehr_diag_pd_rbd_only_all",
) -> None:
    """Phase 3: RBD × predictor interaction analysis via LRT."""
    print("\n" + "=" * 70)
    print("PHASE 3: RBD × Cognitive/TMT/PRS Interaction Analysis (LRT)")
    print("=" * 70)

    # Load base frame
    print(f"\nLoading data from {file_name}...")
    frame_base = build_analysis_frame(file_name)

    # Verify RBD risk group exists; if not, create from z-score tertiles
    if RBD_TERTILE_COL not in frame_base.df.columns:
        print(f"[INFO] Creating {RBD_TERTILE_COL} from z-score tertiles...")
        rbd_z = frame_base.df["rbd_zscore"]
        p33 = rbd_z.quantile(1 / 3)
        p67 = rbd_z.quantile(2 / 3)

        rbd_groups = pd.cut(
            rbd_z,
            bins=[-np.inf, p33, p67, np.inf],
            labels=["Low", "Mid", "High"],
            include_lowest=True,
        )
        frame_base.df[RBD_TERTILE_COL] = rbd_groups

    # Check for alcohol and smoking columns
    alcohol_col = "cov_alcohol"
    smoking_col = "cov_smoking"

    if alcohol_col not in frame_base.df.columns:
        # Fallback to baseline visit (formerly instance 0)
        if "cov_alcohol_20117_bl" in frame_base.df.columns:
            print(f"[INFO] Using cov_alcohol_20117_bl as {alcohol_col}")
            frame_base.df[alcohol_col] = frame_base.df["cov_alcohol_20117_bl"]
        else:
            raise ValueError(
                f"{alcohol_col} and cov_alcohol_20117_bl not found in dataset"
            )

    if smoking_col not in frame_base.df.columns:
        # Fallback to baseline visit (formerly instance 0)
        if "cov_smoking_20116_bl" in frame_base.df.columns:
            print(f"[INFO] Using cov_smoking_20116_bl as {smoking_col}")
            frame_base.df[smoking_col] = frame_base.df["cov_smoking_20116_bl"]
        else:
            raise ValueError(
                f"{smoking_col} and cov_smoking_20116_bl not found in dataset"
            )

    results_summary = {
        "analysis": "phase3_rbd_interaction_lrt",
        "file_name": file_name,
        "cohorts": {},
    }

    # ── Cohort B: Cognitive + Interaction ───────────────────────────────────
    print("\n" + "-" * 70)
    print("COHORT B: Cognitive Variables (RBD Interaction LRT)")
    print("-" * 70)
    frame_cog, cog_stats = filter_cohort(frame_base, "cognitive_interaction")
    print(
        f"N={cog_stats['n_total']:,}, cases={cog_stats['n_cases']}, "
        f"{cog_stats['pct_complete']:.1f}% complete"
    )

    cognitive_results = []
    for col, label in COGNITIVE_VARS.items():
        result = analyze_variable_interaction(
            frame_cog, col, label, "cognitive", INTERACTION_CONFOUNDERS
        )
        cognitive_results.append(result)

    results_summary["cohorts"]["cognitive"] = {
        "n": cog_stats["n_total"],
        "n_cases": cog_stats["n_cases"],
        "variables": len(cognitive_results),
    }

    # ── Cohort D: TMT + Interaction ────────────────────────────────────────
    print("\n" + "-" * 70)
    print("COHORT D: Trail Making Test (RBD Interaction LRT)")
    print("-" * 70)
    frame_tmt, tmt_stats = filter_cohort(frame_base, "tmt_interaction")
    print(
        f"N={tmt_stats['n_total']:,}, cases={tmt_stats['n_cases']}, "
        f"{tmt_stats['pct_complete']:.1f}% complete"
    )

    tmt_results = []
    for col, label in TMT_VARS.items():
        result = analyze_variable_interaction(
            frame_tmt, col, label, "tmt", INTERACTION_CONFOUNDERS
        )
        tmt_results.append(result)

    results_summary["cohorts"]["tmt"] = {
        "n": tmt_stats["n_total"],
        "n_cases": tmt_stats["n_cases"],
        "variables": len(tmt_results),
    }

    # ── Cohort C: Genetic + Interaction ────────────────────────────────────
    print("\n" + "-" * 70)
    print("COHORT C: Polygenic Risk Score (RBD Interaction LRT)")
    print("-" * 70)
    frame_gen, gen_stats = filter_cohort(frame_base, "genetic_interaction")
    print(
        f"N={gen_stats['n_total']:,}, cases={gen_stats['n_cases']}, "
        f"{gen_stats['pct_complete']:.1f}% complete"
    )

    genetic_results = []
    for col, label in GENETIC_VARS.items():
        result = analyze_variable_interaction(
            frame_gen, col, label, "genetic", INTERACTION_CONFOUNDERS_GENETIC
        )
        genetic_results.append(result)

    results_summary["cohorts"]["genetic"] = {
        "n": gen_stats["n_total"],
        "n_cases": gen_stats["n_cases"],
        "variables": len(genetic_results),
    }

    # Save CSVs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_cog = pd.DataFrame(cognitive_results)
    csv_cog = OUTPUT_DIR / "rbd_interaction_cognitive.csv"
    df_cog.to_csv(csv_cog, index=False)
    print(f"\n[OK] Saved: {csv_cog.name}")

    df_tmt = pd.DataFrame(tmt_results)
    csv_tmt = OUTPUT_DIR / "rbd_interaction_tmt.csv"
    df_tmt.to_csv(csv_tmt, index=False)
    print(f"[OK] Saved: {csv_tmt.name}")

    df_gen = pd.DataFrame(genetic_results)
    csv_gen = OUTPUT_DIR / "rbd_interaction_genetic.csv"
    df_gen.to_csv(csv_gen, index=False)
    print(f"[OK] Saved: {csv_gen.name}")

    # Save summary
    with open(OUTPUT_DIR / "rbd_interaction_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"[OK] Saved: rbd_interaction_summary.json")

    print("\n" + "=" * 70)
    print("Phase 3 complete!")
    print(f"Results: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    run_rbd_interaction_analysis()
