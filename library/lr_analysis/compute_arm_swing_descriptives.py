"""
Compute descriptive statistics for arm swing variables stratified by RBD risk groups.
Output table for manuscript methods/results section.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Set up sys.path BEFORE importing project modules
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import numpy as np

from config.config import config as project_config
from library.lr_analysis.config import (
    ARM_SWING_VARS,
    RBD_TERTILE_COL,
)
from library.lr_analysis.data_prep import build_analysis_frame, filter_cohort

# ── Output path ────────────────────────────────────────────────────────────────

OUTPUT_DIR = project_config["results"]["root"] / "lr_analysis" / "arm_swing"


def compute_descriptives() -> pd.DataFrame:
    """Compute descriptive stats for arm swing by RBD group."""
    print("\nLoading data and filtering to arm_swing cohort...")
    frame_base = build_analysis_frame()

    # Create percentile_3g RBD groups if not present
    if RBD_TERTILE_COL not in frame_base.df.columns:
        print(f"  Creating {RBD_TERTILE_COL} using percentile method...")
        rbd_prob = frame_base.df["abk_rbd_score_mean"]
        p90 = rbd_prob.quantile(0.90)
        p99 = rbd_prob.quantile(0.99)

        rbd_groups = pd.cut(
            rbd_prob,
            bins=[-np.inf, p90, p99, np.inf],
            labels=["Low (0-90%)", "Mid (90-99%)", "High (99-100%)"],
            include_lowest=True,
        )
        frame_base.df[RBD_TERTILE_COL] = rbd_groups
        print(f"    Thresholds: p90={p90:.4f}, p99={p99:.4f}")
        print(f"    Group distribution:\n{rbd_groups.value_counts().sort_index()}")

    # Filter to complete-case cohort
    frame_arm, _ = filter_cohort(frame_base, "arm_swing_interaction")

    # Compute descriptive stats by RBD group
    print("\nComputing descriptive statistics...")

    # Get RBD groups in sorted order (Low, Mid/Intermediate, High)
    rbd_groups = sorted(frame_arm.df[RBD_TERTILE_COL].dropna().unique())
    print(f"  RBD groups found: {rbd_groups}")

    rows = []

    for col, label in ARM_SWING_VARS.items():
        for rbd_group in rbd_groups:
            subset = frame_arm.df[frame_arm.df[RBD_TERTILE_COL] == rbd_group][col]
            subset_clean = subset.dropna()

            if len(subset_clean) > 0:
                row = {
                    "Variable": label,
                    "RBD Group": rbd_group,
                    "N": len(subset_clean),
                    "Mean": round(float(subset_clean.mean()), 4),
                    "SD": round(float(subset_clean.std()), 4),
                    "Median": round(float(subset_clean.median()), 4),
                    "Q1": round(float(subset_clean.quantile(0.25)), 4),
                    "Q3": round(float(subset_clean.quantile(0.75)), 4),
                    "Min": round(float(subset_clean.min()), 4),
                    "Max": round(float(subset_clean.max()), 4),
                }
                rows.append(row)

    df_descriptives = pd.DataFrame(rows)

    # Save CSV
    output_path = OUTPUT_DIR / "arm_swing_descriptives_by_rbd.csv"
    df_descriptives.to_csv(output_path, index=False)
    print(f"\n[OK] Saved: {output_path.name}")

    return df_descriptives


if __name__ == "__main__":
    compute_descriptives()
