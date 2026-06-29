"""
Arm Swing Gait Analysis: Secondary LR analyses for motor biomarkers.

Analyzes 4 arm swing amplitude variables (jacket/wrist, mean/variance) plus 1 PCA composite.
Runs Phases 1–3 (univariate OR, LR profiles, interaction testing).
All cohorts use consistent sample (arm_swing_interaction complete-case set).
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
from sklearn.decomposition import PCA

from config.config import RBD_RISK_COLORS, config as project_config
from library.lr_analysis.config import (
    AGE_COL,
    ARM_SWING_VARS,
    INTERACTION_CONFOUNDERS,
    RBD_TERTILE_COL,
    ZSCORE_THRESHOLD_GRID,
)
from library.lr_analysis.data_prep import build_analysis_frame, filter_cohort
from library.lr_analysis.lr_metrics import (
    compute_logistic_or_cohort,
    compute_lr_profile_by_age,
    compute_lr_profile_by_rbd_strata,
    compute_rbd_interaction_test,
)

# ── Output paths ───────────────────────────────────────────────────────────────

RESULTS_SUBDIR = "lr_analysis"
OUTPUT_DIR = project_config["results"]["root"] / RESULTS_SUBDIR / "arm_swing"
FIGURES_DIR = OUTPUT_DIR / "figures"


# ── PCA composite ──────────────────────────────────────────────────────────────


def _compute_pca_composite(
    df: pd.DataFrame,
    is_case: pd.Series,
    arm_swing_cols: list[str],
) -> tuple[pd.Series, dict]:
    """Fit PCA(n_components=1) on control subjects, transform all.

    Args:
        df: DataFrame with arm swing columns
        is_case: Boolean series (True = case, False = control)
        arm_swing_cols: List of column names

    Returns:
        Tuple: (pc1_series, metadata_dict with variance_explained and loadings)
    """
    # Extract control data
    control_mask = is_case == False
    control_data = df.loc[control_mask, arm_swing_cols].dropna()

    if len(control_data) < 2:
        warnings.warn("Fewer than 2 controls with complete arm swing data; PCA not viable")
        return pd.Series(float("nan"), index=df.index), {
            "variance_explained": float("nan"),
            "loadings": {col: float("nan") for col in arm_swing_cols},
        }

    # Fit PCA on controls only (no-leakage)
    pca = PCA(n_components=1)
    pca.fit(control_data)

    # Transform all subjects (will have NaN for rows with any missing arm_swing value)
    all_data = df[arm_swing_cols].copy()
    pc1_full = pd.Series(float("nan"), index=df.index)

    # Transform only complete cases
    complete_mask = all_data.notna().all(axis=1)
    if complete_mask.sum() > 0:
        pc1_transformed = pca.transform(all_data[complete_mask])
        # Flatten the result (PCA returns shape (n, 1), squeeze to 1D)
        pc1_full.loc[complete_mask] = pc1_transformed.ravel()

    # Extract loadings
    loadings = {
        col: float(pca.components_[0, i])
        for i, col in enumerate(arm_swing_cols)
    }

    metadata = {
        "variance_explained": float(pca.explained_variance_ratio_[0]),
        "loadings": loadings,
    }

    return pc1_full, metadata


def _zscore_series(series: pd.Series, is_case: pd.Series) -> pd.Series:
    """Z-score using control distribution only (no-leakage)."""
    control_mask = is_case == False
    valid = series[control_mask].notna()
    if valid.sum() < 2:
        return pd.Series(float("nan"), index=series.index)

    mu = float(series[control_mask][valid].mean())
    sigma = float(series[control_mask][valid].std())
    if sigma == 0:
        sigma = 1.0

    return (series - mu) / sigma


# ── Phase 1: Logistic OR ───────────────────────────────────────────────────────


def run_phase1(
    frame: object,
    vars_dict: dict[str, str],
    output_dir: Path,
) -> pd.DataFrame:
    """Compute crude and adjusted logistic OR for each variable."""
    print("\n" + "-" * 70)
    print("PHASE 1: Logistic Odds Ratios")
    print("-" * 70)

    results = []

    for col, label in vars_dict.items():
        print(f"  {label} ({col})...")

        # Crude OR
        crude_result, _ = compute_logistic_or_cohort(
            frame.df, frame.is_case, col, "arm_swing", adjusted=False
        )

        # Adjusted OR
        adj_result, _ = compute_logistic_or_cohort(
            frame.df, frame.is_case, col, "arm_swing", adjusted=True
        )

        row = {
            "variable": col,
            "label": label,
            "crude_or": round(crude_result.or_estimate, 6),
            "crude_lci": round(crude_result.or_lci, 6),
            "crude_uci": round(crude_result.or_uci, 6),
            "crude_p": round(crude_result.p_value, 6),
            "adjusted_or": round(adj_result.or_estimate, 6),
            "adjusted_lci": round(adj_result.or_lci, 6),
            "adjusted_uci": round(adj_result.or_uci, 6),
            "adjusted_p": round(adj_result.p_value, 6),
            "n_total": adj_result.n,
            "n_cases": adj_result.n_cases,
            "n_controls": adj_result.n - adj_result.n_cases,
            "converged": adj_result.converged,
        }
        results.append(row)

    df_results = pd.DataFrame(results)
    output_path = output_dir / "arm_swing_or_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\n[OK] Saved: {output_path.name}")

    return df_results


# ── Phase 2: LR Profiles ───────────────────────────────────────────────────────


def run_phase2(
    frame: object,
    vars_dict: dict[str, str],
    output_dir: Path,
) -> None:
    """Compute LR+ / LR- profiles across thresholds, stratified by age and RBD."""
    print("\n" + "-" * 70)
    print("PHASE 2: LR Profile Analysis (Age & RBD Stratification)")
    print("-" * 70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for col, label in vars_dict.items():
        print(f"  {label} ({col})...")

        # Z-score the variable (control distribution)
        z_series = _zscore_series(frame.df[col], frame.is_case)
        frame_working = frame.df.copy()
        frame_working[f"{col}_zscore"] = z_series

        # LR profile by age
        df_by_age = compute_lr_profile_by_age(
            frame_working,
            frame.is_case,
            f"{col}_zscore",
            age_col=AGE_COL,
            thresholds=ZSCORE_THRESHOLD_GRID,
        )

        # LR profile by RBD
        df_by_rbd = compute_lr_profile_by_rbd_strata(
            frame_working,
            frame.is_case,
            f"{col}_zscore",
            rbd_strata_col=RBD_TERTILE_COL,
            thresholds=ZSCORE_THRESHOLD_GRID,
        )

        # Save CSVs
        safe_col_name = col.replace("_", "-")
        csv_age = output_dir / f"lr_profile_{safe_col_name}_by_age.csv"
        csv_rbd = output_dir / f"lr_profile_{safe_col_name}_by_rbd.csv"

        df_by_age.to_csv(csv_age, index=False)
        df_by_rbd.to_csv(csv_rbd, index=False)

        # Figures (reuse plotting functions from integration runner)
        _plot_lr_by_age(df_by_age, label, FIGURES_DIR / f"lr_{safe_col_name}_by_age.png")
        _plot_lr_by_rbd(df_by_rbd, label, FIGURES_DIR / f"lr_{safe_col_name}_by_rbd.png")

    print(f"\n[OK] Saved figures to: {FIGURES_DIR}")


def _plot_lr_by_age(df_lr: pd.DataFrame, variable_label: str, output_path: Path) -> None:
    """Plot LR+ and LR- by age group."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    age_groups = sorted(df_lr["age_group"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(age_groups)))

    # LR+
    for age_group, color in zip(age_groups, colors):
        subset = df_lr[df_lr["age_group"] == age_group].sort_values("threshold")
        ax1.plot(
            subset["threshold"],
            subset["lr_pos"],
            marker="o",
            label=age_group,
            color=color,
            alpha=0.7,
        )
        ax1.fill_between(
            subset["threshold"],
            subset["lr_pos_lci"],
            subset["lr_pos_uci"],
            color=color,
            alpha=0.2,
        )

    ax1.axhline(y=1, color="red", linestyle="--", linewidth=1, label="LR=1")
    ax1.set_xlabel("Z-score Threshold", fontsize=11)
    ax1.set_ylabel("LR+", fontsize=11)
    ax1.set_title(f"{variable_label}\nPositive LR by Age Group", fontsize=12)
    ax1.legend(fontsize=9, loc="best")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # LR-
    for age_group, color in zip(age_groups, colors):
        subset = df_lr[df_lr["age_group"] == age_group].sort_values("threshold")
        ax2.plot(
            subset["threshold"],
            subset["lr_neg"],
            marker="s",
            label=age_group,
            color=color,
            alpha=0.7,
        )
        ax2.fill_between(
            subset["threshold"],
            subset["lr_neg_lci"],
            subset["lr_neg_uci"],
            color=color,
            alpha=0.2,
        )

    ax2.axhline(y=1, color="red", linestyle="--", linewidth=1, label="LR=1")
    ax2.set_xlabel("Z-score Threshold", fontsize=11)
    ax2.set_ylabel("LR-", fontsize=11)
    ax2.set_title(f"{variable_label}\nNegative LR by Age Group", fontsize=12)
    ax2.legend(fontsize=9, loc="best")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_lr_by_rbd(df_lr: pd.DataFrame, variable_label: str, output_path: Path) -> None:
    """Plot LR+ and LR- by RBD group."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    rbd_strata = ["Low", "Mid", "High"]
    colors_rbd = {"Low": RBD_RISK_COLORS["Low"], "Mid": RBD_RISK_COLORS["Mid"], "High": RBD_RISK_COLORS["High"]}

    # LR+
    for stratum in rbd_strata:
        subset = df_lr[df_lr["rbd_stratum"] == stratum].sort_values("threshold")
        if len(subset) == 0:
            continue
        ax1.plot(
            subset["threshold"],
            subset["lr_pos"],
            marker="o",
            label=f"RBD {stratum}",
            color=colors_rbd[stratum],
            linewidth=2,
            alpha=0.8,
        )
        ax1.fill_between(
            subset["threshold"],
            subset["lr_pos_lci"],
            subset["lr_pos_uci"],
            color=colors_rbd[stratum],
            alpha=0.2,
        )

    ax1.axhline(y=1, color="black", linestyle="--", linewidth=1, label="LR=1")
    ax1.set_xlabel("Z-score Threshold", fontsize=11)
    ax1.set_ylabel("LR+", fontsize=11)
    ax1.set_title(f"{variable_label}\nPositive LR by RBD Risk Group", fontsize=12)
    ax1.legend(fontsize=10, loc="best")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # LR-
    for stratum in rbd_strata:
        subset = df_lr[df_lr["rbd_stratum"] == stratum].sort_values("threshold")
        if len(subset) == 0:
            continue
        ax2.plot(
            subset["threshold"],
            subset["lr_neg"],
            marker="s",
            label=f"RBD {stratum}",
            color=colors_rbd[stratum],
            linewidth=2,
            alpha=0.8,
        )
        ax2.fill_between(
            subset["threshold"],
            subset["lr_neg_lci"],
            subset["lr_neg_uci"],
            color=colors_rbd[stratum],
            alpha=0.2,
        )

    ax2.axhline(y=1, color="black", linestyle="--", linewidth=1, label="LR=1")
    ax2.set_xlabel("Z-score Threshold", fontsize=11)
    ax2.set_ylabel("LR-", fontsize=11)
    ax2.set_title(f"{variable_label}\nNegative LR by RBD Risk Group", fontsize=12)
    ax2.legend(fontsize=10, loc="best")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Phase 3: Interaction Testing ───────────────────────────────────────────────


def run_phase3(
    frame: object,
    vars_dict: dict[str, str],
    output_dir: Path,
) -> pd.DataFrame:
    """Test RBD × variable interaction via LRT."""
    print("\n" + "-" * 70)
    print("PHASE 3: RBD × Marker Interaction (LRT)")
    print("-" * 70)

    results = []

    for col, label in vars_dict.items():
        print(f"  {label} ({col})...")

        result = compute_rbd_interaction_test(
            frame.df,
            frame.is_case,
            col,
            RBD_TERTILE_COL,
            INTERACTION_CONFOUNDERS,
            label,
            "arm_swing",
        )
        results.append(result.to_dict())

    df_results = pd.DataFrame(results)
    output_path = output_dir / "arm_swing_interaction.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\n[OK] Saved: {output_path.name}")

    return df_results


# ── Main runner ────────────────────────────────────────────────────────────────


def run_arm_swing_analysis(file_name: str = "ehr_diag_pd_rbd_only_all") -> None:
    """Arm swing secondary LR analysis: Phases 1–3."""
    print("\n" + "=" * 70)
    print("ARM SWING GAIT ANALYSIS: Secondary LR Analyses")
    print("=" * 70)

    # Load base frame
    print(f"\nLoading data from {file_name}...")
    frame_base = build_analysis_frame(file_name)

    # Create RBD tertiles if not present
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

    # Filter to arm swing cohort (interaction cohort for consistency across phases)
    print("\nFiltering to arm_swing_interaction cohort (complete-case)...")
    frame_arm, arm_stats = filter_cohort(frame_base, "arm_swing_interaction")
    print(
        f"N={arm_stats['n_total']:,}, cases={arm_stats['n_cases']}, "
        f"{arm_stats['pct_complete']:.1f}% complete"
    )

    if arm_stats["n_cases"] < 10:
        warnings.warn(
            f"Only {arm_stats['n_cases']} cases in arm swing cohort; "
            "power may be insufficient for interaction testing"
        )

    # Compute PCA composite
    print("\nComputing PCA composite (fit on controls only)...")
    arm_swing_cols = list(ARM_SWING_VARS.keys())
    pc1_series, pca_metadata = _compute_pca_composite(
        frame_arm.df, frame_arm.is_case, arm_swing_cols
    )
    frame_arm.df["arm_swing_pca_pc1"] = pc1_series

    # Build variables dict with PCA
    vars_dict = {
        **ARM_SWING_VARS,
        "arm_swing_pca_pc1": "Arm Swing PCA Composite (PC1)",
    }

    print(f"PC1 variance explained: {pca_metadata['variance_explained']:.1%}")
    print("PC1 loadings:", {k: f"{v:.3f}" for k, v in pca_metadata['loadings'].items()})

    # Run phases
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_phase1 = run_phase1(frame_arm, vars_dict, OUTPUT_DIR)
    run_phase2(frame_arm, vars_dict, OUTPUT_DIR)
    df_phase3 = run_phase3(frame_arm, vars_dict, OUTPUT_DIR)

    # Save PCA metadata
    pca_path = OUTPUT_DIR / "pca_metadata.json"
    with open(pca_path, "w") as f:
        json.dump(pca_metadata, f, indent=2)
    print(f"[OK] Saved: {pca_path.name}")

    # Summary
    print("\n" + "=" * 70)
    n_sig = (df_phase3["lrt_p"] < 0.05).sum()
    print(f"Significant interactions (LRT p < 0.05): {n_sig} / {len(df_phase3)}")
    print(f"Results directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    run_arm_swing_analysis()
