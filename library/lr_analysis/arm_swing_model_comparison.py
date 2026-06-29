"""
Model comparison: RBD vs Arm Swing vs Combined for incident PD prediction.

Compares three logistic regression models with goodness-of-fit and
classification metrics (R², Adjusted R², AUC, Sensitivity, Specificity, etc.)
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score, brier_score_loss
)
import statsmodels.api as sm
from statsmodels.formula.api import logit

from config.config import config as project_config
from library.lr_analysis.config import (
    RESULTS_SUBDIR, INTERACTION_CONFOUNDERS,
    ARM_SWING_VARS, RBD_ZSCORE_COL
)
from library.lr_analysis.data_prep import build_analysis_frame, filter_cohort

_FIG_DPI = 300


def compute_mcfadden_r2(model_fitted) -> float:
    """Compute McFadden's pseudo R²."""
    ll_full = model_fitted.llf
    ll_null = model_fitted.llnull
    return 1 - (ll_full / ll_null)


def compute_adjusted_r2(model_fitted, n_obs: int) -> float:
    """Compute adjusted McFadden's R² (adjusted for number of parameters)."""
    r2 = compute_mcfadden_r2(model_fitted)
    k = model_fitted.df_model
    return 1 - ((1 - r2) * (n_obs - 1) / (n_obs - k - 1))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> dict[str, float]:
    """Compute classification metrics at specified threshold.

    Parameters
    ----------
    y_true : np.ndarray
        Binary outcomes (0/1)
    y_pred_proba : np.ndarray
        Predicted probabilities
    threshold : float
        Classification threshold (default 0.5)

    Returns
    -------
    dict
        Classification metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    n = len(y_true)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=np.nan)
    accuracy = (tp + tn) / n
    auc = roc_auc_score(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba)

    # Specificity and sensitivity at optimal threshold (Youden)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    youden_idx = np.argmax(tpr - fpr)
    youden_threshold = thresholds[youden_idx]

    return {
        "n": n,
        "n_cases": int(y_true.sum()),
        "n_controls": int((1 - y_true).sum()),
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "brier_score": brier,
        "youden_threshold": youden_threshold,
    }


def fit_model(
    df: pd.DataFrame,
    formula: str,
    model_name: str
) -> tuple[sm.BinomialResults, dict]:
    """Fit logistic regression model and compute metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Data with all required columns
    formula : str
        Patsy formula for logit model
    model_name : str
        Model name for reporting

    Returns
    -------
    tuple
        (fitted_model, metrics_dict)
    """
    print(f"\n  Fitting {model_name}...")

    # Fit model
    model = logit(formula, data=df)
    result = model.fit(disp=False)

    # Compute goodness-of-fit metrics
    n_obs = len(df)
    mcfadden_r2 = compute_mcfadden_r2(result)
    adj_r2 = compute_adjusted_r2(result, n_obs)

    # Classification metrics
    y_true = df["outcome"].values
    y_pred_proba = result.predict(df)
    clf_metrics = compute_classification_metrics(y_true, y_pred_proba)

    metrics = {
        "model_name": model_name,
        "n_obs": n_obs,
        "n_cases": clf_metrics["n_cases"],
        "n_controls": clf_metrics["n_controls"],
        "n_params": result.df_model + 1,
        "ll_full": result.llf,
        "ll_null": result.llnull,
        "aic": result.aic,
        "bic": result.bic,
        "mcfadden_r2": mcfadden_r2,
        "adjusted_r2": adj_r2,
        "accuracy": clf_metrics["accuracy"],
        "sensitivity": clf_metrics["sensitivity"],
        "specificity": clf_metrics["specificity"],
        "precision": clf_metrics["precision"],
        "recall": clf_metrics["recall"],
        "f1": clf_metrics["f1"],
        "auc": clf_metrics["auc"],
        "brier_score": clf_metrics["brier_score"],
        "youden_threshold": clf_metrics["youden_threshold"],
    }

    print(f"    McFadden R²: {mcfadden_r2:.4f}")
    print(f"    Adjusted R²: {adj_r2:.4f}")
    print(f"    AUC-ROC: {clf_metrics['auc']:.4f}")
    print(f"    Accuracy: {clf_metrics['accuracy']:.4f}")

    return result, metrics


def likelihood_ratio_test(
    model_reduced: sm.BinomialResults,
    model_full: sm.BinomialResults
) -> dict:
    """Perform likelihood ratio test comparing nested models.

    Parameters
    ----------
    model_reduced : sm.BinomialResults
        Reduced (simpler) model
    model_full : sm.BinomialResults
        Full (more complex) model

    Returns
    -------
    dict
        LRT results
    """
    ll_reduced = model_reduced.llf
    ll_full = model_full.llf
    lrt_stat = -2 * (ll_reduced - ll_full)

    # df = difference in number of parameters
    df = model_full.df_model - model_reduced.df_model

    # p-value from chi-square distribution
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(lrt_stat, df)

    return {
        "lrt_stat": lrt_stat,
        "df": df,
        "p_value": p_value,
    }


def run_model_comparison() -> None:
    """Execute complete model comparison analysis."""
    results_dir = project_config["results"]["root"] / RESULTS_SUBDIR
    output_dir = results_dir / "arm_swing"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Arm Swing Model Comparison: RBD vs Arm Swing vs Combined")
    print("=" * 70)

    # Load and prepare data
    print("\n[Data Preparation]")
    frame_base = build_analysis_frame()
    frame_arm, _ = filter_cohort(frame_base, "arm_swing")
    df = frame_arm.df.copy()

    # Add outcome column
    df["outcome"] = frame_arm.is_case.astype(int)

    # Create z-scored predictor for arm swing composite
    arm_swing_col = "arm_swing_pca_pc1"
    if arm_swing_col not in df.columns:
        print(f"  Computing PCA composite...")
        from library.lr_analysis.lr_arm_swing_runner import _compute_pca_composite
        pc1_series, pca_meta = _compute_pca_composite(
            df, ~frame_arm.is_case, list(ARM_SWING_VARS.keys())
        )
        df[arm_swing_col] = pc1_series

    # Z-score the arm swing composite (on controls only, no leakage)
    control_mask = ~frame_arm.is_case
    mean_as = df.loc[control_mask, arm_swing_col].mean()
    std_as = df.loc[control_mask, arm_swing_col].std()
    df[f"{arm_swing_col}_zscore"] = (df[arm_swing_col] - mean_as) / std_as

    # Z-score RBD (on controls only)
    mean_rbd = df.loc[control_mask, RBD_ZSCORE_COL].mean()
    std_rbd = df.loc[control_mask, RBD_ZSCORE_COL].std()
    df[f"{RBD_ZSCORE_COL}_zscore"] = (df[RBD_ZSCORE_COL] - mean_rbd) / std_rbd

    # Confounders formula
    confounders_str = " + ".join(INTERACTION_CONFOUNDERS)

    print(f"  N: {len(df)}")
    print(f"  N cases: {frame_arm.is_case.sum()}")
    print(f"  N controls: {(~frame_arm.is_case).sum()}")

    # Fit three models
    print("\n[Model Fitting]")

    model1, metrics1 = fit_model(
        df,
        f"outcome ~ {RBD_ZSCORE_COL}_zscore + {confounders_str}",
        "Model 1: RBD Only"
    )

    model2, metrics2 = fit_model(
        df,
        f"outcome ~ {arm_swing_col}_zscore + {confounders_str}",
        "Model 2: Arm Swing Only"
    )

    model3, metrics3 = fit_model(
        df,
        f"outcome ~ {RBD_ZSCORE_COL}_zscore + {arm_swing_col}_zscore + {confounders_str}",
        "Model 3: RBD + Arm Swing"
    )

    # Likelihood ratio tests
    print("\n[Likelihood Ratio Tests]")

    lrt_12 = likelihood_ratio_test(model1, model3)  # RBD vs Combined (arm swing improvement)
    print(f"\n  Model 1 (RBD) vs Model 3 (Combined):")
    print(f"    LRT chi2({lrt_12['df']}) = {lrt_12['lrt_stat']:.4f}, p = {lrt_12['p_value']:.2e}")

    lrt_23 = likelihood_ratio_test(model2, model3)  # Arm Swing vs Combined (RBD improvement)
    print(f"\n  Model 2 (Arm Swing) vs Model 3 (Combined):")
    print(f"    LRT chi2({lrt_23['df']}) = {lrt_23['lrt_stat']:.4f}, p = {lrt_23['p_value']:.2e}")

    # Build comparison table
    print("\n[Building Comparison Tables]")

    comparison_df = pd.DataFrame([metrics1, metrics2, metrics3])
    comparison_csv = output_dir / "model_comparison_metrics.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"  Comparison metrics: {comparison_csv}")

    # Classification metrics table
    clf_cols = [
        "model_name", "auc", "accuracy", "sensitivity", "specificity",
        "precision", "f1", "brier_score"
    ]
    clf_df = comparison_df[clf_cols].copy()
    clf_csv = output_dir / "model_comparison_classification.csv"
    clf_df.to_csv(clf_csv, index=False)
    print(f"  Classification metrics: {clf_csv}")

    # Goodness-of-fit table
    gof_cols = [
        "model_name", "n_obs", "n_params", "ll_full", "aic", "bic",
        "mcfadden_r2", "adjusted_r2"
    ]
    gof_df = comparison_df[gof_cols].copy()
    gof_csv = output_dir / "model_comparison_goodness_of_fit.csv"
    gof_df.to_csv(gof_csv, index=False)
    print(f"  Goodness-of-fit metrics: {gof_csv}")

    # LRT results
    lrt_df = pd.DataFrame([
        {
            "comparison": "Model 1 (RBD) vs Model 3 (Combined)",
            "lrt_stat": lrt_12["lrt_stat"],
            "df": lrt_12["df"],
            "p_value": lrt_12["p_value"],
            "interpretation": "Arm swing adds significant predictive value"
            if lrt_12["p_value"] < 0.05 else "No significant improvement"
        },
        {
            "comparison": "Model 2 (Arm Swing) vs Model 3 (Combined)",
            "lrt_stat": lrt_23["lrt_stat"],
            "df": lrt_23["df"],
            "p_value": lrt_23["p_value"],
            "interpretation": "RBD adds significant predictive value"
            if lrt_23["p_value"] < 0.05 else "No significant improvement"
        }
    ])
    lrt_csv = output_dir / "model_comparison_lrt.csv"
    lrt_df.to_csv(lrt_csv, index=False)
    print(f"  LRT results: {lrt_csv}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nGoodness-of-Fit Comparison:")
    print(gof_df.to_string(index=False))

    print("\n\nClassification Metrics Comparison:")
    print(clf_df.to_string(index=False))

    print("\n\nModel Improvement Tests:")
    print(lrt_df.to_string(index=False))

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_model_comparison()
