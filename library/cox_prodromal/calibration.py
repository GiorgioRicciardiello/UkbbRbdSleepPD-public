"""
Calibration assessment for Cox models.

Provides calibration slope, calibration-in-the-large, and decile
calibration plots. Essential when claiming screening relevance.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def calibration_slope(
    predicted_risk: np.ndarray,
    observed_event: np.ndarray,
    time: np.ndarray,
    time_horizon: float,
) -> Dict[str, float]:
    """
    Calibration slope: logistic regression of observed binary outcome
    (event within time_horizon) on log(predicted risk).

    Ideal slope = 1.0.
    slope > 1 -> underfitting; slope < 1 -> overfitting.

    Parameters
    ----------
    predicted_risk : np.ndarray
        Predicted partial hazard or risk score.
    observed_event : np.ndarray
        Binary event indicator (0/1).
    time : np.ndarray
        Follow-up time (years).
    time_horizon : float
        Time horizon for binary outcome definition.

    Returns
    -------
    dict
        Keys: slope, intercept, slope_se, slope_p.
    """
    try:
        from statsmodels.api import Logit, add_constant
    except ImportError:
        return {
            "slope": np.nan, "intercept": np.nan,
            "slope_se": np.nan, "slope_p": np.nan,
        }

    # Binary outcome: event occurred within time_horizon
    observed = ((observed_event == 1) & (time <= time_horizon)).astype(int)

    # Filter out subjects with zero predicted risk to avoid log(0)
    mask = predicted_risk > 0
    if mask.sum() < 20:
        return {
            "slope": np.nan, "intercept": np.nan,
            "slope_se": np.nan, "slope_p": np.nan,
        }

    log_risk = np.log(predicted_risk[mask])
    y = observed[mask]

    X = add_constant(log_risk)
    try:
        model = Logit(y, X).fit(disp=0, maxiter=100)
        return {
            "slope": float(model.params[1]),
            "intercept": float(model.params[0]),
            "slope_se": float(model.bse[1]),
            "slope_p": float(model.pvalues[1]),
        }
    except Exception:
        return {
            "slope": np.nan, "intercept": np.nan,
            "slope_se": np.nan, "slope_p": np.nan,
        }


def calibration_in_the_large(
    predicted_risk: np.ndarray,
    observed_event: np.ndarray,
    time: np.ndarray,
    time_horizon: float,
) -> Dict[str, float]:
    """
    Calibration-in-the-large: mean predicted risk vs observed event rate.

    Parameters
    ----------
    predicted_risk : np.ndarray
        Predicted partial hazard or risk score.
    observed_event : np.ndarray
        Binary event indicator.
    time : np.ndarray
        Follow-up time (years).
    time_horizon : float
        Time horizon for binary outcome.

    Returns
    -------
    dict
        Keys: mean_predicted, observed_rate, ratio.
    """
    observed = ((observed_event == 1) & (time <= time_horizon)).astype(float)
    mean_pred = float(np.mean(predicted_risk))
    obs_rate = float(np.mean(observed))
    ratio = mean_pred / obs_rate if obs_rate > 0 else np.nan

    return {
        "mean_predicted": mean_pred,
        "observed_rate": obs_rate,
        "ratio": ratio,
    }


def plot_calibration_deciles(
    ax: Axes,
    predicted_risk: np.ndarray,
    observed_event: np.ndarray,
    time: np.ndarray,
    time_horizon: float,
    n_groups: int = 10,
    title: str = "Calibration Plot",
) -> Dict[str, Any]:
    """
    Observed vs predicted risk by decile of predicted risk.

    Plots the calibration curve with ideal (45-degree) line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    predicted_risk : np.ndarray
        Predicted partial hazard.
    observed_event : np.ndarray
        Binary event indicator.
    time : np.ndarray
        Follow-up time (years).
    time_horizon : float
        Time horizon for binary outcome.
    n_groups : int
        Number of risk decile groups (default 10).
    title : str
        Plot title.

    Returns
    -------
    dict
        Decile-level data: mean_predicted, observed_rate per decile.
    """
    observed = ((observed_event == 1) & (time <= time_horizon)).astype(float)

    # Create decile groups
    try:
        decile = pd.qcut(predicted_risk, n_groups, labels=False, duplicates="drop")
    except ValueError:
        decile = pd.cut(predicted_risk, n_groups, labels=False)

    decile_df = pd.DataFrame({
        "predicted": predicted_risk,
        "observed": observed,
        "decile": decile,
    })
    summary = decile_df.groupby("decile").agg(
        mean_predicted=("predicted", "mean"),
        observed_rate=("observed", "mean"),
        n=("observed", "count"),
    ).reset_index()

    # Plot
    ax.scatter(
        summary["mean_predicted"], summary["observed_rate"],
        s=50, zorder=5, color="#2166AC",
    )
    lims = [0, max(
        summary["mean_predicted"].max(),
        summary["observed_rate"].max()
    ) * 1.1]
    ax.plot(lims, lims, "--", color="gray", alpha=0.7, label="Ideal")
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed event rate")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    return {
        "decile_data": summary.to_dict(orient="records"),
    }
