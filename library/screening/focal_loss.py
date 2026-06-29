"""
Focal loss implementation for XGBoost.

Focal loss addresses class imbalance by down-weighting easy examples and
focusing training on hard-to-classify examples:

    FL(p_t) = -α(1-p_t)^γ log(p_t)

where:
  - p_t = probability of true class (0–1)
  - α = balancing factor (typically 0.25)
  - γ = focusing parameter (typically 2.0); higher γ = sharper focus on hard examples

Reference: Lin et al. "Focal Loss for Dense Object Detection" (CVPR 2017)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def focal_loss_objective(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Focal loss objective for XGBoost.

    XGBoost custom objectives receive:
      - y_pred: model predictions (logits, before sigmoid)
      - y_true: binary labels (0/1)

    And must return:
      - gradients: ∂L/∂y_pred
      - hessians: ∂²L/∂y_pred²

    Parameters
    ----------
    y_pred : np.ndarray
        Raw model predictions (logits).
    y_true : np.ndarray
        True labels (0/1).
    alpha : float
        Balancing factor (default 0.25). Higher α down-weights positive class.
    gamma : float
        Focusing parameter (default 2.0). Higher γ focuses on hard examples.

    Returns
    -------
    gradients : np.ndarray
        First derivative of loss w.r.t. y_pred.
    hessians : np.ndarray
        Second derivative of loss w.r.t. y_pred.
    """
    # Convert logits to probabilities
    sigmoid = 1.0 / (1.0 + np.exp(-y_pred))
    p_t = np.where(y_true == 1, sigmoid, 1 - sigmoid)

    # Ensure numerical stability
    p_t = np.clip(p_t, 1e-7, 1 - 1e-7)

    # Focal weight: (1 - p_t)^γ
    focal_weight = (1 - p_t) ** gamma

    # Class balancing weight
    class_weight = np.where(y_true == 1, alpha, 1 - alpha)

    # Focal loss gradient: ∂FL/∂p = -α(1-p)^γ(γp*log(p) + (1-p))
    # In terms of logit y_pred:
    gradient = -class_weight * focal_weight * (
        gamma * p_t * np.log(np.clip(p_t, 1e-7, 1)) +
        (1 - p_t)
    ) * sigmoid * (1 - sigmoid)

    # Second derivative (hessian) — simplified approximation
    hessian = class_weight * focal_weight * sigmoid * (1 - sigmoid)

    return gradient, hessian


def focal_loss_metric(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tuple[str, float]:
    """
    Focal loss metric for evaluation in cross-validation.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted probabilities (0–1).
    y_true : np.ndarray
        True labels (0/1).
    alpha : float
        Balancing factor.
    gamma : float
        Focusing parameter.

    Returns
    -------
    metric_name : str
        Metric name.
    metric_value : float
        Focal loss value (lower is better).
    """
    # Clip predictions for numerical stability
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

    # Focal loss: FL = -α(1-p_t)^γ log(p_t)
    p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
    focal_weight = (1 - p_t) ** gamma
    class_weight = np.where(y_true == 1, alpha, 1 - alpha)

    focal_loss_val = -class_weight * focal_weight * np.log(p_t)
    mean_focal_loss = float(np.mean(focal_loss_val))

    return "focal_loss", mean_focal_loss


def compute_focal_weights(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> np.ndarray:
    """
    Compute sample weights using focal loss formula.

    Can be applied to existing sample_weight vector to focus training
    on hard-to-classify examples.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (0/1).
    y_pred_proba : np.ndarray
        Predicted probabilities (0–1).
    alpha : float
        Class balancing factor.
    gamma : float
        Focusing parameter.

    Returns
    -------
    focal_weights : np.ndarray
        Per-sample focal loss weights.
    """
    # Clip predictions for numerical stability
    y_pred_proba = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)

    # Focal weight: (1 - p_t)^γ
    p_t = np.where(y_true == 1, y_pred_proba, 1 - y_pred_proba)
    focal_weight = (1 - p_t) ** gamma

    # Class balancing weight
    class_weight = np.where(y_true == 1, alpha, 1 - alpha)

    # Focal loss weight = class_weight × focal_weight
    return class_weight * focal_weight
