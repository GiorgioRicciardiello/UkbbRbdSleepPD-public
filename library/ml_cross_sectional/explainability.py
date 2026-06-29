"""
explainability.py
=================

Feature explainability for ml_cross_sectional models.

* SHAP values when the ``shap`` package is available (TreeExplainer for
  trees, LinearExplainer for linear models, KernelExplainer for SVM with
  subsampling). The pipeline gracefully degrades if SHAP is not installed.
* Permutation importance via scikit-learn (always available).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .models.base import ModelBase

# SHAP is optional — fail soft.
try:  # pragma: no cover - import side effect
    import shap  # type: ignore
    _HAS_SHAP = True
except ImportError:  # pragma: no cover
    shap = None  # type: ignore
    _HAS_SHAP = False


@dataclass
class ShapResult:
    """SHAP summary for one model fit.

    ``shap_values`` is the raw per-sample × per-feature array (same row
    order as ``X_eval``). ``X_eval`` is the raw feature matrix the SHAP
    explainer was called on — both are needed for bootstrap CIs and
    interaction plots downstream.
    """

    feature_importance: pd.DataFrame  # cols: feature, mean_abs_shap
    available: bool
    note: str = ""
    shap_values: np.ndarray | None = None    # shape (n, p)
    X_eval: pd.DataFrame | None = None       # shape (n, p)


@dataclass
class PermutationResult:
    """Permutation importance summary."""

    importance: pd.DataFrame  # cols: feature, importance_mean, importance_std


# --- SHAP --------------------------------------------------------------------

def compute_shap(
    model: ModelBase,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    max_kernel_samples: int = 500,
) -> ShapResult:
    """
    Return mean absolute SHAP values per feature.

    Parameters
    ----------
    model :
        A fitted ``ModelBase`` instance.
    X_train :
        Training-fold features (used as background distribution).
    X_test :
        Held-out features for which SHAP values are computed.
    max_kernel_samples :
        Subsample size for KernelExplainer (SVM); SHAP kernel scales O(n²).

    Notes
    -----
    Returns an empty result with ``available=False`` if ``shap`` is not
    installed or if no compatible explainer is found.
    """
    feat_names = list(X_train.columns)
    empty = pd.DataFrame({"feature": feat_names, "mean_abs_shap": np.nan})

    if not _HAS_SHAP:
        return ShapResult(feature_importance=empty, available=False, note="shap not installed")

    est = getattr(model, "estimator_", None)
    if est is None:
        return ShapResult(feature_importance=empty, available=False, note="model not fitted")

    try:
        X_eval_df: pd.DataFrame  # what we return for downstream plotting
        # Tree-based: XGBoost, RandomForest.
        if model.name in ("xgboost", "random_forest"):
            explainer = shap.TreeExplainer(est)
            X_eval = X_test.values
            sv = explainer.shap_values(X_eval)
            # Normalise return shape: newer shap returns either (n, p),
            # a list [class0, class1], or a 3-D array (n, p, 2) for binary
            # RandomForestClassifier. Always reduce to positive class (n, p).
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.asarray(sv)
            if sv.ndim == 3 and sv.shape[-1] == 2:
                sv = sv[:, :, 1]
            X_eval_df = X_test.copy()
        # Linear: LogisticModel, ElasticNetModel (use the estimator's scaled view).
        elif model.name in ("logistic", "elasticnet"):
            scaler = getattr(model, "scaler_", None)
            X_bg = scaler.transform(X_train.values) if scaler is not None else X_train.values
            X_eval = scaler.transform(X_test.values) if scaler is not None else X_test.values
            explainer = shap.LinearExplainer(est, X_bg)
            sv = explainer.shap_values(X_eval)
            # Return the unscaled X_test for plotting (feature axes in raw units).
            X_eval_df = X_test.copy()
        # SVM: kernel explainer with subsampling.
        elif model.name == "svm_rbf":
            scaler = getattr(model, "scaler_", None)
            X_bg = scaler.transform(X_train.values) if scaler is not None else X_train.values
            X_eval = scaler.transform(X_test.values) if scaler is not None else X_test.values
            n_bg = min(100, X_bg.shape[0])
            n_eval = min(max_kernel_samples, X_eval.shape[0])
            rng = np.random.default_rng(42)
            bg_idx = rng.choice(X_bg.shape[0], size=n_bg, replace=False)
            ev_idx = rng.choice(X_eval.shape[0], size=n_eval, replace=False)
            f = lambda x: est.predict_proba(x)[:, 1]
            explainer = shap.KernelExplainer(f, X_bg[bg_idx])
            sv = explainer.shap_values(X_eval[ev_idx], nsamples=100)
            X_eval_df = X_test.iloc[ev_idx].copy().reset_index(drop=True)
        else:
            return ShapResult(feature_importance=empty, available=False,
                              note=f"no SHAP path for model {model.name}")

        sv = np.asarray(sv)
        mean_abs = np.abs(sv).mean(axis=0)
        out = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
        out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        return ShapResult(
            feature_importance=out,
            available=True,
            shap_values=sv,
            X_eval=X_eval_df,
        )
    except Exception as e:  # pragma: no cover - SHAP can be brittle
        return ShapResult(feature_importance=empty, available=False, note=f"shap error: {e}")


# --- Permutation importance --------------------------------------------------

def compute_permutation_importance(
    model: ModelBase,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> PermutationResult:
    """
    Compute permutation importance using ``average_precision`` as the score.

    Wraps the model in a thin shim so sklearn's helper can call ``score``
    on a probability output.
    """
    class _Shim:
        def __init__(self, m: ModelBase) -> None:
            self.m = m
        def fit(self, *a: Any, **k: Any) -> "_Shim":
            return self
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return (self.m.predict_proba(pd.DataFrame(X, columns=X_test.columns)) >= 0.5).astype(int)
        def score(self, X: pd.DataFrame, y: pd.Series) -> float:
            from sklearn.metrics import average_precision_score
            p = self.m.predict_proba(pd.DataFrame(X, columns=X_test.columns))
            return float(average_precision_score(y, p))

    shim = _Shim(model)
    r = permutation_importance(
        shim,
        X_test.values,
        y_test.values,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    out = pd.DataFrame({
        "feature": list(X_test.columns),
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return PermutationResult(importance=out)
