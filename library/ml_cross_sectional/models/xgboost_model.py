"""
XGBoostModel — gradient-boosted trees with scale_pos_weight.

For extreme imbalance (~0.7% prevalence) we let XGBoost handle reweighting
internally via ``scale_pos_weight = n_neg / n_pos`` rather than passing
``sample_weight``. ``eval_metric='aucpr'`` matches our Optuna objective
(average precision).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBClassifier

from .base import ModelBase


class XGBoostModel(ModelBase):
    """XGBoost binary classifier."""

    name = "xgboost"
    supports_sample_weight = True
    supports_optuna = True

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> "XGBoostModel":
        """
        Fit XGBoost. ``scale_pos_weight`` is computed from the training fold
        and overrides any value passed via ``self.params``.
        """
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        spw = (n_neg / n_pos) if n_pos > 0 else 1.0

        defaults: dict[str, Any] = dict(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1.0,
            reg_lambda=1.0,
            reg_alpha=0.0,
            scale_pos_weight=spw,
            random_state=self.random_state,
            n_jobs=1,
            verbosity=0,
        )
        defaults.update(self.params)
        # Always recompute scale_pos_weight from training fold (no leakage).
        defaults["scale_pos_weight"] = spw
        self.estimator_ = XGBClassifier(**defaults)
        # We do NOT pass sample_weight: scale_pos_weight already handles imbalance.
        self.estimator_.fit(X_train.values, y_train.values)
        self.feature_names_cached_ = list(X_train.columns)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Positive-class probabilities."""
        return self.estimator_.predict_proba(X.values)[:, 1]

    def get_optuna_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Tune depth, learning rate, regularisation, and subsampling."""
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        }
