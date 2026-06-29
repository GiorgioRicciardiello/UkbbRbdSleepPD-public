"""
LogisticModel — plain L2 logistic regression with class_weight='balanced'.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import ModelBase


class LogisticModel(ModelBase):
    """L2 logistic regression with standardised features."""

    name = "logistic"
    supports_sample_weight = True
    supports_optuna = True

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> "LogisticModel":
        """
        Fit a standardised logistic regression. ``random_state=42`` for the
        solver, fixed in ``__init__``.
        """
        self.scaler_ = StandardScaler().fit(X_train.values)
        Xs = self.scaler_.transform(X_train.values)
        defaults: dict[str, Any] = dict(
            solver="lbfgs",
            class_weight="balanced",
            C=1.0,
            max_iter=2000,
            random_state=self.random_state,
        )
        defaults.update(self.params)
        self.estimator_ = LogisticRegression(**defaults)
        self.estimator_.fit(Xs, y_train.values, sample_weight=sample_weight)
        self.feature_names_cached_ = list(X_train.columns)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Positive-class probabilities."""
        Xs = self.scaler_.transform(X.values)
        return self.estimator_.predict_proba(Xs)[:, 1]

    def get_optuna_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Tune the inverse regularisation strength ``C``."""
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
