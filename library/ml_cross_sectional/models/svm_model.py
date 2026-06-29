"""
SVMModel — RBF support-vector classifier with class_weight='balanced'.

SVM is by far the slowest model in the suite (O(n^2) to O(n^3)). For the
full 580k cohort it would be intractable; in pipeline.py we either subsample
the training fold for SVM or skip it on large runs. The wrapper itself does
not subsample — that decision lives in the trainer.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .base import ModelBase


class SVMModel(ModelBase):
    """RBF SVM, calibrated for probability output."""

    name = "svm_rbf"
    supports_sample_weight = True
    supports_optuna = True

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> "SVMModel":
        """Standardise and fit an RBF SVM with probability output."""
        self.scaler_ = StandardScaler().fit(X_train.values)
        Xs = self.scaler_.transform(X_train.values)
        defaults: dict[str, Any] = dict(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            cache_size=500,
            random_state=self.random_state,
        )
        defaults.update(self.params)
        self.estimator_ = SVC(**defaults)
        self.estimator_.fit(Xs, y_train.values, sample_weight=sample_weight)
        self.feature_names_cached_ = list(X_train.columns)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Positive-class probabilities (Platt scaling)."""
        Xs = self.scaler_.transform(X.values)
        return self.estimator_.predict_proba(Xs)[:, 1]

    def get_optuna_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Tune ``C`` and ``gamma`` (both log-spaced)."""
        return {
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        }
