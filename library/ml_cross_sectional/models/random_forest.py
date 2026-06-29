"""
RandomForestModel — random forest with class_weight='balanced_subsample'.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .base import ModelBase


class RandomForestModel(ModelBase):
    """Random forest binary classifier."""

    name = "random_forest"
    supports_sample_weight = True
    supports_optuna = True

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> "RandomForestModel":
        """Fit RF with bootstrap subsample reweighting per tree."""
        defaults: dict[str, Any] = dict(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=self.random_state,
        )
        defaults.update(self.params)
        self.estimator_ = RandomForestClassifier(**defaults)
        self.estimator_.fit(X_train.values, y_train.values, sample_weight=sample_weight)
        self.feature_names_cached_ = list(X_train.columns)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Positive-class probabilities."""
        return self.estimator_.predict_proba(X.values)[:, 1]

    def get_optuna_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Tune number of trees, depth, and split/leaf sizes."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
