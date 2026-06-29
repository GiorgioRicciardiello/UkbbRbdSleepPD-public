"""
models/base.py
==============

Abstract model interface for the ml_cross_sectional pipeline.

Every concrete model must:

* Set ``name``, ``supports_sample_weight``, ``supports_optuna`` class attrs.
* Accept a hyperparameter dict via ``set_params``.
* Implement ``fit``, ``predict_proba``, ``get_optuna_params``.

The fit signature deliberately accepts a single ``sample_weight`` array so
the trainer can pass class-imbalance weights uniformly across all models.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import optuna
import pandas as pd


class ModelBase(ABC):
    """Common interface every concrete model must implement."""

    #: Display name (used as a key in result files).
    name: str = "base"

    #: Whether the underlying estimator accepts ``sample_weight`` in ``fit``.
    supports_sample_weight: bool = True

    #: Whether the model exposes tunable hyperparameters via Optuna.
    supports_optuna: bool = True

    def __init__(self, params: dict[str, Any] | None = None, random_state: int = 42):
        """
        Parameters
        ----------
        params :
            Hyperparameter dict. Each subclass merges this with its defaults.
        random_state :
            Master random seed (passed through to the underlying estimator).
        """
        self.random_state = random_state
        self.params: dict[str, Any] = dict(params) if params else {}
        self.estimator_ = None  # populated in fit()

    # ------------------------------------------------------------------
    # Abstract surface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> "ModelBase":
        """Fit the underlying estimator. Returns ``self``."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return the positive-class probabilities, shape ``(n,)``."""

    @abstractmethod
    def get_optuna_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """
        Sample a hyperparameter dict from an Optuna trial.

        The dict is consumed by ``self.__class__(params=...)`` to construct
        the candidate model for one inner-CV iteration.
        """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def feature_names_(self) -> list[str]:
        """Return the column names seen at fit time, if available."""
        if hasattr(self.estimator_, "feature_names_in_"):
            return list(self.estimator_.feature_names_in_)
        return []
