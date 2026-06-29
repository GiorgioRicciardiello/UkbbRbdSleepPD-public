"""
Abstract base class for training paradigms.

Each paradigm implements a single method that receives the full training fold
(incident cases + prevalent cases + controls) and returns the subset of rows
to use for training, along with per-sample weights.

The outer CV loop, preprocessing, XGBoost tuning, and evaluation are
paradigm-agnostic and handled entirely in ``main.py``.

Contract
--------
``prepare_training_data`` returns a DataFrame that:
  - Contains only rows from the input ``df_fold_train``
  - Has a ``y_label`` column: 1 = case (as defined by the paradigm), 0 = control
  - Has a ``sample_weight`` column: per-sample float weight ≥ 0
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseParadigm(ABC):
    """Abstract interface for a training paradigm."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in result tables and file names."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-sentence description of the paradigm's training strategy."""

    @abstractmethod
    def prepare_training_data(
        self,
        df_fold_train: pd.DataFrame,
        controls_per_case: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """
        Select and weight training rows from the fold's training set.

        Parameters
        ----------
        df_fold_train : pd.DataFrame
            Full training fold.  Contains columns ``y_incident``,
            ``y_prevalent``, ``y_control``, and all feature columns.
        controls_per_case : int
            Target number of controls per case to sample (e.g. 10).
        rng : np.random.Generator
            Fold-specific seeded random generator for reproducibility.

        Returns
        -------
        pd.DataFrame
            Subset of ``df_fold_train`` with two added columns:
            - ``y_label``: int (1 = case, 0 = control)
            - ``sample_weight``: float (≥ 0)
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
