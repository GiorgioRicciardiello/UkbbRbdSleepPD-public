"""
Paradigm 3: Weighted training (rebalancing prevalent vs incident).

Strategy
--------
Both prevalent and incident PD cases are used as positives, but prevalent
cases receive a down-weighted sample weight ``alpha`` < 1.0 while incident
cases receive weight 1.0.  Controls also receive weight 1.0.

The ``alpha`` parameter controls how much the training objective is pulled
toward incident-aligned patterns.  A sweep over alpha values (e.g. 0.1 to
0.5) can be performed by instantiating multiple copies of this paradigm.

Weight assignment:
  - incident case   → 1.0
  - prevalent case  → alpha  (default: config.DEFAULT_PREVALENT_WEIGHT = 0.3)
  - control         → 1.0

Rationale
---------
Prevalent PD cases are under dopaminergic medication at actigraphy time,
confounding actigraphy-derived features.  Down-weighting them limits their
influence on motor-related features while preserving their contribution to
HES-derived prodromal markers (constipation, anosmia, etc.), which are
largely unaffected by medication.  This is the recommended primary paradigm.

Note on sample_weight + XGBoost:
  XGBoost's ``scale_pos_weight`` is NOT set here because ``sample_weight``
  already encodes relative importance.  Setting both would double-count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from library.screening.config import DEFAULT_PREVALENT_WEIGHT
from library.screening.matching import match_controls, split_cases_controls
from library.screening.paradigms.base import BaseParadigm


class WeightedTrainingParadigm(BaseParadigm):
    """Paradigm 3: prevalent + incident, prevalent down-weighted by alpha."""

    def __init__(self, alpha: float = DEFAULT_PREVALENT_WEIGHT) -> None:
        """
        Parameters
        ----------
        alpha : float
            Sample weight for prevalent cases (0 < alpha ≤ 1.0).
            Incident cases always receive weight 1.0.
        """
        if not (0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1]; got {alpha}")
        self.alpha = alpha

    @property
    def name(self) -> str:
        return f"p3_weighted_a{self.alpha:.2f}".replace(".", "")

    @property
    def description(self) -> str:
        return (
            f"Prevalent + incident as cases; prevalent weight={self.alpha:.2f}, "
            f"incident weight=1.0; controls matched 1:N per combined case count."
        )

    def prepare_training_data(
        self,
        df_fold_train: pd.DataFrame,
        controls_per_case: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """
        Combine cases, sample controls, assign differential weights.

        Parameters
        ----------
        df_fold_train : pd.DataFrame
        controls_per_case : int
        rng : np.random.Generator

        Returns
        -------
        pd.DataFrame
            With ``y_label`` (0/1) and ``sample_weight`` columns.
        """
        case_mask = df_fold_train["y_incident"] | df_fold_train["y_prevalent"]
        df_cases, df_controls = split_cases_controls(df_fold_train, case_mask)

        df_train = match_controls(df_cases, df_controls, controls_per_case, rng)
        df_train = df_train.copy()

        # Label: 1 if any PD case, 0 if control
        df_train["y_label"] = (
            df_train["y_incident"] | df_train["y_prevalent"]
        ).astype(int)

        # Differential weights: incident=1.0, prevalent=alpha, control=1.0
        df_train["sample_weight"] = np.where(
            df_train["y_prevalent"],
            self.alpha,
            1.0,
        )

        return df_train
