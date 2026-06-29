"""
Paradigm 1: Combined training (prevalent + incident as cases).

Strategy
--------
Both prevalent and incident PD cases are treated as positives (y_label=1).
Controls are randomly sampled at ``controls_per_case`` per combined case count.
All samples receive uniform weight = 1.0.

Rationale
---------
Maximises the number of positive training examples under severe class
imbalance (~0.7% incidence).  Prevalent cases provide stronger, fully
expressed disease signal that anchors the feature space even though they
are pharmacologically confounded at actigraphy time.
Evaluation is restricted to incident cases in the test fold.

Limitation
----------
The model may learn late-stage disease features (motor symptoms, medication
effects) from prevalent cases that do not generalise to prodromal prediction.
Compare against Paradigm 3 (weighted) to quantify this risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from library.screening.matching import match_controls, split_cases_controls
from library.screening.paradigms.base import BaseParadigm


class CombinedTrainingParadigm(BaseParadigm):
    """Paradigm 1: prevalent + incident = cases, uniform weights."""

    @property
    def name(self) -> str:
        return "p1_combined"

    @property
    def description(self) -> str:
        return (
            "Prevalent + incident PD as cases, uniform weights, "
            "controls matched 1:N per combined case count."
        )

    def prepare_training_data(
        self,
        df_fold_train: pd.DataFrame,
        controls_per_case: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """
        Combine incident and prevalent cases, sample controls, assign uniform weights.

        Parameters
        ----------
        df_fold_train : pd.DataFrame
        controls_per_case : int
        rng : np.random.Generator

        Returns
        -------
        pd.DataFrame
            With ``y_label`` (0/1) and ``sample_weight`` (all 1.0) columns.
        """
        case_mask = df_fold_train["y_incident"] | df_fold_train["y_prevalent"]
        df_cases, df_controls = split_cases_controls(df_fold_train, case_mask)

        df_train = match_controls(df_cases, df_controls, controls_per_case, rng)

        # Assign labels: cases=1, controls=0
        df_train = df_train.copy()
        df_train["y_label"] = (
            df_train["y_incident"] | df_train["y_prevalent"]
        ).astype(int)
        df_train["sample_weight"] = 1.0

        return df_train
