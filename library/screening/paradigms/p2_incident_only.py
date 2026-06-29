"""
Paradigm 2: Incident-only training (strict prospective learning).

Strategy
--------
Only incident PD cases are used as positives; prevalent cases are excluded
from the training set entirely.  Controls are matched 1:N to incident cases.
All samples receive uniform weight = 1.0.

Rationale
---------
Full alignment between training objective and evaluation target: the model
learns only from subjects who had not yet developed PD at the time of
actigraphy, so the learned features reflect prodromal physiology rather than
established disease.  This is the methodologically cleanest paradigm for
prospective risk prediction.

Limitation
----------
Severely reduced positive count relative to paradigm 1 (no prevalent signal),
which may degrade gradient quality in tree boosting and produce unstable
decision boundaries.  Use as a reference upper bound for prospective validity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from library.screening.matching import match_controls, split_cases_controls
from library.screening.paradigms.base import BaseParadigm


class IncidentOnlyParadigm(BaseParadigm):
    """Paradigm 2: incident PD only as cases, uniform weights."""

    @property
    def name(self) -> str:
        return "p2_incident_only"

    @property
    def description(self) -> str:
        return (
            "Incident PD only as cases, prevalent cases excluded, "
            "controls matched 1:N per incident case count."
        )

    def prepare_training_data(
        self,
        df_fold_train: pd.DataFrame,
        controls_per_case: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """
        Use only incident cases; exclude prevalent from training entirely.

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
        case_mask = df_fold_train["y_incident"]   # prevalent explicitly excluded
        df_cases, df_controls = split_cases_controls(df_fold_train, case_mask)

        df_train = match_controls(df_cases, df_controls, controls_per_case, rng)

        df_train = df_train.copy()
        df_train["y_label"] = df_train["y_incident"].astype(int)
        df_train["sample_weight"] = 1.0

        return df_train
