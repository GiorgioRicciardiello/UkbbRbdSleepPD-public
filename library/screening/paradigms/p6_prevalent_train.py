"""
Paradigm 6: Prevalent-train / incident-validate strategy.


Deprecate:
  Outcome column [incident] = 'outcome_1a_pd_only__incident'  value_counts: {False: 87667, True: 448}
  Outcome column [prevalent] = 'outcome_1a_pd_only__prevalent'  value_counts: {False: 87975, True: 140}
  Outcome column [control] = 'control'  value_counts: {True: 86973, False: 1142}
  Slimmed DataFrame: 87561 rows × 22 columns (features=18, labels=4, original width dropped)

  few prevalent cases for proper training


Strategy
--------
The model is trained exclusively on prevalent PD cases vs controls.
Evaluation is performed on incident cases (test fold), same as all other
paradigms, to assess transfer from cross-sectional disease detection to
prospective risk prediction.

Rationale
---------
Prevalent cases provide high signal-to-noise for disease feature learning,
which is valuable under severe incidence imbalance.  The hypothesis is that
some feature patterns learned from established PD generalise to prodromal risk.
This is most likely to hold for HES-derived prodromal markers (constipation,
anosmia, dream enactment) which represent genuine early-stage features and are
ascertained from the same data source regardless of disease stage.

Critical caveat
---------------
Prevalent PD at actigraphy time → active disease with motor symptoms +
dopaminergic medication → actigraphy features (movement metrics) are
confounded.  If actigraphy features dominate, the model learns motor
signatures that do NOT transfer to prodromal prediction, and performance on
the incident test fold will be lower than Paradigm 1.  Use this paradigm as
an exploratory/auxiliary comparison only, not as the primary model.

For the inner CV, the training and inner-validation folds also use prevalent
cases as positives, keeping the inner objective consistent with the outer
training objective.


"""
from __future__ import annotations

import numpy as np
import pandas as pd

from library.screening.matching import match_controls, split_cases_controls
from library.screening.paradigms.base import BaseParadigm


class PrevalentTrainParadigm(BaseParadigm):
    """Paradigm 6: train on prevalent cases, evaluate on incident."""

    @property
    def name(self) -> str:
        return "p6_prevalent_train"

    @property
    def description(self) -> str:
        return (
            "Training on prevalent PD cases only; evaluation on incident test fold. "
            "Tests cross-sectional → prodromal feature transfer."
        )

    def prepare_training_data(
        self,
        df_fold_train: pd.DataFrame,
        controls_per_case: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """
        Use only prevalent cases as positives in training.

        Parameters
        ----------
        df_fold_train : pd.DataFrame
        controls_per_case : int
        rng : np.random.Generator

        Returns
        -------
        pd.DataFrame
            With ``y_label`` (0/1) and ``sample_weight`` (all 1.0) columns.
            Incident cases from the training fold are excluded entirely.
        """
        case_mask = df_fold_train["y_prevalent"]   # incident excluded
        df_cases, df_controls = split_cases_controls(df_fold_train, case_mask)

        if len(df_cases) == 0:
            raise RuntimeError(
                "No prevalent cases in this training fold. "
                "Cannot run Paradigm 6 — check cohort composition."
            )

        df_train = match_controls(df_cases, df_controls, controls_per_case, rng)
        df_train = df_train.copy()

        df_train["y_label"] = df_train["y_prevalent"].astype(int)
        df_train["sample_weight"] = 1.0

        return df_train
