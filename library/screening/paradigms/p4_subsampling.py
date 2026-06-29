"""
Paradigm 4: Subsampling (variable case-to-control ratio).

Strategy
--------
Both prevalent and incident cases are used as positives (same as Paradigm 1)
but the control sampling ratio is decoupled from the fixed 1:10 default.
This paradigm tests whether a tighter ratio (e.g. 1:5) improves gradient
quality by reducing the numerical dominance of the negative class, relative
to the looser 1:10 matching used elsewhere.

Note: when ``ratio`` equals ``CONTROLS_PER_CASE`` from config, this paradigm
is equivalent to Paradigm 1 with the same seed and should produce identical
results.  The expected use case is a sweep over ratio in {3, 5, 10, 20}.

All samples receive uniform weight = 1.0 (contrast with Paradigm 3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from library.screening.matching import match_controls, split_cases_controls
from library.screening.paradigms.base import BaseParadigm


class SubsamplingParadigm(BaseParadigm):
    """Paradigm 4: combined cases, variable control ratio, uniform weights."""

    def __init__(self, ratio: int = 5) -> None:
        """
        Parameters
        ----------
        ratio : int
            Controls sampled per combined case in this paradigm.
            Overrides the global ``controls_per_case`` passed to
            ``prepare_training_data`` so the paradigm is self-contained.
        """
        if ratio < 1:
            raise ValueError(f"ratio must be ≥ 1; got {ratio}")
        self.ratio = ratio

    @property
    def name(self) -> str:
        return f"p4_subsample_r{self.ratio}"

    @property
    def description(self) -> str:
        return (
            f"Prevalent + incident as cases, uniform weights, "
            f"controls sampled at 1:{self.ratio} (overrides global ratio)."
        )

    def prepare_training_data(
        self,
        df_fold_train: pd.DataFrame,
        controls_per_case: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """
        Combine cases, sample controls at ``self.ratio``, uniform weights.

        The ``controls_per_case`` argument from the outer loop is intentionally
        overridden by ``self.ratio`` so the paradigm controls its own ratio.

        Parameters
        ----------
        df_fold_train : pd.DataFrame
        controls_per_case : int
            Ignored; paradigm uses ``self.ratio`` instead.
        rng : np.random.Generator

        Returns
        -------
        pd.DataFrame
            With ``y_label`` (0/1) and ``sample_weight`` (all 1.0) columns.
        """
        case_mask = df_fold_train["y_incident"] | df_fold_train["y_prevalent"]
        df_cases, df_controls = split_cases_controls(df_fold_train, case_mask)

        df_train = match_controls(df_cases, df_controls, self.ratio, rng)
        df_train = df_train.copy()

        df_train["y_label"] = (
            df_train["y_incident"] | df_train["y_prevalent"]
        ).astype(int)
        df_train["sample_weight"] = 1.0

        return df_train
