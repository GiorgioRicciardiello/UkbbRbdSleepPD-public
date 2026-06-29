"""
Case-control matching within CV folds.

Implements random 1:N control matching without replacement.
Matching is performed independently per fold to avoid information leakage
between training and test sets.

Assumption: no propensity-score matching is applied — controls are sampled
uniformly from the available pool.  Age/sex matching can be layered in as a
future extension if distributional imbalance is observed.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def match_controls(
    df_cases: pd.DataFrame,
    df_controls: pd.DataFrame,
    controls_per_case: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Randomly sample ``controls_per_case`` controls per case without replacement.

    If fewer controls are available than requested, all controls are returned
    and a warning is emitted.

    Parameters
    ----------
    df_cases : pd.DataFrame
        Case rows (incident and/or prevalent depending on paradigm).
    df_controls : pd.DataFrame
        Full control pool from the training fold.
    controls_per_case : int
        Target number of controls per case (e.g. 10).
    rng : np.random.Generator
        Seeded random generator; pass a fold-specific seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Concatenation of all cases + sampled controls.
        Row order: cases first, then controls.
    """
    n_cases = len(df_cases)
    n_controls_needed = n_cases * controls_per_case
    n_controls_available = len(df_controls)

    if n_controls_available == 0:
        raise ValueError("No controls available in this fold for matching.")

    if n_controls_needed > n_controls_available:
        logger.warning(
            "Requested %d controls (%d cases × %d) but only %d available. "
            "Using all controls (ratio: 1:%.1f).",
            n_controls_needed, n_cases, controls_per_case,
            n_controls_available,
            n_controls_available / max(n_cases, 1),
        )
        df_sampled_controls = df_controls
    else:
        sample_idx = rng.choice(
            n_controls_available,
            size=n_controls_needed,
            replace=False,
        )
        df_sampled_controls = df_controls.iloc[sample_idx]

    df_matched = pd.concat([df_cases, df_sampled_controls], ignore_index=True)
    logger.debug(
        "Matched dataset: %d cases + %d controls = %d total.",
        n_cases, len(df_sampled_controls), len(df_matched),
    )
    return df_matched


def split_cases_controls(
    df_fold: pd.DataFrame,
    case_mask: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a fold DataFrame into cases and controls.

    Parameters
    ----------
    df_fold : pd.DataFrame
        Training fold rows.
    case_mask : pd.Series
        Boolean Series (index aligned to ``df_fold``) where True = case.

    Returns
    -------
    df_cases : pd.DataFrame
    df_controls : pd.DataFrame
    """
    df_cases = df_fold[case_mask].copy()
    df_controls = df_fold[df_fold["y_control"]].copy()
    return df_cases, df_controls
