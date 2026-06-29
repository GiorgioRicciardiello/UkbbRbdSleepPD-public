"""
Feature engineering and preprocessing pipeline.

Builds an sklearn ColumnTransformer that handles:
  - Numeric features: median imputation (fit on train, applied to test)
  - Binary prodromal features: constant imputation with 0 (absent = not diagnosed)
  - PRS score: median imputation (absent for non-European subjects)
  - TMT ratio: median imputation + pre-computed missingness flag passed through
  - RBD categorical (rg_pctl3): one-hot encoding with fixed category order

All transformers must be fit on the training fold only to prevent data leakage.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from library.screening.config import (
    DEMO_FEATURES,
    PRODROMAL_BINARY,
    PRS_FEATURES,
    RBD_CATEGORICAL,
    RBD_CONTINUOUS,
    TMT_FEATURES,
    TMT_MISSING_FLAG,
)

logger = logging.getLogger(__name__)

# ── Fixed category order for rg_pctl3 one-hot encoding ───────────────────────
# Explicit ordering ensures consistent column positions across all folds.
RBD_CATEGORY_ORDER: List[str] = [
    "Low (0,90%)",
    "Intermediate (90,99%)",
    "High (99,100%)",
]


def build_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    """
    Build a ColumnTransformer for the feature matrix.

    The transformer is not yet fitted — call ``.fit_transform(X_train)``
    on the training fold and ``.transform(X_test)`` on the test fold.

    Parameters
    ----------
    feature_cols : list[str]
        Columns present in the data (subset of ALL_FEATURE_COLS).

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessing pipeline.
    """
    # ── Subsets restricted to what is actually present ────────────────────────
    numeric_cols = [
        c for c in (DEMO_FEATURES + RBD_CONTINUOUS + TMT_FEATURES)
        if c in feature_cols
    ]
    prs_cols = [c for c in PRS_FEATURES if c in feature_cols]
    binary_cols = [c for c in PRODROMAL_BINARY if c in feature_cols]
    tmt_flag_cols = [TMT_MISSING_FLAG] if TMT_MISSING_FLAG in feature_cols else []
    rbd_cat_cols = [c for c in RBD_CATEGORICAL if c in feature_cols]

    transformers = []

    if numeric_cols:
        # Median imputation preserves distribution under MCAR assumption.
        # fit on training fold → applied to test fold without leakage.
        transformers.append((
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_cols,
        ))

    if prs_cols:
        # PRS is missing for non-European or unmatched subjects.
        # Median imputation is defensible; a missingness indicator could be
        # added as a future refinement if PRS-missingness correlates with outcome.
        transformers.append((
            "prs",
            SimpleImputer(strategy="median"),
            prs_cols,
        ))

    if binary_cols:
        # Missing binary prodromal = not diagnosed in HES → impute as 0.
        # This is the standard assumption for HES-derived binary indicators:
        # absence of a code is treated as absence of the condition.
        transformers.append((
            "binary_prodromal",
            SimpleImputer(strategy="constant", fill_value=0),
            binary_cols,
        ))

    if tmt_flag_cols:
        # tmt_missing is already a 0/1 indicator; pass through unchanged.
        transformers.append((
            "tmt_flag",
            SimpleImputer(strategy="constant", fill_value=0),
            tmt_flag_cols,
        ))

    if rbd_cat_cols:
        # One-hot encode with fixed category order so feature indices are
        # stable across folds.  rg_pctl3 is present for all subjects (including
        # prevalent cases); handle_unknown='ignore' is a safety guard only.
        ohe = OneHotEncoder(
            categories=[RBD_CATEGORY_ORDER],
            handle_unknown="ignore",
            sparse_output=False,
        )
        transformers.append((
            "rbd_categorical",
            ohe,
            rbd_cat_cols,
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",    # drop any column not explicitly listed
        verbose_feature_names_out=True,
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer, feature_cols: List[str]) -> List[str]:
    """
    Return feature names after transformation (for SHAP labelling).

    Must be called after ``preprocessor.fit``.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted transformer.
    feature_cols : list[str]
        Original feature columns (unused but kept for API symmetry).

    Returns
    -------
    list[str]
        Output feature names in column order.
    """
    return list(preprocessor.get_feature_names_out())


def extract_feature_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Slice ``df`` to the requested feature columns.

    Returns a DataFrame (not a numpy array) so that the ColumnTransformer
    can address columns by name.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
        Columns to extract; must be a subset of ``df.columns``.

    Returns
    -------
    pd.DataFrame
        Shape (n_samples, len(feature_cols)).
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from DataFrame: {missing}")
    return df[feature_cols].copy()
