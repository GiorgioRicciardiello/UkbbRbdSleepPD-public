from __future__ import annotations

import numpy as np
import pandas as pd


def make_high_vs_low(
    s: pd.Series,
    mode: str = "merge_medium_with_low"
) -> pd.Series:
    """
    Create binary exposure for primary contrast: High vs (Low [+ Medium]).

    Parameters
    ----------
    s : pd.Series
        Input risk group labels (e.g. low / medium / high).
    mode : str
        One of:
        - "merge_medium_with_low" (DEFAULT): low+medium=0, high=1
        - "drop_medium": low=0, high=1, medium=NaN
        - "include_medium": low=0, medium=0, high=1 (alias of merge, explicit)

    Returns
    -------
    pd.Series
        Binary exposure suitable for Cox/logistic models.
    """

    out = pd.Series(np.nan, index=s.index, dtype="float")

    is_low = s.apply(_is_low_label)
    is_medium = s.apply(_is_medium_label)
    is_high = s.apply(_is_high_label)

    if mode == "merge_medium_with_low":
        out[is_low | is_medium] = 0.0
        out[is_high] = 1.0

    elif mode == "drop_medium":
        out[is_low] = 0.0
        out[is_high] = 1.0
        # medium stays NaN -> caller must dropna()

    elif mode == "include_medium":
        # Explicit alias for clarity in pipelines / reviewers
        out[is_low] = 0.0
        out[is_medium] = 1.0
        out[is_high] = 2.0

    else:
        raise ValueError(
            "mode must be one of "
            "{'merge_medium_with_low', 'drop_medium', 'include_medium'}"
        )

    return out


def _is_high_label(x: str) -> bool:
    """Heuristic: treat any label starting with 'High' as high-risk."""
    return isinstance(x, str) and x.strip().lower().startswith("high")


def _is_low_label(x: str) -> bool:
    """Heuristic: treat any label starting with 'Low' as low-risk."""
    return isinstance(x, str) and x.strip().lower().startswith("low")


def _is_medium_label(x: str) -> bool:
    """Heuristic: treat any label starting with 'Low' as low-risk."""
    return isinstance(x, str) and x.strip().lower().startswith("intermediate")


