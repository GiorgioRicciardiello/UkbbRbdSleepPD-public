"""
matching.py
===========

Case-control matching for the P1 Combined training paradigm.

Implements random 1:N matching without replacement within each outer CV fold.
Adapted from library/screening/matching.py.

Design
------
* Controls are defined by the outcome-agnostic ``control`` column (boolean):
  subjects with no PD, AD, or dementia diagnosis and no neurological exclusion.
* Cases are subjects with y_incident == 1 OR y_prevalent == 1.
* Matching is per-fold, using a fold-specific RNG seeded from
  ``random_state + fold_idx`` to guarantee reproducibility.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def split_cases_controls(
    y_incident: pd.Series,
    y_prevalent: pd.Series,
    y_control: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Partition row indices into case indices and control indices.

    Parameters
    ----------
    y_incident :
        Binary series; 1 = incident case (diagnosed after actigraphy baseline).
    y_prevalent :
        Binary series; 1 = prevalent case (diagnosed before baseline).
    y_control :
        Boolean series; True = valid control (no neurological disease).

    Returns
    -------
    case_idx :
        Integer indices of subjects that are incident OR prevalent positives.
    control_idx :
        Integer indices of subjects with ``y_control == True`` that are NOT cases.

    Notes
    -----
    A subject flagged as both a case (incident|prevalent) and a control is
    treated as a case. This guards against rare data-quality artefacts.
    """
    is_case = (y_incident.astype(int) | y_prevalent.astype(int)).astype(bool)
    is_ctrl = y_control.astype(bool) & ~is_case

    case_idx = np.where(is_case)[0]
    control_idx = np.where(is_ctrl)[0]

    if len(case_idx) == 0:
        warnings.warn("No cases found in the provided fold.", stacklevel=2)
    if len(control_idx) == 0:
        warnings.warn("No controls found in the provided fold.", stacklevel=2)

    return case_idx, control_idx


def match_controls(
    case_idx: np.ndarray,
    control_idx: np.ndarray,
    controls_per_case: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample controls without replacement to achieve 1:N ratio.

    Parameters
    ----------
    case_idx :
        Indices of case subjects (output of ``split_cases_controls``).
    control_idx :
        Indices of eligible control subjects.
    controls_per_case :
        Desired number of controls per case (N in 1:N matching).
    rng :
        Seeded NumPy Generator for reproducible sampling.

    Returns
    -------
    selected_control_idx :
        Subset of ``control_idx``; size = min(needed, available).

    Notes
    -----
    If the control pool is smaller than ``n_cases * controls_per_case``, all
    available controls are used and a warning is emitted.
    """
    n_needed = len(case_idx) * controls_per_case
    n_available = len(control_idx)

    if n_available == 0:
        return np.array([], dtype=int)

    if n_available < n_needed:
        warnings.warn(
            f"Control pool ({n_available}) < required ({n_needed}). "
            "Using all available controls.",
            stacklevel=2,
        )
        return control_idx.copy()

    selected = rng.choice(control_idx, size=n_needed, replace=False)
    return selected
