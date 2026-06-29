"""
ml_cross_sectional
==================

Modular ML binary classification library for cross-sectional PD vs non-PD.

The dataset is treated cross-sectionally: a subject is positive if they ever
developed PD between wear-time start and the censor date, and negative
otherwise. Time-to-event is included as a *feature* (not a survival target),
because the temporal distance between actigraphy recording and PD diagnosis
carries information about converter speed.

Pipeline order
--------------
1. ``dataset.convert_to_cross_sectional``  -> flat (X, y) source frame
2. ``features.get_feature_matrix``         -> (X, y) with NaN time_to_event for controls
3. ``training.NestedCVTrainer``            -> nested CV + Optuna over models
4. ``metrics.compute_all_metrics``         -> per-fold metrics
5. ``explainability``                      -> SHAP / permutation importance
6. ``storage.ResultsWriter``               -> CSV / JSON dump per run
"""
from __future__ import annotations

__all__ = [
    "dataset",
    "features",
    "training",
    "metrics",
    "explainability",
    "storage",
    "models",
]
