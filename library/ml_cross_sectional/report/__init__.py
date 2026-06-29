"""
report
======

Reporting utilities for ml_cross_sectional:

* ``distribution`` — feature and outcome distribution tables (global + per-fold).
* ``cross_fs_plots`` — 8-panel cross-feature-set comparison figures analogous
  to the paradigm-comparison figures in ``results/screening_paradigms/``.
"""
from .distribution import (
    feature_distribution_by_class,
    fold_composition_table,
    oof_distribution_summary,
)
from .cross_fs_plots import generate_all_cross_fs_plots

__all__ = [
    "feature_distribution_by_class",
    "fold_composition_table",
    "oof_distribution_summary",
    "generate_all_cross_fs_plots",
]
