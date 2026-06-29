"""
results_collector
=================

Aggregate and report ML cross-sectional results across models and TTE modes.

Submodules
----------
* ``collector`` — discover and load persisted run artifacts
* ``tables``    — publication-ready summary tables (metrics, SHAP, permutation)
* ``figures``   — multi-panel ROC + confusion-matrix figure
* ``runner``    — entry point that generates all outputs for one TTE mode
"""
from __future__ import annotations

__all__ = [
    "collector",
    "tables",
    "figures",
    "runner",
]
