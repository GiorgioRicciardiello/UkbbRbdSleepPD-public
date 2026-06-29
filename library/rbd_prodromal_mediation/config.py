"""
Mediation-specific configuration.

Inherits shared constants from cox_config.py. Adds only parameters
unique to the mediation analysis (RBD high-risk percentile, inconsistency
flag logic).
"""
from __future__ import annotations

from typing import Dict, List

from library.cox_prodromal.cox_config import (
    BASE_COVARIATES,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    MIN_PREVALENCE_FOR_BINARY,
    PRIMARY_OUTCOME,
    PRODROMAL_BINARY_VARS,
    PRODROMAL_VARS,
    RIDGE_PENALIZER,
)

# Re-export inherited constants so callers import from one place
__all__ = [
    "BASE_COVARIATES",
    "BOOTSTRAP_N",
    "BOOTSTRAP_SEED",
    "MIN_PREVALENCE_FOR_BINARY",
    "PRIMARY_OUTCOME",
    "PRODROMAL_BINARY_VARS",
    "PRODROMAL_VARS",
    "RIDGE_PENALIZER",
    "RBD_HIGH_PERCENTILE",
    "RBD_3G_PERCENTILES",
    "INTERPRETATION_A_VARS",
    "INTERPRETATION_C_VARS",
]

# ── Mediation-specific parameters ─────────────────────────────────────────

# Percentile threshold for binary high-RBD encoding (Model 1b)
RBD_HIGH_PERCENTILE: float = 99.0

# Percentile thresholds for 3-group encoding (supplementary b-path)
RBD_3G_PERCENTILES: tuple = (90.0, 99.0)

# ── Interpretation variable sets ──────────────────────────────────────────
# Interpretation A: binary prodromal markers (HES/medication derived)
INTERPRETATION_A_VARS: Dict[str, str] = dict(PRODROMAL_BINARY_VARS)

# Interpretation C: continuous cognitive markers
INTERPRETATION_C_VARS: Dict[str, str] = dict(PRODROMAL_VARS)

# Combined for convenience
ALL_MEDIATION_VARS: Dict[str, str] = {**INTERPRETATION_A_VARS, **INTERPRETATION_C_VARS}
