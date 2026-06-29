"""
outcomes.py
===========

Explicit outcome definition registry for cross-sectional classification tasks.

Each outcome specifies:
- A binary outcome column (label at any time point)
- An optional event date column (for time-to-event analysis)
- A human-readable description

This centralizes outcome definitions and makes them transparent and modifiable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class OutcomeConfig:
    """Configuration for a binary outcome variable."""

    outcome_col: str
    """Column name of the binary outcome (0/1)."""

    event_date_col: str
    """Column name of the event date (if applicable)."""

    description: str
    """Human-readable description of the outcome."""


#: Available outcomes: {name → OutcomeConfig}.
#: Customize by adding new outcomes or modifying existing ones.
OUTCOMES: Final[dict[str, OutcomeConfig]] = {
    "pd_only": OutcomeConfig(
        outcome_col="outcome_1a_pd_only",
        event_date_col="outcome_1a_pd_only_date",
        description="PD (Parkinson's Disease) only — diagnosed with PD, no AD or DLB",
    ),
    "pd_ad": OutcomeConfig(
        outcome_col="outcome_1b_pd_ad",
        event_date_col="outcome_1b_pd_ad_date",
        description="PD or AD — diagnosed with either Parkinson's or Alzheimer's disease",
    ),
    "ad_only": OutcomeConfig(
        outcome_col="outcome_4a_ad_only",
        event_date_col="outcome_4a_ad_only_date",
        description="AD only — diagnosed with Alzheimer's disease only",
    ),
    "other_dementia": OutcomeConfig(
        outcome_col="outcome_2a_otherdementia",
        event_date_col="outcome_2a_otherdementia_date",
        description="Other dementia (excl. PD, AD, DLB)",
    ),
    "pd_other_dementia": OutcomeConfig(
        outcome_col="outcome_2b_pd_otherdementia",
        event_date_col="outcome_2b_pd_otherdementia_date",
        description="PD or other dementia (excl. AD, DLB)",
    ),
    "dlb_only": OutcomeConfig(
        outcome_col="outcome_3a_dlb_only",
        event_date_col="outcome_3a_dlb_only_date",
        description="DLB only — diagnosed with Lewy body dementia only",
    ),
}

#: Default outcome (used if none specified).
DEFAULT_OUTCOME: Final[str] = "pd_only"


def get_outcome_config(name: str) -> OutcomeConfig:
    """
    Retrieve outcome configuration by name.

    Parameters
    ----------
    name : str
        Outcome name (e.g., "pd_only", "pd_ad", "ad_only").

    Returns
    -------
    OutcomeConfig
        Configuration specifying outcome and date columns.

    Raises
    ------
    ValueError
        If the outcome name is not registered.
    """
    if name not in OUTCOMES:
        raise ValueError(
            f"Unknown outcome: {name!r}. "
            f"Available: {list(OUTCOMES.keys())}"
        )
    return OUTCOMES[name]


def list_outcomes() -> dict[str, str]:
    """Return a dictionary of {outcome_name: description} for easy reference."""
    return {name: cfg.description for name, cfg in OUTCOMES.items()}
