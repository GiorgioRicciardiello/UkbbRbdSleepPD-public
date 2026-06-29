"""
feature_sets.py
===============

Configuration for multi-feature-set experiments.

Each feature set specifies which demographic, RBD, genetic, and prodromal
variables to include in the model.
"""
from __future__ import annotations

from typing import Final

#: Explicit prodromal marker feature names (8 binary HES-derived markers).
#: Matches the screening model's prodromal feature set.
PRODROMAL_MARKERS: Final[tuple[str, ...]] = (
    'prodromal_anosmia_bl',
    'prodromal_anxiety_bl',
    'prodromal_constipation_bl',
    'prodromal_depression_bl',
    'prodromal_dream_enactment_bl',
    'prodromal_erectile_dysfunction_bl',
    'prodromal_hyposmia_bl',
    'prodromal_orthostatic_bl',
)

#: Feature set specifications: {name → {component → [columns]}}.
FEATURE_SETS: Final[dict[str, dict]] = {
    "rbd_alone": {
        "label": "RBD alone + demographics",
        "demographics": ["cov_age_recruitment_21022", "cov_sex_31", "bmi_21001_bl"],
        "rbd": ["abk_rbd_score_mean"],
        "prs": [],
        "pcs": [],
        "prodromal": [],
    },
    "rbd_prodromal": {
        "label": "RBD + Prodromal (8 markers) + demographics",
        "demographics": ["cov_age_recruitment_21022", "cov_sex_31", "bmi_21001_bl"],
        "rbd": ["abk_rbd_score_mean"],
        "prs": [],
        "pcs": [],
        "prodromal": list(PRODROMAL_MARKERS),
    },
    "rbd_prs": {
        "label": "RBD + PRS + demographics",
        "demographics": ["cov_age_recruitment_21022", "cov_sex_31", "bmi_21001_bl"],
        "rbd": ["abk_rbd_score_mean"],
        "prs": ["prs_score_pd"],
        "pcs": [
            # "prs_pc1", "prs_pc2", "prs_pc3", "prs_pc4", "prs_pc5",
            #     "prs_pc6", "prs_pc7", "prs_pc8", "prs_pc9", "prs_pc10"
        ],
        "prodromal": [],
    },
    "rbd_prs_prodromal": {
        "label": "RBD + PRS + Prodromal (8 markers) + demographics",
        "demographics": ["cov_age_recruitment_21022", "cov_sex_31", "bmi_21001_bl"],
        "rbd": ["abk_rbd_score_mean"],
        "prs": ["prs_score_pd"],
        "pcs": [
            # "prs_pc1", "prs_pc2", "prs_pc3", "prs_pc4", "prs_pc5",
            #     "prs_pc6", "prs_pc7", "prs_pc8", "prs_pc9", "prs_pc10"
        ],
        "prodromal": list(PRODROMAL_MARKERS),
    },
    "rbd_trail_ratio": {
        "label": "RBD + Trail Making Test Ratio (TMT-B/A) + demographics",
        "demographics": ["cov_age_recruitment_21022", "cov_sex_31", "bmi_21001_bl"],
        "rbd": ["abk_rbd_score_mean"],
        "prs": [],
        "pcs": [],
        "tmt": ["cog_tmt_ratio_log_bl"],
        "prodromal": [],
    },
}

#: Default feature set (used if none specified).
DEFAULT_FEATURE_SET: Final[str] = "rbd_alone"


def get_feature_set(name: str) -> dict:
    """Retrieve feature set config by name."""
    if name not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set: {name!r}. Available: {list(FEATURE_SETS.keys())}"
        )
    return FEATURE_SETS[name]


def build_feature_list(fs: dict) -> tuple[str, ...]:
    """
    Concatenate all feature components into a single ordered list.

    Components are assembled in order: demographics, rbd, prs, pcs, prodromal, tmt.
    """
    features = (
        fs.get("demographics", []) +
        fs.get("rbd", []) +
        fs.get("prs", []) +
        fs.get("pcs", []) +
        fs.get("prodromal", []) +
        fs.get("tmt", [])
    )
    return tuple(features)
