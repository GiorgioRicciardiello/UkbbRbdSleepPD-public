"""
Categorical reference group selection.

Convention: Always use lowest-risk category as reference (HR=1.0).
For RBD groups: Low
For prodromal cognitive: Low (lowest cognitive score)
For prodromal binary: No (absence of symptom)
"""

from typing import List


def pick_reference_category(cols_list: List[str]) -> str:
    """
    Identify lowest-risk category from dummy-encoded column names.

    Searches for substrings (in order of priority):
    1. "low" — RBD risk groups, cognitive markers
    2. "never" — Smoking/alcohol status
    3. "no" — Binary markers (column ends with "_no")
    4. Falls back to first alphabetically

    Parameters
    ----------
    cols_list : list[str]
        Column names from pd.get_dummies(..., drop_first=False)
        Format: "prefix_category1", "prefix_category2", ...

    Returns
    -------
    str
        Column name to drop (reference category)

    Examples
    --------
    >>> cols = ["rbd_High", "rbd_Low", "rbd_Mid"]
    >>> pick_reference_category(cols)
    'rbd_Low'

    >>> cols = ["prod_No", "prod_Yes"]
    >>> pick_reference_category(cols)
    'prod_No'
    """
    for col in sorted(cols_list):
        col_lower = col.lower()
        # Check for explicit low-risk markers
        if "low" in col_lower or "never" in col_lower:
            return col
        # Check for binary "no"
        if col_lower.split("_")[-1] == "no":
            return col

    # Fallback: first alphabetically
    return sorted(cols_list)[0]
