from pathlib import Path
import pandas as pd
from typing import Iterable, Union, List, Optional, Tuple, Set
from tabulate import tabulate


from pathlib import Path
from typing import Iterable, Union
import numpy as np
import pandas as pd
from tabulate import tabulate


def report_outcomes_by_flags(
    df: pd.DataFrame,
    outcomes: list[str],
    flags: Union[str, Iterable[str]],
    verbose: bool = True,
    add_percent: bool = True,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Generate an outcome-indexed report with counts (and optional percentages)
    for one or more boolean flag columns.
    """

    if isinstance(flags, str):
        flags = [flags]
    else:
        flags = list(flags)

    # ------------------------------------------------------------
    # Validate outcomes
    # ------------------------------------------------------------
    missing_outcomes = [o for o in outcomes if o not in df.columns]
    if missing_outcomes:
        # Deferred outcomes (e.g. outcome_5a_pd_med) may not exist yet
        # at early pipeline stages. Skip them instead of raising.
        outcomes = [o for o in outcomes if o not in missing_outcomes]
        if not outcomes:
            raise KeyError(f"No outcome columns found in DataFrame.")

    # ------------------------------------------------------------
    # Core counts
    # ------------------------------------------------------------
    report = pd.DataFrame(
        {"n_outcome": df[outcomes].sum()},
        index=outcomes,
    )

    for flag in flags:
        if flag not in df.columns:
            raise KeyError(f"Flag column not found: {flag}")

        # NaN-safe boolean mask
        mask = (
            pd.to_numeric(df[flag], errors="coerce")
            .fillna(0)
            .astype(bool)
        )

        report[f"n_{flag}"] = df.loc[mask, outcomes].sum()

        if add_percent:
            denom = report["n_outcome"].replace(0, np.nan)
            report[f"pct_{flag}"] = (
                report[f"n_{flag}"] / denom * 100
            ).round(2)

    if verbose:
        print(
            tabulate(
                report.reset_index(names="outcome"),
                headers="keys",
                tablefmt="psql",
                showindex=False,
            )
        )

    # ------------------------------------------------------------
    # Optional save
    # ------------------------------------------------------------
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path)

    return report

from typing import List, Tuple
import pandas as pd
from tabulate import tabulate


def build_ukbb_rename_map(
    columns: List[str],
    code_to_name: dict,
    prefix: str = "p",
    print_table: bool = True,
    suffix_map: Optional[dict] = None,
) -> Tuple[dict, pd.DataFrame]:
    """
    Build a rename map for UKBB columns preserving instance suffixes.

    Parameters
    ----------
    columns : list[str]
        Source column names to scan.
    code_to_name : dict
        ``{field_code: new_base_name}`` mapping (field code without prefix).
    prefix : str
        UKBB field prefix (default ``"p"``).
    print_table : bool
        Print the rename table to stdout.
    suffix_map : dict | None
        Optional ``{old_suffix: new_suffix}`` substitution applied to the
        instance suffix of each matched column, e.g.
        ``{"_i0": "_bl", "_i2": "_fu"}``.  The substitution matches either the
        whole suffix (``_i0`` → ``_bl``) or its leading segment for
        array-indexed fields (``_i0_a3`` → ``_bl_a3``).  Suffixes not present in
        the map are preserved verbatim (e.g. ``_i1``, ``_i3``).  When ``None``
        (default) suffixes are passed through unchanged, so existing callers are
        unaffected.

    Returns
    -------
    rename_dict : dict
        {old_col: new_col}
    df_table_rows : pd.DataFrame
        Table with columns:
        ['UKBB column', 'Original column', 'Renamed column']
    """

    rename_dict = {}
    table_rows = []

    for code, new_name in code_to_name.items():
        code_str = f"{prefix}{code}"
        # Match exact field code: "p53_i0" but NOT "p5364_i0"
        # Column must equal code_str exactly OR start with code_str + "_"
        matched_cols = [
            col for col in columns
            if col == code_str or col.startswith(f"{code_str}_")
        ]

        for old_col in matched_cols:
            suffix = old_col[len(code_str):]  # preserves _iX, _iX_aY

            # Optional instance-suffix substitution (e.g. _i0 -> _bl, _i2 -> _fu).
            # Match the whole suffix or its leading instance segment so that
            # array-indexed fields (_i0_a3) keep their trailing _aY part.
            if suffix_map:
                for old_sfx, new_sfx in suffix_map.items():
                    if suffix == old_sfx or suffix.startswith(f"{old_sfx}_"):
                        suffix = new_sfx + suffix[len(old_sfx):]
                        break

            new_col = f"{new_name}{suffix}"

            rename_dict[old_col] = new_col
            table_rows.append([code, old_col, new_col])

    # --------------------------------------------------
    # SAFETY: handle zero matches
    # --------------------------------------------------
    if not table_rows:
        df_table_rows = pd.DataFrame(
            columns=["UKBB column", "Original column", "Renamed column"]
        )
        return rename_dict, df_table_rows

    # Normal case
    df_table_rows = pd.DataFrame(
        table_rows,
        columns=["UKBB column", "Original column", "Renamed column"],
    )

    if print_table:
        print(
            tabulate(
                df_table_rows,
                headers="keys",
                tablefmt="github",
                showindex=False,
            )
        )

    return rename_dict, df_table_rows
