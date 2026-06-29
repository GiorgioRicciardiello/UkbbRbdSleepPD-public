"""
Medication Flags

Detects self-reported medication use by drug family from the merged UK Biobank
dataset and adds per-subject binary flags and earliest-report dates.

Two coding layers are supported automatically:
  - data_coding_4   : verbatim medication codes (field 20003, large int codes)
  - Broad codings   : category-level fields (e.g. data_coding_100628 "Laxatives"=6)

The lookup table (field × coding_name × meaning × code) is built internally
from the raw data-sheet files; callers only supply the directory path.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Drug families of interest (keywords substring-matched against 'meaning')
# ---------------------------------------------------------------------------

DRUG_FAMILIES: dict[str, list[str]] = {
    "pd_medications": [
        # Dopamine precursors + combinations
        "levodopa", "co-careldopa", "co-beneldopa",
        "sinemet", "madopar", "stalevo", "duodopa",
        # MAO-B inhibitors
        "selegiline", "rasagiline",
        # COMT inhibitors
        "entacapone", "tolcapone",
        # Dopamine agonists
        "bromocriptine", "pergolide", "cabergoline",
        "pramipexole", "ropinirole", "rotigotine", "apomorphine",
        # NMDA antagonist
        "amantadine",
    ],
    "pde5_inhibitors": [
        "sildenafil", "tadalafil",
        "vardenafil", "avanafil",
    ],
    "orthostatic_hypotension": [
        "midodrine", "fludrocortisone",
        "droxidopa",
    ],
    "laxatives": [
        "lactulose", "macrogol", "movicol",
        "senna", "bisacodyl", "psyllium",
        "ispaghula", "docusate",
    ],
    "depression": [
        # SSRIs
        "fluoxetine", "sertraline", "paroxetine", "citalopram",
        "escitalopram", "fluvoxamine",
        # SNRIs
        "venlafaxine", "duloxetine", "desvenlafaxine",
        # Atypical/other antidepressants
        "bupropion", "mirtazapine", "trazodone",
        # Tricyclic antidepressants
        "amitriptyline", "imipramine", "doxepin", "nortriptyline",
        "clomipramine", "trimipramine", "desipramine",
        # MAOIs
        "phenelzine", "tranylcypromine", "moclobemide", "isocarboxazid",
    ],
    "anxiety": [
        # Benzodiazepines
        "diazepam", "lorazepam", "alprazolam", "clonazepam",
        "oxazepam", "temazepam", "chlordiazepoxide", "bromazepam",
        "midazolam", "nitrazepam",
        # Other anxiolytics
        "buspirone", "hydroxyzine",
    ],
}

# Default raw-data file names (identical for all UKBB application exports)
_DEFAULT_DATA_DICT_FILENAME = "app45551_20251118060954.dataset.data_dictionary.csv"
_DEFAULT_CODINGS_FILENAME = "app45551_20251118060954.dataset.codings.csv"

# "macrogol" keyword should not match topical skin preparations
_MACROGOL_TOPICAL_PATTERN = re.compile(
    r"ointment|cream|bath|lauromacrogol|cetomacrogol", re.IGNORECASE
)

# Regex to extract instance index k from a column like "p20003_i2_a0"
_INSTANCE_RE = re.compile(r"_i(\d+)_")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_df_map(
    data_sheet_dir: Path,
    df_merged_columns: list[str],
    data_dict_filename: str,
    codings_filename: str,
) -> pd.DataFrame:
    """Build the field × coding_name × meaning × code lookup table.

    Reads the raw data-dictionary and codings CSV files, filters to medication
    fields that are actually present in df_merged, and returns their merged
    lookup.

    Args:
        data_sheet_dir: Directory containing the raw UKBB data-sheet files.
        df_merged_columns: Column names of the merged dataset (used to identify
                           which field prefixes are present).
        data_dict_filename: Filename of the data dictionary CSV.
        codings_filename: Filename of the codings CSV.

    Returns:
        DataFrame with columns [field, coding_name, meaning, code].
        ``field`` is the column prefix in df_merged (e.g. "p20003").
    """
    # Identify medication field prefixes present in df_merged
    df_data_dict = pd.read_csv(
        data_sheet_dir / data_dict_filename, low_memory=False
    )
    df_data_dict["field"] = df_data_dict["name"].apply(
        lambda x: x.split("_")[0]
    )

    # Restrict to fields that (a) have a coding_name and (b) are present as
    # column prefixes in df_merged
    present_prefixes = {c.split("_")[0] for c in df_merged_columns}
    df_data_dict_med = (
        df_data_dict.loc[
            df_data_dict["field"].isin(present_prefixes)
            & df_data_dict["coding_name"].notna(),
            ["coding_name", "field"],
        ]
        .drop_duplicates(subset="field", keep="first")
    )

    # Load coding dictionary and filter to medication codings
    df_coding_dict = pd.read_csv(
        data_sheet_dir / codings_filename, low_memory=False
    )
    df_coding_dict_med = df_coding_dict.loc[
        df_coding_dict["coding_name"].isin(df_data_dict_med["coding_name"]),
        ["coding_name", "meaning", "code"],
    ]

    df_map = pd.merge(df_data_dict_med, df_coding_dict_med, on="coding_name", how="left")
    return df_map


def _build_family_code_map(
    df_map: pd.DataFrame,
    drug_families: dict[str, list[str]],
) -> dict[str, dict[str, set[int]]]:
    """Build mapping: field_prefix → family → set of matching integer codes.

    Args:
        df_map: Table with columns [field, coding_name, meaning, code].
        drug_families: Family name → list of substring keywords.

    Returns:
        ``{field_prefix: {family: {code, ...}}}``.
    """
    result: dict[str, dict[str, set[int]]] = {}

    for field_prefix, grp in df_map.groupby("field"):
        family_codes: dict[str, set[int]] = {}
        meanings = grp["meaning"].astype(str)
        codes = grp["code"]

        for family, keywords in drug_families.items():
            matched: set[int] = set()
            for kw in keywords:
                mask = meanings.str.contains(kw, case=False, regex=False, na=False)
                # Exclude topical macrogol preparations
                if kw.lower() == "macrogol":
                    topical = meanings.str.contains(
                        _MACROGOL_TOPICAL_PATTERN.pattern,
                        case=False, regex=True, na=False,
                    )
                    mask = mask & ~topical
                matched.update(codes[mask].dropna().astype(int).tolist())

            if matched:
                family_codes[family] = matched

        if family_codes:
            result[field_prefix] = family_codes

    return result


def _instance_date_map(df: pd.DataFrame) -> dict[int, str]:
    """Map instance index k → follow_up_date_i{k} column name.

    Field 53 (date of assessment visit) is stored as ``p53_i{k}`` in the raw
    UKBB export and renamed to ``follow_up_date_i{k}`` in the pipeline
    (see build_ukb_dataset.py).  Note: no ``_a0`` suffix — field 53 is a
    scalar per instance, not an array field.

    Args:
        df: The merged UK Biobank DataFrame.

    Returns:
        ``{instance_index: column_name}``, e.g. ``{0: "follow_up_date_i0"}``.
    """
    pattern = re.compile(r"^follow_up_date_i(\d+)$")
    return {
        int(m.group(1)): col
        for col in df.columns
        if (m := pattern.match(col))
    }


def _get_field_columns(df: pd.DataFrame, field_prefix: str) -> list[str]:
    """Return all df columns whose name starts with ``field_prefix_``."""
    return [c for c in df.columns if c.startswith(field_prefix + "_")]


def _parse_instance(col_name: str) -> Optional[int]:
    """Extract the instance index from a column name like ``p20003_i2_a0``."""
    m = _INSTANCE_RE.search(col_name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def add_medication_flags(
    df: pd.DataFrame,
    data_sheet_dir: Path,
    data_dict_filename: str = _DEFAULT_DATA_DICT_FILENAME,
    codings_filename: str = _DEFAULT_CODINGS_FILENAME,
    col_prefix: str = "med_",
    save_dir: Optional[Path] = None,
    verbose: bool = True,
    overwrite: bool = True,
) -> pd.DataFrame:
    """Add binary medication-family flags and earliest report dates to df.

    For each family defined in DRUG_FAMILIES, the function:
      1. Builds the field × coding_name × meaning × code lookup internally.
      2. Resolves which numeric codes correspond to each family's keywords.
      3. Scans medication columns in df for those codes (vectorized).
      4. Writes a boolean flag (ever reported) and the earliest assessment-
         centre visit date (field 53, renamed to follow_up_date_i{k}).

    Assumptions:
      - Medication columns are named ``p{field}_i{k}_a{j}`` with float64
        values (numeric codes after _convert_data_types).
      - Negative codes (-1, -3, -7) are sentinel non-answers and will not
        match any drug keyword.
      - Visit dates exist as ``follow_up_date_i{k}`` columns (field 53,
        scalar per instance — no ``_a{j}`` suffix).

    Args:
        df: Merged UK Biobank DataFrame (subject-level).
        data_sheet_dir: Directory containing the raw UKBB data-sheet CSVs.
        data_dict_filename: Data-dictionary CSV filename (default matches the
                            standard UKBB export naming convention).
        codings_filename: Codings CSV filename (same convention).
        col_prefix: Prefix for output column names (default ``"med_"``).
        save_dir: If provided, saves a summary log CSV here.
        verbose: Whether to print progress and per-family counts.

    Returns:
        New DataFrame with added columns per family:
          - ``{col_prefix}{family}``       : bool (True = ever reported)
          - ``{col_prefix}{family}_date``  : pd.Timestamp | NaT (earliest report)
    """
    # if not overwrite and
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Build lookup table from raw files
    # ------------------------------------------------------------------
    if verbose:
        print("  Building medication lookup table from raw data-sheet files...")

    df_map = _build_df_map(
        data_sheet_dir=data_sheet_dir,
        df_merged_columns=list(df.columns),
        data_dict_filename=data_dict_filename,
        codings_filename=codings_filename,
    )
    if verbose:
        print(f"  Lookup: {len(df_map):,} rows, "
              f"{df_map['field'].nunique()} fields, "
              f"{df_map['coding_name'].nunique()} codings")

    # ------------------------------------------------------------------
    # 2. Build field_prefix → family → code set
    # ------------------------------------------------------------------
    family_code_map = _build_family_code_map(df_map, DRUG_FAMILIES)

    if verbose:
        print("  Family-code mapping:")
        for fp, fam_dict in family_code_map.items():
            for fam, codes in fam_dict.items():
                print(f"    {fp:12s}  {fam:30s}  {len(codes):3d} codes")

    # Warn about families with no codes found anywhere
    matched_families = {f for fd in family_code_map.values() for f in fd}
    for family in DRUG_FAMILIES:
        if family not in matched_families:
            warnings.warn(
                f"[medications] Family '{family}' matched no codes in the "
                "coding dictionary. Check keywords."
            )

    # ------------------------------------------------------------------
    # 3. Instance → date column mapping
    # ------------------------------------------------------------------
    inst_date = _instance_date_map(df)
    if not inst_date and verbose:
        print("  Warning: no follow_up_date_i* columns found (field 53); dates will be NaT.")

    # ------------------------------------------------------------------
    # 4. Vectorized detection
    # ------------------------------------------------------------------
    family_flag: dict[str, pd.Series] = {
        f: pd.Series(False, index=df.index) for f in DRUG_FAMILIES
    }
    family_date: dict[str, pd.Series] = {
        f: pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        for f in DRUG_FAMILIES
    }

    for field_prefix, fam_dict in family_code_map.items():
        cols = _get_field_columns(df, field_prefix)
        if not cols:
            continue

        # Group columns by instance
        instance_cols: dict[int, list[str]] = {}
        for col in cols:
            inst = _parse_instance(col)
            if inst is not None:
                instance_cols.setdefault(inst, []).append(col)

        for family, codes in fam_dict.items():
            codes_float = {float(c) for c in codes}

            for inst, inst_col_list in instance_cols.items():
                match_mask = df[inst_col_list].isin(codes_float).any(axis=1)
                family_flag[family] = family_flag[family] | match_mask

                date_col = inst_date.get(inst)
                if date_col and date_col in df.columns:
                    visit_date = pd.to_datetime(df[date_col], errors="coerce")
                    candidate = visit_date.where(match_mask)
                    # Element-wise minimum, ignoring NaT
                    family_date[family] = family_date[family].combine(
                        candidate,
                        lambda a, b: (
                            min(x for x in (a, b) if pd.notna(x))
                            if any(pd.notna(x) for x in (a, b))
                            else pd.NaT
                        ),
                    )

    # ------------------------------------------------------------------
    # 5. Assign output columns and log
    # ------------------------------------------------------------------
    log_rows = []

    if verbose:
        header = f"  {'Family':<35} {'reported':>10} {'with_date':>10} {'both':>6}"
        print("\n" + header)
        print("  " + "-" * (len(header) - 2))

    for family in DRUG_FAMILIES:
        flag_col = col_prefix + family
        date_col = col_prefix + family + "_date"

        df[flag_col] = family_flag[family].astype(bool)
        df[date_col] = family_date[family]

        n_reported  = int(df[flag_col].sum())
        n_with_date = int(df[date_col].notna().sum())
        n_both      = int((df[flag_col] & df[date_col].notna()).sum())

        if verbose:
            print(f"  {family:<35} {n_reported:>10,} {n_with_date:>10,} {n_both:>6,}")

        log_rows.append({
            "family":      family,
            "n_reported":  n_reported,
            "n_with_date": n_with_date,
            "n_both":      n_both,
        })

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        log_path = save_dir / "medication_flags_log.csv"
        pd.DataFrame(log_rows).to_csv(log_path, index=False)
        if verbose:
            print(f"\n  Log saved to {log_path}")

    return df
