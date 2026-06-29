#!/usr/bin/env python3
"""
Folder-level backup audit: Minerva vs neuron

Compares project folder sizes (GB) using pre-generated CSVs.
Designed for HPC-scale data where per-file comparison is not feasible.

Input CSV format (headerless or with header):
    project,size
Example:
    project_A,120G
    project_B,87G

Output:
    CSV with status per project:
    OK | SIZE_MISMATCH | MISSING_ON_NEURON | EXTRA_ON_NEURON
"""

from pathlib import Path
import re
import pandas as pd

# ------------------------------------------------------------------
# CONFIG — EDIT THESE
# ------------------------------------------------------------------

MINERVA_CSV = Path("minerva_old_lm_folder_sizes.csv")
NEURON_CSV  = Path("neuron_old_lm_folder_sizes.csv")
OUTPUT_CSV = Path("old_lm_backup_audit.csv")

SIZE_TOLERANCE_GB = 1.0   # acceptable GB difference due to rounding

# ------------------------------------------------------------------
# INTERNALS
# ------------------------------------------------------------------

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGTP])?B?\s*$", re.I)
_UNIT_TO_GB = {
    None: 1.0,
    "K": 1 / (1024**2),
    "M": 1 / 1024,
    "G": 1.0,
    "T": 1024.0,
    "P": 1024.0 * 1024.0,
}


def _parse_size_to_gb(val):
    """Convert '120G', '512M', '1T' → GB (float)."""
    if pd.isna(val):
        return None

    s = str(val).strip()
    m = _SIZE_RE.match(s)
    if not m:
        return None

    size = float(m.group(1))
    unit = m.group(2).upper() if m.group(2) else None
    return size * _UNIT_TO_GB[unit]


def _read_folder_csv(path: Path, label: str) -> pd.DataFrame:
    """
    Reads CSV of project folder sizes.
    Normalizes to:
        project | size_<label>_gb
    """
    df = pd.read_csv(path, header=None)

    # Detect header
    if df.shape[1] >= 2:
        if "project" in str(df.iloc[0, 0]).lower():
            df = pd.read_csv(path)

    df = df.iloc[:, :2]
    df.columns = ["project", f"size_{label}"]

    df["project"] = df["project"].astype(str).str.strip()
    df[f"size_{label}_gb"] = df[f"size_{label}"].apply(_parse_size_to_gb)

    # Deduplicate: keep largest (safety for repeated du runs)
    df = (
        df.sort_values(f"size_{label}_gb", ascending=False)
        .drop_duplicates("project")
        .reset_index(drop=True)
    )

    return df[["project", f"size_{label}_gb"]]


def run_backup_audit():
    df_minerva = _read_folder_csv(MINERVA_CSV, "minerva")
    df_neuron  = _read_folder_csv(NEURON_CSV, "neuron")

    df = df_minerva.merge(df_neuron, on="project", how="outer")

    # Status logic
    df["status"] = "OK"

    df.loc[df["size_neuron_gb"].isna(), "status"] = "MISSING_ON_NEURON"
    df.loc[df["size_minerva_gb"].isna(), "status"] = "EXTRA_ON_NEURON"

    both = df["size_minerva_gb"].notna() & df["size_neuron_gb"].notna()
    mismatch = (df["size_minerva_gb"] - df["size_neuron_gb"]).abs() > SIZE_TOLERANCE_GB
    df.loc[both & mismatch, "status"] = "SIZE_MISMATCH"

    # Diagnostics
    df["diff_gb"] = df["size_neuron_gb"] - df["size_minerva_gb"]

    # Order for human review
    priority = {
        "MISSING_ON_NEURON": 0,
        "SIZE_MISMATCH": 1,
        "EXTRA_ON_NEURON": 2,
        "OK": 3,
    }
    df["priority"] = df["status"].map(priority)
    df = df.sort_values(["priority", "project"]).drop(columns="priority")

    # Save
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    # Summary
    print("Backup audit complete")
    print(df["status"].value_counts())
    print(f"Output written to: {OUTPUT_CSV.resolve()}")


# ------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------

if __name__ == "__main__":
    run_backup_audit()
