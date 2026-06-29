"""
Automated case/control exports in CSV, Feather, Parquet.
"""

import pandas as pd
from pathlib import Path


def save_cases(df, outcome, outdir):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    df[df[outcome] == True].to_csv(outdir / f"cases_{outcome}.csv", index=False)


def save_controls(df, outdir):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    df[df["control"] == True].to_csv(outdir / "controls_clean.csv", index=False)


def save_parquet(df, filename, outdir):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    df.to_parquet(outdir / filename)
