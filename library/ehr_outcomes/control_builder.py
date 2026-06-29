"""
Defines neuro-exclusion rules and clean universal controls.
"""

import pandas as pd
from config.config import outcomes, neuro_exclusion_codes



def neuro_exclusion(df: pd.DataFrame, icd_col:str='41270') -> pd.Series:
    icd_cols = [c for c in df.columns if icd_col in c]
    pattern = "|".join(neuro_exclusion_codes)
    return df[icd_cols].astype(str).apply(
        lambda r: r.str.contains(pattern, regex=True).any(),
        axis=1
    )


def define_controls(df: pd.DataFrame, outcomes:list[str]) -> pd.DataFrame:
    df = df.copy()
    df["neuro_exclude"] = neuro_exclusion(df)


    no_outcomes = (df[outcomes].sum(axis=1) == 0)

    clean_neuro = (
        (~df["PD_flag"]) &
        (~df["AD_flag"]) &
        (~df["DEM_flag"]) &
        (~df["neuro_exclude"])
    )

    df["control"] = no_outcomes & clean_neuro
    return df
