from matplotlib.lines import Line2D
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from library.column_registry import col_incident, rename_legacy_columns


df = pd.read_parquet(r"C:\Users\riccig01\OneDrive - The Mount Sinai Hospital\Projects\ukbb_pd_lang_data\ehr_diag_pd_rbd_all.parquet")



config = {
    'pp': {
        'final_dir': Path(r'ukbb_pd_lang_data'),
        'thresholds': {
            'root': Path(r'ukbb_pd_lang_data\risk_thresholds'),
        }
    }
}


features = [
    "WASO", "SE", "AI10", "AI10_w", "AI10_REM", "AI10_REM_w", "AI10_NREM", "AI10_NREM_w",
    "AI30", "AI30_w", "AI30_REM", "AI30_REM_w", "AI30_NREM", "AI30_NREM_w",
    "AI60", "AI60_w", "AI60_REM", "AI60_REM_w", "AI60_NREM", "AI60_NREM_w",
    "TA0.5", "TA0.5_w", "TA0.5_REM", "TA0.5_REM_w", "TA0.5_NREM", "TA0.5_NREM_w",
    "TA1", "TA1_w", "TA1_REM", "TA1_REM_w", "TA1_NREM", "TA1_NREM_w",
    "TA1.5", "TA1.5_w", "TA1.5_REM", "TA1.5_REM_w", "TA1.5_NREM", "TA1.5_NREM_w",
    "SIB0", "SIB0_w", "SIB0_REM", "SIB0_REM_w", "SIB0_NREM", "SIB0_NREM_w",
    "SIB1", "SIB1_w", "SIB1_REM", "SIB1_REM_w", "SIB1_NREM", "SIB1_NREM_w",
    "SIB5", "SIB5_w", "SIB5_REM", "SIB5_REM_w", "SIB5_NREM", "SIB5_NREM_w",
    "LIB60", "LIB60_w", "LIB60_REM", "LIB60_REM_w", "LIB60_NREM", "LIB60_NREM_w",
    "LIB120", "LIB120_w", "LIB120_REM", "LIB120_REM_w", "LIB120_NREM", "LIB120_NREM_w",
    "LIB300", "LIB300_w", "LIB300_REM", "LIB300_REM_w", "LIB300_NREM", "LIB300_NREM_w",
    "MMAS", "MMAS_w", "MMAS_REM", "MMAS_REM_w", "MMAS_NREM", "MMAS_NREM_w",
    "T_avg_w", "T_avg_REM", "T_avg_REM_w", "T_avg_NREM", "T_avg_NREM_w",
    "T_std_w", "T_std_REM", "T_std_REM_w", "T_std_NREM", "T_std_NREM_w",
    "HP_A_ac", "HP_A_ac_w", "HP_A_ac_REM", "HP_A_ac_REM_w", "HP_A_ac_NREM", "HP_A_ac_NREM_w",
    "HP_M_ac", "HP_M_ac_w", "HP_M_ac_REM", "HP_M_ac_REM_w", "HP_M_ac_NREM", "HP_M_ac_NREM_w",
    "HP_C_ac", "HP_C_ac_w", "HP_C_ac_REM", "HP_C_ac_REM_w", "HP_C_ac_NREM", "HP_C_ac_NREM_w"
]


def load_prodromal_dataset(
    file_name: str = "ehr_diag_pd_rbd_only_all",
) -> Tuple[dict, pd.DataFrame]:
    """
    Load and clean the production cohort dataset (ABK model).

    Reads from the canonical final directories defined in config.
    ABK outputs are promoted there by
    ``run_merge_ukbb_rbd.py::promote_abk_to_final()``.

    Parameters
    ----------
    file_name : str
        Base name of the parquet / threshold collection.

    Returns
    -------
    thresholds : dict
        Nested dict of risk thresholds by method and outcome.
    df : pd.DataFrame
        Subject-level DataFrame with ``rbd_prob``, covariates,
        survival columns, and risk group columns.
    """
    dir_final = config["pp"]["final_dir"]
    dir_thresh = config["pp"]["thresholds"]["root"]

    thresholds, df_risk = get_clean_risk_data(
        file_name=file_name,
        thresholds_root=dir_thresh,
        final_dir=dir_final,
    )
    df_risk = make_subject_level(df_risk, id_col="eid", prob_col="abk_rbd_score_mean")

    return thresholds, df_risk

def make_subject_level(df, id_col="eid", prob_col="prob_mean"):
    """
    Generates a subject-level DataFrame by aggregating by id_col and select the first value of prob_col.
    prob_col is expected to be the *average* probability across the nights, so it's a single value

    This function groups the input DataFrame by a specified identifier column, retaining
    the first occurrence of each identifier. It also renames a given probability column
    to a predefined name in the resulting DataFrame for standardization purposes.

    :param df: The input pandas DataFrame to process.
    :type df: pandas.DataFrame
    :param id_col: The name of the column used as the unique identifier for grouping.
        Defaults to "eid".
    :type id_col: str, optional
    :param prob_col: The name of the column containing probability values, which will
        be renamed in the resulting DataFrame. Defaults to "prob_mean".
    :type prob_col: str, optional
    :return: A new pandas DataFrame containing one entry per unique identifier, with
        the specified probability column renamed to "rbd_prob".
    :rtype: pandas.DataFrame
    """
    df_subj = (
        df.groupby(id_col, as_index=False)
          .first()
    )
    df_subj = df_subj.rename(columns={prob_col: "rbd_prob"})
    return df_subj


def get_clean_risk_data(thresholds_root: Optional[Path] = None,
                        final_dir: Optional[Path] = None,
                        file_name: str = 'file_name_risk_data') -> tuple[dict, pd.DataFrame]:
    """
    Retrieve the normalized risk thresholds and the associated dataframe for a given file_name.
    Thresholds are in the structure:
                            <method>            <outcome>
        thresholds.get('percentile_3g').get('outcome_1a_pd_only')


    Args:
        file_name (str): The identifier for the file (e.g., 'ehr_diag_pd_rbd_only_val').

    Returns:
        tuple[dict, pd.DataFrame]: A tuple containing the normalized thresholds dictionary
                                   and the loaded DataFrame.
    """
    # 1. Paths from config
    if thresholds_root is None:
        thresholds_root = config['pp']['thresholds']['root']
    if final_dir is None:
        final_dir = config['pp']['final_dir']

    # 2. Construct paths
    # Collection.json is inside a folder named after the file_name
    collection_path = thresholds_root / file_name / 'risk_collection.json'
    parquet_path = final_dir / f"{file_name}.parquet"

    # 3. Load Thresholds (Collection)
    if not collection_path.exists():
        raise FileNotFoundError(f"Risk collection not found at: {collection_path}")

    with open(collection_path, 'r') as f:
        thresholds = json.load(f)

    # 4. Load DataFrame
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataframe parquet not found at: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    # remove the features from the matrix
    col_feat = [f for f in features if f in df.columns]
    if len(col_feat) > 0:
        df = df.drop(columns=col_feat)
    # Exclude neurologically ineligible subjects (prevalent neuro disease at baseline).
    # train_sleep is NOT applied here: the ABK model was not trained on any UKBB subject,
    # so all actigraphy participants are valid for analysis regardless of train_sleep.
    df = df[df['neuro_exclude'] == 0].copy()
    print(f'  Excluded neuro_exclude subjects. Remaining: {df.shape[0]:,}')

    # Exclude subjects with poor actigraphy recording quality.
    # acc_bad_quality = True if ANY of: insufficient wear time (p90015), failed calibration
    # (p90016), not calibrated on own data (p90017), daylight-savings crossover (p90018),
    # unreliable device size (p90002), or non-zero recording problems (p90180).
    # Subjects failing these criteria produced unreliable actigraphy signals; their RBD
    # probability scores cannot be trusted for the survival analysis.
    if 'acc_bad_quality' in df.columns:
        n_before_acc = df.shape[0]
        df = df[df['acc_bad_quality'] != True].copy()
        n_excl_acc = n_before_acc - df.shape[0]
        print(f'  Excluded acc_bad_quality subjects: {n_excl_acc:,}. Remaining: {df.shape[0]:,}')
    else:
        print('  Warning: acc_bad_quality column not found — quality exclusion skipped.')

    # Exclude subjects doing night shifts at the time of actigraphy (instance 2).
    # See library/risk/risk_helpers.py for full rationale.
    _night_shift_col = 'shift_any_i2_p3426'
    if _night_shift_col in df.columns:
        n_before_ns = df.shape[0]
        df = df[df[_night_shift_col] != 1].copy()
        n_excl_ns = n_before_ns - df.shape[0]
        print(f'  Excluded night-shift (i2) subjects: {n_excl_ns:,}. Remaining: {df.shape[0]:,}')
    else:
        print(f'  Warning: {_night_shift_col} column not found — night-shift exclusion skipped.')

    # Migrate legacy column names to the new __-separated convention.
    # This is a safety net for stale parquet files that have not been
    # regenerated after the naming convention change.
    df = rename_legacy_columns(df)

    return thresholds, df
