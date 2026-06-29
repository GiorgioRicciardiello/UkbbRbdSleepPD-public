import pandas as pd
from pathlib import Path
from config.config import  outcomes
from library.ehr_outcomes.utils import report_outcomes_by_flags


def add_data_split_flags(df_ehr: pd.DataFrame=None,
                     df_val_pd: pd.DataFrame=None,
                     df_val_dlb: pd.DataFrame=None,
                     df_train_sleep_model:pd.DataFrame=None,
                        verbose: bool=False,
                    save_dir:Path = None,
                         ) -> pd.DataFrame:
    """
    Generates a processed dataset by integrating various datasets, renaming fields, and filtering
    entries based on the presence of sleep features. The processed dataset includes validation
    flags, renamed metadata fields, and a summary statistics table of key variables.

    :param df_ehr: Original EHR dataframe with participant records.
    :param df_val_pd: Dataframe for validation purposes, includes participant IDs flagged for PD.
    :param df_val_dlb: Dataframe for validation purposes, includes participant IDs flagged for DLB.
    :param df_train_sleep_model: Dataframe containing IDs of participants used for model training.
    :param path_out: Output path where the processed dataframe will be saved in parquet format.
    :return: A processed Pandas DataFrame containing filtered and updated participant records with
        added flags and renamed columns.
    """

    # ------------------------------------------------------------
    # 3. Validation flags for PD and DLB
    # ------------------------------------------------------------
    print(f'Flagging participants for PD and DLB:')
    df_ehr["val_pd"] = df_ehr["eid"].isin(df_val_pd["eid"])
    df_ehr["val_dlb"] = df_ehr["eid"].isin(df_val_dlb["eid"])
    # ------------------------------------------------------------
    # 3B. Flag records used for training
    # ------------------------------------------------------------
    print(f'Flagging participants used for training:')
    df_train_sleep_model_ied = df_train_sleep_model.loc[~df_train_sleep_model["eid"].isna(), "eid"]
    df_train_sleep_model_ied = df_train_sleep_model_ied.astype(int)
    df_ehr['train_sleep'] = df_ehr['eid'].isin(df_train_sleep_model_ied)
    # this indexes are only valid when visting number is 0
    # df_ehr.loc[(df_ehr['train_sleep'] == True) &
    #            (df_ehr['visit_number'] != 0), 'train_sleep'] = False

    report = report_outcomes_by_flags(df=df_ehr,
         outcomes=outcomes,
        verbose=verbose,
         flags=['val_pd', 'val_dlb', 'train_sleep'],
         output_path=save_dir.joinpath("splits_models.csv") if save_dir else None)

    return df_ehr


