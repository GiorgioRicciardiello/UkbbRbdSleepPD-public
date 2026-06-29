from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from config.config import config

from library.cox_prodromal.cox_config import (
    BASE_COVARIATES,
    PRODROMAL_BINARY_VARS,
    PRODROMAL_VARS,
)
from library.cox_prodromal.data_prep import (
    build_availability_table,
    build_extended_covariates,
    filter_active_variables,
    load_prodromal_dataset,
)

from library.cox_prodromal.utils import save_table

# Columns to retain for Nick's analysis.
# RBD score: abk_rbd_score_mean (averaged nightly score per subject).
# Smoking/alcohol: baseline instance (_i0) field names.
# Outcomes updated 2026-03-30: otherdementia -> vasculardementia; dlb and pd_med removed.
needed_cols = [
  'eid',
  'abk_rbd_score',
  'abk_rbd_score_mean',
  'acc_bad_quality',
  'acc_calib_mean_temp_90170',
  'acc_friday_avg_90023',
  'acc_light_day_avg_40044_i0',
  'acc_light_day_hr_avg_40032_i0',
  'acc_light_overall_40048_i0',
  'acc_light_weekday_hr_avg_40036_i0',
  'acc_light_weekend_hr_avg_40040_i0',
  'acc_monday_avg_90019',
  'acc_mvpa_day_avg_40045_i0',
  'acc_mvpa_day_hr_avg_40033_i0',
  'acc_mvpa_overall_40049_i0',
  'acc_mvpa_weekday_hr_avg_40037_i0',
  'acc_mvpa_weekend_hr_avg_40041_i0',
  'acc_non_wear_duration_90052',
  'acc_overall_acc_avg',
  'acc_saturday_avg_90024',
  'acc_sed_day_avg_40043_i0',
  'acc_sed_day_hr_avg_40031_i0',
  'acc_sed_overall_40047_i0',
  'acc_sed_weekday_hr_avg_40035_i0',
  'acc_sed_weekend_hr_avg_40039_i0',
  'acc_sleep_day_avg_40042_i0',
  'acc_sleep_day_hr_avg_40030_i0',
  'acc_sleep_overall_40046_i0',
  'acc_sleep_weekday_hr_avg_40034_i0',
  'acc_sleep_weekend_hr_avg_40038_i0',
  'acc_sunday_avg_90025',
  'acc_temp_max_90195',
  'acc_temp_mean_90192',
  'acc_temp_min_90194',
  'acc_temp_std_90193',
  'acc_thursday_avg_90022',
  'acc_tuesday_avg_90020',
  'acc_wear_duration_90051',
  'acc_wednesday_avg_90021',
  'ad_flag__surv_event',
  'arm_swing_amplitude_mean_j',
  'arm_swing_amplitude_mean_w',
  'arm_swing_amplitude_var_j',
  'arm_swing_amplitude_var_w',
  'cov_bmi',
  'control',
  'cov_age_recruitment_21022',
  'cov_alcohol',
  'cov_fi_questions_attempted_20128_bl',
  'cov_fluid_intelligence_20016_bl',
  'cov_numeric_memory_max_20240_bl',
  'cov_pairs_status_20244_bl',
  'cov_react_time_mean_20023_bl',
  'cov_react_time_mean_20023_i1',
  'cov_react_time_mean_20023_fu',
  'cov_react_time_mean_20023_i3',
  'cov_sex_31',
  'cov_smoking',

 'cov_recp_location_54_bl',
 'cov_recp_location_54_i1',
 'cov_recp_location_54_fu',
 'cov_recp_location_54_i3',
 
 
  'death_date',
  'death_flag',
  'dem_flag__surv_event',
  'hes_gap_pre_baseline_years',
  'cog_tmt_ratio_log_bl',
  'neuro_exclude',
  'outcome_1a_pd_only',
  'outcome_1a_pd_only__incident',
  'outcome_1a_pd_only__prevalent',
  'outcome_1a_pd_only__surv_days',
  'outcome_1a_pd_only__surv_event',
  # 'outcome_1b_pd_ad__incident',
  # 'outcome_1b_pd_ad__surv_days',
  # 'outcome_1b_pd_ad__surv_event',
  # 'outcome_2a_vasculardementia__incident',
  # 'outcome_2a_vasculardementia__surv_days',
  # 'outcome_2a_vasculardementia__surv_event',
  # 'outcome_2b_pd_vasculardementia__incident',
  # 'outcome_2b_pd_vasculardementia__surv_days',
  # 'outcome_2b_pd_vasculardementia__surv_event',
  # 'outcome_4a_ad_only__incident',
  # 'outcome_4a_ad_only__surv_days',
  # 'outcome_4a_ad_only__surv_event',
  'pd_flag__surv_event',
  'prodromal_anosmia_bl',
  'prodromal_anosmia_bl_date',
  'prodromal_anosmia_bl_source',
  'prodromal_anxiety_bl',
  'prodromal_anxiety_bl_date',
  'prodromal_anxiety_bl_source',
  'prodromal_constipation_bl',
  'prodromal_constipation_bl_date',
  'prodromal_constipation_bl_source',
  'prodromal_depression_bl',
  'prodromal_depression_bl_date',
  'prodromal_depression_bl_source',
  'prodromal_dream_enactment_bl',
  'prodromal_dream_enactment_bl_date',
  'prodromal_dream_enactment_bl_source',
  'prodromal_erectile_dysfunction_bl',
  'prodromal_erectile_dysfunction_bl_date',
  'prodromal_erectile_dysfunction_bl_source',
  'prodromal_hyposmia_bl',
  'prodromal_hyposmia_bl_date',
  'prodromal_hyposmia_bl_source',
  'prodromal_orthostatic_bl',
  'prodromal_orthostatic_bl_date',
  'prodromal_orthostatic_bl_source',
  'rg_pctl2',
  'rg_pctl3',
  'rg_q4',
  'tmt1_dur_baseline',
  'tmt1_err_baseline',
  'tmt2_dur_baseline',
  'tmt2_err_baseline',
  'tmt_lag_days',
  'tmt_missing',
  'tmt_ratio_baseline',
  'tmt_ratio_clinic_i2',
  'tmt_ratio_clinic_i3',
  'tmt_ratio_online_i0',
  'tmt_ratio_online_i1',
  'tmt_source_baseline',
  'wear_time_start',
    'visit_number',
    'gba_carrier',
    'gba_biallelic',
    'gba_cn',
    'gba_deletion_in_gba',
    'gba_allele'

]

# %% Sharing for the genetics analysis
needed_cols = list(set(needed_cols))

thresholds, df_risk = load_prodromal_dataset()

# ── 2. Load data ───────────────────────────────────────────────────
print("[1/7] Loading data ...")
print(f'\t\t Risk Data Dim: {df_risk.shape} | Unique subjects: {df_risk['id'].nunique()}')
print("[2/7] Preparing covariates ...")
df_risk, extended_covariates = build_extended_covariates(df_risk, BASE_COVARIATES)

# ── 3. Data availability ───────────────────────────────────────────
print("[3/7] Filter needed columns ...")
# Column-prune df_risk to reduce IPC pickle overhead (~600 MB → ~100 MB)
# needed_cols = sorted(list(set(needed_cols)))
needed_cols = list(set(needed_cols))
df_risk_slim = df_risk[needed_cols].copy()

df_risk_slim['eid', 'control']

# %% GBA count analysis
cols_gba = [col for col in df_risk.columns if 'gba' in col]

# df_gba = df_risk[cols_gba + ['outcome_1a_pd_only__incident', 'rg_pctl2']]

# GBA carrier counts by risk group and incident PD status
# cols_gba = ["gba_carrier"]  # or however you've named it
rbd_risk_group = 'rg_pctl2'

df_gba = df_risk[
    (df_risk['gba_carrier'].notna()) &
    (df_risk['outcome_1a_pd_only__incident'].notna())
    ].copy()

# Then add risk group (will have NaN for subjects without stratification)
# or filter to analytical cohort first
df_gba = df_gba[
    (df_gba['outcome_1a_pd_only__incident'].astype(bool)) |
    (df_gba['control'].fillna(False).astype(bool))
    ]

print(f"\nGBA analysis cohort: {len(df_gba):,} subjects")
print(f"Incident PD: {df_gba['outcome_1a_pd_only__incident'].sum()}")
print(f"GBA carriers: {(df_gba['gba_carrier'] == 1).sum()}")


# 1. Overall GBA carrier count
n_gba = (df_gba["gba_carrier"] == 1).sum()
n_total = len(df_gba)
pct_gba = 100 * n_gba / n_total
print(f"Total GBA carriers: {n_gba:,} / {n_total:,} ({pct_gba:.2f}%)\n")

# 2. Counts by risk group
print("GBA carriers by risk group:")
gba_by_group = df_gba.groupby(rbd_risk_group)["gba_carrier"].agg(
    ["sum", "count", lambda x: 100 * x.sum() / len(x)]
)
gba_by_group.columns = ["GBA_carriers", "N", "Pct"]
print(gba_by_group)

# 3. Counts by incident PD
print("\nGBA carriers by incident PD status:")
gba_by_pd = df_gba.groupby("outcome_1a_pd_only__incident")["gba_carrier"].agg(
    ["sum", "count", lambda x: 100 * x.sum() / len(x)]
)
gba_by_pd.columns = ["GBA_carriers", "N", "Pct"]
print(gba_by_pd)

# 4. Cross-tabulation: risk group × incident PD
print("\nCross-tabulation (Group × Incident PD × GBA):")
crosstab = pd.crosstab(
    [df_gba[rbd_risk_group], df_gba["outcome_1a_pd_only__incident"]],
    df_gba["gba_carrier"],
    margins=True
)
print(crosstab)

# 5. Event counts per cell (risk group × GBA carrier)
print("\nIncident PD events per cell (Group × GBA carrier):")
events_by_cell = df_gba[df_gba["outcome_1a_pd_only__incident"] == 1].groupby(
    [rbd_risk_group, "gba_carrier"]
).size().unstack(fill_value=0)
print(events_by_cell)
print(f"\nMinimum events per cell: {events_by_cell.min().min()}")



# %%
import json

# assuming your DataFrame is df
columns = df_risk.columns.tolist()
columns = sorted(columns)

# save to JSON file
with open(r"...\data\pp\res_build_final_dataset\columns.json", "w") as f:
    json.dump(columns, f, indent=4)

df_risk_slim['strata'] = df_risk_slim['rg_pctl3'].map({
'Low (0,90%)': 'low',
    'Intermediate (90,99%)': 'low',
    'High (99,100%)': 'high'
})

df_risk_slim.to_parquet(r'...\Projects\ukbb_pd_lang_data\ehr_diag_pd_rbd_all.parquet', index=False)

# %% Sharing for the gait PD Temperature study

df_full_data = pd.read_parquet(config['pp']['final_dir'].joinpath("ehr_diag_pd_rbd_only_all.parquet"))
df_full_data = df_full_data[needed_cols]

df_full_data.to_parquet(r'...\Projects\ukbb_pd_lang_data\pd_walk_temperature\pd_walk_temp.parquet', index=False)


# -- EHR dataset (raw feature dataset, pre-survival processing) ----------------------
# ukb_final_dataset: 101K rows — raw UKB covariates and EHR fields before merging
# # with RBD scores. Useful for covariate inspection and descriptive statistics.
# EHR_PATH = ROOT / "data/pp/data_sheet/ukb_final_dataset.parquet"
#
# df_ehr = pd.read_parquet(EHR_PATH)
# cols_ehr = [col for col in df_ehr.columns if col in cols_need_set]
# df_ehr = df_ehr[cols_ehr]
#
# print(f"EHR: {len(df_ehr):,} rows, {len(df_ehr.columns)} columns")
# missing_ehr = cols_need_set - set(df_ehr.columns)
# if missing_ehr:
#     print(f"  Missing from EHR: {sorted(missing_ehr)}")
#
# OUT_EHR = ROOT / "data/pp/res_build_final_dataset/nick_ehr.parquet"
# df_ehr.to_parquet(OUT_EHR, index=False)
# print(f"Saved EHR -> {OUT_EHR}")
