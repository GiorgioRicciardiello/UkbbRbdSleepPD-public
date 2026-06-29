"""
Author: Giorgio Ricciardiello
        giocrm@stanford.edu
configurations parameters for the paths
"""
import os
from pathlib import Path

# ── RBD risk group color palette (single source of truth for all plots) ──────
# Three-group scheme: Low / Intermediate (Mid) / High
RBD_RISK_COLORS: dict = {
    "High":         "#f78b8b",  # coral-pink
    "Intermediate": "#fab46f",  # orange
    "Mid":          "#fab46f",  # alias for Intermediate
    "Low":          "#8ec7ea",  # sky-blue
}

# Combined RBD × prodromal variants (darker shades for interaction plots)
RBD_RISK_COLORS_COMBINED: dict = {
    "High_Yes":  "#d44040",  # High RBD + prodromal Yes  (darkened coral)
    "High":      "#f78b8b",  # High RBD alone
    "Low_No":    "#4a8fbd",  # Low  RBD + prodromal No   (darkened blue)
    "Low":       "#8ec7ea",  # Low  RBD alone
    "Mid_Yes":   "#d47020",  # Mid  RBD + prodromal Yes  (darkened orange)
    "Mid":       "#fab46f",  # Mid  RBD alone
}

# import shutil

# Define root path
root_path = Path(__file__).resolve().parents[1]
# Define raw data path
data_path = root_path.joinpath('data')
# data paths
data_raw_path = data_path.joinpath('raw')
data_pp_path = data_path.joinpath('pp')
path_actig_extracted = data_path.joinpath('actig_extracted_features')
# results path
data_res = root_path.joinpath('results')
data_logs = data_res.joinpath('logs')
path_project_rar = root_path.joinpath('models')
path_thresholds =  data_path.joinpath('risk_thresholds')

# External UKBB data-sheet location (controlled-access; NOT shipped with this repo).
# Set the UKBB_DATA_ROOT environment variable to your local UKBB data-sheet
# directory. Falls back to <repo>/data/ukb_datasheet if the variable is unset.
#   Linux/macOS:  export UKBB_DATA_ROOT=/path/to/ukbb_datasheet
#   Windows (PowerShell):  $env:UKBB_DATA_ROOT="C:\path\to\ukbb_datasheet"
_ukbb_data_root = os.environ.get("UKBB_DATA_ROOT")
path_raw_data_sheet = Path(_ukbb_data_root) if _ukbb_data_root else data_path.joinpath("ukb_datasheet")
path_pp_data_sheet = data_pp_path.joinpath('data_sheet')
config = {
    # directory paths
    'paths': {
        'root': root_path,
        'data': data_path,
        'raw': data_raw_path,
        'actig_extracted': {
            'root': path_actig_extracted,
            'first_batch': path_actig_extracted.joinpath('first_batch'),
            'second_batch': path_actig_extracted.joinpath('second_batch'),
            'third_batch': path_actig_extracted.joinpath('third_batch'),
            'fourth_batch': path_actig_extracted.joinpath('fourth_batch'),
            'remaining': path_actig_extracted.joinpath('DataRemaining'),   # alias
            'dataremaining': path_actig_extracted.joinpath('DataRemaining'),
            'data_only_sleep_rbd': path_actig_extracted.joinpath('DataOnlySleepRBD'),
            'merged': path_actig_extracted.joinpath('merged'),
            'merged_sleep': path_actig_extracted.joinpath('merged', 'F_Sleep_abk_merged.parquet'),
            'rbd_scores': path_actig_extracted.joinpath('merged', 'RBD_Sleep_Score_merged.parquet'),
            'merged_gait': path_actig_extracted.joinpath('merged', 'F_gait_abk_merged.parquet'),
        },
        'data_sheet': {
            'dir_input':  path_raw_data_sheet,
            'dir_out': path_pp_data_sheet,
            'dir_csv': path_pp_data_sheet.joinpath('ukb_final_dataset.csv'),
            'dir_parquet': path_pp_data_sheet.joinpath('ukb_final_dataset.parquet'),
            'formal_name_csv':  path_pp_data_sheet.joinpath('formal_names_cols.csv'),
            'formal_name_json': path_pp_data_sheet.joinpath('formal_names_cols.json'),
            'withdraw_subjects': path_raw_data_sheet.joinpath('withdraw_subjects_ukb_20260310.csv'),
        },
        # 'data_sheet': data_raw_path.joinpath('data_sheet', 'ukb_sliced_data.parquet'),
        'data_sheet_big': data_raw_path.joinpath('data_sheet', 'ukb_full_ehr_dataset.csv'),
        # Upstream actigraphy (MATLAB) feature-extraction output dir; relative default.
        'abk_matlab_out_dir': data_path.joinpath('actig_matlab_out'),
        'genetics': {
            'prs': data_pp_path / 'genetics' / 'UKB_PD_RBD_PRScs_IDmatched_unrelated_eur_project97043.tsv',
            'gba': data_pp_path / 'genetics' / 'UKB_GBA_Gauchian_unrelated_European_matched_IDs_forukb674793.tsv',
        },
    },

    'pp': {
        'root': data_pp_path,
        'ehr_split_flags': data_pp_path.joinpath('ehr_split_flags', 'ehr_split_flags.parquet'),
        'ehr_split_flags_features': data_pp_path.joinpath('ehr_split_flags_features', 'ehr_split_flags_features.parquet'),
        'ehr_split_flags_features_rbd': data_pp_path.joinpath('ehr_split_flags_features', 'ehr_split_flags_features_rbd.parquet'),
        'data_sheet': data_pp_path.joinpath('data_sheet', 'data_sheet.parquet'),
        # 'rbd_pred_diag': data_pp_path.joinpath('ehr_diag_pd_rbd.parquet'),
        'final_dir': data_pp_path.joinpath('res_build_final_dataset'),
        'interim': data_pp_path.joinpath('interim'),
        'thresholds': {
            'root': path_thresholds,
            'collection': 'risk_collection.json',
            'percentile_2g': 'risk_percentile_2g.json',
            'percentile_3g': 'risk_percentile_3g.json',
            'roc': 'risk_roc.json',
            'pr': 'risk_pr.json',
            'f1': 'risk_f1.json',
            'surv': 'risk_surv.json',
            'quartile': 'risk_quartile.json',
        },
        'age_groups': data_pp_path.joinpath('age_groups'),
        'rbd_scores': data_pp_path.joinpath('rbd_scores').joinpath('rbd_scores.parquet'),
        # ''
    },

    'validation': {
        'outcome_pd': data_path.joinpath('validation_outcomes', 'ukbb_5000_outcome1_pd.csv'),
        'outcome_dlb': data_path.joinpath('validation_outcomes', 'ukbb_5000_outcome2_dlb.csv'),
    },

    'train_data': data_path.joinpath('train_data', 'training_data_AX_sleepmodel.csv'),


    # 'data_sheet':  data_pp_path.joinpath('data_sheet', 'data_sheet.parquet'),
    'accele_data_sheet': data_pp_path.joinpath('data_sheet', 'accele_3gp_ukbb.parquet'),
    'accele_columns_definitions': data_pp_path.joinpath('ukbb_field_info.csv'),


    'rar_rbd_models': {
        'script': root_path.joinpath('notebook', 'generate_rbd_predictions_from_features.py'),
        'rar_model': path_project_rar.joinpath('final_model_rar.pkl'),
        'rar_sleep': path_project_rar.joinpath('final_model_sleep.pkl'),
    },


    'results': {
        'root': data_res,
        'logs': data_logs,
        'diagnosis_definition':  data_res.joinpath('diagnosis_definition'),
        'rbd': data_res.joinpath('rbd', 'rbd.parquet'),
        'risk_summary': data_path.joinpath('risk_summary'),
        'thresholds': data_res.joinpath('thresholds'),
    },

}


# Authoritative source: library.column_registry.METHOD_TO_RISK_SUFFIX
from library.column_registry import METHOD_TO_RISK_SUFFIX as method_to_risk_suffix  # noqa: E402

# %% ehr
outcomes = [
    # PD pathways
    "outcome_1a_pd_only",
    "outcome_1b_pd_ad",
    "outcome_2a_vasculardementia",
    "outcome_2b_pd_vasculardementia",
    # "outcome_any_neurodegenerative",
    # Synucleinopathies
    # "outcome_3a_dlb_only",
    # "outcome_3b_msa_only",
    "outcome_4a_ad_only",
    # High-specificity PD: G20 diagnosis + PD medication (Field 20003)
    # "outcome_5a_pd_med",
    # Optional: demyelinating/autoimmune comparator
    # "outcome_4a_ms_only",
]

outcomes_formal_names = {
    "outcome_1a_pd_only": "Parkinson’s disease",
    "outcome_1b_pd_ad": "Parkinson’s disease with Alzheimer’s disease",
    "outcome_2a_vasculardementia": "Vascular dementias",
    "outcome_2b_pd_vasculardementia": "Parkinson’s disease with Vascular dementias",
    # "outcome_3a_dlb_only": "Dementia with Lewy bodies",
    "outcome_4a_ad_only": "Alzheimer’s disease",
    # "outcome_5a_pd_med": "Parkinson’s disease (medication-confirmed)"
}

outcomes_short_names = {
    "outcome_1a_pd_only": "PD",
    "outcome_1b_pd_ad": "PD + AD",
    "outcome_2a_vasculardementia": "Vascular Dementia",
    "outcome_2b_pd_vasculardementia": "PD + Vascular Dementia",
    # "outcome_3a_dlb_only": "DLB",
    "outcome_4a_ad_only": "AD",
}


# Authoritative mapping of disease → UKB date field columns.
# Primary: algo-defined adjudicated fields (p42xxx, coverage ~2024).
# Fallback: first-occurrence ICD-10 fields (p13xxx, coverage ~2025).
# Priority rule: use primary date if not NaT, else use fallback.
# Used by library.ehr_outcomes.outcome_flags.add_outcome_flags.
DISEASE_DATE_COLS: dict = {
    # G20 Parkinson’s disease
    #   primary  p42032 = Date of Parkinson’s disease report (algo-defined)
    #   fallback p131022 = Date of first G20 occurrence (first-occurrence)
    "pd": {"primary": "p42032", "fallback": "p131022"},
    # G30 Alzheimer’s disease
    #   primary  p42020 = Date of Alzheimer’s disease report (algo-defined)
    #   fallback p131036 = Date of first G30 occurrence (first-occurrence)
    "ad": {"primary": "p42020", "fallback": "p131036"},
    # F01 Vascular dementia
    #   primary  p42022 = Date of vascular dementia report (algo-defined)
    #   fallback p130838 = Date of first F01 vascular dementia occurrence
    "dem": {"primary": "p42022", "fallback": "p130838"},
}


# “Non-Alzheimer’s, non-PD dementia was defined using ICD-10 codes F01, F020–F022,
# F024, F028, F03 — excluding AD (G30/F00*), PD dementia (F023), and DLB (G31.8)
# to ensure etiologically disjoint outcome groups.”
# Reference: Wilkinson et al. 2019 (PMC6497624)


# Questionnaire. The 4-item questionnaire screened for:
# (1) dream enactment using the  RBD-Innsbruck summary question19: kicking or hitting during sleep due to defensive dreaming;
# -- https://biobank.ndph.ox.ac.uk/ukb/field.cgi?id=30557
# (2) Hyposmia (anosmia): reduced smell or taste compared to Vasculars or previous ability17;
# (3) Constipation: requiring straining or laxative use20;
# (4) Orthostatic symptoms, possibly related to hypotension 20. Each item allowed responses of No (0), Don’t know (0.5), or Yes (1).
questionnaire_codes = {
    'dream_enactment': ['G4752'],  # ICD10, '30557' field Frequency of 'acting out dreams'
    'constipation': ['K590'],  # ICD10
    'anosmia': ['R430'],  # ICD10
    'hyposmia': ['G520'],  # ICD10
    'Orthostatic': ['I951'],
    'erectile_dysfunction': [
        'N5201',
        'N521',
        'F5221',  # organic ED
        'N529',   # Psychogenic ED
        # not considering N52.3
    ],

    # Mood disorders
    'depression': [
        'F32',  # Depressive episode
        'F33',  # Recurrent depressive disorder
        'F34',  # Persistent mood disorders (incl. dysthymia)
        'F38',  # Vascular mood disorders
        'F39',  # Unspecified mood disorder
    ],

    # Anxiety disorders
    'anxiety': [
        'F40',  # Phobic anxiety disorders
        'F41',  # Vascular anxiety disorders (incl. GAD, panic)
    ],

    # 'executive_function': [],
    # 'reaction_time': [],
    # 'psychomotor_test' : [],
    # 'trail_making_test': []
}



# ICD-10 exclusion list
neuro_exclusion_codes = [
    # Atypical parkinsonism / parkinson-plus
    # -----------------------------------------------------s
    'G21', 'G210', 'G211', 'G212', 'G213', 'G214', 'G218', 'G219',
    'G22', 'G23', 'G230', 'G231', 'G232', 'G233', 'G238', 'G239',
    'G24', 'G250', 'G251', 'G252', 'G253', 'G254', 'G255', 'G256', 'G259',


    # -----------------------------------------------------
    # All-cause dementia (EXCEPT when used as OUTCOME)
    # -----------------------------------------------------
    # These codes should exclude *prevalent dementia only*
    "G31",
    "F051",                 # Delirium superimposed on dementia
    "F106",                 # Alcohol-related dementia
    "I673",                 # Binswanger disease


    # Neurodegenerative diseases
    'G10', 'G11', 'G110', 'G111', 'G112', 'G113', 'G114', 'G118', 'G119',
    'G12', 'G122', 'G1220', 'G1221', 'G1222', 'G1229',
    'G13', 'G130', 'G131', 'G132', 'G138', 'G139',
    'G32', 'G320', 'G321', 'G328', 'G329',

    # Demyelinating diseases
    'G35',
    'G36', 'G360', 'G361', 'G368', 'G369',
    'G37', 'G370', 'G371', 'G372', 'G373', 'G375', 'G378', 'G379',

    # Epilepsy / Seizure
    'G40', 'G400', 'G401', 'G402', 'G403', 'G404', 'G405', 'G406', 'G407', 'G408', 'G409',
    'G41', 'G410', 'G411', 'G412', 'G418', 'G419',
    'R560', 'R568',

    # Encephalitis / Encephalopathy
    'G04', 'G040', 'G041', 'G042', 'G048', 'G049',
    'G05', 'G050', 'G051', 'G052', 'G058', 'G059',
    'G934', 'G938', 'G939',

    # Narcolepsy
    'G474', 'G4740', 'G4741', 'G4742',

    # Multiple System Atrophy (MSA)
    'G232',

    # Multiple Sclerosis (MS)
]


# %%

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

# %% For the RAR RBD model, the predictions need to be done in the minerva server
# %% for minerva

root_path = Path(__file__).resolve().parents[1]

path_data = data_path
path_models = root_path.joinpath('models')
path_project_rar = path_models.joinpath('RAR_pipeline-main')
path_results = root_path.joinpath('results')

config_rbd = {
    'rar_models': {
        'rar_model': path_project_rar.joinpath('final_model_rar.pkl'),
        'rar_sleep': path_project_rar.joinpath('final_model_sleep.pkl'),
    },

}






