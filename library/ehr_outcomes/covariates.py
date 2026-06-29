"""
Universal control definition for outcome-based epidemiological analysis.

Controls must:
    - Have NO outcomes for ANY of the defined outcome categories
    - Have NO PD / AD / Dementia flags
    - Have NO neurological exclusion codes
"""
from typing import List
import numpy as np
import pandas as pd
from config.config import questionnaire_codes
from library.column_registry import col_surv_time
from library.ehr_outcomes.utils import build_ukbb_rename_map
from pathlib import Path
from tabulate import tabulate
from typing import Iterable, Optional, Set
import warnings
import re
from typing import Iterable
from library.ehr_outcomes.utils import report_outcomes_by_flags
from config.config import outcomes

# ------------------------------------------------------------------
# Lifestyle covariate candidates (visit-ordered; earlier visits preferred)
# ------------------------------------------------------------------
_SMOKING_CANDIDATES: list[str] = [
    "cov_smoking_20116_bl",
    "cov_smoking_20116_i1",
    "cov_smoking_20116_fu",
    "cov_smoking_20116_i3",
]

_ALCOHOL_CANDIDATES: list[str] = [
    "cov_alcohol_20117_bl",
    "cov_alcohol_20117_i1",
    "cov_alcohol_20117_fu",
    "cov_alcohol_20117_i3",
]


# ------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------

def add_covariates(
    df: pd.DataFrame,
    save_dir: Path = None,
    verbose: bool = True,
    overwrite:bool = True,
) -> pd.DataFrame:
    """
    Add HES-based covariate flags and first diagnosis dates
    using ICD-10 (41270) and diagnosis dates (41280).
    """

    if verbose:
        print("Processing covariates (HES only)...")

    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    path_file = save_dir.joinpath("4_covariates.parquet")
    if not overwrite and path_file.exists():
        df = pd.read_parquet(path_file)
        return df

    df = df.copy()


    df = derive_reaction_time_metrics(
        df,
        rt_field="p20023",
        round_field="p404",
        pilot_round_field="p10147",
        verbose=True,
        drop_round_level=True,
    )

    # ------------------------------------------------------------------
    # Trial Making covariates
    # ------------------------------------------------------------------
    df = trail_making_covariates(df=df, save_dir=save_dir)

    # ------------------------------------------------------------------
    # Symbol Digital Substitution
    # ------------------------------------------------------------------
    df = symbol_digit_substitution_covariates(df)

    # ------------------------------------------------------------------
    # Prospective Memory (field 6373)
    # ------------------------------------------------------------------
    df = prospective_memory_covariates(df)


    # ------------------------------------------------------------------
    # HES diagnostic covariates (ICD-10)
    # ------------------------------------------------------------------
    df = add_alpha_syn_covariates(
        df=df,
        questionnaire_codes=questionnaire_codes,
        icd10_col="p41270",
        icd10_date_cols=sorted([c for c in df.columns if c.startswith("p41280")]),
    )

    # ------------------------------------------------------------------
    # rename_columns
    # ------------------------------------------------------------------
    code_to_name = {
        # ========== BASELINE DEMOGRAPHICS ==========
        "31": "cov_sex_31",
        "34": "cov_year_birth_34",
        "52": "cov_month_birth_52",
        "21022": "cov_age_recruitment_21022",
        "21000": "cov_ethnicity_21000",

        # ========== RECEPTION ==========
        "53": "cov_recp_date_53",
        "54": "cov_recp_location_54",
        "55": "cov_recp_month_55",
        "20118": "cov_recp_density_20118",
        "21003": "cov_recp_age_21003",

        # ========== SLEEP ==========
        "1160": "cov_sleep_duration_1160",
        "1170": "cov_getting_up_1170",
        "1180": "cov_chronotype_1180",
        "1190": "cov_naps_day_1190",
        "1200": "cov_sleeplessness_insomnia_1200",
        "1210": "cov_snoring_1210",
        "1220": "cov_daytime_dozing_1220",

        # Sleep disturbance questionnaire (online assessment)
        "30544": "cov_sleep_wrong_time_30544",
        "30545": "cov_sleep_late_bedtime_30545",
        "30546": "cov_asleep_normal_waking_30546",
        "30547": "cov_sleep_early_bedtime_30547",
        "30548": "cov_wake_early_alert_30548",
        "30549": "cov_knee_buckle_30549",
        "30550": "cov_jaw_sag_30550",
        "30551": "cov_head_drop_30551",
        "30552": "cov_arm_weakness_30552",
        "30553": "cov_speech_slur_30553",
        "30554": "cov_fall_ground_30554",
        "30555": "cov_sleepwalk_freq_30555",
        "30556": "cov_teeth_grind_freq_30556",
        "30557": "cov_dream_enactment_freq_30557",
        "30558": "cov_violent_sleep_freq_30558",
        "30559": "cov_nightmare_freq_30559",
        "30560": "cov_dream_recall_freq_30560",
        "30561": "cov_seizure_sleep_30561",
        "30562": "cov_snoring_occur_30562",
        "30563": "cov_snoring_volume_30563",
        "30564": "cov_snoring_freq_30564",
        "30565": "cov_snoring_affect_others_30565",
        "30566": "cov_breath_stop_freq_30566",
        "30567": "cov_tired_after_sleep_30567",
        "30568": "cov_tired_daytime_30568",
        "30569": "cov_fall_asleep_driving_30569",
        "30570": "cov_fall_asleep_drive_freq_30570",
        "30571": "cov_motor_accident_sleepy_30571",
        "32121": "cov_sleep_quest_start_32121",
        "32122": "cov_sleep_quest_complete_32122",

        # ========== PHYSICAL ACTIVITY ==========
        # Walking
        "864": "cov_days_walked_864",
        "874": "cov_dur_walks_874",
        "924": "cov_walking_pace_924",
        "971": "cov_freq_walk_pleasure_971",
        "981": "cov_dur_walk_pleasure_981",
        # Moderate activity
        "884": "cov_days_mod_activity_884",
        "894": "cov_dur_mod_activity_894",
        # Vigorous activity
        "904": "cov_days_vig_activity_904",
        "914": "cov_dur_vig_activity_914",
        # Strenuous sports
        "991": "cov_freq_stren_sports_991",
        "1001": "cov_dur_stren_sports_1001",
        # DIY activities
        "1011": "cov_freq_light_diy_1011",
        "1021": "cov_dur_light_diy_1021",
        "2624": "cov_freq_heavy_diy_2624",
        "2634": "cov_dur_heavy_diy_2634",
        # Other exercise
        "3637": "cov_freq_other_exercise_3637",
        "3647": "cov_dur_other_exercise_3647",
        # Stair climbing
        "943": "cov_freq_stair_climbing_943",
        # Sedentary behavior
        "1070": "cov_dur_tv_1070",
        "1080": "cov_dur_computer_1080",
        "1090": "cov_dur_driving_1090",
        "1100": "cov_drive_fast_1100",
        # Types (categorical)
        "6162": "cov_transport_types_6162",
        "6164": "cov_phys_activity_types_6164",

        # ========== EMPLOYMENT (retained for add_shift_worker) ==========
        "826": "cov_shift_826",
        "3426": "cov_night_shift_job_3426",
        "22650": "cov_night_shifts_history_22650",

        # ========== BODY SIZE / COMPOSITION ==========
        "21001": "bmi_21001",
        "21002": "cov_weight_21002",
        "23104": "bmi_imp_23104",
        "23106": "cov_body_impedance_23106",
        "6218": "cov_body_impedance_manual_6218",

        # ========== COGNITIVE ==========
        "20016": "cov_fluid_intelligence_20016",
        "20023": "cov_react_time_mean_20023",
        "20128": "cov_fi_questions_attempted_20128",
        "20240": "cov_numeric_memory_max_20240",
        "20244": "cov_pairs_status_20244",

        # ========== TMT — Online raw source fields ==========
        # Registered for traceability; dropped below after trail_making_covariates()
        # produces the named tmt_* columns consumed by select_tmt_baseline().
        "20156": "tmt1_dur_raw_online_20156",     # TMT-A duration [seconds]
        "20157": "tmt2_dur_raw_online_20157",     # TMT-B duration [seconds]
        "20247": "tmt1_err_raw_online_20247",     # TMT-A errors
        "20248": "tmt2_err_raw_online_20248",     # TMT-B errors
        "20246": "tmt_complete_raw_online_20246", # Completion status (0 = both done)
        "20136": "tmt_date_raw_online_20136",     # Assessment date

        # ========== ACCELEROMETER CALIBRATION / QUALITY ==========
        "90051": "acc_wear_duration_90051",
        "90052": "acc_non_wear_duration_90052",
        "90170": "acc_calib_mean_temp_90170",
        "90192": "acc_temp_mean_90192",
        "90193": "acc_temp_std_90193",
        "90194": "acc_temp_min_90194",
        "90195": "acc_temp_max_90195",

        # ========== LFE STYLE ==========
        "20117": 'cov_alcohol_20117',
        '20116': 'cov_smoking_20116',

        # ========== DERIVED ACCELEROMETRY ==========
        # Day hour averages
        "40030": "acc_sleep_day_hr_avg_40030",
        "40031": "acc_sed_day_hr_avg_40031",
        "40032": "acc_light_day_hr_avg_40032",
        "40033": "acc_mvpa_day_hr_avg_40033",
        # Weekday hour averages
        "40034": "acc_sleep_weekday_hr_avg_40034",
        "40035": "acc_sed_weekday_hr_avg_40035",
        "40036": "acc_light_weekday_hr_avg_40036",
        "40037": "acc_mvpa_weekday_hr_avg_40037",
        # Weekend hour averages
        "40038": "acc_sleep_weekend_hr_avg_40038",
        "40039": "acc_sed_weekend_hr_avg_40039",
        "40040": "acc_light_weekend_hr_avg_40040",
        "40041": "acc_mvpa_weekend_hr_avg_40041",
        # Day averages
        "40042": "acc_sleep_day_avg_40042",
        "40043": "acc_sed_day_avg_40043",
        "40044": "acc_light_day_avg_40044",
        "40045": "acc_mvpa_day_avg_40045",
        # Overall averages
        "40046": "acc_sleep_overall_40046",
        "40047": "acc_sed_overall_40047",
        "40048": "acc_light_overall_40048",
        "40049": "acc_mvpa_overall_40049",
        "90012": "acc_overall_acc_avg",
        # Day-of-week acceleration averages
        "90019": "acc_monday_avg_90019",
        "90020": "acc_tuesday_avg_90020",
        "90021": "acc_wednesday_avg_90021",
        "90022": "acc_thursday_avg_90022",
        "90023": "acc_friday_avg_90023",
        "90024": "acc_saturday_avg_90024",
        "90025": "acc_sunday_avg_90025",
        # Acceleration SD (overall wear period)
        "90013": "acc_overall_std_90013",
        # Hourly acceleration averages (00:00–23:59)
        "90027": "acc_hour_00_90027",
        "90028": "acc_hour_01_90028",
        "90029": "acc_hour_02_90029",
        "90030": "acc_hour_03_90030",
        "90031": "acc_hour_04_90031",
        "90032": "acc_hour_05_90032",
        "90033": "acc_hour_06_90033",
        "90034": "acc_hour_07_90034",
        "90035": "acc_hour_08_90035",
        "90036": "acc_hour_09_90036",
        "90037": "acc_hour_10_90037",
        "90038": "acc_hour_11_90038",
        "90039": "acc_hour_12_90039",
        "90040": "acc_hour_13_90040",
        "90041": "acc_hour_14_90041",
        "90042": "acc_hour_15_90042",
        "90043": "acc_hour_16_90043",
        "90044": "acc_hour_17_90044",
        "90045": "acc_hour_18_90045",
        "90046": "acc_hour_19_90046",
        "90047": "acc_hour_20_90047",
        "90048": "acc_hour_21_90048",
        "90049": "acc_hour_22_90049",
        "90050": "acc_hour_23_90050",
        # No-wear-time bias-adjusted summary statistics
        "90087": "acc_nowear_avg_90087",
        "90088": "acc_nowear_std_90088",
        "90089": "acc_nowear_median_90089",
        "90090": "acc_nowear_min_90090",
        "90091": "acc_nowear_max_90091",

        # ========== GP PRESCRIPTIONS (IRS) ==========
        "42039": "cov_gp_prescriptions_irs_42039",

    }

    # Instance-suffix convention: recruitment visit (i0) -> baseline (_bl),
    # imaging visit (i2) -> follow-up (_fu).  i1/i3 retain numeric notation.
    # Applied at the source rename so all persistent parquet columns are
    # consistently named; downstream code references _bl / _fu.
    INSTANCE_SUFFIX_MAP: dict[str, str] = {"_i0": "_bl", "_i2": "_fu"}
    columns_formalizer,df_columns_formalizer  = build_ukbb_rename_map(
        columns=[*df.columns],
        code_to_name=code_to_name,
        suffix_map=INSTANCE_SUFFIX_MAP,
    )
    df_columns_formalizer.to_csv(save_dir.joinpath('cov_table_renames.csv'), index=False)

    # rename
    df.rename(columns=columns_formalizer, inplace=True)

    # ------------------------------------------------------------------
    # DROP UNUSED COLUMNS
    # ------------------------------------------------------------------
    # These columns have been processed into aggregates or are not needed
    
    cols_to_drop = []
    
    # 1. ICD-10/ICD-9 columns (already used for outcome flags)
    cols_to_drop += [col for col in df.columns if col.startswith(("p41270",
                                                                  "p41280",
                                                                  "p41201",
                                                                  "p41202",
                                                                  "p41203",
                                                                  "p41204",
                                                                  "p41205",
                                                                  "p41262",
                                                                  "p41263",
                                                                  "p41271",
                                                                  "p41281"))]
    
    # 2. Raw accelerometer columns (keep only the renamed ones)
    #    - p90002: Data problem indicator
    #    - p90010: Start time (we have wear_time_start)
    #    - p90012: kept → acc_overall_acc_avg (in code_to_name)
    #    - p90013: kept → acc_overall_std_90013 (in code_to_name)
    #    - p90015-p90018: quality/calibration flags (used for exclusions, then dropped)
    #    - p90019-p90025: kept → acc_{day}_avg_* (in code_to_name)
    #    - p90026: Daily acceleration average (redundant with overall)
    #    - p90027-p90050: kept → acc_hour_XX_* (in code_to_name)
    #    - p90053-p90086: Wear duration by day/hour (too granular)
    #    - p90087-p90091: kept → acc_nowear_* (in code_to_name)
    #    - p90092-p90158: Fraction acceleration distributions (too granular)
    #    - p90159-p90177: Calibration coefficients (too granular)
    #    - p90179-p90191: Device stats (not needed)
    raw_acc_prefixes = (
        "p90002", "p90010",
        "p90015", "p90016", "p90017", "p90018",
        "p90026",
        "p90053", "p90054", "p90055", "p90056", "p90057", "p90058", "p90059",
        "p90060", "p90061", "p90062", "p90063", "p90064", "p90065", "p90066", "p90067",
        "p90068", "p90069", "p90070", "p90071", "p90072", "p90073", "p90074", "p90075",
        "p90076", "p90077", "p90078", "p90079", "p90080", "p90081", "p90082", "p90083",
        "p90084", "p90085", "p90086",
        "p90092", "p90093", "p90094", "p90095", "p90096", "p90097", "p90098", "p90099",
        "p90100", "p90101", "p90102", "p90103", "p90104", "p90105", "p90106", "p90107",
        "p90108", "p90109", "p90110", "p90111", "p90112", "p90113", "p90114", "p90115",
        "p90116", "p90117", "p90118", "p90119", "p90120", "p90121", "p90122", "p90123",
        "p90124", "p90125", "p90126", "p90127", "p90128", "p90129", "p90130", "p90131",
        "p90132", "p90133", "p90134", "p90135", "p90136", "p90137", "p90138", "p90139",
        "p90140", "p90141", "p90142", "p90143", "p90144", "p90145", "p90146", "p90147",
        "p90148", "p90149", "p90150", "p90151", "p90152", "p90153", "p90154", "p90155",
        "p90156", "p90157", "p90158", "p90159", "p90160", "p90161", "p90162", "p90163",
        "p90164", "p90165", "p90166", "p90167", "p90168", "p90169", "p90171",
        "p90172", "p90173", "p90174", "p90175", "p90176", "p90177", "p90179", "p90180",
        "p90181", "p90182", "p90183", "p90184", "p90185", "p90186", "p90187", "p90188",
        "p90189", "p90190", "p90191",
    )
    cols_to_drop += [col for col in df.columns if col.startswith(raw_acc_prefixes)]
    
    # 3. Fluid Intelligence individual items (have aggregate cov_fluid_intelligence_20016)
    #    p4924-p5867: FI1-FI13 individual test items and durations
    fi_prefixes = (
        "p4924", "p4935", "p4936", "p4946", "p4947", "p4957", "p4958", "p4968", "p4969",
        "p4979", "p4980", "p4990", "p4991", "p5001", "p5002", "p5012", "p5013",
        "p5556", "p5557", "p5699", "p5700", "p5779", "p5780", "p5790", "p5791",
        "p5866", "p5867",
    )
    cols_to_drop += [col for col in df.columns if col.startswith(fi_prefixes)]
    # Also drop misnamed FI columns (5556/5557 were incorrectly mapped to reception)
    cols_to_drop += [col for col in df.columns if "5556" in col or "5557" in col]
    
    # 4. Reaction time round-level data (have aggregates cov_rt_*)
    #    p403: Duration to first press in each round (main)
    #    p10141: Duration to first press in each round (pilot)
    cols_to_drop += [col for col in df.columns if col.startswith(("p403_", "p10141_"))]
    
    # 5. Pilot physical activity (redundant with main measures)
    cols_to_drop += [col for col in df.columns if col.startswith(("p10953", "p10962", "p10971"))]
    
    # 6. Trail making raw source fields (processed into tmt_* by trail_making_covariates)
    #    p6770-p6773: Per-step errors/intervals
    #    p6348-p6351: Clinic raw duration/errors (deciseconds)
    #    p20156/20157/20247/20248/20246/20136: Online raw fields (seconds)
    cols_to_drop += [col for col in df.columns if col.startswith((
        "p6770", "p6771", "p6772", "p6773",
        "p6348", "p6349", "p6350", "p6351",
        "p20156", "p20157", "p20247", "p20248", "p20246", "p20136",
    ))]
    
    # 7. Numeric memory timestamp (not needed, have max digits)
    cols_to_drop += [col for col in df.columns if col.startswith("p20138")]
    
    # Deduplicate and drop
    cols_to_drop = list(set(cols_to_drop))
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    if verbose:
        print(f"  Dropping {len(cols_to_drop)} unused columns...")
    
    df.drop(columns=cols_to_drop, inplace=True)

    if verbose:
        print(f"  Remaining columns: {len(df.columns)}")

    # ------------------------------------------------------------------
    # TMT baseline selection
    # Must run after trail_making_covariates() (per-instance tmt columns)
    # and after rename (wear_time_start is already present from filtering).
    # ------------------------------------------------------------------
    if verbose:
        print("\n  Selecting TMT baseline instance...")
    df = select_tmt_baseline(df, wear_col="wear_time_start")
    # Baseline TMT log-ratio (named per _bl convention; consumed by
    # _TMT_ALIASES and the cognitive _bl block in add_cognitive_latest_per_subject).
    df["cog_tmt_ratio_log_bl"] = np.log(df["tmt_ratio_baseline"])

    # ------------------------------------------------------------------
    # Lifestyle covariates: smoking and alcohol
    # Must run after rename (needs cov_smoking_20116_bl etc.).
    # ------------------------------------------------------------------
    if verbose:
        print("\n  Building lifestyle covariates (smoking, alcohol)...")
    df = prepare_lifestyle_covariates(df)

    # ------------------------------------------------------------------
    # BMI consolidation
    # Combines bmi_21001 and bmi_imp_23104 across instances to fill missing
    # ------------------------------------------------------------------
    if verbose:
        print("\n  Consolidating BMI measures...")
    df = combine_bmi_measures(df, verbose=verbose)

    # ------------------------------------------------------------------
    # BMI imputation (k-NN hot-deck for subjects missing in all visits)
    # ------------------------------------------------------------------
    if verbose:
        print("\n  Imputing remaining missing BMI via k-NN on controls...")
    df = impute_bmi_knn(df, k=5, verbose=verbose)

    # ------------------------------------------------------------------
    # Genetics: PRS scores and GBA carrier status
    # ------------------------------------------------------------------
    from config.config import config as _config
    _genetics = _config.get("paths", {}).get("genetics", {})

    _prs_path = _genetics.get("prs")
    if _prs_path is not None:
        df = merge_prs_scores(df=df, prs_path=_prs_path, verbose=verbose)
    elif verbose:
        print("  [SKIP] PRS scores: config['paths']['genetics']['prs'] not found.")

    _gba_path = _genetics.get("gba")
    if _gba_path is not None:
        df = merge_gba_carrier(df=df, gba_path=_gba_path, verbose=verbose)
    elif verbose:
        print("  [SKIP] GBA: config['paths']['genetics']['gba'] not found.")

    # ------------------------------------------------------------------
    # Optional save
    # ------------------------------------------------------------------

    if verbose:
        print("Done.")

    covariates = list(questionnaire_codes.keys())


    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        # visit=0
        # report = report_outcomes_by_flags(df=df,
        #                                   outcomes=outcomes,
        #                                   verbose=verbose,
        #                                   flags=[],
        #                                   output_path= save_dir / "log_covariates.csv" if save_dir else None)

    return df


def merge_prs_scores(
    df: pd.DataFrame,
    prs_path: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """Merge polygenic risk scores (PRS) for PD and RBD into the main cohort.

    Columns added:
        prs_score_pd  — PD PRS z-score (``PD_PRS_zscore``)
        prs_score_rbd — RBD PRS z-score (``RBD_PRS_zscore``)
        prs_pc1 … prs_pc10 — Genetic ancestry principal components

    The merge is a *left join* on ``eid`` so all cohort subjects are
    retained; subjects without genetic data receive NaN for PRS columns.

    Args:
        df: Main cohort DataFrame with an ``eid`` column.
        prs_path: Absolute path to the TSV file with columns
            ``IID, PD_PRS_zscore, RBD_PRS_zscore, pc1..pc10``.
        verbose: Print merge diagnostics when True.

    Returns:
        DataFrame with PRS columns appended.
    """
    if verbose:
        print(f"  Merging PRS scores from: {prs_path}")

    if not Path(prs_path).exists():
        raise FileNotFoundError(f"PRS score file not found: {prs_path}")

    pc_cols = [f"pc{k}" for k in range(1, 11)]
    prs_df = pd.read_csv(
        prs_path,
        sep="\t",
        usecols=["IID", "PD_PRS_zscore", "RBD_PRS_zscore"] + pc_cols,
        dtype={"IID": np.int64},
    )

    rename_map: dict[str, str] = {
        "IID": "eid",
        "PD_PRS_zscore": "prs_score_pd",
        "RBD_PRS_zscore": "prs_score_rbd",
        **{f"pc{k}": f"prs_pc{k}" for k in range(1, 11)},
    }
    prs_df = prs_df.rename(columns=rename_map)

    # Coerce eid type to match main df before merge
    prs_df["eid"] = prs_df["eid"].astype(df["eid"].dtype)

    n_before = len(df)
    df = df.merge(prs_df, on="eid", how="left")
    n_matched = df["prs_score_pd"].notna().sum()

    if verbose:
        print(
            f"  PRS merge: {n_matched:,}/{n_before:,} subjects matched "
            f"({n_before - n_matched:,} without genetic data -> NaN)"
        )

    return df


def merge_gba_carrier(
    df: pd.DataFrame,
    gba_path: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """Merge GBA/GBAP1 carrier status from Gauchian pipeline output.

    Columns added:
        gba_carrier         — is_carrier flag (0/1) for GBAP1-like variant exon9-11
        gba_biallelic       — is_biallelic flag (0/1; NaN when not called)
        gba_cn              — total copy number GBA+GBAP1 (float32; NaN when not called)
        gba_deletion_in_gba — deletion breakpoint in GBA gene (0/1; NaN when no deletion)
        gba_allele          — GBAP1-like alleles called in exon9-11 (string)

    Left join on ``eid``; subjects without genetic data receive NaN.

    Args:
        df: Main cohort DataFrame with an ``eid`` column.
        gba_path: Path to the comma-separated GBA TSV (Gauchian output).
        verbose: Print merge diagnostics when True.

    Returns:
        DataFrame with GBA columns appended.
    """
    if verbose:
        print(f"  Merging GBA carrier data from: {gba_path}")

    if not Path(gba_path).exists():
        raise FileNotFoundError(f"GBA data file not found: {gba_path}")

    gba_df = pd.read_csv(gba_path, sep=",", dtype=str)

    rename_map: dict[str, str] = {
        "ID": "eid",
        "is_carrier(GBAP1-like_variant_exon9-11)": "gba_carrier",
        "is_biallelic(GBAP1-like_variant_exon9-11)": "gba_biallelic",
        "CN(GBA+GBAP1)": "gba_cn",
        "deletion_breakpoint_in_GBA": "gba_deletion_in_gba",
        "GBAP1-like_variant_exon9-11": "gba_allele",
    }
    gba_df = gba_df.rename(columns=rename_map).drop(columns=["Sample"], errors="ignore")

    for bool_col in ("gba_carrier", "gba_biallelic", "gba_deletion_in_gba"):
        if bool_col in gba_df.columns:
            gba_df[bool_col] = (
                gba_df[bool_col]
                .map({"True": 1, "False": 0})
                .astype("Int8")
            )

    gba_df["gba_cn"] = pd.to_numeric(gba_df["gba_cn"], errors="coerce").astype("float32")
    gba_df["eid"] = pd.to_numeric(gba_df["eid"], errors="coerce").astype(np.int64)
    gba_df["eid"] = gba_df["eid"].astype(df["eid"].dtype)

    cols_to_merge = ["eid", "gba_carrier", "gba_biallelic", "gba_cn",
                     "gba_deletion_in_gba", "gba_allele"]
    n_before = len(df)
    df = df.merge(gba_df[cols_to_merge], on="eid", how="left")

    n_genotyped = df["gba_carrier"].notna().sum()
    n_carriers = (df["gba_carrier"] == 1).sum()

    if verbose:
        pct_carriers = 100.0 * n_carriers / n_genotyped if n_genotyped > 0 else float("nan")
        print(
            f"  GBA merge: {n_genotyped:,}/{n_before:,} subjects with genetic data "
            f"({n_before - n_genotyped:,} without -> NaN)\n"
            f"  GBA carriers (is_carrier=1): {n_carriers:,}/{n_genotyped:,} "
            f"({pct_carriers:.1f}% of genotyped)"
        )

    return df


def add_alpha_syn_covariates(
        df: pd.DataFrame,
        questionnaire_codes: dict,
        icd10_col: str = "p41270",
        icd10_date_cols: Iterable[str] = (),
        verbose: bool = True,
        baseline_col: str = "wear_time_start",
) -> pd.DataFrame:
    """
    Add alpha-synucleinopathy-related HES covariates:
    - earliest diagnosis date
    - binary flag (ever diagnosed)
    - HES activity gap (computed from p41280 before columns are dropped)
    """

    def _earliest_hes_date(
            row: pd.Series,
            codes: Iterable[str],
            icd10_col: str = "p41270",
            icd10_date_cols: Iterable[str] = (),
    ) -> Optional[pd.Timestamp]:
        """
        Return earliest HES diagnosis date for any ICD-10 code in `codes`.

        Uses UKBB array alignment between p41270 and p41280_aX columns.
        """
        if pd.isna(row.get(icd10_col)):
            return pd.NaT

        codes_set: Set[str] = set(codes)
        icd_codes = str(row[icd10_col]).split("|")

        dates = []
        for idx, code in enumerate(icd_codes):
            if code in codes_set:
                date_col = f"p41280_a{idx}"
                if date_col in icd10_date_cols:
                    d = row.get(date_col)
                    if pd.notna(d):
                        dates.append(pd.to_datetime(d, errors="coerce"))

        if not dates:
            return pd.NaT

        return min(dates)


    for covar, codes in questionnaire_codes.items():
        if verbose:
            print(f"-> Adding covariate: {covar}")

        date_col = f"{covar}_hes_date"
        flag_col = f"{covar}_hes"

        df[date_col] = df.apply(
            _earliest_hes_date,
            axis=1,
            codes=codes,
            icd10_col=icd10_col,
            icd10_date_cols=icd10_date_cols,
        )

        df[flag_col] = df[date_col].notna().astype(int)

        if verbose:
            print(f"   {flag_col}: {df[flag_col].sum():,} cases")

    # Compute HES activity gap while p41280 columns are still available
    # (they will be dropped later in add_covariates after this function returns)
    df = compute_hes_activity_gap(
        df,
        baseline_col=baseline_col,
        hes_date_prefix="p41280"
    )

    return df


# ---------------------------------------------------------------------------
# HES activity gap: per-subject measurement reliability indicator
# ---------------------------------------------------------------------------

def compute_hes_activity_gap(
    df: pd.DataFrame,
    baseline_col: str = "wear_time_start",
    hes_date_prefix: str = "p41280",
) -> pd.DataFrame:
    """
    Compute per-subject HES activity gap for prodromal marker reliability.

    For each subject, identifies the most recent HES record date that predates
    actigraphy baseline, then computes the gap between that date and
    ``wear_time_start``.  A large gap means the subject had limited hospital
    contact before actigraphy, so a "no constipation/depression/etc." label
    based on HES absence is less verifiable.

    Epidemiological use
    -------------------
    Subjects with ``hes_gap_pre_baseline_years > threshold`` (typically 4 years)
    have a pre-baseline window that is not well covered by hospital records.
    These subjects are NOT excluded from the primary analysis — HES-unexposed is
    a valid exposure category — but they are flagged so that sensitivity analyses
    can restrict to subjects with adequate HES coverage and verify that results
    are not driven by misclassified unexposed individuals.

    NaN gap (``hes_last_pre_baseline_date`` = NaT) means the subject has NO
    pre-baseline HES record at all.  These subjects are treated as having the
    maximum possible uncertainty and will be excluded from the HES-active
    sensitivity analysis regardless of threshold.

    Args:
        df: DataFrame containing raw HES date columns (p41280_a*) and the
            baseline column.  Must have ``baseline_col`` as datetime.
        baseline_col: Column name of the actigraphy wear start date.
        hes_date_prefix: Prefix identifying HES diagnosis date columns
            (default ``"p41280"``).

    Returns:
        DataFrame with two new columns:
            - ``hes_last_pre_baseline_date`` — most recent pre-baseline HES date
            - ``hes_gap_pre_baseline_years`` — gap in years (NaN if no record)
    """
    baseline: pd.Series = pd.to_datetime(df[baseline_col], errors="coerce")

    hes_date_cols = [c for c in df.columns if hes_date_prefix in c]

    if not hes_date_cols:
        warnings.warn(
            f"No HES date columns matching prefix '{hes_date_prefix}' found. "
            "hes_gap_pre_baseline_years will be NaN for all subjects."
        )
        df = df.copy()
        df["hes_last_pre_baseline_date"] = pd.NaT
        df["hes_gap_pre_baseline_years"] = np.nan
        return df

    # Stack all HES date columns into a single DataFrame, then mask any date
    # that falls on or after baseline to NaT.  Row-wise max gives the most
    # recent confirmed pre-baseline hospital contact.
    # Memory note: p41280_a* can have 200+ columns; this is a one-time step.
    all_dates = pd.concat(
        [pd.to_datetime(df[c], errors="coerce") for c in hes_date_cols],
        axis=1,
    )
    # Boolean mask: True where date < baseline (broadcast scalar baseline_col)
    pre_mask = all_dates.lt(baseline.values[:, None])
    all_dates_pre = all_dates.where(pre_mask, other=pd.NaT)

    last_pre = all_dates_pre.max(axis=1)   # NaT where no pre-baseline record

    df = df.copy()
    df["hes_last_pre_baseline_date"] = last_pre
    df["hes_gap_pre_baseline_years"] = (
        (baseline - last_pre).dt.days / 365.25
    )

    if df["hes_gap_pre_baseline_years"].notna().any():
        gap_stats = df["hes_gap_pre_baseline_years"].describe()
        pct_no_record = round(100 * df["hes_gap_pre_baseline_years"].isna().mean(), 1)
        print(
            f"  [HES gap] median={gap_stats['50%']:.1f}y  "
            f"max={gap_stats['max']:.1f}y  "
            f"no pre-baseline record: {pct_no_record}%"
        )

    return df


# ---------------------------------------------------------------------------
# Prodromal marker mapping: HES covariates × medication families
# ---------------------------------------------------------------------------

PRODROMAL_MAP: dict[str, tuple[str, str | None]] = {
    # marker_name:         (hes_flag_col,                 med_flag_col or None)
    "constipation":         ("constipation_hes",           "med_laxatives"),
    "depression":           ("depression_hes",             "med_depression"),
    "anxiety":              ("anxiety_hes",                "med_anxiety"),
    "orthostatic":          ("Orthostatic_hes",            "med_orthostatic_hypotension"),
    "erectile_dysfunction": ("erectile_dysfunction_hes",    "med_pde5_inhibitors"),
    "dream_enactment":      ("dream_enactment_hes",        None),
    "anosmia":              ("anosmia_hes",                 None),
    "hyposmia":             ("hyposmia_hes",                None),
}


def merge_prodromal_markers(
    df: pd.DataFrame,
    baseline_col: str = "wear_time_start",
    surv_days_col: Optional[str] = None,
    post_window_start_days: int = 182,
    verbose: bool = True,
    save_dir: Path = None,
) -> pd.DataFrame:
    """
    Merge HES diagnosis flags with medication flags into prodromal markers,
    in two temporal passes anchored on the actigraphy baseline.

    Pass 1 — baseline (``_bl``): pre-baseline prevalence
    ----------------------------------------------------
    Prodromal markers are, by definition, symptoms that precede disease onset.
    Including post-baseline diagnoses (conditions that emerged *after*
    actigraphy) would conflate true prodromal exposure with reverse causation:
    early PD pathology could itself cause the symptom (e.g. constipation driven
    by enteric neurodegeneration) during follow-up, making it appear associated
    with PD when it is in fact a consequence of it.  Restricting to pre-baseline
    evidence ensures ``prodromal_{marker}_bl = 1`` reflects a condition
    *prevalent at the time of measurement*.  Prevalent prodromal cases are the
    intended exposed group and are NOT excluded.

    Pass 2 — post-baseline incident (``_post``)
    -------------------------------------------
    New-onset markers arising during follow-up, for descriptive
    characterisation of incident prodromal burden.  An event counts as incident
    post-baseline when its earliest HES or medication date falls in the window
    ``(baseline + post_window_start_days, baseline + surv_days]`` AND the subject
    had no pre-baseline evidence (``_bl == 0``).  The window closes at each
    subject's PD-only survival time (event or censoring) so that markers arising
    after PD diagnosis are not counted (reverse-causation guard).

    Prevalent-PD subjects (``surv_days`` is NaN) have no valid incident
    observation window; their ``_post`` flags and ``prodromal_burden_post`` are
    set to **NaN** (not 0) to remain distinguishable from true negatives and to
    be excluded from post-baseline burden analyses.

    Missing dates
    -------------
    If a record flag is True but its date is NaT (date column missing or
    unparseable), the evidence is treated as unverifiable and conservatively set
    to False for that subject in both passes.

    For each entry in PRODROMAL_MAP, creates:
      - ``prodromal_{marker}_bl``        — int (1 = pre-baseline HES/med evidence)
      - ``prodromal_{marker}_bl_date``   — Timestamp (earliest pre-baseline evidence)
      - ``prodromal_{marker}_bl_source`` — str: ``"hes"`` | ``"med"`` | ``"both"`` | NaN
      - ``prodromal_{marker}_post``      — float (1/0; NaN for prevalent-PD subjects)
      - ``prodromal_{marker}_post_date`` — Timestamp (earliest post-baseline evidence)
    plus one aggregate:
      - ``prodromal_burden_post``        — count of incident post-baseline markers
        (NaN for prevalent-PD subjects)

    Original ``{covar}_hes*`` and ``med_{family}*`` columns are preserved.

    Args:
        df: DataFrame with HES covariate and medication flag columns already added.
            Must contain ``baseline_col`` and ``surv_days_col``.
        baseline_col: Actigraphy wear start date column (default ``"wear_time_start"``).
        surv_days_col: Survival-time column (days from baseline) used as the
            post-baseline window end.  Defaults to the PD-only survival column
            via ``col_surv_time("outcome_1a_pd_only")``.
        post_window_start_days: Days after baseline at which the post-baseline
            window opens (default 182 ≈ 6 months).
        verbose: Print per-marker summary table.
        save_dir: Optional directory to save the log as an Excel file.

    Returns:
        DataFrame with added ``prodromal_*_bl`` and ``prodromal_*_post`` columns.
    """
    if baseline_col not in df.columns:
        raise KeyError(
            f"Baseline column '{baseline_col}' not found in DataFrame. "
            "Ensure actigraphy filtering runs before prodromal marker merging."
        )

    if surv_days_col is None:
        surv_days_col = col_surv_time("outcome_1a_pd_only")
    if surv_days_col not in df.columns:
        raise KeyError(
            f"Survival-time column '{surv_days_col}' not found in DataFrame. "
            "Ensure add_outcome_flags() runs before merge_prodromal_markers() so "
            "the post-baseline window end is available."
        )

    baseline: pd.Series = pd.to_datetime(df[baseline_col], errors="coerce")

    # HES activity gap is already computed in add_alpha_syn_covariates()
    # before p41280 columns are dropped. Check that it exists.
    if "hes_gap_pre_baseline_years" not in df.columns:
        raise ValueError(
            "hes_gap_pre_baseline_years column not found. "
            "Ensure add_alpha_syn_covariates() has been called (via add_covariates()) "
            "before merge_prodromal_markers()."
        )
    df = df.copy()
    log_rows: list[dict] = []

    # ── Post-baseline window bounds (per subject) ─────────────────────────────
    # surv_days is in days from baseline; NaN for prevalent-PD subjects, who
    # therefore have no valid incident window (post flags set to NaN below).
    surv_days: pd.Series = pd.to_numeric(df[surv_days_col], errors="coerce")
    valid_post: pd.Series = surv_days.notna()
    post_start: pd.Series = baseline + pd.Timedelta(days=post_window_start_days)
    post_end: pd.Series = baseline + pd.to_timedelta(surv_days, unit="D")  # NaT where surv NaN

    for marker, (hes_col, med_col) in PRODROMAL_MAP.items():
        hes_date_col = f"{hes_col}_date"
        out_flag = f"prodromal_{marker}_bl"
        out_date = f"prodromal_{marker}_bl_date"
        out_src = f"prodromal_{marker}_bl_source"
        out_flag_post = f"prodromal_{marker}_post"
        out_date_post = f"prodromal_{marker}_post_date"

        # ── HES evidence ──────────────────────────────────────────────────
        # hes_flag_raw: any HES record exists (regardless of timing)
        # hes_date:     earliest ICD-10 date for this condition
        # hes_flag_pre: record exists AND predates actigraphy baseline
        #
        # Subjects with hes_flag_raw=True but hes_date=NaT cannot be
        # verified as pre-baseline and are conservatively reclassified to 0.
        if hes_col in df.columns:
            hes_flag_raw = df[hes_col].astype(bool)
        else:
            hes_flag_raw = pd.Series(False, index=df.index)

        if hes_date_col in df.columns:
            hes_date = pd.to_datetime(df[hes_date_col], errors="coerce")
        else:
            hes_date = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

        # Restrict to pre-baseline: date must be known and < wear_time_start
        hes_pre_baseline: pd.Series = hes_date < baseline   # NaT → False
        hes_flag = hes_flag_raw & hes_pre_baseline

        n_hes_post = int((hes_flag_raw & ~hes_pre_baseline).sum())

        # ── Medication evidence ───────────────────────────────────────────
        # Same logic: restrict to pre-baseline prescriptions.
        # If the medication date column is absent, dates are all NaT and all
        # medication flags are conservatively excluded (logged as missing date).
        if med_col is not None and med_col in df.columns:
            med_flag_raw = df[med_col].astype(bool)
            med_date_col = f"{med_col}_date"
            if med_date_col in df.columns:
                med_date = pd.to_datetime(df[med_date_col], errors="coerce")
            else:
                med_date = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
                print(
                    f"  Warning [{marker}]: medication date column '{med_date_col}' "
                    "not found — all medication evidence excluded (cannot verify "
                    "pre-baseline timing)."
                )
        else:
            med_flag_raw = pd.Series(False, index=df.index)
            med_date = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

        med_pre_baseline: pd.Series = med_date < baseline   # NaT → False
        med_flag = med_flag_raw & med_pre_baseline

        n_med_post = int((med_flag_raw & ~med_pre_baseline).sum())

        # ── PASS 1: baseline flag (pre-baseline HES OR pre-baseline med) ──
        bl_flag_bool: pd.Series = hes_flag | med_flag
        df[out_flag] = bl_flag_bool.astype(int)

        # ── Earliest pre-baseline evidence date ───────────────────────────
        # Use NaT for any source that did not contribute pre-baseline evidence.
        hes_date_pre = hes_date.where(hes_flag, pd.NaT)
        med_date_pre = med_date.where(med_flag, pd.NaT)
        df[out_date] = pd.concat([hes_date_pre, med_date_pre], axis=1).min(axis=1)

        # ── Source attribution ────────────────────────────────────────────
        src = pd.Series(pd.NA, index=df.index, dtype="string")
        src[hes_flag & ~med_flag] = "hes"
        src[~hes_flag & med_flag] = "med"
        src[hes_flag & med_flag] = "both"
        df[out_src] = src

        # ── PASS 2: incident post-baseline marker ─────────────────────────
        # Evidence dated strictly after post_start and on/before post_end
        # (subject's PD-only survival/censoring date), with no pre-baseline
        # evidence (incident only).  NaT dates → False via the comparisons.
        hes_post: pd.Series = (hes_date > post_start) & (hes_date <= post_end)
        med_post: pd.Series = (med_date > post_start) & (med_date <= post_end)
        incident_post: pd.Series = (~bl_flag_bool) & (hes_post | med_post)

        # Float flag: 1/0 for subjects with a valid window; NaN for prevalent-PD
        # subjects (surv_days NaN) so they stay distinct from true negatives.
        df[out_flag_post] = incident_post.astype("float64").where(valid_post, np.nan)

        # Earliest qualifying post-baseline evidence date (NaT otherwise).
        hes_date_post = hes_date.where(hes_post, pd.NaT)
        med_date_post = med_date.where(med_post, pd.NaT)
        post_date = pd.concat([hes_date_post, med_date_post], axis=1).min(axis=1)
        df[out_date_post] = post_date.where(incident_post & valid_post, pd.NaT)

        # ── Logging ───────────────────────────────────────────────────────
        n_hes_only = int((hes_flag & ~med_flag).sum())
        n_med_only = int((~hes_flag & med_flag).sum())
        n_both = int((hes_flag & med_flag).sum())
        n_total = int(df[out_flag].sum())
        n_post = int((incident_post & valid_post).sum())

        log_rows.append({
            "marker": marker,
            "hes_only": n_hes_only,
            "med_only": n_med_only,
            "both": n_both,
            "total_pre_baseline": n_total,
            "excluded_post_baseline_hes": n_hes_post,
            "excluded_post_baseline_med": n_med_post,
            "incident_post_baseline": n_post,
        })

    # ── Aggregate: incident post-baseline prodromal burden ────────────────
    # Row-sum of the 8 _post flags.  skipna=False ⇒ NaN whenever any _post is
    # NaN; since all _post columns are NaN together (prevalent-PD subjects) or
    # all present, this yields NaN exactly for prevalent-PD subjects.
    post_cols = [f"prodromal_{m}_post" for m in PRODROMAL_MAP]
    df["prodromal_burden_post"] = df[post_cols].sum(axis=1, skipna=False)

    df_log = pd.DataFrame(log_rows)
    if verbose:
        n_prevalent_pd = int((~valid_post).sum())
        print("\n" + "=" * 70)
        print("PRODROMAL MARKER MERGE (HES + Medication, two-pass)")
        print("Pass 1 (_bl): pre-baseline prevalence | Pass 2 (_post): incident")
        print(f"Post-window: (baseline +{post_window_start_days}d, baseline + {surv_days_col}]")
        print(f"Prevalent-PD subjects with NaN survival (post=NaN): {n_prevalent_pd:,}")
        print("=" * 70)
        print(tabulate(df_log, headers="keys", tablefmt="github", showindex=False))
        print("=" * 70)
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        df_log.to_excel(save_dir / "prodromal_markers.xlsx", index=False)

    return df


# def trail_making_covariates(
#     df: pd.DataFrame,
#     eid_col: str = "eid",
#     save_dir: Path = None,
#     instances: Iterable[int] = (2, 3),
# ) -> pd.DataFrame:
#     """
#     Build Trail Making covariates (errors, accuracy, completion time)
#     using UK Biobank fields:
#
#       p6348 : duration trail 1
#       p6349 : errors trail 1 (arrayed)
#       p6350 : duration trail 2
#       p6351 : errors trail 2 (arrayed)
#     The rest of the columns for each trial and step are these but we do not need them,
#     we need the total. So we just rename them.
#     6348	Duration to complete numeric path (trail #1)	Trail making
#     6349	Total errors traversing numeric path (trail #1)	Trail making
#     6350	Duration to complete alphanumeric path (trail #2)	Trail making
#     6351	Total errors traversing alphanumeric path (trail #2)	Trail making
#     6770	Errors before selecting correct item in numeric path (trail #1)	Trail making
#     6771	Errors before selecting correct item in alphanumeric path (trail #2)	Trail making
#     6772	Interval between previous point and current one in numeric path (trail #1)	Trail making
#     6773	Interval between previous point and current one in alphanumeric path (trail #2)	Trail making
#
#
#
#     Output: one row per eid, wide format.
#     """
#
#     # here we are filtering by the total
#     col_trails = [col for col in df.columns if (col.startswith('p6349'))]
#     if not col_trails:
#         raise ValueError("No trail making columns (p6349**) found in dataframe.")
#
#     instances = sorted({
#         int(m.group(1))
#         for s in col_trails
#         for m in [re.search(r'_i(\d+)_', s)]
#         if m
#     })
#
#     # ---------------------------------------------------------
#     # Field maps
#     # ---------------------------------------------------------
#     trial_map = {
#         "trail1": {
#             "errors": "p6349",
#             "duration": "p6348",
#         },
#         "trail2": {
#             "errors": "p6351",
#             "duration": "p6350",
#         },
#     }
#
#     out = df[[eid_col]].copy()
#
#     for trial, fields in trial_map.items():
#         for inst in instances:
#             i = f"i{inst}"
#
#             err_col = f"{fields['errors']}_{i}"
#             time_col = f"{fields['duration']}_{i}"
#
#             if err_col in df.columns:
#                 out[f"trial_making_errors_hes{trial}_{i}"] = pd.to_numeric(
#                     df[err_col], errors="coerce"
#                 )
#
#             if time_col in df.columns:
#                 out[f"hes_trial_making_duration_hes{trial}_{i}"] = pd.to_numeric(
#                     df[time_col], errors="coerce"
#                 )
#     col_trails = [col for col in df.columns if col.startswith('p677')]
#     df = df.drop(columns=col_trails)
#
#     _ = report_trail_making_covariates(
#         df=df,
#         output_path=save_dir / "trail_making_report.csv",
#         verbose=True,
#     )
#     return df


# ------------------------------------------------------------------
# Trail Making constants
# ------------------------------------------------------------------
_TMT_CLINIC_DECISEC_TO_SEC: float = 0.1   # p6348/p6350 are in deciseconds
_TMT_DUR_MIN_S: float = 5.0               # < 5s is physiologically implausible
_TMT_DUR_MAX_S: float = 600.0             # > 10 min is invalid
_TMT_RATIO_MIN: float = 1.0              # Trail-2 must always take ≥ Trail-1


def trail_making_covariates(
    df: pd.DataFrame,
    eid_col: str = "eid",
    save_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Extract all available Trail Making Test measurements from UK Biobank.

    Sources
    -------
    Online (instances i0, i1):
        p20156  Trail-1 (numeric path) duration  [seconds]
        p20157  Trail-2 (alphanumeric path) duration  [seconds]
        p20247  Trail-1 total errors
        p20248  Trail-2 total errors
        p20246  Completion status (0 = both trails completed)
        p20136  Exact datetime of assessment

    In-clinic (instances i2, i3):
        p6348   Trail-1 duration  [deciseconds → converted to seconds ÷ 10]
        p6350   Trail-2 duration  [deciseconds → converted to seconds ÷ 10]
        p6349   Trail-1 errors
        p6351   Trail-2 errors
        follow_up_date_iX  Assessment visit date

    No temporal filtering is applied.  Baseline selection relative to
    wear_time_start is performed by select_tmt_baseline() called from
    add_covariates() immediately after this function.

    Validity guards (per trail, per instance independently):
        Online: p20246 != 0 → both trail values NaN for that instance.
        Duration < _TMT_DUR_MIN_S or > _TMT_DUR_MAX_S → NaN.
        Ratio < _TMT_RATIO_MIN → NaN.

    Output columns
    --------------
    tmt1_dur_{source}_{inst}     Trail-1 duration (seconds)
    tmt2_dur_{source}_{inst}     Trail-2 duration (seconds)
    tmt1_err_{source}_{inst}     Trail-1 errors
    tmt2_err_{source}_{inst}     Trail-2 errors
    tmt_ratio_{source}_{inst}    Trail-2 / Trail-1 duration ratio
    tmt_date_{source}_{inst}     Assessment date
    """
    out = df[[eid_col]].copy()

    # ------------------------------------------------------------------
    # Online: instances i0, i1
    # ------------------------------------------------------------------
    _online = {
        "dur_tmt1": "p20156",
        "dur_tmt2": "p20157",
        "err_tmt1": "p20247",
        "err_tmt2": "p20248",
        "complete": "p20246",
        "date":     "p20136",
    }

    for inst in (0, 1):
        tag = f"i{inst}"
        complete_col = f"{_online['complete']}_{tag}"

        # completion mask: 0 = both trails completed
        if complete_col in df.columns:
            completed: pd.Series = pd.to_numeric(df[complete_col], errors="coerce") == 0
        else:
            completed = pd.Series(False, index=df.index)

        date_col = f"{_online['date']}_{tag}"
        if date_col in df.columns:
            out[f"tmt_date_online_{tag}"] = pd.to_datetime(df[date_col], errors="coerce")

        for trail, field_key in (("tmt1", "dur_tmt1"), ("tmt2", "dur_tmt2")):
            src_col = f"{_online[field_key]}_{tag}"
            if src_col not in df.columns:
                continue
            dur = pd.to_numeric(df[src_col], errors="coerce")
            dur = dur.where(completed)
            dur = dur.where((dur >= _TMT_DUR_MIN_S) & (dur <= _TMT_DUR_MAX_S))
            out[f"{trail}_dur_online_{tag}"] = dur

        for trail, field_key in (("tmt1", "err_tmt1"), ("tmt2", "err_tmt2")):
            src_col = f"{_online[field_key]}_{tag}"
            if src_col not in df.columns:
                continue
            err = pd.to_numeric(df[src_col], errors="coerce")
            out[f"{trail}_err_online_{tag}"] = err.where(completed)

        dur1_col, dur2_col = f"tmt1_dur_online_{tag}", f"tmt2_dur_online_{tag}"
        if dur1_col in out.columns and dur2_col in out.columns:
            ratio = out[dur2_col] / out[dur1_col]
            out[f"tmt_ratio_online_{tag}"] = ratio.where(ratio >= _TMT_RATIO_MIN)

    # ------------------------------------------------------------------
    # In-clinic: instances i2, i3
    # ------------------------------------------------------------------
    _clinic = {
        "dur_tmt1": "p6348",
        "dur_tmt2": "p6350",
        "err_tmt1": "p6349",
        "err_tmt2": "p6351",
    }

    for inst in (2, 3):
        tag = f"i{inst}"

        date_col = f"follow_up_date_{tag}"
        if date_col in df.columns:
            out[f"tmt_date_clinic_{tag}"] = pd.to_datetime(df[date_col], errors="coerce")

        for trail, field_key in (("tmt1", "dur_tmt1"), ("tmt2", "dur_tmt2")):
            src_col = f"{_clinic[field_key]}_{tag}"
            if src_col not in df.columns:
                continue
            # deciseconds → seconds
            dur = pd.to_numeric(df[src_col], errors="coerce") * _TMT_CLINIC_DECISEC_TO_SEC
            dur = dur.where((dur >= _TMT_DUR_MIN_S) & (dur <= _TMT_DUR_MAX_S))
            out[f"{trail}_dur_clinic_{tag}"] = dur

        for trail, field_key in (("tmt1", "err_tmt1"), ("tmt2", "err_tmt2")):
            src_col = f"{_clinic[field_key]}_{tag}"
            if src_col not in df.columns:
                continue
            out[f"{trail}_err_clinic_{tag}"] = pd.to_numeric(df[src_col], errors="coerce")

        dur1_col, dur2_col = f"tmt1_dur_clinic_{tag}", f"tmt2_dur_clinic_{tag}"
        if dur1_col in out.columns and dur2_col in out.columns:
            ratio = out[dur2_col] / out[dur1_col]
            out[f"tmt_ratio_clinic_{tag}"] = ratio.where(ratio >= _TMT_RATIO_MIN)

    # ------------------------------------------------------------------
    # Merge and report
    # ------------------------------------------------------------------
    df_merged = df.merge(out, on=eid_col, how="left")

    tmt_cols = [c for c in out.columns if c != eid_col]
    print(f"  Trail Making: {len(tmt_cols)} columns created: {tmt_cols}")

    if save_dir:
        report_trail_making_covariates(
            df=df_merged,
            output_path=save_dir / "trail_making_report.csv",
            verbose=True,
        )
        ratio_cols = [c for c in tmt_cols if c.startswith("tmt_ratio")]
        report_outcomes_by_flags(
            df=df_merged,
            outcomes=outcomes,
            verbose=True,
            flags=ratio_cols,
            output_path=save_dir / "trail_making_report_outcomes.csv",
        )

    return df_merged


def symbol_digit_substitution_covariates(
    df: pd.DataFrame,
    eid_col: str = "eid",
    instances: tuple[int, ...] = (0, 1),
) -> pd.DataFrame:
    """
    Reduce UK Biobank Symbol Digit Substitution (Category 122)
    to interpretable, epidemiologically valid covariates.

    Outputs (per instance):
      - sds_duration_iX (minutes)
      - sds_correct_iX
      - sds_attempted_iX
      - sds_accuracy_iX
      - sds_correct_per_min_iX
    """

    out = df[[eid_col]].copy()

    for i in instances:
        dur_col = f"p20134_i{i}"
        correct_col = f"p20159_i{i}"
        attempt_col = f"p20195_i{i}"

        if not all(c in df.columns for c in [dur_col, correct_col, attempt_col]):
            continue

        duration_min = pd.to_numeric(df[dur_col], errors="coerce") / 60.0
        correct = pd.to_numeric(df[correct_col], errors="coerce")
        attempted = pd.to_numeric(df[attempt_col], errors="coerce")

        accuracy = correct / attempted
        accuracy = accuracy.where((attempted > 0) & (accuracy <= 1))

        correct_per_min = correct / duration_min
        correct_per_min = correct_per_min.where(duration_min > 0)

        out[f"cov_sds_duration_i{i}"] = duration_min
        out[f"cov_sds_correct_i{i}"] = correct
        out[f"cov_sds_attempted_i{i}"] = attempted
        out[f"cov_sds_accuracy_i{i}"] = accuracy
        out[f"cov_sds_correct_per_min_i{i}"] = correct_per_min

    df = df.merge(
        out,
        on="eid",
        how="left",
    )

    drop_prefixes = ("p20129", "p20130", "p20131", "p20132", "p20133", "p20134",
                     "p20159", "p20195", "p20230")

    df = df.drop(
        columns=[c for c in df.columns if c.startswith(drop_prefixes)],
        errors="ignore",
    )

    return df


def derive_reaction_time_metrics(
        df: pd.DataFrame,
        rt_field: str = "p20023",
        round_field: str = "p404",
        pilot_round_field: str = "p10147",
        verbose: bool = True,
        drop_round_level: bool = False,
) -> pd.DataFrame:
    """
    Derive reaction-time and performance metrics from UK Biobank
    Snap / reaction time task.

    Focuses on:
      - Global mean RT (Field 20023)
      - Per-round first-press RT distributions (Field 404 / 10147)

    Ignores card identity fields.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    rt_field : str
        Base field for mean reaction time (default p20023).
    round_field : str
        Base field for per-round RT (main assessment).
    pilot_round_field : str
        Base field for pilot per-round RT.
    drop_round_level : bool
        Drop round-level columns after aggregation.
    """

    # --------------------------------------------------
    # 1. Mean reaction time (already aggregated)
    # --------------------------------------------------
    rt_cols = [c for c in df.columns if c.startswith(f"{rt_field}_i")]

    for col in rt_cols:
        visit = col.split("_i")[-1]
        df[f"cov_rt_mean_i{visit}"] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------------
    # 2. Per-round RT aggregation (main + pilot)
    # --------------------------------------------------
    def _aggregate_rounds(prefix: str, label: str):
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            return []

        arr = df[cols].apply(pd.to_numeric, errors="coerce")
        valid = arr.where(arr > 0)  # RT must be positive

        base_cols = []

        df[f"cov_rt_round_median_{label}"] = valid.median(axis=1)
        df[f"cov_rt_round_mean_{label}"] = valid.mean(axis=1)
        # df[f"rt_round_iqr_{label}"] = valid.quantile(0.75, axis=1) - valid.quantile(0.25, axis=1)
        # df[f"rt_round_min_{label}"] = valid.min(axis=1)
        # df[f"rt_round_max_{label}"] = valid.max(axis=1)
        df[f"cov_rt_round_n_valid_{label}"] = valid.notna().sum(axis=1)

        base_cols.extend(cols)
        return base_cols

    main_round_cols = _aggregate_rounds(round_field, "main")
    pilot_round_cols = _aggregate_rounds(pilot_round_field, "pilot")
    cols_card_indices = [c for c in df.columns if c.startswith(("p401_", "p402_", "p10139_", "p10140_"))]
    # --------------------------------------------------
    # 3. Optional cleanup
    # --------------------------------------------------
    if drop_round_level:
        df = df.drop(columns=main_round_cols +
                             pilot_round_cols +
                            cols_card_indices,
                     errors="ignore")

    if verbose:
        print("Reaction time metrics created:")
        print("  -> Mean RT per visit: rt_mean_iX")
        print("  -> Round-level metrics: median / IQR / min / max / n_valid")

    return df


# %% Report functions


def report_trail_making_covariates(
    df: pd.DataFrame,
    eid_col: str = "eid",
    prefix_patterns: tuple[str, ...] = (
        "tmt1_dur_",
        "tmt2_dur_",
        "tmt1_err_",
        "tmt2_err_",
        "tmt_ratio_",
    ),
    verbose: bool = True,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Generate descriptive statistics for Trail Making covariates.

    Reports, per variable:
      - N total
      - N observed
      - % missing
      - mean, SD
      - median, IQR
      - min, max
    """

    # ---------------------------------------------------------
    # Identify trail-making columns
    # ---------------------------------------------------------
    tm_cols = [
        c for c in df.columns
        if any(c.startswith(p) for p in prefix_patterns)
    ]

    if not tm_cols:
        raise ValueError("No Trail Making covariate columns found.")

    rows = []
    n_total = df.shape[0]

    for col in tm_cols:
        x = pd.to_numeric(df[col], errors="coerce")

        n_obs = int(x.notna().sum())
        miss_pct = round(100 * (n_total - n_obs) / n_total, 2)

        rows.append({
            "variable": col,
            "N_total": n_total,
            "N_observed": n_obs,
            "missing_pct": miss_pct,
            "mean": x.mean(),
            "sd": x.std(),
            "median": x.median(),
            "iqr": x.quantile(0.75) - x.quantile(0.25),
            "min": x.min(),
            "max": x.max(),
        })

    report_df = pd.DataFrame(rows).sort_values("variable")

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------
    if verbose:
        print("\n" + "=" * 100)
        print("TRAIL MAKING COVARIATES REPORT")
        print("=" * 100)
        print(
            tabulate(
                report_df,
                headers="keys",
                tablefmt="github",
                showindex=False,
                floatfmt=".3f",
            )
        )
        print("=" * 100)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(output_path, index=False)

    return report_df


# ------------------------------------------------------------------
# Lifestyle covariates: smoking and alcohol
# ------------------------------------------------------------------

def prepare_lifestyle_covariates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build best-available ``cov_smoking`` and ``cov_alcohol`` columns.

    Iterates visit instances in order (i0 → i3) and fills forward using the
    first non-null, non-negative observation.  Negative codes (e.g. -3 = prefer
    not to answer) are treated as missing.

    This function must be called AFTER column renaming in ``add_covariates()``
    so that instance-suffixed names (e.g. ``cov_smoking_20116_bl``) are present.

    Args:
        df: DataFrame with renamed covariate columns.

    Returns:
        DataFrame with ``cov_smoking`` and ``cov_alcohol`` columns added.
    """
    df = df.copy()

    for out_col, candidates in [
        ("cov_smoking", _SMOKING_CANDIDATES),
        ("cov_alcohol", _ALCOHOL_CANDIDATES),
    ]:
        if out_col in df.columns:
            print(f"  Lifestyle covariate '{out_col}' already present — skipping.")
            continue

        series = pd.Series(np.nan, index=df.index, dtype=float)
        for col in candidates:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                vals = vals.where(vals >= 0, np.nan)
                series = series.fillna(vals)

        df[out_col] = series
        n_obs = int(df[out_col].notna().sum())
        pct = n_obs / len(df) * 100
        print(f"  Lifestyle covariate '{out_col}': {n_obs:,} obs ({pct:.1f}%)")

    return df


# ------------------------------------------------------------------
# BMI consolidation (combine multiple sources to fill missing)
# ------------------------------------------------------------------

def combine_bmi_measures(
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Derive a single baseline BMI covariate (cov_bmi) by chaining
    available BMI sources in priority order.

    Combines two BMI measurement types (manual p21001 and bioimpedance-derived
    p23104) across four visit instances (i0–i3). Fills missing values by
    cascading through a prioritized list, preferring baseline (_bl) and manual
    measurement (21001) over later visits and impedance-derived measures.

    Priority (earliest / most direct measurement first):
        1. bmi_21001_bl  — manual BMI, baseline visit (i0)
        2. bmi_imp_23104_bl — bioimpedance BMI, baseline visit (i0)
        3. bmi_21001_i1 / bmi_imp_23104_i1  — visit 1 fallback
        4. bmi_21001_fu / bmi_imp_23104_fu  — follow-up visit (i2) fallback
        5. bmi_21001_i3 / bmi_imp_23104_i3  — visit 3 fallback

    At baseline (_bl), bmi_21001 and bmi_imp_23104 are nearly perfectly
    correlated (r=0.9999, mean abs diff=0.031 kg/m²) and have complementary
    coverage; combining them reduces missing at baseline from ~0.2% to ~0.18%.

    This function must be called AFTER column renaming in ``add_covariates()``
    so that renamed instance-suffixed names (e.g. ``bmi_21001_bl``) are present.

    Args:
        df: DataFrame with renamed covariate columns from ``add_covariates()``.
        verbose: If True, print missingness counts at each step.

    Returns:
        DataFrame with new column ``cov_bmi`` (float64).
    """
    df = df.copy()

    # Initialize new BMI column
    cov_bmi = pd.Series(np.nan, index=df.index, dtype=float)
    n_total = len(df)

    # Priority order for fill chain
    fill_chain: list[str] = [
        "bmi_21001_bl",
        "bmi_imp_23104_bl",
        "bmi_21001_i1",
        "bmi_imp_23104_i1",
        "bmi_21001_fu",
        "bmi_imp_23104_fu",
        "bmi_21001_i3",
        "bmi_imp_23104_i3",
    ]

    if verbose:
        print(f"    Starting fill: missing = {cov_bmi.isna().sum():,} / {n_total:,}")

    for col in fill_chain:
        if col not in df.columns:
            if verbose:
                print(f"    Skipping {col:20} (not present)")
            continue

        n_before = cov_bmi.notna().sum()
        cov_bmi = cov_bmi.fillna(df[col])
        n_after = cov_bmi.notna().sum()
        n_filled = n_after - n_before

        if verbose and n_filled > 0:
            print(f"    {col:20}: filled {n_filled:6} -> total {n_after:6} / {n_total:,}")

    df["cov_bmi"] = cov_bmi
    n_final = cov_bmi.notna().sum()
    pct_final = n_final / n_total * 100
    pct_missing = (1 - pct_final / 100) * 100

    if verbose:
        print(f"    Final BMI: {n_final:,} / {n_total:,} ({pct_final:.2f}% obs, {pct_missing:.2f}% missing)")

    return df


# ------------------------------------------------------------------
# BMI imputation via k-NN hot-deck matching on controls
# ------------------------------------------------------------------

def impute_bmi_knn(
    df: pd.DataFrame,
    k: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Impute missing cov_bmi values using k-nearest-neighbour hot-deck
    matching on controls with observed BMI.

    For each subject with missing cov_bmi, the k most similar controls
    (by Euclidean distance in standardised [age, sex] space) are identified.
    The imputed value is the mean of those k donors' cov_bmi values.

    Matching features: cov_age_recruitment_21022, cov_sex_31 (both 100%
    observed for all missing-BMI subjects per data audit).

    Args:
        df: DataFrame with cov_bmi, cov_age_recruitment_21022, cov_sex_31,
            control columns (post-rename from add_covariates).
        k: Number of nearest neighbours to average (default 5).
        verbose: Print imputation audit log.

    Returns:
        DataFrame with missing cov_bmi values imputed.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    df = df.copy()

    # Early exit if no missing BMI
    if df['cov_bmi'].notna().all():
        if verbose:
            print("    No missing BMI to impute.")
        return df

    # Matching features (confirmed 100% observed for missing-BMI subjects)
    match_features = ['cov_age_recruitment_21022', 'cov_sex_31']

    # Subset 1: Donor pool (controls with observed BMI and complete matching features)
    donors = df[
        (df['control'] == 1) &
        (df['cov_bmi'].notna()) &
        (df[match_features].notna().all(axis=1))
    ].copy()

    n_donors = len(donors)
    if n_donors == 0:
        if verbose:
            print("    WARNING: No donors available for imputation.")
        return df

    # Subset 2: Targets (subjects with missing BMI and complete matching features)
    targets_mask = (
        (df['cov_bmi'].isna()) &
        (df[match_features].notna().all(axis=1))
    )
    n_targets = targets_mask.sum()

    if n_targets == 0:
        if verbose:
            print("    No subjects with missing BMI and complete matching features.")
        return df

    targets_idx = df[targets_mask].index

    # Standardise matching features (fit on donors, transform both)
    scaler = StandardScaler()
    donors_scaled = scaler.fit_transform(donors[match_features])
    targets_scaled = scaler.transform(df.loc[targets_idx, match_features])

    # k-NN: find k nearest donors for each target
    knn = NearestNeighbors(n_neighbors=min(k, n_donors), metric='euclidean')
    knn.fit(donors_scaled)
    distances, indices = knn.kneighbors(targets_scaled)

    # Impute: mean BMI of k nearest donors
    donors_arr = donors['cov_bmi'].values
    imputed_values = donors_arr[indices].mean(axis=1)

    # Fill in original dataframe
    df.loc[targets_idx, 'cov_bmi'] = imputed_values

    # Audit log
    if verbose:
        print(f"    k-NN imputation (k={min(k, n_donors)}):")
        print(f"      Donors available: {n_donors:,}")
        print(f"      Targets imputed: {n_targets}")
        print(f"      Imputed BMI: mean={imputed_values.mean():.2f}, "
              f"std={imputed_values.std():.2f}, "
              f"min={imputed_values.min():.2f}, max={imputed_values.max():.2f}")
        remaining_missing = df['cov_bmi'].isna().sum()
        if remaining_missing > 0:
            print(f"      Remaining missing: {remaining_missing} "
                  f"(no matching features available)")

    return df


# ------------------------------------------------------------------
# TMT baseline selection (from per-instance trail-making columns)
# ------------------------------------------------------------------

def select_tmt_baseline(
    df: pd.DataFrame,
    wear_col: str = "wear_time_start",
    baseline_window_days: int = 730,
) -> pd.DataFrame:
    """
    Select the epidemiologically valid TMT baseline measurement per subject.

    Evaluates all TMT instances from ``trail_making_covariates()`` and selects
    the best one whose assessment date falls within ±baseline_window_days of
    ``wear_time_start``.

    Selection priority
    ------------------
    1. Source: clinic (in-person) preferred over online.
    2. Tiebreak: assessment closest in time to wear_time_start.

    Epidemiological rationale
    -------------------------
    The actigraphy wear window (2013–2015) and online TMT i0 (2014–2015) are
    concurrent.  A ±730-day window captures the same baseline health state.
    Instances i1 and i3 (2021–2023) fall during follow-up and are excluded to
    prevent reverse causation.

    Args:
        df: DataFrame with tmt_* columns from ``trail_making_covariates()``
            and a datetime ``wear_col``.
        wear_col: Column containing actigraphy wear start datetime.
        baseline_window_days: Maximum absolute lag (days) from wear_time_start.

    Returns:
        DataFrame with added columns:
            tmt1_dur_baseline     Trail-1 duration at baseline (seconds)
            tmt2_dur_baseline     Trail-2 duration at baseline (seconds)
            tmt1_err_baseline     Trail-1 errors at baseline
            tmt2_err_baseline     Trail-2 errors at baseline
            tmt_ratio_baseline    Trail-2 / Trail-1 ratio at baseline
            tmt_lag_days          Signed days from assessment to wear_time_start
            tmt_source_baseline   Instance used (e.g. "online_i0", "clinic_i2")
            tmt_missing           True if no eligible instance within window
    """
    _CANDIDATES: list[tuple[str, str]] = [
        ("clinic", "i2"),
        ("clinic", "i3"),
        ("online", "i0"),
        ("online", "i1"),
    ]
    _SOURCE_PRIORITY: dict[str, float] = {"clinic": 0.0, "online": 1.0}

    wear = pd.to_datetime(df[wear_col], errors="coerce")

    best_priority = pd.Series(np.inf, index=df.index, dtype=float)
    best_abs_lag  = pd.Series(np.inf, index=df.index, dtype=float)
    best_source   = pd.Series(pd.NA,  index=df.index, dtype=object)

    out_metric_cols = ["tmt1_dur", "tmt2_dur", "tmt1_err", "tmt2_err", "tmt_ratio"]
    best_metrics: dict[str, pd.Series] = {
        col: pd.Series(np.nan, index=df.index, dtype=float)
        for col in out_metric_cols
    }
    best_lag = pd.Series(np.nan, index=df.index, dtype=float)

    for source, inst in _CANDIDATES:
        date_col = f"tmt_date_{source}_{inst}"
        if date_col not in df.columns:
            continue

        assessment_date = pd.to_datetime(df[date_col], errors="coerce")
        lag     = (assessment_date - wear).dt.days.astype(float)
        abs_lag = lag.abs()
        within  = abs_lag <= baseline_window_days

        ratio_col = f"tmt_ratio_{source}_{inst}"
        has_ratio = (
            pd.to_numeric(df[ratio_col], errors="coerce").notna()
            if ratio_col in df.columns
            else pd.Series(False, index=df.index)
        )

        eligible    = within & has_ratio
        src_priority = _SOURCE_PRIORITY[source]

        is_better = eligible & (
            (src_priority < best_priority)
            | ((src_priority == best_priority) & (abs_lag < best_abs_lag))
        )

        best_priority = best_priority.where(~is_better, src_priority)
        best_abs_lag  = best_abs_lag.where(~is_better, abs_lag)
        best_lag      = best_lag.where(~is_better, lag)
        best_source   = best_source.where(~is_better, f"{source}_{inst}")

        for metric in out_metric_cols:
            src_metric_col = f"{metric}_{source}_{inst}"
            vals = (
                pd.to_numeric(df[src_metric_col], errors="coerce")
                if src_metric_col in df.columns
                else pd.Series(np.nan, index=df.index, dtype=float)
            )
            best_metrics[metric] = best_metrics[metric].where(~is_better, vals)

    df = df.copy()
    for metric, vals in best_metrics.items():
        df[f"{metric}_baseline"] = vals

    df["tmt_lag_days"]        = best_lag
    df["tmt_source_baseline"] = best_source
    df["tmt_missing"]         = best_source.isna()

    n_assigned = int(df["tmt_source_baseline"].notna().sum())
    n_total    = len(df)
    src_counts = df["tmt_source_baseline"].value_counts().to_dict()
    print(
        f"  TMT baseline: {n_assigned:,}/{n_total:,} assigned "
        f"({100 * n_assigned / n_total:.1f}%) | "
        f"missing: {n_total - n_assigned:,} | sources: {src_counts}"
    )

    return df


def prospective_memory_covariates(
    df: pd.DataFrame,
    instances: tuple[int, ...] = (0, 1),
) -> pd.DataFrame:
    """
    Process UK Biobank Prospective Memory field (6373).

    UKBB field 6373 coding:
        1  = Correct
        0  = Incorrect
       -1  = Prefer not to answer  → NaN
       -3  = Not answered          → NaN

    Args:
        df: Input DataFrame with raw p6373_iX columns.
        instances: Visit instances to process (UKBB: 0 and 1 only).

    Returns:
        DataFrame with cov_prospective_memory_6373_iX (0/1) added
        and source p6373_iX columns dropped.
    """
    df = df.copy()
    valid_map = {1: 1, 0: 0}

    for i in instances:
        src_col = f"p6373_i{i}"
        dst_col = f"cov_prospective_memory_6373_i{i}"
        if src_col not in df.columns:
            continue
        raw = pd.to_numeric(df[src_col], errors="coerce")
        df[dst_col] = raw.map(valid_map)

    drop_cols = [f"p6373_i{i}" for i in instances if f"p6373_i{i}" in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df


# ---------------------------------------------------------------------------
# Cognitive variable registry used by add_cognitive_latest_per_subject()
# ---------------------------------------------------------------------------
# Instance-number -> column suffix label.  Mirrors the INSTANCE_SUFFIX_MAP
# applied in add_covariates(): i0 -> baseline (_bl), i2 -> follow-up (_fu).
# i1/i3 retain numeric notation.  Used to resolve per-instance column names
# for variables whose source field was renamed by build_ukbb_rename_map().
_INST_LABEL: dict[int, str] = {0: "bl", 1: "i1", 2: "fu", 3: "i3"}

# Each entry uses a single ``{S}`` placeholder for the instance suffix.
# ``suffix_renamed`` marks whether the source field is in ``code_to_name`` and
# was therefore renamed by the suffix_map (i0->bl, i2->fu).  Derived columns
# (cov_rt_mean, cov_sds_*, cov_prospective_memory) are NOT renamed and keep the
# raw ``i{n}`` notation; see add_cognitive_latest_per_subject() for resolution.
_COG_VARS: list[dict] = [
    {
        "base": "fluid_intelligence",
        "label": "Fluid Intelligence",
        "unit": "score 0-13 (correct answers)",
        "pattern": "cov_fluid_intelligence_20016_{S}",
        "instances": [0, 1, 2, 3],
        "suffix_renamed": True,
    },
    {
        "base": "react_time",
        "label": "Mean Reaction Time",
        "unit": "milliseconds",
        "pattern": "cov_rt_mean_{S}",
        "instances": [0, 1, 2, 3],
        "suffix_renamed": False,
    },
    {
        "base": "fi_questions",
        "label": "FI Questions Attempted",
        "unit": "count",
        "pattern": "cov_fi_questions_attempted_20128_{S}",
        "instances": [0, 1, 2, 3],
        "suffix_renamed": True,
    },
    {
        "base": "numeric_memory",
        "label": "Numeric Memory",
        "unit": "max digits recalled",
        "pattern": "cov_numeric_memory_max_20240_{S}",
        "instances": [0, 1, 2, 3],
        "suffix_renamed": True,
    },
    {
        "base": "pairs_status",
        "label": "Pairs Matching",
        "unit": "errors (field 20244)",
        "pattern": "cov_pairs_status_20244_{S}",
        "instances": [0, 1, 2, 3],
        "suffix_renamed": True,
    },
    {
        "base": "sds_correct_per_min",
        "label": "SDS Correct per Minute",
        "unit": "correct answers/minute",
        "pattern": "cov_sds_correct_per_min_{S}",
        "instances": [0, 1],
        "suffix_renamed": False,
    },
    {
        "base": "sds_accuracy",
        "label": "SDS Accuracy",
        "unit": "proportion 0-1",
        "pattern": "cov_sds_accuracy_{S}",
        "instances": [0, 1],
        "suffix_renamed": False,
    },
    {
        "base": "prospective_memory",
        "label": "Prospective Memory",
        "unit": "binary (0=incorrect, 1=correct)",
        "pattern": "cov_prospective_memory_6373_{S}",
        "instances": [0, 1],
        "suffix_renamed": False,
    },
]

_TMT_ALIASES: list[dict] = [
    {
        "base": "tmt1_dur",
        "label": "TMT-A Duration",
        "unit": "seconds",
        "source_col": "tmt1_dur_baseline",
    },
    {
        "base": "tmt2_dur",
        "label": "TMT-B Duration",
        "unit": "seconds",
        "source_col": "tmt2_dur_baseline",
    },
    {
        "base": "tmt_ratio_log",
        "label": "TMT-B/A Ratio (log)",
        "unit": "log(seconds/seconds)",
        "source_col": "cog_tmt_ratio_log_bl",
    },
]


def add_cognitive_latest_per_subject(
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Select the latest available cognitive assessment per subject across UKBB instances.

    For each variable in _COG_VARS, finds the most recent non-null observation
    (based on follow_up_date_iX) and writes three columns:
        cog_{base}_latest           — measurement value
        cog_{base}_latest_date      — assessment date at that instance
        cog_{base}_latest_instance  — instance label (e.g. "i1")

    TMT variables are aliased from tmt*_baseline (already selected by
    select_tmt_baseline() using clinic-preferred, proximity-to-wear-time logic).

    # TODO: After first run, validate units/ranges in the debug table below
    #       and confirm N_valid matches expected non-missing rates.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # Non-TMT: walk instances latest→earliest, fill first valid per subject
    # ------------------------------------------------------------------
    for var in _COG_VARS:
        base = var["base"]
        pattern = var["pattern"]
        suffix_renamed = var["suffix_renamed"]
        instances_desc = sorted(var["instances"], reverse=True)  # latest first

        latest_val: pd.Series = pd.Series(np.nan, index=df.index, dtype=float)
        latest_date: pd.Series = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        latest_inst: pd.Series = pd.Series(pd.NA, index=df.index, dtype=object)

        for i in instances_desc:
            # Renamed fields (in code_to_name) use the _bl/_fu labels applied by
            # the suffix_map; derived fields keep raw i{n} notation.
            suffix = _INST_LABEL.get(i, f"i{i}") if suffix_renamed else f"i{i}"
            val_col = pattern.format(S=suffix)
            # Visit date keeps raw instance notation (created in the extractor as
            # follow_up_date_i{n}; never passed through the suffix_map).
            date_col = f"follow_up_date_i{i}"
            if val_col not in df.columns or date_col not in df.columns:
                continue

            vals = pd.to_numeric(df[val_col], errors="coerce")
            dates = pd.to_datetime(df[date_col], errors="coerce")

            not_filled = latest_val.isna()
            has_valid = vals.notna() & dates.notna()
            update = not_filled & has_valid

            latest_val = latest_val.where(~update, vals)
            latest_date = latest_date.where(~update, dates)
            # Report the raw UKB instance (i0..i3) for unambiguous provenance,
            # independent of the _bl/_fu column-naming convention.
            latest_inst = latest_inst.where(~update, f"i{i}")

        df[f"cog_{base}_latest"] = latest_val
        df[f"cog_{base}_latest_date"] = latest_date
        df[f"cog_{base}_latest_instance"] = latest_inst

    # ------------------------------------------------------------------
    # TMT: alias from existing baseline columns (clinic-preferred selection)
    # ------------------------------------------------------------------
    for alias in _TMT_ALIASES:
        base = alias["base"]
        src = alias["source_col"]
        if src in df.columns:
            df[f"cog_{base}_latest"] = df[src].copy()
        if "tmt_source_baseline" in df.columns:
            df[f"cog_{base}_latest_instance"] = df["tmt_source_baseline"].copy()

    # ------------------------------------------------------------------
    # Visit-anchored cognitive columns: baseline (_bl = i0), follow-up
    # (_fu = i2), and change (_delta = fu - bl).
    #
    # Distinct from the cog_*_latest columns above (latest-available, any
    # visit): these are fixed to specific UKB instances so that adjusted Cox
    # models and Table 1 use a temporally consistent baseline/follow-up.
    #
    # Source columns:
    #   - Fields in code_to_name were renamed by the suffix_map (i0->_bl,
    #     i2->_fu): cov_fluid_intelligence_20016_*, cov_react_time_mean_20023_*,
    #     cov_numeric_memory_max_20240_*, cov_pairs_status_20244_*.
    #   - TMT columns are derived (not renamed): tmt*_baseline keep their name;
    #     follow-up TMT reads the clinic_i2 columns directly.
    #   - Numeric memory and pairs matching are collected at i0 only — no i2
    #     column exists, so _fu is intentionally all-NaN.
    # All reads are guarded: an absent source yields an all-NaN column.
    # ------------------------------------------------------------------
    def _num(col: str) -> pd.Series:
        """Coerce a source column to numeric, or all-NaN if absent."""
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype=float)

    # Baseline (_bl = recruitment visit i0)
    df["cog_fluid_intelligence_bl"] = _num("cov_fluid_intelligence_20016_bl")
    df["cog_react_time_bl"]         = _num("cov_react_time_mean_20023_bl")
    df["cog_numeric_memory_bl"]     = _num("cov_numeric_memory_max_20240_bl")
    df["cog_pairs_matching_bl"]     = _num("cov_pairs_status_20244_bl")
    df["cog_fi_questions_bl"]       = _num("cov_fi_questions_attempted_20128_bl")
    df["cog_tmt1_dur_bl"]           = _num("tmt1_dur_baseline")
    df["cog_tmt2_dur_bl"]           = _num("tmt2_dur_baseline")
    # cog_tmt_ratio_log_bl already created in add_covariates() (np.log of the
    # selected baseline ratio); ensure presence for self-containment.
    if "cog_tmt_ratio_log_bl" not in df.columns:
        df["cog_tmt_ratio_log_bl"] = _num("tmt_ratio_baseline").apply(
            lambda r: np.log(r) if pd.notna(r) and r > 0 else np.nan
        )

    # Follow-up (_fu = imaging visit i2)
    df["cog_fluid_intelligence_fu"] = _num("cov_fluid_intelligence_20016_fu")
    df["cog_react_time_fu"]         = _num("cov_react_time_mean_20023_fu")
    df["cog_numeric_memory_fu"]     = pd.Series(np.nan, index=df.index, dtype=float)  # i0 only
    df["cog_pairs_matching_fu"]     = pd.Series(np.nan, index=df.index, dtype=float)  # i0 only
    df["cog_tmt1_dur_fu"]           = _num("tmt1_dur_clinic_i2")
    df["cog_tmt2_dur_fu"]           = _num("tmt2_dur_clinic_i2")
    ratio_fu = _num("tmt_ratio_clinic_i2")
    df["cog_tmt_ratio_log_fu"]      = np.log(ratio_fu.where(ratio_fu > 0))

    # Change (_delta = follow-up - baseline); FI and RT only.
    # No TMT delta: baseline TMT is ~99% online_i0, follow-up is clinic_i2 —
    # a paradigm mismatch that would confound administration with cognition.
    # NaN propagates where either endpoint is missing (no imputation).
    df["cog_fluid_intelligence_delta"] = (
        df["cog_fluid_intelligence_fu"] - df["cog_fluid_intelligence_bl"]
    )
    df["cog_react_time_delta"] = df["cog_react_time_fu"] - df["cog_react_time_bl"]

    # ------------------------------------------------------------------
    # DEBUG: print coverage and descriptive stats to terminal
    # TODO: Validate units and value ranges — confirm median/IQR are
    #       plausible for each measure before running the strata analysis.
    # ------------------------------------------------------------------
    if verbose:
        print("\n[DEBUG] Cognitive latest column summary (Step 4c):")
        header = f"  {'Variable':<35} {'N_valid':>8}  {'Median':>10}  {'[P25, P75]':<25}  Unit"
        sep = f"  {'-'*35} {'-'*8}  {'-'*10}  {'-'*25}  {'-'*35}"
        print(header)
        print(sep)

        all_vars = [(v["base"], v["unit"]) for v in _COG_VARS] + \
                   [(a["base"], a["unit"]) for a in _TMT_ALIASES]

        for base, unit in all_vars:
            col = f"cog_{base}_latest"
            if col not in df.columns:
                print(f"  {base:<35} {'MISSING':>8}")
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            n = int(series.notna().sum())
            if n == 0:
                print(f"  {base:<35} {0:>8,}  {'N/A':>10}  {'[N/A, N/A]':<25}  {unit}")
                continue
            p25 = float(series.quantile(0.25))
            med = float(series.quantile(0.50))
            p75 = float(series.quantile(0.75))
            iqr_str = f"[{p25:.2f}, {p75:.2f}]"
            print(f"  {base:<35} {n:>8,}  {med:>10.2f}  {iqr_str:<25}  {unit}")

        print("\n  Instance distribution per variable:")
        for var in _COG_VARS:
            base = var["base"]
            inst_col = f"cog_{base}_latest_instance"
            if inst_col in df.columns:
                dist = df[inst_col].value_counts().to_dict()
                print(f"    {base}: {dist}")

    return df
