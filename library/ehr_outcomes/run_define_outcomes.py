from library.ehr_outcomes.outcome_flags import add_outcome_flags
from library.ehr_outcomes.control_builder import define_controls
from library.ehr_outcomes.competing_risk import define_time_zero, build_competing_risk_events
from library.ehr_outcomes.survival_dataset import build_wide_survival, build_long_survival
from library.ehr_outcomes.export_tools import save_cases, save_controls, save_parquet
import pandas as pd
from config.config import outcomes

def define_outcomes_ehr(df:pd.DataFrame=None):

    # 2. Outcome flags
    df = add_outcome_flags(df)

    # 3. Neuro-exclusion and clean control definition
    df = define_controls(df=df, outcomes=outcomes)

    # 4. Time-zero + competing risk model
    df = define_time_zero(df, "enroll_date")
    df = build_competing_risk_events(df)

    # 5. Survival datasets
    wide = build_wide_survival(df)
    long = build_long_survival(df)

    # 6. Export
    save_parquet(wide, "survival_wide.parquet", "outputs/")
    save_parquet(long, "survival_long.parquet", "outputs/")
    save_controls(df, "outputs/")

    for outcome in [
        "Outcome_1a_PD_only",
        "Outcome_1b_PD_AD",
        "Outcome_1c_PD_Dementia",
        "Outcome_2a_OtherDementia",
        "Outcome_2b_PD_AD",
        "Outcome_2c_PD_OtherDementia",
    ]:
        save_cases(df, outcome, "outputs/")


if __name__ == "__main__":
    define_outcomes_ehr(df=None)