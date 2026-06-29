from config.config import config, outcomes
import pandas as pd
from library.risk.risk_helpers import (make_subject_level,
                                       load_and_normalize_thresholds,
                                       )
from library.risk.survival_analysis import survival_panels_per_outcome, survival_attrition_summary
from library.column_registry import METHOD_TO_RISK_SUFFIX


if __name__ == "__main__":
    df_risk = pd.read_parquet(config["pp"]["rbd_pred_diag"])
    COMPUTE_ATTRITION = True
    thresholds = load_and_normalize_thresholds(config["pp"]["thresholds"])
    # print_thresholds(thresholds)
    val_flags = {o: df_risk["val"] == 1 for o in outcomes}
    df_subj = make_subject_level(df_risk, id_col="eid", prob_col="prob_mean")

    df_subj = df_subj[~df_subj['train_sleep']]
    if COMPUTE_ATTRITION:
        attrition_summary = pd.DataFrame()
        for outcome in outcomes:
            df_attrition = survival_attrition_summary(df_subj, outcome)
            attrition_summary = pd.concat([attrition_summary, df_attrition])

    survival_stats = survival_panels_per_outcome(
        df=df_subj,
        outcomes=outcomes,
        # thresholds=thresholds,
        # prob_col="rbd_prob",
        methods=["percentile", "roc", "pr", "f1", "surv", "quartile"]
,
        save_path=None
    )

