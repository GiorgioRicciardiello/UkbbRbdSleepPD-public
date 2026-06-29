"""
This module generates separate plots for different outcomes. It visualizes the
methods' predictions and outcomes across validation and non-validation cohorts.
Plots are tailored for each outcome, with rows representing methods and columns
representing validation and non-validation data splits.

Functions:
- plot_methods_separately_per_outcome: Creates figures with histograms and
  threshold-based risk group divisions for each outcome.

"""

from config.config import config, outcomes
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from library.risk.risk_helpers import (load_and_normalize_thresholds,
                                        print_thresholds,
                                        make_subject_level,
                                       plot_rbd_thresholds_methods_separately_per_outcome)





if __name__ == "__main__":
    df_risk = pd.read_parquet(config["pp"]["rbd_pred_diag"])


    thresholds = load_and_normalize_thresholds(config["pp"]["thresholds"])
    print_thresholds(thresholds)
    val_flags = {o: df_risk["val"] == 1 for o in outcomes}
    df_subj = make_subject_level(df_risk, id_col="eid", prob_col="prob_mean")

    df_subj = df_subj[~df_subj['train_sleep']]
    plot_rbd_thresholds_methods_separately_per_outcome(
        df=df_subj,
        outcomes=outcomes,
        thresholds=thresholds,
        prob_col="rbd_prob",
        save_path=None
    )