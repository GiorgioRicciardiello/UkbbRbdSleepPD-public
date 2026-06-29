"""
This module provides functions for statistical analysis and visualization
of risk stratification data, including computation of risk ratio, confidence
intervals, p-values, and creation of forest plots for visualizing results.

Functions:
- compute_rr_ci: Compute the risk ratio (RR) and 95% confidence interval (CI)
  using a standard log method for a 2×2 contingency table.
- fisher_p: Perform Fisher's Exact Test to calculate p-values for contingency
  tables.
- forest_cell_plot: Generate a forest plot for a single contingency table
  and annotate it with detailed statistical information.
- forest_panels_per_outcome: Create comprehensive forest plots for multiple
  methods and outcomes, arranged in panels, and optionally save the results
  to the specified location.

The module is tailored to process dataframes containing risk and control
information, and visualize results stratified by validation status.

Dependencies include libraries such as pandas, matplotlib, numpy, and scipy.
"""
from config.config import config, outcomes
import pandas as pd
from library.risk.risk_helpers import (make_subject_level,
                                       load_and_normalize_thresholds,
                                       print_thresholds,
                                       )
from library.risk.forest_plot import forest_panels_per_outcome


if __name__ == "__main__":
    df_risk = pd.read_parquet(config["pp"]["rbd_pred_diag"])

    thresholds = load_and_normalize_thresholds(config["pp"]["thresholds"])
    print_thresholds(thresholds)
    val_flags = {o: df_risk["val"] == 1 for o in outcomes}
    df_subj = make_subject_level(df_risk, id_col="eid", prob_col="prob_mean")

    df_subj = df_subj[~df_subj['train_sleep']]
    forest_panels_per_outcome(
        df=df_subj,
        outcomes=outcomes,
        thresholds=thresholds,
        prob_col="rbd_prob",
        save_path=None
    )