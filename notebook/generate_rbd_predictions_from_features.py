"""
===============================================================================
Sleep Prediction Pipeline
===============================================================================

This script loads pre-trained diagnostic models, applies them to extracted
sleep-related features, and generates prediction reports. It supports both
per-epoch and per-subject majority-voted classification, and outputs summary
statistics in tabular and CSV formats.



Modules:
- Loads models from pickle files
- Cleans and prepares feature data
- Generates probabilistic and hard predictions
- Computes overall and grouped prediction summaries
- Saves results to disk for downstream analysis

Author: Giorgio
Date: September 2025
Usage: Run as a standalone script after configuring paths in config/config.py
===============================================================================
"""
from config.config import features, config
import pickle
import numpy as np
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from tabulate import tabulate


def load_model(model_path:Path):
    print(f"Loading model from {model_path} ...")
    with open(str(model_path), "rb") as f:
        model = pickle.load(f)

    # print model properties
    print(getattr(model, '__sklearn_version__', 'Not available'))
    print(type(model.estimators_[0][0]))
    print(dir(model.estimators_[0][0]))
    print(type(model))
    print(model.n_features_in_)
    print(model.feature_importances_)
    print(model.feature_names_in_)

    return model



def report_predictions(df: pd.DataFrame,
                       group_col: str = "ID",
                       pred_col: str = "prediction",
                       output_path: Path = None):
    """
    Report overall prediction counts and percentages,
    and compute majority-voted prediction per group.
    Optionally save both summaries to a single CSV file.

    :param df: DataFrame containing predictions
    :param group_col: Column to group by (e.g., subject ID or night)
    :param pred_col: Column with prediction values
    :param output_path: Optional Path to save combined CSV report
    """
    # Overall summary
    print("🔹 Overall prediction counts and percentages:\n")
    counts = df[pred_col].value_counts()
    percentages = df[pred_col].value_counts(normalize=True) * 100
    df_summary_across_nights = pd.DataFrame({
        'Prediction': counts.index,
        'Count': counts.values,
        'Percentage (%)': percentages.values.round(2),
        'Method': 'Across nights'
    })
    print(tabulate(df_summary_across_nights, headers='keys', tablefmt='github', showindex=False))
    print()

    # Majority vote summary
    df_summary_single_subject = pd.DataFrame()
    if group_col in df.columns:
        majority = (
            df.groupby(group_col)[pred_col]
            .agg(lambda x: x.value_counts().idxmax())
            .reset_index()
            .rename(columns={pred_col: "Majority Prediction"})
        )

        counts = majority["Majority Prediction"].value_counts()
        percentages = majority["Majority Prediction"].value_counts(normalize=True) * 100
        df_summary_single_subject = pd.DataFrame({
            'Prediction': counts.index,
            'Count': counts.values,
            'Percentage (%)': percentages.values.round(2),
            'Method': 'Within Subject Majority Vote'
        })

        print("🔹 Majority-voted prediction counts and percentages:\n")
        print(tabulate(df_summary_single_subject, headers='keys', tablefmt='github', showindex=False))
        print()

    # Combine and save
    df_summary = pd.concat([df_summary_across_nights, df_summary_single_subject], axis=0)

    if output_path:
        output_path = output_path.joinpath("rbd_prediction_summary.csv")
        df_summary.to_csv(output_path, index=False)
        print(f"📁 Saved summary to: {output_path}")


if __name__ == "__main__":
    # %% Paths
    path_model_rar = config.get('rar_rbd_models')['rar_model']
    path_model_sleep = config.get('rar_rbd_models')['rar_sleep']

    path_ehr = config.get('paths')['data_sheet']['dir_parquet']
    path_sleep_features = config.get('paths')['actig_extracted']['merged_sleep']
    path_abk_rbd_scores= config.get('paths')['actig_extracted']['rbd_scores']
    # path_features = config.get('paths')['actig_extracted']['merged_sleep']
    path_out_rbd_scores = config.get('pp')['rbd_scores']

    # ehr_split_flags_features = config["pp"]["ehr_split_flags_features"]    # -> created in generate dataset, unique eid nights id, important
    # ehr_split_flags_features_rbd = config["pp"]["ehr_split_flags_features_rbd"]
    # %% Load models
    model_rar = load_model(path_model_rar)
    model_sleep = load_model(path_model_sleep)

    # %% load the data
    df_ehr = pd.read_parquet(path_ehr)
    # IDS WITH ACTIGRAPHY - get only the ids with acceleration data from the UKKB

    ehr_eids = df_ehr.loc[df_ehr['wear_time_start'].notna(), 'eid']

    # SLEEP FEATURES -
    df_sleep_features = pd.read_parquet(path_sleep_features)
    df_sleep_features['eid'] = df_sleep_features['eid'].astype(int)
    df_sleep_features = df_sleep_features.reset_index(drop=True)
    # ABK RBD SCORE -
    df_abk_rbd_scores = pd.read_parquet(path_abk_rbd_scores)
    df_abk_rbd_scores['eid'] = df_abk_rbd_scores['eid'].astype(int)

    # %% Filter the data
    df_sleep_features = df_sleep_features.loc[df_sleep_features['eid'].isin(ehr_eids), :]
    df_features = df_sleep_features[features]
    df_features = df_features.replace([np.inf, -np.inf], np.nan).fillna(0)

    # %% Remove bad nights based on featues
    # T_avg >= 27, 12 >= TST >= 3, nonwear < 2, exclude_var

    # %% Generate the predictions
    if not config.get('pp')['rbd_scores'].exists():
        print(f'Computing RBD Predictions....')
        predictions = model_sleep.predict_proba(df_features)
        # df_pred = pd.DataFrame(predictions, columns=['PredictionClass_0', 'PredictionClass_1'])
        # df_pred['prediction'] = model_sleep.predict(df_features)

        df_pred = pd.DataFrame({
            'eid': df_sleep_features['eid'].values,
            'ID': df_sleep_features['ID'].values,
            'Date': df_sleep_features['Date'].values,
            'PredictionClass_0': predictions[:, 0],  # -> Control class
            'PredictionClass_1': predictions[:, 1],  # -> iRBD class
            'prediction': model_sleep.predict(df_features)
        })

        for df in [df_sleep_features, df_abk_rbd_scores]:
            df['Date'] = df['Date'].astype(int)
            df['eid'] = df['eid'].astype(int)
            df['ID'] = df['ID'].astype(str).str.strip()

        print("Duplicates in df_sleep_features:",
              df_sleep_features.duplicated(['eid', 'ID', 'Date']).sum())

        print("Duplicates in df_pred:",
              df_pred.duplicated(['eid', 'ID', 'Date']).sum())

        print("Duplicates in df_abk:",
              df_abk_rbd_scores.duplicated(['eid', 'ID', 'Date']).sum())


        # include abk rbd scores
        df_pred_id_abk = pd.merge(
            left=df_pred,
            right=df_abk_rbd_scores[['eid', 'ID', 'Date', 'visit_number', 'iRBD_Sleep_Score']],
            on=['eid', 'ID', 'Date'],
            how='left'
        )

        # now we want to give the abk rbd scores to the predictions
        df_pred_id['Date'] = df_pred_id['Date'].astype(int)
        df_pred_id['eid'] = df_pred_id['eid'].astype(int)
        df_pred_id['ID'] = df_pred_id['ID'].astype(str).str.strip()

        df_abk_rbd_scores['Date'] = df_abk_rbd_scores['Date'].astype(int)
        df_abk_rbd_scores['eid'] = df_abk_rbd_scores['eid'].astype(int)
        df_abk_rbd_scores['ID'] = df_abk_rbd_scores['ID'].astype(str).str.strip()

        df_pred_id_abk = pd.merge(
            left=df_pred_id,
            right=df_abk_rbd_scores[['eid', 'ID', 'Date', 'visit_number','iRBD_Sleep_Score']],
            on=['eid', 'ID', 'Date'],
            how='left'
        )

        report_predictions(df=df_pred_id_abk,
                           output_path=path_out_rbd_scores.parent)

        df_pred_id_abk.rename(columns={
                                'prediction': 'rbd_bin',
                                 'PredictionClass_0': 'rbd_prob_class0',
                                 'PredictionClass_1': 'rbd_prob_class1',
                                'iRBD_Sleep_Score': 'abk_rbd_score'}, inplace=True)

        df_pred_id_abk.columns = [col.lower() for col in df_pred_id_abk.columns]


        df_pred_id_abk.to_parquet(path_out_rbd_scores)
        print(f'Predictions saved in parquet file')
    else:
        print(f'Predictions already exist. Loading from file')
        df_pred_id_abk = pd.read_parquet(path_out_rbd_scores)
        report_predictions(df=df_pred_id_abk)


    # %% ========== COMPARE THE SCORES BETWEEN KATARIINA RBD AND ANDREAS
    # generate plot to compare the rbd

    df = df_pred_id_abk.dropna(subset=['abk_rbd_score'])


    df_subj = (
        df.groupby('id', as_index=False)
        .agg(
            rbd_prob_mean=('rbd_prob_class1', 'mean'),
            abk_rbd_score_mean=('abk_rbd_score', 'mean'),
            rbd_bin=('rbd_bin', 'max')  # ever-positive
        )
    )
    # logits from the probabilities
    eps = 1e-6  # small constant

    p = df_subj['rbd_prob_mean'].clip(eps, 1 - eps)
    df_subj['rbd_logit'] = np.log(p / (1 - p))




    from matplotlib import pyplot as plt
    import seaborn as sns

    # %%
    plt.figure(figsize=(6, 6))
    for label, color in [(0, 'tab:blue'), (1, 'tab:orange')]:
        mask = df_subj['rbd_bin'] == label
        plt.scatter(
            x=df_subj.loc[mask, 'rbd_prob_mean'],
            y=df_subj.loc[mask, 'abk_rbd_score_mean'],
            s=3,
            alpha=0.25,
            label=f'rbd_bin = {label}',
            c=color
        )
    plt.xlabel('Mean RBD probability')
    plt.ylabel('ABK RBD score mean')
    plt.title('RBD probability vs ABK RBD score (subject-level)')
    plt.legend(markerscale=3, frameon=False)
    plt.show()

    #%% Distribution of RBD scores hue by binary predictions from Kat model
    # Goal: Show how good is the separation between ABK RBD scores predictions and Kat model
    plt.figure(figsize=(7, 4))
    sns.kdeplot(
        data=df_subj,
        x='abk_rbd_score_mean',
        hue='rbd_bin',
        common_norm=False,
        fill=True,
        alpha=0.35
    )
    plt.xlabel('Mean ABK RBD sleep score (subject-level)')
    plt.ylabel('Density (within each RBD group)')
    plt.title('RBD sleep score distribution stratified by RBD prediction')
    plt.legend(title='RBD binary prediction', labels=['Negative (0)', 'Positive (1)'])
    plt.show()

    # %% Ambulatory disorder quadrant plot
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt


    def plot_rbd_quadrant_ambulatory_disorder(
            df,
            x_col,
            prob_col='rbd_prob_mean',
            bin_col='rbd_bin',
            use_logit=True,
            x_thr=0.0,
            prob_thr=0.5,
            eps=1e-6,
            figsize=(6, 6)
    ):
        """
        Quadrant plot comparing IRBD score vs RBD model output.

        Parameters
        ----------
        df : pd.DataFrame
            Subject-level dataframe
        x_col : str
            Column for IRBD / ABK score (x-axis)
        prob_col : str
            Probability column (used if use_logit=False)
        bin_col : str
            Binary RBD label (0/1)
        use_logit : bool
            If True, use logit(probability) on y-axis
        x_thr : float
            Threshold on x-axis
        prob_thr : float
            Probability threshold (converted to logit if needed)
        eps : float
            Clipping value for probabilities
        figsize : tuple
            Figure size
        """

        x = df[x_col]

        if use_logit:
            p = df[prob_col].clip(eps, 1 - eps)
            y = np.log(p / (1 - p))
            y_thr = np.log(prob_thr / (1 - prob_thr))
            y_label = 'RBD model logit score'
        else:
            y = df[prob_col]
            y_thr = prob_thr
            y_label = 'RBD prediction probability'

        plt.figure(figsize=figsize)

        # Controls
        m0 = df[bin_col] == 0
        plt.scatter(
            x[m0],
            y[m0],
            s=20,
            facecolors='none',
            edgecolors='tab:blue',
            label='Control'
        )

        # Cases
        m1 = df[bin_col] == 1
        plt.scatter(
            x[m1],
            y[m1],
            s=25,
            marker='D',
            c='tab:orange',
            alpha=0.8,
            label='RBD case'
        )

        # Threshold lines
        plt.axvline(x_thr, ls='--', c='k', lw=1)
        plt.axhline(y_thr, ls='--', c='k', lw=1)

        # Quadrant shading
        xlim = plt.xlim()
        ylim = plt.ylim()

        plt.fill_betweenx(
            [y_thr, ylim[1]],
            x_thr, xlim[1],
            color='lightgrey',
            alpha=0.3
        )
        plt.fill_betweenx(
            [ylim[0], y_thr],
            xlim[0], x_thr,
            color='lightgrey',
            alpha=0.3
        )

        # Labels
        plt.xlabel('IRBD sleep score (actigraphy)')
        plt.ylabel(y_label)
        plt.title('Agreement between IRBD score and RBD model')

        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()

        # Quadrant counts
        q = pd.DataFrame({
            'IRBD_pos': x > x_thr,
            'RBD_pos': y > y_thr
        })

        return q.value_counts()


    q_counts = plot_rbd_quadrant_ambulatory_disorder(
        df=df_subj,
        x_col='abk_rbd_score_mean',
        prob_col='rbd_prob_mean',
        bin_col='rbd_bin',
        use_logit=True,
        x_thr=0.0,
        prob_thr=0.4
    )
    df_subj[['rbd_logit', 'abk_rbd_score_mean']].corr(method='spearman')



    # %% logit vs score error plot
    from typing import Dict, Tuple
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


    def plot_logit_score_fit(
            df: pd.DataFrame,
            x_col: str,
            y_col: str,
            bin_col: str = 'rbd_bin',
            subject_col: str = 'eid',
            poly_order: int = 3,
            figsize: Tuple[int, int] = (8, 4)):
        """
        Fit a polynomial regression between score and logit, compute ML-style
        regression errors and correlations, and plot with points colored by rbd_bin.
        """

        # Drop missing
        d = df[[subject_col, x_col, y_col, bin_col]].dropna().copy()

        # Sort by score
        d = d.sort_values(x_col)

        x: np.ndarray = d[x_col].to_numpy()
        y: np.ndarray = d[y_col].to_numpy()
        b: np.ndarray = d[bin_col].to_numpy()

        # Polynomial fit (global)
        coeffs: np.ndarray = np.polyfit(x, y, poly_order)
        poly = np.poly1d(coeffs)
        y_hat: np.ndarray = poly(x)

        # Errors
        residuals: np.ndarray = y - y_hat
        abs_residuals: np.ndarray = np.abs(residuals)

        # ML regression metrics
        mae: float = mean_absolute_error(y, y_hat)
        rmse: float = mean_squared_error(y, y_hat, squared=False)
        r2: float = r2_score(y, y_hat)

        # MAE confidence interval (95%)
        mae_ci: Tuple[float, float] = stats.t.interval(
            0.95,
            len(abs_residuals) - 1,
            loc=mae,
            scale=stats.sem(abs_residuals)
        )

        # Correlations
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)

        # Plot
        plt.figure(figsize=figsize)

        for label, color, name in [
            (0, 'tab:blue', 'Control'),
            (1, 'tab:orange', 'RBD case')
        ]:
            m = b == label
            plt.scatter(
                x[m],
                y[m],
                s=10,
                alpha=0.3,
                c=color,
                label=name
            )

        plt.plot(
            x,
            y_hat,
            color='black',
            linewidth=2,
            label=f'{poly_order}rd-order fit'
        )

        plt.xlabel('IRBD / ABK score')
        plt.ylabel('RBD model logit')
        plt.title(
            f'Logit vs score ({poly_order}rd-order fit)\n'
            f'MAE={mae:.3f} '
            f'(95% CI [{mae_ci[0]:.3f}, {mae_ci[1]:.3f}]), '
            f'RMSE={rmse:.3f}, R²={r2:.3f}'
        )

        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()

        return {
            'coefficients': coeffs,
            'mae': mae,
            'mae_ci_95': mae_ci,
            'rmse': rmse,
            'r2': r2,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'residuals': residuals
        }


    out = plot_logit_score_fit(
        df=df_subj,
        x_col='abk_rbd_score_mean',
        y_col='rbd_logit',
        subject_col='id'
    )







