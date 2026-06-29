import pandas as pd
import numpy as np

class IncidencePrevalenceCalculator:
    """
    Compute incidence and prevalence by age bands for UK Biobank-style data.
    """

    def __init__(self, df, birth_col="34-0.0", diag_date_col="131022-0.0", censor_col=None):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Input dataset with at least birth year and diagnosis date.
        birth_col : str
            Column containing year of birth (UKB: 34-0.0).
        diag_date_col : str
            Column containing date of first diagnosis (e.g. 131022-0.0 = G20).
        censor_col : str, optional
            Column with last follow-up date / censoring date if available.
        """
        self.df = df.copy()
        self.birth_col = birth_col
        self.diag_date_col = diag_date_col
        self.censor_col = censor_col

        # Convert diagnosis dates to datetime
        self.df[self.diag_date_col] = pd.to_datetime(self.df[self.diag_date_col], errors="coerce")

        if censor_col and censor_col in df.columns:
            self.df[censor_col] = pd.to_datetime(self.df[censor_col], errors="coerce")
        else:
            # fallback censoring: today
            self.df["censor_date"] = pd.to_datetime("today")
            self.censor_col = "censor_date"

    def age_at(self, col_date):
        """Compute age at a given date col (diagnosis or censoring)."""
        return (self.df[col_date].dt.year - self.df[self.birth_col]).astype("float")

    def compute(self, age_bands):
        """
        Calculate incidence and prevalence for custom age bands.

        Parameters
        ----------
        age_bands : list of tuples
            Each tuple is (start_age, end_age).

        Returns
        -------
        pd.DataFrame
            Results table with incidence, prevalence counts and proportions.
        """
        results = []
        diag_age = self.age_at(self.diag_date_col)
        censor_age = self.age_at(self.censor_col)

        for start, end in age_bands:
            band_mask = (censor_age >= start)  # people who reached this age
            eligible = self.df[band_mask]

            # Prevalence = already diagnosed before or during the band
            prev_cases = ((diag_age < end) & (diag_age.notna()))[band_mask].sum()

            # Incidence = diagnosed within the band interval
            inc_cases = ((diag_age >= start) & (diag_age < end))[band_mask].sum()

            results.append({
                "age_band": f"{start}-{end}",
                "n_population": eligible.shape[0],
                "prevalence_cases": prev_cases,
                "prevalence_prop": prev_cases / eligible.shape[0] if eligible.shape[0] else np.nan,
                "incidence_cases": inc_cases,
                "incidence_rate": inc_cases / eligible.shape[0] if eligible.shape[0] else np.nan
            })

        return pd.DataFrame(results)
