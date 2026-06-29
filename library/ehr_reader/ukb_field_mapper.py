"""
UK Biobank Field Mapper

This module provides the UkbFieldMapper class for identifying and mapping
UK Biobank field codes to their corresponding CSV files.
"""

from typing import Dict, Set, Tuple
from pathlib import Path
import pandas as pd
import warnings
import re

class UkbFieldMapper:
    """
    Handles field identification and CSV mapping for UK Biobank data.

    This class extracts field codes of interest from the data dictionary,
    maps them to their corresponding CSV files, and identifies which fields
    are available in the dataset.

    Attributes:
        df_data_dictionary: UK Biobank data dictionary DataFrame
        csv_files: Dictionary mapping CSV names to their text file paths
        field_groups: Dictionary mapping category names to sets of field names
        all_field_ids: Set of all field IDs to extract
        df_matched: DataFrame with matched fields and their CSV locations
    """

    def __init__(self, df_data_dictionary: pd.DataFrame, csv_files: Dict[str, Path]):
        """
        Initialize the UkbFieldMapper.

        Args:
            df_data_dictionary: UK Biobank data dictionary DataFrame
            csv_files: Dictionary mapping CSV names to their text file paths
        """
        self.df_data_dictionary = df_data_dictionary
        self.csv_files = csv_files
        self.field_groups = None
        self.all_field_ids = None
        self.df_matched = None

    def get_codes_of_interest(self) -> Tuple[Dict[str, Set[str]], Set[str]]:
        """
        Extract sets of UK Biobank field codes of interest from the data dictionary.

        Returns:
            Tuple of (field_code_groups, all_field_ids) where:
                - field_code_groups: Dictionary mapping category name -> set of field names
                - all_field_ids: Set of all field names to extract
        """
        # Clean and normalize
        df = self.df_data_dictionary.copy()
        df = df.loc[~df["field_code"].isna()]
        df["field_code"] = df["field_code"].astype(int)

        # Helper for folder-based extraction
        def codes_from_folder(folder_path: str,
                              column:str='folder_path',
                              search:str='exact') -> Set[int]:
            if search == 'exact':
                return set(
                    df.loc[df[column] == folder_path, "field_code"]

                    .dropna()
                    .astype(int)
                    .tolist()
                )
            else:
                # literal string match so we disable regex
                return set(
                    df.loc[df[column].str.contains(folder_path,  regex=False), "field_code"]
                    .dropna()
                    .astype(int)
                    .tolist()
                )

        def codes_from_substring(
                df: pd.DataFrame,
                substrings: Tuple[str, ...],
                search_cols: Tuple[str, ...] = ("name", "folder_path"),
                case_insensitive: bool = True,
        ) -> Set[int]:
            """
            Extract field_codes where any of the given substrings appear
            in specified columns (e.g. name, folder_path).

            Args:
                df: UKB data dictionary dataframe
                substrings: substrings to search for (e.g. ('accelerometer', 'acceleration'))
                search_cols: columns to search in
                case_insensitive: whether matching is case-insensitive

            Returns:
                Set of matching field_codes
            """
            flags = 0 if not case_insensitive else re.IGNORECASE
            pattern = "|".join(map(re.escape, substrings))

            mask = False
            for col in search_cols:
                if col in df.columns:
                    mask = mask | df[col].astype(str).str.contains(pattern, flags=flags, regex=True)

            return set(df.loc[mask, "field_code"].dropna().astype(int))

        # ------------------------------------------------------------------
        # Explicit field groups
        # ------------------------------------------------------------------
        demographic_field_ids = {
            31, 33, 34, 52, 53,
            21000, 21001, 21002, 21003, 21022
        }

        icd10_field_ids = {41202, 41204}

        work_shift_employment = {22650, 3426, 826}

        # ------------------------------------------------------------------
        # Folder-based groups
        # ------------------------------------------------------------------
        reception = codes_from_folder(
            "Assessment centre > Recruitment > Reception"
        )

        fluid_reasoning = codes_from_folder(
            "Assessment centre > Cognitive function > Fluid intelligence / reasoning"
        )

        reaction_time = codes_from_folder(
            "Assessment centre > Cognitive function > Reaction time"
        )

        sleep_assessment = codes_from_folder(
            "Assessment centre > Touchscreen > Lifestyle and environment > Sleep"
        )

        physical_activity = codes_from_folder(
            "Assessment centre > Touchscreen > Lifestyle and environment > Physical activity"
        )

        diagnosis = codes_from_folder(
            'Health-related outcomes > Hospital inpatient > Summary Diagnoses'
        )

        memory = codes_from_folder('Online follow-up > Cognitive function online > Numeric memory'
                                   )
        pairs_matching = codes_from_folder('Online follow-up > Cognitive function online > Pairs matching')

        impedance_bmi = codes_from_folder(folder_path='Impedance of whole body',
                                          column='title',
                                          search='contains')
        bmi_manual = codes_from_folder(folder_path='Body mass index (BMI)',
                                          column='title',
                                          search='contains')

        smoking = codes_from_folder(folder_path='Assessment centre > Touchscreen > Lifestyle and environment > Smoking',
                                    column='folder_path',
                                    search='exact'
                                    )

        alcohol = codes_from_folder(folder_path='Assessment centre > Touchscreen > Lifestyle and environment > Alcohol',
                                    column='folder_path',
                                    search='exact'
                                    )
        medication = codes_from_folder(folder_path='Medication',
                                       column='folder_path',
                                       search='contains')

        medical_conditions = codes_from_folder(folder_path='Assessment centre > Verbal interview > Medical conditions',
                                               column='folder_path',
                                               search='exact')

        death = codes_from_folder(folder_path='Health-related outcomes > Death register',
                                  column='folder_path',
                                  search='exact')

        sleep_dist = codes_from_folder(folder_path='Online follow-up > Sleep > Sleep disturbances',
                                  column='folder_path',
                                  search='exact')

        trail_making_title = codes_from_folder(folder_path='Trail making',
                                         column='title',
                                         search='contains')

        trail_making_folder_path = codes_from_folder(folder_path='Trail making',
                                         column='folder_path',
                                         search='contains')
        trail_making = trail_making_title.union(trail_making_folder_path)

        # algo defined outcomes for pd, ad and vascular dementia, our main outcomes
        algo_pd = {42032, 42033}
        algo_ad = {42020, 42021}
        algo_vascular_dementia = {42022, 42023}
        # because algo define are not until 2025, we include the followings that are up to 2025,
        # the idea is to use algo define until we have no more and the rest comes from these ones:
        pd_g20 = codes_from_folder(folder_path='G20',
                                  column='title',
                                  search='contains')

        ad_g30 = codes_from_folder(folder_path='G30',
                                  column='title',
                                  search='contains')
        dem_fo1 = {130838, 130839}  # Date and source for vascular dementia

        outcome_pd = algo_pd.union(pd_g20)
        outcome_ad = algo_ad.union(ad_g30)
        outcome_dem = algo_vascular_dementia.union(dem_fo1)  # vascular dementia outcome

        # ------------------------------------------------------------------
        # Range-based groups
        # ------------------------------------------------------------------
        symbol_digit_substitution_correct = set(df.loc[df['field_code'] == 20159, 'field_code'])
        symbol_digit_substitution_attempted = set(df.loc[df['field_code'] == 20195, 'field_code'])
        symbol_digit_substitution_duration = set(df.loc[df['field_code'] == 20230, 'field_code'])
        symbol_digit_substitution_matches_correct = set(df.loc[df['field_code'] == 23324, 'field_code'])
        symbol_digit = set().union(
            symbol_digit_substitution_correct,
            symbol_digit_substitution_attempted,
            symbol_digit_substitution_duration,
            symbol_digit_substitution_matches_correct
        )

        prospective_memory = set(df.loc[df['field_code'] == 6373, 'field_code'])

        mean_time_correct_matches = set(df.loc[df['field_code'] == 20023, 'field_code'])
        number_fi_questions_within_time_limit = set(df.loc[df['field_code'] == 20128, 'field_code'])
        accelerometry = codes_from_substring(
            df,
            substrings=(
                "accelerometer",
                "acceleration",
                "accelerometry",
                "Axial acceleration",
            ),
            search_cols=("name", "folder_path"),
        )

        matrix_pattern_completion = set(df.loc[df['field_code'] == 20159, 'field_code'])

        gp_prescriptions_irs = set(df.loc[df['field_code'] == 42039, 'field_code'])


        # ------------------------------------------------------------------
        # From field code to name in dataset
        # ------------------------------------------------------------------
        dem_names = self.df_data_dictionary.loc[
            self.df_data_dictionary['field_code'].isin(demographic_field_ids), 'name'
        ].tolist()

        diag_name = self.df_data_dictionary.loc[
            self.df_data_dictionary['field_code'].isin(diagnosis), 'name'
        ].tolist()

        work_name = self.df_data_dictionary.loc[
            self.df_data_dictionary['field_code'].isin(work_shift_employment), 'name'
        ].tolist()

        reception_name = self.df_data_dictionary.loc[
            self.df_data_dictionary['field_code'].isin(reception), 'name'
        ].tolist()

        fluid_name = self.df_data_dictionary.loc[
            self.df_data_dictionary['field_code'].isin(fluid_reasoning), 'name'
        ].tolist()

        reaction_name = self.df_data_dictionary.loc[
            self.df_data_dictionary['field_code'].isin(reaction_time), 'name'
        ].tolist()

        sleep_name = self.df_data_dictionary.loc[
            self.df_data_dictionary['field_code'].isin(sleep_assessment), 'name'
        ].tolist()

        physical_name = self.df_data_dictionary.loc[
            self.df_data_dictionary['field_code'].isin(physical_activity), 'name'
        ].tolist()

        trail_making_name = self.df_data_dictionary.loc[
            self.df_data_dictionary['field_code'].isin(trail_making), 'name'
        ].tolist()

        accel_name = self.df_data_dictionary.loc[
            self.df_data_dictionary["field_code"].isin(accelerometry), "name"
        ].tolist()

        memory_name = self.df_data_dictionary.loc[
            self.df_data_dictionary["field_code"].isin(memory), "name"
        ].tolist()

        pairs_matching_name = self.df_data_dictionary.loc[
            self.df_data_dictionary["field_code"].isin(pairs_matching), "name"
        ].tolist()

        symbol_digit_name = self.df_data_dictionary.loc[
            self.df_data_dictionary["field_code"].isin(symbol_digit), "name"
        ].tolist()

        prospective_memory_name = self.df_data_dictionary.loc[
            self.df_data_dictionary["field_code"].isin(prospective_memory), "name"
        ].tolist()

        matrix_pattern_completion_name = self.df_data_dictionary.loc[
            self.df_data_dictionary["field_code"].isin(matrix_pattern_completion), "name"
        ].tolist()

        mean_time_correct_matches_name = self.df_data_dictionary.loc[
            self.df_data_dictionary["field_code"].isin(mean_time_correct_matches), "name"
        ].tolist()

        number_fi_questions_within_time_limit_name = self.df_data_dictionary.loc[
            self.df_data_dictionary["field_code"].isin(number_fi_questions_within_time_limit), "name"
        ].tolist()

        impedance_bmi_name = self.df_data_dictionary.loc[
            self.df_data_dictionary["field_code"].isin(impedance_bmi), "name"
        ].tolist()

        bmi_manual_name = self.df_data_dictionary.loc[
            self.df_data_dictionary["field_code"].isin(bmi_manual), "name"
        ].tolist()

        alcohol_name = self.df_data_dictionary.loc[
            self.df_data_dictionary["field_code"].isin(alcohol), "name"
        ].tolist()

        smoking_name = self.df_data_dictionary.loc[
            self.df_data_dictionary["field_code"].isin(smoking), "name"
        ].tolist()

        medication_name = self.df_data_dictionary.loc[
            self.df_data_dictionary["field_code"].isin(medication), "name"
        ].tolist()

        death_name = self.df_data_dictionary.loc[
            self.df_data_dictionary["field_code"].isin(death), "name"
        ].tolist()

        medical_conditions_name = self.df_data_dictionary.loc[
            self.df_data_dictionary["field_code"].isin(medical_conditions), "name"
        ].tolist()

        outcome_pd_name = self.df_data_dictionary.loc[
            self.df_data_dictionary["field_code"].isin(outcome_pd), "name"
        ].tolist()

        outcome_ad_name = self.df_data_dictionary.loc[
            self.df_data_dictionary["field_code"].isin(outcome_ad), "name"
        ].tolist()

        outcome_dem_name = self.df_data_dictionary.loc[
            self.df_data_dictionary["field_code"].isin(outcome_dem), "name"
        ].tolist()

        sleep_dist_name = self.df_data_dictionary.loc[
            self.df_data_dictionary["field_code"].isin(sleep_dist), "name"
        ].tolist()

        gp_prescriptions_irs_name = self.df_data_dictionary.loc[
            self.df_data_dictionary["field_code"].isin(gp_prescriptions_irs), "name"
        ].tolist()

        # ------------------------------------------------------------------
        # Combine and return
        # ------------------------------------------------------------------
        field_code_groups = {
            "demographics": set(dem_names),
            "icd10": set(diag_name),
            "work_shift_employment": set(work_name),
            "fluid_reasoning": set(fluid_name),
            'reception': set(reception_name),
            "reaction_time": set(reaction_name),
            "sleep_assessment": set(sleep_name),
            "physical_activity": set(physical_name),
            "trail_making_name": set(trail_making_name),
            'prospective_memory': set(prospective_memory_name),
            'matrix_pattern_completion': set(matrix_pattern_completion_name),
            'mean_time_correct_matches': set(mean_time_correct_matches_name),
            'number_fi_questions_within_time_limit_name': set(number_fi_questions_within_time_limit),
            "accelerometry": set(accel_name),
            'alcohol': set(alcohol_name),
            'smoking': set(smoking_name),
            "memory": set(memory_name),
            "pairs_matching_name": set(pairs_matching_name),
            "symbol_digit_name": set(symbol_digit_name),
            'impedance_bmi': set(impedance_bmi_name),
            'bmi_manual': set(bmi_manual_name),
            'death': set(death_name),
            'medication': set(medication_name),
            'medical_conditions': set(medical_conditions_name),
            'gp_prescriptions_irs': set(gp_prescriptions_irs_name),
            'outcome_pd': set(outcome_pd_name),
            'outcome_ad': set(outcome_ad_name),
            'outcome_dem': set(outcome_dem_name),
            'sleep_dist_name': set(sleep_dist_name),
        }

        print("Field code groups:")
        n = 0
        for field, values in field_code_groups.items():
            print(f"\t{field}: {len(values):,}")
            n += len(values)
        print(f"Total columns: {n:,}")

        all_field_ids: Set[str] = set().union(*field_code_groups.values())

        # Store for later use
        self.field_groups = field_code_groups
        self.all_field_ids = all_field_ids

        return field_code_groups, all_field_ids

    def map_fields_to_csv(self, all_field_ids: Set[str]) -> pd.DataFrame:
        """
        Map field IDs to their corresponding CSV files and return matched fields.

        Args:
            all_field_ids: Set of field IDs to map

        Returns:
            DataFrame with columns: csv_name, column_name, is_match
        """
        df_cols_avail = self._fields_to_frame()
        df_matched = self._match_fields_to_frame(df_cols_avail, all_field_ids)
        df_matched = df_matched[df_matched['is_match'] == True].copy()
        self.df_matched = df_matched
        return df_matched

    def get_matched_fields(self) -> pd.DataFrame:
        """Return the DataFrame of matched fields."""
        return self.df_matched

    def _fields_to_frame(self) -> pd.DataFrame:
        """Get all columns of each .txt file mapped into a single dataframe."""
        print("\nExtracting column fields from .txt files...")
        rows = []

        for csv_name, txt_file in self.csv_files.items():
            if not txt_file.exists():
                print(f"Warning: {txt_file} not found, skipping...")
                continue

            print(f"Reading column names from {txt_file}...")

            with open(txt_file, "r") as f:
                columns = [line.strip() for line in f if line.strip()]

            for col in columns:
                rows.append({"csv_name": csv_name, "column_name": col})

        return pd.DataFrame(rows)

    def _match_fields_to_frame(
        self,
        df_cols_avail: pd.DataFrame,
        all_field_ids: Set[str],
        field_col: str = "column_name",
    ) -> pd.DataFrame:
        """Add columns indicating whether a field was requested and matched."""
        df = df_cols_avail.copy()
        df["is_match"] = df[field_col].isin(all_field_ids)

        matched_count = df["is_match"].sum()
        requested_count = len(all_field_ids)

        print(f'\nFrom all fields to search: {requested_count:,}')
        print(f'We discovered {matched_count:,} fields.')

        if matched_count != requested_count:
            missing = requested_count - matched_count
            warnings.warn(
                f'{missing} fields were not found. '
                f'Remember to append with the current data so they match'
            )

        return df
