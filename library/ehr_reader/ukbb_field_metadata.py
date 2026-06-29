"""
UK Biobank (UKBB) Field Metadata Extractor
=========================================

This module provides a class-based interface to:
1. Extract UKBB field IDs from a dataset (CSV header).
2. Query the UKBB Showcase website to retrieve field descriptions and categories.
3. Persist the results as CSV and JSON.
4. Reload existing labels directly from CSV or JSON when available.

The implementation is rate-limited and robust to partial failures.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
from config.config import config

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------

BASE_URL: str = "https://biobank.ndph.ox.ac.uk/ukb/search.cgi"

HEADERS: Dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (UKBB-metadata-scraper/1.0)"
}


# ---------------------------------------------------------------------
# CLASS
# ---------------------------------------------------------------------

class UKBBFieldMetadataExtractor:
    """
    Extracts UK Biobank field metadata (description and category)
    from the UKBB Showcase website and manages persistence.

    Parameters
    ----------
    data_sheet_csv : Path
        Path to the CSV file containing UKBB columns (EHR or phenotype sheet).
        Only the header is read.
    output_csv : Path
        Destination path for the extracted metadata CSV.
    output_json : Path
        Destination path for the extracted metadata JSON.
    sleep_seconds : float, optional
        Delay between successive HTTP requests (rate limiting), by default 1.0.
    timeout : int, optional
        Timeout for HTTP requests in seconds, by default 20.
    """

    def __init__(
        self,
        data_sheet_csv: Path,
        output_csv: Path,
        output_json: Path,
        sleep_seconds: float = 1.0,
        timeout: int = 20,
    ) -> None:
        self.data_sheet_csv = data_sheet_csv
        self.output_csv = output_csv
        self.output_json = output_json
        self.sleep_seconds = sleep_seconds
        self.timeout = timeout

    # -----------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------

    def run(self, force: bool = False) -> pd.DataFrame:
        """
        Main entry point. Extracts field IDs, fetches metadata,
        and saves results unless already present.

        Parameters
        ----------
        force : bool, optional
            If True, ignore existing outputs and re-query UKBB, by default False.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: field_id, description, category.
        """
        field_ids = self.extract_field_ids()

        if not force:
            existing = self.load_labels()
            if existing is not None and existing.shape[0] == len(field_ids):
                return existing

        df_fields = self.build_metadata_table(field_ids)
        self.save_outputs(df_fields)
        return df_fields

    def extract_field_ids(self) -> List[int]:
        """
        Extract unique UKBB field IDs from the header of the data sheet.

        Returns
        -------
        List[int]
            Unique field IDs found in the dataset.
        """
        if not self.data_sheet_csv.exists():
            raise FileNotFoundError(
                f"Data sheet must be first constructed and saved in:\n"
                f"{self.data_sheet_csv}\nNot Found"
            )

        df_header = pd.read_csv(self.data_sheet_csv, nrows=1)
        cols = list(df_header.columns)

        field_ids: List[int] = []
        for col in cols:
            if col == "eid":
                continue
            try:
                fid = int(col.split("_")[0].replace("p", ""))
                field_ids.append(fid)
            except ValueError:
                continue

        return sorted(set(field_ids))

    def load_labels(
        self,
        source: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load existing labels from CSV or JSON.

        Parameters
        ----------
        source : {"csv", "json", None}, optional
            Explicit source to load from. If None, auto-detect.

        Returns
        -------
        Optional[pd.DataFrame]
            Loaded metadata DataFrame, or None if unavailable.
        """
        if source in (None, "csv") and self.output_csv.exists():
            return pd.read_csv(self.output_csv)

        if source in (None, "json") and self.output_json.exists():
            with open(self.output_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            return (
                pd.DataFrame.from_dict(data, orient="index")
                .reset_index()
                .rename(columns={"index": "field_id"})
            )

        return None

    # -----------------------------------------------------------------
    # CORE LOGIC
    # -----------------------------------------------------------------

    def build_metadata_table(self, field_ids: Iterable[int]) -> pd.DataFrame:
        """
        Fetch metadata for a collection of field IDs.

        Parameters
        ----------
        field_ids : Iterable[int]
            UKBB field IDs to query.

        Returns
        -------
        pd.DataFrame
            Metadata table.
        """
        rows: List[Dict[str, Union[int, str, None]]] = []

        field_ids = list(field_ids)
        for i, fid in enumerate(field_ids, 1):
            try:
                rows.append(self.fetch_field_metadata(fid))
            except Exception as e:
                rows.append({
                    "field_id": fid,
                    "description": None,
                    "category": None,
                    "error": str(e),
                })

            if i % 25 == 0:
                print(f"[INFO] {i}/{len(field_ids)} fields processed")

            time.sleep(self.sleep_seconds)

        return pd.DataFrame(rows)

    def fetch_field_metadata(self, field_id: int) -> Dict[str, Optional[str]]:
        """
        Query UKBB Showcase for a single field ID.

        Parameters
        ----------
        field_id : int
            UKBB field ID.

        Returns
        -------
        Dict[str, Optional[str]]
            Dictionary with keys: field_id, description, category.
        """
        params = {
            "wot": 0,
            "srch": field_id,
            "yfirst": 2000,
            "ylast": 2025,
        }

        response = requests.get(
            BASE_URL,
            params=params,
            headers=HEADERS,
            timeout=self.timeout,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        row = soup.find("tr", id=f"f{field_id}")

        if row is None:
            return {"field_id": field_id, "description": None, "category": None}

        cells = row.find_all("td")
        return {
            "field_id": field_id,
            "description": cells[1].get_text(strip=True),
            "category": cells[2].get_text(strip=True),
        }

    # -----------------------------------------------------------------
    # PERSISTENCE
    # -----------------------------------------------------------------

    def save_outputs(self, df: pd.DataFrame) -> None:
        """
        Save metadata to CSV and JSON.

        Parameters
        ----------
        df : pd.DataFrame
            Metadata DataFrame.
        """
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(self.output_csv, index=False)

        field_dict = (
            df.set_index("field_id")[["description", "category"]]
            .to_dict(orient="index")
        )

        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(field_dict, f, indent=2)

        print(f"[INFO] CSV  saved to: {self.output_csv}")
        print(f"[INFO] JSON saved to: {self.output_json}")


# ---------------------------------------------------------------------
# USAGE EXAMPLE
# ---------------------------------------------------------------------

if __name__ == "__main__":
    extractor = UKBBFieldMetadataExtractor(
        data_sheet_csv=config.get('paths')['data_sheet']['dir_csv'],
        output_csv=config.get('paths')['data_sheet']['formal_name_csv'],
        output_json=config.get('paths')['data_sheet']['formal_name_json'],
    )
    df_fields = extractor.run()
