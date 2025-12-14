# apmt_dashboard/data_processing/data_processor.py

import pandas as pd
import numpy as np
import re
import streamlit as st

from utils.helpers import yn, to_num, one_hot_multiselect, coalesce_first
from utils.data_quality import clean_and_validate
from .panel_manager import PanelDataManager
from .calculations import (
    calculate_herd_metrics,
    calculate_pl_metrics,
    calculate_food_security,
    calculate_climate_resilience,
)


class APMTDataProcessor:
    """
    Core data preparation pipeline for the APMT longitudinal dashboard.

    Responsibilities:
    - Run global cleaning/validation.
    - Normalize key identifiers and dates.
    - Create standardized flags (KPMD registration, treatment arm).
    - Delegate metric calculations.
    - Initialize panel structure via PanelDataManager.
    """

    def __init__(self, df: pd.DataFrame):
        # 1. Clean and validate raw data
        clean_df, issues = clean_and_validate(df)
        self.df = clean_df
        self.dq_issues = issues

        # 2. Basic structural cleanups
        self._basic_cleanups()

        # 3. Build column mapping from multiple possible labels
        self.column_mapping = self._build_column_mapping()

        # 4. Standardize key variables (dates, KPMD, treatment, yes/no blocks)
        self.enhanced_standardize_data_optimized()

        # 5. Calculate thematic metrics
        self.calculate_herd_metrics()
        self.calculate_pl_metrics()
        self.calculate_food_security()
        self.calculate_climate_resilience()

        # 6. Panel data handling
        self.panel_manager = PanelDataManager(self.df)
        self.df = self.panel_manager.df
        self.is_panel_data = self.panel_manager.panel_structure["is_panel"]
        self.panel_summary = self.panel_manager.get_panel_summary()

    # ------------------------------------------------------------------
    # Basic cleanups
    # ------------------------------------------------------------------
    def _basic_cleanups(self):
        """Basic data cleaning operations."""
        # Ensure phone / ID-like fields are treated as strings
        for col in self.df.columns:
            lower = col.lower()
            if any(
                k in lower
                for k in ["phone", "telephone", "household id", "_id", "_uuid"]
            ):
                self.df[col] = self.df[col].astype(str)

        # Normalize column names: collapse whitespace
        self.df.columns = [re.sub(r"\s+", " ", c).strip() for c in self.df.columns]

    # ------------------------------------------------------------------
    # Column mapping
    # ------------------------------------------------------------------
    def _build_column_mapping(self):
        """Build mapping of standard column names to actual column names."""
        mapping = {}

        mapping["county"] = coalesce_first(self.df, ["County", "county", "COUNTY"])

        gender_col = coalesce_first(
            self.df, ["Gender", "gender", "GENDER", "Select respondent name"]
        )
        mapping["gender"] = gender_col
        if gender_col == "Select respondent name":
            st.warning(
                "⚠️ Using 'Select respondent name' as the gender field. "
                "Please verify this is intentional for this dataset."
            )

        mapping["kpmd_registration"] = coalesce_first(
            self.df,
            [
                "A8. Are you registered to KPMD programs?",
                "KPMD registration",
                "Registered to KPMD",
            ],
        )

        mapping["household_type"] = coalesce_first(
            self.df,
            [
                "Selection of the household",
                "Household type",
                "Treatment/Control",
            ],
        )

        mapping["gps_lat"] = coalesce_first(
            self.df,
            ["_GPS Coordinates_latitude", "GPS Latitude", "Latitude"],
        )
        mapping["gps_lon"] = coalesce_first(
            self.df,
            ["_GPS Coordinates_longitude", "GPS Longitude", "Longitude"],
        )

        # Extend here with other standardized fields as needed
        return mapping

    def _find_columns_pattern(self, pattern):
        """Find columns matching a regex pattern."""
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return []
        return [c for c in self.df.columns if rx.search(c)]

    # ------------------------------------------------------------------
    # Standardization (defensible, no fabricated timing)
    # ------------------------------------------------------------------
    def enhanced_standardize_data_optimized(self):
        """
        Optimized standardization to avoid fragmented DataFrame warnings.

        Key points for defensibility:
        - Interview date:
          * Uses best-available date column.
          * If no valid date, we DO NOT create fake months; we use NA and warn.
        - KPMD registration / treatment:
          * When structure is missing, values are set to NA (not 0).
          * Avoids silently classifying all households as non-KPMD/control.
        - Yes/no blocks:
          * Converted using `yn` helper, keeping space for NA where appropriate.
        """
        try:
            df = self.df.copy()

            # ---- 1. Interview date: choose best available source ----
            def _coerce_date(s):
                return pd.to_datetime(s, errors="coerce", dayfirst=True)

            date_candidates = [
                c
                for c in ["int_date", "_submission_time", "start", "end"]
                if c in df.columns
            ]
            new_columns = {}

            best_col = None
            best_non_na = 0
            parsed_dates = {}

            for c in date_candidates:
                parsed = _coerce_date(df[c])
                parsed_dates[c] = parsed
                non_na = parsed.notna().sum()
                if non_na > best_non_na:
                    best_non_na = non_na
                    best_col = c

            if best_col is not None and best_non_na > 0:
                int_date = parsed_dates[best_col]
                new_columns["int_date_std"] = int_date
                new_columns["month"] = int_date.dt.to_period("M").astype(str)
                new_columns["year"] = int_date.dt.year
                # UI caption about primary interview date removed to keep layout clean
            else:
                # No valid date – do NOT fabricate sequential months
                st.warning(
                    "⚠️ No valid interview date could be parsed from any of "
                    "['int_date', '_submission_time', 'start', 'end']. "
                    "Setting 'int_date_std', 'month', and 'year' to NA. "
                    "Wave-based panel analysis will rely on other identifiers."
                )
                idx = df.index
                new_columns["int_date_std"] = pd.Series(
                    pd.NaT, index=idx, dtype="datetime64[ns]"
                )
                new_columns["month"] = pd.Series(pd.NA, index=idx, dtype="string")
                new_columns["year"] = pd.Series(pd.NA, index=idx)

            # ---- 2. KPMD registration flag (no silent 0 default) ----
            kpmd_col = self.column_mapping.get("kpmd_registration")
            idx = df.index
            if kpmd_col and kpmd_col in df.columns:
                kpmd_series = df[kpmd_col].apply(yn)
                # Allow missing as NA; use pandas nullable integer
                try:
                    new_columns["kpmd_registered"] = kpmd_series.astype("Int64")
                except TypeError:
                    new_columns["kpmd_registered"] = kpmd_series
            else:
                st.info(
                    "ℹ️ KPMD registration field not found. "
                    "'kpmd_registered' set to NA for all records."
                )
                new_columns["kpmd_registered"] = pd.Series(
                    [pd.NA] * len(df), index=idx, dtype="Int64"
                )

            # ---- 3. Treatment status flag (no silent 0 default) ----
            arm_col = self.column_mapping.get("household_type")
            if arm_col and arm_col in df.columns:
                arm_str = df[arm_col].astype(str).str.lower()
                # Mark as treatment if the value explicitly contains "treat"
                is_treatment = arm_str.str.contains("treat", na=False)
                new_columns["is_treatment"] = is_treatment.astype("Int64")
            else:
                st.info(
                    "ℹ️ Treatment/Control field not found. "
                    "'is_treatment' set to NA for all records."
                )
                new_columns["is_treatment"] = pd.Series(
                    [pd.NA] * len(df), index=idx, dtype="Int64"
                )

            # ---- 4. Convert blocks of yes/no questions ----
            yn_patterns = [
                r"^C1\.",  # e.g. livestock ownership / constraints
                r"^C2\.",
                r"^D1\..*vaccinate",
                r"^D3\..*treat",
                r"^D4\..*deworm",
                r"^B5a\.",
                r"^B6a\.",
                r"^J1\.",
            ]

            for pat in yn_patterns:
                for col in self._find_columns_pattern(pat):
                    if col in df.columns:
                        # Keep NA where `yn` cannot interpret the response
                        df[col] = df[col].apply(yn)

            # ---- 5. Attach all new columns at once ----
            for col_name, col_data in new_columns.items():
                df[col_name] = col_data

            self.df = df

        except Exception as e:
            st.warning(f"Some data standardization issues occurred: {str(e)}")

            # Ensure minimum required columns exist with safe defaults
            if "month" not in self.df.columns:
                self.df["month"] = pd.Series(
                    pd.NA, index=self.df.index, dtype="string"
                )
            if "kpmd_registered" not in self.df.columns:
                self.df["kpmd_registered"] = pd.Series(
                    [pd.NA] * len(self.df), index=self.df.index, dtype="Int64"
                )
            if "is_treatment" not in self.df.columns:
                self.df["is_treatment"] = pd.Series(
                    [pd.NA] * len(self.df), index=self.df.index, dtype="Int64"
                )

    # Keep the original method for backward compatibility
    def enhanced_standardize_data(self):
        """Legacy method - calls optimized version."""
        return self.enhanced_standardize_data_optimized()

    # Delegate calculation methods to calculations.py
    calculate_herd_metrics = calculate_herd_metrics
    calculate_pl_metrics = calculate_pl_metrics
    calculate_food_security = calculate_food_security
    calculate_climate_resilience = calculate_climate_resilience