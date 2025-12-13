# apmt_dashboard/data_processing/panel_manager.py

import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st


class PanelDataManager:
    """Enhanced panel data analysis for longitudinal tracking."""

    def __init__(self, df: pd.DataFrame):
        # Work on a copy to avoid side effects
        self.df = df.copy()
        self.attrition_df = pd.DataFrame()

        # Detect basic panel structure
        self.panel_structure = self._detect_panel_structure()

        # Create identifiers and time variables
        self._create_panel_identifiers()

        # Compute panel metrics (changes, DiD helpers, attrition)
        self._calculate_panel_metrics()

    # ------------------------------------------------------------------
    # PANEL STRUCTURE DETECTION
    # ------------------------------------------------------------------
    def _detect_panel_structure(self):
        """Detect panel data components in the dataset."""
        structure = {
            "is_panel": False,
            "hhid_col": None,
            "wave_col": None,
            "date_col": None,
            "waves": [],
            "households": 0,
            "observations_per_hh": {},
            "time_periods": [],
            "balanced": False,
        }

        # ---- 1. Detect household ID ----
        hh_candidates = [
            "Household ID",
            "household_id",
            "HHID",
            "_id",
            "_uuid",
            "respondent_id",
        ]
        for col in hh_candidates:
            if col in self.df.columns and self.df[col].notna().any():
                structure["hhid_col"] = col
                break

        # ---- 2. Detect date column ----
        # Prefer the standardized 'int_date_std' if it exists
        if "int_date_std" in self.df.columns and self.df["int_date_std"].notna().any():
            structure["date_col"] = "int_date_std"
        else:
            date_candidates = ["int_date", "_submission_time", "start", "end"]
            for col in date_candidates:
                if col in self.df.columns and self.df[col].notna().any():
                    structure["date_col"] = col
                    break

        # ---- 3. Decide if we have a panel structure ----
        if structure["hhid_col"] and structure["date_col"]:
            try:
                # Normalize date to datetime
                if structure["date_col"] == "int_date_std":
                    date_parsed = pd.to_datetime(
                        self.df["int_date_std"], errors="coerce"
                    )
                else:
                    date_parsed = pd.to_datetime(
                        self.df[structure["date_col"]], errors="coerce"
                    )

                self.df["_date_parsed"] = date_parsed
                self.df["_month_year"] = self.df["_date_parsed"].dt.to_period("M")

                time_periods = (
                    self.df["_month_year"].dropna().sort_values().unique().tolist()
                )
                structure["time_periods"] = time_periods
                structure["waves"] = [str(p) for p in time_periods]

                # If we have at least 2 distinct months, treat as panel
                if len(time_periods) > 0:
                    structure["is_panel"] = True

            except Exception as e:
                st.warning(
                    f"⚠️ Could not parse panel dates reliably "
                    f"from '{structure['date_col']}': {e}"
                )
                structure["is_panel"] = False

            if structure["is_panel"]:
                hhid_col = structure["hhid_col"]
                hh_counts = self.df[hhid_col].value_counts()

                structure["households"] = int(hh_counts.size)
                structure["observations_per_hh"] = {
                    "min": int(hh_counts.min()) if not hh_counts.empty else 0,
                    "max": int(hh_counts.max()) if not hh_counts.empty else 0,
                    "mean": float(hh_counts.mean()) if not hh_counts.empty else 0.0,
                    "std": float(hh_counts.std()) if not hh_counts.empty else 0.0,
                }
                structure["balanced"] = (
                    structure["observations_per_hh"]["min"]
                    == structure["observations_per_hh"]["max"]
                    and structure["observations_per_hh"]["min"] > 0
                )

        return structure

    # ------------------------------------------------------------------
    # IDENTIFIERS AND WAVES
    # ------------------------------------------------------------------
    def _create_panel_identifiers(self):
        """Create consistent panel identifiers for households and waves."""
        # ----- Household ID -----
        if self.panel_structure['hhid_col']:
            # Use the detected HHID column
            self.df['panel_hhid'] = (
                self.df[self.panel_structure['hhid_col']]
                .astype(str)
                .str.strip()
            )
        else:
            # Fallback: synthesize ID using County + GPS if available
            if (
                'County' in self.df.columns and
                '_GPS Coordinates_latitude' in self.df.columns and
                '_GPS Coordinates_longitude' in self.df.columns
            ):
                county_code = self.df['County'].astype(str).str.slice(0, 3)

                lat = (
                    self.df['_GPS Coordinates_latitude']
                    .astype(str)
                    .str.replace('.', '', regex=False)
                    .str.slice(-4)
                )
                lon = (
                    self.df['_GPS Coordinates_longitude']
                    .astype(str)
                    .str.replace('.', '', regex=False)
                    .str.slice(-4)
                )

                # Row-wise concatenation => each row gets a unique synthetic ID
                self.df['panel_hhid'] = county_code + '_' + lat + lon
            else:
                # Last resort: index-based ID (still stable within this dataset)
                self.df['panel_hhid'] = self.df.index.astype(str)

        # ----- Wave / time identifiers -----
        # Primary wave label
        if '_month_year' in self.df.columns:
            self.df['panel_wave'] = self.df['_month_year'].astype(str)
        else:
            self.df['panel_wave'] = 'Wave1'

        # Numeric order of waves for sorting and diffs
        self.df['panel_wave_num'] = self.df['panel_wave'].factorize()[0] + 1

        # Optional quarter label if a parsed date exists
        if '_date_parsed' in self.df.columns:
            self.df['panel_quarter'] = (
                self.df['_date_parsed'].dt.year.astype(str)
                + 'Q'
                + self.df['_date_parsed'].dt.quarter.astype(str)
            )

        # Unique panel observation ID (HHID + wave)
        self.df['panel_id'] = self.df['panel_hhid'] + '_' + self.df['panel_wave'].astype(str)

        # Always sort by HH and wave for consistent longitudinal calculations
        self.df = self.df.sort_values(['panel_hhid', 'panel_wave_num'])

    def _calculate_panel_metrics(self):
        """
        Calculate panel-specific metrics:
        - Within-household change and % change for key variables
        - Cumulative change from baseline (first observed value per HH)
        - Treatment timing (ever treated, post treatment, time since treatment)
        - Attrition / retention across waves (via _calculate_attrition_metrics)
        """
        if not self.panel_structure['is_panel']:
            return self.df

        # Ensure sorted order
        self.df = self.df.sort_values(['panel_hhid', 'panel_wave_num'])

        # Metrics where within-household movement is meaningful
        key_metrics = [
            'total_sr',
            'net_profit',
            'rcsi_30',
            'birth_rate_per_100',
            'income_kpmd',
            'income_non_kpmd',
            'total_revenue',
            'total_costs'
        ]

        for metric in key_metrics:
            if metric in self.df.columns:
                s = pd.to_numeric(self.df[metric], errors='coerce')

                # Wave-to-wave change within each household
                self.df[f'{metric}_change'] = (
                    s.groupby(self.df['panel_hhid']).diff()
                )

                # Percentage change
                self.df[f'{metric}_pct_change'] = (
                    s.groupby(self.df['panel_hhid']).pct_change() * 100
                )

                # Cumulative change from baseline (first observed value per household)
                baseline = (
                    s.groupby(self.df['panel_hhid'])
                    .transform(lambda x: x.ffill().bfill().iloc[0])
                )
                self.df[f'{metric}_cumulative'] = s - baseline

        # ----- Treatment timing (for DiD-style views) -----
        if 'kpmd_registered' in self.df.columns and 'panel_wave_num' in self.df.columns:
            treated = self.df[self.df['kpmd_registered'] == 1]

            # First wave when each HH is observed as registered
            first_reg_wave = treated.groupby('panel_hhid')['panel_wave_num'].min()

            # Ever treated flag
            self.df['ever_treated'] = self.df['panel_hhid'].isin(first_reg_wave.index).astype(int)

            # Map first treatment wave back to all rows of that HH
            self.df['first_kpmd_wave'] = self.df['panel_hhid'].map(first_reg_wave)

            # Post-treatment indicator
            self.df['post_treatment'] = (
                (self.df['panel_wave_num'] >= self.df['first_kpmd_wave'])
                & (self.df['first_kpmd_wave'].notna())
            ).astype(int)

            # Time since treatment (0 pre-treatment, 1,2,3... after)
            self.df['time_since_treatment'] = (
                self.df['panel_wave_num'] - self.df['first_kpmd_wave']
            )

        # ----- Attrition / retention -----
        if self.panel_structure['is_panel'] and len(self.panel_structure['waves']) > 1:
            self._calculate_attrition_metrics()

        return self.df

    def _calculate_attrition_metrics(self):
        """Calculate attrition and retention metrics between successive waves."""
        waves = (
            self.df["panel_wave"]
            .dropna()
            .unique()
            .tolist()
        )

        if len(waves) <= 1:
            return

        # Ensure waves are ordered chronologically using month_year mapping if available
        if "_month_year" in self.df.columns:
            period_map = (
                self.df[["_month_year", "panel_wave"]]
                .dropna()
                .drop_duplicates()
                .set_index("panel_wave")["_month_year"]
            )
            waves_sorted = sorted(waves, key=lambda w: period_map.get(w, w))
        else:
            waves_sorted = sorted(waves)

        attrition_data = []
        for i in range(len(waves_sorted) - 1):
            wave_from = waves_sorted[i]
            wave_to = waves_sorted[i + 1]

            hhs_from = set(
                self.df[self.df["panel_wave"] == wave_from]["panel_hhid"]
            )
            hhs_to = set(
                self.df[self.df["panel_wave"] == wave_to]["panel_hhid"]
            )

            stayed = len(hhs_from.intersection(hhs_to))
            attrited = len(hhs_from - hhs_to)
            new = len(hhs_to - hhs_from)
            total_from = len(hhs_from)

            attrition_data.append(
                {
                    "from_wave": wave_from,
                    "to_wave": wave_to,
                    "stayed": stayed,
                    "attrited": attrited,
                    "new": new,
                    "attrition_rate": (
                        attrited / total_from * 100 if total_from > 0 else 0.0
                    ),
                    "retention_rate": (
                        stayed / total_from * 100 if total_from > 0 else 0.0
                    ),
                }
            )

        self.attrition_df = pd.DataFrame(attrition_data)

    # ------------------------------------------------------------------
    # PUBLIC SUMMARY
    # ------------------------------------------------------------------
    def get_panel_summary(self):
        """Generate panel data summary (markdown string)."""
        if not self.panel_structure["is_panel"]:
            return (
                "## No panel structure detected\n\n"
                "Dataset appears to be cross-sectional or lacks reliable time identifiers."
            )

        obs_stats = self.panel_structure.get("observations_per_hh", {})
        time_periods = self.panel_structure.get("waves", [])

        summary = f"""
        ## 📊 Panel Data Structure

        ### Basic Information
        - **Total Households**: {self.panel_structure['households']:,}
        - **Total Observations**: {len(self.df):,}
        - **Time Periods**: {len(time_periods)}
        - **Periods Covered**: {', '.join(sorted(time_periods))}

        ### Panel Characteristics
        - **Observations per HH**:
          - Min: {obs_stats.get('min', 0)}
          - Max: {obs_stats.get('max', 0)}
          - Mean: {obs_stats.get('mean', 0.0):.1f} ± {obs_stats.get('std', 0.0):.1f}
        - **Panel Type**: {'Balanced' if self.panel_structure['balanced'] else 'Unbalanced'}
        """

        if hasattr(self, "attrition_df") and not self.attrition_df.empty:
            avg_attrition = self.attrition_df["attrition_rate"].mean()
            summary += f"\n- **Average Attrition Rate**: {avg_attrition:.1f}% between waves"

        return summary
