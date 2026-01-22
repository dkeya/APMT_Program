# apmt_dashboard/data_processing/panel_manager.py

import pandas as pd
import numpy as np
import streamlit as st
import re


class PanelDataManager:
    """
    Enhanced panel data analysis for longitudinal tracking.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.attrition_df = pd.DataFrame()

        self.panel_structure = self._detect_panel_structure()
        self._create_panel_identifiers()
        self._calculate_panel_metrics()

    def _parse_mixed_datetime_series(self, s: pd.Series) -> pd.Series:
        if s is None:
            return pd.Series(dtype="datetime64[ns]")

        if pd.api.types.is_datetime64_any_dtype(s):
            return pd.to_datetime(s, errors="coerce")

        try:
            return pd.to_datetime(s, errors="coerce", format="mixed")
        except TypeError:
            pass

        def _parse_one(x):
            if pd.isna(x):
                return pd.NaT
            x = str(x).strip()
            if not x:
                return pd.NaT
            if re.match(r"^\d{4}-\d{2}-\d{2}", x):
                return pd.to_datetime(x, errors="coerce", yearfirst=True)
            return pd.to_datetime(x, errors="coerce", dayfirst=True)

        return s.apply(_parse_one)

    def _detect_panel_structure(self):
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

        hh_candidates = ["Household ID", "household_id", "HHID", "_id", "_uuid", "respondent_id"]
        for col in hh_candidates:
            if col in self.df.columns and self.df[col].notna().any():
                structure["hhid_col"] = col
                break

        if "int_date_std" in self.df.columns and self.df["int_date_std"].notna().any():
            structure["date_col"] = "int_date_std"
        else:
            date_candidates = ["int_date", "_submission_time", "start", "end"]
            for col in date_candidates:
                if col in self.df.columns and self.df[col].notna().any():
                    structure["date_col"] = col
                    break

        if structure["hhid_col"] and structure["date_col"]:
            try:
                date_parsed = self._parse_mixed_datetime_series(self.df[structure["date_col"]])
                self.df["_date_parsed"] = date_parsed
                self.df["_month_year"] = self.df["_date_parsed"].dt.to_period("M")

                time_periods = self.df["_month_year"].dropna().sort_values().unique().tolist()
                structure["time_periods"] = time_periods
                structure["waves"] = [str(p) for p in time_periods]
                structure["is_panel"] = len(time_periods) >= 2

            except Exception as e:
                st.warning(f"⚠️ Could not parse panel dates reliably from '{structure['date_col']}': {e}")
                structure["is_panel"] = False

            hhid_col = structure["hhid_col"]
            hh_counts = self.df[hhid_col].value_counts(dropna=True)

            structure["households"] = int(hh_counts.size)
            structure["observations_per_hh"] = {
                "min": int(hh_counts.min()) if not hh_counts.empty else 0,
                "max": int(hh_counts.max()) if not hh_counts.empty else 0,
                "mean": float(hh_counts.mean()) if not hh_counts.empty else 0.0,
                "std": float(hh_counts.std()) if not hh_counts.empty else 0.0,
            }
            structure["balanced"] = (
                structure["observations_per_hh"]["min"] == structure["observations_per_hh"]["max"]
                and structure["observations_per_hh"]["min"] > 0
            )

        return structure

    def _create_panel_identifiers(self):
        # ----- Household ID -----
        if self.panel_structure["hhid_col"]:
            self.df["panel_hhid"] = self.df[self.panel_structure["hhid_col"]].astype(str).str.strip()
        else:
            if (
                "County" in self.df.columns
                and "_GPS Coordinates_latitude" in self.df.columns
                and "_GPS Coordinates_longitude" in self.df.columns
            ):
                county_code = self.df["County"].astype(str).str.slice(0, 3)

                lat = (
                    self.df["_GPS Coordinates_latitude"]
                    .astype(str)
                    .str.replace(".", "", regex=False)
                    .str.slice(-4)
                )
                lon = (
                    self.df["_GPS Coordinates_longitude"]
                    .astype(str)
                    .str.replace(".", "", regex=False)
                    .str.slice(-4)
                )
                self.df["panel_hhid"] = county_code + "_" + lat + lon
            else:
                self.df["panel_hhid"] = self.df.index.astype(str)

        # ----- Wave -----
        if "_month_year" in self.df.columns:
            wave_str = self.df["_month_year"].astype(str).replace("NaT", np.nan)
            self.df["panel_wave"] = wave_str
        else:
            self.df["panel_wave"] = pd.NA

        # ----- Wave numeric order -----
        if "_month_year" in self.df.columns:
            ordered_periods = self.df["_month_year"].dropna().sort_values().unique().tolist()
            wave_order = {str(p): i + 1 for i, p in enumerate(ordered_periods)}
        else:
            unique_waves = sorted([w for w in self.df["panel_wave"].dropna().unique().tolist()])
            wave_order = {w: i + 1 for i, w in enumerate(unique_waves)}

        self.df["panel_wave_num"] = self.df["panel_wave"].map(wave_order).astype("Int64")

        # Optional quarter label
        if "_date_parsed" in self.df.columns:
            yr = self.df["_date_parsed"].dt.year.astype("Int64").astype(str)
            q = self.df["_date_parsed"].dt.quarter.astype("Int64").astype(str)
            self.df["panel_quarter"] = yr + "Q" + q

        # HH-wave key
        self.df["panel_id"] = self.df["panel_hhid"].astype(str) + "_" + self.df["panel_wave"].astype(str)

        # Unique per row
        self.df["panel_row_id"] = self.df["panel_hhid"].astype(str) + "_" + self.df.index.astype(str)

        # Duplicates within HH-wave
        self.df["dup_within_wave"] = (
            self.df.groupby(["panel_hhid", "panel_wave"])["panel_row_id"].transform("size") > 1
        ).astype(int)

        # Deterministic ordering
        sort_cols = ["panel_hhid", "panel_wave_num"]
        if "_submission_time" in self.df.columns:
            self.df["_submission_time_parsed"] = self._parse_mixed_datetime_series(self.df["_submission_time"])
            sort_cols.append("_submission_time_parsed")
        elif "_date_parsed" in self.df.columns:
            sort_cols.append("_date_parsed")
        else:
            if "__index__" not in self.df.columns:
                self.df["__index__"] = self.df.index
            sort_cols.append("__index__")

        self.df = self.df.sort_values(sort_cols, kind="mergesort")

        self.df["panel_within_wave_seq"] = (
            self.df.groupby(["panel_hhid", "panel_wave"]).cumcount().astype(int) + 1
        )

    def _calculate_panel_metrics(self):
        if not self.panel_structure["is_panel"]:
            return self.df

        self.df = self.df.sort_values(["panel_hhid", "panel_wave_num", "panel_within_wave_seq"], kind="mergesort")

        key_metrics = [
            "total_sr",
            "net_profit",
            "rcsi_30",
            "birth_rate_per_100",
            "income_kpmd",
            "income_non_kpmd",
            "total_revenue",
            "total_costs",
        ]

        for metric in key_metrics:
            if metric in self.df.columns:
                s = pd.to_numeric(self.df[metric], errors="coerce")
                self.df[f"{metric}_change"] = s.groupby(self.df["panel_hhid"]).diff()
                self.df[f"{metric}_pct_change"] = s.groupby(self.df["panel_hhid"]).pct_change() * 100

                baseline = s.groupby(self.df["panel_hhid"]).transform(
                    lambda x: x.ffill().bfill().iloc[0] if x.notna().any() else np.nan
                )
                self.df[f"{metric}_cumulative"] = s - baseline

        # Treatment timing helpers (NA-safe int conversion)
        if "kpmd_registered" in self.df.columns and "panel_wave_num" in self.df.columns:
            treated = self.df[self.df["kpmd_registered"] == 1]
            first_reg_wave = treated.groupby("panel_hhid")["panel_wave_num"].min()

            self.df["ever_treated"] = self.df["panel_hhid"].isin(first_reg_wave.index).astype(int)
            self.df["first_kpmd_wave"] = self.df["panel_hhid"].map(first_reg_wave).astype("Int64")

            wave_num = self.df["panel_wave_num"].astype("Int64")
            first_wave = self.df["first_kpmd_wave"].astype("Int64")

            # ✅ THIS IS WHERE YOU WERE CRASHING BEFORE:
            # You had: (...) .astype(int) but the boolean could contain <NA>.
            cond = (
                wave_num.notna()
                & first_wave.notna()
                & (wave_num >= first_wave)
            )
            self.df["post_treatment"] = cond.astype(int)

            self.df["time_since_treatment"] = (wave_num - first_wave).astype("Int64")

        if len(self.panel_structure.get("waves", [])) >= 2:
            self._calculate_attrition_metrics()

        return self.df

    def _calculate_attrition_metrics(self):
        waves = self.df["panel_wave"].dropna().unique().tolist()
        waves = [w for w in waves if str(w).lower() != "nat"]

        if len(waves) <= 1:
            return

        if "_month_year" in self.df.columns:
            period_map = (
                self.df[["_month_year", "panel_wave"]]
                .dropna()
                .drop_duplicates()
                .set_index("panel_wave")["_month_year"]
            )
            waves_sorted = sorted(waves, key=lambda w: period_map.get(w, pd.Period("1900-01", freq="M")))
        else:
            waves_sorted = sorted(waves)

        attrition_data = []
        for i in range(len(waves_sorted) - 1):
            wave_from = waves_sorted[i]
            wave_to = waves_sorted[i + 1]

            hhs_from = set(self.df[self.df["panel_wave"] == wave_from]["panel_hhid"])
            hhs_to = set(self.df[self.df["panel_wave"] == wave_to]["panel_hhid"])

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
                    "attrition_rate": (attrited / total_from * 100) if total_from > 0 else 0.0,
                    "retention_rate": (stayed / total_from * 100) if total_from > 0 else 0.0,
                }
            )

        self.attrition_df = pd.DataFrame(attrition_data)

    def get_panel_summary(self):
        if not self.panel_structure["is_panel"]:
            return (
                "## No panel structure detected\n\n"
                "Dataset appears cross-sectional or lacks reliable multi-period time identifiers."
            )

        obs_stats = self.panel_structure.get("observations_per_hh", {})
        time_periods = self.panel_structure.get("waves", [])

        hh_wave_unique = int(self.df["panel_id"].nunique()) if "panel_id" in self.df.columns else 0
        dup_rows = int((self.df.get("dup_within_wave", 0) == 1).sum()) if "dup_within_wave" in self.df.columns else 0

        summary = f"""
## 📊 Panel Data Structure

### Basic Information
- **Total Households**: {self.panel_structure['households']:,}
- **Total Records (rows)**: {len(self.df):,}
- **Unique HH-Wave Observations (panel_id.nunique)**: {hh_wave_unique:,}
- **Rows that are duplicates within HH-Wave**: {dup_rows:,}
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
