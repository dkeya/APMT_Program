# apmt_dashboard/pages/01_Field_Outlook.py
import streamlit as st
import pandas as pd
import plotly.express as px
import geopandas as gpd
import pydeck as pdk
import json
from pathlib import Path
import numpy as np
import re

from components.comparison_cards import create_comparison_cards
from components.charts import create_time_series_chart, create_distribution_chart
from utils.helpers import coalesce_first

try:
    from utils.geo_utils import ensure_geo_assets
except ImportError:
    from ..utils.geo_utils import ensure_geo_assets


def _parse_mixed_datetime_series(s: pd.Series) -> pd.Series:
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


def _best_date_column(df: pd.DataFrame):
    """
    Choose the date column that yields the most valid parsed datetimes.
    Preference order is still sensible, but ONLY if it parses well.
    """
    candidates = [c for c in ["int_date_std", "_submission_time", "int_date", "start", "end"] if c in df.columns]
    best = None
    best_non_na = -1
    best_parsed = None

    for c in candidates:
        parsed = _parse_mixed_datetime_series(df[c])
        non_na = int(parsed.notna().sum())
        if non_na > best_non_na:
            best_non_na = non_na
            best = c
            best_parsed = parsed

    return best, best_parsed


def render_field_outlook(processor):
    st.header("🧭 Field & Data Outlook")
    df = processor.df

    # ---------- TOP METRICS ----------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records (Rows)", len(df))

    with col2:
        best_col, parsed = _best_date_column(df)
        latest = parsed.max() if parsed is not None else None

        st.metric(
            "Latest Submission",
            latest.strftime("%Y-%m-%d") if (latest is not None and pd.notna(latest)) else "N/A"
        )
        if best_col:
            st.caption(f"Using date field: **{best_col}**")

    with col3:
        counties_covered = int(df["County"].nunique()) if "County" in df.columns else 0
        st.metric("Counties Covered", counties_covered)

    with col4:
        if "kpmd_registered" in df.columns:
            kp = pd.to_numeric(df["kpmd_registered"], errors="coerce")
            kpmd_submissions = int(kp.fillna(0).clip(lower=0).sum())
        else:
            kpmd_submissions = 0
        st.metric("KPMD Submissions", kpmd_submissions)

    # ---------- SUBMISSIONS OVER TIME ----------
    left, right = st.columns([0.8, 0.2])
    with left:
        st.subheader("Submissions Over Time")
    with right:
        gran = st.selectbox(
            "Granularity",
            ["Daily", "Weekly", "Monthly"],
            index=0,
            label_visibility="collapsed",
            key="field_granularity"
        )

    best_col, parsed = _best_date_column(df)
    if best_col and parsed is not None and parsed.notna().any():
        tmp = df.copy()
        tmp["__date"] = parsed
        tmp = tmp[tmp["__date"].notna()].copy()

        if not tmp.empty:
            if gran == "Daily":
                tmp["__bucket"] = tmp["__date"].dt.date
                x_label, title = "Date", "Daily Submission Volume"
            elif gran == "Weekly":
                tmp["__bucket"] = tmp["__date"].dt.to_period("W").dt.start_time.dt.date
                x_label, title = "Week (start)", "Weekly Submission Volume"
            else:
                tmp["__bucket"] = tmp["__date"].dt.to_period("M").dt.to_timestamp()
                x_label, title = "Month", "Monthly Submission Volume"

            series = tmp.groupby("__bucket").size().reset_index(name="Submissions").sort_values("__bucket")

            if len(series) > 0:
                fig = create_time_series_chart(series, x_col="__bucket", y_col="Submissions", title=title, markers=True)
                if fig:
                    fig.update_layout(xaxis_title=x_label, yaxis_title="Submissions")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No dated submissions available to plot.")
        else:
            st.info("No valid dates after parsing.")
    else:
        if "month" in df.columns and not df["month"].isna().all():
            monthly = df.groupby("month").size().reset_index(name="Submissions").sort_values("month")
            fig = create_time_series_chart(monthly, x_col="month", y_col="Submissions", title="Monthly Submission Volume", markers=True)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No time information available to show submissions over time.")

    # ---------- PANEL DATA OVERVIEW ----------
    if hasattr(processor, "is_panel_data") and processor.is_panel_data:
        st.subheader("📊 Panel Data Overview")
        col1, col2, col3 = st.columns(3)

        if "panel_hhid" in df.columns:
            with col1:
                st.metric("Unique Households", df["panel_hhid"].nunique())

        if "panel_wave" in df.columns:
            with col2:
                st.metric("Time Periods", df["panel_wave"].nunique())

        if "panel_hhid" in df.columns:
            obs_per_hh = df.groupby("panel_hhid").size()
            with col3:
                st.metric("Avg Obs per HH", f"{obs_per_hh.mean():.1f}")

        if "panel_wave" in df.columns:
            wave_dist = df["panel_wave"].value_counts().sort_index()
            wave_df = pd.DataFrame({"Time Period": wave_dist.index, "Count": wave_dist.values})
            fig = px.bar(wave_df, x="Time Period", y="Count", title="Observations by Time Period", labels={"Count": "Number of Observations"})
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Submissions by County and KPMD Status")
    if "County" in df.columns and "kpmd_registered" in df.columns:
        tmp = df.copy()
        tmp["kpmd_registered"] = pd.to_numeric(tmp["kpmd_registered"], errors="coerce").fillna(0).astype(int)

        county_kpmd = tmp.groupby(["County", "kpmd_registered"]).size().reset_index(name="count")
        county_kpmd["kpmd_status"] = county_kpmd["kpmd_registered"].map({1: "KPMD", 0: "Non-KPMD"})

        fig = px.bar(
            county_kpmd,
            x="County",
            y="count",
            color="kpmd_status",
            title="Submissions by County and KPMD Status",
            barmode="group",
            labels={"count": "Submissions", "kpmd_status": "KPMD Status"},
        )
        fig.update_traces(text=county_kpmd["count"], textposition="outside")
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("County or KPMD data not available.")

    st.subheader("Household Locations")
    render_household_map(processor)


def render_household_map(processor):
    # keep your existing implementation here (unchanged)
    # (If you want, paste it and I’ll make the TextLayer labels fully robust too.)
    df = processor.df
    st.info("Map rendering section unchanged in this patch.")
