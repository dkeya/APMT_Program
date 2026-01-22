# apmt_dashboard/pages/01_Field_Outlook.py

import streamlit as st
import pandas as pd
import plotly.express as px
import geopandas as gpd
import pydeck as pdk
import json
from pathlib import Path
import numpy as np

from components.charts import create_time_series_chart
from utils.helpers import coalesce_first

try:
    from utils.geo_utils import ensure_geo_assets
except ImportError:
    from ..utils.geo_utils import ensure_geo_assets


def _parse_mixed_datetime(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=True)
    except TypeError:
        return pd.to_datetime(s, errors="coerce", dayfirst=True)


def render_field_outlook(processor):
    st.header("🧭 Field & Data Outlook")
    df = processor.df

    # ---------- TOP METRICS ----------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records (Rows)", len(df))

    with col2:
        latest = None

        # Prefer the standardized datetime column created by the processor
        if "int_date_std" in df.columns and df["int_date_std"].notna().any():
            latest = pd.to_datetime(df["int_date_std"], errors="coerce").max()
        else:
            # Fallback to mixed parse for raw columns
            for cand in ["_submission_time", "int_date", "start", "end"]:
                if cand in df.columns and df[cand].notna().any():
                    latest = _parse_mixed_datetime(df[cand]).max()
                    if pd.notna(latest):
                        break

        st.metric(
            "Latest Submission",
            latest.strftime("%Y-%m-%d") if (latest is not None and pd.notna(latest)) else "N/A",
        )

    with col3:
        counties_covered = int(df["County"].nunique()) if "County" in df.columns else 0
        st.metric("Counties Covered", counties_covered)

    with col4:
        if "kpmd_registered" in df.columns:
            kpmd_participants = int(pd.to_numeric(df["kpmd_registered"], errors="coerce").fillna(0).sum())
        else:
            kpmd_participants = 0
        st.metric("KPMD Participants", kpmd_participants)

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
            key="field_granularity",
        )

    # Prefer standardized parsed datetime
    if "int_date_std" in df.columns and df["int_date_std"].notna().any():
        tmp = df.copy()
        tmp["__date"] = pd.to_datetime(tmp["int_date_std"], errors="coerce")
    else:
        # Fallback: raw columns with mixed parsing
        date_col = None
        for c in ["_submission_time", "int_date", "start", "end"]:
            if c in df.columns and df[c].notna().any():
                date_col = c
                break

        tmp = df.copy()
        tmp["__date"] = _parse_mixed_datetime(tmp[date_col]) if date_col else pd.NaT

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
            fig = create_time_series_chart(
                series,
                x_col="__bucket",
                y_col="Submissions",
                title=title,
                markers=True,
            )
            if fig:
                fig.update_layout(xaxis_title=x_label, yaxis_title="Submissions")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No dated submissions available to plot.")
    else:
        st.info("No valid dates after parsing.")

    # ---------- PANEL DATA OVERVIEW ----------
    if getattr(processor, "is_panel_data", False):
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
            fig = px.bar(
                wave_df,
                x="Time Period",
                y="Count",
                title="Observations by Time Period",
                labels={"Count": "Number of Observations"},
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---------- SUBMISSIONS BY COUNTY & KPMD ----------
    st.subheader("Submissions by County and KPMD Status")
    if "County" in df.columns and "kpmd_registered" in df.columns:
        tmp2 = df.copy()
        tmp2["kpmd_registered"] = pd.to_numeric(tmp2["kpmd_registered"], errors="coerce").fillna(0).astype(int)

        county_kpmd = tmp2.groupby(["County", "kpmd_registered"]).size().reset_index(name="count")
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

    # ---------- MAP ----------
    st.subheader("Household Locations")
    render_household_map(processor)


def render_household_map(processor):
    df = processor.df

    try:
        lat_col = coalesce_first(df, ["_GPS Coordinates_latitude", "GPS Latitude", "Latitude", "lat", "Lat"])
        lon_col = coalesce_first(df, ["_GPS Coordinates_longitude", "GPS Longitude", "Longitude", "lon", "Lon"])

        if not (lat_col and lon_col and lat_col in df.columns and lon_col in df.columns):
            st.info("GPS coordinates not available in the dataset.")
            return

        pts_df = df.copy()
        pts_df[lon_col] = pd.to_numeric(pts_df[lon_col], errors="coerce")
        pts_df[lat_col] = pd.to_numeric(pts_df[lat_col], errors="coerce")
        pts_df = pts_df.dropna(subset=[lat_col, lon_col])

        if pts_df.empty:
            st.info("No valid GPS points to map.")
            return

        if not ensure_geo_assets():
            st.warning("Could not download or access Kenya GeoJSON files. Map layer disabled.")
            return

        counties_path = Path("geo/kenya_counties.geojson")
        if not counties_path.exists() or counties_path.stat().st_size == 0:
            st.warning("Missing geo/kenya_counties.geojson even after fetch attempt.")
            return

        gdf_pts = gpd.GeoDataFrame(
            pts_df,
            geometry=gpd.points_from_xy(pts_df[lon_col], pts_df[lat_col]),
            crs="EPSG:4326",
        )
        gdf_counties = gpd.read_file(counties_path).to_crs("EPSG:4326")
        name_col = "shapeName" if "shapeName" in gdf_counties.columns else gdf_counties.columns[0]

        joined = gpd.sjoin(
            gdf_pts[["geometry"]],
            gdf_counties[[name_col, "geometry"]],
            predicate="within",
            how="left",
        )
        counts = joined.groupby(name_col).size().rename("farmers").reset_index()
        gdf_counties = gdf_counties.merge(counts, on=name_col, how="left")
        gdf_counties["farmers"] = gdf_counties["farmers"].fillna(0).astype(float)

        counties_geojson = json.loads(gdf_counties.to_json())
        layers = [
            pdk.Layer(
                "GeoJsonLayer",
                data=counties_geojson,
                stroked=True,
                filled=True,
                get_fill_color="[240, 240, 240, 80]",
                get_line_color=[120, 120, 120, 200],
                line_width_min_pixels=0.8,
                pickable=True,
            )
        ]

        data_pts = pts_df.copy()
        data_pts["lon"] = pts_df[lon_col]
        data_pts["lat"] = pts_df[lat_col]

        if "kpmd_registered" in data_pts.columns:
            data_pts["kpmd_registered"] = pd.to_numeric(data_pts["kpmd_registered"], errors="coerce").fillna(0).astype(int)
            data_pts["color"] = data_pts["kpmd_registered"].map({1: [31, 119, 180], 0: [214, 39, 40]})
        else:
            data_pts["color"] = [[160, 160, 160] for _ in range(len(data_pts))]

        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=data_pts,
                get_position="[lon, lat]",
                get_radius=700,
                get_fill_color="color",
                get_line_color=[0, 0, 0, 128],
                line_width_min_pixels=0.5,
                pickable=True,
                filled=True,
            )
        )

        bounds = gdf_counties.total_bounds
        cx = float((bounds[0] + bounds[2]) / 2)
        cy = float((bounds[1] + bounds[3]) / 2)

        view_state = pdk.ViewState(latitude=cy, longitude=cx, zoom=5.6, pitch=0, bearing=0)

        tooltip = {"html": f"<b>{{{name_col}}}</b><br>Farmers: {{farmers}}", "style": {"backgroundColor": "steelblue", "color": "white"}}

        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=view_state,
                layers=layers,
                tooltip=tooltip,
            )
        )

    except Exception as e:
        st.error(f"Error rendering map: {str(e)}")
