# apmt_dashboard/pages/01_Field_Outlook.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import pydeck as pdk
import json
from pathlib import Path
import numpy as np

from components.comparison_cards import create_comparison_cards
from components.charts import create_time_series_chart, create_distribution_chart
from utils.helpers import coalesce_first

# Geo utils (with safe import pattern if you ever package it differently)
try:
    from utils.geo_utils import ensure_geo_assets
except ImportError:
    from ..utils.geo_utils import ensure_geo_assets


def render_field_outlook(processor):
    """Render the Field Outlook dashboard page."""
    st.header("🧭 Field & Data Outlook")

    df = processor.df

    # ---------- TOP METRICS ----------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Submissions", len(df))

    with col2:
        latest = None
        cand = None
        # Prefer standardized date if available
        if '_submission_time' in df.columns:
            cand = '_submission_time'
        elif 'int_date_std' in df.columns:
            cand = 'int_date_std'
        elif 'int_date' in df.columns:
            cand = 'int_date'
        elif 'start' in df.columns:
            cand = 'start'

        if cand:
            try:
                latest = pd.to_datetime(df[cand], errors='coerce').max()
            except Exception:
                latest = None

        st.metric(
            "Latest Submission",
            latest.strftime("%Y-%m-%d") if (latest is not None and pd.notna(latest)) else "N/A"
        )

    with col3:
        counties_covered = int(df['County'].nunique()) if 'County' in df.columns else 0
        st.metric("Counties Covered", counties_covered)

    with col4:
        if 'kpmd_registered' in df.columns:
            try:
                kpmd_participants = int(pd.to_numeric(df['kpmd_registered'], errors='coerce').fillna(0).sum())
            except Exception:
                kpmd_participants = 0
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
            key="field_granularity"
        )

    date_col = None
    if '_submission_time' in df.columns:
        date_col = '_submission_time'
    elif 'int_date_std' in df.columns:
        date_col = 'int_date_std'
    elif 'int_date' in df.columns:
        date_col = 'int_date'
    elif 'start' in df.columns:
        date_col = 'start'

    if date_col:
        tmp = df.copy()
        tmp['__date'] = pd.to_datetime(tmp[date_col], errors='coerce')
        tmp = tmp[tmp['__date'].notna()].copy()

        if not tmp.empty:
            if gran == "Daily":
                tmp['__bucket'] = tmp['__date'].dt.date
                x_label, title = "Date", "Daily Submission Volume"
            elif gran == "Weekly":
                tmp['__bucket'] = tmp['__date'].dt.to_period('W').dt.start_time.dt.date
                x_label, title = "Week (start)", "Weekly Submission Volume"
            else:  # Monthly
                tmp['__bucket'] = tmp['__date'].dt.to_period('M').dt.to_timestamp()
                x_label, title = "Month", "Monthly Submission Volume"

            series = (
                tmp.groupby('__bucket')
                .size()
                .reset_index(name='Submissions')
                .sort_values('__bucket')
            )

            if len(series) > 0:
                fig = create_time_series_chart(
                    series,
                    x_col='__bucket',
                    y_col='Submissions',
                    title=title,
                    markers=True
                )
                if fig:
                    fig.update_layout(xaxis_title=x_label, yaxis_title="Submissions")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No dated submissions available to plot.")
        else:
            st.info("No valid dates after parsing.")
    else:
        # Fallback: use 'month' categorical if present
        if 'month' in df.columns and not df['month'].isna().all():
            monthly = (
                df.groupby('month')
                .size()
                .reset_index(name='Submissions')
                .sort_values('month')
            )
            fig = create_time_series_chart(
                monthly,
                x_col='month',
                y_col='Submissions',
                title='Monthly Submission Volume',
                markers=True
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No time information available to show submissions over time.")

    # ---------- PANEL DATA OVERVIEW ----------
    if hasattr(processor, 'is_panel_data') and processor.is_panel_data:
        st.subheader("📊 Panel Data Overview")
        col1, col2, col3 = st.columns(3)

        if 'panel_hhid' in df.columns:
            with col1:
                st.metric("Unique Households", df['panel_hhid'].nunique())

        if 'panel_wave' in df.columns:
            with col2:
                st.metric("Time Periods", df['panel_wave'].nunique())

        if 'panel_hhid' in df.columns:
            obs_per_hh = df.groupby('panel_hhid').size()
            with col3:
                st.metric("Avg Obs per HH", f"{obs_per_hh.mean():.1f}")

        # Show time period distribution
        if 'panel_wave' in df.columns:
            wave_dist = df['panel_wave'].value_counts().sort_index()
            wave_df = pd.DataFrame({
                'Time Period': wave_dist.index,
                'Count': wave_dist.values
            })
            fig = px.bar(
                wave_df,
                x='Time Period',
                y='Count',
                title='Observations by Time Period',
                labels={'Count': 'Number of Observations'}
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---------- SUBMISSIONS BY COUNTY & KPMD ----------
    st.subheader("Submissions by County and KPMD Status")
    if 'County' in df.columns and 'kpmd_registered' in df.columns:
        tmp = df.copy()
        tmp['kpmd_registered'] = pd.to_numeric(tmp['kpmd_registered'], errors='coerce').fillna(0).astype(int)

        county_kpmd = (
            tmp.groupby(['County', 'kpmd_registered'])
            .size()
            .reset_index(name='count')
        )
        county_kpmd['kpmd_status'] = county_kpmd['kpmd_registered'].map({1: 'KPMD', 0: 'Non-KPMD'})

        fig = px.bar(
            county_kpmd,
            x='County',
            y='count',
            color='kpmd_status',
            title='Submissions by County and KPMD Status',
            barmode='group',
            labels={'count': 'Submissions', 'kpmd_status': 'KPMD Status'}
        )
        fig.update_traces(text=county_kpmd['count'], textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("County or KPMD data not available.")

    # ---------- HOUSEHOLD LOCATIONS MAP ----------
    st.subheader("Household Locations")
    render_household_map(processor)


def render_household_map(processor):
    """Render interactive household map."""
    df = processor.df

    try:
        # ---- Detect GPS columns robustly ----
        lat_col = coalesce_first(
            df,
            ['_GPS Coordinates_latitude', 'GPS Latitude', 'Latitude', 'lat', 'Lat']
        )
        lon_col = coalesce_first(
            df,
            ['_GPS Coordinates_longitude', 'GPS Longitude', 'Longitude', 'lon', 'Lon']
        )

        if not (lat_col and lon_col and lat_col in df.columns and lon_col in df.columns):
            st.info("GPS coordinates not available in the dataset.")
            return

        pts_df = df.dropna(subset=[lat_col, lon_col]).copy()
        pts_df[lon_col] = pd.to_numeric(pts_df[lon_col], errors='coerce')
        pts_df[lat_col] = pd.to_numeric(pts_df[lat_col], errors='coerce')
        pts_df = pts_df.dropna(subset=[lat_col, lon_col])

        if len(pts_df) == 0:
            st.info("No valid GPS points to map.")
            return

        # ---- Ensure GeoJSON assets are available ----
        assets_ok = ensure_geo_assets()
        if not assets_ok:
            st.warning("Could not download or access Kenya GeoJSON files. Map layer disabled.")
            return

        counties_path = Path("geo/kenya_counties.geojson")
        if not counties_path.exists() or counties_path.stat().st_size == 0:
            st.warning("Missing geo/kenya_counties.geojson even after fetch attempt.")
            return

        # ---- Build GeoDataFrames ----
        gdf_pts = gpd.GeoDataFrame(
            pts_df,
            geometry=gpd.points_from_xy(pts_df[lon_col], pts_df[lat_col]),
            crs="EPSG:4326",
        )

        gdf_counties = gpd.read_file(counties_path).to_crs("EPSG:4326")
        name_col = "shapeName" if "shapeName" in gdf_counties.columns else gdf_counties.columns[0]

        # Spatial join to count households per county
        joined = gpd.sjoin(
            gdf_pts[["geometry"]],
            gdf_counties[[name_col, "geometry"]],
            predicate="within",
            how="left",
        )
        counts = joined.groupby(name_col).size().rename("farmers").reset_index()
        gdf_counties = gdf_counties.merge(counts, on=name_col, how="left")
        gdf_counties["farmers"] = gdf_counties["farmers"].fillna(0).astype(float)

        # ---- Map controls ----
        with st.expander("Map Settings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                show_points = st.checkbox("Show household points", value=True)
            with col2:
                show_labels = st.checkbox("Show county labels", value=True)

            v = gdf_counties["farmers"].astype(float)
            vmax = float(v.max()) if np.isfinite(v.max()) else 0.0
            int_max = int(vmax) if vmax > 0 else 0
            default_val = int(round(vmax * 0.05)) if vmax > 0 else 0
            default_val = max(0, min(default_val, int_max))

            if int_max > 0:
                threshold = st.slider(
                    "Minimum farmers to highlight",
                    min_value=0,
                    max_value=int_max,
                    value=default_val,
                    help="Counties with fewer farmers will be dimmed",
                )
            else:
                threshold = 0
                st.caption("All counties have 0 recorded farmers in this view.")

        # ---- Color gradient ----
        v = gdf_counties["farmers"].astype(float)
        vmin, vmax = float(v.min()), float(v.max())
        span = (vmax - vmin) if vmax > vmin else 1.0
        t = (v - vmin) / span

        # Base RGB: light orange → dark red
        r0, g0, b0 = 254, 240, 217
        r1, g1, b1 = 165, 15, 21

        gdf_counties["r"] = (r0 + t * (r1 - r0)).round().clip(0, 255).astype(int)
        gdf_counties["g"] = (g0 + t * (g1 - g0)).round().clip(0, 255).astype(int)
        gdf_counties["b"] = (b0 + t * (b1 - b0)).round().clip(0, 255).astype(int)

        # Alpha: dim counties below threshold
        if threshold > 0:
            gdf_counties["alpha"] = np.where(gdf_counties["farmers"] >= threshold, 180, 40)
        else:
            gdf_counties["alpha"] = 180

        # ---- Prepare layers ----
        layers = []

        counties_geojson = json.loads(gdf_counties.to_json())
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                data=counties_geojson,
                stroked=True,
                filled=True,
                get_fill_color="[properties.r, properties.g, properties.b, properties.alpha]",
                get_line_color=[120, 120, 120, 200],
                line_width_min_pixels=0.8,
                pickable=True,
            )
        )

        # Household points layer
        has_kpmd = "kpmd_registered" in pts_df.columns
        if show_points:
            data_pts = pts_df.copy()
            data_pts["lon"] = pd.to_numeric(data_pts[lon_col], errors='coerce')
            data_pts["lat"] = pd.to_numeric(data_pts[lat_col], errors='coerce')
            data_pts = data_pts.dropna(subset=["lon", "lat"])

            if has_kpmd:
                data_pts["kpmd_registered"] = pd.to_numeric(
                    data_pts["kpmd_registered"], errors="coerce"
                ).fillna(0).astype(int)
                data_pts["color"] = data_pts["kpmd_registered"].map({
                    1: [31, 119, 180],   # blue
                    0: [214, 39, 40],    # red
                })
            else:
                data_pts["color"] = [[160, 160, 160] for _ in range(len(data_pts))]

            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=data_pts[['lon', 'lat', 'color']].to_dict(orient='records'),
                    get_position='[lon, lat]',
                    get_radius=700,
                    get_fill_color='color',
                    get_line_color=[0, 0, 0, 128],
                    line_width_min_pixels=0.5,
                    pickable=True,
                    filled=True,
                )
            )

        # County label layer
        if show_labels:
            label_df = gdf_counties[[name_col, "geometry", "farmers"]].copy()
            # Representative points for stable labeling
            rep_points = label_df.geometry.representative_point()
            label_df["lon"] = rep_points.x
            label_df["lat"] = rep_points.y

            layers.append(
                pdk.Layer(
                    "TextLayer",
                    data=label_df.to_dict(orient='records'),
                    get_position='[lon, lat]',
                    get_text=f'properties["{name_col}"]' if name_col == "shapeName" else f'properties["{name_col}"]',
                    get_size=10,
                    get_color=[0, 0, 0, 220],
                    get_angle=0,
                    get_alignment_baseline="'bottom'",
                )
            )

        # ---- View state ----
        bounds = gdf_counties.total_bounds  # [minx, miny, maxx, maxy]
        cx = float((bounds[0] + bounds[2]) / 2)
        cy = float((bounds[1] + bounds[3]) / 2)

        view_state = pdk.ViewState(
            latitude=cy,
            longitude=cx,
            zoom=5.6,
            pitch=0,
            bearing=0
        )

        # ---- Render map ----
        tooltip = None
        if "shapeName" in gdf_counties.columns:
            tooltip = {
                "html": "<b>{shapeName}</b><br>Farmers: {farmers}",
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }

        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=view_state,
                layers=layers,
                tooltip=tooltip,
            )
        )

        # ---- Legend ----
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Legend:**")
        if has_kpmd:
            with col2:
                st.markdown("🔵 KPMD Households")
            with col3:
                st.markdown("🔴 Non-KPMD Households")

    except Exception as e:
        st.error(f"Error rendering map: {str(e)}")
        st.info("To enable the interactive map, ensure all dependencies (geopandas, pydeck, shapely) are installed "
                "and that the GeoJSON assets can be downloaded.")
