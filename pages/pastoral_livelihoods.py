# apmt_dashboard/pages/03_Pastoral_Livelihoods.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from components.comparison_cards import create_comparison_cards
from components.charts import (
    create_comparison_bar_chart,
    create_pie_chart,
    create_time_series_chart,
    create_distribution_chart,
    create_box_plot,
)
from utils.helpers import coalesce_first, to_num, one_hot_multiselect
from utils.stats import lsmeans_by_group
from data_processing.calculations import calculate_pl_metrics


def render_pastoral_livelihoods(processor):
    """Render the Pastoral Livelihoods dashboard page."""
    st.header("💰 Pastoral Livelihoods")

    # Ensure P&L metrics exist on the processor dataframe
    calculate_pl_metrics(processor)

    # Work on a local copy of the (already globally filtered) data
    df = processor.df
    is_panel_data = bool(getattr(processor, "is_panel_data", False))
    selected_periods = None

    # ---- Time period selector for panel data ----
    if is_panel_data and "panel_wave" in df.columns:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📅 Time Period Selection")

        time_periods = sorted(df["panel_wave"].dropna().unique())
        selected_periods = st.sidebar.multiselect(
            "Select time periods to include",
            time_periods,
            default=time_periods,
        )

        if selected_periods:
            # Filter only if user excludes some waves
            if set(selected_periods) != set(time_periods):
                df = df[df["panel_wave"].isin(selected_periods)]
            st.caption(
                "**Time periods included:** "
                + ", ".join(str(p) for p in sorted(selected_periods))
                + f"  (Total observations: {len(df)})"
            )

    tab1, tab2, tab3 = st.tabs(
        [
            "Household Income Segmentation (Monthly)",
            "Access to Markets",
            "Price Information Access",
        ]
    )

    with tab1:
        render_income_segmentation_tab(df, is_panel_data)

    with tab2:
        render_market_access_tab(df)

    with tab3:
        render_price_info_tab(df)


# -------------------------------------------------------------------
# TAB 1: Income Segmentation
# -------------------------------------------------------------------
def render_income_segmentation_tab(df: pd.DataFrame, is_panel_data: bool):
    """Render income segmentation tab."""
    st.subheader("Household Income Segmentation (Monthly)")

    # Show time period info if panel data
    if is_panel_data and "panel_wave" in df.columns:
        period_info = df["panel_wave"].value_counts().sort_index()
        st.caption(
            "**Time periods represented on this view:** "
            + ", ".join(str(p) for p in period_info.index)
            + f"  (Total observations: {len(df)})"
        )

    # Income columns mapping
    income_cols_map = {
        "income_kpmd": "KPMD Livestock Income",
        "income_non_kpmd": "Non-KPMD Livestock Income",
        "income_feed": "Feed Income",
    }
    base_cols = list(income_cols_map.keys()) + ["kpmd_registered"]

    if not all(c in df.columns for c in base_cols):
        missing = [c for c in base_cols if c not in df.columns]
        st.info(f"Income fields missing: {', '.join(missing)}")
        return

    # Prepare income data
    inc = df[base_cols].copy()
    for c in income_cols_map.keys():
        inc[c] = to_num(inc[c])

    # Check for entirely NaN columns
    all_nan = [k for k, v in income_cols_map.items() if inc[k].notna().sum() == 0]
    if all_nan:
        st.warning("No data for: " + ", ".join(income_cols_map[k] for k in all_nan))

    # Pie chart: average mix across available values only
    avg_mix = []
    for k, label in income_cols_map.items():
        s = inc[k].dropna()
        if len(s):
            avg_mix.append((label, s.mean()))

    if avg_mix:
        names, vals = zip(*avg_mix)
        fig = create_pie_chart(
            pd.DataFrame({"Source": names, "Value": vals}),
            names_col="Source",
            values_col="Value",
            title="Average Household Income Mix",
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Time series view if panel data
    if is_panel_data and "panel_wave" in df.columns:
        st.subheader("Income Trends Over Time")

        ts_income = (
            df.groupby(["panel_wave", "kpmd_registered"])[list(income_cols_map.keys())]
            .mean()
            .reset_index()
        )
        ts_income["KPMD Status"] = ts_income["kpmd_registered"].map(
            {1: "KPMD", 0: "Non-KPMD"}
        )

        for income_source, label in income_cols_map.items():
            if income_source in ts_income.columns:
                fig_ts = create_time_series_chart(
                    ts_income,
                    x_col="panel_wave",
                    y_col=income_source,
                    color_col="KPMD Status",
                    title=f"{label} Trends Over Time",
                    markers=True,
                )
                if fig_ts:
                    st.plotly_chart(fig_ts, use_container_width=True)

    # Grouped bar: mean by KPMD
    melted = (
        inc.rename(columns=income_cols_map)
        .melt(id_vars=["kpmd_registered"], var_name="Income Type", value_name="KES")
    )
    melted["KES"] = to_num(melted["KES"])

    grp = (
        melted.dropna(subset=["KES"])
        .groupby(["kpmd_registered", "Income Type"])["KES"]
        .mean()
        .reset_index()
    )

    if len(grp):
        grp["KPMD Status"] = grp["kpmd_registered"].map({1: "KPMD", 0: "Non-KPMD"})

        fig2 = create_comparison_bar_chart(
            grp,
            x_col="Income Type",
            y_col="KES",
            color_col="KPMD Status",
            title="Average Income by KPMD Registration",
            barmode="group",
            text_format="{:.0f}",
            y_title="KES",
        )

        if fig2:
            st.plotly_chart(fig2, use_container_width=True)

    # LSMeans calculation (adjusted for key covariates if present)
    controls = []
    for candidate in ["County", "Gender", "total_sr", "month", "panel_wave"]:
        if candidate in df.columns and candidate != "kpmd_registered":
            controls.append(candidate)

    ls_notes = []
    for col, label in income_cols_map.items():
        df_lsm = inc[["kpmd_registered", col]].dropna()
        if len(df_lsm) >= 2 and df_lsm[col].var() > 0:
            lsm = lsmeans_by_group(df_lsm, col, "kpmd_registered", controls)
            if isinstance(lsm, dict):
                ls_notes.append(
                    f"{label} — LSMean KPMD: {lsm.get(1, np.nan):,.0f}, "
                    f"Non-KPMD: {lsm.get(0, np.nan):,.0f}"
                )

    if ls_notes:
        st.caption("Adjusted (LSMeans): " + " | ".join(ls_notes))


# -------------------------------------------------------------------
# TAB 2: Access to Markets
# -------------------------------------------------------------------
def render_market_access_tab(df: pd.DataFrame):
    """Render market access tab."""
    st.subheader("Access to Markets")

    f1 = coalesce_first(
        df,
        [
            "F1. How far did you travel to sell small ruminants in Kilometers last month?",
        ],
    )

    if not f1:
        st.info("F1 (distance to market) not available in this dataset.")
        return

    try:
        dfD = df.copy()
        dfD[f1] = to_num(dfD[f1])

        # Comparison cards
        create_comparison_cards(dfD, f1, "Distance to Market (km)", "{:.1f} km")

        # Distribution histogram
        fig = create_distribution_chart(
            dfD.dropna(subset=[f1]),
            col=f1,
            title="Distribution of Distance to Market (km)",
            nbins=20,
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Box plot by KPMD status
        if "kpmd_registered" in dfD.columns:
            box = dfD.dropna(subset=[f1])
            if len(box):
                box["KPMD Status"] = box["kpmd_registered"].map(
                    {1: "KPMD", 0: "Non-KPMD"}
                )

                fig2 = create_box_plot(
                    box,
                    x_col="KPMD Status",
                    y_col=f1,
                    title="Distance to Market by KPMD Registration",
                )
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.warning(f"Error processing Access to Markets: {e}")


# -------------------------------------------------------------------
# TAB 3: Price Information Access
# -------------------------------------------------------------------
def render_price_info_tab(df: pd.DataFrame):
    """Render price information access tab."""
    st.subheader("Price Information Access")

    f2 = coalesce_first(
        df,
        [
            "F2. Did you get information about livestock prices prior to selling in the last three months?",
            "F2. Did you get information about livestock prices prior to selling in the last 3 months?",
            "F2. Did you get livestock price information before selling in the last three months?",
            "F2. Did you get livestock price information before selling in the last 3 months?",
            "F2. Did you get price information prior to selling (last 3 months)?",
            "F2. Price information access (3 months)",
            "price_info",
            "price_information",
            "Got price information?",
        ],
    )

    if not f2:
        st.info("F2 (price information) not found in this dataset.")
        return

    try:
        tmp = df.copy()

        # Map responses to binary
        s = tmp[f2].astype(str).str.strip().str.lower()
        mapped = s.map(
            {
                "yes": 1,
                "y": 1,
                "true": 1,
                "t": 1,
                "1": 1,
                "ndio": 1,
                "ndiyo": 1,
                "no": 0,
                "n": 0,
                "false": 0,
                "f": 0,
                "0": 0,
                "la": 0,
                "hapana": 0,
            }
        )

        # Fallback: numeric interpretation
        if mapped.notna().sum() == 0:
            as_num = pd.to_numeric(tmp[f2], errors="coerce")
            mapped = (as_num > 0).astype("Int64")

        mask = tmp[f2].notna()
        tmp = tmp.loc[mask].copy()
        tmp["price_info"] = mapped.loc[mask].fillna(0).astype(int)

        # Debug view
        with st.expander("Debug: raw responses for price info (F2)", expanded=False):
            vc = df[f2].value_counts(dropna=False)
            st.write(vc.to_frame("count"))

        # Comparison cards (share of households with price info)
        create_comparison_cards(
            tmp,
            "price_info",
            "Households Accessing Price Info",
            "{:.1%}",
        )

        st.caption(
            f"Denominator uses households with a valid response in **{f2}** "
            f"(n={len(tmp):,})."
        )

        if tmp["price_info"].sum() == 0 and len(tmp) > 0:
            st.info(
                "All valid responses are 'No' (or 0) in the current filter. "
                "If you expected non-zero, check the raw value distribution above."
            )

    except Exception as e:
        st.warning(f"Error processing Price Information Access: {e}")
