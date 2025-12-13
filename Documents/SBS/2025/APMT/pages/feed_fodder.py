# apmt_dashboard/pages/04_Feed_Fodder.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from components.comparison_cards import create_comparison_cards
from components.charts import (
    create_comparison_bar_chart,
    create_time_series_chart,
    create_distribution_chart,
    create_box_plot,
)
from utils.helpers import coalesce_first, to_num, yn, one_hot_multiselect


def render_feed_fodder(processor):
    """Render the Feed & Fodder dashboard page."""
    st.header("🌾 Feed & Fodder")

    df = processor.df

    # ------------------------------------------------------------------
    # Detect key columns (robustly)
    # ------------------------------------------------------------------
    # Feed expenditure: try generic derived col, then questionnaire-like names
    feed_exp_col = coalesce_first(
        df,
        [
            "Feed_Expenditure",
            "feed_expenditure",
            "Total feed expenditure",
            "Total Feed Expenditure",
            "B5b. How much did you spend on fodder in the last 1 month?",
            "B5b. How much did you spend on fodder in the last one month?",
        ],
    )

    # Fodder purchase & production yes/no columns
    fodder_purchase_col = coalesce_first(
        df,
        [
            "B5a. Did you purchase fodder in the last 1 month?",
            "B5a. Did you purchase fodder in the last one month?",
            "B5a. Did you purchase fodder last 1 month?",
            "B5a. Did you purchase fodder in last 1 month?",
        ],
    )

    fodder_production_col = coalesce_first(
        df,
        [
            "B6a. Did you produce any fodder?",
            "B6a. Did you produce fodder?",
            "B6a. Did you produce any fodder in the last 1 month?",
        ],
    )

    # Herd size column (for per-animal metrics)
    herd_col = coalesce_first(df, ["total_sr", "Total_SR", "total_sr_equivalent"])

    # ------------------------------------------------------------------
    # Top metrics
    # ------------------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        if feed_exp_col and feed_exp_col in df.columns:
            feed_vals = to_num(df[feed_exp_col])
            avg_feed_exp = feed_vals.mean()
            st.metric("Average Monthly Feed Expenditure", f"KES {avg_feed_exp:,.0f}")
        else:
            st.metric("Average Monthly Feed Expenditure", "N/A")

    with col2:
        if fodder_purchase_col and fodder_purchase_col in df.columns:
            purchase_rate = df[fodder_purchase_col].apply(yn).mean() * 100
            st.metric("Fodder Purchase Rate", f"{purchase_rate:.1f}%")
        else:
            st.metric("Fodder Purchase Rate", "N/A")

    with col3:
        if fodder_production_col and fodder_production_col in df.columns:
            prod_rate = df[fodder_production_col].apply(yn).mean() * 100
            st.metric("Fodder Production Rate", f"{prod_rate:.1f}%")
        else:
            st.metric("Fodder Production Rate", "N/A")

    # ------------------------------------------------------------------
    # Panel selector (if panel data)
    # ------------------------------------------------------------------
    original_df = None
    selected_wave = "All periods"
    if hasattr(processor, "is_panel_data") and processor.is_panel_data and "panel_wave" in df.columns:
        col_l, col_r = st.columns([3, 1])
        with col_l:
            st.subheader("Feed & Fodder Overview")
        with col_r:
            selected_wave = st.selectbox(
                "Select time period",
                ["All periods"] + sorted(df["panel_wave"].dropna().unique().tolist()),
                key="feed_wave_select",
            )

        if selected_wave != "All periods":
            original_df = df.copy()
            df = df[df["panel_wave"] == selected_wave]
            st.caption(f"Filtered to: **{selected_wave}**")

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Overview",
            "Feed Expenditure",
            "Fodder Access & Production",
            "Feed Types & Cost Efficiency",
        ]
    )

    with tab1:
        _render_feed_overview(df, feed_exp_col, fodder_purchase_col, fodder_production_col)

    with tab2:
        _render_feed_expenditure(df, feed_exp_col, herd_col, selected_wave)

    with tab3:
        _render_fodder_access_production(df, fodder_purchase_col, fodder_production_col)

    with tab4:
        _render_feed_types_and_efficiency(df, herd_col, feed_exp_col)

    # Restore df if we filtered by wave
    if original_df is not None:
        processor.df = original_df


# ----------------------------------------------------------------------
# TAB 1: Overview
# ----------------------------------------------------------------------
def _render_feed_overview(df, feed_exp_col, fodder_purchase_col, fodder_production_col):
    st.subheader("Feed & Fodder Snapshot")

    # Simple counts: households purchasing / producing
    cols = st.columns(3)

    with cols[0]:
        st.markdown("**Households**")
        st.write(f"Total households in view: **{len(df):,}**")

    with cols[1]:
        if fodder_purchase_col and fodder_purchase_col in df.columns:
            purchase_flag = df[fodder_purchase_col].apply(yn)
            st.write(f"Purchasing fodder: **{int(purchase_flag.sum()):,}**")
        else:
            st.write("Purchasing fodder: _N/A_")

    with cols[2]:
        if fodder_production_col and fodder_production_col in df.columns:
            prod_flag = df[fodder_production_col].apply(yn)
            st.write(f"Producing fodder: **{int(prod_flag.sum()):,}**")
        else:
            st.write("Producing fodder: _N/A_")

    st.markdown("---")

    # Distribution of feed expenditure
    if feed_exp_col and feed_exp_col in df.columns:
        st.markdown("### Distribution of Feed Expenditure")
        work = df.copy()
        work["_feed_exp"] = to_num(work[feed_exp_col])
        work = work[work["_feed_exp"].notna()]

        if len(work) > 0:
            fig = create_distribution_chart(
                work,
                col="_feed_exp",
                title="Household Feed Expenditure (KES)",
                nbins=30,
            )
            if fig:
                fig.update_layout(xaxis_title="Feed expenditure (KES)", yaxis_title="Households")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No valid feed expenditure values to plot.")

    # County-level summary (purchase + production)
    if "County" in df.columns and (fodder_purchase_col or fodder_production_col):
        st.markdown("### Fodder Access by County")

        work = df.copy()

        if fodder_purchase_col and fodder_purchase_col in work.columns:
            work["purchased_fodder"] = work[fodder_purchase_col].apply(yn)
        else:
            work["purchased_fodder"] = np.nan

        if fodder_production_col and fodder_production_col in work.columns:
            work["produced_fodder"] = work[fodder_production_col].apply(yn)
        else:
            work["produced_fodder"] = np.nan

        grp = (
            work.groupby("County")
            .agg(
                hh_count=("County", "size"),
                purchase_rate=("purchased_fodder", lambda s: s.mean() * 100 if s.notna().any() else np.nan),
                production_rate=("produced_fodder", lambda s: s.mean() * 100 if s.notna().any() else np.nan),
            )
            .reset_index()
        )

        grp = grp.dropna(subset=["purchase_rate", "production_rate"], how="all")

        if not grp.empty:
            melted = grp.melt(
                id_vars=["County"],
                value_vars=["purchase_rate", "production_rate"],
                var_name="Type",
                value_name="Rate",
            ).dropna(subset=["Rate"])

            type_map = {
                "purchase_rate": "Purchase rate",
                "production_rate": "Production rate",
            }
            melted["Type"] = melted["Type"].map(type_map)

            fig = create_comparison_bar_chart(
                melted,
                x_col="County",
                y_col="Rate",
                color_col="Type",
                title="Fodder Purchase & Production Rates by County",
                barmode="group",
                text_format="{:.1f}",
                x_title="County",
                y_title="Rate (%)",
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No valid county-level fodder access information available.")


# ----------------------------------------------------------------------
# TAB 2: Feed Expenditure
# ----------------------------------------------------------------------
def _render_feed_expenditure(df, feed_exp_col, herd_col, selected_wave: str):
    st.subheader("Feed Expenditure Patterns")

    if not (feed_exp_col and feed_exp_col in df.columns):
        st.info("No feed expenditure column found in the dataset.")
        return

    work = df.copy()
    work["_feed_exp"] = to_num(work[feed_exp_col])

    if "_feed_exp" not in work.columns or work["_feed_exp"].notna().sum() == 0:
        st.info("No valid feed expenditure values to analyse.")
        return

    # Feed expenditure distribution by KPMD status
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Average Feed Expenditure by KPMD Status**")
        if "kpmd_registered" in work.columns:
            # Use comparison cards for KPMD vs non-KPMD
            create_comparison_cards(
                work,
                metric_col="_feed_exp",
                title="Feed Expenditure",
                format_str="KES {:,.0f}",
                group_col="kpmd_registered",
            )
        else:
            st.info("KPMD registration flag not available.")

    with col2:
        st.markdown("**Average Feed Expenditure by County**")
        if "County" in work.columns:
            grp = (
                work.groupby("County")["_feed_exp"]
                .mean()
                .reset_index(name="Average feed expenditure (KES)")
                .sort_values("Average feed expenditure (KES)", ascending=False)
            )

            fig = create_comparison_bar_chart(
                grp,
                x_col="County",
                y_col="Average feed expenditure (KES)",
                color_col=None,
                title="Average Monthly Feed Expenditure by County",
                text_format="{:,.0f}",
                x_title="County",
                y_title="Average feed expenditure (KES)",
            )
            if fig:
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("County variable not available.")

    st.markdown("---")

    # Panel trend over time if panel data available and user is viewing "All periods"
    if (
        selected_wave == "All periods"
        and "panel_wave" in work.columns
        and hasattr(processor := None, "is_panel_data")  # just to avoid linter; actual flag is handled in caller
    ):
        # caller already checked panel-ness; we just need df columns here
        st.markdown("### Feed Expenditure Trends Over Time")
        ts = (
            work.groupby("panel_wave")["_feed_exp"]
            .mean()
            .reset_index(name="Average feed expenditure (KES)")
            .sort_values("panel_wave")
        )
        if not ts.empty:
            fig = create_time_series_chart(
                ts,
                x_col="panel_wave",
                y_col="Average feed expenditure (KES)",
                title="Average Feed Expenditure Over Time",
                markers=True,
            )
            if fig:
                fig.update_layout(xaxis_title="Time period", yaxis_title="Average feed expenditure (KES)")
                st.plotly_chart(fig, use_container_width=True)

    # Per-animal feed cost
    st.markdown("### Feed Cost per Animal")

    if herd_col and herd_col in work.columns:
        herd_vals = to_num(work[herd_col])
        work = work.assign(
            _herd_size=herd_vals.replace(0, np.nan),
        )
        work["feed_cost_per_animal"] = work["_feed_exp"] / work["_herd_size"]

        valid = work[work["feed_cost_per_animal"].notna()].copy()

        if not valid.empty:
            create_comparison_cards(
                valid,
                metric_col="feed_cost_per_animal",
                title="Feed Cost per Animal",
                format_str="KES {:,.0f}",
                group_col="kpmd_registered" if "kpmd_registered" in valid.columns else "kpmd_registered",
            )

            fig = create_distribution_chart(
                valid,
                col="feed_cost_per_animal",
                title="Distribution of Feed Cost per Animal (KES)",
                nbins=30,
            )
            if fig:
                fig.update_layout(xaxis_title="Feed cost per animal (KES)", yaxis_title="Households")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No valid herd size values to compute feed cost per animal.")
    else:
        st.info("Total herd size column not found; cannot compute feed cost per animal.")


# ----------------------------------------------------------------------
# TAB 3: Fodder Access & Production
# ----------------------------------------------------------------------
def _render_fodder_access_production(df, fodder_purchase_col, fodder_production_col):
    st.subheader("Fodder Access & Production")

    if not (fodder_purchase_col or fodder_production_col):
        st.info("No fodder purchase/production indicators found in the dataset.")
        return

    work = df.copy()

    if fodder_purchase_col and fodder_purchase_col in work.columns:
        work["purchased_fodder"] = work[fodder_purchase_col].apply(yn)
    else:
        work["purchased_fodder"] = np.nan

    if fodder_production_col and fodder_production_col in work.columns:
        work["produced_fodder"] = work[fodder_production_col].apply(yn)
    else:
        work["produced_fodder"] = np.nan

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Fodder Purchase & Production by KPMD Status**")
        if "kpmd_registered" in work.columns:
            grp = (
                work.groupby("kpmd_registered")
                .agg(
                    purchase_rate=("purchased_fodder", lambda s: s.mean() * 100 if s.notna().any() else np.nan),
                    production_rate=("produced_fodder", lambda s: s.mean() * 100 if s.notna().any() else np.nan),
                )
                .reset_index()
            )
            grp["KPMD Status"] = grp["kpmd_registered"].map({1: "KPMD", 0: "Non-KPMD"})

            melted = grp.melt(
                id_vars=["KPMD Status"],
                value_vars=["purchase_rate", "production_rate"],
                var_name="Type",
                value_name="Rate",
            ).dropna(subset=["Rate"])

            type_map = {
                "purchase_rate": "Purchase rate",
                "production_rate": "Production rate",
            }
            melted["Type"] = melted["Type"].map(type_map)

            if not melted.empty:
                fig = create_comparison_bar_chart(
                    melted,
                    x_col="KPMD Status",
                    y_col="Rate",
                    color_col="Type",
                    title="Fodder Access by KPMD Registration",
                    barmode="group",
                    text_format="{:.1f}",
                    x_title="KPMD Status",
                    y_title="Rate (%)",
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No non-missing fodder access rates to show.")
        else:
            st.info("KPMD registration flag not available.")

    with col2:
        st.markdown("**Fodder Access by Gender (if available)**")
        if "Gender" in work.columns:
            grp_g = (
                work.groupby("Gender")
                .agg(
                    purchase_rate=("purchased_fodder", lambda s: s.mean() * 100 if s.notna().any() else np.nan),
                    production_rate=("produced_fodder", lambda s: s.mean() * 100 if s.notna().any() else np.nan),
                )
                .reset_index()
            )

            melted_g = grp_g.melt(
                id_vars=["Gender"],
                value_vars=["purchase_rate", "production_rate"],
                var_name="Type",
                value_name="Rate",
            ).dropna(subset=["Rate"])

            type_map = {
                "purchase_rate": "Purchase rate",
                "production_rate": "Production rate",
            }
            melted_g["Type"] = melted_g["Type"].map(type_map)

            if not melted_g.empty:
                fig = create_comparison_bar_chart(
                    melted_g,
                    x_col="Gender",
                    y_col="Rate",
                    color_col="Type",
                    title="Fodder Access by Gender",
                    barmode="group",
                    text_format="{:.1f}",
                    x_title="Gender",
                    y_title="Rate (%)",
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Gender column not available.")

    st.markdown("---")

    # Simple cross-tab view
    if fodder_purchase_col and fodder_production_col:
        st.markdown("### Combined Access Matrix")
        access_df = work[["purchased_fodder", "produced_fodder"]].dropna(how="all")
        if not access_df.empty:
            access_df = access_df.assign(
                Purchase=lambda d: d["purchased_fodder"].map({1: "Yes", 0: "No"}),
                Production=lambda d: d["produced_fodder"].map({1: "Yes", 0: "No"}),
            )

            ctab = pd.crosstab(access_df["Purchase"], access_df["Production"], normalize="all") * 100
            st.dataframe(ctab.style.format("{:.1f}%"))
        else:
            st.info("Not enough data to show combined access matrix.")


# ----------------------------------------------------------------------
# TAB 4: Feed Types & Cost Efficiency
# ----------------------------------------------------------------------
def _render_feed_types_and_efficiency(df, herd_col, feed_exp_col):
    st.subheader("Feed Types & Cost Efficiency")

    # Detect multi-select / text columns that look like "types / sources of fodder"
    fodder_type_candidates = []
    for c in df.columns:
        lc = c.lower()
        if "fodder" in lc and any(t in lc for t in ["type", "source", "kind", "variety"]):
            fodder_type_candidates.append(c)

    if fodder_type_candidates:
        st.markdown("### Fodder Types / Sources")

        for col in fodder_type_candidates:
            st.markdown(f"**{col}**")
            oh = one_hot_multiselect(df[col])
            if not oh.empty:
                counts = oh.sum().sort_values(ascending=False).reset_index()
                counts.columns = ["Fodder type", "Households"]

                fig = create_comparison_bar_chart(
                    counts,
                    x_col="Fodder type",
                    y_col="Households",
                    color_col=None,
                    title=f"Reported fodder types for: {col}",
                    barmode="group",
                    text_format="{:.0f}",
                    x_title="Fodder type",
                    y_title="Number of households",
                )
                if fig:
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No valid values found in column: {col}")
    else:
        st.info("No fodder type/source columns detected (based on column names).")

    st.markdown("---")

    # Feed share in total costs (if total_costs is available)
    st.markdown("### Feed Share in Total Costs")

    if feed_exp_col and feed_exp_col in df.columns and "total_costs" in df.columns:
        work = df.copy()
        work["_feed_exp"] = to_num(work[feed_exp_col])
        work["_total_costs"] = to_num(work["total_costs"])

        work = work.replace([np.inf, -np.inf], np.nan)
        work["feed_cost_share"] = work["_feed_exp"] / work["_total_costs"].replace(0, np.nan)

        valid = work[work["feed_cost_share"].notna()].copy()

        if not valid.empty:
            create_comparison_cards(
                valid,
                metric_col="feed_cost_share",
                title="Feed Cost Share in Total Costs",
                format_str="{:.2f}",
                group_col="kpmd_registered" if "kpmd_registered" in valid.columns else "kpmd_registered",
            )

            fig = create_distribution_chart(
                valid,
                col="feed_cost_share",
                title="Distribution of Feed Cost Share",
                nbins=30,
            )
            if fig:
                fig.update_layout(
                    xaxis_title="Feed cost share of total costs",
                    yaxis_title="Households",
                )
                st.plotly_chart(fig, use_container_width=True)

            if "County" in valid.columns:
                grp = (
                    valid.groupby("County")["feed_cost_share"]
                    .mean()
                    .reset_index(name="Average feed cost share")
                    .sort_values("Average feed cost share", ascending=False)
                )
                fig2 = create_comparison_bar_chart(
                    grp,
                    x_col="County",
                    y_col="Average feed cost share",
                    color_col=None,
                    title="Average Feed Cost Share by County",
                    barmode="group",
                    text_format="{:.2f}",
                    x_title="County",
                    y_title="Average feed cost share",
                )
                if fig2:
                    fig2.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No valid observations to compute feed cost share.")
    else:
        st.info("Cannot compute feed cost share — need both feed expenditure and total_costs columns.")
