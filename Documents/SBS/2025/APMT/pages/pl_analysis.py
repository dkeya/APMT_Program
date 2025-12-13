# pages/pl_analysis.py - UPDATED & REFINED VERSION
"""
Profit & Loss Analysis Page
Extracted from the original DashboardRenderer.render_pl_analysis method
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

# Import from our modular structure
from components.comparison_cards import create_comparison_cards
from utils.helpers import yn, to_num, coalesce_first
from utils.stats import lsmeans_by_group
from data_processing.calculations import calculate_pl_metrics


def render_pl_analysis(data_processor):
    """
    Render the Profit & Loss Analysis page.
    
    Args:
        data_processor: APMTDataProcessor instance with processed data.
    """
    st.header("💰 Profit & Loss Analysis")
    
    # Ensure P&L metrics are up to date
    calculate_pl_metrics(data_processor)
    
    # Get dataframe from processor
    df = data_processor.df
    
    # ENHANCED: Add time period selector for panel data (non-destructive)
    original_df = None  # To store original if we filter
    selected_wave = "All periods"
    
    if getattr(data_processor, "is_panel_data", False) and "panel_wave" in df.columns:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Overall Profitability")
        with col2:
            selected_wave = st.selectbox(
                "Select time period",
                ["All periods"] + sorted(df["panel_wave"].dropna().unique().tolist()),
                key="pl_wave_select",
            )

            if selected_wave != "All periods":
                original_df = df.copy()
                df = df[df["panel_wave"] == selected_wave]
                st.caption(f"Showing data for: {selected_wave}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overall Profitability", "Revenue Analysis", "Cost Analysis", "Channel Comparison"]
    )

    with tab1:
        _render_overall_profitability(
            df,
            data_processor,
            selected_wave=selected_wave,
        )

    with tab2:
        _render_revenue_analysis(df)

    with tab3:
        _render_cost_analysis(df)

    with tab4:
        _render_channel_comparison(df)

    # Restore original dataframe if we filtered it for the view
    if original_df is not None:
        data_processor.df = original_df


def _render_overall_profitability(df: pd.DataFrame, data_processor, selected_wave: str):
    """Render overall profitability tab."""
    st.subheader("Overall Profitability")

    if df.empty:
        st.info("No data available for the selected filters.")
        return

    # Helper to determine covariates for LSMeans
    def _controls_for_lsmeans(group_col=None):
        candidates = ["County", "Gender", "total_sr", "month", "panel_wave"]
        return [c for c in candidates if c in df.columns and c != group_col]

    col1, col2, col3, col4 = st.columns(4)

    # --- Average Net Profit & LSMeans ---
    with col1:
        avg_profit = df["net_profit"].mean()
        st.metric(
            "Average Net Profit (KES)",
            f"{(avg_profit if pd.notna(avg_profit) else 0):,.0f}",
        )

        if "kpmd_registered" in df.columns:
            controls = _controls_for_lsmeans(group_col="kpmd_registered")
            df_lsm = df.dropna(subset=["net_profit"])
            if not df_lsm.empty:
                lsm = lsmeans_by_group(
                    df_lsm, "net_profit", "kpmd_registered", controls
                )
                if isinstance(lsm, dict):
                    kpmd_lsm = lsm.get(1, np.nan)
                    non_kpmd_lsm = lsm.get(0, np.nan)
                    st.caption(
                        f"Adjusted (LSMean) — KPMD: {kpmd_lsm:,.0f} | Non-KPMD: {non_kpmd_lsm:,.0f}"
                    )

    # --- Average Profit Margin ---
    with col2:
        avg_margin = df["profit_margin"].mean()
        st.metric(
            "Average Profit Margin (%)",
            f"{(avg_margin if pd.notna(avg_margin) else 0):.1f}%",
        )

    # --- Share of Profitable Households ---
    with col3:
        profitable_hhs = (df["net_profit"] > 0).sum()
        total_hhs = len(df)
        pct = (profitable_hhs / total_hhs * 100) if total_hhs else 0
        st.metric("Profitable Households", f"{pct:.1f}%")

    # --- Average Revenue ---
    with col4:
        avg_revenue = df["total_revenue"].mean()
        st.metric(
            "Average Monthly Revenue (KES)",
            f"{(avg_revenue if pd.notna(avg_revenue) else 0):,.0f}",
        )

    # -----------------------------
    # Profit Distribution
    # -----------------------------
    st.subheader("Profit Distribution")
    col1, col2 = st.columns(2)

    with col1:
        if "net_profit" in df.columns:
            fig = px.histogram(
                df,
                x="net_profit",
                title="Distribution of Net Profit",
                labels={"net_profit": "Net Profit (KES)"},
            )
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Net profit data not available.")

    with col2:
        if "kpmd_registered" in df.columns:
            tmp = df.copy()
            tmp["KPMD Status"] = tmp["kpmd_registered"].map(
                {1: "KPMD", 0: "Non-KPMD"}
            )
            tmp["KPMD Status"] = pd.Categorical(
                tmp["KPMD Status"],
                categories=["Non-KPMD", "KPMD"],
                ordered=True,
            )

            if tmp["KPMD Status"].notna().any():
                fig = px.box(
                    tmp,
                    x="KPMD Status",
                    y="net_profit",
                    color="KPMD Status",
                    category_orders={"KPMD Status": ["Non-KPMD", "KPMD"]},
                    title="Profit Distribution by Registration",
                    labels={
                        "KPMD Status": "Registration",
                        "net_profit": "Net Profit (KES)",
                    },
                )
                fig.update_layout(legend_title_text="Registration")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("KPMD registration information is missing for this subset.")
        else:
            st.info("KPMD registration field is not available in the dataset.")

    # -----------------------------
    # Profit Trends Over Time (Panel)
    # -----------------------------
    if (
        getattr(data_processor, "is_panel_data", False)
        and "panel_wave" in df.columns
        and selected_wave == "All periods"
        and "kpmd_registered" in df.columns
    ):
        st.subheader("Profit Trends Over Time")

        profit_ts = (
            df.groupby(["panel_wave", "kpmd_registered"])["net_profit"]
            .mean()
            .reset_index()
        )

        if not profit_ts.empty:
            profit_ts["KPMD Status"] = profit_ts["kpmd_registered"].map(
                {1: "KPMD", 0: "Non-KPMD"}
            )

            fig_ts = px.line(
                profit_ts,
                x="panel_wave",
                y="net_profit",
                color="KPMD Status",
                title="Average Net Profit Over Time",
                markers=True,
                labels={"net_profit": "Average Net Profit (KES)", "panel_wave": "Wave"},
            )
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info("No panel time-series data available for net profit.")

    # -----------------------------
    # Profitability by County
    # -----------------------------
    if "County" in df.columns:
        st.subheader("Profitability by County")

        county_profit = (
            df.groupby("County", dropna=True)["net_profit"]
            .agg(["mean", "count"])
            .reset_index()
        )
        # Only show counties with at least a small sample size
        county_profit = county_profit[county_profit["count"] >= 3]

        if len(county_profit) > 0:
            fig = px.bar(
                county_profit,
                x="County",
                y="mean",
                title="Average Net Profit by County",
                labels={"mean": "Average Net Profit (KES)"},
                color="mean",
            )
            fig.update_traces(
                text=county_profit["mean"].round(0), textposition="outside"
            )
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data per county to show a meaningful comparison.")


def _render_revenue_analysis(df: pd.DataFrame):
    """Render revenue analysis tab."""
    st.subheader("Revenue Analysis")

    if df.empty:
        st.info("No data available for the selected filters.")
        return

    # Revenue composition (all revenue sub-components except total)
    revenue_cols = [
        c for c in df.columns if "revenue" in c.lower() and c != "total_revenue"
    ]
    if revenue_cols:
        avg_comp = df[revenue_cols].mean().sort_values(ascending=False)
        if not avg_comp.empty:
            fig = px.pie(
                values=avg_comp.values,
                names=avg_comp.index,
                title="Average Revenue Composition",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No detailed revenue components available for composition analysis.")

    # Income by source (KPMD livestock, Non-KPMD livestock, Feed)
    if all(
        c in df.columns for c in ["income_kpmd", "income_non_kpmd", "income_feed"]
    ):
        comp = df[["income_kpmd", "income_non_kpmd", "income_feed"]].mean().reset_index()
        comp.columns = ["Source", "KES"]
        comp["Source"] = comp["Source"].map(
            {
                "income_kpmd": "KPMD Livestock",
                "income_non_kpmd": "Non-KPMD Livestock",
                "income_feed": "Feed",
            }
        )
        fig2 = px.bar(
            comp,
            x="Source",
            y="KES",
            title="Average Income by Source (All Households)",
            labels={"KES": "Average Monthly Income (KES)"},
        )
        fig2.update_traces(text=comp["KES"].round(0), textposition="outside")
        fig2.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
        st.plotly_chart(fig2, use_container_width=True)

    # Revenue by KPMD status
    if "kpmd_registered" in df.columns:
        rc = (
            df.groupby("kpmd_registered")["total_revenue"]
            .mean()
            .reset_index()
        )
        if not rc.empty:
            rc["KPMD_Status"] = rc["kpmd_registered"].map(
                {1: "KPMD", 0: "Non-KPMD"}
            )
            fig = px.bar(
                rc,
                x="KPMD_Status",
                y="total_revenue",
                title="Average Revenue by KPMD Status",
                labels={"total_revenue": "Average Revenue (KES)"},
            )
            fig.update_traces(
                text=rc["total_revenue"].round(0), textposition="outside"
            )
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
            st.plotly_chart(fig, use_container_width=True)


def _render_cost_analysis(df: pd.DataFrame):
    """Render cost analysis tab."""
    st.subheader("Cost Structure Analysis")

    if df.empty:
        st.info("No data available for the selected filters.")
        return

    # Cost composition (all cost components except total_costs)
    cost_cols = [c for c in df.columns if "costs" in c.lower() and c != "total_costs"]
    if cost_cols:
        avg_cost = df[cost_cols].mean().sort_values(ascending=False)
        if not avg_cost.empty:
            fig = px.bar(
                x=avg_cost.index,
                y=avg_cost.values,
                title="Average Cost Composition",
                labels={"x": "Cost Category", "y": "Average Cost (KES)"},
            )
            fig.update_traces(text=avg_cost.round(0), textposition="outside")
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cost Efficiency")
    col1, col2 = st.columns(2)

    # --- Cost per animal ---
    with col1:
        if "total_costs" in df.columns and "total_sr" in df.columns:
            tmp = df.copy()
            tmp["cost_per_animal"] = tmp["total_costs"] / tmp["total_sr"].replace(
                0, np.nan
            )
            valid = tmp[tmp["cost_per_animal"].notna()]
            if len(valid) > 0:
                create_comparison_cards(
                    valid,
                    "cost_per_animal",
                    "Cost per Animal",
                    "KES {:.0f}",
                )

    # --- Cost-to-revenue ratio ---
    with col2:
        if "total_revenue" in df.columns and "total_costs" in df.columns:
            tmp = df.copy()
            tmp["cost_ratio"] = tmp["total_costs"] / tmp[
                "total_revenue"
            ].replace(0, np.nan)
            valid = tmp[tmp["cost_ratio"].notna()]
            if len(valid) > 0:
                create_comparison_cards(
                    valid,
                    "cost_ratio",
                    "Cost-to-Revenue Ratio",
                    "{:.2f}",
                )


def _render_channel_comparison(df: pd.DataFrame):
    """Render channel comparison & breakeven analysis tab."""
    st.subheader("Channel Profitability Comparison")

    if df.empty:
        st.info("No data available for the selected filters.")
        return

    # Check for channel-specific profit margin columns (optional, if ever added)
    channel_cols = ["sheep_kpmd_profit_margin", "sheep_non_kpmd_profit_margin"]
    available = [c for c in channel_cols if c in df.columns]

    if available and "kpmd_registered" in df.columns:
        rows = []
        for col in available:
            channel_name = " ".join(col.split("_")[:3]).title()
            for s in [0, 1]:
                sub = df[df["kpmd_registered"] == s]
                if len(sub) == 0:
                    continue
                rows.append(
                    {
                        "Channel": channel_name,
                        "Profit_Margin": sub[col].mean(),
                        "KPMD_Status": "KPMD Registered" if s == 1 else "Non-KPMD Registered",
                    }
                )

        if rows:
            ch_df = pd.DataFrame(rows)
            fig = px.bar(
                ch_df,
                x="Channel",
                y="Profit_Margin",
                color="KPMD_Status",
                title="Channel Profit Margins by KPMD Registration",
                barmode="group",
                labels={"Profit_Margin": "Profit Margin (%)"},
            )
            fig.update_traces(
                text=ch_df["Profit_Margin"].round(1), textposition="outside"
            )
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Channel-level profit margin metrics are not available in the current dataset."
        )

    # -----------------------------
    # Breakeven Analysis
    # -----------------------------
    st.subheader("Breakeven Analysis")

    data = df.copy()
    data["breakeven_status"] = np.where(
        data["net_profit"] >= 0, "Profitable", "Loss-making"
    )

    if "kpmd_registered" in data.columns:
        if data["kpmd_registered"].notna().any():
            # Normalize by row (each KPMD status), ensure both statuses present as columns
            pivot = (
                pd.crosstab(
                    data["kpmd_registered"],
                    data["breakeven_status"],
                    normalize="index",
                )
                * 100
            )
            pivot = pivot.reindex(
                columns=["Profitable", "Loss-making"], fill_value=0
            ).reset_index()
            pivot["KPMD_Status"] = pivot["kpmd_registered"].map(
                {1: "KPMD", 0: "Non-KPMD"}
            )

            melted = pivot.melt(
                id_vars=["KPMD_Status"],
                value_vars=["Profitable", "Loss-making"],
                var_name="Status",
                value_name="Percentage",
            )

            fig = px.bar(
                melted,
                x="KPMD_Status",
                y="Percentage",
                color="Status",
                title="Breakeven Status by KPMD Registration",
                barmode="stack",
                labels={"Percentage": "Households (%)"},
            )
            fig.update_traces(
                text=melted["Percentage"].round(1), textposition="outside"
            )
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("KPMD registration is missing for this subset; cannot show breakeven split.")
    else:
        st.info("KPMD registration field not available; breakeven by registration cannot be computed.")


# Helper function for LSMean formatting (kept from original)
def fmt_lsmean_note(lsm):
    try:
        return f'<div class="lsm-note">LSMean (adjusted): {lsm}</div>'
    except Exception:
        return ""
