# apmt_dashboard/pages/02_Pastoral_Productivity.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from components.comparison_cards import create_comparison_cards
from components.charts import (
    create_comparison_bar_chart,
    create_distribution_chart,
    create_box_plot,
)
from utils.helpers import coalesce_first, to_num, yn
from data_processing.calculations import calculate_herd_metrics


def render_pastoral_productivity(processor):
    """Render the Pastoral Productivity dashboard page."""
    st.header("🐑 Pastoral Productivity")

    # Calculate herd metrics (adds total_sr, rates, etc.)
    calculate_herd_metrics(processor)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(
        ["Herd Composition", "Animal Health Indicators", "SR Productivity Indicators"]
    )

    # ---------- Tab 1: Herd Composition ----------
    with tab1:
        render_herd_composition_tab(processor)

    # ---------- Tab 2: Animal Health ----------
    with tab2:
        render_animal_health_tab(processor)

    # ---------- Tab 3: SR Productivity (per 100 head) ----------
    with tab3:
        render_productivity_tab(processor)


def render_herd_composition_tab(processor):
    """Render herd composition tab."""
    df = processor.df

    st.subheader("Herd Structure & Size")
    st.write("**Average Animals Owned**")

    # Get column names from processor mapping or use defaults
    sheep_col = (
        processor.column_mapping.get("total_sheep")
        if hasattr(processor, "column_mapping")
        else None
    ) or "total_sheep"

    goats_col = (
        processor.column_mapping.get("total_goats")
        if hasattr(processor, "column_mapping")
        else None
    ) or "total_goats"

    col1, col2 = st.columns(2)
    with col1:
        if sheep_col in df.columns:
            create_comparison_cards(df, sheep_col, "Average Sheep", "{:.1f}")
        else:
            st.info("Sheep data not available.")

    with col2:
        if goats_col in df.columns:
            create_comparison_cards(df, goats_col, "Average Goats", "{:.1f}")
        else:
            st.info("Goat data not available.")

    # Female/Male percentages
    if "pct_female" in df.columns:
        st.write("**Percentage Female Stock**")
        create_comparison_cards(df, "pct_female", "Female Stock %", "{:.1f}%")

    if "pct_male" in df.columns:
        st.write("**Percentage Male Stock**")
        create_comparison_cards(df, "pct_male", "Male Stock %", "{:.1f}%")

    # Herd composition visualization
    if all(col in df.columns for col in [sheep_col, goats_col]) and "kpmd_registered" in df.columns:
        st.subheader("Herd Composition by KPMD Status")
        try:
            need = ["kpmd_registered", sheep_col, goats_col]
            dfh = df[need].copy()
            dfh = dfh.rename(columns={sheep_col: "Sheep", goats_col: "Goats"})

            # Melt for visualization
            long = dfh.melt(
                id_vars=["kpmd_registered"],
                value_vars=["Sheep", "Goats"],
                var_name="Species",
                value_name="Count",
            )
            long["Count"] = pd.to_numeric(long["Count"], errors="coerce")

            # Group by KPMD status and species
            comp = (
                long.groupby(["kpmd_registered", "Species"])["Count"]
                .mean()
                .reset_index()
            )
            comp["KPMD_Status"] = comp["kpmd_registered"].map(
                {1: "KPMD", 0: "Non-KPMD"}
            )

            fig = create_comparison_bar_chart(
                comp,
                x_col="KPMD_Status",
                y_col="Count",
                color_col="Species",
                title="Average Herd Composition by KPMD Status",
                barmode="group",
                text_format="{:.1f}",
                y_title="Average Count",
            )

            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Add time series view if panel data
            if (
                hasattr(processor, "is_panel_data")
                and processor.is_panel_data
                and "panel_wave" in df.columns
            ):
                st.subheader("Herd Composition Trends Over Time")

                trend_data = (
                    df.groupby(["panel_wave", "kpmd_registered"])[[sheep_col, goats_col]]
                    .mean()
                    .reset_index()
                )
                trend_data["KPMD Status"] = trend_data["kpmd_registered"].map(
                    {1: "KPMD", 0: "Non-KPMD"}
                )

                # Sheep trends
                fig_sheep = px.line(
                    trend_data,
                    x="panel_wave",
                    y=sheep_col,
                    color="KPMD Status",
                    title="Sheep Count Trends Over Time",
                    markers=True,
                )
                st.plotly_chart(fig_sheep, use_container_width=True)

                # Goat trends
                fig_goat = px.line(
                    trend_data,
                    x="panel_wave",
                    y=goats_col,
                    color="KPMD Status",
                    title="Goat Count Trends Over Time",
                    markers=True,
                )
                st.plotly_chart(fig_goat, use_container_width=True)

        except Exception as e:
            st.info(f"Herd composition visualization not available: {str(e)}")
    else:
        st.info("Herd composition data not available for visualization.")


def render_animal_health_tab(processor):
    """Render animal health indicators tab."""
    df = processor.df
    st.subheader("Animal Health Indicators")

    has_kpmd = "kpmd_registered" in df.columns

    # ---- Vaccination Rate ----
    vacc_col = (
        processor.column_mapping.get("vaccination")
        if hasattr(processor, "column_mapping")
        else None
    )
    if not vacc_col:
        # Fallback if not mapped
        vacc_col = None

    if vacc_col and vacc_col in df.columns and has_kpmd:
        st.write("**Vaccination Rate**")
        base = df.copy()
        base["vaccinated"] = base[vacc_col].apply(yn)

        tmp = base[["kpmd_registered", "vaccinated"]].copy()

        rows = []
        for s in [0, 1]:
            sub = tmp[tmp["kpmd_registered"] == s]["vaccinated"]
            rate = (
                sub.eq(1).sum() / sub.notna().sum() * 100
                if sub.notna().any()
                else np.nan
            )
            rows.append(
                {
                    "KPMD_Status": "KPMD" if s == 1 else "Non-KPMD",
                    "Rate": rate,
                }
            )

        df_rate = pd.DataFrame(rows).dropna(subset=["Rate"])

        if not df_rate.empty:
            fig = create_comparison_bar_chart(
                df_rate,
                x_col="KPMD_Status",
                y_col="Rate",
                title="Vaccination Rate",
                text_format="{:.1f}%",
                y_title="Percentage",
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    elif vacc_col and vacc_col in df.columns and not has_kpmd:
        st.info("Vaccination data available, but KPMD status is missing for comparison.")
    else:
        st.info("Vaccination data not available.")

    # ---- Treatment Rate ----
    treat_col = None
    if hasattr(processor, "column_mapping"):
        treat_col = processor.column_mapping.get("treat_small_ruminants")

    if not treat_col:
        treat_col = "D3. Did you treat small ruminants for disease in the last month?"

    if treat_col in df.columns and has_kpmd:
        st.write("**Treatment Rate**")
        base = df.copy()
        base["treated"] = base[treat_col].apply(yn)

        rows = []
        for s in [0, 1]:
            sub = base[base["kpmd_registered"] == s]["treated"]
            rate = (
                sub.eq(1).sum() / sub.notna().sum() * 100
                if sub.notna().any()
                else np.nan
            )
            rows.append(
                {
                    "KPMD_Status": "KPMD" if s == 1 else "Non-KPMD",
                    "Rate": rate,
                }
            )

        df_rate = pd.DataFrame(rows).dropna(subset=["Rate"])

        if not df_rate.empty:
            fig = create_comparison_bar_chart(
                df_rate,
                x_col="KPMD_Status",
                y_col="Rate",
                title="Treatment Rate",
                text_format="{:.1f}%",
                y_title="Percentage",
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    elif treat_col in df.columns and not has_kpmd:
        st.info("Disease treatment data available, but KPMD status is missing for comparison.")
    else:
        st.info("Disease treatment data not available.")

    # ---- Deworming Rate ----
    deworm_col = None
    if hasattr(processor, "column_mapping"):
        deworm_col = processor.column_mapping.get("deworm_small_ruminants")

    if not deworm_col:
        deworm_col = "D4. Did you deworm your small ruminants last month?"

    if deworm_col in df.columns and has_kpmd:
        st.write("**Deworming Rate**")
        base = df.copy()
        base["dewormed"] = base[deworm_col].apply(yn)

        rows = []
        for s in [0, 1]:
            sub = base[base["kpmd_registered"] == s]["dewormed"]
            rate = (
                sub.eq(1).sum() / sub.notna().sum() * 100
                if sub.notna().any()
                else np.nan
            )
            rows.append(
                {
                    "KPMD_Status": "KPMD" if s == 1 else "Non-KPMD",
                    "Rate": rate,
                }
            )

        df_rate = pd.DataFrame(rows).dropna(subset=["Rate"])

        if not df_rate.empty:
            fig = create_comparison_bar_chart(
                df_rate,
                x_col="KPMD_Status",
                y_col="Rate",
                title="Deworming Rate",
                text_format="{:.1f}%",
                y_title="Percentage",
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    elif deworm_col in df.columns and not has_kpmd:
        st.info("Deworming data available, but KPMD status is missing for comparison.")
    else:
        st.info("Deworming data not available.")

    # ---------- Disease Analysis ----------
    st.subheader("Disease Analysis")

    def _disease_rates(cols):
        if not has_kpmd:
            return pd.DataFrame()

        rows = []
        for col in cols:
            if col not in df.columns:
                continue
            # Clean label
            name = col.split("/")[-1].strip()
            name = " ".join(name.split())

            for s in [0, 1]:
                sub_raw = df[df["kpmd_registered"] == s][col]
                v = sub_raw.apply(yn)  # convert Yes/No, 0/1, etc. to 0/1
                rate = (
                    v.eq(1).sum() / v.notna().sum() * 100
                    if v.notna().any()
                    else np.nan
                )
                rows.append(
                    {
                        "Disease": name,
                        "Rate": rate,
                        "KPMD_Status": "KPMD" if s == 1 else "Non-KPMD",
                    }
                )

        dfp = pd.DataFrame(rows).dropna(subset=["Rate"])
        if not dfp.empty:
            dfp = (
                dfp.groupby(["Disease", "KPMD_Status"])["Rate"]
                .mean()
                .reset_index()
            )
        return dfp

    # Vaccination diseases
    if hasattr(processor, "vacc_disease_cols") and processor.vacc_disease_cols and has_kpmd:
        dfp = _disease_rates(processor.vacc_disease_cols)
        if not dfp.empty:
            fig = create_comparison_bar_chart(
                dfp,
                x_col="Disease",
                y_col="Rate",
                color_col="KPMD_Status",
                title="Vaccination Diseases by KPMD Status (%)",
                barmode="group",
                text_format="{:.1f}%",
                y_title="Percentage",
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    elif hasattr(processor, "vacc_disease_cols") and processor.vacc_disease_cols:
        st.info("Vaccination disease data available, but KPMD status is missing.")
    else:
        st.info("Vaccination disease data not available.")

    # Treatment diseases
    if hasattr(processor, "treat_disease_cols") and processor.treat_disease_cols and has_kpmd:
        dfp = _disease_rates(processor.treat_disease_cols)
        if not dfp.empty:
            fig = create_comparison_bar_chart(
                dfp,
                x_col="Disease",
                y_col="Rate",
                color_col="KPMD_Status",
                title="Treatment Diseases by KPMD Status (%)",
                barmode="group",
                text_format="{:.1f}%",
                y_title="Percentage",
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    elif hasattr(processor, "treat_disease_cols") and processor.treat_disease_cols:
        st.info("Treatment disease data available, but KPMD status is missing.")
    else:
        st.info("Treatment disease data not available.")

    # Vaccination providers
    prov_col = None
    if hasattr(processor, "column_mapping"):
        prov_col = processor.column_mapping.get("vaccination_provider")

    if not prov_col:
        prov_col = "D2. Who performed the small ruminants vaccinations in the last month?"

    if prov_col in df.columns and has_kpmd:
        try:
            provider_counts = (
                df[["kpmd_registered", prov_col]]
                .dropna(subset=[prov_col])
                .groupby(["kpmd_registered", prov_col])
                .size()
                .reset_index(name="count")
            )
            provider_counts["KPMD_Status"] = provider_counts["kpmd_registered"].map(
                {1: "KPMD", 0: "Non-KPMD"}
            )

            fig = create_comparison_bar_chart(
                provider_counts,
                x_col="KPMD_Status",
                y_col="count",
                color_col=prov_col,
                title="Vaccination Providers by KPMD Status",
                text_format="{:.0f}",
                y_title="Count",
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Vaccination provider data not available.")
    elif prov_col in df.columns and not has_kpmd:
        st.info("Vaccination provider data available, but KPMD status is missing.")
    else:
        st.info("Vaccination provider data not available.")


def render_productivity_tab(processor):
    """Render productivity indicators tab."""
    df = processor.df
    st.subheader("Small Ruminant Productivity Indicators")

    # Birth, mortality, and loss rates
    if "birth_rate_per_100" in df.columns:
        st.write("**Birth Rate (per 100 head)**")
        create_comparison_cards(df, "birth_rate_per_100", "Birth Rate", "{:.1f}")

    if "mortality_rate_per_100" in df.columns:
        st.write("**Mortality Rate (per 100 head)**")
        create_comparison_cards(df, "mortality_rate_per_100", "Mortality Rate", "{:.1f}")

    if "loss_rate_per_100" in df.columns:
        st.write("**Loss Rate (per 100 head)**")
        create_comparison_cards(df, "loss_rate_per_100", "Loss Rate", "{:.1f}")

    def _weighted_mean(rate_col, weight_col, df_group):
        """Weighted mean with total_sr as weight; fallback to simple mean."""
        r = pd.to_numeric(df_group[rate_col], errors="coerce")
        w = pd.to_numeric(df_group[weight_col], errors="coerce")
        if w.notna().sum() == 0 or w.fillna(0).sum() == 0:
            return r.mean()
        return (r.fillna(0) * w.fillna(0)).sum() / w.fillna(0).sum()

    has_rates = all(
        c in df.columns
        for c in ["birth_rate_per_100", "mortality_rate_per_100", "loss_rate_per_100"]
    )
    weight_col = "total_sr" if "total_sr" in df.columns else None
    has_kpmd = "kpmd_registered" in df.columns

    if has_rates and has_kpmd:
        st.subheader("Productivity Rates by KPMD Status")
        try:
            rows = []
            for s in [0, 1]:
                sub = df[df["kpmd_registered"] == s]

                if weight_col:
                    b = _weighted_mean("birth_rate_per_100", weight_col, sub)
                    m = _weighted_mean("mortality_rate_per_100", weight_col, sub)
                    l = _weighted_mean("loss_rate_per_100", weight_col, sub)
                else:
                    b = pd.to_numeric(
                        sub["birth_rate_per_100"], errors="coerce"
                    ).mean()
                    m = pd.to_numeric(
                        sub["mortality_rate_per_100"], errors="coerce"
                    ).mean()
                    l = pd.to_numeric(
                        sub["loss_rate_per_100"], errors="coerce"
                    ).mean()

                rows.append(
                    {
                        "KPMD_Status": "KPMD" if s == 1 else "Non-KPMD",
                        "Birth Rate": b,
                        "Mortality Rate": m,
                        "Loss Rate": l,
                    }
                )

            prod = pd.DataFrame(rows).melt(
                id_vars=["KPMD_Status"], var_name="Metric", value_name="Rate"
            )

            fig = create_comparison_bar_chart(
                prod,
                x_col="KPMD_Status",
                y_col="Rate",
                color_col="Metric",
                title="Productivity Rates by KPMD Status (per 100 head)",
                barmode="group",
                text_format="{:.1f}",
                y_title="Rate per 100 head",
            )

            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Add trend analysis if panel data
            if (
                hasattr(processor, "is_panel_data")
                and processor.is_panel_data
                and "panel_wave" in df.columns
            ):
                st.subheader("Productivity Trends Over Time")

                trend_data = (
                    df.groupby(["panel_wave", "kpmd_registered"])[
                        [
                            "birth_rate_per_100",
                            "mortality_rate_per_100",
                            "loss_rate_per_100",
                        ]
                    ]
                    .mean()
                    .reset_index()
                )
                trend_data["KPMD Status"] = trend_data["kpmd_registered"].map(
                    {1: "KPMD", 0: "Non-KPMD"}
                )

                metrics_to_plot = [
                    ("birth_rate_per_100", "Birth Rate"),
                    ("mortality_rate_per_100", "Mortality Rate"),
                    ("loss_rate_per_100", "Loss Rate"),
                ]

                for metric_col, metric_name in metrics_to_plot:
                    if metric_col in trend_data.columns:
                        fig_trend = px.line(
                            trend_data,
                            x="panel_wave",
                            y=metric_col,
                            color="KPMD Status",
                            title=f"{metric_name} Trends Over Time",
                            markers=True,
                            labels={metric_col: f"{metric_name} per 100"},
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)

        except Exception as e:
            st.info(
                f"Productivity rate data not available for visualization: {str(e)}"
            )
    elif has_rates and not has_kpmd:
        st.info(
            "Productivity rate data available, but KPMD status is missing for comparison."
        )
    else:
        st.info("Productivity rate data not available for visualization.")
