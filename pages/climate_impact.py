# apmt_dashboard/pages/climate_impact.py
import streamlit as st
import pandas as pd
import plotly.express as px

from components.comparison_cards import create_comparison_cards
from components.charts import create_comparison_bar_chart
from utils.helpers import yn, one_hot_multiselect
from data_processing.calculations import calculate_climate_resilience


def render_climate_impact(processor):
    """Render the Climate Impact dashboard page."""
    st.header("🌦️ Climate Impact")

    # Calculate climate resilience metrics (defensive)
    try:
        calculate_climate_resilience(processor)
    except Exception as e:
        st.info(f"Could not compute derived climate resilience metrics: {e}")

    tab1, tab2, tab3 = st.tabs(
        ["Adaptation Measures", "Barriers to Adaptation", "Climate Resilience"]
    )

    with tab1:
        render_adaptation_measures_tab(processor)

    with tab2:
        render_barriers_tab(processor)

    with tab3:
        render_climate_resilience_tab(processor)


# --------------------------------------------------------------------
# TAB 1: Adaptation measures
# --------------------------------------------------------------------
def render_adaptation_measures_tab(processor):
    """Render adaptation measures tab."""
    st.subheader("Adaptation Measures")

    colmap = getattr(processor, "column_mapping", {}) or {}
    j1 = colmap.get("adaptation_measures")

    # --- Overall adaptation rate / cards ---
    if j1 and j1 in processor.df.columns:
        tmp = processor.df.copy()
        tmp["adapted"] = tmp[j1].apply(yn).astype(int)

        if "kpmd_registered" in tmp.columns:
            # Use comparison cards (KPMD vs Non-KPMD)
            create_comparison_cards(tmp, "adapted", "Adaptation Rate", "{:.1%}")
        else:
            # Fallback: overall rate only
            rate = tmp["adapted"].mean() if len(tmp) else 0.0
            st.metric("Adaptation Rate (all households)", f"{rate:.1%}")
    else:
        st.info("Climate adaptation data (J1) not available")

    # --- Adaptation strategies ---
    st.subheader("Adaptation Strategies")

    j2_stem = "J2. Which adapatations measures are you using?"  # spelling matches questionnaire
    strategy_cols = [
        c
        for c in processor.df.columns
        if c.startswith(j2_stem + "/") and "Other" not in c
    ]

    # Helper: prepare “statuses” depending on presence of kpmd_registered
    has_kpmd = "kpmd_registered" in processor.df.columns

    if strategy_cols:
        rows = []

        if has_kpmd:
            statuses = [0, 1]
            label_fn = lambda s: "KPMD" if s == 1 else "Non-KPMD"
        else:
            statuses = [None]
            label_fn = lambda s: "All households"

        for c in strategy_cols:
            name = c.split("/")[-1]
            for s in statuses:
                if s is None:
                    sub = processor.df
                else:
                    sub = processor.df[processor.df["kpmd_registered"] == s]

                if len(sub) == 0:
                    rate = 0.0
                else:
                    rate = (
                        pd.to_numeric(
                            sub[c].astype(str).replace({"1": 1, "0": 0}),
                            errors="coerce",
                        )
                        .fillna(0)
                        .mean()
                        * 100
                    )

                rows.append(
                    {
                        "Strategy": name,
                        "Usage_Rate": rate,
                        "KPMD_Status": label_fn(s),
                    }
                )

        dfp = pd.DataFrame(rows)

        fig = create_comparison_bar_chart(
            dfp,
            x_col="Strategy",
            y_col="Usage_Rate",
            color_col="KPMD_Status",
            title="Adaptation Strategies by KPMD Status (%)",
            barmode="group",
            text_format="{:.1f}%",
            y_title="Usage Rate (%)",
        )

        if fig:
            st.plotly_chart(fig, use_container_width=True)

    elif j2_stem in processor.df.columns:
        # Try to parse as a single multiselect column
        dummies = one_hot_multiselect(processor.df[j2_stem])

        if not dummies.empty:
            if has_kpmd:
                tmp = pd.concat(
                    [processor.df[["kpmd_registered"]], dummies], axis=1
                )
                long = tmp.melt(
                    id_vars=["kpmd_registered"],
                    var_name="Strategy",
                    value_name="flag",
                )
                agg = (
                    long.groupby(["Strategy", "kpmd_registered"])["flag"]
                    .mean()
                    .mul(100)
                    .reset_index()
                )
                agg["KPMD_Status"] = agg["kpmd_registered"].map(
                    {1: "KPMD", 0: "Non-KPMD"}
                )
            else:
                # No KPMD; just overall
                long = dummies.melt(
                    var_name="Strategy", value_name="flag"
                )
                agg = (
                    long.groupby("Strategy")["flag"]
                    .mean()
                    .mul(100)
                    .reset_index()
                )
                agg["KPMD_Status"] = "All households"

            fig = create_comparison_bar_chart(
                agg,
                x_col="Strategy",
                y_col="flag",
                color_col="KPMD_Status",
                title="Adaptation Strategies by KPMD Status (%)",
                barmode="group",
                text_format="{:.1f}%",
                y_title="Usage Rate (%)",
            )

            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Adaptation strategy data (J2) not available.")
    else:
        st.info("Adaptation strategy data (J2) not available.")


# --------------------------------------------------------------------
# TAB 2: Barriers to adaptation
# --------------------------------------------------------------------
def render_barriers_tab(processor):
    """Render barriers to adaptation tab."""
    st.subheader("Barriers to Adaptation")

    colmap = getattr(processor, "column_mapping", {}) or {}
    j1 = colmap.get("adaptation_measures")  # yes/no for adapting
    j3_stem = "J3. Why not?"

    # Base: non-adapting households if J1 is available, else all households
    if j1 and j1 in processor.df.columns:
        base = processor.df[processor.df[j1].apply(yn) == 0].copy()
        if base.empty:
            st.info("No non-adapting households found in the data; showing all households instead.")
            base = processor.df.copy()
    else:
        base = processor.df.copy()
        st.info("Could not filter to non-adapting households – showing all data.")

    barrier_cols = [
        c
        for c in base.columns
        if c.startswith(j3_stem + "/") and "Other" not in c
    ]

    has_kpmd = "kpmd_registered" in base.columns

    if barrier_cols:
        rows = []

        if has_kpmd:
            statuses = [0, 1]
            label_fn = lambda s: "KPMD" if s == 1 else "Non-KPMD"
        else:
            statuses = [None]
            label_fn = lambda s: "All households"

        for c in barrier_cols:
            name = c.split("/")[-1]
            for s in statuses:
                if s is None:
                    sub = base
                else:
                    sub = base[base["kpmd_registered"] == s]

                if len(sub) == 0:
                    rate = 0.0
                else:
                    rate = (
                        pd.to_numeric(
                            sub[c].astype(str).replace({"1": 1, "0": 0}),
                            errors="coerce",
                        )
                        .fillna(0)
                        .mean()
                        * 100
                    )

                rows.append(
                    {
                        "Barrier": name,
                        "Rate": rate,
                        "KPMD_Status": label_fn(s),
                    }
                )

        dfp = pd.DataFrame(rows)

        fig = create_comparison_bar_chart(
            dfp,
            x_col="Barrier",
            y_col="Rate",
            color_col="KPMD_Status",
            title="Barriers to Adaptation by KPMD Status (%)",
            barmode="group",
            text_format="{:.1f}%",
            y_title="Percentage",
        )

        if fig:
            st.plotly_chart(fig, use_container_width=True)

    elif j3_stem in base.columns:
        # Try to parse as multiselect
        dummies = one_hot_multiselect(base[j3_stem])

        if not dummies.empty:
            if has_kpmd:
                tmp = pd.concat(
                    [base[["kpmd_registered"]], dummies], axis=1
                )
                long = tmp.melt(
                    id_vars=["kpmd_registered"],
                    var_name="Barrier",
                    value_name="flag",
                )
                agg = (
                    long.groupby(["Barrier", "kpmd_registered"])["flag"]
                    .mean()
                    .mul(100)
                    .reset_index()
                )
                agg["KPMD_Status"] = agg["kpmd_registered"].map(
                    {1: "KPMD", 0: "Non-KPMD"}
                )
            else:
                long = dummies.melt(
                    var_name="Barrier", value_name="flag"
                )
                agg = (
                    long.groupby("Barrier")["flag"]
                    .mean()
                    .mul(100)
                    .reset_index()
                )
                agg["KPMD_Status"] = "All households"

            fig = create_comparison_bar_chart(
                agg,
                x_col="Barrier",
                y_col="flag",
                color_col="KPMD_Status",
                title="Barriers to Adaptation by KPMD Status (%)",
                barmode="group",
                text_format="{:.1f}%",
                y_title="Percentage",
            )

            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No barrier data available for non-adapting households.")
    else:
        st.info("Barriers to adaptation data (J3) not available.")


# --------------------------------------------------------------------
# TAB 3: Climate resilience indicators
# --------------------------------------------------------------------
def render_climate_resilience_tab(processor):
    """Render climate resilience indicators tab."""
    st.subheader("Climate Resilience Indicators")

    col1, col2, col3 = st.columns(3)

    # Overall KPMD participation
    with col1:
        if "kpmd_registered" in processor.df.columns:
            kpmd_participation = processor.df["kpmd_registered"].mean() * 100
            st.metric("Overall KPMD Participation Rate", f"{kpmd_participation:.1f}%")

    # Average resilience score
    with col2:
        if "resilience_score" in processor.df.columns:
            resilience_data = processor.df["resilience_score"].dropna()
            if len(resilience_data) > 0:
                avg_resilience = resilience_data.mean()
                st.metric("Average Resilience Score", f"{avg_resilience:.1f}")

    # Adaptation score (usually a 0–1 index)
    with col3:
        if "adaptation_score" in processor.df.columns:
            adaptation_vals = processor.df["adaptation_score"].dropna()
            if len(adaptation_vals) > 0:
                adaptation_rate = adaptation_vals.mean() * 100
                st.metric(
                    "Households Implementing Adaptation",
                    f"{adaptation_rate:.1f}%",
                )

    # Additional resilience indicators
    st.subheader("Additional Resilience Indicators")

    if "total_sr" in processor.df.columns:
        sr_clean = pd.to_numeric(
            processor.df["total_sr"], errors="coerce"
        ).dropna()
        if len(sr_clean) > 0:
            herd_size_median = sr_clean.median()
            large_herds = (sr_clean > herd_size_median).mean() * 100

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Median Herd Size", f"{herd_size_median:.1f}")
            with col2:
                st.metric(
                    "Households with Above-Median Herd Size",
                    f"{large_herds:.1f}%",
                )

    # Resilience by KPMD status
    if (
        "resilience_score" in processor.df.columns
        and "kpmd_registered" in processor.df.columns
    ):
        st.subheader("Resilience by KPMD Status")

        base = processor.df[
            processor.df["resilience_score"].notna()
        ].copy()
        if not base.empty:
            resilience_by_kpmd = (
                base.groupby("kpmd_registered")["resilience_score"]
                .agg(["mean", "count"])
                .reset_index()
            )
            resilience_by_kpmd["KPMD Status"] = resilience_by_kpmd[
                "kpmd_registered"
            ].map({1: "KPMD", 0: "Non-KPMD"})

            fig = create_comparison_bar_chart(
                resilience_by_kpmd,
                x_col="KPMD Status",
                y_col="mean",
                title="Average Resilience Score by KPMD Status",
                text_format="{:.1f}",
                y_title="Resilience Score",
            )

            if fig:
                st.plotly_chart(fig, use_container_width=True)

    # Panel data trends if available
    if (
        getattr(processor, "is_panel_data", False)
        and "panel_wave" in processor.df.columns
        and "resilience_score" in processor.df.columns
    ):
        st.subheader("Resilience Trends Over Time")

        trend_base = processor.df[
            processor.df["resilience_score"].notna()
        ].copy()
        if not trend_base.empty:
            if "kpmd_registered" in trend_base.columns:
                resilience_trend = (
                    trend_base.groupby(["panel_wave", "kpmd_registered"])[
                        "resilience_score"
                    ]
                    .mean()
                    .reset_index()
                )
                resilience_trend["KPMD Status"] = resilience_trend[
                    "kpmd_registered"
                ].map({1: "KPMD", 0: "Non-KPMD"})
                color = "KPMD Status"
            else:
                resilience_trend = (
                    trend_base.groupby("panel_wave")["resilience_score"]
                    .mean()
                    .reset_index()
                )
                color = None

            fig = px.line(
                resilience_trend,
                x="panel_wave",
                y="resilience_score",
                color=color,
                title="Resilience Score Trends Over Time",
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)
