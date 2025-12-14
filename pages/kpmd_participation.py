# apmt_dashboard/pages/kpmd_participation.py
import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

from components.charts import create_comparison_bar_chart, create_distribution_chart
from utils.helpers import coalesce_first, to_num, yn


def render_kpmd_participation(processor):
    """Render the KPMD Participation dashboard page."""
    st.header("🤝 KPMD Participation")

    # Time period info for panel data
    if getattr(processor, "is_panel_data", False) and "panel_wave" in processor.df.columns:
        st.caption(f"**Analysis across:** {processor.df['panel_wave'].nunique()} time periods")

    # ------------------------------------------------------------------
    # Months in KPMD (A9)
    # ------------------------------------------------------------------
    months_col = coalesce_first(processor.df, ["A9. For how many months have you been participating in KPMD?"])

    if months_col:
        try:
            months_num = to_num(processor.df[months_col])
            months_clean = months_num.fillna(0)

            col1, col2, col3 = st.columns(3)

            with col1:
                avg_months = months_clean.mean()
                st.metric("Average Months in KPMD", f"{avg_months:.1f}")

            with col2:
                median_months = months_clean.median()
                st.metric("Median Months in KPMD", f"{median_months:.1f}")

            with col3:
                max_months = months_clean.max()
                st.metric("Maximum Months in KPMD", f"{max_months:.0f}")

            # Distribution (using a small helper DataFrame to ensure numeric)
            df_plot = pd.DataFrame({"Months in KPMD": months_clean})
            fig = create_distribution_chart(
                df_plot,
                col="Months in KPMD",
                title="Distribution of Months in KPMD",
                nbins=15,
            )
            if fig:
                fig.update_layout(xaxis_title="Months in KPMD")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.info(f"Could not parse A9 months: {e}")

    # ------------------------------------------------------------------
    # Trainings & interventions
    # ------------------------------------------------------------------
    render_trainings_section(processor)
    render_interventions_section(processor)

    # ------------------------------------------------------------------
    # Participation trends over time (panel)
    # ------------------------------------------------------------------
    if getattr(processor, "is_panel_data", False) and "panel_wave" in processor.df.columns:
        render_participation_trends(processor)


# ----------------------------------------------------------------------
# Trainings (B1)
# ----------------------------------------------------------------------
def render_trainings_section(processor):
    """Render trainings received section."""
    st.subheader("B1. Trainings Received (last 1 month)")

    b1_stem = (
        "B1. Have you received any of the following through KPMD in the past 1 month? "
        "(select all that apply)"
    )
    b1_cols = [c for c in processor.df.columns if c.startswith(b1_stem + "/")]

    if not b1_cols:
        st.info("B1 training option columns not found")
        return

    has_kpmd = "kpmd_registered" in processor.df.columns
    rows = []

    if has_kpmd:
        statuses = [0, 1]
        label_fn = lambda s: "KPMD" if s == 1 else "Non-KPMD"
    else:
        statuses = [None]
        label_fn = lambda s: "All households"

    for c in b1_cols:
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
                    "Training": name,
                    "Rate": rate,
                    "KPMD_Status": label_fn(s),
                }
            )

    dfp = pd.DataFrame(rows)

    fig = create_comparison_bar_chart(
        dfp,
        x_col="Training",
        y_col="Rate",
        color_col="KPMD_Status",
        title="KPMD Trainings in last 1 month (%)",
        barmode="group",
        text_format="{:.1f}%",
        y_title="Percentage",
    )

    if fig:
        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------
# Interventions (B2)
# ----------------------------------------------------------------------
def render_interventions_section(processor):
    """Render interventions received section."""
    st.subheader("B2. Interventions Received (last 1 month)")

    b2_stem = (
        "B2. Have you received any of the following through KPMD in the past 1 month? "
        "(select all that apply)"
    )
    b2_cols = [c for c in processor.df.columns if c.startswith(b2_stem + "/")]

    if not b2_cols:
        st.info("B2 intervention option columns not found")
        return

    has_kpmd = "kpmd_registered" in processor.df.columns
    rows = []

    if has_kpmd:
        statuses = [0, 1]
        label_fn = lambda s: "KPMD" if s == 1 else "Non-KPMD"
    else:
        statuses = [None]
        label_fn = lambda s: "All households"

    for c in b2_cols:
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
                    "Intervention": name,
                    "Rate": rate,
                    "KPMD_Status": label_fn(s),
                }
            )

    dfp = pd.DataFrame(rows)

    fig = create_comparison_bar_chart(
        dfp,
        x_col="Intervention",
        y_col="Rate",
        color_col="KPMD_Status",
        title="KPMD Interventions in last 1 month (%)",
        barmode="group",
        text_format="{:.1f}%",
        y_title="Percentage",
    )

    if fig:
        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------
# Participation trends over time (panel)
# ----------------------------------------------------------------------
def render_participation_trends(processor):
    """Render KPMD participation trends over time."""
    st.subheader("KPMD Participation Trends Over Time")

    if "kpmd_registered" not in processor.df.columns or "panel_wave" not in processor.df.columns:
        st.info("Panel wave or KPMD registration data not available for trends.")
        return

    # Aggregate participation rate & counts by time period
    participation_trend = (
        processor.df.groupby("panel_wave")["kpmd_registered"]
        .agg(["mean", "count"])
        .reset_index()
    )
    if participation_trend.empty:
        st.info("No panel participation data available.")
        return

    participation_trend["mean_pct"] = participation_trend["mean"] * 100.0

    # Use secondary y-axis: line = participation rate, bars = count of households
    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        subplot_titles=["KPMD Participation Rate and Sample Size Over Time"],
    )

    # Line: participation rate (%)
    fig.add_trace(
        px.line(
            participation_trend,
            x="panel_wave",
            y="mean_pct",
            markers=True,
        ).data[0],
        secondary_y=True,
    )

    # Bar: number of households
    fig.add_trace(
        px.bar(
            participation_trend,
            x="panel_wave",
            y="count",
        ).data[0],
        secondary_y=False,
    )

    fig.update_layout(
        title_text="KPMD Participation Rate Over Time",
        legend_title_text="",
    )

    fig.update_yaxes(
        title_text="Number of Households", secondary_y=False
    )
    fig.update_yaxes(
        title_text="Participation Rate (%)", secondary_y=True
    )
    fig.update_xaxes(title_text="Time Period")

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------------
    # New vs returning participants analysis
    # --------------------------------------------------------------
    if "panel_hhid" in processor.df.columns:
        st.subheader("New KPMD Participants Over Time")

        # Identify first wave at which each household is registered
        reg = processor.df[processor.df["kpmd_registered"] == 1].copy()
        if reg.empty:
            st.info("No registered KPMD participants found for new-participant analysis.")
            return

        first_reg = (
            reg.groupby("panel_hhid")["panel_wave"]
            .min()
            .reset_index()
            .rename(columns={"panel_wave": "first_kpmd_wave"})
        )

        new_participants = (
            first_reg.groupby("first_kpmd_wave")["panel_hhid"]
            .nunique()
            .reset_index()
            .rename(
                columns={
                    "first_kpmd_wave": "wave",
                    "panel_hhid": "new_participants",
                }
            )
        )

        if not new_participants.empty:
            fig2 = px.bar(
                new_participants,
                x="wave",
                y="new_participants",
                title="New KPMD Participants by Time Period",
                labels={"wave": "Time Period", "new_participants": "New Participants"},
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No new KPMD participants identified across time periods.")
