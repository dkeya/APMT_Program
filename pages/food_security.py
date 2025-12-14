# apmt_dashboard/pages/food_security.py
import streamlit as st
import pandas as pd
import plotly.express as px


def render_food_security(processor):
    """Render the Food Security (rCSI) page."""
    df = processor.df

    st.header("🍚 Food Security — Reduced Coping Strategies Index (30-day)")

    # ---------------- Top summary metrics ----------------
    col1, col2, col3 = st.columns(3)

    with col1:
        if "rcsi_30" in df.columns:
            avg_rcsi = pd.to_numeric(df["rcsi_30"], errors="coerce").mean()
            if pd.notna(avg_rcsi):
                st.metric("Average rCSI-30", f"{avg_rcsi:.1f}")
            else:
                st.metric("Average rCSI-30", "N/A")
        else:
            st.metric("Average rCSI-30", "N/A")

    with col2:
        if "food_worry" in df.columns:
            worry_rate = pd.to_numeric(df["food_worry"], errors="coerce").mean() * 100
            st.metric("Households Worried about Food", f"{worry_rate:.1f}%")
        else:
            st.metric("Households Worried about Food", "N/A")

    with col3:
        if "insured_sr" in df.columns:
            avg_insured = pd.to_numeric(df["insured_sr"], errors="coerce").mean()
            st.metric("Avg. # SR Insured", f"{avg_insured:.1f}")
        else:
            st.metric("Avg. # SR Insured", "N/A")

    # ---------------- Overall rCSI distribution ----------------
    if "rcsi_30" in df.columns:
        rcsi = pd.to_numeric(df["rcsi_30"], errors="coerce")
        rcsi = rcsi[rcsi.notna()]

        if len(rcsi) > 0:
            st.subheader("Distribution of rCSI (30 days)")
            fig = px.histogram(
                rcsi,
                x=rcsi,
                nbins=30,
                title="Distribution of rCSI (30 days)",
                labels={"x": "rCSI score (30 days)", "y": "Count"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("rCSI exists but all values are missing or non-numeric.")
    else:
        st.info("No rCSI (rcsi_30) column found in the dataset.")

    # ---------------- rCSI by KPMD registration ----------------
    st.subheader("🍚 Food Security — Reduced Coping Strategies Index (30-day)")

    if "rcsi_30" in df.columns and "kpmd_registered" in df.columns:
        df_rcsi = df[["rcsi_30", "kpmd_registered"]].copy()
        df_rcsi["rcsi_30"] = pd.to_numeric(df_rcsi["rcsi_30"], errors="coerce")
        df_rcsi = df_rcsi[df_rcsi["rcsi_30"].notna()]

        if df_rcsi.empty:
            st.info("No rCSI values available for registration comparison.")
            return

        df_rcsi["KPMD Status"] = df_rcsi["kpmd_registered"].map({1: "KPMD", 0: "Non-KPMD"})
        df_rcsi["KPMD Status"] = pd.Categorical(
            df_rcsi["KPMD Status"],
            categories=["Non-KPMD", "KPMD"],
            ordered=True,
        )

        # Summary cards
        colA, colB = st.columns(2)
        with colA:
            m_kpmd = df_rcsi[df_rcsi["KPMD Status"] == "KPMD"]["rcsi_30"].mean()
            st.metric(
                "KPMD — Avg rCSI (30d)",
                f"{m_kpmd:.1f}" if pd.notna(m_kpmd) else "N/A",
            )
        with colB:
            m_non = df_rcsi[df_rcsi["KPMD Status"] == "Non-KPMD"]["rcsi_30"].mean()
            st.metric(
                "Non-KPMD — Avg rCSI (30d)",
                f"{m_non:.1f}" if pd.notna(m_non) else "N/A",
            )

        # Box plot
        fig2 = px.box(
            df_rcsi,
            x="KPMD Status",
            y="rcsi_30",
            color="KPMD Status",
            category_orders={"KPMD Status": ["Non-KPMD", "KPMD"]},
            labels={"KPMD Status": "Registration", "rcsi_30": "rCSI (30-day)"},
            title="rCSI by Registration",
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ---------------- rCSI trends over time (if panel) ----------------
        if getattr(processor, "is_panel_data", False) and "panel_wave" in df.columns:
            st.subheader("Food Security Trends Over Time")

            trend = (
                df.copy()
                .assign(
                    rcsi_30=pd.to_numeric(df["rcsi_30"], errors="coerce"),
                    kpmd_registered=df["kpmd_registered"],
                )
                .dropna(subset=["rcsi_30", "panel_wave"])
            )

            if not trend.empty:
                trend["KPMD Status"] = trend["kpmd_registered"].map(
                    {1: "KPMD", 0: "Non-KPMD"}
                )
                fig3 = px.line(
                    trend,
                    x="panel_wave",
                    y="rcsi_30",
                    color="KPMD Status",
                    markers=True,
                    title="rCSI Trends Over Time by KPMD Status",
                    labels={"rcsi_30": "rCSI score (30 days)", "panel_wave": "Time period"},
                )
                st.plotly_chart(fig3, use_container_width=True)

    else:
        st.info("Missing rCSI or registration fields to plot rCSI by Registration.")
