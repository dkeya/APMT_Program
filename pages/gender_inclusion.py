# apmt_dashboard/pages/gender_inclusion.py
import streamlit as st
import pandas as pd

from components.comparison_cards import create_comparison_cards
from components.charts import create_comparison_bar_chart
from utils.helpers import yn


def render_gender_inclusion(processor):
    """Render the Gender Inclusion dashboard page."""
    st.header("♀️ Gender Inclusion")

    tab1, tab2, tab3 = st.tabs(["Decision Making", "KPMD Participation", "Income Control"])

    with tab1:
        render_decision_making_tab(processor)

    with tab2:
        render_kpmd_participation_tab(processor)

    with tab3:
        render_income_control_tab(processor)


# --------- helper detectors ---------


def _detect_decision_stem(df: pd.DataFrame) -> str | None:
    """
    Infer the stem for 'Who makes decisions about livestock sale' from column names.

    Looks for columns that:
      - contain '/Head', '/Spouse', '/Daughter', '/Wife', or '/Son'
      - AND have 'decision' in the prefix text.
    """
    candidate_roles = ["/Head", "/Spouse", "/Daughter", "/Wife", "/Son"]
    stems = set()

    for c in df.columns:
        lower = str(c).lower()
        if "decision" in lower:
            for role in candidate_roles:
                if role in c:
                    stems.add(str(c).split("/", 1)[0])
                    break

    # Fallback: look for "who makes" wording
    if not stems:
        for c in df.columns:
            lower = str(c).lower()
            if "who makes" in lower:
                for role in candidate_roles:
                    if role in c:
                        stems.add(str(c).split("/", 1)[0])
                        break

    return sorted(stems)[0] if stems else None


def _detect_income_control_stem(df: pd.DataFrame) -> str | None:
    """
    Infer the stem for 'Who controls the income from livestock sales' from column names.

    Looks for columns that:
      - contain '/Head', '/Spouse', '/Daughter', '/Wife', or '/Son'
      - AND have 'control' or 'income control' in the prefix text.
    """
    candidate_roles = ["/Head", "/Spouse", "/Daughter", "/Wife", "/Son"]
    stems = set()

    for c in df.columns:
        lower = str(c).lower()
        if ("control" in lower) or ("income" in lower and "who" in lower):
            for role in candidate_roles:
                if role in c:
                    stems.add(str(c).split("/", 1)[0])
                    break

    # Fallback: any '/Head' column with "income" and "control" in the question text
    if not stems:
        for c in df.columns:
            lower = str(c).lower()
            if "income" in lower and "control" in lower:
                for role in candidate_roles:
                    if role in c:
                        stems.add(str(c).split("/", 1)[0])
                        break

    return sorted(stems)[0] if stems else None


# --------- tabs ---------


def render_decision_making_tab(processor):
    """Render decision making tab."""
    st.subheader("Livestock Sale Decision Making")

    df = processor.df

    # 1) Try mapping from processor.gender_columns, if present
    decision_col = ""
    if hasattr(processor, "gender_columns"):
        decision_col = processor.gender_columns.get("decision_making", "") or ""

    # 2) If mapping missing/empty, auto-detect from column names
    if not decision_col:
        decision_col = _detect_decision_stem(df)

    if not decision_col:
        st.info("Decision making data not available (no suitable columns found).")
        return

    # Sub-columns like ".../Head", ".../Spouse", etc.
    decision_cols = [
        c
        for c in df.columns
        if str(c).startswith(decision_col) and "Other" not in str(c) and "/" in str(c)
    ]

    if not decision_cols:
        st.info("Decision making role columns not found.")
        return

    rows = []
    for c in decision_cols:
        role = str(c).split("/")[-1]
        for s in [0, 1]:
            sub = df[df["kpmd_registered"] == s]
            if len(sub):
                rate = (
                    pd.to_numeric(
                        sub[c].astype(str).replace({"1": 1, "0": 0}),
                        errors="coerce",
                    )
                    .fillna(0)
                    .mean()
                    * 100
                )
            else:
                rate = 0.0

            rows.append(
                {
                    "Role": role,
                    "Involvement_Rate": rate,
                    "KPMD_Status": "KPMD" if s == 1 else "Non-KPMD",
                }
            )

    dfp = pd.DataFrame(rows)

    if dfp.empty:
        st.info("No decision making data after processing.")
        return

    # Women's involvement summary
    women_roles = ["Spouse", "Daughter", "Wife"]
    women_df = dfp[dfp["Role"].isin(women_roles)]

    if not women_df.empty:
        st.write("**Women's Involvement in Decision Making**")
        women_summary = women_df.groupby("KPMD_Status")["Involvement_Rate"].mean().reset_index()

        for _, row in women_summary.iterrows():
            st.metric(
                f"{row['KPMD_Status']} - Women Involvement",
                f"{row['Involvement_Rate']:.1f}%",
            )

    # Visualization
    fig = create_comparison_bar_chart(
        dfp,
        x_col="Role",
        y_col="Involvement_Rate",
        color_col="KPMD_Status",
        title="Decision Making Roles by KPMD Status (%)",
        barmode="group",
        text_format="{:.1f}%",
        y_title="Involvement Rate (%)",
    )

    if fig:
        st.plotly_chart(fig, use_container_width=True)

def render_kpmd_participation_tab(processor):
    """Render KPMD participation by gender tab."""
    st.subheader("KPMD Participation by Gender")

    df = processor.df

    # ----- Basic stacked chart by gender -----
    if "Gender" in df.columns and "kpmd_registered" in df.columns:
        g = df[df["Gender"].notna()]

        if len(g) > 0:
            ct = pd.crosstab(g["Gender"], g["kpmd_registered"], normalize="index") * 100
            ct = ct.reset_index().rename(columns={0: "Non-KPMD", 1: "KPMD"})

            melted = ct.melt(
                id_vars=["Gender"],
                value_vars=["KPMD", "Non-KPMD"],
                var_name="KPMD_Status",
                value_name="Percentage",
            )

            fig = create_comparison_bar_chart(
                melted,
                x_col="Gender",
                y_col="Percentage",
                color_col="KPMD_Status",
                title="KPMD Participation by Gender (%)",
                barmode="stack",
                text_format="{:.1f}%",
                y_title="Percentage",
            )

            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No gender data available.")
            return
    else:
        st.info("Gender or KPMD data not available.")
        return

    # ----- Female-headed households in KPMD -----
    st.subheader("Female-Headed Households in KPMD")

    # Try to find a 'household head' column
    hh_head_col = None
    if hasattr(processor, "gender_columns"):
        hh_head_col = processor.gender_columns.get("household_head", "")

    # If mapping not found, search heuristically
    if not hh_head_col or hh_head_col not in df.columns:
        for c in df.columns:
            lc = c.lower()
            if "head" in lc and "house" in lc:
                hh_head_col = c
                break

    df2 = df[df["Gender"].notna()].copy()

    if hh_head_col and hh_head_col in df2.columns:
        # Use explicit household-head indicator
        heads_mask = df2[hh_head_col].apply(yn) == 1
    else:
        # Fallback: treat all respondents as heads
        heads_mask = pd.Series(True, index=df2.index)

    female_heads = df2[(df2["Gender"] == "Female") & heads_mask]
    male_heads = df2[(df2["Gender"] == "Male") & heads_mask]

    if not female_heads.empty:
        pct_kpmd_f = female_heads["kpmd_registered"].mean() * 100
        # Main headline metric
        st.metric("Female-Headed Households in KPMD", f"{pct_kpmd_f:.1f}%")

        # Optional comparison vs male-headed
        if not male_heads.empty:
            pct_kpmd_m = male_heads["kpmd_registered"].mean() * 100
            diff = pct_kpmd_f - pct_kpmd_m

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Female-Headed KPMD", f"{pct_kpmd_f:.1f}%")
            with col2:
                st.metric("Male-Headed KPMD", f"{pct_kpmd_m:.1f}%")

            if diff > 0:
                st.success(
                    f"Female-headed households are {diff:.1f}% more likely to participate in KPMD"
                )
            elif diff < 0:
                st.warning(
                    f"Female-headed households are {abs(diff):.1f}% less likely to participate in KPMD"
                )
    else:
        st.info("No female-headed households found in the data.")

def render_income_control_tab(processor):
    """Render income control tab."""
    st.subheader("Income Control and Usage")

    df = processor.df

    # 1) Try mapping from processor.gender_columns, if present
    income_col = (
        processor.gender_columns.get("income_control", "")
        if hasattr(processor, "gender_columns")
        else ""
    )

    # 2) If missing, auto-detect from column names
    if not income_col:
        income_col = _detect_income_control_stem(df)

    if not income_col:
        st.info("Income control data not available (no suitable columns found).")
        return

    income_cols = [
        c
        for c in df.columns
        if str(c).startswith(income_col) and "Other" not in str(c) and "/" in str(c)
    ]

    if not income_cols:
        st.info("Income control role columns not found.")
        return

    rows = []
    for c in income_cols:
        role = str(c).split("/")[-1]
        for s in [0, 1]:
            sub = df[df["kpmd_registered"] == s]
            if len(sub):
                rate = (
                    pd.to_numeric(
                        sub[c].astype(str).replace({"1": 1, "0": 0}),
                        errors="coerce",
                    )
                    .fillna(0)
                    .mean()
                    * 100
                )
            else:
                rate = 0.0

            rows.append(
                {
                    "Role": role,
                    "Control_Rate": rate,
                    "KPMD_Status": "KPMD" if s == 1 else "Non-KPMD",
                }
            )

    dfp = pd.DataFrame(rows)

    if dfp.empty:
        st.info("No income control data after processing.")
        return

    # Women's control summary
    women_roles = ["Spouse", "Daughter", "Wife"]
    women_df = dfp[dfp["Role"].isin(women_roles)]

    if not women_df.empty:
        st.write("**Women's Control Over Livestock Income**")
        women_summary = women_df.groupby("KPMD_Status")["Control_Rate"].mean().reset_index()

        for _, row in women_summary.iterrows():
            st.metric(
                f"{row['KPMD_Status']} - Women Control",
                f"{row['Control_Rate']:.1f}%",
            )

    fig = create_comparison_bar_chart(
        dfp,
        x_col="Role",
        y_col="Control_Rate",
        color_col="KPMD_Status",
        title="Income Control Roles by KPMD Status (%)",
        barmode="group",
        text_format="{:.1f}%",
        y_title="Control Rate (%)",
    )

    if fig:
        st.plotly_chart(fig, use_container_width=True)
