# apmt_dashboard/pages/county_comparator.py
import streamlit as st
import pandas as pd

from utils.helpers import yn, to_num


def render_county_comparator(processor):
    """Render the County Comparator dashboard page."""
    st.header("📊 County Comparator")

    # Basic sanity checks
    if "County" not in processor.df.columns:
        st.info("County column missing.")
        return

    counties = sorted(processor.df["County"].dropna().unique())
    if len(counties) < 2:
        st.info("Need at least two counties for comparison.")
        return

    # County selection
    col1, col2 = st.columns(2)
    left = col1.selectbox("Left county", counties, key="cmpL")
    right = col2.selectbox(
        "Right county",
        counties,
        index=1 if len(counties) > 1 else 0,
        key="cmpR",
    )

    if left == right:
        st.info("Select two different counties to compare.")
        return

    def slice_county(c):
        return processor.df[processor.df["County"] == c]

    A, B = slice_county(left), slice_county(right)

    # ------------------------------------------------------------------
    # Core comparison metrics
    # ------------------------------------------------------------------
    st.subheader(f"Comparison: {left} vs {right}")

    metrics = [
        # label, column, format, function(df [, col])
        ("Households", "", "{:,}", lambda df: len(df)),
        (
            "KPMD Participation",
            "kpmd_registered",
            "{:.1%}",
            lambda df, col: df[col].mean() if col in df.columns else 0,
        ),
        (
            "Avg Total SR",
            "total_sr",
            "{:.1f}",
            lambda df, col: df[col].mean() if col in df.columns else 0,
        ),
        (
            "Avg Net Profit (KES)",
            "net_profit",
            "KES {:,.0f}",
            lambda df, col: df[col].mean() if col in df.columns else 0,
        ),
    ]

    # Optional metrics using mapping from processor (if present)
    mapping = getattr(processor, "column_mapping", {}) or {}

    additional_metrics = [
        ("Vaccination Rate", mapping.get("vaccination", ""), "{:.1%}"),
        (
            "Fodder Purchase Rate",
            "B5a. Did you purchase fodder in the last 1 month?",
            "{:.1%}",
        ),
        (
            "Adaptation Measures",
            mapping.get("adaptation_measures", ""),
            "{:.1%}",
        ),
    ]

    for label, col, fmt in additional_metrics:
        if col and col in processor.df.columns:
            # Use default argument to avoid late-binding closure bug
            metrics.append(
                (
                    label,
                    col,
                    fmt,
                    lambda df, col=col: df[col].apply(yn).mean()
                    if col in df.columns
                    else 0,
                )
            )

    # Display side-by-side metrics
    colA, colB = st.columns(2)

    for label, col, fmt, func in metrics:
        with colA:
            if col:
                value_left = func(A, col)
            else:
                value_left = func(A)
            st.metric(f"{label} — {left}", fmt.format(value_left))

        with colB:
            if col:
                value_right = func(B, col)
            else:
                value_right = func(B)
            st.metric(f"{label} — {right}", fmt.format(value_right))

    # ------------------------------------------------------------------
    # Price comparisons
    # ------------------------------------------------------------------
    st.subheader("Price Comparisons")

    price_metrics = [
        ("Avg Sheep Price (KPMD)", "E1c. What was the average price per sheep last month?"),
        ("Avg Goat Price (KPMD)", "E2c. What was the average price per goat last month?"),
        ("Avg Sheep Price (Non-KPMD)", "E3d. What was the average price per sheep last month?"),
        ("Avg Goat Price (Non-KPMD)", "E4d. What was the average price per goat last month?"),
    ]

    price_colA, price_colB = st.columns(2)

    for label, price_col in price_metrics:
        if price_col in processor.df.columns:
            with price_colA:
                left_vals = to_num(A[price_col]) if price_col in A.columns else pd.Series(dtype=float)
                left_price = left_vals.mean() if not left_vals.empty else float("nan")
                st.metric(
                    f"{label} — {left}",
                    f"KES {left_price:,.0f}" if pd.notna(left_price) else "N/A",
                )

            with price_colB:
                right_vals = to_num(B[price_col]) if price_col in B.columns else pd.Series(dtype=float)
                right_price = right_vals.mean() if not right_vals.empty else float("nan")
                st.metric(
                    f"{label} — {right}",
                    f"KES {right_price:,.0f}" if pd.notna(right_price) else "N/A",
                )

    # ------------------------------------------------------------------
    # Difference analysis
    # ------------------------------------------------------------------
    st.subheader("Difference Analysis")

    diff_col1, diff_col2, diff_col3 = st.columns(3)

    with diff_col1:
        if "kpmd_registered" in processor.df.columns:
            left_kpmd = A["kpmd_registered"].mean() * 100
            right_kpmd = B["kpmd_registered"].mean() * 100
            diff = right_kpmd - left_kpmd
            st.metric("KPMD Participation Difference", f"{diff:+.1f}%")

    with diff_col2:
        if "total_sr" in processor.df.columns:
            left_sr = A["total_sr"].mean()
            right_sr = B["total_sr"].mean()
            diff = right_sr - left_sr
            st.metric("Avg Herd Size Difference", f"{diff:+.1f}")

    with diff_col3:
        if "net_profit" in processor.df.columns:
            left_profit = A["net_profit"].mean()
            right_profit = B["net_profit"].mean()
            diff = right_profit - left_profit
            st.metric("Avg Profit Difference", f"KES {diff:+,.0f}")
