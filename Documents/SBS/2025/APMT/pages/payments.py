# apmt_dashboard/pages/payments.py
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import create_comparison_bar_chart
from utils.helpers import one_hot_multiselect


def render_payments(processor):
    """Render the Payments dashboard page."""
    st.header("💸 Payment Methods")

    def _norm(s: str) -> str:
        """Normalize strings for fuzzy matching."""
        return re.sub(r"\s+", " ", str(s)).strip().lower()

    # Stems correspond to the multi-select questions in the questionnaire
    stems = [
        (
            "Sheep – KPMD",
            "E1g. How were you paid by the KPMD off-takers last  month? [Select all that apply]",
        ),
        (
            "Goats – KPMD",
            "E2g. How were you paid by the KPMD off-takers last  month? [Select all that apply]",
        ),
        (
            "Sheep – Other",
            "E3h. How were you paid by the non-KPMD off-takers last  month? [Select all that apply]",
        ),
        (
            "Goats – Other",
            "E4h. How were you paid by the non-KPMD off-takers  last  month? [Select all that apply]",
        ),
    ]

    rows = []

    # Normalized lookup for exact stem → original column name (when stored as single text field)
    cols_norm = {_norm(c): c for c in processor.df.columns}

    for label, stem in stems:
        stem_n = _norm(stem)

        # --- 1. Find expanded multi-select columns (stem/... form) ---
        subcols = []
        for c in processor.df.columns:
            c_n = _norm(c)
            if c_n.startswith(stem_n) and "/" in c:
                subcols.append(c)

        # --- 2. Classify payment modes (mobile / cash) over the expanded columns ---
        mobile_cols, cash_cols = [], []
        for c in subcols:
            suffix = _norm(c.split("/", 1)[1])
            if ("mobile" in suffix) or ("m-pesa" in suffix) or ("mpesa" in suffix):
                mobile_cols.append(c)
            if "cash" in suffix:
                cash_cols.append(c)

        # --- 3. If stored as a single multi-select text column, prepare to one-hot it ---
        single_col = cols_norm.get(stem_n) if stem_n in cols_norm else None

        mobile_series = None
        cash_series = None

        # From expanded binary columns
        if mobile_cols:
            mobile_series = (
                processor.df[mobile_cols]
                .astype(str)
                .replace({"1": 1, "0": 0})
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .max(axis=1)
            )
        if cash_cols:
            cash_series = (
                processor.df[cash_cols]
                .astype(str)
                .replace({"1": 1, "0": 0})
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .max(axis=1)
            )

        # Fallback: one-hot from a single multi-select text column
        if (mobile_series is None or cash_series is None) and (single_col is not None):
            dummies = one_hot_multiselect(processor.df[single_col])

            if mobile_series is None:
                tok = next(
                    (
                        t
                        for t in dummies.columns
                        if _norm(t).startswith("mobile") or "mpesa" in _norm(t)
                    ),
                    None,
                )
                mobile_series = dummies.get(
                    tok, pd.Series(0, index=processor.df.index)
                )

            if cash_series is None:
                tok = next(
                    (t for t in dummies.columns if _norm(t).startswith("cash")),
                    None,
                )
                cash_series = dummies.get(
                    tok, pd.Series(0, index=processor.df.index)
                )

        # Hard fallback to zeros if still missing
        if mobile_series is None:
            mobile_series = pd.Series(0, index=processor.df.index)
        if cash_series is None:
            cash_series = pd.Series(0, index=processor.df.index)

        # Build block-level frame
        tmp_cols = ["kpmd_registered"]
        if "County" in processor.df.columns:
            tmp_cols.append("County")

        tmp = processor.df[tmp_cols].copy()
        tmp["block"] = label

        tmp["mobile"] = (
            pd.to_numeric(mobile_series, errors="coerce")
            .fillna(0)
            .clip(0, 1)
            .astype(int)
        )
        tmp["cash"] = (
            pd.to_numeric(cash_series, errors="coerce")
            .fillna(0)
            .clip(0, 1)
            .astype(int)
        )
        tmp["both"] = ((tmp["mobile"] == 1) & (tmp["cash"] == 1)).astype(int)

        rows.append(tmp)

    if not rows:
        st.info("No payment method columns found.")
        return

    payment = pd.concat(rows, ignore_index=True)

    # ------------------------------------------------------------------
    # 1) Payment method mix by species/channel × KPMD
    # ------------------------------------------------------------------
    st.subheader("Payment Method Mix by Channel and KPMD Status")

    grp = payment.groupby(["block", "kpmd_registered"], dropna=False)
    summary = pd.DataFrame(
        {
            "Mobile share": grp["mobile"].mean() * 100.0,
            "Cash share": grp["cash"].mean() * 100.0,
            "Both share": grp["both"].mean() * 100.0,
        }
    ).reset_index()

    summary["KPMD Status"] = summary["kpmd_registered"].map(
        {1: "KPMD", 0: "Non-KPMD"}
    )

    long = summary.melt(
        id_vars=["block", "KPMD Status"],
        value_vars=["Cash share", "Mobile share", "Both share"],
        var_name="Method",
        value_name="Share",
    )

    if long["Share"].dropna().sum() == 0 or long.dropna(subset=["Share"]).empty:
        st.warning("No non-zero payment shares detected. Check column names in your dataset.")
        return

    fig = px.bar(
        long,
        x="block",
        y="Share",
        color="Method",
        barmode="group",
        facet_col="KPMD Status",
        title="Payment method mix by channel/species and KPMD",
    )
    fig.update_traces(text=long["Share"].round(1), textposition="outside")
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 2) Digital adoption by County × KPMD
    # ------------------------------------------------------------------
    st.subheader("Digital Adoption by County")

    if "County" not in payment.columns:
        payment["County"] = "(Unspecified)"

    # Create simple household ID if none exists
    if "hhid" not in payment.columns:
        payment["hhid"] = range(len(payment))

    # Aggregate to HH level (any mobile/cash/both within HH)
    agg = (
        payment.groupby(
            ["hhid", "County", "kpmd_registered"], dropna=False
        )[["mobile", "cash", "both"]]
        .max()
        .reset_index()
    )

    agg["payer"] = ((agg["mobile"] == 1) | (agg["cash"] == 1)).astype(int)
    agg["digital"] = ((agg["mobile"] == 1) | (agg["both"] == 1)).astype(int)

    # Controls
    min_payers = st.number_input(
        "Minimum payers per County × KPMD to include",
        min_value=0,
        value=5,
        step=1,
    )
    include_subthreshold = st.checkbox(
        "Include groups below threshold", value=False
    )

    # Summarise among payers only
    payers = agg[agg["payer"] == 1].copy()
    summary = (
        payers.groupby(["County", "kpmd_registered"], dropna=False)
        .agg(
            n=("hhid", "nunique"),
            digital_share=(
                "digital",
                lambda s: s.mean() * 100.0,
            ),
        )
        .reset_index()
    )
    summary["KPMD Status"] = summary["kpmd_registered"].map(
        {1: "KPMD", 0: "Non-KPMD"}
    )

    # Filter by threshold
    below = summary[summary["n"] < min_payers][
        ["County", "KPMD Status", "n"]
    ]
    shown = summary if include_subthreshold else summary[summary["n"] >= min_payers]

    if shown.empty:
        st.info("No groups meet the minimum payer threshold.")
        return

    shown = shown.copy()
    shown["label"] = shown.apply(
        lambda r: f"{r['digital_share']:.1f}% (n={int(r['n'])})", axis=1
    )

    fix_axis = st.checkbox("Fix y-axis to 0–100%", value=True)

    fig2 = create_comparison_bar_chart(
        shown,
        x_col="County",
        y_col="digital_share",
        color_col="KPMD Status",
        title="Digital Payment Adoption by County and KPMD Status (%)",
        barmode="group",
        text_format="{:.1f}%",
        y_title="Digital Adoption Rate (%)",
    )

    if fig2:
        if fix_axis:
            fig2.update_yaxes(range=[0, 100])
        st.plotly_chart(fig2, use_container_width=True)

    # Denominator notes
    denom_note = " | ".join(
        [
            f"{r['County']} – {r['KPMD Status']}: n={int(r['n'])}"
            for _, r in shown[["County", "KPMD Status", "n"]].iterrows()
        ]
    )
    st.caption("Denominator: payers only (household-level). " + denom_note)

    if not include_subthreshold and not below.empty:
        dropped = " | ".join(
            [
                f"{r['County']} – {r['KPMD Status']}: n={int(r['n'])}"
                for _, r in below.iterrows()
            ]
        )
        st.caption("Hidden (below threshold): " + dropped)
