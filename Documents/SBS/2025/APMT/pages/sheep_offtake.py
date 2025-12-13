# apmt_dashboard/pages/sheep_offtake.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re

from components.comparison_cards import create_comparison_cards
from components.charts import create_comparison_bar_chart, create_box_plot
from utils.helpers import coalesce_first, to_num, yn
from utils.stats import lsmeans_by_group


def render_sheep_offtake(processor):
    """Render the Sheep Offtake dashboard page."""
    return render_offtake_analysis(processor, 'sheep')


def render_offtake_analysis(processor, species='sheep'):
    """
    Generic offtake analysis for sheep or goats.

    Args:
        processor: Data processor instance with .df
        species: 'sheep' or 'goats'
    """
    df = processor.df
    title_species = 'Sheep' if species.lower().startswith('sheep') else 'Goats'
    st.header(f"🚚 Offtake Analysis - {title_species}")

    # Prefixes from questionnaire
    if species.lower().startswith('sheep'):
        kpmd_prefix, non_kpmd_prefix = 'E1', 'E3'
    else:
        kpmd_prefix, non_kpmd_prefix = 'E2', 'E4'

    # Column mapping from processor (preferred)
    mapping = getattr(processor, 'offtake_col_mapping', {}) or {}
    if species.lower().startswith('sheep'):
        kpmd_sold_col = mapping.get('sheep_kpmd_sold')
        non_kpmd_sold_col = mapping.get('sheep_non_kpmd_sold')
    else:
        kpmd_sold_col = mapping.get('goat_kpmd_sold')
        non_kpmd_sold_col = mapping.get('goat_non_kpmd_sold')

    # Internal helper to construct questionnaire column texts
    def _sales_cols(_species, _kpmd_prefix, _non_kpmd_prefix):
        if _species.lower().startswith('sheep'):
            price_kpmd = f"{_kpmd_prefix}c. What was the average price per sheep last month?"
            price_non = f"{_non_kpmd_prefix}d. What was the average price per sheep last month?"
            age_kpmd = f"{_kpmd_prefix}d. What was the typical age in months of the sheep when sold to KPMD off-takers last month?"
            age_non = f"{_non_kpmd_prefix}e. What was the typical age in months of the sheep when sold to non-KPMD off-takers last month?"
            wt_kpmd = f"{_kpmd_prefix}f. What was the typical weight in kilos of sheep sold last month?"
            wt_non = f"{_non_kpmd_prefix}g. What was the typical weight in kilos of sheep sold last month?"
            breed_kpmd_stem = f"{_kpmd_prefix}i. What breeds of sheep did you sell? [Select all that apply]"
            breed_non_stem = f"{_non_kpmd_prefix}j. What breeds of sheep did you sell? [Select all that apply]"
            buyers_non_stem = f"{_non_kpmd_prefix}a. To whom did you sell sheep? [Select all that apply]"
        else:
            price_kpmd = f"{_kpmd_prefix}c. What was the average price per goat last month?"
            price_non = f"{_non_kpmd_prefix}d. What was the average price per goat last month?"
            age_kpmd = f"{_kpmd_prefix}d. What was the typical age in months of the goats when sold to KPMD off-takers last month?"
            age_non = f"{_non_kpmd_prefix}e. What was the typical age in months of the goats when sold to non-KPMD off-takers last month?"
            wt_kpmd = f"{_kpmd_prefix}f. What was the typical weight in kilos of goats sold last month?"
            wt_non = f"{_non_kpmd_prefix}g. What was the typical weight in kilos of goats sold last month?"
            breed_kpmd_stem = f"{_kpmd_prefix}i. What breeds of goats did you sell? [Select all that apply]"
            breed_non_stem = f"{_non_kpmd_prefix}j. What breeds of goats did you sell? [Select all that apply]"
            buyers_non_stem = f"{_non_kpmd_prefix}a. To whom did you sell goats? [Select all that apply]"

        return (
            price_kpmd,
            price_non,
            age_kpmd,
            age_non,
            wt_kpmd,
            wt_non,
            breed_kpmd_stem,
            breed_non_stem,
            buyers_non_stem,
        )

    (
        price_kpmd_col,
        price_non_col,
        age_kpmd_col,
        age_non_col,
        wt_kpmd_col,
        wt_non_col,
        breed_kpmd_stem,
        breed_non_stem,
        buyers_non_stem,
    ) = _sales_cols(species, kpmd_prefix, non_kpmd_prefix)

    tab1, tab2, tab3 = st.tabs(["Sales Volume", "Price Analysis", "Transaction Details"])

    # ---------- Tab 1: Sales Volume ----------
    with tab1:
        render_sales_volume_tab(processor, species, title_species, kpmd_sold_col, non_kpmd_sold_col)

    # ---------- Tab 2: Price Analysis ----------
    with tab2:
        render_price_analysis_tab(
            processor,
            species,
            title_species,
            kpmd_sold_col,
            non_kpmd_sold_col,
            price_kpmd_col,
            price_non_col,
        )

    # ---------- Tab 3: Transaction Details ----------
    with tab3:
        render_transaction_details_tab(
            processor,
            species,
            title_species,
            kpmd_sold_col,
            non_kpmd_sold_col,
            age_kpmd_col,
            age_non_col,
            wt_kpmd_col,
            wt_non_col,
            breed_kpmd_stem,
            breed_non_stem,
            buyers_non_stem,
        )


def render_sales_volume_tab(processor, species, title_species, kpmd_sold_col, non_kpmd_sold_col):
    """Render sales volume tab."""
    df = processor.df
    st.subheader("Sales Volume Analysis")

    st.write("**Households Selling to Different Channels**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**KPMD Channel Sales**")
        if kpmd_sold_col and kpmd_sold_col in df.columns:
            tmp = df.copy()
            tmp['sold_kpmd'] = tmp[kpmd_sold_col].apply(yn).astype(int)
            # 0/1 → mean → formatted as percentage
            create_comparison_cards(tmp, 'sold_kpmd', 'KPMD Channel Sales', '{:.1%}')
        else:
            st.info("No KPMD channel sales indicator found for this species.")

    with col2:
        st.markdown("**Non-KPMD Channel Sales**")
        if non_kpmd_sold_col and non_kpmd_sold_col in df.columns:
            tmp = df.copy()
            tmp['sold_non_kpmd'] = tmp[non_kpmd_sold_col].apply(yn).astype(int)
            create_comparison_cards(tmp, 'sold_non_kpmd', 'Non-KPMD Channel Sales', '{:.1%}')
        else:
            st.info("No Non-KPMD channel sales indicator found for this species.")

    # Sales channel participation visualization
    if kpmd_sold_col and non_kpmd_sold_col and 'kpmd_registered' in df.columns:
        st.subheader("Sales Channel Participation by KPMD Status")
        try:
            df_offtake = df[['kpmd_registered']].copy()

            if kpmd_sold_col in df.columns:
                df_offtake['KPMD_Channel'] = df[kpmd_sold_col].apply(yn).astype(int)
            if non_kpmd_sold_col in df.columns:
                df_offtake['Non_KPMD_Channel'] = df[non_kpmd_sold_col].apply(yn).astype(int)

            participation_data = []
            for s in [0, 1]:
                sub = df_offtake[df_offtake['kpmd_registered'] == s]
                kpmd_status = 'KPMD' if s == 1 else 'Non-KPMD'

                if 'KPMD_Channel' in sub.columns and len(sub) > 0:
                    kpmd_rate = sub['KPMD_Channel'].mean() * 100
                    participation_data.append({
                        'KPMD_Status': kpmd_status,
                        'Channel': 'KPMD Channel',
                        'Participation_Rate': kpmd_rate
                    })

                if 'Non_KPMD_Channel' in sub.columns and len(sub) > 0:
                    non_kpmd_rate = sub['Non_KPMD_Channel'].mean() * 100
                    participation_data.append({
                        'KPMD_Status': kpmd_status,
                        'Channel': 'Non-KPMD Channel',
                        'Participation_Rate': non_kpmd_rate
                    })

            if participation_data:
                participation_df = pd.DataFrame(participation_data)
                fig = create_comparison_bar_chart(
                    participation_df,
                    x_col='KPMD_Status',
                    y_col='Participation_Rate',
                    color_col='Channel',
                    title=f'{title_species} Sales Channel Participation by KPMD Status',
                    barmode='group',
                    text_format='{:.1f}%',
                    y_title='Participation Rate (%)'
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Sales channel visualization not available: {str(e)}")
    else:
        st.info("Not enough information to compare sales channel participation by KPMD status.")


def render_price_analysis_tab(processor, species, title_species,
                              kpmd_sold_col, non_kpmd_sold_col,
                              price_kpmd_col, price_non_col):
    """Render price analysis tab."""
    df = processor.df
    st.subheader("Price Analysis")

    if 'kpmd_registered' not in df.columns:
        st.info("KPMD registration indicator not available; cannot stratify prices.")
        return

    # Build seller flags
    sold_kpmd = None
    sold_non = None
    if kpmd_sold_col and kpmd_sold_col in df.columns:
        sold_kpmd = df[kpmd_sold_col].apply(yn).astype(int) == 1
    if non_kpmd_sold_col and non_kpmd_sold_col in df.columns:
        sold_non = df[non_kpmd_sold_col].apply(yn).astype(int) == 1

    price_data = []

    # KPMD prices
    if price_kpmd_col in df.columns:
        mask = df[price_kpmd_col].notna()
        if sold_kpmd is not None:
            mask &= sold_kpmd
        df_kpmd = df.loc[mask].copy()

        if not df_kpmd.empty:
            for s in [0, 1]:
                sub = df_kpmd[df_kpmd['kpmd_registered'] == s]
                vals = to_num(sub[price_kpmd_col]).dropna()
                price_data.extend([
                    {
                        'Channel': 'KPMD',
                        'Price': v,
                        'KPMD_Status': 'KPMD Registered' if s == 1 else 'Non-KPMD Registered'
                    }
                    for v in vals
                ])

    # Non-KPMD prices
    if price_non_col in df.columns:
        mask = df[price_non_col].notna()
        if sold_non is not None:
            mask &= sold_non
        df_non = df.loc[mask].copy()

        if not df_non.empty:
            for s in [0, 1]:
                sub = df_non[df_non['kpmd_registered'] == s]
                vals = to_num(sub[price_non_col]).dropna()
                price_data.extend([
                    {
                        'Channel': 'Non-KPMD',
                        'Price': v,
                        'KPMD_Status': 'KPMD Registered' if s == 1 else 'Non-KPMD Registered'
                    }
                    for v in vals
                ])

    if price_data:
        dfp = pd.DataFrame(price_data)
        fig = create_box_plot(
            dfp,
            x_col='Channel',
            y_col='Price',
            color_col='KPMD_Status',
            title=f'{title_species} Price Distribution by Channel and KPMD Registration'
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Price data for {title_species} not available.")


def render_transaction_details_tab(processor, species, title_species,
                                   kpmd_sold_col, non_kpmd_sold_col,
                                   age_kpmd_col, age_non_col,
                                   wt_kpmd_col, wt_non_col,
                                   breed_kpmd_stem, breed_non_stem, buyers_non_stem):
    """Render transaction details tab."""
    df = processor.df
    st.subheader("Transaction Details (Age at Sale)")

    if 'kpmd_registered' not in df.columns:
        st.info("KPMD registration indicator not available; cannot stratify transaction details.")
        return

    # ---- Age at sale analysis ----
    age_data = []

    if age_kpmd_col in df.columns:
        for s in [0, 1]:
            sub = df[df['kpmd_registered'] == s]
            vals = to_num(sub[age_kpmd_col]).dropna()
            # Filter to reasonable age range (1–120 months)
            vals = vals[vals.between(1, 120)]
            age_data.extend([
                {
                    'Channel': 'KPMD',
                    'Age': v,
                    'KPMD_Status': 'KPMD Registered' if s == 1 else 'Non-KPMD Registered'
                }
                for v in vals
            ])

    if age_non_col in df.columns:
        for s in [0, 1]:
            sub = df[df['kpmd_registered'] == s]
            vals = to_num(sub[age_non_col]).dropna()
            vals = vals[vals.between(1, 120)]
            age_data.extend([
                {
                    'Channel': 'Non-KPMD',
                    'Age': v,
                    'KPMD_Status': 'KPMD Registered' if s == 1 else 'Non-KPMD Registered'
                }
                for v in vals
            ])

    if age_data:
        dfp = pd.DataFrame(age_data)
        fig = create_box_plot(
            dfp,
            x_col='Channel',
            y_col='Age',
            color_col='KPMD_Status',
            title=f'{title_species} Age at Sale by Channel and KPMD Registration'
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Age at sale data for {title_species} not available.")

    # ---- Weights analysis ----
    st.subheader("Weights Analysis")
    render_weights_analysis(
        processor,
        species,
        title_species,
        kpmd_sold_col,
        non_kpmd_sold_col,
        wt_kpmd_col,
        wt_non_col
    )

    # ---- Breeds analysis ----
    st.subheader("Breeds Sold")
    render_breeds_analysis(processor, breed_kpmd_stem, breed_non_stem, title_species)

    # ---- Buyer types analysis ----
    st.subheader("Non-KPMD Buyer Types")
    render_buyer_analysis(processor, buyers_non_stem)


def render_weights_analysis(processor, species, title_species,
                            kpmd_sold_col, non_kpmd_sold_col,
                            wt_kpmd_col, wt_non_col):
    """Render weights analysis."""
    df = processor.df

    if 'kpmd_registered' not in df.columns:
        st.info("KPMD registration indicator not available; cannot stratify weights.")
        return

    # Build seller flags
    sold_kpmd = None
    sold_non = None
    if kpmd_sold_col and kpmd_sold_col in df.columns:
        sold_kpmd = df[kpmd_sold_col].apply(yn).astype(int) == 1
    if non_kpmd_sold_col and non_kpmd_sold_col in df.columns:
        sold_non = df[non_kpmd_sold_col].apply(yn).astype(int) == 1

    wt_rows = []

    # KPMD channel weights
    if wt_kpmd_col in df.columns:
        for s in [0, 1]:
            mask = df['kpmd_registered'] == s
            if sold_kpmd is not None:
                mask &= sold_kpmd
            sub = df.loc[mask]
            if wt_kpmd_col in sub.columns:
                w = to_num(sub[wt_kpmd_col]).dropna()
                # Reasonable liveweight range (kg)
                w = w[w.between(10, 100)]
                wt_rows.extend([
                    {
                        'Channel': 'KPMD',
                        'Weight (kg)': v,
                        'KPMD_Status': 'KPMD' if s == 1 else 'Non-KPMD'
                    }
                    for v in w
                ])

    # Non-KPMD channel weights
    if wt_non_col in df.columns:
        for s in [0, 1]:
            mask = df['kpmd_registered'] == s
            if sold_non is not None:
                mask &= sold_non
            sub = df.loc[mask]
            if wt_non_col in sub.columns:
                w = to_num(sub[wt_non_col]).dropna()
                w = w[w.between(10, 100)]
                wt_rows.extend([
                    {
                        'Channel': 'Non-KPMD',
                        'Weight (kg)': v,
                        'KPMD_Status': 'KPMD' if s == 1 else 'Non-KPMD'
                    }
                    for v in w
                ])

    if wt_rows:
        d_w = pd.DataFrame(wt_rows)
        fig = create_box_plot(
            d_w,
            x_col='Channel',
            y_col='Weight (kg)',
            color_col='KPMD_Status',
            title=f'{title_species} Typical Weights by Channel and KPMD Status'
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Weight data for {title_species} not available.")


def render_breeds_analysis(processor, breed_kpmd_stem, breed_non_stem, title_species):
    """Render breeds analysis."""
    df = processor.df

    bkp_cols = [c for c in df.columns if c.startswith(breed_kpmd_stem + "/")]
    bno_cols = [c for c in df.columns if c.startswith(breed_non_stem + "/")]

    breed_data = []

    # KPMD breeds
    for c in bkp_cols:
        name = c.split('/')[-1]
        vals = pd.to_numeric(
            df[c].astype(str).replace({'1': 1, '0': 0}),
            errors='coerce'
        ).fillna(0)
        rate = vals.mean() * 100
        breed_data.append({'Breed': name, 'Rate': rate, 'Channel': 'KPMD'})

    # Non-KPMD breeds
    for c in bno_cols:
        name = c.split('/')[-1]
        vals = pd.to_numeric(
            df[c].astype(str).replace({'1': 1, '0': 0}),
            errors='coerce'
        ).fillna(0)
        rate = vals.mean() * 100
        breed_data.append({'Breed': name, 'Rate': rate, 'Channel': 'Non-KPMD'})

    if breed_data:
        breed_df = pd.DataFrame(breed_data)
        fig = create_comparison_bar_chart(
            breed_df,
            x_col='Breed',
            y_col='Rate',
            color_col='Channel',
            title=f'{title_species} Breeds Sold by Channel (%)',
            barmode='group',
            text_format='{:.1f}%',
            y_title='Percentage'
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Breed selection columns not available.")


def render_buyer_analysis(processor, buyers_non_stem):
    """Render buyer analysis."""
    df = processor.df

    buyers_cols = [c for c in df.columns if c.startswith(buyers_non_stem + "/")]

    if buyers_cols:
        buyer_data = []
        for c in buyers_cols:
            name = c.split('/')[-1]
            vals = pd.to_numeric(
                df[c].astype(str).replace({'1': 1, '0': 0}),
                errors='coerce'
            ).fillna(0)
            rate = vals.mean() * 100
            buyer_data.append({'Buyer': name, 'Rate': rate})

        buyer_df = pd.DataFrame(buyer_data)
        fig = create_comparison_bar_chart(
            buyer_df,
            x_col='Buyer',
            y_col='Rate',
            title='Non-KPMD Buyer Mix (%)',
            text_format='{:.1f}%',
            y_title='Percentage'
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Buyer mix columns not available.")
