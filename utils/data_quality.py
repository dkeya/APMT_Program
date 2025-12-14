# apmt_dashboard/utils/data_quality.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

# Fixed import - works with or without __init__.py files
try:
    from utils.helpers import to_num, coalesce_first, _iqr_outlier_mask
except ImportError:
    from .helpers import to_num, coalesce_first, _iqr_outlier_mask

def clean_and_validate(df: pd.DataFrame):
    """
    Returns: (clean_df, issues)
    """
    issues = []
    work = df.copy()

    # Strip whitespace from column names and string cells
    work.columns = [re.sub(r'\s+', ' ', c).strip() for c in work.columns]
    obj_cols = work.select_dtypes(include=['object']).columns.tolist()
    for c in obj_cols:
        work[c] = work[c].astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan})

    # Deduplicate exact rows
    before = len(work)
    work = work.drop_duplicates()
    dupes_removed = before - len(work)
    if dupes_removed > 0:
        issues.append(f"Removed {dupes_removed} duplicate rows.")

    # Coerce likely numeric columns by hint
    numeric_hints = [
        'price', 'cost', 'number', 'quantity', 'bales', 'weight', 'months', 'rate', 'total',
        'insured', 'premium', 'revenue', 'times', 'transport', 'distance', 'profit', 'margin'
    ]
    for c in work.columns:
        if any(h in c.lower() for h in numeric_hints):
            work[c] = to_num(work[c])

    # Harmonize dates - Fixed with dayfirst=True
    for c in ['int_date','_submission_time','start','end']:
        if c in work.columns:
            work[c] = pd.to_datetime(work[c], errors='coerce', dayfirst=True)

    # Range validations (collect warnings)
    def add_issue(mask, msg, suggestion=None):
        try:
            cnt = int(pd.Series(mask).fillna(False).sum())
        except Exception:
            cnt = 0
        if cnt > 0:
            issues.append(f"{msg}: {cnt} rows" + (f" — {suggestion}" if suggestion else ""))

    # GPS sanity
    lat_col = coalesce_first(work, ['_GPS Coordinates_latitude','GPS Latitude','Latitude'])
    lon_col = coalesce_first(work, ['_GPS Coordinates_longitude','GPS Longitude','Longitude'])
    if lat_col and lon_col:
        bad_lat = ~pd.to_numeric(work[lat_col], errors='coerce').between(-4.7, 5.0)
        bad_lon = ~pd.to_numeric(work[lon_col], errors='coerce').between(33.0, 42.5)
        add_issue(bad_lat | bad_lon, "Out-of-bounds GPS coordinates", "check data entry")

    # Negative values checks
    for c in work.columns:
        cl = c.lower()
        if any(k in cl for k in ['price','cost','revenue','premium','transport','profit','weight']):
            bad = pd.to_numeric(work[c], errors='coerce') < 0
            add_issue(bad.fillna(False), f"Negative values in '{c}'", "should be ≥ 0")

    # Unrealistic weights
    for c in work.columns:
        if 'weight' in c.lower():
            over = pd.to_numeric(work[c], errors='coerce') > 120
            add_issue(over.fillna(False), f"Unusually large weights in '{c}'", "verify units (kg)")

    # Missing key fields
    if 'County' in work.columns:
        add_issue(work['County'].isna() | (work['County'].astype(str).str.strip() == ''), "Missing County")

    for c in ['County','Gender','month']:
        if c in work.columns:
            work[c] = work[c].astype('category')

    return work, issues

def render_data_quality_section(df: pd.DataFrame, issues: list):
    """
    Combined, folded section:
    - Top: Validation issues list (warnings)
    - Visuals: missingness, duplicates, outliers by column, and quick distributions
    Always folded by default (expanded=False).
    """
    with st.expander("🧹 Data Quality Overview", expanded=False):
        # ---- Issues / Validation report
        if issues:
            for it in issues:
                st.warning(it)
        else:
            st.success("No major data validation issues detected.")

        st.markdown("---")

        # Missingness (columns on the X axis)
        try:
            miss_pct = df.isna().mean().mul(100).sort_values(ascending=False)
            miss_df_full = miss_pct.reset_index()
            miss_df_full.columns = ['Column','Missing %']

            # Plot ONLY columns with >0% missing
            miss_df = miss_df_full[miss_df_full['Missing %'] > 0].copy()

            if not miss_df.empty:
                fig = px.bar(
                    miss_df,
                    x='Column',
                    y='Missing %',
                    title='Missing Data (%) by Column',
                )
                fig.update_traces(marker_line_width=0, hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>")
                fig.update_layout(
                    xaxis={
                        'categoryorder': 'total descending',
                        'tickangle': -60,
                        'automargin': True
                    },
                    yaxis={'rangemode': 'tozero'},
                    bargap=0.15,
                    height=min(1200, max(500, 20 * (len(miss_df) > 35) + 600)),
                    margin=dict(l=60, r=30, t=60, b=260)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No missing values detected.")

            # Full table (including 0% columns) for audit
            with st.expander("Show full missingness table (includes 0%)", expanded=False):
                st.dataframe(miss_df_full.reset_index(drop=True))
        except Exception as e:
            st.info(f"Missingness scan skipped: {e}")

        # Duplicates
        try:
            dup_rows = int(df.duplicated().sum())
            st.metric("Duplicate rows", f"{dup_rows:,}")
        except Exception:
            pass

        # Outliers (numeric only)
        try:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if num_cols:
                out_counts = []
                for c in num_cols:
                    mask = _iqr_outlier_mask(df[c])
                    out_counts.append({'Column': c, 'Outliers': int(mask.sum())})
                out_df = pd.DataFrame(out_counts).sort_values('Outliers', ascending=False)

                # Plot ONLY non-zero outlier columns
                nonzero = out_df[out_df['Outliers'] > 0].copy()

                if not nonzero.empty:
                    fig2 = px.bar(
                        nonzero,
                        y='Column',
                        x='Outliers',
                        orientation='h',
                        title='Outlier Counts (numeric columns with >0 outliers)'
                    )
                    fig2.update_layout(
                        height=min(1200, max(450, 18 * len(nonzero))),
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No IQR-based outliers detected in numeric columns.")

                # Full table (includes zeros) for reference
                with st.expander("Show full outlier counts table (includes zero-outlier columns)", expanded=False):
                    st.dataframe(out_df.reset_index(drop=True))
        except Exception as e:
            st.info(f"Outlier scan skipped: {e}")

        # Quick distributions (lightweight; top few numeric cols by variance)
        try:
            if num_cols:
                var_sorted = pd.Series({c: pd.to_numeric(df[c], errors='coerce').var() for c in num_cols}).sort_values(ascending=False)
                top_plot = [c for c in var_sorted.index[:4] if c in df.columns]
                if top_plot:
                    st.caption("Quick distributions (top-variance numeric columns)")
                    for c in top_plot:
                        fig3 = px.histogram(df, x=c, nbins=30, title=f'Distribution — {c}')
                        st.plotly_chart(fig3, use_container_width=True)
        except Exception:
            pass