import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pydeck as pdk
from pathlib import Path
import json
import re
import io
import os
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

# -------------------------------------------------
# Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="APMT Project Insights",
    page_icon="🐑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .kpmd-card { background-color: #e8f4fd; border-left: 4px solid #1f77b4; }
    .non-kpmd-card { background-color: #fde8e8; border-left: 4px solid #ff6b6b; }
    .warning-card { background-color: #fff3cd; padding: 1rem; border-radius: 10px; border-left: 4px solid #ffc107; }
    .profit-positive { color: #28a745; font-weight: bold; }
    .profit-negative { color: #dc3545; font-weight: bold; }
    .lsm-note { font-size: 0.85rem; color: #555; margin-top: 0.25rem; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
_YES = {'yes','y','true','1','t','aye','yeah'}
_NO  = {'no','n','false','0','f','nah'}

def yn(x):
    if pd.isna(x): return 0
    if isinstance(x, (int, float, np.integer, np.floating)): return 1 if float(x) == 1 else 0
    if isinstance(x, bool): return 1 if x else 0
    s = str(x).strip().lower()
    if s in _YES: return 1
    if s in _NO: return 0
    if s.startswith('yes'): return 1
    if s.startswith('no'): return 0
    return 0

def to_num(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '').str.strip(), errors='coerce')

def one_hot_multiselect(series: pd.Series) -> pd.DataFrame:
    if series.dropna().empty: return pd.DataFrame(index=series.index)
    tokens_list, pattern = [], re.compile(r'\s*\|\s*|\s*;\s*|\s*,\s*|\s*/\s*|\s{2,}')
    for val in series.fillna(''):
        if not isinstance(val, str):
            tokens_list.append([]); continue
        tokens = [t.strip() for t in pattern.split(val) if t.strip() != '']
        tokens_list.append(tokens)
    uniques = sorted({tok for toks in tokens_list for tok in toks})
    if not uniques: return pd.DataFrame(index=series.index)
    data = {tok: [1 if tok in toks else 0 for toks in tokens_list] for tok in uniques}
    return pd.DataFrame(data, index=series.index).astype(int)

def coalesce_first(df, candidates):
    if not isinstance(df, pd.DataFrame): return None
    for c in candidates:
        if c in df.columns: return c
    return None

# ---------- LSMeans utilities ----------
def _design_matrix(df, y_col, group_col=None, controls=None, dropna=True):
    if controls is None: controls = []
    work = df.copy()

    if y_col not in work.columns:
        return None, None, {}
    y = pd.to_numeric(work[y_col], errors='coerce')

    cols_to_use = []
    if group_col and group_col in work.columns:
        cols_to_use.append(group_col)
    for c in controls:
        if c in work.columns and c != group_col:
            cols_to_use.append(c)

    X_parts = []
    meta = {'intercept': True, 'group': None, 'group_levels': [], 'control_cols': []}

    X_parts.append(pd.Series(1.0, index=work.index, name='Intercept'))

    if group_col and group_col in work.columns:
        g = work[group_col]
        if pd.api.types.is_numeric_dtype(g) and set(pd.unique(g.dropna())) <= {0,1}:
            g1 = pd.to_numeric(g, errors='coerce').fillna(0.0)
            colname = f'{group_col}_1'
            X_parts.append(pd.Series(g1, index=work.index, name=colname))
            meta['group'] = group_col
            meta['group_levels'] = [0,1]
            meta['group_dummy_cols'] = {1: colname}
        else:
            g = g.astype('category')
            dummies = pd.get_dummies(g, prefix=group_col, drop_first=True)
            X_parts.append(dummies)
            meta['group'] = group_col
            levels = list(g.cat.categories)
            meta['group_levels'] = levels
            meta['group_dummy_cols'] = {}
            for lvl in levels[1:]:
                meta['group_dummy_cols'][lvl] = f"{group_col}_{lvl}"

    for c in controls:
        if c == group_col or c not in work.columns: continue
        s = work[c]
        if pd.api.types.is_numeric_dtype(s):
            X_parts.append(pd.Series(pd.to_numeric(s, errors='coerce'), index=work.index, name=c))
            meta['control_cols'].append(c)
        else:
            s = s.astype('category')
            dummies = pd.get_dummies(s, prefix=c, drop_first=True)
            if dummies.shape[1] > 0:
                X_parts.append(dummies)
                meta['control_cols'].extend(list(dummies.columns))

    X = pd.concat(X_parts, axis=1)
    data = pd.concat([y.rename('y'), X], axis=1)
    if dropna:
        data = data.dropna(axis=0, how='any')
    if data.shape[0] < X.shape[1] + 1:
        return None, None, {}
    y_clean = data['y'].values.astype(float)
    X_clean = data.drop(columns=['y']).values.astype(float)
    meta['X_cols'] = list(data.drop(columns=['y']).columns)
    meta['X_means'] = data.drop(columns=['y']).mean(axis=0).to_dict()
    return y_clean, X_clean, meta

def _ols_beta(y, X):
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta
    except Exception:
        return None

def lsmeans_by_group(df, y_col, group_col, controls=None):
    if controls is None: controls = []
    if y_col not in df.columns or group_col not in df.columns:
        try:
            return df.groupby(group_col)[y_col].mean().to_dict()
        except Exception:
            return None
    y, X, meta = _design_matrix(df, y_col, group_col, controls)
    if y is None or X is None:
        try:
            return df.groupby(group_col)[y_col].mean().to_dict()
        except Exception:
            return None
    beta = _ols_beta(y, X)
    if beta is None:
        try:
            return df.groupby(group_col)[y_col].mean().to_dict()
        except Exception:
            return None

    X_cols = meta['X_cols']
    X_means = meta['X_means']
    g = meta.get('group')
    g_levels = meta.get('group_levels', [])
    g_dummy_cols = meta.get('group_dummy_cols', {})

    results = {}
    for lvl in g_levels:
        xbar = np.array([X_means.get(c, 0.0) for c in X_cols], dtype=float)
        if g is not None:
            if set(g_levels) <= {0,1}:
                colname = g_dummy_cols.get(1, f"{g}_1")
                if colname in X_cols:
                    idx = X_cols.index(colname)
                    xbar[idx] = 1.0 if lvl == 1 else 0.0
            else:
                for col in g_dummy_cols.values():
                    if col in X_cols:
                        xbar[X_cols.index(col)] = 0.0
                if lvl in g_dummy_cols:
                    colname = g_dummy_cols[lvl]
                    if colname in X_cols:
                        xbar[X_cols.index(colname)] = 1.0
        results[lvl] = float(np.dot(xbar, beta))
    return results

def fmt_lsmean_note(lsm):
    try:
        return f'<div class="lsm-note">LSMean (adjusted): {lsm}</div>'
    except Exception:
        return ""

# -------------------------------------------------
# Data Loading (Auto)
# -------------------------------------------------
def _resolve_data_path() -> str:
    return str((Path(__file__).resolve().parent / "APMT_Longitudinal_Survey.csv"))

DATA_PATH = _resolve_data_path()

@st.cache_data(ttl=900, show_spinner=False)
def load_apmt_csv(path: str) -> pd.DataFrame:
    encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1', 'ISO-8859-1', 'windows-1252']
    for enc in encodings:
        try: return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError: continue
        except FileNotFoundError: raise
        except Exception: continue
    for enc in encodings:
        try: return pd.read_csv(path, encoding=enc, sep=None, engine='python')
        except UnicodeDecodeError: continue
        except FileNotFoundError: raise
        except Exception: continue
    try: return pd.read_csv(path, encoding='utf-8', errors='replace')
    except FileNotFoundError: raise
    except Exception: return pd.read_csv(path, encoding='latin-1', errors='replace')

# -------------------------------------------------
# Geo assets: ensure county/sub-county files exist (auto-download if missing)
# -------------------------------------------------
def ensure_geo_assets():
    import os, requests
    os.makedirs("geo", exist_ok=True)
    assets = {
        "geo/kenya_counties.geojson":    "https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/gbOpen/KEN/ADM1/geoBoundaries-KEN-ADM1.geojson",
        "geo/kenya_subcounties.geojson": "https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/gbOpen/KEN/ADM2/geoBoundaries-KEN-ADM2.geojson",
    }

    missing = [p for p in assets if not (os.path.exists(p) and os.path.getsize(p) > 0)]
    if not missing:
        return True  # already present

    for path in missing:
        url = assets[path]
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            st.warning(f"Could not fetch {os.path.basename(path)}: {e}")
            return False
    return True

# -------------------------------------------------
# Data Cleaning, Validation & Quality (Merged, folded)
# -------------------------------------------------
def clean_and_validate(df: pd.DataFrame):
    """
    Returns: (clean_df, issues)
    NOTE: we do not render UI here anymore; UI is in render_data_quality_section(...).
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

    # Harmonize dates
    for c in ['int_date','_submission_time','start','end']:
        if c in work.columns:
            work[c] = pd.to_datetime(work[c], errors='coerce', infer_datetime_format=True)

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

def _iqr_outlier_mask(s: pd.Series):
    s = pd.to_numeric(s, errors='coerce')
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=s.index)
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    return (s < lower) | (s > upper)

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
                    height=min(1200, max(500, 20 * (len(miss_df) > 35) + 600)),  # a bit taller if many cols
                    margin=dict(l=60, r=30, t=60, b=260)  # big bottom margin for rotated labels
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
                # Build full counts (keep this complete for the table)
                out_counts = []
                for c in num_cols:
                    mask = _iqr_outlier_mask(df[c])
                    out_counts.append({'Column': c, 'Outliers': int(mask.sum())})
                out_df = pd.DataFrame(out_counts).sort_values('Outliers', ascending=False)

                # Plot ONLY non-zero outlier columns
                nonzero = out_df[out_df['Outliers'] > 0].copy()

                if not nonzero.empty:
                    # horizontal bars scale better with long labels
                    fig2 = px.bar(
                        nonzero,
                        y='Column',
                        x='Outliers',
                        orientation='h',
                        title='Outlier Counts (numeric columns with >0 outliers)'
                    )
                    fig2.update_layout(
                        height=min(1200, max(450, 18 * len(nonzero))),
                        yaxis={'categoryorder': 'total ascending'}  # largest at top
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

# -------------------------------------------------
# Data Processor
# -------------------------------------------------
class APMTDataProcessor:
    def __init__(self, df):
        # run cleaning/validation first (now returns (df, issues))
        clean_df, issues = clean_and_validate(df)
        self.df = clean_df
        self.dq_issues = issues
        self._basic_cleanups()
        self.column_mapping = self._build_column_mapping()
        self.enhanced_standardize_data()

    def _basic_cleanups(self):
        for col in self.df.columns:
            if any(k in col.lower() for k in ['phone', 'telephone', 'household id', '_id', '_uuid']):
                self.df[col] = self.df[col].astype(str)
        self.df.columns = [re.sub(r'\s+', ' ', c).strip() for c in self.df.columns]

    def _build_column_mapping(self):
        mapping = {}
        mapping['county'] = coalesce_first(self.df, ['County','county','COUNTY'])
        mapping['gender'] = coalesce_first(self.df, ['Gender','gender','GENDER','Select respondent name'])
        mapping['kpmd_registration'] = coalesce_first(self.df, [
            'A8. Are you registered to KPMD programs?','KPMD registration','Registered to KPMD'
        ])
        mapping['household_type'] = coalesce_first(self.df, [
            'Selection of the household','Household type','Treatment/Control'
        ])
        mapping['gps_lat'] = coalesce_first(self.df, ['_GPS Coordinates_latitude','GPS Latitude','Latitude'])
        mapping['gps_lon'] = coalesce_first(self.df, ['_GPS Coordinates_longitude','GPS Longitude','Longitude'])
        mapping['vaccination'] = coalesce_first(self.df, [
            'D1. Did you vaccinate your small ruminants in the last month?',
            'D1. Did you vaccinate your small ruminants livestock in the last month?',
            'D1. Did you vaccinate your small ruminants livestock last month?'
        ])
        mapping['vaccination_diseases'] = self._find_columns_pattern(r'D1c\..*vaccinate')
        mapping['treatment_diseases'] = self._find_columns_pattern(r'D3c\..*(treat|disease)')
        mapping['fodder_purchase'] = coalesce_first(self.df, ['B5a. Did you purchase fodder in the last 1 month?'])
        mapping['feed_sources'] = self._find_columns_pattern(r'B5b\..*buy feeds')
        mapping['sheep_kpmd_sales'] = self._find_columns_pattern(r'^E1\..*sheep.*KPMD|^E1\.$')
        mapping['goat_kpmd_sales']  = self._find_columns_pattern(r'^E2\..*goat.*KPMD|^E2\.$')
        mapping['sheep_non_kpmd_sales'] = self._find_columns_pattern(r'^E3\..*sheep|^E3\.$')
        mapping['goat_non_kpmd_sales']  = self._find_columns_pattern(r'^E4\..*goat|^E4\.$')
        mapping['decision_making'] = coalesce_first(self.df, [
            'G1.Who in the household makes the decision for livestock sale?  [Select all that apply]',
            'G1.Who in the household makes the decision for livestock sale? [Select all that apply]'
        ])
        mapping['income_control'] = coalesce_first(self.df, [
            'G2. Who in the household uses the income from the livestock sale? [Select all that apply]'
        ])
        mapping['adaptation_measures'] = coalesce_first(self.df, [
            'J1. Have you made any adaptation measures last month due to drought shocks?',
            'J1. Have you made any adaptation measures last month due to drought  shocks?'
        ])
        mapping['adaptation_strategies'] = self._find_columns_pattern(r'J2\..*adaptations?')
        mapping['barriers'] = self._find_columns_pattern(r'J3\..*Why not')
        return mapping

    def _find_columns_pattern(self, pattern):
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return []
        return [c for c in self.df.columns if rx.search(c)]

    def enhanced_standardize_data(self):
        try:
            def _coerce_date(s): return pd.to_datetime(s, errors='coerce', dayfirst=True, infer_datetime_format=True)
            date_candidates = [c for c in ['int_date','_submission_time','start','end'] if c in self.df.columns]
            date_parsed = False
            for c in date_candidates:
                parsed = _coerce_date(self.df[c])
                if parsed.notna().any():
                    self.df['int_date_std'] = parsed
                    self.df.loc[parsed.notna(), 'month'] = parsed.dt.to_period('M').astype(str)
                    self.df.loc[parsed.notna(), 'year'] = parsed.dt.year
                    date_parsed = True
                    break
            if not date_parsed:
                self.df['month'] = [f"2024-{i:02d}" for i in range(1, min(len(self.df)+1, 13))]

            kpmd_col = self.column_mapping['kpmd_registration']
            self.df['kpmd_registered'] = self.df[kpmd_col].apply(yn).astype(int) if kpmd_col else 0

            arm_col = self.column_mapping['household_type']
            self.df['is_treatment'] = (
                self.df[arm_col].astype(str).str.contains('Treatment', case=False, na=False).astype(int)
            ) if arm_col else 0

            for pat in [r'^C1\.', r'^C2\.', r'^D1\..*vaccinate', r'^D3\..*treat', r'^D4\..*deworm', r'^B5a\.', r'^B6a\.', r'^J1\.']:
                for col in self._find_columns_pattern(pat):
                    self.df[col] = self.df[col].apply(yn).astype(int)

            # Keep price fields numeric but DO NOT zero-fill them.
            price_patterns = [
                r'^E1c\..*price.*sheep',   # Sheep KPMD price
                r'^E2c\..*price.*goat',    # Goat  KPMD price
                r'^E3d\..*price.*sheep',   # Sheep Non-KPMD price
                r'^E4d\..*price.*goat',    # Goat  Non-KPMD price
            ]

            # Other numeric fields can still be coerced and zero-filled.
            other_numeric_patterns = [
                r'B3b.*cost.*herding', r'B4b\..*cost', r'B5c\..*price.*bale', r'B5d\..*Number.*bales.*purchased',
                r'B6b\..*Quantity.*harvested', r'B6d\..*price.*sell', r'B6e\..*Number.*bales.*sold',
                r'D1a\..*vaccinated', r'D1b\..*cost.*vaccination', r'D3a\..*sick.*treated', r'D3b\..*cost.*treatment',
                r'D4a\..*cost.*deworming',
                r'^E[1-4][bh]\..*(How many|times)',  # quantities
                r'^E[1-4]f\..*weight|^E[1-4]g\..*weight|^E[1-4]h\..*transport',
                r'^F1\.'
            ]

            # Apply conversions
            for pat in price_patterns:
                for col in self._find_columns_pattern(pat):
                    self.df[col] = to_num(self.df[col])  # keep NaN if not provided

            for pat in other_numeric_patterns:
                for col in self._find_columns_pattern(pat):
                    self.df[col] = to_num(self.df[col]).fillna(0)

            self.enhanced_disease_mapping()
            self.enhanced_feed_calculation()
            self.enhanced_offtake_mapping()
            self.enhanced_gender_mapping()
            self.calculate_climate_resilience()
            self.calculate_food_security()  # rCSI + insurance/worry
        except Exception as e:
            st.warning(f"Some data standardization issues occurred: {str(e)}")
            if 'month' not in self.df.columns: self.df['month'] = [f"2024-{i:02d}" for i in range(1, min(len(self.df)+1, 13))]
            if 'kpmd_registered' not in self.df.columns: self.df['kpmd_registered'] = 0
            if 'is_treatment' not in self.df.columns: self.df['is_treatment'] = 0

    def enhanced_disease_mapping(self):
        self.vacc_disease_cols = [c for c in self.df.columns if c.startswith('D1c. ') and '/' in c]
        self.treat_disease_cols = [c for c in self.df.columns if c.startswith('D3c. ') and '/' in c]
        for col in self.vacc_disease_cols + self.treat_disease_cols:
            self.df[col] = pd.to_numeric(self.df[col].astype(str).replace({'1':1,'0':0}), errors='coerce').fillna(0).astype(int)

    def enhanced_feed_calculation(self):
        price_col = coalesce_first(self.df, ['B5c. What was the price per 15 kg bale in the last 1 month?'])
        qty_col = coalesce_first(self.df, ['B5d. Number of 15 kg bales purchased in the last 1 month?'])
        self.df['Feed_Expenditure'] = (
            to_num(self.df[price_col]).fillna(0) * to_num(self.df[qty_col]).fillna(0)
        ) if price_col and qty_col else 0

    def enhanced_offtake_mapping(self):
        self.offtake_col_mapping = {}
        sheep_kpmd = coalesce_first(self.df, ['E1. Did you sell sheep to KPMD off-takers last  month?']) or coalesce_first(self.df, [c for c in self._find_columns_pattern(r'^E1\..*sell sheep')])
        if sheep_kpmd: self.offtake_col_mapping['sheep_kpmd_sold'] = sheep_kpmd
        sheep_non = coalesce_first(self.df, ['E3. Did you sell sheep to non-KPMD off-takers last  month?']) or coalesce_first(self.df, [c for c in self._find_columns_pattern(r'^E3\..*sell sheep')])
        if sheep_non: self.offtake_col_mapping['sheep_non_kpmd_sold'] = sheep_non
        goat_kpmd = coalesce_first(self.df, ['E2. Did you sell goats to KPMD off-takers last  month?']) or coalesce_first(self.df, [c for c in self._find_columns_pattern(r'^E2\..*sell goats')])
        if goat_kpmd: self.offtake_col_mapping['goat_kpmd_sold'] = goat_kpmd
        goat_non = coalesce_first(self.df, ['E4. Did you sell goats to non-KPMD off-takers last  month?']) or coalesce_first(self.df, [c for c in self._find_columns_pattern(r'^E4\..*sell goats')])
        if goat_non: self.offtake_col_mapping['goat_non_kpmd_sold'] = goat_non

    def enhanced_gender_mapping(self):
        self.gender_columns = {
            'decision_making': coalesce_first(self.df, [
                'G1.Who in the household makes the decision for livestock sale?  [Select all that apply]',
                'G1.Who in the household makes the decision for livestock sale? [Select all that apply]'
            ]) or '',
            'income_control': coalesce_first(self.df, [
                'G2. Who in the household uses the income from the livestock sale? [Select all that apply]'
            ]) or '',
            'household_head': coalesce_first(self.df, ['A17. Are you the head of this household?']) or ''
        }

    def calculate_climate_resilience(self):
        a_col = self.column_mapping.get('adaptation_measures')
        if a_col and a_col in self.df.columns: self.df['adaptation_score'] = self.df[a_col].apply(yn)
        if 'kpmd_registered' in self.df.columns: self.df['kpmd_resilience_bonus'] = self.df['kpmd_registered'] * 0.5
        components = [c for c in ['adaptation_score','kpmd_resilience_bonus'] if c in self.df.columns]
        if components: self.df['resilience_score'] = self.df[components].sum(axis=1)

    def calculate_food_security(self):
        ins_num = coalesce_first(self.df, ['Number of your small ruminants livestock  insured?'])
        ins_cost = coalesce_first(self.df, ['Cost of insurance premiums in Ksh?'])
        if ins_num: self.df['insured_sr'] = to_num(self.df[ins_num]).fillna(0)
        if ins_cost: self.df['insurance_premium'] = to_num(self.df[ins_cost]).fillna(0)

        worry = coalesce_first(self.df, ['I1. In the past 30 days, did you worry that your household would not have enough food?'])
        if worry: self.df['food_worry'] = self.df[worry].apply(yn).astype(int)

        freq_map_30_raw = {
            'never (0 days)': 0,
            'rarely (1–3 days)': 2, 'rarely (1-3 days)': 2, 'rarely (1 — 3 days)': 2,
            'sometimes (4–10 days)': 7, 'sometimes (4-10 days)': 7,
            'often (11–20 days)': 15, 'often (11-20 days)': 15,
            'very often (21–30 days)': 25, 'very often (21-30 days)': 25,
        }
        def _canon(s: str) -> str:
            return re.sub(r'\s+', ' ', str(s).strip().lower().replace('–', '-').replace('—', '-'))
        freq_map_30 = {_canon(k): v for k, v in freq_map_30_raw.items()}

        def map_freq(colname):
            if not colname or colname not in self.df.columns: return pd.Series(0, index=self.df.index)
            s = self.df[colname].astype(str).map(_canon)
            return s.map(freq_map_30).fillna(0)

        c12 = coalesce_first(self.df, ['12. Did your household rely on less preferred or less expensive foods?'])
        c13 = coalesce_first(self.df, ['13. Did your household borrow food, or rely on help from friends/relatives?'])
        c14 = coalesce_first(self.df, ['14. Did your household limit portion sizes at mealtimes?'])
        c15 = coalesce_first(self.df, ['15. Did your household reduce the number of meals eaten in a day?'])
        c16 = coalesce_first(self.df, ['I6. Did adults in your household restrict their consumption so that small children could eat?'])

        d12 = map_freq(c12)
        d13 = map_freq(c13)
        d14 = map_freq(c14)
        d15 = map_freq(c15)
        d16 = map_freq(c16)

        w12, w13, w14, w15, w16 = 1, 2, 1, 1, 3
        self.df['rcsi_30'] = (w12*d12 + w13*d13 + w14*d14 + w15*d15 + w16*d16)

    def calculate_herd_metrics(self):
        try:
            for col in ['total_sheep','total_goats','total_sr','pct_female','pct_male',
                        'total_births','total_mortality','total_losses',
                        'birth_rate_per_100','mortality_rate_per_100','loss_rate_per_100']:
                if col not in self.df.columns: self.df[col] = 0.0

            sheep_cols = [c for c in [
                'C3. Number of Rams currently owned (total: at home + away + relatives/friends)',
                'C3. Number of Ewes currently owned (total: at home + away + relatives/friends)'
            ] if c in self.df.columns]
            goat_cols = [c for c in [
                'C3. Number of Bucks currently owned (total: at home + away + relatives/friends)',
                'C3. Number of Does currently owned (total: at home + away + relatives/friends)'
            ] if c in self.df.columns]
            for col in sheep_cols + goat_cols:
                self.df[col] = to_num(self.df[col]).fillna(0)

            if sheep_cols: self.df['total_sheep'] = self.df[sheep_cols].sum(axis=1)
            if goat_cols:  self.df['total_goats'] = self.df[goat_cols].sum(axis=1)
            self.df['total_sr'] = self.df['total_sheep'] + self.df['total_goats']

            female_sheep_col = 'C3. Number of Ewes currently owned (total: at home + away + relatives/friends)'
            female_goat_col  = 'C3. Number of Does currently owned (total: at home + away + relatives/friends)'
            male_sheep_col   = 'C3. Number of Rams currently owned (total: at home + away + relatives/friends)'
            male_goat_col    = 'C3. Number of Bucks currently owned (total: at home + away + relatives/friends)'
            female_sheep = to_num(self.df[female_sheep_col]).fillna(0) if female_sheep_col in self.df.columns else 0
            female_goats = to_num(self.df[female_goat_col]).fillna(0) if female_goat_col in self.df.columns else 0
            male_sheep   = to_num(self.df[male_sheep_col]).fillna(0) if male_sheep_col in self.df.columns else 0
            male_goats   = to_num(self.df[male_goat_col]).fillna(0) if male_goat_col in self.df.columns else 0

            total_female = female_sheep + female_goats
            total_male   = male_sheep + male_goats

            valid = (self.df['total_sr'] > 0)
            self.df.loc[valid, 'pct_female'] = (total_female[valid] / self.df.loc[valid, 'total_sr'] * 100).clip(0,100)
            self.df.loc[valid, 'pct_male']   = (total_male[valid] / self.df.loc[valid, 'total_sr'] * 100).clip(0,100)

            def existing(cols): return [c for c in cols if c in self.df.columns]
            birth_cols = existing([
                'C4. Number of Rams born in the last 1 month','C4. Number of Ewes born in the last 1 month',
                'C4. Number of Bucks born in the last 1 month','C4. Number of Does born in the last 1 month'
            ])
            mort_cols = existing([
                'C5. Number of Rams that died in the last 1 month','C5. Number of Ewes that died in the last 1 month',
                'C5. Number of Bucks that died in the last 1 month','C5. Number of Does that died in the last 1 month'
            ])
            loss_cols = existing([
                'C6. Number of Rams lost/not found or lost to wild animals in the last 1 month',
                'C6. Number of Ewes lost/not found or lost to wild animals in the last 1 month',
                'C6. Number of Bucks lost/not found or lost to wild animals in the last 1 month',
                'C6. Number of Does lost/not found or lost to wild animals in the last 1 month'
            ])
            for c in birth_cols + mort_cols + loss_cols:
                self.df[c] = to_num(self.df[c]).fillna(0)
            if birth_cols: self.df['total_births'] = self.df[birth_cols].sum(axis=1)
            if mort_cols:  self.df['total_mortality'] = self.df[mort_cols].sum(axis=1)
            if loss_cols:  self.df['total_losses'] = self.df[loss_cols].sum(axis=1)

            self.df.loc[valid, 'birth_rate_per_100']     = self.df.loc[valid, 'total_births']     / self.df.loc[valid, 'total_sr'] * 100
            self.df.loc[valid, 'mortality_rate_per_100'] = self.df.loc[valid, 'total_mortality']  / self.df.loc[valid, 'total_sr'] * 100
            self.df.loc[valid, 'loss_rate_per_100']      = self.df.loc[valid, 'total_losses']     / self.df.loc[valid, 'total_sr'] * 100
            self.df.loc[~valid, ['birth_rate_per_100','mortality_rate_per_100','loss_rate_per_100']] = 0
        except Exception as e:
            st.warning(f"Some herd metrics could not be calculated: {str(e)}")
            for col in ['total_sheep','total_goats','total_sr','pct_female','pct_male',
                        'total_births','total_mortality','total_losses',
                        'birth_rate_per_100','mortality_rate_per_100','loss_rate_per_100']:
                if col not in self.df.columns: self.df[col] = 0.0

    def calculate_pl_metrics(self):
        """
        Calculate Profit & Loss metrics with robust column detection.

        Revenue now follows your rule strictly:
        Household Income = (sheep_qty × sheep_price) + (goat_qty × goat_price)
        computed separately for KPMD and Non-KPMD channels (no 'times' multiplier).

        Also records which columns were used in self._income_debug for quick inspection.
        """
        try:
            z = pd.Series(0.0, index=self.df.index)
            self._income_debug = {}  # for UI debugging

            def _pick_qty_price(species: str, kpmd: bool):
                # species: 'sheep' or 'goat'; kpmd: True for KPMD (E1/E2), False for Non-KPMD (E3/E4)
                if species == 'sheep' and kpmd:
                    qty_exact  = ['E1a. How many sheep did you sell to KPMD off-takers  last month?']
                    qty_pats   = [r'^E1a\..*(how many|number).*(sheep).*sell.*KPMD', r'^E1\..*how many.*sheep.*KPMD']
                    price_exact= ['E1c. What was the average price per sheep last month?']
                    price_pats = [r'^E1c\..*(average|avg).*price.*sheep', r'^E1\..*price.*sheep.*KPMD']
                elif species == 'goat' and kpmd:
                    qty_exact  = ['E2a. How many goats did you sell to KPMD off-takers  last month?']
                    qty_pats   = [r'^E2a\..*(how many|number).*(goat).*sell.*KPMD', r'^E2\..*how many.*goat.*KPMD']
                    price_exact= ['E2c. What was the average price per goat last month?']
                    price_pats = [r'^E2c\..*(average|avg).*price.*goat', r'^E2\..*price.*goat.*KPMD']
                elif species == 'sheep' and not kpmd:
                    qty_exact  = ['E3b. How many sheep did you sell to non-KPMD off-takers  last month?']
                    qty_pats   = [r'^E3b\..*(how many|number).*(sheep).*sell.*non.*KPMD', r'^E3\..*how many.*sheep.*non']
                    price_exact= ['E3d. What was the average price per sheep last month?']
                    price_pats = [r'^E3d\..*(average|avg).*price.*sheep', r'^E3\..*price.*sheep.*non']
                else:  # goats, non-KPMD
                    qty_exact  = ['E4b. How many goats did you sell to non-KPMD off-takers  last month?']
                    qty_pats   = [r'^E4b\..*(how many|number).*(goat).*sell.*non.*KPMD', r'^E4\..*how many.*goat.*non']
                    price_exact= ['E4d. What was the average price per goat last month?']
                    price_pats = [r'^E4d\..*(average|avg).*price.*goat', r'^E4\..*price.*goat.*non']

                # exact match first
                qty_col = next((c for c in qty_exact if c in self.df.columns), None)
                price_col = next((c for c in price_exact if c in self.df.columns), None)

                # regex fallback
                if qty_col is None:
                    for pat in qty_pats:
                        hits = [c for c in self.df.columns if re.search(pat, c, flags=re.IGNORECASE)]
                        if hits:
                            qty_col = hits[0]
                            break
                if price_col is None:
                    for pat in price_pats:
                        hits = [c for c in self.df.columns if re.search(pat, c, flags=re.IGNORECASE)]
                        if hits:
                            price_col = hits[0]
                            break
                return qty_col, price_col

            # ---- Revenue sources (strict qty × price, no “times”) ----
            sk_qty, sk_price = _pick_qty_price('sheep', True)
            gk_qty, gk_price = _pick_qty_price('goat', True)
            sn_qty, sn_price = _pick_qty_price('sheep', False)
            gn_qty, gn_price = _pick_qty_price('goat', False)

            # Convert & compute
            if sk_qty and sk_price:
                self.df['sheep_kpmd_revenue'] = to_num(self.df[sk_qty]).fillna(0) * to_num(self.df[sk_price]).fillna(0)
            else:
                self.df['sheep_kpmd_revenue'] = 0.0
            if gk_qty and gk_price:
                self.df['goat_kpmd_revenue'] = to_num(self.df[gk_qty]).fillna(0) * to_num(self.df[gk_price]).fillna(0)
            else:
                self.df['goat_kpmd_revenue'] = 0.0
            if sn_qty and sn_price:
                self.df['sheep_non_kpmd_revenue'] = to_num(self.df[sn_qty]).fillna(0) * to_num(self.df[sn_price]).fillna(0)
            else:
                self.df['sheep_non_kpmd_revenue'] = 0.0
            if gn_qty and gn_price:
                self.df['goat_non_kpmd_revenue'] = to_num(self.df[gn_qty]).fillna(0) * to_num(self.df[gn_price]).fillna(0)
            else:
                self.df['goat_non_kpmd_revenue'] = 0.0

            # Record what we used (for a quick UI debug table)
            self._income_debug['channels'] = [
                {'Channel':'Sheep KPMD',     'Qty':sk_qty, 'Price':sk_price},
                {'Channel':'Goat KPMD',      'Qty':gk_qty, 'Price':gk_price},
                {'Channel':'Sheep Non-KPMD', 'Qty':sn_qty, 'Price':sn_price},
                {'Channel':'Goat Non-KPMD',  'Qty':gn_qty, 'Price':gn_price},
            ]

            # ------ Feed income (unchanged) ------
            if all(c in self.df.columns for c in ['B6d. At What price did you sell a 15 kg bale last month?','B6e. Number of 15 kg bales sold in the last 1 month?']):
                self.df['fodder_revenue'] = (
                    to_num(self.df['B6d. At What price did you sell a 15 kg bale last month?']).fillna(0) *
                    to_num(self.df['B6e. Number of 15 kg bales sold in the last 1 month?']).fillna(0)
                )
            elif 'fodder_revenue' not in self.df.columns:
                self.df['fodder_revenue'] = 0.0

            # Totals and segmented incomes
            revenue_components = [
                'sheep_kpmd_revenue','goat_kpmd_revenue',
                'sheep_non_kpmd_revenue','goat_non_kpmd_revenue','fodder_revenue'
            ]
            self.df['total_revenue']   = self.df[revenue_components].sum(axis=1)
            self.df['income_kpmd']     = self.df['sheep_kpmd_revenue'] + self.df['goat_kpmd_revenue']
            self.df['income_non_kpmd'] = self.df['sheep_non_kpmd_revenue'] + self.df['goat_non_kpmd_revenue']
            self.df['income_feed']     = self.df['fodder_revenue']

            # -------- Costs (same structure as before) --------
            cost_components = []
            if 'Feed_Expenditure' in self.df.columns:
                self.df['feed_costs'] = to_num(self.df['Feed_Expenditure']).fillna(0); cost_components.append('feed_costs')
            if 'B3b. What was the cost of herding per month (Ksh)?' in self.df.columns:
                self.df['herding_costs'] = to_num(self.df['B3b. What was the cost of herding per month (Ksh)?']).fillna(0); cost_components.append('herding_costs')

            vet_costs = []
            if 'D1b. What was the cost of small ruminants vaccination in KSH per animal in the last month?' in self.df.columns:
                self.df['vaccination_costs'] = to_num(self.df['D1b. What was the cost of small ruminants vaccination in KSH per animal in the last month?']).fillna(0); vet_costs.append('vaccination_costs')
            if 'D3b. What was the total cost of treatment in KSH last month?' in self.df.columns:
                self.df['treatment_costs'] = to_num(self.df['D3b. What was the total cost of treatment in KSH last month?']).fillna(0); vet_costs.append('treatment_costs')
            if 'D4a. What was the total of cost of deworming in KSH last month?' in self.df.columns:
                self.df['deworming_costs'] = to_num(self.df['D4a. What was the total of cost of deworming in KSH last month?']).fillna(0); vet_costs.append('deworming_costs')
            if vet_costs:
                self.df['vet_costs'] = self.df[vet_costs].sum(axis=1); cost_components.append('vet_costs')

            transport_cols = [
                'E1h. What was the transport cost to  the market per sheep last month?',
                'E2h. What was the transport cost to  the market per goat last month?',
                'E3i. What was the transport cost to  the market per sheep last month?',
                'E4i. What was the transport cost to  the market per goat last month?'
            ]
            existing_transport = [c for c in transport_cols if c in self.df.columns]
            if existing_transport:
                for c in existing_transport: self.df[c] = to_num(self.df[c]).fillna(0)
                self.df['transport_costs'] = self.df[existing_transport].sum(axis=1); cost_components.append('transport_costs')

            other_costs_cols = [
                'B4b. What is the total cost of fencing(Ksh)?',
                'B4b. What is the total monthly cost of use of minerals(Ksh)?',
                'B4b. What is the total monthly cost of catration of small ruminants(Ksh)?',
                'B4b. What is the total monthly cost of hoof trimming(Ksh)?',
                'B4b. What is the total monthly cost of cleaning the pens(Ksh)?',
                'B4b. What is the total monthly cost of ear tagging(Ksh)?',
                'B4b. What is the total monthly cost of water(Ksh)?',
                'B4b. What is the total monthly cost of spraying of acaricides(Ksh)?'
            ]
            existing_other = [c for c in other_costs_cols if c in self.df.columns]
            if existing_other:
                for c in existing_other: self.df[c] = to_num(self.df[c]).fillna(0)
                self.df['other_costs'] = self.df[existing_other].sum(axis=1)
                cost_components.append('other_costs')

            self.df['total_costs'] = self.df[cost_components].sum(axis=1) if cost_components else 0.0
            self.df['net_profit']  = self.df['total_revenue'] - self.df['total_costs']
            valid_revenue = self.df['total_revenue'] > 0
            self.df.loc[valid_revenue, 'profit_margin'] = (
                self.df.loc[valid_revenue, 'net_profit'] / self.df.loc[valid_revenue, 'total_revenue'] * 100
            )

            # Safety net: ensure presence of key columns
            must_have = [
                'sheep_kpmd_revenue','goat_kpmd_revenue',
                'sheep_non_kpmd_revenue','goat_non_kpmd_revenue',
                'fodder_revenue','total_revenue','total_costs',
                'net_profit','profit_margin','income_kpmd','income_non_kpmd','income_feed'
            ]
            for c in must_have:
                if c not in self.df.columns:
                    self.df[c] = 0.0
        except Exception as e:
            st.warning(f"P&L metric calculation issue: {e}")

# -------------------------------------------------
# Dashboard Renderer
# -------------------------------------------------
class DashboardRenderer:
    def __init__(self, data_processor):
        self.dp = data_processor

    @property
    def df(self):
        return self.dp.df

    def _controls_for_lsmeans(self, group_col=None):
        candidates = ['County', 'Gender', 'total_sr', 'month']
        return [c for c in candidates if c in self.df.columns and c != group_col]

    def _with_kpmd_status(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with a categorical 'KPMD Status' column for clean legends/axes."""
        df = data.copy()
        if 'kpmd_registered' in df.columns:
            df['KPMD Status'] = df['kpmd_registered'].map({1: 'KPMD', 0: 'Non-KPMD'})
            df['KPMD Status'] = pd.Categorical(df['KPMD Status'], categories=['Non-KPMD', 'KPMD'], ordered=True)
        return df

    def create_comparison_cards(self, data, metric_col, title, format_str="{:.1f}"):
        try:
            if metric_col not in data.columns:
                st.info(f"Column '{metric_col}' not found in dataset.")
                return
            kpmd_data = data[data['kpmd_registered'] == 1]
            non_kpmd_data = data[data['kpmd_registered'] == 0]
            col1, col2 = st.columns(2)
            controls = self._controls_for_lsmeans(group_col='kpmd_registered')
            lsm = lsmeans_by_group(data.dropna(subset=[metric_col]), metric_col, 'kpmd_registered', controls=controls) or {}

            with col1:
                v = kpmd_data[metric_col].mean() if (metric_col in kpmd_data.columns and len(kpmd_data) > 0) else 0
                txt = format_str.format(v if pd.notna(v) else 0)
                st.markdown(f"""
                <div class="metric-card kpmd-card">
                    <h4>KPMD Registered</h4>
                    <h3>{txt}</h3>
                    <small>n={len(kpmd_data)}</small>
                    {fmt_lsmean_note(format_str.format(lsm.get(1, v)) if isinstance(lsm, dict) else "")}
                </div>
                """, unsafe_allow_html=True)

            with col2:
                v = non_kpmd_data[metric_col].mean() if (metric_col in non_kpmd_data.columns and len(non_kpmd_data) > 0) else 0
                txt = format_str.format(v if pd.notna(v) else 0)
                st.markdown(f"""
                <div class="metric-card non-kpmd-card">
                    <h4>Non-KPMD</h4>
                    <h3>{txt}</h3>
                    <small>n={len(non_kpmd_data)}</small>
                    {fmt_lsmean_note(format_str.format(lsm.get(0, v)) if isinstance(lsm, dict) else "")}
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not create comparison cards for {metric_col}: {e}")

    # ---------------- NEW: Pastoral Livelihoods (as tabs) ----------------
    def render_pastoral_livelihoods(self):
        st.header("🏠 Pastoral Livelihoods")
        self.dp.calculate_pl_metrics()

        tab1, tab2, tab3 = st.tabs([
            "Household Income Segmentation (Monthly)",
            "Access to Markets",
            "Price Information Access"
        ])

        # --- Tab 1: Income Segmentation ---
        with tab1:
            st.subheader("Household Income Segmentation (Monthly)")
            # ---- Ensure columns exist before slicing (belt-and-braces) ----
            for c in ['income_kpmd','income_non_kpmd','income_feed','kpmd_registered']:
                if c not in self.df.columns:
                    self.df[c] = 0.0

            try:
                inc = self.df[['income_kpmd','income_non_kpmd','income_feed','kpmd_registered']].copy()
                inc.rename(columns={
                    'income_kpmd':'KPMD Livestock Income',
                    'income_non_kpmd':'Non-KPMD Livestock Income',
                    'income_feed':'Feed Income'
                }, inplace=True)

                avg_comp = inc[['KPMD Livestock Income','Non-KPMD Livestock Income','Feed Income']].mean()
                fig = px.pie(values=avg_comp.values, names=avg_comp.index, title='Average Household Income Mix')
                st.plotly_chart(fig, use_container_width=True)

                melted = inc.melt(id_vars=['kpmd_registered'], var_name='Income Type', value_name='KES')
                grp = melted.groupby(['kpmd_registered','Income Type'])['KES'].mean().reset_index()
                grp['KPMD Status'] = grp['kpmd_registered'].map({1:'KPMD',0:'Non-KPMD'})
                fig2 = px.bar(grp, x='Income Type', y='KES', color='KPMD Status', barmode='group',
                        title='Average Income by KPMD Registration')
                fig2.update_traces(text=grp['KES'].round(0), textposition='outside')
                fig2.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig2, use_container_width=True)

                controls = self._controls_for_lsmeans(group_col='kpmd_registered')
                ls_notes = []
                for inc_type, col in [('KPMD Livestock Income','income_kpmd'),
                                    ('Non-KPMD Livestock Income','income_non_kpmd'),
                                    ('Feed Income','income_feed')]:
                    lsm = lsmeans_by_group(inc.dropna(subset=[inc_type]).assign(**{
                        # map back to original column names for lsmeans
                        col: inc[inc_type]
                    }), col, 'kpmd_registered', controls=controls)
                    if isinstance(lsm, dict):
                        ls_notes.append(f"{inc_type} — LSMean KPMD: {lsm.get(1, np.nan):,.0f}, Non-KPMD: {lsm.get(0, np.nan):,.0f}")
                if ls_notes:
                    st.caption("Adjusted (LSMeans): " + " | ".join(ls_notes))
            except KeyError as e:
                st.warning(f"Error processing income segmentation: missing {list(e.args)}")

        # --- Tab 2: Access to Markets ---
        with tab2:
            st.subheader("Access to Markets")
            f1 = coalesce_first(self.df, ['F1. How far did you travel to sell small ruminants in Kilometers last month?'])
            if f1:
                try:
                    self.create_comparison_cards(self.df, f1, "Distance to Market (km)", "{:.1f} km")
                    fig = px.histogram(self.df, x=f1, nbins=20, title='Distribution of Distance to Market (km)',
                                       labels={f1:'Distance (km)'})
                    st.plotly_chart(fig, use_container_width=True)
                    if 'kpmd_registered' in self.df.columns:
                        fig2 = px.box(self.df, x='kpmd_registered', y=f1, color='kpmd_registered',
                                      labels={'kpmd_registered':'KPMD Registered', f1:'Distance (km)'},
                                      title='Distance to Market by KPMD Registration')
                        st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.warning(f"Error processing Access to Markets: {e}")
            else:
                st.info("F1 distance to market not available in this dataset")

        # --- Tab 3: Price Information Access ---
        with tab3:
            st.subheader("Price Information Access")
            f2 = coalesce_first(self.df, ['F2. Did you get information about livestock prices prior to selling in the last three months?'])
            if f2:
                try:
                    tmp = self.df.copy()
                    tmp['price_info'] = tmp[f2].apply(yn).astype(int)
                    self.create_comparison_cards(tmp, 'price_info', 'Households Accessing Price Info', '{:.1%}')
                except Exception as e:
                    st.warning(f"Error processing Price Information Access: {e}")
            else:
                st.info("F2 (price information) not available")

    # ---------------- KPMD Participation ----------------
    def render_kpmd_participation(self):
        st.header("🤝 KPMD Participation")

        months_col = coalesce_first(self.df, ['A9. For how many months have you been participating in KPMD?'])
        if months_col:
            try:
                tmp = to_num(self.df[months_col]).fillna(0)
                st.metric("Average Months in KPMD", f"{tmp.mean():.1f}")
                fig = px.histogram(self.df, x=months_col, title='Distribution of Months in KPMD',
                                   labels={months_col:'Months'}, nbins=10)
                fig.update_traces(textposition='outside')
                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Could not parse A9 months")

        st.subheader("B1. Trainings Received (last 1 month)")
        b1_stem = "B1. Have you received any of the following through KPMD in the past 1 month? (select all that apply)"
        b1_cols = [c for c in self.df.columns if c.startswith(b1_stem + "/")]
        if b1_cols:
            rows=[]
            for c in b1_cols:
                name = c.split('/')[-1]
                for s in [0,1]:
                    sub = self.df[self.df['kpmd_registered']==s]
                    rate = pd.to_numeric(sub[c].astype(str).replace({'1':1,'0':0}), errors='coerce').fillna(0).mean() * 100 if len(sub) else 0
                    rows.append({'Training': name, 'Rate': rate, 'KPMD Status':'KPMD' if s==1 else 'Non-KPMD'})
            dfp = pd.DataFrame(rows)
            fig = px.bar(dfp, x='Training', y='Rate', color='KPMD Status', barmode='group',
                         title='KPMD Trainings in last 1 month (%)')
            fig.update_traces(text=dfp['Rate'].round(1), textposition='outside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("B1 training option columns not found")

        st.subheader("B2. Interventions Received (last 1 month)")
        b2_stem = "B2. Have you received any of the following through KPMD in the past 1 month? (select all that apply)"
        b2_cols = [c for c in self.df.columns if c.startswith(b2_stem + "/")]
        if b2_cols:
            rows=[]
            for c in b2_cols:
                name = c.split('/')[-1]
                for s in [0,1]:
                    sub = self.df[self.df['kpmd_registered']==s]
                    rate = pd.to_numeric(sub[c].astype(str).replace({'1':1,'0':0}), errors='coerce').fillna(0).mean() * 100 if len(sub) else 0
                    rows.append({'Intervention': name, 'Rate': rate, 'KPMD Status':'KPMD' if s==1 else 'Non-KPMD'})
            dfp = pd.DataFrame(rows)
            fig = px.bar(dfp, x='Intervention', y='Rate', color='KPMD Status', barmode='group',
                         title='KPMD Interventions in last 1 month (%)')
            fig.update_traces(text=dfp['Rate'].round(1), textposition='outside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("B2 intervention option columns not found")

    # ---------------- P&L Analysis ----------------
    def render_pl_analysis(self):
        st.header("💰 Profit & Loss Analysis")
        self.dp.calculate_pl_metrics()

        tab1, tab2, tab3, tab4 = st.tabs(["Overall Profitability", "Revenue Analysis", "Cost Analysis", "Channel Comparison"])

        with tab1:
            st.subheader("Overall Profitability")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_profit = self.df['net_profit'].mean()
                st.metric("Average Net Profit (KES)", f"{(avg_profit if pd.notna(avg_profit) else 0):,.0f}")
                if 'kpmd_registered' in self.df.columns:
                    controls = self._controls_for_lsmeans(group_col='kpmd_registered')
                    lsm = lsmeans_by_group(self.df.dropna(subset=['net_profit']), 'net_profit', 'kpmd_registered', controls)
                    if isinstance(lsm, dict):
                        st.caption(f"Adjusted (LSMean) — KPMD: {lsm.get(1, np.nan):,.0f} | Non-KPMD: {lsm.get(0, np.nan):,.0f}")
            with col2:
                avg_margin = self.df['profit_margin'].mean()
                st.metric("Average Profit Margin (%)", f"{(avg_margin if pd.notna(avg_margin) else 0):.1f}%")
            with col3:
                profitable_hhs = (self.df['net_profit'] > 0).sum()
                total_hhs = len(self.df)
                pct = (profitable_hhs / total_hhs * 100) if total_hhs else 0
                st.metric("Profitable Households", f"{pct:.1f}%")
            with col4:
                avg_revenue = self.df['total_revenue'].mean()
                st.metric("Average Monthly Revenue (KES)", f"{(avg_revenue if pd.notna(avg_revenue) else 0):,.0f}")

            st.subheader("Profit Distribution")
            col1, col2 = st.columns(2)

            # Left: keep the histogram as-is
            with col1:
                fig = px.histogram(
                    self.df,
                    x='net_profit',
                    title='Distribution of Net Profit',
                    labels={'net_profit': 'Net Profit (KES)'}
                )
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)

            # Right: Box by clean Registration labels
            with col2:
                if 'kpmd_registered' in self.df.columns:
                    tmp = self._with_kpmd_status(self.df)
                    fig = px.box(
                        tmp,
                        x='KPMD Status',
                        y='net_profit',
                        color='KPMD Status',
                        category_orders={'KPMD Status': ['Non-KPMD', 'KPMD']},
                        title='Profit Distribution by Registration',
                        labels={'KPMD Status': 'Registration', 'net_profit': 'Net Profit (KES)'}
                    )
                    fig.update_layout(legend_title_text='Registration')
                    st.plotly_chart(fig, use_container_width=True)

            if 'County' in self.df.columns:
                st.subheader("Profitability by County")
                county_profit = self.df.groupby('County', dropna=True)['net_profit'].agg(['mean','count']).reset_index()
                county_profit = county_profit[county_profit['count'] >= 3]
                if len(county_profit) > 0:
                    fig = px.bar(county_profit, x='County', y='mean',
                                 title='Average Net Profit by County',
                                 labels={'mean': 'Average Net Profit (KES)'}, color='mean')
                    fig.update_traces(text=county_profit['mean'].round(0), textposition='outside')
                    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                    st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Revenue Analysis")
            revenue_cols = [c for c in self.df.columns if 'revenue' in c.lower() and c != 'total_revenue']
            if revenue_cols:
                avg_comp = self.df[revenue_cols].mean().sort_values(ascending=False)
                fig = px.pie(values=avg_comp.values, names=avg_comp.index, title='Average Revenue Composition')
                st.plotly_chart(fig, use_container_width=True)
            if all(c in self.df.columns for c in ['income_kpmd','income_non_kpmd','income_feed']):
                comp = self.df[['income_kpmd','income_non_kpmd','income_feed']].mean().reset_index()
                comp.columns = ['Source','KES']
                comp['Source'] = comp['Source'].map({
                    'income_kpmd':'KPMD Livestock', 'income_non_kpmd':'Non-KPMD Livestock', 'income_feed':'Feed'
                })
                fig2 = px.bar(comp, x='Source', y='KES', title='Average Income by Source (All Households)')
                fig2.update_traces(text=comp['KES'].round(0), textposition='outside')
                fig2.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig2, use_container_width=True)

            if 'kpmd_registered' in self.df.columns:
                rc = self.df.groupby('kpmd_registered')['total_revenue'].mean().reset_index()
                rc['KPMD_Status'] = rc['kpmd_registered'].map({1: 'KPMD', 0: 'Non-KPMD'})
                fig = px.bar(rc, x='KPMD_Status', y='total_revenue', title='Average Revenue by KPMD Status',
                             labels={'total_revenue': 'Average Revenue (KES)'})
                fig.update_traces(text=rc['total_revenue'].round(0), textposition='outside')
                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Cost Structure Analysis")
            cost_cols = [c for c in self.df.columns if 'costs' in c.lower() and c != 'total_costs']
            if cost_cols:
                avg_cost = self.df[cost_cols].mean().sort_values(ascending=False)
                fig = px.bar(avg_cost, title='Average Cost Composition',
                             labels={'value': 'Average Cost (KES)', 'index': 'Cost Category'})
                fig.update_traces(text=avg_cost.round(0), textposition='outside')
                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Cost Efficiency")
            col1, col2 = st.columns(2)
            with col1:
                if 'total_costs' in self.df.columns and 'total_sr' in self.df.columns:
                    self.df['cost_per_animal'] = self.df['total_costs'] / self.df['total_sr'].replace(0, np.nan)
                    valid = self.df[self.df['cost_per_animal'].notna()]
                    if len(valid) > 0:
                        self.create_comparison_cards(valid, 'cost_per_animal', 'Cost per Animal', 'KES {:.0f}')
            with col2:
                if 'total_revenue' in self.df.columns and 'total_costs' in self.df.columns:
                    self.df['cost_ratio'] = self.df['total_costs'] / self.df['total_revenue'].replace(0, np.nan)
                    valid = self.df[self.df['cost_ratio'].notna()]
                    if len(valid) > 0:
                        self.create_comparison_cards(valid, 'cost_ratio', 'Cost-to-Revenue Ratio', '{:.2f}')

        with tab4:
            st.subheader("Channel Profitability Comparison")
            channel_cols = ['sheep_kpmd_profit_margin', 'sheep_non_kpmd_profit_margin']
            available = [c for c in channel_cols if c in self.df.columns]
            if available:
                rows = []
                for col in available:
                    channel_name = ' '.join(col.split('_')[:3]).title()
                    for s in [0, 1]:
                        sub = self.df[self.df['kpmd_registered'] == s]
                        rows.append({'Channel': channel_name, 'Profit_Margin': sub[col].mean(),
                                     'KPMD_Status': 'KPMD Registered' if s == 1 else 'Non-KPMD Registered'})
                ch_df = pd.DataFrame(rows)
                fig = px.bar(ch_df, x='Channel', y='Profit_Margin', color='KPMD_Status',
                             title='Channel Profit Margins by KPMD Registration',
                             barmode='group', labels={'Profit_Margin': 'Profit Margin (%)'})
                fig.update_traces(text=ch_df['Profit_Margin'].round(1), textposition='outside')
                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Breakeven Analysis")
            data = self.df.copy()
            data['breakeven_status'] = np.where(data['net_profit'] >= 0, 'Profitable', 'Loss-making')
            if 'kpmd_registered' in data.columns:
                pivot = pd.crosstab(data['kpmd_registered'], data['breakeven_status'], normalize='index') * 100
                pivot = pivot.reset_index(); pivot['KPMD_Status'] = pivot['kpmd_registered'].map({1:'KPMD',0:'Non-KPMD'})
                melted = pivot.melt(id_vars=['KPMD_Status'], value_vars=['Profitable','Loss-making'],
                                    var_name='Status', value_name='Percentage')
                fig = px.bar(melted, x='KPMD_Status', y='Percentage', color='Status',
                             title='Breakeven Status by KPMD Registration', barmode='stack')
                fig.update_traces(text=melted['Percentage'].round(1), textposition='outside')
                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig, use_container_width=True)

    # ---------------- Field & Data Outlook ----------------
    def render_field_outlook(self):
        st.header("🧭 Field & Data Outlook")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Submissions", len(self.df))
        with col2:
            latest = None
            cand = '_submission_time' if '_submission_time' in self.df.columns else 'int_date_std' if 'int_date_std' in self.df.columns else None
            if cand:
                try: latest = pd.to_datetime(self.df[cand], errors='coerce').max()
                except Exception: latest=None
            st.metric("Latest Submission", latest.strftime("%Y-%m-%d") if (latest is not None and pd.notna(latest)) else "N/A")
        with col3: st.metric("Counties Covered", int(self.df['County'].nunique()) if 'County' in self.df.columns else 0)
        with col4: st.metric("KPMD Participants", int(self.df['kpmd_registered'].sum()) if 'kpmd_registered' in self.df.columns else 0)

        left, right = st.columns([0.8, 0.2])
        with left: st.subheader("Submissions Over Time")
        with right:
            gran = st.selectbox("Granularity", ["Daily","Weekly","Monthly"], index=0, label_visibility="collapsed")

        date_col = '_submission_time' if '_submission_time' in self.df.columns else 'int_date_std' if 'int_date_std' in self.df.columns else None
        if date_col:
            tmp = self.df.copy()
            tmp['__date'] = pd.to_datetime(tmp[date_col], errors='coerce')
            tmp = tmp[tmp['__date'].notna()].copy()
            if gran == "Daily":
                tmp['__bucket'] = tmp['__date'].dt.date; x_label, title = "Date", "Daily Submission Volume"
            elif gran == "Weekly":
                tmp['__bucket'] = tmp['__date'].dt.to_period('W').dt.start_time.dt.date; x_label, title = "Week (start)", "Weekly Submission Volume"
            else:
                tmp['__bucket'] = tmp['__date'].dt.to_period('M').dt.to_timestamp(); x_label, title = "Month", "Monthly Submission Volume"
            series = tmp.groupby('__bucket').size().reset_index(name='Submissions').sort_values('__bucket')
            if len(series) > 0:
                fig = px.line(series, x='__bucket', y='Submissions', title=title, markers=True, labels={'__bucket':x_label})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No dated submissions available to plot.")
        else:
            if 'month' in self.df.columns and not self.df['month'].isna().all():
                monthly = self.df.groupby('month').size().reset_index(name='Submissions').sort_values('month')
                fig = px.line(monthly, x='month', y='Submissions', title='Monthly Submission Volume', markers=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No time information available.")

        st.subheader("Submissions by County and KPMD Status")
        if 'County' in self.df.columns and 'kpmd_registered' in self.df.columns:
            county_kpmd = self.df.groupby(['County','kpmd_registered']).size().reset_index(name='count')
            county_kpmd['kpmd_status'] = county_kpmd['kpmd_registered'].map({1:'KPMD',0:'Non-KPMD'})
            fig = px.bar(county_kpmd, x='County', y='count', color='kpmd_status',
                         title='Submissions by County and KPMD Status', barmode='group')
            fig.update_traces(text=county_kpmd['count'], textposition='outside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("County or KPMD data not available")

        # ---------------- Household Locations ----------------
        st.subheader("Household Locations")

        try:
            import geopandas as gpd
            from shapely.geometry import Point
            import json
            import pydeck as pdk
            from pathlib import Path
        except Exception:
            st.info(
                "To enable the interactive map, install:\n"
                "  py -3 -m pip install geopandas shapely pydeck\n"
                "Then rerun the app."
            )
        else:
            lat_col = '_GPS Coordinates_latitude'
            lon_col = '_GPS Coordinates_longitude'

            if all(c in self.df.columns for c in (lat_col, lon_col)):
                # 1) Points GDF
                pts_df = self.df.dropna(subset=[lat_col, lon_col]).copy()
                if len(pts_df) == 0:
                    st.info("No GPS points to map.")
                else:
                    gdf_pts = gpd.GeoDataFrame(
                        pts_df,
                        geometry=gpd.points_from_xy(pts_df[lon_col].astype(float), pts_df[lat_col].astype(float)),
                        crs="EPSG:4326",
                    )

                    # 2) Load county polygons (geoBoundaries)
                    counties_path = Path("geo/kenya_counties.geojson")
                    if not counties_path.exists():
                        st.warning("Missing geo/kenya_counties.geojson. Run:  py -3 scripts\\fetch_kenya_geo.py")
                    else:
                        gdf_counties = gpd.read_file(counties_path).to_crs("EPSG:4326")
                        name_col = "shapeName" if "shapeName" in gdf_counties.columns else gdf_counties.columns[0]

                        # 3) Spatial join → farmer counts per county
                        joined = gpd.sjoin(
                            gdf_pts[["geometry"]],
                            gdf_counties[[name_col, "geometry"]],
                            predicate="within",
                            how="left",
                        )
                        counts = joined.groupby(name_col).size().rename("farmers").reset_index()
                        gdf_counties = gdf_counties.merge(counts, on=name_col, how="left").fillna({"farmers": 0})

                        # 4) Build colors (OrRd-ish ramp)
                        v = gdf_counties["farmers"].astype(float)
                        vmin, vmax = float(v.min()), float(v.max())
                        span = (vmax - vmin) if vmax > vmin else 1.0
                        t = (v - vmin) / span
                        # start (very light) → end (dark red)
                        r0, g0, b0 = 254, 240, 217
                        r1, g1, b1 = 165,  15,  21
                        gdf_counties["r"] = (r0 + t * (r1 - r0)).round().clip(0,255).astype(int)
                        gdf_counties["g"] = (g0 + t * (g1 - g0)).round().clip(0,255).astype(int)
                        gdf_counties["b"] = (b0 + t * (b1 - b0)).round().clip(0,255).astype(int)

                        # 5) Text anchor points for labels
                        reps = gdf_counties.representative_point()
                        gdf_labels = gpd.GeoDataFrame(
                            {
                                "name": gdf_counties[name_col].astype(str),
                                "farmers": gdf_counties["farmers"].astype(int),
                                "lon": reps.x,
                                "lat": reps.y,
                            },
                            geometry=reps,
                            crs="EPSG:4326",
                        )
                        gdf_labels["label"] = gdf_labels.apply(lambda r: f"{r['name']}\n{r['farmers']}", axis=1)

                        # 6) Focus & layer controls (collapsed by default)
                        max_count = int(vmax) if np.isfinite(vmax) else 0
                        with st.expander("Map focus & layers", expanded=False):
                            st.caption("Highlight areas with data")
                            focus_mode = st.radio(
                                "Highlight areas with data",
                                ["Normal", "Dim areas below threshold", "Hide areas below threshold"],
                                index=1,
                                horizontal=True,
                                label_visibility="collapsed",
                                key="map_focus_mode",
                            )

                            if max_count > 0 and focus_mode != "Normal":
                                threshold = st.slider(
                                    "Minimum # of farmers to focus",
                                    0, max_count, max(1, int(round(max_count * 0.05))),
                                    key="map_focus_threshold",
                                )
                            else:
                                threshold = 0

                            c1, c2 = st.columns(2)
                            with c1:
                                show_points = st.checkbox("Show household points", value=True, key="map_show_points")
                            with c2:
                                show_labels = st.checkbox("Show county labels", value=True, key="map_show_labels")

                        # 7) Prepare polygons according to focus mode
                        gdf_c = gdf_counties.copy()

                        if focus_mode == "Hide areas below threshold" and max_count > 0:
                            gdf_c = gdf_c[gdf_c["farmers"] >= threshold].copy()
                            # If everything got filtered out, fall back to original so we don't render a blank map
                            if gdf_c.empty:
                                gdf_c = gdf_counties.copy()
                            gdf_c["a"] = 220  # solid fill for what remains
                        elif focus_mode == "Dim areas below threshold" and max_count > 0:
                            # Give high alpha (solid) to focused areas; dim the rest
                            gdf_c["a"] = np.where(gdf_c["farmers"] >= threshold, 220, 40).astype(int)
                        else:
                            # Normal: uniform but still nicely visible
                            gdf_c["a"] = 180

                        # 8) Labels built from the SAME set used for the polygons after filtering
                        reps = gdf_c.representative_point()
                        gdf_labels = gpd.GeoDataFrame(
                            {
                                "name": gdf_c[name_col].astype(str),
                                "farmers": gdf_c["farmers"].astype(int),
                                "lon": reps.x,
                                "lat": reps.y,
                            },
                            geometry=reps,
                            crs="EPSG:4326",
                        )
                        gdf_labels["label"] = gdf_labels.apply(lambda r: f"{r['name']}\n{r['farmers']}", axis=1)

                        # 9) Build layers
                        layers = []

                        # Choropleth polygons (use per-feature alpha 'a'; colors already computed)
                        counties_geojson = json.loads(gdf_c.to_json())
                        layers.append(
                            pdk.Layer(
                                "GeoJsonLayer",
                                data=counties_geojson,
                                stroked=True,
                                filled=True,
                                get_fill_color="[properties.r, properties.g, properties.b, properties.a]",
                                get_line_color=[120, 120, 120, 180],
                                line_width_min_pixels=0.8,
                                pickable=True,
                            )
                        )

                        # Labels (optional)
                        if show_labels and not gdf_labels.empty:
                            layers.append(
                                pdk.Layer(
                                    "TextLayer",
                                    data=gdf_labels[["lon", "lat", "label"]].to_dict(orient="records"),
                                    get_position="[lon, lat]",
                                    get_text="label",
                                    get_size=10,
                                    get_color=[30, 30, 30],
                                    get_alignment_baseline="'center'",
                                    billboard=True,
                                )
                            )

                        # Household points (optional)
                        if show_points and not gdf_pts.empty:
                            has_kpmd = "kpmd_registered" in gdf_pts.columns
                            data_pts = gdf_pts.assign(
                                r=lambda d: d["kpmd_registered"].map({1: 31, 0: 214}) if has_kpmd else 160,
                                g=lambda d: d["kpmd_registered"].map({1:119, 0:  39}) if has_kpmd else 160,
                                b=lambda d: d["kpmd_registered"].map({1:180, 0:  40}) if has_kpmd else 160,
                                lon=lambda d: d.geometry.x,
                                lat=lambda d: d.geometry.y,
                            )
                            layers.append(
                                pdk.Layer(
                                    "ScatterplotLayer",
                                    data=data_pts[["lon", "lat", "r", "g", "b"]].to_dict(orient="records"),
                                    get_position="[lon, lat]",
                                    get_radius=700,
                                    get_fill_color="[r, g, b, 200]",
                                    pickable=True,
                                )
                            )

                        # 10) Kenya-centric view — fit to the possibly filtered polygons to reinforce the focus
                        bounds = gdf_c.total_bounds  # [minx, miny, maxx, maxy]
                        cx = float((bounds[0] + bounds[2]) / 2)
                        cy = float((bounds[1] + bounds[3]) / 2)
                        view_state = pdk.ViewState(latitude=cy, longitude=cx, zoom=5.6, pitch=0, bearing=0)

                        # 11) Render
                        st.pydeck_chart(
                            pdk.Deck(
                                map_style="mapbox://styles/mapbox/light-v9",
                                initial_view_state=view_state,
                                layers=layers,
                                tooltip={"text": "{name}\nFarmers: {farmers}"},
                            )
                        )

    # ---------------- Pastoral Productivity ----------------
    def render_pastoral_productivity(self):
        st.header("🐑 Pastoral Productivity")
        self.dp.calculate_herd_metrics()

        tab1, tab2, tab3 = st.tabs(["Herd Composition", "Animal Health Indicators", "SR Productivity Indicators"])

        with tab1:
            st.subheader("Herd Structure & Size")
            st.write("**Average Animals Owned**")
            col1, col2 = st.columns(2)
            with col1:
                if 'total_sheep' in self.df.columns:
                    self.create_comparison_cards(self.df, 'total_sheep', 'Average Sheep', '{:.1f}')
                else:
                    st.info("Sheep data not available")
            with col2:
                if 'total_goats' in self.df.columns:
                    self.create_comparison_cards(self.df, 'total_goats', 'Average Goats', '{:.1f}')
                else:
                    st.info("Goat data not available")

            if 'pct_female' in self.df.columns:
                st.write("**Percentage Female Stock**")
                self.create_comparison_cards(self.df, 'pct_female', 'Female Stock %', '{:.1f}%')
            if 'pct_male' in self.df.columns:
                st.write("**Percentage Male Stock**")
                self.create_comparison_cards(self.df, 'pct_male', 'Male Stock %', '{:.1f}%')

            if all(col in self.df.columns for col in ['total_sheep','total_goats','kpmd_registered']):
                st.subheader("Herd Composition by KPMD Status")
                try:
                    comp = self.df.groupby('kpmd_registered')[['total_sheep','total_goats']].mean().reset_index()
                    comp['kpmd_status'] = comp['kpmd_registered'].map({1:'KPMD',0:'Non-KPMD'})
                    melted = comp.melt(id_vars=['kpmd_status'], value_vars=['total_sheep','total_goats'],
                                       var_name='Species', value_name='Average Count')
                    melted['Species'] = melted['Species'].map({'total_sheep':'Sheep','total_goats':'Goats'})
                    fig = px.bar(melted, x='kpmd_status', y='Average Count', color='Species',
                                 title='Average Herd Composition by KPMD Status', barmode='group')
                    fig.update_traces(text=melted['Average Count'].round(1), textposition='outside')
                    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("Herd composition data not available")
            else:
                st.info("Herd composition data not available for visualization")

        with tab2:
            st.subheader("Animal Health Indicators")
            vacc_col = self.dp.column_mapping.get('vaccination')
            if vacc_col and vacc_col in self.df.columns:
                vacc_data = self.df.copy()
                vacc_data['vaccinated'] = vacc_data[vacc_col].apply(yn).astype(int)
                self.create_comparison_cards(vacc_data, 'vaccinated', 'Vaccination Rate', '{:.1%}')
            else:
                st.info("Vaccination data not available")

            treat_col = 'D3. Did you treat small ruminants for disease in the last month?'
            if treat_col in self.df.columns:
                treat_data = self.df.copy()
                treat_data['treated'] = treat_data[treat_col].apply(yn).astype(int)
                self.create_comparison_cards(treat_data, 'treated', 'Treatment Rate', '{:.1%}')
            else:
                st.info("Disease treatment data not available")

            deworm_col = 'D4. Did you deworm your small ruminants last month?'
            if deworm_col in self.df.columns:
                deworm_data = self.df.copy()
                deworm_data['dewormed'] = deworm_data[deworm_col].apply(yn).astype(int)
                self.create_comparison_cards(deworm_data, 'dewormed', 'Deworming Rate', '{:.1%}')
            else:
                st.info("Deworming data not available")

            st.subheader("Disease Analysis")
            if hasattr(self.dp, 'vacc_disease_cols') and self.dp.vacc_disease_cols:
                rows=[]
                for col in self.dp.vacc_disease_cols:
                    name = col.split('/')[-1]
                    for s in [0,1]:
                        sub = self.df[self.df['kpmd_registered']==s]
                        if len(sub)>0:
                            rate = sub[col].mean() * 100
                            rows.append({'Disease':name,'Rate':rate,'KPMD_Status':'KPMD' if s==1 else 'Non-KPMD'})
                if rows:
                    dfp = pd.DataFrame(rows)
                    fig = px.bar(dfp, x='Disease', y='Rate', color='KPMD_Status',
                                 title='Vaccination Diseases by KPMD Status (%)', barmode='group')
                    fig.update_traces(text=dfp['Rate'].round(1), textposition='outside')
                    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Vaccination disease data not available")

            if hasattr(self.dp, 'treat_disease_cols') and self.dp.treat_disease_cols:
                rows=[]
                for col in self.dp.treat_disease_cols:
                    name = col.split('/')[-1]
                    for s in [0,1]:
                        sub = self.df[self.df['kpmd_registered']==s]
                        if len(sub)>0:
                            rate = sub[col].mean() * 100
                            rows.append({'Disease':name,'Rate':rate,'KPMD_Status':'KPMD' if s==1 else 'Non-KPMD'})
                if rows:
                    dfp = pd.DataFrame(rows)
                    fig = px.bar(dfp, x='Disease', y='Rate', color='KPMD_Status',
                                 title='Treatment Diseases by KPMD Status (%)', barmode='group')
                    fig.update_traces(text=dfp['Rate'].round(1), textposition='outside')
                    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Treatment disease data not available")

            prov_col = 'D2. Who performed the small ruminants vaccinations in the last month?'
            if prov_col in self.df.columns:
                try:
                    provider_counts = self.df.groupby(['kpmd_registered', prov_col]).size().reset_index(name='count')
                    provider_counts['KPMD_Status'] = provider_counts['kpmd_registered'].map({1:'KPMD',0:'Non-KPMD'})
                    fig = px.bar(provider_counts, x='KPMD_Status', y='count', color=prov_col,
                                 title='Vaccination Providers by KPMD Status')
                    fig.update_traces(text=provider_counts['count'], textposition='outside')
                    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("Vaccination provider data not available")
            else:
                st.info("Vaccination provider data not available")

        with tab3:
            st.subheader("Small Ruminant Productivity Indicators")
            if 'birth_rate_per_100' in self.df.columns:
                self.create_comparison_cards(self.df, 'birth_rate_per_100', 'Birth Rate', '{:.1f}')
            if 'mortality_rate_per_100' in self.df.columns:
                self.create_comparison_cards(self.df, 'mortality_rate_per_100', 'Mortality Rate', '{:.1f}')
            if 'loss_rate_per_100' in self.df.columns:
                self.create_comparison_cards(self.df, 'loss_rate_per_100', 'Loss Rate', '{:.1f}')

            if all(c in self.df.columns for c in ['birth_rate_per_100','mortality_rate_per_100','loss_rate_per_100','kpmd_registered']):
                st.subheader("Productivity Rates by KPMD Status")
                try:
                    prod = self.df.groupby('kpmd_registered')[['birth_rate_per_100','mortality_rate_per_100','loss_rate_per_100']].mean().reset_index()
                    prod['KPMD_Status'] = prod['kpmd_registered'].map({1: 'KPMD', 0: 'Non-KPMD'})
                    m = prod.melt(id_vars=['KPMD_Status'],
                                  value_vars=['birth_rate_per_100','mortality_rate_per_100','loss_rate_per_100'],
                                  var_name='Metric', value_name='Rate')
                    m['Metric'] = m['Metric'].map({'birth_rate_per_100':'Birth Rate','mortality_rate_per_100':'Mortality Rate','loss_rate_per_100':'Loss Rate'})
                    fig = px.bar(m, x='KPMD_Status', y='Rate', color='Metric',
                                 title='Productivity Rates by KPMD Status (per 100 head)', barmode='group',
                                 text=m['Rate'].round(1))
                    fig.update_traces(textposition='outside')
                    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("Productivity rate data not available for visualization")

    # ---------------- Feed & Fodder ----------------
    def render_feed_fodder(self):
        st.header("🌾 Feed & Fodder")
        tab1, tab2, tab3 = st.tabs(["Feed Purchase", "Fodder Production", "Feed Economics"])

        with tab1:
            st.subheader("Feed Purchase Patterns")
            col = 'B5a. Did you purchase fodder in the last 1 month?'
            if col in self.df.columns:
                tmp = self.df.copy(); tmp['purchased'] = tmp[col].apply(yn).astype(int)
                self.create_comparison_cards(tmp, 'purchased', 'Purchase Rate', '{:.1%}')
            else:
                st.info("Fodder purchase data not available")

            st.subheader("Feed Purchase Sources")
            source_cols = [c for c in self.df.columns if c.startswith('B5b. Where did you buy feeds in the last 1 month?/') and 'Other' not in c]
            if source_cols:
                rows=[]
                for c in source_cols:
                    name = c.split('/')[-1]
                    for s in [0,1]:
                        sub = self.df[self.df['kpmd_registered']==s]
                        rate = pd.to_numeric(sub[c].astype(str).replace({'1':1,'0':0}), errors='coerce').fillna(0).mean() * 100 if len(sub) else 0
                        rows.append({'Source': name, 'Rate': rate, 'KPMD_Status': 'KPMD' if s==1 else 'Non-KPMD'})
                dfp = pd.DataFrame(rows)
                fig = px.bar(dfp, x='Source', y='Rate', color='KPMD_Status',
                             title='Feed Purchase Sources by KPMD Status (%)', barmode='group')
                fig.update_traces(text=dfp['Rate'].round(1), textposition='outside')
                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feed source data not available")

        with tab2:
            st.subheader("Fodder Production")
            col = 'B6a. Did you produce any fodder?'
            if col in self.df.columns:
                tmp = self.df.copy(); tmp['produced'] = tmp[col].apply(yn).astype(int)
                self.create_comparison_cards(tmp, 'produced', 'Production Rate', '{:.1%}')
            else:
                st.info("Fodder production data not available")

            qty = 'B6b. Quantity of feeds harvested in the last 1 month (15 kg bales)?'
            if qty in self.df.columns:
                self.create_comparison_cards(self.df, qty, 'Harvested Bales', '{:.1f}')
            else:
                st.info("Fodder harvest quantity data not available")

        with tab3:
            st.subheader("Feed Economics")
            price = 'B5c. What was the price per 15 kg bale in the last 1 month?'
            if price in self.df.columns:
                self.create_comparison_cards(self.df, price, 'Price per Bale', 'KES {:.0f}')
            else:
                st.info("Fodder price data not available")

            if 'Feed_Expenditure' in self.df.columns:
                exp = self.df[self.df['Feed_Expenditure'] > 0]
                if len(exp) > 0: self.create_comparison_cards(exp, 'Feed_Expenditure', 'Feed Expenditure', 'KES {:.0f}')
                else: st.info("No households reported feed expenditure")
            else:
                st.info("Feed expenditure data not available")

    # ---------------- Offtake Analysis ----------------
    def render_offtake_analysis(self, species='sheep'):
        title_species = 'Sheep' if species.lower().startswith('sheep') else 'Goats'
        st.header(f"🚚 Offtake Analysis - {title_species}")

        if species.lower().startswith('sheep'):
            kpmd_prefix, non_kpmd_prefix = 'E1', 'E3'
        else:
            kpmd_prefix, non_kpmd_prefix = 'E2', 'E4'

        mapping = getattr(self.dp, 'offtake_col_mapping', {}) or {}
        if species.lower().startswith('sheep'):
            kpmd_sold_col = mapping.get('sheep_kpmd_sold'); non_kpmd_sold_col = mapping.get('sheep_non_kpmd_sold')
        else:
            kpmd_sold_col = mapping.get('goat_kpmd_sold'); non_kpmd_sold_col = mapping.get('goat_non_kpmd_sold')

        def _sales_cols(species, kpmd_prefix, non_kpmd_prefix):
            if species.lower().startswith('sheep'):
                price_kpmd = f"{kpmd_prefix}c. What was the average price per sheep last month?"
                price_non  = f"{non_kpmd_prefix}d. What was the average price per sheep last month?"
                age_kpmd   = f"{kpmd_prefix}d. What was the typical age in months of the sheep when sold to KPMD off-takers last month?"
                age_non    = f"{non_kpmd_prefix}e. What was the typical age in months of the sheep when sold to non-KPMD off-takers last month?"
                wt_kpmd    = f"{kpmd_prefix}f. What was the typical weight in kilos of sheep sold last month?"
                wt_non     = f"{non_kpmd_prefix}g. What was the typical weight in kilos of sheep sold last month?"
                breed_kpmd_stem = f"{kpmd_prefix}i. What breeds of sheep did you sell? [Select all that apply]"
                breed_non_stem  = f"{non_kpmd_prefix}j. What breeds of sheep did you sell? [Select all that apply]"
                buyers_non_stem = f"{non_kpmd_prefix}a. To whom did you sell sheep? [Select all that apply]"
            else:
                price_kpmd = f"{kpmd_prefix}c. What was the average price per goat last month?"
                price_non  = f"{non_kpmd_prefix}d. What was the average price per goat last month?"
                age_kpmd   = f"{kpmd_prefix}d. What was the typical age in months of the goats when sold to KPMD off-takers last month?"
                age_non    = f"{non_kpmd_prefix}e. What was the typical age in months of the goats when sold to non-KPMD off-takers last month?"
                wt_kpmd    = f"{kpmd_prefix}f. What was the typical weight in kilos of goats sold last month?"
                wt_non     = f"{non_kpmd_prefix}g. What was the typical weight in kilos of goats sold last month?"
                breed_kpmd_stem = f"{kpmd_prefix}i. What breeds of goats did you sell? [Select all that apply]"
                breed_non_stem  = f"{non_kpmd_prefix}j. What breeds of goats did you sell? [Select all that apply]"
                buyers_non_stem = f"{non_kpmd_prefix}a. To whom did you sell goats? [Select all that apply]"
            return price_kpmd, price_non, age_kpmd, age_non, wt_kpmd, wt_non, breed_kpmd_stem, breed_non_stem, buyers_non_stem

        price_kpmd_col, price_non_col, age_kpmd_col, age_non_col, wt_kpmd_col, wt_non_col, breed_kpmd_stem, breed_non_stem, buyers_non_stem = _sales_cols(species, kpmd_prefix, non_kpmd_prefix)

        tab1, tab2, tab3, tab4 = st.tabs(["Sales Volume", "Price Analysis", "Transaction Details", "Weights • Breeds • Buyers"])

        with tab1:
            st.subheader("Sales Volume Analysis")
            if kpmd_sold_col and kpmd_sold_col in self.df.columns:
                tmp = self.df.copy(); tmp['sold_kpmd'] = tmp[kpmd_sold_col].apply(yn).astype(int)
                self.create_comparison_cards(tmp, 'sold_kpmd', f'KPMD Sales Rate ({title_species})', '{:.1%}')
            else:
                st.info(f"KPMD sales data for {title_species} not available")
            if non_kpmd_sold_col and non_kpmd_sold_col in self.df.columns:
                tmp = self.df.copy(); tmp['sold_non_kpmd'] = tmp[non_kpmd_sold_col].apply(yn).astype(int)
                self.create_comparison_cards(tmp, 'sold_non_kpmd', f'Non-KPMD Sales Rate ({title_species})', '{:.1%}')
            else:
                st.info(f"Non-KPMD sales data for {title_species} not available")

        with tab2:
            st.subheader("Price Analysis")

            # Build seller flags for masks
            sold_kpmd = None
            sold_non  = None
            if kpmd_sold_col and kpmd_sold_col in self.df.columns:
                sold_kpmd = self.df[kpmd_sold_col].apply(yn).astype(int) == 1
            if non_kpmd_sold_col and non_kpmd_sold_col in self.df.columns:
                sold_non = self.df[non_kpmd_sold_col].apply(yn).astype(int) == 1

            # Subset for KPMD prices: must have non-missing price AND (if available) sold_kpmd == True
            df_kpmd = pd.DataFrame(columns=self.df.columns)
            if price_kpmd_col in self.df.columns:
                mask = self.df[price_kpmd_col].notna()
                if sold_kpmd is not None:
                    mask &= sold_kpmd
                df_kpmd = self.df.loc[mask].copy()

            # Subset for Non-KPMD prices: must have non-missing price AND (if available) sold_non == True
            df_non = pd.DataFrame(columns=self.df.columns)
            if price_non_col in self.df.columns:
                mask = self.df[price_non_col].notna()
                if sold_non is not None:
                    mask &= sold_non
                df_non = self.df.loc[mask].copy()

            # Build boxplot dataframe from the filtered subsets
            price_data = []
            if not df_kpmd.empty:
                for s in [0, 1]:
                    sub = df_kpmd[df_kpmd['kpmd_registered'] == s]
                    vals = to_num(sub[price_kpmd_col]).dropna()
                    price_data += [
                        {'Channel': 'KPMD', 'Price': v, 'KPMD_Status': 'KPMD Registered' if s == 1 else 'Non-KPMD Registered'}
                        for v in vals
                    ]
            if not df_non.empty:
                for s in [0, 1]:
                    sub = df_non[df_non['kpmd_registered'] == s]
                    vals = to_num(sub[price_non_col]).dropna()
                    price_data += [
                        {'Channel': 'Non-KPMD', 'Price': v, 'KPMD_Status': 'KPMD Registered' if s == 1 else 'Non-KPMD Registered'}
                        for v in vals
                    ]

            if price_data:
                dfp = pd.DataFrame(price_data)
                fig = px.box(dfp, x='Channel', y='Price', color='KPMD_Status',
                            title=f'{title_species} Price Distribution by Channel and KPMD Registration')
                st.plotly_chart(fig, use_container_width=True)

                # LSMeans computed ONLY on the filtered subsets
                controls = self._controls_for_lsmeans(group_col='kpmd_registered')

                if not df_kpmd.empty and price_kpmd_col in df_kpmd.columns:
                    lsm_k = lsmeans_by_group(df_kpmd.dropna(subset=[price_kpmd_col]), price_kpmd_col, 'kpmd_registered', controls)
                    if isinstance(lsm_k, dict):
                        st.caption(
                            f"KPMD price LSMean — KPMD: {lsm_k.get(1, np.nan):,.0f} | Non-KPMD: {lsm_k.get(0, np.nan):,.0f}"
                        )

                if not df_non.empty and price_non_col in df_non.columns:
                    lsm_n = lsmeans_by_group(df_non.dropna(subset=[price_non_col]), price_non_col, 'kpmd_registered', controls)
                    if isinstance(lsm_n, dict):
                        st.caption(
                            f"Non-KPMD price LSMean — KPMD: {lsm_n.get(1, np.nan):,.0f} | Non-KPMD: {lsm_n.get(0, np.nan):,.0f}"
                        )
            else:
                st.info(f"Price data for {title_species} not available")

        with tab3:
            st.subheader("Transaction Details (Age at Sale)")
            age_data = []
            if age_kpmd_col in self.df.columns:
                for s in [0,1]:
                    sub = self.df[self.df['kpmd_registered']==s]
                    vals = to_num(sub[age_kpmd_col]).dropna()
                    age_data += [{'Channel':'KPMD','Age':v,'KPMD_Status':'KPMD Registered' if s==1 else 'Non-KPMD Registered'} for v in vals]
            if age_non_col in self.df.columns:
                for s in [0,1]:
                    sub = self.df[self.df['kpmd_registered']==s]
                    vals = to_num(sub[age_non_col]).dropna()
                    age_data += [{'Channel':'Non-KPMD','Age':v,'KPMD_Status':'KPMD Registered' if s==1 else 'Non-KPMD Registered'} for v in vals]
            if age_data:
                dfp = pd.DataFrame(age_data)
                fig = px.box(dfp, x='Channel', y='Age', color='KPMD_Status',
                             title=f'{title_species} Age at Sale by Channel and KPMD Registration')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Age at sale data for {title_species} not available")

        with tab4:
            st.subheader("Weights")
            wt_rows=[]
            if wt_kpmd_col in self.df.columns:
                for s in [0,1]:
                    sub = self.df[self.df['kpmd_registered']==s]
                    w = to_num(sub[wt_kpmd_col]).dropna()
                    wt_rows += [{'Channel':'KPMD','Weight (kg)':v,'KPMD_Status':'KPMD' if s==1 else 'Non-KPMD'} for v in w]
            if wt_non_col in self.df.columns:
                for s in [0,1]:
                    sub = self.df[self.df['kpmd_registered']==s]
                    w = to_num(sub[wt_non_col]).dropna()
                    wt_rows += [{'Channel':'Non-KPMD','Weight (kg)':v,'KPMD_Status':'KPMD' if s==1 else 'Non-KPMD'} for v in w]
            if wt_rows:
                d_w = pd.DataFrame(wt_rows)
                figw = px.box(d_w, x='Channel', y='Weight (kg)', color='KPMD_Status',
                              title=f'{title_species} Typical Weights by Channel and KPMD Status')
                st.plotly_chart(figw, use_container_width=True)
            else:
                st.info("Weight columns not available")

            st.subheader("Breeds Sold")
            bkp_cols = [c for c in self.df.columns if c.startswith(breed_kpmd_stem + "/")]
            bno_cols = [c for c in self.df.columns if c.startswith(breed_non_stem + "/")]
            def _rate_table(cols, label):
                rows=[]
                for c in cols:
                    name = c.split('/')[-1]
                    rate = pd.to_numeric(self.df[c].astype(str).replace({'1':1,'0':0}), errors='coerce').fillna(0).mean() * 100
                    rows.append({'Breed': name, 'Rate': rate, 'Channel': label})
                return pd.DataFrame(rows)
            breed_df = pd.DataFrame()
            if bkp_cols: breed_df = pd.concat([breed_df, _rate_table(bkp_cols, 'KPMD')], ignore_index=True)
            if bno_cols: breed_df = pd.concat([breed_df, _rate_table(bno_cols, 'Non-KPMD')], ignore_index=True)
            if not breed_df.empty:
                figb = px.bar(breed_df, x='Breed', y='Rate', color='Channel',
                              barmode='group', title=f'{title_species} Breeds Sold by Channel (%)')
                figb.update_traces(text=breed_df['Rate'].round(1), textposition='outside')
                figb.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(figb, use_container_width=True)
            else:
                st.info("Breed selection columns not available")

            st.subheader("Non-KPMD Buyer Types")
            buyers_cols = [c for c in self.df.columns if c.startswith(buyers_non_stem + "/")]
            if buyers_cols:
                rows=[]
                for c in buyers_cols:
                    name = c.split('/')[-1]
                    rate = pd.to_numeric(self.df[c].astype(str).replace({'1':1,'0':0}), errors='coerce').fillna(0).mean() * 100
                    rows.append({'Buyer': name, 'Rate': rate})
                d_buy = pd.DataFrame(rows)
                fig_buy = px.bar(d_buy, x='Buyer', y='Rate', title='Non-KPMD Buyer Mix (%)')
                fig_buy.update_traces(text=d_buy['Rate'].round(1), textposition='outside')
                fig_buy.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig_buy, use_container_width=True)
            else:
                st.info("Buyer mix columns not available")

    # ---------------- Payments ----------------
    def render_payments(self):
        st.header("💸 Payment Methods")

        def _norm(s: str) -> str: return re.sub(r'\s+', ' ', str(s)).strip().lower()
        stems = [
            ('Sheep – KPMD',  'E1g. How were you paid by the KPMD off-takers last  month? [Select all that apply]'),
            ('Goats – KPMD',  'E2g. How were you paid by the KPMD off-takers last  month? [Select all that apply]'),
            ('Sheep – Other', 'E3h. How were you paid by the non-KPMD off-takers last  month? [Select all that apply]'),
            ('Goats – Other', 'E4h. How were you paid by the non-KPMD off-takers  last  month? [Select all that apply]'),
        ]
        rows = []; cols_norm = defaultdict(list)
        for c in self.df.columns:
            cols_norm[_norm(c)].append(c)

        for label, stem in stems:
            stem_n = _norm(stem)
            subcols = []
            for c in self.df.columns:
                c_n = _norm(c)
                if c_n.startswith(stem_n) and '/' in c: subcols.append(c)
            mobile_cols, cash_cols = [], []
            for c in subcols:
                suffix = _norm(c.split('/', 1)[1])
                if ('mobile' in suffix) or ('m-pesa' in suffix) or ('mpesa' in suffix): mobile_cols.append(c)
                if 'cash' in suffix: cash_cols.append(c)
            cands = cols_norm.get(stem_n, [])
            single_col = max(cands, key=len) if cands else None

            mobile_series = None; cash_series = None
            if mobile_cols:
                mobile_series = (self.df[mobile_cols].astype(str).replace({'1':1,'0':0})
                                 .apply(pd.to_numeric, errors='coerce').fillna(0).max(axis=1))
            if cash_cols:
                cash_series = (self.df[cash_cols].astype(str).replace({'1':1,'0':0})
                               .apply(pd.to_numeric, errors='coerce').fillna(0).max(axis=1))
            if (mobile_series is None or cash_series is None) and (single_col is not None):
                dummies = one_hot_multiselect(self.df[single_col])
                if mobile_series is None:
                    tok = next((t for t in dummies.columns if _norm(t).startswith('mobile') or 'mpesa' in _norm(t)), None)
                    mobile_series = dummies.get(tok, pd.Series(0, index=self.df.index))
                if cash_series is None:
                    tok = next((t for t in dummies.columns if _norm(t).startswith('cash')), None)
                    cash_series = dummies.get(tok, pd.Series(0, index=self.df.index))
            if mobile_series is None: mobile_series = pd.Series(0, index=self.df.index)
            if cash_series is None: cash_series = pd.Series(0, index=self.df.index)

            tmp_cols = ['kpmd_registered'] + (['County'] if 'County' in self.df.columns else [])
            tmp = self.df[tmp_cols].copy()
            tmp['block']  = label
            tmp['mobile'] = pd.to_numeric(mobile_series, errors='coerce').fillna(0).clip(0,1).astype(int)
            tmp['cash']   = pd.to_numeric(cash_series,   errors='coerce').fillna(0).clip(0,1).astype(int)
            tmp['both']   = ((tmp['mobile']==1) & (tmp['cash']==1)).astype(int)
            rows.append(tmp)

        if not rows:
            st.info("No payment method columns found")
            return

        payment = pd.concat(rows, ignore_index=True)
        grp = payment.groupby(['block','kpmd_registered'], dropna=False)
        summary = pd.DataFrame({
            'Mobile share': grp['mobile'].mean() * 100,
            'Cash share':   grp['cash'].mean() * 100,
            'Both share':   grp['both'].mean() * 100
        }).reset_index()
        summary['KPMD Status'] = summary['kpmd_registered'].map({1:'KPMD',0:'Non-KPMD'})
        long = summary.melt(id_vars=['block','KPMD Status'], value_vars=['Cash share','Mobile share','Both share'],
                            var_name='Method', value_name='Share')
        if long['Share'].sum() == 0 or long.dropna(subset=['Share']).empty:
            st.warning("No non-zero payment shares detected. Check column names in your CSV."); return
        fig = px.bar(long, x='block', y='Share', color='Method', barmode='group', facet_col='KPMD Status',
                     title='Payment method mix by channel/species and KPMD')
        fig.update_traces(text=long['Share'].round(1), textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Digital adoption by county (Mobile or Both)")
        county = payment.copy(); county['digital'] = ((county['mobile']==1) | (county['both']==1)).astype(int)
        if 'County' in county.columns:
            county_summary = county.groupby(['County','kpmd_registered'])[['digital']].mean().mul(100).reset_index()
            county_summary['KPMD Status'] = county_summary['kpmd_registered'].map({1:'KPMD',0:'Non-KPMD'})
            if county_summary['digital'].sum() == 0:
                st.info("No digital payments found in the data.")
            else:
                fig2 = px.bar(county_summary, x='County', y='digital', color='KPMD Status',
                              barmode='group', title='Digital share (%)')
                fig2.update_traces(text=county_summary['digital'].round(1), textposition='outside')
                fig2.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("County column not available for county split")

    # ---------------- County Comparator ----------------
    def render_county_compare(self):
        st.header("📊 County Comparator")
        if 'County' not in self.df.columns:
            st.info("County column missing"); return
        counties = sorted(self.df['County'].dropna().unique())
        if len(counties) < 2:
            st.info("Need at least two counties for comparison"); return
        c1, c2 = st.columns(2)
        left  = c1.selectbox("Left county", counties, key='cmpL')
        right = c2.selectbox("Right county", counties, index=1 if len(counties)>1 else 0, key='cmpR')

        def slice_county(c): return self.df[self.df['County']==c]
        metrics = [
            ('KPMD participation','kpmd_registered','{:.0%}'),
            ('Avg price sheep (KPMD)','E1c. What was the average price per sheep last month?','{:.0f}'),
            ('Avg price goats (KPMD)','E2c. What was the average price per goat last month?','{:.0f}'),
            ('Vaccination rate', self.dp.column_mapping.get('vaccination') or '', '{:.0%}'),
            ('Fodder purchase rate','B5a. Did you purchase fodder in the last 1 month?','{:.0%}')
        ]
        colA, colB = st.columns(2); A, B = slice_county(left), slice_county(right)
        for (label, col, fmt) in metrics:
            with colA:
                v = pd.to_numeric(A[col].apply(yn) if col and col in A.columns and A[col].dtype=='O' else A[col] if col and col in A.columns else pd.Series(dtype=float), errors='coerce').mean()
                st.metric(f"{label} — {left}", (fmt.format(v) if pd.notna(v) else "N/A"))
            with colB:
                v = pd.to_numeric(B[col].apply(yn) if col and col in B.columns and B[col].dtype=='O' else B[col] if col and col in B.columns else pd.Series(dtype=float), errors='coerce').mean()
                st.metric(f"{label} — {right}", (fmt.format(v) if pd.notna(v) else "N/A"))

    # ---------------- Gender Inclusion ----------------
    def render_gender_inclusion(self):
        st.header("♀️ Gender Inclusion")
        tab1, tab2, tab3 = st.tabs(["Decision Making", "KPMD Participation", "Income Control"])

        with tab1:
            st.subheader("Livestock Sale Decision Making")
            decision_col = self.dp.gender_columns.get('decision_making','')
            if decision_col and decision_col in self.df.columns:
                decision_cols = [c for c in self.df.columns if c.startswith(decision_col) and 'Other' not in c and '/' in c]
                if decision_cols:
                    rows=[]
                    for c in decision_cols:
                        role = c.split('/')[-1]
                        for s in [0,1]:
                            sub = self.df[self.df['kpmd_registered']==s]
                            rate = pd.to_numeric(sub[c].astype(str).replace({'1':1,'0':0}), errors='coerce').fillna(0).mean() * 100 if len(sub) else 0
                            rows.append({'Role': role, 'Involvement_Rate': rate, 'KPMD_Status': 'KPMD' if s==1 else 'Non-KPMD'})
                    dfp = pd.DataFrame(rows)
                    women_roles = ['Spouse','Daughter']
                    if not dfp.empty:
                        women = dfp[dfp['Role'].isin(women_roles)].groupby('KPMD_Status')['Involvement_Rate'].mean().reset_index()
                        st.write("**Women's Involvement in Decision Making**")
                        for _, r in women.iterrows():
                            st.metric(f"{r['KPMD_Status']} - Women Involvement", f"{r['Involvement_Rate']:.1f}%")
                        fig = px.bar(dfp, x='Role', y='Involvement_Rate', color='KPMD_Status',
                                     title='Decision Making Roles by KPMD Status (%)', barmode='group')
                        fig.update_traces(text=dfp['Involvement_Rate'].round(1), textposition='outside')
                        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                        st.plotly_chart(fig, use_container_width=True)
                    else: st.info("No decision making data")
                else: st.info("Decision making role columns not found")
            else: st.info("Decision making data not available")

        with tab2:
            st.subheader("KPMD Participation by Gender")
            if 'Gender' in self.df.columns and 'kpmd_registered' in self.df.columns:
                g = self.df[self.df['Gender'].notna()]
                if len(g)>0:
                    ct = pd.crosstab(g['Gender'], g['kpmd_registered'], normalize='index') * 100
                    ct = ct.reset_index().rename(columns={0:'Non-KPMD',1:'KPMD'})
                    melted = ct.melt(id_vars=['Gender'], value_vars=['KPMD','Non-KPMD'], var_name='KPMD_Status', value_name='Percentage')
                    fig = px.bar(melted, x='Gender', y='Percentage', color='KPMD_Status',
                                 title='KPMD Participation by Gender (%)', barmode='stack')
                    fig.update_traces(text=melted['Percentage'].round(1), textposition='outside')
                    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                    st.plotly_chart(fig, use_container_width=True)
                    hh_head_col = self.dp.gender_columns.get('household_head','')
                    if hh_head_col and hh_head_col in self.df.columns:
                        female_heads = self.df[(self.df['Gender']=='Female') & (self.df[hh_head_col].apply(yn)==1)]
                        if len(female_heads)>0:
                            pct = female_heads['kpmd_registered'].mean() * 100
                            st.metric("Female-Headed Households in KPMD", f"{pct:.1f}%")
                        else: st.info("No female-headed households found")
                else: st.info("No gender data available")
            else: st.info("Gender or KPMD data not available")

        with tab3:
            st.subheader("Income Control and Usage")
            income_col = self.dp.gender_columns.get('income_control','')
            if income_col and income_col in self.df.columns:
                income_cols = [c for c in self.df.columns if c.startswith(income_col) and 'Other' not in c and '/' in c]
                if income_cols:
                    rows=[]
                    for c in income_cols:
                        role = c.split('/')[-1]
                        for s in [0,1]:
                            sub = self.df[self.df['kpmd_registered']==s]
                            rate = pd.to_numeric(sub[c].astype(str).replace({'1':1,'0':0}), errors='coerce').fillna(0).mean() * 100 if len(sub) else 0
                            rows.append({'Role': role, 'Control_Rate': rate, 'KPMD_Status': 'KPMD' if s==1 else 'Non-KPMD'})
                    dfp = pd.DataFrame(rows)
                    if not dfp.empty:
                        women_roles = ['Spouse','Daughter']
                        women = dfp[dfp['Role'].isin(women_roles)].groupby('KPMD_Status')['Control_Rate'].mean().reset_index()
                        st.write("**Women's Control Over Livestock Income**")
                        for _, r in women.iterrows():
                            st.metric(f"{r['KPMD_Status']} - Women Control", f"{r['Control_Rate']:.1f}%")
                        fig = px.bar(dfp, x='Role', y='Control_Rate', color='KPMD_Status',
                                     title='Income Control Roles by KPMD Status (%)', barmode='group')
                        fig.update_traces(text=dfp['Control_Rate'].round(1), textposition='outside')
                        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                        st.plotly_chart(fig, use_container_width=True)
                    else: st.info("No income control data")
                else: st.info("Income control role columns not found")
            else: st.info("Income control data not available")

    # ---------------- Climate Impact ----------------
    def render_climate_impact(self):
        st.header("🌦️ Climate Impact")
        tab1, tab2, tab3 = st.tabs(["Adaptation Measures", "Barriers to Adaptation", "Climate Resilience"])

        with tab1:
            st.subheader("Adaptation Measures")
            j1 = self.dp.column_mapping.get('adaptation_measures')
            if j1 and j1 in self.df.columns:
                tmp = self.df.copy(); tmp['adapted'] = tmp[j1].apply(yn).astype(int)
                self.create_comparison_cards(tmp, 'adapted', 'Adaptation Rate', '{:.1%}')
            else:
                st.info("Climate adaptation data (J1) not available")

            j2_stem = 'J2. Which adapatations measures are you using?'
            strategy_cols = [c for c in self.df.columns if c.startswith(j2_stem + '/') and 'Other' not in c]
            if strategy_cols:
                rows=[]
                for c in strategy_cols:
                    name = c.split('/')[-1]
                    for s in [0,1]:
                        sub = self.df[self.df['kpmd_registered']==s]
                        rate = pd.to_numeric(sub[c].astype(str).replace({'1':1,'0':0}), errors='coerce').fillna(0).mean() * 100 if len(sub) else 0
                        rows.append({'Strategy': name, 'Usage_Rate': rate, 'KPMD_Status': 'KPMD' if s==1 else 'Non-KPMD'})
                dfp = pd.DataFrame(rows)
                fig = px.bar(dfp, x='Strategy', y='Usage_Rate', color='KPMD_Status',
                             title='Adaptation Strategies by KPMD Status (%)', barmode='group')
                fig.update_traces(text=dfp['Usage_Rate'].round(1), textposition='outside')
                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig, use_container_width=True)
            elif j2_stem in self.df.columns:
                dummies = one_hot_multiselect(self.df[j2_stem])
                if not dummies.empty:
                    tmp = pd.concat([self.df[['kpmd_registered']], dummies], axis=1)
                    long = tmp.melt(id_vars=['kpmd_registered'], var_name='Strategy', value_name='flag')
                    agg = long.groupby(['Strategy','kpmd_registered'])['flag'].mean().mul(100).reset_index()
                    agg['KPMD_Status'] = agg['kpmd_registered'].map({1:'KPMD',0:'Non-KPMD'})
                    fig = px.bar(agg, x='Strategy', y='flag', color='KPMD_Status',
                                 title='Adaptation Strategies by KPMD Status (%)', barmode='group')
                    fig.update_yaxes(title='Usage_Rate')
                    fig.update_traces(text=agg['flag'].round(1), textposition='outside')
                    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Adaptation strategy data (J2) not available")
            else:
                st.info("Adaptation strategy data (J2) not available")

        with tab2:
            st.subheader("Barriers to Adaptation")
            j3_stem = 'J3. Why not?'
            base = self.df
            if self.dp.column_mapping.get('adaptation_measures') and self.dp.column_mapping['adaptation_measures'] in self.df.columns:
                base = base[base[self.dp.column_mapping['adaptation_measures']].apply(yn) == 0]
            barrier_cols = [c for c in base.columns if c.startswith(j3_stem + '/') and 'Other' not in c]
            if barrier_cols:
                rows=[]
                for c in barrier_cols:
                    name = c.split('/')[-1]
                    for s in [0,1]:
                        sub = base[base['kpmd_registered']==s]
                        rate = pd.to_numeric(sub[c].astype(str).replace({'1':1,'0':0}), errors='coerce').fillna(0).mean() * 100 if len(sub) else 0
                        rows.append({'Barrier': name, 'Rate': rate, 'KPMD_Status': 'KPMD' if s==1 else 'Non-KPMD'})
                dfp = pd.DataFrame(rows)
                fig = px.bar(dfp, x='Barrier', y='Rate', color='KPMD_Status',
                             title='Barriers to Adaptation by KPMD Status (%)', barmode='group')
                fig.update_traces(text=dfp['Rate'].round(1), textposition='outside')
                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig, use_container_width=True)
            elif j3_stem in base.columns:
                dummies = one_hot_multiselect(base[j3_stem])
                if not dummies.empty:
                    tmp = pd.concat([base[['kpmd_registered']], dummies], axis=1)
                    long = tmp.melt(id_vars=['kpmd_registered'], var_name='Barrier', value_name='flag')
                    agg = long.groupby(['Barrier','kpmd_registered'])['flag'].mean().mul(100).reset_index()
                    agg['KPMD_Status'] = agg['kpmd_registered'].map({1:'KPMD',0:'Non-KPMD'})
                    fig = px.bar(agg, x='Barrier', y='flag', color='KPMD_Status',
                                 title='Barriers to Adaptation by KPMD Status (%)', barmode='group')
                    fig.update_yaxes(title='Rate')
                    fig.update_traces(text=agg['flag'].round(1), textposition='outside')
                    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No barrier data available for non-adapting households")
            else:
                st.info("Barriers to adaptation data (J3) not available")

        with tab3:
            st.subheader("Climate Resilience Indicators")
            if 'kpmd_registered' in self.df.columns:
                kpmd_participation = self.df['kpmd_registered'].mean() * 100
                st.metric("Overall KPMD Participation Rate", f"{kpmd_participation:.1f}%")
            if 'resilience_score' in self.df.columns:
                resilience_data = self.df[self.df['resilience_score'].notna()]
                if len(resilience_data)>0:
                    self.create_comparison_cards(resilience_data, 'resilience_score', 'Resilience Score', '{:.1f}')
                else:
                    st.info("No resilience score data available")
            if 'adaptation_score' in self.df.columns:
                st.metric("Households Implementing Adaptation", f"{(self.df['adaptation_score'].mean()*100):.1f}%")
            if 'total_sr' in self.df.columns and 'Feed_Expenditure' in self.df.columns:
                large_herds = (self.df['total_sr'] > self.df['total_sr'].median()).mean() * 100
                st.metric("Households with Above-Median Herd Size", f"{large_herds:.1f}%")
            st.info("Additional climate resilience indicators will be displayed as data becomes available")

    # ---------------- Food Security (rCSI) ----------------
    def render_food_security(self):
        st.header("🍚 Food Security — Reduced Coping Strategies Index (30-day)")

        # Top summary metrics
        col1, col2, col3 = st.columns(3)
        if 'rcsi_30' in self.df.columns:
            col1.metric("Average rCSI-30", f"{self.df['rcsi_30'].mean():.1f}")
        worry_col = 'food_worry' if 'food_worry' in self.df.columns else None
        if worry_col:
            col2.metric("Households Worried about Food", f"{(self.df[worry_col].mean()*100):.1f}%")
        if 'insured_sr' in self.df.columns:
            col3.metric("Avg. # SR Insured", f"{self.df['insured_sr'].mean():.1f}")

        # Overall rCSI distribution
        if 'rcsi_30' in self.df.columns:
            fig = px.histogram(self.df, x='rcsi_30', nbins=30, title='Distribution of rCSI (30 days)')
            st.plotly_chart(fig, use_container_width=True)

        # ---------------- rCSI by Registration (30-day) ---------------- 
        st.subheader("🍚 Food Security — Reduced Coping Strategies Index (30-day)")

        if 'rcsi_30' in self.df.columns and 'kpmd_registered' in self.df.columns:
            df_rcsi = self.df[['rcsi_30', 'kpmd_registered']].copy()
            df_rcsi = df_rcsi[df_rcsi['rcsi_30'].notna()]
            if len(df_rcsi) == 0:
                st.info("No rCSI values available.")
                return

            # Add clean categorical for legend/axis
            df_rcsi = self._with_kpmd_status(df_rcsi)

            # Summary cards
            colA, colB = st.columns(2)
            with colA:
                m = df_rcsi[df_rcsi['KPMD Status'] == 'KPMD']['rcsi_30'].mean()
                st.metric("KPMD — Avg rCSI (30d)", f"{m:.1f}" if pd.notna(m) else "N/A")
            with colB:
                m = df_rcsi[df_rcsi['KPMD Status'] == 'Non-KPMD']['rcsi_30'].mean()
                st.metric("Non-KPMD — Avg rCSI (30d)", f"{m:.1f}" if pd.notna(m) else "N/A")

            # Box plot with ordered categories
            fig2 = px.box(
                df_rcsi,
                x='KPMD Status',
                y='rcsi_30',
                color='KPMD Status',
                category_orders={'KPMD Status': ['Non-KPMD', 'KPMD']},
                labels={'KPMD Status': 'Registration', 'rcsi_30': 'rCSI (30-day)'},
                title='rCSI by Registration'
            )
            fig2.update_layout(legend_title_text='Registration')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Missing rCSI or registration fields to plot rCSI by Registration.")

# -------------------------------------------------
# Main App
# -------------------------------------------------
def main():
    st.title("APMT Project Insights")
    st.markdown('<div class="main-header">Pastoral Market Transformation Monitoring</div>', unsafe_allow_html=True)

    # --- Auto-load dataset (no uploader) ---
    st.sidebar.header("Data Source")
    st.sidebar.write("Auto-loaded file:")
    st.sidebar.code(os.path.abspath(DATA_PATH))

    if st.sidebar.button("Reload data"):
        load_apmt_csv.clear()
        st.sidebar.success("Cache cleared. Reloading…")
        st.rerun()

    try:
        df = load_apmt_csv(DATA_PATH)
        st.success(f"Data loaded successfully: {len(df):,} records")

        # Ensure the GeoJSON base maps are present (downloads once if missing)
        with st.spinner("Preparing base maps…"):
            ok_geo = ensure_geo_assets()
        if not ok_geo:
            st.warning("Base maps unavailable — county/sub-county outlines will be hidden.")

        with st.expander("Data Preview", expanded=False):

            st.write(f"Columns detected ({len(df.columns)}):")
            st.write(list(df.columns))
            st.write(f"Total records: {len(df)}")
            st.dataframe(df.head(10))

        # Build processor/renderer on the loaded df
        processor = APMTDataProcessor(df)
        renderer = DashboardRenderer(processor)

        # NEW: Data Cleaning & Quality section (merged & folded)
        render_data_quality_section(processor.df, processor.dq_issues)

        # ---------- Sidebar: FILTERS ----------
        st.sidebar.header("Global Filters")
        # (kept exactly as before – “Select Here” expander + cascading filters)

        # --- Helper: compute date bounds from the CURRENT (unfiltered) df ---
        def _compute_date_bounds(df_for_bounds: pd.DataFrame):
            for cand in ['int_date_std', '_submission_time', 'start', 'end']:
                if cand in df_for_bounds.columns:
                    s = pd.to_datetime(df_for_bounds[cand], errors='coerce')
                    if s.notna().any():
                        return (s.min().date(), s.max().date(), cand)
            return (datetime(2024, 1, 1).date(), datetime.today().date(), None)

        _min_date, _max_date, _date_col_found = _compute_date_bounds(df)
        if _min_date > _max_date:
            _min_date, _max_date = _max_date, _min_date

        head_hash = pd.util.hash_pandas_object(df.head(10), index=False).sum() if len(df) else 0
        tail_hash = pd.util.hash_pandas_object(df.tail(10), index=False).sum() if len(df) else 0
        data_sig = (len(df), str(_min_date), str(_max_date), int(head_hash), int(tail_hash))

        if st.session_state.get('data_sig') != data_sig:
            st.session_state['data_sig'] = data_sig
            if not st.session_state.get('date_range_is_custom', False):
                st.session_state['date_range'] = (_min_date, _max_date)

        if 'date_range' not in st.session_state:
            st.session_state['date_range'] = (_min_date, _max_date)

        def _clamp_to_bounds(dr_tuple):
            try:
                a, b = dr_tuple
                a = max(_min_date, a)
                b = min(_max_date, b)
                if a > b:
                    a, b = _min_date, _max_date
                return (a, b)
            except Exception:
                return (_min_date, _max_date)

        st.session_state['date_range'] = _clamp_to_bounds(st.session_state['date_range'])

        def _get_state(k, default):
            return st.session_state.get(k, default)

        county_default = 'All'
        subcounty_default = 'All'
        kpmd_default = 'All'
        gender_default = 'All'
        date_default = (_min_date, _max_date)

        filters_active = (
            _get_state('county', county_default) != county_default or
            _get_state('subcounty', subcounty_default) != subcounty_default or
            _get_state('kpmd_filter', kpmd_default) != kpmd_default or
            _get_state('gender', gender_default) != gender_default or
            (
                isinstance(_get_state('date_range', date_default), tuple) and
                _get_state('date_range', date_default) != date_default
            )
        )

        with st.sidebar.expander("Select Here", expanded=filters_active):
            # ----- County → Sub-County (cascading) -----
            if 'County' in processor.df.columns:
                counties = ['All'] + sorted(processor.df['County'].dropna().unique())
                selected_county = st.selectbox("Select County", counties, key="county")

                if selected_county != 'All':
                    processor.df = processor.df[processor.df['County'] == selected_county]

                    sub_col = coalesce_first(
                        df,
                        ['Sub County', 'Sub-County', 'Subcounty', 'Sub-county', 'SubCounty', 'Sub county']
                    )
                    if sub_col and sub_col in df.columns:
                        sub_opts = ['All'] + sorted(
                            df.loc[df['County'] == selected_county, sub_col].dropna().unique()
                        )
                        selected_sub = st.selectbox("Select Sub-County", sub_opts, key="subcounty")
                        if selected_sub != 'All':
                            processor.df = processor.df[processor.df[sub_col] == selected_sub]
            else:
                selected_county = 'All'

            # ----- KPMD status -----
            kpmd_filter = st.selectbox("KPMD Status", ['All', 'Registered', 'Not Registered'], key="kpmd_filter")
            if kpmd_filter == 'Registered':
                processor.df = processor.df[processor.df['kpmd_registered'] == 1]
            elif kpmd_filter == 'Not Registered':
                processor.df = processor.df[processor.df['kpmd_registered'] == 0]

            # ----- Gender -----
            if 'Gender' in processor.df.columns:
                genders = ['All'] + sorted(processor.df['Gender'].dropna().unique())
                selected_gender = st.selectbox("Select Gender", genders, key="gender")
                if selected_gender != 'All':
                    processor.df = processor.df[processor.df['Gender'] == selected_gender]

            # ----- Date range -----
            def _mark_date_custom():
                st.session_state['date_range_is_custom'] = True

            if _date_col_found is not None:
                cols_reset = st.columns([1, 1.2, 2])
                with cols_reset[0]:
                    if st.button("Reset dates"):
                        st.session_state['date_range'] = (_min_date, _max_date)
                        st.session_state['date_range_is_custom'] = False

                date_range = st.date_input(
                    "Select Date Range",
                    value=st.session_state['date_range'],
                    min_value=_min_date,
                    max_value=_max_date,
                    key="date_range",
                    on_change=_mark_date_custom
                )

                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                    processor.df = processor.df[
                        (pd.to_datetime(processor.df[_date_col_found], errors='coerce') >= start) &
                        (pd.to_datetime(processor.df[_date_col_found], errors='coerce') <= end)
                    ]
            else:
                st.info("No time information available for date filtering.")

        # ---------- Sidebar: NAVIGATION ----------
        st.sidebar.markdown(
            '<div style="color:#dc3545; font-weight:700; font-size:1rem; margin-bottom:0.25rem;">'
            'Navigate Here <span style="font-size:1.1rem; line-height:1;">👇</span>'
            '</div>',
            unsafe_allow_html=True
        )

        # Main pages list (P&L Analysis is intentionally NOT included here)
        MAIN_PAGES = [
            "Field Outlook",
            "Pastoral Productivity",
            "Pastoral Livelihoods",
            "Feed & Fodder",
            "Sheep Offtake",
            "Goat Offtake",
            "Payments",
            "County Comparator",
            "Gender Inclusion",
            "Climate Impact",
            "KPMD Participation",
            "Food Security (rCSI – 30d)",
        ]

        # Defaults
        if "nav_page" not in st.session_state:
            st.session_state["nav_page"] = "Field Outlook"
        if "nav_page_radio" not in st.session_state:
            st.session_state["nav_page_radio"] = "Field Outlook"

        # Radio for the standard pages (excludes P&L Analysis)
        selected_from_radio = st.sidebar.radio(
            "Select Dashboard Page",
            MAIN_PAGES,
            key="nav_page_radio"
        )

        # If user is not currently on P&L Analysis, keep nav_page in sync with the radio
        if st.session_state.get("nav_page") != "P&L Analysis":
            st.session_state["nav_page"] = selected_from_radio

        # Separate, top-level entry for P&L Analysis (same level as the radio label)
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 💰 P&L Analysis")
        if st.sidebar.button("Open P&L Analysis", use_container_width=True):
            st.session_state["nav_page"] = "P&L Analysis"

        # Optional: while on P&L Analysis, show a quick way back to the radio pages
        if st.session_state.get("nav_page") == "P&L Analysis":
            def _back_to_pages():
                st.session_state["nav_page"] = st.session_state.get("nav_page_radio", MAIN_PAGES[0])
            st.sidebar.button("← Back to pages", on_click=_back_to_pages, use_container_width=True)

        # ---- Render the chosen page ----
        page = st.session_state["nav_page"]

        if page == "Field Outlook":
            renderer.render_field_outlook()
        elif page == "Pastoral Productivity":
            renderer.render_pastoral_productivity()
        elif page == "Pastoral Livelihoods":
            renderer.render_pastoral_livelihoods()
        elif page == "Feed & Fodder":
            renderer.render_feed_fodder()
        elif page == "Sheep Offtake":
            renderer.render_offtake_analysis('sheep')
        elif page == "Goat Offtake":
            renderer.render_offtake_analysis('goats')
        elif page == "Payments":
            renderer.render_payments()
        elif page == "County Comparator":
            renderer.render_county_compare()
        elif page == "Gender Inclusion":
            renderer.render_gender_inclusion()
        elif page == "Climate Impact":
            renderer.render_climate_impact()
        elif page == "KPMD Participation":
            renderer.render_kpmd_participation()
        elif page == "Food Security (rCSI – 30d)":
            renderer.render_food_security()
        elif page == "P&L Analysis":
            renderer.render_pl_analysis()

    except FileNotFoundError:
        st.error("The specified data file was not found.")
        st.code(DATA_PATH)
        st.info("Please check the path or ensure the file exists at this location.")
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.info("Please check that your CSV file matches the expected APMT data format")

if __name__ == "__main__":
    main()