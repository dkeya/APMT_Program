# app.py — APMT Longitudinal Dashboard (auto-load + fixes)
# -----------------------------------------------------------------------------------------
# Changes in this version:
# - Auto-loads CSV from the provided path (no file uploader).
# - Adds "Reload data" button (clears cache and reruns).
# - Fixes coalesce_first helper.
# - Keeps the pydeck map style and previous chart ordering (county/KPMD chart between
#   Monthly Submissions and the Household Locations map).

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pydeck as pdk
import re
import io
import os
import warnings

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
    .kpmd-card {
        background-color: #e8f4fd;
        border-left: 4px solid #1f77b4;
    }
    .non-kpmd-card {
        background-color: #fde8e8;
        border-left: 4px solid #ff6b6b;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
    }
    .profit-positive {
        color: #28a745;
        font-weight: bold;
    }
    .profit-negative {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
_YES = {'yes','y','true','1','t','aye','yeah'}
_NO  = {'no','n','false','0','f','nah'}

def yn(x):
    """
    Robust yes/no to 1/0.
    Accepts strings (Yes/No variations), booleans, numeric (1/0), and NaN.
    Anything non-yes returns 0.
    """
    if pd.isna(x):
        return 0
    if isinstance(x, (int, float, np.integer, np.floating)):
        return 1 if float(x) == 1 else 0
    if isinstance(x, bool):
        return 1 if x else 0
    s = str(x).strip().lower()
    if s in _YES:
        return 1
    if s in _NO:
        return 0
    if s.startswith('yes'):
        return 1
    if s.startswith('no'):
        return 0
    return 0

def to_num(series):
    """Coerce to numeric safely, stripping commas and spaces."""
    return pd.to_numeric(series.astype(str).str.replace(',', '').str.strip(), errors='coerce')

def one_hot_multiselect(series: pd.Series) -> pd.DataFrame:
    """
    Robustly one-hot a single multi-select text column.
    Splits on | ; , / or 2+ spaces. Trims whitespace. Ignores empty tokens.
    Returns int dummies (0/1).
    """
    if series.dropna().empty:
        return pd.DataFrame(index=series.index)

    tokens_list = []
    pattern = re.compile(r'\s*\|\s*|\s*;\s*|\s*,\s*|\s*/\s*|\s{2,}')
    for val in series.fillna(''):
        if not isinstance(val, str):
            tokens_list.append([])
            continue
        tokens = [t.strip() for t in pattern.split(val) if t.strip() != '']
        tokens_list.append(tokens)

    uniques = sorted({tok for toks in tokens_list for tok in toks})
    if not uniques:
        return pd.DataFrame(index=series.index)

    data = {tok: [1 if tok in toks else 0 for toks in tokens_list] for tok in uniques}
    return pd.DataFrame(data, index=series.index).astype(int)

def coalesce_first(df, candidates):
    """Return the first existing column name from candidates, else None."""
    if not isinstance(df, pd.DataFrame):
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -------------------------------------------------
# Data Loading (Auto)
# -------------------------------------------------
DATA_PATH = r"C:\Users\dkeya\Documents\SBS\2025\APMT\APMT_Longitudinal_Survey.csv"

@st.cache_data(ttl=900, show_spinner=False)
def load_apmt_csv(path: str) -> pd.DataFrame:
    """
    Robust CSV loader with multiple encodings and fallback character replacement.
    Auto-detects delimiter when possible.
    """
    encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1', 'ISO-8859-1', 'windows-1252']
    # Try straightforward reads first
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            raise
        except Exception:
            continue
    # Try python engine with sep=None (delimiter sniff)
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine='python')
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            raise
        except Exception:
            continue
    # Final fallback with character replacement
    try:
        return pd.read_csv(path, encoding='utf-8', errors='replace')
    except FileNotFoundError:
        raise
    except Exception:
        # As last resort try latin-1 with replacement
        return pd.read_csv(path, encoding='latin-1', errors='replace')

# -------------------------------------------------
# Data Processor
# -------------------------------------------------
class APMTDataProcessor:
    def __init__(self, df):
        self.df = df.copy()
        self._basic_cleanups()
        self.column_mapping = self._build_column_mapping()
        self.enhanced_standardize_data()  # Use enhanced version

    # ---------- basic cleanup before mapping ----------
    def _basic_cleanups(self):
        # Treat likely ID/phone-like fields as strings (preserve leading zeros)
        for col in self.df.columns:
            if any(k in col.lower() for k in ['phone', 'telephone', 'household id', '_id', '_uuid']):
                self.df[col] = self.df[col].astype(str)

        # Strip column whitespace and normalize accidental double spaces
        self.df.columns = [re.sub(r'\s+', ' ', c).strip() for c in self.df.columns]

    # ---------- flexible mapping ----------
    def _build_column_mapping(self):
        mapping = {}
        # Core identifiers
        mapping['county'] = coalesce_first(self.df, ['County', 'county', 'COUNTY'])
        mapping['gender'] = coalesce_first(self.df, ['Gender', 'gender', 'GENDER', 'Select respondent name'])
        mapping['kpmd_registration'] = coalesce_first(self.df, [
            'A8. Are you registered to KPMD programs?',
            'KPMD registration',
            'Registered to KPMD'
        ])
        mapping['household_type'] = coalesce_first(self.df, [
            'Selection of the household',
            'Household type',
            'Treatment/Control'
        ])

        # GPS columns (prefer canonical underscored forms from sample)
        mapping['gps_lat'] = coalesce_first(self.df, ['_GPS Coordinates_latitude', 'GPS Latitude', 'Latitude'])
        mapping['gps_lon'] = coalesce_first(self.df, ['_GPS Coordinates_longitude', 'GPS Longitude', 'Longitude'])

        # Herd composition
        mapping['herd_sheep'] = self._find_columns_pattern(r'C3\..*(Ewes|Rams)|C3\..*sheep')
        mapping['herd_goats'] = self._find_columns_pattern(r'C3\..*(Does|Bucks)|C3\..*goat')
        mapping['births'] = self._find_columns_pattern(r'C4\..*(born|born in the last)')
        mapping['deaths'] = self._find_columns_pattern(r'C5\..*(died|death)')
        mapping['losses'] = self._find_columns_pattern(r'C6\..*(lost|not found|wild)')

        # Animal health
        mapping['vaccination'] = coalesce_first(self.df, [
            'D1. Did you vaccinate your small ruminants in the last month?',
            'D1. Did you vaccinate your small ruminants livestock in the last month?',
            'D1. Did you vaccinate your small ruminants livestock last month?'
        ])
        mapping['vaccination_diseases'] = self._find_columns_pattern(r'D1c\..*vaccinate')
        mapping['treatment_diseases'] = self._find_columns_pattern(r'D3c\..*(treat|disease)')

        # Feed and fodder
        mapping['fodder_purchase'] = coalesce_first(self.df, [
            'B5a. Did you purchase fodder in the last 1 month?'
        ])
        mapping['feed_sources'] = self._find_columns_pattern(r'B5b\..*buy feeds')

        # Offtake patterns
        mapping['sheep_kpmd_sales'] = self._find_columns_pattern(r'^E1\..*sheep.*KPMD|^E1\.$')
        mapping['goat_kpmd_sales'] = self._find_columns_pattern(r'^E2\..*goat.*KPMD|^E2\.$')
        mapping['sheep_non_kpmd_sales'] = self._find_columns_pattern(r'^E3\..*sheep|^E3\.$')
        mapping['goat_non_kpmd_sales'] = self._find_columns_pattern(r'^E4\..*goat|^E4\.$')

        # Gender & decision
        mapping['decision_making'] = coalesce_first(self.df, [
            'G1.Who in the household makes the decision for livestock sale?  [Select all that apply]',
            'G1.Who in the household makes the decision for livestock sale? [Select all that apply]'
        ])
        mapping['income_control'] = coalesce_first(self.df, [
            'G2. Who in the household uses the income from the livestock sale? [Select all that apply]'
        ])

        # Climate adaptation
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

    # ---------- standardization & feature engineering ----------
    def enhanced_standardize_data(self):
        try:
            # 1) Robust date parsing
            def _coerce_date(s):
                return pd.to_datetime(s, errors='coerce', dayfirst=True, infer_datetime_format=True)

            date_candidates = [c for c in ['int_date', '_submission_time', 'start', 'end'] if c in self.df.columns]
            date_parsed = False
            for c in date_candidates:
                parsed = _coerce_date(self.df[c])
                if parsed.notna().any():
                    self.df['int_date_std'] = parsed
                    self.df.loc[parsed.notna(), 'month'] = self.df.loc[parsed.notna(), 'int_date_std'].dt.to_period('M').astype(str)
                    self.df.loc[parsed.notna(), 'year'] = self.df.loc[parsed.notna(), 'int_date_std'].dt.year
                    date_parsed = True
                    break
            if not date_parsed:
                self.df['month'] = [f"2024-{i:02d}" for i in range(1, min(len(self.df) + 1, 13))]

            # 2) KPMD registration flag
            kpmd_col = self.column_mapping['kpmd_registration']
            if kpmd_col:
                self.df['kpmd_registered'] = self.df[kpmd_col].apply(yn).astype(int)
            else:
                self.df['kpmd_registered'] = 0

            # 3) Household treatment flag
            arm_col = self.column_mapping['household_type']
            if arm_col:
                self.df['is_treatment'] = self.df[arm_col].astype(str).str.contains('Treatment', case=False, na=False).astype(int)
            else:
                self.df['is_treatment'] = 0

            # 4) Boolean coercion families
            bool_patterns = [
                r'^C1\.', r'^C2\.', r'^D1\..*vaccinate', r'^D3\..*treat', r'^D4\..*deworm',
                r'^B5a\.', r'^B6a\.', r'^J1\.'
            ]
            for pat in bool_patterns:
                for col in self._find_columns_pattern(pat):
                    self.df[col] = self.df[col].apply(yn).astype(int)

            # 5) Numeric coercion
            numeric_patterns = [
                r'B3b.*cost.*herding', r'B4b\..*cost', r'B5c\..*price.*bale', r'B5d\..*Number.*bales.*purchased',
                r'B6b\..*Quantity.*harvested', r'B6d\..*price.*sell', r'B6e\..*Number.*bales.*sold',
                r'D1a\..*vaccinated', r'D1b\..*cost.*vaccination', r'D3a\..*sick.*treated', r'D3b\..*cost.*treatment',
                r'D4a\..*cost.*deworming',
                r'^E[1-4][chi]\..*price|^E[1-4][bh]\..*(How many|times)|^E[1-4]f\..*weight|^E[1-4]h\..*transport'
            ]
            for pat in numeric_patterns:
                for col in self._find_columns_pattern(pat):
                    self.df[col] = to_num(self.df[col]).fillna(0)

            # 6) Disease mapping
            self.enhanced_disease_mapping()

            # 7) Feed expenditure
            self.enhanced_feed_calculation()

            # 8) Offtake mapping
            self.enhanced_offtake_mapping()

            # 9) Gender mapping
            self.enhanced_gender_mapping()

            # 10) Climate resilience
            self.calculate_climate_resilience()

        except Exception as e:
            st.warning(f"Some data standardization issues occurred: {str(e)}")
            if 'month' not in self.df.columns:
                self.df['month'] = [f"2024-{i:02d}" for i in range(1, min(len(self.df) + 1, 13))]
            if 'kpmd_registered' not in self.df.columns:
                self.df['kpmd_registered'] = 0
            if 'is_treatment' not in self.df.columns:
                self.df['is_treatment'] = 0

    def enhanced_disease_mapping(self):
        self.vacc_disease_cols = [c for c in self.df.columns if c.startswith('D1c. ') and '/' in c]
        self.treat_disease_cols = [c for c in self.df.columns if c.startswith('D3c. ') and '/' in c]
        for col in self.vacc_disease_cols + self.treat_disease_cols:
            self.df[col] = pd.to_numeric(self.df[col].astype(str).replace({'1': 1, '0': 0}), errors='coerce').fillna(0).astype(int)

    def enhanced_feed_calculation(self):
        price_col = coalesce_first(self.df, ['B5c. What was the price per 15 kg bale in the last 1 month?'])
        qty_col = coalesce_first(self.df, ['B5d. Number of 15 kg bales purchased in the last 1 month?'])
        if price_col and qty_col:
            self.df['Feed_Expenditure'] = to_num(self.df[price_col]).fillna(0) * to_num(self.df[qty_col]).fillna(0)
        else:
            self.df['Feed_Expenditure'] = 0

    def enhanced_offtake_mapping(self):
        self.offtake_col_mapping = {}
        sheep_kpmd = coalesce_first(self.df, ['E1. Did you sell sheep to KPMD off-takers last  month?'])
        if not sheep_kpmd:
            sheep_kpmd = coalesce_first(self.df, [c for c in self._find_columns_pattern(r'^E1\..*sell sheep')])
        if sheep_kpmd:
            self.offtake_col_mapping['sheep_kpmd_sold'] = sheep_kpmd

        sheep_non = coalesce_first(self.df, ['E3. Did you sell sheep to non-KPMD off-takers last  month?'])
        if not sheep_non:
            sheep_non = coalesce_first(self.df, [c for c in self._find_columns_pattern(r'^E3\..*sell sheep')])
        if sheep_non:
            self.offtake_col_mapping['sheep_non_kpmd_sold'] = sheep_non

        goat_kpmd = coalesce_first(self.df, ['E2. Did you sell goats to KPMD off-takers last  month?'])
        if not goat_kpmd:
            goat_kpmd = coalesce_first(self.df, [c for c in self._find_columns_pattern(r'^E2\..*sell goats')])
        if goat_kpmd:
            self.offtake_col_mapping['goat_kpmd_sold'] = goat_kpmd

        goat_non = coalesce_first(self.df, ['E4. Did you sell goats to non-KPMD off-takers last  month?'])
        if not goat_non:
            goat_non = coalesce_first(self.df, [c for c in self._find_columns_pattern(r'^E4\..*sell goats')])
        if goat_non:
            self.offtake_col_mapping['goat_non_kpmd_sold'] = goat_non

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
        if a_col:
            self.df['adaptation_score'] = self.df[a_col].apply(yn)
        if 'kpmd_registered' in self.df.columns:
            self.df['kpmd_resilience_bonus'] = self.df['kpmd_registered'] * 0.5
        components = [c for c in ['adaptation_score', 'kpmd_resilience_bonus'] if c in self.df.columns]
        if components:
            self.df['resilience_score'] = self.df[components].sum(axis=1)

    def calculate_herd_metrics(self):
        try:
            for col in ['total_sheep','total_goats','total_sr','pct_female',
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

            if sheep_cols:
                self.df['total_sheep'] = self.df[sheep_cols].sum(axis=1)
            if goat_cols:
                self.df['total_goats'] = self.df[goat_cols].sum(axis=1)
            self.df['total_sr'] = self.df['total_sheep'] + self.df['total_goats']

            female_sheep_col = 'C3. Number of Ewes currently owned (total: at home + away + relatives/friends)'
            female_goat_col = 'C3. Number of Does currently owned (total: at home + away + relatives/friends)'
            female_sheep = to_num(self.df[female_sheep_col]).fillna(0) if female_sheep_col in self.df.columns else 0
            female_goats = to_num(self.df[female_goat_col]).fillna(0) if female_goat_col in self.df.columns else 0
            total_female = female_sheep + female_goats

            valid = (self.df['total_sr'] > 0) & (pd.to_numeric(total_female, errors='coerce') >= 0)
            self.df.loc[valid, 'pct_female'] = (total_female[valid] / self.df.loc[valid, 'total_sr'] * 100)
            self.df.loc[~valid, 'pct_female'] = 0
            self.df['pct_female'] = self.df['pct_female'].clip(0, 100)

            def existing(cols):
                return [c for c in cols if c in self.df.columns]

            birth_cols = existing([
                'C4. Number of Rams born in the last 1 month',
                'C4. Number of Ewes born in the last 1 month',
                'C4. Number of Bucks born in the last 1 month',
                'C4. Number of Does born in the last 1 month'
            ])
            mort_cols = existing([
                'C5. Number of Rams that died in the last 1 month',
                'C5. Number of Ewes that died in the last 1 month',
                'C5. Number of Bucks that died in the last 1 month',
                'C5. Number of Does that died in the last 1 month'
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

            valid = self.df['total_sr'] > 0
            self.df.loc[valid, 'birth_rate_per_100'] = self.df.loc[valid, 'total_births'] / self.df.loc[valid, 'total_sr'] * 100
            self.df.loc[valid, 'mortality_rate_per_100'] = self.df.loc[valid, 'total_mortality'] / self.df.loc[valid, 'total_sr'] * 100
            self.df.loc[valid, 'loss_rate_per_100'] = self.df.loc[valid, 'total_losses'] / self.df.loc[valid, 'total_sr'] * 100
            self.df.loc[~valid, ['birth_rate_per_100','mortality_rate_per_100','loss_rate_per_100']] = 0

        except Exception as e:
            st.warning(f"Some herd metrics could not be calculated: {str(e)}")
            for col in ['total_sheep','total_goats','total_sr','pct_female',
                        'total_births','total_mortality','total_losses',
                        'birth_rate_per_100','mortality_rate_per_100','loss_rate_per_100']:
                if col not in self.df.columns: self.df[col] = 0.0

    def calculate_pl_metrics(self):
        """Calculate Profit & Loss metrics for each household"""
        try:
            self.df['total_revenue'] = 0.0
            self.df['total_costs'] = 0.0
            self.df['net_profit'] = 0.0
            self.df['profit_margin'] = 0.0

            revenue_components = []

            # KPMD (E1/E2)
            if all(c in self.df.columns for c in ['E1a. How many sheep did you sell to KPMD off-takers  last month?', 'E1c. What was the average price per sheep last month?']):
                self.df['sheep_kpmd_revenue'] = to_num(self.df['E1a. How many sheep did you sell to KPMD off-takers  last month?']).fillna(0) * \
                                                to_num(self.df['E1c. What was the average price per sheep last month?']).fillna(0)
                revenue_components.append('sheep_kpmd_revenue')

            if all(c in self.df.columns for c in ['E2a. How many goats did you sell to KPMD off-takers  last month?', 'E2c. What was the average price per goat last month?']):
                self.df['goat_kpmd_revenue'] = to_num(self.df['E2a. How many goats did you sell to KPMD off-takers  last month?']).fillna(0) * \
                                               to_num(self.df['E2c. What was the average price per goat last month?']).fillna(0)
                revenue_components.append('goat_kpmd_revenue')

            # Non-KPMD (E3/E4)
            if all(c in self.df.columns for c in ['E3b. How many sheep did you sell to non-KPMD off-takers  last month?', 'E3d. What was the average price per sheep last month?']):
                self.df['sheep_non_kpmd_revenue'] = to_num(self.df['E3b. How many sheep did you sell to non-KPMD off-takers  last month?']).fillna(0) * \
                                                    to_num(self.df['E3d. What was the average price per sheep last month?']).fillna(0)
                revenue_components.append('sheep_non_kpmd_revenue')

            if all(c in self.df.columns for c in ['E4b. How many goats did you sell to non-KPMD off-takers  last month?', 'E4d. What was the average price per goat last month?']):
                self.df['goat_non_kpmd_revenue'] = to_num(self.df['E4b. How many goats did you sell to non-KPMD off-takers  last month?']).fillna(0) * \
                                                   to_num(self.df['E4d. What was the average price per goat last month?']).fillna(0)
                revenue_components.append('goat_non_kpmd_revenue')

            # Fodder sales revenue
            if all(c in self.df.columns for c in ['B6d. At What price did you sell a 15 kg bale last month?','B6e. Number of 15 kg bales sold in the last 1 month?']):
                self.df['fodder_revenue'] = to_num(self.df['B6d. At What price did you sell a 15 kg bale last month?']).fillna(0) * \
                                            to_num(self.df['B6e. Number of 15 kg bales sold in the last 1 month?']).fillna(0)
                revenue_components.append('fodder_revenue')

            if revenue_components:
                self.df['total_revenue'] = self.df[revenue_components].sum(axis=1)

            # COSTS
            cost_components = []
            if 'Feed_Expenditure' in self.df.columns:
                self.df['feed_costs'] = to_num(self.df['Feed_Expenditure']).fillna(0)
                cost_components.append('feed_costs')

            if 'B3b. What was the cost of herding per month (Ksh)?' in self.df.columns:
                self.df['herding_costs'] = to_num(self.df['B3b. What was the cost of herding per month (Ksh)?']).fillna(0)
                cost_components.append('herding_costs')

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
                self.df['transport_costs'] = self.df[existing_transport].sum(axis=1)
                cost_components.append('transport_costs')

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

            if cost_components:
                self.df['total_costs'] = self.df[cost_components].sum(axis=1)

            self.df['net_profit'] = self.df['total_revenue'] - self.df['total_costs']

            valid_revenue = self.df['total_revenue'] > 0
            self.df.loc[valid_revenue, 'profit_margin'] = (
                self.df.loc[valid_revenue, 'net_profit'] / self.df.loc[valid_revenue, 'total_revenue'] * 100
            )

            if all(c in self.df.columns for c in ['sheep_kpmd_revenue', 'transport_costs']):
                self.df['sheep_kpmd_profit_margin'] = (
                    (self.df['sheep_kpmd_revenue'] - self.df['transport_costs'] * 0.5) /
                    self.df['sheep_kpmd_revenue'].replace(0, np.nan) * 100
                ).replace([np.inf, -np.inf], 0).fillna(0)
            if all(c in self.df.columns for c in ['sheep_non_kpmd_revenue', 'transport_costs']):
                self.df['sheep_non_kpmd_profit_margin'] = (
                    (self.df['sheep_non_kpmd_revenue'] - self.df['transport_costs'] * 0.5) /
                    self.df['sheep_non_kpmd_revenue'].replace(0, np.nan) * 100
                ).replace([np.inf, -np.inf], 0).fillna(0)

        except Exception as e:
            st.warning(f"Some P&L metrics could not be calculated: {str(e)}")

# -------------------------------------------------
# Dashboard Renderer
# -------------------------------------------------
class DashboardRenderer:
    def __init__(self, data_processor):
        self.dp = data_processor

    @property
    def df(self):
        return self.dp.df

    def create_comparison_cards(self, data, metric_col, title, format_str="{:.1f}"):
        try:
            kpmd_data = data[data['kpmd_registered'] == 1]
            non_kpmd_data = data[data['kpmd_registered'] == 0]

            col1, col2 = st.columns(2)
            with col1:
                v = kpmd_data[metric_col].mean() if (metric_col in kpmd_data and len(kpmd_data) > 0) else 0
                st.markdown(f"""
                <div class="metric-card kpmd-card">
                    <h4>KPMD Registered</h4>
                    <h3>{format_str.format(v if pd.notna(v) else 0)}</h3>
                    <small>n={len(kpmd_data)}</small>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                v = non_kpmd_data[metric_col].mean() if (metric_col in non_kpmd_data and len(non_kpmd_data) > 0) else 0
                st.markdown(f"""
                <div class="metric-card non-kpmd-card">
                    <h4>Non-KPMD</h4>
                    <h3>{format_str.format(v if pd.notna(v) else 0)}</h3>
                    <small>n={len(non_kpmd_data)}</small>
                </div>
                """, unsafe_allow_html=True)
        except Exception:
            st.warning(f"Could not create comparison cards for {metric_col}")

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
            with col1:
                fig = px.histogram(self.df, x='net_profit', title='Distribution of Net Profit',
                                   labels={'net_profit': 'Net Profit (KES)'})
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                if 'kpmd_registered' in self.df.columns:
                    fig = px.box(self.df, x='kpmd_registered', y='net_profit',
                                 title='Profit Distribution by KPMD Status',
                                 labels={'kpmd_registered': 'KPMD Registered', 'net_profit': 'Net Profit (KES)'},
                                 color='kpmd_registered')
                    st.plotly_chart(fig, use_container_width=True)

            if 'County' in self.df.columns:
                st.subheader("Profitability by County")
                county_profit = self.df.groupby('County', dropna=True)['net_profit'].agg(['mean', 'count']).reset_index()
                county_profit = county_profit[county_profit['count'] >= 3]
                if len(county_profit) > 0:
                    fig = px.bar(county_profit, x='County', y='mean',
                                 title='Average Net Profit by County',
                                 labels={'mean': 'Average Net Profit (KES)'}, color='mean')
                    st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Revenue Analysis")
            revenue_cols = [c for c in self.df.columns if 'revenue' in c.lower() and c != 'total_revenue']
            if revenue_cols:
                avg_comp = self.df[revenue_cols].mean().sort_values(ascending=False)
                fig = px.pie(values=avg_comp.values, names=avg_comp.index, title='Average Revenue Composition')
                st.plotly_chart(fig, use_container_width=True)

            if 'kpmd_registered' in self.df.columns:
                rc = self.df.groupby('kpmd_registered')['total_revenue'].mean().reset_index()
                rc['KPMD_Status'] = rc['kpmd_registered'].map({1: 'KPMD', 0: 'Non-KPMD'})
                fig = px.bar(rc, x='KPMD_Status', y='total_revenue', title='Average Revenue by KPMD Status',
                             labels={'total_revenue': 'Average Revenue (KES)'})
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Cost Structure Analysis")
            cost_cols = [c for c in self.df.columns if 'costs' in c.lower() and c != 'total_costs']
            if cost_cols:
                avg_cost = self.df[cost_cols].mean().sort_values(ascending=False)
                fig = px.bar(avg_cost, title='Average Cost Composition',
                             labels={'value': 'Average Cost (KES)', 'index': 'Cost Category'})
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
                        rows.append({
                            'Channel': channel_name,
                            'Profit_Margin': sub[col].mean(),
                            'KPMD_Status': 'KPMD Registered' if s == 1 else 'Non-KPMD Registered'
                        })
                ch_df = pd.DataFrame(rows)
                fig = px.bar(ch_df, x='Channel', y='Profit_Margin', color='KPMD_Status',
                             title='Channel Profit Margins by KPMD Registration',
                             barmode='group', labels={'Profit_Margin': 'Profit Margin (%)'})
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Breakeven Analysis")
            data = self.df.copy()
            data['breakeven_status'] = np.where(data['net_profit'] >= 0, 'Profitable', 'Loss-making')
            if 'kpmd_registered' in data.columns:
                pivot = pd.crosstab(data['kpmd_registered'], data['breakeven_status'], normalize='index') * 100
                pivot = pivot.reset_index()
                pivot['KPMD_Status'] = pivot['kpmd_registered'].map({1: 'KPMD', 0: 'Non-KPMD'})
                melted = pivot.melt(id_vars=['KPMD_Status'], value_vars=['Profitable', 'Loss-making'],
                                    var_name='Status', value_name='Percentage')
                fig = px.bar(melted, x='KPMD_Status', y='Percentage', color='Status',
                             title='Breakeven Status by KPMD Registration', barmode='stack')
                st.plotly_chart(fig, use_container_width=True)

    # ---------------- Field & Data Outlook ----------------
    def render_field_outlook(self):
        st.header("🧭 Field & Data Outlook")

        # --- KPI row ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Submissions", len(self.df))
        with col2:
            latest = None
            cand = (
                '_submission_time' if '_submission_time' in self.df.columns
                else 'int_date_std' if 'int_date_std' in self.df.columns
                else None
            )
            if cand:
                try:
                    tmp = pd.to_datetime(self.df[cand], errors='coerce')
                    latest = tmp.max()
                except Exception:
                    latest = None
            st.metric("Latest Submission", latest.strftime("%Y-%m-%d") if pd.notna(latest) else "N/A")
        with col3:
            c = self.df['County'].nunique() if 'County' in self.df.columns else 0
            st.metric("Counties Covered", int(c))
        with col4:
            k = self.df['kpmd_registered'].sum() if 'kpmd_registered' in self.df.columns else 0
            st.metric("KPMD Participants", int(k))

        # --- Submissions over time ---
        left, right = st.columns([0.8, 0.2])
        with left:
            st.subheader("Submissions Over Time")
        with right:
            gran = st.selectbox(
                "Granularity", ["Daily", "Weekly", "Monthly"],
                index=0,                      # default to Daily
                label_visibility="collapsed"  # keep it compact
            )

        # pick the best date column available
        date_col = (
            '_submission_time' if '_submission_time' in self.df.columns
            else 'int_date_std' if 'int_date_std' in self.df.columns
            else None
        )

        if date_col:
            tmp = self.df.copy()
            tmp['__date'] = pd.to_datetime(tmp[date_col], errors='coerce')
            tmp = tmp[tmp['__date'].notna()].copy()

            if gran == "Daily":
                tmp['__bucket'] = tmp['__date'].dt.date
                x_label, title = "Date", "Daily Submission Volume"
            elif gran == "Weekly":
                tmp['__bucket'] = tmp['__date'].dt.to_period('W').dt.start_time.dt.date
                x_label, title = "Week (start)", "Weekly Submission Volume"
            else:  # Monthly
                tmp['__bucket'] = tmp['__date'].dt.to_period('M').dt.to_timestamp()
                x_label, title = "Month", "Monthly Submission Volume"

            series = (tmp.groupby('__bucket').size()
                        .reset_index(name='Submissions')
                        .sort_values('__bucket'))

            if len(series) > 0:
                fig = px.line(
                    series, x='__bucket', y='Submissions',
                    title=title, markers=True,
                    labels={'__bucket': x_label}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No dated submissions available to plot.")
        else:
            # Fallback to existing monthly column if no timestamp exists
            if 'month' in self.df.columns and not self.df['month'].isna().all():
                monthly = (self.df.groupby('month').size()
                        .reset_index(name='Submissions')
                        .sort_values('month'))
                fig = px.line(monthly, x='month', y='Submissions',
                            title='Monthly Submission Volume', markers=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No time information available to show submissions over time.")

        # --- Submissions by County and KPMD Status ---
        st.subheader("Submissions by County and KPMD Status")
        if 'County' in self.df.columns and 'kpmd_registered' in self.df.columns:
            county_kpmd = (
                self.df.groupby(['County', 'kpmd_registered'])
                    .size()
                    .reset_index(name='count')
            )
            county_kpmd['kpmd_status'] = county_kpmd['kpmd_registered'].map({1: 'KPMD', 0: 'Non-KPMD'})
            fig = px.bar(
                county_kpmd,
                x='County',
                y='count',
                color='kpmd_status',
                title='Submissions by County and KPMD Status',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("County or KPMD data not available for this analysis")

        # --- Household locations map (after the county chart) ---
        st.subheader("Household Locations")
        lat_col = '_GPS Coordinates_latitude'
        lon_col = '_GPS Coordinates_longitude'
        if lat_col in self.df.columns and lon_col in self.df.columns:
            map_df = self.df.dropna(subset=[lat_col, lon_col]).copy()
            map_df.rename(columns={lat_col: 'lat', lon_col: 'lon'}, inplace=True)
            if 'kpmd_registered' in map_df.columns:
                map_df['r'] = np.where(map_df['kpmd_registered'] == 1, 31, 214)
                map_df['g'] = np.where(map_df['kpmd_registered'] == 1, 119, 39)
                map_df['b'] = np.where(map_df['kpmd_registered'] == 1, 180, 40)
            else:
                map_df['r'], map_df['g'], map_df['b'] = 160, 160, 160
            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                initial_view_state=pdk.ViewState(
                    latitude=map_df['lat'].mean() if len(map_df) > 0 else -1.29,
                    longitude=map_df['lon'].mean() if len(map_df) > 0 else 36.82,
                    zoom=7, pitch=0
                ),
                layers=[pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df,
                    get_position='[lon, lat]',
                    get_radius=900,
                    get_fill_color='[r, g, b]',
                    pickable=True
                )]
            ))
        else:
            st.info("GPS coordinates not available for mapping")

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
            else:
                st.info("Female stock percentage data not available")

            if all(col in self.df.columns for col in ['total_sheep', 'total_goats', 'kpmd_registered']):
                st.subheader("Herd Composition by KPMD Status")
                try:
                    comp = self.df.groupby('kpmd_registered')[['total_sheep', 'total_goats']].mean().reset_index()
                    comp['kpmd_status'] = comp['kpmd_registered'].map({1: 'KPMD', 0: 'Non-KPMD'})
                    melted = comp.melt(id_vars=['kpmd_status'], value_vars=['total_sheep', 'total_goats'],
                                       var_name='Species', value_name='Average Count')
                    melted['Species'] = melted['Species'].map({'total_sheep': 'Sheep', 'total_goats': 'Goats'})
                    fig = px.bar(melted, x='kpmd_status', y='Average Count', color='Species',
                                 title='Average Herd Composition by KPMD Status', barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("Herd composition data not available for visualization")
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
                rows = []
                for col in self.dp.vacc_disease_cols:
                    name = col.split('/')[-1]
                    for s in [0, 1]:
                        sub = self.df[self.df['kpmd_registered'] == s]
                        if len(sub) > 0:
                            rate = sub[col].mean() * 100
                            rows.append({'Disease': name, 'Rate': rate, 'KPMD_Status': 'KPMD' if s == 1 else 'Non-KPMD'})
                if rows:
                    dfp = pd.DataFrame(rows)
                    fig = px.bar(dfp, x='Disease', y='Rate', color='KPMD_Status',
                                 title='Vaccination Diseases by KPMD Status (%)', barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Vaccination disease data not available in this dataset")

            if hasattr(self.dp, 'treat_disease_cols') and self.dp.treat_disease_cols:
                rows = []
                for col in self.dp.treat_disease_cols:
                    name = col.split('/')[-1]
                    for s in [0, 1]:
                        sub = self.df[self.df['kpmd_registered'] == s]
                        if len(sub) > 0:
                            rate = sub[col].mean() * 100
                            rows.append({'Disease': name, 'Rate': rate, 'KPMD_Status': 'KPMD' if s == 1 else 'Non-KPMD'})
                if rows:
                    dfp = pd.DataFrame(rows)
                    fig = px.bar(dfp, x='Disease', y='Rate', color='KPMD_Status',
                                 title='Treatment Diseases by KPMD Status (%)', barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Treatment disease data not available in this dataset")

            prov_col = 'D2. Who performed the small ruminants vaccinations in the last month?'
            if prov_col in self.df.columns:
                try:
                    provider_counts = self.df.groupby(['kpmd_registered', prov_col]).size().reset_index(name='count')
                    provider_counts['KPMD_Status'] = provider_counts['kpmd_registered'].map({1: 'KPMD', 0: 'Non-KPMD'})
                    fig = px.bar(provider_counts, x='KPMD_Status', y='count', color=prov_col,
                                 title='Vaccination Providers by KPMD Status')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("Vaccination provider data not available")
            else:
                st.info("Vaccination provider data not available")

        with tab3:
            st.subheader("Small Ruminant Productivity Indicators")
            if 'birth_rate_per_100' in self.df.columns:
                self.create_comparison_cards(self.df, 'birth_rate_per_100', 'Birth Rate', '{:.1f}')
            else:
                st.info("Birth rate data not available")
            if 'mortality_rate_per_100' in self.df.columns:
                self.create_comparison_cards(self.df, 'mortality_rate_per_100', 'Mortality Rate', '{:.1f}')
            else:
                st.info("Mortality rate data not available")
            if 'loss_rate_per_100' in self.df.columns:
                self.create_comparison_cards(self.df, 'loss_rate_per_100', 'Loss Rate', '{:.1f}')
            else:
                st.info("Loss rate data not available")

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
                                 title='Productivity Rates by KPMD Status (per 100 head)', barmode='group')
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
                tmp = self.df.copy()
                tmp['purchased'] = tmp[col].apply(yn).astype(int)
                self.create_comparison_cards(tmp, 'purchased', 'Purchase Rate', '{:.1%}')
            else:
                st.info("Fodder purchase data not available")

            st.subheader("Feed Purchase Sources")
            source_cols = [c for c in self.df.columns if c.startswith('B5b. Where did you buy feeds in the last 1 month?/') and 'Other' not in c]
            if source_cols:
                rows = []
                for c in source_cols:
                    name = c.split('/')[-1]
                    for s in [0,1]:
                        sub = self.df[self.df['kpmd_registered'] == s]
                        rate = pd.to_numeric(sub[c].astype(str).replace({'1':1,'0':0}), errors='coerce').fillna(0).mean() * 100 if len(sub) else 0
                        rows.append({'Source': name, 'Rate': rate, 'KPMD_Status': 'KPMD' if s==1 else 'Non-KPMD'})
                dfp = pd.DataFrame(rows)
                fig = px.bar(dfp, x='Source', y='Rate', color='KPMD_Status',
                             title='Feed Purchase Sources by KPMD Status (%)', barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feed source data not available")

        with tab2:
            st.subheader("Fodder Production")
            col = 'B6a. Did you produce any fodder?'
            if col in self.df.columns:
                tmp = self.df.copy()
                tmp['produced'] = tmp[col].apply(yn).astype(int)
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
                if len(exp) > 0:
                    self.create_comparison_cards(exp, 'Feed_Expenditure', 'Feed Expenditure', 'KES {:.0f}')
                else:
                    st.info("No households reported feed expenditure")
            else:
                st.info("Feed expenditure data not available - required columns missing")

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
            kpmd_sold_col = mapping.get('sheep_kpmd_sold')
            non_kpmd_sold_col = mapping.get('sheep_non_kpmd_sold')
        else:
            kpmd_sold_col = mapping.get('goat_kpmd_sold')
            non_kpmd_sold_col = mapping.get('goat_non_kpmd_sold')

        def _sales_cols(species, kpmd_prefix, non_kpmd_prefix):
            if species.lower().startswith('sheep'):
                price_kpmd = f"{kpmd_prefix}c. What was the average price per sheep last month?"
                price_non  = f"{non_kpmd_prefix}d. What was the average price per sheep last month?"
                age_kpmd   = f"{kpmd_prefix}d. What was the typical age in months of the sheep when sold to KPMD off-takers last month?"
                age_non    = f"{non_kpmd_prefix}e. What was the typical age in months of the sheep when sold to non-KPMD off-takers last month?"
            else:
                price_kpmd = f"{kpmd_prefix}c. What was the average price per goat last month?"
                price_non  = f"{non_kpmd_prefix}d. What was the average price per goat last month?"
                age_kpmd   = f"{kpmd_prefix}d. What was the typical age in months of the goats when sold to KPMD off-takers last month?"
                age_non    = f"{non_kpmd_prefix}e. What was the typical age in months of the goats when sold to non-KPMD off-takers last month?"
            return price_kpmd, price_non, age_kpmd, age_non

        price_kpmd_col, price_non_col, age_kpmd_col, age_non_col = _sales_cols(species, kpmd_prefix, non_kpmd_prefix)

        tab1, tab2, tab3 = st.tabs(["Sales Volume", "Price Analysis", "Transaction Details"])

        with tab1:
            st.subheader("Sales Volume Analysis")
            if kpmd_sold_col and kpmd_sold_col in self.df.columns:
                tmp = self.df.copy()
                tmp['sold_kpmd'] = tmp[kpmd_sold_col].apply(yn).astype(int)
                self.create_comparison_cards(tmp, 'sold_kpmd', f'KPMD Sales Rate ({title_species})', '{:.1%}')
            else:
                st.info(f"KPMD sales data for {title_species} not available")

            if non_kpmd_sold_col and non_kpmd_sold_col in self.df.columns:
                tmp = self.df.copy()
                tmp['sold_non_kpmd'] = tmp[non_kpmd_sold_col].apply(yn).astype(int)
                self.create_comparison_cards(tmp, 'sold_non_kpmd', f'Non-KPMD Sales Rate ({title_species})', '{:.1%}')
            else:
                st.info(f"Non-KPMD sales data for {title_species} not available")

        with tab2:
            st.subheader("Price Analysis")
            price_data = []
            if price_kpmd_col in self.df.columns:
                for s in [0,1]:
                    sub = self.df[self.df['kpmd_registered'] == s]
                    vals = to_num(sub[price_kpmd_col]).dropna()
                    price_data += [{'Channel':'KPMD','Price':v,'KPMD_Status':'KPMD Registered' if s==1 else 'Non-KPMD Registered'} for v in vals]
            if price_non_col in self.df.columns:
                for s in [0,1]:
                    sub = self.df[self.df['kpmd_registered'] == s]
                    vals = to_num(sub[price_non_col]).dropna()
                    price_data += [{'Channel':'Non-KPMD','Price':v,'KPMD_Status':'KPMD Registered' if s==1 else 'Non-KPMD Registered'} for v in vals]
            if price_data:
                dfp = pd.DataFrame(price_data)
                fig = px.box(dfp, x='Channel', y='Price', color='KPMD_Status',
                             title=f'{title_species} Price Distribution by Channel and KPMD Registration')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Price data for {title_species} not available")

        with tab3:
            st.subheader("Transaction Details")
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

    # ---------------- Payments ----------------
    def render_payments(self):
        st.header("💸 Payment Methods")

        # helper: normalize whitespace/case for reliable matching
        def _norm(s: str) -> str:
            return re.sub(r'\s+', ' ', str(s)).strip().lower()

        # Use space-normalized stems (single spaces)
        stems = [
            ('Sheep – KPMD',  'E1g. How were you paid by the KPMD off-takers last month? [Select all that apply]'),
            ('Goats – KPMD',  'E2g. How were you paid by the KPMD off-takers last month? [Select all that apply]'),
            ('Sheep – Other', 'E3h. How were you paid by the non-KPMD off-takers last month? [Select all that apply]'),
            ('Goats – Other', 'E4h. How were you paid by the non-KPMD off-takers last month? [Select all that apply]'),
        ]

        rows = []
        cols_norm = {_norm(c): c for c in self.df.columns}  # map normalized -> original

        for label, stem in stems:
            stem_n = _norm(stem)

            # 1) find all sub-option columns like "<stem>/<option>"
            #    by comparing *normalized* strings
            subcols = []
            for c in self.df.columns:
                c_n = _norm(c)
                if c_n.startswith(stem_n) and '/' in c:
                    subcols.append(c)

            # classify subcols into mobile/cash based on suffix text
            mobile_cols, cash_cols = [], []
            for c in subcols:
                suffix = _norm(c.split('/', 1)[1])
                if ('mobile' in suffix) or ('m-pesa' in suffix) or ('mpesa' in suffix):
                    mobile_cols.append(c)
                if 'cash' in suffix:
                    cash_cols.append(c)

            mobile_series = None
            cash_series = None

            if mobile_cols:
                mobile_series = (self.df[mobile_cols].astype(str).replace({'1': 1, '0': 0})
                                .apply(pd.to_numeric, errors='coerce').fillna(0).max(axis=1))
            if cash_cols:
                cash_series = (self.df[cash_cols].astype(str).replace({'1': 1, '0': 0})
                            .apply(pd.to_numeric, errors='coerce').fillna(0).max(axis=1))

            # 2) fallback: a single multi-select cell column equal to the stem
            if mobile_series is None or cash_series is None:
                # look up any column whose normalized name equals the normalized stem
                single_col = cols_norm.get(stem_n, None)
                if single_col is not None:
                    dummies = one_hot_multiselect(self.df[single_col])
                    if mobile_series is None:
                        tok = next((t for t in dummies.columns
                                    if _norm(t).startswith('mobile') or 'mpesa' in _norm(t)), None)
                        mobile_series = dummies.get(tok, pd.Series(0, index=self.df.index))
                    if cash_series is None:
                        tok = next((t for t in dummies.columns if _norm(t).startswith('cash')), None)
                        cash_series = dummies.get(tok, pd.Series(0, index=self.df.index))

            # 3) final fallback to zeros
            if mobile_series is None:
                mobile_series = pd.Series(0, index=self.df.index)
            if cash_series is None:
                cash_series = pd.Series(0, index=self.df.index)

            tmp_cols = ['kpmd_registered'] + (['County'] if 'County' in self.df.columns else [])
            tmp = self.df[tmp_cols].copy()
            tmp['block']  = label
            tmp['mobile'] = pd.to_numeric(mobile_series, errors='coerce').fillna(0).clip(0, 1).astype(int)
            tmp['cash']   = pd.to_numeric(cash_series,   errors='coerce').fillna(0).clip(0, 1).astype(int)
            tmp['both']   = ((tmp['mobile'] == 1) & (tmp['cash'] == 1)).astype(int)
            rows.append(tmp)

        if not rows:
            st.info("No payment method columns found")
            return

        payment = pd.concat(rows, ignore_index=True)

        grp = payment.groupby(['block', 'kpmd_registered'], dropna=False)
        summary = pd.DataFrame({
            'Mobile share': grp['mobile'].mean() * 100,
            'Cash share':   grp['cash'].mean() * 100,
            'Both share':   grp['both'].mean() * 100
        }).reset_index()
        summary['KPMD Status'] = summary['kpmd_registered'].map({1: 'KPMD', 0: 'Non-KPMD'})

        long = summary.melt(
            id_vars=['block', 'KPMD Status'],
            value_vars=['Cash share', 'Mobile share', 'Both share'],
            var_name='Method',
            value_name='Share'
        )

        if long['Share'].sum() == 0 or long.dropna(subset=['Share']).empty:
            st.warning("No non-zero payment shares detected. Check column names in your CSV.")
            return

        fig = px.bar(long, x='block', y='Share', color='Method',
                    barmode='group', facet_col='KPMD Status',
                    title='Payment method mix by channel/species and KPMD')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Digital adoption by county (Mobile or Both)")
        county = payment.copy()
        county['digital'] = ((county['mobile'] == 1) | (county['both'] == 1)).astype(int)
        if 'County' in county.columns:
            county_summary = county.groupby(['County', 'kpmd_registered'])[['digital']].mean().mul(100).reset_index()
            county_summary['KPMD Status'] = county_summary['kpmd_registered'].map({1: 'KPMD', 0: 'Non-KPMD'})
            if county_summary['digital'].sum() == 0:
                st.info("No digital payments found in the data.")
            else:
                fig2 = px.bar(county_summary, x='County', y='digital', color='KPMD Status',
                            barmode='group', title='Digital share (%)')
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("County column not available for county split")

    # ---------------- County Comparator ----------------
    def render_county_compare(self):
        st.header("📊 County Comparator")
        if 'County' not in self.df.columns:
            st.info("County column missing")
            return
        counties = sorted(self.df['County'].dropna().unique())
        if len(counties) < 2:
            st.info("Need at least two counties for comparison")
            return
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
        colA, colB = st.columns(2)
        A, B = slice_county(left), slice_county(right)
        for (label, col, fmt) in metrics:
            with colA:
                if col and col in A.columns:
                    v = pd.to_numeric(A[col].apply(yn) if A[col].dtype=='O' else A[col], errors='coerce').mean()
                else:
                    v = np.nan
                st.metric(f"{label} — {left}", (fmt.format(v) if pd.notna(v) else "N/A"))
            with colB:
                if col and col in B.columns:
                    v = pd.to_numeric(B[col].apply(yn) if B[col].dtype=='O' else B[col], errors='coerce').mean()
                else:
                    v = np.nan
                st.metric(f"{label} — {right}", (fmt.format(v) if pd.notna(v) else "N/A"))

    # ---------------- Gender Inclusion ----------------
    def render_gender_inclusion(self):
        st.header("♀️ Gender Inclusion")

        tab1, tab2, tab3 = st.tabs(["Decision Making", "KPMD Participation", "Income Control"])

        with tab1:
            st.subheader("Livestock Sale Decision Making")
            decision_col = self.dp.gender_columns.get('decision_making', '')
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
                    women_roles = ['Spouse', 'Daughter']
                    if not dfp.empty:
                        women = dfp[dfp['Role'].isin(women_roles)].groupby('KPMD_Status')['Involvement_Rate'].mean().reset_index()
                        st.write("**Women's Involvement in Decision Making**")
                        for _, r in women.iterrows():
                            st.metric(f"{r['KPMD_Status']} - Women Involvement", f"{r['Involvement_Rate']:.1f}%")
                        fig = px.bar(dfp, x='Role', y='Involvement_Rate', color='KPMD_Status',
                                     title='Decision Making Roles by KPMD Status (%)', barmode='group')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No decision making data available after processing")
                else:
                    st.info("Decision making role columns not found")
            else:
                st.info("Decision making data not available in this dataset")

        with tab2:
            st.subheader("KPMD Participation by Gender")
            if 'Gender' in self.df.columns and 'kpmd_registered' in self.df.columns:
                g = self.df[self.df['Gender'].notna()]
                if len(g) > 0:
                    ct = pd.crosstab(g['Gender'], g['kpmd_registered'], normalize='index') * 100
                    ct = ct.reset_index().rename(columns={0:'Non-KPMD',1:'KPMD'})
                    melted = ct.melt(id_vars=['Gender'], value_vars=['KPMD','Non-KPMD'], var_name='KPMD_Status', value_name='Percentage')
                    fig = px.bar(melted, x='Gender', y='Percentage', color='KPMD_Status',
                                 title='KPMD Participation by Gender (%)', barmode='stack')
                    st.plotly_chart(fig, use_container_width=True)
                    hh_head_col = self.dp.gender_columns.get('household_head','')
                    if hh_head_col and hh_head_col in self.df.columns:
                        female_heads = self.df[(self.df['Gender'] == 'Female') & (self.df[hh_head_col].apply(yn) == 1)]
                        if len(female_heads) > 0:
                            pct = female_heads['kpmd_registered'].mean() * 100
                            st.metric("Female-Headed Households in KPMD", f"{pct:.1f}%")
                        else:
                            st.info("No female-headed households found in the data")
                else:
                    st.info("No gender data available for analysis")
            else:
                st.info("Gender or KPMD data not available")

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
                        women_roles = ['Spouse', 'Daughter']
                        women = dfp[dfp['Role'].isin(women_roles)].groupby('KPMD_Status')['Control_Rate'].mean().reset_index()
                        st.write("**Women's Control Over Livestock Income**")
                        for _, r in women.iterrows():
                            st.metric(f"{r['KPMD_Status']} - Women Control", f"{r['Control_Rate']:.1f}%")
                        fig = px.bar(dfp, x='Role', y='Control_Rate', color='KPMD_Status',
                                     title='Income Control Roles by KPMD Status (%)', barmode='group')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No income control data available after processing")
                else:
                    st.info("Income control role columns not found")
            else:
                st.info("Income control data not available in this dataset")

    # ---------------- Climate Impact ----------------
    def render_climate_impact(self):
        st.header("🌦️ Climate Impact")
        tab1, tab2, tab3 = st.tabs(["Adaptation Measures", "Barriers to Adaptation", "Climate Resilience"])

        with tab1:
            st.subheader("Adaptation Measures")
            j1 = self.dp.column_mapping.get('adaptation_measures')
            if j1 and j1 in self.df.columns:
                tmp = self.df.copy()
                tmp['adapted'] = tmp[j1].apply(yn).astype(int)
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
                rows = []
                for c in barrier_cols:
                    name = c.split('/')[-1]
                    for s in [0,1]:
                        sub = base[base['kpmd_registered']==s]
                        rate = pd.to_numeric(sub[c].astype(str).replace({'1':1,'0':0}), errors='coerce').fillna(0).mean() * 100 if len(sub) else 0
                        rows.append({'Barrier': name, 'Rate': rate, 'KPMD_Status': 'KPMD' if s==1 else 'Non-KPMD'})
                dfp = pd.DataFrame(rows)
                fig = px.bar(dfp, x='Barrier', y='Rate', color='KPMD_Status',
                             title='Barriers to Adaptation by KPMD Status (%)', barmode='group')
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
                if len(resilience_data) > 0:
                    self.create_comparison_cards(resilience_data, 'resilience_score', 'Resilience Score', '{:.1f}')
                else:
                    st.info("No resilience score data available")

            if 'adaptation_score' in self.df.columns:
                st.metric("Households Implementing Adaptation", f"{(self.df['adaptation_score'].mean()*100):.1f}%")

            if 'total_sr' in self.df.columns and 'Feed_Expenditure' in self.df.columns:
                large_herds = (self.df['total_sr'] > self.df['total_sr'].median()).mean() * 100
                st.metric("Households with Above-Median Herd Size", f"{large_herds:.1f}%")

            st.info("Additional climate resilience indicators will be displayed as data becomes available")

# -------------------------------------------------
# Main App
# -------------------------------------------------
def main():
    st.title("APMT Project Insights")
    st.markdown('<div class="main-header">Pastoral Market Transformation Monitoring</div>', unsafe_allow_html=True)

    # --- Auto-load dataset (no uploader) ---
    st.sidebar.header("Data Source")
    st.sidebar.write("Auto-loaded file:")
    st.sidebar.code(DATA_PATH)

    # Reload button to clear cache and rerun
    if st.sidebar.button("Reload data"):
        load_apmt_csv.clear()  # clears @st.cache_data
        st.sidebar.success("Cache cleared. Reloading…")
        st.rerun()

    try:
        df = load_apmt_csv(DATA_PATH)
        st.success(f"Data loaded successfully: {len(df):,} records")

        with st.expander("Data Preview", expanded=False):
            st.write(f"Columns detected ({len(df.columns)}):")
            st.write(list(df.columns))
            st.write(f"Total records: {len(df)}")
            st.dataframe(df.head(10))

        # Initialize data processor and renderer
        processor = APMTDataProcessor(df)
        renderer = DashboardRenderer(processor)

        # ---------- Sidebar: FILTERS (collapsed unless active) ----------
        st.sidebar.header("Global Filters")

        # Defaults for “inactive” state
        county_default = 'All'
        subcounty_default = 'All'
        kpmd_default = 'All'
        gender_default = 'All'

        # Date defaults (full range on current data)
        if 'int_date_std' in processor.df.columns:
            _min_date = pd.to_datetime(processor.df['int_date_std'], errors='coerce').min()
            _max_date = pd.to_datetime(processor.df['int_date_std'], errors='coerce').max()
        else:
            _min_date, _max_date = datetime(2024, 1, 1), datetime.today()

        date_default = (
            _min_date.date() if pd.notna(_min_date) else datetime(2024, 1, 1).date(),
            _max_date.date() if pd.notna(_max_date) else datetime.today().date()
        )

        def _get_state(k, default):
            return st.session_state.get(k, default)

        # Open filters if any control is “active”
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
            if 'int_date_std' in processor.df.columns:
                try:
                    date_range = st.date_input(
                        "Select Date Range",
                        value=_get_state('date_range', date_default),
                        min_value=date_default[0],
                        max_value=date_default[1],
                        key="date_range"
                    )
                    if isinstance(date_range, tuple) and len(date_range) == 2:
                        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                        processor.df = processor.df[
                            (pd.to_datetime(processor.df['int_date_std'], errors='coerce') >= start) &
                            (pd.to_datetime(processor.df['int_date_std'], errors='coerce') <= end)
                        ]
                except Exception:
                    st.warning("Date filtering not available")

        # ---------- Sidebar: NAVIGATION (ALWAYS VISIBLE, RED HEADER) ----------
        st.sidebar.markdown(
            '<div style="color:#dc3545; font-weight:700; font-size:1rem; margin-bottom:0.25rem;">'
            'Navigate Here <span style="font-size:1.1rem; line-height:1;">👇</span>'
            '</div>',
            unsafe_allow_html=True
        )

        # Order with P&L Analysis LAST
        pages = [
            "Field Outlook",
            "Pastoral Productivity",
            "Feed & Fodder",
            "Sheep Offtake",
            "Goat Offtake",
            "Payments",
            "County Comparator",
            "Gender Inclusion",
            "Climate Impact",
            "P&L Analysis",  # last
        ]

        if 'nav_page' not in st.session_state:
            st.session_state['nav_page'] = "Field Outlook"

        page = st.sidebar.radio(
            "Select Dashboard Page",
            pages,
            key="nav_page"
        )

        # Render
        if page == "Field Outlook":
            renderer.render_field_outlook()
        elif page == "Pastoral Productivity":
            renderer.render_pastoral_productivity()
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
        elif page == "P&L Analysis":
            renderer.render_pl_analysis()

        # Export
        st.sidebar.header("Data Export")
        csv = processor.df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download Filtered Data (CSV)",
            data=csv,
            file_name=f"apmt_filtered_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    except FileNotFoundError:
        st.error("The specified data file was not found.")
        st.code(DATA_PATH)
        st.info("Please check the path or ensure the file exists at this location.")
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.info("Please check that your CSV file matches the expected APMT data format")

if __name__ == "__main__":
    main()