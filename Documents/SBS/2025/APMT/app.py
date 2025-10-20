# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pydeck as pdk
import re
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------
# Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="APMT Program",
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
    .tab-container {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def yn(x):
    """Robust yes/no to 1/0."""
    return 1 if str(x).strip().lower() in {'yes', 'y', '1', 'true'} else 0

def _sales_cols(species, kpmd_prefix, non_kpmd_prefix):
    """Resolve price & age column names for sheep vs goats."""
    if species.lower() == 'sheep':
        price_kpmd = f"{kpmd_prefix}c. What was the average price per sheep last month?"
        price_non  = f"{non_kpmd_prefix}d. What was the average price per sheep last month?"
        age_kpmd   = f"{kpmd_prefix}d. What was the typical age in months of the sheep when sold to KPMD off-takers last month?"
        age_non    = f"{non_kpmd_prefix}e. What was the typical age in months of the sheep when sold to non-KPMD off-takers last month?"
    else:  # goats
        price_kpmd = f"{kpmd_prefix}c. What was the average price per goat last month?"
        price_non  = f"{non_kpmd_prefix}d. What was the average price per goat last month?"
        age_kpmd   = f"{kpmd_prefix}d. What was the typical age in months of the goats when sold to KPMD off-takers last month?"
        age_non    = f"{non_kpmd_prefix}e. What was the typical age in months of the goats when sold to non-KPMD off-takers last month?"
    return price_kpmd, price_non, age_kpmd, age_non

def one_hot_multiselect(series: pd.Series) -> pd.DataFrame:
    """
    Robustly one-hot a single multi-select text column.
    Splits on | ; , / or 2+ spaces. Trims whitespace.
    Ignores empty tokens. Returns 0/1 dtype.
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
    data = {}
    for tok in uniques:
        data[tok] = [1 if tok in toks else 0 for toks in tokens_list]
    df = pd.DataFrame(data, index=series.index).astype(int)
    return df

def try_get_col(df: pd.DataFrame, candidates):
    """Return the first existing column name from candidates (list/tuple), else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -------------------------------------------------
# Data Processor
# -------------------------------------------------
class APMTDataProcessor:
    def __init__(self, df):
        self.df = df.copy()
        self.subcounty_col = None
        self.standardize_data()
        self.add_rcsi()  # try to create/standardize rCSI if possible
        
    def standardize_data(self):
        """Standardize column names and data types"""
        try:
            # Convert dates - handle timezone issues
            date_cols = ['start', 'end', 'int_date', '_submission_time']
            for col in date_cols:
                if col in self.df.columns:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    if pd.api.types.is_datetime64tz_dtype(self.df[col]):
                        self.df[col] = self.df[col].dt.tz_convert(None)
                    self.df[col] = self.df[col].dt.tz_localize(None)

            # Extract month and year
            if 'int_date' in self.df.columns:
                self.df['month'] = self.df['int_date'].dt.to_period('M').astype(str)
                self.df['year'] = self.df['int_date'].dt.year
            
            # KPMD registration standardization
            a8_col = 'A8. Are you registered to KPMD programs?'
            if a8_col in self.df.columns:
                self.df['kpmd_registered'] = self.df[a8_col].apply(yn).astype(int)
            else:
                self.df['kpmd_registered'] = 0  # Default if column doesn't exist
            
            # Standardize common boolean columns
            bool_cols = [
                'C1. Do you own any sheep?', 'C2. Do you own any goats?', 
                'D1. Did you vaccinate your small ruminants livestock in the last month?',
                'D3. Did you treat small ruminants for disease in the last month?',
                'D4. Did you deworm your small ruminants livestock last month?'
            ]
            for col in bool_cols:
                if col in self.df.columns:
                    self.df[col] = self.df[col].apply(yn).astype(int)
            
            # Numeric columns (strip commas safely)
            numeric_cols = [
                'B3bWhat was the cost of herding per month (Ksh)?',
                'B4b. What is the total cost (Ksh)?',
                'B5c. What was the price per 15 kg bale in the last 1 month?', 
                'B5d. Number of 15 kg bales purchased in the last 1 month?',
                'B6b. Quantity of feeds harvested in the last 1 month (15 kg bales)?',
                'B6d. At What price did you sell a 15 kg bale last month?',
                'B6e. Number of 15 kg bales sold in the last 1 month?',
                'D1a. How many small ruminants were vaccinated in the last month?',
                'D1b. What was the cost of small ruminants vaccination in KSH per animal in the last month?',
                'D3a. How many small ruminants livestock were sick and treated in the last one month?',
                'D3b. What was the total cost of treatment in KSH last month?',
                'D4a. What was the total of cost of deworming in KSH last month?'
            ]
            for col in numeric_cols:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col].astype(str).str.replace(',', ''), errors='coerce')

            # Derived feed expenditure
            if {'B5c. What was the price per 15 kg bale in the last 1 month?',
                'B5d. Number of 15 kg bales purchased in the last 1 month?'}.issubset(self.df.columns):
                self.df['Feed_Expenditure'] = (
                    self.df['B5c. What was the price per 15 kg bale in the last 1 month?'] *
                    self.df['B5d. Number of 15 kg bales purchased in the last 1 month?'].fillna(0)
                )
            else:
                self.df['Feed_Expenditure'] = np.nan
            
            # Treatment flag (Arm)
            if 'Selection of the household' in self.df.columns:
                self.df['is_treatment'] = self.df['Selection of the household'].astype(str).str.contains('Treatment', na=False).astype(int)
            else:
                self.df['is_treatment'] = 0

            # Disease binary normalization for any existing one-hot disease columns
            self.standardize_disease_names()

            # Detect subcounty column
            self.subcounty_col = try_get_col(self.df, [
                'Subcounty','Sub-County','Sub County','Sub_County','Sub county'
            ])
            
        except Exception as e:
            st.warning(f"Some data standardization issues occurred: {str(e)}")
        
    def standardize_disease_names(self):
        """Standardize disease names across vaccination and treatment columns (if they exist as dummies)"""
        try:
            vacc_cols = [c for c in self.df.columns if 'D1c. What diseases did you vaccinate' in c]
            treat_cols = [c for c in self.df.columns if 'D3c. What type of disease did you treat' in c]
            for col in vacc_cols + treat_cols:
                self.df[col] = (self.df[col]
                                .astype(str)
                                .str.strip()
                                .replace({'1':1,'0':0,'Yes':1,'No':0,'yes':1,'no':0})
                                .apply(lambda x: 1 if str(x).strip()=='1' else 0)
                               ).astype(int)
        except Exception as e:
            st.warning(f"Some disease standardization issues occurred: {str(e)}")

    def add_rcsi(self):
        """
        Create/standardize rCSI score:
        - If a column named 'rCSI' (case-insensitive) exists, coerce to numeric.
        - Else try to compute from common question patterns using standard weights:
          less preferred foods (1), borrow food (2), limit portion size (1),
          reduce meals (1), restrict adult consumption (3).
        """
        # direct column
        rcsi_col = None
        for c in self.df.columns:
            if c.strip().lower() == 'rcsi':
                rcsi_col = c
                break
        if rcsi_col is not None:
            self.df['rCSI_score'] = pd.to_numeric(self.df[rcsi_col], errors='coerce')
            return

        # try to build from components (frequency per last 7 days)
        # regex candidates (very permissive)
        c_less_pref  = [c for c in self.df.columns if re.search(r'less\s*preferred', c, re.I)]
        c_borrow     = [c for c in self.df.columns if re.search(r'borrow\s*food', c, re.I)]
        c_limit_port = [c for c in self.df.columns if re.search(r'limit(ed)?\s*portion', c, re.I)]
        c_reduce_meal= [c for c in self.df.columns if re.search(r'reduce(d)?\s*meals?', c, re.I)]
        c_restrict   = [c for c in self.df.columns if re.search(r'restrict(ed)?\s*adult', c, re.I)]

        def col_to_num(cands):
            if not cands: return None
            return pd.to_numeric(self.df[cands[0]], errors='coerce')

        L = col_to_num(c_less_pref)
        B = col_to_num(c_borrow)
        P = col_to_num(c_limit_port)
        M = col_to_num(c_reduce_meal)
        R = col_to_num(c_restrict)

        if any(x is not None for x in [L,B,P,M,R]):
            L = L if L is not None else 0
            B = B if B is not None else 0
            P = P if P is not None else 0
            M = M if M is not None else 0
            R = R if R is not None else 0
            self.df['rCSI_score'] = (L*1 + B*2 + P*1 + M*1 + R*3)
        else:
            # not available
            self.df['rCSI_score'] = np.nan
        
    def calculate_herd_metrics(self):
        """Calculate comprehensive herd composition and productivity metrics with error handling"""
        try:
            # Initialize defaults
            for col in ['total_sheep','total_goats','total_sr','pct_female',
                        'total_births','total_mortality','total_losses',
                        'birth_rate_per_100','mortality_rate_per_100','loss_rate_per_100']:
                if col not in self.df.columns:
                    self.df[col] = 0

            # Columns
            sheep_cols = []
            if 'C3. Number of Rams currently owned (total: at home + away + relatives/friends)' in self.df.columns:
                sheep_cols.append('C3. Number of Rams currently owned (total: at home + away + relatives/friends)')
            if 'C3. Number of Ewes currently owned (total: at home + away + relatives/friends)' in self.df.columns:
                sheep_cols.append('C3. Number of Ewes currently owned (total: at home + away + relatives/friends)')

            goat_cols = []
            if 'C3. Number of Bucks currently owned (total: at home + away + relatives/friends)' in self.df.columns:
                goat_cols.append('C3. Number of Bucks currently owned (total: at home + away + relatives/friends)')
            if 'C3. Number of Does currently owned (total: at home + away + relatives/friends)' in self.df.columns:
                goat_cols.append('C3. Number of Does currently owned (total: at home + away + relatives/friends)')

            birth_cols = []
            for col in ['C4. Number of Rams born in the last 1 month',
                        'C4. Number of Ewes born in the last 1 month',
                        'C4. Number of Bucks born in the last 1 month',
                        'C4. Number of Does born in the last 1 month']:
                if col in self.df.columns: birth_cols.append(col)

            mortality_cols = []
            for col in ['C5. Number of Rams that died in the last 1 month',
                        'C5. Number of Ewes that died in the last 1 month',
                        'C5. Number of Bucks that died in the last 1 month',
                        'C5. Number of Does that died in the last 1 month']:
                if col in self.df.columns: mortality_cols.append(col)

            loss_cols = []
            for col in ['C6. Number of Rams lost/not found or lost to wild animals in the last 1 month',
                        'C6. Number of Ewes lost/not found or lost to wild animals in the last 1 month',
                        'C6. Number of Bucks lost/not found or lost to wild animals in the last 1 month',
                        'C6. Number of Does lost/not found or lost to wild animals in the last 1 month']:
                if col in self.df.columns: loss_cols.append(col)

            # Ensure numeric
            numeric_like = sheep_cols + goat_cols + birth_cols + mortality_cols + loss_cols
            for c in numeric_like:
                if c in self.df.columns:
                    self.df[c] = pd.to_numeric(self.df[c], errors='coerce')

            # Totals
            if sheep_cols:
                self.df['total_sheep'] = self.df[sheep_cols].sum(axis=1, skipna=True).fillna(0)
            if goat_cols:
                self.df['total_goats'] = self.df[goat_cols].sum(axis=1, skipna=True).fillna(0)
            self.df['total_sr'] = self.df['total_sheep'] + self.df['total_goats']

            # Female percentage
            female_sheep = self.df['C3. Number of Ewes currently owned (total: at home + away + relatives/friends)'].fillna(0) if 'C3. Number of Ewes currently owned (total: at home + away + relatives/friends)' in self.df.columns else 0
            female_goats = self.df['C3. Number of Does currently owned (total: at home + away + relatives/friends)'].fillna(0) if 'C3. Number of Does currently owned (total: at home + away + relatives/friends)' in self.df.columns else 0
            valid = self.df['total_sr'] > 0
            self.df.loc[valid, 'pct_female'] = ((pd.Series(female_sheep)[valid] + pd.Series(female_goats)[valid]) /
                                                self.df.loc[valid, 'total_sr'] * 100)

            # Births, mortality, losses
            if birth_cols:
                self.df['total_births'] = self.df[birth_cols].sum(axis=1, skipna=True).fillna(0)
            if mortality_cols:
                self.df['total_mortality'] = self.df[mortality_cols].sum(axis=1, skipna=True).fillna(0)
            if loss_cols:
                self.df['total_losses'] = self.df[loss_cols].sum(axis=1, skipna=True).fillna(0)

            # Rates
            valid = self.df['total_sr'] > 0
            self.df.loc[valid, 'birth_rate_per_100'] = self.df.loc[valid, 'total_births'] / self.df.loc[valid, 'total_sr'] * 100
            self.df.loc[valid, 'mortality_rate_per_100'] = self.df.loc[valid, 'total_mortality'] / self.df.loc[valid, 'total_sr'] * 100
            self.df.loc[valid, 'loss_rate_per_100'] = self.df.loc[valid, 'total_losses'] / self.df.loc[valid, 'total_sr'] * 100

        except Exception as e:
            st.warning(f"Some herd metrics could not be calculated: {str(e)}")
            for col in ['total_sheep','total_goats','total_sr','pct_female',
                        'total_births','total_mortality','total_losses',
                        'birth_rate_per_100','mortality_rate_per_100','loss_rate_per_100']:
                if col not in self.df.columns:
                    self.df[col] = 0
        
        return self.df

    # ---------- P&L helpers ----------
    def _find_count_cols(self, species='sheep'):
        """Best-effort discovery of 'number sold' columns per channel."""
        sp = 'sheep' if species.lower().startswith('sheep') else 'goat'
        # Search by prefixes and species names
        patterns = [
            (r'^E1.*(how many|number).*sheep', 'sheep_kpmd'),
            (r'^E3.*(how many|number).*sheep', 'sheep_non'),
            (r'^E2.*(how many|number).*goat',  'goat_kpmd'),
            (r'^E4.*(how many|number).*goat',  'goat_non'),
        ]
        out = {'sheep_kpmd':None,'sheep_non':None,'goat_kpmd':None,'goat_non':None}
        for col in self.df.columns:
            c = col.lower()
            for pat, key in patterns:
                if re.search(pat, c):
                    out[key] = col
        # If generic "How many did you sell to KPMD last month?" exists per species
        if out['sheep_kpmd'] is None:
            for col in self.df.columns:
                if re.search(r'kpmd.*(how many|number).*sheep', col, re.I):
                    out['sheep_kpmd'] = col
                    break
        if out['sheep_non'] is None:
            for col in self.df.columns:
                if re.search(r'non-?kpm[d]?.*(how many|number).*sheep', col, re.I):
                    out['sheep_non'] = col
                    break
        if out['goat_kpmd'] is None:
            for col in self.df.columns:
                if re.search(r'kpmd.*(how many|number).*goat', col, re.I):
                    out['goat_kpmd'] = col
                    break
        if out['goat_non'] is None:
            for col in self.df.columns:
                if re.search(r'non-?kpm[d]?.*(how many|number).*goat', col, re.I):
                    out['goat_non'] = col
                    break
        return out

    def _price_cols(self):
        return {
            'sheep_kpmd': 'E1c. What was the average price per sheep last month?',
            'sheep_non' : 'E3d. What was the average price per sheep last month?',
            'goat_kpmd' : 'E2c. What was the average price per goat last month?',
            'goat_non'  : 'E4d. What was the average price per goat last month?',
        }

    def compute_pnl(self, focus='all'):
        """
        Build a P&L per household based on available columns. Returns:
        - pnl_df: per-household breakdown
        - summary: dict of totals/means
        """
        df = self.df.copy()

        # Counts & prices
        count_cols = self._find_count_cols()
        price_cols = self._price_cols()

        # Coerce numeric safely
        for key, col in count_cols.items():
            if col and col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        for key, col in price_cols.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Compute revenues for each stream (only if both count & price exist)
        def rev(count_key, price_key):
            ccol = count_cols.get(count_key)
            pcol = price_cols.get(price_key)
            if ccol and (pcol in df.columns) and (ccol in df.columns):
                return (df[ccol].fillna(0) * df[pcol].fillna(0))
            return pd.Series(0, index=df.index)

        rev_sheep_kpmd = rev('sheep_kpmd','sheep_kpmd')
        rev_sheep_non  = rev('sheep_non','sheep_non')
        rev_goat_kpmd  = rev('goat_kpmd','goat_kpmd')
        rev_goat_non   = rev('goat_non','goat_non')

        # Select focus
        if focus == 'sheep':
            total_revenue = rev_sheep_kpmd + rev_sheep_non
        elif focus == 'goats':
            total_revenue = rev_goat_kpmd + rev_goat_non
        else:  # all
            total_revenue = rev_sheep_kpmd + rev_sheep_non + rev_goat_kpmd + rev_goat_non

        # Costs (available fields)
        feed = df.get('Feed_Expenditure', pd.Series(np.nan, index=df.index)).fillna(0)
        vacc_n = df.get('D1a. How many small ruminants were vaccinated in the last month?', pd.Series(0, index=df.index)).fillna(0)
        vacc_c = df.get('D1b. What was the cost of small ruminants vaccination in KSH per animal in the last month?', pd.Series(0, index=df.index)).fillna(0)
        vacc_cost = vacc_n * vacc_c
        treat_cost = df.get('D3b. What was the total cost of treatment in KSH last month?', pd.Series(0, index=df.index)).fillna(0)
        deworm_cost = df.get('D4a. What was the total of cost of deworming in KSH last month?', pd.Series(0, index=df.index)).fillna(0)
        herding_cost = df.get('B3bWhat was the cost of herding per month (Ksh)?', pd.Series(0, index=df.index)).fillna(0)
        other_cost = df.get('B4b. What is the total cost (Ksh)?', pd.Series(0, index=df.index)).fillna(0)  # generic bucket if provided

        total_costs = feed + vacc_cost + treat_cost + deworm_cost + herding_cost + other_cost
        gross_profit = total_revenue - total_costs

        pnl_cols = {
            'Revenue – Sheep (KPMD)': rev_sheep_kpmd,
            'Revenue – Sheep (Other)': rev_sheep_non,
            'Revenue – Goats (KPMD)': rev_goat_kpmd,
            'Revenue – Goats (Other)': rev_goat_non,
            'Total Revenue': total_revenue,
            'Feed Cost': feed,
            'Vaccination Cost': vacc_cost,
            'Treatment Cost': treat_cost,
            'Deworming Cost': deworm_cost,
            'Herding Cost': herding_cost,
            'Other Cost': other_cost,
            'Total Costs': total_costs,
            'Gross Profit': gross_profit
        }

        pnl_df = pd.DataFrame(pnl_cols, index=df.index)
        # append KPMD & County/Subcounty for grouping and display
        pnl_df['kpmd_registered'] = df.get('kpmd_registered', 0)
        pnl_df['County'] = df.get('County', np.nan)
        pnl_df['Subcounty'] = df.get(try_get_col(df, ['Subcounty','Sub-County','Sub County','Sub_County','Sub county']) or 'Subcounty', np.nan)

        # Summary
        mean_row = pnl_df[['Total Revenue','Total Costs','Gross Profit']].mean(numeric_only=True)
        margin = (mean_row['Gross Profit'] / mean_row['Total Revenue'] * 100) if mean_row['Total Revenue'] and not np.isnan(mean_row['Total Revenue']) else np.nan

        # KPMD vs Non-KPMD means
        grp = pnl_df.groupby('kpmd_registered')[['Total Revenue','Total Costs','Gross Profit']].mean().rename(index={1:'KPMD',0:'Non-KPMD'})
        summary = {
            'avg_revenue': mean_row['Total Revenue'],
            'avg_costs': mean_row['Total Costs'],
            'avg_profit': mean_row['Gross Profit'],
            'avg_margin_pct': margin,
            'by_kpmd': grp
        }
        return pnl_df, summary

# -------------------------------------------------
# Dashboard Renderer
# -------------------------------------------------
class DashboardRenderer:
    def __init__(self, data_processor):
        self.dp = data_processor  # keep processor only

    @property
    def df(self):
        """Always return the latest filtered/augmented frame."""
        return self.dp.df
        
    def create_comparison_cards(self, data, metric_col, title, format_str="{:.1f}"):
        """Create side-by-side cards for KPMD vs Non-KPMD comparison"""
        try:
            kpmd_data = data[data['kpmd_registered'] == 1]
            non_kpmd_data = data[data['kpmd_registered'] == 0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                kpmd_value = kpmd_data[metric_col].mean() if len(kpmd_data) > 0 and metric_col in kpmd_data.columns else 0
                st.markdown(f"""
                <div class="metric-card kpmd-card">
                    <h4>KPMD Registered</h4>
                    <h3>{format_str.format(kpmd_value)}</h3>
                    <small>n={len(kpmd_data)}</small>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                non_kpmd_value = non_kpmd_data[metric_col].mean() if len(non_kpmd_data) > 0 and metric_col in non_kpmd_data.columns else 0
                st.markdown(f"""
                <div class="metric-card non-kpmd-card">
                    <h4>Non-KPMD</h4>
                    <h3>{format_str.format(non_kpmd_value)}</h3>
                    <small>n={len(non_kpmd_data)}</small>
                </div>
                """, unsafe_allow_html=True)
        except Exception:
            st.warning(f"Could not create comparison cards for {metric_col}")
    
    # ---------------- Field & Data Outlook ----------------
    def render_field_outlook(self):
        st.header("🧭 Field & Data Outlook")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_submissions = len(self.df)
            st.metric("Total Submissions", total_submissions)
            
        with col2:
            latest_submission = self.df['_submission_time'].max() if '_submission_time' in self.df.columns else None
            st.metric("Latest Submission", latest_submission.strftime("%Y-%m-%d") if pd.notna(latest_submission) else "N/A")
            
        with col3:
            counties_covered = self.df['County'].nunique() if 'County' in self.df.columns else 0
            st.metric("Counties Covered", counties_covered)
            
        with col4:
            kpmd_participants = self.df['kpmd_registered'].sum() if 'kpmd_registered' in self.df.columns else 0
            st.metric("KPMD Participants", kpmd_participants)

        # rCSI
        if 'rCSI_score' in self.df.columns and self.df['rCSI_score'].notna().any():
            avg_rcsi = self.df['rCSI_score'].mean()
            st.metric("Avg rCSI (lower is better)", f"{avg_rcsi:.1f}")
        
        # Submissions over time
        st.subheader("Submissions Over Time")
        if 'month' in self.df.columns:
            monthly_subs = self.df.groupby('month').size().reset_index(name='count')
            fig = px.line(monthly_subs, x='month', y='count', title='Monthly Submissions')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Month data not available for time series analysis")
        
        # County coverage with KPMD comparison
        st.subheader("County Coverage - KPMD vs Non-KPMD")
        if 'County' in self.df.columns and 'kpmd_registered' in self.df.columns:
            county_kpmd = self.df.groupby(['County', 'kpmd_registered']).size().reset_index(name='count')
            county_kpmd['kpmd_status'] = county_kpmd['kpmd_registered'].map({1: 'KPMD', 0: 'Non-KPMD'})
            fig = px.bar(county_kpmd, x='County', y='count', color='kpmd_status', 
                        title='Submissions by County and KPMD Status', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("County or KPMD data not available for this analysis")
        
        # GPS Map (pydeck with numeric RGB columns)
        st.subheader("Household Locations")
        lat_col = '_GPS Coordinates_latitude'
        lon_col = '_GPS Coordinates_longitude'
        if lat_col in self.df.columns and lon_col in self.df.columns:
            map_df = self.df.dropna(subset=[lat_col, lon_col]).copy()
            map_df.rename(columns={lat_col:'lat', lon_col:'lon'}, inplace=True)
            # derive r,g,b scalar columns (avoid lists in cells)
            if 'kpmd_registered' in map_df.columns:
                map_df['r'] = np.where(map_df['kpmd_registered']==1, 31, 214)
                map_df['g'] = np.where(map_df['kpmd_registered']==1,119, 39)
                map_df['b'] = np.where(map_df['kpmd_registered']==1,180, 40)
            else:
                map_df['r'], map_df['g'], map_df['b'] = 160, 160, 160
            
            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                initial_view_state=pdk.ViewState(
                    latitude=map_df['lat'].mean() if len(map_df)>0 else -1.29,
                    longitude=map_df['lon'].mean() if len(map_df)>0 else 36.82,
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
        
        # Calculate comprehensive metrics
        self.dp.calculate_herd_metrics()
        
        tab1, tab2, tab3 = st.tabs(["Herd Composition", "Animal Health Indicators", "SR Productivity Indicators"])
        
        with tab1:
            st.subheader("Herd Structure & Size")
            
            # Average animals owned - KPMD comparison
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
            
            # Percentage female stock
            if 'pct_female' in self.df.columns:
                st.write("**Percentage Female Stock**")
                self.create_comparison_cards(self.df, 'pct_female', 'Female Stock %', '{:.1f}%')
            else:
                st.info("Female stock percentage data not available")
            
            # Herd composition visualization
            if all(col in self.df.columns for col in ['total_sheep','total_goats','kpmd_registered']):
                st.subheader("Herd Composition by KPMD Status")
                try:
                    comp_data = self.df.groupby('kpmd_registered')[['total_sheep', 'total_goats']].mean().reset_index()
                    comp_data['kpmd_status'] = comp_data['kpmd_registered'].map({1: 'KPMD', 0: 'Non-KPMD'})
                    comp_melted = comp_data.melt(id_vars=['kpmd_status'], value_vars=['total_sheep', 'total_goats'],
                                               var_name='Species', value_name='Average Count')
                    comp_melted['Species'] = comp_melted['Species'].map({'total_sheep': 'Sheep', 'total_goats': 'Goats'})
                    
                    fig = px.bar(comp_melted, x='kpmd_status', y='Average Count', color='Species',
                                title='Average Herd Composition by KPMD Status', barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("Herd composition data not available for visualization")
            else:
                st.info("Herd composition data not available for visualization")
        
        with tab2:
            st.subheader("Animal Health Indicators")
            
            # Vaccination coverage
            if 'D1. Did you vaccinate your small ruminants livestock in the last month?' in self.df.columns:
                st.write("**Vaccination Coverage**")
                vacc_data = self.df.copy()
                vacc_data['vaccinated'] = vacc_data['D1. Did you vaccinate your small ruminants livestock in the last month?']
                self.create_comparison_cards(vacc_data, 'vaccinated', 'Vaccination Rate', '{:.1%}')
            else:
                st.info("Vaccination data not available")
            
            # Disease treatment
            if 'D3. Did you treat small ruminants for disease in the last month?' in self.df.columns:
                st.write("**Disease Treatment Rate**")
                treat_data = self.df.copy()
                treat_data['treated'] = treat_data['D3. Did you treat small ruminants for disease in the last month?']
                self.create_comparison_cards(treat_data, 'treated', 'Treatment Rate', '{:.1%}')
            else:
                st.info("Disease treatment data not available")
            
            # Deworming coverage
            if 'D4. Did you deworm your small ruminants livestock last month?' in self.df.columns:
                st.write("**Deworming Coverage**")
                deworm_data = self.df.copy()
                deworm_data['dewormed'] = deworm_data['D4. Did you deworm your small ruminants livestock last month?']
                self.create_comparison_cards(deworm_data, 'dewormed', 'Deworming Rate', '{:.1%}')
            else:
                st.info("Deworming data not available")
            
            # Disease analysis
            st.subheader("Disease Analysis")
            
            # Vaccination diseases
            vacc_disease_cols = [col for col in self.df.columns if 'D1c. What diseases did you vaccinate' in col]
            if vacc_disease_cols:
                try:
                    vacc_disease_data = []
                    for col in vacc_disease_cols:
                        disease_name = col.split('/')[-1] if '/' in col else col
                        for kpmd_status in [0, 1]:
                            subset = self.df[self.df['kpmd_registered'] == kpmd_status]
                            rate = subset[col].mean() * 100 if len(subset) > 0 else 0
                            vacc_disease_data.append({
                                'Disease': disease_name,
                                'Rate': rate,
                                'KPMD_Status': 'KPMD' if kpmd_status == 1 else 'Non-KPMD'
                            })
                    
                    vacc_disease_df = pd.DataFrame(vacc_disease_data)
                    fig = px.bar(vacc_disease_df, x='Disease', y='Rate', color='KPMD_Status',
                                title='Vaccination Diseases by KPMD Status (%)', barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("Vaccination disease data not available")
            else:
                st.info("Vaccination disease data not available")
            
            # Treatment diseases
            treat_disease_cols = [col for col in self.df.columns if 'D3c. What type of disease did you treat' in col]
            if treat_disease_cols:
                try:
                    treat_disease_data = []
                    for col in treat_disease_cols:
                        disease_name = col.split('/')[-1] if '/' in col else col
                        for kpmd_status in [0, 1]:
                            subset = self.df[self.df['kpmd_registered'] == kpmd_status]
                            rate = subset[col].mean() * 100 if len(subset) > 0 else 0
                            treat_disease_data.append({
                                'Disease': disease_name,
                                'Rate': rate,
                                'KPMD_Status': 'KPMD' if kpmd_status == 1 else 'Non-KPMD'
                            })
                    
                    treat_disease_df = pd.DataFrame(treat_disease_data)
                    fig = px.bar(treat_disease_df, x='Disease', y='Rate', color='KPMD_Status',
                                title='Treatment Diseases by KPMD Status (%)', barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("Treatment disease data not available")
            else:
                st.info("Treatment disease data not available")
            
            # Vaccination providers
            if 'D2. Who performed the small ruminants vaccinations in the last month?' in self.df.columns:
                st.subheader("Vaccination Service Providers")
                try:
                    provider_counts = self.df.groupby(['kpmd_registered', 'D2. Who performed the small ruminants vaccinations in the last month?']).size().reset_index(name='count')
                    provider_counts['KPMD_Status'] = provider_counts['kpmd_registered'].map({1: 'KPMD', 0: 'Non-KPMD'})
                    fig = px.bar(provider_counts, x='KPMD_Status', y='count', 
                                color='D2. Who performed the small ruminants vaccinations in the last month?',
                                title='Vaccination Providers by KPMD Status')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("Vaccination provider data not available")
            else:
                st.info("Vaccination provider data not available")
        
        with tab3:
            st.subheader("Small Ruminant Productivity Indicators")
            
            # Birth rates
            if 'birth_rate_per_100' in self.df.columns:
                st.write("**Birth Rates (per 100 head)**")
                self.create_comparison_cards(self.df, 'birth_rate_per_100', 'Birth Rate', '{:.1f}')
            else:
                st.info("Birth rate data not available")
            
            # Mortality rates
            if 'mortality_rate_per_100' in self.df.columns:
                st.write("**Mortality Rates (per 100 head)**")
                self.create_comparison_cards(self.df, 'mortality_rate_per_100', 'Mortality Rate', '{:.1f}')
            else:
                st.info("Mortality rate data not available")
            
            # Loss rates
            if 'loss_rate_per_100' in self.df.columns:
                st.write("**Loss Rates (per 100 head)**")
                self.create_comparison_cards(self.df, 'loss_rate_per_100', 'Loss Rate', '{:.1f}')
            else:
                st.info("Loss rate data not available")
            
            # Combined visualization
            if all(col in self.df.columns for col in ['birth_rate_per_100', 'mortality_rate_per_100', 'loss_rate_per_100', 'kpmd_registered']):
                st.subheader("Productivity Rates by KPMD Status")
                try:
                    productivity_data = self.df.groupby('kpmd_registered')[['birth_rate_per_100', 'mortality_rate_per_100', 'loss_rate_per_100']].mean().reset_index()
                    productivity_data['KPMD_Status'] = productivity_data['kpmd_registered'].map({1: 'KPMD', 0: 'Non-KPMD'})
                    productivity_melted = productivity_data.melt(id_vars=['KPMD_Status'], 
                                                               value_vars=['birth_rate_per_100', 'mortality_rate_per_100', 'loss_rate_per_100'],
                                                               var_name='Metric', value_name='Rate')
                    productivity_melted['Metric'] = productivity_melted['Metric'].map({
                        'birth_rate_per_100': 'Birth Rate',
                        'mortality_rate_per_100': 'Mortality Rate', 
                        'loss_rate_per_100': 'Loss Rate'
                    })
                    
                    fig = px.bar(productivity_melted, x='KPMD_Status', y='Rate', color='Metric',
                                title='Productivity Rates by KPMD Status (per 100 head)', barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("Productivity rate data not available for visualization")
            else:
                st.info("Productivity rate data not available for visualization")
    
    # ---------------- Feed & Fodder ----------------
    def render_feed_fodder(self):
        st.header("🌾 Feed & Fodder")
        
        tab1, tab2, tab3 = st.tabs(["Feed Purchase", "Fodder Production", "Feed Economics"])
        
        with tab1:
            st.subheader("Feed Purchase Patterns")
            
            if 'B5a. Did you purchase fodder in the last 1 month?' in self.df.columns:
                st.write("**Households Purchasing Fodder**")
                purchase_data = self.df.copy()
                purchase_data['purchased'] = (purchase_data['B5a. Did you purchase fodder in the last 1 month?'].apply(yn)).astype(int)
                self.create_comparison_cards(purchase_data, 'purchased', 'Purchase Rate', '{:.1%}')
            else:
                st.info("Fodder purchase data not available")
            
            # Feed sources
            st.subheader("Feed Purchase Sources")
            source_cols = [col for col in self.df.columns if 'B5b. Where did you buy feeds' in col and 'Other' not in col]
            if source_cols:
                source_data = []
                for col in source_cols:
                    source_name = col.split('/')[-1]
                    for kpmd_status in [0, 1]:
                        subset = self.df[self.df['kpmd_registered'] == kpmd_status]
                        rate = (subset[col].astype(str).replace({'1':1,'0':0}).apply(pd.to_numeric, errors='coerce').fillna(0).mean() * 100)
                        source_data.append({
                            'Source': source_name,
                            'Rate': rate,
                            'KPMD_Status': 'KPMD' if kpmd_status == 1 else 'Non-KPMD'
                        })
                
                source_df = pd.DataFrame(source_data)
                fig = px.bar(source_df, x='Source', y='Rate', color='KPMD_Status',
                            title='Feed Purchase Sources by KPMD Status (%)', barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feed source data not available")
        
        with tab2:
            st.subheader("Fodder Production")
            
            if 'B6a. Did you produce any fodder?' in self.df.columns:
                st.write("**Households Producing Fodder**")
                production_data = self.df.copy()
                production_data['produced'] = (production_data['B6a. Did you produce any fodder?'].apply(yn)).astype(int)
                self.create_comparison_cards(production_data, 'produced', 'Production Rate', '{:.1%}')
            else:
                st.info("Fodder production data not available")
            
            if 'B6b. Quantity of feeds harvested in the last 1 month (15 kg bales)?' in self.df.columns:
                st.write("**Average Fodder Harvested (bales)**")
                self.create_comparison_cards(self.df, 'B6b. Quantity of feeds harvested in the last 1 month (15 kg bales)?', 
                                           'Harvested Bales', '{:.1f}')
            else:
                st.info("Fodder harvest quantity data not available")
        
        with tab3:
            st.subheader("Feed Economics")
            
            if 'B5c. What was the price per 15 kg bale in the last 1 month?' in self.df.columns:
                st.write("**Average Purchase Price per Bale (KES)**")
                self.create_comparison_cards(self.df, 'B5c. What was the price per 15 kg bale in the last 1 month?', 
                                           'Price per Bale', 'KES {:.0f}')
            else:
                st.info("Fodder price data not available")
            
            # Feed expenditure
            if 'Feed_Expenditure' in self.df.columns:
                st.write("**Average Feed Expenditure (KES)**")
                self.create_comparison_cards(self.df, 'Feed_Expenditure', 'Feed Expenditure', 'KES {:.0f}')
            else:
                st.info("Feed expenditure data not available")
    
    # ---------------- Offtake Analysis ----------------
    def render_offtake_analysis(self, species='sheep'):
        st.header(f"🚚 Offtake Analysis - {species.title()}")
        
        # Determine column prefixes based on species
        if species.lower() == 'sheep':
            kpmd_prefix = 'E1'
            non_kpmd_prefix = 'E3'
        else:
            kpmd_prefix = 'E2'
            non_kpmd_prefix = 'E4'
        
        price_kpmd_col, price_non_col, age_kpmd_col, age_non_col = _sales_cols(species, kpmd_prefix, non_kpmd_prefix)

        tab1, tab2, tab3 = st.tabs(["Sales Volume", "Price Analysis", "Transaction Details"])
        
        with tab1:
            st.subheader("Sales Volume Analysis")
            
            # KPMD Sales
            kpmd_sold_col = f"{kpmd_prefix}. Did you sell {species} to KPMD off-takers last month?"
            if kpmd_sold_col in self.df.columns:
                st.write(f"**Households Selling to KPMD - {species.title()}**")
                kpmd_sales_data = self.df.copy()
                kpmd_sales_data['sold_kpmd'] = (kpmd_sales_data[kpmd_sold_col].apply(yn)).astype(int)
                self.create_comparison_cards(kpmd_sales_data, 'sold_kpmd', f'KPMD Sales Rate', '{:.1%}')
            else:
                st.info(f"KPMD sales data for {species} not available")
            
            # Non-KPMD Sales
            non_kpmd_sold_col = f"{non_kpmd_prefix}. Did you sell {species} to non-KPMD off-takers last month?"
            if non_kpmd_sold_col in self.df.columns:
                st.write(f"**Households Selling to Non-KPMD - {species.title()}**")
                non_kpmd_sales_data = self.df.copy()
                non_kpmd_sales_data['sold_non_kpmd'] = (non_kpmd_sales_data[non_kpmd_sold_col].apply(yn)).astype(int)
                self.create_comparison_cards(non_kpmd_sales_data, 'sold_non_kpmd', f'Non-KPMD Sales Rate', '{:.1%}')
            else:
                st.info(f"Non-KPMD sales data for {species} not available")
        
        with tab2:
            st.subheader("Price Analysis")
            
            # Price comparison
            price_data = []
            
            # KPMD prices
            if price_kpmd_col in self.df.columns:
                for kpmd_status in [0, 1]:
                    subset = self.df[self.df['kpmd_registered'] == kpmd_status]
                    kpmd_prices = pd.to_numeric(subset[price_kpmd_col], errors='coerce').dropna()
                    if len(kpmd_prices) > 0:
                        price_data.extend([{
                            'Channel': 'KPMD', 
                            'Price': price,
                            'KPMD_Status': 'KPMD Registered' if kpmd_status == 1 else 'Non-KPMD Registered'
                        } for price in kpmd_prices])
            
            # Non-KPMD prices
            if price_non_col in self.df.columns:
                for kpmd_status in [0, 1]:
                    subset = self.df[self.df['kpmd_registered'] == kpmd_status]
                    non_kpmd_prices = pd.to_numeric(subset[price_non_col], errors='coerce').dropna()
                    if len(non_kpmd_prices) > 0:
                        price_data.extend([{
                            'Channel': 'Non-KPMD', 
                            'Price': price,
                            'KPMD_Status': 'KPMD Registered' if kpmd_status == 1 else 'Non-KPMD Registered'
                        } for price in non_kpmd_prices])
            
            if price_data:
                price_df = pd.DataFrame(price_data)
                fig = px.box(price_df, x='Channel', y='Price', color='KPMD_Status',
                            title=f'{species.title()} Price Distribution by Channel and KPMD Registration')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Price data for {species} not available")
        
        with tab3:
            st.subheader("Transaction Details")
            
            # Age at sale analysis
            age_data = []
            
            # KPMD age
            if age_kpmd_col in self.df.columns:
                for kpmd_status in [0, 1]:
                    subset = self.df[self.df['kpmd_registered'] == kpmd_status]
                    kpmd_ages = pd.to_numeric(subset[age_kpmd_col], errors='coerce').dropna()
                    if len(kpmd_ages) > 0:
                        age_data.extend([{
                            'Channel': 'KPMD',
                            'Age': age,
                            'KPMD_Status': 'KPMD Registered' if kpmd_status == 1 else 'Non-KPMD Registered'
                        } for age in kpmd_ages])
            
            # Non-KPMD age
            if age_non_col in self.df.columns:
                for kpmd_status in [0, 1]:
                    subset = self.df[self.df['kpmd_registered'] == kpmd_status]
                    non_kpmd_ages = pd.to_numeric(subset[age_non_col], errors='coerce').dropna()
                    if len(non_kpmd_ages) > 0:
                        age_data.extend([{
                            'Channel': 'Non-KPMD',
                            'Age': age,
                            'KPMD_Status': 'KPMD Registered' if kpmd_status == 1 else 'Non-KPMD Registered'
                        } for age in non_kpmd_ages])
            
            if age_data:
                age_df = pd.DataFrame(age_data)
                fig = px.box(age_df, x='Channel', y='Age', color='KPMD_Status',
                            title=f'{species.title()} Age at Sale by Channel and KPMD Registration')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Age at sale data for {species} not available")

    # ---------------- Payments ----------------
    def render_payments(self):
        st.header("💸 Payment Methods")

        blocks = [
            ('Sheep – KPMD',  'E1g. How were you paid by the KPMD off-takers last  month? [Select all that apply]'),
            ('Goats – KPMD',  'E2g. How were you paid by the KPMD off-takers last  month? [Select all that apply]'),
            ('Sheep – Other', 'E3h. How were you paid by the non-KPMD off-takers last  month? [Select all that apply]'),
            ('Goats – Other', 'E4h. How were you paid by the non-KPMD off-takers last  month? [Select all that apply]')
        ]
        rows=[]
        for label, stem in blocks:
            mobile_cols=[c for c in self.df.columns if c.startswith(stem) and c.endswith('/Mobile payment e.g M-PESA')]
            cash_cols  =[c for c in self.df.columns if c.startswith(stem) and c.endswith('/Cash')]

            mobile_series = None
            cash_series = None

            if mobile_cols:
                mobile_series = (self.df[mobile_cols]
                                 .astype(str).replace({'1':1,'0':0})
                                 .apply(pd.to_numeric, errors='coerce')
                                 .fillna(0).max(axis=1))
            if cash_cols:
                cash_series = (self.df[cash_cols]
                               .astype(str).replace({'1':1,'0':0})
                               .apply(pd.to_numeric, errors='coerce')
                               .fillna(0).max(axis=1))

            if mobile_series is None or cash_series is None:
                if stem in self.df.columns:
                    dummies = one_hot_multiselect(self.df[stem])
                    if mobile_series is None:
                        mobile_token = None
                        for tok in dummies.columns:
                            if tok.lower().startswith('mobile') or 'mpesa' in tok.lower() or 'm-pesa' in tok.lower():
                                mobile_token = tok; break
                        mobile_series = dummies.get(mobile_token, pd.Series(0, index=self.df.index))
                    if cash_series is None:
                        cash_token = None
                        for tok in dummies.columns:
                            if tok.lower().startswith('cash'):
                                cash_token = tok; break
                        cash_series = dummies.get(cash_token, pd.Series(0, index=self.df.index))
                else:
                    mobile_series = mobile_series if mobile_series is not None else pd.Series(0, index=self.df.index)
                    cash_series   = cash_series   if cash_series   is not None else pd.Series(0, index=self.df.index)

            tmp = self.df[['kpmd_registered','County']].copy()
            tmp['block']=label
            tmp['mobile']=mobile_series.fillna(0).clip(0,1).astype(int)
            tmp['cash']=cash_series.fillna(0).clip(0,1).astype(int)
            tmp['both']=((tmp['mobile']==1) & (tmp['cash']==1)).astype(int)
            rows.append(tmp)

        if not rows:
            st.info("No payment method columns found")
            return

        payment = pd.concat(rows, ignore_index=True)

        # Aggregate shares by block and KPMD
        grp = payment.groupby(['block','kpmd_registered'])
        summary = pd.DataFrame({
            'Mobile share': grp['mobile'].mean()*100,
            'Cash share': grp['cash'].mean()*100,
            'Both share': grp['both'].mean()*100
        }).reset_index()
        summary['KPMD Status'] = summary['kpmd_registered'].map({1:'KPMD',0:'Non-KPMD'})

        long = summary.melt(
            id_vars=['block','KPMD Status'],
            value_vars=['Cash share','Mobile share','Both share'],
            var_name='Method',
            value_name='Share'
        )
        fig = px.bar(
            long, x='block', y='Share', color='Method',
            barmode='group', facet_col='KPMD Status',
            title='Payment method mix by channel/species and KPMD'
        )
        st.plotly_chart(fig, use_container_width=True)

        # county split view
        st.subheader("Digital adoption by county (Mobile or Both)")
        county = payment.copy()
        county['digital'] = ((county['mobile']==1) | (county['both']==1)).astype(int)
        county_summary = county.groupby(['County','kpmd_registered'])[['digital']].mean().mul(100).reset_index()
        county_summary['KPMD Status']=county_summary['kpmd_registered'].map({1:'KPMD',0:'Non-KPMD'})
        fig2 = px.bar(county_summary, x='County', y='digital', color='KPMD Status', barmode='group', title='Digital share (%)')
        st.plotly_chart(fig2, use_container_width=True)

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
            ('Vaccination rate','D1. Did you vaccinate your small ruminants livestock in the last month?','{:.0%}'),
            ('Fodder purchase rate','B5a. Did you purchase fodder in the last 1 month?','{:.0%}')
        ]
        colA, colB = st.columns(2)
        A, B = slice_county(left), slice_county(right)
        for (label, col, fmt) in metrics:
            with colA:
                v = pd.to_numeric(A[col].apply(yn) if col in A and A[col].dtype=='O' else A[col] if col in A else np.nan, errors='coerce').mean()
                st.metric(f"{label} — {left}", (fmt.format(v) if pd.notna(v) else "N/A"))
            with colB:
                v = pd.to_numeric(B[col].apply(yn) if col in B and B[col].dtype=='O' else B[col] if col in B else np.nan, errors='coerce').mean()
                st.metric(f"{label} — {right}", (fmt.format(v) if pd.notna(v) else "N/A"))

    # ---------------- Gender Inclusion ----------------
    def render_gender_inclusion(self):
        st.header("♀️ Gender Inclusion")
        
        tab1, tab2, tab3 = st.tabs(["Decision Making", "KPMD Participation", "Income Control"])
        
        with tab1:
            st.subheader("Livestock Sale Decision Making")
            
            if 'G1.Who in the household makes the decision for livestock sale?' in self.df.columns:
                decision_cols = [col for col in self.df.columns if 'G1.Who in the household makes the decision for livestock sale?' in col and 'Other' not in col]
                if decision_cols:
                    decision_data = []
                    for col in decision_cols:
                        role = col.split('/')[-1]
                        for kpmd_status in [0, 1]:
                            subset = self.df[self.df['kpmd_registered'] == kpmd_status]
                            involvement_rate = (subset[col].astype(str).replace({'1':1,'0':0}).astype(float).fillna(0).mean() * 100)
                            decision_data.append({
                                'Role': role,
                                'Involvement_Rate': involvement_rate,
                                'KPMD_Status': 'KPMD' if kpmd_status == 1 else 'Non-KPMD'
                            })
                    
                    decision_df = pd.DataFrame(decision_data)
                    
                    # Women's involvement (Spouse + Daughter)
                    women_roles = ['Spouse', 'Daughter']
                    women_involvement = decision_df[decision_df['Role'].isin(women_roles)].groupby('KPMD_Status')['Involvement_Rate'].mean().reset_index()
                    
                    st.write("**Women's Involvement in Decision Making**")
                    for _, row in women_involvement.iterrows():
                        st.metric(f"{row['KPMD_Status']} - Women Involvement", f"{row['Involvement_Rate']:.1f}%")
                    
                    # Detailed role breakdown
                    fig = px.bar(decision_df, x='Role', y='Involvement_Rate', color='KPMD_Status',
                                title='Decision Making Roles by KPMD Status (%)', barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Decision making role data not available")
            else:
                st.info("Decision making data not available")
        
        with tab2:
            st.subheader("KPMD Participation by Gender")
            
            if 'Gender' in self.df.columns:
                gender_kpmd = pd.crosstab(self.df['Gender'], self.df['kpmd_registered'], normalize='index') * 100
                gender_kpmd = gender_kpmd.reset_index()
                gender_kpmd_melted = gender_kpmd.melt(id_vars=['Gender'], var_name='KPMD_Status', value_name='Percentage')
                gender_kpmd_melted['KPMD_Status'] = gender_kpmd_melted['KPMD_Status'].map({1: 'KPMD', 0: 'Non-KPMD'})
                
                fig = px.bar(gender_kpmd_melted, x='Gender', y='Percentage', color='KPMD_Status',
                            title='KPMD Participation by Gender (%)', barmode='stack')
                st.plotly_chart(fig, use_container_width=True)
                
                # Female-headed households in KPMD
                if 'A17. Are you the head of this household?' in self.df.columns:
                    female_heads = self.df[(self.df['Gender'] == 'Female') & 
                                         (self.df['A17. Are you the head of this household?'].apply(yn) == 1)]
                    if len(female_heads) > 0:
                        female_heads_kpmd = female_heads['kpmd_registered'].mean() * 100
                        st.metric("Female-Headed Households in KPMD", f"{female_heads_kpmd:.1f}%")
                    else:
                        st.info("No female-headed households found in the data")
                else:
                    st.info("Household head data not available")
            else:
                st.info("Gender data not available")
        
        with tab3:
            st.subheader("Income Control and Usage")
            
            if 'G2. Who in the household uses the income from the livestock sale?' in self.df.columns:
                income_cols = [col for col in self.df.columns if 'G2. Who in the household uses the income from the livestock sale?' in col and 'Other' not in col]
                if income_cols:
                    income_data = []
                    for col in income_cols:
                        role = col.split('/')[-1]
                        for kpmd_status in [0, 1]:
                            subset = self.df[self.df['kpmd_registered'] == kpmd_status]
                            control_rate = (subset[col].astype(str).replace({'1':1,'0':0}).astype(float).fillna(0).mean() * 100)
                            income_data.append({
                                'Role': role,
                                'Control_Rate': control_rate,
                                'KPMD_Status': 'KPMD' if kpmd_status == 1 else 'Non-KPMD'
                            })
                    
                    income_df = pd.DataFrame(income_data)
                    
                    # Women's control (Spouse + Daughter)
                    women_roles = ['Spouse', 'Daughter']
                    women_control = income_df[income_df['Role'].isin(women_roles)].groupby('KPMD_Status')['Control_Rate'].mean().reset_index()
                    
                    st.write("**Women's Control Over Livestock Income**")
                    for _, row in women_control.iterrows():
                        st.metric(f"{row['KPMD_Status']} - Women Control", f"{row['Control_Rate']:.1f}%")
                    
                    # Detailed control breakdown
                    fig = px.bar(income_df, x='Role', y='Control_Rate', color='KPMD_Status',
                                title='Income Control Roles by KPMD Status (%)', barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Income control role data not available")
            else:
                st.info("Income control data not available")
    
    # ---------------- Climate Impact ----------------
    def render_climate_impact(self):
        st.header("🌦️ Climate Impact")
        
        tab1, tab2, tab3 = st.tabs(["Adaptation Measures", "Barriers to Adaptation", "Climate Resilience"])
        
        with tab1:
            st.subheader("Adaptation Measures")
            
            if 'J1. Have you made any adaptation measures last month due to drought shocks?' in self.df.columns:
                st.write("**Households Implementing Adaptation Measures**")
                adaptation_data = self.df.copy()
                adaptation_data['adapted'] = (adaptation_data['J1. Have you made any adaptation measures last month due to drought shocks?'].apply(yn)).astype(int)
                self.create_comparison_cards(adaptation_data, 'adapted', 'Adaptation Rate', '{:.1%}')
            else:
                st.info("Climate adaptation data (J1) not available")
            
            # ---- Adaptation strategies (J2) ----
            j2_stem = 'J2. Which adapatations measures are you using?'
            strategy_cols = [col for col in self.df.columns if j2_stem in col and 'Other' not in col and col != j2_stem]
            if strategy_cols:
                st.subheader("Adaptation Strategies by KPMD Status")
                strategy_data = []
                for col in strategy_cols:
                    strategy_name = col.split('/')[-1]
                    for kpmd_status in [0, 1]:
                        subset = self.df[self.df['kpmd_registered'] == kpmd_status]
                        rate = (subset[col].astype(str).replace({'1':1,'0':0}).astype(float).fillna(0).mean() * 100)
                        strategy_data.append({
                            'Strategy': strategy_name,
                            'Usage_Rate': rate,
                            'KPMD_Status': 'KPMD' if kpmd_status == 1 else 'Non-KPMD'
                        })
                strategy_df = pd.DataFrame(strategy_data)
                fig = px.bar(strategy_df, x='Strategy', y='Usage_Rate', color='KPMD_Status',
                            title='Adaptation Strategies by KPMD Status (%)', barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            elif j2_stem in self.df.columns:
                st.subheader("Adaptation Strategies by KPMD Status")
                dummies = one_hot_multiselect(self.df[j2_stem])
                if not dummies.empty:
                    tmp = pd.concat([self.df[['kpmd_registered']], dummies], axis=1)
                    long = tmp.melt(id_vars=['kpmd_registered'], var_name='Strategy', value_name='flag')
                    agg = long.groupby(['Strategy','kpmd_registered'])['flag'].mean().mul(100).reset_index()
                    agg['KPMD_Status'] = agg['kpmd_registered'].map({1:'KPMD',0:'Non-KPMD'})
                    fig = px.bar(agg, x='Strategy', y='flag', color='KPMD_Status', title='Adaptation Strategies by KPMD Status (%)', barmode='group')
                    fig.update_yaxes(title='Usage_Rate')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Adaptation strategy data (J2) not available")
            else:
                st.info("Adaptation strategy data (J2) not available")
        
        with tab2:
            st.subheader("Barriers to Adaptation")
            j3_stem = 'J3. Why not?'
            barrier_cols = [col for col in self.df.columns if j3_stem in col and 'Other' not in col and col != j3_stem]
            base = self.df
            if 'J1. Have you made any adaptation measures last month due to drought shocks?' in base.columns:
                base = base[base['J1. Have you made any adaptation measures last month due to drought shocks?'].apply(yn) == 0]
            
            if barrier_cols:
                barrier_data = []
                for col in barrier_cols:
                    barrier_name = col.split('/')[-1]
                    for kpmd_status in [0, 1]:
                        subset = base[base['kpmd_registered'] == kpmd_status]
                        if len(subset) > 0:
                            rate = (subset[col].astype(str).replace({'1':1,'0':0}).astype(float).fillna(0).mean() * 100)
                            barrier_data.append({
                                'Barrier': barrier_name,
                                'Rate': rate,
                                'KPMD_Status': 'KPMD' if kpmd_status == 1 else 'Non-KPMD'
                            })
                if barrier_data:
                    barrier_df = pd.DataFrame(barrier_data)
                    fig = px.bar(barrier_df, x='Barrier', y='Rate', color='KPMD_Status',
                                title='Barriers to Adaptation by KPMD Status (%)', barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No barrier data available for non-adapting households")
            elif j3_stem in self.df.columns:
                dummies = one_hot_multiselect(base[j3_stem])
                if not dummies.empty:
                    tmp = pd.concat([base[['kpmd_registered']], dummies], axis=1)
                    long = tmp.melt(id_vars=['kpmd_registered'], var_name='Barrier', value_name='flag')
                    agg = long.groupby(['Barrier','kpmd_registered'])['flag'].mean().mul(100).reset_index()
                    agg['KPMD_Status'] = agg['kpmd_registered'].map({1:'KPMD',0:'Non-KPMD'})
                    fig = px.bar(agg, x='Barrier', y='flag', color='KPMD_Status', title='Barriers to Adaptation by KPMD Status (%)', barmode='group')
                    fig.update_yaxes(title='Rate')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No barrier data available for non-adapting households")
            else:
                st.info("Barriers to adaptation data (J3) not available")
        
        with tab3:
            st.subheader("Climate Resilience Indicators")
            
            if 'kpmd_registered' in self.df.columns:
                st.write("**KPMD Participation as Resilience Factor**")
                kpmd_participation = self.df['kpmd_registered'].mean() * 100
                st.metric("Overall KPMD Participation Rate", f"{kpmd_participation:.1f}%")
            else:
                st.info("KPMD participation data not available")
            
            st.info("Additional climate resilience indicators will be displayed as data becomes available")

    # ---------------- P&L ----------------
    def render_pnl(self, ruminant_focus='all'):
        st.header("📒 SR Profit & Loss (P&L)")

        focus_label = {'all':'All SR','sheep':'Sheep','goats':'Goats'}.get(ruminant_focus, 'All SR')
        st.caption(f"Ruminant focus: **{focus_label}** (set in Global Filters)")

        pnl_df, summary = self.dp.compute_pnl(focus=ruminant_focus)

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Avg Revenue (KES)", f"{summary['avg_revenue']:.0f}" if pd.notna(summary['avg_revenue']) else "N/A")
        with k2:
            st.metric("Avg Costs (KES)", f"{summary['avg_costs']:.0f}" if pd.notna(summary['avg_costs']) else "N/A")
        with k3:
            st.metric("Avg Gross Profit (KES)", f"{summary['avg_profit']:.0f}" if pd.notna(summary['avg_profit']) else "N/A")
        with k4:
            st.metric("Avg Gross Margin (%)", f"{summary['avg_margin_pct']:.1f}%" if pd.notna(summary['avg_margin_pct']) else "N/A")

        # Breakdown chart
        st.subheader("Average P&L Breakdown")
        avg_breakdown = pnl_df[['Revenue – Sheep (KPMD)', 'Revenue – Sheep (Other)',
                                'Revenue – Goats (KPMD)', 'Revenue – Goats (Other)',
                                'Feed Cost','Vaccination Cost','Treatment Cost','Deworming Cost','Herding Cost','Other Cost']].mean(numeric_only=True)
        # Filter out zero-only categories to keep charts clean
        avg_breakdown = avg_breakdown[avg_breakdown.abs() > 0]
        if not avg_breakdown.empty:
            plot_df = avg_breakdown.reset_index()
            plot_df.columns = ['Item','KES']
            plot_df['Type'] = np.where(plot_df['Item'].str.contains('Revenue'), 'Revenue', 'Cost')
            fig = px.bar(plot_df, x='Item', y='KES', color='Type', title='Average Revenues & Costs')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No revenue/cost data available to plot.")

        # KPMD vs Non-KPMD averages
        st.subheader("KPMD vs Non-KPMD (Averages)")
        if isinstance(summary.get('by_kpmd'), pd.DataFrame) and not summary['by_kpmd'].empty:
            st.dataframe(summary['by_kpmd'].round(0))
        else:
            st.info("No KPMD/Non-KPMD breakdown available.")

        # Detailed table
        st.subheader("Household-level P&L (filtered sample)")
        show_cols = ['County','Subcounty','Total Revenue','Total Costs','Gross Profit',
                     'Feed Cost','Vaccination Cost','Treatment Cost','Deworming Cost','Herding Cost','Other Cost','kpmd_registered']
        show_cols = [c for c in show_cols if c in pnl_df.columns]
        st.dataframe(pnl_df[show_cols].round(0))

# -------------------------------------------------
# Main App
# -------------------------------------------------
def main():
    st.title("APMT Program Insights")
    st.markdown('<div class="main-header">Pastoral Market Transformation Monitoring</div>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload APMT Data CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success(f"Data loaded successfully: {len(df)} records")
            
            # Initialize data processor and renderer
            processor = APMTDataProcessor(df)
            renderer = DashboardRenderer(processor)
            
            # Sidebar filters - use processor.df consistently
            st.sidebar.header("Global Filters")

            # Ruminant Focus (affects P&L primarily, and info caption)
            focus_choice = st.sidebar.selectbox("Ruminant Focus", ['All SR','Sheep','Goats'])
            focus_key = {'All SR':'all', 'Sheep':'sheep', 'Goats':'goats'}[focus_choice]
            
            # County filter
            if 'County' in processor.df.columns:
                counties = ['All'] + sorted(processor.df['County'].dropna().unique())
                selected_county = st.sidebar.selectbox("Select County", counties)
                if selected_county != 'All':
                    processor.df = processor.df[processor.df['County'] == selected_county]

                # Subcounty filter (appears after County selection)
                subcounty_col = processor.subcounty_col
                if subcounty_col and selected_county != 'All':
                    sub_df = processor.df
                    subs = ['All'] + sorted(sub_df[subcounty_col].dropna().unique())
                    selected_sub = st.sidebar.selectbox("Select Subcounty", subs)
                    if selected_sub != 'All':
                        processor.df = processor.df[processor.df[subcounty_col] == selected_sub]
            else:
                selected_county = 'All'
            
            # KPMD filter
            kpmd_filter = st.sidebar.selectbox("KPMD Status", ['All', 'Registered', 'Not Registered'])
            if kpmd_filter == 'Registered':
                processor.df = processor.df[processor.df['kpmd_registered'] == 1]
            elif kpmd_filter == 'Not Registered':
                processor.df = processor.df[processor.df['kpmd_registered'] == 0]
            
            # Gender filter
            if 'Gender' in processor.df.columns:
                genders = ['All'] + sorted(processor.df['Gender'].dropna().unique())
                selected_gender = st.sidebar.selectbox("Select Gender", genders)
                if selected_gender != 'All':
                    processor.df = processor.df[processor.df['Gender'] == selected_gender]
            
            # Date range filter
            if 'int_date' in processor.df.columns:
                try:
                    min_date = pd.to_datetime(processor.df['int_date']).min()
                    max_date = pd.to_datetime(processor.df['int_date']).max()
                    date_range = st.sidebar.date_input(
                        "Select Date Range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date
                    )
                    if isinstance(date_range, tuple) and len(date_range) == 2:
                        processor.df = processor.df[
                            (processor.df['int_date'] >= pd.to_datetime(date_range[0])) & 
                            (processor.df['int_date'] <= pd.to_datetime(date_range[1]))
                        ]
                except Exception:
                    st.sidebar.warning("Date filtering not available")
            
            # Optional ruminant focus note in sidebar
            st.sidebar.caption(f"Ruminant Focus is currently **{focus_choice}**. This affects the P&L page.")

            # Data preview (post-filters)
            with st.expander("Data Preview"):
                st.write(f"Columns: {list(processor.df.columns)}")
                st.dataframe(processor.df.head())
            
            # Navigation
            st.sidebar.header("Dashboard Navigation")
            page = st.sidebar.selectbox(
                "Select Dashboard Page",
                ["Field Outlook", "Pastoral Productivity", "Feed & Fodder", 
                 "Sheep Offtake", "Goat Offtake", "Payments", "County Comparator", "Gender Inclusion", "Climate Impact", "SR P&L"]
            )
            
            # Render selected page
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
            elif page == "SR P&L":
                renderer.render_pnl(ruminant_focus=focus_key)
                
            # Data download
            st.sidebar.header("Data Export")
            csv = processor.df.to_csv(index=False)
            st.sidebar.download_button(
                label="Download Filtered Data (CSV)",
                data=csv,
                file_name=f"apmt_filtered_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
                
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.info("Please check that your CSV file matches the expected APMT data format")
    else:
        st.info("👆 Please upload an APMT data CSV file to begin")
        
        # Show sample data structure expectations
        st.subheader("Expected Data Structure")
        st.markdown("""
        Your CSV should contain these key columns:
        - `Household ID`, `County`, `Gender`, **Subcounty**
        - `A8. Are you registered to KPMD programs?`
        - `Selection of the household` (Treatment/Control)
        - `int_date`, `_submission_time`
        - GPS coordinates columns
        - Sections B, C, D, E, G, J variables as described in the questionnaire
        - Optional: `rCSI` (or the 5 component questions to compute it)

        **Note:** The dashboard will handle missing columns gracefully and show informative messages.
        """)

if __name__ == "__main__":
    main()