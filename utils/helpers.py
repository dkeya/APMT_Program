# apmt_dashboard/utils/helpers.py
import pandas as pd
import numpy as np
import re

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

def _iqr_outlier_mask(s: pd.Series):
    """Identify outliers using IQR method."""
    s = pd.to_numeric(s, errors='coerce')
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=s.index)
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    return (s < lower) | (s > upper)