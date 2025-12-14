# apmt_dashboard/utils/stats.py
import pandas as pd
import numpy as np

def _design_matrix(df, y_col, group_col=None, controls=None, dropna=True):
    """Create design matrix for LSMeans calculation."""
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
    """Calculate OLS beta coefficients."""
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta
    except Exception:
        return None

def lsmeans_by_group(df, y_col, group_col, controls=None):
    """Calculate Least Squares Means by group with controls."""
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
    """Format LSMean note for HTML display."""
    try:
        return f'<div class="lsm-note">LSMean (adjusted): {lsm}</div>'
    except Exception:
        return ""