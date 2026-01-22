# apmt_dashboard/utils/helpers.py
import pandas as pd
import numpy as np
import re

_YES = {"yes", "y", "true", "1", "t", "aye", "yeah"}
_NO = {"no", "n", "false", "0", "f", "nah"}


def yn(x):
    """
    Convert common Yes/No encodings to 1/0 with NA-safe behavior.

    IMPORTANT:
    - Returns pd.NA when the value is missing/blank/unknown (instead of defaulting to 0).
    - This prevents silently treating missing responses as "No".
    """
    if pd.isna(x):
        return pd.NA

    # Numeric (including numpy types)
    if isinstance(x, (int, float, np.integer, np.floating)):
        # Treat exactly 1 as Yes, exactly 0 as No; other values -> NA
        try:
            v = float(x)
        except Exception:
            return pd.NA
        if v == 1.0:
            return 1
        if v == 0.0:
            return 0
        return pd.NA

    # Bool
    if isinstance(x, bool):
        return 1 if x else 0

    s = str(x).strip().lower()

    # Empty strings -> NA
    if s == "" or s in {"nan", "none", "null"}:
        return pd.NA

    # Direct matches
    if s in _YES:
        return 1
    if s in _NO:
        return 0

    # Prefix matches (handles "yes, ..." and "no, ...")
    if s.startswith("yes"):
        return 1
    if s.startswith("no"):
        return 0

    # Anything else unknown -> NA
    return pd.NA


def to_num(series: pd.Series) -> pd.Series:
    """
    Robust numeric coercion:
    - Removes commas and spaces
    - Handles percent strings like '12.5%' by converting to 12.5 (not 0.125)
    """
    if series is None:
        return pd.Series(dtype="float64")

    s = series.astype(str).str.strip()

    # Treat common null-like tokens as NaN
    s = s.replace({"nan": np.nan, "None": np.nan, "none": np.nan, "NULL": np.nan, "null": np.nan})

    # Remove thousands separators and spaces
    s = s.str.replace(",", "", regex=False).str.replace(" ", "", regex=False)

    # Remove trailing percent sign (keep numeric scale as written)
    s = s.str.replace("%", "", regex=False)

    return pd.to_numeric(s, errors="coerce")


def one_hot_multiselect(series: pd.Series) -> pd.DataFrame:
    """
    One-hot encode multi-select strings split by common delimiters.
    Returns an Int64 (nullable) DataFrame.
    """
    if series is None or series.dropna().empty:
        return pd.DataFrame(index=getattr(series, "index", None))

    tokens_list = []
    pattern = re.compile(r"\s*\|\s*|\s*;\s*|\s*,\s*|\s*/\s*|\s{2,}")

    for val in series.fillna(""):
        if not isinstance(val, str):
            tokens_list.append([])
            continue
        tokens = [t.strip() for t in pattern.split(val) if t.strip() != ""]
        tokens_list.append(tokens)

    uniques = sorted({tok for toks in tokens_list for tok in toks})
    if not uniques:
        return pd.DataFrame(index=series.index)

    data = {tok: [1 if tok in toks else 0 for toks in tokens_list] for tok in uniques}
    return pd.DataFrame(data, index=series.index).astype("Int64")


def coalesce_first(df, candidates):
    if not isinstance(df, pd.DataFrame):
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _iqr_outlier_mask(s: pd.Series):
    """Identify outliers using the IQR method."""
    s = pd.to_numeric(s, errors="coerce")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=s.index)
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (s < lower) | (s > upper)
