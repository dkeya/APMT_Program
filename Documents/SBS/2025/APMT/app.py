from __future__ import annotations
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import pydeck as pdk

# =========================================================
# ---------------------- SETTINGS -------------------------
# =========================================================

st.set_page_config(
    page_title="APMT Program Insights",
    page_icon="🐐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Make Altair faster on larger data
alt.data_transformers.disable_max_rows()

# Project constants (edit here if the instrument changes)
COUNTY_COL = "County"
SUBCOUNTY_COL = "Sub county"
HH_ID_COL = "Household ID"
DATE_COL = "int_date"     # has +03:00 suffix in your sample
GENDER_COL = "Gender"

GPS_LAT_COL = "_GPS Coordinates_latitude"
GPS_LON_COL = "_GPS Coordinates_longitude"
GPS_ALT_COL = "_GPS Coordinates_altitude"
GPS_PREC_COL = "_GPS Coordinates_precision"

# Ownership (C-section)
SHEEP_RAMS = "C3. Number of Rams currently owned (total: at home + away + relatives/friends)"
SHEEP_EWES = "C3. Number of Ewes currently owned (total: at home + away + relatives/friends)"
GOAT_BUCKS = "C3. Number of Bucks currently owned (total: at home + away + relatives/friends)"
GOAT_DOES = "C3. Number of Does currently owned (total: at home + away + relatives/friends)"

# Births/Deaths/Loss/Slaughter
R_BORN, R_DIED, R_LOST, R_SLAUGHT = (
    "C4. Number of Rams born in the last 1 month",
    "C5. Number of Rams that died in the last 1 month",
    "C6. Number of Rams lost/not found or lost to wild animals in the last 1 month",
    "C7. Number of Rams slaughtered for home consumption in the last 1 month",
)
E_BORN, E_DIED, E_LOST, E_SLAUGHT = (
    "C4. Number of Ewes born in the last 1 month",
    "C5. Number of Ewes that died in the last 1 month",
    "C6. Number of Ewes lost/not found or lost to wild animals in the last 1 month",
    "C7. Number of Ewes slaughtered for home consumption in the last 1 month",
)
B_BORN, B_DIED, B_LOST, B_SLAUGHT = (
    "C4. Number of Bucks born in the last 1 month",
    "C5. Number of Bucks that died in the last 1 month",
    "C6. Number of Bucks lost/not found or lost to wild animals in the last 1 month",
    "C7. Number of Bucks slaughtered for home consumption in the last 1 month",
)
D_BORN, D_DIED, D_LOST, D_SLAUGHT = (
    "C4. Number of Does born in the last 1 month",
    "C5. Number of Does that died in the last 1 month",
    "C6. Number of Does lost/not found or lost to wild animals in the last 1 month",
    "C7. Number of Does slaughtered for home consumption in the last 1 month",
)

MEN_OWN = "C8. How many small ruminants are owned by Men in the last 1 month?"
WOMEN_OWN = "C9. How many small ruminants are owned & controlled by Women in the last 1 month?"
WATER_ACCESS = "C10. Do you have access to water for your livestock?"

# KPMD registration and program exposure
KPMD_REG = "A8. Are you registered to KPMD programs?"

# Health (D-section)
VACC_ANY = "D1. Did you vaccinate your small ruminants livestock in the last month?"
VACC_COUNT = "D1a. How many small ruminants were vaccinated in the last month?"
DEWORM_ANY = "D4. Did you deworm your small ruminants livestock last month?"
TREAT_ANY = "D3. Did you treat small ruminants for disease in the last month?"
TREAT_COUNT = "D3a. How many small ruminants livestock were sick and treated in the last one month?"

# Sales (E-section)
S_SOLD_KPMD = "E1. Did you sell sheep to KPMD off-takers last  month?"
S_N_KPMD = "E1a. How many sheep did you sell to KPMD off-takers  last month?"
S_PRICE_KPMD = "E1c. What was the average price per sheep last month?"
S_TRANS_KPMD = "E1h. What was the transport cost to  the market per sheep last month?"

G_SOLD_KPMD = "E2. Did you sell goats to KPMD off-takers last  month?"
G_N_KPMD = "E2a. How many goats did you sell to KPMD off-takers  last month?"
G_PRICE_KPMD = "E2c. What was the average price per goat last month?"
G_TRANS_KPMD = "E2h. What was the transport cost to  the market per goat last month?"

S_SOLD_NON = "E3. Did you sell sheep to non-KPMD off-takers last  month?"
S_N_NON = "E3b. How many sheep did you sell to non-KPMD off-takers  last month?"
S_PRICE_NON = "E3d. What was the average price per sheep last month?"
S_TRANS_NON = "E3i. What was the transport cost to  the market per sheep last month?"

G_SOLD_NON = "E4. Did you sell goats to non-KPMD off-takers last  month?"
G_N_NON = "E4b. How many goats did you sell to non-KPMD off-takers  last month?"
G_PRICE_NON = "E4d. What was the average price per goat last month?"
G_TRANS_NON = "E4i. What was the transport cost to  the market per goat last month?"

# Market access (F)
DIST_KM = "F1. How far did you sell small ruminants  livestock last month in Kilometres?"
PRICE_INFO = "F2. Did you get information about livestock prices prior to selling in the last three months?"

# Gender decision & control (G)
G1_MULTI = "G1.Who in the household makes the decision for livestock sale?  [Select all that apply]"
G2_MULTI = "G2. Who in the household uses the income from the livestock sale? [Select all that apply]"
G_ROLES = ["Head", "Spouse", "Daughter", "Son", "Other member"]

# Credit & Insurance (H)
CREDIT_30D = "H1. Have you or any member of your household  applied for credit  during the last 30 days ? (either formal or informal institutions )"
CREDIT_TIMES = "H3. Number of times  borrowed in the 30 days"
CREDIT_VALUE = "H4. Average value of credit borrowed in  Ksh"
INS_30D = "H5. Have you ever accessed livestock insurance  in the last 30 days?"
INS_COUNT = "Number of your small ruminants livestock  insured?"
INS_PREMIUM = "Cost of insurance premiums in Ksh?"

# Food security (I) - rCSI items
RCSI_12 = "12. Did your household rely on less preferred or less expensive foods?"
RCSI_13 = "13. Did your household borrow food, or rely on help from friends/relatives?"
RCSI_14 = "14. Did your household limit portion sizes at mealtimes?"
RCSI_15 = "15. Did your household reduce the number of meals eaten in a day?"
RCSI_I6 = "I6. Did adults in your household restrict their consumption so that small children could eat?"
WORRY = "I1. In the past 30 days, did you worry that your household would not have enough food?"

# ---- rCSI band order (centralized) ----
RCSI_ORDER = ["Minimal", "Stress", "Crisis", "Emergency", "Catastrophic"]

# Adaptation (J)
ADAPT_ANY = "J1. Have you made any adaptation measures last month due to drought  shocks?"
J2_MULTI = "J2. Which adapatations measures are you using?"
J2_OPTIONS = [
    "Increased mobility (distance & frequency)",
    "Purchase of fodder",
    "Change in water management",
    "Diversify to other livelihoods",
    "Banking livestock assets (sell and bank saving)",
    "Herd destocking and restocking",
    "Reduce herd size",
    "Use stored fodder",
    "Other",
]

# =========================================================
# ---------------------- UTILITIES ------------------------
# =========================================================

def yes_no_to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map({"yes": True, "no": False})

def _normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize smart punctuation and spacing in object columns."""
    if df.empty:
        return df
    replace_map = {
        "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\xa0": " ",
    }
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols):
        df[obj_cols] = df[obj_cols].apply(
            lambda s: s.astype(str)
                       .replace(replace_map, regex=True)
                       .str.replace(r"\s+", " ", regex=True)
                       .str.strip()
        )
    return df

def _safe_num_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return numeric series or zeros aligned to index if col missing."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(0, index=df.index, dtype=float)

def _fmt_median_int(series: pd.Series, default: int = 0) -> int:
    """Format median safely as int; returns default if NaN/empty."""
    if series is None or len(series) == 0:
        return default
    med = pd.to_numeric(series, errors="coerce").median(skipna=True)
    return default if pd.isna(med) else int(med)

# ===== STRICT, DETERMINISTIC DATA LOADING (no guessing) =====

DATA_BASENAME = "APMT_data"                 # exact base name expected
DATA_EXT_PRIORITY = [".xlsx", ".csv"]       # tried in this order, same folder as the app

# Minimal, explicit aliasing for core fields only (no fuzzy matching)
_HEADER_ALIASES = {
    "County": ["County"],
    "Sub county": ["Sub county", "Sub-County", "Subcounty"],
    "Household ID": ["Household ID", "HH ID", "HHID"],
    "int_date": ["int_date", "Interview date", "Interview Date", "int-date"],
}

# strict minimum to render filters/KPIs
_REQUIRED_FOR_APP = [COUNTY_COL, SUBCOUNTY_COL]

def _resolve_data_path() -> Path:
    """Return the absolute path to APMT_data with a whitelisted extension, or raise."""
    folder = Path(__file__).resolve().parent
    for ext in DATA_EXT_PRIORITY:
        p = (folder / f"{DATA_BASENAME}{ext}")
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Expected '{DATA_BASENAME}.xlsx' or '{DATA_BASENAME}.csv' next to this script."
    )

def _apply_header_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trim/normalize spacing, then map explicit aliases -> canonical names.
    Tight and reviewable: only the entries above.
    """
    clean_cols = []
    for c in df.columns:
        cc = str(c).replace("\xa0", " ")
        cc = re.sub(r"\s+", " ", cc).strip()
        clean_cols.append(cc)
    df.columns = clean_cols

    rename_map = {}
    current = set(df.columns)
    for canonical, variants in _HEADER_ALIASES.items():
        if canonical in current:
            continue
        for v in variants:
            if v in current:
                rename_map[v] = canonical
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def _validate_required(df: pd.DataFrame):
    missing = [c for c in _REQUIRED_FOR_APP if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
            + ". Please ensure your file uses the canonical headers."
        )

# ---- Named waves (explicit) ----
WAVE_LABELS = [
    ("Baseline", "2025-06-01", "2025-07-31"),
    ("Midline",  "2025-08-01", "2025-09-30"),
    ("Endline",  "2025-10-01", "2025-12-31"),
]

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """
    Deterministic loader:
    - Look for APMT_data.xlsx (preferred) or APMT_data.csv (same folder).
    - Read once with a clear code path (no retries/sniffing).
    - Apply explicit header aliases and validate required columns.
    - Parse dates and coerce numeric fields used later.
    """
    path = _resolve_data_path()

    # Read (one clear path per extension)
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="cp1252", engine="python")
    else:
        raise ValueError(f"Unsupported extension: {path.suffix}")

    # Light text normalization so filters/contains behave
    df = _normalize_text_columns(df)

    # Canonicalize headers and validate
    df = _apply_header_aliases(df)
    _validate_required(df)

    # Parse date (keeps timezone suffix if present)
    if DATE_COL in df.columns:
        df["int_date_parsed"] = pd.to_datetime(
            df[DATE_COL].astype(str).str.replace(" ", "T", regex=False),
            errors="coerce",
            utc=False,  # we will just remove any tz below
        )
        # If tz-aware, strip timezone to make tz-naive
        try:
            if getattr(df["int_date_parsed"].dt, "tz", None) is not None:
                df["int_date_parsed"] = df["int_date_parsed"].dt.tz_localize(None)
        except Exception:
            pass

    # Month string (still fine with tz removed)
    df["wave_month"] = df["int_date_parsed"].dt.to_period("M").astype(str)

    # Named wave labels (safe now that int_date_parsed is tz-naive)
    def _wave_name(d):
        if pd.isna(d):
            return None
        for name, start, end in WAVE_LABELS:
            if pd.to_datetime(start) <= d <= pd.to_datetime(end):
                return name
        return "Unlabeled"

    df["wave_label"] = df["int_date_parsed"].apply(_wave_name)

    # Coerce numeric fields actually used later
    num_cols = [
        SHEEP_RAMS,SHEEP_EWES,GOAT_BUCKS,GOAT_DOES,
        R_BORN,R_DIED,R_LOST,R_SLAUGHT,
        E_BORN,E_DIED,E_LOST,E_SLAUGHT,
        B_BORN,B_DIED,B_LOST,B_SLAUGHT,
        D_BORN,D_DIED,D_LOST,D_SLAUGHT,
        MEN_OWN,WOMEN_OWN,
        VACC_COUNT, TREAT_COUNT,
        S_N_KPMD, S_PRICE_KPMD, S_TRANS_KPMD,
        G_N_KPMD, G_PRICE_KPMD, G_TRANS_KPMD,
        S_N_NON,  S_PRICE_NON,  S_TRANS_NON,
        G_N_NON,  G_PRICE_NON,  G_TRANS_NON,
        DIST_KM, CREDIT_TIMES, CREDIT_VALUE, INS_COUNT, INS_PREMIUM,
        GPS_ALT_COL, GPS_PREC_COL, GPS_LAT_COL, GPS_LON_COL
    ]
    for c in [c for c in num_cols if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Standardize Yes/No to booleans for key indicators
    for col in [KPMD_REG, VACC_ANY, DEWORM_ANY, TREAT_ANY, S_SOLD_KPMD, G_SOLD_KPMD,
                S_SOLD_NON, G_SOLD_NON, PRICE_INFO, CREDIT_30D, INS_30D, ADAPT_ANY,
                WATER_ACCESS, WORRY]:
        if col in df.columns:
            df[col + "_bool"] = df[col].astype(str).str.strip().str.lower().map({"yes": True, "no": False})

    return df

# =========================================================
# ---------------------- LOAD DATA ------------------------
# =========================================================

df_raw = load_data()
df = df_raw.copy()

# =========================================================
# ------------------- DERIVED FIELDS ----------------------
# =========================================================

def compute_rcsi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute reduced CSI (rCSI) using standard weights."""
    for col in [RCSI_12, RCSI_13, RCSI_14, RCSI_15, RCSI_I6]:
        if col in df.columns:
            df[col + "_n"] = df[col].apply(lambda x: (
                0 if not isinstance(x, str) else
                0 if x.strip().lower().startswith("never") else
                2 if x.strip().lower().startswith("rarely") else
                7 if x.strip().lower().startswith("sometimes") else
                15 if x.strip().lower().startswith("often") else
                26 if x.strip().lower().startswith("very often") else 0
            ))
        else:
            df[col + "_n"] = 0
    df["rCSI"] = (
        df.get(RCSI_12 + "_n", 0) * 1
        + df.get(RCSI_13 + "_n", 0) * 2
        + df.get(RCSI_14 + "_n", 0) * 1
        + df.get(RCSI_15 + "_n", 0) * 1
        + df.get(RCSI_I6 + "_n", 0) * 3
    )
    df["rCSI_band"] = pd.cut(
        df["rCSI"], bins=[-0.1, 3, 9, 18, 30, 1000],
        labels=RCSI_ORDER
    )
    return df

def compute_herd(df: pd.DataFrame) -> pd.DataFrame:
    df["sheep_owned"] = df.get(SHEEP_RAMS, 0) + df.get(SHEEP_EWES, 0)
    df["goats_owned"] = df.get(GOAT_BUCKS, 0) + df.get(GOAT_DOES, 0)
    df["sr_total_owned"] = df["sheep_owned"] + df["goats_owned"]

    df["births_total"] = df.get(R_BORN, 0) + df.get(E_BORN, 0) + df.get(B_BORN, 0) + df.get(D_BORN, 0)
    df["deaths_total"] = df.get(R_DIED, 0) + df.get(E_DIED, 0) + df.get(B_DIED, 0) + df.get(D_DIED, 0)
    df["loss_total"] = df.get(R_LOST, 0) + df.get(E_LOST, 0) + df.get(B_LOST, 0) + df.get(D_LOST, 0)
    df["slaughter_total"] = df.get(R_SLAUGHT, 0) + df.get(E_SLAUGHT, 0) + df.get(B_SLAUGHT, 0) + df.get(D_SLAUGHT, 0)
    df["net_change_reported"] = df["births_total"] - (df["deaths_total"] + df["loss_total"] + df["slaughter_total"])

    denom = df[["sr_total_owned"]].replace({0: np.nan})
    df["men_control_share"] = df.get(MEN_OWN, 0) / denom["sr_total_owned"]
    df["women_control_share"] = df.get(WOMEN_OWN, 0) / denom["sr_total_owned"]
    return df

def compute_health(df: pd.DataFrame) -> pd.DataFrame:
    for col in [VACC_ANY, DEWORM_ANY, TREAT_ANY]:
        if col in df.columns and col + "_bool" not in df.columns:
            df[col + "_bool"] = yes_no_to_bool(df[col])
    return df

def compute_finance(df: pd.DataFrame) -> pd.DataFrame:
    for col in [CREDIT_30D, INS_30D]:
        if col in df.columns and (col + "_bool") not in df.columns:
            df[col + "_bool"] = df[col].astype(str).str.strip().str.lower().map({"yes": True, "no": False})
    for col in [CREDIT_TIMES, CREDIT_VALUE, INS_COUNT, INS_PREMIUM]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def compute_sales(df: pd.DataFrame) -> pd.DataFrame:
    s_kpmd_n = _safe_num_series(df, S_N_KPMD)
    g_kpmd_n = _safe_num_series(df, G_N_KPMD)
    s_non_n  = _safe_num_series(df, S_N_NON)
    g_non_n  = _safe_num_series(df, G_N_NON)

    s_kpmd_p = _safe_num_series(df, S_PRICE_KPMD)
    g_kpmd_p = _safe_num_series(df, G_PRICE_KPMD)
    s_non_p  = _safe_num_series(df, S_PRICE_NON)
    g_non_p  = _safe_num_series(df, G_PRICE_NON)

    s_kpmd_t = _safe_num_series(df, S_TRANS_KPMD)
    g_kpmd_t = _safe_num_series(df, G_TRANS_KPMD)
    s_non_t  = _safe_num_series(df, S_TRANS_NON)
    g_non_t  = _safe_num_series(df, G_TRANS_NON)

    df["sheep_sold_kpmd"] = s_kpmd_n.fillna(0)
    df["goat_sold_kpmd"]  = g_kpmd_n.fillna(0)
    df["sheep_sold_non"]  = s_non_n.fillna(0)
    df["goat_sold_non"]   = g_non_n.fillna(0)

    df["rev_sheep_kpmd"] = s_kpmd_n.fillna(0) * s_kpmd_p.fillna(0)
    df["rev_goat_kpmd"]  = g_kpmd_n.fillna(0) * g_kpmd_p.fillna(0)
    df["rev_sheep_non"]  = s_non_n.fillna(0)  * s_non_p.fillna(0)
    df["rev_goat_non"]   = g_non_n.fillna(0)  * g_non_p.fillna(0)
    df["rev_total"] = df[["rev_sheep_kpmd","rev_goat_kpmd","rev_sheep_non","rev_goat_non"]].sum(axis=1)

    df["transport_cost_total"] = (
        s_kpmd_t.fillna(0) * s_kpmd_n.fillna(0)
        + g_kpmd_t.fillna(0) * g_kpmd_n.fillna(0)
        + s_non_t.fillna(0)  * s_non_n.fillna(0)
        + g_non_t.fillna(0)  * g_non_n.fillna(0)
    )

    df["animals_sold_total"] = df[["sheep_sold_kpmd","goat_sold_kpmd","sheep_sold_non","goat_sold_non"]].sum(axis=1)
    return df

# Derived fields
df = compute_rcsi(df)
df = compute_herd(df)
df = compute_health(df)
df = compute_sales(df)
df = compute_finance(df)

# =========================================================
# -------------------- SIDEBAR FILTERS --------------------
# =========================================================

st.sidebar.title("Filters")

# Ruminant Focus (keep label exactly as requested: All / Sheep / Goat)
species_focus = st.sidebar.selectbox(
    "Ruminant Focus",
    options=["All", "Sheep", "Goat"],
    index=0,
    help="Scope ownership, herd events, sales, and revenue to a specific ruminant. Health/finance/rCSI/maps/exports remain global."
)
_focus = "Sheep" if species_focus == "Sheep" else "Goat" if species_focus == "Goat" else "All"

# Optional toggles (incrementals)
gender_split = st.sidebar.checkbox("Disaggregate some charts by Gender", value=False)
color_hotspots = st.sidebar.checkbox("Color map by treated cases (last month)", value=False)
market_split = st.sidebar.checkbox("Split Markets & Sales by KPMD registration", value=False)

# Ensure filter columns exist
for _col in [COUNTY_COL, SUBCOUNTY_COL]:
    if _col not in df.columns:
        df[_col] = np.nan

counties = sorted([c for c in df[COUNTY_COL].dropna().unique().tolist() if c])

# Collapsed multiselects via popover for County & Sub-county
with st.sidebar.popover("County"):
    sel_counties = st.multiselect(
        "County",
        options=counties,
        default=counties,
        label_visibility="collapsed",
    )

sub_options = sorted(df.loc[df[COUNTY_COL].isin(sel_counties), SUBCOUNTY_COL].dropna().unique().tolist())
with st.sidebar.popover("Sub-county"):
    sel_subcounties = st.multiselect(
        "Sub-county",
        options=sub_options,
        default=sub_options,
        label_visibility="collapsed",
    )

# Date range
if "int_date_parsed" in df.columns:
    min_d = pd.to_datetime(df["int_date_parsed"]).min()
    max_d = pd.to_datetime(df["int_date_parsed"]).max()
    start_d, end_d = st.sidebar.date_input(
        "Interview date range",
        value=(min_d.date() if pd.notna(min_d) else None, max_d.date() if pd.notna(max_d) else None),
        help="Filter by interview date"
    )
else:
    start_d = end_d = None

kpm_reg_filter = st.sidebar.selectbox("KPMD registered?", options=["All", "Yes", "No"], index=0)

# Apply filters
mask = df[COUNTY_COL].isin(sel_counties) & df[SUBCOUNTY_COL].isin(sel_subcounties)
if start_d and end_d and "int_date_parsed" in df.columns:
    mask &= df["int_date_parsed"].dt.date.between(start_d, end_d)
if kpm_reg_filter != "All" and KPMD_REG in df.columns:
    mask &= df[KPMD_REG].astype(str).str.lower().eq(kpm_reg_filter.lower())

dff = df.loc[mask].copy()
households = dff[HH_ID_COL].nunique() if HH_ID_COL in dff.columns else dff.shape[0]

# Helper: apply species scope (no changes to ingestion or other logic)
def _scope_cols(d: pd.DataFrame, focus: str) -> pd.DataFrame:
    d = d.copy()

    # Always return a numeric Series aligned to d.index (never a raw int)
    def s(col: str) -> pd.Series:
        return _safe_num_series(d, col).fillna(0)

    if focus == "Sheep":
        d["owned_scoped"]      = s("sheep_owned")
        d["births_scoped"]     = s(R_BORN) + s(E_BORN)
        d["deaths_scoped"]     = s(R_DIED) + s(E_DIED)
        d["loss_scoped"]       = s(R_LOST) + s(E_LOST)
        d["slaughter_scoped"]  = s(R_SLAUGHT) + s(E_SLAUGHT)
        d["sold_scoped"]       = s("sheep_sold_kpmd") + s("sheep_sold_non")
        d["rev_scoped"]        = s("rev_sheep_kpmd") + s("rev_sheep_non")
        d["transport_scoped"]  = s(S_TRANS_KPMD) * s(S_N_KPMD) + s(S_TRANS_NON) * s(S_N_NON)
        d["owned_label"]       = "Sheep owned (median)"

    elif focus == "Goat":
        d["owned_scoped"]      = s("goats_owned")
        d["births_scoped"]     = s(B_BORN) + s(D_BORN)
        d["deaths_scoped"]     = s(B_DIED) + s(D_DIED)
        d["loss_scoped"]       = s(B_LOST) + s(D_LOST)
        d["slaughter_scoped"]  = s(B_SLAUGHT) + s(D_SLAUGHT)
        d["sold_scoped"]       = s("goat_sold_kpmd") + s("goat_sold_non")
        d["rev_scoped"]        = s("rev_goat_kpmd") + s("rev_goat_non")
        d["transport_scoped"]  = s(G_TRANS_KPMD) * s(G_N_KPMD) + s(G_TRANS_NON) * s(G_N_NON)
        d["owned_label"]       = "Goats owned (median)"

    else:
        # All (global totals)
        d["owned_scoped"]      = s("sr_total_owned")
        d["births_scoped"]     = s("births_total")
        d["deaths_scoped"]     = s("deaths_total")
        d["loss_scoped"]       = s("loss_total")
        d["slaughter_scoped"]  = s("slaughter_total")
        d["sold_scoped"]       = s("animals_sold_total")
        d["rev_scoped"]        = s("rev_total")
        d["transport_scoped"]  = s("transport_cost_total")
        d["owned_label"]       = "Animals owned (median)"

    d["net_change_scoped"] = d["births_scoped"] - (d["deaths_scoped"] + d["loss_scoped"] + d["slaughter_scoped"])
    return d

dff_scoped = _scope_cols(dff, _focus)

# =========================================================
# ---------------------- HEADER KPIs ----------------------
# =========================================================

st.title("APMT Program Insights")
st.caption("Longitudinal panel insights on small-ruminant ownership, health, market participation, and resilience across Kajiado, Samburu, and Narok")

col_a, col_b, col_c, col_d, col_e = st.columns(5)
with col_a:
    st.metric("Households (filtered)", f"{households:,}")
with col_b:
    owned_med_int = _fmt_median_int(dff_scoped["owned_scoped"]) if "owned_scoped" in dff_scoped else 0
    owned_label = (dff_scoped["owned_label"].iloc[0] if ("owned_label" in dff_scoped and not dff_scoped.empty)
                   else "Animals owned (median)")
    st.metric(owned_label, f"{owned_med_int}")
with col_c:
    vacc_rate = (dff[VACC_ANY + "_bool"].mean()*100) if VACC_ANY + "_bool" in dff else 0
    st.metric("Vaccinated in last month", f"{vacc_rate:0.1f}%")
with col_d:
    sold_any = (dff_scoped["sold_scoped"] > 0).mean()*100 if "sold_scoped" in dff_scoped else 0
    st.metric("Sold animals (any channel)", f"{sold_any:0.1f}%")
with col_e:
    rcsi_mean = dff["rCSI"].mean() if "rCSI" in dff and len(dff["rCSI"].dropna()) else np.nan
    st.metric("Avg rCSI", f"{rcsi_mean:.1f}" if pd.notna(rcsi_mean) else "—")

# ---- Incremental: % female stock KPIs (global) ----
col_f1, col_f2 = st.columns(2)
with col_f1:
    if "sheep_owned" in dff.columns and SHEEP_EWES in dff.columns:
        denom = dff["sheep_owned"].replace({0: np.nan}).sum()
        if pd.notna(denom) and denom > 0:
            sheep_female_share = (dff[SHEEP_EWES].sum() / denom) * 100
            st.metric("Sheep: % female", f"{sheep_female_share:0.1f}%")
with col_f2:
    if "goats_owned" in dff.columns and GOAT_DOES in dff.columns:
        denom = dff["goats_owned"].replace({0: np.nan}).sum()
        if pd.notna(denom) and denom > 0:
            goat_female_share = (dff[GOAT_DOES].sum() / denom) * 100
            st.metric("Goats: % female", f"{goat_female_share:0.1f}%")

st.divider()

# =========================================================
# ---------------------- VISUAL SECTIONS ------------------
# =========================================================

def section_header(title: str, help_text: str = ""):
    st.subheader(title, help=help_text)

def vbar(df_in, x, y, color=None, title="", tooltip=None):
    if tooltip is None:
        tooltip = [x, y] + ([color] if color else [])
    chart = alt.Chart(df_in).mark_bar().encode(
        x=alt.X(x, sort="-y"),
        y=alt.Y(y),
        color=color if color else alt.value("#4c78a8"),
        tooltip=tooltip
    ).properties(height=320, title=title)
    st.altair_chart(chart, use_container_width=True)

def line(df_in, x, y, color=None, title="", tooltip=None):
    if tooltip is None:
        tooltip = [x, y] + ([color] if color else [])
    enc = {"x": alt.X(x), "y": alt.Y(y), "tooltip": tooltip}
    if color:
        enc["color"] = color
    chart = alt.Chart(df_in).mark_line(point=True).encode(**enc)\
        .properties(height=320, title=title)
    st.altair_chart(chart, use_container_width=True)

# ---------- Common sections (shown to all) ----------

# 1) Herd dynamics (scoped by Ruminant Focus)
section_header("Herd Dynamics", "Ownership, births/deaths/losses, net change, water access")
c1, c2, c3 = st.columns([1,1,1])

with c1:
    if not dff_scoped.empty:
        if gender_split and GENDER_COL in dff_scoped.columns and dff_scoped[GENDER_COL].notna().any():
            by_geo = (
                dff_scoped.dropna(subset=[GENDER_COL])
                .groupby([COUNTY_COL, GENDER_COL])[["owned_scoped"]]
                .median(numeric_only=True)
                .reset_index()
                .rename(columns={"owned_scoped": "owned"})
            )
            title = (dff_scoped["owned_label"].iloc[0] if "owned_label" in dff_scoped and not dff_scoped.empty
                     else "Animals owned (median)")
            vbar(by_geo, x=COUNTY_COL, y="owned", color=GENDER_COL, title=f"{title} by County (by Gender)")
        else:
            by_geo = dff_scoped.groupby([COUNTY_COL])[["owned_scoped"]].median(numeric_only=True).reset_index()
            if not by_geo.empty:
                title = (dff_scoped["owned_label"].iloc[0] if "owned_label" in dff_scoped and not dff_scoped.empty
                         else "Animals owned (median)")
                vbar(by_geo.rename(columns={"owned_scoped": "owned"}), x=COUNTY_COL, y="owned", title=title)

with c2:
    dff_scoped["net_change_scoped"] = dff_scoped.get("net_change_scoped", 0)
    by_geo2 = dff_scoped.groupby([COUNTY_COL])[["births_scoped","deaths_scoped","loss_scoped","slaughter_scoped","net_change_scoped"]] \
                        .median(numeric_only=True).reset_index()
    if not by_geo2.empty:
        hm = by_geo2.melt(id_vars=[COUNTY_COL], var_name="event", value_name="count")
        vbar(hm, x="event", y="count", color="County", title="Median monthly herd events (by County)")

with c3:
    # Water access remains global (not species-scoped)
    if WATER_ACCESS in dff.columns:
        water_rate = dff[WATER_ACCESS + "_bool"].mean()*100 if WATER_ACCESS + "_bool" in dff else \
                     yes_no_to_bool(dff[WATER_ACCESS]).mean()*100
        st.metric("Water access (share)", f"{water_rate:0.1f}%")
    gender_df = dff[[COUNTY_COL,"men_control_share","women_control_share"]].replace([np.inf, -np.inf], np.nan).dropna()
    if not gender_df.empty:
        gender_m = gender_df.groupby(COUNTY_COL)[["men_control_share","women_control_share"]].median().reset_index().melt(
            id_vars=COUNTY_COL, var_name="group", value_name="share"
        )
        vbar(gender_m, x="group", y="share", color="County", title="Median control share by gender (County)")

# ---- Incremental: KPMD split for scoped births (example) ----
if KPMD_REG in dff.columns and "births_scoped" in dff_scoped.columns and "wave_month" in dff_scoped.columns:
    births_kpmd = dff_scoped.groupby(["wave_month", KPMD_REG])["births_scoped"].median().reset_index()
    section_header("KPMD Split (Scoped Metric)", "Median births by month, split by KPMD registration")
    line(births_kpmd, x="wave_month", y="births_scoped", color=KPMD_REG, title="Births (median) by KPMD status")

# 2) Animal health (global)
section_header("Animal Health", "Vaccination, treatment, deworming and diseases (last month)")
h1, h2, h3 = st.columns(3)
with h1:
    if VACC_ANY + "_bool" in dff:
        v_rate = dff[VACC_ANY + "_bool"].mean()*100
        st.metric("Vaccinated (last month)", f"{v_rate:0.1f}%")
    if DEWORM_ANY + "_bool" in dff:
        d_rate = dff[DEWORM_ANY + "_bool"].mean()*100
        st.metric("Dewormed (last month)", f"{d_rate:0.1f}%")
with h2:
    if TREAT_ANY + "_bool" in dff:
        t_rate = dff[TREAT_ANY + "_bool"].mean()*100
        st.metric("Treated ill animals (last month)", f"{t_rate:0.1f}%")
    if VACC_COUNT in dff:
        st.metric("Vaccinated animals (median)", f"{_fmt_median_int(dff[VACC_COUNT])}")
with h3:
    if TREAT_COUNT in dff:
        st.metric("Treated animals (median)", f"{_fmt_median_int(dff[TREAT_COUNT])}")

# 3) Markets & sales (scoped)
section_header("Markets & Sales", "Sales to KPMD vs non-KPMD, revenue, transport, distance, KPMD exposure")
m1, m2, m3 = st.columns(3)

def _sales_split_by_kpmd_for_focus(dff_scoped: pd.DataFrame, focus: str) -> pd.DataFrame:
    """Return dataframe with totals split by KPMD registration for current focus."""
    if KPMD_REG not in dff_scoped.columns:
        return pd.DataFrame()
    if focus == "All":
        sheep = (dff_scoped["sheep_sold_kpmd"] + dff_scoped["sheep_sold_non"]).groupby(dff_scoped[KPMD_REG]).sum()
        goat  = (dff_scoped["goat_sold_kpmd"]  + dff_scoped["goat_sold_non"]).groupby(dff_scoped[KPMD_REG]).sum()
        out = [{"species": "Sheep", KPMD_REG: k, "count": v} for k, v in sheep.items()] + \
              [{"species": "Goat",  KPMD_REG: k, "count": v} for k, v in goat.items()]
        return pd.DataFrame(out)
    else:
        return dff_scoped.groupby(KPMD_REG)["sold_scoped"].sum().reset_index().rename(columns={"sold_scoped": "count"})

def _revenue_split_by_kpmd_for_focus(dff_scoped: pd.DataFrame, focus: str) -> pd.DataFrame:
    if KPMD_REG not in dff_scoped.columns:
        return pd.DataFrame()
    if focus == "All":
        out = []
        for sp, col in [("Sheep", ["rev_sheep_kpmd","rev_sheep_non"]),
                        ("Goat",  ["rev_goat_kpmd","rev_goat_non"])]:
            ser = (dff_scoped[col[0]] + dff_scoped[col[1]]).groupby(dff_scoped[KPMD_REG]).sum()
            out += [{"species": sp, KPMD_REG: k, "revenue_ksh": v} for k, v in ser.items()]
        return pd.DataFrame(out)
    else:
        return dff_scoped.groupby(KPMD_REG)["rev_scoped"].sum().reset_index().rename(columns={"rev_scoped": "revenue_ksh"})

with m1:
    if market_split and KPMD_REG in dff_scoped.columns:
        if _focus == "All":
            sales = _sales_split_by_kpmd_for_focus(dff_scoped, _focus)
            vbar(sales, x="species", y="count", color=KPMD_REG, title="Animals sold (count) by KPMD registration")
        else:
            sales = _sales_split_by_kpmd_for_focus(dff_scoped, _focus)
            vbar(sales, x=KPMD_REG, y="count", title=f"{_focus} sold (count) by KPMD registration")
    else:
        if _focus == "All":
            sales = pd.DataFrame({
                "channel": ["KPMD","KPMD","Non-KPMD","Non-KPMD"],
                "species": ["Sheep","Goat","Sheep","Goat"],
                "count": [
                    dff.get("sheep_sold_kpmd", pd.Series([], dtype=float)).sum(),
                    dff.get("goat_sold_kpmd",  pd.Series([], dtype=float)).sum(),
                    dff.get("sheep_sold_non",  pd.Series([], dtype=float)).sum(),
                    dff.get("goat_sold_non",   pd.Series([], dtype=float)).sum(),
                ]
            })
            vbar(sales, x="species", y="count", color="channel", title="Animals sold (count)")
        elif _focus == "Sheep":
            sales = pd.DataFrame({
                "channel": ["KPMD","Non-KPMD"],
                "count": [
                    dff.get("sheep_sold_kpmd", pd.Series([], dtype=float)).sum(),
                    dff.get("sheep_sold_non",  pd.Series([], dtype=float)).sum(),
                ]
            })
            vbar(sales, x="channel", y="count", title="Sheep sold (count)")
        else:  # Goat
            sales = pd.DataFrame({
                "channel": ["KPMD","Non-KPMD"],
                "count": [
                    dff.get("goat_sold_kpmd", pd.Series([], dtype=float)).sum(),
                    dff.get("goat_sold_non",  pd.Series([], dtype=float)).sum(),
                ]
            })
            vbar(sales, x="channel", y="count", title="Goats sold (count)")

with m2:
    if market_split and KPMD_REG in dff_scoped.columns:
        if _focus == "All":
            rev = _revenue_split_by_kpmd_for_focus(dff_scoped, _focus)
            vbar(rev, x="species", y="revenue_ksh", color=KPMD_REG, title="Revenue (KSh) by KPMD registration")
        else:
            rev = _revenue_split_by_kpmd_for_focus(dff_scoped, _focus)
            vbar(rev, x=KPMD_REG, y="revenue_ksh", title=f"Revenue (KSh) — {_focus} by KPMD registration")
    else:
        if _focus == "All":
            rev = pd.DataFrame({
                "channel": ["KPMD","KPMD","Non-KPMD","Non-KPMD"],
                "species": ["Sheep","Goat","Sheep","Goat"],
                "revenue_ksh": [
                    dff.get("rev_sheep_kpmd", pd.Series([], dtype=float)).sum(),
                    dff.get("rev_goat_kpmd",  pd.Series([], dtype=float)).sum(),
                    dff.get("rev_sheep_non",  pd.Series([], dtype=float)).sum(),
                    dff.get("rev_goat_non",   pd.Series([], dtype=float)).sum(),
                ]
            })
            vbar(rev, x="species", y="revenue_ksh", color="channel", title="Revenue (KSh)")
        elif _focus == "Sheep":
            rev = pd.DataFrame({
                "channel": ["KPMD","Non-KPMD"],
                "revenue_ksh": [
                    dff.get("rev_sheep_kpmd", pd.Series([], dtype=float)).sum(),
                    dff.get("rev_sheep_non",  pd.Series([], dtype=float)).sum(),
                ]
            })
            vbar(rev, x="channel", y="revenue_ksh", title="Revenue (KSh) — Sheep")
        else:
            rev = pd.DataFrame({
                "channel": ["KPMD","Non-KPMD"],
                "revenue_ksh": [
                    dff.get("rev_goat_kpmd", pd.Series([], dtype=float)).sum(),
                    dff.get("rev_goat_non",  pd.Series([], dtype=float)).sum(),
                ]
            })
            vbar(rev, x="channel", y="revenue_ksh", title="Revenue (KSh) — Goats")

with m3:
    km = dff[DIST_KM].dropna() if DIST_KM in dff else pd.Series([], dtype=float)
    if len(km) > 0:
        st.metric("Median distance to market (km)", f"{km.median():.1f}")
    if KPMD_REG in dff and HH_ID_COL in dff:
        exp = dff.groupby(KPMD_REG)[HH_ID_COL].nunique().reset_index().rename(columns={HH_ID_COL:"households"})
        vbar(exp, x=KPMD_REG, y="households", title="Households by KPMD registration")

# 4) Gender decision & control (global)
section_header("Gender Decision & Use of Income", "Share of decision makers and income controllers")
g1, g2 = st.columns(2)
if G1_MULTI in dff and G2_MULTI in dff:
    def counts_from_multiselect(col: str, options: List[str]) -> pd.DataFrame:
        tmp = {}
        s = dff[col].astype(str).fillna("")
        for opt in options:
            tmp[opt] = s.str.contains(re.escape(opt), case=False, na=False).astype(int)
        return pd.DataFrame(tmp)

    g1_df = counts_from_multiselect(G1_MULTI, G_ROLES).sum().reset_index()
    g1_df.columns = ["role","count"]
    vbar(g1_df, x="role", y="count", title="Who decides livestock sale?")

    g2_df = counts_from_multiselect(G2_MULTI, G_ROLES).sum().reset_index()
    g2_df.columns = ["role","count"]
    vbar(g2_df, x="role", y="count", title="Who uses income from sales?")

# 5) Food security (rCSI) — global
section_header("Food Security (rCSI)", "Reduced Coping Strategies Index (higher = worse)")
fs1, fs2 = st.columns(2)
with fs1:
    if "rCSI" in dff and not dff.empty:
        if gender_split and GENDER_COL in dff.columns and dff[GENDER_COL].notna().any():
            tmp = dff.dropna(subset=[GENDER_COL]).groupby([COUNTY_COL, GENDER_COL])["rCSI"].median().reset_index()
            vbar(tmp, x=COUNTY_COL, y="rCSI", color=GENDER_COL, title="Median rCSI by County (by Gender)")
        else:
            by_c = dff.groupby(COUNTY_COL)["rCSI"].median().reset_index()
            vbar(by_c, x=COUNTY_COL, y="rCSI", title="Median rCSI by County")
with fs2:
    # Safer rCSI band ordering (Minimal -> Catastrophic)
    if "rCSI_band" in dff:
        bands = (dff["rCSI_band"]
                 .astype("category")
                 .cat.set_categories(RCSI_ORDER, ordered=True)
                 .value_counts(dropna=False)
                 .rename_axis("band")
                 .reset_index(name="households"))
        vbar(bands, x="band", y="households", title="rCSI bands (count)")

# 6) Credit & Insurance — global
section_header("Finance & Insurance", "Credit access and livestock insurance (last 30 days)")
fi1, fi2, fi3 = st.columns(3)
with fi1:
    if CREDIT_30D + "_bool" in dff:
        st.metric("Applied for credit (30d)", f"{dff[CREDIT_30D + '_bool'].mean()*100:.1f}%")
    if CREDIT_VALUE in dff:
        avg_credit = dff[CREDIT_VALUE].dropna().mean()
        st.metric("Avg credit value (KSh)", f"{avg_credit:,.0f}" if pd.notna(avg_credit) else "—")
with fi2:
    if INS_30D + "_bool" in dff:
        st.metric("Accessed insurance (30d)", f"{dff[INS_30D + '_bool'].mean()*100:.1f}%")
    if INS_COUNT in dff:
        st.metric("Livestock insured (median)", f"{_fmt_median_int(dff[INS_COUNT])}")
with fi3:
    if INS_PREMIUM in dff:
        st.metric("Premium (median KSh)", f"{_fmt_median_int(dff[INS_PREMIUM])}")

# 7) Adaptation strategies — global
section_header("Adaptation to Shocks", "Adopted strategies last month")
if J2_MULTI in dff:
    j2_counts = {}
    s = dff[J2_MULTI].astype(str).fillna("")
    for opt in J2_OPTIONS:
        j2_counts[opt] = s.str.contains(re.escape(opt), case=False, na=False).sum()
    j2_df = pd.DataFrame({"strategy": list(j2_counts.keys()), "households": list(j2_counts.values())})
    vbar(j2_df, x="strategy", y="households", title="Adopted strategies")

# ---- Incremental: Resilience grouping + trend ----
ADAPTIVE = {
    "Increased mobility (distance & frequency)",
    "Purchase of fodder",
    "Change in water management",
    "Use stored fodder",
    "Herd destocking and restocking",
}
EROSIVE = {"Reduce herd size", "Banking livestock assets (sell and bank saving)"}

if "wave_month" in dff.columns and J2_MULTI in dff.columns:
    s_ad = dff[J2_MULTI].astype(str).fillna("")
    df_ad = pd.DataFrame({
        "adaptive": s_ad.str.contains("|".join(map(re.escape, ADAPTIVE)), case=False, na=False),
        "erosive":  s_ad.str.contains("|".join(map(re.escape, EROSIVE)),  case=False, na=False),
        "wave_month": dff["wave_month"]
    })
    trend_ad = df_ad.groupby("wave_month")[["adaptive","erosive"]].mean().reset_index().melt(
        id_vars="wave_month", var_name="type", value_name="share"
    )
    section_header("Adaptation Trends", "Share of households adopting adaptive vs erosive strategies")
    line(trend_ad, x="wave_month", y="share", color="type", title="Adaptation shares over time")

# 8) Map (GPS) — global
section_header("Map of Respondents", "Locations based on captured GPS")
if GPS_LAT_COL in dff and GPS_LON_COL in dff:
    geo = dff[[GPS_LAT_COL, GPS_LON_COL, COUNTY_COL, SUBCOUNTY_COL]].dropna().copy()
    geo = geo.rename(columns={GPS_LAT_COL:"lat", GPS_LON_COL:"lon"})
    if not geo.empty:
        # Robust hotspot coloring: precompute color columns
        if color_hotspots and TREAT_COUNT in dff.columns:
            geo["treated"] = dff.loc[geo.index, TREAT_COUNT].fillna(0).astype(float)
            geo["color_r"] = np.minimum(255, (geo["treated"] * 10).astype(int))
            layers = [pdk.Layer(
                "ScatterplotLayer",
                data=geo,
                get_position='[lon, lat]',
                get_radius=2000,
                pickable=True,
                get_fill_color='[color_r, 0, 0, 160]',
            )]
        else:
            layers = [pdk.Layer(
                "ScatterplotLayer", data=geo,
                get_position='[lon, lat]',
                get_radius=2000, pickable=True
            )]

        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=geo["lat"].mean(), longitude=geo["lon"].mean(), zoom=6, pitch=0
            ),
            layers=layers,
            tooltip={"text": "{County}\n{Sub county}"}
        ))
    else:
        st.info("No GPS coordinates available in the filtered data.")

# =========================================================
# ---------------- LONGITUDINAL & TRENDS ------------------
# =========================================================

section_header("Longitudinal Trends", "Follow households across waves / months (panel views)")
if "wave_month" in dff and HH_ID_COL in dff:

    # ---- Incremental: Monthly submissions volume (global) ----
    sub_counts = dff.groupby("wave_month").size().reset_index(name="submissions")
    if not sub_counts.empty:
        vbar(sub_counts, x="wave_month", y="submissions", title="Submitted records per month")

    if _focus == "All":
        trend_options = [
            ("sr_total_owned", "Animals owned (median)"),
            ("births_total", "Births (median)"),
            ("deaths_total", "Deaths (median)"),
            ("animals_sold_total", "Animals sold (median)"),
            ("rev_total", "Revenue (median KSh)"),
            ("transport_cost_total", "Transport cost (median KSh)"),
            ("rCSI", "rCSI (median)"),
        ]
    else:
        trend_options = [
            ("owned_scoped", "Owned (median)"),
            ("births_scoped", "Births (median)"),
            ("deaths_scoped", "Deaths (median)"),
            ("sold_scoped", "Animals sold (median)"),
            ("rev_scoped", "Revenue (median KSh)"),
            ("transport_scoped", "Transport cost (median KSh)"),
            ("rCSI", "rCSI (median)"),  # rCSI is global but can be trended
        ]

    metric_keys = [k for k, _ in trend_options]
    labels = {k: lbl for k, lbl in trend_options}

    choice_metric = st.selectbox(
        "Metric to trend by month",
        options=metric_keys,
        index=0,
        format_func=lambda k: labels.get(k, k)
    )

    # Choose the right dataframe for the selected metric
    source_df = dff_scoped if choice_metric in [
        "owned_scoped","births_scoped","deaths_scoped","sold_scoped","rev_scoped","transport_scoped"
    ] else dff

    trend = source_df.groupby(["wave_month"])[choice_metric].median(numeric_only=True).reset_index()
    if not trend.empty:
        line(trend, x="wave_month", y=choice_metric, title=f"Median {labels.get(choice_metric, choice_metric)} over time")

    # -------- Household panel (scoped + global) --------
    panel_base_cols = ["int_date_parsed", "wave_month"]
    if "wave_label" in dff.columns:
        panel_base_cols.append("wave_label")

    global_cols = []
    if "rCSI" in dff.columns: global_cols.append("rCSI")
    if "men_control_share" in dff.columns: global_cols.append("men_control_share")
    if "women_control_share" in dff.columns: global_cols.append("women_control_share")

    # Ensure panel has household id and geography (if present)
    panel_extra_cols = []
    if HH_ID_COL in dff.columns: panel_extra_cols.append(HH_ID_COL)
    if COUNTY_COL in dff.columns: panel_extra_cols.append(COUNTY_COL)
    if SUBCOUNTY_COL in dff.columns: panel_extra_cols.append(SUBCOUNTY_COL)

    scoped_cols = ["owned_scoped","births_scoped","deaths_scoped","sold_scoped","rev_scoped","transport_scoped"]

    # Merge global columns from dff to dff_scoped (safe on HH_ID + date)
    join_keys = []
    if HH_ID_COL in dff.columns: join_keys.append(HH_ID_COL)
    if "int_date_parsed" in dff.columns: join_keys.append("int_date_parsed")

    rhs_cols = list(set((join_keys or []) + global_cols))
    rhs = dff[rhs_cols].copy() if rhs_cols else pd.DataFrame()

    panel_df = dff_scoped.copy()
    if rhs.shape[1] > 0 and len(join_keys) > 0:
        panel_df = panel_df.merge(rhs, on=join_keys, how="left")

    show_cols = [c for c in panel_extra_cols + panel_base_cols + scoped_cols + global_cols if c in panel_df.columns]

    # Household selector (uses filtered set)
    hh_options = sorted(dff[HH_ID_COL].dropna().unique().tolist()) if HH_ID_COL in dff else []
    hh_default_index = 0 if hh_options else None
    hh_for_panel = st.selectbox(
        "Household to inspect (panel)",
        options=hh_options,
        index=hh_default_index
    )

    # ✅ Correct slicing with .loc[...] and a boolean mask
    if hh_for_panel and HH_ID_COL in panel_df.columns:
        hh_mask = panel_df[HH_ID_COL].astype(str).eq(str(hh_for_panel))
        hh_panel = panel_df.loc[hh_mask]
    else:
        hh_panel = panel_df.iloc[0:0]  # empty fallback

    # Keep sort & render
    hh_panel = hh_panel.sort_values("int_date_parsed") if "int_date_parsed" in hh_panel.columns else hh_panel
    if show_cols:
        st.dataframe(hh_panel[show_cols].reset_index(drop=True))
    else:
        st.info("No panel columns available to display for the selected household.")

# =========================================================
# ------------------ DATA QUALITY CHECKS ------------------
# =========================================================

section_header("Data Quality", "Missingness, outliers, impossible values")
dq1, dq2, dq3 = st.columns(3)
with dq1:
    req_cols = [HH_ID_COL, COUNTY_COL, SUBCOUNTY_COL, DATE_COL, "sr_total_owned"]
    missing = {c: dff[c].isna().mean()*100 if c in dff else 100.0 for c in req_cols}
    miss_df = pd.DataFrame({"field": list(missing.keys()), "missing_%": list(missing.values())})
    vbar(miss_df, x="field", y="missing_%", title="Missingness (%)")

with dq2:
    flags = pd.DataFrame({
        "neg_ownership": (dff["sr_total_owned"] < 0).sum() if "sr_total_owned" in dff else 0,
        "age>200? (if present)": (dff.get("A12. Age in years", pd.Series([0])).fillna(0) > 200).sum(),
        "distance>500km?": (dff.get(DIST_KM, pd.Series([0])).fillna(0) > 500).sum(),
    }, index=[0]).T.reset_index()
    flags.columns = ["check","count"]
    vbar(flags, x="check", y="count", title="Outlier flags (count)")

with dq3:
    if "rev_total" in dff:
        pct95 = dff["rev_total"].quantile(0.95)
        st.metric("Revenue (95th pct, KSh)", f"{pct95:,.0f}" if pd.notna(pct95) else "—")

# =========================================================
# ----------------- DOWNLOADS / EXPORTS -------------------
# =========================================================

section_header("Exports", "Download filtered data and KPI snapshots")

def to_csv_download(df_in: pd.DataFrame, name: str):
    csv_data = df_in.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download {name}.csv",
        data=csv_data,
        file_name=f"{name}.csv",
        mime="text/csv",
        use_container_width=True
    )

# Filtered raw rows (GLOBAL filter context, not species-sliced rows)
to_csv_download(dff, "apmt_filtered_rows")

# KPI snapshot — include wave_label if available (easier pivoting downstream)
if not dff.empty:
    agg_dict = {
        "sr_total_owned":"median",
        "rCSI":"median",
        "rev_total":"sum",
    }
    if VACC_ANY + "_bool" in dff:
        agg_dict[VACC_ANY + "_bool"] = "mean"

    group_cols = [COUNTY_COL]
    if "wave_label" in dff.columns:
        group_cols.append("wave_label")

    if HH_ID_COL in dff:
        hh_series = dff.groupby(group_cols)[HH_ID_COL].nunique()
    else:
        hh_series = dff.groupby(group_cols).size()

    kpi_by = dff.groupby(group_cols).agg(agg_dict).rename(columns={
        "sr_total_owned":"sr_owned_med",
        "rCSI":"rCSI_med",
        "rev_total":"rev_sum",
        VACC_ANY + "_bool":"vacc_rate"
    }).reset_index()

    if isinstance(hh_series, pd.Series):
        kpi_by = kpi_by.merge(hh_series.rename("hh"), on=group_cols, how="left")
    if "vacc_rate" in kpi_by.columns:
        kpi_by["vacc_rate"] = kpi_by["vacc_rate"] * 100

    to_csv_download(kpi_by, "apmt_kpi")

# ------------------------ END ----------------------------