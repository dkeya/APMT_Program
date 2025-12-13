# apmt_dashboard/data_processing/data_loader.py

import pandas as pd
import streamlit as st
from utils.geo_utils import ensure_geo_assets  # re-exported for app.py


@st.cache_data(ttl=900, show_spinner=False)
def load_apmt_csv(path: str) -> pd.DataFrame:
    """
    Load the APMT longitudinal CSV with robust encoding handling.

    - Tries a sequence of common encodings.
    - Falls back to the Python engine with automatic separator detection.
    - As a last resort, uses errors='replace' and warns the user.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded APMT dataset.
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1", "ISO-8859-1", "windows-1252"]

    # 1) Normal read with standard engine
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            # Previously: st.caption(f"✅ Loaded APMT CSV using encoding: `{enc}`")
            return df
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            # This is a real error – surface it immediately
            raise
        except Exception:
            # Other read errors – try the next encoding
            continue

    # 2) Try again with python engine and automatic separator detection
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
            # Previously:
            # st.caption(
            #     f"✅ Loaded APMT CSV using encoding: `{enc}` "
            #     "(python engine, automatic separator detection)"
            # )
            return df
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            raise
        except Exception:
            continue

    # 3) Last-resort fallbacks with errors='replace'
    st.warning(
        "⚠️ Could not cleanly decode the file using standard encodings. "
        "Falling back to `errors='replace'`. Please inspect text labels for "
        "any garbled characters (especially non-ASCII characters)."
    )

    try:
        df = pd.read_csv(path, encoding="utf-8", errors="replace")
        # Previously:
        # st.caption("Loaded APMT CSV using encoding: `utf-8` with `errors='replace'`.")
        return df
    except FileNotFoundError:
        raise
    except Exception:
        df = pd.read_csv(path, encoding="latin-1", errors="replace")
        # Previously:
        # st.caption("Loaded APMT CSV using encoding: `latin-1` with `errors='replace'`.")
        return df
