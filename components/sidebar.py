# apmt_dashboard/components/sidebar.py
import streamlit as st
import pandas as pd
from datetime import datetime
from utils.helpers import coalesce_first

def create_sidebar(processor):
    """Create sidebar with filters and navigation."""
    
    # ---------- Sidebar: FILTERS ----------
    st.sidebar.header("Global Filters")
    
    # Store original dataframe for bounds calculation
    original_df = processor.original_df if hasattr(processor, 'original_df') else processor.df.copy()
    
    # Helper function to compute date bounds
    def _compute_date_bounds(df_for_bounds):
        for cand in ['int_date_std', '_submission_time', 'start', 'end']:
            if cand in df_for_bounds.columns:
                s = pd.to_datetime(df_for_bounds[cand], errors='coerce')
                if s.notna().any():
                    return (s.min().date(), s.max().date(), cand)
        return (datetime(2024, 1, 1).date(), datetime.today().date(), None)

    # Compute date bounds
    _min_date, _max_date, _date_col_found = _compute_date_bounds(original_df)
    if _min_date > _max_date:
        _min_date, _max_date = _max_date, _min_date

    # Initialize session state for filters
    if 'data_sig' not in st.session_state:
        st.session_state['data_sig'] = None
    
    head_hash = pd.util.hash_pandas_object(original_df.head(10), index=False).sum() if len(original_df) else 0
    tail_hash = pd.util.hash_pandas_object(original_df.tail(10), index=False).sum() if len(original_df) else 0
    data_sig = (len(original_df), str(_min_date), str(_max_date), int(head_hash), int(tail_hash))

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

    # Check if filters are active
    filters_active = False
    
    with st.sidebar.expander("Select Here", expanded=filters_active):
        # ----- County → Sub-County (cascading) -----
        if 'County' in processor.df.columns:
            counties = ['All'] + sorted(processor.df['County'].dropna().unique())
            selected_county = st.selectbox("Select County", counties, key="county")

            if selected_county != 'All':
                processor.df = processor.df[processor.df['County'] == selected_county]
                filters_active = True

                sub_col = coalesce_first(
                    original_df,
                    ['Sub County', 'Sub-County', 'Subcounty', 'Sub-county', 'SubCounty', 'Sub county']
                )
                if sub_col and sub_col in original_df.columns:
                    sub_opts = ['All'] + sorted(
                        original_df.loc[original_df['County'] == selected_county, sub_col].dropna().unique()
                    )
                    selected_sub = st.selectbox("Select Sub-County", sub_opts, key="subcounty")
                    if selected_sub != 'All':
                        processor.df = processor.df[processor.df[sub_col] == selected_sub]
                        filters_active = True

        # ----- KPMD status -----
        kpmd_filter = st.selectbox("KPMD Status", ['All', 'Registered', 'Not Registered'], key="kpmd_filter")
        if kpmd_filter == 'Registered':
            processor.df = processor.df[processor.df['kpmd_registered'] == 1]
            filters_active = True
        elif kpmd_filter == 'Not Registered':
            processor.df = processor.df[processor.df['kpmd_registered'] == 0]
            filters_active = True

        # ----- Gender -----
        if 'Gender' in processor.df.columns:
            genders = ['All'] + sorted(processor.df['Gender'].dropna().unique())
            selected_gender = st.selectbox("Select Gender", genders, key="gender")
            if selected_gender != 'All':
                processor.df = processor.df[processor.df['Gender'] == selected_gender]
                filters_active = True

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
                filters_active = True
        else:
            st.info("No time information available for date filtering.")

    # ---------- Sidebar: NAVIGATION ----------
    st.sidebar.markdown(
        '<div style="color:#dc3545; font-weight:700; font-size:1rem; margin-bottom:0.25rem;">'
        'Navigate Here <span style="font-size:1.1rem; line-height:1;">👇</span>'
        '</div>',
        unsafe_allow_html=True
    )

    # Main pages list (Panel Analysis removed; it will have its own section)
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
        "Food Security",
    ]

    # Initialize session state for navigation
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "Field Outlook"
    if "nav_page_radio" not in st.session_state:
        st.session_state["nav_page_radio"] = "Field Outlook"

    # Radio for standard pages
    selected_from_radio = st.sidebar.radio(
        "Select Dashboard Page",
        MAIN_PAGES,
        key="nav_page_radio"
    )

    # Sync navigation state, but do NOT override if currently on Panel or P&L
    if st.session_state.get("nav_page") not in ["P&L Analysis", "Panel Analysis"]:
        st.session_state["nav_page"] = selected_from_radio

    # ---------- Separate section: Panel Analysis ----------
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Panel Analysis")
    if st.sidebar.button("Open Panel Analysis", use_container_width=True, key="open_panel_analysis"):
        st.session_state["nav_page"] = "Panel Analysis"

    if st.session_state.get("nav_page") == "Panel Analysis":
        def _back_to_pages_panel():
            st.session_state["nav_page"] = st.session_state.get("nav_page_radio", MAIN_PAGES[0])
        st.sidebar.button("← Back to pages", on_click=_back_to_pages_panel, use_container_width=True, key="back_from_panel")

    # ---------- Separate section: P&L Analysis ----------
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💰 P&L Analysis")
    if st.sidebar.button("Open P&L Analysis", use_container_width=True, key="open_pl_analysis"):
        st.session_state["nav_page"] = "P&L Analysis"

    # Back button when on P&L Analysis
    if st.session_state.get("nav_page") == "P&L Analysis":
        def _back_to_pages():
            st.session_state["nav_page"] = st.session_state.get("nav_page_radio", MAIN_PAGES[0])
        st.sidebar.button("← Back to pages", on_click=_back_to_pages, use_container_width=True, key="back_from_pl")

    # Return selected page
    return st.session_state["nav_page"]
