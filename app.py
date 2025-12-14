# apmt_dashboard/app.py
import streamlit as st
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import from our modules
from data_processing.data_loader import load_apmt_csv
from data_processing.data_processor import APMTDataProcessor
from data_processing.calculations import calculate_all_metrics
from utils.data_quality import render_data_quality_section
from utils.geo_utils import ensure_geo_assets
from components.sidebar import create_sidebar

# Page imports
from pages.field_outlook import render_field_outlook
from pages.pastoral_productivity import render_pastoral_productivity
from pages.pastoral_livelihoods import render_pastoral_livelihoods
from pages.feed_fodder import render_feed_fodder
from pages.sheep_offtake import render_sheep_offtake
from pages.goat_offtake import render_goat_offtake
from pages.payments import render_payments
from pages.county_comparator import render_county_comparator
from pages.gender_inclusion import render_gender_inclusion
from pages.climate_impact import render_climate_impact
from pages.kpmd_participation import render_kpmd_participation
from pages.food_security import render_food_security
from pages.panel_analysis import render_panel_analysis
from pages.pl_analysis import render_pl_analysis

# -------------------------------------------------
# Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="APMT Panel Data Dashboard",
    page_icon="🐑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🔒 Hide default Streamlit multipage navigation in the sidebar
st.markdown(
    """
    <style>
        /* Hide the built-in sidebar page navigation */
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------
def load_css():
    css_path = os.path.join(project_root, "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback CSS
        st.markdown(
            """
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
            .warning-card {
                background-color: #fff3cd;
                padding: 1rem;
                border-radius: 10px;
                border-left: 4px solid #ffc107;
            }
            .profit-positive { color: #28a745; font-weight: bold; }
            .profit-negative { color: #dc3545; font-weight: bold; }
            .lsm-note { font-size: 0.85rem; color: #555; margin-top: 0.25rem; }
            .panel-wave { color: #6f42c1; font-weight: bold; }
            .did-estimate {
                background-color: #e8f5e8;
                padding: 10px;
                border-radius: 5px;
                border-left: 4px solid #28a745;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )

# -------------------------------------------------
# Main App Function
# -------------------------------------------------
def main():
    # Load CSS
    load_css()

    st.title("APMT Project Insights")
    st.markdown(
        '<div class="main-header">Pastoral Market Transformation Monitoring</div>',
        unsafe_allow_html=True,
    )

    # --- Auto-load dataset ---
    st.sidebar.header("Data Source")
    data_path = os.path.join(project_root, "APMT_Longitudinal_Survey.csv")

    if not os.path.exists(data_path):
        st.error(f"Data file not found at: {data_path}")
        st.info(
            "Please ensure APMT_Longitudinal_Survey.csv is in the project root directory."
        )
        return

    st.sidebar.write("Auto-loaded file:")
    st.sidebar.code(os.path.abspath(data_path))

    if st.sidebar.button("Reload data"):
        st.cache_data.clear()
        st.sidebar.success("Cache cleared. Reloading…")
        st.rerun()

    try:
        # Load data
        df = load_apmt_csv(data_path)
        st.success(f"Data loaded successfully: {len(df):,} records")

        # Prepare geo assets (county / sub-county GeoJSON)
        with st.spinner("Preparing base maps…"):
            ok_geo = ensure_geo_assets()
        if not ok_geo:
            st.warning(
                "Base maps unavailable — county/sub-county outlines will be hidden."
            )

        # Show data preview
        with st.expander("Data Preview", expanded=False):
            st.write(f"Columns detected ({len(df.columns)}):")
            st.write(list(df.columns))
            st.write(f"Total records: {len(df)}")
            st.dataframe(df.head(10))

        # Process data
        processor = APMTDataProcessor(df)

        # Calculate all metrics (herd, offtake, profits, climate, food security, etc.)
        calculate_all_metrics(processor)

        # Data quality section
        render_data_quality_section(processor.df, processor.dq_issues)

        # Create sidebar filters and get selected page
        selected_page = create_sidebar(processor)

        # Render selected page
        if selected_page == "Field Outlook":
            render_field_outlook(processor)
        elif selected_page == "Pastoral Productivity":
            render_pastoral_productivity(processor)
        elif selected_page == "Pastoral Livelihoods":
            render_pastoral_livelihoods(processor)
        elif selected_page == "Feed & Fodder":
            render_feed_fodder(processor)
        elif selected_page == "Sheep Offtake":
            render_sheep_offtake(processor)
        elif selected_page == "Goat Offtake":
            render_goat_offtake(processor)
        elif selected_page == "Payments":
            render_payments(processor)
        elif selected_page == "County Comparator":
            render_county_comparator(processor)
        elif selected_page == "Gender Inclusion":
            render_gender_inclusion(processor)
        elif selected_page == "Climate Impact":
            render_climate_impact(processor)
        elif selected_page == "KPMD Participation":
            render_kpmd_participation(processor)
        elif selected_page == "Food Security":
            render_food_security(processor)
        elif selected_page == "Panel Analysis":
            render_panel_analysis(processor)
        elif selected_page == "P&L Analysis":
            render_pl_analysis(processor)

    except FileNotFoundError:
        st.error("The specified data file was not found.")
        st.code(data_path)
        st.info("Please check the path or ensure the file exists at this location.")
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        import traceback

        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
