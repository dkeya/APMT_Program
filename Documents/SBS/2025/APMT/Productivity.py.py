# pages/1_📈_Productivity.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from app import load_data, create_derived_metrics

st.set_page_config(page_title="Productivity Dashboard", layout="wide")

st.title("🐑 Pastoral Productivity Dashboard")

# Load data (uses cached data from main app)
if 'df' not in st.session_state:
    # Try to load data automatically
    df = load_data()
    if df is not None:
        st.session_state.df = create_derived_metrics(df)

if 'df' in st.session_state:
    df = st.session_state.df
    
    # Productivity metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_herd = df['total_sr_owned'].mean()
        st.metric("Avg Herd Size", f"{avg_herd:.1f}")
    
    with col2:
        mortality = df['mortality_rate'].mean()
        st.metric("Avg Mortality Rate", f"{mortality:.1f}%")
    
    with col3:
        birth_rate = (df['total_sr_born'].sum() / df['total_sr_owned'].sum() * 100) if df['total_sr_owned'].sum() > 0 else 0
        st.metric("Birth Rate", f"{birth_rate:.1f}%")
    
    with col4:
        offtake_rate = (df['total_sales'].sum() / df['total_sr_owned'].sum() * 100) if df['total_sr_owned'].sum() > 0 else 0
        st.metric("Offtake Rate", f"{offtake_rate:.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Herd composition
        if 'County' in df.columns:
            herd_by_county = df.groupby('County')['total_sr_owned'].mean().reset_index()
            fig = px.bar(herd_by_county, x='County', y='total_sr_owned', 
                        title="Average Herd Size by County")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sales comparison
        if all(col in df.columns for col in ['total_kpmd_sales', 'total_non_kpmd_sales']):
            kpmd_sales = df['total_kpmd_sales'].sum()
            non_kpmd_sales = df['total_non_kpmd_sales'].sum()
            
            fig = go.Figure(data=[
                go.Bar(name='KPMD Sales', x=['Sales'], y=[kpmd_sales], marker_color='blue'),
                go.Bar(name='Non-KPMD Sales', x=['Sales'], y=[non_kpmd_sales], marker_color='red')
            ])
            fig.update_layout(title="Sales Channel Comparison", barmode='group')
            st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Please upload data in the main dashboard first")