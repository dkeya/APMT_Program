# apmt_dashboard/components/comparison_cards.py
import streamlit as st
import pandas as pd
import numpy as np
from utils.stats import lsmeans_by_group, fmt_lsmean_note

def create_comparison_cards(data, metric_col, title, format_str="{:.1f}", group_col='kpmd_registered'):
    """Create comparison cards for KPMD vs Non-KPMD groups."""
    try:
        if metric_col not in data.columns:
            st.info(f"Column '{metric_col}' not found in dataset.")
            return
        
        kpmd_data = data[data[group_col] == 1]
        non_kpmd_data = data[data[group_col] == 0]
        
        # Get controls for LSMeans
        controls = []
        for candidate in ['County', 'Gender', 'total_sr', 'month', 'panel_wave']:
            if candidate in data.columns and candidate != group_col:
                controls.append(candidate)
        
        lsm = lsmeans_by_group(data.dropna(subset=[metric_col]), metric_col, group_col, controls=controls) or {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            v = kpmd_data[metric_col].mean() if (metric_col in kpmd_data.columns and len(kpmd_data) > 0) else 0
            txt = format_str.format(v if pd.notna(v) else 0)
            st.markdown(f"""
            <div class="metric-card kpmd-card">
                <h4>KPMD Registered</h4>
                <h3>{txt}</h3>
                <small>n={len(kpmd_data)}</small>
                {fmt_lsmean_note(format_str.format(lsm.get(1, v)) if isinstance(lsm, dict) else "")}
            </div>
            """, unsafe_allow_html=True)

        with col2:
            v = non_kpmd_data[metric_col].mean() if (metric_col in non_kpmd_data.columns and len(non_kpmd_data) > 0) else 0
            txt = format_str.format(v if pd.notna(v) else 0)
            st.markdown(f"""
            <div class="metric-card non-kpmd-card">
                <h4>Non-KPMD</h4>
                <h3>{txt}</h3>
                <small>n={len(non_kpmd_data)}</small>
                {fmt_lsmean_note(format_str.format(lsm.get(0, v)) if isinstance(lsm, dict) else "")}
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.warning(f"Could not create comparison cards for {metric_col}: {e}")