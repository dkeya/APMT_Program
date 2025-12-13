# apmt_dashboard/pages/panel_analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from components.charts import create_time_series_chart, create_comparison_bar_chart


def render_panel_analysis(processor):
    """Render the Panel Analysis dashboard page."""
    st.header("📈 Longitudinal Panel Analysis")
    
    # Check if panel data is available
    if not getattr(processor, "is_panel_data", False):
        render_no_panel_data_message(processor)
        return
    
    # Panel summary
    with st.expander("📊 Panel Data Structure", expanded=True):
        render_panel_summary(processor)
    
    # Tabs for different panel analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Household Trajectories", 
        "Difference-in-Differences",
        "Attrition Analysis",
        "Wave Comparisons",
        "Time-Series Analysis"
    ])
    
    with tab1:
        render_household_trajectories(processor)
    
    with tab2:
        render_difference_in_differences(processor)
    
    with tab3:
        render_attrition_analysis(processor)
    
    with tab4:
        render_wave_comparisons(processor)
    
    with tab5:
        render_time_series_analysis(processor)


def render_no_panel_data_message(processor):
    """Render message when no panel data is available."""
    st.info("""
    ### No panel data structure detected
    
    To enable longitudinal analysis, your dataset needs:
    1. **Household ID** column (e.g., 'household_id', 'HHID')
    2. **Time period** information (date column)
    
    Current dataset appears to be cross-sectional or missing identifiers.
    """)
    
    # Show what we detected
    with st.expander("Data Structure Details"):
        cols = list(processor.df.columns)
        st.write("**Available Columns (first 20):**", cols[:20])
        if "Household ID" in cols:
            st.write(f"**Household ID detected:** Yes ({processor.df['Household ID'].nunique()} unique households)")
        if "int_date" in cols:
            st.write(f"**Date column detected:** Yes ({processor.df['int_date'].nunique()} unique dates)")


def render_panel_summary(processor):
    """Render panel data summary."""
    if not hasattr(processor, "panel_manager"):
        st.info("Panel manager not available")
        return
    
    structure = processor.panel_manager.panel_structure
    
    st.markdown(f"""
    ### Basic Information
    - **Total Households**: {structure['households']:,}
    - **Total Observations**: {len(processor.df):,}
    - **Time Periods**: {len(structure['waves'])} 
    - **Periods Covered**: {', '.join(sorted(structure['waves']))}
    
    ### Panel Characteristics
    - **Observations per HH**: 
      - Min: {structure['observations_per_hh']['min']}
      - Max: {structure['observations_per_hh']['max']}
      - Mean: {structure['observations_per_hh']['mean']:.1f} ± {structure['observations_per_hh']['std']:.1f}
    - **Panel Type**: {'Balanced' if structure['balanced'] else 'Unbalanced'}
    """)
    
    # Show wave distribution
    if "panel_wave" in processor.df.columns:
        wave_counts = processor.df["panel_wave"].value_counts().sort_index()
        fig = px.bar(
            x=wave_counts.index, 
            y=wave_counts.values,
            title="Observations per Time Period",
            labels={"x": "Time Period", "y": "Number of Observations"}
        )
        st.plotly_chart(fig, use_container_width=True)


def render_household_trajectories(processor):
    """Render household trajectories analysis."""
    st.subheader("Household Trajectories Over Time")
    
    df = processor.df.copy()
    
    # Select metric to track
    metric_options = [
        "total_sr", "net_profit", "rcsi_30", "birth_rate_per_100",
        "income_kpmd", "income_non_kpmd", "total_revenue", "total_costs"
    ]
    available_metrics = [m for m in metric_options if m in df.columns]
    
    if not available_metrics:
        st.info("No trajectory metrics available.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        selected_metric = st.selectbox("Select metric to track", available_metrics)
    with col2:
        show_treated_only = st.checkbox("Show treated households only", value=False)
    
    plot_data = df.copy()
    
    if show_treated_only and "ever_treated" in plot_data.columns:
        plot_data = plot_data[plot_data["ever_treated"] == 1]
    
    if plot_data.empty:
        st.info("No data available after applying filters.")
        return
    
    # Limit number of households for readability
    if "panel_hhid" in plot_data.columns:
        unique_hhs = plot_data["panel_hhid"].unique()
        if len(unique_hhs) > 30:
            st.warning(f"Showing 30 random households out of {len(unique_hhs)} for clarity.")
            sample_hhs = np.random.choice(unique_hhs, 30, replace=False)
            plot_data = plot_data[plot_data["panel_hhid"].isin(sample_hhs)]
    
    if "panel_wave" in plot_data.columns and "panel_hhid" in plot_data.columns:
        # Build hover data dynamically from existing columns
        hover_cols = [c for c in ["County", "kpmd_registered", "panel_wave_num", "panel_quarter"] if c in plot_data.columns]
        hover_data = hover_cols if hover_cols else None
        
        fig = px.line(
            plot_data,
            x="panel_wave",
            y=selected_metric,
            color="panel_hhid",
            title=f"{selected_metric.replace('_', ' ').title()} Trajectories",
            labels={
                "panel_wave": "Time Period",
                selected_metric: selected_metric.replace("_", " ").title()
            },
            hover_data=hover_data
        )
        
        # Add treatment status averages if available
        if "kpmd_registered" in plot_data.columns:
            avg_by_wave = (
                plot_data.groupby(["panel_wave", "kpmd_registered"])[selected_metric]
                .mean()
                .reset_index()
            )
            avg_by_wave["KPMD Status"] = avg_by_wave["kpmd_registered"].map({1: "KPMD", 0: "Non-KPMD"})
            
            kpmd_mask = avg_by_wave["kpmd_registered"] == 1
            non_kpmd_mask = avg_by_wave["kpmd_registered"] == 0
            
            if kpmd_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=avg_by_wave.loc[kpmd_mask, "panel_wave"],
                        y=avg_by_wave.loc[kpmd_mask, selected_metric],
                        mode="lines+markers",
                        line=dict(width=4, color="red", dash="dash"),
                        name="KPMD Average",
                        showlegend=True
                    )
                )
            
            if non_kpmd_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=avg_by_wave.loc[non_kpmd_mask, "panel_wave"],
                        y=avg_by_wave.loc[non_kpmd_mask, selected_metric],
                        mode="lines+markers",
                        line=dict(width=4, color="blue", dash="dash"),
                        name="Non-KPMD Average",
                        showlegend=True
                    )
                )
        
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

def render_difference_in_differences(processor):
    """Render difference-in-differences analysis."""
    st.subheader("Difference-in-Differences Analysis")
    
    df = processor.df.copy()
    
    required_cols = ["ever_treated", "panel_wave"]
    if not all(c in df.columns for c in required_cols):
        st.info("DiD requires treatment timing and panel wave information (ever_treated, panel_wave).")
        return
    
    # Select outcome variable
    outcome_options = [
        "net_profit", "total_revenue", "rcsi_30", "total_sr",
        "income_kpmd", "income_non_kpmd", "profit_margin"
    ]
    available_outcomes = [o for o in outcome_options if o in df.columns]
    
    if not available_outcomes:
        st.info("No outcome variables available for DiD.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        selected_outcome = st.selectbox("Select outcome variable", available_outcomes)
    
    with col2:
        time_periods = sorted(df["panel_wave"].dropna().unique().tolist())
        if len(time_periods) < 2:
            st.info("Need at least 2 time periods for DiD.")
            return
        
        before_period = st.selectbox("Before period", time_periods[:-1], index=0)
        after_period = st.selectbox("After period", time_periods[1:], index=len(time_periods) - 2)
    
    # Sanity check on ordering
    if time_periods.index(before_period) >= time_periods.index(after_period):
        st.warning("The 'Before' period must come before the 'After' period in time.")
        return
    
    # Restrict to the two selected waves
    df_sub = df[df["panel_wave"].isin([before_period, after_period])].copy()
    df_sub = df_sub.dropna(subset=[selected_outcome, "panel_wave", "ever_treated"])
    
    if df_sub.empty:
        st.info("No data available for the selected periods and outcome.")
        return
    
    if "panel_hhid" not in df_sub.columns:
        st.info("DiD requires a stable household identifier (panel_hhid).")
        return
    
    # Aggregate to household x wave x treatment (mean within cell if duplicates)
    hh_wave = (
        df_sub
        .groupby(["panel_hhid", "panel_wave", "ever_treated"])[selected_outcome]
        .mean()
        .reset_index()
    )
    
    # Pivot to wide (before / after columns)
    pivot = hh_wave.pivot_table(
        index=["panel_hhid", "ever_treated"],
        columns="panel_wave",
        values=selected_outcome
    )
    
    if before_period not in pivot.columns or after_period not in pivot.columns:
        st.info("Insufficient data to form complete before/after pairs.")
        return
    
    pivot = pivot[[before_period, after_period]].dropna()
    if pivot.empty:
        st.info("No households observed in both selected periods.")
        return
    
    pivot["delta"] = pivot[after_period] - pivot[before_period]
    pivot = pivot.reset_index()
    
    treated = pivot[pivot["ever_treated"] == 1]["delta"]
    control = pivot[pivot["ever_treated"] == 0]["delta"]
    
    if treated.empty or control.empty:
        st.info("Need both treated and control households with observations in both periods.")
        return
    
    # Compute means and DiD estimate
    treat_pre = hh_wave[(hh_wave["panel_wave"] == before_period) & (hh_wave["ever_treated"] == 1)][selected_outcome].mean()
    treat_post = hh_wave[(hh_wave["panel_wave"] == after_period) & (hh_wave["ever_treated"] == 1)][selected_outcome].mean()
    control_pre = hh_wave[(hh_wave["panel_wave"] == before_period) & (hh_wave["ever_treated"] == 0)][selected_outcome].mean()
    control_post = hh_wave[(hh_wave["panel_wave"] == after_period) & (hh_wave["ever_treated"] == 0)][selected_outcome].mean()
    
    treat_diff = treat_post - treat_pre
    control_diff = control_post - control_pre
    did_estimate = treat_diff - control_diff
    
    # Simple t-test on change scores
    t_stat, p_value = stats.ttest_ind(
        treated, control, equal_var=False, nan_policy="omit"
    )
    
    # Build summary table (as plain strings, no Styler)
    summary_df = pd.DataFrame({
        "Group": ["Treated (ever_treated=1)", "Control (ever_treated=0)"],
        f"Mean {selected_outcome} (Before {before_period})": [treat_pre, control_pre],
        f"Mean {selected_outcome} (After {after_period})": [treat_post, control_post],
        "Change (After - Before)": [treat_diff, control_diff],
    })
    
    # Create a display version with formatted strings for numeric columns
    display_df = summary_df.copy()
    num_cols = display_df.columns.drop("Group")
    for col in num_cols:
        display_df[col] = display_df[col].apply(
            lambda x: f"{x:,.2f}" if pd.notna(x) else ""
        )
    
    st.markdown("### DiD Summary Table")
    st.dataframe(display_df)
    
    st.markdown("### DiD Estimate")
    st.markdown(f"""
    - **Outcome**: `{selected_outcome}`
    - **Before period**: `{before_period}`
    - **After period**: `{after_period}`
    - **Treated change**: {treat_diff:,.2f}
    - **Control change**: {control_diff:,.2f}
    - **Difference-in-Differences (DiD) estimate**: **{did_estimate:,.2f}**
    - **t-statistic (Δtreated vs Δcontrol)**: {t_stat:,.2f}
    - **p-value**: {p_value:,.4f}
    """)
    
    st.caption(
        "Interpretation: The DiD estimate is the additional change in the treated group relative to the control group, "
        "after netting out baseline differences and common time trends. The p-value is from a t-test comparing household-level change scores."
    )

def render_attrition_analysis(processor):
    """Render attrition analysis."""
    st.subheader("Attrition Analysis")
    
    if not hasattr(processor, "panel_manager"):
        st.info("Panel manager not available for attrition analysis.")
        return
    
    if not hasattr(processor.panel_manager, "attrition_df") or processor.panel_manager.attrition_df.empty:
        st.info("Need at least 2 time periods for attrition analysis.")
        return
    
    attrition_df = processor.panel_manager.attrition_df.copy()
    
    st.markdown("### Household Transitions Between Time Periods")
    
    # Round numeric columns and add percent-as-string columns for display
    if "attrition_rate" in attrition_df.columns:
        attrition_df["attrition_rate"] = attrition_df["attrition_rate"].round(1)
    if "retention_rate" in attrition_df.columns:
        attrition_df["retention_rate"] = attrition_df["retention_rate"].round(1)
    
    st.dataframe(attrition_df)
    
    # Plot attrition rates
    if "attrition_rate" in attrition_df.columns:
        fig1 = px.bar(
            attrition_df,
            x="from_wave",
            y="attrition_rate",
            title="Attrition Rate Between Time Periods",
            text=attrition_df["attrition_rate"].round(1),
            labels={"from_wave": "From Period", "attrition_rate": "Attrition Rate (%)"}
        )
        fig1.update_traces(textposition="outside")
        st.plotly_chart(fig1, use_container_width=True)
    
    # Plot retention rates
    if "retention_rate" in attrition_df.columns:
        fig2 = px.bar(
            attrition_df,
            x="from_wave",
            y="retention_rate",
            title="Retention Rate Between Time Periods",
            text=attrition_df["retention_rate"].round(1),
            labels={"from_wave": "From Period", "retention_rate": "Retention Rate (%)"}
        )
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)


def render_wave_comparisons(processor):
    """Render wave comparisons."""
    st.subheader("Time Period Comparisons")
    
    df = processor.df.copy()
    
    metric_options = [
        "net_profit", "total_revenue", "rcsi_30", "total_sr",
        "kpmd_registered", "income_kpmd", "income_non_kpmd",
        "birth_rate_per_100", "mortality_rate_per_100"
    ]
    available_metrics = [m for m in metric_options if m in df.columns]
    
    if not available_metrics:
        st.info("No metrics available for time period comparison.")
        return
    
    selected_metrics = st.multiselect(
        "Select metrics to compare across time periods",
        available_metrics,
        default=available_metrics[:3] if len(available_metrics) >= 3 else available_metrics
    )
    
    if not selected_metrics:
        return
    
    for metric in selected_metrics:
        st.markdown(f"### {metric.replace('_', ' ').title()}")
        
        if "panel_wave" in df.columns and "kpmd_registered" in df.columns:
            wave_means = (
                df.groupby(["panel_wave", "kpmd_registered"])[metric]
                .mean()
                .reset_index()
            )
            wave_means["KPMD Status"] = wave_means["kpmd_registered"].map({1: "KPMD", 0: "Non-KPMD"})
            
            fig = create_time_series_chart(
                wave_means,
                x_col="panel_wave",
                y_col=metric,
                color_col="KPMD Status",
                title=f"{metric.replace('_', ' ').title()} by Time Period and KPMD Status",
                markers=True
            )
            
            if fig:
                # Add overall mean as bar chart (transparent overlay)
                overall_means = (
                    df.groupby("panel_wave")[metric]
                    .mean()
                    .reset_index()
                )
                fig.add_trace(
                    go.Bar(
                        x=overall_means["panel_wave"],
                        y=overall_means[metric],
                        name="Overall Mean",
                        opacity=0.3,
                        marker_color="gray"
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)


def render_time_series_analysis(processor):
    """Render time series analysis."""
    st.subheader("Time-Series Analysis")
    
    df = processor.df.copy()
    
    variable_options = [
        "net_profit", "total_revenue", "total_costs", "rcsi_30",
        "total_sr", "income_kpmd", "income_non_kpmd"
    ]
    available_vars = [v for v in variable_options if v in df.columns]
    
    if not available_vars:
        st.info("No variables available for time-series analysis.")
        return
    
    selected_vars = st.multiselect(
        "Select variables for time-series analysis",
        available_vars,
        default=available_vars[:2] if len(available_vars) >= 2 else available_vars
    )
    
    if not selected_vars:
        return
    
    for var in selected_vars:
        st.markdown(f"### {var.replace('_', ' ').title()}")
        
        if "panel_wave" in df.columns:
            ts_data = (
                df.groupby("panel_wave")[var]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            
            # Standard error and 95% CI, safe handling when count <= 1 or std is NaN
            ts_data["se"] = ts_data.apply(
                lambda row: row["std"] / np.sqrt(row["count"])
                if row["count"] > 1 and pd.notna(row["std"])
                else 0,
                axis=1
            )
            ts_data["ci_lower"] = ts_data["mean"] - 1.96 * ts_data["se"]
            ts_data["ci_upper"] = ts_data["mean"] + 1.96 * ts_data["se"]
            
            fig = go.Figure()
            
            # Mean line
            fig.add_trace(go.Scatter(
                x=ts_data["panel_wave"],
                y=ts_data["mean"],
                mode="lines+markers",
                name="Mean",
                line=dict(color="blue", width=2)
            ))
            
            # Confidence interval (band)
            fig.add_trace(go.Scatter(
                x=ts_data["panel_wave"].tolist() + ts_data["panel_wave"].tolist()[::-1],
                y=ts_data["ci_upper"].tolist() + ts_data["ci_lower"].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(0, 100, 255, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% CI",
                showlegend=True
            ))
            
            fig.update_layout(
                title=f"{var.replace('_', ' ').title()} Time Series with Confidence Intervals",
                xaxis_title="Time Period",
                yaxis_title=var.replace("_", " ").title(),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
