# apmt_dashboard/components/charts.py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st

def create_comparison_bar_chart(data, x_col, y_col, color_col=None, title="", barmode='group', 
                               text_format="{:.1f}", x_title=None, y_title=None):
    """Create a comparison bar chart."""
    if data.empty:
        return None
    
    fig = px.bar(
        data, 
        x=x_col, 
        y=y_col, 
        color=color_col,
        title=title,
        barmode=barmode,
        text=data[y_col].apply(lambda x: text_format.format(x))
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        uniformtext_minsize=8, 
        uniformtext_mode='hide',
        xaxis_title=x_title or x_col,
        yaxis_title=y_title or y_col
    )
    
    return fig

def create_time_series_chart(data, x_col, y_col, color_col=None, title="", markers=True):
    """Create a time series chart."""
    if data.empty:
        return None
    
    fig = px.line(
        data,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        markers=markers
    )
    
    return fig

def create_distribution_chart(data, col, title="", nbins=30, color_col=None):
    """Create a distribution histogram."""
    if data.empty or col not in data.columns:
        return None
    
    fig = px.histogram(
        data,
        x=col,
        nbins=nbins,
        title=title,
        color=color_col
    )
    
    return fig

def create_box_plot(data, x_col, y_col, color_col=None, title=""):
    """Create a box plot."""
    if data.empty:
        return None
    
    fig = px.box(
        data,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title
    )
    
    return fig

def create_pie_chart(data, names_col, values_col, title=""):
    """Create a pie chart."""
    if data.empty:
        return None
    
    fig = px.pie(
        data,
        names=names_col,
        values=values_col,
        title=title
    )
    
    return fig

def create_scatter_plot(data, x_col, y_col, color_col=None, size_col=None, title="", hover_data=None):
    """Create a scatter plot."""
    if data.empty:
        return None
    
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        title=title,
        hover_data=hover_data
    )
    
    return fig

def create_choropleth_map(gdf, geojson_data, color_col, title="", color_scale="Viridis"):
    """Create a choropleth map."""
    if gdf.empty:
        return None
    
    fig = px.choropleth_mapbox(
        gdf,
        geojson=geojson_data,
        locations=gdf.index,
        color=color_col,
        mapbox_style="carto-positron",
        zoom=5,
        center={"lat": gdf.geometry.centroid.y.mean(), "lon": gdf.geometry.centroid.x.mean()},
        opacity=0.5,
        title=title,
        color_continuous_scale=color_scale
    )
    
    fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    
    return fig

# NEW FUNCTIONS ADDED FOR P&L ANALYSIS
def create_profit_distribution_chart(data, profit_col='net_profit', title='Distribution of Net Profit'):
    """Create a profit distribution histogram."""
    if data.empty or profit_col not in data.columns:
        return None
    
    fig = px.histogram(
        data,
        x=profit_col,
        title=title,
        labels={profit_col: 'Net Profit (KES)'}
    )
    fig.update_layout(bargap=0.1)
    return fig

def create_profit_by_kpmd_chart(data, profit_col='net_profit', kpmd_col='kpmd_registered'):
    """Create box plot of profit by KPMD status."""
    if data.empty or profit_col not in data.columns or kpmd_col not in data.columns:
        return None
    
    # Create KPMD Status column for visualization
    tmp = data.copy()
    tmp['KPMD Status'] = tmp[kpmd_col].map({1: 'KPMD', 0: 'Non-KPMD'})
    tmp['KPMD Status'] = pd.Categorical(tmp['KPMD Status'], categories=['Non-KPMD', 'KPMD'], ordered=True)
    
    fig = px.box(
        tmp,
        x='KPMD Status',
        y=profit_col,
        color='KPMD Status',
        category_orders={'KPMD Status': ['Non-KPMD', 'KPMD']},
        title='Profit Distribution by Registration',
        labels={'KPMD Status': 'Registration', profit_col: 'Net Profit (KES)'}
    )
    fig.update_layout(legend_title_text='Registration')
    return fig

def create_revenue_composition_chart(data, revenue_cols, title='Average Revenue Composition'):
    """Create pie chart of revenue composition."""
    if data.empty or not revenue_cols:
        return None
    
    # Get average composition
    avg_comp = data[revenue_cols].mean().sort_values(ascending=False)
    fig = px.pie(values=avg_comp.values, names=avg_comp.index, title=title)
    return fig

def create_cost_structure_chart(data, cost_cols, title='Average Cost Composition'):
    """Create bar chart of cost structure."""
    if data.empty or not cost_cols:
        return None
    
    avg_cost = data[cost_cols].mean().sort_values(ascending=False)
    fig = px.bar(
        x=avg_cost.index,
        y=avg_cost.values,
        title=title,
        labels={'x': 'Cost Category', 'y': 'Average Cost (KES)'}
    )
    fig.update_traces(text=avg_cost.round(0), textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    return fig

def create_profit_trend_chart(data, time_col, profit_col='net_profit', kpmd_col='kpmd_registered', 
                            title='Average Net Profit Over Time'):
    """Create time series chart of profit trends."""
    if data.empty or time_col not in data.columns or profit_col not in data.columns:
        return None
    
    # Calculate profit by time period and KPMD status
    profit_ts = data.groupby([time_col, kpmd_col])[profit_col].mean().reset_index()
    profit_ts['KPMD Status'] = profit_ts[kpmd_col].map({1: 'KPMD', 0: 'Non-KPMD'})
    
    fig = px.line(profit_ts, x=time_col, y=profit_col, color='KPMD Status',
                 title=title, markers=True)
    return fig