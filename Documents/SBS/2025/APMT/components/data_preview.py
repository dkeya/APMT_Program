# apmt_dashboard/components/data_preview.py
import streamlit as st
import pandas as pd

def show_data_preview(df, title="Data Preview", max_rows=10):
    """Display a data preview with basic statistics."""
    with st.expander(title, expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
            st.metric("Total Columns", f"{len(df.columns):,}")
        
        with col2:
            if 'kpmd_registered' in df.columns:
                kpmd_count = df['kpmd_registered'].sum()
                st.metric("KPMD Registered", f"{kpmd_count:,}")
                st.metric("Non-KPMD", f"{len(df) - kpmd_count:,}")
        
        st.write("First few records:")
        st.dataframe(df.head(max_rows))
        
        if st.checkbox("Show column summary"):
            col_summary = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null Count': df.notna().sum(),
                'Null Count': df.isna().sum(),
                'Null %': (df.isna().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_summary)

def show_column_statistics(df, column):
    """Display statistics for a specific column."""
    if column not in df.columns:
        st.warning(f"Column '{column}' not found in data.")
        return
    
    st.subheader(f"Statistics for: {column}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Count", f"{df[column].count():,}")
    
    with col2:
        st.metric("Null Count", f"{df[column].isna().sum():,}")
    
    if pd.api.types.is_numeric_dtype(df[column]):
        with col3:
            st.metric("Mean", f"{df[column].mean():.2f}")
        
        with col4:
            st.metric("Std Dev", f"{df[column].std():.2f}")
        
        # Additional stats
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Min", f"{df[column].min():.2f}")
        with col6:
            st.metric("25%", f"{df[column].quantile(0.25):.2f}")
        with col7:
            st.metric("Median", f"{df[column].median():.2f}")
        with col8:
            st.metric("75%", f"{df[column].quantile(0.75):.2f}")
    
    elif pd.api.types.is_datetime64_any_dtype(df[column]):
        with col3:
            st.metric("Earliest", df[column].min().strftime("%Y-%m-%d"))
        
        with col4:
            st.metric("Latest", df[column].max().strftime("%Y-%m-%d"))
    
    # Value counts for categorical
    if df[column].nunique() < 50:
        st.subheader("Value Distribution")
        value_counts = df[column].value_counts(dropna=False).reset_index()
        value_counts.columns = ['Value', 'Count']
        value_counts['Percentage'] = (value_counts['Count'] / len(df) * 100).round(2)
        st.dataframe(value_counts)

def show_data_quality_metrics(df):
    """Display data quality metrics."""
    st.subheader("Data Quality Metrics")
    
    metrics = []
    
    # Duplicates
    duplicate_rows = df.duplicated().sum()
    metrics.append(("Duplicate Rows", f"{duplicate_rows:,}", "Rows with identical values across all columns"))
    
    # Missing values
    total_cells = df.size
    missing_cells = df.isna().sum().sum()
    missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0
    metrics.append(("Missing Values", f"{missing_cells:,} ({missing_pct:.1f}%)", "Empty or null values"))
    
    # Columns with high missing
    high_missing = (df.isna().mean() > 0.5).sum()
    metrics.append(("Columns >50% Missing", f"{high_missing:,}", "Columns with majority missing data"))
    
    # Display metrics
    for i in range(0, len(metrics), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(metrics):
                name, value, desc = metrics[i + j]
                with cols[j]:
                    st.metric(name, value)
                    st.caption(desc)