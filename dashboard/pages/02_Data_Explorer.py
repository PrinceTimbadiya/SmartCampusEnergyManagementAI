"""
Data Explorer Page
Smart Campus Energy Management Dashboard
Student: Prince Timbadiya
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.dashboard_utils import *

st.markdown(
    """
<div class="section-header">📊 Data Explorer</div>
""",
    unsafe_allow_html=True,
)

# Load data
df = load_dataset()

# Filters
st.markdown("### 🔍 Filter Data")
col1, col2, col3 = st.columns(3)

with col1:
    buildings = st.multiselect(
        "Building", options=df["Building"].unique(), default=df["Building"].unique()
    )

with col2:
    date_range = st.date_input(
        "Date Range",
        value=[df["Date"].min(), df["Date"].max()],
        min_value=df["Date"].min(),
        max_value=df["Date"].max(),
    )

with col3:
    temp_range = st.slider(
        "Temperature Range (°C)",
        min_value=float(df["Average_Temperature_C"].min()),
        max_value=float(df["Average_Temperature_C"].max()),
        value=(
            float(df["Average_Temperature_C"].min()),
            float(df["Average_Temperature_C"].max()),
        ),
    )

# Apply filters
filtered_df = df.copy()
if buildings:
    filtered_df = filtered_df[filtered_df["Building"].isin(buildings)]
if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df["Date"] >= pd.to_datetime(date_range[0]))
        & (filtered_df["Date"] <= pd.to_datetime(date_range[1]))
    ]
filtered_df = filtered_df[
    (filtered_df["Average_Temperature_C"] >= temp_range[0])
    & (filtered_df["Average_Temperature_C"] <= temp_range[1])
]

# Summary
st.markdown(f"### 📈 Dataset Summary ({len(filtered_df)} records)")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", len(filtered_df))
with col2:
    st.metric("Buildings", filtered_df["Building"].nunique())
with col3:
    st.metric(
        "Avg Consumption",
        f"{filtered_df['Electricity_Consumption_kWh'].mean():.2f} kWh",
    )
with col4:
    st.metric("Avg Temp", f"{filtered_df['Average_Temperature_C'].mean():.1f}°C")

# Preview
st.markdown("### 📄 Data Preview")
st.dataframe(
    filtered_df,
    use_container_width=True,
    column_config={
        "Date": st.column_config.DatetimeColumn("Date"),
        "Building": st.column_config.TextColumn("Building"),
        "Electricity_Consumption_kWh": st.column_config.NumberColumn(
            "Consumption (kWh)", format="%.2f"
        ),
        "Occupancy_Count": st.column_config.NumberColumn("Occupancy"),
        "Average_Temperature_C": st.column_config.NumberColumn(
            "Temperature (°C)", format="%.1f"
        ),
    },
)

# Statistics
st.markdown("### 📊 Statistics")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Numerical Statistics")
    st.dataframe(filtered_df.describe().round(2))

with col2:
    st.markdown("#### Building-wise Statistics")
    building_stats = (
        filtered_df.groupby("Building")["Electricity_Consumption_kWh"]
        .agg(["mean", "min", "max", "std"])
        .round(2)
    )
    st.dataframe(building_stats)

# Download
st.markdown("### 💾 Download Data")
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="📥 Download CSV",
    data=csv,
    file_name="campus_energy_data.csv",
    mime="text/csv",
    use_container_width=True,
)
