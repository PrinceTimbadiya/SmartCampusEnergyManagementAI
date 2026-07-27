"""
Main Streamlit Application (Dashboard Only)
Smart Campus Energy Management Dashboard
Student: Prince Timbadiya
Date: July 2026
"""

import streamlit as st
import sys
import pandas as pd
import plotly.express as px
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import utilities
from utils.dashboard_utils import *

# Page configuration
st.set_page_config(
    page_title="Smart Campus Energy Management",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",  # Collapse sidebar by default
)

# Define BASE_DIR
BASE_DIR = Path(__file__).parent


# Load CSS
def load_css():
    css_path = BASE_DIR / "css" / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()

# Initialize session state
if "predictions" not in st.session_state:
    st.session_state.predictions = []


# Load data
@st.cache_data
def load_data():
    return load_dataset()


df = load_data()
stats = get_summary_stats(df)

# ============================================
# MAIN CONTENT - DASHBOARD
# ============================================

# Hero Banner
st.markdown(
    """
<div class="hero-banner">
    <h1>⚡ Smart Campus Energy Management</h1>
    <div class="subtitle">AI-Enabled Energy Management for Sustainable Campuses</div>
    <div class="student-info">
        <strong>Prince Timbadiya</strong> · BCA · Microsoft + 1M1B Green Skills & Applied AI Internship
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# KPI Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
    <div class="kpi-card blue">
        <div class="kpi-icon">📊</div>
        <div class="kpi-label">Total Records</div>
        <div class="kpi-value">{stats['total_records']}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
    <div class="kpi-card green">
        <div class="kpi-icon">🏢</div>
        <div class="kpi-label">Buildings</div>
        <div class="kpi-value">{stats['buildings']}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
    <div class="kpi-card gold">
        <div class="kpi-icon">⚡</div>
        <div class="kpi-label">Avg Consumption</div>
        <div class="kpi-value">{stats['avg_consumption']}</div>
        <div style="font-size:0.8rem;color:#7f8c8d;">kWh</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
    <div class="kpi-card danger">
        <div class="kpi-icon">🔥</div>
        <div class="kpi-label">Max Consumption</div>
        <div class="kpi-value">{stats['max_consumption']}</div>
        <div style="font-size:0.8rem;color:#7f8c8d;">kWh</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Quick Statistics
st.markdown("### 📊 Quick Statistics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Average Temperature",
        f"{df['Average_Temperature_C'].mean():.1f}°C",
        delta=f"{df['Average_Temperature_C'].max():.1f}°C max",
    )

with col2:
    st.metric(
        "Average Occupancy",
        f"{int(df['Occupancy_Count'].mean())} persons",
        delta=f"{int(df['Occupancy_Count'].max())} max",
    )

with col3:
    st.metric("Date Range", stats["date_range"])

st.markdown("---")

# Energy Consumption Chart
st.markdown("### 📈 Energy Consumption Trend")

# Daily trend
daily_avg = df.groupby("Date")["Electricity_Consumption_kWh"].mean().reset_index()

fig = px.line(
    daily_avg,
    x="Date",
    y="Electricity_Consumption_kWh",
    title="Daily Average Electricity Consumption",
    labels={"Electricity_Consumption_kWh": "Consumption (kWh)"},
)
fig.update_layout(
    xaxis_title="Date", yaxis_title="Average Consumption (kWh)", hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Building Comparison
st.markdown("### 🏢 Building-wise Consumption")

col1, col2 = st.columns(2)

with col1:
    # Bar Chart
    building_avg = (
        df.groupby("Building")["Electricity_Consumption_kWh"].mean().reset_index()
    )
    fig = px.bar(
        building_avg,
        x="Building",
        y="Electricity_Consumption_kWh",
        title="Average Consumption by Building",
        color="Building",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Pie Chart
    building_share = (
        df.groupby("Building")["Electricity_Consumption_kWh"].sum().reset_index()
    )
    fig = px.pie(
        building_share,
        values="Electricity_Consumption_kWh",
        names="Building",
        title="Energy Consumption Share by Building",
        hole=0.3,
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Correlation Analysis
st.markdown("### 🔗 Correlation Analysis")

col1, col2 = st.columns(2)

with col1:
    # Scatter: Occupancy vs Consumption
    fig = px.scatter(
        df,
        x="Occupancy_Count",
        y="Electricity_Consumption_kWh",
        color="Building",
        title="Occupancy vs Electricity Consumption",
        trendline="ols",
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Scatter: Temperature vs Consumption
    fig = px.scatter(
        df,
        x="Average_Temperature_C",
        y="Electricity_Consumption_kWh",
        color="Building",
        title="Temperature vs Electricity Consumption",
        trendline="ols",
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Energy Tips
st.markdown("### 💡 Quick Energy Tips")
tips = get_energy_tips()[:4]
cols = st.columns(4)
for i, tip in enumerate(tips):
    with cols[i]:
        st.info(tip)

st.markdown("---")

# Footer
st.markdown(
    """
<div class="footer">
    <strong>Smart Campus Energy Management</strong> · Developed by <strong>Prince Timbadiya</strong> · 
    Microsoft + 1M1B Green Skills & Applied AI Internship · 2026
</div>
""",
    unsafe_allow_html=True,
)
