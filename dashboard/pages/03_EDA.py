"""
EDA Analysis Page
Smart Campus Energy Management Dashboard
Student: Prince Timbadiya
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.dashboard_utils import *

st.markdown(
    """
<div class="section-header">📈 Exploratory Data Analysis</div>
""",
    unsafe_allow_html=True,
)

df = load_dataset()

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Trend Analysis", "🏢 Building Analysis", "📉 Distribution", "🔗 Correlations"]
)

# Tab 1: Trend Analysis
with tab1:
    st.markdown("#### Daily Consumption Trend")

    fig = px.line(
        df.groupby("Date")["Electricity_Consumption_kWh"].mean().reset_index(),
        x="Date",
        y="Electricity_Consumption_kWh",
        title="Daily Average Electricity Consumption",
        labels={"Electricity_Consumption_kWh": "Consumption (kWh)"},
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Average Consumption (kWh)",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
    <div class="chart-container">
        <div class="chart-insight">
            <strong>📌 Insight:</strong> Electricity consumption shows an increasing trend from May to August,
            with fluctuations corresponding to temperature variations and occupancy patterns.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Monthly Trend")

    df["Month"] = df["Date"].dt.month
    monthly_avg = (
        df.groupby("Month")["Electricity_Consumption_kWh"].mean().reset_index()
    )
    monthly_avg["Month"] = monthly_avg["Month"].map(
        {5: "May", 6: "June", 7: "July", 8: "August"}
    )

    fig = px.bar(
        monthly_avg,
        x="Month",
        y="Electricity_Consumption_kWh",
        title="Monthly Average Consumption",
        color="Electricity_Consumption_kWh",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Building Analysis
with tab2:
    st.markdown("#### Consumption by Building")

    fig = px.bar(
        df.groupby("Building")["Electricity_Consumption_kWh"].mean().reset_index(),
        x="Building",
        y="Electricity_Consumption_kWh",
        title="Average Consumption by Building",
        color="Building",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Building-wise Energy Share")

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

# Tab 3: Distribution
with tab3:
    st.markdown("#### Consumption Distribution")

    fig = px.histogram(
        df,
        x="Electricity_Consumption_kWh",
        nbins=30,
        title="Distribution of Electricity Consumption",
        color_discrete_sequence=["#2E86AB"],
    )
    fig.add_vline(
        x=df["Electricity_Consumption_kWh"].mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {df['Electricity_Consumption_kWh'].mean():.1f}",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Temperature Distribution")

    fig = px.histogram(
        df,
        x="Average_Temperature_C",
        nbins=20,
        title="Distribution of Temperature",
        color_discrete_sequence=["#F18F01"],
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 4: Correlations
with tab4:
    st.markdown("#### Correlation Matrix")

    corr_df = df[
        ["Electricity_Consumption_kWh", "Occupancy_Count", "Average_Temperature_C"]
    ].corr()

    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Occupancy vs Consumption")

    fig = px.scatter(
        df,
        x="Occupancy_Count",
        y="Electricity_Consumption_kWh",
        color="Building",
        title="Occupancy vs Electricity Consumption",
        trendline="ols",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Temperature vs Consumption")

    fig = px.scatter(
        df,
        x="Average_Temperature_C",
        y="Electricity_Consumption_kWh",
        color="Building",
        title="Temperature vs Electricity Consumption",
        trendline="ols",
    )
    st.plotly_chart(fig, use_container_width=True)
