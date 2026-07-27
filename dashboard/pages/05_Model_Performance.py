"""
Model Performance Page
Smart Campus Energy Management Dashboard
Student: Prince Timbadiya
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from utils.dashboard_utils import *

st.markdown(
    """
<div class="section-header">📉 Model Performance</div>
""",
    unsafe_allow_html=True,
)

model = load_model()
# Load metrics
metrics = load_metrics()

if metrics is None:
    st.warning("⚠️ Model metrics not found. Please train the model first.")
    st.stop()

best_model = metrics.get("best_model", "N/A")
model_metrics = metrics.get("metrics", {})
all_models = metrics.get("all_models", {})

# KPI Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Best Model", best_model)

with col2:
    st.metric("R² Score", f"{model_metrics.get('R2', 0):.4f}")

with col3:
    st.metric("RMSE", f"{model_metrics.get('RMSE', 0):.2f} kWh")

with col4:
    st.metric("MAE", f"{model_metrics.get('MAE', 0):.2f} kWh")

st.markdown("---")

# Model Comparison
st.markdown("### 📊 Model Comparison")

if all_models:
    comparison_df = pd.DataFrame(all_models).T
    comparison_df = comparison_df.sort_values("R2", ascending=False)

    # Metrics selection
    # metric_to_show = st.selectbox(
    #     "Select Metric to Compare", options=["R2", "RMSE", "MAE", "MAPE", "CV_Mean"]
    # )
    metric_to_show = st.selectbox(
        "Select Metric to Compare", options=["R2", "RMSE", "MAE", "MSE"]
    )

    fig = px.bar(
        comparison_df.reset_index(),
        x="index",
        y=metric_to_show,
        title=f"{metric_to_show} Comparison",
        color="index",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(xaxis_title="Model", yaxis_title=metric_to_show)
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.markdown("### 📋 Detailed Comparison")
    st.dataframe(comparison_df, use_container_width=True)

# Feature Importance
# st.markdown("### 🔍 Feature Importance")

# if model is not None and hasattr(model, "feature_importances_"):
#     # Try to load feature columns
#     try:
#         with open("models/feature_columns.json", "r") as f:
#             feature_names = json.load(f)
#     except:
#         feature_names = [
#             "Occupancy",
#             "Temperature",
#             "Building",
#             "DayOfWeek",
#             "Month",
#             "IsWeekend",
#             "Lag_1",
#             "Rolling_Mean_3",
#             "Rolling_Mean_7",
#         ]

#     importance = model.feature_importances_
#     importance_df = pd.DataFrame(
#         {"Feature": feature_names, "Importance": importance}
#     ).sort_values("Importance", ascending=True)

#     fig = px.bar(
#         importance_df,
#         x="Importance",
#         y="Feature",
#         orientation="h",
#         title="Feature Importance Ranking",
#         color="Importance",
#         color_continuous_scale="Blues",
#     )
#     st.plotly_chart(fig, use_container_width=True)

# # Learning Curve
# st.markdown("### 📈 Learning Curve")

# # Try to load or generate learning curve
# st.info("Learning curve visualization would be shown here based on training history")
