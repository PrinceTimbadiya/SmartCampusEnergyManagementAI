"""
AI Prediction Page
Smart Campus Energy Management Dashboard
Student: Prince Timbadiya
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from utils.dashboard_utils import *

st.markdown(
    """
<div class="section-header">🤖 AI Energy Prediction</div>
""",
    unsafe_allow_html=True,
)

# Load model and scaler
model = load_model()
scaler = load_scaler()
encoder = load_encoder()
df = load_dataset()

if model is None or scaler is None:
    st.warning("⚠️ Model not trained. Please train the model first.")
    st.info("""
To train the model, run:
```bash
python src/train_model.py
Then restart the dashboard.
""")
    st.stop()

st.markdown("### 📝 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    building = st.selectbox(
        "🏢 Building", options=["Academic Block", "Library", "Canteen", "Hostel"]
    )

occupancy = st.number_input(
    "👥 Occupancy Count", min_value=0, max_value=500, value=300, step=10
)

temperature = st.slider(
    "🌡️ Temperature (°C)", min_value=28.0, max_value=43.0, value=37.0, step=0.5
)

with col2:
    day_of_week = st.selectbox(
        "📅 Day of Week",
        options=[
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
    )
day_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}

month = st.selectbox("📆 Month", options=["May", "June", "July", "August"])
month_map = {"May": 5, "June": 6, "July": 7, "August": 8}

is_weekend = st.checkbox("🗓️ Weekend")

# Lag features
st.markdown("### 📊 Historical Data (Optional)")
col1, col2, col3 = st.columns(3)

with col1:
    lag_1 = st.number_input(
        "Previous Day Consumption (kWh)",
        min_value=50.0,
        max_value=250.0,
        value=150.0,
        step=5.0,
    )

with col2:
    rolling_3 = st.number_input(
        "3-Day Rolling Average (kWh)",
        min_value=50.0,
        max_value=250.0,
        value=155.0,
        step=5.0,
    )

with col3:
    rolling_7 = st.number_input(
        "7-Day Rolling Average (kWh)",
        min_value=50.0,
        max_value=250.0,
        value=160.0,
        step=5.0,
    )

prediction = None

# Predict button
if st.button("🔮 Predict", use_container_width=True):
    with st.spinner("Making prediction..."):

        # Encode building
        building_encoded = encoder.transform([building])[0]

        # Create features
        features = np.array(
            [
                occupancy,
                temperature,
                building_encoded,
                day_map[day_of_week],
                month_map[month],
                1 if is_weekend else 0,
                lag_1,
                rolling_3,
                rolling_7,
            ]
        ).reshape(1, -1)

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]

        # Store in session
        st.session_state.last_prediction = {
            "building": building,
            "prediction": prediction,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

if prediction is not None:

    # Display result
    st.markdown("---")
    st.markdown("### 🎯 Prediction Result")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        status, status_class = get_status(prediction, building)
    recommendation = get_recommendation(building, prediction)

    st.markdown(
        f"""

    <div class="prediction-card"> <div class="prediction-label">Predicted Consumption</div> <div class="prediction-value">{prediction:.2f}</div> <div class="prediction-unit">kWh</div> <div> <span class="status-badge {status_class}">{status}</span> </div> <div style="margin-top:15px; font-size:1rem; color:#2c3e50;"> {recommendation} </div> </div> """,
        unsafe_allow_html=True,
    )
    # Additional info
    st.markdown("### 📋 Additional Information")

    safe_limits = {"Academic Block": 180, "Library": 120, "Canteen": 155, "Hostel": 210}
    limit = safe_limits.get(building, 200)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Safe Limit", f"{limit} kWh")
    with col2:
        st.metric("Difference", f"{prediction - limit:.2f} kWh")
    with col3:
        st.metric("Capacity", f"{(prediction/limit*100):.1f}%")

# Prediction history
# if st.session_state.predictions:
#     st.markdown("---")
#     st.markdown("### 📜 Prediction History")
if "predictions" not in st.session_state:
    st.session_state.predictions = []

history_df = pd.DataFrame(st.session_state.predictions)
st.dataframe(history_df, use_container_width=True)

if st.button("Clear History"):
    st.session_state.predictions = []
    st.rerun()

# Energy tips
st.markdown("---")
st.markdown("### 💡 Quick Energy Tips")
tips = get_energy_tips()[:3]
cols = st.columns(3)
for i, tip in enumerate(tips):
    with cols[i]:
        st.info(tip)
