"""
Dashboard Utility Functions
Smart Campus Energy Management System
Student: Prince Timbadiya
Date: June 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set plotting style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("Set2")


def load_dataset():
    """Load cleaned dataset"""
    data_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "processed"
        / "electricity_data_cleaned.csv"
    )

    if data_path.exists():
        df = pd.read_csv(data_path)
        # df["Date"] = pd.to_datetime(df["Date"])
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        return df
    else:
        # Create sample data if file doesn't exist
        return create_sample_data()


def create_sample_data():
    """Create sample data for demonstration"""
    dates = pd.date_range("2026-05- ", "2026-08-16", freq="D")
    buildings = ["Academic Block", "Library", "Canteen", "Hostel"]

    data = []
    for date in dates:
        for building in buildings:
            if building == "Academic Block":
                base = 150 + (date - dates[0]).days * 0.3
                base += np.random.normal(0, 8)
            elif building == "Library":
                base = 95 + (date - dates[0]).days * 0.15
                base += np.random.normal(0, 5)
            elif building == "Canteen":
                base = 120 + (date - dates[0]).days * 0.2
                base += np.random.normal(0, 6)
            else:  # Hostel
                base = 180 + (date - dates[0]).days * 0.25
                base += np.random.normal(0, 10)

            occupancy = {
                "Academic Block": 320 + np.random.normal(0, 20),
                "Library": 120 + np.random.normal(0, 15),
                "Canteen": 250 + np.random.normal(0, 25),
                "Hostel": 400 + np.random.normal(0, 30),
            }[building]

            temp = 34 + (date - dates[0]).days * 0.04 + np.random.normal(0, 2)

            data.append(
                {
                    "Date": date,
                    "Building": building,
                    "Electricity_Consumption_kWh": round(max(70, base), 2),
                    "Occupancy_Count": int(max(50, occupancy)),
                    "Average_Temperature_C": round(max(28, min(43, temp)), 1),
                }
            )

    return pd.DataFrame(data)


def load_model():
    """Load trained model"""
    # model_path = Path(__file__).parent.parent.parent / "models" / "best_model.pkl"
    model_path = (
        Path(__file__).parent.parent.parent / "models" / "energy_prediction_model.pkl"
    )

    if model_path.exists():
        try:
            model = joblib.load(model_path)
            return model
        except:
            return None
    return None


def load_scaler():
    """Load scaler"""
    scaler_path = Path(__file__).parent.parent.parent / "models" / "scaler.pkl"

    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
            return scaler
        except:
            return None
    return None


def load_encoder():
    """Load encoder"""
    encoder_path = Path(__file__).parent.parent.parent / "models" / "encoder.pkl"

    if encoder_path.exists():
        try:
            encoder = joblib.load(encoder_path)
            return encoder
        except:
            return None
    return None


def load_metrics():
    """Load model metrics"""
    metrics_path = Path(__file__).parent.parent.parent / "models" / "model_metrics.json"

    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except:
            return None
    return None


def get_building_avg(df):
    """Get average consumption by building"""
    return df.groupby("Building")["Electricity_Consumption_kWh"].mean().to_dict()


def get_summary_stats(df):
    """Get summary statistics"""
    return {
        "total_records": len(df),
        "buildings": df["Building"].nunique(),
        "date_range": f"{df['Date'].min().strftime('%b %d')} - {df['Date'].max().strftime('%b %d, %Y')}",
        "avg_consumption": round(df["Electricity_Consumption_kWh"].mean(), 2),
        "max_consumption": round(df["Electricity_Consumption_kWh"].max(), 2),
        "min_consumption": round(df["Electricity_Consumption_kWh"].min(), 2),
    }


def get_building_stats(df):
    """Get building-wise statistics"""
    stats = (
        df.groupby("Building")
        .agg(
            {
                "Electricity_Consumption_kWh": ["mean", "min", "max", "std"],
                "Occupancy_Count": ["mean", "min", "max"],
            }
        )
        .round(2)
    )
    return stats


def predict_consumption(model, scaler, features):
    """Make prediction using loaded model"""
    if model is None or scaler is None:
        return None

    try:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        return prediction
    except:
        return None


def format_prediction(prediction):
    """Format prediction for display"""
    if prediction is None:
        return "N/A"
    return f"{prediction:.2f} kWh"


def get_status(prediction, building):
    """Get status based on prediction"""
    safe_limits = {"Academic Block": 180, "Library": 120, "Canteen": 155, "Hostel": 210}

    limit = safe_limits.get(building, 200)

    if prediction <= limit * 0.9:
        return "Safe", "safe"
    elif prediction <= limit:
        return "Caution", "warning"
    else:
        return "Alert", "danger"


def get_recommendation(building, prediction):
    """Get recommendation based on prediction"""
    safe_limits = {"Academic Block": 180, "Library": 120, "Canteen": 155, "Hostel": 210}

    limit = safe_limits.get(building, 200)

    if prediction > limit:
        return f"⚠️ {building} expected to exceed safe limit. Consider reducing non-essential load."
    elif prediction > limit * 0.9:
        return f"📊 {building} usage approaching limit. Monitor closely."
    else:
        return f"✅ {building} usage within safe limits. Continue monitoring."


def get_energy_tips():
    """Get energy saving tips"""
    return [
        "💡 Turn off lights when leaving rooms",
        "🌡️ Set AC temperature to 24°C for optimal efficiency",
        "🔌 Unplug devices when not in use",
        "💻 Enable power-saving mode on computers",
        "🌿 Use natural lighting when possible",
        "📊 Monitor energy usage regularly",
        "🚿 Reduce hot water usage",
        "❄️ Clean AC filters monthly",
        "🔦 Use LED lighting throughout campus",
        "🏢 Conduct regular energy audits",
    ]


def get_building_list():
    """Get list of buildings"""
    return ["Academic Block", "Library", "Canteen", "Hostel"]
