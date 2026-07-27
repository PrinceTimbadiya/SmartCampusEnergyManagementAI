"""
Prediction module for Smart Campus Energy Management System
Author: Prince Timbadiya
Date: June 2026
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

from config import MODEL_PATH, SCALER_PATH, SAFE_LIMITS, BUILDINGS
from preprocessing import feature_engineering, prepare_features


class EnergyPredictor:
    """
    Energy Consumption Prediction Class
    """

    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        """
        Initialize predictor with trained model and scaler

        Parameters:
        model_path (str): Path to trained model
        scaler_path (str): Path to fitted scaler
        """
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print("✅ Model and scaler loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
            self.scaler = None

    def predict_single(
        self,
        building,
        occupancy,
        temperature,
        date=None,
        lag_1=None,
        rolling_mean_3=None,
        rolling_mean_7=None,
    ):
        """
        Predict energy consumption for a single building

        Parameters:
        building (str): Building name
        occupancy (int): Occupancy count
        temperature (float): Average temperature
        date (datetime): Date for prediction (default: today)
        lag_1 (float): Previous day's consumption
        rolling_mean_3 (float): 3-day rolling average
        rolling_mean_7 (float): 7-day rolling average

        Returns:
        dict: Prediction results
        """
        if self.model is None:
            return {"error": "Model not loaded"}

        # Prepare input data
        if date is None:
            date = datetime.now()

        # Create a single record
        data = {
            "Building": [building],
            "Occupancy_Count": [occupancy],
            "Average_Temperature_C": [temperature],
            "Date": [date],
        }

        df = pd.DataFrame(data)

        # Feature engineering
        df_engineered = feature_engineering(df)

        # Use provided lag values or estimate
        if lag_1 is None:
            df_engineered["Lag_1"] = df_engineered["Electricity_Consumption_kWh"] * 0.95
        else:
            df_engineered["Lag_1"] = lag_1

        if rolling_mean_3 is None:
            df_engineered["Rolling_Mean_3"] = (
                df_engineered["Electricity_Consumption_kWh"] * 0.97
            )
        else:
            df_engineered["Rolling_Mean_3"] = rolling_mean_3

        if rolling_mean_7 is None:
            df_engineered["Rolling_Mean_7"] = (
                df_engineered["Electricity_Consumption_kWh"] * 0.98
            )
        else:
            df_engineered["Rolling_Mean_7"] = rolling_mean_7

        # Prepare features
        X = prepare_features(df_engineered, for_prediction=True)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Make prediction
        prediction = self.model.predict(X_scaled)[0]

        # Check against safe limit
        safe_limit = SAFE_LIMITS.get(building, 200)
        is_alert = prediction > safe_limit

        # Generate recommendation
        if is_alert:
            recommendation = f"⚠️ {building} expected to exceed safe limit. Consider reducing non-essential load."
        else:
            recommendation = f"✅ {building} usage within safe limits."

        return {
            "building": building,
            "predicted_consumption": round(prediction, 2),
            "safe_limit": safe_limit,
            "occupancy": occupancy,
            "temperature": temperature,
            "is_alert": is_alert,
            "recommendation": recommendation,
            "date": date.strftime("%Y-%m-%d"),
        }

    def predict_building_day(self, building, date=None):
        """
        Predict energy consumption for a building for a specific day

        Parameters:
        building (str): Building name
        date (datetime): Date to predict (default: today)

        Returns:
        dict: Prediction results
        """
        # Get average values for the building
        avg_occupancy = {
            "Academic Block": 335,
            "Library": 128,
            "Canteen": 278,
            "Hostel": 420,
        }

        avg_temp = {
            "Academic Block": 37.2,
            "Library": 36.8,
            "Canteen": 37.5,
            "Hostel": 36.9,
        }

        if date is None:
            date = datetime.now()

        return self.predict_single(
            building=building,
            occupancy=avg_occupancy.get(building, 300),
            temperature=avg_temp.get(building, 37),
            date=date,
        )

    def predict_all_buildings(self, date=None):
        """
        Predict energy consumption for all buildings

        Parameters:
        date (datetime): Date to predict (default: today)

        Returns:
        pd.DataFrame: Predictions for all buildings
        """
        results = []

        for building in BUILDINGS:
            prediction = self.predict_building_day(building, date)
            results.append(prediction)

        return pd.DataFrame(results)

    def predict_future_week(self, building=None):
        """
        Predict energy consumption for the next week

        Parameters:
        building (str): Building name (default: all buildings)

        Returns:
        pd.DataFrame: 7-day predictions
        """
        results = []

        for i in range(7):
            date = datetime.now() + timedelta(days=i)

            if building:
                # Predict for specific building
                pred = self.predict_building_day(building, date)
                results.append(pred)
            else:
                # Predict for all buildings
                day_predictions = self.predict_all_buildings(date)
                results.extend(day_predictions.to_dict("records"))

        return pd.DataFrame(results)


def predict_usage(
    building, occupancy, temperature, model_path=MODEL_PATH, scaler_path=SCALER_PATH
):
    """
    Convenience function to predict energy usage

    Parameters:
    building (str): Building name
    occupancy (int): Occupancy count
    temperature (float): Average temperature
    model_path (str): Path to model
    scaler_path (str): Path to scaler

    Returns:
    dict: Prediction results
    """
    predictor = EnergyPredictor(model_path, scaler_path)
    return predictor.predict_single(building, occupancy, temperature)


def generate_energy_report(predictions_df):
    """
    Generate energy report from predictions

    Parameters:
    predictions_df (pd.DataFrame): Predictions data

    Returns:
    str: Formatted report
    """
    report = []
    report.append("=" * 60)
    report.append("  ENERGY CONSUMPTION PREDICTION REPORT")
    report.append("=" * 60)
    report.append(f"\nReport Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("-" * 40)

    for _, row in predictions_df.iterrows():
        report.append(f"\n{row['building']}:")
        report.append(f"  Predicted Usage: {row['predicted_consumption']} kWh")
        report.append(f"  Safe Limit: {row['safe_limit']} kWh")
        report.append(f"  Status: {'⚠️ ALERT' if row['is_alert'] else '✅ NORMAL'}")
        report.append(f"  Recommendation: {row['recommendation']}")

    report.append("\n" + "=" * 60)

    # Summary statistics
    total_predicted = predictions_df["predicted_consumption"].sum()
    total_limit = predictions_df["safe_limit"].sum()
    alerts = predictions_df["is_alert"].sum()

    report.append("\n📊 SUMMARY:")
    report.append(f"  Total Predicted Usage: {total_predicted:.2f} kWh")
    report.append(f"  Total Safe Limit: {total_limit:.2f} kWh")
    report.append(f"  Alerts Generated: {alerts}")
    report.append("-" * 40)

    return "\n".join(report)


def main():
    """
    Test prediction functionality
    """
    print("\n" + "=" * 60)
    print("  SMART CAMPUS ENERGY MANAGEMENT - PREDICTION TEST")
    print("=" * 60)

    # Initialize predictor
    predictor = EnergyPredictor()

    if predictor.model is None:
        print("❌ Please train the model first using train_model.py")
        return

    # Test single prediction
    print("\n📊 Single Building Prediction:")
    print("-" * 40)
    result = predictor.predict_single(
        building="Academic Block",
        occupancy=350,
        temperature=40,
        date=datetime(2026, 6, 20),
    )

    for key, value in result.items():
        print(f"  {key}: {value}")

    # Test all buildings
    print("\n\n📊 All Buildings Prediction:")
    print("-" * 40)
    predictions = predictor.predict_all_buildings()
    print(predictions.to_string())

    # Generate report
    print("\n\n📄 Energy Report:")
    print("-" * 40)
    report = generate_energy_report(predictions)
    print(report)

    print("\n" + "=" * 60)
    print("✅ PREDICTION TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
