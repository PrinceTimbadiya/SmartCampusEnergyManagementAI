"""
Model training module for Smart Campus Energy Management System
Author: Prince Timbadiya
Date: June 2026
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

from config import MODEL_PATH, SCALER_PATH, METRICS_PATH
from preprocessing import prepare_data_for_model, clean_dataset, load_raw_data
from utils import save_metrics


def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models and compare performance

    Parameters:
    X_train, X_test: Training and test features
    y_train, y_test: Training and test targets

    Returns:
    dict: Trained models
    dict: Model performance metrics
    """
    print("\n🚀 Training Models...")
    print("-" * 40)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n📊 Training {name}...")

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Store results
        results[name] = {
            "MAE": round(mae, 2),
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4),
        }

        trained_models[name] = model

        print(f"   ✅ MAE: {mae:.2f}")
        print(f"   ✅ RMSE: {rmse:.2f}")
        print(f"   ✅ R²: {r2:.4f}")

    return trained_models, results


def select_best_model(models, results, X_train, y_train):
    """
    Select the best model based on R² score

    Parameters:
    models (dict): Trained models
    results (dict): Model metrics
    X_train: Training features
    y_train: Training target

    Returns:
    object: Best model
    str: Best model name
    """
    # Find best model by R²
    best_model_name = max(results, key=lambda x: results[x]["R2"])
    best_model = models[best_model_name]

    # Retrain best model on full training data
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   R² Score: {results[best_model_name]['R2']}")

    best_model.fit(X_train, y_train)

    return best_model, best_model_name


def save_model(model, scaler, metrics, model_path, scaler_path, metrics_path):
    """
    Save trained model, scaler, and metrics

    Parameters:
    model: Trained model
    scaler: Fitted scaler
    metrics (dict): Model metrics
    model_path (str): Path to save model
    scaler_path (str): Path to save scaler
    metrics_path (str): Path to save metrics
    """
    # Save model
    joblib.dump(model, model_path)
    print(f"✅ Model saved: {model_path}")

    # Save scaler
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved: {scaler_path}")

    # Save metrics
    save_metrics(metrics, metrics_path)
    print(f"✅ Metrics saved: {metrics_path}")


def main():
    """
    Main training pipeline
    """
    from config import RAW_DATA_FILE, CLEANED_DATA_FILE

    print("\n" + "=" * 60)
    print("  SMART CAMPUS ENERGY MANAGEMENT - MODEL TRAINING")
    print("=" * 60)

    # 1. Load raw data
    print("\n📂 Step 1: Loading Data")
    print("-" * 40)
    df_raw = load_raw_data(RAW_DATA_FILE)

    # 2. Clean data
    print("\n🧹 Step 2: Cleaning Data")
    print("-" * 40)
    df_cleaned = clean_dataset(df_raw)

    # ----------------------------------------
    # Create and Save Label Encoder
    # ----------------------------------------

    encoder = LabelEncoder()
    encoder.fit(df_cleaned["Building"])

    encoder_path = MODEL_PATH.parent / "encoder.pkl"

    joblib.dump(encoder, encoder_path)

    print(f"✅ Encoder saved: {encoder_path}")

    # Save cleaned data
    df_cleaned.to_csv(CLEANED_DATA_FILE, index=False)
    print(f"✅ Cleaned data saved: {CLEANED_DATA_FILE}")

    # 3. Prepare data for model
    print("\n🔧 Step 3: Preparing Data for Model")
    print("-" * 40)
    X_train, X_test, y_train, y_test, scaler = prepare_data_for_model(df_cleaned)

    # 4. Train models
    print("\n🤖 Step 4: Training Models")
    print("-" * 40)
    models, results = train_models(X_train, X_test, y_train, y_test)

    # 5. Select best model
    print("\n🏆 Step 5: Selecting Best Model")
    print("-" * 40)
    best_model, best_model_name = select_best_model(models, results, X_train, y_train)

    # 6. Save model
    print("\n💾 Step 6: Saving Model")
    print("-" * 40)

    # Prepare metrics to save
    metrics_to_save = {
        "best_model": best_model_name,
        "performance": results[best_model_name],
        "all_models": results,
        "test_size": 0.2,
        "features_used": [
            "Occupancy_Count",
            "Average_Temperature_C",
            "Building_encoded",
            "DayOfWeek",
            "Month",
            "IsWeekend",
            "Lag_1",
            "Rolling_Mean_3",
            "Rolling_Mean_7",
        ],
    }

    save_model(
        best_model, scaler, metrics_to_save, MODEL_PATH, SCALER_PATH, METRICS_PATH
    )

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)

    # Print final summary
    print("\n📊 Model Performance Summary:")
    print("-" * 40)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  MAE: {metrics['MAE']} kWh")
        print(f"  RMSE: {metrics['RMSE']} kWh")
        print(f"  R²: {metrics['R2']}")

    return best_model, scaler, results


if __name__ == "__main__":
    main()
