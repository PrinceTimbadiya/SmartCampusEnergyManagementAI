"""
Configuration file for Smart Campus Energy Management System
Author: Prince Timbadiya
Date: June 2026
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "energy_prediction_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

# Output paths
OUTPUT_DIR = BASE_DIR / "outputs"
VIZ_DIR = OUTPUT_DIR / "visualizations"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Data file names
RAW_DATA_FILE = RAW_DATA_DIR / "electricity_data_raw.csv"
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "electricity_data_cleaned.csv"
CLEANED_DATA_EXCEL = PROCESSED_DATA_DIR / "electricity_data_cleaned.xlsx"

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Building names
BUILDINGS = ["Academic Block", "Library", "Canteen", "Hostel"]

# Feature columns
FEATURE_COLUMNS = [
    "Occupancy_Count",
    "Average_Temperature_C",
    "Building_encoded",
    "DayOfWeek",
    "Month",
    "IsWeekend",
]

# Target column
TARGET_COLUMN = "Electricity_Consumption_kWh"

# Safe limits (kWh)
SAFE_LIMITS = {"Academic Block": 180, "Library": 120, "Canteen": 155, "Hostel": 210}


# Create directories if they don't exist
def create_directories():
    """Create all necessary directories"""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        OUTPUT_DIR,
        VIZ_DIR,
        REPORTS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Call on import
create_directories()
