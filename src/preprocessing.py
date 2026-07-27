"""
Data preprocessing module for Smart Campus Energy Management System
Author: Prince Timbadiya
Date: June 2026
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


def load_raw_data(file_path):
    """
    Load raw data from CSV

    Parameters:
    file_path (str): Path to raw data file

    Returns:
    pd.DataFrame: Raw data
    """
    df = pd.read_csv(file_path)
    print(f"✅ Raw data loaded: {len(df)} records")
    return df


def check_missing_values(df):
    """
    Check for missing values in dataset

    Parameters:
    df (pd.DataFrame): Data to check

    Returns:
    dict: Missing value summary
    """
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100

    missing_summary = pd.DataFrame(
        {"Missing_Count": missing, "Missing_Percent": missing_percent}
    )

    print("\n📊 Missing Values Summary:")
    print(missing_summary)

    return missing_summary


def handle_missing_values(df):
    """
    Handle missing values in dataset

    Parameters:
    df (pd.DataFrame): Data to clean

    Returns:
    pd.DataFrame: Cleaned data
    """
    df_cleaned = df.copy()

    # Fill missing Electricity with building mean
    for building in df["Building"].unique():
        mask = (df_cleaned["Building"] == building) & (
            df_cleaned["Electricity_Consumption_kWh"].isnull()
        )
        if mask.any():
            mean_val = df_cleaned[df_cleaned["Building"] == building][
                "Electricity_Consumption_kWh"
            ].mean()
            df_cleaned.loc[mask, "Electricity_Consumption_kWh"] = mean_val

    # Fill missing Occupancy with previous day's value
    df_cleaned = df_cleaned.sort_values(["Building", "Date"])
    df_cleaned["Occupancy_Count"] = df_cleaned.groupby("Building")[
        "Occupancy_Count"
    ].fillna(method="ffill")

    # If any still missing, fill with overall mean
    df_cleaned["Occupancy_Count"] = df_cleaned["Occupancy_Count"].fillna(
        df_cleaned["Occupancy_Count"].mean()
    )

    print(f"✅ Missing values handled: {df.isnull().sum().sum()} values filled")
    return df_cleaned


def check_duplicates(df):
    """
    Check for duplicate records

    Parameters:
    df (pd.DataFrame): Data to check

    Returns:
    int: Number of duplicate records
    """
    duplicates = df.duplicated(subset=["Date", "Building"]).sum()
    if duplicates > 0:
        print(f"⚠️ Found {duplicates} duplicate records")
    else:
        print("✅ No duplicate records found")
    return duplicates


def remove_duplicates(df):
    """
    Remove duplicate records

    Parameters:
    df (pd.DataFrame): Data to clean

    Returns:
    pd.DataFrame: Data without duplicates
    """
    return df.drop_duplicates(subset=["Date", "Building"])


def standardize_dates(df):
    """
    Standardize date format

    Parameters:
    df (pd.DataFrame): Data to clean

    Returns:
    pd.DataFrame: Data with standardized dates
    """
    df_cleaned = df.copy()
    # df_cleaned["Date"] = pd.to_datetime(df_cleaned["Date"])
    df_cleaned["Date"] = pd.to_datetime(df_cleaned["Date"], dayfirst=True)
    df_cleaned["Date"] = df_cleaned["Date"].dt.strftime("%Y-%m-%d")
    return df_cleaned


def detect_outliers(df, column, threshold=2.5):
    """
    Detect outliers using z-score method

    Parameters:
    df (pd.DataFrame): Data to check
    column (str): Column to check
    threshold (float): Z-score threshold

    Returns:
    pd.DataFrame: Records with outliers
    """
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    outliers = df[z_scores > threshold]

    if len(outliers) > 0:
        print(f"⚠️ Found {len(outliers)} outliers in {column}")
    else:
        print(f"✅ No outliers found in {column}")

    return outliers


def feature_engineering(df):
    """
    Create new features for ML model

    Parameters:
    df (pd.DataFrame): Data to transform

    Returns:
    pd.DataFrame: Data with engineered features
    """
    df_engineered = df.copy()

    # Convert Date to datetime
    df_engineered["Date"] = pd.to_datetime(df_engineered["Date"])

    # Extract date features
    df_engineered["DayOfWeek"] = df_engineered[
        "Date"
    ].dt.dayofweek  # Monday=0, Sunday=6
    df_engineered["Month"] = df_engineered["Date"].dt.month
    df_engineered["Day"] = df_engineered["Date"].dt.day
    df_engineered["IsWeekend"] = df_engineered["DayOfWeek"].isin([5, 6]).astype(int)

    # Encode Building
    label_encoder = LabelEncoder()
    df_engineered["Building_encoded"] = label_encoder.fit_transform(
        df_engineered["Building"]
    )

    # Create lag features (previous day's consumption)
    df_engineered = df_engineered.sort_values(["Building", "Date"])
    df_engineered["Lag_1"] = df_engineered.groupby("Building")[
        "Electricity_Consumption_kWh"
    ].shift(1)

    # Create rolling averages
    df_engineered["Rolling_Mean_3"] = (
        df_engineered.groupby("Building")["Electricity_Consumption_kWh"]
        .rolling(3)
        .mean()
        .reset_index(0, drop=True)
    )
    df_engineered["Rolling_Mean_7"] = (
        df_engineered.groupby("Building")["Electricity_Consumption_kWh"]
        .rolling(7)
        .mean()
        .reset_index(0, drop=True)
    )

    # Drop rows with NaN from lag features
    df_engineered = df_engineered.dropna()

    print(f"✅ Feature engineering complete: {len(df_engineered.columns)} features")
    return df_engineered


def prepare_features(df, for_prediction=False):
    """
    Prepare features for model training/prediction

    Parameters:
    df (pd.DataFrame): Data to prepare
    for_prediction (bool): If True, return features only

    Returns:
    tuple: X features, y target
    """
    # Define feature columns
    feature_cols = [
        "Occupancy_Count",
        "Average_Temperature_C",
        "Building_encoded",
        "DayOfWeek",
        "Month",
        "IsWeekend",
        "Lag_1",
        "Rolling_Mean_3",
        "Rolling_Mean_7",
    ]

    X = df[feature_cols]

    if for_prediction:
        return X

    y = df["Electricity_Consumption_kWh"]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets

    Parameters:
    X (pd.DataFrame): Features
    y (pd.Series): Target
    test_size (float): Proportion for test set
    random_state (int): Random seed

    Returns:
    tuple: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"✅ Data split: {len(X_train)} train, {len(X_test)} test")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler

    Parameters:
    X_train (pd.DataFrame): Training features
    X_test (pd.DataFrame): Test features

    Returns:
    tuple: Scaled X_train, scaled X_test, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("✅ Features scaled successfully")
    return X_train_scaled, X_test_scaled, scaler


def clean_dataset(df):
    """
    Complete data cleaning pipeline

    Parameters:
    df (pd.DataFrame): Raw data

    Returns:
    pd.DataFrame: Cleaned data
    """
    print("\n🔧 Starting Data Cleaning Pipeline...")
    print("-" * 40)

    # 1. Standardize dates
    df = standardize_dates(df)

    # 2. Check missing values
    missing = check_missing_values(df)

    # 3. Handle missing values
    df = handle_missing_values(df)

    # 4. Check duplicates
    duplicates = check_duplicates(df)

    # 5. Remove duplicates
    if duplicates > 0:
        df = remove_duplicates(df)

    # 6. Check outliers
    outlier_cols = ["Electricity_Consumption_kWh", "Occupancy_Count"]
    for col in outlier_cols:
        outliers = detect_outliers(df, col)
        if len(outliers) > 0:
            print(f"⚠️ {len(outliers)} outliers detected in {col}")
            print(outliers[["Building", "Date", col]].head())

    print("\n✅ Data cleaning complete!")
    print(f"Final data shape: {df.shape}")

    return df


def prepare_data_for_model(df):
    """
    Complete data preparation pipeline for ML model

    Parameters:
    df (pd.DataFrame): Cleaned data

    Returns:
    tuple: X_train, X_test, y_train, y_test, scaler
    """
    print("\n🔧 Preparing Data for ML Model...")
    print("-" * 40)

    # 1. Feature engineering
    df_engineered = feature_engineering(df)

    # 2. Prepare features
    X, y = prepare_features(df_engineered)

    # 3. Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 4. Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("\n✅ Data preparation complete!")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
