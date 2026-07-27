"""
Utility functions for Smart Campus Energy Management System
Author: Prince Timbadiya
Date: June 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from pathlib import Path


def load_data(file_path):
    """
    Load data from CSV file

    Parameters:
    file_path (str or Path): Path to CSV file

    Returns:
    pd.DataFrame: Loaded data
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Data loaded successfully: {len(df)} records")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def save_data(df, file_path):
    """
    Save data to CSV file

    Parameters:
    df (pd.DataFrame): Data to save
    file_path (str or Path): Path to save CSV file
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"✅ Data saved successfully: {file_path}")
    except Exception as e:
        print(f"❌ Error saving data: {e}")


def get_building_stats(df):
    """
    Get statistics for each building

    Parameters:
    df (pd.DataFrame): Dataframe with building column

    Returns:
    pd.DataFrame: Building statistics
    """
    stats = (
        df.groupby("Building")
        .agg(
            {
                "Electricity_Consumption_kWh": ["mean", "min", "max", "std"],
                "Occupancy_Count": ["mean", "min", "max"],
                "Average_Temperature_C": ["mean", "min", "max"],
            }
        )
        .round(2)
    )
    return stats


def get_summary_stats(df):
    """
    Get overall summary statistics

    Parameters:
    df (pd.DataFrame): Dataframe

    Returns:
    pd.DataFrame: Summary statistics
    """
    return df.describe().round(2)


def save_metrics(metrics, file_path):
    """
    Save model metrics to JSON file

    Parameters:
    metrics (dict): Metrics to save
    file_path (str or Path): Path to save JSON file
    """
    try:
        with open(file_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"✅ Metrics saved: {file_path}")
    except Exception as e:
        print(f"❌ Error saving metrics: {e}")


def load_metrics(file_path):
    """
    Load metrics from JSON file

    Parameters:
    file_path (str or Path): Path to JSON file

    Returns:
    dict: Loaded metrics
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading metrics: {e}")
        return None


def create_visualization(df, plot_type, title, x_col, y_col, hue_col=None):
    """
    Create and save visualization

    Parameters:
    df (pd.DataFrame): Data to visualize
    plot_type (str): Type of plot (line, bar, scatter, box)
    title (str): Plot title
    x_col (str): X-axis column
    y_col (str): Y-axis column
    hue_col (str): Color grouping column

    Returns:
    matplotlib.figure.Figure: Created figure
    """
    plt.figure(figsize=(12, 6))

    if plot_type == "line":
        sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col)
    elif plot_type == "bar":
        sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col)
    elif plot_type == "scatter":
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col)
    elif plot_type == "box":
        sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col)
    elif plot_type == "heatmap":
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", center=0)

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.tight_layout()

    return plt.gcf()


def detect_anomalies(df, column, threshold=2):
    """
    Detect anomalies using z-score

    Parameters:
    df (pd.DataFrame): Dataframe
    column (str): Column to check
    threshold (float): Z-score threshold

    Returns:
    pd.DataFrame: Anomalies detected
    """
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    anomalies = df[z_scores > threshold]
    return anomalies


def generate_alerts(df, safe_limits):
    """
    Generate alerts based on safe limits

    Parameters:
    df (pd.DataFrame): Dataframe with predictions
    safe_limits (dict): Safe limits per building

    Returns:
    list: Alert messages
    """
    alerts = []

    for building, limit in safe_limits.items():
        building_data = df[df["Building"] == building]
        if not building_data.empty:
            max_usage = building_data["Electricity_Consumption_kWh"].max()
            if max_usage > limit:
                alerts.append(
                    f"⚠️ {building} usage exceeded safe limit: {max_usage:.1f} kWh (Limit: {limit} kWh)"
                )

    return alerts


def format_date_range(df):
    """Format date range information"""
    date_min = df["Date"].min()
    date_max = df["Date"].max()
    return f"{date_min} to {date_max}"


def print_section_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)
