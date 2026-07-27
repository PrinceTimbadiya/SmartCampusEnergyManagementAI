"""
Dashboard module for Smart Campus Energy Management System
Author: Prince Timbadiya
Date: June 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

from config import BUILDINGS, SAFE_LIMITS
from utils import create_visualization, get_building_stats, generate_alerts
from predict import EnergyPredictor


class EnergyDashboard:
    """
    Energy Management Dashboard
    """

    def __init__(self):
        """Initialize dashboard"""
        self.predictor = EnergyPredictor()

        # Set up matplotlib style
        plt.style.use("seaborn-v0_8-darkgrid")
        self.figsize = (12, 6)
        self.colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

    def plot_building_comparison(self, df):
        """
        Create building comparison chart

        Parameters:
        df (pd.DataFrame): Data with Building and Electricity_Consumption_kWh

        Returns:
        matplotlib.figure.Figure: Plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Calculate average consumption by building
        avg_consumption = df.groupby("Building")["Electricity_Consumption_kWh"].mean()

        # Create bar chart
        bars = ax.bar(
            avg_consumption.index, avg_consumption.values, color=self.colors, alpha=0.7
        )

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f} kWh",
                ha="center",
                va="bottom",
            )

        ax.set_xlabel("Building", fontsize=12)
        ax.set_ylabel("Average Consumption (kWh)", fontsize=12)
        ax.set_title(
            "Average Energy Consumption by Building", fontsize=16, fontweight="bold"
        )

        # Add safe limit lines
        for building in BUILDINGS:
            if building in SAFE_LIMITS:
                ax.axhline(
                    y=SAFE_LIMITS[building], color="red", linestyle="--", alpha=0.5
                )

        plt.tight_layout()
        return fig

    def plot_trend_analysis(self, df):
        """
        Create trend analysis chart

        Parameters:
        df (pd.DataFrame): Data with Date and Electricity_Consumption_kWh

        Returns:
        matplotlib.figure.Figure: Plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Group by date and calculate average consumption
        daily_avg = df.groupby("Date")["Electricity_Consumption_kWh"].mean()

        # Create line chart
        ax.plot(daily_avg.index, daily_avg.values, color="#2E86AB", linewidth=2)

        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Average Consumption (kWh)", fontsize=12)
        ax.set_title("Daily Energy Consumption Trend", fontsize=16, fontweight="bold")
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    def plot_occupancy_relationship(self, df):
        """
        Create occupancy vs consumption scatter plot

        Parameters:
        df (pd.DataFrame): Data with Occupancy_Count and Electricity_Consumption_kWh

        Returns:
        matplotlib.figure.Figure: Plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Create scatter plot
        for i, building in enumerate(BUILDINGS):
            subset = df[df["Building"] == building]
            ax.scatter(
                subset["Occupancy_Count"],
                subset["Electricity_Consumption_kWh"],
                label=building,
                alpha=0.6,
                s=50,
            )

        ax.set_xlabel("Occupancy Count", fontsize=12)
        ax.set_ylabel("Electricity Consumption (kWh)", fontsize=12)
        ax.set_title(
            "Occupancy vs Electricity Consumption", fontsize=16, fontweight="bold"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_temperature_relationship(self, df):
        """
        Create temperature vs consumption scatter plot

        Parameters:
        df (pd.DataFrame): Data with Average_Temperature_C and Electricity_Consumption_kWh

        Returns:
        matplotlib.figure.Figure: Plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Create scatter plot
        for i, building in enumerate(BUILDINGS):
            subset = df[df["Building"] == building]
            ax.scatter(
                subset["Average_Temperature_C"],
                subset["Electricity_Consumption_kWh"],
                label=building,
                alpha=0.6,
                s=50,
            )

        ax.set_xlabel("Average Temperature (°C)", fontsize=12)
        ax.set_ylabel("Electricity Consumption (kWh)", fontsize=12)
        ax.set_title(
            "Temperature vs Electricity Consumption", fontsize=16, fontweight="bold"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_heatmap(self, df):
        """
        Create correlation heatmap

        Parameters:
        df (pd.DataFrame): Data with numerical columns

        Returns:
        matplotlib.figure.Figure: Plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Select numerical columns
        numeric_cols = [
            "Electricity_Consumption_kWh",
            "Occupancy_Count",
            "Average_Temperature_C",
        ]
        corr = df[numeric_cols].corr()

        # Create heatmap
        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
        )

        ax.set_title("Correlation Matrix", fontsize=16, fontweight="bold")

        plt.tight_layout()
        return fig

    def create_predictions_plot(self, df_predictions):
        """
        Create predictions visualization

        Parameters:
        df_predictions (pd.DataFrame): Prediction results

        Returns:
        matplotlib.figure.Figure: Plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Create bar chart
        buildings = df_predictions["building"]
        predictions = df_predictions["predicted_consumption"]
        limits = df_predictions["safe_limit"]

        x = np.arange(len(buildings))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2, predictions, width, label="Predicted", color="#2E86AB"
        )
        bars2 = ax.bar(
            x + width / 2, limits, width, label="Safe Limit", color="#F18F01"
        )

        ax.set_xlabel("Building", fontsize=12)
        ax.set_ylabel("Consumption (kWh)", fontsize=12)
        ax.set_title("Predicted vs Safe Limit", fontsize=16, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(buildings)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add alert markers
        for i, (pred, limit, alert) in enumerate(
            zip(predictions, limits, df_predictions["is_alert"])
        ):
            if alert:
                ax.plot(
                    i - width / 2,
                    pred,
                    "r*",
                    markersize=15,
                    label="Alert" if i == 0 else "",
                )

        plt.tight_layout()
        return fig

    def create_summary_cards(self, df):
        """
        Create summary statistics cards

        Parameters:
        df (pd.DataFrame): Data

        Returns:
        dict: Summary statistics
        """
        summary = {
            "total_consumption": df["Electricity_Consumption_kWh"].sum(),
            "avg_consumption": df["Electricity_Consumption_kWh"].mean(),
            "max_consumption": df["Electricity_Consumption_kWh"].max(),
            "min_consumption": df["Electricity_Consumption_kWh"].min(),
            "total_records": len(df),
            "buildings": df["Building"].nunique(),
            "date_range": f"{df['Date'].min()} to {df['Date'].max()}",
        }

        return summary

    def display_dashboard(self, df):
        """
        Display full dashboard

        Parameters:
        df (pd.DataFrame): Data to visualize
        """
        print("\n" + "=" * 60)
        print("  SMART CAMPUS ENERGY MANAGEMENT DASHBOARD")
        print("=" * 60)

        # Summary cards
        print("\n📊 QUICK STATISTICS:")
        print("-" * 40)
        summary = self.create_summary_cards(df)
        print(f"Total Consumption: {summary['total_consumption']:.2f} kWh")
        print(f"Average Consumption: {summary['avg_consumption']:.2f} kWh")
        print(f"Maximum Consumption: {summary['max_consumption']:.2f} kWh")
        print(f"Minimum Consumption: {summary['min_consumption']:.2f} kWh")
        print(f"Records: {summary['total_records']}")
        print(f"Buildings: {summary['buildings']}")
        print(f"Date Range: {summary['date_range']}")

        # Generate visualizations
        print("\n📊 GENERATING VISUALIZATIONS:")
        print("-" * 40)

        try:
            # Plot 1: Building comparison
            fig1 = self.plot_building_comparison(df)
            print("✅ Building comparison plot created")

            # Plot 2: Trend analysis
            fig2 = self.plot_trend_analysis(df)
            print("✅ Trend analysis plot created")

            # Plot 3: Occupancy relationship
            fig3 = self.plot_occupancy_relationship(df)
            print("✅ Occupancy relationship plot created")

            # Plot 4: Temperature relationship
            fig4 = self.plot_temperature_relationship(df)
            print("✅ Temperature relationship plot created")

            # Plot 5: Heatmap
            fig5 = self.plot_heatmap(df)
            print("✅ Correlation heatmap created")

            # Show all plots
            plt.show()

        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")

        # Generate predictions if predictor is available
        if self.predictor.model is not None:
            print("\n🔮 GENERATING PREDICTIONS:")
            print("-" * 40)

            try:
                predictions = self.predictor.predict_all_buildings()
                print("✅ Predictions generated")

                # Plot predictions
                fig6 = self.create_predictions_plot(predictions)
                print("✅ Predictions plot created")

                plt.show()

                # Display predictions
                print("\n📊 Prediction Results:")
                print(predictions.to_string(index=False))

            except Exception as e:
                print(f"⚠️ Prediction error: {e}")
                print("   (Model may not be trained yet)")
        else:
            print("\n⚠️ Prediction model not available.")
            print("   Please train the model first using train_model.py")


def main():
    """
    Main dashboard function
    """
    from config import CLEANED_DATA_FILE
    from utils import load_data

    print("\n" + "=" * 60)
    print("  SMART CAMPUS ENERGY MANAGEMENT - DASHBOARD")
    print("=" * 60)

    # Load data
    df = load_data(CLEANED_DATA_FILE)

    if df is None:
        print("❌ No data available. Please run data preparation first.")
        return

    # Convert date
    df["Date"] = pd.to_datetime(df["Date"])

    # Create dashboard
    dashboard = EnergyDashboard()
    dashboard.display_dashboard(df)

    print("\n" + "=" * 60)
    print("✅ DASHBOARD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
