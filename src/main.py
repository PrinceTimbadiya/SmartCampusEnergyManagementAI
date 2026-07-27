"""
Main application for Smart Campus Energy Management System
Author: Prince Timbadiya
Date: June 2026
"""

import sys
import os
from datetime import datetime
import pandas as pd

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import CLEANED_DATA_FILE
from utils import load_data, print_section_header
from train_model import main as train_main
from predict import EnergyPredictor, generate_energy_report
from dashboard import EnergyDashboard


def show_menu():
    """Display main menu"""
    print("\n" + "=" * 60)
    print("  SMART CAMPUS ENERGY MANAGEMENT SYSTEM")
    print("  Microsoft + 1M1B Green Skills & Applied AI Internship")
    print("  Student: Prince Timbadiya")
    print("=" * 60)
    print("\nPlease select an option:")
    print("  1. Train Model")
    print("  2. View Dashboard")
    print("  3. Make Predictions")
    print("  4. Generate Energy Report")
    print("  5. Exit")
    print("-" * 40)


def view_dashboard():
    """View dashboard option"""
    print_section_header("VIEW DASHBOARD")

    dashboard = EnergyDashboard()
    df = load_data(CLEANED_DATA_FILE)

    if df is not None:
        df["Date"] = pd.to_datetime(df["Date"])
        dashboard.display_dashboard(df)
    else:
        print("❌ No data available. Please ensure data exists in data/processed/")


def make_predictions():
    """Make predictions option"""
    print_section_header("MAKE PREDICTIONS")

    predictor = EnergyPredictor()

    if predictor.model is None:
        print("❌ Model not trained. Please train the model first (Option 1)")
        return

    print("\nAvailable Buildings:")
    from config import BUILDINGS

    for i, building in enumerate(BUILDINGS, 1):
        print(f"  {i}. {building}")

    try:
        choice = int(input("\nSelect building (1-4): ")) - 1
        if 0 <= choice < len(BUILDINGS):
            building = BUILDINGS[choice]

            # Get input parameters
            occupancy = int(input(f"Enter occupancy count for {building}: "))
            temperature = float(input("Enter average temperature (°C): "))

            # Make prediction
            result = predictor.predict_single(building, occupancy, temperature)

            print("\n" + "=" * 40)
            print("  PREDICTION RESULT")
            print("=" * 40)
            for key, value in result.items():
                print(f"  {key}: {value}")
            print("=" * 40)
        else:
            print("❌ Invalid selection")
    except ValueError:
        print("❌ Invalid input. Please enter numbers only.")


def generate_report():
    """Generate energy report option"""
    print_section_header("GENERATE ENERGY REPORT")

    predictor = EnergyPredictor()

    if predictor.model is None:
        print("❌ Model not trained. Please train the model first (Option 1)")
        return

    predictions = predictor.predict_all_buildings()
    report = generate_energy_report(predictions)
    print(report)

    # Save report to file
    from config import REPORTS_DIR

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = REPORTS_DIR / f"energy_report_{timestamp}.txt"

    with open(report_file, "w") as f:
        f.write(report)
    print(f"\n✅ Report saved to: {report_file}")


def main():
    """Main application loop"""
    while True:
        show_menu()
        choice = input("\nEnter your choice (1-5): ")

        if choice == "1":
            train_main()
        elif choice == "2":
            view_dashboard()
        elif choice == "3":
            make_predictions()
        elif choice == "4":
            generate_report()
        elif choice == "5":
            print("\n👋 Thank you for using Smart Campus Energy Management System!")
            print("   Developed by Prince Timbadiya")
            sys.exit()
        else:
            print("❌ Invalid choice. Please enter a number between 1 and 5.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
