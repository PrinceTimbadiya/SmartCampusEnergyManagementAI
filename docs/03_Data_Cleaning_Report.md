2. DATA CLEANING

   2.1 Data Cleaning Steps Applied
   Step Action Details
   1 Date Standardization Converted all dates to YYYY-MM-DD format
   2 Missing Value Check Scanned all columns for null/empty values
   3 Duplicate Removal Checked for duplicate records
   4 Outlier Detection Identified values outside expected ranges
   5 Consistency Check Verified building names are uniform
   6 Data Type Validation Ensured correct data types for each column
   7 Range Validation Checked values are within realistic ranges

   2.2 Missing Values Report
   Column Missing Count Percentage Action
   Date 0 0% No action needed
   Building 0 0% No action needed
   Electricity_Consumption_kWh 2 0.4% Filled with building average
   Occupancy_Count 1 0.2% Filled with previous day value
   Average_Temperature_C 0 0% No action needed

   2.3 Outlier Detection Report
   Building Column Outliers Detected Action
   Academic Block Electricity 3 records > 195 kWh Verified - peak usage days
   Hostel Electricity 2 records > 215 kWh Verified - summer peak
   Hostel Occupancy 1 record > 460 Verified - special event
   Canteen Electricity 2 records < 115 kWh Verified - low attendance days

   2.4 Duplicate Records
   Finding Count Action
   Duplicate rows found 0 No action needed
   Duplicate date-building 0 No action needed

   2.5 Data Quality Score
   Metric Score
   Completeness 99.4%
   Consistency 100%
   Accuracy 98%
   Timeliness 100%
   Overall Quality 99.35%
