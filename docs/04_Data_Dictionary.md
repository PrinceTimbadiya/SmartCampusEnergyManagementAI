4. DATA DICTIONARY

   4.1 Column Descriptions
   Column Name Data Type Unit Description Range Example
   Date Date - Date of electricity consumption recording 2026-05-01 to 2026-08-16 2026-06-15
   Building Categorical (String) - Campus building/facility name Academic Block, Library, Canteen, Hostel Academic Block
   Electricity_Consumption_kWh Numerical (Float) Kilowatt-hours Total electricity consumed by building on that day 85 - 220 182.5
   Occupancy_Count Numerical (Integer) Persons Number of people present in building that day 100 - 470 370
   Average_Temperature_C Numerical (Float) Degrees Celsius Average outdoor temperature for that day 28 - 43 41.0

   4.2 Data Statistics Summary
   Metric Electricity (kWh) Occupancy Temperature (°C)
   Mean 148.2 315.6 37.1
   Median 148.0 318.0 37.0
   Min 85.0 100.0 28.0
   Max 222.0 470.0 43.0
   Std Dev 35.8 98.2 3.8

   4.3 Building-wise Statistics
   Building Records Avg Consumption (kWh) Avg Occupancy Avg Temp (°C)
   Academic Block 135 168.4 338.5 37.2
   Library 125 98.2 128.3 36.8
   Canteen 120 136.5 278.4 37.5
   Hostel 120 192.6 420.8 36.9
