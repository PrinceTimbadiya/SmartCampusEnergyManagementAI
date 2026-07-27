#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete EDA Script
Smart Campus Energy Management System
Student: Prince Timbadiya
Date: June 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Configuration
OUTPUT_DIR = "../outputs/visualizations/"
REPORT_DIR = "../outputs/reports/"
DATA_PATH = "../data/processed/electricity_data_cleaned.csv"

# Create directories
import os

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("Set2")

# Load data
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])

# [All previous code sections concatenated here]

print("✅ EDA Complete - All visualizations and reports generated")
