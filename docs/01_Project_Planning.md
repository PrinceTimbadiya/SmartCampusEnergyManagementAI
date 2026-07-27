# PHASE 1: PROJECT PLANNING

## Smart Campus Energy Management using Artificial Intelligence

### Microsoft + 1M1B Green Skills & Applied AI Internship

**Student:** Prince Timbadiya  
**Course:** Bachelor of Computer Applications (BCA)  
**Date:** June 2026

---

## 1. PROJECT SCOPE

### 1.1 Project Overview

This project aims to develop an AI-enabled energy management system for a sustainable college campus. The system leverages machine learning to predict electricity consumption patterns across different campus buildings, enabling proactive energy optimization and cost reduction.

### 1.2 Problem Domain

The campus faces significant energy inefficiencies:

- Academic Block electricity usage surged from 145 kWh to 182 kWh (+25%)
- Hostel consumption peaked at 210-215 kWh on 16-17 June
- Consumption correlates with temperature (34°C → 41°C)
- Lights and fans remain ON in empty classrooms
- No predictive analytics or automated controls exist

### 1.3 Solution Domain

The proposed Smart Campus Energy Management System addresses these challenges through:

- **Predictive Intelligence**: AI models forecast next-hour and next-day energy demand
- **Automated Monitoring**: Real-time tracking of consumption patterns
- **Proactive Alerting**: Early warnings before wastage occurs
- **Data-Driven Recommendations**: Specific, actionable suggestions for energy savings

### 1.4 Target Users

- College Administration (decision-makers)
- Facility Management Team (operational users)
- Students (awareness and behavior change)
- Sustainability Committee (policy development)

### 1.5 Geographic Scope

- Academic Block (classrooms, offices)
- Library
- Canteen
- Hostel

---

## 2. PROJECT OBJECTIVES

### 2.1 Primary Objectives

| #   | Objective                       | Description                                                                        |
| --- | ------------------------------- | ---------------------------------------------------------------------------------- |
| 1   | **Predict Energy Consumption**  | Develop ML models to forecast building-level energy usage (next-hour and next-day) |
| 2   | **Identify Anomalies**          | Detect abnormal consumption patterns and peak usage events                         |
| 3   | **Automate Recommendations**    | Generate real-time energy-saving suggestions based on predictions                  |
| 4   | **Visualize Insights**          | Create interactive dashboards for monitoring and decision-making                   |
| 5   | **Enable Proactive Management** | Shift from reactive alerts to proactive prevention                                 |

### 2.2 Secondary Objectives

| #   | Objective                    | Description                                            |
| --- | ---------------------------- | ------------------------------------------------------ |
| 6   | **Sustainability Reporting** | Track carbon footprint reduction and SDG contributions |
| 7   | **Cost Optimization**        | Identify and reduce unnecessary operational expenses   |
| 8   | **Behavioral Change**        | Promote energy-conscious culture across campus         |
| 9   | **Scalability Planning**     | Design system extensible to other sustainability areas |

### 2.3 Key Performance Indicators (KPIs)

| KPI                 | Target                   | Measurement                               |
| ------------------- | ------------------------ | ----------------------------------------- |
| Energy Reduction    | 25-30% decrease          | kWh comparison (baseline vs. implemented) |
| Prediction Accuracy | >85%                     | R² Score, MAE                             |
| Anomaly Detection   | <5% false positives      | Precision-Recall metrics                  |
| User Adoption       | 80% admin usage          | Dashboard login analytics                 |
| Cost Savings        | £40,000-£60,000 annually | Energy bill comparison                    |

---

## 3. DELIVERABLES

### 3.1 Technical Deliverables

| Phase | Deliverable      | Description                        | Format             |
| ----- | ---------------- | ---------------------------------- | ------------------ |
| 1     | Project Plan     | Scope, objectives, roadmap         | PDF                |
| 2     | Dataset          | Raw, cleaned, data dictionary      | CSV, Excel         |
| 3     | Python Code      | Complete working project           | .py files          |
| 4     | EDA Report       | Visualizations and insights        | PDF, HTML          |
| 5     | Dashboard        | Professional interactive dashboard | Power BI, Python   |
| 6     | Diagrams         | Architecture, workflow, UML        | Editable (draw.io) |
| 7     | ML Model         | Trained model with evaluation      | .pkl, Jupyter      |
| 8     | Documentation    | Complete project report            | PDF, DOCX          |
| 9     | PowerPoint       | Professional presentation          | PPTX               |
| 10    | Final Submission | Complete ZIP package               | ZIP                |

### 3.2 Data Deliverables

| Deliverable       | Description                                   |
| ----------------- | --------------------------------------------- |
| Raw Dataset       | Original 50 observations                      |
| Cleaned Dataset   | Missing values handled, consistent formatting |
| Data Dictionary   | Column descriptions, units, data types        |
| Indicator Mapping | Sustainability indicators for measurement     |

### 3.3 Model Deliverables

| Deliverable         | Description                                 |
| ------------------- | ------------------------------------------- |
| Regression Model    | Trained on 50-records, predicts consumption |
| Model Evaluation    | Accuracy metrics, visualizations            |
| Feature Engineering | Lag values, rolling averages, hour-of-day   |

### 3.4 Documentation Deliverables

| Deliverable        | Description                        |
| ------------------ | ---------------------------------- |
| README.md          | Project overview, setup guide      |
| User Manual        | How to use the system              |
| Developer Guide    | Code structure, modification guide |
| Installation Guide | Step-by-step setup instructions    |

---

## 4. FOLDER STRUCTURE

```
SmartCampusEnergyManagement/
│
├── data/
│   ├── raw/
│   │   └── electricity_data_raw.csv
│   ├── processed/
│   │   ├── electricity_data_cleaned.csv
│   │   └── electricity_data_cleaned.xlsx
│   └── data_dictionary.md
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   ├── dashboard.py
│   └── main.py
│
├── models/
│   ├── energy_prediction_model.pkl
│   ├── scaler.pkl
│   └── model_metrics.json
│
├── outputs/
│   ├── visualizations/
│   │   ├── trend_analysis.png
│   │   ├── building_comparison.png
│   │   ├── correlation_heatmap.png
│   │   └── prediction_results.png
│   ├── dashboard/
│   │   └── energy_dashboard.png
│   └── reports/
│       └── eda_report.html
│
├── docs/
│   ├── Project_Documentation.pdf
│   ├── User_Manual.pdf
│   └── Developer_Guide.pdf
│
├── presentations/
│   └── Smart_Campus_Energy_Management.pptx
│
├── requirements.txt
├── README.md
├── setup.py
└── .gitignore
```

---

## 5. DEVELOPMENT ROADMAP

### 5.1 Phase Timeline

| Phase        | Activity            | Duration | Status     |
| ------------ | ------------------- | -------- | ---------- |
| **Phase 1**  | Project Planning    | 2 days   | ✅ Current |
| **Phase 2**  | Dataset Preparation | 2 days   | 🔄 Next    |
| **Phase 3**  | Python Project      | 4 days   | ⏳ Planned |
| **Phase 4**  | EDA & Visualization | 3 days   | ⏳ Planned |
| **Phase 5**  | Dashboard Creation  | 3 days   | ⏳ Planned |
| **Phase 6**  | Diagrams            | 2 days   | ⏳ Planned |
| **Phase 7**  | Machine Learning    | 4 days   | ⏳ Planned |
| **Phase 8**  | Documentation       | 3 days   | ⏳ Planned |
| **Phase 9**  | PowerPoint          | 2 days   | ⏳ Planned |
| **Phase 10** | Final Submission    | 2 days   | ⏳ Planned |

### 5.2 Technology Stack

| Category            | Tools                 | Purpose                     |
| ------------------- | --------------------- | --------------------------- |
| **Programming**     | Python 3.9+           | Core development            |
| **ML Framework**    | Scikit-learn          | Regression model            |
| **Data Processing** | Pandas, NumPy         | Data cleaning, manipulation |
| **Visualization**   | Matplotlib, Seaborn   | Charts and plots            |
| **Dashboard**       | Plotly Dash, Power BI | Interactive dashboard       |
| **Notebook**        | Jupyter/Google Colab  | Development and testing     |
| **Version Control** | Git                   | Code management             |
| **Documentation**   | Markdown, LaTeX       | Reports and documents       |
| **Presentation**    | PowerPoint            | Final presentation          |

### 5.3 Critical Success Factors

1. **Data Quality**: Accurate historical consumption records
2. **Model Accuracy**: Reliable predictions (>85% accuracy)
3. **User-Friendly Interface**: Simple and intuitive dashboard
4. **Actionable Insights**: Clear recommendations for action
5. **Sustainability Alignment**: Contribution to SDG 7 (Clean Energy)

---

## 6. RISK ASSESSMENT

| Risk                  | Probability | Impact | Mitigation                      |
| --------------------- | ----------- | ------ | ------------------------------- |
| Data quality issues   | Medium      | High   | Multiple validation checks      |
| Model overfitting     | Medium      | Medium | Cross-validation, test split    |
| User adoption         | Low         | Medium | User-friendly design, training  |
| Technical integration | Low         | Low    | Modular design, documentation   |
| Scope creep           | Medium      | Medium | Clear deliverables, phase gates |

---

## 7. SUSTAINABILITY CONTRIBUTION

### 7.1 SDG Alignment

| SDG                                  | Contribution                                    |
| ------------------------------------ | ----------------------------------------------- |
| **SDG 7** (Clean Energy)             | Direct contribution through energy optimization |
| **SDG 11** (Sustainable Cities)      | Sustainable campus infrastructure               |
| **SDG 13** (Climate Action)          | Reduced carbon emissions                        |
| **SDG 12** (Responsible Consumption) | Efficient resource use                          |

### 7.2 Environmental Impact

- **Energy Reduction**: 25-30% reduction in consumption
- **Carbon Savings**: ~20-25% reduction in CO₂ emissions
- **Resource Optimization**: Efficient cooling and lighting
- **Waste Reduction**: Proactive management of wastage

### 7.3 Economic Impact

- **Cost Savings**: £40,000-£60,000 annually
- **ROI**: 18-24 months for hardware investment
- **Peak Load Shifting**: Reduced demand charges
- **Predictive Maintenance**: Lower equipment replacement costs

---

## 8. NEXT STEPS

### Phase 2: Dataset Preparation

**What will be covered:**

- Raw dataset with 50 records
- Data cleaning and validation
- Missing value handling
- Outlier detection
- Excel and CSV generation
- Data dictionary creation

**Expected Output:**

- `electricity_data_raw.csv`
- `electricity_data_cleaned.csv`
- `electricity_data_cleaned.xlsx`
- `data_dictionary.md`
- Data cleaning report

---

## 9. APPROVAL & CONFIRMATION

**Project Owner:** Prince Timbadiya  
**Date:** June 2026  
**Status:** Phase 1 Complete - Ready for Review

---

## 10. SETUP

### `setup.py`

```python
"""
Setup file for Smart Campus Energy Management System
"""

from setuptools import setup, find_packages

setup(
    name='smart-campus-energy-management',
    version='1.0.0',
    author='Prince Timbadiya',
    author_email='prince@example.com',
    description='AI-Enabled Energy Management for Sustainable Campus',
    packages=find_packages(),
    install_requires=[
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'scikit-learn>=1.3.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'joblib>=1.3.0',
        'openpyxl>=3.1.0'
    ],
    python_requires='>=3.9',
)
```
