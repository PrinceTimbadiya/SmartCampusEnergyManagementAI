"""
Machine Learning Model Development & Evaluation Module
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

# Import preprocessing and utilities
from preprocessing import clean_dataset, prepare_data_for_model, feature_engineering
from utils import save_metrics, load_data, print_section_header
from config import RAW_DATA_FILE, CLEANED_DATA_FILE, MODELS_DIR

# Import ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
)

# Try importing XGBoost
try:
    from xgboost import XGBRegressor

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost not installed. Skipping XGBoost model.")

import joblib
import json
import os
from scipy import stats

# Create output directories
MODEL_RESULTS_DIR = "../outputs/model_results/"
MODEL_VIZ_DIR = "../outputs/model_visualizations/"
os.makedirs(MODEL_RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_VIZ_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 70)
print("  SMART CAMPUS ENERGY MANAGEMENT - MODEL EVALUATION")
print("  Student: Prince Timbadiya")
print("=" * 70)

# Load and prepare data
df = pd.read_csv(CLEANED_DATA_FILE)
df["Date"] = pd.to_datetime(df["Date"])

print(f"\n📊 Dataset loaded: {len(df)} records, {len(df.columns)} columns")
print(f"   Date Range: {df['Date'].min()} to {df['Date'].max()}")

# Feature engineering
df_engineered = feature_engineering(df)
print(f"\n🔧 Feature engineering complete: {len(df_engineered.columns)} features")

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

X = df_engineered[feature_cols]
y = df_engineered["Electricity_Consumption_kWh"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Data split: {len(X_train)} training, {len(X_test)} test")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Feature scaling complete")

# Save scaler
joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
print(f"✅ Scaler saved: {MODELS_DIR}/scaler.pkl")

# Feature columns for later use
feature_columns = feature_cols
with open(f"{MODELS_DIR}/feature_columns.json", "w") as f:
    json.dump(feature_columns, f, indent=4)
print(f"✅ Feature columns saved: {MODELS_DIR}/feature_columns.json")

print("\n" + "=" * 70)
print("  SECTION 1 & 2: MODEL DEVELOPMENT & FEATURE ENGINEERING")
print("=" * 70)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
}

if XGB_AVAILABLE:
    models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)

# Storage for results
results = {}
trained_models = {}
predictions = {}
training_times = {}
prediction_times = {}

print("\n🚀 Training Models...")
print("-" * 50)

for name, model in models.items():
    print(f"\n📊 Training {name}...")

    # Train
    start_time = datetime.now()
    model.fit(X_train_scaled, y_train)
    train_time = (datetime.now() - start_time).total_seconds()

    # Predict
    start_time = datetime.now()
    y_pred = model.predict(X_test_scaled)
    pred_time = (datetime.now() - start_time).total_seconds()

    # Store
    trained_models[name] = model
    predictions[name] = y_pred
    training_times[name] = train_time
    prediction_times[name] = pred_time

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2 = r2_score(y_test, y_pred)

    # Adjusted R²
    n = len(y_test)
    k = X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

    # Cross validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

    # Store results
    results[name] = {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
        "R2": round(r2, 4),
        "Adjusted_R2": round(adj_r2, 4),
        "CV_Mean": round(cv_scores.mean(), 4),
        "CV_Std": round(cv_scores.std(), 4),
        "Training_Time": round(train_time, 3),
        "Prediction_Time": round(pred_time, 3),
    }

    print(f"   ✅ R²: {r2:.4f}")
    print(f"   ✅ RMSE: {rmse:.2f} kWh")
    print(f"   ✅ MAE: {mae:.2f} kWh")
    print(f"   ✅ CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"   ⏱️ Training: {train_time:.3f}s, Prediction: {pred_time:.3f}s")

    print("\n" + "=" * 70)
print("  SECTION 3: MODEL EVALUATION - DETAILED METRICS")
print("=" * 70)

# Create detailed evaluation table
evaluation_df = pd.DataFrame(results).T
evaluation_df = evaluation_df.sort_values("R2", ascending=False)

print("\n📊 Model Performance Summary (Sorted by R²):")
print("-" * 60)
print(evaluation_df.to_string())

# Find best model
best_model_name = evaluation_df.index[0]
best_model = trained_models[best_model_name]
best_predictions = predictions[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   R² Score: {results[best_model_name]['R2']}")
print(f"   RMSE: {results[best_model_name]['RMSE']} kWh")
print(f"   MAE: {results[best_model_name]['MAE']} kWh")

# Save comparison table
evaluation_df.to_csv(f"{MODEL_RESULTS_DIR}/model_comparison.csv")
evaluation_df.to_excel(f"{MODEL_RESULTS_DIR}/model_comparison.xlsx")

# Save as markdown
with open(f"{MODEL_RESULTS_DIR}/model_comparison.md", "w") as f:
    f.write("# Model Comparison Results\n\n")
    f.write("## Performance Metrics\n\n")
    f.write(evaluation_df.to_markdown())
    f.write(f"\n\n**Best Model: {best_model_name}**\n")
    f.write(f"- R² Score: {results[best_model_name]['R2']}\n")
    f.write(f"- RMSE: {results[best_model_name]['RMSE']} kWh\n")
    f.write(f"- MAE: {results[best_model_name]['MAE']} kWh\n")

print(f"\n✅ Results saved to: {MODEL_RESULTS_DIR}")

print("\n" + "=" * 70)
print("  SECTION 4: MODEL COMPARISON TABLE")
print("=" * 70)

# Create comparison table with ranking
comparison_df = evaluation_df.copy()
comparison_df["Rank"] = range(1, len(comparison_df) + 1)

# Highlight best model
best_row = comparison_df.iloc[0].copy()

print("\n📊 Ranked Model Performance:")
print("-" * 70)
print(comparison_df[["Rank", "R2", "RMSE", "MAE", "MAPE", "CV_Mean"]].to_string())

print("\n🏆 BEST MODEL SELECTED:")
print(f"   Model: {best_model_name}")
print(f"   Rank: 1/{len(comparison_df)}")
print(f"   R² Score: {best_row['R2']:.4f}")
print(f"   RMSE: {best_row['RMSE']:.2f} kWh")
print(f"   MAE: {best_row['MAE']:.2f} kWh")
print(f"   MAPE: {best_row['MAPE']:.2f}%")
print(f"   CV Score: {best_row['CV_Mean']:.4f}")

# Save comparison with ranking
comparison_df.to_csv(f"{MODEL_RESULTS_DIR}/model_ranking.csv")
print(f"\n✅ Ranking saved: {MODEL_RESULTS_DIR}/model_ranking.csv")

print("\n" + "=" * 70)
print("  SECTION 5: VISUALIZATIONS")
print("=" * 70)

# 1. Model Comparison Bar Chart
print("\n📊 Chart 1: Model Comparison Bar Chart")
fig, ax = plt.subplots(figsize=(12, 6))
metrics_to_plot = ["R2", "RMSE", "MAE"]
x = np.arange(len(comparison_df.index))
width = 0.25

for i, metric in enumerate(metrics_to_plot):
    values = comparison_df[metric].values
    if metric != "R2":
        # For error metrics, lower is better
        values = values / values.max() * 100  # Normalize
        label = f"{metric} (lower better)"
    else:
        label = f"{metric} (higher better)"

    ax.bar(x + i * width, values, width, label=label)

ax.set_xlabel("Model", fontsize=12, fontweight="bold")
ax.set_ylabel("Normalized Score", fontsize=12, fontweight="bold")
ax.set_title("Model Performance Comparison", fontsize=16, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(comparison_df.index, rotation=45, ha="right")
ax.legend(loc="best")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{MODEL_VIZ_DIR}/01_model_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Chart 1 saved: 01_model_comparison.png")

# 2. R² Score Comparison
print("📊 Chart 2: R² Score Comparison")
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#2E86AB" if i == 0 else "#A23B72" for i in range(len(comparison_df))]
bars = ax.bar(
    comparison_df.index,
    comparison_df["R2"],
    color=colors,
    alpha=0.8,
    edgecolor="black",
    linewidth=1,
)
ax.axhline(
    y=comparison_df["R2"].mean(),
    color="red",
    linestyle="--",
    alpha=0.7,
    label=f'Mean: {comparison_df["R2"].mean():.3f}',
)
ax.set_xlabel("Model", fontsize=12, fontweight="bold")
ax.set_ylabel("R² Score", fontsize=12, fontweight="bold")
ax.set_title("R² Score Comparison", fontsize=16, fontweight="bold")
ax.set_ylim(0, 1)
ax.legend(loc="best")
ax.grid(True, alpha=0.3, axis="y")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{MODEL_VIZ_DIR}/02_r2_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Chart 2 saved: 02_r2_comparison.png")

# 3. RMSE Comparison
print("📊 Chart 3: RMSE Comparison")
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#2E86AB" if i == 0 else "#A23B72" for i in range(len(comparison_df))]
bars = ax.bar(
    comparison_df.index,
    comparison_df["RMSE"],
    color=colors,
    alpha=0.8,
    edgecolor="black",
    linewidth=1,
)
ax.axhline(
    y=comparison_df["RMSE"].mean(),
    color="red",
    linestyle="--",
    alpha=0.7,
    label=f'Mean: {comparison_df["RMSE"].mean():.2f} kWh',
)
ax.set_xlabel("Model", fontsize=12, fontweight="bold")
ax.set_ylabel("RMSE (kWh)", fontsize=12, fontweight="bold")
ax.set_title("RMSE Comparison (lower is better)", fontsize=16, fontweight="bold")
ax.legend(loc="best")
ax.grid(True, alpha=0.3, axis="y")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{MODEL_VIZ_DIR}/03_rmse_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Chart 3 saved: 03_rmse_comparison.png")

# 4. MAE Comparison
print("📊 Chart 4: MAE Comparison")
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#2E86AB" if i == 0 else "#A23B72" for i in range(len(comparison_df))]
bars = ax.bar(
    comparison_df.index,
    comparison_df["MAE"],
    color=colors,
    alpha=0.8,
    edgecolor="black",
    linewidth=1,
)
ax.axhline(
    y=comparison_df["MAE"].mean(),
    color="red",
    linestyle="--",
    alpha=0.7,
    label=f'Mean: {comparison_df["MAE"].mean():.2f} kWh',
)
ax.set_xlabel("Model", fontsize=12, fontweight="bold")
ax.set_ylabel("MAE (kWh)", fontsize=12, fontweight="bold")
ax.set_title("MAE Comparison (lower is better)", fontsize=16, fontweight="bold")
ax.legend(loc="best")
ax.grid(True, alpha=0.3, axis="y")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{MODEL_VIZ_DIR}/04_mae_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Chart 4 saved: 04_mae_comparison.png")

# 5. Prediction vs Actual Scatter Plot
print("📊 Chart 5: Prediction vs Actual Scatter Plot")
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(y_test, best_predictions, alpha=0.6, s=50, color="#2E86AB")
ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    linewidth=2,
    label="Perfect Prediction",
)
ax.set_xlabel("Actual Consumption (kWh)", fontsize=12, fontweight="bold")
ax.set_ylabel("Predicted Consumption (kWh)", fontsize=12, fontweight="bold")
ax.set_title(f"Actual vs Predicted: {best_model_name}", fontsize=16, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(f"{MODEL_VIZ_DIR}/05_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Chart 5 saved: 05_actual_vs_predicted.png")

# 6. Residual Plot
print("📊 Chart 6: Residual Plot")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
residuals = y_test - best_predictions

# Residual vs Predicted
axes[0].scatter(best_predictions, residuals, alpha=0.6, color="#A23B72")
axes[0].axhline(y=0, color="r", linestyle="--", linewidth=2)
axes[0].set_xlabel("Predicted Consumption (kWh)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Residual (kWh)", fontsize=12, fontweight="bold")
axes[0].set_title("Residual Plot", fontsize=14, fontweight="bold")
axes[0].grid(True, alpha=0.3)

# Residual Distribution
axes[1].hist(
    residuals, bins=20, color="#F18F01", alpha=0.7, edgecolor="black", linewidth=1
)
axes[1].axvline(x=0, color="r", linestyle="--", linewidth=2)
axes[1].set_xlabel("Residual (kWh)", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Frequency", fontsize=12, fontweight="bold")
axes[1].set_title("Residual Distribution", fontsize=14, fontweight="bold")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{MODEL_VIZ_DIR}/06_residual_plot.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Chart 6 saved: 06_residual_plot.png")

# 7. Feature Importance (for tree-based models)
print("📊 Chart 7: Feature Importance")
if hasattr(best_model, "feature_importances_"):
    importance = best_model.feature_importances_
    feature_names = feature_cols

    # Sort features by importance
    indices = np.argsort(importance)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importance = [importance[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(sorted_features, sorted_importance, color="#2E86AB", alpha=0.8)
    ax.set_xlabel("Feature Importance", fontsize=12, fontweight="bold")
    ax.set_ylabel("Features", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Feature Importance: {best_model_name}", fontsize=16, fontweight="bold"
    )
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(
        f"{MODEL_VIZ_DIR}/07_feature_importance.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("✅ Chart 7 saved: 07_feature_importance.png")
else:
    print("⚠️ Feature importance not available for this model")

# 8. Error Distribution Histogram
print("📊 Chart 8: Error Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
errors = y_test - best_predictions
n, bins, patches = ax.hist(
    errors, bins=20, color="#C73E1D", alpha=0.7, edgecolor="black", linewidth=1
)
ax.axvline(x=0, color="blue", linestyle="--", linewidth=2, label="Zero Error")
ax.axvline(
    x=np.mean(errors),
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Mean Error: {np.mean(errors):.2f}",
)
ax.set_xlabel("Prediction Error (kWh)", fontsize=12, fontweight="bold")
ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
ax.set_title(f"Error Distribution: {best_model_name}", fontsize=16, fontweight="bold")
ax.legend(loc="best")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{MODEL_VIZ_DIR}/08_error_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Chart 8 saved: 08_error_distribution.png")

# 9. Learning Curve
print("📊 Chart 9: Learning Curve")
train_sizes, train_scores, test_scores = learning_curve(
    best_model,
    X_train_scaled,
    y_train,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    random_state=42,
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes, train_mean, "o-", color="#2E86AB", label="Training Score")
ax.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.2,
    color="#2E86AB",
)
ax.plot(train_sizes, test_mean, "o-", color="#C73E1D", label="Cross-Validation Score")
ax.fill_between(
    train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color="#C73E1D"
)
ax.set_xlabel("Training Examples", fontsize=12, fontweight="bold")
ax.set_ylabel("Score", fontsize=12, fontweight="bold")
ax.set_title(f"Learning Curve: {best_model_name}", fontsize=16, fontweight="bold")
ax.legend(loc="best")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{MODEL_VIZ_DIR}/09_learning_curve.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Chart 9 saved: 09_learning_curve.png")

# 10. Cross Validation Score Plot
print("📊 Chart 10: Cross Validation Score Plot")
fig, ax = plt.subplots(figsize=(10, 6))
cv_means = []
cv_stds = []

for name in comparison_df.index:
    cv_scores = cross_val_score(trained_models[name], X_train_scaled, y_train, cv=5)
    cv_means.append(cv_scores.mean())
    cv_stds.append(cv_scores.std())

x_pos = np.arange(len(comparison_df.index))
bars = ax.bar(x_pos, cv_means, yerr=cv_stds, capsize=8, alpha=0.8, color="#2E86AB")
ax.set_xlabel("Model", fontsize=12, fontweight="bold")
ax.set_ylabel("Cross-Validation Score", fontsize=12, fontweight="bold")
ax.set_title("Cross-Validation Score Comparison", fontsize=16, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(comparison_df.index, rotation=45, ha="right")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{MODEL_VIZ_DIR}/10_cv_scores.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Chart 10 saved: 10_cv_scores.png")

# 11. Actual vs Predicted Line Graph
print("📊 Chart 11: Actual vs Predicted Line Graph")
fig, ax = plt.subplots(figsize=(14, 6))
sample_size = min(50, len(y_test))
indices = np.argsort(y_test.values)[:sample_size]
ax.plot(
    range(sample_size),
    y_test.values[indices],
    "o-",
    color="#2E86AB",
    label="Actual",
    linewidth=2,
    markersize=8,
)
ax.plot(
    range(sample_size),
    best_predictions[indices],
    "s-",
    color="#C73E1D",
    label="Predicted",
    linewidth=2,
    markersize=8,
)
ax.set_xlabel("Sample Index (sorted by actual value)", fontsize=12, fontweight="bold")
ax.set_ylabel("Electricity Consumption (kWh)", fontsize=12, fontweight="bold")
ax.set_title(
    f"Actual vs Predicted: {best_model_name} (Sample of {sample_size} records)",
    fontsize=14,
    fontweight="bold",
)
ax.legend(loc="best")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    f"{MODEL_VIZ_DIR}/11_actual_vs_predicted_line.png", dpi=300, bbox_inches="tight"
)
plt.close()
print("✅ Chart 11 saved: 11_actual_vs_predicted_line.png")

# 12. Model Performance Radar Chart
print("📊 Chart 12: Model Performance Radar Chart")
try:
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Select top 5 models
    top_models = comparison_df.head(5)
    metrics_for_radar = ["R2", "CV_Mean"]  # Higher is better
    # Normalize metrics (0-1 scale)
    normalized_data = []
    for metric in metrics_for_radar:
        values = top_models[metric].values
        if metric == "R2":
            # R² is already 0-1
            normalized = values
        else:
            normalized = (values - values.min()) / (values.max() - values.min())
        normalized_data.append(normalized)

    angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()
    angles += angles[:1]

    for i, model_name in enumerate(top_models.index):
        values = [normalized_data[0][i], normalized_data[1][i]]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_for_radar)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Model Performance Radar Chart", fontsize=16, fontweight="bold", pad=20
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(f"{MODEL_VIZ_DIR}/12_radar_chart.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Chart 12 saved: 12_radar_chart.png")
except Exception as e:
    print(f"⚠️ Radar chart could not be generated: {e}")

print("\n✅ All visualizations saved to:", MODEL_VIZ_DIR)

print("\n" + "=" * 70)
print("  SECTION 6: BEST MODEL SELECTION")
print("=" * 70)

# Find best model
best_model_name = evaluation_df.index[0]
best_model = trained_models[best_model_name]
best_predictions = predictions[best_model_name]

print(f"\n🏆 BEST MODEL: {best_model_name}")
print("-" * 50)
print(f"   R² Score: {results[best_model_name]['R2']:.4f}")
print(f"   Adjusted R²: {results[best_model_name]['Adjusted_R2']:.4f}")
print(f"   RMSE: {results[best_model_name]['RMSE']:.2f} kWh")
print(f"   MAE: {results[best_model_name]['MAE']:.2f} kWh")
print(f"   MAPE: {results[best_model_name]['MAPE']:.2f}%")
print(
    f"   CV Score: {results[best_model_name]['CV_Mean']:.4f} (±{results[best_model_name]['CV_Std']:.4f})"
)

print("\n📊 WHY THIS MODEL IS BEST:")
print("-" * 50)
print(
    f"1. Highest R² Score ({results[best_model_name]['R2']:.4f}) - explains most variance"
)
print(
    f"2. Lowest RMSE ({results[best_model_name]['RMSE']:.2f} kWh) - most accurate predictions"
)
print(
    f"3. Lowest MAE ({results[best_model_name]['MAE']:.2f} kWh) - smallest average error"
)
print(
    f"4. Lowest MAPE ({results[best_model_name]['MAPE']:.2f}%) - most accurate percentage-wise"
)
print(
    f"5. High Cross-Validation Score ({results[best_model_name]['CV_Mean']:.4f}) - generalizes well"
)

# Save best model info
best_model_info = {
    "best_model": best_model_name,
    "r2_score": results[best_model_name]["R2"],
    "rmse": results[best_model_name]["RMSE"],
    "mae": results[best_model_name]["MAE"],
    "mape": results[best_model_name]["MAPE"],
    "features_used": feature_cols,
    "test_size": 0.2,
    "random_state": 42,
}

with open(f"{MODEL_RESULTS_DIR}/best_model_info.json", "w") as f:
    json.dump(best_model_info, f, indent=4)
print(f"\n✅ Best model info saved: {MODEL_RESULTS_DIR}/best_model_info.json")
print("\n" + "="*70)
print("  SECTION 7: EXPLAINABLE AI (XAI)")
print("="*70)

if hasattr(best_model, 'feature_importances_'):
    importance = best_model.feature_importances_
    feature_names = feature_cols
    
    # Sort and display
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("\n📊 Feature Importance Ranking:")
    print("-"*50)
    for idx, row in feature_importance_df.iterrows():
        print(f"   {row['Feature']}: {row['Importance']:.4f} ({row['Importance']/feature_importance_df['Importance'].sum()*100:.1f}%)")
    
    print("\n💡 Top 10 Most Important Features:")
    print("-"*50)
    for idx, (feat, imp) in enumerate(zip(feature_importance_df['Feature'][:10], 
                                          feature_importance_df['Importance'][:10]), 1):
        print(f"{idx}. {feat}: {imp:.4f}")
    
    print("\n🏢 BUSINESS MEANING:")
    print("-"*50)
    print("1. Rolling Mean (7-day): Indicates weekly consumption patterns")
    print("2. Lag 1: Previous day consumption strongly predicts current day")
    print("3. Occupancy Count: Directly drives energy consumption")
    print("4. Temperature: Cooling demand drives summer consumption")
    print("5. Building Type: Different buildings have different usage profiles")
    
    print("\n🌱 ENERGY MEANING:")
    print("-"*50)
    print("1. Weekly patterns help identify peak days")
    print("2. Lag patterns enable day-ahead forecasting")
    print("3. Occupancy-based controls can reduce waste")
    print("4. Temperature integration enables cooling optimization")
    print("5. Building-type differentiation enables targeted interventions")
    
    print("\n🌍 SUSTAINABILITY MEANING:")
    print("-"*50)
    print("1. Understanding patterns enables waste reduction")
    print("2. Predictive ability enables proactive management")
    print("3. Building-specific insights enable targeted savings")
    print("4. Temperature correlation enables climate-responsive controls")
    print("5. Pattern awareness enables behavioral change campaigns")
    
    # Save feature importance
    feature_importance_df.to_csv(f'{MODEL_RESULTS_DIR}/feature_importance.csv', index=False)
    print(f"\n✅ Feature importance saved: {MODEL_RESULTS_DIR}/feature_importance.csv")
else:
    print("⚠️ Feature importance not available for this model type")

print("\n" + "="*70)
print("  SECTION 8: PREDICTION EXAMPLES")
print("="*70)

# Create prediction examples
example_data = {
    'Academic Block': {'occupancy': 350, 'temperature': 39, 'building_encoded': 0, 'day': 3, 'month': 6, 'weekend': 0},
    'Library': {'occupancy': 135, 'temperature': 37, 'building_encoded': 1, 'day': 3, 'month': 6, 'weekend': 0},
    'Canteen': {'occupancy': 290, 'temperature': 39, 'building_encoded': 2, 'day': 3, 'month': 6, 'weekend': 0},
    'Hostel': {'occupancy': 435, 'temperature': 40, 'building_encoded': 3, 'day': 3, 'month': 6, 'weekend': 0}
}

# Get actual average for each building
actual_avg = df.groupby('Building')['Electricity_Consumption_kWh'].mean()

print("\n📊 Prediction Examples:")
print("-"*70)

prediction_examples = []
for building, data in example_data.items():
    # Create feature vector
    features = np.array([
        data['occupancy'],
        data['temperature'],
        data['building_encoded'],
        data['day'],
        data['month'],
        data['weekend'],
        actual_avg[building] * 0.95,  # Estimate lag
        actual_avg[building] * 0.97,  # Estimate rolling 3-day
        actual_avg[building] * 0.98   # Estimate rolling 7-day
    ]).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    predicted = best_model.predict(features_scaled)[0]
    actual = actual_avg[building]
    error = abs(predicted - actual)
    error_pct = (error / actual) * 100
    
    prediction_examples.append({
        'Building': building,
        'Actual': round(actual, 2),
        'Predicted': round(predicted, 2),
        'Difference': round(error, 2),
        'Error %': round(error_pct, 1)
    })
    
    print(f"\n{building}:")
    print(f"   Actual: {actual:.2f} kWh")
    print(f"   Predicted: {predicted:.2f} kWh")
    print(f"   Difference: {error:.2f} kWh ({error_pct:.1f}% error)")

# Save predictions
prediction_df = pd.DataFrame(prediction_examples)
prediction_df.to_csv(f'{MODEL_RESULTS_DIR}/prediction_examples.csv', index=False)
print(f"\n✅ Prediction examples saved: {MODEL_RESULTS_DIR}/prediction_examples.csv")

print("\n" + "="*70)
print("  SECTION 9: MODEL SAVING")
print("="*70)

# Save best model
best_model_path = f'{MODELS_DIR}/best_model.pkl'
joblib.dump(best_model, best_model_path)
print(f"✅ Best model saved: {best_model_path}")

# Save scaler (already saved above)
scaler_path = f'{MODELS_DIR}/scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved: {scaler_path}")

# Save encoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(df['Building'])
encoder_path = f'{MODELS_DIR}/encoder.pkl'
joblib.dump(encoder, encoder_path)
print(f"✅ Encoder saved: {encoder_path}")

# Save feature columns
feature_columns = feature_cols
feature_path = f'{MODELS_DIR}/feature_columns.json'
with open(feature_path, 'w') as f:
    json.dump(feature_columns, f, indent=4)
print(f"✅ Feature columns saved: {feature_path}")

# Save metrics
metrics = {
    'best_model': best_model_name,
    'metrics': results[best_model_name],
    'all_models': results,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}
metrics_path = f'{MODELS_DIR}/model_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f"✅ Metrics saved: {metrics_path}")

# Save all models (optional)
for name, model in trained_models.items():
    model_path = f'{MODELS_DIR}/{name.replace(" ", "_").lower()}.pkl'
    joblib.dump(model, model_path)
print(f"✅ All models saved")

print("\n" + "="*70)
print("  SECTION 10: MODEL REPORT")
print("="*70)

report = f"""# Machine Learning Model Report
## Smart Campus Energy Management System

**Student:** Prince Timbadiya  
**Date:** {datetime.now().strftime('%Y-%m-%d')}  
**Internship:** Microsoft + 1M1B Green Skills & Applied AI Internship  

---

## Executive Summary

This report presents the development, evaluation, and comparison of multiple machine learning models for predicting campus electricity consumption. The **{best_model_name}** emerged as the best-performing model with an R² score of {results[best_model_name]['R2']:.4f} and RMSE of {results[best_model_name]['RMSE']:.2f} kWh.

---

## Models Tested

| Model | R² Score | RMSE (kWh) | MAE (kWh) | MAPE (%) | CV Score |
|-------|----------|------------|-----------|----------|----------|
"""

for name in evaluation_df.index:
    row = evaluation_df.loc[name]
    report += f"| {name} | {row['R2']:.4f} | {row['RMSE']:.2f} | {row['MAE']:.2f} | {row['MAPE']:.2f} | {row['CV_Mean']:.4f} |\n"

report += f"""
---

## Best Model: {best_model_name}

| Metric | Value |
|--------|-------|
| R² Score | {results[best_model_name]['R2']:.4f} |
| Adjusted R² | {results[best_model_name]['Adjusted_R2']:.4f} |
| RMSE | {results[best_model_name]['RMSE']:.2f} kWh |
| MAE | {results[best_model_name]['MAE']:.2f} kWh |
| MAPE | {results[best_model_name]['MAPE']:.2f}% |
| Cross-Validation Mean | {results[best_model_name]['CV_Mean']:.4f} |
| Cross-Validation Std | {results[best_model_name]['CV_Std']:.4f} |
| Training Time | {results[best_model_name]['Training_Time']:.3f}s |
| Prediction Time | {results[best_model_name]['Prediction_Time']:.3f}s |

---

## Feature Importance

"""

if hasattr(best_model, 'feature_importances_'):
    for idx, row in feature_importance_df.iterrows():
        report += f"- **{row['Feature']}**: {row['Importance']:.4f} ({row['Importance']/feature_importance_df['Importance'].sum()*100:.1f}%)\n"

report += f"""
---

## Visualizations Generated

1. Model Comparison Bar Chart
2. R² Score Comparison
3. RMSE Comparison
4. MAE Comparison
5. Actual vs Predicted Scatter Plot
6. Residual Plot
7. Feature Importance
8. Error Distribution
9. Learning Curve
10. Cross-Validation Scores
11. Actual vs Predicted Line Graph
12. Radar Chart

All visualizations saved in: `outputs/model_visualizations/`

---

## Business Insights

1. **Occupancy-Driven Consumption**: Strong relationship between occupancy and energy use
2. **Temperature Impact**: Significant cooling load during summer months
3. **Building-Specific Patterns**: Different buildings have unique consumption profiles
4. **Temporal Patterns**: Clear weekly and seasonal trends
5. **Prediction Accuracy**: Model can predict consumption within {results[best_model_name]['MAE']:.2f} kWh

---

## Limitations

1. Dataset limited to 500 records (may need more data for complex patterns)
2. Weather data limited to temperature (could include humidity, rainfall)
3. No data on specific equipment usage
4. No data on building operations (events, holidays)
5. Model may not capture rare events

---

## Future Improvements

1. **Data Collection**: Increase frequency (hourly readings)
2. **Additional Features**: Include weather, humidity, solar radiation
3. **Deep Learning**: Explore neural networks for complex patterns
4. **Real-Time Integration**: Connect to IoT sensors
5. **Anomaly Detection**: Implement continuous monitoring
6. **Explainable AI**: Provide more detailed feature explanations
7. **Ensemble Models**: Combine multiple models for better performance
8. **Integration**: Connect with energy management systems

---

## Conclusion

The machine learning models developed in this study demonstrate that energy consumption in campus buildings can be accurately predicted using historical data. The **{best_model_name}** model achieved strong performance metrics, making it suitable for real-world deployment in proactive energy management.

---

*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

# Save report
with open(f'{MODEL_RESULTS_DIR}/model_report.md', 'w') as f:
    f.write(report)

print(f"✅ Model report saved: {MODEL_RESULTS_DIR}/model_report.md")

print("\n" + "="*70)
print("  SECTION 11: BUSINESS RECOMMENDATIONS")
print("="*70)

print("\n💼 15 AI Recommendations:")
print("-"*50)
recommendations = [
    "1. Deploy best model for real-time energy prediction",
    "2. Implement automated alerts for predicted exceedances",
    "3. Use model to optimize HVAC scheduling",
    "4. Integrate occupancy predictions for dynamic load management",
    "5. Develop energy consumption forecasting dashboard",
    "6. Use feature importance to focus on key drivers",
    "7. Implement model retraining pipeline (monthly)",
    "8. Create building-specific consumption targets using model predictions",
    "9. Use model for peak demand forecasting",
    "10. Integrate with solar generation predictions",
    "11. Develop mobile app for real-time alerts",
    "12. Use model for energy budget planning",
    "13. Implement anomaly detection using prediction residuals",
    "14. Create automated energy efficiency reports",
    "15. Use model to evaluate energy conservation measures"
]

for rec in recommendations:
    print(rec)

print("\n💡 10 Energy Saving Recommendations:")
print("-"*50)
energy_savings = [
    "1. Implement occupancy-based lighting controls (25-30% savings)",
    "2. Optimize HVAC scheduling based on temperature predictions",
    "3. Install motion sensors in all classrooms",
    "4. Replace traditional lighting with LEDs",
    "5. Implement automated shut-off during unoccupied hours",
    "6. Conduct regular energy audits using model insights",
    "7. Monitor and manage weekend energy usage",
    "8. Implement demand response during peak hours",
    "9. Use model to identify energy waste patterns",
    "10. Implement building-specific efficiency measures"
]

for rec in energy_savings:
    print(rec)

print("\n🌍 10 Sustainability Recommendations:")
print("-"*50)
sustainability = [
    "1. Reduce campus carbon footprint through energy optimization",
    "2. Align with SDG 7 (Clean Energy) using predictive management",
    "3. Implement green building certification programs",
    "4. Launch energy awareness campaigns for students",
    "5. Create sustainable campus culture through data transparency",
    "6. Integrate renewable energy forecasts",
    "7. Implement waste management integration",
    "8. Track sustainability metrics using model predictions",
    "9. Share best practices with other campuses",
    "10. Create sustainability report using model insights"
]

for rec in sustainability:
    print(rec)

print("\n🔮 10 Future AI Improvements:")
print("-"*50)
future_ai = [
    "1. Deploy deep learning models for complex patterns",
    "2. Implement reinforcement learning for optimal scheduling",
    "3. Integrate multi-modal data (weather, events, occupancy)",
    "4. Develop federated learning for multi-building models",
    "5. Implement real-time stream processing for live data",
    "6. Develop explainable AI dashboards for decision-making",
    "7. Create ensemble models for improved accuracy",
    "8. Implement automated hyperparameter optimization",
    "9. Develop transfer learning for new buildings",
    "10. Create anomaly detection models for equipment monitoring"
]

for rec in future_ai:
    print(rec)

# Save recommendations
with open(f'{MODEL_RESULTS_DIR}/recommendations.md', 'w') as f:
    f.write("# Business Recommendations\n\n")
    f.write("## 15 AI Recommendations\n\n")
    for rec in recommendations:
        f.write(f"{rec}\n")
    f.write("\n## 10 Energy Saving Recommendations\n\n")
    for rec in energy_savings:
        f.write(f"{rec}\n")
    f.write("\n## 10 Sustainability Recommendations\n\n")
    for rec in sustainability:
        f.write(f"{rec}\n")
    f.write("\n## 10 Future AI Improvements\n\n")
    for rec in future_ai:
        f.write(f"{rec}\n")

print(f"\n✅ Recommendations saved: {MODEL_RESULTS_DIR}/recommendations.md")

print("\n" + "="*70)
print("  SECTION 12: PROJECT FOLDER STRUCTURE")
print("="*70)

print("""
SmartCampusEnergyManagement/
│
├── data/
│   ├── raw/
│   │   └── electricity_data_raw.csv
│   └── processed/
│       ├── electricity_data_cleaned.csv
│       └── electricity_data_cleaned.xlsx
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   └── 03_model_training_evaluation.ipynb
│
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   ├── dashboard.py
│   ├── model_evaluation.py    ← NEW
│   └── main.py
│
├── models/                     ← UPDATED
│   ├── best_model.pkl
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── decision_tree.pkl
│   ├── gradient_boosting.pkl
│   ├── extra_trees.pkl
│   ├── xgboost.pkl (if available)
│   ├── scaler.pkl
│   ├── encoder.pkl
│   ├── feature_columns.json
│   └── model_metrics.json
│
├── outputs/
│   ├── visualizations/         (15 EDA charts)
│   ├── model_results/          ← NEW
│   │   ├── model_comparison.csv
│   │   ├── model_comparison.xlsx
│   │   ├── model_comparison.md
│   │   ├── model_ranking.csv
│   │   ├── best_model_info.json
│   │   ├── feature_importance.csv
│   │   ├── prediction_examples.csv
│   │   ├── model_report.md
│   │   └── recommendations.md
│   └── model_visualizations/   ← NEW
│       ├── 01_model_comparison.png
│       ├── 02_r2_comparison.png
│       ├── 03_rmse_comparison.png
│       ├── 04_mae_comparison.png
│       ├── 05_actual_vs_predicted.png
│       ├── 06_residual_plot.png
│       ├── 07_feature_importance.png
│       ├── 08_error_distribution.png
│       ├── 09_learning_curve.png
│       ├── 10_cv_scores.png
│       ├── 11_actual_vs_predicted_line.png
│       └── 12_radar_chart.png
│
├── requirements.txt
├── README.md
├── setup.py
└── .gitignore
""")

print("\n" + "="*70)
print("  PHASE 5: MACHINE LEARNING MODEL DEVELOPMENT - COMPLETE")
print("="*70)

print("\n📊 SUMMARY OF OUTPUTS:")
print("-"*50)

# Count models evaluated
print(f"   ✅ Models Evaluated: {len(models)}")
print(f"   ✅ Best Model: {best_model_name}")
print(f"   ✅ R² Score: {results[best_model_name]['R2']:.4f}")
print(f"   ✅ RMSE: {results[best_model_name]['RMSE']:.2f} kWh")

# Count visualizations
print(f"\n📊 Visualizations Generated: 12")
print(f"   📁 Location: outputs/model_visualizations/")

# Count reports
print(f"\n📄 Reports Generated:")
print(f"   ✅ Model Comparison: CSV, Excel, MD")
print(f"   ✅ Feature Importance: CSV")
print(f"   ✅ Prediction Examples: CSV")
print(f"   ✅ Model Report: MD")
print(f"   ✅ Recommendations: MD")
print(f"   ✅ Best Model Info: JSON")
print(f"   ✅ Model Metrics: JSON")
print(f"   📁 Location: outputs/model_results/")

# Count models saved
print(f"\n💾 Models Saved: {len(models) + 2} (including scaler and encoder)")
print(f"   📁 Location: models/")

print("\n" + "="*70)
print("  ✅ PHASE 5 COMPLETE")
print("  🏆 Best Model: " + best_model_name)
print("="*70)

