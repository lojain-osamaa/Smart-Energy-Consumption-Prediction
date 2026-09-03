"""
train.py
--------
Training script for the Smart Energy AI project.
Loads the cleaned dataset, performs feature engineering, trains three regression
models, and saves the best model (Linear Regression) as a deployment bundle.

Run from the project root directory:
    python src/train.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add src to path so we can import data_preprocessing
sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import (
    load_raw_data, clean_data, create_time_features, save_cleaned_data
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# File paths (relative to project root)
RAW_DATA_PATH     = "data/raw/energydata_complete.csv"
CLEANED_DATA_PATH = "data/processed/energydata_cleaned.csv"
MODEL_PATH        = "models/final_model.joblib"

TARGET        = "Appliances"
TIME_FEATURES = ["hour", "day_of_week", "month", "is_weekend"]


def main():
    print("Smart Energy AI - Training Script")
    print("=" * 50)

    # --- Load and clean data ---
    print("\nLoading raw data...")
    df = load_raw_data(RAW_DATA_PATH)
    print(f"  Raw shape: {df.shape}")

    print("Cleaning data...")
    df = clean_data(df)
    print(f"  Cleaned shape: {df.shape}")

    # Save cleaned data
    save_cleaned_data(df, CLEANED_DATA_PATH)

    # Create time-based features
    print("Creating time-based features...")
    df = create_time_features(df)

    # --- Identify features ---
    excluded = [TARGET, "date"]
    original_features = [c for c in df.columns if c not in excluded + TIME_FEATURES]
    all_features = original_features + TIME_FEATURES

    # --- Chronological train/test split ---
    print("\nSplitting data chronologically (80/20)...")
    split_index = int(len(df) * 0.80)

    X_train_full = df[all_features].iloc[:split_index]
    X_test_full  = df[all_features].iloc[split_index:]
    y_train      = df[TARGET].iloc[:split_index]
    y_test       = df[TARGET].iloc[split_index:]

    print(f"  Training data shape: {X_train_full.shape}")
    print(f"  Testing data shape:  {X_test_full.shape}")

    # --- Feature selection (training data only) ---
    print("\nSelecting features using Random Forest importance (training data only)...")
    rf_selector = RandomForestRegressor(
        n_estimators=60, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_selector.fit(X_train_full, y_train)

    importance_df = pd.DataFrame({
        "Feature": X_train_full.columns,
        "Importance": rf_selector.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # Select top 15 original + all 4 time features
    top_k = 15
    selected_original = [
        f for f in importance_df["Feature"] if f in original_features
    ][:top_k]
    final_features = selected_original + TIME_FEATURES

    print(f"  Selected {len(selected_original)} original + 4 time = {len(final_features)} features")

    X_train = X_train_full[final_features]
    X_test  = X_test_full[final_features]

    # --- Scale features (fit on training data only) ---
    print("Scaling features (StandardScaler fit on training data only)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # --- Train three regression models ---
    print("\nTraining models...")

    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)
    lin_preds = linear_model.predict(X_test_scaled)
    print("  LinearRegression trained.")

    rf_model = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_model.fit(X_train.values, y_train)
    rf_preds = rf_model.predict(X_test.values)
    print("  RandomForestRegressor trained.")

    gb_model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        random_state=RANDOM_STATE
    )
    gb_model.fit(X_train.values, y_train)
    gb_preds = gb_model.predict(X_test.values)
    print("  GradientBoostingRegressor trained.")

    # --- Cross-validation (TimeSeriesSplit) ---
    print("\nRunning TimeSeriesSplit cross-validation...")
    tscv = TimeSeriesSplit(n_splits=3)
    cv_results = {}
    for name, model, Xtr in [
        ("Linear Regression", linear_model, X_train_scaled),
        ("Random Forest",     rf_model,     X_train.values),
        ("Gradient Boosting", gb_model,     X_train.values),
    ]:
        scores = cross_val_score(
            model, Xtr, y_train,
            cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=1
        )
        cv_results[name] = -scores.mean()
        print(f"  {name}: CV RMSE = {cv_results[name]:.2f} Wh")

    # --- Evaluate on test set ---
    print("\nEvaluating on test set...")
    predictions = {
        "Linear Regression": lin_preds,
        "Random Forest":     rf_preds,
        "Gradient Boosting": gb_preds,
    }
    results = []
    for name, preds in predictions.items():
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        results.append({"Model": name, "MAE": round(mae, 2),
                        "RMSE": round(rmse, 2), "R2": round(r2, 4)})
        print(f"  {name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    best_model_name = results_df.iloc[0]["Model"]
    print(f"\n  Best model by RMSE: {best_model_name}")

    # --- Compute classification threshold (from training data only) ---
    threshold = float(y_train.quantile(0.90))

    # --- Save the final model (Linear Regression) ---
    print("\nSaving final model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Re-train final model on all training data
    final_model = LinearRegression()
    final_model.fit(X_train_scaled, y_train)

    deployment_bundle = {
        "model": final_model,
        "scaler": scaler,
        "features": final_features,
        "uses_scaled_input": True,
        "model_type": "LinearRegression",
        "threshold_high_consumption": threshold,
    }
    joblib.dump(deployment_bundle, MODEL_PATH)
    print(f"  Saved: {MODEL_PATH}")
    print(f"  Features: {final_features}")
    print(f"  Threshold: {threshold:.2f} Wh")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
