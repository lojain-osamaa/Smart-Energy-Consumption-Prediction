"""
data_preprocessing.py
---------------------
Utility functions for loading and cleaning the Smart Energy AI dataset.
Used by train.py and optionally by the notebooks.
"""

import pandas as pd
import numpy as np
import os


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw CSV dataset from the given path."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset:
    - Convert the date column to datetime (dayfirst=True format)
    - Drop rows with invalid dates if any
    - Retain outliers (they are legitimate sensor readings)

    Returns the cleaned DataFrame.
    """
    df = df.copy()

    # Convert date string to datetime (format is DD-MM-YYYY HH:MM)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    # Remove rows with invalid dates
    invalid_count = df["date"].isnull().sum()
    if invalid_count > 0:
        print(f"  Removing {invalid_count} rows with invalid dates.")
        df = df.dropna(subset=["date"]).reset_index(drop=True)

    # Sort by date to ensure chronological order
    df = df.sort_values("date").reset_index(drop=True)

    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from the date column.
    These features capture daily and seasonal usage patterns.

    Returns DataFrame with new columns: hour, day_of_week, month, is_weekend.
    """
    df = df.copy()

    # Extract time components from the date column
    df["hour"]        = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek   # 0=Monday, 6=Sunday
    df["month"]       = df["date"].dt.month
    df["is_weekend"]  = (df["date"].dt.dayofweek >= 5).astype(int)

    return df


def get_features_and_target(df: pd.DataFrame, selected_features: list, target: str = "Appliances"):
    """
    Return X (feature matrix) and y (target series) for a given feature list.
    """
    X = df[selected_features]
    y = df[target]
    return X, y


def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """Save the cleaned DataFrame to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")
    print(f"  Rows: {df.shape[0]}")
    print(f"  Columns: {df.shape[1]}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
