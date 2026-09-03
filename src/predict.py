"""
predict.py
----------
Prediction utility for the Smart Energy AI project.
Loads the saved deployment bundle and makes predictions for new input data.

Can be used as a module or run directly for a quick example.
"""

import os
import numpy as np
import pandas as pd
import joblib

# Default path to the saved model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_model.joblib")


def load_model(model_path: str = MODEL_PATH) -> dict:
    """Load the saved deployment bundle from disk."""
    bundle = joblib.load(model_path)
    return bundle


def prepare_input(input_data: dict, feature_names: list) -> np.ndarray:
    """
    Convert a dictionary of input values to a numpy array.
    The columns are ordered to match the training feature list.

    Parameters
    ----------
    input_data : dict
        Dictionary with feature names as keys and numeric values.
    feature_names : list
        Ordered list of feature names from the deployment bundle.

    Returns
    -------
    np.ndarray
        2D array with shape (1, n_features), ready for the scaler.
    """
    row = [input_data[f] for f in feature_names]
    return np.array(row, dtype=float).reshape(1, -1)


def predict(input_data: dict, bundle: dict = None, model_path: str = MODEL_PATH) -> dict:
    """
    Make a prediction for a single observation.

    Parameters
    ----------
    input_data : dict
        Dictionary containing raw feature values.
    bundle : dict, optional
        Pre-loaded deployment bundle. If None, it is loaded from model_path.
    model_path : str
        Path to the saved .joblib file. Used only if bundle is None.

    Returns
    -------
    dict
        predicted_wh      : predicted appliance energy consumption (Wh)
        is_high           : True if predicted value exceeds the threshold
        threshold_wh      : the high-consumption threshold in Wh
    """
    # Load the model if not provided
    if bundle is None:
        bundle = load_model(model_path)

    model     = bundle["model"]
    scaler    = bundle["scaler"]
    features  = bundle["features"]
    threshold = bundle["threshold_high_consumption"]

    # Prepare and scale the input
    X_raw    = prepare_input(input_data, features)
    X_scaled = scaler.transform(X_raw)

    # Make the prediction
    prediction = float(model.predict(X_scaled)[0])
    prediction = max(prediction, 0.0)  # energy cannot be negative

    return {
        "predicted_wh": round(prediction, 2),
        "is_high":       prediction > threshold,
        "threshold_wh":  threshold,
    }


if __name__ == "__main__":
    # Example prediction with typical indoor/outdoor values
    example_input = {
        "T3":          21.0,
        "RH_3":        39.0,
        "Press_mm_hg": 755.0,
        "T8":          21.5,
        "RH_5":        50.0,
        "RH_2":        40.0,
        "lights":       0.0,
        "RH_4":        39.0,
        "T5":          19.5,
        "Tdewpoint":    4.0,
        "RH_6":        80.0,
        "RH_1":        40.0,
        "RH_8":        42.0,
        "RH_9":        41.0,
        "T7":          20.5,
        "hour":        18,
        "day_of_week":  1,
        "month":        3,
        "is_weekend":   0,
    }

    print("Smart Energy AI - Example Prediction")
    print("=" * 40)
    result = predict(example_input)
    print(f"Predicted Appliance Energy: {result['predicted_wh']} Wh")
    print(f"High Consumption Alert:     {result['is_high']}")
    print(f"High Consumption Threshold: {result['threshold_wh']} Wh")
