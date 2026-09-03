"""FastAPI application for the Smart Energy AI historical-data dashboard."""

from datetime import date, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "final_model.joblib"
DATA_PATH = BASE_DIR / "data" / "processed" / "energydata_cleaned.csv"
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
app = FastAPI(title="Smart Energy AI API", version="1.1.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def load_bundle():
    """Load the deployment bundle created by src/train.py."""
    result = joblib.load(MODEL_PATH)
    required = {"model", "scaler", "features", "threshold_high_consumption"}
    missing = required.difference(result)
    if missing:
        raise ValueError(f"Model bundle is missing: {', '.join(sorted(missing))}")
    return result


def load_data():
    """Load historical measurements and derive training-compatible time fields."""
    data = pd.read_csv(DATA_PATH)
    if not {"date", "Appliances"}.issubset(data.columns):
        raise ValueError("Processed data must contain date and Appliances columns.")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    data["hour"] = data["date"].dt.hour
    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    return data


try:
    bundle = load_bundle()
    model, scaler = bundle["model"], bundle["scaler"]
    FEATURES = list(bundle["features"])
    THRESHOLD = float(bundle["threshold_high_consumption"])
    model_error = None
except Exception as exc:
    bundle = model = scaler = None
    FEATURES, THRESHOLD, model_error = [], None, str(exc)
try:
    df_data, data_error = load_data(), None
except Exception as exc:
    df_data, data_error = pd.DataFrame(), str(exc)


class PredictionInput(BaseModel):
    """Raw sensor values; time features are derived from the selected date and time."""
    selected_date: date
    selected_time: time
    T3: float
    RH_3: float = Field(ge=0, le=100)
    T8: float
    RH_5: float = Field(ge=0, le=100)
    RH_2: float = Field(ge=0, le=100)
    lights: float = Field(ge=0)
    RH_4: float = Field(ge=0, le=100)
    T5: float
    RH_1: float = Field(ge=0, le=100)
    RH_6: float = Field(ge=0, le=100)
    RH_8: float = Field(ge=0, le=100)
    RH_7: float = Field(ge=0, le=100)
    RH_out: float = Field(ge=0, le=100)
    T4: float
    Press_mm_hg: float = Field(gt=0)


def require_data():
    if df_data.empty:
        raise HTTPException(503, detail=f"Historical data is unavailable: {data_error}")
    return df_data


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/health")
def health():
    return {"status": "ok" if bundle and not df_data.empty else "degraded",
            "model_loaded": bundle is not None, "data_loaded": not df_data.empty,
            "model_error": model_error, "data_error": data_error}


@app.get("/api/features")
def get_features():
    return {"required_features": FEATURES}


@app.get("/api/info")
def get_info():
    return {"model_type": bundle.get("model_type", type(model).__name__) if bundle else None,
            "num_features": len(FEATURES), "features": FEATURES,
            "threshold_wh": round(THRESHOLD, 2) if THRESHOLD is not None else None,
            "historical_data": True}


@app.post("/api/predict")
def predict(payload: PredictionInput):
    if not bundle:
        raise HTTPException(503, detail=f"Prediction model is unavailable: {model_error}")
    timestamp = pd.Timestamp.combine(payload.selected_date, payload.selected_time)
    values = payload.model_dump()
    values.update(hour=timestamp.hour, day_of_week=timestamp.dayofweek, month=timestamp.month,
                  is_weekend=int(timestamp.dayofweek >= 5))
    missing = [feature for feature in FEATURES if feature not in values]
    if missing:
        raise HTTPException(500, detail=f"Deployment input mapping is missing: {', '.join(missing)}")
    try:
        raw = pd.DataFrame([[values[feature] for feature in FEATURES]], columns=FEATURES)
        predicted = max(float(model.predict(scaler.transform(raw) if scaler is not None else raw)[0]), 0.0)
    except Exception as exc:
        raise HTTPException(500, detail=f"Prediction failed: {exc}") from exc
    level = "High" if predicted >= THRESHOLD else "Normal"
    return {"success": True, "predicted_wh": round(predicted, 2), "predicted_kwh": round(predicted / 1000, 4),
            "level": level, "threshold_wh": round(THRESHOLD, 2),
            "message": "High consumption expected for the selected conditions." if level == "High"
                       else "Consumption is expected to be below the high-consumption threshold.",
            "derived_time": {key: values[key] for key in ("hour", "day_of_week", "month", "is_weekend")}}


@app.get("/api/analytics/summary")
def analytics_summary():
    data = require_data()
    peak_hour = int(data.groupby("hour")["Appliances"].mean().idxmax())
    latest = data.iloc[-1]
    return {"average_consumption_wh": round(float(data["Appliances"].mean()), 2),
            "latest_consumption_wh": round(float(latest["Appliances"]), 2),
            "latest_recorded_at": latest["date"].isoformat(), "peak_hour": f"{peak_hour:02d}:00"}


@app.get("/api/analytics/hourly")
def hourly_analytics():
    grouped = require_data().groupby("hour")["Appliances"].mean().reindex(range(24))
    return {"hours": list(range(24)), "averages": grouped.round(2).fillna(0).tolist()}


@app.get("/api/analytics/daily")
def daily_analytics():
    grouped = require_data().groupby("day_of_week")["Appliances"].mean().reindex(range(7))
    return {"days": DAY_NAMES, "averages": grouped.round(2).fillna(0).tolist()}


@app.get("/api/analytics/weekly")
def weekly_analytics():
    values = require_data().set_index("date")["Appliances"].resample("W-MON").mean()
    return {"periods": values.index.strftime("%Y-%m-%d").tolist(), "averages": values.round(2).tolist()}


@app.get("/api/analytics/monthly")
def monthly_analytics():
    values = require_data().groupby("month")["Appliances"].mean().reindex(range(1, 13))
    return {"months": list(range(1, 13)), "averages": values.round(2).fillna(0).tolist()}


@app.get("/api/analytics/heatmap")
def heatmap_analytics():
    values = require_data().pivot_table(index="day_of_week", columns="hour", values="Appliances", aggfunc="mean")
    values = values.reindex(index=range(7), columns=range(24)).round(2).fillna(0)
    return {"days": DAY_NAMES, "hours": list(range(24)), "values": values.values.tolist()}


@app.get("/api/analytics/peak-hours")
def peak_hours():
    values = require_data().groupby("hour")["Appliances"].mean().sort_values(ascending=False).head(3)
    return {"peak_hours": [{"hour": f"{int(hour):02d}:00", "average_wh": round(float(value), 2)}
                           for hour, value in values.items()]}


@app.get("/api/model/performance")
def model_performance():
    data = require_data()
    if not bundle:
        raise HTTPException(503, detail=f"Prediction model is unavailable: {model_error}")
    test = data.iloc[int(len(data) * 0.8):]
    missing = [feature for feature in FEATURES if feature not in test]
    if missing:
        raise HTTPException(500, detail=f"Evaluation data lacks: {', '.join(missing)}")
    predicted = model.predict(scaler.transform(test[FEATURES]) if scaler is not None else test[FEATURES])
    actual = test["Appliances"].to_numpy()
    nonzero = actual != 0
    mape = np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100 if nonzero.any() else None
    return {"MAE": round(float(mean_absolute_error(actual, predicted)), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(actual, predicted))), 2),
            "R2": round(float(r2_score(actual, predicted)), 4),
            "MAPE": round(float(mape), 2) if mape is not None else None,
            "actual_subset": actual[-100:].round(2).tolist(),
            "predicted_subset": np.round(predicted[-100:], 2).tolist()}


@app.get("/api/model/comparison")
def model_comparison():
    return {"available": False,
            "message": "Model comparison results were not saved with this project, so they are not displayed."}
