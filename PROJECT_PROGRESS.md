# Smart Energy AI — Project Progress

## Project overview

- Objective: predict aggregate household appliance consumption in Wh from historical sensor readings.
- Task: regression, with a high-consumption alert based on the training-set 90th-percentile threshold.
- Data: Appliances Energy Prediction Dataset in data/raw and data/processed.
- Saved deployment model: models/final_model.joblib, a Linear Regression bundle with StandardScaler and 19 ordered features.
- Primary GUI: FastAPI + HTML/CSS/Vanilla JavaScript + Plotly.js. streamlit_app.py is a legacy prototype only.

## Verified ML integration

- The bundle is the source of truth for model, scaler, feature ordering, and alert threshold.
- Required input columns: T3, RH_3, T8, RH_5, RH_2, lights, RH_4, T5, RH_1, RH_6, RH_8, RH_9, T7, Press_mm_hg, Tdewpoint, hour, day_of_week, month, is_weekend.
- The prediction route derives hour, day_of_week, month, and is_weekend from the selected date and time. Missing features are not silently set to zero.
- Historical time fields are derived when data loads, matching src/data_preprocessing.py.

## Completed

- Inspected the existing model, train/predict/preprocessing code, datasets, existing FastAPI UI, and documentation.
- Kept the saved model unchanged; no retraining occurred.
- Reworked app.py for validated input, clear unavailable-resource errors, and real data/model integration.
- Added weekly, monthly, heatmap, and peak-hour analytics endpoints.
- Replaced the old approximate heatmap with an actual day-of-week by hour aggregation.
- Updated the prediction form to use date/time and server-side feature derivation.
- Corrected the form after runtime inspection showed that the saved model expects RH_out, T4, and RH_7 rather than RH_9, T7, and Tdewpoint.
- Added an honest unavailable state for model comparison because no machine-readable comparison results were saved.
- Reduced requirements to the FastAPI app dependencies and added Jinja2.

## Main API routes

- GET / serves the web dashboard.
- GET /api/health reports model and data availability.
- POST /api/predict runs real saved-model inference.
- GET /api/analytics/summary, hourly, daily, weekly, monthly, heatmap, and peak-hours expose historical calculations.
- GET /api/model/performance calculates metrics on the same chronological 80/20 split used in training.

## Files modified

- app.py
- templates/index.html
- static/js/app.js
- static/js/charts.js
- requirements.txt
- run.bat
- README.md

## Testing status

- Python syntax check passed: python -m compileall -q app.py src.
- End-to-end API/UI testing is pending. The current MSYS2 Python 3.14 interpreter lacks Pandas, NumPy, scikit-learn, FastAPI, Joblib, and Uvicorn. Normal pip selects source packages that do not complete in this environment.

## Run instructions

1. Install standard CPython 3.11–3.13, or compatible MSYS2 Python packages.
2. Create a virtual environment and install requirements.
3. Start: python -m uvicorn app:app --reload.
4. Open http://localhost:8000.

## Limitations

- The data is historical, not live smart-meter data.
- Predictions are aggregate household appliance consumption, not individual-appliance monitoring.
- Model-comparison results are not displayed because no saved comparison table exists.

## Next action

After supported dependencies are available, start the server and test health, prediction, and dashboard routes; then update this file with the end-to-end test result.
