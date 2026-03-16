# Hydrovision Mekong Pro

A Flask-based spatial analytics dashboard for Mekong monitoring stations.

## Features
- World map base layer with Mekong basin overlay from GeoJSON
- Station markers built from `data_schema.py`
- Station hover tooltips and click-to-pin metadata card
- Feature filter buttons for Rainfall, Discharge, Water Level, and Total Suspended Solids
- Explore builder with per-row custom date ranges
- Visualization dock with persistent cards
- Analysis dock with automated narrative insights
- Prediction dock with time-series smoothing forecasts and confidence bands

## Project structure
- `app.py` - Flask application entry point
- `services/` - backend services for data loading, chart generation, analysis, and prediction
- `templates/index.html` - UI layout
- `static/css/styles.css` - styling
- `static/js/app.js` - front-end behaviour
- `data/filled_dataset/` - station CSV files
- `data/data_schema.py` - station metadata and feature availability
- `data/mekong_basin.geojson` - Mekong basin map overlay

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.
