from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

from services.analysis_service import AnalysisService
from services.chart_service import ChartService
from services.data_loader import DataRepository
from services.prediction_service import PredictionService


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
DATASET_DIR = DATA_DIR / 'filled_dataset'
SCHEMA_PATH = DATA_DIR / 'data_schema.py'
GEOJSON_PATH = DATA_DIR / 'mekong_basin.geojson'


repository = DataRepository(DATASET_DIR, SCHEMA_PATH, GEOJSON_PATH)
chart_service = ChartService(repository)
analysis_service = AnalysisService(repository, chart_service)
prediction_service = PredictionService(repository)

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/bootstrap')
def bootstrap():
    return jsonify(repository.bootstrap_payload())


@app.route('/api/mekong-geojson')
def mekong_geojson():
    return GEOJSON_PATH.read_text(encoding='utf-8'), 200, {'Content-Type': 'application/json'}


@app.post('/api/visualize')
def visualize():
    payload = request.get_json(silent=True) or {}
    try:
        result = chart_service.generate_chart(payload)
        return jsonify({'ok': True, 'result': result})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.post('/api/analyze')
def analyze():
    payload = request.get_json(silent=True) or {}
    try:
        result = analysis_service.analyse(payload)
        return jsonify({'ok': True, 'result': result})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.post('/api/predict')
def predict():
    payload = request.get_json(silent=True) or {}
    station = str(payload.get('station', '')).strip()
    feature = str(payload.get('feature', '')).strip()
    horizon = int(payload.get('horizon', 7))
    try:
        result = prediction_service.predict(station, feature, horizon)
        return jsonify({'ok': True, 'result': result})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=8000)
