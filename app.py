from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

from services.analysis_service import AnalysisService
from services.chart_service import ChartService
from services.data_loader import DataRepository, MultiDataRepository
from services.prediction_service import PredictionService


BASE_DIR = Path(__file__).resolve().parent

# Mekong dataset paths
MEKONG_DIR = BASE_DIR / 'data' / 'Mekong'
MEKONG_DATASET_DIR = MEKONG_DIR / 'filled_dataset'
MEKONG_SCHEMA_PATH = MEKONG_DIR / 'data_schema.py'
MEKONG_GEOJSON_PATH = MEKONG_DIR / 'mekong_basin.geojson'

# LamaH dataset paths
LAMAH_DIR = BASE_DIR / 'data' / 'LamaH'
LAMAH_DATASET_DIR = LAMAH_DIR / 'filled_dataset'
LAMAH_SCHEMA_PATH = LAMAH_DIR / 'data_schema.py'
LAMAH_GEOJSON_PATH = LAMAH_DIR / 'lamah_countries.geojson'

print('Loading Mekong dataset...')
mekong_repo = DataRepository(MEKONG_DATASET_DIR, MEKONG_SCHEMA_PATH, MEKONG_GEOJSON_PATH, dataset='mekong')
print(f'  Mekong: {len(mekong_repo.station_index)} stations loaded.')

print('Loading LamaH dataset (857 stations — this may take ~30s)...')
lamah_repo = DataRepository(
    LAMAH_DATASET_DIR, LAMAH_SCHEMA_PATH, None,
    dataset='lamah',
)
print(f'  LamaH: {len(lamah_repo.station_index)} stations loaded.')

repository = MultiDataRepository([mekong_repo, lamah_repo])
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


@app.route('/api/datasets')
def datasets():
    data_dir = BASE_DIR / 'data'
    names = sorted(
        d.name for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith('_') and not d.name.startswith('.')
    )
    return jsonify(names)


@app.route('/api/mekong-geojson')
def mekong_geojson():
    return MEKONG_GEOJSON_PATH.read_text(encoding='utf-8'), 200, {'Content-Type': 'application/json'}


@app.route('/api/lamah-geojson')
def lamah_geojson():
    return LAMAH_GEOJSON_PATH.read_text(encoding='utf-8'), 200, {'Content-Type': 'application/json'}


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
