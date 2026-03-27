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
prediction_service = PredictionService(repository, data_dir=BASE_DIR / 'data')

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


MEKONG_FEATURE_DIRS = {'Water_Discharge', 'Water_Level', 'Rainfall', 'Total_Suspended_Solids'}
NON_MODEL_DIRS = MEKONG_FEATURE_DIRS | {'LamaH_daily'}


@app.route('/api/predict-models')
def predict_models():
    """Return the union of all model names found in prediction_results directories."""
    models = set()
    for base_dir in [LAMAH_DIR, MEKONG_DIR]:
        for sub in ['station_predictions', 'station_predictions_future']:
            parent = base_dir / 'prediction_results' / sub
            if not parent.is_dir():
                continue
            for d in parent.iterdir():
                if not d.is_dir():
                    continue
                if d.name in MEKONG_FEATURE_DIRS:
                    # Mekong feature folder — children are model dirs
                    for model_dir in d.iterdir():
                        if model_dir.is_dir() and model_dir.name not in NON_MODEL_DIRS:
                            models.add(model_dir.name)
                elif d.name not in NON_MODEL_DIRS:
                    models.add(d.name)
    return jsonify(sorted(models))


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


@app.post('/api/analyze-free')
def analyze_free():
    payload = request.get_json(silent=True) or {}
    try:
        result = analysis_service.analyse_free(payload)
        return jsonify({'ok': True, 'result': result})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.post('/api/analyze-free-multi')
def analyze_free_multi():
    payload = request.get_json(silent=True) or {}
    try:
        result = analysis_service.analyse_free_multi(payload)
        return jsonify({'ok': True, 'result': result})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.route('/api/predict-stations')
def predict_stations():
    model = request.args.get('model', 'FlowNet').strip()
    result = {
        'lamah': {'historical': [], 'future': []},
        'mekong': {'historical': [], 'future': []},
    }

    # LamaH historical
    lamah_hist_dir = LAMAH_DIR / 'prediction_results' / 'station_predictions' / model
    if lamah_hist_dir.is_dir():
        result['lamah']['historical'] = sorted(p.stem for p in lamah_hist_dir.glob('*.csv'))

    # LamaH future — FlowNet has an extra LamaH_daily subfolder; others have CSVs directly
    lamah_future_dir = LAMAH_DIR / 'prediction_results' / 'station_predictions_future' / model
    if lamah_future_dir.is_dir():
        csvs = list(lamah_future_dir.glob('*.csv'))
        if not csvs and (lamah_future_dir / 'LamaH_daily').is_dir():
            csvs = list((lamah_future_dir / 'LamaH_daily').glob('*.csv'))
        result['lamah']['future'] = sorted(p.stem for p in csvs)

    # Mekong historical
    mekong_hist_base = MEKONG_DIR / 'prediction_results' / 'station_predictions'
    if mekong_hist_base.is_dir():
        seen = set()
        for feat_dir in mekong_hist_base.iterdir():
            if not feat_dir.is_dir():
                continue
            model_dir = feat_dir / model
            if model_dir.is_dir():
                for p in model_dir.glob('*.csv'):
                    seen.add(p.stem)
        result['mekong']['historical'] = sorted(seen)

    # Mekong future
    mekong_future_base = MEKONG_DIR / 'prediction_results' / 'station_predictions_future'
    if mekong_future_base.is_dir():
        seen = set()
        for feat_dir in mekong_future_base.iterdir():
            if not feat_dir.is_dir():
                continue
            model_dir = feat_dir / model
            if model_dir.is_dir():
                for p in model_dir.glob('*.csv'):
                    seen.add(p.stem)
        result['mekong']['future'] = sorted(seen)

    return jsonify(result)


@app.post('/api/predict')
def predict():
    payload = request.get_json(silent=True) or {}
    station = str(payload.get('station', '')).strip()
    feature = str(payload.get('feature', '')).strip()
    horizon = int(payload.get('horizon', 7))
    model = str(payload.get('model', 'FlowNet')).strip()
    mode = str(payload.get('mode', 'future')).strip()
    analysis = bool(payload.get('analysis', True))
    try:
        result = prediction_service.predict(station, feature, horizon, model, mode, analysis)
        return jsonify({'ok': True, 'result': result})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=8000)
