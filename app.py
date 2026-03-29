from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

load_dotenv()

from services.analysis_service import AnalysisService
from services.chart_service import ChartService
from services.comparison_service import ComparisonService
from services.data_loader import DataRepository, MultiDataRepository, SeriesRequest
from services.index_service import IndexService
from services.network_service import NetworkService
from services.prediction_service import PredictionService
from services.extreme_service import ExtremeService
from services.quality_service import QualityService
from services.risk_service import RiskService
from services.scenario_service import ScenarioService


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
index_service = IndexService(repository)
comparison_service = ComparisonService(repository)
network_service = NetworkService(repository)
scenario_service = ScenarioService(repository, data_dir=BASE_DIR / 'data')
quality_service = QualityService(repository, data_dir=BASE_DIR / 'data')
extreme_service = ExtremeService(repository)
risk_service = RiskService(repository)

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


@app.route('/api/indices')
def indices():
    station = request.args.get('station', '').strip()
    if not station:
        return jsonify({'ok': False, 'error': 'station parameter required'}), 400
    try:
        result = index_service.compute_for_station(station)
        return jsonify({'ok': True, 'result': result})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.post('/api/export-csv')
def export_csv():
    payload = request.get_json(silent=True) or {}
    series_list = payload.get('series', [])
    if not series_list:
        return jsonify({'ok': False, 'error': 'series array required'}), 400
    try:
        frames = []
        for s in series_list:
            station = str(s.get('station', '')).strip()
            feature = str(s.get('feature', '')).strip()
            if not station or not feature:
                continue
            repo = next((r for r in repository.repos if station in r.station_index), None)
            if repo is None:
                continue
            fd = repo.station_index[station]['feature_details'].get(feature)
            if fd is None:
                continue
            req = SeriesRequest(
                station=station, feature=feature,
                start_date=s.get('start_date') or fd['start_date'],
                end_date=s.get('end_date') or fd['end_date'],
            )
            df = repo.get_feature_series(req)
            df.insert(0, 'Station', station)
            df.insert(1, 'Feature', feature)
            df['Unit'] = repo.feature_units.get(feature, '')
            frames.append(df)
        if not frames:
            raise ValueError('No data found for the given series.')
        combined = pd.concat(frames, ignore_index=True)
        first = series_list[0]
        safe = lambda v: str(v).replace('/', '_').replace(' ', '_')
        filename = f"{safe(first.get('station', 'data'))}_{safe(first.get('feature', 'export'))}.csv"
        return Response(
            combined.to_csv(index=False),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.post('/api/compare')
def compare():
    payload = request.get_json(silent=True) or {}
    dataset = str(payload.get('dataset', 'mekong')).strip()
    feature = str(payload.get('feature', '')).strip()
    year = payload.get('year')
    if year is not None:
        try:
            year = int(year)
        except (ValueError, TypeError):
            year = None
    component = str(payload.get('component', 'all')).strip()
    if not feature:
        return jsonify({'ok': False, 'error': 'feature parameter required'}), 400
    try:
        if component == 'correlation':
            result = comparison_service.compute_correlation_matrix(dataset, feature)
        elif component == 'leaderboard':
            result = comparison_service.compute_anomaly_leaderboard(dataset, feature, year)
        elif component == 'summary':
            result = comparison_service.compute_basin_summary(dataset, feature)
        else:
            result = comparison_service.compare(dataset, feature, year)
        return jsonify({'ok': True, 'result': result})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.route('/api/network')
def network():
    dataset = request.args.get('dataset', 'mekong').strip()
    try:
        result = network_service.compute_full_network(dataset)
        return jsonify({'ok': True, 'result': result})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.route('/api/network/travel-times')
def network_travel_times():
    dataset = request.args.get('dataset', 'mekong').strip()
    try:
        result = network_service.compute_travel_times(dataset)
        return jsonify({'ok': True, 'result': result})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.route('/api/network/contribution')
def network_contribution():
    station = request.args.get('station', '').strip()
    dataset = request.args.get('dataset', 'mekong').strip()
    if not station:
        return jsonify({'ok': False, 'error': 'station parameter required'}), 400
    try:
        result = network_service.compute_contribution(station, dataset)
        return jsonify({'ok': True, 'result': result})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


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


@app.post('/api/scenario')
def scenario():
    payload = request.get_json(silent=True) or {}
    station = str(payload.get('station', '')).strip()
    target_feature = str(payload.get('target_feature', '')).strip()
    driver_feature = str(payload.get('driver_feature', target_feature)).strip()
    scale_pct = float(payload.get('scale_pct', 20))
    duration_months = int(payload.get('duration_months', 3))
    start_offset = int(payload.get('start_offset', 0))
    model = str(payload.get('model', 'FlowNet')).strip()
    horizon = int(payload.get('horizon', 12))
    if not station or not target_feature:
        return jsonify({'ok': False, 'error': 'station and target_feature required'}), 400
    try:
        result = scenario_service.run_scenario(
            station, target_feature, driver_feature,
            scale_pct, duration_months, start_offset, model, horizon,
        )
        return jsonify({'ok': True, 'result': result})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.route('/api/quality/completeness')
def quality_completeness():
    station = request.args.get('station', '').strip()
    feature = request.args.get('feature', '').strip()
    if not station or not feature:
        return jsonify({'ok': False, 'error': 'station and feature required'}), 400
    try:
        return jsonify({'ok': True, 'result': quality_service.completeness(station, feature)})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.route('/api/quality/imputation')
def quality_imputation():
    dataset = request.args.get('dataset', 'mekong').strip()
    feature = request.args.get('feature', '').strip() or None
    try:
        return jsonify({'ok': True, 'result': quality_service.imputation_summary(dataset, feature)})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.route('/api/quality/gaps')
def quality_gaps():
    station = request.args.get('station', '').strip()
    feature = request.args.get('feature', '').strip()
    if not station or not feature:
        return jsonify({'ok': False, 'error': 'station and feature required'}), 400
    try:
        return jsonify({'ok': True, 'result': quality_service.gaps(station, feature)})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.route('/api/quality/anomalies')
def quality_anomalies():
    station = request.args.get('station', '').strip()
    feature = request.args.get('feature', '').strip()
    z_thresh = float(request.args.get('z_thresh', 3.0))
    if not station or not feature:
        return jsonify({'ok': False, 'error': 'station and feature required'}), 400
    try:
        return jsonify({'ok': True, 'result': quality_service.anomaly_candidates(station, feature, z_thresh)})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.post('/api/quality/flag')
def quality_flag():
    payload = request.get_json(silent=True) or {}
    station = str(payload.get('station', '')).strip()
    feature = str(payload.get('feature', '')).strip()
    date_str = str(payload.get('date', '')).strip()
    flag = str(payload.get('flag', '')).strip()
    if not station or not feature or not date_str or not flag:
        return jsonify({'ok': False, 'error': 'station, feature, date, flag required'}), 400
    try:
        return jsonify({'ok': True, 'result': quality_service.save_flag(station, feature, date_str, flag)})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.route('/api/quality/flags')
def quality_flags():
    try:
        return jsonify({'ok': True, 'result': quality_service.flag_summary()})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.route('/api/extreme')
def extreme():
    station = request.args.get('station', '').strip()
    feature = request.args.get('feature', '').strip()
    distribution = request.args.get('distribution', 'gev').strip()
    if not station or not feature:
        return jsonify({'ok': False, 'error': 'station and feature required'}), 400
    try:
        result = extreme_service.compute(station, feature, distribution)
        return jsonify({'ok': True, 'result': result})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.route('/api/risk')
def risk():
    dataset = request.args.get('dataset', 'mekong').strip()
    feature = request.args.get('feature', '').strip()
    lookback = int(request.args.get('lookback', 30))
    if not feature:
        return jsonify({'ok': False, 'error': 'feature parameter required'}), 400
    try:
        result = risk_service.compute_risk_map(dataset, feature, lookback)
        return jsonify({'ok': True, 'result': result})
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=8000)
