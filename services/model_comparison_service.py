from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import markdown
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io

from .analysis_service import _gemini_generate
from .data_loader import SeriesRequest


MODELS = [
    ('FlowNet',  '#60a5fa', 'rgba(96,165,250,0.3)',  'circle'),
    ('LSTM',     '#34d399', 'rgba(52,211,153,0.3)',   'square'),
    ('PatchTST', '#f59e0b', 'rgba(245,158,11,0.3)',   'diamond'),
    ('DLinear',  '#f87171', 'rgba(248,113,113,0.3)',  'triangle-up'),
]

FEATURE_FOLDER = {
    'Discharge': 'Water_Discharge',
    'Water_Level': 'Water_Level',
    'Rainfall': 'Rainfall',
    'Total_Suspended_Solids': 'Total_Suspended_Solids',
}


def _generate_modelcompare_analysis(result: Dict[str, Any]) -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_google_gemini_api_key_here' or not api_key.strip():
        return _fallback_modelcompare_analysis(result)
    try:
        s = result.get('stats', {})
        prompt = f"""Act as a professional hydrologist comparing multiple forecast models for operational use.
Write the response in markdown and structure it exactly as follows:

**Executive Summary**
2-3 sentences summarising which model performs best, how large the skill differences are, and what that means for forecast confidence.

**Detailed Insights**
- **Best Model:** identify the best-performing model using the reported metrics.
- **Performance Spread:** compare RMSE and MAPE across the available models.
- **Forecast Confidence:** explain what the spread between models implies for confidence in the forecast.
- **Operational Interpretation:** state how this comparison should be used in practice.

Rules:
- Use professional hydrological language.
- Always cite specific numbers from the provided results.
- Replace underscores with spaces.
- Do not include any introduction or sign-off outside the two sections.

Comparison title: {str(result.get('title', '')).replace('_', ' ')}
Best model: {s.get('best_model_by_rmse')}
Horizon: {s.get('horizon_days')} days
Models: {s.get('models')}
"""
        return markdown.markdown(_gemini_generate(api_key, prompt))
    except Exception:
        return _fallback_modelcompare_analysis(result)


def _fallback_modelcompare_analysis(result: Dict[str, Any]) -> str:
    s = result.get('stats', {}) or {}
    title = str(result.get('title', '')).replace('_', ' ')
    best = s.get('best_model_by_rmse', 'n/a')
    horizon = s.get('horizon_days')
    models = [m for m in (s.get('models') or []) if m.get('source_note') != 'no_data']

    def _rmse(model):
        try:
            return float(model.get('RMSE'))
        except Exception:
            return None

    ranked = sorted((m for m in models if _rmse(m) is not None), key=_rmse)
    best_row = ranked[0] if ranked else None
    second_row = ranked[1] if len(ranked) > 1 else None
    spread_text = (
        f'The lowest RMSE is {best_row.get("RMSE")} for {best_row.get("Model")}, compared with {second_row.get("RMSE")} for {second_row.get("Model")}.'
        if best_row and second_row else
        'Only one model has usable trained output for this station-feature combination, so cross-model spread is limited.'
    )

    mape_bits = ', '.join(
        f'{m.get("Model")} {m.get("MAPE")}'
        for m in models[:4]
    ) or 'no valid MAPE values'

    return (
        '<p><strong>Executive Summary</strong></p>'
        f'<p>The multi-model comparison for <strong>{title}</strong> evaluates a {horizon}-day forecast horizon and identifies <strong>{best}</strong> as the best performer by RMSE. '
        f'This comparison provides a screening view of relative model skill rather than a full uncertainty quantification framework.</p>'
        '<p><strong>Detailed Insights</strong></p>'
        '<ul>'
        f'<li><strong>Best Model:</strong> The current best model by RMSE is {best}, based on the reported historical fit metrics.</li>'
        f'<li><strong>Performance Spread:</strong> {spread_text} Reported MAPE values are {mape_bits}.</li>'
        '<li><strong>Forecast Confidence:</strong> When multiple models produce similar errors, forecast confidence is stronger; when their skill differs materially, the forecast should be interpreted with more caution.</li>'
        '<li><strong>Operational Interpretation:</strong> Use the best-RMSE model as the default operational candidate, but review the model spread and plotted forecast divergence before making high-stakes decisions.</li>'
        '</ul>'
    )


class ModelComparisonService:
    """
    Multi-Model Forecast Comparison using pre-trained ML models.

    Loads FlowNet, LSTM, PatchTST, and DLinear predictions from the
    station_predictions (historical fit for RMSE/MAPE) and
    station_predictions_future (forecast) CSV files.
    """

    def __init__(self, repository, data_dir: str | Path = 'data') -> None:
        self.repo = repository
        self.data_dir = Path(data_dir)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _find_repo(self, dataset: str):
        if hasattr(self.repo, 'repos'):
            return next((r for r in self.repo.repos if r.dataset == dataset), None)
        if getattr(self.repo, 'dataset', '') == dataset:
            return self.repo
        return None

    def _get_dataset(self, station: str) -> Optional[str]:
        if hasattr(self.repo, '_station_to_repo'):
            r = self.repo._station_to_repo.get(station)
            return r.dataset if r else None
        return getattr(self.repo, 'dataset', None)

    def _feature_folder(self, feature: str) -> str:
        return FEATURE_FOLDER.get(feature, feature)

    def _load_actual(self, station: str, feature: str) -> Optional[pd.DataFrame]:
        """Load full historical daily series as DataFrame with Timestamp/Value columns."""
        dataset = self._get_dataset(station)
        repo = self._find_repo(dataset) if dataset else None
        if repo is None:
            return None
        try:
            fd = repo.station_index[station]['feature_details'][feature]
            req = SeriesRequest(
                station=station, feature=feature,
                start_date=fd['start_date'], end_date=fd['end_date'],
            )
            df = repo.get_feature_series(req)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            return df[['Timestamp', 'Value']].dropna().sort_values('Timestamp').reset_index(drop=True)
        except Exception:
            return None

    def _load_future(self, station: str, feature: str, model: str, horizon: int) -> Optional[np.ndarray]:
        """Load future forecast from station_predictions_future CSVs."""
        dataset = self._get_dataset(station)
        if dataset == 'mekong':
            path = (self.data_dir / 'Mekong' / 'prediction_results' / 'station_predictions_future'
                    / self._feature_folder(feature) / model / f'{station}.csv')
        elif dataset == 'lamah':
            path = (self.data_dir / 'LamaH' / 'prediction_results' / 'station_predictions_future'
                    / model / f'{station}.csv')
            if not path.exists():
                path = (self.data_dir / 'LamaH' / 'prediction_results' / 'station_predictions_future'
                        / model / 'LamaH_daily' / f'{station}.csv')
        else:
            return None
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, nrows=1)
            return df.iloc[0, :horizon].values.astype(float)
        except Exception:
            return None

    def _load_historical_fit(
        self, station: str, feature: str, model: str, horizon: int, actual_df: pd.DataFrame
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, float]:
        """
        Load rolling predictions from station_predictions and compute RMSE/MAPE.
        Returns (fit_dates, fit_values, rmse, mape).
        """
        dataset = self._get_dataset(station)
        if dataset == 'mekong':
            path = (self.data_dir / 'Mekong' / 'prediction_results' / 'station_predictions'
                    / self._feature_folder(feature) / model / f'{station}.csv')
        elif dataset == 'lamah':
            path = (self.data_dir / 'LamaH' / 'prediction_results' / 'station_predictions'
                    / model / f'{station}.csv')
        else:
            return None, None, float('nan'), float('nan')

        if not path.exists():
            return None, None, float('nan'), float('nan')
        try:
            pred_df = pd.read_csv(path)
            n_windows, n_horizons = len(pred_df), len(pred_df.columns)
            actual_h = min(horizon, n_horizons)
            col_idx = actual_h - 1

            last_date = actual_df['Timestamp'].max()
            eval_start = last_date - pd.Timedelta(days=n_windows + n_horizons - 2)
            dates = pd.date_range(
                start=eval_start + pd.Timedelta(days=actual_h - 1),
                periods=n_windows, freq='D',
            )
            values = pred_df.iloc[:, col_idx].values.astype(float)
            fit_df = pd.DataFrame({'Timestamp': dates, 'ModelFit': values})

            actual_indexed = actual_df.set_index('Timestamp')['Value']
            fit_df = fit_df[fit_df['Timestamp'].isin(actual_indexed.index)].reset_index(drop=True)
            if fit_df.empty:
                return None, None, float('nan'), float('nan')

            actuals = actual_indexed.reindex(fit_df['Timestamp']).values
            preds = fit_df['ModelFit'].values
            residuals = actuals - preds
            rmse = float(np.sqrt(np.nanmean(residuals ** 2)))

            act_mean = float(np.nanmean(np.abs(actuals)))
            threshold = max(act_mean * 0.01, 1e-6)
            valid = np.abs(actuals) > threshold
            mape = float(np.nanmean(np.abs(residuals[valid] / actuals[valid])) * 100) if valid.sum() > 0 else float('nan')

            return fit_df['Timestamp'].values, fit_df['ModelFit'].values, rmse, mape
        except Exception:
            return None, None, float('nan'), float('nan')

    # ── main entry ─────────────────────────────────────────────────────────────

    def compare(
        self,
        dataset: str,
        station: str,
        feature: str,
        horizon: int = 14,
        include_analysis: bool = False,
        capability_service=None,
    ) -> Dict[str, Any]:
        repo = self._find_repo(dataset)
        if repo is None:
            raise ValueError(f"Dataset '{dataset}' not found.")

        actual_df = self._load_actual(station, feature)
        if actual_df is None or len(actual_df) < 60:
            raise ValueError(f"Insufficient data for '{station}' / '{feature}'.")

        horizon = max(1, min(horizon, 30))
        unit = repo.feature_units.get(feature, '')
        station_name = repo.station_index[station].get('name', station)
        feature_label = feature.replace('_', ' ').title()

        # Show last 365 days of actual history on chart
        display_days = 365
        plot_actual = actual_df.iloc[-display_days:] if len(actual_df) > display_days else actual_df
        last_date = actual_df['Timestamp'].max()
        last_value = float(actual_df.loc[actual_df['Timestamp'] == last_date, 'Value'].iloc[0])
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
        # Prepend last actual point so forecast lines connect to history
        connected_dates = [last_date] + list(future_dates)
        # Clip negatives if feature is physically non-negative (e.g. discharge)
        non_negative = float(actual_df['Value'].dropna().min()) >= 0


        DARK_BG = '#07111f'
        TEXT = '#e5eefc'

        fig = go.Figure()

        # Actual historical line
        fig.add_trace(go.Scatter(
            x=plot_actual['Timestamp'].tolist(),
            y=plot_actual['Value'].tolist(),
            mode='lines',
            name='Actual',
            line=dict(color='rgba(148,163,184,0.8)', width=1.5),
            hovertemplate='%{x|%Y-%m-%d}: %{y:.3f} ' + unit + '<extra></extra>',
        ))

        metrics: List[Dict[str, str]] = []

        for model_name, color, color_faint, symbol in MODELS:
            # Historical fit for metrics
            fit_dates, fit_vals, rmse, mape = self._load_historical_fit(
                station, feature, model_name, horizon, actual_df
            )
            # Future forecast
            future_vals = self._load_future(station, feature, model_name, horizon)

            has_fit = fit_dates is not None and fit_vals is not None
            has_future = future_vals is not None

            if not has_fit and not has_future:
                metrics.append({
                    'Model': model_name, 'RMSE': 'n/a', 'MAPE': 'n/a',
                    'source_note': 'no_data',
                })
                continue

            # Plot historical fit (last 365 days of fit window)
            if has_fit:
                fit_df = pd.DataFrame({'ts': fit_dates, 'val': fit_vals})
                fit_df = fit_df[fit_df['ts'] >= plot_actual['Timestamp'].min()]
                if not fit_df.empty:
                    fig.add_trace(go.Scatter(
                        x=fit_df['ts'].tolist(), y=fit_df['val'].tolist(),
                        mode='lines', name=f'{model_name} fit',
                        line=dict(color=color_faint, width=1),
                        showlegend=False, hoverinfo='skip',
                    ))

            # Plot future forecast (prepend last actual point to connect)
            if has_future:
                if non_negative:
                    future_vals = np.clip(future_vals, 0, None)
                fig.add_trace(go.Scatter(
                    x=connected_dates, y=[last_value] + future_vals.tolist(),
                    mode='lines+markers', name=model_name,
                    line=dict(color=color, width=1.5),
                    marker=dict(size=4, color=color, symbol=symbol),
                    hovertemplate=f'{model_name} %{{x|%Y-%m-%d}}: %{{y:.3f}} {unit}<extra></extra>',
                ))

            metrics.append({
                'Model': model_name,
                'RMSE': f'{rmse:.3f}' if not np.isnan(rmse) else 'n/a',
                'MAPE': f'{mape:.1f}%' if not np.isnan(mape) else 'n/a',
                'source_note': 'trained_model',
            })

        # Forecast start line
        fig.update_layout(
            paper_bgcolor=DARK_BG,
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, sans-serif', color=TEXT, size=12),
            title=dict(
                text=f'Multi-Model Forecast Comparison — {feature_label} · {station_name}',
                font=dict(size=14, color=TEXT),
                x=0.5, xanchor='center',
            ),
            xaxis=dict(
                title='Date', gridcolor='rgba(148,163,184,0.08)',
                showgrid=True, zeroline=False,
            ),
            yaxis=dict(
                title=f'{feature_label} ({unit})' if unit else feature_label,
                gridcolor='rgba(148,163,184,0.08)',
                showgrid=True, zeroline=False,
            ),
            legend=dict(
                orientation='v', yanchor='top', y=0.99,
                xanchor='left', x=0.01,
                bgcolor='rgba(7,17,31,0.82)',
                bordercolor='rgba(148,163,184,0.15)',
                borderwidth=1, font=dict(size=10),
            ),
            shapes=[dict(
                type='line', x0=last_date, x1=last_date,
                y0=0, y1=1, yref='paper',
                line=dict(color='rgba(148,163,184,0.4)', dash='dot', width=1.5),
            )],
            annotations=[dict(
                x=last_date, y=1.04, yref='paper',
                text='↑ Forecast', showarrow=False, xanchor='center',
                font=dict(color='rgba(148,163,184,0.6)', size=10),
            )],
            margin=dict(l=60, r=20, t=60, b=50),
        )

        # Guard: if every model has no data, raise an informative error
        all_no_data = all(m.get('source_note') == 'no_data' for m in metrics)
        if all_no_data:
            raise ValueError(
                f"No trained model artifacts found for station '{station}' / feature '{feature}'. "
                f"Only stations with pre-computed prediction CSVs are supported for model comparison."
            )

        valid_rmse = [(m['Model'], float(m['RMSE'])) for m in metrics if m['RMSE'] not in ('n/a', '—')]
        best_model = min(valid_rmse, key=lambda x: x[1])[0] if valid_rmse else 'n/a'

        # ── Zoom figure: last 90 days + forecast ──────────────────────────────
        zoom_actual = actual_df[actual_df['Timestamp'] >= last_date - pd.Timedelta(days=90)]
        fig_zoom = go.Figure()
        fig_zoom.add_trace(go.Scatter(
            x=zoom_actual['Timestamp'].tolist(),
            y=zoom_actual['Value'].tolist(),
            mode='lines', name='Actual',
            line=dict(color='rgba(148,163,184,0.8)', width=1.5),
            hovertemplate='%{x|%Y-%m-%d}: %{y:.3f} ' + unit + '<extra></extra>',
        ))
        for model_name, color, color_faint, symbol in MODELS:
            future_vals = self._load_future(station, feature, model_name, horizon)
            if future_vals is not None:
                if non_negative:
                    future_vals = np.clip(future_vals, 0, None)
                fig_zoom.add_trace(go.Scatter(
                    x=connected_dates, y=[last_value] + future_vals.tolist(),
                    mode='lines+markers', name=model_name,
                    line=dict(color=color, width=1.5),
                    marker=dict(size=4, color=color, symbol=symbol),
                    hovertemplate=f'{model_name} %{{x|%Y-%m-%d}}: %{{y:.3f}} {unit}<extra></extra>',
                ))
        fig_zoom.update_layout(
            paper_bgcolor=DARK_BG,
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, sans-serif', color=TEXT, size=12),
            title=dict(
                text=f'Last 90 Days + Forecast — {feature_label} · {station_name}',
                font=dict(size=14, color=TEXT),
                x=0.5, xanchor='center',
            ),
            xaxis=dict(title='Date', gridcolor='rgba(148,163,184,0.08)', showgrid=True, zeroline=False),
            yaxis=dict(
                title=f'{feature_label} ({unit})' if unit else feature_label,
                gridcolor='rgba(148,163,184,0.08)', showgrid=True, zeroline=False,
            ),
            legend=dict(
                orientation='v', yanchor='top', y=0.99,
                xanchor='left', x=0.01,
                bgcolor='rgba(7,17,31,0.82)',
                bordercolor='rgba(148,163,184,0.15)',
                borderwidth=1, font=dict(size=10),
            ),
            shapes=[dict(
                type='line', x0=last_date, x1=last_date,
                y0=0, y1=1, yref='paper',
                line=dict(color='rgba(148,163,184,0.4)', dash='dot', width=1.5),
            )],
            annotations=[dict(
                x=last_date, y=1.04, yref='paper',
                text='↑ Forecast', showarrow=False, xanchor='center',
                font=dict(color='rgba(148,163,184,0.6)', size=10),
            )],
            margin=dict(l=60, r=20, t=60, b=50),
        )

        result = {
            'title': f'Model Comparison · {station_name}',
            'subtitle': f'{feature_label} · {horizon}-day forecast · best: {best_model}',
            'figure': plotly.io.to_json(fig),
            'figure_zoom': plotly.io.to_json(fig_zoom),
            'stats': {
                'models': metrics,
                'best_model_by_rmse': best_model,
                'horizon_days': horizon,
            },
        }

        if include_analysis:
            analysis = _generate_modelcompare_analysis(result)
            if analysis:
                result['analysis'] = analysis

        return result
