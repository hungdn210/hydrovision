from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io
import scipy.stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

from .data_loader import SeriesRequest


class ModelComparisonService:
    """
    Multi-Model Forecast Comparison.

    Fits three models on the historical monthly record and projects forward:
      1. Holt-Winters (additive trend + seasonal, damped)
      2. ARIMA(2,1,2)
      3. Linear trend extrapolation

    For each model computes in-sample RMSE, MAPE, and AIC (where available).
    Returns a Plotly figure showing all three forecast lines against history
    plus a metrics summary table.
    """

    def __init__(self, repository) -> None:
        self.repo = repository

    # ── helpers ────────────────────────────────────────────────────────────────

    def _find_repo(self, dataset: str):
        if hasattr(self.repo, 'repos'):
            return next((r for r in self.repo.repos if r.dataset == dataset), None)
        if getattr(self.repo, 'dataset', '') == dataset:
            return self.repo
        return None

    def _load_series(self, repo, station: str, feature: str) -> Optional[pd.Series]:
        try:
            fd = repo.station_index[station]['feature_details'][feature]
            req = SeriesRequest(
                station=station, feature=feature,
                start_date=fd['start_date'], end_date=fd['end_date'],
            )
            df = repo.get_feature_series(req)
            ts = df.set_index('Timestamp')['Value'].sort_index()
            ts.index = pd.to_datetime(ts.index)
            return ts.dropna()
        except Exception:
            return None

    # ── main entry ─────────────────────────────────────────────────────────────

    def compare(
        self,
        dataset: str,
        station: str,
        feature: str,
        horizon: int = 12,
    ) -> Dict[str, Any]:
        repo = self._find_repo(dataset)
        if repo is None:
            raise ValueError(f"Dataset '{dataset}' not found.")

        ts = self._load_series(repo, station, feature)
        if ts is None or len(ts) < 60:
            raise ValueError(f"Insufficient data for '{station}' / '{feature}'.")

        unit = repo.feature_units.get(feature, '')
        station_name = repo.station_index[station].get('name', station)
        feature_label = feature.replace('_', ' ').title()

        # Resample to monthly means for stability
        monthly = ts.resample('ME').mean().dropna()
        if len(monthly) < 24:
            raise ValueError("Need at least 24 months of data for model comparison.")

        values = monthly.values.astype(float)
        dates = list(monthly.index)
        n = len(values)
        seasonal_period = 12

        horizon = max(1, min(horizon, 36))

        # Future dates
        last_date = dates[-1]
        future_dates = pd.date_range(
            start=last_date + pd.offsets.MonthEnd(1),
            periods=horizon, freq='ME',
        )

        DARK_BG = '#07111f'
        TEXT = '#e5eefc'

        fig = go.Figure()

        # Historical line
        fig.add_trace(go.Scatter(
            x=dates, y=values.tolist(),
            mode='lines',
            name='Historical monthly mean',
            line=dict(color='rgba(148,163,184,0.7)', width=1.5),
            hovertemplate='%{x|%Y-%m}: %{y:.3f} ' + unit + '<extra></extra>',
        ))

        metrics: List[Dict[str, str]] = []

        # ── Model 1: Holt-Winters ─────────────────────────────────────────────
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                hw_fit = ExponentialSmoothing(
                    values,
                    trend='add', seasonal='add',
                    seasonal_periods=seasonal_period,
                    damped_trend=True,
                    initialization_method='estimated',
                ).fit(optimized=True)
            hw_fitted = hw_fit.fittedvalues
            hw_forecast = hw_fit.forecast(horizon)
            res = values - hw_fitted
            hw_rmse = float(np.sqrt(np.mean(res ** 2)))
            mask = np.abs(values) > 1e-6
            hw_mape = float(np.mean(np.abs(res[mask] / values[mask])) * 100) if mask.any() else float('nan')
            hw_aic = float(hw_fit.aic)

            # Fitted line (faint)
            fig.add_trace(go.Scatter(
                x=dates, y=hw_fitted.tolist(),
                mode='lines', name='HW in-sample',
                line=dict(color='rgba(96,165,250,0.3)', width=1),
                showlegend=False, hoverinfo='skip',
            ))
            # Forecast line
            fig.add_trace(go.Scatter(
                x=list(future_dates), y=hw_forecast.tolist(),
                mode='lines+markers', name='Holt-Winters',
                line=dict(color='#60a5fa', width=2.5, dash='dash'),
                marker=dict(size=5, color='#60a5fa'),
                hovertemplate='HW %{x|%Y-%m}: %{y:.3f} ' + unit + '<extra></extra>',
            ))
            metrics.append({
                'Model': 'Holt-Winters',
                'RMSE': f'{hw_rmse:.3f}',
                'MAPE': f'{hw_mape:.1f}%' if not np.isnan(hw_mape) else 'n/a',
                'AIC': f'{hw_aic:.1f}',
            })
        except Exception:
            metrics.append({'Model': 'Holt-Winters', 'RMSE': 'failed', 'MAPE': '—', 'AIC': '—'})

        # ── Model 2: ARIMA(2,1,2) ─────────────────────────────────────────────
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                arima_fit = ARIMA(values, order=(2, 1, 2)).fit()
            arima_fitted = arima_fit.fittedvalues
            arima_fc = arima_fit.forecast(horizon)
            # residuals from index 1 onward (differencing drops first obs)
            res = values[1:] - arima_fitted[1:]
            arima_rmse = float(np.sqrt(np.mean(res ** 2)))
            mask = np.abs(values[1:]) > 1e-6
            arima_mape = float(np.mean(np.abs(res[mask] / values[1:][mask])) * 100) if mask.any() else float('nan')
            arima_aic = float(arima_fit.aic)

            fig.add_trace(go.Scatter(
                x=dates, y=arima_fitted.tolist(),
                mode='lines', name='ARIMA in-sample',
                line=dict(color='rgba(52,211,153,0.3)', width=1),
                showlegend=False, hoverinfo='skip',
            ))
            fig.add_trace(go.Scatter(
                x=list(future_dates), y=arima_fc.tolist(),
                mode='lines+markers', name='ARIMA(2,1,2)',
                line=dict(color='#34d399', width=2.5, dash='dot'),
                marker=dict(size=5, color='#34d399'),
                hovertemplate='ARIMA %{x|%Y-%m}: %{y:.3f} ' + unit + '<extra></extra>',
            ))
            metrics.append({
                'Model': 'ARIMA(2,1,2)',
                'RMSE': f'{arima_rmse:.3f}',
                'MAPE': f'{arima_mape:.1f}%' if not np.isnan(arima_mape) else 'n/a',
                'AIC': f'{arima_aic:.1f}',
            })
        except Exception:
            metrics.append({'Model': 'ARIMA(2,1,2)', 'RMSE': 'failed', 'MAPE': '—', 'AIC': '—'})

        # ── Model 3: Linear trend ─────────────────────────────────────────────
        try:
            t = np.arange(n, dtype=float)
            slope, intercept, r_val, _, _ = scipy.stats.linregress(t, values)
            lin_fitted = slope * t + intercept
            lin_fc = slope * np.arange(n, n + horizon, dtype=float) + intercept
            res = values - lin_fitted
            lin_rmse = float(np.sqrt(np.mean(res ** 2)))
            mask = np.abs(values) > 1e-6
            lin_mape = float(np.mean(np.abs(res[mask] / values[mask])) * 100) if mask.any() else float('nan')

            fig.add_trace(go.Scatter(
                x=dates, y=lin_fitted.tolist(),
                mode='lines', name='Linear in-sample',
                line=dict(color='rgba(248,113,113,0.3)', width=1),
                showlegend=False, hoverinfo='skip',
            ))
            fig.add_trace(go.Scatter(
                x=list(future_dates), y=lin_fc.tolist(),
                mode='lines+markers', name='Linear Trend',
                line=dict(color='#f87171', width=2.5, dash='dashdot'),
                marker=dict(size=5, color='#f87171'),
                hovertemplate='Linear %{x|%Y-%m}: %{y:.3f} ' + unit + '<extra></extra>',
            ))
            metrics.append({
                'Model': 'Linear Trend',
                'RMSE': f'{lin_rmse:.3f}',
                'MAPE': f'{lin_mape:.1f}%' if not np.isnan(lin_mape) else 'n/a',
                'AIC': 'n/a',
            })
        except Exception:
            metrics.append({'Model': 'Linear Trend', 'RMSE': 'failed', 'MAPE': '—', 'AIC': '—'})

        # ── Forecast-start vertical line ──────────────────────────────────────
        shapes = [dict(
            type='line', x0=last_date, x1=last_date,
            y0=0, y1=1, yref='paper',
            line=dict(color='rgba(148,163,184,0.4)', dash='dot', width=1.5),
        )]
        annotations = [dict(
            x=last_date, y=1.04, yref='paper',
            text='↑ Forecast', showarrow=False, xanchor='center',
            font=dict(color='rgba(148,163,184,0.6)', size=10),
        )]

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
            shapes=shapes, annotations=annotations,
            margin=dict(l=60, r=20, t=60, b=50),
        )

        # Best model by RMSE (exclude failed)
        valid = [(m['Model'], float(m['RMSE'])) for m in metrics
                 if m['RMSE'] not in ('failed', '—')]
        best_model = min(valid, key=lambda x: x[1])[0] if valid else 'n/a'

        return {
            'title': f'Model Comparison · {station_name}',
            'subtitle': (
                f'{feature_label} · {horizon}-month forecast · '
                f'{len(monthly)} months of history · best: {best_model}'
            ),
            'figure': plotly.io.to_json(fig),
            'stats': {
                'models': metrics,
                'best_model_by_rmse': best_model,
                'horizon_months': horizon,
                'n_months_historical': len(monthly),
            },
        }
