from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io
from plotly.subplots import make_subplots
from .data_loader import SeriesRequest


def _generate_scenario_analysis(scenario_result: Dict[str, Any]) -> str:
    """Generate AI analysis of scenario results using Gemini."""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return ''

    try:
        from google import genai
        client = genai.Client(api_key=api_key)

        station = scenario_result.get('station', '').replace('_', ' ')
        target = scenario_result.get('target_feature', '').replace('_', ' ')
        driver = scenario_result.get('driver_feature', '').replace('_', ' ')
        scale = scenario_result.get('scale_pct', 0)
        duration = scenario_result.get('duration_months', 0)
        stats = scenario_result.get('stats', {})
        sensitivity = scenario_result.get('sensitivity', {})

        prompt = f"""
Analyze this hydrological scenario simulation result and provide 2-3 bullet-point insights:

**Scenario Details:**
- Station: {station}
- Target variable: {target}
- Driver variable: {driver}
- Driver change: {scale:+.0f}%
- Application duration: {duration} month(s)

**Results:**
- Mean impact: {stats.get('mean_delta', 0):+.2f} units
- Peak impact: {stats.get('max_delta', 0):+.2f} units
- Average % change: {stats.get('mean_delta_pct', 0):+.1f}%
- Elasticity (sensitivity): {sensitivity.get('elasticity', 0):.2f}
- Dominant lag: {sensitivity.get('dominant_lag', 0)} month(s)
- Model type: {sensitivity.get('model_type', 'unknown')}

Provide brief, actionable insights about the scenario impact. Make clear this is a statistical response model, not a physics simulation. Format as bullet points.
"""

        response = client.models.generate_content(
            model='gemini-3.1-flash-lite-preview',
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        # Silently fail — analysis is optional
        return ''


class ScenarioService:
    """
    What-If scenario modelling.

    Workflow:
      1. Load historical target and driver series and resample to monthly means.
      2. Remove monthly climatology to form relative anomalies.
      3. Fit a distributed-lag response model with target persistence and
         driver lags across the previous 0..3 months.
      4. Load the baseline forecast from the prediction CSV files, or fall back
         to a simple monthly projection.
      5. Inject a driver shock during the selected months and propagate it
         through the lagged response model to obtain month-varying target impacts.
      6. Return both series + a Plotly figure.
    """

    # Map Mekong internal feature names → prediction folder names
    _MEKONG_FEAT_FOLDER = {
        'Discharge': 'Water_Discharge',
        'Water_Level': 'Water_Level',
        'Rainfall': 'Rainfall',
        'Total_Suspended_Solids': 'Total_Suspended_Solids',
    }
    _MAX_DRIVER_LAGS = 3

    def __init__(self, repository, data_dir: str | Path = 'data') -> None:
        self.repo = repository
        self.data_dir = Path(data_dir)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _find_repo(self, station: str):
        """Return the DataRepository that owns station, or None."""
        if hasattr(self.repo, 'repos'):
            return next((r for r in self.repo.repos if station in r.station_index), None)
        if station in getattr(self.repo, 'station_index', {}):
            return self.repo
        return None

    def _load_series(self, repo, station: str, feature: str) -> pd.Series:
        fd = repo.station_index[station]['feature_details'][feature]
        req = SeriesRequest(
            station=station, feature=feature,
            start_date=fd['start_date'], end_date=fd['end_date'],
        )
        df = repo.get_feature_series(req)
        ts = df.set_index('Timestamp')['Value'].sort_index()
        ts.index = pd.to_datetime(ts.index)
        return ts

    def _mekong_feat_folder(self, feature: str) -> str:
        return self._MEKONG_FEAT_FOLDER.get(feature, feature)

    def _load_csv_forecast(self, repo, station: str, feature: str, model: str, horizon: int) -> pd.Series | None:
        dataset = repo.dataset
        if dataset == 'mekong':
            path = (self.data_dir / 'Mekong' / 'prediction_results' / 'station_predictions_future'
                    / self._mekong_feat_folder(feature) / model / f'{station}.csv')
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
            vals = df.iloc[0, :horizon].values.astype(float)
            return pd.Series(vals, name='forecast')
        except Exception:
            return None

    def _simple_forecast(self, ts_monthly: pd.Series, horizon: int) -> pd.Series:
        """Flat mean of last 12 months as naive fallback."""
        baseline_val = float(ts_monthly.tail(12).mean())
        return pd.Series([baseline_val] * horizon, name='forecast')

    def _monthly_relative_anomaly(self, ts: pd.Series) -> pd.Series:
        """Relative anomaly versus monthly climatology."""
        ts = ts.dropna()
        if ts.empty:
            return pd.Series(dtype=float)
        df = ts.to_frame('value')
        df['month'] = df.index.month
        clim = df.groupby('month')['value'].mean()
        denom = df['month'].map(clim).astype(float)
        anomaly = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
        safe = denom.abs() > 1e-10
        anomaly.loc[safe] = (df.loc[safe, 'value'] - denom.loc[safe]) / denom.loc[safe].abs()
        return anomaly.rename('anomaly')

    def _compute_sensitivity(self, target_monthly: pd.Series, driver_monthly: pd.Series, direct: bool = False) -> Dict[str, Any]:
        """
        Fit a distributed-lag anomaly response model.
        """
        if direct:
            return {
                'beta': 1.0,
                'elasticity': 1.0,
                'r_value': 1.0,
                'p_value': 0.0,
                'n_months': 0,
                'direct': True,
                'model_type': 'direct_scaling',
                'driver_lag_coeffs': [1.0],
                'target_persistence': 0.0,
                'dominant_lag': 0,
                'cumulative_response': 1.0,
                'residual_std': 0.0,
                'fit_r2': 1.0,
                'used_fallback': False,
            }

        aligned = pd.concat([target_monthly.rename('target'), driver_monthly.rename('driver')], axis=1).dropna()
        if len(aligned) < 24:
            return {
                'beta': 0.0,
                'elasticity': 0.0,
                'r_value': 0.0,
                'p_value': 1.0,
                'n_months': len(aligned),
                'direct': False,
                'model_type': 'insufficient_data',
                'driver_lag_coeffs': [0.0] * (self._MAX_DRIVER_LAGS + 1),
                'target_persistence': 0.0,
                'dominant_lag': 0,
                'cumulative_response': 0.0,
                'residual_std': 0.0,
                'fit_r2': 0.0,
                'used_fallback': True,
            }

        target_anom = self._monthly_relative_anomaly(aligned['target'])
        driver_anom = self._monthly_relative_anomaly(aligned['driver'])
        model_df = pd.DataFrame({'target_anom': target_anom, 'driver_anom': driver_anom}).dropna()
        model_df['target_lag1'] = model_df['target_anom'].shift(1)
        for lag in range(self._MAX_DRIVER_LAGS + 1):
            model_df[f'driver_lag_{lag}'] = model_df['driver_anom'].shift(lag)
        model_df = model_df.dropna()

        if len(model_df) < 18:
            return {
                'beta': 0.0,
                'elasticity': 0.0,
                'r_value': 0.0,
                'p_value': 1.0,
                'n_months': len(model_df),
                'direct': False,
                'model_type': 'insufficient_data',
                'driver_lag_coeffs': [0.0] * (self._MAX_DRIVER_LAGS + 1),
                'target_persistence': 0.0,
                'dominant_lag': 0,
                'cumulative_response': 0.0,
                'residual_std': 0.0,
                'fit_r2': 0.0,
                'used_fallback': True,
            }

        feature_cols = ['target_lag1'] + [f'driver_lag_{lag}' for lag in range(self._MAX_DRIVER_LAGS + 1)]
        X = model_df[feature_cols].to_numpy(dtype=float)
        y = model_df['target_anom'].to_numpy(dtype=float)
        coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        phi = float(coefs[0])
        driver_lag_coeffs = [float(v) for v in coefs[1:]]
        fitted = X @ coefs
        residuals = y - fitted
        residual_std = float(np.nanstd(residuals)) if len(residuals) else 0.0
        fit_r2 = float(1 - np.sum(residuals ** 2) / np.sum((y - y.mean()) ** 2)) if len(y) and np.sum((y - y.mean()) ** 2) > 0 else 0.0
        fit_corr = float(np.corrcoef(y, fitted)[0, 1]) if len(y) > 1 else 0.0
        if np.isnan(fit_corr):
            fit_corr = 0.0
        cumulative_response = float(np.sum(driver_lag_coeffs))
        dominant_lag = int(np.argmax(np.abs(driver_lag_coeffs))) if driver_lag_coeffs else 0
        mean_driver = float(aligned['driver'].mean())
        mean_target = float(aligned['target'].mean())
        elasticity = cumulative_response * mean_driver / mean_target if mean_target != 0 else cumulative_response
        return {
            'beta': float(cumulative_response),
            'elasticity': float(elasticity),
            'r_value': fit_corr,
            'p_value': 1.0,
            'n_months': len(model_df),
            'direct': False,
            'model_type': 'distributed_lag_anomaly_response',
            'driver_lag_coeffs': driver_lag_coeffs,
            'target_persistence': phi,
            'dominant_lag': dominant_lag,
            'cumulative_response': cumulative_response,
            'residual_std': residual_std,
            'fit_r2': fit_r2,
            'used_fallback': abs(fit_corr) < 0.2,
        }

    def _simulate_scenario_response(
        self,
        baseline: pd.Series,
        scale_pct: float,
        duration_months: int,
        start_offset: int,
        sensitivity: Dict[str, Any],
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        n = len(baseline)
        driver_shock = np.zeros(n, dtype=float)
        window_end = min(start_offset + duration_months, n)
        driver_shock[start_offset:window_end] = scale_pct / 100.0

        response = np.zeros(n, dtype=float)
        if sensitivity.get('direct'):
            response[start_offset:window_end] = scale_pct / 100.0
        else:
            phi = float(sensitivity.get('target_persistence', 0.0))
            lag_coefs = list(sensitivity.get('driver_lag_coeffs', []))
            if not lag_coefs:
                lag_coefs = [0.0] * (self._MAX_DRIVER_LAGS + 1)
            for t in range(n):
                propagated = (phi * response[t - 1]) if t > 0 else 0.0
                for lag, coef in enumerate(lag_coefs):
                    if t - lag >= 0:
                        propagated += float(coef) * driver_shock[t - lag]
                response[t] = np.clip(propagated, -0.95, 3.0)

        scenario = (baseline * (1.0 + response)).clip(lower=0).rename('scenario')
        delta = (scenario - baseline).rename('delta')
        delta_pct = pd.Series(response * 100.0, index=baseline.index, name='delta_pct')
        return scenario, delta, delta_pct

    def _relationship_strong_enough(self, sensitivity: Dict[str, Any]) -> bool:
        if sensitivity.get('direct'):
            return True
        if sensitivity.get('model_type') != 'distributed_lag_anomaly_response':
            return False
        if sensitivity.get('n_months', 0) < 24:
            return False
        if abs(float(sensitivity.get('r_value', 0.0))) < 0.35:
            return False
        if float(sensitivity.get('fit_r2', 0.0)) < 0.12:
            return False
        if abs(float(sensitivity.get('cumulative_response', 0.0))) < 0.03:
            return False
        return True

    # ── main entry point ─────────────────────────────────────────────────────

    def run_scenario(
        self,
        station: str,
        target_feature: str,
        driver_feature: str,
        scale_pct: float,          # e.g. 20 = +20%, -30 = −30%
        duration_months: int,      # how many forecast months the scenario covers
        start_offset: int,         # 0 = first forecast month, 1 = second, etc.
        model: str = 'FlowNet',
        horizon: int = 12,
        include_analysis: bool = False,
    ) -> Dict[str, Any]:
        repo = self._find_repo(station)
        if repo is None:
            raise ValueError(f"Station '{station}' not found.")

        meta = repo.station_index[station]
        features = meta.get('features', [])
        if target_feature not in features:
            raise ValueError(f"'{target_feature}' not available for {station}.")

        unit = repo.feature_units.get(target_feature, '')
        driver_label = driver_feature if driver_feature in features else target_feature

        # ── 1. Load historical target series ─────────────────────────────────
        target_ts = self._load_series(repo, station, target_feature)
        target_monthly = target_ts.resample('MS').mean().dropna()

        # ── 2. Compute sensitivity ────────────────────────────────────────────
        if driver_feature != target_feature:
            if driver_feature not in features:
                raise ValueError(f"Driver feature '{driver_feature}' is not available for station {station}.")
            driver_ts = self._load_series(repo, station, driver_feature)
            driver_monthly = driver_ts.resample('MS').mean().dropna()
        else:
            # Direct scaling — driver IS the target
            driver_monthly = target_monthly.copy()

        is_direct = (driver_feature == target_feature)
        sensitivity = self._compute_sensitivity(target_monthly, driver_monthly, direct=is_direct)
        if not self._relationship_strong_enough(sensitivity):
            raise ValueError('Relationship too weak for a reliable scenario estimate.')

        # ── 3. Build baseline forecast ────────────────────────────────────────
        effective_horizon = max(horizon, start_offset + duration_months)
        csv_fc = self._load_csv_forecast(repo, station, target_feature, model, effective_horizon)

        last_date = target_ts.index[-1]
        freq = repo.feature_frequency.get(target_feature, 'daily')
        if freq == 'monthly':
            forecast_index = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=effective_horizon, freq='MS',
            )
        else:
            # Even for daily series work at monthly resolution for scenario
            forecast_index = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=effective_horizon, freq='MS',
            )

        if csv_fc is not None:
            n = min(len(csv_fc), effective_horizon)
            baseline_vals = csv_fc.values[:n].tolist()
            # Pad to effective_horizon with last value if needed
            while len(baseline_vals) < effective_horizon:
                baseline_vals.append(baseline_vals[-1])
        else:
            baseline_vals = self._simple_forecast(target_monthly, effective_horizon).tolist()

        baseline = pd.Series(baseline_vals[:effective_horizon], index=forecast_index[:effective_horizon], name='baseline')

        # ── 4. Apply scenario ─────────────────────────────────────────────────
        window_start = start_offset
        window_end = min(start_offset + duration_months, effective_horizon)
        scenario, delta, delta_pct = self._simulate_scenario_response(
            baseline, scale_pct, duration_months, start_offset, sensitivity
        )

        # ── 5. Build figure ───────────────────────────────────────────────────
        figure = self._build_figure(
            target_monthly, baseline, scenario, delta, delta_pct,
            station, target_feature, driver_label, scale_pct,
            duration_months, start_offset, unit, sensitivity,
        )

        # ── 6. Build summary stats ────────────────────────────────────────────
        window_baseline = baseline.iloc[window_start:window_end]
        window_scenario = scenario.iloc[window_start:window_end]
        mean_delta = float((window_scenario - window_baseline).mean())
        max_delta = float((window_scenario - window_baseline).abs().max())
        mean_delta_pct = float(delta_pct.iloc[window_start:window_end].mean()) if not delta_pct.iloc[window_start:window_end].isna().all() else 0.0

        baseline_source = 'trained_model_csv' if csv_fc is not None else 'statistical_mean_fallback'
        result = {
            'station': station,
            'target_feature': target_feature,
            'driver_feature': driver_feature,
            'scale_pct': scale_pct,
            'duration_months': duration_months,
            'start_offset': start_offset,
            'model': model,
            'unit': unit,
            'sensitivity': sensitivity,
            'baseline': [{'date': str(d.date()), 'value': round(float(v), 4)} for d, v in baseline.items()],
            'scenario': [{'date': str(d.date()), 'value': round(float(v), 4)} for d, v in scenario.items()],
            'delta_abs': [{'date': str(d.date()), 'value': round(float(v), 4)} for d, v in delta.items()],
            'delta_pct': [{'date': str(d.date()), 'pct': round(float(p), 2)} for d, p in delta_pct.items()],
            'model_note': (
                ('Driver shocks are propagated through a monthly distributed-lag anomaly response model.'
                 + (' Historical fit is weak, so interpret results cautiously.' if sensitivity.get('used_fallback') else ''))
                if not sensitivity.get('direct')
                else 'Direct scaling is applied because the driver and target are the same variable.'
            ),
            'stats': {
                'mean_delta': round(mean_delta, 3),
                'max_delta': round(max_delta, 3),
                'mean_delta_pct': round(mean_delta_pct, 2),
            },
            'figure': plotly.io.to_json(figure),
            'csv_used': csv_fc is not None,          # kept for backward compatibility
            'baseline_source': baseline_source,
        }

        # Generate analysis if requested
        if include_analysis:
            analysis = _generate_scenario_analysis(result)
            if analysis:
                result['analysis'] = analysis

        return result

    # ── figure builder ────────────────────────────────────────────────────────

    def _build_figure(
        self,
        history: pd.Series,
        baseline: pd.Series,
        scenario: pd.Series,
        delta: pd.Series,
        delta_pct: pd.Series,
        station: str, target_feat: str, driver_feat: str,
        scale_pct: float, duration_months: int, start_offset: int,
        unit: str, sensitivity: Dict[str, Any],
    ) -> go.Figure:
        sign = '+' if scale_pct >= 0 else ''
        driver_label = driver_feat.replace('_', ' ')
        target_label = target_feat.replace('_', ' ')

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.68, 0.32],
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=(
                f'{target_label} forecast: baseline versus scenario',
                f'Absolute change from baseline ({unit})' if unit else 'Absolute change from baseline',
            ),
        )

        GRID = 'rgba(148,163,184,0.16)'
        TEXT = '#334155'
        TITLE = '#0f172a'
        SOFT = '#94a3b8'
        BLUE = '#2563eb'
        ORANGE = '#ea580c'
        GREEN_POS = '#059669'
        RED_NEG = '#dc2626'
        SCENARIO_FILL = 'rgba(234,88,12,0.10)'

        # Baseline forecast
        fig.add_trace(go.Scatter(
            x=baseline.index, y=baseline.values,
            mode='lines', name='Baseline forecast',
            line=dict(color=BLUE, width=2.2, dash='dash'),
            hovertemplate='%{x|%b %Y}<br>%{y:.2f} ' + unit + '<extra>Baseline</extra>',
        ), row=1, col=1)

        # Scenario forecast
        fig.add_trace(go.Scatter(
            x=scenario.index, y=scenario.values,
            mode='lines+markers', name=f'Scenario ({sign}{scale_pct:.0f}% {driver_label})',
            line=dict(color=ORANGE, width=2.8),
            marker=dict(size=6, color=ORANGE),
            hovertemplate='%{x|%b %Y}<br>%{y:.2f} ' + unit + '<extra>Scenario</extra>',
        ), row=1, col=1)

        # Filled area between baseline and scenario
        fig.add_trace(go.Scatter(
            x=pd.concat([scenario, baseline[::-1]]).index.tolist(),
            y=pd.concat([scenario, baseline[::-1]]).values.tolist(),
            fill='toself', fillcolor=SCENARIO_FILL,
            line=dict(width=0), showlegend=False,
            hoverinfo='skip',
        ), row=1, col=1)

        # Shade scenario window
        window_dates = baseline.index[start_offset: start_offset + duration_months]
        if len(window_dates) > 0:
            fig.add_vrect(
                x0=window_dates[0], x1=window_dates[-1],
                fillcolor='rgba(234,88,12,0.06)',
                line_width=0,
                row=1, col=1,
            )
            fig.add_vrect(
                x0=window_dates[0], x1=window_dates[-1],
                fillcolor='rgba(234,88,12,0.06)',
                line_width=0,
                row=2, col=1,
            )
            fig.add_annotation(
                x=window_dates[0],
                y=1.02,
                yref='paper',
                text='Intervention window',
                showarrow=False,
                xanchor='left',
                font=dict(size=10, color='#9a3412'),
                bgcolor='rgba(255,255,255,0.82)',
                bordercolor='rgba(234,88,12,0.18)',
                borderwidth=1,
                borderpad=4,
            )

        # Absolute delta bars
        bar_colors = [GREEN_POS if v >= 0 else RED_NEG for v in delta.values]
        fig.add_trace(go.Bar(
            x=delta.index, y=delta.values,
            name='Scenario impact',
            marker_color=bar_colors,
            hovertemplate='%{x|%b %Y}<br>%{y:+.2f} ' + unit + '<extra>Impact</extra>',
        ), row=2, col=1)

        # Zero line for delta
        fig.add_hline(y=0, line_dash='solid', line_color=GRID, line_width=1, row=2, col=1)

        # Elasticity annotation
        mean_pct = float(delta_pct.iloc[start_offset: start_offset + duration_months].mean()) if len(delta_pct) else 0.0
        elast_txt = (
            f"Applied effect: about {mean_pct:+.1f}% during the intervention window  |  "
            f"dominant lag: {sensitivity.get('dominant_lag', 0)} month(s)  |  "
            f"fit R={sensitivity['r_value']:.2f}  |  "
            f"R²={sensitivity.get('fit_r2', 0.0):.2f}"
            if not sensitivity.get('direct')
            else f'Direct scaling applied: about {mean_pct:+.1f}% during the intervention window'
        )

        fig.update_layout(
            title=dict(
                text='Scenario Projection Overview',
                x=0.02,
                xanchor='left',
                font=dict(size=16, color=TITLE),
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(248,250,252,0.72)',
            font=dict(family='Inter, sans-serif', color=TEXT, size=12),
            legend=dict(
                orientation='h', yanchor='bottom', y=1.03,
                xanchor='left', x=0,
                font=dict(size=11),
                bgcolor='rgba(255,255,255,0.70)',
                bordercolor='rgba(148,163,184,0.18)',
                borderwidth=1,
            ),
            margin=dict(l=56, r=26, t=86, b=80),
            hovermode='x unified',
            hoverlabel=dict(bgcolor='#ffffff', bordercolor='rgba(148,163,184,0.28)', font=dict(color='#0f172a', size=12)),
            height=560,
            annotations=[
                dict(
                    text=elast_txt,
                    xref='paper', yref='paper',
                    x=0.02, y=-0.12,
                    xanchor='left', yanchor='top',
                    font=dict(size=10, color=TEXT),
                    showarrow=False,
                ),
            ],
        )
        axis_style = dict(
            gridcolor=GRID, zerolinecolor=GRID,
            linecolor=GRID, tickfont=dict(size=10, color=TEXT),
        )
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style)
        fig.update_yaxes(title_text=unit or target_label, row=1, col=1)
        fig.update_yaxes(title_text=unit or 'Change', row=2, col=1)
        for ann in fig['layout']['annotations'][:2]:
            ann['font'] = dict(size=12, color=TITLE)

        return fig
