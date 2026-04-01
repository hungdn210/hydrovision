from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io
from plotly.subplots import make_subplots
import scipy.stats

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

Provide brief, actionable insights about the scenario impact. Format as bullet points.
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
      1. Load full historical series for the target feature (e.g. Discharge) and
         the driver feature (e.g. Rainfall / Precipitation) at a station.
      2. Resample both to monthly means and compute a linear sensitivity
         coefficient:  beta = cov(target, driver) / var(driver)
         Elasticity (dimensionless) = beta * mean_driver / mean_target
      3. Load the pre-computed baseline forecast from the prediction CSV files.
         If none is found, fall back to a simple exponential-smoothing projection.
      4. Apply the scenario for `duration_months` starting at `start_offset`:
             scenario[t] = baseline[t] * (1 + elasticity * scale_pct / 100)
      5. Return both series + a Plotly figure (history + overlay + delta subplot).
    """

    # Map Mekong internal feature names → prediction folder names
    _MEKONG_FEAT_FOLDER = {
        'Discharge': 'Water_Discharge',
        'Water_Level': 'Water_Level',
        'Rainfall': 'Rainfall',
        'Total_Suspended_Solids': 'Total_Suspended_Solids',
    }

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

    def _compute_sensitivity(self, target_monthly: pd.Series, driver_monthly: pd.Series, direct: bool = False) -> Dict[str, Any]:
        """
        Compute linear sensitivity between driver and target.
        Returns dict with: beta, elasticity, r_value, p_value, n_months.
        """
        if direct:
            # Direct scaling — elasticity is 1 by definition
            return {'beta': 1.0, 'elasticity': 1.0, 'r_value': 1.0, 'p_value': 0.0, 'n_months': 0, 'direct': True}

        aligned = pd.concat([target_monthly.rename('target'), driver_monthly.rename('driver')], axis=1).dropna()
        if len(aligned) < 12:
            return {'beta': 0.0, 'elasticity': 0.0, 'r_value': 0.0, 'p_value': 1.0, 'n_months': len(aligned), 'direct': False}

        slope, intercept, r_value, p_value, _ = scipy.stats.linregress(aligned['driver'], aligned['target'])
        mean_driver = float(aligned['driver'].mean())
        mean_target = float(aligned['target'].mean())
        elasticity = slope * mean_driver / mean_target if mean_target != 0 else 0.0
        return {
            'beta': float(slope),
            'elasticity': float(elasticity),
            'r_value': float(r_value),
            'p_value': float(p_value),
            'n_months': len(aligned),
            'direct': False,
        }

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
        if driver_feature != target_feature and driver_feature in features:
            driver_ts = self._load_series(repo, station, driver_feature)
            driver_monthly = driver_ts.resample('MS').mean().dropna()
        else:
            # Direct scaling — driver IS the target
            driver_monthly = target_monthly.copy()
            driver_feature = target_feature  # normalise

        is_direct = (driver_feature == target_feature)
        sensitivity = self._compute_sensitivity(target_monthly, driver_monthly, direct=is_direct)
        elasticity = sensitivity['elasticity']

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
        scenario = baseline.copy().rename('scenario')
        window_start = start_offset
        window_end = min(start_offset + duration_months, effective_horizon)
        scale_factor = elasticity * (scale_pct / 100.0)
        scenario.iloc[window_start:window_end] = baseline.iloc[window_start:window_end] * (1.0 + scale_factor)

        delta = (scenario - baseline).rename('delta')
        delta_pct = ((delta / baseline.replace(0, np.nan)) * 100).rename('delta_pct')

        # ── 5. Build figure ───────────────────────────────────────────────────
        figure = self._build_figure(
            target_monthly, baseline, scenario, delta_pct,
            station, target_feature, driver_label, scale_pct,
            duration_months, start_offset, unit, sensitivity,
        )

        # ── 6. Build summary stats ────────────────────────────────────────────
        window_baseline = baseline.iloc[window_start:window_end]
        window_scenario = scenario.iloc[window_start:window_end]
        mean_delta = float((window_scenario - window_baseline).mean())
        max_delta = float((window_scenario - window_baseline).abs().max())
        mean_delta_pct = float(delta_pct.iloc[window_start:window_end].mean()) if not delta_pct.iloc[window_start:window_end].isna().all() else 0.0

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
            'delta': [{'date': str(d.date()), 'pct': round(float(p), 2)} for d, p in delta_pct.items()],
            'stats': {
                'mean_delta': round(mean_delta, 3),
                'max_delta': round(max_delta, 3),
                'mean_delta_pct': round(mean_delta_pct, 2),
            },
            'figure': plotly.io.to_json(figure),
            'csv_used': csv_fc is not None,
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
        delta_pct: pd.Series,
        station: str, target_feat: str, driver_feat: str,
        scale_pct: float, duration_months: int, start_offset: int,
        unit: str, sensitivity: Dict[str, Any],
    ) -> go.Figure:

        # Show last 24 months of history for context
        history_ctx = history.tail(24)
        sign = '+' if scale_pct >= 0 else ''
        driver_label = driver_feat.replace('_', ' ')
        target_label = target_feat.replace('_', ' ')

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.68, 0.32],
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=(
                f'{target_label} — Baseline vs. Scenario Forecast',
                f'Delta (%) from baseline',
            ),
        )

        DARK_BG = '#07111f'
        GRID = 'rgba(148,163,184,0.12)'
        TEXT = '#e5eefc'
        SOFT = '#9db0d1'
        BLUE = '#38bdf8'
        ORANGE = '#fb923c'
        GREEN_POS = '#34d399'
        RED_NEG = '#f87171'
        SCENARIO_FILL = 'rgba(251,146,60,0.08)'

        # History
        fig.add_trace(go.Scatter(
            x=history_ctx.index, y=history_ctx.values,
            mode='lines', name='Historical',
            line=dict(color=SOFT, width=1.5),
            hovertemplate='%{x|%b %Y}<br>%{y:.2f} ' + unit + '<extra>Historical</extra>',
        ), row=1, col=1)

        # Baseline forecast
        fig.add_trace(go.Scatter(
            x=baseline.index, y=baseline.values,
            mode='lines', name='Baseline forecast',
            line=dict(color=BLUE, width=2, dash='dot'),
            hovertemplate='%{x|%b %Y}<br>%{y:.2f} ' + unit + '<extra>Baseline</extra>',
        ), row=1, col=1)

        # Scenario forecast
        fig.add_trace(go.Scatter(
            x=scenario.index, y=scenario.values,
            mode='lines', name=f'Scenario ({sign}{scale_pct:.0f}% {driver_label})',
            line=dict(color=ORANGE, width=2.5),
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
                fillcolor='rgba(251,146,60,0.06)',
                line_width=0,
                row=1, col=1,
            )
            fig.add_vrect(
                x0=window_dates[0], x1=window_dates[-1],
                fillcolor='rgba(251,146,60,0.06)',
                line_width=0,
                row=2, col=1,
            )

        # Delta % bars
        bar_colors = [GREEN_POS if v >= 0 else RED_NEG for v in delta_pct.values]
        fig.add_trace(go.Bar(
            x=delta_pct.index, y=delta_pct.values,
            name='Delta (%)',
            marker_color=bar_colors,
            hovertemplate='%{x|%b %Y}<br>%{y:+.1f}%<extra>Δ%</extra>',
        ), row=2, col=1)

        # Zero line for delta
        fig.add_hline(y=0, line_dash='solid', line_color=GRID, line_width=1, row=2, col=1)

        # Elasticity annotation
        elast_txt = (
            f"Elasticity: {sensitivity['elasticity']:.2f}  |  "
            f"R={sensitivity['r_value']:.2f}  |  "
            f"n={sensitivity['n_months']} months"
            if not sensitivity.get('direct')
            else 'Direct scaling (driver = target)'
        )

        fig.update_layout(
            paper_bgcolor=DARK_BG,
            plot_bgcolor=DARK_BG,
            font=dict(family='Inter, sans-serif', color=TEXT, size=12),
            legend=dict(
                orientation='h', yanchor='bottom', y=1.02,
                xanchor='left', x=0,
                font=dict(size=11), bgcolor='rgba(0,0,0,0)',
            ),
            margin=dict(l=10, r=20, t=60, b=80),
            hovermode='x unified',
            annotations=[
                dict(
                    text=elast_txt,
                    xref='paper', yref='paper',
                    x=0.98, y=-0.10,
                    xanchor='right', yanchor='top',
                    font=dict(size=10, color=SOFT),
                    showarrow=False,
                ),
            ],
        )
        axis_style = dict(
            gridcolor=GRID, zerolinecolor=GRID,
            linecolor=GRID, tickfont=dict(size=10, color=SOFT),
        )
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style)
        fig.update_yaxes(title_text=unit or target_label, row=1, col=1)
        fig.update_yaxes(title_text='Δ%', row=2, col=1)
        for ann in fig['layout']['annotations'][:2]:
            ann['font'] = dict(size=12, color=SOFT)

        return fig
