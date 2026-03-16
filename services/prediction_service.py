from __future__ import annotations

import os
from typing import Any, Dict
import warnings

import markdown
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .data_loader import DataRepository, SeriesRequest


class PredictionService:
    def __init__(self, repository: DataRepository) -> None:
        self.repository = repository

    def predict(self, station: str, feature: str, horizon: int) -> Dict[str, Any]:
        if horizon <= 0:
            raise ValueError('Prediction horizon must be greater than zero.')
        meta = self.repository.get_station_metadata(station)
        if feature not in meta.get('features', []):
            raise ValueError(f'{feature} is not available for {station}')

        detail = meta['feature_details'][feature]
        request = SeriesRequest(
            station=station,
            feature=feature,
            start_date=detail['start_date'],
            end_date=detail['end_date'],
        )
        series = self.repository.get_feature_series(request)
        frequency = self.repository.feature_frequency.get(feature, 'daily')
        model_frame = self._prepare_training_frame(series, frequency)
        model_frame = model_frame.dropna(subset=['Value'])
        if len(model_frame) < 10:
            raise ValueError('Not enough observations are available for prediction.')

        fitted, forecast_index, forecast_values, lower, upper, residual_std = self._forecast(model_frame, frequency, horizon)
        figure = self._build_figure(model_frame, forecast_index, forecast_values, lower, upper, station, feature)
        figure_zoom = self._build_zoom_figure(model_frame, forecast_index, forecast_values, lower, upper, station, feature, frequency)

        insight = self._prediction_summary(model_frame, forecast_values, residual_std, frequency, horizon, station, feature)
        return {
            'station': station,
            'feature': feature,
            'horizon': horizon,
            'frequency': frequency,
            'figure': plotly.io.to_json(figure),
            'figure_zoom': plotly.io.to_json(figure_zoom),
            'title': f'Prediction · {feature} · {station}',
            'summary': insight,
        }

    def _prepare_training_frame(self, series: pd.DataFrame, frequency: str) -> pd.DataFrame:
        frame = series[['Timestamp', 'Value']].copy().sort_values('Timestamp')
        frame = frame.set_index('Timestamp')
        if frequency == 'monthly':
            frame = frame.resample('MS').mean()
            frame = frame.tail(120)
        else:
            frame = frame.resample('D').mean().interpolate(limit_direction='both')
            frame = frame.tail(1500)
        return frame.reset_index()

    def _forecast(self, frame: pd.DataFrame, frequency: str, horizon: int):
        indexed = frame.set_index('Timestamp')['Value']
        seasonal_periods = None
        trend = 'add'
        seasonal = None
        damped = True
        if frequency == 'monthly' and len(indexed) >= 24:
            seasonal = 'add'
            seasonal_periods = 12
        elif frequency == 'daily' and len(indexed) >= 60:
            seasonal = 'add'
            seasonal_periods = 7

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model = ExponentialSmoothing(
                    indexed,
                    trend=trend,
                    damped_trend=damped,
                    seasonal=seasonal,
                    seasonal_periods=seasonal_periods,
                    initialization_method='estimated',
                )
                fit = model.fit(optimized=True, use_brute=True)
                forecast_values = fit.forecast(horizon)
            residuals = indexed - fit.fittedvalues.reindex(indexed.index)
            residual_std = float(np.nanstd(residuals)) if len(residuals) else 0.0
        except Exception:
            fit = None
            last_value = float(indexed.iloc[-1])
            forecast_index = self._future_index(indexed.index[-1], frequency, horizon)
            forecast_values = pd.Series([last_value] * horizon, index=forecast_index)
            residual_std = float(np.nanstd(indexed.values)) if len(indexed) else 0.0
            lower = forecast_values - 1.96 * residual_std
            upper = forecast_values + 1.96 * residual_std
            return fit, forecast_index, forecast_values, lower, upper, residual_std

        forecast_index = forecast_values.index
        step = np.sqrt(np.arange(1, horizon + 1))
        margin = 1.96 * residual_std * step
        lower = forecast_values.values - margin
        upper = forecast_values.values + margin
        lower = pd.Series(lower, index=forecast_index)
        upper = pd.Series(upper, index=forecast_index)
        return fit, forecast_index, forecast_values, lower, upper, residual_std

    def _future_index(self, last_timestamp: pd.Timestamp, frequency: str, horizon: int):
        if frequency == 'monthly':
            return pd.date_range(start=last_timestamp + pd.offsets.MonthBegin(1), periods=horizon, freq='MS')
        return pd.date_range(start=last_timestamp + pd.Timedelta(days=1), periods=horizon, freq='D')

    def _build_figure(
        self,
        frame: pd.DataFrame,
        forecast_index,
        forecast_values,
        lower,
        upper,
        station: str,
        feature: str,
    ) -> go.Figure:
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=frame['Timestamp'],
                y=frame['Value'],
                mode='lines',
                name='Historical',
                line={'width': 2.2, 'color': '#38bdf8'},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=forecast_index,
                y=upper,
                mode='lines',
                line={'width': 0},
                showlegend=False,
                hoverinfo='skip',
            )
        )
        figure.add_trace(
            go.Scatter(
                x=forecast_index,
                y=lower,
                mode='lines',
                line={'width': 0},
                fill='tonexty',
                fillcolor='rgba(245, 158, 11, 0.18)',
                name='Confidence band',
                hoverinfo='skip',
            )
        )
        figure.add_trace(
            go.Scatter(
                x=forecast_index,
                y=forecast_values,
                mode='lines+markers',
                name='Forecast',
                line={'width': 2.4, 'color': '#f59e0b', 'dash': 'dash'},
            )
        )
        figure.update_layout(
            template='plotly_white',
            title={'text': f'Prediction · {feature} · {station}', 'x': 0.5, 'xanchor': 'center'},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#ffffff',
            font={'family': 'Inter, Arial, sans-serif', 'size': 12, 'color': '#0f172a'},
            margin={'l': 46, 'r': 28, 't': 72, 'b': 48},
            hovermode='x unified',
            legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'left', 'x': 0},
        )
        figure.update_xaxes(showgrid=True, gridcolor='rgba(148,163,184,0.22)')
        figure.update_yaxes(showgrid=True, gridcolor='rgba(148,163,184,0.22)', title=f'{feature} ({self.repository.feature_units.get(feature, "")})')
        return figure

    def _build_zoom_figure(
        self,
        frame: pd.DataFrame,
        forecast_index,
        forecast_values,
        lower,
        upper,
        station: str,
        feature: str,
        frequency: str,
    ) -> go.Figure:
        # Slice to last 1 year of historical data
        cutoff = frame['Timestamp'].max() - (pd.DateOffset(months=12) if frequency == 'monthly' else pd.Timedelta(days=365))
        zoom_frame = frame[frame['Timestamp'] >= cutoff]

        # Bridge: prepend last historical point to forecast traces so lines connect with no gap
        last_hist_ts = zoom_frame['Timestamp'].iloc[-1]
        last_hist_val = float(zoom_frame['Value'].iloc[-1])
        bridge_x = [last_hist_ts] + list(forecast_index)
        bridge_forecast = [last_hist_val] + list(forecast_values.values)
        bridge_upper = [last_hist_val] + list(upper.values)
        bridge_lower = [last_hist_val] + list(lower.values)

        # Explicit x-axis range: start of zoom window to end of forecast + small padding
        if frequency == 'monthly':
            x_end = pd.Timestamp(forecast_index[-1]) + pd.DateOffset(months=1)
        else:
            x_end = pd.Timestamp(forecast_index[-1]) + pd.Timedelta(days=3)
        x_start = pd.Timestamp(zoom_frame['Timestamp'].iloc[0])

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=zoom_frame['Timestamp'],
                y=zoom_frame['Value'],
                mode='lines',
                name='Historical (last 1 year)',
                line={'width': 2.2, 'color': '#38bdf8'},
            )
        )
        # Confidence band using bridged coordinates
        figure.add_trace(
            go.Scatter(
                x=bridge_x,
                y=bridge_upper,
                mode='lines',
                line={'width': 0},
                showlegend=False,
                hoverinfo='skip',
            )
        )
        figure.add_trace(
            go.Scatter(
                x=bridge_x,
                y=bridge_lower,
                mode='lines',
                line={'width': 0},
                fill='tonexty',
                fillcolor='rgba(245, 158, 11, 0.18)',
                name='Confidence band',
                hoverinfo='skip',
            )
        )
        # Forecast line bridged from last historical point
        figure.add_trace(
            go.Scatter(
                x=bridge_x,
                y=bridge_forecast,
                mode='lines+markers',
                name='Forecast',
                line={'width': 2.4, 'color': '#f59e0b', 'dash': 'dash'},
                marker={'size': 6},
            )
        )
        figure.update_layout(
            template='plotly_white',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#ffffff',
            font={'family': 'Inter, Arial, sans-serif', 'size': 12, 'color': '#0f172a'},
            margin={'l': 46, 'r': 28, 't': 28, 'b': 48},
            hovermode='x unified',
            legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'left', 'x': 0},
            # Vertical line marking start of forecast
            shapes=[{
                'type': 'line',
                'x0': last_hist_ts, 'x1': last_hist_ts,
                'y0': 0, 'y1': 1,
                'yref': 'paper',
                'line': {'color': 'rgba(245,158,11,0.5)', 'width': 1.5, 'dash': 'dot'},
            }],
            annotations=[{
                'x': last_hist_ts,
                'y': 1,
                'yref': 'paper',
                'text': 'Forecast start',
                'showarrow': False,
                'font': {'size': 10, 'color': '#f59e0b'},
                'xanchor': 'left',
                'xshift': 4,
            }],
        )
        figure.update_xaxes(
            showgrid=True,
            gridcolor='rgba(148,163,184,0.22)',
            range=[x_start.strftime('%Y-%m-%d'), x_end.strftime('%Y-%m-%d')],
            autorange=False,
        )
        figure.update_yaxes(showgrid=True, gridcolor='rgba(148,163,184,0.22)', title=f'{feature} ({self.repository.feature_units.get(feature, "")})')
        return figure

    def _prediction_summary(
        self,
        frame: pd.DataFrame,
        forecast_values: pd.Series,
        residual_std: float,
        frequency: str,
        horizon: int,
        station: str,
        feature: str,
    ) -> str:
        last_actual = float(frame['Value'].iloc[-1])
        first_forecast = float(forecast_values.iloc[0])
        last_forecast = float(forecast_values.iloc[-1])
        forecast_min = float(forecast_values.min())
        forecast_max = float(forecast_values.max())
        direction = 'upward' if last_forecast > last_actual else 'downward' if last_forecast < last_actual else 'flat'
        unit = self.repository.feature_units.get(feature, '')
        period_label = 'month' if frequency == 'monthly' else 'day'
        pct_change = ((last_forecast - last_actual) / last_actual * 100) if last_actual != 0 else 0.0

        # Historical stats for context
        hist_mean = float(frame['Value'].mean())
        hist_std = float(frame['Value'].std())
        hist_min = float(frame['Value'].min())
        hist_max = float(frame['Value'].max())

        base_summary = (
            f'The forecast for {feature} at {station} spans the next {horizon} {period_label}(s). '
            f'The model projects a {direction} trajectory from the latest historical value of {last_actual:.2f} {unit} '
            f'toward approximately {last_forecast:.2f} {unit} by the end of the horizon ({pct_change:+.1f}%). '
            f'Forecast range: {forecast_min:.2f}–{forecast_max:.2f} {unit}. '
            f'Confidence band based on residual std {residual_std:.2f}.'
        )

        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key or api_key == 'your_google_gemini_api_key_here' or not api_key.strip():
            return base_summary + '\n\n*(Note: Set GEMINI_API_KEY in the .env file to enable AI-powered prediction analysis)*'

        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = (
                'Act as a professional hydrologist for the Mekong River Commission. '
                'You have just run a time-series forecast model (Holt-Winters Exponential Smoothing) on hydrological data. '
                'Analyse the prediction results below and provide a concise, insightful report.\n\n'
                'Structure your response exactly as follows:\n'
                '1. A brief **Executive Summary** paragraph (2-3 sentences) summarising the forecast outlook.\n'
                '2. A **Detailed Analysis** section with exactly 5 bullet points. Each bullet must start with a bold **Heading:** and cover:\n'
                '   - **Forecast Trajectory**: Direction, magnitude, and rate of change.\n'
                '   - **Comparison to Historical Baseline**: How the forecast compares to historical mean/range.\n'
                '   - **Uncertainty & Confidence**: Interpretation of the confidence band width and residual spread.\n'
                '   - **Seasonal Context**: Whether the forecast aligns with or deviates from expected seasonal patterns.\n'
                '   - **Operational Implications**: Practical meaning for water resource management or flood/drought risk.\n\n'
                'IMPORTANT: Use pretty names (e.g. "Ban Chot" not "Ban_Chot"). Be analytical and specific — cite the actual numbers provided.\n'
                'Do NOT include any introduction, sign-off, or markdown code blocks.\n\n'
                f'Prediction Data:\n'
                f'- Station: {station.replace("_", " ")}\n'
                f'- Feature: {feature.replace("_", " ")} (unit: {unit})\n'
                f'- Forecast horizon: {horizon} {period_label}(s)\n'
                f'- Last historical value: {last_actual:.2f} {unit}\n'
                f'- First forecast value: {first_forecast:.2f} {unit}\n'
                f'- Last forecast value: {last_forecast:.2f} {unit} ({pct_change:+.1f}% change)\n'
                f'- Forecast range: {forecast_min:.2f}–{forecast_max:.2f} {unit}\n'
                f'- Residual std (confidence spread): {residual_std:.2f} {unit}\n'
                f'- Historical mean: {hist_mean:.2f} {unit}\n'
                f'- Historical std: {hist_std:.2f} {unit}\n'
                f'- Historical min/max: {hist_min:.2f}/{hist_max:.2f} {unit}\n'
                f'- Overall forecast direction: {direction}\n'
            )
            response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            html_content = markdown.markdown(response.text.strip())
            return '🧠 Analysis:\n' + html_content
        except Exception as e:
            return base_summary + f'\n\n*(AI Analysis Failed: {str(e)})*'
