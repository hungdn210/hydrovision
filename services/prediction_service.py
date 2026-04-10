from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
import warnings

import markdown
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import acf, pacf

from .analysis_service import _gemini_generate
from .data_loader import DataRepository, SeriesRequest

MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


class PredictionService:
    def __init__(self, repository: DataRepository, data_dir: str | Path = 'data') -> None:
        self.repository = repository
        self.data_dir = Path(data_dir)

    def _get_dataset(self, station: str) -> str | None:
        dataset = getattr(self.repository, 'dataset', None)
        if dataset is None and hasattr(self.repository, '_station_to_repo'):
            repo = self.repository._station_to_repo.get(station)
            dataset = repo.dataset if repo else None
        return dataset

    def _mekong_feature_folder(self, feature: str) -> str:
        return {'Discharge': 'Water_Discharge', 'Water_Level': 'Water_Level',
                'Rainfall': 'Rainfall', 'Total_Suspended_Solids': 'Total_Suspended_Solids'}.get(feature, feature)

    def _load_csv_predictions(self, station: str, feature: str, model: str, horizon: int) -> pd.Series | None:
        """Load future forecast (station_predictions_future). Returns Series of horizon values or None."""
        dataset = self._get_dataset(station)
        if dataset == 'mekong':
            csv_path = (self.data_dir / 'Mekong' / 'prediction_results' / 'station_predictions_future'
                        / self._mekong_feature_folder(feature) / model / f'{station}.csv')
        elif dataset == 'lamah':
            csv_path = (self.data_dir / 'LamaH' / 'prediction_results' / 'station_predictions_future'
                        / model / f'{station}.csv')
            # FlowNet has an extra LamaH_daily subfolder
            if not csv_path.exists():
                csv_path = (self.data_dir / 'LamaH' / 'prediction_results' / 'station_predictions_future'
                            / model / 'LamaH_daily' / f'{station}.csv')
        else:
            return None
        if not csv_path.exists():
            return None
        try:
            df = pd.read_csv(csv_path, nrows=1)
            values = df.iloc[0, :horizon].values
            return pd.Series(values, name='forecast')
        except Exception:
            return None

    def _load_historical_fit(
        self, station: str, feature: str, model: str, horizon_h: int, actual_frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, int, float, float] | None:
        """
        Load rolling predictions from station_predictions and stitch horizon_h across all windows.
        Returns (fit_df, actual_h, rmse, mape) where fit_df has columns [Timestamp, ModelFit].
        Date formula: date(window_i, horizon_k) = eval_start + (i-1) + (k-1)
                      eval_start = last_actual_date - (n_windows + n_horizons - 2)
        """
        dataset = self._get_dataset(station)
        if dataset == 'mekong':
            csv_path = (self.data_dir / 'Mekong' / 'prediction_results' / 'station_predictions'
                        / self._mekong_feature_folder(feature) / model / f'{station}.csv')
        elif dataset == 'lamah':
            csv_path = (self.data_dir / 'LamaH' / 'prediction_results' / 'station_predictions'
                        / model / f'{station}.csv')
        else:
            return None
        if not csv_path.exists():
            return None
        try:
            pred_df = pd.read_csv(csv_path)
            n_windows, n_horizons = len(pred_df), len(pred_df.columns)
            actual_h = min(horizon_h, n_horizons)   # clamp to available horizons
            col_idx = actual_h - 1

            last_date = actual_frame['Timestamp'].max()
            eval_start = last_date - pd.Timedelta(days=n_windows + n_horizons - 2)

            # Dates for window 1..n_windows using horizon actual_h
            dates = pd.date_range(
                start=eval_start + pd.Timedelta(days=actual_h - 1),
                periods=n_windows, freq='D',
            )
            values = pred_df.iloc[:, col_idx].values
            fit_df = pd.DataFrame({'Timestamp': dates, 'ModelFit': values.astype(float)})

            # Align with actual data and compute metrics
            actual_indexed = actual_frame.set_index('Timestamp')['Value']
            fit_df = fit_df[fit_df['Timestamp'].isin(actual_indexed.index)].reset_index(drop=True)
            if fit_df.empty:
                return None

            actuals = actual_indexed.reindex(fit_df['Timestamp']).values
            preds = fit_df['ModelFit'].values
            residuals = actuals - preds
            rmse = float(np.sqrt(np.nanmean(residuals ** 2)))
            # MAPE: only use rows where actual is meaningfully non-zero (> 1% of mean)
            act_mean = float(np.nanmean(np.abs(actuals)))
            threshold = max(act_mean * 0.01, 1e-6)
            valid = np.abs(actuals) > threshold
            if valid.sum() > 0:
                mape = float(np.nanmean(np.abs(residuals[valid] / actuals[valid])) * 100)
            else:
                mape = float('nan')

            return fit_df, actual_h, rmse, mape
        except Exception:
            return None

    def predict(self, station: str, feature: str, horizon: int, model: str = 'FlowNet', mode: str = 'future', analysis: bool = True) -> Dict[str, Any]:
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

        # Historical fit CSVs have at most 30 horizon columns — cap for that loader only
        hist_horizon = min(horizon, 30)

        # Always load the historical rolling fit (needed for metrics in both modes)
        hist_result = self._load_historical_fit(station, feature, model, hist_horizon, model_frame)
        hist_fit_df, actual_h, rmse, mape = hist_result if hist_result else (None, hist_horizon, float('nan'), float('nan'))

        unit = self.repository.feature_units.get(feature, '')

        # ── HISTORICAL MODE ───────────────────────────────────────────────────
        if mode == 'historical':
            if hist_fit_df is None or hist_fit_df.empty:
                raise ValueError(f'No historical predictions found for {station} / {feature} / {model}.')
            figure = self._build_historical_figure(model_frame, hist_fit_df, actual_h, station, feature)
            figure_zoom = self._build_historical_zoom_figure(model_frame, hist_fit_df, actual_h, feature)
            diag_result = self._build_diagnostics_figure(
                model_frame, feature, frequency, hist_fit_df=hist_fit_df, source_label=model
            )
            insight = self._historical_summary(model_frame, hist_fit_df, rmse, mape, actual_h, station, feature) if analysis else None
            return {
                'station': station, 'feature': feature, 'horizon': actual_h,
                'frequency': frequency,
                'source_type': 'trained_model',
                'figure': plotly.io.to_json(figure),
                'figure_zoom': plotly.io.to_json(figure_zoom),
                'figure_diagnostics': plotly.io.to_json(diag_result['figure']) if diag_result else None,
                'diagnostics_summary': diag_result['summary'] if diag_result else None,
                'title': f'Historical Fit · {feature} · {station}',
                'summary': insight,
                'model_metrics': {
                    'rmse': round(rmse, 3) if not np.isnan(rmse) else None,
                    'mape': round(mape, 1) if (not np.isnan(mape) and mape <= 500) else None,
                },
            }

        # ── FUTURE MODE ───────────────────────────────────────────────────────
        effective_horizon = min(horizon, 30)
        csv_forecast = self._load_csv_predictions(station, feature, model, effective_horizon)

        if csv_forecast is not None:
            source_type = 'trained_model'
            last_timestamp = model_frame['Timestamp'].iloc[-1]
            forecast_index = self._future_index(last_timestamp, frequency, len(csv_forecast))
            forecast_values = pd.Series(csv_forecast.values.astype(float), index=forecast_index)
            residual_std = rmse if not np.isnan(rmse) else float(np.nanstd(model_frame['Value'].values))
            # Compute CI band from residual_std
            step = np.sqrt(np.arange(1, len(forecast_values) + 1))
            margin = 1.96 * residual_std * step
            non_negative = float(model_frame['Value'].min()) >= 0
            lower = pd.Series(forecast_values.values - margin, index=forecast_index)
            upper = pd.Series(forecast_values.values + margin, index=forecast_index)
            if non_negative:
                lower = lower.clip(lower=0)
        else:
            source_type = 'statistical_fallback'
            _, forecast_index, forecast_values, lower, upper, residual_std, rmse, mape = self._forecast(model_frame, frequency, effective_horizon)
            hist_fit_df = None

        figure = self._build_figure(model_frame, forecast_index, forecast_values, feature, actual_h, lower=lower, upper=upper)
        figure_zoom = self._build_zoom_figure(model_frame, forecast_index, forecast_values, feature, frequency, actual_h, lower=lower, upper=upper)
        diag_result = self._build_diagnostics_figure(
            model_frame, feature, frequency, hist_fit_df=hist_fit_df, source_label=model if hist_fit_df is not None else None
        )

        if analysis:
            fc = model_frame.copy()
            fc['Month'] = pd.to_datetime(fc['Timestamp']).dt.month
            monthly_normals = fc.groupby('Month')['Value'].mean().to_dict()
            forecast_df = pd.DataFrame({'value': forecast_values.values}, index=forecast_index)
            forecast_df['month'] = forecast_df.index.month
            forecast_by_month = forecast_df.groupby('month')['value'].mean()
            clim_lines = []
            for m in sorted(forecast_by_month.index):
                norm = monthly_normals.get(m)
                fcast = float(forecast_by_month[m])
                if norm and norm != 0:
                    pct = (fcast - norm) / abs(norm) * 100
                    direction = 'above' if pct >= 0 else 'below'
                    clim_lines.append(
                        f"  {MONTH_NAMES[m - 1]}: forecast {fcast:.2f} {unit} vs normal {norm:.2f} {unit} "
                        f"({abs(pct):.1f}% {direction} climatological normal)"
                    )
            climatological_context = '\n'.join(clim_lines) if clim_lines else 'Insufficient history for monthly normals.'
            insight = self._prediction_summary(
                model_frame, forecast_values, residual_std, rmse, mape,
                climatological_context, frequency, horizon, station, feature,
            )
        else:
            insight = None
        return {
            'station': station, 'feature': feature, 'horizon': horizon,
            'frequency': frequency,
            'source_type': source_type,
            'figure': plotly.io.to_json(figure),
            'figure_zoom': plotly.io.to_json(figure_zoom),
            'figure_diagnostics': plotly.io.to_json(diag_result['figure']) if diag_result else None,
            'diagnostics_summary': diag_result['summary'] if diag_result else None,
            'title': f'Future Forecast · {feature} · {station}',
            'summary': insight,
            'model_metrics': {
                'rmse': round(rmse, 3) if not np.isnan(rmse) else None,
                'mape': round(mape, 1) if (not np.isnan(mape) and mape <= 500) else None,
            },
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

        non_negative = float(indexed.min()) >= 0

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
            rmse = float(np.sqrt(np.nanmean(residuals ** 2)))
            nonzero = indexed[indexed != 0]
            if len(nonzero):
                mape = float(np.nanmean(np.abs((nonzero - fit.fittedvalues.reindex(nonzero.index)) / nonzero)) * 100)
            else:
                mape = float('nan')
        except Exception:
            fit = None
            last_value = float(indexed.iloc[-1])
            forecast_index = self._future_index(indexed.index[-1], frequency, horizon)
            forecast_values = pd.Series([last_value] * horizon, index=forecast_index)
            residual_std = float(np.nanstd(indexed.values)) if len(indexed) else 0.0
            rmse = residual_std
            mape = float('nan')
            lower = forecast_values - 1.96 * residual_std
            upper = forecast_values + 1.96 * residual_std
            if non_negative:
                lower = lower.clip(lower=0)
            return fit, forecast_index, forecast_values, lower, upper, residual_std, rmse, mape

        forecast_index = forecast_values.index
        step = np.sqrt(np.arange(1, horizon + 1))
        margin = 1.96 * residual_std * step
        lower = pd.Series(forecast_values.values - margin, index=forecast_index)
        upper = pd.Series(forecast_values.values + margin, index=forecast_index)
        if non_negative:
            lower = lower.clip(lower=0)
        return fit, forecast_index, forecast_values, lower, upper, residual_std, rmse, mape

    def _future_index(self, last_timestamp: pd.Timestamp, frequency: str, horizon: int):
        if frequency == 'monthly':
            return pd.date_range(start=last_timestamp + pd.offsets.MonthBegin(1), periods=horizon, freq='MS')
        return pd.date_range(start=last_timestamp + pd.Timedelta(days=1), periods=horizon, freq='D')

    def _build_figure(
        self,
        frame: pd.DataFrame,
        forecast_index,
        forecast_values,
        feature: str,
        actual_h: int = 1,
        lower=None,
        upper=None,
    ) -> go.Figure:
        unit = self.repository.feature_units.get(feature, '')
        last_hist_ts = frame['Timestamp'].iloc[-1]
        last_hist_val = float(frame['Value'].iloc[-1])
        bridge_x = [last_hist_ts] + list(forecast_index)
        bridge_forecast = [last_hist_val] + list(forecast_values.values)
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=frame['Timestamp'], y=frame['Value'],
            mode='lines', name='Actual',
            line={'width': 1.8, 'color': '#38bdf8'},
            fill='tozeroy', fillcolor='rgba(56,189,248,0.07)',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Actual: %{y:.3f} ' + unit + '<extra></extra>',
        ))
        figure.add_trace(go.Scatter(
            x=bridge_x, y=bridge_forecast,
            mode='lines', name=f'Forecast (H={actual_h})',
            line={'width': 2.6, 'color': '#f59e0b', 'dash': 'dash'},
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Forecast: %{y:.3f} ' + unit + '<extra></extra>',
        ))
        # CI band (trace index 2) — toggled by the CI toggle in the UI
        if lower is not None and upper is not None:
            bridge_upper = [last_hist_val] + list(upper.values)
            bridge_lower = [last_hist_val] + list(lower.values)
            x_band = bridge_x + bridge_x[::-1]
            y_band = bridge_upper + bridge_lower[::-1]
            figure.add_trace(go.Scatter(
                x=x_band, y=y_band,
                fill='toself', fillcolor='rgba(245,158,11,0.13)',
                line={'color': 'rgba(245,158,11,0.3)', 'width': 0.8},
                mode='lines', name='95% CI',
                hoverinfo='skip',
            ))
        figure.update_layout(
            template='plotly_white',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(248,250,252,0.6)',
            font={'family': 'Inter, Arial, sans-serif', 'size': 12, 'color': '#334155'},
            margin={'l': 50, 'r': 28, 't': 48, 'b': 48},
            hovermode='x unified',
            hoverlabel={'bgcolor': '#1e293b', 'bordercolor': '#334155', 'font': {'color': '#f1f5f9', 'size': 12}},
            legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'left', 'x': 0,
                    'font': {'size': 11}, 'bgcolor': 'rgba(0,0,0,0)'},
        )
        figure.update_xaxes(showgrid=True, gridcolor='rgba(148,163,184,0.18)', tickfont={'size': 11, 'color': '#64748b'})
        figure.update_yaxes(showgrid=True, gridcolor='rgba(148,163,184,0.18)',
                            title=f'{feature.replace("_"," ")} ({unit})',
                            tickfont={'size': 11, 'color': '#64748b'})
        return figure

    def _build_zoom_figure(
        self,
        frame: pd.DataFrame,
        forecast_index,
        forecast_values,
        feature: str,
        frequency: str,
        actual_h: int = 1,
        lower=None,
        upper=None,
    ) -> go.Figure:
        unit = self.repository.feature_units.get(feature, '')

        cutoff = frame['Timestamp'].max() - (pd.DateOffset(months=6) if frequency == 'monthly' else pd.Timedelta(days=90))
        zoom_frame = frame[frame['Timestamp'] >= cutoff]

        last_hist_ts = zoom_frame['Timestamp'].iloc[-1]
        last_hist_val = float(zoom_frame['Value'].iloc[-1])
        bridge_x = [last_hist_ts] + list(forecast_index)
        bridge_forecast = [last_hist_val] + list(forecast_values.values)

        if frequency == 'monthly':
            x_end = pd.Timestamp(forecast_index[-1]) + pd.DateOffset(months=1)
        else:
            x_end = pd.Timestamp(forecast_index[-1]) + pd.Timedelta(days=3)
        x_start = pd.Timestamp(zoom_frame['Timestamp'].iloc[0])

        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=zoom_frame['Timestamp'], y=zoom_frame['Value'],
            mode='lines', name='Actual (last 3 months)',
            line={'width': 2, 'color': '#38bdf8'},
            fill='tozeroy', fillcolor='rgba(56,189,248,0.09)',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Actual: %{y:.3f} ' + unit + '<extra></extra>',
        ))
        figure.add_trace(go.Scatter(
            x=bridge_x, y=bridge_forecast,
            mode='lines+markers', name=f'Forecast (H={actual_h})',
            line={'width': 2.2, 'color': '#f59e0b', 'dash': 'dash'},
            marker={'size': 8, 'color': '#f59e0b', 'line': {'width': 1.5, 'color': 'white'}},
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Forecast: %{y:.3f} ' + unit + '<extra></extra>',
        ))
        # CI band (trace index 2) — toggled by the CI toggle in the UI
        if lower is not None and upper is not None:
            bridge_upper = [last_hist_val] + list(upper.values)
            bridge_lower = [last_hist_val] + list(lower.values)
            x_band = bridge_x + bridge_x[::-1]
            y_band = bridge_upper + bridge_lower[::-1]
            figure.add_trace(go.Scatter(
                x=x_band, y=y_band,
                fill='toself', fillcolor='rgba(245,158,11,0.13)',
                line={'color': 'rgba(245,158,11,0.3)', 'width': 0.8},
                mode='lines', name='95% CI',
                hoverinfo='skip',
            ))
        figure.update_layout(
            template='plotly_white',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(248,250,252,0.6)',
            font={'family': 'Inter, Arial, sans-serif', 'size': 12, 'color': '#334155'},
            margin={'l': 50, 'r': 28, 't': 28, 'b': 48},
            hovermode='x unified',
            hoverlabel={'bgcolor': '#1e293b', 'bordercolor': '#334155', 'font': {'color': '#f1f5f9', 'size': 12}},
            legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'left', 'x': 0,
                    'font': {'size': 11}, 'bgcolor': 'rgba(0,0,0,0)'},
            shapes=[{
                'type': 'line', 'x0': last_hist_ts, 'x1': last_hist_ts,
                'y0': 0, 'y1': 1, 'yref': 'paper',
                'line': {'color': 'rgba(245,158,11,0.6)', 'width': 1.5, 'dash': 'dot'},
            }],
            annotations=[{
                'x': last_hist_ts, 'y': 1, 'yref': 'paper',
                'text': 'Forecast start', 'showarrow': False,
                'font': {'size': 10, 'color': '#f59e0b'},
                'xanchor': 'left', 'xshift': 5,
            }],
        )
        figure.update_xaxes(
            showgrid=True, gridcolor='rgba(148,163,184,0.18)',
            range=[x_start.strftime('%Y-%m-%d'), x_end.strftime('%Y-%m-%d')],
            autorange=False, tickfont={'size': 11, 'color': '#64748b'},
        )
        figure.update_yaxes(showgrid=True, gridcolor='rgba(148,163,184,0.18)',
                            title=f'{feature.replace("_"," ")} ({unit})',
                            tickfont={'size': 11, 'color': '#64748b'})
        return figure

    def _build_historical_figure(
        self, frame: pd.DataFrame, hist_fit_df: pd.DataFrame,
        actual_h: int, station: str, feature: str,
    ) -> go.Figure:
        unit = self.repository.feature_units.get(feature, '')
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frame['Timestamp'], y=frame['Value'],
            mode='lines', name='Actual',
            line={'width': 1.8, 'color': '#38bdf8'},
            fill='tozeroy', fillcolor='rgba(56,189,248,0.07)',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Actual: %{y:.3f} ' + unit + '<extra></extra>',
        ))
        fig.add_trace(go.Scatter(
            x=hist_fit_df['Timestamp'], y=hist_fit_df['ModelFit'],
            mode='lines', name=f'Model fit (H={actual_h})',
            line={'width': 1.8, 'color': '#f472b6'},
            opacity=0.85,
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Model fit: %{y:.3f} ' + unit + '<extra></extra>',
        ))
        fig.update_layout(
            template='plotly_white',
            title={'text': f'Historical Fit · {feature.replace("_"," ")} · {station.replace("_"," ")}',
                   'x': 0.5, 'xanchor': 'center', 'y': 0.97, 'yanchor': 'top',
                   'font': {'size': 14, 'color': '#0f172a'}},
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(248,250,252,0.6)',
            font={'family': 'Inter, Arial, sans-serif', 'size': 12, 'color': '#334155'},
            margin={'l': 50, 'r': 28, 't': 96, 'b': 48},
            hovermode='x unified',
            hoverlabel={'bgcolor': '#1e293b', 'bordercolor': '#334155', 'font': {'color': '#f1f5f9', 'size': 12}},
            legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'left', 'x': 0,
                    'font': {'size': 11}, 'bgcolor': 'rgba(0,0,0,0)'},
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148,163,184,0.18)', tickfont={'size': 11, 'color': '#64748b'})
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148,163,184,0.18)',
                         title=f'{feature.replace("_"," ")} ({unit})',
                         tickfont={'size': 11, 'color': '#64748b'})
        return fig

    def _build_historical_zoom_figure(
        self, frame: pd.DataFrame, hist_fit_df: pd.DataFrame,
        actual_h: int, feature: str,
    ) -> go.Figure:
        unit = self.repository.feature_units.get(feature, '')
        # Zoom to the window covered by hist_fit_df
        x_start = hist_fit_df['Timestamp'].min()
        zoom_frame = frame[frame['Timestamp'] >= x_start]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=zoom_frame['Timestamp'], y=zoom_frame['Value'],
            mode='lines', name='Actual',
            line={'width': 2, 'color': '#38bdf8'},
            fill='tozeroy', fillcolor='rgba(56,189,248,0.09)',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Actual: %{y:.3f} ' + unit + '<extra></extra>',
        ))
        fig.add_trace(go.Scatter(
            x=hist_fit_df['Timestamp'], y=hist_fit_df['ModelFit'],
            mode='lines', name=f'Model fit (H={actual_h})',
            line={'width': 1.8, 'color': '#f472b6'},
            opacity=0.85,
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Model fit: %{y:.3f} ' + unit + '<extra></extra>',
        ))
        fig.update_layout(
            template='plotly_white',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(248,250,252,0.6)',
            font={'family': 'Inter, Arial, sans-serif', 'size': 12, 'color': '#334155'},
            margin={'l': 50, 'r': 28, 't': 28, 'b': 48},
            hovermode='x unified',
            hoverlabel={'bgcolor': '#1e293b', 'bordercolor': '#334155', 'font': {'color': '#f1f5f9', 'size': 12}},
            legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'left', 'x': 0,
                    'font': {'size': 11}, 'bgcolor': 'rgba(0,0,0,0)'},
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148,163,184,0.18)',
                         range=[x_start.strftime('%Y-%m-%d'),
                                hist_fit_df['Timestamp'].max().strftime('%Y-%m-%d')],
                         autorange=False, tickfont={'size': 11, 'color': '#64748b'})
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148,163,184,0.18)',
                         title=f'{feature.replace("_"," ")} ({unit})',
                         tickfont={'size': 11, 'color': '#64748b'})
        return fig

    def _build_diagnostics_figure(
        self,
        frame: pd.DataFrame,
        feature: str,
        frequency: str,
        hist_fit_df: pd.DataFrame | None = None,
        source_label: str | None = None,
    ):
        """
        Residual diagnostics for the trained-model historical fit when available,
        otherwise for the Holt-Winters in-sample fit.
        Returns {'figure': Plotly Figure, 'summary': str} or None if fitting fails.
        """
        indexed = frame.set_index('Timestamp')['Value'].dropna()
        if len(indexed) < 20:
            return None

        residuals = None
        diagnostics_source = f'{source_label} historical fit' if source_label and hist_fit_df is not None else 'Holt-Winters baseline'

        if hist_fit_df is not None and not hist_fit_df.empty:
            actual_on_fit = indexed.reindex(pd.to_datetime(hist_fit_df['Timestamp']))
            model_fit = pd.Series(
                hist_fit_df['ModelFit'].to_numpy(dtype=float),
                index=pd.to_datetime(hist_fit_df['Timestamp']),
            )
            residuals = (actual_on_fit - model_fit).dropna()

        if residuals is None or len(residuals) < 10:
            seasonal_periods = None
            seasonal = None
            if frequency == 'monthly' and len(indexed) >= 24:
                seasonal = 'add'
                seasonal_periods = 12
            elif frequency == 'daily' and len(indexed) >= 60:
                seasonal = 'add'
                seasonal_periods = 7

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    fit = ExponentialSmoothing(
                        indexed,
                        trend='add', damped_trend=True,
                        seasonal=seasonal,
                        seasonal_periods=seasonal_periods,
                        initialization_method='estimated',
                    ).fit(optimized=True)
                residuals = (indexed - fit.fittedvalues.reindex(indexed.index)).dropna()
                diagnostics_source = 'Holt-Winters baseline'
            except Exception:
                return None

        if len(residuals) < 10:
            return None

        n_lags = min(40, len(residuals) // 2 - 1)
        conf_bound = 1.96 / np.sqrt(len(residuals))

        try:
            acf_vals = acf(residuals, nlags=n_lags, fft=True)
            pacf_vals = pacf(residuals, nlags=n_lags, method='ywm')
        except Exception:
            return None

        # Exclude lag 0 from ACF/PACF plots (always 1.0 — distorts y-axis)
        lags_plot = list(range(1, len(acf_vals)))
        acf_plot = acf_vals[1:].tolist()
        pacf_plot = pacf_vals[1:].tolist()

        unit = self.repository.feature_units.get(feature, '')
        LIGHT_BG = 'rgba(248,250,252,0.6)'

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                f'Residuals over time ({diagnostics_source})',
                f'ACF (lags 1–{n_lags},  95% CI ±{conf_bound:.3f})',
                f'PACF (lags 1–{n_lags},  95% CI ±{conf_bound:.3f})',
            ],
            vertical_spacing=0.14,
            row_heights=[0.38, 0.31, 0.31],
        )

        # Panel 1 — Residuals time series
        res_vals = residuals.values.tolist()
        res_dates = list(residuals.index)
        colors = ['rgba(248,113,113,0.85)' if r < 0 else 'rgba(96,165,250,0.85)' for r in res_vals]
        fig.add_trace(go.Bar(
            x=res_dates, y=res_vals,
            marker_color=colors,
            name='Residual',
            hovertemplate='%{x|%Y-%m-%d}: %{y:.3f} ' + unit + '<extra></extra>',
        ), row=1, col=1)
        fig.add_hline(y=0, line_color='rgba(100,116,139,0.5)', line_width=1, row=1, col=1)

        # Panel 2 — ACF (lag 0 excluded)
        sig_colors_acf = [
            'rgba(96,165,250,0.9)' if abs(v) > conf_bound else 'rgba(96,165,250,0.45)'
            for v in acf_plot
        ]
        fig.add_trace(go.Bar(
            x=lags_plot, y=acf_plot,
            marker_color=sig_colors_acf,
            name='ACF',
            hovertemplate='Lag %{x}: %{y:.3f}<extra></extra>',
        ), row=2, col=1)
        fig.add_hrect(y0=-conf_bound, y1=conf_bound,
                      fillcolor='rgba(148,163,184,0.08)', line_width=0, row=2, col=1)
        for bound in [conf_bound, -conf_bound]:
            fig.add_hline(y=bound, line_color='rgba(245,158,11,0.7)',
                          line_width=1.2, line_dash='dash', row=2, col=1)

        # Panel 3 — PACF (lag 0 excluded)
        sig_colors_pacf = [
            'rgba(52,211,153,0.9)' if abs(v) > conf_bound else 'rgba(52,211,153,0.45)'
            for v in pacf_plot
        ]
        fig.add_trace(go.Bar(
            x=lags_plot, y=pacf_plot,
            marker_color=sig_colors_pacf,
            name='PACF',
            hovertemplate='Lag %{x}: %{y:.3f}<extra></extra>',
        ), row=3, col=1)
        fig.add_hrect(y0=-conf_bound, y1=conf_bound,
                      fillcolor='rgba(148,163,184,0.08)', line_width=0, row=3, col=1)
        for bound in [conf_bound, -conf_bound]:
            fig.add_hline(y=bound, line_color='rgba(245,158,11,0.7)',
                          line_width=1.2, line_dash='dash', row=3, col=1)

        fig.update_layout(
            template='plotly_white',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor=LIGHT_BG,
            font={'family': 'Inter, Arial, sans-serif', 'size': 11, 'color': '#334155'},
            showlegend=False,
            height=580,
            margin={'l': 55, 'r': 28, 't': 60, 'b': 44},
            bargap=0.15,
        )
        for i in range(1, 4):
            fig.update_xaxes(showgrid=True, gridcolor='rgba(148,163,184,0.15)',
                             tickfont={'size': 10, 'color': '#64748b'}, row=i, col=1)
            fig.update_yaxes(showgrid=True, gridcolor='rgba(148,163,184,0.15)',
                             tickfont={'size': 10, 'color': '#64748b'}, row=i, col=1)
        fig.update_xaxes(title_text='Lag', title_font={'size': 10}, row=2, col=1)
        fig.update_xaxes(title_text='Lag', title_font={'size': 10}, row=3, col=1)
        fig.update_yaxes(title_text=f'Residual ({unit})', title_font={'size': 10}, row=1, col=1)

        # Style subplot titles
        for ann in fig.layout.annotations:
            ann.font.size = 11
            ann.font.color = '#475569'

        # ── Diagnostics summary ───────────────────────────────────────────────
        res_mean = float(np.mean(res_vals))
        res_std = float(np.std(res_vals))
        sig_acf_lags = [lags_plot[i] for i, v in enumerate(acf_plot) if abs(v) > conf_bound]
        sig_pacf_lags = [lags_plot[i] for i, v in enumerate(pacf_plot) if abs(v) > conf_bound]
        summary = self._diagnostics_summary(
            feature, unit, res_mean, res_std, conf_bound,
            sig_acf_lags, sig_pacf_lags, n_lags,
            diagnostics_source=diagnostics_source,
        )

        return {'figure': fig, 'summary': summary}

    def _diagnostics_summary(
        self,
        feature: str,
        unit: str,
        res_mean: float,
        res_std: float,
        conf_bound: float,
        sig_acf_lags: list,
        sig_pacf_lags: list,
        n_lags: int,
        diagnostics_source: str,
    ) -> str:
        bias_desc = (
            f'mean residual of {res_mean:+.3f} {unit}, indicating a slight '
            + ('positive bias (model under-predicts on average)' if res_mean > 0 else 'negative bias (model over-predicts on average)')
        ) if abs(res_mean) > 0.01 * res_std else f'near-zero mean residual ({res_mean:+.4f} {unit}), suggesting no systematic bias'

        if not sig_acf_lags:
            acf_desc = f'No significant ACF spikes detected across all {n_lags} lags — residuals behave as white noise.'
        else:
            acf_desc = (
                f'Significant ACF at lag{"s" if len(sig_acf_lags) > 1 else ""} '
                f'{", ".join(str(l) for l in sig_acf_lags[:5])}'
                + (' (and more)' if len(sig_acf_lags) > 5 else '')
                + ' — the model has not fully captured temporal structure in the data.'
            )

        if not sig_pacf_lags:
            pacf_desc = f'No significant PACF spikes — no direct lag dependencies remain after accounting for shorter lags.'
        else:
            pacf_desc = (
                f'Significant PACF at lag{"s" if len(sig_pacf_lags) > 1 else ""} '
                f'{", ".join(str(l) for l in sig_pacf_lags[:5])}'
                + (' (and more)' if len(sig_pacf_lags) > 5 else '')
                + ' — consider an AR component at those lags.'
            )

        overall = 'well-specified' if (not sig_acf_lags and not sig_pacf_lags) else 'improvable'
        overall_note = (
            'Residuals approximate white noise, supporting forecast reliability.'
            if overall == 'well-specified'
            else 'Residual structure suggests the model may benefit from higher-order components or differencing.'
        )

        # ── Local fallback HTML (always computed) ─────────────────────────────
        # Relative spread: how large is residual std vs CI bound
        spread_note = (
            f'The 95% confidence band of ±{conf_bound:.3f} {unit} is relatively tight, '
            f'suggesting point forecasts carry low uncertainty.'
            if res_std < conf_bound * 0.6
            else f'The 95% confidence band of ±{conf_bound:.3f} {unit} is moderately wide — '
            f'point forecasts should be interpreted with caution.'
        )
        acf_severity = (
            '' if not sig_acf_lags
            else f' AR terms at lag{"s" if len(sig_acf_lags) > 1 else ""} '
            f'{", ".join(str(l) for l in sig_acf_lags[:3])} may improve model fit.'
        )
        fallback_html = (
            f'<p><strong>Residual Diagnostics — {feature.replace("_", " ")}</strong></p>'
            f'<ul>'
            f'<li><strong>Diagnostics Source:</strong> Residuals computed from the <strong>{diagnostics_source}</strong>.</li>'
            f'<li><strong>Bias:</strong> The model shows a {bias_desc}.</li>'
            f'<li><strong>Residual Spread:</strong> Standard deviation is {res_std:.3f} {unit}. {spread_note}</li>'
            f'<li><strong>Autocorrelation (ACF):</strong> {acf_desc}{acf_severity}</li>'
            f'<li><strong>Partial Autocorrelation (PACF):</strong> {pacf_desc}</li>'
            f'<li><strong>Overall:</strong> Model fit is <strong>{overall}</strong>. {overall_note}</li>'
            f'</ul>'
        )

        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key or api_key == 'your_google_gemini_api_key_here' or not api_key.strip():
            return fallback_html

        try:
            prompt = (
                'Act as a professional hydrologist interpreting residual diagnostics from a time-series model.\n\n'
                'RESPONSE FORMAT (STRICT — bullet points only, no intro or sign-off):\n\n'
                '**Residual Diagnostics**\n\n'
                '- **Bias Assessment**: [Interpret mean residual; is there systematic over/under-prediction?]\n'
                '- **Residual Spread**: [Interpret std; is the model error large or small relative to the variable?]\n'
                '- **Autocorrelation Structure (ACF)**: [Are residuals white noise or is temporal structure left over?]\n'
                '- **Partial Autocorrelation (PACF)**: [What lag orders are significant? What does that suggest?]\n'
                '- **Model Adequacy**: [Is the model well-specified or should enhancements be considered?]\n'
                '- **Practical Implication**: [How does residual quality affect the reliability of forecasts?]\n\n'
                'RULES: Cite numbers. Each bullet 1-2 sentences.\n\n'
                f'Variable: {feature.replace("_", " ")} ({unit})\n'
                f'Diagnostics source: {diagnostics_source}\n'
                f'Residual mean: {res_mean:+.4f} {unit}\n'
                f'Residual std: {res_std:.3f} {unit}\n'
                f'95% CI bound: ±{conf_bound:.3f}\n'
                f'Significant ACF lags (excluding lag 0): {sig_acf_lags if sig_acf_lags else "none"}\n'
                f'Significant PACF lags (excluding lag 0): {sig_pacf_lags if sig_pacf_lags else "none"}\n'
                f'Total lags evaluated: {n_lags}\n'
            )
            return markdown.markdown(_gemini_generate(api_key, prompt))
        except Exception:
            return fallback_html

    def _historical_summary(
        self, frame: pd.DataFrame, hist_fit_df: pd.DataFrame,
        rmse: float, mape: float, actual_h: int, station: str, feature: str,
    ) -> str:
        unit = self.repository.feature_units.get(feature, '')
        n_windows = len(hist_fit_df)
        date_start = hist_fit_df['Timestamp'].min().strftime('%Y-%m-%d')
        date_end = hist_fit_df['Timestamp'].max().strftime('%Y-%m-%d')
        hist_mean = float(frame['Value'].mean())
        rmse_str = f'{rmse:.3f} {unit}' if not np.isnan(rmse) else 'n/a'
        mape_str = f'{mape:.1f}%' if (not np.isnan(mape) and mape <= 500) else 'n/a'

        # ── Accuracy grading ──────────────────────────────────────────────────
        if not np.isnan(mape) and mape <= 500:
            if mape < 10:
                acc_grade, acc_label = 'strong', 'Strong'
            elif mape < 20:
                acc_grade, acc_label = 'moderate', 'Moderate'
            else:
                acc_grade, acc_label = 'poor', 'Poor'
        else:
            acc_grade, acc_label = 'indeterminate', 'Indeterminate'

        rmse_pct_of_mean = (
            f' ({rmse / hist_mean * 100:.1f}% of historical mean)'
            if hist_mean > 0 and not np.isnan(rmse)
            else ''
        )
        horizon_note = (
            f'At H={actual_h}, each prediction is made {actual_h} step(s) ahead — '
            + ('a short horizon typical of near-term operational forecasting.'
               if actual_h <= 7
               else 'a longer horizon where error accumulation is expected.')
        )
        coverage_note = (
            f'{n_windows} rolling evaluation windows span {date_start} to {date_end}, '
            + ('providing broad temporal coverage for a reliable accuracy estimate.'
               if n_windows > 200
               else 'covering a moderate evaluation window.')
        )
        recommendation = {
            'strong': 'Model fit is sufficient for operational use with reasonable confidence.',
            'moderate': 'Model fit is acceptable — supplement forecasts with local expert judgement before operational use.',
            'poor': 'Model fit is weak — treat forecasts as indicative only and validate against current observations.',
            'indeterminate': 'Accuracy could not be computed reliably — interpret results with caution.',
        }[acc_grade]

        fallback_html = (
            f'<p><strong>Historical Fit Assessment — {feature.replace("_", " ")} · {station.replace("_", " ")}</strong></p>'
            f'<ul>'
            f'<li><strong>Overall Accuracy:</strong> RMSE = {rmse_str}{rmse_pct_of_mean}, '
            f'MAPE = {mape_str} — accuracy grade is <strong>{acc_label}</strong>.</li>'
            f'<li><strong>Horizon Effect:</strong> {horizon_note}</li>'
            f'<li><strong>Coverage Period:</strong> {coverage_note}</li>'
            f'<li><strong>Historical Reference:</strong> Historical mean is {hist_mean:.3f} {unit}, '
            f'providing the baseline against which model error is measured.</li>'
            f'<li><strong>Operational Recommendation:</strong> {recommendation}</li>'
            f'</ul>'
        )

        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key or api_key == 'your_google_gemini_api_key_here' or not api_key.strip():
            return fallback_html

        try:
            prompt = (
                'Act as a professional hydrologist reviewing a model backtesting evaluation.\n\n'
                'RESPONSE FORMAT (STRICT — bullet points only):\n\n'
                '**Historical Fit Assessment**\n\n'
                '- **Overall Accuracy**: [Interpret RMSE and MAPE; is the model fit strong, moderate, or poor?]\n'
                '- **Bias & Systematic Errors**: [Does the model consistently over- or under-predict? Any drift?]\n'
                '- **Horizon Effect**: [What does H={h} mean operationally — how far ahead is the model predicting each point?]\n'
                '- **Coverage Period**: [What do {n} evaluation windows over {start}→{end} tell us about reliability?]\n'
                '- **Operational Recommendation**: [Should this model be trusted for decisions? Any caveats?]\n\n'
                'RULES: Cite numbers. No intro, no sign-off. Each bullet 1-2 sentences.\n\n'
                f'Station: {station.replace("_", " ")}\nFeature: {feature.replace("_", " ")} ({unit})\n'
                f'Horizon H: {actual_h}\nWindows evaluated: {n_windows} ({date_start} → {date_end})\n'
                f'RMSE: {rmse_str}\nMAPE: {mape_str}\nHistorical mean: {hist_mean:.3f} {unit}\n'
            ).replace('{h}', str(actual_h)).replace('{n}', str(n_windows)).replace('{start}', date_start).replace('{end}', date_end)
            html_content = markdown.markdown(_gemini_generate(api_key, prompt))
            return html_content
        except Exception:
            return fallback_html

    def _prediction_summary(
        self,
        frame: pd.DataFrame,
        forecast_values: pd.Series,
        residual_std: float,
        rmse: float,
        mape: float,
        climatological_context: str,
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
        hist_mean = float(frame['Value'].mean())
        hist_std = float(frame['Value'].std())
        hist_min = float(frame['Value'].min())
        hist_max = float(frame['Value'].max())
        mape_str = f'{mape:.1f}%' if not np.isnan(mape) else 'n/a'
        rmse_str = f'{rmse:.3f} {unit}'

        # ── Historical context classification ─────────────────────────────────
        hist_deviation = ((last_forecast - hist_mean) / hist_mean * 100) if hist_mean != 0 else 0.0
        if last_forecast > hist_mean + 1.5 * hist_std:
            hist_context = f'well above the historical mean ({hist_deviation:+.1f}% above)'
        elif last_forecast > hist_mean + 0.5 * hist_std:
            hist_context = f'above the historical mean ({hist_deviation:+.1f}% above)'
        elif last_forecast > hist_mean - 0.5 * hist_std:
            hist_context = f'near the historical mean ({hist_deviation:+.1f}%)'
        elif last_forecast > hist_mean - 1.5 * hist_std:
            hist_context = f'below the historical mean ({hist_deviation:+.1f}% below)'
        else:
            hist_context = f'well below the historical mean ({abs(hist_deviation):.1f}% below)'

        # ── Volatility note ───────────────────────────────────────────────────
        forecast_range = forecast_max - forecast_min
        hist_range = hist_max - hist_min
        range_pct_of_hist = (forecast_range / hist_range * 100) if hist_range > 0 else 0.0
        volatility_note = (
            f'The forecast spans {forecast_min:.2f}–{forecast_max:.2f} {unit} '
            f'(range: {forecast_range:.2f} {unit}, {range_pct_of_hist:.0f}% of the historical range), '
            + ('indicating high intra-period variability.' if range_pct_of_hist > 60
               else 'indicating moderate intra-period variability.' if range_pct_of_hist > 25
               else 'indicating relatively stable conditions ahead.')
        )

        # ── Confidence note ───────────────────────────────────────────────────
        if not np.isnan(mape) and mape <= 500:
            if mape < 10:
                conf_label = 'high'
            elif mape < 20:
                conf_label = 'moderate'
            else:
                conf_label = 'low'
        else:
            conf_label = 'indeterminate'
        conf_note = (
            f'RMSE = {rmse_str}, MAPE = {mape_str} — forecast confidence is <strong>{conf_label}</strong>. '
            f'Each forecast point carries ±{residual_std:.3f} {unit} (1σ residual spread from backtesting).'
        )

        fallback_html = (
            f'<p><strong>Executive Summary</strong></p>'
            f'<p>The {horizon}-{period_label} forecast for <strong>{feature.replace("_", " ")}</strong> '
            f'at <strong>{station.replace("_", " ")}</strong> projects a <strong>{direction}</strong> '
            f'trajectory of {pct_change:+.1f}%, moving from {last_actual:.2f} {unit} to {last_forecast:.2f} {unit} '
            f'by the end of the horizon. The end-of-horizon value is {hist_context}. '
            f'Forecast confidence is <strong>{conf_label}</strong> based on historical backtesting '
            f'(RMSE {rmse_str}, MAPE {mape_str}). '
            f'{volatility_note}</p>'

            f'<p><strong>Trajectory and Magnitude</strong></p>'
            f'<p>The forecast moves from the last observed value of {last_actual:.2f} {unit} '
            f'to {last_forecast:.2f} {unit} at the end of the {horizon}-{period_label} horizon, '
            f'representing a {pct_change:+.1f}% change. '
            f'The first forecast step is {first_forecast:.2f} {unit}, '
            + (f'indicating an {"immediate rise" if first_forecast > last_actual else "immediate decline"} from the current observed level. '
               if abs(first_forecast - last_actual) > 0.01 * abs(last_actual) else
               'indicating near-continuity from the current observed level. ')
            + f'The rate of change averages {abs(last_forecast - last_actual) / max(horizon, 1):.3f} {unit} per {period_label} across the forecast window. '
            f'Peak forecast value within the horizon is {forecast_max:.2f} {unit} and the minimum is {forecast_min:.2f} {unit}.</p>'

            f'<p><strong>Historical Comparison</strong></p>'
            f'<p>The end-of-horizon forecast value of {last_forecast:.2f} {unit} is {hist_context}. '
            f'For reference, the full historical record spans {hist_min:.2f}–{hist_max:.2f} {unit} '
            f'with a mean of {hist_mean:.2f} {unit} and standard deviation of {hist_std:.2f} {unit}. '
            + (f'The forecast end-value is {abs(last_forecast - hist_mean) / hist_std:.1f} standard deviations from the historical mean, '
               + ('placing it in an unusual range that warrants additional scrutiny. '
                  if abs(last_forecast - hist_mean) / hist_std > 2 else
                  'within the normal range of historical variability. ')
               if hist_std > 0 else '')
            + f'The forecast range ({forecast_min:.2f}–{forecast_max:.2f} {unit}) covers {range_pct_of_hist:.0f}% of the full historical range, '
            + ('suggesting high within-period variability — the forecast is not flat and conditions are expected to change significantly during the horizon. '
               if range_pct_of_hist > 60 else
               'suggesting moderate variability within the forecast period. '
               if range_pct_of_hist > 25 else
               'suggesting relatively stable conditions throughout the forecast period. ')
            + '</p>'

            f'<p><strong>Model Confidence and Uncertainty</strong></p>'
            f'<p>{conf_note} '
            f'The residual standard deviation of ±{residual_std:.3f} {unit} from backtesting provides a practical uncertainty estimate: '
            f'approximately 68% of actual future values should fall within ±{residual_std:.3f} {unit} of the forecast line, '
            f'and approximately 95% within ±{2*residual_std:.3f} {unit}, assuming the model error distribution is approximately normal and stationary. '
            + ('With a MAPE below 10%, this model demonstrates strong historical accuracy and its forecasts can be used with reasonable operational confidence. '
               if conf_label == 'high' else
               'With a MAPE between 10–20%, this model shows moderate accuracy — treat the forecast as a directional guide rather than a precise prediction. '
               if conf_label == 'moderate' else
               'With a MAPE above 20%, this model shows weaker accuracy on this station–feature combination. '
               'Use the forecast for general directional guidance only and monitor against real-time observations closely. '
               if conf_label == 'low' else
               'Model confidence could not be assessed from available backtesting data. ')
            + '</p>'

            f'<p><strong>Seasonal and Climatological Context</strong></p>'
            f'<p>The forecast covers {horizon} {period_label}(s) ahead. '
            'Comparing each forecast step to the historical monthly climatological normal for that calendar period reveals whether the model is projecting above- or below-normal conditions. '
            'Significant positive deviations from climatology may indicate an anomalous wet period, upstream release event, or lagged monsoon response. '
            'Significant negative deviations may indicate drought conditions, reduced upstream inflow, or seasonal dry-season onset ahead of schedule. '
            'The monthly climatological context should be inspected in the chart alongside the numerical forecast values.</p>'

            f'<p><strong>Operational Risk Assessment</strong></p>'
            '<ul>'
            + (f'<li><strong>High-flow risk:</strong> The forecast maximum of {forecast_max:.2f} {unit} '
               + (f'approaches or exceeds the historical 90th percentile — elevated flood preparedness is recommended. '
                  if forecast_max > hist_mean + 1.28 * hist_std else
                  'is within the normal historical range. ')
               + '</li>'
               if hist_std > 0 else '')
            + (f'<li><strong>Low-flow risk:</strong> The forecast minimum of {forecast_min:.2f} {unit} '
               + (f'falls near or below the historical 10th percentile — water scarcity and ecological minimum-flow conditions should be monitored. '
                  if forecast_min < hist_mean - 1.28 * hist_std else
                  'remains within the normal historical range. ')
               + '</li>'
               if hist_std > 0 else '')
            + f'<li><strong>Data quality and limitations:</strong> This forecast is generated from a pre-trained ML or statistical model based on historical observations. '
            f'It does not account for real-time basin interventions (dam releases, diversions), extreme events outside the training distribution, '
            f'or sudden shifts in catchment behaviour. Always cross-reference with real-time gauge readings and operational advisories before high-stakes decisions.</li>'
            '</ul>'
        )

        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key or api_key == 'your_google_gemini_api_key_here' or not api_key.strip():
            return fallback_html

        try:
            prompt = (
                'Act as a professional hydrologist and water resource specialist writing a detailed technical forecast report.\n\n'
                'RESPONSE FORMAT (STRICT):\n'
                'Use exactly these sections:\n\n'
                '**Executive Summary**\n'
                '4-5 sentences. State the forecast direction and total magnitude of change, where the end-of-horizon value sits relative to the historical distribution, '
                'the model confidence level (RMSE and MAPE), the forecast volatility, and the primary operational implication.\n\n'
                '**Trajectory and Magnitude**\n'
                'A paragraph (4-5 sentences). Describe the full trajectory: starting value, first step, end value, rate of change per period, peak and trough within the horizon. '
                'Interpret whether the change is gradual or rapid. State what this trajectory implies for the hydrological variable (rising limb, recession, plateau).\n\n'
                '**Historical Comparison**\n'
                'A paragraph (4-5 sentences). Compare forecast values against the full historical record statistics (mean, std, min, max). '
                'Express the end-of-horizon value in terms of standard deviations from the mean or historical percentile rank. '
                'State whether the forecast represents normal, above-normal, or below-normal conditions and what this implies operationally.\n\n'
                '**Model Confidence and Uncertainty**\n'
                'A paragraph (4-5 sentences). Interpret RMSE and MAPE in detail — classify confidence as high/moderate/low with justification. '
                'Derive practical uncertainty bounds from the residual standard deviation (±1σ = 68%, ±2σ = 95%). '
                'Discuss factors that could cause actual values to deviate more than the model error suggests.\n\n'
                '**Seasonal and Climatic Context**\n'
                'A paragraph (3-4 sentences). Compare each forecast step to the historical monthly climatological normal using the provided context. '
                'Identify which months are projected above or below normal and by how much. '
                'Discuss plausible climate drivers (monsoon phase, ENSO, seasonal transition) that may explain deviations.\n\n'
                '**Operational Risk Assessment**\n'
                'Exactly 3 bullet points:\n'
                '- **High-flow risk:** assess flood or high-water risk based on forecast maximum vs. historical distribution.\n'
                '- **Low-flow risk:** assess drought or water scarcity risk based on forecast minimum vs. historical distribution.\n'
                '- **Decision recommendations:** state specific operational actions this forecast should trigger (reservoir management, irrigation scheduling, flood alerts, ecological minimum-flow monitoring).\n\n'
                '**Data Quality and Limitations**\n'
                'Exactly 3 bullet points covering: model training assumptions, factors not captured (real-time interventions, extreme events outside training range), '
                'and recommended cross-checks with real-time observations.\n\n'
                'RULES:\n'
                '• Use professional hydrological language throughout\n'
                '• Replace underscores with spaces in all names\n'
                '• Cite actual numbers from the data in every section\n'
                '• No introductions, sign-offs, apologies, or code blocks\n'
                '• Be specific — avoid vague qualifiers without numeric support\n\n'
                f'Station: {station.replace("_", " ")}\n'
                f'Feature: {feature.replace("_", " ")} ({unit})\n'
                f'Horizon: {horizon} {period_label}(s)\n'
                f'Last historical value: {last_actual:.2f} {unit}\n'
                f'First forecast value: {first_forecast:.2f} {unit}\n'
                f'Last forecast value: {last_forecast:.2f} {unit} ({pct_change:+.1f}% change)\n'
                f'Forecast min/max: {forecast_min:.2f} / {forecast_max:.2f} {unit}\n'
                f'Forecast range width: {forecast_max - forecast_min:.2f} {unit}\n'
                f'Residual std: {residual_std:.3f} {unit}\n'
                f'RMSE (model error): {rmse_str}\n'
                f'MAPE (accuracy %): {mape_str}\n'
                f'Historical statistics:\n'
                f'  - Mean: {hist_mean:.2f} {unit}\n'
                f'  - Std Dev: {hist_std:.2f} {unit}\n'
                f'  - Min: {hist_min:.2f} {unit}\n'
                f'  - Max: {hist_max:.2f} {unit}\n'
                f'  - Range: {hist_max - hist_min:.2f} {unit}\n\n'
                f'Monthly climatological normals (forecast vs historical):\n{climatological_context}\n'
            )
            html_content = markdown.markdown(_gemini_generate(api_key, prompt))
            return html_content
        except Exception:
            return fallback_html
