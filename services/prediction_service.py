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
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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
        non_negative = float(model_frame['Value'].min()) >= 0

        # ── HISTORICAL MODE ───────────────────────────────────────────────────
        if mode == 'historical':
            if hist_fit_df is None or hist_fit_df.empty:
                raise ValueError(f'No historical predictions found for {station} / {feature} / {model}.')
            figure = self._build_historical_figure(model_frame, hist_fit_df, actual_h, station, feature)
            figure_zoom = self._build_historical_zoom_figure(model_frame, hist_fit_df, actual_h, feature)
            insight = self._historical_summary(model_frame, hist_fit_df, rmse, mape, actual_h, station, feature) if analysis else None
            return {
                'station': station, 'feature': feature, 'horizon': actual_h,
                'frequency': frequency,
                'figure': plotly.io.to_json(figure),
                'figure_zoom': plotly.io.to_json(figure_zoom),
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
            last_timestamp = model_frame['Timestamp'].iloc[-1]
            forecast_index = self._future_index(last_timestamp, frequency, len(csv_forecast))
            forecast_values = pd.Series(csv_forecast.values.astype(float), index=forecast_index)
            residual_std = rmse if not np.isnan(rmse) else float(np.nanstd(model_frame['Value'].values))
            step = np.sqrt(np.arange(1, effective_horizon + 1))
            lower = pd.Series(forecast_values.values - 1.96 * residual_std * step, index=forecast_index)
            upper = pd.Series(forecast_values.values + 1.96 * residual_std * step, index=forecast_index)
            if non_negative:
                lower = lower.clip(lower=0)
        else:
            _, forecast_index, forecast_values, lower, upper, residual_std, rmse, mape = self._forecast(model_frame, frequency, effective_horizon)
            hist_fit_df = None

        figure = self._build_figure(model_frame, forecast_index, forecast_values, lower, upper,
                                    feature, hist_fit_df, actual_h)
        figure_zoom = self._build_zoom_figure(model_frame, forecast_index, forecast_values, lower, upper,
                                              station, feature, frequency, hist_fit_df, actual_h)

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
            'figure': plotly.io.to_json(figure),
            'figure_zoom': plotly.io.to_json(figure_zoom),
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
        lower,
        upper,
        feature: str,
        hist_fit_df: pd.DataFrame | None = None,
        actual_h: int = 1,
    ) -> go.Figure:
        unit = self.repository.feature_units.get(feature, '')
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=frame['Timestamp'], y=frame['Value'],
            mode='lines', name='Actual',
            line={'width': 2, 'color': '#38bdf8'},
        ))
        if hist_fit_df is not None and not hist_fit_df.empty:
            figure.add_trace(go.Scatter(
                x=hist_fit_df['Timestamp'], y=hist_fit_df['ModelFit'],
                mode='lines', name=f'Model fit (H={actual_h})',
                line={'width': 1.5, 'color': '#a78bfa'},
                opacity=0.85,
            ))
        # Confidence band
        figure.add_trace(go.Scatter(
            x=forecast_index, y=upper,
            mode='lines', line={'width': 0},
            showlegend=False, hoverinfo='skip',
        ))
        figure.add_trace(go.Scatter(
            x=forecast_index, y=lower,
            mode='lines', line={'width': 0},
            fill='tonexty', fillcolor='rgba(245,158,11,0.15)',
            name='Confidence band', hoverinfo='skip',
        ))
        figure.add_trace(go.Scatter(
            x=forecast_index, y=forecast_values,
            mode='lines+markers', name=f'Forecast (H={actual_h})',
            line={'width': 2.4, 'color': '#f59e0b', 'dash': 'dash'},
            marker={'size': 5},
        ))
        figure.update_layout(
            template='plotly_white',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#ffffff',
            font={'family': 'Inter, Arial, sans-serif', 'size': 12, 'color': '#0f172a'},
            margin={'l': 46, 'r': 28, 't': 48, 'b': 48},
            hovermode='x unified',
            legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'left', 'x': 0},
        )
        figure.update_xaxes(showgrid=True, gridcolor='rgba(148,163,184,0.22)')
        figure.update_yaxes(showgrid=True, gridcolor='rgba(148,163,184,0.22)',
                            title=f'{feature.replace("_"," ")} ({unit})')
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
        hist_fit_df: pd.DataFrame | None = None,
        actual_h: int = 1,
    ) -> go.Figure:
        unit = self.repository.feature_units.get(feature, '')

        # Show last ~2 years of history to align with model fit window
        cutoff = frame['Timestamp'].max() - (pd.DateOffset(months=24) if frequency == 'monthly' else pd.Timedelta(days=730))
        zoom_frame = frame[frame['Timestamp'] >= cutoff]

        last_hist_ts = zoom_frame['Timestamp'].iloc[-1]
        last_hist_val = float(zoom_frame['Value'].iloc[-1])
        bridge_x = [last_hist_ts] + list(forecast_index)
        bridge_forecast = [last_hist_val] + list(forecast_values.values)
        bridge_upper = [last_hist_val] + list(upper.values)
        bridge_lower = [last_hist_val] + list(lower.values)

        if frequency == 'monthly':
            x_end = pd.Timestamp(forecast_index[-1]) + pd.DateOffset(months=1)
        else:
            x_end = pd.Timestamp(forecast_index[-1]) + pd.Timedelta(days=3)
        x_start = pd.Timestamp(zoom_frame['Timestamp'].iloc[0])

        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=zoom_frame['Timestamp'], y=zoom_frame['Value'],
            mode='lines', name='Actual (last 2 years)',
            line={'width': 2, 'color': '#38bdf8'},
        ))
        # Model fit overlay (clipped to zoom window)
        if hist_fit_df is not None and not hist_fit_df.empty:
            fit_zoom = hist_fit_df[hist_fit_df['Timestamp'] >= cutoff]
            if not fit_zoom.empty:
                figure.add_trace(go.Scatter(
                    x=fit_zoom['Timestamp'], y=fit_zoom['ModelFit'],
                    mode='lines', name=f'Model fit (H={actual_h})',
                    line={'width': 1.5, 'color': '#a78bfa'},
                    opacity=0.85,
                ))
        # Confidence band (bridged)
        figure.add_trace(go.Scatter(
            x=bridge_x, y=bridge_upper,
            mode='lines', line={'width': 0},
            showlegend=False, hoverinfo='skip',
        ))
        figure.add_trace(go.Scatter(
            x=bridge_x, y=bridge_lower,
            mode='lines', line={'width': 0},
            fill='tonexty', fillcolor='rgba(245,158,11,0.15)',
            name='Confidence band', hoverinfo='skip',
        ))
        figure.add_trace(go.Scatter(
            x=bridge_x, y=bridge_forecast,
            mode='lines+markers', name=f'Forecast (H={actual_h})',
            line={'width': 2.4, 'color': '#f59e0b', 'dash': 'dash'},
            marker={'size': 6},
        ))
        figure.update_layout(
            template='plotly_white',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#ffffff',
            font={'family': 'Inter, Arial, sans-serif', 'size': 12, 'color': '#0f172a'},
            margin={'l': 46, 'r': 28, 't': 28, 'b': 48},
            hovermode='x unified',
            legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'left', 'x': 0},
            shapes=[{
                'type': 'line', 'x0': last_hist_ts, 'x1': last_hist_ts,
                'y0': 0, 'y1': 1, 'yref': 'paper',
                'line': {'color': 'rgba(245,158,11,0.5)', 'width': 1.5, 'dash': 'dot'},
            }],
            annotations=[{
                'x': last_hist_ts, 'y': 1, 'yref': 'paper',
                'text': 'Forecast start', 'showarrow': False,
                'font': {'size': 10, 'color': '#f59e0b'},
                'xanchor': 'left', 'xshift': 4,
            }],
        )
        figure.update_xaxes(
            showgrid=True, gridcolor='rgba(148,163,184,0.22)',
            range=[x_start.strftime('%Y-%m-%d'), x_end.strftime('%Y-%m-%d')],
            autorange=False,
        )
        figure.update_yaxes(showgrid=True, gridcolor='rgba(148,163,184,0.22)',
                            title=f'{feature.replace("_"," ")} ({unit})')
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
            line={'width': 2, 'color': '#38bdf8'},
        ))
        fig.add_trace(go.Scatter(
            x=hist_fit_df['Timestamp'], y=hist_fit_df['ModelFit'],
            mode='lines', name=f'Model fit (H={actual_h})',
            line={'width': 1.8, 'color': '#a78bfa'},
            opacity=0.9,
        ))
        fig.update_layout(
            template='plotly_white',
            title={'text': f'Historical Fit · {feature.replace("_"," ")} · {station.replace("_"," ")}', 'x': 0.5, 'xanchor': 'center', 'y': 0.97, 'yanchor': 'top'},
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#ffffff',
            font={'family': 'Inter, Arial, sans-serif', 'size': 12, 'color': '#0f172a'},
            margin={'l': 46, 'r': 28, 't': 96, 'b': 48},
            hovermode='x unified',
            legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'left', 'x': 0},
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148,163,184,0.22)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148,163,184,0.22)',
                         title=f'{feature.replace("_"," ")} ({unit})')
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
        ))
        fig.add_trace(go.Scatter(
            x=hist_fit_df['Timestamp'], y=hist_fit_df['ModelFit'],
            mode='lines', name=f'Model fit (H={actual_h})',
            line={'width': 1.8, 'color': '#a78bfa'},
            opacity=0.9,
        ))
        fig.update_layout(
            template='plotly_white',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#ffffff',
            font={'family': 'Inter, Arial, sans-serif', 'size': 12, 'color': '#0f172a'},
            margin={'l': 46, 'r': 28, 't': 28, 'b': 48},
            hovermode='x unified',
            legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'left', 'x': 0},
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148,163,184,0.22)',
                         range=[x_start.strftime('%Y-%m-%d'),
                                hist_fit_df['Timestamp'].max().strftime('%Y-%m-%d')],
                         autorange=False)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148,163,184,0.22)',
                         title=f'{feature.replace("_"," ")} ({unit})')
        return fig

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

        base = (
            f'Historical fit for {feature} at {station} using H={actual_h} over {n_windows} windows '
            f'({date_start} → {date_end}). RMSE: {rmse_str}, MAPE: {mape_str}. '
            f'Historical mean: {hist_mean:.3f} {unit}.'
        )

        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key or api_key == 'your_google_gemini_api_key_here' or not api_key.strip():
            return base

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
        except Exception as e:
            return base + f'\n\n*(AI Analysis Failed: {str(e)})*'

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

        base_summary = (
            f'The forecast for {feature} at {station} spans the next {horizon} {period_label}(s). '
            f'The model projects a {direction} trajectory from {last_actual:.2f} {unit} '
            f'to {last_forecast:.2f} {unit} ({pct_change:+.1f}%). '
            f'Model fit: RMSE {rmse_str}, MAPE {mape_str}. '
            f'Forecast range: {forecast_min:.2f}–{forecast_max:.2f} {unit}.'
        )

        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key or api_key == 'your_google_gemini_api_key_here' or not api_key.strip():
            return base_summary + '\n\n*(Note: Set GEMINI_API_KEY in the .env file to enable AI-powered prediction analysis)*'

        try:
            prompt = (
                'Act as a professional hydrologist and water resource specialist. '
                'Analyze the forecast results and provide a comprehensive, data-driven report.\n\n'
                'RESPONSE FORMAT (STRICT):\n'
                'Start with exactly this structure:\n\n'
                '**Executive Summary**\n'
                '[2-3 sentences overview of the forecast outlook and key implications]\n\n'
                '**Forecast Insights**\n\n'
                '- **Trajectory & Magnitude**: [Direction of change, percentage change, rate per period, and what this means for the variable]\n'
                '- **Historical Comparison**: [How forecast values rank against historical min/mean/max; cite percentiles if significant]\n'
                '- **Volatility & Range**: [Expected variability, widest swings, and what conditions trigger high/low values]\n'
                '- **Model Confidence**: [Interpret RMSE and MAPE; assess forecast reliability and uncertainty margins]\n'
                '- **Seasonal & Climatic Context**: [Compare forecast months to historical monthly normals; cite specific deviations and months]\n'
                '- **Operational Risk Assessment**: [Flood/drought risk, water availability, irrigation implications, or ecosystem effects]\n'
                '- **Data Quality Notes**: [Limitations of the model, assumptions, or factors not captured in the forecast]\n\n'
                'RULES:\n'
                '• Use ONLY bullet points (prefix each with "-" and bold the heading)\n'
                '• Replace underscores with spaces in names\n'
                '• Cite actual numbers from the data\n'
                '• No introductions, sign-offs, apologies, or code blocks\n'
                '• Be specific: avoid vague terms like "moderate" or "notable"\n'
                '• Each bullet should be 1-2 sentences of focused analysis\n\n'
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
        except Exception as e:
            return base_summary + f'\n\n*(AI Analysis Failed: {str(e)})*'
