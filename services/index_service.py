"""
index_service.py
~~~~~~~~~~~~~~~~
Computes drought and flood indices for stations:
  - SPI-{scale}: Standardized Precipitation Index (Gamma-distribution fitting)
  - Flow Anomaly Index: % deviation from long-run monthly climatology

Each result carries an alert level: normal | watch | warning | critical
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_loader import DataRepository, SeriesRequest
from .feature_registry import is_flow, is_precip


# ── Alert level helpers ──────────────────────────────────────────────────────

_ALERT_ORDER = {'normal': 0, 'watch': 1, 'warning': 2, 'critical': 3}

ALERT_META = {
    'normal':   {'color': '#22c55e', 'label': 'Normal'},
    'watch':    {'color': '#f59e0b', 'label': 'Watch'},
    'warning':  {'color': '#f97316', 'label': 'Warning'},
    'critical': {'color': '#ef4444', 'label': 'Critical'},
}


def _worst_level(*levels: str) -> str:
    return max(levels, key=lambda lv: _ALERT_ORDER.get(lv, 0), default='normal')


def _spi_level(spi: float) -> Tuple[str, str]:
    if spi >= 2.0:
        return 'critical', 'Extremely Wet'
    if spi >= 1.5:
        return 'warning', 'Very Wet'
    if spi >= 1.0:
        return 'watch', 'Moderately Wet'
    if spi > -1.0:
        return 'normal', 'Near Normal'
    if spi > -1.5:
        return 'watch', 'Moderately Dry'
    if spi > -2.0:
        return 'warning', 'Severely Dry'
    return 'critical', 'Extremely Dry'


def _flow_level(pct: float) -> Tuple[str, str]:
    if pct >= 100.0:
        return 'critical', 'Extreme Flood'
    if pct >= 50.0:
        return 'warning', 'Flood Risk'
    if pct >= 20.0:
        return 'watch', 'Above Normal'
    if pct > -20.0:
        return 'normal', 'Normal'
    if pct > -50.0:
        return 'watch', 'Below Normal'
    if pct > -70.0:
        return 'warning', 'Low Flow'
    return 'critical', 'Critically Low'



# ── Main service ─────────────────────────────────────────────────────────────

class IndexService:
    """Compute SPI and Flow Anomaly indices for any station in the repository."""

    def __init__(self, repository: DataRepository) -> None:
        self.repository = repository

    # ── Feature discovery ────────────────────────────────────────────────────

    def _find_rain_feature(self, station: str) -> Optional[str]:
        meta = self.repository.get_station_metadata(station)
        for f in meta['features']:
            if is_precip(f):
                return f
        return None

    def _find_flow_feature(self, station: str) -> Optional[str]:
        meta = self.repository.get_station_metadata(station)
        for f in meta['features']:
            if is_flow(f):
                return f
        return None

    # ── SPI computation ──────────────────────────────────────────────────────

    def compute_spi(self, station: str, feature: str, scale: int = 3) -> Dict[str, Any]:
        """
        Compute SPI-{scale} (default 3-month) for a rainfall feature.

        Steps:
          1. Resample raw data to monthly totals.
          2. Build rolling {scale}-month accumulations.
          3. For each calendar month fit a Gamma distribution to historical values.
          4. Convert each observed value to a standard-normal z-score via the CDF.
        """
        from scipy.stats import gamma as gamma_dist, norm as norm_dist

        meta = self.repository.get_station_metadata(station)
        fd = meta['feature_details'][feature]
        req = SeriesRequest(
            station=station, feature=feature,
            start_date=fd['start_date'], end_date=fd['end_date'],
        )
        df = self.repository.get_feature_series(req)

        # Monthly totals (sum — precipitation is accumulative)
        monthly = (
            df.set_index('Timestamp')['Value']
            .resample('MS').sum()
            .dropna()
        )
        if len(monthly) < max(24, scale * 2):
            raise ValueError(
                f'Need at least {max(24, scale * 2)} months of data for SPI-{scale} '
                f'(got {len(monthly)}).'
            )

        # Rolling {scale}-month accumulation
        rolling = monthly.rolling(scale).sum().dropna()

        # Per-calendar-month Gamma fit → z-score
        spi_series = pd.Series(np.nan, index=rolling.index, dtype=float)
        for cal_month in range(1, 13):
            mask = rolling.index.month == cal_month
            if mask.sum() < 6:
                continue
            vals = rolling[mask].values.copy()
            vals = np.where(vals <= 0, 1e-6, vals)  # guard against zero rainfall
            try:
                shape, loc, sc = gamma_dist.fit(vals, floc=0)
                cdf = gamma_dist.cdf(vals, shape, loc=loc, scale=sc)
                cdf = np.clip(cdf, 0.0013, 0.9987)   # avoid ±3 σ infinity
                spi_series[mask] = norm_dist.ppf(cdf)
            except Exception:
                continue

        spi_series = spi_series.dropna()
        if spi_series.empty:
            raise ValueError('SPI computation failed — insufficient data per calendar month.')

        current_spi = float(spi_series.iloc[-1])
        level, label = _spi_level(current_spi)

        # Thin series for frontend sparkline (max 120 points)
        thin_idx = np.round(np.linspace(0, len(spi_series) - 1, min(120, len(spi_series)))).astype(int)
        series_out = [
            {'date': str(spi_series.index[i].date()), 'spi': round(float(spi_series.iloc[i]), 3)}
            for i in thin_idx
        ]

        return {
            'value': round(current_spi, 3),
            'level': level,
            'label': label,
            'scale': scale,
            'feature': feature,
            'latest_date': str(spi_series.index[-1].date()),
            'series': series_out,
        }

    # ── Flow Anomaly computation ──────────────────────────────────────────────

    def compute_flow_anomaly(self, station: str, feature: str) -> Dict[str, Any]:
        """
        Compute monthly flow anomaly as % deviation from long-run climatology.

        Returns the current (latest month) anomaly plus a full anomaly series.
        """
        meta = self.repository.get_station_metadata(station)
        fd = meta['feature_details'][feature]
        req = SeriesRequest(
            station=station, feature=feature,
            start_date=fd['start_date'], end_date=fd['end_date'],
        )
        df = self.repository.get_feature_series(req)
        unit = self.repository.feature_units.get(feature, '')

        # Monthly means
        monthly = (
            df.set_index('Timestamp')['Value']
            .resample('MS').mean()
            .dropna()
            .reset_index()
        )
        monthly.columns = ['Timestamp', 'Value']

        if len(monthly) < 12:
            raise ValueError(
                f'Need at least 12 months of data for Flow Anomaly (got {len(monthly)}).'
            )

        # Long-run climatology per calendar month
        monthly['CalMonth'] = monthly['Timestamp'].dt.month
        clim = monthly.groupby('CalMonth')['Value'].mean()
        monthly['Clim'] = monthly['CalMonth'].map(clim)
        monthly['Anomaly_pct'] = (
            (monthly['Value'] - monthly['Clim']) / (monthly['Clim'].abs() + 1e-12) * 100
        )

        latest = monthly.iloc[-1]
        current_pct = float(latest['Anomaly_pct'])
        level, label = _flow_level(current_pct)

        # Percentile of current value in historical distribution
        hist_vals = monthly['Value'].values
        current_val = float(latest['Value'])
        percentile = float(np.mean(hist_vals <= current_val) * 100)

        # Trend: slope of last 6 months
        recent = monthly['Value'].iloc[-6:].values
        if len(recent) >= 3:
            slope = float(np.polyfit(range(len(recent)), recent, 1)[0])
            trend = 'rising' if slope > 0.01 * abs(recent.mean()) else (
                'falling' if slope < -0.01 * abs(recent.mean()) else 'stable'
            )
        else:
            trend = 'stable'

        # Thin series for frontend
        thin_idx = np.round(np.linspace(0, len(monthly) - 1, min(120, len(monthly)))).astype(int)
        series_out = [
            {
                'date': str(monthly['Timestamp'].iloc[i].date()),
                'anomaly_pct': round(float(monthly['Anomaly_pct'].iloc[i]), 1),
                'value': round(float(monthly['Value'].iloc[i]), 3),
            }
            for i in thin_idx
        ]

        return {
            'anomaly_pct': round(current_pct, 1),
            'current_value': round(current_val, 3),
            'climatology_mean': round(float(latest['Clim']), 3),
            'percentile': round(percentile, 1),
            'trend': trend,
            'level': level,
            'label': label,
            'feature': feature,
            'unit': unit,
            'latest_date': str(latest['Timestamp'].date()),
            'series': series_out,
        }

    # ── Combined entry point ─────────────────────────────────────────────────

    def compute_for_station(self, station: str) -> Dict[str, Any]:
        """
        Compute all available indices for a station.
        Gracefully skips indices when data/features are unavailable.
        """
        result: Dict[str, Any] = {
            'station': station,
            'spi': None,
            'flow': None,
            'worst_level': 'normal',
            'errors': [],
        }

        rain_feature = self._find_rain_feature(station)
        flow_feature = self._find_flow_feature(station)

        if rain_feature:
            try:
                result['spi'] = self.compute_spi(station, rain_feature)
            except Exception as exc:
                result['errors'].append(f'SPI: {exc}')

        if flow_feature:
            try:
                result['flow'] = self.compute_flow_anomaly(station, flow_feature)
            except Exception as exc:
                result['errors'].append(f'Flow: {exc}')

        levels: List[str] = []
        if result['spi']:
            levels.append(result['spi']['level'])
        if result['flow']:
            levels.append(result['flow']['level'])

        result['worst_level'] = _worst_level(*levels) if levels else 'normal'
        return result
