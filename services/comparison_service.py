"""
comparison_service.py
~~~~~~~~~~~~~~~~~~~~~
Basin-level comparison analytics:
  1. Correlation matrix  — Pearson r between all station pairs for one feature
  2. Anomaly leaderboard — stations ranked by deviation from climatology for a year
  3. Basin-wide summary  — aggregated statistics across all stations
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_loader import DataRepository, MultiDataRepository, SeriesRequest


def _generate_comparison_analysis(result: Dict[str, Any], feature: str) -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return ''
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        summary = result.get('summary', {})
        leaderboard = (result.get('leaderboard') or {}).get('rows', [])[:5]
        prompt = f"""Analyze this basin comparison result and provide 3 concise bullet-point insights:

Feature analyzed: {feature}
Basin summary: {summary}
Top anomaly stations: {leaderboard}

Focus on: spatial patterns, stations with highest anomalies, and what the basin-wide trends suggest for water resources. Use **bold** for key terms."""
        resp = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
        return resp.text.strip()
    except Exception:
        return ''


class ComparisonService:
    # Max stations included in the correlation matrix (N² cost)
    CORR_CAP = {'mekong': 65, 'lamah': 50}
    # Skip per-station trend for large datasets to keep summary fast
    TREND_CAP = 200

    def __init__(self, repository: MultiDataRepository) -> None:
        self.repository = repository

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _repo_for(self, dataset: str) -> DataRepository:
        for repo in self.repository.repos:
            if repo.dataset == dataset:
                return repo
        raise ValueError(f'Unknown dataset: {dataset!r}')

    def _stations_for_feature(
        self, dataset: str, feature: str
    ) -> List[Tuple[str, int, Dict]]:
        """
        Return [(station_id, observations, meta)] sorted by observations desc.
        Uses pre-built index — no CSV reads.
        """
        repo = self._repo_for(dataset)
        result = []
        for name, meta in repo.station_index.items():
            fd = meta['feature_details'].get(feature)
            if fd:
                result.append((name, int(fd['observations']), meta))
        return sorted(result, key=lambda x: -x[1])

    def _monthly_series(self, repo: DataRepository, station: str, feature: str) -> pd.Series:
        meta = repo.station_index[station]
        fd = meta['feature_details'][feature]
        req = SeriesRequest(
            station=station, feature=feature,
            start_date=fd['start_date'], end_date=fd['end_date'],
        )
        df = repo.get_feature_series(req)
        return (
            df.set_index('Timestamp')['Value']
            .resample('MS').mean()
        )

    # ── 1. Correlation matrix ────────────────────────────────────────────────

    def compute_correlation_matrix(self, dataset: str, feature: str) -> Dict[str, Any]:
        """
        Pearson correlation matrix across all stations (capped for LamaH).
        Resamples every station to monthly means then calls DataFrame.corr().
        """
        repo = self._repo_for(dataset)
        cap = self.CORR_CAP.get(dataset, 65)
        all_info = self._stations_for_feature(dataset, feature)
        total_available = len(all_info)

        if total_available < 2:
            raise ValueError(f'Need at least 2 stations with {feature} data in {dataset}.')

        selected = all_info[:cap]
        unit = repo.feature_units.get(feature, '')

        # Build aligned monthly DataFrame
        frames: Dict[str, pd.Series] = {}
        for name, _, _ in selected:
            try:
                frames[name] = self._monthly_series(repo, name, feature)
            except Exception:
                pass

        if len(frames) < 2:
            raise ValueError('Could not load data for enough stations.')

        combined = pd.DataFrame(frames)
        corr = combined.corr(method='pearson', min_periods=12)

        station_ids = list(corr.columns)
        pretty = [
            (repo.station_index[s].get('name', s) or s).replace('_', ' ')
            for s in station_ids
        ]
        matrix = [
            [None if pd.isna(v) else round(float(v), 3) for v in row]
            for row in corr.values.tolist()
        ]

        # Compute mean correlation per station (excluding self)
        mean_corrs = []
        n = len(station_ids)
        for i in range(n):
            others = [matrix[i][j] for j in range(n) if i != j and matrix[i][j] is not None]
            mean_corrs.append(round(float(np.mean(others)), 3) if others else None)

        return {
            'stations': pretty,
            'station_ids': station_ids,
            'matrix': matrix,
            'mean_correlations': mean_corrs,
            'feature': feature,
            'unit': unit,
            'dataset': dataset,
            'n_stations': len(station_ids),
            'capped': len(selected) < total_available,
            'total_available': total_available,
        }

    # ── 2. Anomaly leaderboard ───────────────────────────────────────────────

    def compute_anomaly_leaderboard(
        self,
        dataset: str,
        feature: str,
        year: Optional[int] = None,
        top_n: int = 25,
    ) -> Dict[str, Any]:
        """
        Rank all stations by |anomaly %| for the given year vs long-run climatology.
        If year is None, the most recent complete calendar year is used.
        """
        repo = self._repo_for(dataset)
        all_info = self._stations_for_feature(dataset, feature)
        unit = repo.feature_units.get(feature, '')

        rows: List[Dict] = []
        resolved_year: Optional[int] = year

        for station_name, _, meta in all_info:
            try:
                monthly = self._monthly_series(repo, station_name, feature).dropna().reset_index()
                monthly.columns = ['Timestamp', 'Value']
                monthly['Year'] = monthly['Timestamp'].dt.year
                monthly['CalMonth'] = monthly['Timestamp'].dt.month

                # Resolve target year once from the first station if not provided
                if resolved_year is None:
                    year_counts = monthly.groupby('Year')['Value'].count()
                    complete = year_counts[year_counts >= 9].index
                    if len(complete) == 0:
                        continue
                    resolved_year = int(complete[-1])

                year_data = monthly[monthly['Year'] == resolved_year]
                if len(year_data) < 6:
                    continue

                year_mean = float(year_data['Value'].mean())
                # Climatology = mean of all calendar-month averages (avoids seasonal bias)
                clim_mean = float(monthly.groupby('CalMonth')['Value'].mean().mean())
                if abs(clim_mean) < 1e-10:
                    continue

                anomaly_pct = (year_mean - clim_mean) / abs(clim_mean) * 100

                abs_pct = abs(anomaly_pct)
                level = (
                    'critical' if abs_pct >= 70 else
                    'warning'  if abs_pct >= 40 else
                    'watch'    if abs_pct >= 20 else
                    'normal'
                )
                display_name = (meta.get('name', station_name) or station_name).replace('_', ' ')

                rows.append({
                    'station': station_name,
                    'name': display_name,
                    'anomaly_pct': round(anomaly_pct, 1),
                    'year_mean': round(year_mean, 3),
                    'clim_mean': round(clim_mean, 3),
                    'level': level,
                    'direction': 'above' if anomaly_pct >= 0 else 'below',
                    'unit': unit,
                })
            except Exception:
                continue

        rows.sort(key=lambda r: abs(r['anomaly_pct']), reverse=True)

        return {
            'year': resolved_year,
            'rows': rows[:top_n],
            'total_stations': len(rows),
            'above_normal': sum(1 for r in rows if r['direction'] == 'above'),
            'below_normal': sum(1 for r in rows if r['direction'] == 'below'),
            'feature': feature,
            'unit': unit,
            'dataset': dataset,
        }

    # ── 3. Basin-wide summary ────────────────────────────────────────────────

    def compute_basin_summary(self, dataset: str, feature: str) -> Dict[str, Any]:
        """
        Aggregate statistics across all stations using the pre-built index
        (no CSV loads for basic stats) plus optional trend from CSVs.
        """
        repo = self._repo_for(dataset)
        all_info = self._stations_for_feature(dataset, feature)
        unit = repo.feature_units.get(feature, '')

        if not all_info:
            raise ValueError(f'No stations with {feature} data in {dataset}.')

        # ── Basic stats from pre-built index (fast, no CSV) ──────────────────
        station_means: List[Tuple[str, str, float]] = []
        total_obs = 0
        total_imputed = 0

        for name, obs, meta in all_info:
            fd = meta['feature_details'].get(feature, {})
            mean_val = fd.get('mean')
            if mean_val is None:
                continue
            display = (meta.get('name', name) or name).replace('_', ' ')
            station_means.append((name, display, float(mean_val)))
            total_obs += int(fd.get('observations', 0))
            total_imputed += int(fd.get('imputed_points', 0))

        if not station_means:
            raise ValueError('No mean values available for any station.')

        means = [m for _, _, m in station_means]
        mean_of_means = float(np.mean(means))
        std_of_means = float(np.std(means))
        spatial_cv = round(std_of_means / (abs(mean_of_means) + 1e-12) * 100, 1)

        sorted_by_mean = sorted(station_means, key=lambda x: -x[2])
        highest = sorted_by_mean[0]
        lowest = sorted_by_mean[-1]

        avg_imputation_pct = round(total_imputed / max(total_obs, 1) * 100, 1)

        # ── Trend computation (from CSVs, skipped for very large datasets) ───
        trends = {'rising': 0, 'stable': 0, 'falling': 0}
        compute_trends = len(all_info) <= self.TREND_CAP

        if compute_trends:
            for name, _, meta in all_info:
                try:
                    monthly = self._monthly_series(repo, name, feature).dropna()
                    if len(monthly) < 12:
                        continue
                    x = np.arange(len(monthly))
                    slope = float(np.polyfit(x, monthly.values, 1)[0])
                    rel = slope / (abs(monthly.mean()) + 1e-12)
                    if rel > 0.001:
                        trends['rising'] += 1
                    elif rel < -0.001:
                        trends['falling'] += 1
                    else:
                        trends['stable'] += 1
                except Exception:
                    continue

        # ── Distribution histogram (30 bins) ─────────────────────────────────
        counts, edges = np.histogram(means, bins=min(30, len(means)))

        # ── Percentile bands ─────────────────────────────────────────────────
        p10 = float(np.percentile(means, 10))
        p25 = float(np.percentile(means, 25))
        p75 = float(np.percentile(means, 75))
        p90 = float(np.percentile(means, 90))

        return {
            'dataset': dataset,
            'feature': feature,
            'unit': unit,
            'active_stations': len(station_means),
            'total_stations': len(all_info),
            'basin_mean': round(mean_of_means, 3),
            'basin_median': round(float(np.median(means)), 3),
            'basin_std': round(std_of_means, 3),
            'basin_min': round(float(np.min(means)), 3),
            'basin_max': round(float(np.max(means)), 3),
            'p10': round(p10, 3),
            'p25': round(p25, 3),
            'p75': round(p75, 3),
            'p90': round(p90, 3),
            'spatial_cv_pct': spatial_cv,
            'highest_station': {'id': highest[0], 'name': highest[1], 'mean': round(highest[2], 3)},
            'lowest_station': {'id': lowest[0], 'name': lowest[1], 'mean': round(lowest[2], 3)},
            'trends': trends,
            'trends_computed': compute_trends,
            'avg_imputation_pct': avg_imputation_pct,
            'total_observations': total_obs,
            'histogram': {
                'counts': counts.tolist(),
                'edges': [round(float(e), 3) for e in edges.tolist()],
            },
        }

    # ── Combined entry point ─────────────────────────────────────────────────

    def compare(self, dataset: str, feature: str, year: Optional[int] = None, include_analysis: bool = False) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            'correlation': None,
            'leaderboard': None,
            'summary': None,
            'errors': [],
        }
        for key, fn in [
            ('correlation', lambda: self.compute_correlation_matrix(dataset, feature)),
            ('leaderboard', lambda: self.compute_anomaly_leaderboard(dataset, feature, year)),
            ('summary',     lambda: self.compute_basin_summary(dataset, feature)),
        ]:
            try:
                result[key] = fn()
            except Exception as exc:
                result['errors'].append(f'{key}: {exc}')
        if include_analysis:
            analysis = _generate_comparison_analysis(result, feature)
            if analysis:
                result['analysis'] = analysis
        return result
