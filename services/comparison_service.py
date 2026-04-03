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

import markdown
import numpy as np
import pandas as pd

from .analysis_service import _gemini_generate
from .data_loader import DataRepository, MultiDataRepository, SeriesRequest


def _generate_component_analysis(component: str, data: Dict[str, Any], feature: str) -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    fallback = _fallback_component_analysis(component, data, feature)
    if not api_key or api_key == 'your_google_gemini_api_key_here' or not api_key.strip():
        return fallback
    try:
        prompt = _component_prompt(component, data, feature)
        return markdown.markdown(_gemini_generate(api_key, prompt))
    except Exception:
        return _fallback_component_analysis(component, data, feature)


def _component_prompt(component: str, data: Dict[str, Any], feature: str) -> str:
    feature_label = feature.replace('_', ' ')
    if component == 'correlation':
        return (
            'Act as a professional hydrologist interpreting a basin correlation matrix.\n\n'
            'RESPONSE FORMAT (STRICT):\n'
            'Use markdown with exactly these sections:\n'
            '## Matrix Overview\n2-3 sentences.\n'
            '## Strongest Relationships\nExactly 4 bullet points.\n'
            '## Spatial Interpretation\nExactly 4 bullet points.\n'
            '## Operational Implications\nExactly 3 bullet points.\n\n'
            'RULES:\n'
            '- Cite specific correlation values.\n'
            '- Replace underscores with spaces.\n'
            '- Focus on cross-station coherence and what it implies hydrologically.\n'
            '- No intro before the first heading and no conclusion after the last bullet.\n\n'
            f'Feature: {feature_label}\n\n'
            f'{_format_correlation_for_prompt(data)}\n'
        )
    if component == 'leaderboard':
        return (
            'Act as a professional hydrologist interpreting a basin anomaly leaderboard.\n\n'
            'RESPONSE FORMAT (STRICT):\n'
            'Use markdown with exactly these sections:\n'
            '## Year Context\n2-3 sentences.\n'
            '## Highest Anomaly Stations\nExactly 4 bullet points.\n'
            '## Basin Balance\nExactly 4 bullet points.\n'
            '## Operational Implications\nExactly 3 bullet points.\n\n'
            'RULES:\n'
            '- Cite anomaly percentages and annual mean versus climatology.\n'
            '- Replace underscores with spaces.\n'
            '- Distinguish above-normal and below-normal conditions clearly.\n'
            '- No intro before the first heading and no conclusion after the last bullet.\n\n'
            f'Feature: {feature_label}\n\n'
            f'{_format_leaderboard_for_prompt((data or {}).get("rows", []))}\n'
            f'\nMetadata: year={data.get("year")}, above_normal={data.get("above_normal")}, below_normal={data.get("below_normal")}, total_stations={data.get("total_stations")}\n'
        )
    return (
        'Act as a professional hydrologist interpreting basin summary statistics.\n\n'
        'RESPONSE FORMAT (STRICT):\n'
        'Use markdown with exactly these sections:\n'
        '## Basin Snapshot\n2-3 sentences.\n'
        '## Distribution Structure\nExactly 4 bullet points.\n'
        '## Station Extremes\nExactly 4 bullet points.\n'
        '## Operational Implications\nExactly 3 bullet points.\n\n'
        'RULES:\n'
        '- Cite the reported summary statistics.\n'
        '- Replace underscores with spaces.\n'
        '- Comment on spread, percentiles, extremes, and trends.\n'
        '- No intro before the first heading and no conclusion after the last bullet.\n\n'
        f'Feature: {feature_label}\n\n'
        f'{_format_summary_for_prompt(data)}\n'
    )


def _format_correlation_for_prompt(corr: Dict[str, Any]) -> str:
    if not corr:
        return 'No correlation matrix available.'
    stations = corr.get('stations', [])
    matrix = corr.get('matrix', [])
    mean_corrs = corr.get('mean_correlations', [])
    pairs: List[tuple[float, str, str]] = []
    for i in range(len(stations)):
        for j in range(i + 1, len(stations)):
            v = matrix[i][j] if i < len(matrix) and j < len(matrix[i]) else None
            if v is not None:
                pairs.append((float(v), stations[i], stations[j]))
    pairs.sort(key=lambda x: abs(x[0]), reverse=True)
    top_pairs = pairs[:5]
    top_station = None
    if stations and mean_corrs:
        ranked = [(mean_corrs[i], stations[i]) for i in range(min(len(stations), len(mean_corrs))) if mean_corrs[i] is not None]
        ranked.sort(reverse=True)
        if ranked:
            top_station = ranked[0]
    lines = [
        f"Dataset {corr.get('dataset')} with {corr.get('n_stations')} stations for {corr.get('feature', '')}.",
        f"Capped subset applied: {'yes' if corr.get('capped') else 'no'}; total available stations {corr.get('total_available')}.",
    ]
    if top_station:
        lines.append(f"Highest mean cross-station correlation: {top_station[1]} at {top_station[0]:.3f}.")
    if top_pairs:
        lines.append('Top correlation pairs:')
        lines.extend(f'  - {a} vs {b}: r={v:.3f}' for v, a, b in top_pairs)
    return '\n'.join(lines)


def _format_leaderboard_for_prompt(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return 'No anomaly leaderboard available.'
    lines = ['Top anomaly stations:']
    for row in rows[:8]:
        lines.append(
            f"  - {row['name']}: {row['anomaly_pct']:+.1f}% ({row['direction']} normal), "
            f"year mean {row['year_mean']} {row['unit']} vs climatology {row['clim_mean']} {row['unit']}, level {row['level']}"
        )
    return '\n'.join(lines)


def _format_summary_for_prompt(summary: Dict[str, Any]) -> str:
    if not summary:
        return 'No basin summary available.'
    trends = summary.get('trends', {})
    highest = summary.get('highest_station', {})
    lowest = summary.get('lowest_station', {})
    return (
        f"Dataset: {summary.get('dataset')}\n"
        f"Feature: {summary.get('feature')}\n"
        f"Active stations: {summary.get('active_stations')} of {summary.get('total_stations')}\n"
        f"Basin mean/median/std: {summary.get('basin_mean')} / {summary.get('basin_median')} / {summary.get('basin_std')} {summary.get('unit')}\n"
        f"Range: {summary.get('basin_min')} to {summary.get('basin_max')} {summary.get('unit')}\n"
        f"Percentiles: P10 {summary.get('p10')}, P25 {summary.get('p25')}, P75 {summary.get('p75')}, P90 {summary.get('p90')} {summary.get('unit')}\n"
        f"Spatial CV: {summary.get('spatial_cv_pct')}%\n"
        f"Highest station: {highest.get('name')} ({highest.get('mean')} {summary.get('unit')})\n"
        f"Lowest station: {lowest.get('name')} ({lowest.get('mean')} {summary.get('unit')})\n"
        f"Trend counts: rising {trends.get('rising', 0)}, stable {trends.get('stable', 0)}, falling {trends.get('falling', 0)}\n"
        f"Average imputation: {summary.get('avg_imputation_pct')}%\n"
        f"Total observations: {summary.get('total_observations')}"
    )


def _fallback_component_analysis(component: str, data: Dict[str, Any], feature: str, note: str | None = None) -> str:
    parts: List[str] = []
    feature_label = feature.replace('_', ' ')
    if component == 'correlation':
        corr = data or {}
        parts.append('## Matrix Overview')
        overview = [f'Correlation analysis for **{feature_label}** across the selected basin.']
        if note:
            overview.append(note)
        overview.append(f"The matrix includes {corr.get('n_stations', 0)} station(s) from the {corr.get('dataset', '')} dataset.")
        parts.append(' '.join(overview))
        parts.append('## Strongest Relationships')
        stations = corr.get('stations', [])
        matrix = corr.get('matrix', [])
        pairs: List[tuple[float, str, str]] = []
        for i in range(len(stations)):
            for j in range(i + 1, len(stations)):
                v = matrix[i][j] if i < len(matrix) and j < len(matrix[i]) else None
                if v is not None:
                    pairs.append((float(v), stations[i], stations[j]))
        for value, left, right in sorted(pairs, key=lambda x: abs(x[0]), reverse=True)[:4]:
            parts.append(f'- **{left} vs {right}:** correlation is {value:.3f}, indicating {"strong" if abs(value) >= 0.7 else "moderate"} spatial coherence.')
        parts.append('## Spatial Interpretation')
        mean_corrs = corr.get('mean_correlations', [])
        ranked = [(mean_corrs[i], stations[i]) for i in range(min(len(stations), len(mean_corrs))) if mean_corrs[i] is not None]
        ranked.sort(reverse=True)
        if ranked:
            parts.append(f'- **Most representative station:** {ranked[0][1]} has the highest mean pairwise correlation at {ranked[0][0]:.3f}.')
        parts.append(f'- **Coverage:** capped subset applied: {"yes" if corr.get("capped") else "no"}; total available stations {corr.get("total_available")}.')
        parts.append('- **Interpretation:** stronger positive correlations imply shared basin forcing or synchronized routing behaviour.')
        parts.append('- **Caution:** weak pairs can signal local regulation, tributary independence, or data heterogeneity.')
        parts.append('## Operational Implications')
        parts.append('- **Monitoring:** highly coherent stations are suitable candidates for basin-wide proxy monitoring.')
        parts.append('- **Risk messaging:** weakly aligned subregions should not be represented by one station alone during events.')
        parts.append('- **Model use:** correlation structure can guide regionalization and transferability assumptions.')
        return markdown.markdown('\n'.join(parts))

    if component == 'leaderboard':
        rows = (data or {}).get('rows', [])
        parts.append('## Year Context')
        context = [f'Anomaly leaderboard for **{feature_label}** in {(data or {}).get("year", "the selected year")}.']
        if note:
            context.append(note)
        context.append(
            f"{(data or {}).get('above_normal', 0)} station(s) are above normal and {(data or {}).get('below_normal', 0)} below normal out of {(data or {}).get('total_stations', 0)} ranked station(s)."
        )
        parts.append(' '.join(context))
        parts.append('## Highest Anomaly Stations')
        for row in rows[:4]:
            parts.append(
                f"- **{row['name']}:** {row['anomaly_pct']:+.1f}% relative to climatology, annual mean {row['year_mean']} {row['unit']} versus {row['clim_mean']} {row['unit']} normal, classified as {row['level']}."
            )
        parts.append('## Basin Balance')
        parts.append(f"- **Sign balance:** above-normal stations {(data or {}).get('above_normal', 0)} versus below-normal stations {(data or {}).get('below_normal', 0)}.")
        if rows:
            parts.append(f"- **Largest departure:** {rows[0]['name']} shows the strongest absolute anomaly at {abs(rows[0]['anomaly_pct']):.1f}%.")
        parts.append('- **Interpretation:** mixed signs indicate spatially uneven hydroclimatic forcing rather than a uniform basin-wide shift.')
        parts.append('- **Caution:** anomaly ranking is relative to climatology, not an absolute severity threshold for all stations.')
        parts.append('## Operational Implications')
        parts.append('- **Hotspot prioritization:** warning and critical stations should be checked first for localized stress or surplus conditions.')
        parts.append('- **Narrative control:** separate above-normal and below-normal clusters when communicating basin conditions.')
        parts.append('- **Planning use:** the leaderboard is best used for triage and station targeting, not as a substitute for full hydrograph review.')
        return markdown.markdown('\n'.join(parts))

    summary = data or {}
    parts.append('## Basin Snapshot')
    overview = [f'Basin summary statistics for **{feature_label}** across the selected dataset.']
    if note:
        overview.append(note)
    overview.append(
        f"{summary.get('active_stations', 0)} active station(s) contribute to a basin mean of {summary.get('basin_mean')} {summary.get('unit', '')} and spatial CV of {summary.get('spatial_cv_pct')}%."
    )
    parts.append(' '.join(overview))
    parts.append('## Distribution Structure')
    parts.append(f"- **Central tendency:** mean {summary.get('basin_mean')} {summary.get('unit')} and median {summary.get('basin_median')} {summary.get('unit')}.")
    parts.append(f"- **Spread:** standard deviation {summary.get('basin_std')} {summary.get('unit')} across a range of {summary.get('basin_min')} to {summary.get('basin_max')} {summary.get('unit')}.")
    parts.append(f"- **Percentiles:** P10 {summary.get('p10')}, P25 {summary.get('p25')}, P75 {summary.get('p75')}, P90 {summary.get('p90')} {summary.get('unit')}.")
    parts.append(f"- **Data quality:** average imputation is {summary.get('avg_imputation_pct')}% across {summary.get('total_observations')} observations.")
    parts.append('## Station Extremes')
    parts.append(f"- **Highest station:** {summary.get('highest_station', {}).get('name')} at {summary.get('highest_station', {}).get('mean')} {summary.get('unit')}.")
    parts.append(f"- **Lowest station:** {summary.get('lowest_station', {}).get('name')} at {summary.get('lowest_station', {}).get('mean')} {summary.get('unit')}.")
    if summary.get('trends_computed'):
        trends = summary.get('trends', {})
        parts.append(f"- **Trend mix:** rising {trends.get('rising', 0)}, stable {trends.get('stable', 0)}, falling {trends.get('falling', 0)}.")
    parts.append('- **Interpretation:** wide separation between upper and lower stations suggests strong spatial heterogeneity within the basin.')
    parts.append('## Operational Implications')
    parts.append('- **Network planning:** the distribution spread indicates whether basin management should rely on regional subgroups rather than one basin-average narrative.')
    parts.append('- **Benchmarking:** percentile bands provide a useful baseline for flagging unusually high or low stations in follow-up diagnostics.')
    parts.append('- **Decision use:** combine summary statistics with station-level anomalies before making operational statements.')
    return markdown.markdown('\n'.join(parts))


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

    def with_component_analysis(self, component: str, data: Dict[str, Any], feature: str) -> Dict[str, Any]:
        enriched = dict(data)
        enriched['analysis'] = _generate_component_analysis(component, data, feature)
        return enriched

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

        if resolved_year is None:
            end_date_str = repo.global_time_extent.get('end')
            if end_date_str:
                end_date = pd.to_datetime(end_date_str)
                resolved_year = end_date.year if end_date.month == 12 else end_date.year - 1
            else:
                resolved_year = 2020

        for station_name, _, meta in all_info:
            try:
                monthly = self._monthly_series(repo, station_name, feature).dropna().reset_index()
                monthly.columns = ['Timestamp', 'Value']
                monthly['Year'] = monthly['Timestamp'].dt.year
                monthly['CalMonth'] = monthly['Timestamp'].dt.month

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
                component_result = fn()
                if include_analysis:
                    component_result = self.with_component_analysis(key, component_result, feature)
                result[key] = component_result
            except Exception as exc:
                result['errors'].append(f'{key}: {exc}')
        return result
