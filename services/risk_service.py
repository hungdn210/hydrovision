from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import markdown
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io
import scipy.stats

from .analysis_service import _gemini_generate
from .data_loader import SeriesRequest
from .feature_registry import get_valid_features_for_analysis
from .figure_theme import (
    DARK_BG, TEXT,
    legend_v, MARGIN_MAP,
)


def _fallback_risk_analysis(result: Dict[str, Any]) -> str:
    summary = result.get('summary', {}) or {}
    total = max(int(result.get('n_stations') or 0), 1)
    flood = int(summary.get('flood', 0))
    flood_watch = int(summary.get('flood_watch', 0))
    normal = int(summary.get('normal', 0))
    drought = int(summary.get('drought', 0))
    severe_drought = int(summary.get('severe_drought', 0))
    at_risk = flood + flood_watch + drought + severe_drought
    flood_pct = (flood + flood_watch) / total * 100
    drought_pct = (drought + severe_drought) / total * 100
    at_risk_pct = at_risk / total * 100
    normal_pct = normal / total * 100
    feature = str(result.get('feature', '')).replace('_', ' ')
    dataset = str(result.get('dataset', '')).replace('_', ' ')
    lookback = result.get('lookback')
    mode = result.get('percentile_mode', 'seasonal')
    mode_label = 'seasonal calendar-month' if mode == 'seasonal' else 'full-record'

    # Determine dominant signal
    if flood_pct > drought_pct and flood_pct > 20:
        dominant_signal = 'flood-biased'
        signal_note = 'indicating above-normal hydrological conditions across a significant portion of the basin'
    elif drought_pct > flood_pct and drought_pct > 20:
        dominant_signal = 'drought-biased'
        signal_note = 'pointing to widespread below-normal moisture deficits across the monitored network'
    elif at_risk_pct < 15:
        dominant_signal = 'near-normal'
        signal_note = 'suggesting the basin is currently experiencing relatively stable hydrological conditions'
    else:
        dominant_signal = 'mixed'
        signal_note = 'reflecting spatially heterogeneous forcing with no single dominant direction'

    parts = []
    parts.append(
        f'<p><strong>Executive Summary</strong></p>'
        f'<p>The flood and drought risk screening for <strong>{feature}</strong> across the <strong>{dataset}</strong> dataset '
        f'classified {total} stations against their own historical {mode_label} percentile distributions using a {lookback}-point recent window. '
        f'The basin currently presents a <strong>{dominant_signal}</strong> signal, {signal_note}. '
        f'Overall, {at_risk} stations ({at_risk_pct:.1f}%) are classified outside the normal band: '
        f'{flood + flood_watch} are in elevated flood conditions ({flood_pct:.1f}%) and {drought + severe_drought} are in drought conditions ({drought_pct:.1f}%). '
        f'The remaining {normal} stations ({normal_pct:.1f}%) fall within the P20–P80 normal range, indicating no immediate operational concern for those locations.</p>'
    )

    parts.append('<p><strong>Risk Classification Breakdown</strong></p><ul>')
    parts.append(
        f'<li>Flood Risk (&gt;P95): {flood} station(s) — these are experiencing values exceeding the 95th percentile of historical observations, '
        f'a threshold associated with high-flow events that may require immediate hydrological attention.</li>'
    )
    parts.append(
        f'<li>Flood Watch (P80–P95): {flood_watch} station(s) — elevated but sub-critical conditions; '
        f'continued monitoring is recommended, as conditions could escalate during prolonged precipitation events.</li>'
    )
    parts.append(
        f'<li>Normal (P20–P80): {normal} station(s) — values within the expected seasonal range for the current period; '
        f'no immediate hydrological stress is indicated at these locations.</li>'
    )
    parts.append(
        f'<li>Drought (P5–P20): {drought} station(s) — below-normal conditions that may indicate reduced groundwater recharge, '
        f'lower baseflow contributions, or declining soil moisture reserves.</li>'
    )
    parts.append(
        f'<li>Severe Drought (&lt;P5): {severe_drought} station(s) — critically low values in the lowest 5th percentile of the historical record, '
        f'consistent with severe water deficit conditions requiring priority management attention.</li>'
    )
    parts.append('</ul>')

    parts.append('<p><strong>Threshold Methodology</strong></p><ul>')
    if mode == 'seasonal':
        parts.append(
            '<li>Seasonal percentile mode: thresholds are computed from the same calendar-month subset of the historical record for each station. '
            'This removes the confounding effect of the seasonal discharge cycle, so a normal wet-season value does not erroneously appear as Flood Watch when compared to the full-year distribution. '
            'This approach aligns with WMO flood early-warning operational guidelines.</li>'
        )
    else:
        parts.append(
            '<li>Full-record percentile mode: thresholds are derived from the complete historical record without seasonal stratification. '
            'This can cause seasonal bias — wet-season values may appear elevated relative to the full-year distribution even when conditions are normal for that time of year.</li>'
        )
    parts.append(
        f'<li>Lookback window: the most recent {lookback} data points are averaged to represent current conditions, '
        'smoothing out short-term noise while preserving medium-term hydrological signals.</li>'
    )
    parts.append(
        '<li>Classification boundaries: the five-tier system (Severe Drought / Drought / Normal / Flood Watch / Flood Risk) '
        'uses P5, P20, P80, and P95 breakpoints, which are standard operational thresholds in hydrological monitoring practice.</li>'
    )
    parts.append('</ul>')

    parts.append('<p><strong>Operational Relevance</strong></p><ul>')
    parts.append(
        '<li>Basin triage: use the map as a rapid screening layer to identify priority stations for follow-up analysis. '
        'Stations in Flood Risk or Severe Drought categories should be reviewed first against their full hydrographs before any operational decisions are made.</li>'
    )
    parts.append(
        '<li>Contextual confirmation: risk classification is statistical, not causal. '
        'Confirm any alert-level station with local catchment context, recent rainfall records, and reservoir operations data before escalating to water management authorities.</li>'
    )
    parts.append(
        '<li>Temporal limitation: this screening reflects conditions at the time of computation based on the available record window. '
        'Re-running the analysis after new data ingestion is recommended during active weather events or prolonged dry spells.</li>'
    )
    parts.append('</ul>')

    return ''.join(parts)


def _generate_risk_analysis(result: Dict[str, Any]) -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_google_gemini_api_key_here' or not api_key.strip():
        return _fallback_risk_analysis(result)
    try:
        summary = result.get('summary', {})
        total = max(int(result.get('n_stations') or 0), 1)
        flood = int(summary.get('flood', 0))
        flood_watch = int(summary.get('flood_watch', 0))
        normal = int(summary.get('normal', 0))
        drought = int(summary.get('drought', 0))
        severe_drought = int(summary.get('severe_drought', 0))
        flood_pct = (flood + flood_watch) / total * 100
        drought_pct = (drought + severe_drought) / total * 100
        prompt = f"""Act as a professional hydrologist interpreting a basin-wide flood and drought risk-screening map.
Write the response in markdown and structure it exactly as follows — no text outside these headings:

## Executive Summary
Write 4–5 sentences. State the dominant signal (flood-biased / drought-biased / mixed / near-normal), cite the percentage of at-risk stations, characterise the severity tier breakdown, interpret what this pattern means for basin water availability, and note one key limitation the user should be aware of.

## Risk Classification Breakdown
Exactly 5 bullet points — one per risk tier (Flood Risk, Flood Watch, Normal, Drought, Severe Drought). For each: cite the station count and percentage, interpret the hydrological meaning, and state whether this represents an operationally significant signal.

## Spatial Risk Pattern
Exactly 4 bullet points interpreting spatial coherence: are at-risk stations concentrated in one sub-region or scattered basin-wide? What does clustering imply about forcing mechanisms (e.g., monsoon, ENSO, regulation)? Note any asymmetry between the flood and drought signals spatially.

## Threshold Methodology
Exactly 3 bullet points explaining: (1) how the seasonal/full-record percentile thresholds work, (2) why P5/P20/P80/P95 boundaries were chosen, (3) what the lookback window represents and its effect on the classification.

## Operational Relevance
Exactly 3 bullet points: (1) how water managers should prioritise stations based on tier, (2) what confirmatory steps are needed before operational action, (3) one important caveat about statistical risk classification versus causal diagnosis.

Rules:
- Use professional hydrological language throughout.
- Always cite specific numbers, counts, and percentages from the provided data.
- Replace all underscores with spaces.
- No introductory text before the first heading; no conclusion after the last bullet.

Feature: {str(result.get('feature', '')).replace('_', ' ')}
Dataset: {str(result.get('dataset', '')).replace('_', ' ')}
Lookback: {result.get('lookback')} data points
Percentile mode: {result.get('percentile_mode', 'seasonal')}
Total stations: {total}
Flood Risk: {flood} ({flood_pct:.1f}%), Flood Watch: {flood_watch}, Normal: {normal}, Drought: {drought}, Severe Drought: {severe_drought} ({drought_pct:.1f}%)
Full summary: {summary}
"""
        return markdown.markdown(_gemini_generate(api_key, prompt))
    except Exception:
        return _fallback_risk_analysis(result)


class RiskService:
    """
    Flood and drought risk classification for all stations in a dataset.

    Methodology:
      - Load full historical series for each station that has the target feature.
      - Compute percentile thresholds (5th, 20th, 80th, 95th) from the historical
        record (full-record mode) or from the same calendar-month subset
        (seasonal mode, default).
      - Take the mean of the most recent `lookback` data points as the "current" value.
      - Classify the percentile rank of the current value:
          < 5th  → Severe Drought
          5–20th → Drought
          20–80th→ Normal
          80–95th→ Flood Watch
          > 95th → Flood Risk
      - Return station list + Plotly Scattermapbox figure.

    Seasonal mode (default):
      Percentile thresholds are computed from the same calendar-month subset
      of the historical record.  This removes the seasonal confound that causes
      normal wet-season values to appear as "Flood Watch" when compared against
      the full-year distribution.  Follows standard operational hydrological
      monitoring practice (WMO flood early-warning guidelines).
    """

    # (key, low_pct, high_pct, color, label) — ordered high-to-low for legend
    RISK_LEVELS: List[tuple] = [
        ('flood',          95, 100, '#f87171', 'Flood Risk'),
        ('flood_watch',    80,  95, '#60a5fa', 'Flood Watch'),
        ('normal',         20,  80, '#34d399', 'Normal'),
        ('drought',         5,  20, '#fb923c', 'Drought'),
        ('severe_drought',  0,   5, '#b91c1c', 'Severe Drought'),
    ]

    def __init__(self, repository) -> None:
        self.repo = repository

    # ── helpers ──────────────────────────────────────────────────────────────

    def _find_repo_for_dataset(self, dataset: str):
        if hasattr(self.repo, 'repos'):
            return next(
                (r for r in self.repo.repos if r.dataset == dataset), None
            )
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

    def _classify(self, pct_rank: float) -> tuple:
        """Return (key, color, label) for a given percentile rank."""
        for key, low, high, color, label in self.RISK_LEVELS:
            if pct_rank >= low:
                return key, color, label
        # Fallback (should not happen)
        return 'normal', '#34d399', 'Normal'

    # ── main entry point ─────────────────────────────────────────────────────

    def compute_risk_map(
        self,
        dataset: str,
        feature: str,
        lookback: int = 30,
        include_analysis: bool = False,
        seasonal: bool = True,
    ) -> Dict[str, Any]:
        repo = self._find_repo_for_dataset(dataset)
        if repo is None:
            raise ValueError(f"Dataset '{dataset}' not found.")

        unit = repo.feature_units.get(feature, '')
        stations_with_feature = []
        for s, meta in repo.station_index.items():
            valid_feats = get_valid_features_for_analysis('risk', meta.get('features', []))
            if feature in valid_feats:
                stations_with_feature.append(s)
                

        if not stations_with_feature:
            raise ValueError(f"No stations with feature '{feature}' in '{dataset}'.")

        results: List[Dict[str, Any]] = []
        for station_name in stations_with_feature:
            meta = repo.station_index[station_name]
            lat = meta.get('lat')
            lon = meta.get('lon')
            if lat is None or lon is None:
                continue

            ts = self._load_series(repo, station_name, feature)
            if ts is None or len(ts) < 12:
                continue

            values = ts.values.astype(float)
            lb = min(lookback, len(values))
            recent_mean = float(np.nanmean(values[-lb:]))

            if seasonal:
                # ── Seasonal mode: compare against same calendar-month history ──
                # Determine dominant calendar month in the lookback window
                lookback_months = pd.Series(ts.iloc[-lb:].index.month, dtype='int64')
                cal_month = int(lookback_months.mode().iloc[0])
                historical_season = ts[ts.index.month == cal_month].values.astype(float)
                if len(historical_season) < 3:
                    # Not enough same-month observations — fall back to full record
                    reference = values
                    percentile_mode_station = 'full_record_fallback'
                else:
                    reference = historical_season
                    percentile_mode_station = f'seasonal_month_{cal_month}'
            else:
                reference = values
                percentile_mode_station = 'full_record'

            pct_rank = float(scipy.stats.percentileofscore(reference, recent_mean))
            risk_key, risk_color, risk_label = self._classify(pct_rank)

            results.append({
                'station': station_name,
                'name': meta.get('name', station_name),
                'lat': float(lat),
                'lon': float(lon),
                'risk': risk_key,
                'risk_label': risk_label,
                'color': risk_color,
                'recent_value': round(recent_mean, 3),
                'percentile_rank': round(pct_rank, 1),
                'percentile_mode': percentile_mode_station,
                'unit': unit,
            })

        if not results:
            raise ValueError('No stations with sufficient data for risk computation.')

        # Summary counts per risk level
        summary = {level[0]: 0 for level in self.RISK_LEVELS}
        for r in results:
            summary[r['risk']] = summary.get(r['risk'], 0) + 1

        figure = self._build_map_figure(results, dataset, feature, unit)

        percentile_mode_label = (
            'seasonal (same calendar-month percentiles)' if seasonal
            else 'full-record percentiles'
        )
        result = {
            'dataset': dataset,
            'feature': feature,
            'unit': unit,
            'lookback': lookback,
            'n_stations': len(results),
            'summary': summary,
            'stations': results,
            'figure': plotly.io.to_json(figure),
            'percentile_mode': 'seasonal' if seasonal else 'full_record',
            'method_note': (
                f'Risk classification uses {percentile_mode_label} '
                f'(P5/P20/P80/P95 thresholds). '
                + ('Seasonal mode removes the confounding effect of regular '
                   'seasonal cycles on risk classification. '
                   if seasonal else
                   'Full-record mode does not remove seasonal effects — '
                   'wet-season values may appear elevated relative to the full-year distribution. ')
            ),
        }

        if include_analysis:
            analysis = _generate_risk_analysis(result)
            if analysis:
                result['analysis'] = analysis

        return result

    # ── figure builder ────────────────────────────────────────────────────────

    def _build_map_figure(
        self,
        results: List[Dict[str, Any]],
        dataset: str,
        feature: str,
        unit: str,
    ) -> go.Figure:

        center_lat = float(np.mean([r['lat'] for r in results]))
        center_lon = float(np.mean([r['lon'] for r in results]))

        # Zoom: Mekong basin is narrow, LamaH is broader
        zoom = 4 if dataset == 'mekong' else 6

        fig = go.Figure()

        # One trace per risk level so the legend shows all levels
        for key, _low, _high, color, label in self.RISK_LEVELS:
            group = [r for r in results if r['risk'] == key]
            if not group:
                continue
            lats = [r['lat'] for r in group]
            lons = [r['lon'] for r in group]
            hover = [
                f"<b>{r['name']}</b><br>"
                f"Recent: {r['recent_value']:.2f} {unit}<br>"
                f"Percentile: {r['percentile_rank']:.1f}th<br>"
                f"Status: <b>{r['risk_label']}</b>"
                for r in group
            ]
            fig.add_trace(go.Scattermapbox(
                lat=lats, lon=lons,
                mode='markers',
                marker=dict(size=11, color=color, opacity=0.9),
                text=hover,
                hoverinfo='text',
                name=f'{label} ({len(group)})',
            ))

        feature_label = feature.replace('_', ' ')
        fig.update_layout(
            mapbox=dict(
                style='carto-darkmatter',
                center=dict(lat=center_lat, lon=center_lon),
                zoom=zoom,
            ),
            paper_bgcolor=DARK_BG,
            font=dict(family='Inter, sans-serif', color=TEXT, size=12),
            legend=legend_v(),
            height=680,
            margin=MARGIN_MAP,
            title=dict(
                text=f'{feature_label} Risk Map — {dataset.capitalize()}',
                font=dict(size=14, color=TEXT),
                x=0.5,
                xanchor='center',
            ),
        )
        return fig
