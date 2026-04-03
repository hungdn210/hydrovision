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
    feature = str(result.get('feature', '')).replace('_', ' ')
    dataset = str(result.get('dataset', '')).replace('_', ' ')
    lookback = result.get('lookback')

    return (
        '<p><strong>Executive Summary</strong></p>'
        f'<p>The flood and drought screening for <strong>{feature}</strong> across the <strong>{dataset}</strong> dataset '
        f'used a {lookback}-point recent window and classified {result.get("n_stations")} stations against their own historical percentiles. '
        f'Overall, {at_risk} stations ({at_risk_pct:.1f}%) are outside the normal band, with '
        f'{flood + flood_watch} in elevated flood conditions and {drought + severe_drought} in drought conditions.</p>'
        '<p><strong>Detailed Insights</strong></p>'
        '<ul>'
        f'<li><strong>Risk Balance:</strong> {normal} of {result.get("n_stations")} stations remain normal, while {at_risk} are flagged as hydrologically stressed.</li>'
        f'<li><strong>Flood Signal:</strong> {flood} stations are in Flood Risk and {flood_watch} are in Flood Watch, representing {flood_pct:.1f}% of the monitored network.</li>'
        f'<li><strong>Drought Signal:</strong> {severe_drought} stations are in Severe Drought and {drought} are in Drought, representing {drought_pct:.1f}% of the monitored network.</li>'
        '<li><strong>Operational Interpretation:</strong> Use the map as a basin-screening layer to prioritise station follow-up, but confirm any operational action with station hydrographs and local catchment context.</li>'
        '</ul>'
    )


def _generate_risk_analysis(result: Dict[str, Any]) -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_google_gemini_api_key_here' or not api_key.strip():
        return _fallback_risk_analysis(result)
    try:
        summary = result.get('summary', {})
        prompt = f"""Act as a professional hydrologist interpreting a basin risk-screening map.
Write the response in markdown and structure it exactly as follows:

**Executive Summary**
2-3 sentences summarising the overall flood-versus-drought balance, how many stations are outside normal conditions, and what that means operationally.

**Detailed Insights**
- **Risk Balance:** quantify the split between normal and at-risk stations.
- **Flood Conditions:** interpret the counts in Flood Watch and Flood Risk.
- **Drought Conditions:** interpret the counts in Drought and Severe Drought.
- **Operational Interpretation:** state how water managers should use this map in practice.

Rules:
- Use professional hydrological language.
- Always cite specific numbers from the provided results.
- Replace underscores with spaces.
- Do not include any introduction or sign-off outside the two sections.

Feature: {str(result.get('feature', '')).replace('_', ' ')}
Dataset: {str(result.get('dataset', '')).replace('_', ' ')}
Lookback: {result.get('lookback')} data points
Station risk summary: {summary}
Total stations: {result.get('n_stations')}
"""
        return markdown.markdown(_gemini_generate(api_key, prompt))
    except Exception:
        return _fallback_risk_analysis(result)


class RiskService:
    """
    Flood and drought risk classification for all stations in a dataset.

    Methodology:
      - Load full historical series for each station that has the target feature.
      - Compute percentile thresholds (5th, 20th, 80th, 95th) from the historical record.
      - Take the mean of the most recent `lookback` data points as the "current" value.
      - Classify the percentile rank of the current value against history:
          < 5th  → Severe Drought
          5–20th → Drought
          20–80th→ Normal
          80–95th→ Flood Watch
          > 95th → Flood Risk
      - Return station list + Plotly Scattermapbox figure.
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

            # Percentile rank of recent value vs full history
            pct_rank = float(scipy.stats.percentileofscore(values, recent_mean))
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
                'unit': unit,
            })

        if not results:
            raise ValueError('No stations with sufficient data for risk computation.')

        # Summary counts per risk level
        summary = {level[0]: 0 for level in self.RISK_LEVELS}
        for r in results:
            summary[r['risk']] = summary.get(r['risk'], 0) + 1

        figure = self._build_map_figure(results, dataset, feature, unit)

        result = {
            'dataset': dataset,
            'feature': feature,
            'unit': unit,
            'lookback': lookback,
            'n_stations': len(results),
            'summary': summary,
            'stations': results,
            'figure': plotly.io.to_json(figure),
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

        DARK_BG = '#07111f'
        TEXT = '#e5eefc'

        center_lat = float(np.mean([r['lat'] for r in results]))
        center_lon = float(np.mean([r['lon'] for r in results]))

        # Zoom: Mekong basin is narrow, LamaH is broader
        zoom = 4 if dataset == 'mekong' else 4

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
            legend=dict(
                orientation='v',
                yanchor='top', y=0.99,
                xanchor='left', x=0.01,
                font=dict(size=11),
                bgcolor='rgba(7,17,31,0.82)',
                bordercolor='rgba(148,163,184,0.15)',
                borderwidth=1,
            ),
            height=680,
            margin=dict(l=0, r=0, t=40, b=0),
            title=dict(
                text=f'{feature_label} Risk Map — {dataset.capitalize()}',
                font=dict(size=14, color=TEXT),
                x=0.5,
                xanchor='center',
            ),
        )
        return fig
