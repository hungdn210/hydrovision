from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io
import scipy.stats

from .data_loader import SeriesRequest


def _generate_risk_analysis(result: Dict[str, Any]) -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return ''
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        summary = result.get('summary', {})
        prompt = f"""Analyze this flood/drought risk map result and provide 3 concise bullet-point insights:

Feature: {result.get('feature')} | Dataset: {result.get('dataset')}
Lookback: {result.get('lookback')} days
Station risk summary: {summary}
Total stations: {result.get('n_stations')}

Focus on: current risk conditions, proportion of at-risk stations, and recommendations for water managers. Use **bold** for key terms."""
        resp = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
        return resp.text.strip()
    except Exception:
        return ''


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
        stations_with_feature = [
            s for s, meta in repo.station_index.items()
            if feature in meta.get('features', [])
        ]
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
