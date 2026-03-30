from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io
import scipy.stats

from .data_loader import SeriesRequest


# Risk classification thresholds (percentile-based, same as RiskService)
RISK_LEVELS = [
    (95, '#f87171', 'Flood Risk'),
    (80, '#60a5fa', 'Flood Watch'),
    (20, '#34d399', 'Normal'),
    (5,  '#fb923c', 'Drought'),
    (0,  '#b91c1c', 'Severe Drought'),
]


def _classify_color(pct: float) -> str:
    for threshold, color, _ in RISK_LEVELS:
        if pct >= threshold:
            return color
    return '#34d399'


class AnimationService:
    """
    Animated Time-Series Map.

    Methodology:
      - Load the full time series for every station in the dataset that has
        the chosen feature.
      - Aggregate to yearly means.
      - For each year build a Plotly Scattermapbox frame where each station
        marker is coloured by the percentile rank of that year's value against
        the station's own full historical record.
      - Assemble Plotly animation with play/pause controls and a year slider.
    """

    def __init__(self, repository) -> None:
        self.repo = repository

    # ── helpers ────────────────────────────────────────────────────────────────

    def _find_repo(self, dataset: str):
        if hasattr(self.repo, 'repos'):
            return next((r for r in self.repo.repos if r.dataset == dataset), None)
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

    # ── main entry ─────────────────────────────────────────────────────────────

    def build_animation(
        self,
        dataset: str,
        feature: str,
    ) -> Dict[str, Any]:
        repo = self._find_repo(dataset)
        if repo is None:
            raise ValueError(f"Dataset '{dataset}' not found.")

        unit = repo.feature_units.get(feature, '')

        stations_with_feature = [
            s for s, meta in repo.station_index.items()
            if feature in meta.get('features', [])
        ]
        if not stations_with_feature:
            raise ValueError(f"No stations with feature '{feature}' in '{dataset}'.")

        # Load all station data
        records: List[Dict] = []
        for station in stations_with_feature:
            meta = repo.station_index[station]
            lat, lon = meta.get('lat'), meta.get('lon')
            if lat is None or lon is None:
                continue
            ts = self._load_series(repo, station, feature)
            if ts is None or len(ts) < 365:
                continue
            records.append({
                'station': station,
                'name': meta.get('name', station),
                'lat': float(lat),
                'lon': float(lon),
                'ts': ts,
            })

        if not records:
            raise ValueError('No stations with sufficient data for animation.')

        # Build yearly panel for each station
        all_years: set = set()
        for rec in records:
            annual = rec['ts'].resample('YE').mean().dropna()
            rec['annual'] = annual
            all_years.update(annual.index.year.tolist())

        years = sorted(all_years)

        # Need at least 3 years
        if len(years) < 3:
            raise ValueError('Need at least 3 years of overlapping data for animation.')

        # For percentile ranking, use each station's own full history
        for rec in records:
            rec['all_vals'] = rec['ts'].values.astype(float)

        DARK_BG = '#07111f'
        TEXT = '#e5eefc'
        feature_label = feature.replace('_', ' ').title()

        center_lat = float(np.mean([r['lat'] for r in records]))
        center_lon = float(np.mean([r['lon'] for r in records]))
        # Compute zoom from bounding box so stations fill the viewport
        lat_range = max(r['lat'] for r in records) - min(r['lat'] for r in records)
        lon_range = max(r['lon'] for r in records) - min(r['lon'] for r in records)
        max_range = max(lat_range, lon_range, 0.5)
        zoom = round(max(3.0, min(8.0, math.log2(360 / max_range) - 0.5)), 1)

        # ── Build one frame per year ──────────────────────────────────────────
        frames = []
        slider_steps = []

        for year in years:
            lats, lons, colors, texts, sizes = [], [], [], [], []
            for rec in records:
                if year not in rec['annual'].index.year:
                    continue
                year_val = float(rec['annual'][rec['annual'].index.year == year].iloc[0])
                pct = float(scipy.stats.percentileofscore(rec['all_vals'], year_val))
                color = _classify_color(pct)
                lats.append(rec['lat'])
                lons.append(rec['lon'])
                colors.append(color)
                sizes.append(10)
                texts.append(
                    f"<b>{rec['name']}</b><br>"
                    f"Year: {year}<br>"
                    f"Value: {year_val:.2f} {unit}<br>"
                    f"Percentile: {pct:.1f}th"
                )

            frame_trace = go.Scattermapbox(
                lat=lats, lon=lons,
                mode='markers',
                marker=dict(size=sizes, color=colors, opacity=0.9),
                text=texts,
                hoverinfo='text',
                name=str(year),
            )
            frames.append(go.Frame(data=[frame_trace], name=str(year)))
            slider_steps.append(dict(
                args=[[str(year)], dict(
                    frame=dict(duration=400, redraw=True),
                    mode='immediate',
                    transition=dict(duration=200),
                )],
                label=str(year),
                method='animate',
            ))

        # ── Initial frame (first year) ────────────────────────────────────────
        first = frames[0].data[0] if frames else go.Scattermapbox()

        # Legend traces (invisible anchors)
        legend_traces = []
        for threshold, color, label in RISK_LEVELS:
            legend_traces.append(go.Scattermapbox(
                lat=[None], lon=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=label,
                showlegend=True,
            ))

        fig = go.Figure(
            data=[first] + legend_traces,
            frames=frames,
        )

        fig.update_layout(
            mapbox=dict(
                style='carto-darkmatter',
                center=dict(lat=center_lat, lon=center_lon),
                zoom=zoom,
            ),
            paper_bgcolor=DARK_BG,
            font=dict(family='Inter, sans-serif', color=TEXT, size=12),
            title=dict(
                text=f'Animated {feature_label} Conditions — {dataset.capitalize()}',
                font=dict(size=14, color=TEXT),
                x=0.5, xanchor='center',
            ),
            legend=dict(
                orientation='v',
                yanchor='top', y=0.99,
                xanchor='left', x=0.01,
                bgcolor='rgba(7,17,31,0.82)',
                bordercolor='rgba(148,163,184,0.15)',
                borderwidth=1,
                font=dict(size=11),
            ),
            margin=dict(l=0, r=0, t=50, b=60),
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                y=0.04,
                x=0.5,
                xanchor='center',
                yanchor='bottom',
                bgcolor='rgba(7,17,31,0.85)',
                bordercolor='rgba(148,163,184,0.2)',
                font=dict(color=TEXT, size=12),
                buttons=[
                    dict(
                        label='▶  Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=500, redraw=True),
                            fromcurrent=True,
                            transition=dict(duration=200),
                        )],
                    ),
                    dict(
                        label='⏸  Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                            transition=dict(duration=0),
                        )],
                    ),
                ],
            )],
            sliders=[dict(
                active=0,
                currentvalue=dict(
                    prefix='Year: ',
                    visible=True,
                    xanchor='center',
                    font=dict(color=TEXT, size=13),
                ),
                pad=dict(t=50, b=10),
                bgcolor='rgba(7,17,31,0.6)',
                bordercolor='rgba(148,163,184,0.2)',
                tickcolor=TEXT,
                font=dict(color=TEXT, size=10),
                steps=slider_steps,
            )],
        )

        return {
            'title': f'Animated Map · {feature_label} · {dataset.capitalize()}',
            'subtitle': f'{len(records)} stations · {years[0]}–{years[-1]} · {len(years)} frames',
            'figure': plotly.io.to_json(fig),
            'stats': {
                'n_stations': len(records),
                'n_years': len(years),
                'year_range': [years[0], years[-1]],
                'feature': feature,
                'dataset': dataset,
            },
        }
