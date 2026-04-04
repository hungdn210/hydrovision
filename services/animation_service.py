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


def _classify_label(pct: float) -> str:
    for threshold, _, label in RISK_LEVELS:
        if pct >= threshold:
            return label
    return 'Normal'


def _marker_size(pct: float) -> float:
    if pct >= 95 or pct < 5:
        return 18
    if pct >= 80 or pct < 20:
        return 15
    return 12


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
        SUBTLE_TEXT = '#b6c2d9'
        feature_label = feature.replace('_', ' ').title()

        def make_year_annotation(year: int) -> Dict[str, Any]:
            return dict(
                x=0.985,
                y=0.965,
                xref='paper',
                yref='paper',
                xanchor='right',
                yanchor='top',
                align='right',
                showarrow=False,
                text=f"<b>{year}</b><br><span style='font-size:11px;color:{SUBTLE_TEXT}'>annual basin state</span>",
                font=dict(size=22, color='#f8fbff'),
                bgcolor='rgba(7,17,31,0.78)',
                bordercolor='rgba(148,163,184,0.16)',
                borderwidth=1,
                borderpad=10,
            )

        def make_status_annotation(summary_html: str) -> Dict[str, Any]:
            return dict(
                x=0.015,
                y=0.895,
                xref='paper',
                yref='paper',
                xanchor='left',
                yanchor='top',
                align='left',
                showarrow=False,
                text=summary_html,
                font=dict(size=11, color=TEXT),
                bgcolor='rgba(7,17,31,0.74)',
                bordercolor='rgba(148,163,184,0.16)',
                borderwidth=1,
                borderpad=9,
            )

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
            counts = {label: 0 for _, _, label in RISK_LEVELS}
            for rec in records:
                if year not in rec['annual'].index.year:
                    continue
                year_val = float(rec['annual'][rec['annual'].index.year == year].iloc[0])
                pct = float(scipy.stats.percentileofscore(rec['all_vals'], year_val))
                color = _classify_color(pct)
                label = _classify_label(pct)
                lats.append(rec['lat'])
                lons.append(rec['lon'])
                colors.append(color)
                sizes.append(_marker_size(pct))
                counts[label] += 1
                texts.append(
                    f"<b>{rec['name']}</b><br>"
                    f"Year: {year}<br>"
                    f"Value: {year_val:.2f} {unit}<br>"
                    f"Percentile: {pct:.1f}th<br>"
                    f"Condition: {label}"
                )

            total = max(len(lats), 1)
            wet_share = (counts['Flood Watch'] + counts['Flood Risk']) / total * 100
            dry_share = (counts['Drought'] + counts['Severe Drought']) / total * 100
            dominant_label = max(counts.items(), key=lambda item: item[1])[0]
            status_html = (
                f"<b>{dominant_label}</b><br>"
                f"<span style='color:{SUBTLE_TEXT}'>Wet-risk stations</span> {wet_share:.0f}%"
                f" &nbsp;·&nbsp; "
                f"<span style='color:{SUBTLE_TEXT}'>Dry-risk stations</span> {dry_share:.0f}%"
            )

            glow_trace = go.Scattermapbox(
                lat=lats, lon=lons,
                mode='markers',
                marker=dict(size=[s * 1.95 for s in sizes], color=colors, opacity=0.16),
                hoverinfo='skip',
                showlegend=False,
                name=str(year),
            )
            core_trace = go.Scattermapbox(
                lat=lats, lon=lons,
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors,
                    opacity=0.94,
                ),
                text=texts,
                hoverinfo='text',
                hoverlabel=dict(
                    bgcolor='rgba(7,17,31,0.92)',
                    bordercolor='rgba(148,163,184,0.28)',
                    font=dict(color=TEXT, size=12),
                ),
                showlegend=False,
                name=str(year),
            )
            frames.append(go.Frame(
                data=[glow_trace, core_trace],
                name=str(year),
                layout=go.Layout(annotations=[make_status_annotation(status_html), make_year_annotation(year)]),
            ))
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
        first_glow = frames[0].data[0] if frames else go.Scattermapbox()
        first_core = frames[0].data[1] if frames else go.Scattermapbox()

        # Legend traces (invisible anchors)
        legend_traces = []
        for threshold, color, label in RISK_LEVELS:
            legend_traces.append(go.Scattermapbox(
                lat=[None], lon=[None],
                mode='markers',
                marker=dict(size=11, color=color),
                name=label,
                showlegend=True,
            ))

        fig = go.Figure(
            data=[first_glow, first_core] + legend_traces,
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
                orientation='h',
                yanchor='top', y=0.98,
                xanchor='left', x=0.012,
                bgcolor='rgba(7,17,31,0.62)',
                bordercolor='rgba(148,163,184,0.15)',
                borderwidth=1,
                font=dict(size=11),
                itemclick=False,
                itemdoubleclick=False,
            ),
            height=720,
            margin=dict(l=0, r=0, t=50, b=86),
            annotations=frames[0].layout.annotations if frames else [],
            sliders=[dict(
                active=0,
                currentvalue=dict(
                    visible=False,
                    xanchor='center',
                    font=dict(color=TEXT, size=13),
                ),
                x=0.09,
                y=0.02,
                len=0.82,
                pad=dict(t=18, b=0),
                bgcolor='rgba(7,17,31,0.72)',
                bordercolor='rgba(148,163,184,0.2)',
                tickcolor=TEXT,
                activebgcolor='rgba(96,165,250,0.28)',
                font=dict(color=SUBTLE_TEXT, size=10),
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
