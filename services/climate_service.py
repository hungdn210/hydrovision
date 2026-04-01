from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io
import scipy.stats

from .data_loader import SeriesRequest


def _generate_climate_analysis(result: Dict[str, Any]) -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return ''
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        s = result.get('stats', {})
        prompt = f"""Analyze this climate projection result and provide 3 concise bullet-point insights:

Station/feature: {result.get('title', '')}
Historical trend: {s.get('historical_trend_per_decade')} per decade (R²={s.get('r_squared')}, p={s.get('p_value')})
Projection years: {s.get('projection_years')}
Scenarios: {s.get('scenarios')}

Focus on: trend significance, scenario spread/risk, and practical water management implications. Use **bold** for key terms."""
        resp = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
        return resp.text.strip()
    except Exception:
        return ''


# SSP scenario definitions — delta_temp in °C above baseline,
# discharge_scale / precip_scale as fractional change by end of projection window
SCENARIOS = {
    'SSP1-2.6': {'delta_temp': 1.5, 'q_scale': -0.08, 'p_scale': 0.05, 'color': '#34d399', 'dash': 'dot'},
    'SSP2-4.5': {'delta_temp': 2.5, 'q_scale': -0.16, 'p_scale': 0.09, 'color': '#60a5fa', 'dash': 'dash'},
    'SSP5-8.5': {'delta_temp': 4.5, 'q_scale': -0.30, 'p_scale': 0.15, 'color': '#f87171', 'dash': 'dashdot'},
}

DISCHARGE_KEYWORDS = {'discharge', 'flow', 'runoff', 'streamflow', 'q_'}


class ClimateService:
    """
    Climate Change Impact Projector.

    Methodology (delta-change approach):
      1. Compute the linear trend and inter-annual variability from the full
         historical record (annual means).
      2. Extend the baseline trend into the future projection window.
      3. Apply scenario-specific progressive scaling factors that represent
         the expected direction and magnitude of change under each SSP.
      4. Add uncertainty bands that widen with time, scaled by the warming
         level of each scenario.
      5. Return a Plotly figure with historical data, trend, and projected
         bands for all three SSP scenarios.
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

    @staticmethod
    def _hex_to_rgba(hex_color: str, alpha: float) -> str:
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'

    # ── main entry ─────────────────────────────────────────────────────────────

    def project(
        self,
        dataset: str,
        station: str,
        feature: str,
        projection_years: int = 30,
        include_analysis: bool = False,
    ) -> Dict[str, Any]:
        repo = self._find_repo(dataset)
        if repo is None:
            raise ValueError(f"Dataset '{dataset}' not found.")

        ts = self._load_series(repo, station, feature)
        if ts is None or len(ts) < 730:          # need ~2 years minimum
            raise ValueError(f"Insufficient data for '{station}' / '{feature}'.")

        unit = repo.feature_units.get(feature, '')

        # Annual means
        annual = ts.resample('YE').mean().dropna()
        if len(annual) < 5:
            raise ValueError("Need at least 5 complete years for projection.")

        years_hist = np.array([d.year for d in annual.index], dtype=float)
        vals_hist = annual.values.astype(float)

        # Linear trend over historical record
        slope, intercept, r_val, p_val, _ = scipy.stats.linregress(years_hist, vals_hist)

        # Residual std for uncertainty
        residuals = vals_hist - (slope * years_hist + intercept)
        hist_std = float(np.std(residuals))

        # Determine whether this is a discharge-like or precip-like feature
        fl = feature.lower()
        is_discharge = any(k in fl for k in DISCHARGE_KEYWORDS)
        scale_key = 'q_scale' if is_discharge else 'p_scale'

        # Future years
        last_year = int(years_hist[-1])
        future_years = np.arange(last_year + 1, last_year + projection_years + 1, dtype=float)
        baseline_proj = slope * future_years + intercept   # trend-only projection

        # Progress fraction 0→1 over projection window
        t_frac = (future_years - future_years[0]) / max(projection_years - 1, 1)

        DARK_BG = '#07111f'
        TEXT = '#e5eefc'
        fig = go.Figure()

        # ── Historical annual data ────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=years_hist.astype(int).tolist(),
            y=vals_hist.tolist(),
            mode='lines+markers',
            name='Historical annual mean',
            line=dict(color='#94a3b8', width=1.5),
            marker=dict(size=4),
        ))

        # ── Historical trend line ─────────────────────────────────────────────
        trend_y = (slope * years_hist + intercept).tolist()
        fig.add_trace(go.Scatter(
            x=years_hist.astype(int).tolist(),
            y=trend_y,
            mode='lines',
            name=f'Historical trend  ({slope * 10:+.3f} {unit}/decade)',
            line=dict(color='#e5eefc', width=1.5, dash='dash'),
        ))

        # ── SSP scenario projections ──────────────────────────────────────────
        scenario_stats = {}
        for name, cfg in SCENARIOS.items():
            scale = cfg[scale_key]
            # Progressive fractional change reaching `scale` at end of window
            delta = 1.0 + scale * t_frac
            proj = baseline_proj * delta

            # Uncertainty grows with time and with warming magnitude
            sigma = hist_std * (1.0 + abs(cfg['delta_temp']) * 0.12) * (0.3 + 0.7 * t_frac)
            upper = proj + sigma
            lower = proj - sigma

            color = cfg['color']
            rgba_fill = self._hex_to_rgba(color, 0.12)

            fy_list = future_years.astype(int).tolist()

            # Uncertainty band
            fig.add_trace(go.Scatter(
                x=fy_list + fy_list[::-1],
                y=(upper.tolist()) + (lower.tolist()[::-1]),
                fill='toself',
                fillcolor=rgba_fill,
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False,
                hoverinfo='skip',
                name=f'{name} band',
            ))

            # Central projection line
            fig.add_trace(go.Scatter(
                x=fy_list,
                y=proj.tolist(),
                mode='lines',
                name=f'{name}  (+{cfg["delta_temp"]}°C)',
                line=dict(color=color, width=2, dash=cfg['dash']),
                hovertemplate=f'%{{x}}: %{{y:.2f}} {unit} ({name})<extra></extra>',
            ))

            end_pct = round(scale * 100, 1)
            scenario_stats[name] = {
                'delta_temp_C': cfg['delta_temp'],
                'projected_end_change_pct': end_pct,
                'projected_end_value': round(float(proj[-1]), 3),
            }

        # ── Layout ────────────────────────────────────────────────────────────
        feature_label = feature.replace('_', ' ').title()
        station_name = repo.station_index[station].get('name', station)

        fig.update_layout(
            paper_bgcolor=DARK_BG,
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, sans-serif', color=TEXT, size=12),
            title=dict(
                text=f'Climate Impact Projection — {feature_label} · {station_name}',
                font=dict(size=14, color=TEXT),
                x=0.5, xanchor='center',
            ),
            xaxis=dict(
                title='Year',
                gridcolor='rgba(148,163,184,0.08)',
                showgrid=True,
                zeroline=False,
            ),
            yaxis=dict(
                title=f'{feature_label} ({unit})' if unit else feature_label,
                gridcolor='rgba(148,163,184,0.08)',
                showgrid=True,
                zeroline=False,
            ),
            legend=dict(
                orientation='v',
                yanchor='top', y=0.99,
                xanchor='left', x=0.01,
                bgcolor='rgba(7,17,31,0.82)',
                bordercolor='rgba(148,163,184,0.15)',
                borderwidth=1,
                font=dict(size=10),
            ),
            margin=dict(l=60, r=20, t=50, b=50),
            shapes=[dict(
                type='line',
                x0=last_year, x1=last_year,
                y0=0, y1=1, yref='paper',
                line=dict(color='rgba(148,163,184,0.4)', dash='dot', width=1),
            )],
            annotations=[dict(
                x=last_year, y=1.04, yref='paper',
                xanchor='center', showarrow=False,
                text='↑ Projection',
                font=dict(color='rgba(148,163,184,0.6)', size=10),
            )],
        )

        result = {
            'title': f'Climate Projection · {station_name}',
            'subtitle': (
                f'{feature_label} · {int(years_hist[0])}–{last_year} historical '
                f'+ {projection_years}-year projection'
            ),
            'series': [{'station': station, 'feature': feature,
                        'start_date': str(annual.index[0].date()),
                        'end_date': str(annual.index[-1].date())}],
            'figure': plotly.io.to_json(fig),
            'stats': {
                'n_years_historical': len(annual),
                'historical_trend_per_decade': round(slope * 10, 4),
                'r_squared': round(r_val ** 2, 3),
                'p_value': round(p_val, 4),
                'projection_years': projection_years,
                'scenarios': scenario_stats,
            },
        }

        if include_analysis:
            analysis = _generate_climate_analysis(result)
            if analysis:
                result['analysis'] = analysis

        return result
