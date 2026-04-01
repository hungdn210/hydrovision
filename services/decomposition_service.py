from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL

from .data_loader import SeriesRequest


def _generate_decomp_analysis(result: Dict[str, Any]) -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return ''
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        s = result.get('stats', {})
        prompt = f"""Analyze this STL decomposition result and provide 3 concise bullet-point insights:

Station/feature: {result.get('title', '')}
Record length: {s.get('n_months')} months
Trend strength: {s.get('strength_trend')} (0=no trend, 1=strong trend)
Seasonal strength: {s.get('strength_seasonal')} (0=no seasonality, 1=strong seasonality)
Peak month: {s.get('seasonal_peak_month')}, Trough month: {s.get('seasonal_trough_month')}
Trend slope: {s.get('trend_slope_per_decade')} per decade

Focus on: dominant signal type (trend vs seasonal), seasonal flood/drought timing, and hydrological significance. Use **bold** for key terms."""
        resp = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
        return resp.text.strip()
    except Exception:
        return ''


class DecompositionService:
    """
    STL (Seasonal-Trend decomposition using LOESS) Analysis.

    Methodology:
      - Resample the raw series to monthly means for stability.
      - Apply STL with period=12 (annual seasonality for monthly data).
      - Return a 4-panel Plotly figure: Observed, Trend, Seasonal, Residual.
      - Compute strength-of-trend and strength-of-seasonality indices
        (Wang et al. 2006) which are standard in hydrology papers.
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

    def decompose(
        self,
        dataset: str,
        station: str,
        feature: str,
        include_analysis: bool = False,
    ) -> Dict[str, Any]:
        repo = self._find_repo(dataset)
        if repo is None:
            raise ValueError(f"Dataset '{dataset}' not found.")

        ts = self._load_series(repo, station, feature)
        if ts is None or len(ts) < 60:
            raise ValueError(f"Insufficient data for '{station}' / '{feature}'.")

        unit = repo.feature_units.get(feature, '')
        station_name = repo.station_index[station].get('name', station)
        feature_label = feature.replace('_', ' ').title()

        # Monthly resample
        monthly = ts.resample('ME').mean().dropna()
        if len(monthly) < 36:
            raise ValueError("Need at least 36 months (3 years) for STL decomposition.")

        values = monthly.values.astype(float)
        dates = list(monthly.index)
        period = 12  # annual seasonality

        # ── Run STL ───────────────────────────────────────────────────────────
        stl = STL(values, period=period, robust=True)
        result = stl.fit()

        trend = result.trend
        seasonal = result.seasonal
        residual = result.resid
        observed = values

        # ── Strength metrics (Wang et al. 2006) ───────────────────────────────
        var_resid = float(np.var(residual))
        var_trend_resid = float(np.var(trend + residual))
        var_seasonal_resid = float(np.var(seasonal + residual))

        strength_trend = max(0.0, 1.0 - var_resid / var_trend_resid) if var_trend_resid > 0 else 0.0
        strength_seasonal = max(0.0, 1.0 - var_resid / var_seasonal_resid) if var_seasonal_resid > 0 else 0.0

        # ── Build 4-panel figure ──────────────────────────────────────────────
        DARK_BG = '#07111f'
        TEXT = '#e5eefc'
        GRID = 'rgba(148,163,184,0.08)'

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                f'Observed ({unit})',
                f'Trend  (strength = {strength_trend:.2f})',
                f'Seasonal  (strength = {strength_seasonal:.2f})',
                'Residual',
            ],
            vertical_spacing=0.07,
        )

        common_hover = '%{x|%Y-%m}: %{y:.3f}<extra></extra>'

        # Row 1 — Observed
        fig.add_trace(go.Scatter(
            x=dates, y=observed.tolist(), mode='lines',
            name='Observed',
            line=dict(color='rgba(148,163,184,0.6)', width=1.2),
            hovertemplate=common_hover,
        ), row=1, col=1)

        # Row 2 — Trend
        fig.add_trace(go.Scatter(
            x=dates, y=trend.tolist(), mode='lines',
            name='Trend',
            line=dict(color='#60a5fa', width=2),
            hovertemplate=common_hover,
        ), row=2, col=1)

        # Row 3 — Seasonal
        fig.add_trace(go.Scatter(
            x=dates, y=seasonal.tolist(), mode='lines',
            name='Seasonal',
            line=dict(color='#34d399', width=1.5),
            hovertemplate=common_hover,
            fill='tozeroy', fillcolor='rgba(52,211,153,0.07)',
        ), row=3, col=1)

        # Zero line for seasonal
        fig.add_hline(y=0, line_color='rgba(148,163,184,0.3)', line_width=1, row=3, col=1)

        # Row 4 — Residual
        pos_mask = np.array(residual) >= 0
        fig.add_trace(go.Bar(
            x=[d for d, p in zip(dates, pos_mask) if p],
            y=[r for r, p in zip(residual.tolist(), pos_mask) if p],
            name='Residual (+)',
            marker_color='rgba(96,165,250,0.6)',
            hovertemplate='%{x|%Y-%m}: %{y:.3f}<extra></extra>',
        ), row=4, col=1)
        fig.add_trace(go.Bar(
            x=[d for d, p in zip(dates, pos_mask) if not p],
            y=[r for r, p in zip(residual.tolist(), pos_mask) if not p],
            name='Residual (−)',
            marker_color='rgba(248,113,113,0.6)',
            hovertemplate='%{x|%Y-%m}: %{y:.3f}<extra></extra>',
        ), row=4, col=1)

        # Layout
        fig.update_layout(
            paper_bgcolor=DARK_BG,
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, sans-serif', color=TEXT, size=11),
            title=dict(
                text=f'STL Decomposition — {feature_label} · {station_name}',
                font=dict(size=14, color=TEXT),
                x=0.5, xanchor='center',
            ),
            showlegend=False,
            margin=dict(l=60, r=20, t=80, b=50),
            barmode='relative',
        )

        for i in range(1, 5):
            fig.update_xaxes(
                gridcolor=GRID, zeroline=False,
                tickfont=dict(color=TEXT, size=10),
                row=i, col=1,
            )
            fig.update_yaxes(
                gridcolor=GRID, zeroline=False,
                tickfont=dict(color=TEXT, size=10),
                row=i, col=1,
            )

        # Style subplot titles
        for ann in fig.layout.annotations:
            ann.font.color = TEXT
            ann.font.size = 11

        # ── Seasonal profile (average month values) ───────────────────────────
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        seasonal_profile = [
            float(np.mean([seasonal[i] for i in range(m, len(seasonal), 12)]))
            for m in range(12)
        ]
        peak_month = month_labels[int(np.argmax(seasonal_profile))]
        trough_month = month_labels[int(np.argmin(seasonal_profile))]

        result = {
            'title': f'STL Decomposition · {station_name}',
            'subtitle': (
                f'{feature_label} · {len(monthly)} months · '
                f'trend strength {strength_trend:.2f} · seasonal strength {strength_seasonal:.2f}'
            ),
            'figure': plotly.io.to_json(fig),
            'stats': {
                'n_months': len(monthly),
                'period': period,
                'strength_trend': round(strength_trend, 3),
                'strength_seasonal': round(strength_seasonal, 3),
                'seasonal_peak_month': peak_month,
                'seasonal_trough_month': trough_month,
                'residual_std': round(float(np.std(residual)), 4),
                'trend_slope_per_decade': round(
                    float(np.polyfit(np.arange(len(trend)), trend, 1)[0]) * 120, 4
                ),
            },
        }

        if include_analysis:
            analysis = _generate_decomp_analysis(result)
            if analysis:
                result['analysis'] = analysis

        return result
