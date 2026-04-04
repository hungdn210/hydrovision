from __future__ import annotations

import os
from typing import Any, Dict, Optional

import markdown
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io
import pywt
from plotly.subplots import make_subplots

from .analysis_service import _gemini_generate
from .data_loader import SeriesRequest


def _fallback_wavelet_analysis(result: Dict[str, Any]) -> str:
    s = result.get('stats', {}) or {}
    title = str(result.get('title', '')).replace('_', ' ')
    dom = s.get('dominant_periods_months') or []
    dom_text = ', '.join(str(p) for p in dom) if dom else 'none identified'
    return (
        '<p><strong>Executive Summary</strong></p>'
        f'<p>The wavelet analysis for <strong>{title}</strong> examines how periodic behaviour evolves through time over {s.get("n_months")} months. '
        f'The strongest detected periodicities are {dom_text} months within an analysed range of {s.get("period_range_months")} months.</p>'
        '<p><strong>Detailed Insights</strong></p>'
        '<ul>'
        f'<li><strong>Dominant Cycles:</strong> The leading periodicities are {dom_text} months, indicating the main oscillatory scales in the series.</li>'
        f'<li><strong>Scale Context:</strong> The analysis spans approximately {s.get("period_range_months")} months, allowing comparison between short seasonal cycles and longer inter-annual variability.</li>'
        '<li><strong>Hydrological Interpretation:</strong> Periods near 12 months usually indicate annual monsoonal seasonality, while multi-year bands may reflect broader climate drivers such as ENSO-scale variability.</li>'
        '<li><strong>Operational Interpretation:</strong> Use the dominant bands to inform monitoring windows, seasonal preparedness, and interpretation of whether the system is controlled mainly by annual or multi-year variability.</li>'
        '</ul>'
    )


def _generate_wavelet_analysis(result: Dict[str, Any]) -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_google_gemini_api_key_here' or not api_key.strip():
        return _fallback_wavelet_analysis(result)
    try:
        s = result.get('stats', {})
        prompt = f"""Act as a professional hydrologist interpreting a wavelet power analysis.
Write the response in markdown and structure it exactly as follows:

**Executive Summary**
2-3 sentences summarising the dominant periodicities and whether variability is concentrated at seasonal or multi-year scales.

**Detailed Insights**
- **Dominant Cycles:** interpret the strongest reported periodicities.
- **Scale Context:** explain what the analysed period range captures.
- **Hydrological Interpretation:** relate the periodicities to plausible physical drivers such as monsoon or ENSO-scale variability.
- **Operational Interpretation:** state how this information should be used in planning and monitoring.

Rules:
- Use professional hydrological language.
- Always cite specific numbers from the provided results.
- Replace underscores with spaces.
- Do not include any introduction or sign-off outside the two sections.

Analysis title: {str(result.get('title', '')).replace('_', ' ')}
Record length: {s.get('n_months')} months
Dominant periodicities: {s.get('dominant_periods_months')} months
Period range analyzed: {s.get('period_range_months')} months
Wavelet: {s.get('wavelet')}
"""
        return markdown.markdown(_gemini_generate(api_key, prompt))
    except Exception:
        return _fallback_wavelet_analysis(result)


class WaveletService:
    """
    Continuous Wavelet Transform (CWT) Analysis.

    Methodology:
      - Resample to monthly means and normalise (zero-mean, unit-variance).
      - Apply CWT using the Morlet wavelet across periods 2–120 months.
      - Display power as a scalogram (period vs time heatmap) with cone of
        influence (COI) masking regions affected by edge effects.
      - Compute the global wavelet power spectrum (time-averaged) to identify
        dominant periodicities (ENSO ~36–60 months, annual ~12 months, etc.).
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

    def analyse(
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

        # Monthly means
        monthly = ts.resample('ME').mean().dropna()
        if len(monthly) < 36:
            raise ValueError("Need at least 36 months for wavelet analysis.")

        x = monthly.values.astype(float)
        dates = list(monthly.index)
        N = len(x)
        dt = 1.0  # sampling interval in months

        # Normalise to zero mean and unit std
        x_norm = (x - np.mean(x)) / (np.std(x) + 1e-12)

        # ── CWT using PyWavelets ──────────────────────────────────────────────
        # Periods from 2 to min(N/2, 120) months
        min_period = 2
        max_period = min(N // 2, 120)
        # Logarithmically spaced periods
        periods = np.unique(np.round(
            np.exp(np.linspace(np.log(min_period), np.log(max_period), 80))
        ).astype(int))
        periods = periods[(periods >= min_period) & (periods <= max_period)]

        # PyWavelets CWT: scales = period / (2 * Morlet center frequency)
        # For 'morl', central frequency ≈ 0.8125 (pywt.central_frequency)
        wavelet = 'morl'
        central_freq = pywt.central_frequency(wavelet)
        scales = (central_freq * dt * periods).astype(float)

        coefficients, freqs = pywt.cwt(x_norm, scales, wavelet, sampling_period=dt)
        # Power = |W|^2 / scale  (scale-normalised, Torrence & Compo 1998)
        power = (np.abs(coefficients) ** 2) / scales[:, np.newaxis]

        # ── Cone of Influence (COI) ────────────────────────────────────────────
        # For Morlet wavelet: e-folding time = sqrt(2) * scale
        # COI in time: edge-affected region at each end
        coi_time = np.sqrt(2) * scales / central_freq  # in months
        time_idx = np.arange(N)
        # Mask power outside COI (distance from edges)
        edge_dist = np.minimum(time_idx, N - 1 - time_idx)  # distance from nearest edge
        coi_mask = np.zeros_like(power, dtype=bool)
        for s_idx, coi_t in enumerate(coi_time):
            coi_mask[s_idx, :] = edge_dist < coi_t

        power_masked = np.where(coi_mask, np.nan, power)

        # ── Global power spectrum (time-average, ignoring COI) ────────────────
        valid_counts = (~coi_mask).sum(axis=1)
        global_power = np.where(
            valid_counts > 0,
            np.nansum(power, axis=1) / np.maximum(valid_counts, 1),
            np.nan,
        )

        # Dominant periods (peaks in global spectrum)
        from scipy.signal import argrelmax
        peak_idxs = argrelmax(global_power, order=2)[0]
        if len(peak_idxs) == 0:
            peak_idxs = [int(np.nanargmax(global_power))]
        top_peaks = sorted(peak_idxs, key=lambda i: -global_power[i])[:3]
        dominant_periods = [int(periods[i]) for i in top_peaks if i < len(periods)]

        # ── Build figure ──────────────────────────────────────────────────────
        DARK_BG = '#07111f'
        TEXT = '#e5eefc'
        SOFT = 'rgba(229,238,252,0.72)'
        GRID = 'rgba(148,163,184,0.10)'
        visible_period_max = min(max_period, 60)

        # Downsample and focus the default view on the most interpretable range
        step = max(1, len(periods) // 72)
        p_plot = periods[::step]
        pw_plot = power_masked[::step, :]
        gp_plot = global_power[::step]
        visible_mask = p_plot <= visible_period_max
        p_plot = p_plot[visible_mask]
        pw_plot = pw_plot[visible_mask, :]
        gp_plot = gp_plot[visible_mask]

        valid_power = pw_plot[np.isfinite(pw_plot)]
        zmax = float(np.nanpercentile(valid_power, 98)) if valid_power.size else 1.0

        def _period_label(period: int) -> str:
            if period <= 7:
                return 'Semi-annual scale'
            if period <= 14:
                return 'Annual seasonal scale'
            if period <= 24:
                return '1-2 year variability'
            if period <= 36:
                return '2-3 year variability'
            return 'Multi-year / ENSO-scale'

        dominant_visible = [dp for dp in dominant_periods if dp <= visible_period_max][:3]
        if not dominant_visible and len(p_plot):
            dominant_visible = [int(p_plot[int(np.nanargmax(gp_plot))])]

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.76, 0.24],
            vertical_spacing=0.12,
            subplot_titles=['Cycle Strength Through Time', 'Dominant Periods Overall'],
        )

        # Heatmap guide bands
        period_bands = [
            (5, 7, 'Semi-annual zone', 'rgba(52,211,153,0.07)'),
            (10, 14, 'Annual zone', 'rgba(245,158,11,0.07)'),
            (24, 36, '2-3 year zone', 'rgba(96,165,250,0.07)'),
            (36, 60, 'Multi-year zone', 'rgba(248,113,113,0.06)'),
        ]
        for y0, y1, _label, fill in period_bands:
            if y0 >= visible_period_max:
                continue
            fig.add_hrect(
                y0=y0, y1=min(y1, visible_period_max),
                fillcolor=fill, line_width=0,
                row=1, col=1,
            )

        # Scalogram
        fig.add_trace(go.Heatmap(
            x=dates,
            y=p_plot.tolist(),
            z=pw_plot.tolist(),
            colorscale='Turbo',
            zmin=0,
            zmax=zmax,
            zsmooth='best',
            colorbar=dict(
                title=dict(text='Cycle strength', font=dict(color=TEXT, size=10)),
                tickfont=dict(color=TEXT, size=9),
                len=0.58,
                y=0.74,
                x=1.03,
                thickness=13,
                outlinewidth=0,
            ),
            customdata=[[ _period_label(int(p)) for _ in range(len(dates)) ] for p in p_plot.tolist()],
            hovertemplate='Date: %{x|%Y-%m}<br>Period: %{y} months<br>Interpretation: %{customdata}<br>Cycle strength: %{z:.3f}<extra></extra>',
            name='Cycle strength',
        ), row=1, col=1)

        # Cone of influence boundary: periods above this curve are edge-affected
        coi_boundary = np.clip(edge_dist / np.sqrt(2), float(np.min(p_plot)), float(visible_period_max))
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=coi_boundary.tolist() + [float(visible_period_max)] * len(dates),
            mode='lines',
            fill='toself',
            fillcolor='rgba(7,17,31,0.32)',
            line=dict(width=0),
            hoverinfo='skip',
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=dates,
            y=coi_boundary.tolist(),
            mode='lines',
            line=dict(color='rgba(229,238,252,0.5)', width=1.1, dash='dot'),
            hovertemplate='COI boundary<br>Date: %{x|%Y-%m}<br>Reliable below ~%{y:.1f} months<extra></extra>',
            showlegend=False,
            name='COI boundary',
        ), row=1, col=1)

        PERIOD_COLORS = ['#f59e0b', '#34d399', '#60a5fa']
        for dp, col in zip(dominant_visible[:3], PERIOD_COLORS):
            fig.add_hline(
                y=dp,
                line_color=col,
                line_width=1.4,
                line_dash='dash',
                opacity=0.9,
                row=1, col=1,
            )
            fig.add_annotation(
                x=1.005,
                y=dp,
                xref='paper',
                yref='y',
                text=f'{_period_label(dp)} ({dp} mo)',
                showarrow=False,
                xanchor='left',
                font=dict(color=col, size=10),
            )
        fig.add_annotation(
            x=dates[min(8, len(dates) - 1)],
            y=min(visible_period_max - 3, max(42, visible_period_max * 0.72)),
            xref='x',
            yref='y',
            text='Edge-affected region',
            showarrow=False,
            xanchor='left',
            font=dict(color='rgba(229,238,252,0.68)', size=10),
            bgcolor='rgba(7,17,31,0.55)',
            bordercolor='rgba(229,238,252,0.14)',
            borderwidth=1,
            borderpad=4,
        )

        # Bottom summary: overall strongest periods
        fig.add_trace(go.Scatter(
            x=p_plot.tolist(),
            y=gp_plot.tolist(),
            mode='lines',
            name='Overall cycle strength',
            line=dict(color='#60a5fa', width=2.8),
            fill='tozeroy',
            fillcolor='rgba(96,165,250,0.12)',
            hovertemplate='Period: %{x} months<br>Overall cycle strength: %{y:.3f}<extra></extra>',
        ), row=2, col=1)

        for x0, x1, _label, fill in period_bands:
            if x0 >= visible_period_max:
                continue
            fig.add_vrect(
                x0=x0, x1=min(x1, visible_period_max),
                fillcolor=fill, line_width=0,
                row=2, col=1,
            )

        for dp, col in zip(dominant_visible[:3], PERIOD_COLORS):
            idx = int(np.argmin(np.abs(p_plot - dp)))
            fig.add_vline(
                x=dp,
                line_color=col,
                line_width=1.3,
                line_dash='dash',
                opacity=0.9,
                row=2, col=1,
            )
            fig.add_trace(go.Scatter(
                x=[int(p_plot[idx])],
                y=[float(gp_plot[idx])],
                mode='markers+text',
                text=[f'{int(p_plot[idx])} mo'],
                textposition='top center',
                marker=dict(color=col, size=9, symbol='diamond'),
                textfont=dict(color=col, size=10),
                showlegend=False,
                hovertemplate=f'Period: {int(p_plot[idx])} months<br>Overall cycle strength: %{{y:.3f}}<extra></extra>',
            ), row=2, col=1)

        fig.update_layout(
            paper_bgcolor=DARK_BG,
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, sans-serif', color=TEXT, size=11),
            title=dict(
                text=f'Wavelet Cycle Analysis — {feature_label} · {station_name}',
                font=dict(size=18, color=TEXT),
                x=0.5, xanchor='center',
                y=0.98,
            ),
            showlegend=False,
            margin=dict(l=82, r=110, t=120, b=60),
            height=660,
        )
        fig.add_annotation(
            x=0.0, y=1.13,
            xref='paper', yref='paper',
            showarrow=False,
            xanchor='left',
            align='left',
            font=dict(color=SOFT, size=12),
            text='Shows when repeating cycles were strongest and which cycle lengths dominate overall.',
        )
        fig.add_annotation(
            x=0.0, y=1.08,
            xref='paper', yref='paper',
            showarrow=False,
            xanchor='left',
            align='left',
            font=dict(color='rgba(229,238,252,0.66)', size=10),
            text='How to read: brighter colors indicate stronger repeating behavior. Around 12 months suggests annual seasonality; longer periods suggest multi-year variability.',
        )

        fig.update_yaxes(
            title_text='Period (months)',
            range=[2, visible_period_max],
            tickmode='array',
            tickvals=[2, 3, 6, 12, 24, 36, 48, 60],
            ticktext=['2', '3', '6', '12', '24', '36', '48', '60'],
            gridcolor=GRID,
            zeroline=False,
            tickfont=dict(color=TEXT, size=10),
            row=1, col=1,
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor=GRID,
            zeroline=False,
            tickfont=dict(color=TEXT, size=10),
            tickformat='%Y',
            row=1, col=1,
        )
        fig.update_xaxes(
            title_text='Cycle length (months)',
            range=[2, visible_period_max],
            tickmode='array',
            tickvals=[2, 6, 12, 24, 36, 48, 60],
            ticktext=['2', '6', '12', '24', '36', '48', '60'],
            showgrid=True,
            gridcolor=GRID,
            zeroline=False,
            tickfont=dict(color=TEXT, size=10),
            row=2, col=1,
        )
        fig.update_yaxes(
            title_text='Overall cycle strength',
            showgrid=True,
            gridcolor=GRID,
            zeroline=False,
            tickfont=dict(color=TEXT, size=10),
            row=2, col=1,
        )

        # Style subplot titles only
        for ann in fig.layout.annotations[:2]:
            ann.font.color = TEXT
            ann.font.size = 13

        result = {
            'title': f'Wavelet Analysis · {station_name}',
            'subtitle': (
                f'{feature_label} · {len(monthly)} months · '
                f'dominant periods: {", ".join(str(p) + " mo" for p in dominant_periods[:3])}'
            ),
            'figure': plotly.io.to_json(fig),
            'stats': {
                'n_months': len(monthly),
                'period_range_months': [int(min_period), int(max_period)],
                'dominant_periods_months': dominant_periods[:3],
                'wavelet': 'Morlet',
            },
        }

        if include_analysis:
            analysis = _generate_wavelet_analysis(result)
            if analysis:
                result['analysis'] = analysis

        return result
