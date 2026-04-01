from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io
import pywt
from plotly.subplots import make_subplots

from .data_loader import SeriesRequest


def _generate_wavelet_analysis(result: Dict[str, Any]) -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return ''
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        s = result.get('stats', {})
        prompt = f"""Analyze this wavelet analysis result and provide 3 concise bullet-point insights:

Station/feature: {result.get('title', '')}
Record length: {s.get('n_months')} months
Dominant periodicities: {s.get('dominant_periods_months')} months
Period range analyzed: {s.get('period_range_months')} months

Focus on: dominant cycles and their physical drivers (ENSO ~48-60 months, monsoon ~12 months, multi-decadal), time-varying behavior, and implications for seasonal water resource planning. Use **bold** for key terms."""
        resp = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
        return resp.text.strip()
    except Exception:
        return ''


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

        # Downsample if too many periods (keep ≤60 for performance)
        step = max(1, len(periods) // 60)
        p_plot = periods[::step]
        pw_plot = power_masked[::step, :]
        gp_plot = global_power[::step]

        date_strs = [d.strftime('%Y-%m') for d in dates]

        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.78, 0.22],
            subplot_titles=['Wavelet Power Spectrum (Scalogram)', 'Global Power Spectrum'],
            horizontal_spacing=0.04,
        )

        # Scalogram
        fig.add_trace(go.Heatmap(
            x=date_strs,
            y=p_plot.tolist(),
            z=pw_plot.tolist(),
            colorscale='RdBu_r',
            zsmooth='fast',
            colorbar=dict(
                title=dict(text='Power', font=dict(color=TEXT, size=10)),
                tickfont=dict(color=TEXT, size=9),
                len=0.8, x=0.76, thickness=12,
            ),
            hovertemplate='Date: %{x}<br>Period: %{y} mo<br>Power: %{z:.3f}<extra></extra>',
            name='Power',
        ), row=1, col=1)

        # Dominant period horizontal lines on scalogram
        PERIOD_COLORS = ['#f59e0b', '#34d399', '#f87171']
        period_labels = {12: 'Annual (12 mo)', 36: 'ENSO (36 mo)', 60: 'ENSO (60 mo)'}
        for dp, col in zip(dominant_periods[:3], PERIOD_COLORS):
            label = period_labels.get(dp, f'{dp} mo')
            fig.add_hline(
                y=dp, line_color=col, line_width=1.2, line_dash='dash',
                annotation_text=label,
                annotation_font=dict(color=col, size=9),
                annotation_position='right',
                row=1, col=1,
            )

        # Global power spectrum
        fig.add_trace(go.Scatter(
            x=gp_plot.tolist(),
            y=p_plot.tolist(),
            mode='lines',
            name='Global power',
            line=dict(color='#60a5fa', width=2),
            fill='tozerox',
            fillcolor='rgba(96,165,250,0.15)',
            hovertemplate='Period: %{y} mo<br>Power: %{x:.3f}<extra></extra>',
        ), row=1, col=2)

        # Mark dominant periods on global spectrum
        for dp, col in zip(dominant_periods[:3], PERIOD_COLORS):
            idx = np.argmin(np.abs(p_plot - dp))
            if idx < len(gp_plot):
                fig.add_trace(go.Scatter(
                    x=[gp_plot[idx]],
                    y=[p_plot[idx]],
                    mode='markers',
                    marker=dict(color=col, size=8, symbol='diamond'),
                    showlegend=False,
                    hovertemplate=f'{p_plot[idx]} mo: %{{x:.3f}}<extra></extra>',
                ), row=1, col=2)

        fig.update_layout(
            paper_bgcolor=DARK_BG,
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, sans-serif', color=TEXT, size=11),
            title=dict(
                text=f'Wavelet Analysis (CWT · Morlet) — {feature_label} · {station_name}',
                font=dict(size=14, color=TEXT),
                x=0.5, xanchor='center',
            ),
            showlegend=False,
            margin=dict(l=60, r=80, t=70, b=50),
        )

        # Y-axis: period (log scale looks cleaner)
        fig.update_yaxes(
            title_text='Period (months)', type='log',
            gridcolor='rgba(148,163,184,0.08)', zeroline=False,
            tickfont=dict(color=TEXT, size=10),
            tickvals=[2, 4, 6, 12, 24, 36, 60, 120],
            ticktext=['2', '4', '6', '12', '24', '36', '60', '120'],
            row=1, col=1,
        )
        fig.update_yaxes(
            type='log', gridcolor='rgba(148,163,184,0.08)', zeroline=False,
            tickfont=dict(color=TEXT, size=10),
            tickvals=[2, 4, 6, 12, 24, 36, 60, 120],
            ticktext=['2', '4', '6', '12', '24', '36', '60', '120'],
            row=1, col=2,
        )
        fig.update_xaxes(
            gridcolor='rgba(148,163,184,0.08)', zeroline=False,
            tickfont=dict(color=TEXT, size=10),
            row=1, col=1,
        )
        fig.update_xaxes(
            title_text='Power', gridcolor='rgba(148,163,184,0.08)',
            zeroline=False, tickfont=dict(color=TEXT, size=10),
            row=1, col=2,
        )

        # Style subplot titles
        for ann in fig.layout.annotations:
            ann.font.color = TEXT
            ann.font.size = 11

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
