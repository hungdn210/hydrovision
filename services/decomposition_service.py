from __future__ import annotations

import os
from typing import Any, Dict, Optional

import markdown
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL

from .analysis_service import _gemini_generate
from .data_loader import SeriesRequest
from .figure_theme import (
    GRID_LIGHT, dark_layout, axis_style,
    MARGIN_SUBPLOT, style_subplot_titles,
)


def _fallback_decomp_analysis(result: Dict[str, Any]) -> str:
    s = result.get('stats', {}) or {}
    title = str(result.get('title', '')).replace('_', ' ')
    n_months = s.get('n_months', '?')
    ft = s.get('strength_trend', '?')
    fs = s.get('strength_seasonal', '?')
    peak = s.get('seasonal_peak_month', '?')
    trough = s.get('seasonal_trough_month', '?')
    slope = s.get('trend_slope_per_decade', '?')
    res_std = s.get('residual_std', '?')

    dominance = ''
    try:
        ft_f = float(ft)
        fs_f = float(fs)
        if fs_f >= ft_f and fs_f >= 0.6:
            dominance = (
                f'With seasonal strength F<sub>S</sub> = {fs_f:.2f} and trend strength F<sub>T</sub> = {ft_f:.2f}, '
                'the series is predominantly controlled by recurring annual seasonality. '
                'The annual flood–drought cycle accounts for most of the observed variability, and the long-run trend is a secondary signal.'
            )
        elif ft_f >= fs_f and ft_f >= 0.6:
            dominance = (
                f'With trend strength F<sub>T</sub> = {ft_f:.2f} and seasonal strength F<sub>S</sub> = {fs_f:.2f}, '
                'a persistent long-run directional change is the dominant feature of this series. '
                'This may reflect upstream regulation, climate change, or land-use effects superimposed on the seasonal cycle.'
            )
        else:
            dominance = (
                f'Both trend strength (F<sub>T</sub> = {ft_f:.2f}) and seasonal strength (F<sub>S</sub> = {fs_f:.2f}) are moderate, '
                'suggesting that neither a strong long-run trend nor a highly regular seasonal cycle dominates — '
                'the residual component is relatively large and likely contains episodic events or irregular interannual forcing.'
            )
    except (TypeError, ValueError):
        dominance = f'Trend strength: {ft}, seasonal strength: {fs}.'

    return (
        '<p><strong>Executive Summary</strong></p>'
        f'<p>The STL decomposition for <strong>{title}</strong> separates {n_months} months of observations into Trend, Seasonal, and Residual components '
        'using Seasonal-Trend decomposition via LOESS (Cleveland et al. 1990) with period=12 and robust fitting. '
        f'{dominance} '
        f'The seasonal component peaks in <strong>{peak}</strong> and reaches its annual minimum in <strong>{trough}</strong>, '
        f'while the long-run trend changes at approximately <strong>{slope}</strong> per decade.</p>'

        '<p><strong>Signal Dominance</strong></p>'
        f'<p>The strength indices (Wang et al. 2006) quantify the relative importance of each component. '
        f'Trend strength F<sub>T</sub> = {ft} and seasonal strength F<sub>S</sub> = {fs} (both on a 0–1 scale, where 1 = component fully explains the variance beyond residual noise). '
        'Values above 0.6 are considered strong in the hydrology literature. '
        f'The residual standard deviation of {res_std} indicates the magnitude of unexplained variability — large residuals may point to extreme events, data gaps, or non-periodic climate forcing not captured by the trend or seasonal components.</p>'

        '<p><strong>Seasonal Cycle</strong></p>'
        f'<p>The STL seasonal component captures the repeating annual flood–drought pattern. '
        f'Peak flow typically occurs in <strong>{peak}</strong>, consistent with monsoonal or snowmelt-driven high-water periods at this station. '
        f'The seasonal trough in <strong>{trough}</strong> represents the driest period of the year, which is critical for low-flow planning, irrigation demand, and ecological minimum-flow assessments. '
        'The amplitude of the seasonal component relative to the total signal variance is a direct measure of how predictable year-to-year timing of floods and droughts is at this location.</p>'

        '<p><strong>Trend Analysis</strong></p>'
        f'<p>The fitted LOESS trend evolves smoothly across the {n_months}-month record and changes at approximately <strong>{slope}</strong> per decade. '
        'A positive trend indicates a long-run increase in the variable (possible intensification of the monsoon or upstream land-use change increasing runoff), '
        'while a negative trend may indicate groundwater depletion, upstream abstraction, reservoir impoundment, or progressive drying. '
        'It is important to note that the STL trend is a statistical smoothing, not a causal attribution — the direction should be cross-referenced with basin-management history before drawing conclusions.</p>'

        '<p><strong>Residual Component</strong></p>'
        f'<p>The residual series represents the portion of variability not explained by trend or the regular seasonal cycle. '
        f'A residual standard deviation of {res_std} relative to the total series variance provides a measure of model fit — lower residuals indicate that trend and seasonality together explain most of the signal. '
        'Episodic large-magnitude residuals may correspond to extreme flood or drought years, instrument errors, or the influence of large-scale climate anomalies (e.g. strong El Niño events) that fall outside the regular seasonal pattern. '
        'Inspecting the residual sub-panel for systematic structure (e.g. clustering of positive residuals in specific decades) can help identify non-stationary behaviour or missing explanatory variables.</p>'

        '<p><strong>Operational Relevance</strong></p>'
        '<ul>'
        '<li><strong>Seasonal forecasting:</strong> A high seasonal strength confirms that climatological seasonal forecasts based on historical averages are reliable at this station. '
        f'The identified peak in {peak} and trough in {trough} should anchor flood-preparedness calendars and low-flow contingency plans.</li>'
        f'<li><strong>Trend monitoring:</strong> The observed trend of {slope} per decade should be monitored over successive updates. '
        'If the trend is accelerating, historical design standards (e.g. flood frequency curves assuming stationarity) may underestimate future risk.</li>'
        '<li><strong>Extreme event identification:</strong> Large residuals in specific years can serve as a preliminary screen for extreme or anomalous events, complementing formal extreme-value analysis (GEV / Gumbel).</li>'
        '</ul>'
    )


def _generate_decomp_analysis(result: Dict[str, Any]) -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_google_gemini_api_key_here' or not api_key.strip():
        return _fallback_decomp_analysis(result)
    try:
        s = result.get('stats', {})
        prompt = f"""Act as a professional hydrologist writing a detailed technical interpretation of an STL decomposition analysis.
Write the response in markdown and structure it exactly as follows:

**Executive Summary**
4-5 sentences. Summarise which component (trend, seasonal, residual) dominates the series, cite the strength indices, describe the seasonal timing, interpret the trend direction and magnitude, and state the key operational implication.

**Signal Dominance**
A paragraph (4-5 sentences) interpreting the trend strength (F_T) and seasonal strength (F_S) values in detail. Explain what these indices mean quantitatively (scale 0–1, values above 0.6 are strong). Discuss what the relative magnitudes imply about whether long-run change or recurring seasonality is the primary control. Comment on the residual standard deviation and what it implies about unexplained variability.

**Seasonal Cycle**
A paragraph (4-5 sentences) interpreting the seasonal component. Explain the hydrological meaning of the peak and trough months in the context of monsoon or snowmelt regimes. Discuss the amplitude of seasonality relative to total variance and its implication for flood–drought predictability and operational planning.

**Trend Analysis**
A paragraph (4-5 sentences) interpreting the long-run LOESS trend. Discuss the direction (positive/negative) and magnitude (per-decade slope). List plausible physical causes (regulation, land-use, climate change). Caution that STL trend is a statistical smoothing, not causal attribution, and recommend cross-referencing with basin history.

**Residual Component**
A paragraph (3-4 sentences) interpreting the residual series. Discuss what large residuals may represent (extreme events, data gaps, non-periodic climate forcing). Note whether residual structure may indicate non-stationarity or missing predictors.

**Operational Relevance**
Exactly 3 bullet points:
- **Seasonal forecasting:** how seasonal strength informs reliability of climatological forecasts.
- **Trend monitoring:** how the detected trend should influence design standards and risk assessments going forward.
- **Extreme event screening:** how large residuals can complement formal extreme-value analysis.

Rules:
- Use professional hydrological language throughout.
- Always cite specific numbers from the provided results.
- Replace underscores with spaces.
- Do not include any introduction or sign-off outside the defined sections.

Analysis title: {str(result.get('title', '')).replace('_', ' ')}
Record length: {s.get('n_months')} months
Trend strength (F_T): {s.get('strength_trend')} (0=no trend, 1=strong trend)
Seasonal strength (F_S): {s.get('strength_seasonal')} (0=no seasonality, 1=strong seasonality)
Peak month: {s.get('seasonal_peak_month')}, Trough month: {s.get('seasonal_trough_month')}
Trend slope: {s.get('trend_slope_per_decade')} per decade
Residual std: {s.get('residual_std')}
"""
        return markdown.markdown(_gemini_generate(api_key, prompt))
    except Exception:
        return _fallback_decomp_analysis(result)


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
        fig.update_layout(**dark_layout(
            title=f'STL Decomposition — {feature_label} · {station_name}',
            height=580,
            margin=MARGIN_SUBPLOT,
            show_legend=False,
            barmode='relative',
        ))

        ax = axis_style(grid=GRID_LIGHT)
        for i in range(1, 5):
            fig.update_xaxes(**ax, zeroline=False, row=i, col=1)
            fig.update_yaxes(**ax, zeroline=False, row=i, col=1)

        # Style subplot titles
        style_subplot_titles(fig)

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
