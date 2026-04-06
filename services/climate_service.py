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
from .figure_theme import (
    TEXT, SOFT, GRID_LIGHT,
    dark_layout, axis_style,
    legend_v, MARGIN_STD,
    forecast_divider_shape, forecast_divider_annotation,
)


def _fallback_climate_analysis(result: Dict[str, Any]) -> str:
    s = result.get('stats', {}) or {}
    scenarios = s.get('scenarios', {}) or {}
    title = str(result.get('title', '')).replace('_', ' ')
    trend = s.get('historical_trend_per_decade')
    r2 = s.get('r_squared')
    projection_years = s.get('projection_years')
    scenario_names = list(scenarios.keys())

    if scenario_names:
        high = scenarios.get('SSP5-8.5') or scenarios[scenario_names[-1]]
        low = scenarios.get('SSP1-2.6') or scenarios[scenario_names[0]]
        high_change = high.get('projected_end_change_pct')
        low_change = low.get('projected_end_change_pct')
        high_value = high.get('projected_end_value')
    else:
        high_change = low_change = high_value = None

    scenario_range_html = (
        f'<li><strong>Scenario Range:</strong> By the end of the {projection_years}-year horizon, the low-emissions pathway changes by {low_change}% while the high-emissions pathway changes by {high_change}%, showing the uncertainty envelope tied to future warming.</li>'
        if high_change is not None and low_change is not None else
        '<li><strong>Scenario Range:</strong> Multiple SSP pathways are included to show how future outcomes diverge under different climate forcing assumptions.</li>'
    )
    high_risk_html = (
        f'<li><strong>High-Risk Pathway:</strong> The strongest projected end-state reaches about {high_value} under the most severe scenario, which should be treated as the upper-risk planning case.</li>'
        if high_value is not None else
        '<li><strong>High-Risk Pathway:</strong> The higher-warming scenario provides the stress-test case for planning and infrastructure review.</li>'
    )

    return (
        '<p><strong>Executive Summary</strong></p>'
        f'<p>The climate sensitivity analysis for <strong>{title}</strong> extends the observed historical trend into the next {projection_years} years and applies illustrative SSP delta-scaling brackets. '
        f'The historical trend is {trend:+.4f} per decade with R²={r2}. '
        f'The end-of-period scenario spread indicates how strongly future conditions may diverge under different warming assumptions.</p>'
        f'<p><em>Note: SSP scaling factors are illustrative sensitivity proxies, not CMIP6 ensemble projections. '
        f'Treat these envelopes as directional planning brackets, not deterministic forecasts.</em></p>'
        '<p><strong>Detailed Insights</strong></p>'
        '<ul>'
        f'<li><strong>Historical Trend:</strong> The fitted historical trend is {trend:+.4f} per decade with R²={r2}, which summarises the long-run direction and strength of the observed annual signal.</li>'
        f'{scenario_range_html}'
        f'{high_risk_html}'
        '<li><strong>Operational Interpretation:</strong> Use the scenario spread to test adaptation options and stress-test planning assumptions. '
        'These are sensitivity envelopes — pair them with local hydrological knowledge and, where available, downscaled CMIP6 projections before making infrastructure decisions.</li>'
        '</ul>'
    )


def _generate_climate_analysis(result: Dict[str, Any]) -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_google_gemini_api_key_here' or not api_key.strip():
        return _fallback_climate_analysis(result)
    try:
        s = result.get('stats', {})
        prompt = f"""Act as a professional hydrologist interpreting a climate sensitivity analysis.
Write the response in markdown and structure it exactly as follows:

**Executive Summary**
2-3 sentences summarising the historical trend, the projection horizon, and the significance of the scenario spread.
Always note that SSP scaling factors are illustrative sensitivity brackets, not CMIP6 ensemble projections.

**Detailed Insights**
- **Historical Trend:** interpret the sign, magnitude, and strength of the historical trend.
- **Scenario Spread:** compare low-, medium-, and high-warming sensitivity brackets using the provided end-state changes.
- **Risk Interpretation:** explain what the most severe sensitivity bracket implies for water-resource planning.
- **Operational Interpretation:** state how the sensitivity envelopes should be used in planning and decision-making, and remind the reader these are not CMIP6 projections.

Rules:
- Use professional hydrological language.
- Always cite specific numbers from the provided results.
- Replace underscores with spaces.
- Do not include any introduction or sign-off outside the two sections.
- Use the term "sensitivity envelope" or "sensitivity bracket" rather than "projection" for the future scenarios.
- Acknowledge the illustrative nature of the SSP delta scaling.

Projection title: {str(result.get('title', '')).replace('_', ' ')}
Historical trend: {s.get('historical_trend_per_decade')} per decade (R²={s.get('r_squared')}, p={s.get('p_value')})
Projection years: {s.get('projection_years')}
Scenarios: {s.get('scenarios')}
"""
        return markdown.markdown(_gemini_generate(api_key, prompt))
    except Exception:
        return _fallback_climate_analysis(result)


# ── Methodological disclaimer ─────────────────────────────────────────────────
# The scaling factors below are illustrative delta proxies, NOT derived from
# CMIP6 GCM ensemble output.  They represent plausible directional sensitivity
# brackets tied to IPCC-reported warming levels, but should be treated as
# "what-if" sensitivity envelopes rather than calibrated climate projections.
# For true climate projections, CMIP6 bias-corrected regional outputs
# (e.g. ISIMIP3, CORDEX-SEA) are required.
SSP_DISCLAIMER = (
    'Scaling factors are illustrative delta proxies and do not derive from '
    'CMIP6 GCM ensemble output. These scenarios visualise directional '
    'sensitivity under different warming assumptions and should not be '
    'interpreted as calibrated climate projections.'
)

# SSP scenario definitions — delta_temp in °C above baseline,
# discharge_scale / precip_scale as fractional change by end of projection window
SCENARIOS = {
    'SSP1-2.6 (Low)':  {'delta_temp': 1.5, 'q_scale': -0.08, 'p_scale': 0.05, 'color': '#34d399', 'dash': 'dot'},
    'SSP2-4.5 (Med)':  {'delta_temp': 2.5, 'q_scale': -0.16, 'p_scale': 0.09, 'color': '#60a5fa', 'dash': 'dash'},
    'SSP5-8.5 (High)': {'delta_temp': 4.5, 'q_scale': -0.30, 'p_scale': 0.15, 'color': '#f87171', 'dash': 'dashdot'},
}

from .feature_registry import get_feature_type, FeatureType

class ClimateService:
    """
    Climate Sensitivity Projector (illustrative SSP delta-change approach).

    Methodology:
      1. Compute the linear trend and inter-annual variability from the full
         historical record (annual means).
      2. Extend the baseline trend into the future projection window.
      3. Apply scenario-specific progressive scaling factors that represent
         the expected direction and magnitude of change under each SSP
         warming bracket.
      4. Add uncertainty bands that widen with time, scaled by the warming
         level of each scenario.
      5. Return a Plotly figure with historical data, trend, and projected
         sensitivity envelopes for three SSP-labelled warming brackets.

    IMPORTANT — Academic limitation:
      The SSP scaling factors are illustrative proxies, NOT CMIP6 ensemble
      output. Results are sensitivity envelopes for planning purposes, not
      calibrated climate projections. See SSP_DISCLAIMER for details.
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
        is_discharge = get_feature_type(feature) == FeatureType.FLOW
        scale_key = 'q_scale' if is_discharge else 'p_scale'

        # Future years
        last_year = int(years_hist[-1])
        future_years = np.arange(last_year + 1, last_year + projection_years + 1, dtype=float)
        baseline_proj = slope * future_years + intercept   # trend-only projection

        # Progress fraction 0→1 over projection window
        t_frac = (future_years - future_years[0]) / max(projection_years - 1, 1)

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
            line=dict(color=TEXT, width=1.5, dash='dash'),
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

        _ax = axis_style(grid=GRID_LIGHT)
        fig.update_layout(**dark_layout(
            title=f'Climate Sensitivity Analysis — {feature_label} · {station_name}',
            height=520,
            margin=MARGIN_STD,
            show_legend=True,
            xaxis=dict(**_ax, title='Year', showgrid=True, zeroline=False),
            yaxis=dict(**_ax, title=f'{feature_label} ({unit})' if unit else feature_label,
                       showgrid=True, zeroline=False),
            legend=legend_v(),
            shapes=[forecast_divider_shape(last_year)],
            annotations=[forecast_divider_annotation(last_year, label='↑ Projection')],
        ))

        result = {
            'title': f'Climate Sensitivity Analysis · {station_name}',
            'subtitle': (
                f'{feature_label} · {int(years_hist[0])}–{last_year} historical '
                f'+ {projection_years}-year sensitivity projection'
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
            'method_note': (
                'Linear trend extrapolation + illustrative SSP delta-scaling brackets. '
                'Historical trend fitted by ordinary least squares on annual means.'
            ),
            'caveat': SSP_DISCLAIMER,
            'analysis_strength': 'illustrative_sensitivity',
        }

        if include_analysis:
            analysis = _generate_climate_analysis(result)
            if analysis:
                result['analysis'] = analysis

        return result
