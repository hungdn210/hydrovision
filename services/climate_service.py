from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import markdown
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io
import scipy.stats
from scipy.stats import theilslopes, kendalltau

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
    p_value = s.get('p_value')
    projection_years = s.get('projection_years')
    scenario_names = list(scenarios.keys())

    ssp_low  = scenarios.get('SSP1-2.6') or (scenarios[scenario_names[0]]  if scenario_names else {})
    ssp_mid  = scenarios.get('SSP2-4.5') or (scenarios[scenario_names[1]]  if len(scenario_names) > 1 else {})
    ssp_high = scenarios.get('SSP5-8.5') or (scenarios[scenario_names[-1]] if scenario_names else {})

    low_change  = ssp_low.get('projected_end_change_pct')
    mid_change  = ssp_mid.get('projected_end_change_pct')
    high_change = ssp_high.get('projected_end_change_pct')
    high_value  = ssp_high.get('projected_end_value')
    low_value   = ssp_low.get('projected_end_value')

    # Trend significance note
    try:
        p = float(p_value)
        sig_note = ('statistically significant at the 5% level (p={:.3f})'.format(p) if p < 0.05
                    else 'not statistically significant at the 5% level (p={:.3f}), suggesting the historical signal is weak or noisy'.format(p))
    except (TypeError, ValueError):
        sig_note = 'statistical significance not reported'

    # Scenario divergence
    try:
        spread = abs(float(high_change) - float(low_change))
        spread_note = (
            f'The spread between the low-warming (SSP1-2.6, {low_change:+.1f}%) and high-warming (SSP5-8.5, {high_change:+.1f}%) '
            f'sensitivity brackets is {spread:.1f} percentage points over {projection_years} years, '
            + ('indicating that emission pathway choices have a very large impact on future water availability at this station.'
               if spread > 20 else
               'indicating moderate divergence between pathways — water-resource outcomes remain sensitive to global emissions trajectories.')
        )
    except (TypeError, ValueError):
        spread_note = 'The scenario spread illustrates divergence between low- and high-emissions pathways.'

    return (
        '<p><strong>Executive Summary</strong></p>'
        f'<p>The climate sensitivity analysis for <strong>{title}</strong> extends the observed historical annual signal '
        f'into a {projection_years}-year projection horizon using a delta-change approach applied to three illustrative SSP warming brackets. '
        f"The historical trend estimated by Sen's slope is <strong>{trend:+.4f} per decade</strong> (R²={r2}, Mann-Kendall {sig_note}). "
        + (f'Under the low-emissions pathway (SSP1-2.6, +1.5°C), the projected end-state changes by approximately {low_change:+.1f}%; '
           f'under the high-emissions pathway (SSP5-8.5, +4.5°C), it changes by {high_change:+.1f}%, '
           f'reaching approximately {high_value} by the end of the horizon. '
           if high_change is not None and low_change is not None else '')
        + 'These envelopes are illustrative sensitivity brackets — not CMIP6 ensemble projections — and should be used for directional planning and stress-testing, not as deterministic forecasts.</p>'

        '<p><strong>Historical Trend Analysis</strong></p>'
        f'<p>The fitted linear trend of <strong>{trend:+.4f} per decade</strong> (R²={r2}) characterises the long-run direction and strength of the observed annual signal. '
        + ('A positive trend indicates a long-run increase in the variable — potentially reflecting intensifying monsoon, land-use change increasing runoff, or regional warming affecting evapotranspiration. '
           if trend is not None and float(trend) > 0 else
           'A negative trend indicates a long-run decrease — potentially reflecting declining precipitation, increased upstream abstraction, reservoir regulation, or warming-driven evapotranspiration losses. '
           if trend is not None and float(trend) < 0 else '')
        + f'This trend is {sig_note}. '
        + 'The R² value indicates the fraction of annual variability explained by the linear trend; a low R² suggests that inter-annual variability (ENSO, monsoon variability) dominates over the long-run signal, '
        + 'which should increase caution in extending the trend into the future.</p>'

        '<p><strong>Scenario Sensitivity Brackets</strong></p>'
        f'<p>{spread_note} '
        + (f'Under <strong>SSP1-2.6</strong> (+1.5°C above pre-industrial), the sensitivity bracket projects a {low_change:+.1f}% change by the end of the horizon — the optimistic pathway representing aggressive global mitigation. ' if low_change is not None else '')
        + (f'Under <strong>SSP2-4.5</strong> (+2.5°C), the mid-range bracket projects {mid_change:+.1f}% — the most likely baseline under current policy trajectories. ' if mid_change is not None else '')
        + (f'Under <strong>SSP5-8.5</strong> (+4.5°C), the high-end bracket projects {high_change:+.1f}% — the stress-test case representing fossil-fuel-intensive development with minimal mitigation. ' if high_change is not None else '')
        + 'The scenario spread should be read as the range of plausible futures conditional on global climate policy, not as a probability distribution.</p>'

        '<p><strong>Methodology and Limitations</strong></p>'
        '<ul>'
        '<li><strong>Delta-change method:</strong> The historical trend is extrapolated forward and multiplied by SSP-specific sensitivity scaling factors. '
        'This approach is transparent and reproducible but inherits all assumptions of trend stationarity — it cannot capture non-linear tipping points, '
        'feedback loops, or changes in inter-annual variability that may materialise under high-warming scenarios.</li>'
        '<li><strong>Not CMIP6 projections:</strong> The SSP scaling factors are illustrative delta proxies, not derived from bias-corrected GCM ensemble output. '
        'For infrastructure design requiring probabilistic projections, CMIP6-based regional outputs (e.g. ISIMIP3, CORDEX-SEA) are required.</li>'
        '<li><strong>Stationarity assumption:</strong> The approach assumes the historical trend represents a signal that will continue. '
        'If the series contains a structural break (detectable via change-point analysis), the trend may be unrepresentative of current-regime behaviour.</li>'
        '<li><strong>Local vs global:</strong> SSP scenarios are global-mean temperature pathways. Regional hydrological responses depend on local climate sensitivity, '
        'catchment characteristics, and land-use trajectories that are not captured in this simplified delta approach.</li>'
        '</ul>'

        '<p><strong>Operational Relevance</strong></p>'
        '<ul>'
        '<li><strong>Stress-testing:</strong> Use the SSP5-8.5 bracket as the upper-risk stress-test for infrastructure design, dam safety reviews, and long-horizon water-allocation agreements.</li>'
        '<li><strong>Adaptation planning:</strong> The divergence between SSP pathways quantifies the benefit of emissions mitigation on local water security — this figure is directly relevant to climate-adaptation cost-benefit analyses.</li>'
        '<li><strong>Monitoring trigger:</strong> If observed annual values begin consistently tracking the upper SSP bracket, this may be an early indicator that high-warming impacts are materialising ahead of schedule and should trigger accelerated adaptation planning.</li>'
        '</ul>'
    )


def _generate_climate_analysis(result: Dict[str, Any]) -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_google_gemini_api_key_here' or not api_key.strip():
        return _fallback_climate_analysis(result)
    try:
        s = result.get('stats', {})
        prompt = f"""Act as a professional hydrologist writing a detailed technical interpretation of a climate sensitivity analysis.
Write the response in markdown and structure it exactly as follows:

**Executive Summary**
4-5 sentences. Summarise the historical trend direction, magnitude, and statistical significance; describe the scenario spread between SSP1-2.6 and SSP5-8.5; state the projected end-state range; and note the illustrative nature of the SSP delta-scaling approach.

**Historical Trend Analysis**
A paragraph (4-5 sentences) interpreting the long-run historical trend in detail. Interpret the sign and magnitude of the per-decade change and what it implies about the hydrological system. Interpret R² (fraction of variance explained by the trend) and the p-value (statistical significance). Discuss plausible physical causes of the trend direction (intensifying monsoon, regulation, land-use, warming-driven evapotranspiration). Note the implications if R² is low (inter-annual variability dominates over the trend signal).

**Scenario Sensitivity Brackets**
A paragraph (5-6 sentences) comparing the three SSP sensitivity brackets in detail. For each scenario (SSP1-2.6, SSP2-4.5, SSP5-8.5), cite the projected end-state change percentage and absolute value. Compute and discuss the spread between low and high scenarios. Interpret what each warming pathway implies for water-resource availability or flood risk at this station. Note which scenario represents current global policy trajectory (SSP2-4.5).

**Methodology and Limitations**
Exactly 4 bullet points:
- **Delta-change method:** explain that the historical trend is estimated with Sen's slope (Theil-Sen, robust to outliers), tested with Mann-Kendall tau, then extrapolated and scaled by SSP-specific delta factors. Note the stationarity assumption and absence of non-linear feedbacks.
- **Not CMIP6 projections:** clarify that SSP factors are illustrative proxies, not GCM ensemble output; recommend ISIMIP3 or CORDEX for design applications.
- **Stationarity assumption:** discuss how a structural break in the historical record would affect the reliability of the extended trend.
- **Local vs global:** note that SSP pathways are global-mean targets; local hydrological sensitivity depends on regional climate and catchment characteristics.

**Operational Relevance**
Exactly 3 bullet points:
- **Stress-testing:** how to use SSP5-8.5 as the upper-risk case for infrastructure design.
- **Adaptation planning:** how the scenario spread quantifies the benefit of emissions mitigation on local water security.
- **Monitoring trigger:** how tracking observed values against the scenario bands can serve as an early-warning indicator.

Rules:
- Use professional hydrological language throughout.
- Always cite specific numbers from the provided results.
- Replace underscores with spaces.
- Use "sensitivity bracket" or "sensitivity envelope" — not "projection" — for the future scenarios.
- Explicitly state these are NOT CMIP6 projections.
- Do not include any introduction or sign-off outside the defined sections.

Projection title: {str(result.get('title', '')).replace('_', ' ')}
Historical trend (Sen's slope): {s.get('historical_trend_per_decade')} per decade
R²={s.get('r_squared')}, Mann-Kendall p={s.get('p_value')}
Feature type: {s.get('feature_type')}
Scale key applied: {s.get('scale_key_used')}
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
# scales are fractional changes by end of projection window (feature-type-aware)
#   q_scale    : discharge / streamflow
#   p_scale    : precipitation / rainfall
#   t_scale    : air temperature (positive — warming amplifies the signal)
#   wq_scale   : water quality / unknown (conservative, near-neutral)
SCENARIOS = {
    'SSP1-2.6 (Low)':  {'delta_temp': 1.5, 'q_scale': -0.08, 'p_scale':  0.05, 't_scale': 0.06,  'wq_scale': -0.04, 'color': '#34d399', 'dash': 'dot'},
    'SSP2-4.5 (Med)':  {'delta_temp': 2.5, 'q_scale': -0.16, 'p_scale':  0.09, 't_scale': 0.11,  'wq_scale': -0.07, 'color': '#60a5fa', 'dash': 'dash'},
    'SSP5-8.5 (High)': {'delta_temp': 4.5, 'q_scale': -0.30, 'p_scale':  0.15, 't_scale': 0.20,  'wq_scale': -0.12, 'color': '#f87171', 'dash': 'dashdot'},
}

# Human-readable response note per feature type (shown in chart subtitle)
_FEATURE_RESPONSE_NOTE = {
    'flow':    'discharge response: −8% to −30% by pathway end',
    'precip':  'precipitation response: +5% to +15% by pathway end',
    'temp':    'temperature response: +6% to +20% amplification by pathway end',
    'water_q': 'water quality response: −4% to −12% by pathway end (conservative)',
    'unknown': 'sensitivity scaling: illustrative proxy only',
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

        # ── Trend: Sen's slope (robust to outliers & non-normality) ─────────────
        # Theil-Sen estimator is the standard for hydrological trend analysis
        # (non-parametric, resistant to outliers, paired with Mann-Kendall test).
        res_theil = theilslopes(vals_hist, years_hist)
        slope = float(res_theil.slope)
        intercept = float(res_theil.intercept)

        # Mann-Kendall p-value for significance testing
        tau, p_val = kendalltau(years_hist, vals_hist)

        # R² from OLS kept for display context only (not used for projection)
        _, _, r_val, _, _ = scipy.stats.linregress(years_hist, vals_hist)

        # Residual std for uncertainty bands
        residuals = vals_hist - (slope * years_hist + intercept)
        hist_std = float(np.std(residuals))

        # ── Feature-type-aware scale key ──────────────────────────────────────
        ftype = get_feature_type(feature)
        scale_key = {
            FeatureType.FLOW:    'q_scale',
            FeatureType.PRECIP:  'p_scale',
            FeatureType.TEMP:    't_scale',
            FeatureType.WATER_Q: 'wq_scale',
        }.get(ftype, 'wq_scale')
        response_note = _FEATURE_RESPONSE_NOTE.get(ftype.value, _FEATURE_RESPONSE_NOTE['unknown'])

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
            name=f"Sen's trend  ({slope * 10:+.3f} {unit}/decade)",
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
            title=f'Climate Sensitivity Analysis — {feature_label} · {station_name}  ({response_note})',
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
                'trend_method': "Sen's slope (Theil-Sen estimator)",
                'r_squared': round(r_val ** 2, 3),
                'p_value': round(p_val, 4),
                'p_value_method': 'Mann-Kendall tau',
                'projection_years': projection_years,
                'feature_type': ftype.value,
                'scale_key_used': scale_key,
                'scenarios': scenario_stats,
            },
            'method_note': (
                "Sen's slope (Theil-Sen estimator) on annual means, tested with Mann-Kendall tau. "
                'Projection = trend extrapolation × SSP-specific illustrative delta-scaling.'
            ),
            'caveat': SSP_DISCLAIMER,
            'analysis_strength': 'illustrative_sensitivity',
        }

        if include_analysis:
            analysis = _generate_climate_analysis(result)
            if analysis:
                result['analysis'] = analysis

        return result
