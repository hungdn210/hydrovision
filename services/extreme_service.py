from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import markdown
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io
import scipy.stats
from plotly.subplots import make_subplots

from .data_loader import SeriesRequest
from .feature_registry import get_valid_features_for_analysis
from .figure_theme import (
    SOFT,
    dark_layout, axis_style,
    legend_h, MARGIN_SUBPLOT, style_subplot_titles,
)


def _fallback_extreme_analysis(result: Dict[str, Any], note: str | None = None) -> str:
    station = result.get('station', '').replace('_', ' ')
    feature = result.get('feature', '').replace('_', ' ')
    unit = result.get('unit', '') or 'no unit'
    n_years = result.get('n_years', 0)
    year_range = result.get('year_range', [])
    gev = result.get('gev_params') or {}
    gumbel = result.get('gumbel_params') or {}
    levels = result.get('return_levels', [])
    ci_lower = result.get('ci_lower') or []
    ci_upper = result.get('ci_upper') or []

    def _level(T: int, key: str) -> Any:
        row = next((r for r in levels if r.get('return_period') == T and key in r), None)
        return row.get(key) if row else None

    gev10 = _level(10, 'gev')
    gev50 = _level(50, 'gev')
    gev100 = _level(100, 'gev')
    gumbel10 = _level(10, 'gumbel')
    xi = gev.get('shape')
    if xi is None:
        tail_text = 'No stable GEV tail estimate was available, so interpretation should lean on the simpler Gumbel fit.'
    elif xi > 0.05:
        tail_text = f'The fitted GEV shape parameter is {xi:.3f}, indicating a heavier Fréchet-type upper tail and potentially more severe rare extremes.'
    elif xi < -0.05:
        tail_text = f'The fitted GEV shape parameter is {xi:.3f}, indicating a bounded Weibull-type upper tail rather than an unbounded extreme tail.'
    else:
        tail_text = f'The fitted GEV shape parameter is {xi:.3f}, which is close to Gumbel behaviour and suggests a moderately shaped extreme-value tail.'

    reliability_bits = [f'The analysis uses {n_years} annual maxima']
    if len(year_range) >= 2:
        reliability_bits.append(f'covering {year_range[0]} to {year_range[1]}')
    reliability = ' '.join(reliability_bits) + '.'
    if ci_lower and ci_upper and gev50 is not None:
        try:
            idx50 = [r.get('return_period') for r in levels].index(50)
            reliability += f' The 50-year GEV estimate is {gev50:.2f} {unit} with an approximate 95% interval of {ci_lower[idx50]:.2f} to {ci_upper[idx50]:.2f} {unit}.'
        except Exception:
            pass
    elif n_years < 20:
        reliability += ' The short record length means long return-period estimates should be treated cautiously.'

    bullets = []
    if gev10 is not None or gumbel10 is not None:
        primary_10 = gev10 if gev10 is not None else gumbel10
        primary_50 = gev50 if gev50 is not None else _level(50, 'gumbel')
        primary_100 = gev100 if gev100 is not None else _level(100, 'gumbel')
        bullets.append(
            f'<li><strong>Return Levels:</strong> The fitted extreme-value model suggests a 10-year event near {primary_10:.2f} {unit}'
            + (f', rising to about {primary_50:.2f} {unit} at 50 years' if primary_50 is not None else '')
            + (f' and {primary_100:.2f} {unit} at 100 years.' if primary_100 is not None else '.')
        )
    bullets.append(f'<li><strong>Tail Behaviour:</strong> {tail_text}</li>')
    bullets.append(f'<li><strong>Reliability:</strong> {reliability}</li>')
    bullets.append(
        '<li><strong>Operational Interpretation:</strong> Use the higher return-period estimates as screening-level design guidance, '
        'but pair them with catchment context and hydraulic assessment before making infrastructure or flood-preparedness decisions.</li>'
    )

    note_html = f'<p><em>{note}</em></p>' if note else ''
    return (
        f'<p><strong>Executive Summary</strong></p>'
        f'<p>{feature} at {station} was analysed using annual maxima extreme-value fitting in {unit}. '
        f'The result provides screening-level return-period estimates for rare high-flow events and highlights how confident those estimates are given the available record length.</p>'
        f'<p><strong>Detailed Insights</strong></p>'
        f'<ul>{"".join(bullets)}</ul>'
        f'{note_html}'
    )


def _generate_extreme_analysis(result: Dict[str, Any]) -> str:
    """Generate AI analysis of extreme event results using Gemini."""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return _fallback_extreme_analysis(result)
    try:
        from .analysis_service import _gemini_generate
        station = result.get('station', '').replace('_', ' ')
        feature = result.get('feature', '').replace('_', ' ')
        n_years = result.get('n_years', 0)
        yr = result.get('year_range', [])
        gev = result.get('gev_params')
        gumbel = result.get('gumbel_params')
        levels = result.get('return_levels', [])
        gev_lines = '\n'.join(
            f"- T={r['return_period']} yr: {r.get('gev', 'N/A')} {result.get('unit', '')}"
            for r in levels if 'gev' in r
        ) or '- No stable GEV return levels available.'
        gumbel_lines = '\n'.join(
            f"- T={r['return_period']} yr: {r.get('gumbel', 'N/A')} {result.get('unit', '')}"
            for r in levels if 'gumbel' in r
        ) or '- No Gumbel return levels available.'
        prompt = f"""Act as a professional hydrologist interpreting an extreme value analysis for a station.
Write the response in markdown and structure it exactly as follows:

**Executive Summary**
2-3 sentences summarising the overall flood-risk signal, how extreme the fitted return levels are, and whether the record length is adequate.

**Detailed Insights**
- **Return Levels:** interpret the 10-, 50-, and 100-year return levels using the reported values.
- **Tail Behaviour:** explain what the GEV shape parameter implies about the tail type and extreme-event behaviour.
- **Reliability:** comment on uncertainty, confidence intervals, and whether the record length supports robust inference.
- **Operational Interpretation:** state what the result means for planning, design, or flood preparedness.

Rules:
- Use professional hydrological language.
- Always cite specific numbers from the provided results.
- Replace underscores with spaces.
- Do not include any introduction or sign-off outside the two sections.

Station: {station}
Feature: {feature}
Unit: {result.get('unit', '') or 'no unit'}
Record length: {n_years} years ({yr[0] if yr else '?'}–{yr[1] if len(yr)>1 else '?'})
GEV parameters: {gev if gev else 'Not fitted (unstable)'}
Gumbel parameters: {gumbel}
GEV return levels:
{gev_lines}

Gumbel return levels:
{gumbel_lines}

95% confidence interval lower bounds: {result.get('ci_lower') or 'N/A'}
95% confidence interval upper bounds: {result.get('ci_upper') or 'N/A'}"""
        text = _gemini_generate(api_key, prompt)
        if not text:
            return _fallback_extreme_analysis(result)
        return markdown.markdown(text.strip())
    except Exception:
        return _fallback_extreme_analysis(result)


class ExtremeService:
    """
    Extreme value analysis and return period estimation.

    Workflow:
      1. Extract annual maxima from the full historical series.
      2. Fit GEV (Generalized Extreme Value) and/or Gumbel distributions via MLE.
      3. Compute return levels for standard return periods (2–200 yr).
      4. Bootstrap 95 % CI for GEV return levels (1000 resamples).
      5. Return table + Plotly figure (return-period curve + annual maxima bars).
    """

    RETURN_PERIODS: List[int] = [2, 5, 10, 25, 50, 100, 200]

    def __init__(self, repository) -> None:
        self.repo = repository

    # ── helpers ──────────────────────────────────────────────────────────────

    def _find_repo(self, station: str):
        if hasattr(self.repo, 'repos'):
            return next((r for r in self.repo.repos if station in r.station_index), None)
        if station in getattr(self.repo, 'station_index', {}):
            return self.repo
        return None

    def _load_series(self, repo, station: str, feature: str) -> pd.Series:
        fd = repo.station_index[station]['feature_details'][feature]
        req = SeriesRequest(
            station=station, feature=feature,
            start_date=fd['start_date'], end_date=fd['end_date'],
        )
        df = repo.get_feature_series(req)
        ts = df.set_index('Timestamp')['Value'].sort_index()
        ts.index = pd.to_datetime(ts.index)
        return ts.dropna()

    # ── main entry point ─────────────────────────────────────────────────────

    def compute(
        self,
        station: str,
        feature: str,
        distribution: str = 'gev',
        include_analysis: bool = False,
    ) -> Dict[str, Any]:
        repo = self._find_repo(station)
        if repo is None:
            raise ValueError(f"Station '{station}' not found.")

        meta = repo.station_index[station]
        valid_features = get_valid_features_for_analysis('extreme', meta.get('features', []))
        if feature not in valid_features:
            raise ValueError(f"'{feature}' not valid/available for extreme analysis on {station}.")


        unit = repo.feature_units.get(feature, '')
        ts = self._load_series(repo, station, feature)

        # Annual maxima (positive values only — prevents log-scale issues)
        annual_max = ts.groupby(ts.index.year).max().dropna()
        annual_max = annual_max[annual_max > 0]
        n_years = len(annual_max)
        if n_years < 5:
            raise ValueError(f"Need at least 5 years of positive data (got {n_years}).")

        values = annual_max.values.astype(float)

        # ── Fit GEV ──────────────────────────────────────────────────────────
        # Shape parameter is clamped to [-0.6, 0.6]: values outside this range
        # indicate MLE has not converged to a physically meaningful solution
        # (common with short records <40 yr) and produce astronomical return levels.
        gev_params: Optional[Dict] = None
        try:
            c, loc, scale = scipy.stats.genextreme.fit(values)
            if abs(c) <= 0.6:
                gev_params = {'shape': float(c), 'loc': float(loc), 'scale': float(scale)}
            else:
                gev_params = None  # unreliable fit — fall back to Gumbel only
        except Exception:
            pass

        # ── Fit Gumbel ───────────────────────────────────────────────────────
        gumbel_params: Optional[Dict] = None
        try:
            loc_g, scale_g = scipy.stats.gumbel_r.fit(values)
            gumbel_params = {'loc': float(loc_g), 'scale': float(scale_g)}
        except Exception:
            pass

        # ── Return levels ────────────────────────────────────────────────────
        return_levels = []
        for T in self.RETURN_PERIODS:
            p = 1.0 - 1.0 / T
            row: Dict[str, Any] = {'return_period': T}
            if gev_params:
                rl = scipy.stats.genextreme.ppf(
                    p, gev_params['shape'], gev_params['loc'], gev_params['scale'])
                row['gev'] = round(float(rl), 3)
            if gumbel_params:
                rl = scipy.stats.gumbel_r.ppf(p, gumbel_params['loc'], gumbel_params['scale'])
                row['gumbel'] = round(float(rl), 3)
            return_levels.append(row)

        # ── Empirical plotting positions (Weibull) ────────────────────────────
        sorted_vals = np.sort(values)
        m = np.arange(1, n_years + 1)
        empirical_T = (n_years + 1) / (n_years + 1 - m)  # ascending Weibull

        # ── Smooth fitted curves ──────────────────────────────────────────────
        T_range = np.logspace(np.log10(1.05), np.log10(500), 300)
        p_range = 1.0 - 1.0 / T_range

        gev_curve: Optional[List[float]] = None
        gumbel_curve: Optional[List[float]] = None
        if gev_params:
            gev_curve = scipy.stats.genextreme.ppf(
                p_range, gev_params['shape'], gev_params['loc'], gev_params['scale']
            ).tolist()
        if gumbel_params:
            gumbel_curve = scipy.stats.gumbel_r.ppf(
                p_range, gumbel_params['loc'], gumbel_params['scale']
            ).tolist()

        # ── Bootstrap 95 % CI for GEV ─────────────────────────────────────────
        ci_lower: Optional[List[float]] = None
        ci_upper: Optional[List[float]] = None
        if gev_params and n_years >= 10:
            rng = np.random.default_rng(42)
            boot_levels: List[List[float]] = []
            for _ in range(1000):
                sample = rng.choice(values, size=n_years, replace=True)
                try:
                    c_b, l_b, s_b = scipy.stats.genextreme.fit(sample)
                    lvls = [
                        float(scipy.stats.genextreme.ppf(1 - 1 / T, c_b, l_b, s_b))
                        for T in self.RETURN_PERIODS
                    ]
                    boot_levels.append(lvls)
                except Exception:
                    pass
            if len(boot_levels) >= 10:
                arr = np.array(boot_levels)
                ci_lower = np.percentile(arr, 2.5, axis=0).tolist()
                ci_upper = np.percentile(arr, 97.5, axis=0).tolist()

        # ── Build figure ──────────────────────────────────────────────────────
        figure = self._build_figure(
            annual_max, sorted_vals, empirical_T, T_range,
            gev_curve, gumbel_curve, return_levels, ci_lower, ci_upper,
            station, feature, unit, gev_params, gumbel_params, distribution,
        )

        result = {
            'station': station,
            'feature': feature,
            'unit': unit,
            'n_years': n_years,
            'year_range': [int(annual_max.index[0]), int(annual_max.index[-1])],
            'annual_maxima': [
                {'year': int(y), 'value': round(float(v), 3)}
                for y, v in annual_max.items()
            ],
            'return_levels': return_levels,
            'gev_params': gev_params,
            'gumbel_params': gumbel_params,
            'ci_lower': [round(v, 3) for v in ci_lower] if ci_lower else None,
            'ci_upper': [round(v, 3) for v in ci_upper] if ci_upper else None,
            'figure': plotly.io.to_json(figure),
            'method_note': (
                'Annual maxima fitted to GEV and Gumbel distributions by MLE (Coles 2001). '
                'GEV shape parameter clamped to [-0.6, 0.6] to guard against numerically '
                'unstable fits on short records. '
                'Weibull plotting position i/(n+1) used for empirical return periods (Cunnane 1978). '
                f'95% CI via non-parametric bootstrap with 1000 resamples. '
                + ('L-moment fitting recommended for records < 30 years; '
                   f'MLE used here (n={n_years} yr).' if n_years < 30 else '')
            ),
        }

        if include_analysis:
            analysis = _generate_extreme_analysis(result)
            if analysis:
                result['analysis'] = analysis

        return result

    # ── figure builder ────────────────────────────────────────────────────────

    def _build_figure(
        self,
        annual_max: pd.Series,
        sorted_vals: np.ndarray,
        empirical_T: np.ndarray,
        T_range: np.ndarray,
        gev_curve: Optional[List[float]],
        gumbel_curve: Optional[List[float]],
        return_levels: List[Dict],
        ci_lower: Optional[List[float]],
        ci_upper: Optional[List[float]],
        station: str,
        feature: str,
        unit: str,
        gev_params: Optional[Dict],
        gumbel_params: Optional[Dict],
        distribution: str,
    ) -> go.Figure:

        BLUE   = '#38bdf8'
        ORANGE = '#fb923c'
        GREEN  = '#34d399'
        feature_label = feature.replace('_', ' ')

        show_gev    = distribution in ('gev', 'both')    and gev_params    is not None
        show_gumbel = distribution in ('gumbel', 'both') and gumbel_params is not None

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.60, 0.40],
            vertical_spacing=0.14,
            subplot_titles=(
                f'{feature_label} — Return Period Curve',
                f'Annual Maxima ({int(annual_max.index[0])}–{int(annual_max.index[-1])})',
            ),
        )

        T_pts = [r['return_period'] for r in return_levels]

        # GEV 95 % confidence band
        if show_gev and ci_lower and ci_upper:
            fig.add_trace(go.Scatter(
                x=T_pts + T_pts[::-1],
                y=ci_upper + ci_lower[::-1],
                fill='toself',
                fillcolor='rgba(56,189,248,0.10)',
                line=dict(width=0),
                name='GEV 95% CI',
                showlegend=True,
                hoverinfo='skip',
            ), row=1, col=1)

        # GEV fitted curve
        if show_gev and gev_curve:
            fig.add_trace(go.Scatter(
                x=T_range.tolist(), y=gev_curve,
                mode='lines', name='GEV fit',
                line=dict(color=BLUE, width=2.5),
                hovertemplate='T=%{x:.1f} yr — %{y:.2f} ' + unit + '<extra>GEV</extra>',
            ), row=1, col=1)

        # Gumbel fitted curve
        if show_gumbel and gumbel_curve:
            fig.add_trace(go.Scatter(
                x=T_range.tolist(), y=gumbel_curve,
                mode='lines', name='Gumbel fit',
                line=dict(color=ORANGE, width=2, dash='dash'),
                hovertemplate='T=%{x:.1f} yr — %{y:.2f} ' + unit + '<extra>Gumbel</extra>',
            ), row=1, col=1)

        # Empirical points
        fig.add_trace(go.Scatter(
            x=empirical_T.tolist(), y=sorted_vals.tolist(),
            mode='markers', name='Empirical (Weibull)',
            marker=dict(color=GREEN, size=7, symbol='circle-open', line=dict(width=2)),
            hovertemplate='T=%{x:.1f} yr — %{y:.2f} ' + unit + '<extra>Empirical</extra>',
        ), row=1, col=1)

        # Reference vertical lines at T = 10, 50, 100 yr
        for T_ref in [10, 50, 100]:
            fig.add_shape(
                type='line',
                x0=T_ref, x1=T_ref, y0=0, y1=1,
                xref='x', yref='y domain',
                line=dict(dash='dot', color='rgba(157,176,209,0.30)', width=1),
            )
            fig.add_annotation(
                x=T_ref, y=1.0,
                xref='x', yref='y domain',
                text=f'{T_ref}yr',
                showarrow=False,
                xanchor='center', yanchor='bottom',
                font=dict(size=9, color='rgba(157,176,209,0.65)'),
            )

        # GEV tail-type annotation (bottom-right of return-period panel)
        if gev_params:
            xi = gev_params['shape']
            dist_type = 'Fréchet' if xi > 0.05 else ('Weibull' if xi < -0.05 else 'Gumbel-approx')
            fig.add_annotation(
                x=1.0, y=0.0,
                xref='paper', yref='paper',
                text=f'ξ = {xi:.3f} ({dist_type})  ·  n = {len(sorted_vals)} yr',
                showarrow=False, xanchor='right', yanchor='bottom',
                font=dict(size=9, color=SOFT),
            )

        # Annual maxima bar chart
        fig.add_trace(go.Bar(
            x=annual_max.index.tolist(), y=annual_max.values.tolist(),
            name='Annual max',
            marker_color=BLUE, opacity=0.65,
            hovertemplate='%{x} — %{y:.2f} ' + unit + '<extra>Annual max</extra>',
        ), row=2, col=1)

        # Mean reference line on annual maxima panel
        mean_val = float(annual_max.mean())
        fig.add_shape(
            type='line',
            x0=annual_max.index[0], x1=annual_max.index[-1],
            y0=mean_val, y1=mean_val,
            xref='x2', yref='y2',
            line=dict(dash='dot', color=SOFT, width=1),
        )

        # Sensible y-axis cap: 20× the largest observed annual maximum
        y_cap = float(sorted_vals.max()) * 20

        fig.update_layout(**dark_layout(
            height=620,
            margin=MARGIN_SUBPLOT,
            show_legend=True,
            hovermode='x unified',
        ))
        fig.update_layout(legend=legend_h())

        _ax = axis_style()
        fig.update_xaxes(**_ax)
        fig.update_yaxes(**_ax)
        fig.update_xaxes(
            type='log', title_text='Return Period (years)',
            range=[0, np.log10(500)],
            row=1, col=1,
        )
        fig.update_yaxes(
            title_text=unit or feature_label,
            range=[0, y_cap],
            title_standoff=10,
            row=1, col=1,
        )
        fig.update_xaxes(title_text='Year', row=2, col=1)
        fig.update_yaxes(title_text=f'Max ({unit})' if unit else 'Max', title_standoff=10, row=2, col=1)

        style_subplot_titles(fig)

        return fig
