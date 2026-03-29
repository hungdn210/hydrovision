from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io
import scipy.stats
from plotly.subplots import make_subplots

from .data_loader import SeriesRequest


class ExtremeService:
    """
    Extreme value analysis and return period estimation.

    Workflow:
      1. Extract annual maxima from the full historical series.
      2. Fit GEV (Generalized Extreme Value) and/or Gumbel distributions via MLE.
      3. Compute return levels for standard return periods (2–200 yr).
      4. Bootstrap 95 % CI for GEV return levels (150 resamples).
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
    ) -> Dict[str, Any]:
        repo = self._find_repo(station)
        if repo is None:
            raise ValueError(f"Station '{station}' not found.")

        meta = repo.station_index[station]
        if feature not in meta.get('features', []):
            raise ValueError(f"'{feature}' not available for {station}.")

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
        gev_params: Optional[Dict] = None
        try:
            c, loc, scale = scipy.stats.genextreme.fit(values)
            gev_params = {'shape': float(c), 'loc': float(loc), 'scale': float(scale)}
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
            for _ in range(150):
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

        return {
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
        }

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

        DARK_BG = '#07111f'
        GRID = 'rgba(148,163,184,0.12)'
        TEXT = '#e5eefc'
        SOFT = '#9db0d1'
        BLUE = '#38bdf8'
        ORANGE = '#fb923c'
        GREEN = '#34d399'
        feature_label = feature.replace('_', ' ')

        show_gev = distribution in ('gev', 'both') and gev_params is not None
        show_gumbel = distribution in ('gumbel', 'both') and gumbel_params is not None

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.62, 0.38],
            vertical_spacing=0.10,
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
                hovertemplate='T=%{x:.1f} yr<br>%{y:.2f} ' + unit + '<extra>GEV</extra>',
            ), row=1, col=1)

        # Gumbel fitted curve
        if show_gumbel and gumbel_curve:
            fig.add_trace(go.Scatter(
                x=T_range.tolist(), y=gumbel_curve,
                mode='lines', name='Gumbel fit',
                line=dict(color=ORANGE, width=2, dash='dash'),
                hovertemplate='T=%{x:.1f} yr<br>%{y:.2f} ' + unit + '<extra>Gumbel</extra>',
            ), row=1, col=1)

        # Empirical points
        fig.add_trace(go.Scatter(
            x=empirical_T.tolist(), y=sorted_vals.tolist(),
            mode='markers', name='Empirical (Weibull)',
            marker=dict(color=GREEN, size=7, symbol='circle-open', line=dict(width=2)),
            hovertemplate='T=%{x:.1f} yr<br>%{y:.2f} ' + unit + '<extra>Empirical</extra>',
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

        # GEV tail-type annotation
        if gev_params:
            xi = gev_params['shape']
            dist_type = 'Fréchet' if xi > 0.05 else ('Weibull' if xi < -0.05 else 'Gumbel-approx')
            fig.add_annotation(
                x=1.0, y=0.0,
                xref='paper', yref='paper',
                text=f'ξ={xi:.3f} ({dist_type})  |  n={len(sorted_vals)} yr',
                showarrow=False, xanchor='right', yanchor='bottom',
                font=dict(size=9, color=SOFT),
            )

        # Annual maxima bar chart
        fig.add_trace(go.Bar(
            x=annual_max.index.tolist(), y=annual_max.values.tolist(),
            name='Annual max',
            marker_color=BLUE, opacity=0.65,
            hovertemplate='%{x}<br>%{y:.2f} ' + unit + '<extra>Annual max</extra>',
        ), row=2, col=1)

        # Mean reference line
        mean_val = float(annual_max.mean())
        fig.add_shape(
            type='line',
            x0=annual_max.index[0], x1=annual_max.index[-1],
            y0=mean_val, y1=mean_val,
            xref='x2', yref='y2',
            line=dict(dash='dot', color=SOFT, width=1),
        )

        fig.update_layout(
            paper_bgcolor=DARK_BG,
            plot_bgcolor=DARK_BG,
            font=dict(family='Inter, sans-serif', color=TEXT, size=12),
            legend=dict(
                orientation='h', yanchor='bottom', y=1.02,
                xanchor='left', x=0,
                font=dict(size=11), bgcolor='rgba(0,0,0,0)',
            ),
            margin=dict(l=10, r=10, t=60, b=10),
            hovermode='x unified',
        )
        axis_style = dict(
            gridcolor=GRID, zerolinecolor=GRID,
            linecolor=GRID, tickfont=dict(size=10, color=SOFT),
        )
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style)
        fig.update_xaxes(type='log', title_text='Return Period (years)', row=1, col=1)
        fig.update_yaxes(title_text=unit or feature_label, row=1, col=1)
        fig.update_xaxes(title_text='Year', row=2, col=1)
        fig.update_yaxes(title_text=f'Max ({unit})' if unit else 'Max', row=2, col=1)
        for ann in fig['layout']['annotations'][:2]:
            ann['font'] = dict(size=12, color=SOFT)

        return fig
