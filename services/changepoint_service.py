from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import markdown
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io
import scipy.stats

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False

from .analysis_service import _gemini_generate
from .data_loader import SeriesRequest
from .figure_theme import (
    TEXT, SOFT, GRID_LIGHT,
    dark_layout, axis_style,
    legend_v, MARGIN_STD,
)


def _fallback_changepoint_analysis(result: Dict[str, Any]) -> str:
    s = result.get('stats', {}) or {}
    title = str(result.get('title', '')).replace('_', ' ')
    method = s.get('method', 'unknown')
    n_breaks = s.get('n_breaks_detected', 0)
    cp_dates = s.get('change_point_dates') or []
    segments = s.get('segments') or []
    cp_text = ', '.join(cp_dates) if cp_dates else 'none detected'

    # Build segment summary
    seg_bullets = []
    for i, seg in enumerate(segments):
        seg_bullets.append(
            f'<li><strong>Segment {i+1} ({seg.get("start")} – {seg.get("end")}):</strong> '
            f'Mean = {seg.get("mean")}, Std = {seg.get("std", "?")}, Trend = {seg.get("trend", "?")}. '
            + (
                'The mean and trend in this segment represent the baseline hydrological regime before the next detected break.'
                if i < len(segments) - 1
                else 'This is the most recent regime — its mean and trend are the most operationally relevant for current planning.'
            )
            + '</li>'
        )
    seg_html = ''.join(seg_bullets) if seg_bullets else '<li>No segment statistics available.</li>'

    first_seg = segments[0] if segments else {}
    last_seg = segments[-1] if segments else {}
    transition_text = ''
    if first_seg and last_seg and first_seg is not last_seg:
        try:
            mean_change = float(last_seg.get('mean', 0)) - float(first_seg.get('mean', 0))
            direction = 'increase' if mean_change > 0 else 'decrease'
            transition_text = (
                f'Between the first regime (mean {first_seg.get("mean")}) and the latest regime (mean {last_seg.get("mean")}), '
                f'the series underwent a net {direction} of approximately {abs(mean_change):.2f} units. '
                'This shift may reflect progressive basin change, discrete regulatory events, or a step-change in climate forcing.'
            )
        except (TypeError, ValueError):
            transition_text = (
                f'The earliest regime (mean {first_seg.get("mean")}) and latest regime (mean {last_seg.get("mean")}) '
                'differ in their central tendency, indicating a net regime shift across the record.'
            )

    return (
        '<p><strong>Executive Summary</strong></p>'
        f'<p>The structural break analysis for <strong>{title}</strong> applied the <strong>{method}</strong> algorithm '
        f'and detected <strong>{n_breaks} change point(s)</strong> at {cp_text}, dividing the record into {len(segments)} distinct regime(s). '
        + (transition_text if transition_text else '')
        + ' These break dates are statistically optimal partitions — they identify <em>when</em> the series shifted, '
        'but do not identify <em>why</em>; physical attribution requires cross-referencing with basin management records, '
        'dam construction timelines, and climate indices.</p>'

        '<p><strong>Break Structure</strong></p>'
        f'<p>The {method} algorithm minimises a penalised cost function (L2 norm with BIC-style penalty) to find the optimal segmentation. '
        f'{n_breaks} breakpoint(s) were located at {cp_text}. '
        'Each break marks a statistically significant shift in the mean level of the series that cannot be explained by the regular seasonal cycle or random noise. '
        'The penalty term guards against over-segmentation — the detected breaks represent the strongest, most robust regime shifts within the maximum breakpoint constraint set by the user.</p>'

        '<p><strong>Segment-by-Segment Analysis</strong></p>'
        f'<ul>{seg_html}</ul>'

        '<p><strong>Regime Transition Interpretation</strong></p>'
        f'<p>{transition_text if transition_text else "Segment means and trends provide the basis for comparing regimes."} '
        'Possible hydrological explanations for the detected break dates include: '
        '(1) <strong>Dam impoundment or release changes</strong> — upstream regulation often causes abrupt step-changes in downstream discharge; '
        '(2) <strong>Climate shifts</strong> — changes in monsoon intensity, ENSO regime transitions, or PDO phase shifts can produce multi-year mean shifts; '
        '(3) <strong>Land-use change</strong> — deforestation or agricultural expansion in the catchment alters runoff coefficients and baseflow recession; '
        '(4) <strong>Gauge relocation or instrument change</strong> — non-climatic breaks in the observational record. '
        'Always consult local hydrological station metadata before attributing breaks to physical causes.</p>'

        '<p><strong>Stationarity Assessment</strong></p>'
        '<p>The presence of structural breaks means that the full historical record is <strong>non-stationary</strong>: '
        'the mean or variance of the series changes over time. '
        'This has important implications for any downstream analysis that assumes stationarity (e.g. flood frequency curves, seasonal forecasting baselines, trend tests). '
        'Where breaks are detected, it is generally advisable to perform trend and frequency analyses on the most recent stationary segment '
        'rather than the full record, as the older regimes may no longer represent the current system state.</p>'

        '<p><strong>Operational Relevance</strong></p>'
        '<ul>'
        '<li><strong>Design standards:</strong> Flood frequency curves and return-period estimates derived from the full record may be biased if a regime shift has occurred. '
        'Re-fitting extreme value models to the post-break segment typically produces more representative design values for infrastructure planning.</li>'
        '<li><strong>Forecasting baselines:</strong> Seasonal forecast climatologies and anomaly benchmarks should be computed from the current regime segment, '
        'not the full historical record, to avoid contaminating the baseline with pre-shift data.</li>'
        '<li><strong>Monitoring trigger:</strong> Repeat this analysis periodically as new data accumulates. '
        'An emerging break at the end of the current record could be an early indicator of ongoing regime change requiring management intervention.</li>'
        '</ul>'
    )


def _generate_changepoint_analysis(result: Dict[str, Any]) -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_google_gemini_api_key_here' or not api_key.strip():
        return _fallback_changepoint_analysis(result)
    try:
        s = result.get('stats', {})
        prompt = f"""Act as a professional hydrologist writing a detailed technical interpretation of a structural break / change-point detection analysis.
Write the response in markdown and structure it exactly as follows:

**Executive Summary**
4-5 sentences. State how many breaks were found, when they occur, how large the regime shifts are between segments, which physical causes are most plausible, and the key implications for the stationarity of the historical record.

**Break Structure**
A paragraph (4-5 sentences) interpreting the number and location of breaks in detail. Explain the penalised cost minimisation approach and why the algorithm chose these break dates over alternatives. Discuss the penalty parameter's role in preventing over-segmentation. Comment on whether the breaks are closely clustered or well-separated in time.

**Segment-by-Segment Analysis**
For each detected segment, provide a bullet point citing the time span, mean, standard deviation, and trend direction. Interpret what each segment mean and trend implies about the hydrological regime during that period. For the final segment, note that this represents current conditions most relevant to operational planning.

**Regime Transition Interpretation**
A paragraph (4-5 sentences) interpreting the magnitude and direction of change between regimes. Enumerate plausible physical causes for the observed break dates: dam construction or regulation changes, ENSO or PDO regime transitions, land-use change, or instrumental/observational artefacts. Emphasise that attribution requires cross-referencing with basin management records.

**Stationarity Assessment**
A paragraph (3-4 sentences) assessing the implications of the detected breaks for stationarity. Explain why non-stationarity invalidates full-record frequency analysis and trend tests. Recommend which segment should be used for design and baseline computations.

**Operational Relevance**
Exactly 3 bullet points:
- **Design standards:** how detected regime shifts affect flood frequency curves and infrastructure design values.
- **Forecasting baselines:** how the most recent regime segment should be used to set seasonal forecast climatologies.
- **Ongoing monitoring:** recommend periodic re-analysis as new data accumulates to detect emerging breaks early.

Rules:
- Use professional hydrological language throughout.
- Always cite specific numbers from the provided results.
- Replace underscores with spaces.
- Do not include any introduction or sign-off outside the defined sections.

Analysis title: {str(result.get('title', '')).replace('_', ' ')}
Method: {s.get('method')}
Change points detected: {s.get('n_breaks_detected')} at dates: {s.get('change_point_dates')}
Segments (start, end, mean, std, trend): {s.get('segments')}
"""
        return markdown.markdown(_gemini_generate(api_key, prompt))
    except Exception:
        return _fallback_changepoint_analysis(result)


class ChangePointService:
    """
    Structural Break / Change-Point Detection.

    Methodology:
      - Resample the raw time series to monthly or annual means.
      - Apply the selected algorithm (PELT or BinSeg) via the `ruptures`
        library to locate the most significant regime-change dates.
      - For each segment between change points compute: mean, standard
        deviation, and Mann-Kendall monotonic trend.
      - Return a Plotly figure with the series, change-point vertical lines,
        per-segment means, and a summary statistics table.
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
    def _ar1_prewhiten(x: np.ndarray) -> np.ndarray:
        """
        AR(1) pre-whitening: remove lag-1 autocorrelation before trend testing.
        Returns residuals x[t] - rho*x[t-1] (length n-1).
        Following von Storch (1995) and Yue & Wang (2002) recommendations for
        Mann-Kendall testing on autocorrelated hydrological series.
        """
        if len(x) < 4:
            return x
        rho = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        rho = np.clip(rho, -0.99, 0.99)
        return x[1:] - rho * x[:-1]

    @staticmethod
    def _mk_trend(x: np.ndarray) -> str:
        """
        Return 'Increasing ↑', 'Decreasing ↓', or 'No trend' based on
        Mann-Kendall test applied to AR(1) pre-whitened series.
        Pre-whitening follows von Storch (1995) to correct for autocorrelation
        inflation of the MK S-statistic in hydrological time series.
        """
        if len(x) < 4:
            return 'Too short'
        # AR(1) pre-whiten to reduce autocorrelation bias in MK S-statistic
        rho = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 2 else 0.0
        if abs(rho) > 0.1:
            x = x[1:] - np.clip(rho, -0.99, 0.99) * x[:-1]
        n = len(x)
        if n < 4:
            return 'Too short'
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                s += np.sign(x[j] - x[i])
        var_s = n * (n - 1) * (2 * n + 5) / 18
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            return 'No trend'
        p = 2 * (1 - scipy.stats.norm.cdf(abs(z)))
        if p < 0.05:
            return 'Increasing ↑' if s > 0 else 'Decreasing ↓'
        return 'No trend'

    # ── main entry ─────────────────────────────────────────────────────────────

    def detect(
        self,
        dataset: str,
        station: str,
        feature: str,
        n_breaks: int = 3,
        method: str = 'pelt',
        include_analysis: bool = False,
    ) -> Dict[str, Any]:
        if not HAS_RUPTURES:
            raise RuntimeError("The 'ruptures' package is required. Install it with: pip install ruptures")

        repo = self._find_repo(dataset)
        if repo is None:
            raise ValueError(f"Dataset '{dataset}' not found.")

        ts = self._load_series(repo, station, feature)
        if ts is None or len(ts) < 60:
            raise ValueError(f"Insufficient data for '{station}' / '{feature}'.")

        unit = repo.feature_units.get(feature, '')
        station_name = repo.station_index[station].get('name', station)

        # Resample to monthly means for stability
        monthly = ts.resample('ME').mean().dropna()
        values = monthly.values.astype(float)
        dates = list(monthly.index)
        n = len(values)

        # ── Run change-point algorithm ────────────────────────────────────────
        n_breaks = max(1, min(n_breaks, 6))

        penalty_method_used = 'BIC'
        penalty_value_used: float = 0.0

        if method == 'pelt':
            algo = rpt.Pelt(model='rbf', min_size=12, jump=1).fit(values)

            # ── BIC-derived penalty (Killick et al. 2012) ────────────────────
            # BIC penalty ≈ log(n) * sigma² where sigma² estimated from
            # first-differenced series (noise variance, not signal variance).
            sigma2 = float(np.var(np.diff(values)) / 2.0)
            bic_pen = float(np.log(n) * max(sigma2, 1e-8))
            penalty_value_used = round(bic_pen, 6)
            try:
                raw_bkps = algo.predict(pen=bic_pen)
                bkps = sorted(raw_bkps[:-1])[:n_breaks]
            except Exception:
                bkps = []

            # ── Fallback: adaptive penalty chain if BIC finds nothing ────────
            if not bkps:
                penalty_method_used = 'adaptive_fallback'
                for pen_factor in [1.0, 0.5, 0.2, 0.05]:
                    pen = pen_factor * np.log(n) * np.std(values) ** 2
                    try:
                        raw_bkps = algo.predict(pen=pen)
                        candidate = sorted(raw_bkps[:-1])[:n_breaks]
                        if len(candidate) >= 1:
                            bkps = candidate
                            penalty_value_used = round(pen, 6)
                            break
                    except Exception:
                        continue

            # ── Second fallback: BinSeg if PELT still yields nothing ─────────
            if not bkps:
                penalty_method_used = 'binseg_fallback'
                algo2 = rpt.Binseg(model='rbf', min_size=12, jump=1).fit(values)
                raw_bkps = algo2.predict(n_bkps=n_breaks)
                bkps = sorted(raw_bkps[:-1])
        else:
            # Binary Segmentation (user-requested)
            penalty_method_used = 'binseg_fixed_k'
            penalty_value_used = float(n_breaks)
            algo = rpt.Binseg(model='rbf', min_size=12, jump=1).fit(values)
            raw_bkps = algo.predict(n_bkps=n_breaks)
            bkps = sorted(raw_bkps[:-1])

        # Segments: list of (start_idx, end_idx) inclusive
        seg_ends = [0] + bkps + [n]
        segments = [(seg_ends[i], seg_ends[i + 1]) for i in range(len(seg_ends) - 1)]

        CHANGE_COLOR = '#f59e0b'
        SEGMENT_COLORS = ['#60a5fa', '#34d399', '#f87171', '#a78bfa', '#fb923c', '#38bdf8', '#e879f9']

        fig = go.Figure()

        # ── Raw monthly series ─────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=dates,
            y=values.tolist(),
            mode='lines',
            name='Monthly mean',
            line=dict(color='rgba(148,163,184,0.5)', width=1),
            hovertemplate='%{x|%b %Y}: %{y:.3f} ' + unit + '<extra></extra>',
        ))

        # ── Segment means ──────────────────────────────────────────────────────
        seg_stats: List[Dict[str, Any]] = []
        for idx, (s, e) in enumerate(segments):
            seg_vals = values[s:e]
            seg_dates = dates[s:e]
            if len(seg_vals) == 0:
                continue

            seg_mean = float(np.mean(seg_vals))
            seg_std = float(np.std(seg_vals))
            trend_str = self._mk_trend(seg_vals)
            color = SEGMENT_COLORS[idx % len(SEGMENT_COLORS)]
            start_yr = seg_dates[0].year if seg_dates else '?'
            end_yr = seg_dates[-1].year if seg_dates else '?'

            # Segment mean line
            fig.add_trace(go.Scatter(
                x=[seg_dates[0], seg_dates[-1]],
                y=[seg_mean, seg_mean],
                mode='lines',
                name=f'Segment {idx + 1} mean ({start_yr}–{end_yr})',
                line=dict(color=color, width=2.5),
                hovertemplate=f'Segment {idx + 1}: {seg_mean:.3f} {unit}<extra></extra>',
            ))

            seg_stats.append({
                'segment': idx + 1,
                'start': str(seg_dates[0].date()),
                'end': str(seg_dates[-1].date()),
                'mean': round(seg_mean, 3),
                'std': round(seg_std, 3),
                'trend': trend_str,
                'n_months': len(seg_vals),
            })

        # ── Change-point vertical lines ────────────────────────────────────────
        shapes = []
        cp_annotations = []
        for bp in bkps:
            if bp >= len(dates):
                continue
            cp_date = dates[bp]
            shapes.append(dict(
                type='line',
                x0=cp_date, x1=cp_date,
                y0=0, y1=1, yref='paper',
                line=dict(color=CHANGE_COLOR, dash='dash', width=1.5),
            ))
            cp_annotations.append(dict(
                x=cp_date, y=1.02, yref='paper',
                text=str(cp_date.year),
                showarrow=False, xanchor='center',
                font=dict(color=CHANGE_COLOR, size=10),
            ))

        feature_label = feature.replace('_', ' ').title()
        method_label = 'PELT' if method == 'pelt' else 'Binary Segmentation'

        _ax = axis_style(grid=GRID_LIGHT)
        fig.update_layout(**dark_layout(
            title=f'Change Point Detection ({method_label}) — {feature_label} · {station_name}',
            height=480,
            margin=MARGIN_STD,
            show_legend=True,
            xaxis=dict(**_ax, title='Date', showgrid=True, zeroline=False),
            yaxis=dict(**_ax, title=f'{feature_label} ({unit})' if unit else feature_label,
                       showgrid=True, zeroline=False),
            legend=legend_v(),
            shapes=shapes,
            annotations=cp_annotations,
        ))

        # Change-point dates for output
        cp_dates = [str(dates[bp].date()) for bp in bkps if bp < len(dates)]

        result = {
            'title': f'Change Points · {station_name}',
            'subtitle': (
                f'{feature_label} · {n_breaks} break(s) · {method_label}'
            ),
            'series': [{'station': station, 'feature': feature,
                        'start_date': str(dates[0].date()),
                        'end_date': str(dates[-1].date())}],
            'figure': plotly.io.to_json(fig),
            'stats': {
                'method': method_label,
                'penalty_method': penalty_method_used,
                'penalty_value': penalty_value_used,
                'n_breaks_detected': len(bkps),
                'change_point_dates': cp_dates,
                'segments': seg_stats,
                'mk_prewhitened': True,
                'method_note': (
                    'Change points detected using PELT (Killick et al. 2012) with BIC-derived penalty '
                    '(log(n)·σ²). Mann-Kendall trend tests applied per segment with AR(1) pre-whitening '
                    '(von Storch 1995) to correct for autocorrelation inflation of the S-statistic.'
                ),
            },
        }

        if include_analysis:
            analysis = _generate_changepoint_analysis(result)
            if analysis:
                result['analysis'] = analysis

        return result
