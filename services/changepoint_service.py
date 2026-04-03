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


def _fallback_changepoint_analysis(result: Dict[str, Any]) -> str:
    s = result.get('stats', {}) or {}
    title = str(result.get('title', '')).replace('_', ' ')
    method = s.get('method')
    n_breaks = s.get('n_breaks_detected', 0)
    cp_dates = s.get('change_point_dates') or []
    segments = s.get('segments') or []

    if segments:
        first_seg = segments[0]
        last_seg = segments[-1]
        transition_text = (
            f'The first detected regime spans {first_seg.get("start")} to {first_seg.get("end")} with mean {first_seg.get("mean")} and trend {first_seg.get("trend")}, '
            f'while the latest regime spans {last_seg.get("start")} to {last_seg.get("end")} with mean {last_seg.get("mean")} and trend {last_seg.get("trend")}.'
        )
    else:
        transition_text = 'Segment-level statistics are limited, so the result should be interpreted mainly from the detected break dates.'

    cp_text = ', '.join(cp_dates) if cp_dates else 'none detected'
    return (
        '<p><strong>Executive Summary</strong></p>'
        f'<p>The change-point analysis for <strong>{title}</strong> used the <strong>{method}</strong> method and detected {n_breaks} structural break(s) at {cp_text}. '
        'These break dates indicate candidate shifts in the underlying hydrological regime rather than proof of a specific physical cause.</p>'
        '<p><strong>Detailed Insights</strong></p>'
        '<ul>'
        f'<li><strong>Break Structure:</strong> {n_breaks} change point(s) were detected, separating the series into {len(segments)} segment(s).</li>'
        f'<li><strong>Segment Behaviour:</strong> {transition_text}</li>'
        f'<li><strong>Hydrological Interpretation:</strong> Break dates at {cp_text} may reflect shifts in climate forcing, regulation, land-use change, or observation regime, but they should be cross-checked with basin history.</li>'
        '<li><strong>Operational Interpretation:</strong> Treat the detected breaks as evidence that a single stationary model may be inappropriate across the full record, and consider segment-wise analysis for trend, forecasting, or design studies.</li>'
        '</ul>'
    )


def _generate_changepoint_analysis(result: Dict[str, Any]) -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_google_gemini_api_key_here' or not api_key.strip():
        return _fallback_changepoint_analysis(result)
    try:
        s = result.get('stats', {})
        prompt = f"""Act as a professional hydrologist interpreting a structural-break analysis.
Write the response in markdown and structure it exactly as follows:

**Executive Summary**
2-3 sentences summarising how many breaks were found, when they occur, and why they matter for interpreting the historical record.

**Detailed Insights**
- **Break Structure:** interpret the number and dates of the detected change points.
- **Segment Behaviour:** compare the reported segment means, variability, and trends.
- **Hydrological Interpretation:** explain plausible hydrological meanings of the regime shifts.
- **Operational Interpretation:** state how this result should affect downstream analysis and decision-making.

Rules:
- Use professional hydrological language.
- Always cite specific numbers from the provided results.
- Replace underscores with spaces.
- Do not include any introduction or sign-off outside the two sections.

Analysis title: {str(result.get('title', '')).replace('_', ' ')}
Method: {s.get('method')}
Change points detected: {s.get('n_breaks_detected')} at dates: {s.get('change_point_dates')}
Segments: {s.get('segments')}
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
    def _mk_trend(x: np.ndarray) -> str:
        """Return 'Increasing ↑', 'Decreasing ↓', or 'No trend' based on Mann-Kendall."""
        if len(x) < 4:
            return 'Too short'
        n = len(x)
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

        if method == 'pelt':
            # PELT with RBF cost — try increasingly lower penalties until we get breaks
            algo = rpt.Pelt(model='rbf', min_size=12, jump=1).fit(values)
            bkps = []
            for pen_factor in [1.0, 0.5, 0.2, 0.05]:
                pen = pen_factor * np.log(n) * np.std(values) ** 2
                raw_bkps = algo.predict(pen=pen)
                candidate = sorted(raw_bkps[:-1])[:n_breaks]
                if len(candidate) >= 1:
                    bkps = candidate
                    break
            # If still none, fall back to BinSeg
            if not bkps:
                algo2 = rpt.Binseg(model='rbf', min_size=12, jump=1).fit(values)
                raw_bkps = algo2.predict(n_bkps=n_breaks)
                bkps = sorted(raw_bkps[:-1])
        else:
            # Binary Segmentation
            algo = rpt.Binseg(model='rbf', min_size=12, jump=1).fit(values)
            raw_bkps = algo.predict(n_bkps=n_breaks)
            bkps = sorted(raw_bkps[:-1])

        # Segments: list of (start_idx, end_idx) inclusive
        seg_ends = [0] + bkps + [n]
        segments = [(seg_ends[i], seg_ends[i + 1]) for i in range(len(seg_ends) - 1)]

        DARK_BG = '#07111f'
        TEXT = '#e5eefc'
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
            hovertemplate='%{x|%Y-%m}: %{y:.3f} ' + unit + '<extra></extra>',
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

        fig.update_layout(
            paper_bgcolor=DARK_BG,
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, sans-serif', color=TEXT, size=12),
            title=dict(
                text=f'Change Point Detection ({method_label}) — {feature_label} · {station_name}',
                font=dict(size=14, color=TEXT),
                x=0.5, xanchor='center',
            ),
            xaxis=dict(
                title='Date',
                gridcolor='rgba(148,163,184,0.08)',
                showgrid=True, zeroline=False,
            ),
            yaxis=dict(
                title=f'{feature_label} ({unit})' if unit else feature_label,
                gridcolor='rgba(148,163,184,0.08)',
                showgrid=True, zeroline=False,
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
            shapes=shapes,
            annotations=cp_annotations,
            margin=dict(l=60, r=20, t=50, b=50),
        )

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
                'n_breaks_detected': len(bkps),
                'change_point_dates': cp_dates,
                'segments': seg_stats,
            },
        }

        if include_analysis:
            analysis = _generate_changepoint_analysis(result)
            if analysis:
                result['analysis'] = analysis

        return result
