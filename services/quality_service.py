from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io
from plotly.subplots import make_subplots

import markdown

from .analysis_service import _gemini_generate
from .data_loader import SeriesRequest


def generate_quality_analysis(view: str, result: Dict[str, Any]) -> str:
    api_key = os.getenv('GEMINI_API_KEY')
    fallback = _fallback_quality_analysis(view, result)
    if not api_key or api_key == 'your_google_gemini_api_key_here' or not api_key.strip():
        return fallback
    try:
        prompt = _quality_prompt(view, result)
        return markdown.markdown(_gemini_generate(api_key, prompt))
    except Exception:
        return _fallback_quality_analysis(view, result)


def _quality_prompt(view: str, result: Dict[str, Any]) -> str:
    if view == 'completeness':
        return (
            'Act as a professional hydrologist performing a data-quality audit.\n\n'
            'RESPONSE FORMAT (STRICT):\n'
            'Use markdown with exactly these sections:\n'
            '## Quality Summary\n2-3 sentences.\n'
            '## Coverage Assessment\nExactly 4 bullet points.\n'
            '## Reliability Implications\nExactly 4 bullet points.\n'
            '## Recommended Handling\nExactly 3 bullet points.\n\n'
            'RULES:\n'
            '- Cite the actual completeness statistics.\n'
            '- Replace underscores with spaces.\n'
            '- Keep the tone technical and concise.\n\n'
            f"Station: {result.get('station', '').replace('_', ' ')}\n"
            f"Feature: {result.get('feature', '').replace('_', ' ')}\n"
            f"Overall completeness: {result.get('overall_pct')}%\n"
            f"Missing months: {result.get('missing_months')}\n"
            f"Low-completeness months (<50%): {result.get('low_months')}\n"
            f"Total months evaluated: {result.get('total_months')}\n"
        )
    if view == 'imputation':
        top_rows = (result.get('rows') or [])[:8]
        top_text = '\n'.join(
            f"- {r['name']} · {r['feature'].replace('_', ' ')}: {r['imp_pct']:.1f}% imputed "
            f"({r['imputed']} of {r['observations']})"
            for r in top_rows
        ) or 'No station rows available.'
        return (
            'Act as a professional hydrologist performing a data-quality audit.\n\n'
            'RESPONSE FORMAT (STRICT):\n'
            'Use markdown with exactly these sections:\n'
            '## Quality Summary\n2-3 sentences.\n'
            '## Imputation Distribution\nExactly 4 bullet points.\n'
            '## Reliability Implications\nExactly 4 bullet points.\n'
            '## Recommended Handling\nExactly 3 bullet points.\n\n'
            'RULES:\n'
            '- Cite actual imputation statistics and affected stations.\n'
            '- Replace underscores with spaces.\n'
            '- Keep the tone technical and concise.\n\n'
            f"Dataset: {result.get('dataset')}\n"
            f"Feature filter: {(result.get('feature') or 'all features').replace('_', ' ')}\n"
            f"Overall imputation: {result.get('overall_imp_pct')}%\n"
            f"Total observations: {result.get('total_observations')}\n"
            f"Total imputed: {result.get('total_imputed')}\n"
            f"Stations with imputation: {result.get('stations_with_imputation')}\n"
            f"High-imputation stations (>=20%): {result.get('high_imputation_stations')}\n"
            f"Top affected rows:\n{top_text}\n"
        )
    if view == 'gaps':
        gap_rows = (result.get('gaps') or [])[:8]
        gap_text = '\n'.join(
            f"- {g['start']} to {g['end']}: {g['length']} {g['unit']} ({g['severity']})"
            for g in gap_rows
        ) or 'No gaps recorded.'
        return (
            'Act as a professional hydrologist performing a data-quality audit.\n\n'
            'RESPONSE FORMAT (STRICT):\n'
            'Use markdown with exactly these sections:\n'
            '## Quality Summary\n2-3 sentences.\n'
            '## Gap Structure\nExactly 4 bullet points.\n'
            '## Reliability Implications\nExactly 4 bullet points.\n'
            '## Recommended Handling\nExactly 3 bullet points.\n\n'
            'RULES:\n'
            '- Cite actual gap counts and durations.\n'
            '- Replace underscores with spaces.\n'
            '- Keep the tone technical and concise.\n\n'
            f"Station: {result.get('station', '').replace('_', ' ')}\n"
            f"Feature: {result.get('feature', '').replace('_', ' ')}\n"
            f"Missing share: {result.get('missing_pct')}%\n"
            f"Gap count: {result.get('gap_count')}\n"
            f"Major gaps: {result.get('major')}\n"
            f"Moderate gaps: {result.get('moderate')}\n"
            f"Minor gaps: {result.get('minor')}\n"
            f"Largest gaps:\n{gap_text}\n"
        )
    candidates = (result.get('candidates') or [])[:8]
    cand_text = '\n'.join(
        f"- {c['date']}: value {c['value']} {result.get('unit', '')}, |z|={c['z_score']}, "
        f"{'imputed' if c.get('is_imputed') else 'observed'}, flag={c.get('flag')}"
        for c in candidates
    ) or 'No anomaly candidates.'
    return (
        'Act as a professional hydrologist performing a data-quality audit.\n\n'
        'RESPONSE FORMAT (STRICT):\n'
        'Use markdown with exactly these sections:\n'
        '## Quality Summary\n2-3 sentences.\n'
        '## Candidate Structure\nExactly 4 bullet points.\n'
        '## Reliability Implications\nExactly 4 bullet points.\n'
        '## Recommended Handling\nExactly 3 bullet points.\n\n'
        'RULES:\n'
        '- Cite actual anomaly counts, z-scores, and flag status.\n'
        '- Replace underscores with spaces.\n'
        '- Keep the tone technical and concise.\n\n'
        f"Station: {result.get('station', '').replace('_', ' ')}\n"
        f"Feature: {result.get('feature', '').replace('_', ' ')}\n"
        f"Z threshold: {result.get('z_thresh')}\n"
        f"Mean: {result.get('mean')}\n"
        f"Std: {result.get('std')}\n"
        f"Total candidates: {result.get('total')}\n"
        f"Unflagged: {result.get('unflagged')}\n"
        f"Candidate examples:\n{cand_text}\n"
    )


def _fallback_quality_analysis(view: str, result: Dict[str, Any], note: str | None = None) -> str:
    parts = ['## Quality Summary']
    intro = [f"Quality interpretation for **{view.replace('_', ' ')}**."]
    if note:
        intro.append(note)
    parts.append(' '.join(intro))
    parts.append('## Key Findings')
    if view == 'completeness':
        parts.append(f"- **Coverage:** overall completeness is {result.get('overall_pct')}% across {result.get('total_months')} evaluated months.")
        parts.append(f"- **Missing periods:** {result.get('missing_months')} months are fully missing and {result.get('low_months')} months fall below 50% completeness.")
        parts.append('- **Reliability:** low temporal coverage weakens seasonal and trend interpretation.')
    elif view == 'imputation':
        parts.append(f"- **Overall imputation:** {result.get('overall_imp_pct')}% of {result.get('total_observations')} observations are imputed.")
        parts.append(f"- **Affected stations:** {result.get('stations_with_imputation')} stations contain imputed values; {result.get('high_imputation_stations')} have rates at or above 20%.")
        top = (result.get('rows') or [])[:1]
        if top:
            r = top[0]
            parts.append(f"- **Highest example:** {r['name']} · {r['feature'].replace('_', ' ')} has {r['imp_pct']:.1f}% imputation.")
    elif view == 'gaps':
        parts.append(f"- **Missing share:** {result.get('missing_pct')}% with {result.get('gap_count')} detected gaps.")
        parts.append(f"- **Gap severity:** major {result.get('major')}, moderate {result.get('moderate')}, minor {result.get('minor')}.")
        largest = (result.get('gaps') or [])[:1]
        if largest:
            g = largest[0]
            parts.append(f"- **Largest gap:** {g['start']} to {g['end']} spanning {g['length']} {g['unit']}.")
    else:
        parts.append(f"- **Candidate count:** {result.get('total')} anomaly candidates with {result.get('unflagged')} still unflagged.")
        parts.append(f"- **Thresholding:** candidates were detected at |z| >= {result.get('z_thresh')}.")
        first = (result.get('candidates') or [])[:1]
        if first:
            c = first[0]
            parts.append(f"- **Example:** {c['date']} has value {c['value']} and |z|={c['z_score']}.")
    parts.append('## Recommended Handling')
    parts.append('- **Use with caution:** quality issues should be considered before interpreting hydrological extremes, trends, or model results.')
    parts.append('- **Operational step:** review flagged periods and high-imputation segments before downstream analysis.')
    return markdown.markdown('\n'.join(parts))

MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

DARK_BG   = '#07111f'
GRID      = 'rgba(148,163,184,0.12)'
TEXT      = '#e5eefc'
SOFT      = '#9db0d1'
AXIS_FONT = dict(size=10, color=SOFT)


class QualityService:
    """Data-quality analytics: completeness heatmap, imputation summary,
    gap detection, anomaly-candidate review, and flag persistence."""

    def __init__(self, repository, data_dir: str | Path = 'data') -> None:
        self.repo = repository
        self.data_dir = Path(data_dir)
        self.flags_path = self.data_dir / 'quality_flags.json'

    # ── helpers ──────────────────────────────────────────────────────────────

    def _find_repo(self, station: str):
        if hasattr(self.repo, 'repos'):
            return next((r for r in self.repo.repos if station in r.station_index), None)
        if station in getattr(self.repo, 'station_index', {}):
            return self.repo
        return None

    def _load_series(self, repo, station: str, feature: str) -> pd.Series:
        fd = repo.station_index[station]['feature_details'][feature]
        req = SeriesRequest(station=station, feature=feature,
                            start_date=fd['start_date'], end_date=fd['end_date'])
        df = repo.get_feature_series(req)
        ts = df.set_index('Timestamp')['Value'].sort_index()
        ts.index = pd.to_datetime(ts.index)
        return ts

    def _load_imputed_mask(self, repo, station: str, feature: str) -> pd.Series | None:
        """Return a boolean Series of imputed flags aligned to the series index."""
        df = repo.get_station_dataframe(station)
        imp_col = f'{feature}_imputed'
        if imp_col not in df.columns:
            return None
        mask = df.set_index('Timestamp')[imp_col].fillna('No').str.lower().eq('yes')
        mask.index = pd.to_datetime(mask.index)
        return mask

    def _axis_style(self) -> dict:
        return dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=AXIS_FONT)

    # ── 1. Completeness heatmap ───────────────────────────────────────────────

    def completeness(self, station: str, feature: str) -> Dict[str, Any]:
        repo = self._find_repo(station)
        if repo is None:
            raise ValueError(f"Station '{station}' not found.")
        if feature not in repo.station_index[station].get('features', []):
            raise ValueError(f"'{feature}' not available for {station}.")

        ts = self._load_series(repo, station, feature)
        freq = repo.feature_frequency.get(feature, 'daily')
        unit = repo.feature_units.get(feature, '')

        if freq == 'monthly':
            # For monthly data: 1 obs expected per month
            monthly_obs = ts.resample('MS').count()
            pct_matrix = {}
            for dt, cnt in monthly_obs.items():
                yr, mo = dt.year, dt.month
                pct_matrix.setdefault(yr, {})[mo] = min(100.0, float(cnt) * 100)
        else:
            # Daily: expected = days in month
            monthly_obs = ts.resample('MS').count()
            pct_matrix = {}
            for dt, cnt in monthly_obs.items():
                yr, mo = dt.year, dt.month
                days_in_month = (pd.Timestamp(yr, mo, 1) + pd.offsets.MonthEnd(1)).day
                pct_matrix.setdefault(yr, {})[mo] = round(min(100.0, cnt / days_in_month * 100), 1)

        years = sorted(pct_matrix.keys())
        z = []
        text = []
        for yr in years:
            row_z, row_t = [], []
            for mo in range(1, 13):
                val = pct_matrix.get(yr, {}).get(mo, None)
                row_z.append(val if val is not None else -1)
                row_t.append(f'{val:.0f}%' if val is not None else 'No data')
            z.append(row_z)
            text.append(row_t)

        # Overall stats
        all_vals = [v for row in z for v in row if v >= 0]
        overall_pct = round(float(np.mean(all_vals)), 1) if all_vals else 0.0
        missing_months = sum(1 for row in z for v in row if v < 0)
        low_months = sum(1 for row in z for v in row if 0 <= v < 50)

        colorscale = [
            [0.0,  '#ef4444'],   # 0% — red
            [0.5,  '#f59e0b'],   # 50% — amber
            [0.75, '#84cc16'],   # 75% — lime
            [1.0,  '#22c55e'],   # 100% — green
        ]

        fig = go.Figure(go.Heatmap(
            z=z,
            x=MONTH_ABBR,
            y=[str(yr) for yr in years],
            text=text,
            texttemplate='%{text}',
            colorscale=colorscale,
            zmin=0, zmax=100,
            colorbar=dict(
                title=dict(text='% available', font=dict(size=10, color=SOFT)),
                tickfont=dict(size=10, color=SOFT),
                thickness=12, len=0.7,
            ),
            hovertemplate='%{y} %{x}<br>Completeness: %{text}<extra></extra>',
        ))
        fig.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            font=dict(family='Inter, sans-serif', color=TEXT, size=11),
            margin=dict(l=10, r=10, t=40, b=10),
            title=dict(text=f'Data Completeness · {feature} · {station.replace("_", " ")}',
                       font=dict(size=13, color=TEXT), x=0, xanchor='left', pad=dict(l=4)),
            xaxis=dict(**self._axis_style(), side='top'),
            yaxis=dict(**self._axis_style(), autorange='reversed'),
        )

        return {
            'station': station, 'feature': feature, 'unit': unit,
            'years': years, 'overall_pct': overall_pct,
            'missing_months': missing_months, 'low_months': low_months,
            'total_months': len(all_vals) + missing_months,
            'figure': plotly.io.to_json(fig),
        }

    # ── 2. Imputation summary ────────────────────────────────────────────────

    def imputation_summary(self, dataset: str, feature: str | None = None) -> Dict[str, Any]:
        # Collect per-station imputation stats across the right repo(s)
        repos = self.repo.repos if hasattr(self.repo, 'repos') else [self.repo]
        rows: List[Dict] = []
        for repo in repos:
            if repo.dataset != dataset:
                continue
            for stn, meta in repo.station_index.items():
                for feat, fd in meta.get('feature_details', {}).items():
                    if feature and feat != feature:
                        continue
                    obs = fd.get('observations', 0)
                    imp = fd.get('imputed_points', 0)
                    total = obs
                    imp_pct = round(imp / total * 100, 2) if total > 0 else 0.0
                    rows.append({
                        'station': stn,
                        'name': meta.get('name', stn).replace('_', ' '),
                        'feature': feat,
                        'observations': obs,
                        'imputed': imp,
                        'imp_pct': imp_pct,
                        'country': meta.get('country', ''),
                    })

        if not rows:
            raise ValueError(f"No data found for dataset='{dataset}'" +
                             (f" feature='{feature}'" if feature else '') + '.')

        df = pd.DataFrame(rows).sort_values('imp_pct', ascending=False)

        # Dataset-level stats
        total_obs = int(df['observations'].sum())
        total_imp = int(df['imputed'].sum())
        overall_imp_pct = round(total_imp / total_obs * 100, 2) if total_obs > 0 else 0.0
        stations_with_imputation = int((df['imp_pct'] > 0).sum())
        high_imputation = int((df['imp_pct'] >= 20).sum())

        # Top 20 most-imputed stations for bar chart
        top = df.head(20)
        bar_colors = ['#ef4444' if v >= 20 else '#f59e0b' if v >= 5 else '#38bdf8'
                      for v in top['imp_pct']]

        fig = go.Figure(go.Bar(
            x=top['imp_pct'].tolist(),
            y=top['name'].tolist(),
            orientation='h',
            marker_color=bar_colors,
            hovertemplate='%{y}<br>Imputation: %{x:.1f}%<extra></extra>',
        ))
        feat_label = feature or 'all features'
        fig.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            font=dict(family='Inter, sans-serif', color=TEXT, size=11),
            margin=dict(l=10, r=20, t=40, b=10),
            title=dict(text=f'Top stations by imputation rate · {dataset} · {feat_label}',
                       font=dict(size=13, color=TEXT), x=0, xanchor='left', pad=dict(l=4)),
            xaxis=dict(**self._axis_style(), title='Imputation %'),
            yaxis=dict(**self._axis_style(), autorange='reversed'),
            height=max(300, len(top) * 22 + 80),
        )

        return {
            'dataset': dataset,
            'feature': feature,
            'total_observations': total_obs,
            'total_imputed': total_imp,
            'overall_imp_pct': overall_imp_pct,
            'stations_with_imputation': stations_with_imputation,
            'high_imputation_stations': high_imputation,
            'total_stations': len(df['station'].unique()),
            'rows': df[['station', 'name', 'feature', 'observations', 'imputed', 'imp_pct', 'country']].to_dict('records'),
            'figure': plotly.io.to_json(fig),
        }

    # ── 3. Gap detection ────────────────────────────────────────────────────

    def gaps(self, station: str, feature: str) -> Dict[str, Any]:
        repo = self._find_repo(station)
        if repo is None:
            raise ValueError(f"Station '{station}' not found.")
        if feature not in repo.station_index[station].get('features', []):
            raise ValueError(f"'{feature}' not available for {station}.")

        ts = self._load_series(repo, station, feature)
        freq = repo.feature_frequency.get(feature, 'daily')
        unit = repo.feature_units.get(feature, '')
        pd_freq = 'MS' if freq == 'monthly' else 'D'
        unit_label = 'months' if freq == 'monthly' else 'days'

        # Reindex to full regular grid
        full_idx = pd.date_range(ts.index.min(), ts.index.max(), freq=pd_freq)
        ts_full = ts.reindex(full_idx)

        null_mask = ts_full.isna()
        gap_list: List[Dict] = []

        in_gap = False
        gap_start = None
        for dt, is_null in null_mask.items():
            if is_null and not in_gap:
                in_gap = True
                gap_start = dt
            elif not is_null and in_gap:
                in_gap = False
                length = int((dt - gap_start).days) if freq != 'monthly' else \
                    int((dt.year - gap_start.year) * 12 + dt.month - gap_start.month)
                severity = ('major' if length >= 30 else
                            'moderate' if length >= 7 else 'minor')
                gap_list.append({
                    'start': str(gap_start.date()),
                    'end': str((dt - pd.Timedelta(days=1 if freq != 'monthly' else 0)).date()),
                    'length': length,
                    'unit': unit_label,
                    'severity': severity,
                })

        # If series ends with a gap
        if in_gap:
            length = int((ts_full.index[-1] - gap_start).days) if freq != 'monthly' else \
                int((ts_full.index[-1].year - gap_start.year) * 12 +
                    ts_full.index[-1].month - gap_start.month) + 1
            severity = ('major' if length >= 30 else 'moderate' if length >= 7 else 'minor')
            gap_list.append({
                'start': str(gap_start.date()),
                'end': str(ts_full.index[-1].date()),
                'length': length,
                'unit': unit_label,
                'severity': severity,
            })

        gap_list.sort(key=lambda g: g['length'], reverse=True)

        total_missing = int(null_mask.sum())
        total_pts = len(full_idx)
        missing_pct = round(total_missing / total_pts * 100, 2) if total_pts else 0.0

        # Timeline figure: series with gap regions shaded
        display = ts_full.tail(min(len(ts_full), 2000))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=display.index, y=display.values,
            mode='lines', line=dict(color='#38bdf8', width=1.5),
            name=feature, connectgaps=False,
            hovertemplate='%{x|%Y-%m-%d}<br>%{y:.3f} ' + unit + '<extra></extra>',
        ))
        # Shade top-5 largest gaps
        for g in gap_list[:5]:
            color = ('#ef444430' if g['severity'] == 'major' else
                     '#f59e0b25' if g['severity'] == 'moderate' else '#64748b20')
            fig.add_vrect(x0=g['start'], x1=g['end'],
                          fillcolor=color, line_width=0, layer='below')

        fig.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            font=dict(family='Inter, sans-serif', color=TEXT, size=11),
            margin=dict(l=10, r=10, t=40, b=10),
            title=dict(text=f'Gap Detection · {feature} · {station.replace("_", " ")}',
                       font=dict(size=13, color=TEXT), x=0, xanchor='left', pad=dict(l=4)),
            xaxis=dict(**self._axis_style()),
            yaxis=dict(**self._axis_style(), title=unit or feature),
            showlegend=False,
        )

        return {
            'station': station, 'feature': feature,
            'total_points': total_pts, 'missing_points': total_missing,
            'missing_pct': missing_pct,
            'gap_count': len(gap_list),
            'major': sum(1 for g in gap_list if g['severity'] == 'major'),
            'moderate': sum(1 for g in gap_list if g['severity'] == 'moderate'),
            'minor': sum(1 for g in gap_list if g['severity'] == 'minor'),
            'gaps': gap_list[:50],   # cap at 50 for payload size
            'figure': plotly.io.to_json(fig),
        }

    # ── 4. Anomaly candidates ────────────────────────────────────────────────

    def anomaly_candidates(self, station: str, feature: str, z_thresh: float = 3.0) -> Dict[str, Any]:
        repo = self._find_repo(station)
        if repo is None:
            raise ValueError(f"Station '{station}' not found.")
        if feature not in repo.station_index[station].get('features', []):
            raise ValueError(f"'{feature}' not available for {station}.")

        ts = self._load_series(repo, station, feature)
        unit = repo.feature_units.get(feature, '')
        imp_mask = self._load_imputed_mask(repo, station, feature)

        mean = float(ts.mean())
        std  = float(ts.std())
        if std == 0:
            return {'station': station, 'feature': feature, 'candidates': [], 'total': 0}

        z_scores = ((ts - mean) / std).abs()
        anomaly_idx = z_scores[z_scores >= z_thresh].index

        candidates = []
        existing_flags = self._load_flags().get(station, {}).get(feature, {})

        for dt in anomaly_idx:
            val = float(ts.loc[dt])
            z   = float(z_scores.loc[dt])
            # Context: 3 points before and after
            pos = ts.index.get_loc(dt)
            ctx_before = ts.iloc[max(0, pos - 3): pos].tolist()
            ctx_after  = ts.iloc[pos + 1: pos + 4].tolist()
            is_imputed = bool(imp_mask.loc[dt]) if (imp_mask is not None and dt in imp_mask.index) else False
            date_str = str(dt.date())
            candidates.append({
                'date': date_str,
                'value': round(val, 4),
                'z_score': round(z, 2),
                'above_mean': val > mean,
                'is_imputed': is_imputed,
                'context_before': [round(v, 4) for v in ctx_before],
                'context_after':  [round(v, 4) for v in ctx_after],
                'flag': existing_flags.get(date_str, 'none'),
            })

        # Sort by z-score descending
        candidates.sort(key=lambda c: c['z_score'], reverse=True)

        return {
            'station': station, 'feature': feature, 'unit': unit,
            'z_thresh': z_thresh, 'mean': round(mean, 4), 'std': round(std, 4),
            'total': len(candidates),
            'unflagged': sum(1 for c in candidates if c['flag'] == 'none'),
            'candidates': candidates[:100],   # cap payload at 100
        }

    # ── 5. Flag persistence ─────────────────────────────────────────────────

    def _load_flags(self) -> Dict:
        if self.flags_path.exists():
            try:
                return json.loads(self.flags_path.read_text())
            except Exception:
                pass
        return {}

    def save_flag(self, station: str, feature: str, date_str: str, flag: str) -> Dict[str, Any]:
        """flag must be one of: real, sensor_error, uncertain, none (clears)"""
        allowed = {'real', 'sensor_error', 'uncertain', 'none'}
        if flag not in allowed:
            raise ValueError(f"flag must be one of {allowed}")
        flags = self._load_flags()
        flags.setdefault(station, {}).setdefault(feature, {})
        if flag == 'none':
            flags[station][feature].pop(date_str, None)
        else:
            flags[station][feature][date_str] = flag
        self.flags_path.write_text(json.dumps(flags, indent=2))
        return {'station': station, 'feature': feature, 'date': date_str, 'flag': flag}

    def flag_summary(self) -> Dict[str, Any]:
        flags = self._load_flags()
        total = 0
        breakdown: Dict[str, int] = {'real': 0, 'sensor_error': 0, 'uncertain': 0}
        details = []
        for stn, feats in flags.items():
            for feat, dates in feats.items():
                for d, f in dates.items():
                    total += 1
                    breakdown[f] = breakdown.get(f, 0) + 1
                    details.append({'station': stn, 'feature': feat, 'date': d, 'flag': f})
        details.sort(key=lambda x: x['date'], reverse=True)
        return {'total': total, 'breakdown': breakdown, 'recent': details[:20]}
