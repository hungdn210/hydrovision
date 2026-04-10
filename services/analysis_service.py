from __future__ import annotations

import json
import os
import re
import time
import hashlib
import markdown
from typing import Any, Dict, List
from pathlib import Path

import numpy as np
import pandas as pd

from .chart_service import ChartService
from .data_loader import DataRepository, SeriesRequest

MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Models tried in order; if one fails the next is used automatically.
_GEMINI_MODELS = [
    'gemini-2.0-flash',               # primary — high quota
    'gemini-2.0-flash-lite',          # fallback — lighter
    'gemini-1.5-flash',               # fallback — stable
    'gemini-1.5-flash-8b',            # last resort — smallest
]

_CACHE_PATH = (
    Path(os.environ.get('ANALYSIS_AI_CACHE_PATH', ''))
    if os.environ.get('ANALYSIS_AI_CACHE_PATH')
    else Path(__file__).resolve().parent.parent / 'data' / 'ai_analysis_cache.json'
)
_AI_RESPONSE_CACHE: Dict[str, str] = {}


def _cache_key(api_key: str, prompt: str) -> str:
    return hashlib.sha256(f'{api_key}:{prompt}'.encode('utf-8')).hexdigest()


def _load_ai_cache() -> Dict[str, str]:
    try:
        if _CACHE_PATH.exists():
            data = json.loads(_CACHE_PATH.read_text(encoding='utf-8'))
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def _save_ai_cache(cache: Dict[str, str]) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to temp file then rename to prevent corruption
        tmp = _CACHE_PATH.with_suffix('.tmp')
        tmp.write_text(json.dumps(cache, ensure_ascii=True, indent=2), encoding='utf-8')
        tmp.replace(_CACHE_PATH)
    except Exception:
        pass


_AI_RESPONSE_CACHE = _load_ai_cache()


def _normalise_md(text: str) -> str:
    """Ensure a blank line precedes every markdown list item.

    Python's markdown library requires a blank line between a paragraph (or
    heading) and a ``- ``/``* ``/``1. `` list.  Gemini often omits that blank
    line, which causes all bullet items to be folded into the preceding
    paragraph as plain text.  This one-liner fix is idempotent: if the blank
    line is already present the substitution matches ``\\n\\n- `` and the
    negative-lookbehind prevents adding a third newline.
    """
    return re.sub(r'(?<!\n)\n([-*]|\d+\.) ', r'\n\n\1 ', text)


def _gemini_generate(api_key: str, prompt: str) -> str:
    """Call Gemini with automatic model fallback on any per-model error."""
    from google import genai
    cache_key = _cache_key(api_key, prompt)
    if cache_key in _AI_RESPONSE_CACHE:
        return _AI_RESPONSE_CACHE[cache_key]
    client = genai.Client(api_key=api_key)
    last_exc: Exception | None = None
    for model in _GEMINI_MODELS:
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            text = _normalise_md(response.text.strip())
            _AI_RESPONSE_CACHE[cache_key] = text
            _save_ai_cache(_AI_RESPONSE_CACHE)
            return text
        except Exception as e:
            last_exc = e
            msg = str(e)
            # Back off slightly on quota errors; still try next model regardless
            if '429' in msg or 'RESOURCE_EXHAUSTED' in msg:
                time.sleep(0.6)
            continue  # always try the next model
    raise last_exc  # all models exhausted


class AnalysisService:
    def __init__(self, repository: DataRepository, chart_service: ChartService) -> None:
        self.repository = repository
        self.chart_service = chart_service

    def analyse_free(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Free-form analysis: auto-picks the best graph type from the series list."""
        series = payload.get('series', [])
        if not series:
            raise ValueError('At least one series is required.')
        stations = {s.get('station') for s in series}
        features = {s.get('feature') for s in series}
        if len(series) == 1:
            graph_type = 'Single Category, Single Station Timeline'
        elif len(stations) == 1 and len(features) > 1:
            graph_type = 'Multiple Categories, Single Station Timeline'
        elif len(features) == 1 and len(stations) > 1:
            graph_type = 'Single Category Across Multiple Stations Comparison'
        else:
            graph_type = 'Multiple Categories Across Multiple Stations Comparison'
        return self.analyse({'graph_type': graph_type, 'series': series})

    def analyse_free_multi(self, payload: Dict[str, Any], include_analysis: bool = False) -> Dict[str, Any]:
        """Free-form analysis producing 3 complementary graphs with per-graph analysis."""
        series = payload.get('series', [])
        if not series:
            raise ValueError('At least one series is required.')

        graph_configs = self._pick_three_graphs(series)

        graphs_data = []
        all_unique_frames = []
        seen_frame_keys = set()
        for config in graph_configs:
            try:
                chart_payload = self.chart_service.generate_chart({
                    'graph_type': config['graph_type'],
                    'series': config['series'],
                })
                requests = [SeriesRequest(**item) for item in chart_payload['series']]
                frames = [self.repository.get_feature_series(r) for r in requests]
                
                findings = self._build_findings(frames)
                comparisons = self._build_comparisons(frames)
                climatology = self._build_climatology_anomalies(frames)

                graphs_data.append({
                    'chart_payload': chart_payload,
                    'frames': frames,
                    'label': config['label'],
                    'focus': config['focus'],
                    'findings': findings,
                    'comparisons': comparisons,
                    'climatology': climatology,
                })

                for f in frames:
                    key = (f['Station'].iloc[0], f['Feature'].iloc[0])
                    if key not in seen_frame_keys:
                        seen_frame_keys.add(key)
                        all_unique_frames.append(f)
            except Exception:
                continue

        if not graphs_data:
            raise ValueError('Could not generate any charts for the selected series.')

        benchmark = self._build_dataset_benchmark(all_unique_frames)
        benchmark_analysis = self._compose_benchmark_summary(benchmark)

        summaries = self._compose_multi_graph_summaries(graphs_data, benchmark)

        results = []
        for i, g in enumerate(graphs_data):
            results.append({
                **g['chart_payload'],
                'graph_label': g['label'],
                'analysis': {
                    'summary': summaries[i] if i < len(summaries) else '',
                    'findings': g['findings'],
                    'comparisons': g['comparisons'],
                },
            })

        out: Dict[str, Any] = {'graphs': results, 'benchmark': benchmark, 'benchmark_analysis': benchmark_analysis}

        if include_analysis:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                try:
                    series_desc = ', '.join(
                        f"{s.get('station','?')} / {s.get('feature','?')}" for s in series[:3]
                    )
                    prompt = f"""Analyze these hydrological time series and provide 3 concise bullet-point insights:

Series: {series_desc}
Benchmark summary: {benchmark_analysis[:500] if benchmark_analysis else 'N/A'}

Focus on: key trends, anomalies, and practical water management implications. Use **bold** for key terms."""
                    out['analysis'] = _gemini_generate(api_key, prompt)
                except Exception:
                    pass

        return out

    def _pick_three_graphs(self, series: List[Dict]) -> List[Dict]:
        """Pick 3 complementary graph types based on series composition."""
        unique_stations = list(dict.fromkeys(s.get('station') for s in series))
        unique_features = list(dict.fromkeys(s.get('feature') for s in series))

        if len(series) == 1:
            return [
                {'graph_type': 'Single Category, Single Station Timeline', 'series': series,
                 'label': 'Timeline Overview',
                 'focus': 'long-term trend, seasonality, and temporal patterns over the full period'},
                {'graph_type': 'Year-over-Year Comparison', 'series': series,
                 'label': 'Year-over-Year Comparison',
                 'focus': 'inter-annual variability, anomalous years, and recurring seasonal cycles'},
                {'graph_type': 'Monthly Distribution Box Plot', 'series': series,
                 'label': 'Monthly Distribution',
                 'focus': 'seasonal distribution, spread, median shifts, and outliers across calendar months'},
            ]

        if len(unique_stations) == 1 and len(unique_features) > 1:
            first_series = [series[0]]
            if len(unique_features) == 2:
                graph2 = {'graph_type': 'Correlation Scatter Plot', 'series': series,
                          'label': 'Feature Correlation',
                          'focus': 'correlation strength, linearity, and relationship between the two variables'}
            else:
                graph2 = {'graph_type': 'Monthly Distribution Box Plot', 'series': first_series,
                          'label': 'Monthly Distribution (Primary)',
                          'focus': 'seasonal distribution, spread, and outlier behaviour of the primary variable'}
            return [
                {'graph_type': 'Multiple Categories, Single Station Timeline', 'series': series,
                 'label': 'Multi-Variable Timeline',
                 'focus': 'comparative temporal patterns, co-variation, and divergence across all variables'},
                graph2,
                {'graph_type': 'Year-over-Year Comparison', 'series': first_series,
                 'label': 'Year-over-Year (Primary Variable)',
                 'focus': 'inter-annual variability and anomalous years for the primary variable'},
            ]

        if len(unique_features) == 1 and len(unique_stations) > 1:
            return [
                {'graph_type': 'Single Category Across Multiple Stations Comparison', 'series': series,
                 'label': 'Station Comparison',
                 'focus': 'relative magnitude, trend direction, and divergence across stations'},
                {'graph_type': 'Multi-Station Temporal Heatmap', 'series': series,
                 'label': 'Spatial-Temporal Heatmap',
                 'focus': 'spatial distribution of values over time and regional patterns'},
                {'graph_type': 'Annual Monthly Totals Overview', 'series': [series[0]],
                 'label': 'Monthly Seasonal Profile (Primary Station)',
                 'focus': 'year-by-year seasonal patterns and how monthly climatology has shifted'},
            ]

        # Multiple features, multiple stations
        first_station_series = [s for s in series if s.get('station') == unique_stations[0]]
        first_feature_series = [s for s in series if s.get('feature') == unique_features[0]]
        multi_station = (
            first_feature_series if len({s.get('station') for s in first_feature_series}) > 1 else series[:2]
        )
        multi_var = first_station_series if len(first_station_series) > 1 else [series[0]]
        third_type = 'Multiple Categories, Single Station Timeline' if len(multi_var) > 1 else 'Year-over-Year Comparison'
        third_label = (f'Primary Station Detail — {unique_stations[0].replace("_", " ")}'
                       if len(multi_var) > 1 else 'Year-over-Year')
        third_focus = ('variable interactions and co-evolution at the primary station'
                       if len(multi_var) > 1 else 'inter-annual variability and anomalous years')
        return [
            {'graph_type': 'Multiple Categories Across Multiple Stations Comparison', 'series': series,
             'label': 'Multi-Variable Station Overview',
             'focus': 'overall patterns and relative behaviour across all stations and variables'},
            {'graph_type': 'Single Category Across Multiple Stations Comparison', 'series': multi_station,
             'label': f'Station Comparison — {unique_features[0].replace("_", " ")}',
             'focus': 'station-level differences, relative ranking, and trend divergence for the primary variable'},
            {'graph_type': third_type, 'series': multi_var,
             'label': third_label, 'focus': third_focus},
        ]

    def _compose_multi_graph_summaries(
        self,
        graphs_data: List[Dict],
        benchmark: List[Dict],
    ) -> List[str]:
        """One Gemini call producing N graph-specific analysis sections. Returns list of HTML strings."""
        api_key = os.environ.get("GEMINI_API_KEY")
        fallback = self._fallback_multi_graph_summaries(graphs_data, benchmark)
        if not api_key or api_key == "your_google_gemini_api_key_here" or not api_key.strip():
            return fallback

        graph_blocks = []
        for i, g in enumerate(graphs_data):
            findings_str = self._format_findings_for_prompt(g['findings'])
            clim_str = chr(10).join(g['climatology']) if g['climatology'] else 'Insufficient years for climatological comparison.'
            comp_str = chr(10).join(g['comparisons']) if g['comparisons'] else 'Single series — no cross-series comparison available.'
            block = (
                f"Graph {i + 1}: {g['label']} — focus: {g['focus']}\n"
                f"Statistical Findings:\n{findings_str}\n"
                f"Climatological Anomalies:\n{clim_str}\n"
                f"Cross-Series Correlation Notes:\n{comp_str}"
            )
            graph_blocks.append(block)

        graph_descriptions = '\n\n---\n\n'.join(graph_blocks)
        n = len(graphs_data)

        prompt = (
            "Act as a professional hydrologist for the Mekong River Commission. "
            "You have statistical findings for one or more hydrological series "
            "and must produce separate analyses for each of the following visualisations.\n\n"
            f"Visualisations Data:\n{graph_descriptions}\n\n"
            f"Global Dataset Benchmark (across all selected stations):\n"
            f"{self._format_benchmark_for_prompt(benchmark) if benchmark else 'No benchmark data available.'}\n\n"
            f"Produce exactly {n} analysis sections, one per graph. "
            "Each section must begin with the EXACT marker '## SECTION_N' (N = 1, 2, 3…) "
            "followed by 5 bullet points, each starting with a bold **Heading:** that specifically "
            "addresses what THAT graph type reveals.\n\n"
            "Tailor each section tightly to its focus area. "
            "A Timeline section should discuss trends and seasonality; "
            "a Year-over-Year section MUST reference the climatological anomaly table to name specific years "
            "and state their percentage deviation from the long-run mean (e.g. '2015 was 34% below climatological normal'); "
            "a Box Plot section should focus on spread, median shifts, and outlier months.\n\n"
            "RULES:\n"
            "- Replace underscores with spaces in all station and feature names.\n"
            "- Always cite specific numbers from the statistics — do not be vague.\n"
            "- Use professional hydrological language throughout.\n"
            "- Do NOT include any introduction, conclusion, or sign-off.\n"
            "- Start your response immediately with '## SECTION_1'."
        )

        try:
            raw = _gemini_generate(api_key, prompt)
            parts = re.split(r'##\s+SECTION_\d+\s*', raw)
            sections = [p.strip() for p in parts if p.strip()]
            while len(sections) < n:
                sections.append('')
            return [markdown.markdown(s) if s else '' for s in sections[:n]]
        except Exception:
            return fallback

    def analyse(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        chart_payload = self.chart_service.generate_chart(payload)
        requests = [SeriesRequest(**item) for item in chart_payload['series']]
        series_frames = [self.repository.get_feature_series(request) for request in requests]

        findings = self._build_findings(series_frames)
        comparisons = self._build_comparisons(series_frames)
        climatology = self._build_climatology_anomalies(series_frames)
        benchmark = self._build_dataset_benchmark(series_frames)
        summary = self._compose_summary(series_frames, findings, comparisons, climatology, benchmark)
        benchmark_analysis = self._compose_benchmark_summary(benchmark)

        # If we have a successful summary, suppress the raw findings cards to reduce clutter
        if '<p>' in summary or '<ul>' in summary or '<li>' in summary:
            findings = []

        return {
            **chart_payload,
            'analysis': {
                'summary': summary,
                'findings': findings,
                'comparisons': comparisons,
                'benchmark': benchmark,
                'benchmark_analysis': benchmark_analysis,
            },
        }

    def _compose_summary(self, frames: List[pd.DataFrame], findings: List[Dict], comparisons: List[str], climatology: List[str] = None, benchmark: List[Dict] = None) -> str:
        if len(frames) == 1:
            frame = frames[0]
            station = frame['Station'].iloc[0].replace('_', ' ')
            feature = frame['Feature'].iloc[0].replace('_', ' ')
            start = frame['Timestamp'].min().strftime('%Y-%m-%d')
            end = frame['Timestamp'].max().strftime('%Y-%m-%d')
            mean_val = frame['Value'].mean()
            max_val = frame['Value'].max()
            min_val = frame['Value'].min()
            base_summary = (
                f'{feature} at {station} was analysed from {start} to {end}. '
                f'The average value was {mean_val:.2f}, with a minimum of {min_val:.2f} and a maximum of {max_val:.2f}. '
                f'The insight panel below highlights trend direction, seasonality clues, anomaly counts, and imputation share.'
            )
        else:
            stations = sorted({frame['Station'].iloc[0] for frame in frames})
            features = sorted({frame['Feature'].iloc[0] for frame in frames})
            base_summary = (
                f'This analysis combines {len(frames)} selected series across {len(stations)} station(s) and {len(features)} feature(s). '
                f'The results below summarise data coverage, directional trends, anomaly behaviour, and where overlapping series appear related.'
            )

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key or api_key == "your_google_gemini_api_key_here" or not api_key.strip():
            return self._fallback_summary(frames, findings, comparisons, climatology, benchmark)

        try:
            prompt = (
                "Act as a professional hydrologist for the Mekong River Commission. "
                "You have been given a rich set of statistical findings for one or more hydrological series. "
                "Your task is to produce a thorough, data-driven analysis that a water resource engineer would find genuinely useful.\n\n"
                "Structure your response exactly like this:\n"
                "1. **Executive Summary** — 3-4 sentences covering the big picture: overall behaviour, dominant characteristic (high variability vs stable), and standout period.\n"
                "2. **Detailed Insights** — exactly 6 bullet points, each starting with a bold **Heading:**, covering:\n"
                "   - Trend: Is the long-run direction significant? Reference the slope direction and CV to judge if drift matters.\n"
                "   - Variability & Extremes: Use CV, IQR, Q10/Q90 to characterise how volatile the signal is. Cite the exact record high/low dates and values.\n"
                "   - Wet/Dry Season Asymmetry: Use the monthly mean breakdown to describe how the wet-season peak compares to the dry-season trough. Quantify the ratio if possible.\n"
                "   - Drought & Flood Risk: Use the % of time below Q10 and above Q90, plus longest consecutive dry/wet streaks, to assess frequency and duration of extremes.\n"
                "   - Climatological Anomalies: Use the climatology table to identify which years were furthest from normal. State the year, its percentage deviation, and whether it signals drought or flood conditions.\n"
                "   - Data Integrity: Comment on imputation share and anomaly count. Flag if either is high enough to affect confidence in the findings.\n\n"
                "IMPORTANT RULES:\n"
                "- Always replace underscores with spaces in station and feature names (e.g. 'Ban Chot' not 'Ban_Chot').\n"
                "- Always cite specific numbers from the statistics — do not be vague.\n"
                "- Use professional hydrological language throughout.\n"
                "- Do NOT include any introduction, greeting, or sign-off.\n\n"
                f"Statistical Findings:\n{self._format_findings_for_prompt(findings)}\n\n"
                f"Climatological Anomalies (each year vs long-run mean):\n"
                f"{chr(10).join(climatology) if climatology else 'Insufficient years for climatological comparison.'}\n\n"
                f"Dataset Benchmark (station vs all stations in same dataset):\n"
                f"{self._format_benchmark_for_prompt(benchmark) if benchmark else 'No benchmark data available.'}\n\n"
                f"Series Correlation Notes:\n{chr(10).join(comparisons) if comparisons else 'Single series — no cross-series comparison available.'}"
            )
            html_content = markdown.markdown(_gemini_generate(api_key, prompt))
            return html_content
        except Exception:
            return self._fallback_summary(frames, findings, comparisons, climatology, benchmark)

    def _format_findings_for_prompt(self, findings: List[Dict]) -> str:
        """Format the rich findings dict into a readable block for the Gemini prompt."""
        lines = []
        for f in findings:
            lines.append(f"=== {f['title'].replace('_', ' ')} ===")
            for key, val in f.items():
                if key == 'title':
                    continue
                lines.append(f"  {key}: {val}")
        return '\n'.join(lines)

    def _build_findings(self, frames: List[pd.DataFrame]) -> List[Dict]:
        cards = []
        for frame in frames:
            station = frame['Station'].iloc[0]
            feature = frame['Feature'].iloc[0]
            unit = frame['Unit'].iloc[0]
            values = frame['Value'].dropna()
            observations = len(frame)

            # --- Basic descriptive stats ---
            mean_val = values.mean()
            median_val = values.median()
            std_val = values.std()
            min_row = frame.loc[frame['Value'].idxmin()]
            max_row = frame.loc[frame['Value'].idxmax()]

            # --- Percentiles & spread ---
            q10 = values.quantile(0.10)
            q25 = values.quantile(0.25)
            q75 = values.quantile(0.75)
            q90 = values.quantile(0.90)
            iqr = q75 - q25
            cv = (std_val / mean_val * 100) if mean_val != 0 else float('nan')

            # --- Exceedance frequencies ---
            pct_above_q90 = (values > q90).mean() * 100
            pct_below_q10 = (values < q10).mean() * 100

            # --- Consecutive streaks ---
            dry_streak = self._longest_streak(values, below=mean_val)
            wet_streak = self._longest_streak(values, above=mean_val)

            # --- Annual means ---
            frame_copy = frame.copy()
            frame_copy['Year'] = frame_copy['Timestamp'].dt.year
            annual_means = (
                frame_copy.groupby('Year')['Value'].mean()
                .round(2)
                .to_dict()
            )
            annual_means_str = ', '.join(f"{yr}: {v:.2f}" for yr, v in sorted(annual_means.items()))

            # --- Monthly breakdown ---
            frame_copy['Month'] = frame_copy['Timestamp'].dt.month
            monthly_means = frame_copy.groupby('Month')['Value'].mean()
            monthly_str = ', '.join(
                f"{MONTH_NAMES[m - 1]}: {monthly_means[m]:.2f}"
                for m in range(1, 13) if m in monthly_means.index
            )

            # --- Imputation & anomalies ---
            imputed_share = frame['Imputed'].astype(str).str.lower().eq('yes').mean() * 100
            anomaly_count = self._anomaly_count(frame)

            # --- Trend ---
            trend_text = self._trend_text(frame)

            cards.append({
                'title': f'{station} · {feature}',
                'unit': unit,
                'observations': observations,
                'period': f"{frame['Timestamp'].min().strftime('%Y-%m-%d')} to {frame['Timestamp'].max().strftime('%Y-%m-%d')}",
                'mean': f'{mean_val:.3f} {unit}',
                'median': f'{median_val:.3f} {unit}',
                'std_dev': f'{std_val:.3f} {unit}',
                'cv_percent': f'{cv:.1f}%' if not np.isnan(cv) else 'n/a',
                'record_low': f'{min_row["Value"]:.3f} {unit} on {min_row["Timestamp"].strftime("%Y-%m-%d")}',
                'record_high': f'{max_row["Value"]:.3f} {unit} on {max_row["Timestamp"].strftime("%Y-%m-%d")}',
                'Q10': f'{q10:.3f} {unit}',
                'Q25': f'{q25:.3f} {unit}',
                'Q75': f'{q75:.3f} {unit}',
                'Q90': f'{q90:.3f} {unit}',
                'IQR': f'{iqr:.3f} {unit}',
                'pct_time_above_Q90': f'{pct_above_q90:.1f}%',
                'pct_time_below_Q10': f'{pct_below_q10:.1f}%',
                'longest_consecutive_below_mean_days': dry_streak,
                'longest_consecutive_above_mean_days': wet_streak,
                'trend': trend_text,
                'monthly_means': monthly_str,
                'annual_means': annual_means_str,
                'imputed_pct': f'{imputed_share:.1f}%',
                'anomaly_count': anomaly_count,
            })
        return cards

    def _build_climatology_anomalies(self, frames: List[pd.DataFrame]) -> List[str]:
        """For each series, compute each year's deviation from its long-run climatological mean."""
        notes: List[str] = []
        for frame in frames:
            station = frame['Station'].iloc[0].replace('_', ' ')
            feature = frame['Feature'].iloc[0].replace('_', ' ')

            fc = frame.copy()
            fc['Year'] = fc['Timestamp'].dt.year
            annual = fc.groupby('Year')['Value'].mean()
            if len(annual) < 3:
                continue

            clim_mean = annual.mean()
            if clim_mean == 0 or pd.isna(clim_mean):
                continue

            anomalies = ((annual - clim_mean) / abs(clim_mean) * 100).round(1)

            year_lines = []
            for yr in sorted(annual.index):
                pct = anomalies[yr]
                direction = 'above' if pct >= 0 else 'below'
                year_lines.append(f"    {yr}: {annual[yr]:.3f} ({abs(pct):.1f}% {direction} normal)")

            top_drought = anomalies[anomalies < 0].sort_values().head(1)
            top_surplus = anomalies[anomalies > 0].sort_values(ascending=False).head(1)
            highlights = []
            if not top_drought.empty:
                yr = top_drought.index[0]
                highlights.append(f"  Strongest deficit: {yr} ({abs(top_drought.iloc[0]):.1f}% below normal)")
            if not top_surplus.empty:
                yr = top_surplus.index[0]
                highlights.append(f"  Strongest surplus: {yr} ({top_surplus.iloc[0]:.1f}% above normal)")

            block = (
                f"Climatology — {station} · {feature} "
                f"(long-run mean: {clim_mean:.3f}, {len(annual)} years):\n"
                + '\n'.join(year_lines)
                + ('\n' + '\n'.join(highlights) if highlights else '')
            )
            notes.append(block)
        return notes

    def _build_dataset_benchmark(self, frames: List[pd.DataFrame]) -> List[Dict]:
        """Compare each selected station's mean to the dataset-wide distribution for that feature."""
        results = []
        for frame in frames:
            station_id = frame['Station'].iloc[0]
            feature = frame['Feature'].iloc[0]
            unit = frame['Unit'].iloc[0]

            # Find the repo this station belongs to (works with MultiDataRepository)
            repo = getattr(self.repository, '_station_to_repo', {}).get(station_id)
            if repo is None:
                continue

            # Collect per-station means for this feature from the same dataset
            station_means = []
            for _sid, meta in repo.station_index.items():
                fd = meta.get('feature_details', {}).get(feature)
                if fd is not None and 'mean' in fd and not pd.isna(fd['mean']):
                    station_means.append(fd['mean'])

            if len(station_means) < 2:
                continue

            arr = np.array(station_means)
            dataset_mean = float(np.mean(arr))
            dataset_std = float(np.std(arr))
            dataset_min = float(np.min(arr))
            dataset_q25 = float(np.percentile(arr, 25))
            dataset_median = float(np.median(arr))
            dataset_q75 = float(np.percentile(arr, 75))
            dataset_max = float(np.max(arr))

            station_mean = float(frame['Value'].mean())
            z_score = (station_mean - dataset_mean) / dataset_std if dataset_std > 0 else 0.0
            pct_rank = float(np.sum(arr <= station_mean) / len(arr) * 100)
            pct_diff = (station_mean - dataset_mean) / abs(dataset_mean) * 100 if dataset_mean != 0 else 0.0

            results.append({
                'station': station_id,
                'station_label': station_id.replace('_', ' '),
                'feature': feature,
                'feature_label': feature.replace('_', ' '),
                'dataset': repo.dataset,
                'n_stations': len(station_means),
                'station_mean': round(station_mean, 3),
                'dataset_mean': round(dataset_mean, 3),
                'dataset_std': round(dataset_std, 3),
                'dataset_min': round(dataset_min, 3),
                'dataset_q25': round(dataset_q25, 3),
                'dataset_median': round(dataset_median, 3),
                'dataset_q75': round(dataset_q75, 3),
                'dataset_max': round(dataset_max, 3),
                'z_score': round(z_score, 2),
                'percentile_rank': round(pct_rank, 0),
                'pct_diff': round(pct_diff, 1),
                'unit': unit,
            })
        return results

    def _compose_benchmark_summary(self, benchmark: List[Dict]) -> str:
        """Short paragraph interpreting the dataset benchmark comparisons."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key or api_key == "your_google_gemini_api_key_here" or not api_key.strip() or not benchmark:
            return self._fallback_benchmark_summary(benchmark)
        try:
            data_text = self._format_benchmark_for_prompt(benchmark)
            station_list = ', '.join(b['station_label'].replace('_', ' ') for b in benchmark)
            n = len(benchmark)
            prompt = (
                "Act as a professional hydrologist. "
                "You have a dataset benchmark table comparing selected stations to all stations in their dataset.\n\n"
                f"{data_text}\n\n"
                f"Write exactly {n} bullet point(s), one per station ({station_list}). "
                "Each bullet must start with '- **Station Name:**' (bold, using the exact station name), "
                "followed by one sentence that: states whether the station is above or below the dataset average, "
                "cites the percentile rank and z-score, and explains what this suggests about its hydrological character "
                "(e.g. high-yield alpine catchment, low-flow lowland stream, large mainstem river). "
                "Use professional hydrological language. No introduction, no sign-off, bullets only."
            )
            return markdown.markdown(_gemini_generate(api_key, prompt))
        except Exception:
            return self._fallback_benchmark_summary(benchmark)

    def _format_benchmark_for_prompt(self, benchmarks: List[Dict]) -> str:
        lines = []
        for b in benchmarks:
            direction = 'above' if b['pct_diff'] >= 0 else 'below'
            lines.append(
                f"{b['station_label']} · {b['feature_label']} vs {b['dataset']} dataset "
                f"({b['n_stations']} stations):\n"
                f"  Station mean:    {b['station_mean']} {b['unit']}\n"
                f"  Dataset mean:    {b['dataset_mean']} {b['unit']} "
                f"({abs(b['pct_diff']):.1f}% {direction} dataset average)\n"
                f"  Dataset range:   [{b['dataset_min']}, {b['dataset_max']}] {b['unit']}\n"
                f"  Z-score:         {b['z_score']:+.2f} std devs from dataset mean\n"
                f"  Percentile rank: {b['percentile_rank']:.0f}th among dataset stations"
            )
        return '\n\n'.join(lines)

    def _build_comparisons(self, frames: List[pd.DataFrame]) -> List[str]:
        notes: List[str] = []
        if len(frames) < 2:
            return notes
        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                left = frames[i][['Timestamp', 'Value']].rename(columns={'Value': 'left'})
                right = frames[j][['Timestamp', 'Value']].rename(columns={'Value': 'right'})
                merged = left.merge(right, on='Timestamp', how='inner')
                if len(merged) < 12:
                    continue
                corr = merged['left'].corr(merged['right'])
                if pd.isna(corr):
                    continue
                l_label = f"{frames[i]['Station'].iloc[0]} · {frames[i]['Feature'].iloc[0]}"
                r_label = f"{frames[j]['Station'].iloc[0]} · {frames[j]['Feature'].iloc[0]}"
                if corr >= 0.7:
                    notes.append(f'{l_label} and {r_label} move strongly together over their overlapping period (correlation {corr:.2f}).')
                elif corr <= -0.4:
                    notes.append(f'{l_label} and {r_label} show a meaningful inverse relationship over their overlapping period (correlation {corr:.2f}).')
                else:
                    notes.append(f'{l_label} and {r_label} have weak to moderate alignment over their overlapping period (correlation {corr:.2f}).')
        return notes[:8]

    def _fallback_summary(
        self,
        frames: List[pd.DataFrame],
        findings: List[Dict],
        comparisons: List[str],
        climatology: List[str] | None = None,
        benchmark: List[Dict] | None = None,
    ) -> str:
        parts = ['<p><strong>Local statistical summary</strong></p>', '<ul>']

        for finding in findings[: min(2, len(findings))]:
            parts.append(
                '<li><strong>'
                f"{finding['title'].replace('_', ' ')}:"
                '</strong> '
                f"Mean {finding['mean']}, range {finding['record_low']} to {finding['record_high']}, "
                f"variability {finding['cv_percent']}, trend {finding['trend'].lower()}"
                '</li>'
            )
            parts.append(
                '<li><strong>Seasonality and extremes:</strong> '
                f"Q10 {finding['Q10']}, Q90 {finding['Q90']}, "
                f"{finding['pct_time_below_Q10']} of values below Q10 and {finding['pct_time_above_Q90']} above Q90."
                '</li>'
            )

        if climatology:
            first_block = climatology[0].splitlines()
            parts.append(
                '<li><strong>Climatology:</strong> '
                f"{' '.join(first_block[:3])}"
                '</li>'
            )

        if comparisons:
            parts.append(f'<li><strong>Cross-series relationship:</strong> {comparisons[0]}</li>')

        if benchmark:
            parts.append(f'<li><strong>Dataset benchmark:</strong> {self._benchmark_line(benchmark[0])}</li>')

        parts.append('</ul>')
        return ''.join(parts)

    def _fallback_multi_graph_summaries(self, graphs_data: List[Dict], benchmark: List[Dict]) -> List[str]:
        summaries: List[str] = []
        benchmark_line = self._benchmark_line(benchmark[0]) if benchmark else None
        for g in graphs_data:
            finding = g['findings'][0] if g['findings'] else None
            lines = ['<ul>']
            lines.append(f"<li><strong>Focus:</strong> {g['focus'].capitalize()}.</li>")
            if finding:
                lines.append(
                    '<li><strong>Primary series:</strong> '
                    f"{finding['title'].replace('_', ' ')} shows {finding['trend'].lower()} "
                    f"with mean {finding['mean']} and variability {finding['cv_percent']}.</li>"
                )
                lines.append(
                    '<li><strong>Extremes:</strong> '
                    f"Record low {finding['record_low']}; record high {finding['record_high']}.</li>"
                )
                lines.append(
                    '<li><strong>Seasonal spread:</strong> '
                    f"Q10 {finding['Q10']}, Q90 {finding['Q90']}, IQR {finding['IQR']}.</li>"
                )
            if g['comparisons']:
                lines.append(f"<li><strong>Comparison:</strong> {g['comparisons'][0]}</li>")
            if g['climatology']:
                first_clim = g['climatology'][0].splitlines()[-1].strip()
                lines.append(f'<li><strong>Climatology:</strong> {first_clim}</li>')
            elif benchmark_line:
                lines.append(f'<li><strong>Benchmark:</strong> {benchmark_line}</li>')
            lines.append('</ul>')
            summaries.append(''.join(lines))
        return summaries

    def _fallback_benchmark_summary(self, benchmark: List[Dict]) -> str:
        if not benchmark:
            return ''
        lines = ['<ul>']
        for row in benchmark:
            lines.append(f'<li><strong>{row["station_label"]}:</strong> {self._benchmark_line(row)}</li>')
        lines.append('</ul>')
        return ''.join(lines)

    def _benchmark_line(self, row: Dict[str, Any]) -> str:
        direction = 'above' if row['pct_diff'] >= 0 else 'below'
        return (
            f"{row['feature_label']} mean is {abs(row['pct_diff']):.1f}% {direction} the {row['dataset']} "
            f"dataset average, percentile rank {row['percentile_rank']:.0f}, z-score {row['z_score']:+.2f}."
        )

    def _longest_streak(self, values: pd.Series, above: float = None, below: float = None) -> int:
        if above is not None:
            flags = (values > above).astype(int)
        else:
            flags = (values < below).astype(int)
        max_streak = 0
        current = 0
        for v in flags:
            if v:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    def _trend_text(self, frame: pd.DataFrame) -> str:
        if len(frame) < 3:
            return 'Not enough points to estimate a trend.'
        x = np.arange(len(frame))
        y = frame['Value'].to_numpy(dtype=float)
        slope = np.polyfit(x, y, 1)[0]
        scale = np.nanstd(y)
        if scale == 0 or np.isnan(scale):
            return 'Essentially flat across the selected period.'
        normalized_slope = slope / scale
        if normalized_slope > 0.01:
            return 'Upward trend across the selected period.'
        if normalized_slope < -0.01:
            return 'Downward trend across the selected period.'
        return 'Broadly stable with only a gentle long-run drift.'

    def _anomaly_count(self, frame: pd.DataFrame) -> int:
        std = frame['Value'].std()
        if pd.isna(std) or std == 0:
            return 0
        z = (frame['Value'] - frame['Value'].mean()) / std
        return int((z.abs() >= 2.5).sum())
