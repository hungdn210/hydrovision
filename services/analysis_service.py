from __future__ import annotations

import os
import markdown
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .chart_service import ChartService
from .data_loader import DataRepository, SeriesRequest


class AnalysisService:
    def __init__(self, repository: DataRepository, chart_service: ChartService) -> None:
        self.repository = repository
        self.chart_service = chart_service

    def analyse(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        chart_payload = self.chart_service.generate_chart(payload)
        requests = [SeriesRequest(**item) for item in chart_payload['series']]
        series_frames = [self.repository.get_feature_series(request) for request in requests]

        findings = self._build_findings(series_frames)
        comparisons = self._build_comparisons(series_frames)
        summary = self._compose_summary(series_frames, findings, comparisons)

        # If we have a successful AI summary, suppress the raw findings cards to reduce clutter
        if summary.startswith("✨ AI Analysis:"):
            findings = []

        return {
            **chart_payload,
            'analysis': {
                'summary': summary,
                'findings': findings,
                'comparisons': comparisons,
            },
        }

    def _compose_summary(self, frames: List[pd.DataFrame], findings: List[Dict[str, str]], comparisons: List[str]) -> str:
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
            return base_summary + "\n\n*(Note: Set GEMINI_API_KEY in the .env file to enable advanced AI-powered summaries)*"

        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            
            prompt = (
                "Act as a professional hydrologist for the Mekong River Commission. "
                "Analyze the following hydrological data statistics and series correlations.\n\n"
                "Structure your response exactly like this:\n"
                "1. A clear 'Executive Summary' paragraph (2-3 sentences) giving the big picture of the station's health and behavior.\n"
                "2. A 'Detailed Insights' section with 4-5 high-quality bullet points. Each bullet must start with a bold **Heading:**.\n\n"
                "Focus on interpreting these specific metrics for the reader:\n"
                "- The meaning of the Trend results (stability vs drift).\n"
                "- The timing and intensity of Seasonal shifts.\n"
                "- Explaining variability and Extreme Events (citing specific max/min values from the data).\n"
                "- A note on Data Integrity based on the imputation and anomaly counts.\n"
                "IMPORTANT: Always use 'Pretty Names' for stations and features (e.g., use 'Ban Chot' instead of 'Ban_Chot'). Remove all underscores from names.\n"
                "Use professional formatting, keep it highly analytical, and do NOT include any introduction or sign-off.\n\n"
                f"Statistical Findings provided to you:\n{str(findings).replace('_', ' ')}\n\n"
                f"Relationship Comparisons (if any):\n{str(comparisons).replace('_', ' ')}"
            )
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            # Add a small AI badge or indication
            html_content = markdown.markdown(response.text.strip())
            return "✨ AI Analysis:\n" + html_content
        except Exception as e:
            return base_summary + f"\n\n*(AI Summary Generation Failed: {str(e)})*"

    def _build_findings(self, frames: List[pd.DataFrame]) -> List[Dict[str, str]]:
        cards: List[Dict[str, str]] = []
        for frame in frames:
            station = frame['Station'].iloc[0]
            feature = frame['Feature'].iloc[0]
            unit = frame['Unit'].iloc[0]
            observations = len(frame)
            imputed_share = frame['Imputed'].astype(str).str.lower().eq('yes').mean() * 100
            mean_val = frame['Value'].mean()
            median_val = frame['Value'].median()
            min_row = frame.loc[frame['Value'].idxmin()]
            max_row = frame.loc[frame['Value'].idxmax()]
            trend_text = self._trend_text(frame)
            seasonality_text = self._seasonality_text(frame)
            anomaly_count = self._anomaly_count(frame)

            body = (
                f'{observations:,} observations were included. Average {feature.lower()} was {mean_val:.2f} {unit}, '
                f'with median {median_val:.2f} {unit}. The lowest value was {min_row["Value"]:.2f} on '
                f'{min_row["Timestamp"].strftime("%Y-%m-%d")}, while the highest was {max_row["Value"]:.2f} on '
                f'{max_row["Timestamp"].strftime("%Y-%m-%d")}. {trend_text} {seasonality_text} '
                f'About {imputed_share:.1f}% of the plotted points are imputed, and {anomaly_count} anomaly point(s) were flagged using z-score screening.'
            )
            cards.append({
                'title': f'{station} · {feature}',
                'body': body,
            })
        return cards

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

    def _trend_text(self, frame: pd.DataFrame) -> str:
        if len(frame) < 3:
            return 'There are not enough points to estimate a robust trend.'
        x = np.arange(len(frame))
        y = frame['Value'].to_numpy(dtype=float)
        slope = np.polyfit(x, y, 1)[0]
        scale = np.nanstd(y)
        if scale == 0 or np.isnan(scale):
            return 'The series is essentially flat across the selected period.'
        normalized_slope = slope / scale
        if normalized_slope > 0.01:
            return 'The overall trend is upward across the selected period.'
        if normalized_slope < -0.01:
            return 'The overall trend is downward across the selected period.'
        return 'The overall trend is broadly stable with only a gentle long-run drift.'

    def _seasonality_text(self, frame: pd.DataFrame) -> str:
        if len(frame) < 24:
            return 'The selected range is too short for a strong seasonal reading.'
        month_means = frame.assign(Month=frame['Timestamp'].dt.month).groupby('Month')['Value'].mean()
        if month_means.empty:
            return 'No seasonal signal could be estimated.'
        peak_month = int(month_means.idxmax())
        low_month = int(month_means.idxmin())
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        return f'The seasonal profile peaks around {month_names[peak_month - 1]} and is weakest around {month_names[low_month - 1]}.'

    def _anomaly_count(self, frame: pd.DataFrame) -> int:
        std = frame['Value'].std()
        if pd.isna(std) or std == 0:
            return 0
        z = (frame['Value'] - frame['Value'].mean()) / std
        return int((z.abs() >= 2.5).sum())
