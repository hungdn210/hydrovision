from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

import re
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data_loader import DataRepository, SeriesRequest


GRAPH_TYPES = {
    'Single Category, Single Station Timeline',
    'Multiple Categories, Single Station Timeline',
    'Single Category Across Multiple Stations Comparison',
    'Multiple Categories Across Multiple Stations Comparison',
    'Year-over-Year Comparison',
    'Annual Monthly Totals Overview',
    'Flow Duration Curve',
    'Monthly Distribution Box Plot',
    'Multi-Station Temporal Heatmap',
    'Correlation Scatter Plot',
    'Anomaly Detection Chart',
    'Seasonal Subseries Plot',
    'Calendar Heatmap',
    'Rolling Correlation Chart',
    'Exceedance Probability Curve',
    'STL Decomposition',
    'Change-Point Detection',
    'Wavelet Analysis',
    'Granger Causality',
    'Cross-Correlation Function (CCF)',
}


class ChartService:
    def __init__(self, repository: DataRepository) -> None:
        self.repository = repository
        self.palette = [
            '#38bdf8', '#f59e0b', '#4ade80', '#f472b6',
            '#a78bfa', '#fb923c', '#34d399', '#60a5fa',
            '#fbbf24', '#818cf8', '#e879f9', '#2dd4bf',
        ]

    def validate_payload(self, payload: Dict[str, Any]) -> List[SeriesRequest]:
        graph_type = payload.get('graph_type')
        series = payload.get('series', [])
        if graph_type not in GRAPH_TYPES:
            raise ValueError('Unsupported graph type.')
        if not isinstance(series, list) or not series:
            raise ValueError('At least one series selection is required.')

        requests: List[SeriesRequest] = []
        for item in series:
            request = SeriesRequest(
                station=str(item.get('station', '')).strip(),
                feature=str(item.get('feature', '')).strip(),
                start_date=str(item.get('start_date', '')).strip(),
                end_date=str(item.get('end_date', '')).strip(),
            )
            if not request.station or not request.feature or not request.start_date or not request.end_date:
                raise ValueError('Each series must include station, feature, start_date, and end_date.')
            requests.append(request)

        stations = {req.station for req in requests}
        features = {req.feature for req in requests}

        if graph_type == 'Single Category, Single Station Timeline' and len(requests) != 1:
            raise ValueError('Single Category, Single Station Timeline requires exactly one selection row.')
        if graph_type == 'Multiple Categories, Single Station Timeline' and len(stations) != 1:
            raise ValueError('Multiple Categories, Single Station Timeline requires one station across multiple features.')
        if graph_type == 'Single Category Across Multiple Stations Comparison' and len(features) != 1:
            raise ValueError('Single Category Across Multiple Stations Comparison requires one feature across multiple stations.')
        if graph_type == 'Year-over-Year Comparison' and len(requests) != 1:
            raise ValueError('Year-over-Year Comparison requires exactly one selection row.')
        if graph_type == 'Annual Monthly Totals Overview' and len(requests) != 1:
            raise ValueError('Annual Monthly Totals Overview requires exactly one selection row.')
        if graph_type == 'Flow Duration Curve' and len(requests) != 1:
            raise ValueError('Flow Duration Curve requires exactly one selection row.')
        if graph_type == 'Monthly Distribution Box Plot' and len(requests) != 1:
            raise ValueError('Monthly Distribution Box Plot requires exactly one selection row.')
        if graph_type == 'Multi-Station Temporal Heatmap' and len(features) != 1:
            raise ValueError('Multi-Station Temporal Heatmap requires one feature across multiple stations.')
        if graph_type == 'Correlation Scatter Plot':
            if len(stations) != 1:
                raise ValueError('Correlation Scatter Plot requires all series from the same station.')
            if len(features) != 2:
                raise ValueError('Correlation Scatter Plot requires exactly 2 different features selected.')
        if graph_type == 'Anomaly Detection Chart' and len(requests) != 1:
            raise ValueError('Anomaly Detection Chart requires exactly one selection row.')
        if graph_type == 'Seasonal Subseries Plot' and len(requests) != 1:
            raise ValueError('Seasonal Subseries Plot requires exactly one selection row.')
        if graph_type == 'Calendar Heatmap' and len(requests) != 1:
            raise ValueError('Calendar Heatmap requires exactly one selection row.')
        if graph_type == 'Rolling Correlation Chart':
            if len(stations) != 1:
                raise ValueError('Rolling Correlation Chart requires all series from the same station.')
            if len(features) != 2:
                raise ValueError('Rolling Correlation Chart requires exactly 2 different features.')
        if graph_type == 'Exceedance Probability Curve' and len(requests) != 1:
            raise ValueError('Exceedance Probability Curve requires exactly one selection row.')
        if graph_type == 'STL Decomposition' and len(requests) != 1:
            raise ValueError('STL Decomposition requires exactly one selection row.')
        if graph_type == 'Change-Point Detection' and len(requests) != 1:
            raise ValueError('Change-Point Detection requires exactly one selection row.')
        if graph_type == 'Wavelet Analysis' and len(requests) != 1:
            raise ValueError('Wavelet Analysis requires exactly one selection row.')
        if graph_type == 'Granger Causality':
            if len(stations) != 1:
                raise ValueError('Granger Causality requires all series from the same station.')
            if len(features) != 2:
                raise ValueError('Granger Causality requires exactly 2 different features.')
        if graph_type == 'Cross-Correlation Function (CCF)' and len(requests) != 2:
            raise ValueError('Cross-Correlation Function (CCF) requires exactly 2 series (same or different stations).')

        return requests

    def generate_chart(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        requests = self.validate_payload(payload)
        graph_type = payload['graph_type']

        if graph_type == 'Single Category, Single Station Timeline':
            figure = self._single_category_single_station(requests[0])
        elif graph_type == 'Multiple Categories, Single Station Timeline':
            figure = self._multiple_categories_single_station(requests)
        elif graph_type == 'Single Category Across Multiple Stations Comparison':
            figure = self._single_category_multiple_stations(requests)
        elif graph_type == 'Multiple Categories Across Multiple Stations Comparison':
            figure = self._multiple_categories_multiple_stations(requests)
        elif graph_type == 'Year-over-Year Comparison':
            figure = self._year_over_year(requests[0])
        elif graph_type == 'Annual Monthly Totals Overview':
            figure = self._annual_monthly_totals(requests[0])
        elif graph_type == 'Flow Duration Curve':
            figure = self._flow_duration_curve(requests[0])
        elif graph_type == 'Monthly Distribution Box Plot':
            figure = self._monthly_distribution_box_plot(requests[0])
        elif graph_type == 'Multi-Station Temporal Heatmap':
            figure = self._multi_station_temporal_heatmap(requests)
        elif graph_type == 'Correlation Scatter Plot':
            figure = self._correlation_scatter_plot(requests)
        elif graph_type == 'Anomaly Detection Chart':
            figure = self._anomaly_detection_chart(requests[0])
        elif graph_type == 'Seasonal Subseries Plot':
            figure = self._seasonal_subseries_plot(requests[0])
        elif graph_type == 'Calendar Heatmap':
            figure = self._calendar_heatmap(requests[0])
        elif graph_type == 'Rolling Correlation Chart':
            figure = self._rolling_correlation_chart(requests)
        elif graph_type == 'Exceedance Probability Curve':
            figure = self._exceedance_probability_curve(requests[0])
        elif graph_type == 'STL Decomposition':
            figure = self._stl_decomposition(requests[0])
        elif graph_type == 'Change-Point Detection':
            figure = self._change_point_detection(requests[0])
        elif graph_type == 'Wavelet Analysis':
            figure = self._wavelet_analysis(requests[0])
        elif graph_type == 'Granger Causality':
            figure = self._granger_causality(requests)
        elif graph_type == 'Cross-Correlation Function (CCF)':
            figure = self._cross_correlation_function(requests)
        else:
            raise ValueError('Unsupported graph type.')

        series_payload = [asdict(req) for req in requests]
        title = figure.layout.title.text if (figure.layout.title and figure.layout.title.text) else graph_type
        # If there is a <br> tag, we take everything before it for the clean UI title
        # to avoid technical metadata cluttering the header.
        clean_title = title.split('<br>')[0].split('<BR>')[0]
        # Strip any other remaining tags and whitespace
        clean_title = re.sub(r'<[^>]*>', ' ', clean_title)
        clean_title = re.sub(r'\s+', ' ', clean_title).strip()
        # Card headers already display chart titles; remove in-plot titles to avoid
        # duplicate text and reclaim vertical space inside each card plot.
        figure.update_layout(title=None)
        figure_json = plotly.io.to_json(figure)

        result = {
            'graph_type': graph_type,
            'series': series_payload,
            'figure': figure_json,
            'title': clean_title,
        }
        return result

    def _base_layout(self, figure: go.Figure, title: str) -> None:
        figure.update_layout(
            template='plotly_dark',
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'y': 0.97,
                'yanchor': 'top',
                'font': {'size': 15, 'color': '#ededed', 'family': 'Inter, Arial, sans-serif'},
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Inter, Arial, sans-serif', 'size': 12, 'color': '#a0a0a0'},
            margin={'l': 64, 'r': 40, 't': 108, 'b': 110},
            hovermode='x unified',
            hoverlabel={
                'bgcolor': '#1e293b',
                'bordercolor': '#334155',
                'font': {'color': '#f1f5f9', 'size': 12},
            },
            legend={
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.02,
                'xanchor': 'center',
                'x': 0.5,
                'font': {'size': 11, 'color': '#475569'},
                'bgcolor': 'rgba(0,0,0,0)',
                'borderwidth': 0,
            },
        )
        figure.update_xaxes(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.08)',
            gridwidth=1,
            linecolor='rgba(255,255,255,0.15)',
            zeroline=False,
            automargin=True,
            tickfont={'size': 11, 'color': '#888888'},
            title_standoff=26,
            title_font={'size': 12, 'color': '#a0a0a0'},
        )
        figure.update_yaxes(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.08)',
            gridwidth=1,
            linecolor='rgba(255,255,255,0.15)',
            zeroline=False,
            automargin=True,
            tickfont={'size': 11, 'color': '#888888'},
            title_standoff=16,
            title_font={'size': 12, 'color': '#a0a0a0'},
        )

    @staticmethod
    def _normalize(values: pd.Series) -> pd.Series:
        min_val = values.min()
        max_val = values.max()
        if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
            return pd.Series([0.5] * len(values), index=values.index)
        return (values - min_val) / (max_val - min_val)

    def _single_category_single_station(self, request: SeriesRequest) -> go.Figure:
        df = self.repository.get_feature_series(request)
        unit = self.repository.feature_units.get(request.feature, '')
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df['Timestamp'],
                y=df['Value'],
                mode='lines',
                name=f"{request.station.replace('_', ' ')} · {request.feature.replace('_', ' ')}",
                line={'width': 2, 'color': '#38bdf8', 'shape': 'spline', 'smoothing': 0.3},
                fill='tozeroy',
                fillcolor='rgba(56,189,248,0.12)',
                customdata=df[['Imputed']].to_numpy(),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Value: %{y:.3f} ' + unit + '<br>Imputed: %{customdata[0]}<extra></extra>',
            )
        )
        self._base_layout(fig, f"{request.feature.replace('_', ' ')} Timeline · {request.station.replace('_', ' ')}")
        fig.update_xaxes(title='Date')
        fig.update_yaxes(title=f"{request.feature.replace('_', ' ')} ({unit})" if unit else request.feature.replace('_', ' '))
        return fig

    def _multiple_categories_single_station(self, requests: List[SeriesRequest]) -> go.Figure:
        station = requests[0].station
        fig = make_subplots(specs=[[{'secondary_y': True}]])
        for idx, request in enumerate(requests):
            df = self.repository.get_feature_series(request)
            normalized = self._normalize(df['Value'])
            fig.add_trace(
                go.Scatter(
                    x=df['Timestamp'],
                    y=normalized,
                    mode='lines',
                    name=f"{request.feature.replace('_', ' ')} (normalized)",
                    line={'width': 2.2, 'color': self.palette[idx % len(self.palette)]},
                    hovertemplate='Date: %{x|%Y-%m-%d}<br>Normalized: %{y:.3f}<extra></extra>',
                ),
                secondary_y=bool(idx % 2),
            )
        self._base_layout(fig, f"Multiple Features Timeline · {station.replace('_', ' ')}")
        fig.update_xaxes(title='Date')
        fig.update_yaxes(title='Normalized value', secondary_y=False)
        fig.update_yaxes(title='Normalized value', secondary_y=True)
        return fig

    def _single_category_multiple_stations(self, requests: List[SeriesRequest]) -> go.Figure:
        feature = requests[0].feature
        unit = self.repository.feature_units.get(feature, '')
        fig = go.Figure()
        for idx, request in enumerate(requests):
            df = self.repository.get_feature_series(request)
            fig.add_trace(
                go.Scatter(
                    x=df['Timestamp'],
                    y=df['Value'],
                    mode='lines',
                    name=request.station.replace('_', ' '),
                    line={'width': 2.1, 'color': self.palette[idx % len(self.palette)]},
                    hovertemplate='Date: %{x|%Y-%m-%d}<br>Value: %{y:.3f} ' + unit + '<extra></extra>',
                )
            )
        self._base_layout(fig, f"{feature.replace('_', ' ')} Across Multiple Stations")
        fig.update_layout(
            legend={
                'entrywidth': 90,
                'entrywidthmode': 'pixels',
            },
        )
        fig.update_xaxes(title='Date')
        fig.update_yaxes(title=f"{feature.replace('_', ' ')} ({unit})" if unit else feature.replace('_', ' '))
        return fig

    def _multiple_categories_multiple_stations(self, requests: List[SeriesRequest]) -> go.Figure:
        fig = make_subplots(specs=[[{'secondary_y': True}]])
        for idx, request in enumerate(requests):
            df = self.repository.get_feature_series(request)
            normalized = self._normalize(df['Value'])
            fig.add_trace(
                go.Scatter(
                    x=df['Timestamp'],
                    y=normalized,
                    mode='lines',
                    name=f"{request.station.replace('_', ' ')} · {request.feature.replace('_', ' ')}",
                    line={'width': 1.9, 'color': self.palette[idx % len(self.palette)]},
                    hovertemplate='Date: %{x|%Y-%m-%d}<br>Normalized: %{y:.3f}<extra></extra>',
                ),
                secondary_y=bool(idx % 2),
            )
        self._base_layout(fig, 'Multiple Stations · Multiple Features Comparison')
        fig.update_layout(
            legend={
                'entrywidth': 120,
                'entrywidthmode': 'pixels',
            },
        )
        fig.update_xaxes(title='Date')
        fig.update_yaxes(title='Normalized value', secondary_y=False)
        fig.update_yaxes(title='Normalized value', secondary_y=True)
        return fig

    def _year_over_year(self, request: SeriesRequest) -> go.Figure:
        df = self.repository.get_feature_series(request)
        df['Year'] = df['Timestamp'].dt.year
        df['Month'] = df['Timestamp'].dt.month
        monthly = df.groupby(['Year', 'Month'], as_index=False)['Value'].mean()
        fig = go.Figure()
        for idx, year in enumerate(sorted(monthly['Year'].unique())):
            year_data = monthly[monthly['Year'] == year]
            fig.add_trace(
                go.Scatter(
                    x=year_data['Month'],
                    y=year_data['Value'],
                    mode='lines+markers',
                    name=str(year),
                    line={'width': 2, 'color': self.palette[idx % len(self.palette)]},
                )
            )
        unit = self.repository.feature_units.get(request.feature, '')
        self._base_layout(fig, f"Year-over-Year · {request.feature.replace('_', ' ')} · {request.station.replace('_', ' ')}")
        # Keep legend placement deterministic for dense year traces so identical
        # cards render consistently and never overlap the x-axis title area.
        fig.update_layout(
            legend={
                'entrywidth': 52,
                'entrywidthmode': 'pixels',
            },
        )
        fig.update_xaxes(
            title='Month',
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        )
        fig.update_yaxes(title=f"{request.feature.replace('_', ' ')} ({unit})" if unit else request.feature.replace('_', ' '))
        return fig

    def _annual_monthly_totals(self, request: SeriesRequest) -> go.Figure:
        df = self.repository.get_feature_series(request)
        df['Month'] = df['Timestamp'].dt.month
        totals = df.groupby('Month')['Value'].sum().reindex(range(1, 13), fill_value=0)
        unit = self.repository.feature_units.get(request.feature, '')
        bar_colors = [
            '#93c5fd', '#7dd3fc', '#38bdf8', '#06b6d4',
            '#34d399', '#4ade80', '#facc15', '#fb923c',
            '#f87171', '#ef4444', '#a78bfa', '#818cf8',
        ]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    y=totals.values,
                    text=[f'{value:.1f}' for value in totals.values],
                    textposition='outside',
                    cliponaxis=False,
                    textfont={'size': 10, 'color': '#64748b'},
                    marker={
                        'color': bar_colors,
                        'line': {'width': 0},
                        'opacity': 0.88,
                    },
                    hovertemplate='<b>%{x}</b><br>Total: %{y:.2f} ' + unit + '<extra></extra>',
                )
            ]
        )
        self._base_layout(fig, f"Annual Monthly Totals · {request.feature.replace('_', ' ')} · {request.station.replace('_', ' ')}")
        fig.update_layout(
            height=520,
            margin={'l': 64, 'r': 40, 't': 116, 'b': 90},
        )
        fig.update_xaxes(title='Month')
        fig.update_yaxes(
            title=f"{request.feature.replace('_', ' ')} ({unit})" if unit else request.feature.replace('_', ' '),
            automargin=True,
        )
        return fig

    def _flow_duration_curve(self, request: SeriesRequest) -> go.Figure:
        df = self.repository.get_feature_series(request)
        unit = self.repository.feature_units.get(request.feature, '')
        values = df['Value'].sort_values(ascending=False).reset_index(drop=True)
        n = len(values)
        exceedance = [(i + 1) / (n + 1) * 100 for i in range(n)]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=exceedance,
                y=values,
                mode='lines',
                name=f"{request.station.replace('_', ' ')} · {request.feature.replace('_', ' ')}",
                line={'width': 2.2, 'color': '#38bdf8'},
                fill='tozeroy',
                fillcolor='rgba(56,189,248,0.12)',
                hovertemplate='<b>Exceedance: %{x:.1f}%</b><br>Value: %{y:.3f} ' + unit + '<extra></extra>',
            )
        )
        mean_val = values.mean()
        fig.add_hline(
            y=mean_val,
            line_dash='dash',
            line_color='#f59e0b',
            line_width=1.8,
            annotation_text=f'Mean: {mean_val:.2f} {unit}',
            annotation_position='top right',
            annotation_font_color='#f59e0b',
            annotation_font_size=11,
        )
        self._base_layout(fig, f"Flow Duration Curve · {request.feature.replace('_', ' ')} · {request.station.replace('_', ' ')}")
        fig.update_layout(hovermode='closest')
        fig.update_xaxes(title='Exceedance Probability (%)', range=[0, 100])
        fig.update_yaxes(title=f"{request.feature.replace('_', ' ')} ({unit})" if unit else request.feature.replace('_', ' '))
        return fig

    def _monthly_distribution_box_plot(self, request: SeriesRequest) -> go.Figure:
        df = self.repository.get_feature_series(request)
        unit = self.repository.feature_units.get(request.feature, '')
        df['Month'] = df['Timestamp'].dt.month
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        season_map = {
            1:  ('#93c5fd', 'rgba(147,197,253,0.25)'),
            2:  ('#7dd3fc', 'rgba(125,211,252,0.25)'),
            3:  ('#4ade80', 'rgba(74,222,128,0.25)'),
            4:  ('#22c55e', 'rgba(34,197,94,0.25)'),
            5:  ('#16a34a', 'rgba(22,163,74,0.25)'),
            6:  ('#fbbf24', 'rgba(251,191,36,0.25)'),
            7:  ('#f59e0b', 'rgba(245,158,11,0.25)'),
            8:  ('#d97706', 'rgba(217,119,6,0.25)'),
            9:  ('#fb923c', 'rgba(251,146,60,0.25)'),
            10: ('#f97316', 'rgba(249,115,22,0.25)'),
            11: ('#ea580c', 'rgba(234,88,12,0.25)'),
            12: ('#60a5fa', 'rgba(96,165,250,0.25)'),
        }
        fig = go.Figure()
        for month_num in range(1, 13):
            month_data = df[df['Month'] == month_num]['Value']
            clr, fill = season_map[month_num]
            fig.add_trace(
                go.Box(
                    y=month_data,
                    name=month_labels[month_num - 1],
                    marker={'color': clr, 'size': 3, 'opacity': 0.6},
                    line={'color': clr, 'width': 1.5},
                    fillcolor=fill,
                    boxmean='sd',
                    hoverinfo='y+name',
                )
            )
        self._base_layout(fig, f"Monthly Distribution · {request.feature.replace('_', ' ')} · {request.station.replace('_', ' ')}")
        fig.update_layout(
            hovermode='closest',
            showlegend=False,
        )
        fig.update_xaxes(title='Month')
        fig.update_yaxes(title=f"{request.feature.replace('_', ' ')} ({unit})" if unit else request.feature.replace('_', ' '))
        return fig

    def _multi_station_temporal_heatmap(self, requests: List[SeriesRequest]) -> go.Figure:
        feature = requests[0].feature
        unit = self.repository.feature_units.get(feature, '')

        frames = []
        for request in requests:
            df = self.repository.get_feature_series(request)
            monthly = (
                df.set_index('Timestamp')['Value']
                .resample('MS').mean()
            )
            monthly.name = request.station.replace('_', ' ')
            frames.append(monthly)

        combined = pd.concat(frames, axis=1).T
        # Convert DatetimeIndex columns to "YYYY-MM" strings
        combined.columns = [str(col)[:7] for col in combined.columns]

        station_names = list(combined.index)
        time_labels = list(combined.columns)
        # Replace NaN with None so Plotly renders blank cells correctly
        z_values = [[None if pd.isna(v) else float(v) for v in row]
                    for row in combined.values.tolist()]

        fig = go.Figure(
            data=go.Heatmap(
                z=z_values,
                x=time_labels,
                y=station_names,
                colorscale='Plasma',
                colorbar={
                    'title': {'text': unit, 'side': 'right', 'font': {'size': 11}},
                    'thickness': 14,
                    'len': 0.85,
                    'tickfont': {'size': 10},
                },
                hovertemplate='<b>%{y}</b><br>Period: %{x}<br>Value: %{z:.2f} ' + unit + '<extra></extra>',
            )
        )
        self._base_layout(fig, f"Temporal Heatmap · {feature.replace('_', ' ')} Across Stations")
        fig.update_layout(
            hovermode='closest',
            margin={'l': 120, 'r': 60, 't': 80, 'b': 150},
        )
        fig.update_xaxes(title='Time Period', tickangle=-45)
        fig.update_yaxes(title='', autorange='reversed')
        return fig

    def _correlation_scatter_plot(self, requests: List[SeriesRequest]) -> go.Figure:
        station = requests[0].station
        df_x = self.repository.get_feature_series(requests[0])
        df_y = self.repository.get_feature_series(requests[1])
        feature_x = requests[0].feature
        feature_y = requests[1].feature
        unit_x = self.repository.feature_units.get(feature_x, '')
        unit_y = self.repository.feature_units.get(feature_y, '')

        merged = pd.merge(
            df_x[['Timestamp', 'Value']].rename(columns={'Value': 'X'}),
            df_y[['Timestamp', 'Value']].rename(columns={'Value': 'Y'}),
            on='Timestamp',
            how='inner',
        )
        if merged.empty:
            raise ValueError('No overlapping dates between the two features.')

        merged['Month'] = merged['Timestamp'].dt.month
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        season_colors = {
            12: '#38bdf8', 1: '#38bdf8', 2: '#38bdf8',     # Winter — blue
            3: '#4ade80', 4: '#4ade80', 5: '#4ade80',       # Spring — green
            6: '#f59e0b', 7: '#f59e0b', 8: '#f59e0b',       # Summer — amber
            9: '#ef4444', 10: '#ef4444', 11: '#ef4444',      # Autumn — red
        }
        season_names = {
            12: 'Winter (Dec–Feb)', 1: 'Winter (Dec–Feb)', 2: 'Winter (Dec–Feb)',
            3: 'Spring (Mar–May)', 4: 'Spring (Mar–May)', 5: 'Spring (Mar–May)',
            6: 'Summer (Jun–Aug)', 7: 'Summer (Jun–Aug)', 8: 'Summer (Jun–Aug)',
            9: 'Autumn (Sep–Nov)', 10: 'Autumn (Sep–Nov)', 11: 'Autumn (Sep–Nov)',
        }

        fig = go.Figure()
        added_seasons = set()
        for month_num in range(1, 13):
            subset = merged[merged['Month'] == month_num]
            if subset.empty:
                continue
            season = season_names[month_num]
            show_legend = season not in added_seasons
            added_seasons.add(season)
            fig.add_trace(
                go.Scatter(
                    x=subset['X'],
                    y=subset['Y'],
                    mode='markers',
                    name=season,
                    showlegend=show_legend,
                    legendgroup=season,
                    marker={
                        'color': season_colors[month_num],
                        'size': 5,
                        'opacity': 0.6,
                    },
                    hovertemplate=(
                        f'{feature_x.replace("_", " ")}: %{{x:.2f}} {unit_x}<br>'
                        f'{feature_y.replace("_", " ")}: %{{y:.2f}} {unit_y}<br>'
                        f'Month: {month_labels[month_num - 1]}'
                        '<extra></extra>'
                    ),
                )
            )

        # OLS trend line
        if len(merged) > 1:
            x_vals = merged['X']
            y_vals = merged['Y']
            slope, intercept = pd.Series(y_vals).cov(x_vals) / pd.Series(x_vals).var(), 0
            intercept = y_vals.mean() - slope * x_vals.mean()
            x_line = pd.Series([x_vals.min(), x_vals.max()])
            y_line = slope * x_line + intercept
            corr = merged['X'].corr(merged['Y'])
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode='lines',
                    name=f'Trend (r={corr:.3f})',
                    line={'width': 2, 'color': '#6366f1', 'dash': 'dash'},
                )
            )

        self._base_layout(fig, f"Correlation · {feature_x.replace('_', ' ')} vs {feature_y.replace('_', ' ')} · {station.replace('_', ' ')}")
        fig.update_layout(hovermode='closest')
        fig.update_xaxes(title=f"{feature_x.replace('_', ' ')} ({unit_x})" if unit_x else feature_x.replace('_', ' '))
        fig.update_yaxes(title=f"{feature_y.replace('_', ' ')} ({unit_y})" if unit_y else feature_y.replace('_', ' '))
        return fig

    def _anomaly_detection_chart(self, request: SeriesRequest) -> go.Figure:
        df = self.repository.get_feature_series(request)
        unit = self.repository.feature_units.get(request.feature, '')

        # Aggregate to monthly averages so bars are wide enough to see
        monthly = (
            df.set_index('Timestamp')['Value']
            .resample('ME')
            .mean()
            .dropna()
            .reset_index()
        )
        monthly.columns = ['Timestamp', 'Value']

        # Climatological mean: average of all years for each calendar month (1–12)
        monthly['CalMonth'] = monthly['Timestamp'].dt.month
        clim = monthly.groupby('CalMonth')['Value'].mean()
        monthly['Climatology'] = monthly['CalMonth'].map(clim)
        monthly['Anomaly'] = monthly['Value'] - monthly['Climatology']

        pos = monthly['Anomaly'].clip(lower=0)
        neg = monthly['Anomaly'].clip(upper=0)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=monthly['Timestamp'],
                y=pos,
                name='Above average',
                marker={'color': 'rgba(56,189,248,0.75)', 'line': {'width': 0}},
                hovertemplate='<b>%{x|%Y-%m}</b><br>Anomaly: +%{y:.2f} ' + unit + '<extra></extra>',
            )
        )
        fig.add_trace(
            go.Bar(
                x=monthly['Timestamp'],
                y=neg,
                name='Below average',
                marker={'color': 'rgba(248,113,113,0.75)', 'line': {'width': 0}},
                hovertemplate='<b>%{x|%Y-%m}</b><br>Anomaly: %{y:.2f} ' + unit + '<extra></extra>',
            )
        )
        fig.add_trace(
            go.Scatter(
                x=monthly['Timestamp'],
                y=monthly['Climatology'],
                mode='lines',
                name='Climatological mean',
                line={'width': 1.8, 'color': '#f59e0b', 'dash': 'dot'},
                hovertemplate='<b>%{x|%Y-%m}</b><br>Mean: %{y:.2f} ' + unit + '<extra></extra>',
            )
        )
        self._base_layout(fig, f"Anomaly Detection · {request.feature.replace('_', ' ')} · {request.station.replace('_', ' ')}")
        fig.update_layout(
            hovermode='x unified',
            barmode='relative',
        )
        fig.update_xaxes(title='Month')
        fig.update_yaxes(title=f"Anomaly ({unit})" if unit else 'Anomaly')
        return fig

    def _seasonal_subseries_plot(self, request: SeriesRequest) -> go.Figure:
        df = self.repository.get_feature_series(request)
        unit = self.repository.feature_units.get(request.feature, '')
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        df['Year'] = df['Timestamp'].dt.year
        df['Month'] = df['Timestamp'].dt.month
        monthly = df.groupby(['Year', 'Month'], as_index=False)['Value'].mean()
        years_sorted = sorted(int(y) for y in monthly['Year'].unique())
        n_years = len(years_sorted)
        # Rotate and thin year ticks when the series is long to avoid overlap.
        tick_angle = -45 if n_years > 10 else 0
        step = max(1, int(np.ceil(n_years / 6))) if n_years > 0 else 1
        tickvals = years_sorted[::step]
        ticktext = [str(y) for y in tickvals]
        bottom_margin = 76 if tick_angle else 58

        fig = make_subplots(
            rows=2, cols=6,
            subplot_titles=month_labels,
            shared_yaxes=True,
            horizontal_spacing=0.04,
            vertical_spacing=0.22,
        )
        for i, month_num in enumerate(range(1, 13)):
            row = 1 if i < 6 else 2
            col = (i % 6) + 1
            month_data = monthly[monthly['Month'] == month_num].sort_values('Year')
            if month_data.empty:
                continue
            x_vals = month_data['Year'].values
            y_vals = month_data['Value'].values
            season_clr = ['#93c5fd','#7dd3fc','#4ade80','#22c55e','#16a34a','#fbbf24',
                          '#f59e0b','#d97706','#fb923c','#f97316','#ea580c','#60a5fa'][i]
            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='lines+markers',
                    name=month_labels[i],
                    line={'width': 1.8, 'color': season_clr},
                    marker={'size': 4, 'color': season_clr, 'opacity': 0.8},
                    showlegend=False,
                    hovertemplate=f'<b>{month_labels[i]} %{{x}}</b>: %{{y:.2f}} {unit}<extra></extra>',
                ),
                row=row, col=col,
            )
            if len(x_vals) >= 3:
                coeffs = np.polyfit(x_vals, y_vals, 1)
                y_trend = np.polyval(coeffs, x_vals)
                color = '#ef4444' if coeffs[0] < 0 else '#16a34a'
                fig.add_trace(
                    go.Scatter(
                        x=x_vals, y=y_trend,
                        mode='lines', showlegend=False,
                        line={'width': 1.4, 'color': color, 'dash': 'dash'},
                        hoverinfo='skip',
                    ),
                    row=row, col=col,
                )
        fig.update_layout(
            template='plotly_white',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Inter, Arial, sans-serif', 'size': 11, 'color': '#334155'},
            margin={'l': 56, 'r': 20, 't': 56, 'b': bottom_margin},
            showlegend=False,
            meta={
                'graph_type': 'Seasonal Subseries Plot',
                'seasonal_years': years_sorted,
            },
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor='rgba(148,163,184,0.22)',
            tickformat='d',
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=tick_angle,
            tickfont={'size': 8, 'color': '#64748b'},
            automargin=True,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor='rgba(148,163,184,0.22)',
            tickfont={'size': 10, 'color': '#64748b'},
        )
        for col in range(1, 7):
            fig.update_xaxes(showticklabels=False, row=1, col=col)
            fig.update_xaxes(showticklabels=True, row=2, col=col)
        fig.update_annotations(font={'size': 16, 'color': '#475569'})
        return fig

    def _calendar_heatmap(self, request: SeriesRequest) -> go.Figure:
        df = self.repository.get_feature_series(request)
        unit = self.repository.feature_units.get(request.feature, '')
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        monthly = (
            df.set_index('Timestamp')['Value']
            .resample('MS').mean()
            .dropna()
            .reset_index()
        )
        monthly.columns = ['Timestamp', 'Value']
        monthly['Year'] = monthly['Timestamp'].dt.year
        monthly['Month'] = monthly['Timestamp'].dt.month

        years = sorted(monthly['Year'].unique())
        z = []
        for yr in years:
            row = []
            for m in range(1, 13):
                subset = monthly[(monthly['Year'] == yr) & (monthly['Month'] == m)]['Value']
                row.append(float(subset.iloc[0]) if not subset.empty else None)
            z.append(row)

        n_years = len(years)
        # ~18px per year row, minimum 300px, scales to fill ~80% of a typical panel
        dynamic_height = max(300, n_years * 18 + 160)
        # Show a tick every N years so labels don't overlap (aim for ~15 visible ticks max)
        dtick = max(1, int(np.ceil(n_years / 15)))

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=month_labels,
                y=years,
                colorscale='RdYlBu_r',
                colorbar={
                    'title': {'text': unit, 'side': 'right', 'font': {'size': 11}},
                    'thickness': 14,
                    'len': 0.85,
                    'tickfont': {'size': 10},
                },
                hovertemplate='<b>%{y} · %{x}</b><br>Value: %{z:.3f} ' + unit + '<extra></extra>',
            )
        )
        self._base_layout(fig, f"Calendar Heatmap · {request.feature.replace('_', ' ')} · {request.station.replace('_', ' ')}")
        fig.update_layout(hovermode='closest', margin={'l': 60, 'r': 80, 't': 80, 'b': 60}, height=dynamic_height)
        fig.update_xaxes(title='Month')
        fig.update_yaxes(title='Year', autorange='reversed', tickformat='d', dtick=dtick)
        return fig

    def _station_ranking_bar_chart(self, requests: List[SeriesRequest]) -> go.Figure:
        feature = requests[0].feature
        unit = self.repository.feature_units.get(feature, '')

        rows = []
        for request in requests:
            df = self.repository.get_feature_series(request)
            rows.append({
                'station': request.station.replace('_', ' '),
                'mean': float(df['Value'].mean()),
                'median': float(df['Value'].median()),
                'min': float(df['Value'].min()),
                'max': float(df['Value'].max()),
            })
        # Sort descending so rank 1 (highest mean) is at top
        rows.sort(key=lambda x: x['mean'], reverse=True)

        stations = [r['station'] for r in rows]
        means = [r['mean'] for r in rows]
        medians = [r['median'] for r in rows]
        mins = [r['min'] for r in rows]
        maxes = [r['max'] for r in rows]
        n = len(rows)
        rank_labels = [f'#{i+1:02d}  {station}' for i, station in enumerate(stations)]
        colors = [
            '#0f172a' if i == 0 else '#1d4ed8' if i == 1 else '#2563eb' if i == 2
            else '#60a5fa'
            for i in range(n)
        ]

        fig = go.Figure()
        x_upper = max(maxes) * 1.18 if maxes else 1.0

        for idx, (label, min_v, max_v) in enumerate(zip(rank_labels, mins, maxes)):
            band_fill = 'rgba(248,250,252,0.9)' if idx % 2 == 0 else 'rgba(241,245,249,0.6)'
            fig.add_shape(
                type='rect',
                x0=0,
                x1=x_upper,
                y0=idx - 0.46,
                y1=idx + 0.46,
                xref='x',
                yref='y',
                fillcolor=band_fill,
                line={'width': 0},
                layer='below',
            )
            fig.add_trace(go.Scatter(
                x=[min_v, max_v],
                y=[label, label],
                mode='lines',
                name='Observed range' if idx == 0 else None,
                line={'color': 'rgba(148,163,184,0.42)', 'width': 10},
                hovertemplate='<b>%{y}</b><br>Range: %{customdata[0]:.3f} → %{customdata[1]:.3f} ' + unit + '<extra></extra>',
                customdata=[(min_v, max_v), (min_v, max_v)],
                showlegend=idx == 0,
            ))

        fig.add_trace(go.Scatter(
            y=rank_labels,
            x=mins,
            mode='markers',
            name='Minimum',
            marker={
                'color': 'rgba(255,255,255,0.95)',
                'size': 9,
                'symbol': 'circle',
                'line': {'width': 2, 'color': '#94a3b8'},
            },
            hovertemplate='<b>%{y}</b><br>Min: %{x:.3f} ' + unit + '<extra></extra>',
        ))

        # Median tick
        fig.add_trace(go.Scatter(
            y=rank_labels, x=medians,
            mode='markers',
            name='Median',
            marker={'color': '#f59e0b', 'size': 10, 'symbol': 'line-ns-open',
                    'line': {'width': 3, 'color': '#f59e0b'}},
            hovertemplate='<b>%{y}</b><br>Median: %{x:.3f} ' + unit + '<extra></extra>',
        ))

        fig.add_trace(go.Scatter(
            y=rank_labels,
            x=means,
            mode='markers+text',
            name='Mean',
            marker={
                'color': colors,
                'size': [18 if i == 0 else 16 if i == 1 else 15 if i == 2 else 13 for i in range(n)],
                'symbol': 'circle',
                'line': {'width': 2.4, 'color': 'rgba(255,255,255,0.95)'},
            },
            text=[f'{v:.2f}' for v in means],
            textposition='middle right',
            textfont={'size': 10, 'color': '#334155'},
            cliponaxis=False,
            hovertemplate='<b>%{y}</b><br>Mean: %{x:.3f} ' + unit + '<extra></extra>',
        ))

        # Max diamond
        fig.add_trace(go.Scatter(
            y=rank_labels, x=maxes,
            mode='markers',
            name='Max',
            marker={'color': '#ef4444', 'size': 10, 'symbol': 'diamond',
                    'line': {'width': 1.2, 'color': 'white'}},
            hovertemplate='<b>%{y}</b><br>Max: %{x:.3f} ' + unit + '<extra></extra>',
        ))

        self._base_layout(fig, f"Station Ranking · {feature.replace('_', ' ')}")
        fig.update_layout(
            hovermode='closest',
            plot_bgcolor='#f8fafc',
            paper_bgcolor='rgba(0,0,0,0)',
            height=max(420, n * 64 + 150),
            margin={'l': 188, 'r': 120, 't': 72, 'b': 76},
            legend={
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.01,
                'xanchor': 'left',
                'x': 0.0,
                'font': {'size': 11, 'color': '#475569'},
                'bgcolor': 'rgba(255,255,255,0.7)',
                'borderwidth': 0,
            },
        )
        fig.add_vline(
            x=float(np.mean(means)),
            line_dash='dot',
            line_width=1.5,
            line_color='rgba(245,158,11,0.78)',
            annotation_text='Network mean',
            annotation_position='top right',
            annotation_font={'size': 10, 'color': '#b45309'},
        )
        x_title = f"{feature.replace('_', ' ')} ({unit})" if unit else feature.replace('_', ' ')
        fig.update_xaxes(
            title=x_title,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(148,163,184,0.28)',
            showgrid=True,
            gridcolor='rgba(148,163,184,0.18)',
            tickfont={'size': 11, 'color': '#475569'},
            range=[0, x_upper],
        )
        fig.update_yaxes(
            title='',
            categoryorder='array',
            categoryarray=list(reversed(rank_labels)),
            tickfont={'size': 12, 'color': '#0f172a'},
            automargin=True,
        )
        return fig

    def _rolling_correlation_chart(self, requests: List[SeriesRequest]) -> go.Figure:
        station = requests[0].station
        feature_x = requests[0].feature
        feature_y = requests[1].feature
        df_x = self.repository.get_feature_series(requests[0])
        df_y = self.repository.get_feature_series(requests[1])

        frequency = self.repository.feature_frequency.get(feature_x, 'daily')
        window = 12 if frequency == 'monthly' else 365
        window_label = '12-month' if frequency == 'monthly' else '365-day'

        merged = pd.merge(
            df_x[['Timestamp', 'Value']].rename(columns={'Value': 'X'}),
            df_y[['Timestamp', 'Value']].rename(columns={'Value': 'Y'}),
            on='Timestamp', how='inner',
        ).sort_values('Timestamp')

        min_pts = max(6, window // 6)
        if len(merged) < min_pts:
            raise ValueError(f'Not enough overlapping data for {window_label} rolling correlation.')

        merged['RollingCorr'] = merged['X'].rolling(window=window, min_periods=min_pts).corr(merged['Y'])

        rolling_valid = merged.dropna(subset=['RollingCorr']).copy()
        latest_corr = float(rolling_valid['RollingCorr'].iloc[-1]) if not rolling_valid.empty else None
        mean_corr = float(rolling_valid['RollingCorr'].mean()) if not rolling_valid.empty else None
        strongest_idx = rolling_valid['RollingCorr'].abs().idxmax() if not rolling_valid.empty else None
        strongest_row = rolling_valid.loc[strongest_idx] if strongest_idx is not None else None
        strong_share = float((rolling_valid['RollingCorr'].abs() >= 0.7).mean() * 100) if not rolling_valid.empty else 0.0

        merged['XNorm'] = self._normalize(merged['X'])
        merged['YNorm'] = self._normalize(merged['Y'])

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.72, 0.28],
            vertical_spacing=0.08,
            subplot_titles=(
                f'{window_label.title()} correlation regime',
                'Normalized source signals',
            ),
        )

        for y0, y1, fill, label in [
            (0.7, 1.05, 'rgba(56,189,248,0.08)', 'Strong positive'),
            (0.3, 0.7, 'rgba(34,197,94,0.06)', 'Moderate positive'),
            (-0.3, 0.3, 'rgba(148,163,184,0.05)', 'Weak / unstable'),
            (-0.7, -0.3, 'rgba(251,146,60,0.06)', 'Moderate inverse'),
            (-1.05, -0.7, 'rgba(239,68,68,0.08)', 'Strong inverse'),
        ]:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=fill, line_width=0, row=1, col=1)
            fig.add_annotation(
                xref='paper',
                yref='y1',
                x=1.005,
                y=(y0 + y1) / 2,
                text=label,
                showarrow=False,
                font={'size': 10, 'color': '#94a3b8'},
                xanchor='left',
            )

        for y, dash, color, text in [
            (0.0, 'dash', 'rgba(148,163,184,0.45)', 'Neutral'),
            (0.7, 'dot', 'rgba(56,189,248,0.45)', 'r = 0.7'),
            (-0.7, 'dot', 'rgba(239,68,68,0.45)', 'r = -0.7'),
        ]:
            fig.add_hline(
                y=y,
                line_dash=dash,
                line_color=color,
                line_width=1,
                row=1,
                col=1,
                annotation_text=text,
                annotation_font_size=9,
                annotation_font_color=color.replace('0.45', '1.0'),
            )

        fig.add_trace(
            go.Scatter(
                x=merged['Timestamp'],
                y=merged['RollingCorr'],
                mode='lines',
                name=f'{window_label} rolling r',
                line={'width': 3, 'color': '#8b5cf6'},
                fill='tozeroy',
                fillcolor='rgba(139,92,246,0.12)',
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Rolling r: %{y:.3f}<extra></extra>',
            ),
            row=1,
            col=1,
        )

        if not rolling_valid.empty:
            fig.add_trace(
                go.Scatter(
                    x=[rolling_valid['Timestamp'].iloc[-1]],
                    y=[rolling_valid['RollingCorr'].iloc[-1]],
                    mode='markers',
                    name='Latest',
                    marker={'size': 10, 'color': '#f8fafc', 'line': {'color': '#8b5cf6', 'width': 2}},
                    hovertemplate='<b>Latest</b><br>%{x|%Y-%m-%d}<br>Rolling r: %{y:.3f}<extra></extra>',
                ),
                row=1,
                col=1,
            )
            if strongest_row is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[strongest_row['Timestamp']],
                        y=[strongest_row['RollingCorr']],
                        mode='markers',
                        name='Max |r|',
                        marker={
                            'size': 11,
                            'symbol': 'diamond',
                            'color': '#f59e0b',
                            'line': {'color': '#111827', 'width': 1.5},
                        },
                        hovertemplate='<b>Strongest regime</b><br>%{x|%Y-%m-%d}<br>Rolling r: %{y:.3f}<extra></extra>',
                    ),
                    row=1,
                    col=1,
                )

        fig.add_trace(
            go.Scatter(
                x=merged['Timestamp'],
                y=merged['XNorm'],
                mode='lines',
                name=feature_x.replace('_', ' '),
                line={'width': 1.9, 'color': '#38bdf8'},
                opacity=0.95,
                hovertemplate=f'<b>{feature_x.replace("_", " ")}</b><br>%{{x|%Y-%m-%d}}<br>Normalized: %{{y:.3f}}<extra></extra>',
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=merged['Timestamp'],
                y=merged['YNorm'],
                mode='lines',
                name=feature_y.replace('_', ' '),
                line={'width': 1.9, 'color': '#f59e0b'},
                opacity=0.9,
                hovertemplate=f'<b>{feature_y.replace("_", " ")}</b><br>%{{x|%Y-%m-%d}}<br>Normalized: %{{y:.3f}}<extra></extra>',
            ),
            row=2,
            col=1,
        )

        self._base_layout(
            fig,
            f"Rolling Correlation · {feature_x.replace('_', ' ')} vs {feature_y.replace('_', ' ')} · {station.replace('_', ' ')}",
        )
        fig.update_layout(
            margin={'l': 72, 'r': 122, 't': 118, 'b': 84},
            legend={
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.06,
                'xanchor': 'left',
                'x': 0,
            },
            annotations=list(fig.layout.annotations) + [
                go.layout.Annotation(
                    xref='paper',
                    yref='paper',
                    x=0.995,
                    y=1.13,
                    xanchor='right',
                    yanchor='top',
                    align='left',
                    showarrow=False,
                    bgcolor='rgba(15,23,42,0.82)',
                    bordercolor='rgba(148,163,184,0.26)',
                    borderwidth=1,
                    borderpad=8,
                    font={'size': 10, 'color': '#e2e8f0'},
                    text=(
                        f"<b>Window</b>: {window_label}<br>"
                        f"<b>Latest r</b>: {latest_corr:.3f}<br>"
                        f"<b>Mean r</b>: {mean_corr:.3f}<br>"
                        f"<b>Max |r|</b>: {float(strongest_row['RollingCorr']):.3f}<br>"
                        f"<b>|r| ≥ 0.7</b>: {strong_share:.0f}%"
                    ) if strongest_row is not None and latest_corr is not None and mean_corr is not None else (
                        f"<b>Window</b>: {window_label}<br>"
                        f"<b>Overlapping records</b>: {len(merged):,}"
                    ),
                )
            ],
        )
        fig.update_xaxes(title='Date', row=2, col=1)
        fig.update_yaxes(
            title=f'Pearson r ({window_label} window)',
            range=[-1.05, 1.05],
            tickmode='array',
            tickvals=[-1, -0.7, -0.3, 0, 0.3, 0.7, 1],
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title='Normalized level',
            range=[-0.02, 1.02],
            tickmode='array',
            tickvals=[0, 0.5, 1],
            row=2,
            col=1,
        )
        return fig

    def _exceedance_probability_curve(self, request: SeriesRequest) -> go.Figure:
        df = self.repository.get_feature_series(request)
        unit = self.repository.feature_units.get(request.feature, '')

        values = df['Value'].dropna().sort_values(ascending=False).reset_index(drop=True)
        n = len(values)
        # Weibull plotting positions
        rank = np.arange(1, n + 1)
        exceedance_pct = rank / (n + 1) * 100  # percent

        p10 = float(values.quantile(0.10))
        p50 = float(values.quantile(0.50))
        p90 = float(values.quantile(0.90))
        mean_val = float(values.mean())
        max_val = float(values.iloc[0])
        min_val = float(values.iloc[-1])

        return_periods = [2, 5, 10, 25, 50, 100]
        rp_points = []
        for rp in return_periods:
            prob = 100 / rp
            if exceedance_pct.min() <= prob <= exceedance_pct.max():
                value_at_prob = float(np.interp(prob, exceedance_pct[::-1], values.values[::-1]))
                rp_points.append((rp, prob, value_at_prob))

        fig = go.Figure()
        for x0, x1, fill in [
            (0.05, 1, 'rgba(239,68,68,0.05)'),
            (1, 10, 'rgba(245,158,11,0.05)'),
            (10, 50, 'rgba(56,189,248,0.04)'),
            (50, 100, 'rgba(34,197,94,0.04)'),
        ]:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=fill, line_width=0)

        fig.add_hrect(y0=p90, y1=max_val, fillcolor='rgba(139,92,246,0.06)', line_width=0)
        fig.add_hrect(y0=min_val, y1=p10, fillcolor='rgba(56,189,248,0.05)', line_width=0)

        fig.add_trace(
            go.Scatter(
                x=exceedance_pct,
                y=values.values,
                mode='lines',
                name=request.feature.replace('_', ' '),
                line={'width': 3, 'color': '#8b5cf6', 'shape': 'spline', 'smoothing': 0.45},
                fill='tozeroy',
                fillcolor='rgba(139,92,246,0.12)',
                customdata=np.column_stack((100 / exceedance_pct, rank)),
                hovertemplate=(
                    '<b>Exceedance: %{x:.2f}%</b><br>'
                    'Return period: %{customdata[0]:.1f} years<br>'
                    'Rank: %{customdata[1]:.0f} / ' + str(n) + '<br>'
                    'Value: %{y:.3f} ' + unit + '<extra></extra>'
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=exceedance_pct,
                y=values.values,
                mode='markers',
                name='Observed ranks',
                marker={
                    'size': 4.5,
                    'color': values.values,
                    'colorscale': [[0, '#38bdf8'], [0.55, '#8b5cf6'], [1, '#f59e0b']],
                    'opacity': 0.72,
                    'line': {'width': 0},
                    'showscale': False,
                },
                customdata=100 / exceedance_pct,
                hovertemplate='<b>Observed point</b><br>Exceedance: %{x:.2f}%<br>Return period: %{customdata:.1f} years<br>Value: %{y:.3f} ' + unit + '<extra></extra>',
            )
        )

        if rp_points:
            fig.add_trace(
                go.Scatter(
                    x=[item[1] for item in rp_points],
                    y=[item[2] for item in rp_points],
                    mode='markers+text',
                    name='Design return periods',
                    text=[f'{item[0]}y' for item in rp_points],
                    textposition='top center',
                    marker={
                        'size': 10,
                        'symbol': 'diamond',
                        'color': '#f59e0b',
                        'line': {'color': '#111827', 'width': 1.4},
                    },
                    customdata=[item[0] for item in rp_points],
                    hovertemplate='<b>%{customdata}-year event</b><br>Exceedance: %{x:.2f}%<br>Value: %{y:.3f} ' + unit + '<extra></extra>',
                )
            )

        fig.add_hline(
            y=mean_val,
            line_dash='dash',
            line_color='#f59e0b',
            line_width=1.5,
        )
        for y, color in [
            (p90, 'rgba(139,92,246,0.40)'),
            (p50, 'rgba(148,163,184,0.38)'),
            (p10, 'rgba(56,189,248,0.40)'),
        ]:
            fig.add_hline(
                y=y,
                line_dash='dot',
                line_color=color,
                line_width=1,
            )

        for rp, prob, _value in rp_points:
            fig.add_vline(
                x=prob,
                line_dash='dot',
                line_color='rgba(245,158,11,0.32)',
                line_width=1,
            )

        self._base_layout(fig, f"Exceedance Probability · {request.feature.replace('_', ' ')} · {request.station.replace('_', ' ')}")
        fig.update_layout(
            hovermode='closest',
            margin={'l': 72, 'r': 42, 't': 78, 'b': 88},
            legend={
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.02,
                'xanchor': 'left',
                'x': 0,
            },
            annotations=list(fig.layout.annotations) + [
                go.layout.Annotation(
                    xref='paper',
                    yref='paper',
                    x=0.01,
                    y=0.99,
                    xanchor='left',
                    yanchor='top',
                    align='left',
                    showarrow=False,
                    bgcolor='rgba(15,23,42,0.68)',
                    bordercolor='rgba(148,163,184,0.18)',
                    borderwidth=1,
                    borderpad=6,
                    font={'size': 9, 'color': '#e2e8f0'},
                    text=(
                        f"<b>n</b> {n:,} &nbsp;|&nbsp; "
                        f"<b>Max</b> {max_val:.2f} {unit} &nbsp;|&nbsp; "
                        f"<b>P50</b> {p50:.2f} {unit} &nbsp;|&nbsp; "
                        f"<b>Min</b> {min_val:.2f} {unit}"
                    ),
                )
            ],
        )
        fig.update_xaxes(
            title='Exceedance Probability (%)',
            type='log',
            tickvals=[0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 90],
            ticktext=['0.05', '0.1', '0.5', '1', '2', '5', '10', '20', '50', '90'],
            tickangle=0,
            tickfont={'size': 10, 'color': '#64748b'},
            automargin=True,
            range=[np.log10(max(exceedance_pct.min(), 0.001)), np.log10(min(exceedance_pct.max(), 100))],
        )
        fig.update_yaxes(title=f"{request.feature.replace('_', ' ')} ({unit})" if unit else request.feature.replace('_', ' '))
        return fig

    # ------------------------------------------------------------------ #
    #  Advanced Statistical Analysis — 5 new chart types                  #
    # ------------------------------------------------------------------ #

    def _stl_decomposition(self, request: SeriesRequest) -> go.Figure:
        """Seasonal-Trend decomposition using LOESS (STL). 4-panel subplot."""
        from statsmodels.tsa.seasonal import STL

        df = self.repository.get_feature_series(request)
        unit = self.repository.feature_units.get(request.feature, '')
        frequency = self.repository.feature_frequency.get(request.feature, 'daily')

        # Always work at monthly resolution — keeps STL fast and period=12 natural
        ts = (
            df.set_index('Timestamp')['Value']
            .resample('MS').mean()
            .dropna()
        )
        period = 12
        if len(ts) < 2 * period:
            raise ValueError(
                f'Need at least {2 * period} months of data for STL decomposition '
                f'(got {len(ts)} months).'
            )

        stl = STL(ts, period=period, robust=True)
        res = stl.fit()
        dates = ts.index

        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
            shared_xaxes=True,
            vertical_spacing=0.055,
            row_heights=[0.28, 0.24, 0.24, 0.24],
        )

        # Original
        fig.add_trace(go.Scatter(
            x=dates, y=res.observed,
            mode='lines', name='Original',
            line={'width': 1.8, 'color': '#38bdf8'},
            hovertemplate='<b>%{x|%Y-%m}</b><br>Value: %{y:.3f} ' + unit + '<extra></extra>',
        ), row=1, col=1)

        # Trend
        fig.add_trace(go.Scatter(
            x=dates, y=res.trend,
            mode='lines', name='Trend',
            line={'width': 2.4, 'color': '#f59e0b'},
            hovertemplate='<b>%{x|%Y-%m}</b><br>Trend: %{y:.3f} ' + unit + '<extra></extra>',
        ), row=2, col=1)

        # Seasonal
        fig.add_trace(go.Scatter(
            x=dates, y=res.seasonal,
            mode='lines', name='Seasonal',
            line={'width': 1.8, 'color': '#4ade80'},
            fill='tozeroy',
            fillcolor='rgba(74,222,128,0.1)',
            hovertemplate='<b>%{x|%Y-%m}</b><br>Seasonal: %{y:.3f} ' + unit + '<extra></extra>',
        ), row=3, col=1)

        # Residual — positive/negative bars
        resid = pd.Series(res.resid, index=dates)
        fig.add_trace(go.Bar(
            x=dates, y=resid.clip(lower=0).values,
            name='Residual (+)',
            marker={'color': 'rgba(56,189,248,0.72)', 'line': {'width': 0}},
            hovertemplate='<b>%{x|%Y-%m}</b><br>Residual: +%{y:.3f} ' + unit + '<extra></extra>',
        ), row=4, col=1)
        fig.add_trace(go.Bar(
            x=dates, y=resid.clip(upper=0).values,
            name='Residual (−)',
            marker={'color': 'rgba(248,113,113,0.72)', 'line': {'width': 0}},
            hovertemplate='<b>%{x|%Y-%m}</b><br>Residual: %{y:.3f} ' + unit + '<extra></extra>',
        ), row=4, col=1)

        # Strength metrics
        var_resid = float(np.var(res.resid))
        f_trend = max(0.0, 1.0 - var_resid / (np.var(res.resid + res.trend) + 1e-12))
        f_season = max(0.0, 1.0 - var_resid / (np.var(res.resid + res.seasonal) + 1e-12))
        subtitle = (
            f'Trend strength: {f_trend:.2f} · '
            f'Seasonal strength: {f_season:.2f} · '
            f'Period: 12 months · Robust STL'
        )

        label = f"STL Decomposition · {request.feature.replace('_', ' ')} · {request.station.replace('_', ' ')}"
        freq_note = 'resampled to monthly' if frequency == 'daily' else 'monthly'
        fig.update_layout(
            template='plotly_white',
            title={
                'text': f"{label}<br><sup style='color:#64748b;font-size:11px'>{subtitle} · {freq_note}</sup>",
                'x': 0.5, 'xanchor': 'center', 'y': 0.98, 'yanchor': 'top',
                'font': {'size': 14, 'color': '#0f172a', 'family': 'Inter, Arial, sans-serif'},
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(248,250,252,0.6)',
            font={'family': 'Inter, Arial, sans-serif', 'size': 11, 'color': '#334155'},
            margin={'l': 64, 'r': 40, 't': 132, 'b': 60},
            hovermode='x unified',
            hoverlabel={'bgcolor': '#1e293b', 'bordercolor': '#334155', 'font': {'color': '#f1f5f9', 'size': 12}},
            barmode='relative',
            height=680,
            legend={
                'orientation': 'h', 'yanchor': 'bottom', 'y': 1.08,
                'xanchor': 'center', 'x': 0.5,
                'font': {'size': 11, 'color': '#475569'},
                'bgcolor': 'rgba(0,0,0,0)',
            },
        )
        axis_style = {
            'showgrid': True, 'gridcolor': 'rgba(148,163,184,0.18)',
            'linecolor': 'rgba(148,163,184,0.3)', 'zeroline': False,
            'tickfont': {'size': 10, 'color': '#64748b'},
            'title_font': {'size': 11, 'color': '#475569'},
        }
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style)
        fig.update_yaxes(title_text=unit or 'Value', row=1, col=1)
        fig.update_yaxes(title_text=unit or 'Value', row=2, col=1)
        fig.update_yaxes(title_text=unit or 'Value', row=3, col=1)
        fig.update_yaxes(title_text=unit or 'Value', row=4, col=1)
        fig.update_xaxes(title_text='Date', row=4, col=1)
        return fig

    @staticmethod
    def _cusum_breakpoints(values: np.ndarray, n_bkps: int, min_size: int = 5) -> List[int]:
        """Binary segmentation change-point detection via CUSUM (no external deps)."""
        n = len(values)

        def best_split(start: int, end: int):
            if end - start < 2 * min_size:
                return None, -1.0
            seg = values[start:end]
            cusum = np.cumsum(seg - seg.mean())
            inner = cusum[min_size: len(cusum) - min_size]
            if len(inner) == 0:
                return None, -1.0
            best_local = int(np.argmax(np.abs(inner))) + min_size
            return start + best_local, float(np.abs(cusum[best_local]))

        segments = [(0, n)]
        breakpoints: List[int] = []

        while len(breakpoints) < n_bkps:
            best_bp, best_score, best_seg = None, -1.0, None
            for seg in segments:
                bp, score = best_split(*seg)
                if score > best_score:
                    best_score, best_bp, best_seg = score, bp, seg
            if best_bp is None:
                break
            breakpoints.append(best_bp)
            segments.remove(best_seg)
            segments += [(best_seg[0], best_bp), (best_bp, best_seg[1])]

        return sorted(breakpoints)

    def _change_point_detection(self, request: SeriesRequest) -> go.Figure:
        """Detect structural regime shifts using PELT (ruptures) or CUSUM fallback."""
        try:
            import ruptures as rpt
            _has_ruptures = True
        except ImportError:
            _has_ruptures = False

        df = self.repository.get_feature_series(request)
        unit = self.repository.feature_units.get(request.feature, '')
        frequency = self.repository.feature_frequency.get(request.feature, 'daily')

        # Resample daily → monthly for stability
        if frequency == 'daily':
            ts = (
                df.set_index('Timestamp')['Value']
                .resample('MS').mean().dropna().reset_index()
            )
            ts.columns = ['Timestamp', 'Value']
        else:
            ts = df[['Timestamp', 'Value']].dropna().reset_index(drop=True)

        n = len(ts)
        if n < 10:
            raise ValueError(f'Need at least 10 observations for change-point detection (got {n}).')

        values = ts['Value'].values
        dates = ts['Timestamp'].values

        # Adaptive number of breakpoints
        span_years = max(1.0, (ts['Timestamp'].iloc[-1] - ts['Timestamp'].iloc[0]).days / 365.25)
        n_bkps = min(6, max(2, int(span_years / 3)))
        min_size = max(3, n // (2 * (n_bkps + 1)))

        if _has_ruptures:
            signal = values.reshape(-1, 1)
            algo = rpt.Pelt(model='rbf', min_size=min_size, jump=1).fit(signal)
            raw_bkps = algo.predict(pen=2.0)
            breakpoints = [b for b in raw_bkps if 0 < b < n][:n_bkps]
            method_label = 'PELT · RBF cost'
        else:
            breakpoints = self._cusum_breakpoints(values, n_bkps, min_size)
            method_label = 'Binary Segmentation · CUSUM'

        all_bounds = [0] + breakpoints + [n]
        seg_bg_colors = [
            'rgba(56,189,248,0.07)', 'rgba(74,222,128,0.07)',
            'rgba(167,139,250,0.07)', 'rgba(251,191,36,0.07)',
            'rgba(248,113,113,0.07)', 'rgba(45,212,191,0.07)',
            'rgba(251,146,60,0.07)',
        ]
        segment_means: List[float] = []
        segment_labels: List[str] = []
        segment_midpoints: List[pd.Timestamp] = []
        segment_durations: List[int] = []
        segment_widths: List[float] = []

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.8, 0.2],
            vertical_spacing=0.05,
        )

        for i in range(len(all_bounds) - 1):
            si, ei = all_bounds[i], all_bounds[i + 1]
            seg_vals = values[si:ei]
            seg_dates = dates[si:ei]
            if len(seg_vals) == 0:
                continue
            seg_mean = float(seg_vals.mean())
            d0, d1 = pd.Timestamp(seg_dates[0]), pd.Timestamp(seg_dates[-1])
            segment_means.append(seg_mean)
            segment_labels.append(f'R{i + 1}')
            segment_midpoints.append(d0 + (d1 - d0) / 2)
            segment_durations.append(len(seg_vals))
            seg_span = max(
                pd.Timedelta(days=18) if frequency == 'monthly' else pd.Timedelta(days=45),
                d1 - d0 if d1 > d0 else pd.Timedelta(days=18 if frequency == 'monthly' else 45),
            )
            segment_widths.append(seg_span / pd.Timedelta(milliseconds=1))

            fig.add_vrect(
                x0=d0,
                x1=d1,
                fillcolor=seg_bg_colors[i % len(seg_bg_colors)],
                line_width=0,
                layer='below',
                row=1,
                col=1,
            )
            fig.add_vrect(
                x0=d0,
                x1=d1,
                fillcolor=seg_bg_colors[i % len(seg_bg_colors)],
                line_width=0,
                layer='below',
            )
            fig.add_trace(
                go.Scatter(
                    x=[d0, d1],
                    y=[seg_mean, seg_mean],
                    mode='lines',
                    name='Regime mean',
                    showlegend=(i == 0),
                    line={'width': 2.4, 'color': '#f59e0b', 'dash': 'dash'},
                    hovertemplate=(
                        f'<b>Regime {i + 1}</b><br>'
                        f'Mean: {seg_mean:.3f} {unit}<br>'
                        f'Duration: {len(seg_vals)} points<extra></extra>'
                    ),
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=ts['Timestamp'],
                y=ts['Value'],
                mode='lines',
                name=request.feature.replace('_', ' '),
                line={'width': 2.6, 'color': '#38bdf8'},
                hovertemplate='<b>%{x|%Y-%m}</b><br>Value: %{y:.3f} ' + unit + '<extra></extra>',
            ),
            row=1,
            col=1,
        )

        rolling_window = max(3, min(12, n // 8))
        smoothed = pd.Series(values).rolling(window=rolling_window, center=True, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=ts['Timestamp'],
                y=smoothed,
                mode='lines',
                name='Smoothed context',
                showlegend=False,
                line={'width': 1.8, 'color': '#cbd5e1'},
                opacity=0.75,
                hovertemplate='<b>%{x|%Y-%m}</b><br>Smoothed: %{y:.3f} ' + unit + '<extra></extra>',
            ),
            row=1,
            col=1,
        )

        # Change-point vertical lines
        for bp in breakpoints:
            if 0 < bp < n:
                bp_date = pd.Timestamp(dates[bp])
                fig.add_vline(
                    x=bp_date,
                    line_dash='dash', line_color='rgba(239,68,68,0.8)', line_width=2,
                    row=1,
                    col=1,
                )
                fig.add_annotation(
                    x=bp_date, yref='paper', y=1.015,
                    text='▼', showarrow=False,
                    font={'size': 12, 'color': '#ef4444'},
                    xanchor='center',
                )

        fig.add_trace(
            go.Bar(
                x=segment_midpoints,
                y=segment_means,
                name='Regime mean level',
                showlegend=False,
                marker={
                    'color': segment_means,
                    'colorscale': [[0, '#38bdf8'], [0.5, '#8b5cf6'], [1, '#f59e0b']],
                    'line': {'color': 'rgba(15,23,42,0.85)', 'width': 1},
                },
                width=segment_widths,
                customdata=np.column_stack((segment_labels, segment_durations)),
                hovertemplate=(
                    '<b>%{customdata[0]}</b><br>'
                    'Mean: %{y:.3f} ' + unit + '<br>'
                    'Duration: %{customdata[1]} points<extra></extra>'
                ),
            ),
            row=2,
            col=1,
        )

        self._base_layout(
            fig,
            f"Change-Point Detection · {request.feature.replace('_', ' ')} · {request.station.replace('_', ' ')}",
        )
        overall_mean = float(np.mean(values))
        overall_std = float(np.std(values))
        largest_shift = (
            max(abs(segment_means[i] - segment_means[i - 1]) for i in range(1, len(segment_means)))
            if len(segment_means) > 1 else 0.0
        )
        fig.update_layout(
            margin={'l': 72, 'r': 42, 't': 78, 'b': 82},
            height=500,
            legend={
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.03,
                'xanchor': 'left',
                'x': 0,
            },
            annotations=list(fig.layout.annotations) + [
                go.layout.Annotation(
                    xref='paper',
                    yref='paper',
                    x=0.995,
                    y=1.12,
                    xanchor='right',
                    yanchor='top',
                    align='left',
                    showarrow=False,
                    bgcolor='rgba(15,23,42,0.82)',
                    bordercolor='rgba(148,163,184,0.26)',
                    borderwidth=1,
                    borderpad=6,
                    font={'size': 9, 'color': '#e2e8f0'},
                    text=(
                        f"<b>{method_label}</b><br>"
                        f"Breakpoints: {len(breakpoints)}<br>"
                        f"Mean: {overall_mean:.2f} {unit}<br>"
                        f"Shift: {largest_shift:.2f} {unit}"
                    ),
                )
            ],
        )
        fig.update_xaxes(title='Date', row=2, col=1)
        fig.update_yaxes(
            title=f"{request.feature.replace('_', ' ')} ({unit})" if unit else request.feature.replace('_', ' '),
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title='Regime',
            row=2,
            col=1,
        )
        return fig

    def _wavelet_analysis(self, request: SeriesRequest) -> go.Figure:
        """Continuous Wavelet Transform scalogram using Morlet wavelet."""
        try:
            import pywt
            _has_pywt = True
        except ImportError:
            _has_pywt = False

        df = self.repository.get_feature_series(request)
        unit = self.repository.feature_units.get(request.feature, '')
        frequency = self.repository.feature_frequency.get(request.feature, 'daily')

        # Always resample to monthly for speed and a natural period axis
        ts = (
            df.set_index('Timestamp')['Value']
            .resample('MS').mean()
            .interpolate(method='linear')
            .dropna()
        )
        if len(ts) < 16:
            raise ValueError(
                f'Need at least 16 monthly observations for wavelet analysis (got {len(ts)}).'
            )

        dates = ts.index
        n = len(ts)
        dt = 1.0  # 1 month per step

        # Normalise signal (zero-mean, unit variance)
        signal = (ts.values - ts.values.mean()) / (ts.values.std() + 1e-10)

        # Scale range: 2 → min(n//2, 120) months
        scales = np.arange(2, min(n // 2, 121))

        if _has_pywt:
            coef, freqs = pywt.cwt(signal, scales, 'cmor1.5-1.0', sampling_period=dt)
            power = np.abs(coef) ** 2
            periods = (1.0 / (freqs + 1e-12)).astype(float)
        else:
            # Pure-numpy FFT-based Morlet CWT
            omega0 = 6.0
            x_fft = np.fft.fft(signal, n=n)
            ang_freqs = 2.0 * np.pi * np.fft.fftfreq(n, d=dt)
            power = np.zeros((len(scales), n))
            for i, s in enumerate(scales):
                psi_fft = (np.pi ** -0.25) * np.exp(-0.5 * (s * ang_freqs - omega0) ** 2)
                psi_fft[ang_freqs < 0] = 0.0
                W = np.fft.ifft(x_fft * psi_fft * np.sqrt(2.0 * np.pi * s))
                power[i, :] = np.abs(W) ** 2
            periods = scales.astype(float)

        # Subsample periods for display (cap at 80 rows to keep heatmap readable)
        max_rows = 80
        if len(periods) > max_rows:
            idx = np.round(np.linspace(0, len(periods) - 1, max_rows)).astype(int)
            periods_plot = periods[idx]
            power_plot = power[idx, :]
        else:
            periods_plot = periods
            power_plot = power

        # Human-readable period labels
        def _period_label(p: float) -> str:
            p = int(round(p))
            if p < 12:
                return f'{p}m'
            elif p == 12:
                return '1 yr'
            elif p % 12 == 0:
                return f'{p // 12} yr'
            else:
                return f'{p // 12}yr {p % 12}m'

        period_labels = [_period_label(p) for p in periods_plot]
        date_strs = [str(d)[:7] for d in dates]

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=['Original Signal (monthly)', 'Wavelet Power Spectrum · Morlet CWT'],
            vertical_spacing=0.08,
            row_heights=[0.28, 0.72],
        )

        # Original signal
        fig.add_trace(go.Scatter(
            x=dates, y=ts.values,
            mode='lines', name='Original',
            line={'width': 1.8, 'color': '#38bdf8'},
            hovertemplate='<b>%{x|%Y-%m}</b><br>Value: %{y:.3f} ' + unit + '<extra></extra>',
        ), row=1, col=1)

        # Scalogram
        fig.add_trace(go.Heatmap(
            x=date_strs,
            y=period_labels,
            z=power_plot.tolist(),
            colorscale='Plasma',
            colorbar={
                'title': {'text': 'Power', 'side': 'right', 'font': {'size': 10}},
                'thickness': 14,
                'len': 0.66,
                'y': 0.33,
                'yanchor': 'middle',
                'tickfont': {'size': 9},
            },
            hovertemplate='<b>%{x}</b><br>Period: %{y}<br>Power: %{z:.4f}<extra></extra>',
        ), row=2, col=1)

        # Annotate reference periods on scalogram via shape + annotation
        for ref_p, ref_label in [(12, '1 yr'), (24, '2 yr'), (60, '5 yr')]:
            nearest_label = _period_label(ref_p)
            if nearest_label in period_labels:
                fig.add_shape(
                    type='line',
                    xref='paper', x0=0, x1=1,
                    yref='y2', y0=nearest_label, y1=nearest_label,
                    line={'dash': 'dot', 'color': 'rgba(255,255,255,0.45)', 'width': 1},
                )
                fig.add_annotation(
                    xref='paper', x=1.01,
                    yref='y2', y=nearest_label,
                    text=ref_label,
                    showarrow=False,
                    font={'size': 9, 'color': 'rgba(200,200,200,0.85)'},
                    xanchor='left',
                )

        wavelet_lib = 'pywt · cmor1.5-1.0' if _has_pywt else 'NumPy FFT · Morlet'
        freq_note = 'resampled to monthly' if frequency == 'daily' else 'monthly'

        fig.update_layout(
            template='plotly_white',
            title={
                'text': (
                    f"Wavelet Analysis · {request.feature.replace('_', ' ')} · "
                    f"{request.station.replace('_', ' ')}"
                    f"<br><sup style='color:#64748b;font-size:10px'>"
                    f"{wavelet_lib} · {freq_note}</sup>"
                ),
                'x': 0.5, 'xanchor': 'center', 'y': 0.98, 'yanchor': 'top',
                'font': {'size': 14, 'color': '#0f172a', 'family': 'Inter, Arial, sans-serif'},
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(248,250,252,0.6)',
            font={'family': 'Inter, Arial, sans-serif', 'size': 11, 'color': '#334155'},
            margin={'l': 64, 'r': 80, 't': 90, 'b': 60},
            hovermode='x unified',
            hoverlabel={'bgcolor': '#1e293b', 'bordercolor': '#334155', 'font': {'color': '#f1f5f9', 'size': 12}},
            height=600,
            legend={
                'orientation': 'h', 'yanchor': 'top', 'y': -0.06,
                'xanchor': 'center', 'x': 0.5,
                'font': {'size': 11, 'color': '#475569'},
            },
        )
        fig.update_xaxes(
            showgrid=True, gridcolor='rgba(148,163,184,0.18)',
            tickfont={'size': 10, 'color': '#64748b'},
        )
        fig.update_yaxes(
            showgrid=False,
            tickfont={'size': 10, 'color': '#64748b'},
            title_font={'size': 11, 'color': '#475569'},
        )
        fig.update_yaxes(title_text=unit or 'Value', row=1, col=1)
        fig.update_yaxes(title_text='Period', autorange='reversed', row=2, col=1)
        fig.update_xaxes(title_text='Date', row=2, col=1)
        return fig

    def _granger_causality(self, requests: List[SeriesRequest]) -> go.Figure:
        """Test Granger causality between two features — both directions."""
        from statsmodels.tsa.stattools import grangercausalitytests

        station = requests[0].station
        feat_x, feat_y = requests[0].feature, requests[1].feature
        label_x = feat_x.replace('_', ' ')
        label_y = feat_y.replace('_', ' ')

        df_x = self.repository.get_feature_series(requests[0])
        df_y = self.repository.get_feature_series(requests[1])
        frequency = self.repository.feature_frequency.get(feat_x, 'daily')

        merged = pd.merge(
            df_x[['Timestamp', 'Value']].rename(columns={'Value': feat_x}),
            df_y[['Timestamp', 'Value']].rename(columns={'Value': feat_y}),
            on='Timestamp', how='inner',
        ).sort_values('Timestamp').dropna()

        # For daily data resample to monthly to keep test well-powered
        if frequency == 'daily':
            merged = (
                merged.set_index('Timestamp')
                .resample('MS').mean()
                .dropna()
                .reset_index()
            )

        n = len(merged)
        if n < 30:
            raise ValueError(
                f'Need at least 30 overlapping observations for Granger causality (got {n}).'
            )

        max_lag = min(12, n // 5)
        lags = list(range(1, max_lag + 1))

        # X → Y
        data_xy = merged[[feat_y, feat_x]].values
        res_xy = grangercausalitytests(data_xy, maxlag=max_lag, verbose=False)
        # Y → X
        data_yx = merged[[feat_x, feat_y]].values
        res_yx = grangercausalitytests(data_yx, maxlag=max_lag, verbose=False)

        def _extract(res, lag_list):
            f_stats, p_vals = [], []
            for lag in lag_list:
                f_stat, p_val, _, _ = res[lag][0]['ssr_ftest']
                f_stats.append(float(f_stat))
                p_vals.append(float(p_val))
            return f_stats, p_vals

        f_xy, p_xy = _extract(res_xy, lags)
        f_yx, p_yx = _extract(res_yx, lags)

        def _sig_color(p: float) -> str:
            if p < 0.01:
                return 'rgba(56,189,248,0.88)'   # sky blue — highly significant
            if p < 0.05:
                return 'rgba(74,222,128,0.88)'   # green — significant
            if p < 0.10:
                return 'rgba(251,191,36,0.88)'   # amber — marginal
            return 'rgba(148,163,184,0.45)'      # grey — not significant

        def _sig_stars(p: float) -> str:
            if p < 0.01:
                return '★★★'
            if p < 0.05:
                return '★★'
            if p < 0.10:
                return '★'
            return 'n.s.'

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[
                f'{label_x}  →  {label_y}',
                f'{label_y}  →  {label_x}',
            ],
            vertical_spacing=0.20,
            shared_xaxes=True,
        )

        for row, (f_vals, p_vals, name) in enumerate(
            [(f_xy, p_xy, f'{label_x} → {label_y}'),
             (f_yx, p_yx, f'{label_y} → {label_x}')],
            start=1,
        ):
            colors = [_sig_color(p) for p in p_vals]
            custom = [[p, _sig_stars(p)] for p in p_vals]
            fig.add_trace(go.Bar(
                x=lags, y=f_vals,
                name=name,
                marker={'color': colors, 'line': {'width': 0}},
                customdata=custom,
                hovertemplate=(
                    'Lag %{x}<br>'
                    'F-stat: %{y:.3f}<br>'
                    'p-value: %{customdata[0]:.4f}  %{customdata[1]}'
                    '<extra></extra>'
                ),
            ), row=row, col=1)
            # p=0.05 approximate F threshold (F ≈ 3.84 for large n)
            f_thresh = 3.84
            fig.add_hline(
                y=f_thresh, row=row, col=1,
                line_dash='dash', line_color='rgba(239,68,68,0.55)', line_width=1.5,
            )
            fig.add_annotation(
                xref='paper', x=1.01,
                yref=f'y{row}' if row > 1 else 'y',
                y=f_thresh,
                text='p=0.05',
                showarrow=False,
                font={'size': 9, 'color': '#ef4444'},
                xanchor='left',
            )

        fig.update_layout(
            template='plotly_white',
            title={
                'text': (
                    f"Granger Causality · {label_x} ↔ {label_y} · "
                    f"{station.replace('_', ' ')}"
                ),
                'x': 0.5, 'xanchor': 'center', 'y': 0.98, 'yanchor': 'top',
                'font': {'size': 14, 'color': '#0f172a', 'family': 'Inter, Arial, sans-serif'},
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(248,250,252,0.6)',
            font={'family': 'Inter, Arial, sans-serif', 'size': 11, 'color': '#334155'},
            margin={'l': 64, 'r': 64, 't': 132, 'b': 92},
            hovermode='x unified',
            hoverlabel={'bgcolor': '#1e293b', 'bordercolor': '#334155', 'font': {'color': '#f1f5f9', 'size': 12}},
            height=560,
            barmode='group',
            legend={
                'orientation': 'h', 'yanchor': 'bottom', 'y': 1.16,
                'xanchor': 'center', 'x': 0.5,
                'font': {'size': 11, 'color': '#475569'},
            },
        )
        fig.add_annotation(
            xref='paper', yref='paper', x=0.5, y=-0.20,
            text=(
                '<span style="color:rgba(56,189,248,0.88)">■</span> ★★★ p&lt;0.01 &nbsp;&nbsp;'
                '<span style="color:rgba(74,222,128,0.88)">■</span> ★★ p&lt;0.05 &nbsp;&nbsp;'
                '<span style="color:rgba(251,191,36,0.88)">■</span> ★ p&lt;0.1 &nbsp;&nbsp;'
                '<span style="color:rgba(148,163,184,0.45)">■</span> n.s.'
            ),
            showarrow=False,
            font={'size': 10, 'color': '#64748b'},
            xanchor='center',
        )
        axis_style = {
            'showgrid': True, 'gridcolor': 'rgba(148,163,184,0.18)',
            'tickfont': {'size': 11, 'color': '#64748b'},
            'title_font': {'size': 11, 'color': '#475569'},
        }
        fig.update_xaxes(**axis_style, title_text='Lag', tickmode='array', tickvals=lags, row=2, col=1)
        fig.update_xaxes(**axis_style, tickmode='array', tickvals=lags, row=1, col=1)
        fig.update_yaxes(**axis_style, title_text='F-statistic')
        lag_unit = 'months' if frequency == 'monthly' else 'months (resampled)'
        fig.add_annotation(
            xref='paper', yref='paper', x=0.5, y=1.16,
            text=f'n={n} observations · lags 1–{max_lag} {lag_unit}',
            showarrow=False,
            font={'size': 10, 'color': '#94a3b8'},
            xanchor='center',
        )
        return fig

    def _cross_correlation_function(self, requests: List[SeriesRequest]) -> go.Figure:
        """Cross-Correlation Function (CCF) with 95% confidence bands and peak annotation."""
        req_a, req_b = requests[0], requests[1]
        df_a = self.repository.get_feature_series(req_a)
        df_b = self.repository.get_feature_series(req_b)

        label_a = f"{req_a.station.replace('_', ' ')} · {req_a.feature.replace('_', ' ')}"
        label_b = f"{req_b.station.replace('_', ' ')} · {req_b.feature.replace('_', ' ')}"
        frequency = self.repository.feature_frequency.get(req_a.feature, 'daily')

        merged = pd.merge(
            df_a[['Timestamp', 'Value']].rename(columns={'Value': 'A'}),
            df_b[['Timestamp', 'Value']].rename(columns={'Value': 'B'}),
            on='Timestamp', how='inner',
        ).sort_values('Timestamp').dropna()

        # Resample daily → monthly so lags are interpretable
        if frequency == 'daily':
            merged = (
                merged.set_index('Timestamp')
                .resample('MS').mean()
                .dropna()
                .reset_index()
            )

        n = len(merged)
        if n < 20:
            raise ValueError(
                f'Need at least 20 overlapping observations for CCF (got {n}).'
            )

        max_lag = min(30, n // 4)
        lags = list(range(-max_lag, max_lag + 1))

        # Standardise
        a = ((merged['A'] - merged['A'].mean()) / (merged['A'].std() + 1e-10)).values
        b = ((merged['B'] - merged['B'].mean()) / (merged['B'].std() + 1e-10)).values

        # Compute CCF at each lag
        ccf_vals = []
        for lag in lags:
            if lag >= 0:
                x, y = a[: n - lag], b[lag:]
            else:
                shift = -lag
                x, y = a[shift:], b[: n - shift]
            if len(x) > 1:
                val = float(np.corrcoef(x, y)[0, 1])
                ccf_vals.append(0.0 if np.isnan(val) else val)
            else:
                ccf_vals.append(0.0)

        ci = 1.96 / np.sqrt(n)

        # Find peak (ignoring lag-0 to avoid trivial self-correlation when same feature)
        abs_ccf = np.abs(ccf_vals)
        peak_idx = int(np.argmax(abs_ccf))
        peak_lag = lags[peak_idx]
        peak_corr = ccf_vals[peak_idx]

        lag_unit = 'months'
        if peak_lag > 0:
            direction = f"{label_a.split('·')[1].strip()} leads {label_b.split('·')[1].strip()} by {peak_lag} {lag_unit}"
        elif peak_lag < 0:
            direction = f"{label_b.split('·')[1].strip()} leads {label_a.split('·')[1].strip()} by {abs(peak_lag)} {lag_unit}"
        else:
            direction = 'Simultaneous — peak at lag 0'

        # Bar colors
        colors = []
        for i, v in enumerate(ccf_vals):
            if i == peak_idx:
                colors.append('#f59e0b')           # amber — peak
            elif abs(v) > ci:
                colors.append('rgba(56,189,248,0.8)' if v >= 0 else 'rgba(248,113,113,0.8)')
            else:
                colors.append('rgba(148,163,184,0.4)')

        fig = go.Figure()

        # CI band
        fig.add_hrect(
            y0=-ci, y1=ci,
            fillcolor='rgba(148,163,184,0.1)', line_width=0,
        )
        fig.add_hline(y=ci, line_dash='dot', line_color='rgba(148,163,184,0.55)', line_width=1)
        fig.add_hline(y=-ci, line_dash='dot', line_color='rgba(148,163,184,0.55)', line_width=1)
        fig.add_hline(y=0, line_color='rgba(100,116,139,0.4)', line_width=1)

        # CCF bars
        fig.add_trace(go.Bar(
            x=lags, y=ccf_vals,
            name='CCF',
            marker={'color': colors, 'line': {'width': 0}},
            hovertemplate='Lag %{x}<br>r = %{y:.4f}<extra></extra>',
        ))

        # Peak marker
        fig.add_trace(go.Scatter(
            x=[peak_lag], y=[peak_corr],
            mode='markers+text',
            name=f'Peak r={peak_corr:.3f} @ lag {peak_lag}',
            marker={'color': '#f59e0b', 'size': 14, 'symbol': 'star',
                    'line': {'width': 1.5, 'color': 'white'}},
            text=[f'  r={peak_corr:.3f}'],
            textposition='middle right',
            textfont={'size': 10, 'color': '#f59e0b'},
            hovertemplate=f'Peak: r={peak_corr:.3f} at lag {peak_lag}<extra></extra>',
        ))

        title = f"Cross-Correlation · {label_a} vs {label_b}"
        self._base_layout(fig, title)
        fig.update_layout(
            hovermode='x unified',
            height=440,
            margin={'l': 64, 'r': 64, 't': 126, 'b': 110},
            legend={
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.18,
                'xanchor': 'center',
                'x': 0.5,
                'font': {'size': 11, 'color': '#475569'},
                'bgcolor': 'rgba(0,0,0,0)',
                'borderwidth': 0,
            },
        )
        fig.add_annotation(
            xref='paper', yref='paper', x=0.5, y=1.14,
            text=f'↑ {direction}  ·  95% CI ±{ci:.3f}  ·  n={n}',
            showarrow=False,
            font={'size': 11, 'color': '#f59e0b', 'family': 'Inter, Arial, sans-serif'},
            xanchor='center',
        )
        fig.add_annotation(
            xref='paper', yref='paper', x=0.99, y=0.99,
            text=f'95% CI band',
            showarrow=False,
            font={'size': 9, 'color': '#94a3b8'},
            xanchor='right', yanchor='top',
        )
        lag_label = f'Lag ({lag_unit})'
        if frequency == 'daily':
            lag_label += ' — resampled to monthly'
        fig.update_xaxes(
            title=lag_label,
            zeroline=True, zerolinewidth=1.5,
            zerolinecolor='rgba(148,163,184,0.5)',
        )
        fig.update_yaxes(title='Pearson r', range=[-1.1, 1.1])
        return fig
