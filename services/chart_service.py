from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

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
    'Station Ranking Bar Chart',
    'Rolling Correlation Chart',
    'Exceedance Probability Curve',
}


class ChartService:
    def __init__(self, repository: DataRepository) -> None:
        self.repository = repository
        self.palette = px.colors.qualitative.Plotly

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
        if graph_type == 'Station Ranking Bar Chart' and len(features) != 1:
            raise ValueError('Station Ranking Bar Chart requires one feature across multiple stations.')
        if graph_type == 'Rolling Correlation Chart':
            if len(stations) != 1:
                raise ValueError('Rolling Correlation Chart requires all series from the same station.')
            if len(features) != 2:
                raise ValueError('Rolling Correlation Chart requires exactly 2 different features.')
        if graph_type == 'Exceedance Probability Curve' and len(requests) != 1:
            raise ValueError('Exceedance Probability Curve requires exactly one selection row.')

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
        elif graph_type == 'Station Ranking Bar Chart':
            figure = self._station_ranking_bar_chart(requests)
        elif graph_type == 'Rolling Correlation Chart':
            figure = self._rolling_correlation_chart(requests)
        elif graph_type == 'Exceedance Probability Curve':
            figure = self._exceedance_probability_curve(requests[0])
        else:
            raise ValueError('Unsupported graph type.')

        series_payload = [asdict(req) for req in requests]
        figure_json = plotly.io.to_json(figure)
        return {
            'graph_type': graph_type,
            'series': series_payload,
            'figure': figure_json,
            'title': figure.layout.title.text if figure.layout.title else graph_type,
        }

    def _base_layout(self, figure: go.Figure, title: str) -> None:
        figure.update_layout(
            template='plotly_white',
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'y': 0.98,
                'yanchor': 'top',
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#ffffff',
            font={'family': 'Inter, Arial, sans-serif', 'size': 12, 'color': '#0f172a'},
            margin={'l': 60, 'r': 40, 't': 80, 'b': 150},
            hovermode='x unified',
            legend={
                'orientation': 'h',
                'yanchor': 'top',
                'y': -0.35,
                'xanchor': 'center',
                'x': 0.5,
                'font': {'size': 11},
            },
        )
        figure.update_xaxes(showgrid=True, gridcolor='rgba(148,163,184,0.22)', linecolor='rgba(15,23,42,0.18)')
        figure.update_yaxes(showgrid=True, gridcolor='rgba(148,163,184,0.22)', linecolor='rgba(15,23,42,0.18)')

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
                line={'width': 2.4, 'color': '#2563eb'},
                fill='tozeroy',
                fillcolor='rgba(37,99,235,0.18)',
                customdata=df[['Imputed']].to_numpy(),
                hovertemplate='Date: %{x|%Y-%m-%d}<br>Value: %{y:.3f} ' + unit + '<br>Imputed: %{customdata[0]}<extra></extra>',
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
        fig = go.Figure(
            data=[
                go.Bar(
                    x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    y=totals.values,
                    text=[f'{value:.2f}' for value in totals.values],
                    textposition='auto',
                    marker={'color': '#0ea5e9'},
                )
            ]
        )
        self._base_layout(fig, f"Annual Monthly Totals · {request.feature.replace('_', ' ')} · {request.station.replace('_', ' ')}")
        fig.update_xaxes(title='Month')
        fig.update_yaxes(title=f"{request.feature.replace('_', ' ')} ({unit})" if unit else request.feature.replace('_', ' '))
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
                line={'width': 2.4, 'color': '#2563eb'},
                fill='tozeroy',
                fillcolor='rgba(37,99,235,0.10)',
                hovertemplate='Exceedance: %{x:.1f}%<br>Value: %{y:.3f} ' + unit + '<extra></extra>',
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

        fig = go.Figure()
        for month_num in range(1, 13):
            month_data = df[df['Month'] == month_num]['Value']
            fig.add_trace(
                go.Box(
                    y=month_data,
                    name=month_labels[month_num - 1],
                    marker_color=self.palette[(month_num - 1) % len(self.palette)],
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
                colorscale='Blues',
                colorbar={
                    'title': {'text': f'{unit}', 'side': 'right'},
                    'thickness': 15,
                    'len': 0.9,
                },
                hovertemplate='Station: %{y}<br>Period: %{x}<br>Value: %{z:.2f} ' + unit + '<extra></extra>',
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
                marker_color='rgba(56, 189, 248, 0.8)',
                hovertemplate='Month: %{x|%Y-%m}<br>Anomaly: +%{y:.2f} ' + unit + '<extra></extra>',
            )
        )
        fig.add_trace(
            go.Bar(
                x=monthly['Timestamp'],
                y=neg,
                name='Below average',
                marker_color='rgba(239, 68, 68, 0.8)',
                hovertemplate='Month: %{x|%Y-%m}<br>Anomaly: %{y:.2f} ' + unit + '<extra></extra>',
            )
        )
        fig.add_trace(
            go.Scatter(
                x=monthly['Timestamp'],
                y=monthly['Climatology'],
                mode='lines',
                name='Climatological mean',
                line={'width': 1.5, 'color': '#f59e0b', 'dash': 'dot'},
                hovertemplate='Month: %{x|%Y-%m}<br>Mean: %{y:.2f} ' + unit + '<extra></extra>',
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

        fig = make_subplots(
            rows=2, cols=6,
            subplot_titles=month_labels,
            shared_yaxes=True,
            horizontal_spacing=0.04,
            vertical_spacing=0.18,
        )
        for i, month_num in enumerate(range(1, 13)):
            row = 1 if i < 6 else 2
            col = (i % 6) + 1
            month_data = monthly[monthly['Month'] == month_num].sort_values('Year')
            if month_data.empty:
                continue
            x_vals = month_data['Year'].values
            y_vals = month_data['Value'].values
            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='lines+markers',
                    name=month_labels[i],
                    line={'width': 1.8, 'color': '#2563eb'},
                    marker={'size': 3},
                    showlegend=False,
                    hovertemplate=f'{month_labels[i]} %{{x}}: %{{y:.2f}} {unit}<extra></extra>',
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
            title={
                'text': f"Seasonal Subseries · {request.feature.replace('_', ' ')} · {request.station.replace('_', ' ')}",
                'x': 0.5, 'xanchor': 'center',
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#ffffff',
            font={'family': 'Inter, Arial, sans-serif', 'size': 11, 'color': '#0f172a'},
            margin={'l': 50, 'r': 20, 't': 90, 'b': 50},
            showlegend=False,
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(148,163,184,0.22)', tickformat='d')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(148,163,184,0.22)')
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
                colorscale='Blues',
                colorbar={'title': {'text': unit, 'side': 'right'}, 'thickness': 15, 'len': 0.9},
                hovertemplate='%{y} · %{x}<br>Value: %{z:.3f} ' + unit + '<extra></extra>',
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
        maxes = [r['max'] for r in rows]
        ranks = [f'#{i+1}' for i in range(len(rows))]

        # Color gradient: top rank = deepest blue, lower = lighter
        n = len(rows)
        colors = [f'rgba(37,99,235,{0.9 - 0.5 * i / max(n - 1, 1):.2f})' for i in range(n)]

        fig = go.Figure()

        # Min–Max range lines (thin, faint)
        for i, r in enumerate(rows):
            fig.add_trace(go.Scatter(
                x=[r['min'], r['max']],
                y=[r['station'], r['station']],
                mode='lines',
                line={'color': 'rgba(148,163,184,0.35)', 'width': 6},
                showlegend=False,
                hoverinfo='skip',
            ))

        # Lollipop stems: 0 → mean
        for i, r in enumerate(rows):
            fig.add_trace(go.Scatter(
                x=[0, r['mean']],
                y=[r['station'], r['station']],
                mode='lines',
                line={'color': colors[i], 'width': 3},
                showlegend=False,
                hoverinfo='skip',
            ))

        # Mean dot
        fig.add_trace(go.Scatter(
            y=stations, x=means,
            mode='markers+text',
            name='Mean',
            marker={'color': colors, 'size': 14, 'line': {'width': 2, 'color': 'white'}},
            text=[f'{v:.1f}' for v in means],
            textposition='middle right',
            textfont={'size': 10, 'color': '#94a3b8'},
            hovertemplate='%{y}<br>Mean: %{x:.3f} ' + unit + '<extra></extra>',
        ))

        # Median tick
        fig.add_trace(go.Scatter(
            y=stations, x=medians,
            mode='markers',
            name='Median',
            marker={'color': '#f59e0b', 'size': 10, 'symbol': 'line-ns-open',
                    'line': {'width': 3, 'color': '#f59e0b'}},
            hovertemplate='%{y}<br>Median: %{x:.3f} ' + unit + '<extra></extra>',
        ))

        # Max diamond
        fig.add_trace(go.Scatter(
            y=stations, x=maxes,
            mode='markers',
            name='Max',
            marker={'color': '#ef4444', 'size': 9, 'symbol': 'diamond',
                    'line': {'width': 1, 'color': 'white'}},
            hovertemplate='%{y}<br>Max: %{x:.3f} ' + unit + '<extra></extra>',
        ))

        # Rank labels on y-axis via annotations
        annotations = []
        for i, (station, rank) in enumerate(zip(stations, ranks)):
            annotations.append({
                'x': 0, 'y': station,
                'xref': 'x', 'yref': 'y',
                'text': rank,
                'showarrow': False,
                'xanchor': 'right',
                'xshift': -8,
                'font': {'size': 11, 'color': '#64748b'},
            })

        self._base_layout(fig, f"Station Ranking · {feature.replace('_', ' ')}")
        fig.update_layout(
            hovermode='closest',
            height=max(380, n * 55 + 180),
            margin={'l': 140, 'r': 80, 't': 80, 'b': 70},
            annotations=annotations,
            legend={'orientation': 'h', 'yanchor': 'bottom', 'y': -0.15,
                    'xanchor': 'center', 'x': 0.5},
            bargap=0.4,
        )
        x_title = f"{feature.replace('_', ' ')} ({unit})" if unit else feature.replace('_', ' ')
        fig.update_xaxes(title=x_title, zeroline=True, zerolinewidth=1, zerolinecolor='rgba(148,163,184,0.4)')
        fig.update_yaxes(title='', categoryorder='array', categoryarray=list(reversed(stations)))
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

        fig = go.Figure()
        fig.add_hrect(y0=0.7, y1=1.05, fillcolor='rgba(37,99,235,0.06)', line_width=0)
        fig.add_hrect(y0=-1.05, y1=-0.7, fillcolor='rgba(239,68,68,0.06)', line_width=0)
        fig.add_hline(y=0, line_dash='dash', line_color='rgba(100,100,100,0.35)', line_width=1)
        fig.add_hline(y=0.7, line_dash='dot', line_color='rgba(37,99,235,0.35)', line_width=1,
                      annotation_text='r = 0.7', annotation_font_size=9, annotation_font_color='#2563eb')
        fig.add_hline(y=-0.7, line_dash='dot', line_color='rgba(239,68,68,0.35)', line_width=1,
                      annotation_text='r = −0.7', annotation_font_size=9, annotation_font_color='#ef4444')
        fig.add_trace(
            go.Scatter(
                x=merged['Timestamp'],
                y=merged['RollingCorr'],
                mode='lines',
                name=f'{window_label} rolling r',
                line={'width': 2.2, 'color': '#6366f1'},
                hovertemplate='Date: %{x|%Y-%m-%d}<br>r = %{y:.3f}<extra></extra>',
            )
        )
        self._base_layout(fig, f"Rolling Correlation · {feature_x.replace('_', ' ')} vs {feature_y.replace('_', ' ')} · {station.replace('_', ' ')}")
        fig.update_xaxes(title='Date')
        fig.update_yaxes(title=f'Pearson r ({window_label} window)', range=[-1.05, 1.05])
        return fig

    def _exceedance_probability_curve(self, request: SeriesRequest) -> go.Figure:
        df = self.repository.get_feature_series(request)
        unit = self.repository.feature_units.get(request.feature, '')

        values = df['Value'].dropna().sort_values(ascending=False).reset_index(drop=True)
        n = len(values)
        # Weibull plotting positions
        rank = np.arange(1, n + 1)
        exceedance_pct = rank / (n + 1) * 100  # percent

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=exceedance_pct,
                y=values.values,
                mode='lines',
                name=f"{request.station.replace('_', ' ')} · {request.feature.replace('_', ' ')}",
                line={'width': 2.4, 'color': '#2563eb'},
                fill='tozeroy',
                fillcolor='rgba(37,99,235,0.10)',
                customdata=100 / exceedance_pct,
                hovertemplate='Exceedance: %{x:.2f}%<br>Return period: %{customdata:.1f} yrs<br>Value: %{y:.3f} ' + unit + '<extra></extra>',
            )
        )

        mean_val = float(values.mean())
        fig.add_hline(
            y=mean_val,
            line_dash='dash', line_color='#f59e0b', line_width=1.5,
            annotation_text=f'Mean: {mean_val:.2f} {unit}',
            annotation_position='top right',
            annotation_font_color='#f59e0b',
            annotation_font_size=10,
        )

        # Return period reference lines
        for rp in [2, 5, 10, 25, 50, 100]:
            prob = 100 / rp
            if prob < exceedance_pct.min() or prob > exceedance_pct.max():
                continue
            fig.add_vline(
                x=prob,
                line_dash='dot', line_color='rgba(239,68,68,0.45)', line_width=1,
                annotation_text=f'{rp}yr',
                annotation_position='top',
                annotation_font_size=9,
                annotation_font_color='#ef4444',
            )

        self._base_layout(fig, f"Exceedance Probability · {request.feature.replace('_', ' ')} · {request.station.replace('_', ' ')}")
        fig.update_layout(hovermode='closest')
        fig.update_xaxes(
            title='Exceedance Probability (%)',
            type='log',
            tickvals=[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 50, 70, 90, 99],
            ticktext=['0.01', '0.05', '0.1', '0.5', '1', '2', '5', '10', '20', '30', '50', '70', '90', '99'],
            range=[np.log10(max(exceedance_pct.min(), 0.001)), np.log10(min(exceedance_pct.max(), 100))],
        )
        fig.update_yaxes(title=f"{request.feature.replace('_', ' ')} ({unit})" if unit else request.feature.replace('_', ' '))
        return fig
