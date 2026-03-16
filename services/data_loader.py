from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .schema_loader import SchemaLoader


@dataclass(frozen=True)
class SeriesRequest:
    station: str
    feature: str
    start_date: str
    end_date: str


class DataRepository:
    def __init__(self, dataset_dir: str | Path, schema_path: str | Path, geojson_path: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.geojson_path = Path(geojson_path)
        self.schema_loader = SchemaLoader(schema_path)
        self.schema = self.schema_loader.schema
        self.feature_units: Dict[str, str] = dict(self.schema.get('feature_units', {}))
        self.feature_frequency: Dict[str, str] = dict(self.schema.get('feature_frequency', {}))
        self.city_info: Dict[str, Dict[str, Any]] = dict(self.schema.get('city_info', {}))
        self.city_features: Dict[str, List[str]] = dict(self.schema.get('city_features', {}))
        self.station_names: List[str] = list(self.schema.get('cities', []))
        self.station_index: Dict[str, Dict[str, Any]] = {}
        self.global_feature_counts: Dict[str, int] = {}
        self.global_time_extent: Dict[str, str | None] = {'start': None, 'end': None}
        self._build_index()

    def _build_index(self) -> None:
        feature_counts = {feature: 0 for feature in self.feature_units}
        global_start = None
        global_end = None

        for station in self.station_names:
            file_path = self.dataset_dir / f'{station}.csv'
            city_meta = self.city_info.get(station, {})
            if not file_path.exists():
                continue

            df = self._load_station_dataframe(str(file_path))
            station_features = []
            feature_details: Dict[str, Any] = {}
            latest_timestamp = None

            for feature in self.city_features.get(station, []):
                if feature not in df.columns:
                    continue
                series_df = df[['Timestamp', feature]].copy()
                series_df[feature] = pd.to_numeric(series_df[feature], errors='coerce')
                series_df = series_df.dropna(subset=[feature])
                if series_df.empty:
                    continue

                imputed_col = f'{feature}_imputed'
                if imputed_col in df.columns:
                    imputed_series = df.loc[series_df.index, imputed_col].fillna('No').astype(str)
                    imputed_count = int(imputed_series.str.lower().eq('yes').sum())
                else:
                    imputed_count = 0

                start = series_df['Timestamp'].min()
                end = series_df['Timestamp'].max()
                latest_row = series_df.sort_values('Timestamp').iloc[-1]
                latest_timestamp = max(latest_timestamp, end) if latest_timestamp is not None else end

                feature_details[feature] = {
                    'start_date': start.strftime('%Y-%m-%d'),
                    'end_date': end.strftime('%Y-%m-%d'),
                    'observations': int(len(series_df)),
                    'imputed_points': imputed_count,
                    'latest_value': None if pd.isna(latest_row[feature]) else float(latest_row[feature]),
                    'latest_date': latest_row['Timestamp'].strftime('%Y-%m-%d'),
                    'unit': self.feature_units.get(feature, ''),
                    'frequency': self.feature_frequency.get(feature, 'daily'),
                }
                station_features.append(feature)
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

                global_start = start if global_start is None else min(global_start, start)
                global_end = end if global_end is None else max(global_end, end)

            self.station_index[station] = {
                'station': station,
                'country': city_meta.get('country', 'Unknown'),
                'lat': city_meta.get('lat'),
                'lon': city_meta.get('lon'),
                'features': station_features,
                'feature_details': feature_details,
                'latest_timestamp': latest_timestamp.strftime('%Y-%m-%d') if latest_timestamp is not None else None,
            }

        self.global_feature_counts = feature_counts
        self.global_time_extent = {
            'start': global_start.strftime('%Y-%m-%d') if global_start is not None else None,
            'end': global_end.strftime('%Y-%m-%d') if global_end is not None else None,
        }

    @staticmethod
    @lru_cache(maxsize=128)
    def _load_station_dataframe(file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        if 'Timestamp' not in df.columns:
            raise ValueError(f'Timestamp column missing in {file_path}')
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)
        return df

    def station_exists(self, station: str) -> bool:
        return station in self.station_index

    def feature_available(self, station: str, feature: str) -> bool:
        meta = self.station_index.get(station, {})
        return feature in meta.get('features', [])

    def get_station_metadata(self, station: str) -> Dict[str, Any]:
        if not self.station_exists(station):
            raise KeyError(f'Unknown station: {station}')
        return self.station_index[station]

    def get_station_dataframe(self, station: str) -> pd.DataFrame:
        if not self.station_exists(station):
            raise KeyError(f'Unknown station: {station}')
        file_path = self.dataset_dir / f'{station}.csv'
        if not file_path.exists():
            raise FileNotFoundError(f'Dataset file not found for station {station}')
        return self._load_station_dataframe(str(file_path)).copy()

    def get_feature_series(self, request: SeriesRequest) -> pd.DataFrame:
        if not self.feature_available(request.station, request.feature):
            raise ValueError(f'{request.feature} is not available for {request.station}')

        df = self.get_station_dataframe(request.station)
        feature = request.feature
        imputed_col = f'{feature}_imputed'
        columns = ['Timestamp', feature]
        if imputed_col in df.columns:
            columns.append(imputed_col)
        feature_df = df[columns].copy()
        feature_df[feature] = pd.to_numeric(feature_df[feature], errors='coerce')
        feature_df = feature_df.dropna(subset=[feature])
        if feature_df.empty:
            raise ValueError(f'No non-null {feature} values found for {request.station}')

        start = pd.to_datetime(request.start_date)
        end = pd.to_datetime(request.end_date)
        if pd.isna(start) or pd.isna(end):
            raise ValueError('Invalid start_date or end_date')
        if end < start:
            raise ValueError('end_date must be on or after start_date')

        feature_df = feature_df[(feature_df['Timestamp'] >= start) & (feature_df['Timestamp'] <= end)].copy()
        if feature_df.empty:
            raise ValueError(f'No {feature} data for {request.station} in the selected date range')

        feature_df.rename(columns={feature: 'Value'}, inplace=True)
        if imputed_col in feature_df.columns:
            feature_df['Imputed'] = feature_df[imputed_col].fillna('No').astype(str)
            feature_df.drop(columns=[imputed_col], inplace=True)
        else:
            feature_df['Imputed'] = 'No'
        feature_df['Station'] = request.station
        feature_df['Feature'] = request.feature
        feature_df['Unit'] = self.feature_units.get(request.feature, '')
        return feature_df.reset_index(drop=True)

    def bootstrap_payload(self) -> Dict[str, Any]:
        return {
            'stations': list(self.station_index.values()),
            'station_names': sorted(self.station_index.keys()),
            'features': [feature for feature in self.feature_units.keys()],
            'feature_units': self.feature_units,
            'feature_frequency': self.feature_frequency,
            'feature_counts': self.global_feature_counts,
            'time_extent': self.global_time_extent,
            'graph_types': [
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
            ],
        }
