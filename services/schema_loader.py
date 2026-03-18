from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict


class SchemaLoader:
    def __init__(self, schema_path: str | Path, dataset_key: str = 'mekong') -> None:
        self.schema_path = Path(schema_path)
        self.dataset_key = dataset_key
        if not self.schema_path.exists():
            raise FileNotFoundError(f'Schema file not found: {self.schema_path}')
        self._schema = self._load_schema()

    def _load_schema(self) -> Dict[str, Any]:
        spec = importlib.util.spec_from_file_location('hydrovision_data_schema', self.schema_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Unable to import schema from {self.schema_path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        datasets = getattr(module, 'DATASETS', None)
        if not isinstance(datasets, dict) or self.dataset_key not in datasets:
            raise ValueError(f'Schema file must define DATASETS with a "{self.dataset_key}" entry.')
        data = datasets[self.dataset_key]
        # Normalize field names so DataRepository works with any schema format
        return {
            'cities': data.get('cities') or data.get('stations', []),
            'city_info': data.get('city_info') or data.get('station_info', {}),
            'city_features': data.get('city_features') or data.get('station_features', {}),
            'feature_units': data.get('feature_units', {}),
            'feature_frequency': data.get('feature_frequency', {}),
        }

    @property
    def schema(self) -> Dict[str, Any]:
        return self._schema
