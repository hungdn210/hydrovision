from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict


class SchemaLoader:
    def __init__(self, schema_path: str | Path) -> None:
        self.schema_path = Path(schema_path)
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
        if not isinstance(datasets, dict) or 'mekong' not in datasets:
            raise ValueError('Schema file must define DATASETS with a mekong entry.')
        return datasets['mekong']

    @property
    def schema(self) -> Dict[str, Any]:
        return self._schema
