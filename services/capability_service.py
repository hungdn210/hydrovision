"""
capability_service.py
=====================
Scans the data/prediction_results directory tree at startup and builds an
in-memory capability index so the rest of the app can answer:

    "Does model X have a trained CSV for station Y / feature Z / mode M?"

without touching the filesystem at request time.

Index structure (see CapabilityService.index)
---------------------------------------------
{
  "mekong": {
    "historical": {
      "Water_Discharge": { "FlowNet": {"Ban_Chot", ...}, "LSTM": {...}, ... },
      "Water_Level":     { ... },
    },
    "future": { identical structure },
  },
  "lamah": {
    "historical": {          # LamaH has a flat model directory (no feature sub-folder)
      "_default_feature": { "FlowNet": {"100", "101", ...}, ... }
    },
    "future": { identical flat structure },
  }
}

The public API is intentionally simple; callers never need to touch `index`
directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set

# Mekong prediction dirs sit inside a feature sub-folder
_MEKONG_FEATURE_DIRS = {'Water_Discharge', 'Water_Level', 'Rainfall', 'Total_Suspended_Solids'}
# Sentinel used to represent LamaH's single (unnamed) feature layer
_LAMAH_FEATURE_KEY = '_default_feature'
# Map from Mekong folder names → frontend feature names
_MEKONG_FOLDER_TO_FEATURE = {
    'Water_Discharge': 'Discharge',
    'Water_Level': 'Water_Level',
    'Rainfall': 'Rainfall',
    'Total_Suspended_Solids': 'Total_Suspended_Solids',
}


class CapabilityService:
    """
    Lightweight filesystem scanner for prediction artifacts.

    Call once at startup:

        cap = CapabilityService(data_dir='data')
        cap.scan()          # populates self.index

    Then query at request time with no further filesystem I/O.
    """

    def __init__(self, data_dir: str | Path = 'data') -> None:
        self.data_dir = Path(data_dir)
        self.index: Dict = {}

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def scan(self) -> None:
        """Populate self.index by walking the prediction_results trees."""
        self.index = {
            'mekong': {
                'historical': {},
                'future': {},
            },
            'lamah': {
                'historical': {},
                'future': {},
            },
        }
        self._scan_mekong()
        self._scan_lamah()

    def has_hist(self, dataset: str, station: str, feature: str, model: str) -> bool:
        """Return True if a historical-fit CSV exists for this combo."""
        return station in self._station_set(dataset, 'historical', feature, model)

    def has_future(self, dataset: str, station: str, feature: str, model: str) -> bool:
        """Return True if a future-forecast CSV exists for this combo."""
        return station in self._station_set(dataset, 'future', feature, model)

    def supported_models(
        self,
        dataset: str,
        station: str,
        feature: str,
        mode: str,
    ) -> list[str]:
        """
        Return sorted list of model names that have a CSV for this
        dataset/station/feature/mode combination.
        """
        feat_key = self._feat_key(dataset, feature)
        mode_index = self.index.get(dataset, {}).get(mode, {})
        feat_index = mode_index.get(feat_key, {})
        return sorted(m for m, stations in feat_index.items() if station in stations)

    def supported_stations(
        self,
        dataset: str,
        feature: str,
        model: str,
        mode: str,
    ) -> Set[str]:
        """Return the set of station identifiers for which a CSV exists."""
        feat_key = self._feat_key(dataset, feature)
        return set(
            self.index.get(dataset, {}).get(mode, {}).get(feat_key, {}).get(model, set())
        )

    def bootstrap_capabilities(self) -> dict:
        """
        Return a compact, JSON-serialisable representation for the bootstrap
        payload.  Only includes combinations that have *at least one* station.

        Shape:
        {
          "mekong": {
            "historical": { "Water_Discharge": ["DLinear", "FlowNet", ...], ... },
            "future":     { ... }
          },
          "lamah": {
            "historical": ["CATS", "DLinear", "FlowNet", ...],
            "future":     [...]
          }
        }
        LamaH's flat feature layer is unwrapped (just a list of model names).
        """
        out: dict = {}
        for dataset, modes in self.index.items():
            out[dataset] = {}
            for mode, feat_map in modes.items():
                if dataset == 'lamah':
                    # Flatten — LamaH has only one implicit feature (Discharge)
                    models = sorted(
                        m for m, s in feat_map.get(_LAMAH_FEATURE_KEY, {}).items() if s
                    )
                    out[dataset][mode] = models
                else:
                    out[dataset][mode] = {
                        feat: sorted(m for m, s in model_map.items() if s)
                        for feat, model_map in feat_map.items()
                        if any(s for s in model_map.values())
                    }
        return out

    def station_prediction_features(self) -> dict:
        """
        Returns per-station, per-mode set of features that have at least one trained
        prediction CSV.  Feature names use the frontend convention (e.g. 'Discharge',
        not 'Water_Discharge').

        Shape::

            {
              "mekong": {
                "Kratie": {"historical": ["Discharge", "Water_Level"], "future": [...]},
                ...
              },
              "lamah": {
                "237": {"historical": ["Discharge"], "future": ["Discharge"]},
                ...
              }
            }
        """
        out: dict = {'mekong': {}, 'lamah': {}}

        # Mekong — features live in named sub-folders
        for mode in ('historical', 'future'):
            for folder_name, model_map in self.index.get('mekong', {}).get(mode, {}).items():
                feature_name = _MEKONG_FOLDER_TO_FEATURE.get(folder_name, folder_name)
                for stations in model_map.values():
                    for station in stations:
                        entry = out['mekong'].setdefault(station, {'historical': set(), 'future': set()})
                        entry[mode].add(feature_name)

        # Convert sets → sorted lists
        for station, modes in out['mekong'].items():
            out['mekong'][station] = {m: sorted(s) for m, s in modes.items()}

        # LamaH — single implicit feature (Discharge)
        for mode in ('historical', 'future'):
            for stations in self.index.get('lamah', {}).get(mode, {}).get(_LAMAH_FEATURE_KEY, {}).values():
                for station in stations:
                    entry = out['lamah'].setdefault(station, {'historical': set(), 'future': set()})
                    entry[mode].add('Discharge')

        for station, modes in out['lamah'].items():
            out['lamah'][station] = {m: sorted(s) for m, s in modes.items()}

        return out

    def models_for_station(self, dataset: str, station: str, feature: str) -> dict:
        """
        Return { 'historical': [model, ...], 'future': [model, ...] } for a
        specific station/feature.  Used by /api/predict to decide what models
        to offer.
        """
        return {
            mode: self.supported_models(dataset, station, feature, mode)
            for mode in ('historical', 'future')
        }

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _feat_key(self, dataset: str, feature: str) -> str:
        if dataset == 'lamah':
            return _LAMAH_FEATURE_KEY
        return feature  # Mekong uses actual feature folder names

    def _station_set(self, dataset: str, mode: str, feature: str, model: str) -> Set[str]:
        feat_key = self._feat_key(dataset, feature)
        return self.index.get(dataset, {}).get(mode, {}).get(feat_key, {}).get(model, set())

    # ------------------------------------------------------------------
    # scanning
    # ------------------------------------------------------------------

    def _scan_mekong(self) -> None:
        """
        Mekong layout:
            data/Mekong/prediction_results/station_predictions/<Feature>/<Model>/<station>.csv
            data/Mekong/prediction_results/station_predictions_future/<Feature>/<Model>/<station>.csv
        """
        base = self.data_dir / 'Mekong' / 'prediction_results'
        for sub, mode_key in [
            ('station_predictions',        'historical'),
            ('station_predictions_future', 'future'),
        ]:
            root = base / sub
            if not root.is_dir():
                continue
            for feat_dir in root.iterdir():
                if not feat_dir.is_dir() or feat_dir.name.startswith('.'):
                    continue
                feat = feat_dir.name
                if feat not in self.index['mekong'][mode_key]:
                    self.index['mekong'][mode_key][feat] = {}
                for model_dir in feat_dir.iterdir():
                    if not model_dir.is_dir() or model_dir.name.startswith('.'):
                        continue
                    model = model_dir.name
                    stations = {p.stem for p in model_dir.glob('*.csv')}
                    self.index['mekong'][mode_key][feat][model] = stations

    def _scan_lamah(self) -> None:
        """
        LamaH layout — models sit directly under the sub-folder:
            data/LamaH/prediction_results/station_predictions/<Model>/<station>.csv
            data/LamaH/prediction_results/station_predictions_future/<Model>/<station>.csv
            data/LamaH/prediction_results/station_predictions_future/<Model>/LamaH_daily/<station>.csv
        """
        base = self.data_dir / 'LamaH' / 'prediction_results'
        for sub, mode_key in [
            ('station_predictions',        'historical'),
            ('station_predictions_future', 'future'),
        ]:
            root = base / sub
            if not root.is_dir():
                continue
            if _LAMAH_FEATURE_KEY not in self.index['lamah'][mode_key]:
                self.index['lamah'][mode_key][_LAMAH_FEATURE_KEY] = {}
            for model_dir in root.iterdir():
                if not model_dir.is_dir() or model_dir.name.startswith('.'):
                    continue
                model = model_dir.name
                # CSVs may be directly in model_dir or inside a LamaH_daily/ subfolder
                stations: Set[str] = {p.stem for p in model_dir.glob('*.csv')}
                sub2 = model_dir / 'LamaH_daily'
                if sub2.is_dir():
                    stations |= {p.stem for p in sub2.glob('*.csv')}
                self.index['lamah'][mode_key][_LAMAH_FEATURE_KEY][model] = stations
