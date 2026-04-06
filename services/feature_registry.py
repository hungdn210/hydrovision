"""
feature_registry.py
~~~~~~~~~~~~~~~~~~~
Canonical source of truth for feature classifications and analysis capabilities.
Provides methods to detect the type of a feature and determine which analysis
tools it is mathematically valid for.
"""
from enum import Enum
from typing import Dict, List

class FeatureType(Enum):
    FLOW = "flow"
    PRECIP = "precip"
    TEMP = "temp"
    WATER_Q = "water_q"
    UNKNOWN = "unknown"

_FLOW_KEYWORDS = {'discharge', 'flow', 'streamflow', 'runoff', 'water_discharge', 'water_level', 'waterlevel', 'stage', 'level', 'q_'}
_PRECIP_KEYWORDS = {'rainfall', 'precipitation', 'precip', 'rain', 'prec'}
_TEMP_KEYWORDS = {'temperature', 'temp', 't_max', 't_min', 't_avg'}
_WATER_Q_KEYWORDS = {'total_suspended_solids', 'tss', 'dissolved_oxygen', 'do2', 'ph'}

def get_feature_type(feature_name: str) -> FeatureType:
    fl = feature_name.lower()
    
    if fl in _FLOW_KEYWORDS or any(kw in fl for kw in _FLOW_KEYWORDS if kw != 'q_') or fl.startswith('q_'):
        return FeatureType.FLOW
    if fl in _PRECIP_KEYWORDS or any(kw in fl for kw in _PRECIP_KEYWORDS):
        return FeatureType.PRECIP
    if fl in _TEMP_KEYWORDS or any(kw in fl for kw in _TEMP_KEYWORDS):
        return FeatureType.TEMP
    if fl in _WATER_Q_KEYWORDS or any(kw in fl for kw in _WATER_Q_KEYWORDS):
        return FeatureType.WATER_Q
        
    return FeatureType.UNKNOWN

def is_flow(feature_name: str) -> bool:
    return get_feature_type(feature_name) == FeatureType.FLOW

def is_precip(feature_name: str) -> bool:
    return get_feature_type(feature_name) == FeatureType.PRECIP

ANALYSIS_CAPABILITIES = {
    'risk': [FeatureType.FLOW, FeatureType.PRECIP],
    'extreme': [FeatureType.FLOW, FeatureType.PRECIP],
    'climate': [FeatureType.FLOW, FeatureType.PRECIP, FeatureType.TEMP],
    'index_spi': [FeatureType.PRECIP],
    'index_flow': [FeatureType.FLOW],
    'network': [FeatureType.FLOW],
}

def get_valid_features_for_analysis(analysis_name: str, available_features: List[str]) -> List[str]:
    """Return a list of features from available_features valid for the given analysis."""
    allowed_types = set(ANALYSIS_CAPABILITIES.get(analysis_name, []))
    if not allowed_types:
        return list(available_features)
    return [f for f in available_features if get_feature_type(f) in allowed_types]

def bootstrap_feature_registry(available_features: List[str]) -> Dict:
    """Return the frontend classification map and feature mapping."""
    return {
        'capabilities': {
            key: [t.value for t in types]
            for key, types in ANALYSIS_CAPABILITIES.items()
        },
        'feature_type_map': {
            f: get_feature_type(f).value
            for f in available_features
        }
    }
