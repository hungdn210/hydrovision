import pytest
import pandas as pd
from pathlib import Path
import os
import sys

# Ensure services and data modules can be found
# data/ and services/ both live directly under the hydrovision project root (one level up from tests/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.Mekong.data_schema import DATASETS as MEKONG_SCHEMA
from data.LamaH.data_schema import DATASETS as LAMAH_SCHEMA
from services.data_loader import DataRepository, MultiDataRepository

# Extract station names and dataset roots
MEKONG_ROOT = Path(__file__).parent.parent / "data" / "Mekong" / "filled_dataset"
LAMAH_ROOT = Path(__file__).parent.parent / "data" / "LamaH" / "filled_dataset"

MEKONG_STATIONS = MEKONG_SCHEMA['mekong']['cities']
LAMAH_STATIONS = LAMAH_SCHEMA['lamah']['stations']

# Collect all (dataset, station, root, schema) combinations
ALL_STATIONS = []
for station in MEKONG_STATIONS:
    ALL_STATIONS.append(('mekong', station, MEKONG_ROOT, MEKONG_SCHEMA['mekong']))
for station in LAMAH_STATIONS:
    ALL_STATIONS.append(('lamah', station, LAMAH_ROOT, LAMAH_SCHEMA['lamah']))


@pytest.mark.parametrize("dataset_name, station_name, root_dir, schema_info", ALL_STATIONS)
def test_csv_file_exists(dataset_name, station_name, root_dir, schema_info):
    """Verify that every station declared in the schema has a corresponding physical CSV file."""
    csv_path = root_dir / f"{station_name}.csv"
    assert csv_path.exists(), f"Missing CSV for {dataset_name} station: {station_name}"


@pytest.mark.parametrize("dataset_name, station_name, root_dir, schema_info", ALL_STATIONS)
def test_csv_timestamp_integrity(dataset_name, station_name, root_dir, schema_info):
    """Verify timestamps parse cleanly, are sorted, and are duplicate-free."""
    csv_path = root_dir / f"{station_name}.csv"
    if not csv_path.exists():
        pytest.skip("File does not exist (covered by test_csv_file_exists)")

    # Read only Timestamp to make it fast
    df = pd.read_csv(csv_path, usecols=['Timestamp'])
    
    assert not df['Timestamp'].isna().any(), f"Found NaN values in Timestamp column for {station_name}"
    
    try:
        timestamps = pd.to_datetime(df['Timestamp'])
    except Exception as e:
        pytest.fail(f"Failed to parse timestamps for {station_name}: {e}")

    assert timestamps.is_monotonic_increasing, f"Timestamps are not strictly sorted for {station_name}"
    assert timestamps.is_unique, f"Duplicate timestamps found for {station_name}"


@pytest.mark.parametrize("dataset_name, station_name, root_dir, schema_info", ALL_STATIONS)
def test_schema_feature_alignment(dataset_name, station_name, root_dir, schema_info):
    """Verify schema-declared feature availability matches actual CSV columns."""
    csv_path = root_dir / f"{station_name}.csv"
    if not csv_path.exists():
        pytest.skip("File does not exist (covered by test_csv_file_exists)")

    df = pd.read_csv(csv_path, nrows=0) # Read only headers
    actual_columns = set(df.columns)
    
    # Extract expected features based on dataset structure
    if dataset_name == 'mekong':
        expected_features = set(schema_info['city_features'].get(station_name, []))
    else:
        expected_features = set(schema_info['station_features'].get(station_name, []))

    missing_features = expected_features - actual_columns
    assert not missing_features, f"{station_name} schema declares {missing_features} but they are missing from the CSV headers: {actual_columns}"


def test_bootstrap_matches_data():
    """Verify bootstrap metadata derived from the repository matches the actual data directory realistically."""
    
    repo_mekong = DataRepository(
        dataset_dir=str(MEKONG_ROOT),
        schema_path=str(Path(__file__).parent.parent / "data" / "Mekong" / "data_schema.py"),
        geojson_path=None,
        dataset="mekong"
    )
    repo_lamah = DataRepository(
        dataset_dir=str(LAMAH_ROOT),
        schema_path=str(Path(__file__).parent.parent / "data" / "LamaH" / "data_schema.py"),
        geojson_path=None,
        dataset="lamah"
    )

    multi_repo = MultiDataRepository([repo_mekong, repo_lamah])
    payload = multi_repo.bootstrap_payload()
    
    total_physical_stations = sum(1 for f in MEKONG_ROOT.glob("*.csv") if f.is_file()) + \
                              sum(1 for f in LAMAH_ROOT.glob("*.csv") if f.is_file())

    assert len(payload['stations']) <= total_physical_stations, "Bootstrap loaded more stations than physically available files."
    
    # Must have some feature counts
    assert len(payload['feature_counts']) > 0
    # Must have an end date calculated
    assert payload['time_extent']['end'] is not None
