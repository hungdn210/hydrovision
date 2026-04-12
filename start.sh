#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${HYDROVISION_DATA_DIR:-/opt/render/project/src/data}"

if [[ -n "${R2_BUCKET:-}" ]]; then
  echo "Syncing data from R2 (unchanged files will be skipped)..."
  R2_SYNC_PREFIXES="Mekong/filled_dataset,Mekong/data_schema.py,Mekong/mekong_basin.geojson,Mekong/prediction_results/station_predictions_future.tar.gz,Mekong/prediction_results/station_predictions_h1.tar.gz,LamaH/filled_dataset.tar.gz,LamaH/prediction_results/station_predictions_future.tar.gz" \
    python scripts/sync_r2_data.py

  # Extract each archive only if it has changed since last extraction.
  # A .extracted_<name> marker file stores the byte size at last extraction.
  extract_if_changed() {
    local ARCHIVE="$1"
    local MARKER="${ARCHIVE}.extracted"
    if [[ ! -f "$ARCHIVE" ]]; then return; fi
    local CURRENT_SIZE
    CURRENT_SIZE=$(stat -c%s "$ARCHIVE" 2>/dev/null || stat -f%z "$ARCHIVE")
    if [[ -f "$MARKER" ]] && [[ "$(cat "$MARKER")" == "$CURRENT_SIZE" ]]; then
      echo "Skipping $(basename "$ARCHIVE") (already extracted, unchanged)."
      return
    fi
    echo "Extracting $(basename "$ARCHIVE")..."
    tar -xzf "$ARCHIVE" -C "$(dirname "$ARCHIVE")"
    echo "$CURRENT_SIZE" > "$MARKER"
    echo "Done."
  }

  extract_if_changed "${DATA_DIR}/Mekong/prediction_results/station_predictions_future.tar.gz"
  extract_if_changed "${DATA_DIR}/Mekong/prediction_results/station_predictions_h1.tar.gz"
  extract_if_changed "${DATA_DIR}/LamaH/filled_dataset.tar.gz"
  extract_if_changed "${DATA_DIR}/LamaH/prediction_results/station_predictions_future.tar.gz"
fi

exec gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --timeout 120 --workers 1
