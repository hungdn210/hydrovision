#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${HYDROVISION_DATA_DIR:-/opt/render/project/src/data}"

if [[ -n "${R2_BUCKET:-}" ]]; then
  echo "Syncing data from R2 (unchanged files will be skipped)..."
  R2_SYNC_PREFIXES="Mekong/filled_dataset,Mekong/data_schema.py,Mekong/mekong_basin.geojson,Mekong/prediction_results/station_predictions_future.tar.gz,Mekong/prediction_results/station_predictions_h1.tar.gz,LamaH/prediction_results/station_predictions_future.tar.gz" \
    python scripts/sync_r2_data.py

  # Extract prediction archives
  for ARCHIVE in \
    "${DATA_DIR}/Mekong/prediction_results/station_predictions_future.tar.gz" \
    "${DATA_DIR}/Mekong/prediction_results/station_predictions_h1.tar.gz" \
    "${DATA_DIR}/LamaH/prediction_results/station_predictions_future.tar.gz"; do
    if [[ -f "$ARCHIVE" ]]; then
      DEST_DIR="$(dirname "$ARCHIVE")"
      echo "Extracting $(basename "$ARCHIVE")..."
      tar -xzf "$ARCHIVE" -C "$DEST_DIR"
    fi
  done
  echo "All archives extracted."
fi

exec gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --timeout 120 --workers 1
