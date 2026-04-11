#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${HYDROVISION_DATA_DIR:-/opt/render/project/src/data}"

if [[ -n "${R2_BUCKET:-}" ]]; then
  echo "Syncing data from R2 (unchanged files will be skipped)..."
  R2_SYNC_PREFIXES="Mekong/filled_dataset,Mekong/data_schema.py,Mekong/mekong_basin.geojson,Mekong/prediction_results/station_predictions_future,Mekong/prediction_results/station_predictions_h1.tar.gz,LamaH/prediction_results/station_predictions_future" \
    python scripts/sync_r2_data.py

  # Extract h1 archive if it was just downloaded or not yet extracted
  H1_ARCHIVE="${DATA_DIR}/Mekong/prediction_results/station_predictions_h1.tar.gz"
  H1_DIR="${DATA_DIR}/Mekong/prediction_results/station_predictions_h1"
  if [[ -f "$H1_ARCHIVE" && ! -d "$H1_DIR" ]]; then
    echo "Extracting station_predictions_h1.tar.gz..."
    tar -xzf "$H1_ARCHIVE" -C "${DATA_DIR}/Mekong/prediction_results/"
    echo "Extraction complete."
  fi
fi

exec gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --timeout 120 --workers 1
