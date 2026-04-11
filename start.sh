#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${HYDROVISION_DATA_DIR:-/opt/render/project/src/data}"
MEKONG_SCHEMA="${DATA_DIR}/Mekong/data_schema.py"

if [[ -n "${R2_BUCKET:-}" ]]; then
  echo "Syncing data from R2 (unchanged files will be skipped)..."
  R2_SYNC_PREFIXES="Mekong/filled_dataset,Mekong/data_schema.py,Mekong/mekong_basin.geojson,Mekong/prediction_results/station_predictions_future,Mekong/prediction_results/station_predictions_h1,LamaH/prediction_results/station_predictions_future" \
    python scripts/sync_r2_data.py
fi

exec gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --timeout 120 --workers 1
