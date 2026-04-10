#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${HYDROVISION_DATA_DIR:-/opt/render/project/src/data}"
MEKONG_SCHEMA="${DATA_DIR}/Mekong/data_schema.py"
MEKONG_PREDICTIONS="${DATA_DIR}/Mekong/prediction_results/station_predictions"

if [[ -n "${R2_BUCKET:-}" ]]; then
  if [[ -f "$MEKONG_SCHEMA" && -d "$MEKONG_PREDICTIONS" ]]; then
    echo "Dataset and prediction results already present at ${DATA_DIR}; skipping R2 sync."
  else
    echo "Dataset or prediction results not found at ${DATA_DIR}; syncing from R2 (Mekong only)."
    python scripts/sync_r2_data.py
  fi
fi

exec gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --timeout 120 --workers 1
