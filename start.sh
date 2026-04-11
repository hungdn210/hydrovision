#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${HYDROVISION_DATA_DIR:-/opt/render/project/src/data}"
MEKONG_SCHEMA="${DATA_DIR}/Mekong/data_schema.py"

if [[ -n "${R2_BUCKET:-}" ]]; then
  if [[ -f "$MEKONG_SCHEMA" ]]; then
    echo "Dataset already present at ${DATA_DIR}; skipping R2 sync."
  else
    echo "Dataset not found; syncing core Mekong data from R2 (excluding prediction_results)."
    # Sync only essential files — filled_dataset + top-level schema/geojson
    # prediction_results (~466 MB, thousands of CSVs) is excluded to stay within
    # free-tier startup time limits. The Predict panel is disabled on the live site.
    R2_SYNC_PREFIXES="Mekong/filled_dataset,Mekong/data_schema.py,Mekong/mekong_basin.geojson,Mekong/prediction_results/station_predictions_future" \
      python scripts/sync_r2_data.py
  fi
fi

exec gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --timeout 120 --workers 1
