#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${HYDROVISION_DATA_DIR:-/opt/render/project/src/data}"
MEKONG_SCHEMA="${DATA_DIR}/Mekong/data_schema.py"
LAMAH_SCHEMA="${DATA_DIR}/LamaH/data_schema.py"

if [[ -n "${R2_BUCKET:-}" ]]; then
  if [[ -f "$MEKONG_SCHEMA" || -f "$LAMAH_SCHEMA" ]]; then
    echo "Persistent dataset already present at ${DATA_DIR}; skipping R2 sync."
  else
    echo "Persistent dataset not found at ${DATA_DIR}; syncing from R2."
    python scripts/sync_r2_data.py
  fi
fi

exec gunicorn app:app --bind 0.0.0.0:${PORT:-10000}
