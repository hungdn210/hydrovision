#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${R2_BUCKET:-}" ]]; then
  python scripts/sync_r2_data.py
fi

exec gunicorn app:app --bind 0.0.0.0:${PORT:-10000}
