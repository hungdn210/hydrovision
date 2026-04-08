#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${APP_ROOT:-/opt/hydrovision/app}"
VENV_ROOT="${VENV_ROOT:-/opt/hydrovision/venv}"
DATA_ROOT="${DATA_ROOT:-/opt/hydrovision/data}"

sudo apt update
sudo apt install -y python3 python3-venv python3-pip nginx git unzip

sudo mkdir -p /opt/hydrovision
sudo chown -R "$USER":"$USER" /opt/hydrovision

if [[ ! -d "$VENV_ROOT" ]]; then
  python3 -m venv "$VENV_ROOT"
fi

source "$VENV_ROOT/bin/activate"
pip install --upgrade pip
pip install -r "$APP_ROOT/requirements.txt"

mkdir -p "$DATA_ROOT"

echo "Bootstrap complete."
echo "Next steps:"
echo "  1. Copy deploy/oracle/hydrovision.env.example to /opt/hydrovision/hydrovision.env"
echo "  2. Add your data under $DATA_ROOT or sync it from R2"
echo "  3. Install the systemd service and nginx config from deploy/oracle/"
