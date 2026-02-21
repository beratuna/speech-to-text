#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  osascript -e 'display alert "uv not found" message "Install uv from https://docs.astral.sh/uv/getting-started/installation/ and retry." as critical'
  exit 1
fi

if [ ! -d ".venv" ]; then
  uv sync --dev
fi

PORT=8501
if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  PORT=8502
fi

open "http://localhost:$PORT"
exec .venv/bin/streamlit run app.py --server.headless true --server.port "$PORT"
