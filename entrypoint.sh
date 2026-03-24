#!/bin/sh
set -eu

APP_SERVER="${APP_SERVER:-streamlit}"
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8501}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
QDRANT_URL="${QDRANT_URL:-http://qdrant:6333}"
FASTAPI_APP_MODULE="${FASTAPI_APP_MODULE:-src.api.main:app}"
STREAMLIT_APP_PATH="${STREAMLIT_APP_PATH:-/app/src/ui/app.py}"

export PYTHONPATH=/app/src

wait_for_qdrant() {
  echo "Waiting for Qdrant at ${QDRANT_URL}..."
  python - <<'PY'
import os
import time
from urllib.request import urlopen

qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333").rstrip("/")
health_url = qdrant_url + "/readyz"

for _ in range(60):
    try:
        with urlopen(health_url, timeout=5) as response:
            if 200 <= response.status < 300:
                raise SystemExit(0)
    except Exception:
        time.sleep(2)

raise SystemExit("Qdrant did not become ready in time.")
PY
}

start_fastapi_background() {
  echo "Starting FastAPI backend on ${API_HOST}:${API_PORT}..."
  uvicorn "${FASTAPI_APP_MODULE}" --host "${API_HOST}" --port "${API_PORT}" &
  FASTAPI_PID=$!
  trap 'kill "${FASTAPI_PID}" 2>/dev/null || true' EXIT INT TERM
}

wait_for_qdrant

if [ "${APP_SERVER}" = "streamlit" ]; then
  if [ ! -f "${STREAMLIT_APP_PATH}" ]; then
    echo "Expected Streamlit app at ${STREAMLIT_APP_PATH}, but it does not exist." >&2
    exit 1
  fi
  export VULN_EXPLORER_API_URL="${VULN_EXPLORER_API_URL:-http://127.0.0.1:${API_PORT}}"
  start_fastapi_background
  exec streamlit run "${STREAMLIT_APP_PATH}" --server.address="${APP_HOST}" --server.port="${APP_PORT}"
fi

if [ "${APP_SERVER}" = "fastapi" ]; then
  exec uvicorn "${FASTAPI_APP_MODULE}" --host "${APP_HOST}" --port "${APP_PORT}"
fi

echo "Unsupported APP_SERVER value: ${APP_SERVER}" >&2
exit 1
