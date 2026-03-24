#!/bin/sh
set -eu

QDRANT_URL="${QDRANT_URL:-http://qdrant:6333}"

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

wait_for_qdrant

echo "Running ingestion sync..."
exec python main.py --ingest
