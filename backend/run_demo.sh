#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root relative to this script.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${DEMO_CACHE_DIR:-$ROOT_DIR/frontend/demo_cache}"

if [[ ! -f "$CACHE_DIR/session.json" ]]; then
  echo "[demo] Missing cache snapshot at $CACHE_DIR/session.json" >&2
  echo "[demo] Run: poetry run python scripts/generate_demo_cache.py ..." >&2
  exit 1
fi

cd "$ROOT_DIR"

export DEMO_CACHE_MODE=1
export DEMO_CACHE_DIR="$CACHE_DIR"
export PORT="${PORT:-5001}"

exec poetry run python backend/app.py "$@"
