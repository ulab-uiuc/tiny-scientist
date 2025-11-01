#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root relative to this script.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

# Ensure demo flags are not leaking into the normal instance.
unset DEMO_CACHE_MODE || true
unset DEMO_CACHE_DIR || true

# Default port for live instance
export PORT="${PORT:-5000}"

exec poetry run python backend/app.py "$@"
