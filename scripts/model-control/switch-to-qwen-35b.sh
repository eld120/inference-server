#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://192.168.0.42:8585}"
API_PREFIX="${API_PREFIX:-/api}"
OLD_MODEL="qwen3.6-27b-q4-mtp"
NEW_MODEL="qwen3.6-35b-q4-mtp"

curl -sS -X POST "${BASE_URL}${API_PREFIX}/models/${OLD_MODEL}/unload" >/dev/null || true
curl -fsS \
  -X POST "${BASE_URL}${API_PREFIX}/models/${NEW_MODEL}/load" \
  -H "Content-Type: application/json" \
  -d '{"runtime":"vulkan"}'
echo
