#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://192.168.0.42:8585}"
API_PREFIX="${API_PREFIX:-/api}"
MODEL="qwen3.6-27b-q4-mtp"
RUNTIME="vulkan"

curl -fsS \
  -X POST "${BASE_URL}${API_PREFIX}/models/${MODEL}/load" \
  -H "Content-Type: application/json" \
  -d "{\"runtime\":\"${RUNTIME}\"}"
echo
