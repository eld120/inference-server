#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://192.168.0.42:8585}"
API_PREFIX="${API_PREFIX:-/api}"
MODEL="gemma-4-31b-q4"
RUNTIME="${RUNTIME:-vulkan}"

curl -fsS \
  -X POST "${BASE_URL}${API_PREFIX}/models/${MODEL}/load" \
  -H "Content-Type: application/json" \
  -d "{\"runtime\":\"${RUNTIME}\"}"
echo
