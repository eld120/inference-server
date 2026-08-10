#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://192.168.0.42:8585}"
API_PREFIX="${API_PREFIX:-/api}"
MODEL="qwen3.6-35b-q4-mtp"

curl -fsS \
  -X POST "${BASE_URL}${API_PREFIX}/models/${MODEL}/unload"
echo
