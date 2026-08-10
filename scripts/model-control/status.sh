#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://192.168.0.42:8585}"
API_PREFIX="${API_PREFIX:-/api}"

curl -fsS "${BASE_URL}${API_PREFIX}/status"
echo
