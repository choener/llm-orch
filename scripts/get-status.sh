#!/usr/bin/env bash
# Query llm-orch's admin status endpoint.
# Set APIKEY_LLM_ORCH to your admin API key, defaults to the user's personal
# key.  The admin key is the same key used for API calls (no separate admin key).
set -euo pipefail

: "${APIKEY_LLM_ORCH:?APIKEY_LLM_ORCH is required}"
BASE="${LLM_ORCH_URL:-http://127.0.0.1:8888}"

curl -sS -H "Authorization: Bearer ${APIKEY_LLM_ORCH}" "${BASE}/admin/status" | jq
