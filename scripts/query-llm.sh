#!/usr/bin/env bash
# Quick test: send a chat completion request to llm-orch.
#
# Usage: scripts/query-llm.sh <MODEL>       # prompt from stdin
#        scripts/query-llm.sh <MODEL> <MSG>  # prompt as argument
#
# Examples:
#   echo "hello" | scripts/query-llm.sh qwen
#   scripts/query-llm.sh coder "write a fib function"
#
# Set APIKEY_LLM_ORCH to your API key.
set -euo pipefail

: "${APIKEY_LLM_ORCH:?APIKEY_LLM_ORCH is required}"
BASE="${LLM_ORCH_URL:-http://127.0.0.1:8080}"

if [ $# -lt 1 ]; then
    echo "usage: $0 <MODEL> [MESSAGE]" >&2
    exit 1
fi

MODEL="$1"
shift

if [ $# -ge 1 ]; then
    MSG="$*"
else
    MSG="$(cat)"
fi

PAYLOAD=$(jq -nc \
    --arg model "$MODEL" \
    --arg content "$MSG" \
    '{
        model: $model,
        messages: [{role: "user", content: $content}],
        stream: false,
        max_tokens: 32768
    }')

RESP=$(curl -sS -w '\n%{http_code}' \
    -H "Authorization: Bearer ${APIKEY_LLM_ORCH}" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    "${BASE}/v1/chat/completions")

# Split trailing HTTP status code from body.
HTTP_CODE=$(echo "$RESP" | tail -1)
BODY=$(echo "$RESP" | sed '$d')

if [ "$HTTP_CODE" -ge 400 ]; then
    echo "HTTP $HTTP_CODE:" >&2
    echo "$BODY" >&2
    exit 1
fi

echo "$BODY" | jq
