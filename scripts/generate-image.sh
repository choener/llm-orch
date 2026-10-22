#!/usr/bin/env bash
# Generate an image via llm-orch's /v1/images/generations endpoint.
# Default model: "moss" (MOSS-Image-Base on stable-diffusion.cpp).
#
# Usage:
#   scripts/generate-image.sh [options] [-p PROMPT | -f PROMPT_FILE]
#
# Options:
#   -p PROMPT          Prompt text (inline; overrides -f).
#   -f PROMPT_FILE     Read prompt from file (use "-" for stdin).
#   -m MODEL           Model name (default: moss).
#   -W WIDTH           Image width  (default: 1024).
#   -H HEIGHT          Image height (default: 1024).
#   --size WxH         Shorthand for -W and -H (e.g. 1216x832).
#   -n COUNT           Number of images per request (default: 1).
#   --seed N           RNG seed (-1 = random; default: -1).
#   --cfg-scale F      Classifier-free guidance scale (default: 3.0; MOSS uses 3.0).
#   --steps N          Sampling steps (default: 20).
#   --sampling-method  Sampler, e.g. euler, euler_a (default: euler).
#   --scheduler S      Scheduler, e.g. discrete, karras (default: backend default).
#   --flow-shift F     Flow / time shift (default: backend default; MOSS uses 5).
#   --clip-skip N      CLIP skip (default: backend default).
#   -u NEGATIVE        Negative prompt.
#   --extra JSON       Extra stable-diffusion.cpp native generation parameters,
#                      deep-merged into the request (see "Native parameters").
#   --format FMT       Output format: png | jpeg | webp (default: png).
#   --compression N    Output compression, 0..100 (default: 100).
#   -o OUTDIR          Output directory (default: ./images).
#   -q                 Quiet — print only the output path(s).
#   --dry-run          Print the request JSON and exit without calling the API.
#
# Environment:
#   APIKEY_LLM_ORCH    API key (required).
#   LLM_ORCH_URL       Base URL, default http://127.0.0.1:8080.
#
# Native parameters:
#   /v1/images/generations is the OpenAI-compatible endpoint: sd-server only
#   reads prompt, n, size, output_format and output_compression from it.
#   Everything else (seed, cfg, steps, sampler, negative prompt, lora, hires,
#   tiling, …) has to travel inside the prompt as a native block, which
#   sd-server strips out before conditioning:
#
#       <sd_cpp_extra_args>{"sample_params":{"sample_steps":28}}</sd_cpp_extra_args>
#
#   This script builds that block for you; --extra merges arbitrary native
#   fields into it, so anything from the sdcpp img_gen schema still works:
#
#       --extra '{"lora":[{"path":"flux_offset.safetensors","multiplier":0.6}]}'
#       --extra '{"sample_params":{"guidance":{"slg":{"layers":[22],"scale":0.0"}}}}'
#
#   Images render in well under a minute, so the synchronous endpoint is the
#   right call here.  Should you ever need a render that outlives
#   server.backend_total_timeout_secs, use the async pathway instead:
#   POST /sdcpp/v1/img_gen + poll GET /sdcpp/v1/jobs/{id} (see
#   scripts/generate-video-async.sh for the pattern, and note that the
#   passthrough needs an `X-LLM-Model: <model>` header unless
#   server.default_diffusion_model is set).
#
# Examples:
#   export APIKEY_LLM_ORCH=sk-...
#   scripts/generate-image.sh -p "A cat surfing on a tropical wave"
#   echo "A cat surfing" | scripts/generate-image.sh -f - -q
#   scripts/generate-image.sh -p "Schematic of a REST API" --size 1216x832 -n 4 --seed 42
#   scripts/generate-image.sh -p "An axonometric city block" -o ./shots --steps 30 --cfg-scale 4
#   scripts/generate-image.sh -m flux-1-dev -p "a portrait" --cfg-scale 3.5 --steps 28
set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
: "${APIKEY_LLM_ORCH:?APIKEY_LLM_ORCH is required}"
BASE="${LLM_ORCH_URL:-http://127.0.0.1:8080}"

MODEL="moss"
WIDTH=1024
HEIGHT=1024
COUNT=1
SEED=-1
CFG_SCALE=3.0
STEPS=20
SAMPLING_METHOD="euler"
SCHEDULER=""
FLOW_SHIFT=""
CLIP_SKIP=""
NEGATIVE=""
EXTRA="{}"
OUTPUT_FORMAT="png"
COMPRESSION=100
OUTDIR="./images"
QUIET=false
DRY_RUN=false

# ── Parse arguments ──────────────────────────────────────────────────────────
PROMPT=""
PROMPT_FILE=""

while [ $# -gt 0 ]; do
    case "$1" in
        -p) shift; PROMPT="$1" ;;
        -f) shift; PROMPT_FILE="$1" ;;
        -m) shift; MODEL="$1" ;;
        -W) shift; WIDTH="$1" ;;
        -H) shift; HEIGHT="$1" ;;
        --size) shift; WIDTH="${1%x*}"; HEIGHT="${1#*x}" ;;
        -n) shift; COUNT="$1" ;;
        --seed) shift; SEED="$1" ;;
        --cfg-scale) shift; CFG_SCALE="$1" ;;
        --steps) shift; STEPS="$1" ;;
        --sampling-method) shift; SAMPLING_METHOD="$1" ;;
        --scheduler) shift; SCHEDULER="$1" ;;
        --flow-shift) shift; FLOW_SHIFT="$1" ;;
        --clip-skip) shift; CLIP_SKIP="$1" ;;
        -u) shift; NEGATIVE="$1" ;;
        --extra) shift; EXTRA="$1" ;;
        --format) shift; OUTPUT_FORMAT="$1" ;;
        --compression) shift; COMPRESSION="$1" ;;
        -o) shift; OUTDIR="$1" ;;
        -q) QUIET=true ;;
        --dry-run) DRY_RUN=true ;;
        -h|--help)
            sed -n '/^# Usage:/,/^set -euo pipefail/p' "$0" \
                | sed 's/^# \?//; /set -euo pipefail/d'
            exit 0
            ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

log() { $QUIET || printf '%s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# ── Resolve prompt ───────────────────────────────────────────────────────────
if [ -z "$PROMPT" ] && [ -n "$PROMPT_FILE" ]; then
    if [ "$PROMPT_FILE" = "-" ]; then
        PROMPT=$(cat)
    else
        [ -f "$PROMPT_FILE" ] || die "prompt file not found: ${PROMPT_FILE}"
        PROMPT=$(cat "$PROMPT_FILE")
    fi
fi

if [ -z "$PROMPT" ]; then
    die "prompt is required (-p PROMPT or -f PROMPT_FILE)"
fi

# ── Validate geometry ────────────────────────────────────────────────────────
# MOSS-Image (and the FLUX.2 VAE it shares) downscales 16x, so dimensions must
# be multiples of 16; anything else is silently rounded by the backend and
# shifts the composition.  Warn rather than fail — the render still works.
if [ $(( WIDTH % 16 )) -ne 0 ] || [ $(( HEIGHT % 16 )) -ne 0 ]; then
    echo "warning: ${WIDTH}x${HEIGHT} is not a multiple of 16 — round to it for exact framing" >&2
fi

case "$OUTPUT_FORMAT" in
    png)        EXT=png ;;
    jpeg|jpg)   EXT=jpg; OUTPUT_FORMAT=jpeg ;;
    webp)       EXT=webp ;;
    *) die "unsupported output format: ${OUTPUT_FORMAT} (png | jpeg | webp)" ;;
esac

# ── Build the native (sdcpp) parameter block ─────────────────────────────────
# Omitted fields inherit the sd-server defaults baked into the model's `cmd`.
require_int() { case "$2" in ''|*[!0-9-]*) die "--$1 expects an integer, got '${2}'" ;; esac; }
require_num() { case "$2" in ''|*[!0-9.:-]*) die "--$1 expects a number, got '${2}'" ;; esac; }

jq -e . >/dev/null 2>&1 <<<"$EXTRA" || die "--extra is not valid JSON: ${EXTRA}"
require_int seed "$SEED"
require_int steps "$STEPS"
require_int n "$COUNT"
require_int compression "$COMPRESSION"
require_num cfg-scale "$CFG_SCALE"
[ "$COUNT" -ge 1 ] || die "--n must be at least 1"
[ -z "$FLOW_SHIFT" ] || require_num flow-shift "$FLOW_SHIFT"
[ -z "$CLIP_SKIP" ]  || require_int clip-skip "$CLIP_SKIP"

NATIVE=$(jq -nc \
    --argjson seed "$SEED" \
    --argjson cfg "$CFG_SCALE" \
    --argjson steps "$STEPS" \
    --arg method "$SAMPLING_METHOD" \
    --arg scheduler "$SCHEDULER" \
    --arg flow_shift "$FLOW_SHIFT" \
    --arg clip_skip "$CLIP_SKIP" \
    --arg negative "$NEGATIVE" \
    --argjson extra "$EXTRA" \
    '{
        seed: $seed,
        negative_prompt: $negative,
        sample_params: {
            sample_steps: $steps,
            guidance: { txt_cfg: $cfg }
        }
    }
    | if $method    != "" then .sample_params.sample_method = $method    else . end
    | if $scheduler != "" then .sample_params.scheduler     = $scheduler else . end
    | if $flow_shift != "" then .sample_params.flow_shift   = ($flow_shift | tonumber) else . end
    | if $clip_skip != "" then .clip_skip = ($clip_skip | tonumber) else . end
    | . * $extra')

# The native block rides inside the prompt for the OpenAI-compatible endpoint.
PAYLOAD=$(jq -nc \
    --arg model "$MODEL" \
    --arg prompt "$PROMPT" \
    --arg native "$NATIVE" \
    --arg size "${WIDTH}x${HEIGHT}" \
    --arg format "$OUTPUT_FORMAT" \
    --argjson n "$COUNT" \
    --argjson compression "$COMPRESSION" \
    '{
        model: $model,
        prompt: ($prompt + " <sd_cpp_extra_args>" + $native + "</sd_cpp_extra_args>"),
        n: $n,
        size: $size,
        output_format: $format,
        output_compression: $compression
    }')

if $DRY_RUN; then
    jq . <<<"$PAYLOAD"
    exit 0
fi

log "Generating image with model '${MODEL}'..."
log "  ${COUNT}x ${WIDTH}x${HEIGHT}  steps=${STEPS}  cfg=${CFG_SCALE}  method=${SAMPLING_METHOD:-default}  seed=${SEED}"

# ── Submit ───────────────────────────────────────────────────────────────────
BODY_FILE=$(mktemp)
trap 'rm -f "$BODY_FILE"' EXIT

HTTP_CODE=$(curl -sS -w '%{http_code}' \
    -H "Authorization: Bearer ${APIKEY_LLM_ORCH}" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    -o "$BODY_FILE" \
    "${BASE}/v1/images/generations") || die "could not reach ${BASE} — is llm-orch running?"

if [ "$HTTP_CODE" -ge 400 ] || [ "$HTTP_CODE" = "000" ]; then
    echo "HTTP ${HTTP_CODE}:" >&2
    cat "$BODY_FILE" >&2
    exit 1
fi

jq -e '.data | type == "array" and length > 0' "$BODY_FILE" >/dev/null \
    || die "no images in response: $(head -c 400 "$BODY_FILE")"

# Honour the format the backend actually encoded (it wins over our request).
case "$(jq -r '.output_format // empty' "$BODY_FILE")" in
    png)  EXT=png ;;
    jpeg) EXT=jpg ;;
    webp) EXT=webp ;;
esac

# ── Save output ──────────────────────────────────────────────────────────────
mkdir -p "$OUTDIR"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TOTAL=$(jq -r '.data | length' "$BODY_FILE")
i=0
while [ "$i" -lt "$TOTAL" ]; do
    if [ "$TOTAL" -gt 1 ]; then
        OUTFILE="${OUTDIR}/llm-orch-image-${TIMESTAMP}-$((i + 1)).${EXT}"
    else
        OUTFILE="${OUTDIR}/llm-orch-image-${TIMESTAMP}.${EXT}"
    fi
    # Two renders in the same second must not clobber each other.
    if [ -e "$OUTFILE" ]; then
        dup=1
        while [ -e "${OUTFILE%.*}+${dup}.${EXT}" ]; do dup=$((dup + 1)); done
        OUTFILE="${OUTFILE%.*}+${dup}.${EXT}"
    fi
    jq -r ".data[$i].b64_json" "$BODY_FILE" | base64 -d > "$OUTFILE" \
        || die "failed to write ${OUTFILE}"
    if ! $QUIET; then
        FILE_SIZE=$(stat -c%s "$OUTFILE" 2>/dev/null || stat -f%z "$OUTFILE" 2>/dev/null)
        echo "Done (${FILE_SIZE} bytes): ${OUTFILE}" >&2
    else
        echo "$OUTFILE"
    fi
    i=$((i + 1))
done
