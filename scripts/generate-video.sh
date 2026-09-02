#!/usr/bin/env bash
# Generate a video via llm-orch's /v1/video/generations endpoint.
#
# Usage:
#   scripts/generate-video.sh [options] [-p PROMPT | -f PROMPT_FILE]
#
# Options:
#   -p PROMPT          Prompt text (inline; overrides -f).
#   -f PROMPT_FILE     Read prompt from file (use "-" for stdin).
#   -i IMAGE           Optional init image (first frame). Base64-encoded and
#                      sent as init_image for image-to-video generation.
#   -m MODEL           Model name (default: minimax-h3).
#   -W WIDTH           Video width  (default: 864).
#   -H HEIGHT          Video height (default: 480).
#   --video-frames N   Number of frames (default: 33; aligned to 17k+5 grid).
#   --fps N            Frames per second (default: 24; MiniMax-H3 is fixed at 24).
#   --seed N           RNG seed (-1 = random; default: -1).
#   --cfg-scale F      Classifier-free guidance scale (default: 1.0).
#   -o OUTDIR          Output directory (default: ./videos).
#   -q                Quiet — print only the output path.
#
# Environment:
#   APIKEY_LLM_ORCH    API key (required).
#   LLM_ORCH_URL       Base URL, default http://127.0.0.1:8080.
#
# Examples:
#   export APIKEY_LLM_ORCH=sk-...
#   scripts/generate-video.sh -p "A cat surfing on a tropical wave"
#   echo "A cat surfing" | scripts/generate-video.sh -f -
#   scripts/generate-video.sh -p "A cat surfing" -o /tmp/vids -W 1280 -H 720 --video-frames 65
#   scripts/generate-video.sh -p "An agentic city at dusk" -i ./agentic-city.png
set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
: "${APIKEY_LLM_ORCH:?APIKEY_LLM_ORCH is required}"
BASE="${LLM_ORCH_URL:-http://127.0.0.1:8080}"

MODEL="minimax-h3"
WIDTH=864
HEIGHT=480
VIDEO_FRAMES=33
FPS=24
SEED=-1
CFG_SCALE=1.0
OUTDIR="./videos"
QUIET=false
INIT_IMAGE=""

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
        --video-frames) shift; VIDEO_FRAMES="$1" ;;
        --fps) shift; FPS="$1" ;;
        --seed) shift; SEED="$1" ;;
        --cfg-scale) shift; CFG_SCALE="$1" ;;
        -o) shift; OUTDIR="$1" ;;
        -i) shift; INIT_IMAGE="$1" ;;
        -q) QUIET=true ;;
        -h|--help)
            sed -n '/^# Usage:/,/^set -euo pipefail/p' "$0" \
                | sed 's/^# \?//; /set -euo pipefail/d'
            exit 0
            ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

# ── Resolve prompt ───────────────────────────────────────────────────────────
if [ -z "$PROMPT" ] && [ -n "$PROMPT_FILE" ]; then
    if [ "$PROMPT_FILE" = "-" ]; then
        PROMPT=$(cat)
    else
        PROMPT=$(cat "$PROMPT_FILE")
    fi
fi

if [ -z "$PROMPT" ]; then
    echo "error: prompt is required (-p PROMPT or -f PROMPT_FILE)" >&2
    exit 1
fi

# ── Output path ──────────────────────────────────────────────────────────────
mkdir -p "$OUTDIR"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTFILE="${OUTDIR}/llm-orch-video-${TIMESTAMP}.webm"

# ── Resolve init image ──────────────────────────────────────────────────────
INIT_IMAGE_B64=""
if [ -n "$INIT_IMAGE" ]; then
    if [ ! -f "$INIT_IMAGE" ]; then
        echo "error: init image not found: ${INIT_IMAGE}" >&2
        exit 1
    fi
    INIT_IMAGE_B64=$(base64 -w0 "$INIT_IMAGE")
    if [ -z "$INIT_IMAGE_B64" ]; then
        echo "error: failed to encode init image" >&2
        exit 1
    fi
    if ! $QUIET; then
        echo "  init image: ${INIT_IMAGE} ($(wc -c < "$INIT_IMAGE") bytes)" >&2
    fi
fi

# ── Build JSON payload ───────────────────────────────────────────────────────
if [ -n "$INIT_IMAGE_B64" ]; then
    PAYLOAD=$(jq -nc \
        --arg model "$MODEL" \
        --arg prompt "$PROMPT" \
        --arg init_image "$INIT_IMAGE_B64" \
        --argjson width "$WIDTH" \
        --argjson height "$HEIGHT" \
        --argjson video_frames "$VIDEO_FRAMES" \
        --argjson fps "$FPS" \
        --argjson seed "$SEED" \
        --argjson cfg_scale "$CFG_SCALE" \
        '{
            model: $model,
            prompt: $prompt,
            init_image: $init_image,
            width: $width,
            height: $height,
            video_frames: $video_frames,
            fps: $fps,
            seed: $seed,
            output_format: "webm",
            sample_params: {
                guidance: {
                    txt_cfg: $cfg_scale
                }
            }
        }')
else
    PAYLOAD=$(jq -nc \
        --arg model "$MODEL" \
        --arg prompt "$PROMPT" \
        --argjson width "$WIDTH" \
        --argjson height "$HEIGHT" \
        --argjson video_frames "$VIDEO_FRAMES" \
        --argjson fps "$FPS" \
        --argjson seed "$SEED" \
        --argjson cfg_scale "$CFG_SCALE" \
        '{
            model: $model,
            prompt: $prompt,
            width: $width,
            height: $height,
            video_frames: $video_frames,
            fps: $fps,
            seed: $seed,
            output_format: "webm",
            sample_params: {
                guidance: {
                    txt_cfg: $cfg_scale
                }
            }
        }')
fi

# ── Submit ───────────────────────────────────────────────────────────────────
if ! $QUIET; then
    echo "Generating video with model '${MODEL}'..." >&2
    echo "  frames=${VIDEO_FRAMES}  ${WIDTH}x${HEIGHT}  fps=${FPS}" >&2
    echo "  output: ${OUTFILE}" >&2
fi

HTTP_CODE=""
BODY_FILE=$(mktemp)

HTTP_CODE=$(curl -sS -w '%{http_code}' \
    -H "Authorization: Bearer ${APIKEY_LLM_ORCH}" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    -o "$BODY_FILE" \
    "${BASE}/v1/video/generations")

if [ "$HTTP_CODE" -ge 400 ]; then
    ERROR_MSG=$(cat "$BODY_FILE")
    rm -f "$BODY_FILE"
    echo "HTTP ${HTTP_CODE}:" >&2
    echo "${ERROR_MSG}" >&2
    exit 1
fi

# ── Save output ──────────────────────────────────────────────────────────────
mv "$BODY_FILE" "$OUTFILE"

if ! $QUIET; then
    FILE_SIZE=$(stat -c%s "$OUTFILE" 2>/dev/null || stat -f%z "$OUTFILE" 2>/dev/null)
    echo "Done (${FILE_SIZE} bytes): ${OUTFILE}" >&2
else
    echo "$OUTFILE"
fi