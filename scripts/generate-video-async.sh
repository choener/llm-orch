#!/usr/bin/env bash
# Generate a video through llm-orch's *async* sd-server pathway:
#   POST /sdcpp/v1/vid_gen  ->  202 + job id
#   GET  /sdcpp/v1/jobs/{id} ->  queued | generating | completed | failed | cancelled
#   POST /sdcpp/v1/jobs/{id}/cancel
#
# Why async: /v1/video/generations holds a single backend exchange open for
# the whole render and llm-orch bounds it with server.backend_total_timeout_secs
# (default 900 s).  A render longer than that returns 504 "video generation did
# not complete within 900s" — and llm-orch does NOT cancel the job, so sd-server
# keeps working on the orphan while sd-server's internal queue (max_concurrent: 1)
# blocks every following request.  Here every HTTP request is short: submit,
# poll, fetch.  Wait as long as the render needs, and disconnect any time —
# the job survives (that is the point).
#
# Usage:
#   scripts/generate-video-async.sh [options] [-p PROMPT | -f PROMPT_FILE]
#   scripts/generate-video-async.sh --job-id JOB_ID            # resume + download
#   scripts/generate-video-async.sh --job-id JOB_ID --cancel   # cancel a job
#   scripts/generate-video-async.sh --no-wait -p "..."         # fire and forget
#
# Options (generation — same defaults as generate-video.sh):
#   -p PROMPT          Prompt text (inline; overrides -f).
#   -f PROMPT_FILE     Read prompt from file (use "-" for stdin).
#   -i IMAGE           Init image (first frame, I2VA). Base64-encoded.
#   --end-img IMAGE    Last frame (FL2VA). Base64-encoded.
#   --ref-image IMAGE  Ref2VA reference image; repeatable.
#   -m MODEL           Model name (default: minimax-h3).
#   -W WIDTH           Video width  (default: 864).
#   -H HEIGHT          Video height (default: 480).
#   --video-frames N   Number of frames (default: 33). Note: MiniMax-H3 rounds
#                      up to the 17k+5 grid; the core path then uses the
#                      largest 4n+1 not above it (33 -> 33, 34 -> 33, 32 -> 29).
#   --fps N            Frames per second (default: 24; MiniMax-H3 is fixed at 24).
#   --seed N           RNG seed (-1 = random; default: -1).
#   --cfg-scale F      Classifier-free guidance scale (default: 1.0).
#   --steps N          Sampling steps (default: backend default). Fewer steps
#                      is the cheapest way to fit a render in a time budget.
#   --sampling-method  Sampling method, e.g. euler, euler_a (default: backend default).
#   --format FMT       Container: webm | webp | avi (default: webm).
#   -o OUTDIR          Output directory (default: ./videos).
#   -q                 Quiet — print only the output path (or the job id).
#
# Options (async control):
#   --poll-interval S  Seconds between polls (default: 5).
#   --max-wait S       Give up waiting after S seconds (default: 0 = wait forever).
#   --no-wait          Submit only; print the job id and exit.
#   --job-id ID        Attach to an existing job instead of submitting.
#   --cancel           With --job-id: cancel the job and exit.
#   --cancel-on-timeout Cancel the job when --max-wait expires (default: leave it
#                      running — the render is still worth fetching later).
#   --fail-after N     Consecutive poll failures tolerated (default: 10).
#
# Environment:
#   APIKEY_LLM_ORCH    API key (required).
#   LLM_ORCH_URL       Base URL, default http://127.0.0.1:8080.
#
# Examples:
#   export APIKEY_LLM_ORCH=sk-...
#   scripts/generate-video-async.sh -p "A cat surfing on a tropical wave"
#   echo "A cat surfing" | scripts/generate-video-async.sh -f - --steps 8
#   scripts/generate-video-async.sh -p "An agentic city at dusk" -W 1280 -H 720 --video-frames 124
#   scripts/generate-video-async.sh -p "A cat surfing" --no-wait          # -> job id
#   scripts/generate-video-async.sh --job-id job_01HTXYZVID               # collect later
#   scripts/generate-video-async.sh --job-id job_01HTXYZVID --cancel      # abort it
#
# Notes:
#   - The model is selected with the X-LLM-Model request header, which llm-orch's
#     /sdcpp/v1/* passthrough requires unless server.default_diffusion_model is set.
#   - A poll that fails, is queued behind another request, or is otherwise
#     delayed is treated as "no answer right now" and retried — never as a
#     generation failure.  One flaky poll must not throw away minutes of GPU work.
#   - Ctrl-C leaves the job running.  The job id is printed so you can come back
#     with --job-id.  Use --cancel-on-timeout / --cancel to abort deliberately.
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
OUTPUT_FORMAT="webm"

INIT_IMAGE=""
END_IMAGE=""
REF_IMAGES=()

SAMPLE_METHOD=""
SAMPLE_STEPS=0

POLL_INTERVAL=5
MAX_WAIT=0
NO_WAIT=false
JOB_ID=""
DO_CANCEL=false
CANCEL_ON_TIMEOUT=false
FAIL_AFTER=10
REQ_TIMEOUT=60

# ── Parse arguments ──────────────────────────────────────────────────────────
PROMPT=""
PROMPT_FILE=""

while [ $# -gt 0 ]; do
    case "$1" in
        -p) shift; PROMPT="$1" ;;
        -f) shift; PROMPT_FILE="$1" ;;
        -m) shift; MODEL="$1" ;;
        -i) shift; INIT_IMAGE="$1" ;;
        --end-img) shift; END_IMAGE="$1" ;;
        --ref-image) shift; REF_IMAGES+=("$1") ;;
        -W) shift; WIDTH="$1" ;;
        -H) shift; HEIGHT="$1" ;;
        --video-frames) shift; VIDEO_FRAMES="$1" ;;
        --fps) shift; FPS="$1" ;;
        --seed) shift; SEED="$1" ;;
        --cfg-scale) shift; CFG_SCALE="$1" ;;
        --steps) shift; SAMPLE_STEPS="$1" ;;
        --sampling-method) shift; SAMPLE_METHOD="$1" ;;
        --format) shift; OUTPUT_FORMAT="$1" ;;
        -o) shift; OUTDIR="$1" ;;
        --poll-interval) shift; POLL_INTERVAL="$1" ;;
        --max-wait) shift; MAX_WAIT="$1" ;;
        --fail-after) shift; FAIL_AFTER="$1" ;;
        --no-wait) NO_WAIT=true ;;
        --job-id) shift; JOB_ID="$1" ;;
        --cancel) DO_CANCEL=true ;;
        --cancel-on-timeout) CANCEL_ON_TIMEOUT=true ;;
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

log() { $QUIET || printf '%s\n' "$*" >&2; }
PROGRESS_SHOWN=false
end_progress() { $PROGRESS_SHOWN && { printf '\n' >&2; PROGRESS_SHOWN=false; }; return 0; }

TMPDIR_RUN=$(mktemp -d)
JOB_FILE="${TMPDIR_RUN}/job.json"
cleanup() { rm -rf "$TMPDIR_RUN"; }
trap cleanup EXIT

die() { $PROGRESS_SHOWN && printf '\n' >&2; printf 'error: %s\n' "$*" >&2; exit 1; }

# ── HTTP helpers ─────────────────────────────────────────────────────────────
# api_curl <method> <url> [json-body] -> response body lands in $JOB_FILE and
# the 3-digit status code is printed on stdout ("000" if it never completed).
api_curl() {
    local method="$1" url="$2" body="${3:-}"
    local args=(
        -sS -X "$method" --max-time "$REQ_TIMEOUT"
        -H "Authorization: Bearer ${APIKEY_LLM_ORCH}"
        -H "X-LLM-Model: ${MODEL}"
        -o "$JOB_FILE" -w '%{http_code}'
    )
    if [ -n "$body" ]; then
        args+=(-H 'Content-Type: application/json' -d "$body")
    fi
    local out
    out=$(curl "${args[@]}" "$url" 2>/dev/null) || true
    out=${out//[!0-9]/}
    if [ -z "$out" ]; then
        printf '000'
    else
        printf '%s' "${out: -3}"
    fi
}

# ── Resume / cancel an existing job ─────────────────────────────────────────
if [ -n "$JOB_ID" ]; then
    if $DO_CANCEL; then
        CODE=$(api_curl POST "${BASE}/sdcpp/v1/jobs/${JOB_ID}/cancel" '{}')
        case "$CODE" in
            2*) echo "cancelled job ${JOB_ID}" >&2 ;;
            404) die "job ${JOB_ID} not found (already purged?)" ;;
            409) die "job ${JOB_ID} cannot be cancelled in its current state" ;;
            410) die "job ${JOB_ID} is gone" ;;
            *)   die "cancel failed (HTTP ${CODE}): $(cat "$JOB_FILE" 2>/dev/null)" ;;
        esac
        exit 0
    fi
    NO_WAIT=false
    PROMPT="(attaching to existing job ${JOB_ID})"
fi

# ── Resolve prompt (only when submitting) ────────────────────────────────────
if [ -z "$JOB_ID" ]; then
    if [ -z "$PROMPT" ] && [ -n "$PROMPT_FILE" ]; then
        if [ "$PROMPT_FILE" = "-" ]; then
            PROMPT=$(cat)
        else
            PROMPT=$(cat "$PROMPT_FILE")
        fi
    fi
    if [ -z "$PROMPT" ]; then
        die "prompt is required (-p PROMPT or -f PROMPT_FILE), or --job-id to resume"
    fi
fi

# ── Encode input media ───────────────────────────────────────────────────────
encode_file() {
    local path="$1" label="$2" b64
    [ -f "$path" ] || die "${label} not found: ${path}"
    b64=$(base64 -w0 "$path") || die "failed to encode ${label}: ${path}"
    [ -n "$b64" ] || die "empty ${label}: ${path}"
    log "  ${label}: ${path} ($(wc -c < "$path") bytes)"
    printf '%s' "$b64"
}

INIT_IMAGE_B64=""
[ -n "$INIT_IMAGE" ] && INIT_IMAGE_B64=$(encode_file "$INIT_IMAGE" "init image")
END_IMAGE_B64=""
[ -n "$END_IMAGE" ] && END_IMAGE_B64=$(encode_file "$END_IMAGE" "end image")

REF_IMAGES_JSON="[]"
if [ ${#REF_IMAGES[@]} -gt 0 ]; then
    for img in "${REF_IMAGES[@]}"; do
        ref_b64=$(encode_file "$img" "ref image")
        REF_IMAGES_JSON=$(printf '%s' "$REF_IMAGES_JSON" | jq -c --arg v "$ref_b64" '. + [$v]')
    done
fi

# ── Submit ───────────────────────────────────────────────────────────────────
if [ -z "$JOB_ID" ]; then
    PAYLOAD=$(jq -nc \
        --arg prompt "$PROMPT" \
        --arg output_format "$OUTPUT_FORMAT" \
        --arg init_image "$INIT_IMAGE_B64" \
        --arg end_image "$END_IMAGE_B64" \
        --arg sample_method "$SAMPLE_METHOD" \
        --argjson ref_images "$REF_IMAGES_JSON" \
        --argjson width "$WIDTH" \
        --argjson height "$HEIGHT" \
        --argjson video_frames "$VIDEO_FRAMES" \
        --argjson fps "$FPS" \
        --argjson seed "$SEED" \
        --argjson cfg "$CFG_SCALE" \
        --argjson steps "$SAMPLE_STEPS" \
        '{
            prompt: $prompt,
            width: $width,
            height: $height,
            video_frames: $video_frames,
            fps: $fps,
            seed: $seed,
            output_format: $output_format,
            sample_params: { guidance: { txt_cfg: $cfg } }
        }
        | if $init_image != "" then .init_image = $init_image else . end
        | if $end_image != "" then .end_image = $end_image else . end
        | if ($ref_images | length) > 0 then .ref_images = $ref_images else . end
        | if $sample_method != "" then .sample_params.sample_method = $sample_method else . end
        | if $steps > 0 then .sample_params.sample_steps = $steps else . end')

    if ! $QUIET; then
        echo "Submitting video job to '${MODEL}' (async)..." >&2
        echo "  frames=${VIDEO_FRAMES}  ${WIDTH}x${HEIGHT}  fps=${FPS}  format=${OUTPUT_FORMAT}" >&2
    fi

    CODE=$(api_curl POST "${BASE}/sdcpp/v1/vid_gen" "$PAYLOAD")
    BODY=$(cat "$JOB_FILE" 2>/dev/null || true)
    case "$CODE" in
        2*) : ;;
        000) die "could not reach ${BASE} — is llm-orch running?" ;;
        *)   die "job submission failed (HTTP ${CODE}): ${BODY}" ;;
    esac

    JOB_ID=$(jq -r '.id // empty' <<<"$BODY")
    [ -n "$JOB_ID" ] || die "no job id in submission response: ${BODY}"
    log "  job: ${JOB_ID}"

    if $NO_WAIT; then
        if $QUIET; then
            echo "$JOB_ID"
        else
            echo "Submitted. Collect later with:" >&2
            echo "  $0 --job-id ${JOB_ID}" >&2
        fi
        exit 0
    fi
fi

# ── Poll ─────────────────────────────────────────────────────────────────────
mkdir -p "$OUTDIR"
START=$(date +%s)
deadline=""
[ "$MAX_WAIT" -gt 0 ] && deadline=$((START + MAX_WAIT))

fail_count=0
STATUS="queued"
PREV_STATUS=""
LAST_LOGGED=-1

# Ctrl-C detaches, it does not abort: sd-server keeps rendering, and the job is
# still collectable (or cancellable) with the id printed here.
on_int() {
    end_progress
    printf 'interrupted — job %s keeps running.\n' "$JOB_ID" >&2
    printf '  collect:  %s --job-id %s\n' "$0" "$JOB_ID" >&2
    printf '  cancel:   %s --job-id %s --cancel\n' "$0" "$JOB_ID" >&2
    exit 130
}
trap on_int INT

while :; do
    if [ -n "$deadline" ] && [ "$(date +%s)" -ge "$deadline" ]; then
        if $CANCEL_ON_TIMEOUT; then
            api_curl POST "${BASE}/sdcpp/v1/jobs/${JOB_ID}/cancel" '{}' >/dev/null 2>&1 || true
            die "gave up after ${MAX_WAIT}s and cancelled job ${JOB_ID}"
        fi
        die "gave up waiting after ${MAX_WAIT}s — job ${JOB_ID} is still running; collect it with: $0 --job-id ${JOB_ID}"
    fi

    sleep "$POLL_INTERVAL"

    CODE=$(api_curl GET "${BASE}/sdcpp/v1/jobs/${JOB_ID}")
    if [ "$CODE" != "200" ]; then
        # Transient by design: llm-orch may queue this poll behind another
        # request, sd-server may be mid-render, the job may not be visible yet.
        fail_count=$((fail_count + 1))
        if [ "$fail_count" -eq 1 ]; then
            log "  poll failed (HTTP ${CODE}); retrying"
        fi
        if [ "$fail_count" -ge "$FAIL_AFTER" ]; then
            die "job ${JOB_ID} unreachable after ${fail_count} polls (last HTTP ${CODE}); it may still be running: $0 --job-id ${JOB_ID}"
        fi
        continue
    fi
    if [ "$fail_count" -gt 0 ]; then
        log "  poll reachable again after ${fail_count} failures"
        fail_count=0
    fi

    STATUS=$(jq -r '.status // "unknown"' "$JOB_FILE" 2>/dev/null || echo "invalid")
    ELAPSED=$(( $(date +%s) - START ))

    case "$STATUS" in
        completed)
            break
            ;;
        failed)
            end_progress
            jq -r '"video generation failed: " + ((.error.code // "?") + ": " + (.error.message // "no detail"))' "$JOB_FILE" >&2
            exit 1
            ;;
        cancelled)
            die "job ${JOB_ID} was cancelled"
            ;;
        queued|generating)
            qp=$(jq -r '.queue_position // 0' "$JOB_FILE" 2>/dev/null || echo 0)
            step=$(jq -r 'if .preview then "\(.preview.step // 0)/\(.preview.total_steps // 0)" else "" end' "$JOB_FILE" 2>/dev/null)
            if ! $QUIET; then
                # \r progress only on a terminal.  When stderr is a pipe or a
                # log file, in-place updates would pile up into one monstrous
                # line — emit a plain line on status change and at 30 s beats.
                if [ -t 2 ]; then
                    printf '\r  [%4ds] %-10s%s' "$ELAPSED" "$STATUS" "${qp:+ (queue ${qp})}${step:+ step ${step}}        " >&2
                    PROGRESS_SHOWN=true
                elif [ "$STATUS" != "$PREV_STATUS" ] || [ $(( ELAPSED - LAST_LOGGED )) -ge 30 ]; then
                    log "  [${ELAPSED}s] ${STATUS}${qp:+ (queue ${qp})}${step:+ step ${step}}"
                    LAST_LOGGED=$ELAPSED
                    PREV_STATUS=$STATUS
                fi
            fi
            continue
            ;;
        invalid)
            log "  unpollable job response"
            continue
            ;;
        *)
            log "  unexpected job status '${STATUS}'"
            continue
            ;;
    esac
done

if ! $QUIET; then end_progress; fi

# ── Save output ──────────────────────────────────────────────────────────────
EXT=$(jq -r '.result.output_format // empty' "$JOB_FILE")
case "$EXT" in
    webm|webp|avi) : ;;
    *) case "$(jq -r '.result.mime_type // ""' "$JOB_FILE")" in
           image/webp) EXT=webp ;;
           video/x-msvideo) EXT=avi ;;
           *) EXT=webm ;;
       esac ;;
esac

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTFILE="${OUTDIR}/llm-orch-video-${TIMESTAMP}.${EXT}"

if ! jq -e '.result.b64_json | type == "string"' "$JOB_FILE" >/dev/null 2>&1; then
    die "completed job has no result payload: $(head -c 400 "$JOB_FILE")"
fi

jq -r '.result.b64_json' "$JOB_FILE" | base64 -d > "$OUTFILE" || die "failed to write ${OUTFILE}"

FILE_SIZE=$(stat -c%s "$OUTFILE" 2>/dev/null || stat -f%z "$OUTFILE" 2>/dev/null || echo "?")
FRAMES=$(jq -r '.result.frame_count // "?"' "$JOB_FILE")
FPS_OUT=$(jq -r '.result.fps // "?"' "$JOB_FILE")

if ! $QUIET; then
    echo "Done in $(( $(date +%s) - START ))s: ${FILE_SIZE} bytes, ${FRAMES} frames @ ${FPS_OUT} fps" >&2
    echo "${OUTFILE}" >&2
else
    echo "$OUTFILE"
fi
