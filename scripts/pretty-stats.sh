#!/usr/bin/env bash
# pretty-stats.sh — human-friendly rendering of an llm-orch /v1/stats document.
#
# Fetches the auth-gated /v1/stats endpoint of an llm-orch instance and
# renders the per-model token/latency counters and the GPU snapshot as
# aligned tables.  With --file it renders a saved document instead (the
# stats.json the daemon persists, or an MQTT-published stats payload — an
# optional `rates` block is rendered when present).
#
# Environment:
#   LLM_STATS_URL      llm-orch base URL (e.g. http://127.0.0.1:8888) or the
#                      full /v1/stats URL
#   LLM_STATS_APIKEY   an API key llm-orch accepts
#
# Options:
#   --raw              print the untouched JSON document, pretty-printed
#   --watch [SECS]     redraw every SECS seconds (default 5); fetch errors are
#                      reported and retried on the next tick
#   --file FILE        render a saved stats document instead of the endpoint
#   --model PATTERN    keep only models whose name matches PATTERN (regex)
#   -h, --help         show this help
#
# Dependencies: bash, curl, jq.

set -euo pipefail

PROG=$(basename "$0")

die() { printf '%s: error: %s\n' "$PROG" "$*" >&2; exit 1; }

usage() {
  cat <<EOF
Usage: ${PROG} [--raw] [--watch [SECS]] [--file FILE] [--model PATTERN]

Render an llm-orch /v1/stats document (per-model token and latency
counters, GPU snapshot) as aligned tables.

  --raw              print the untouched JSON document, pretty-printed
  --watch [SECS]     redraw every SECS seconds (default 5)
  --file FILE        render a saved stats document instead of the endpoint
  --model PATTERN     keep only models whose name matches PATTERN (regex)
  -h, --help          show this help

Environment:
  LLM_STATS_URL      llm-orch base URL (e.g. http://127.0.0.1:8888) or the
                     full /v1/stats URL
  LLM_STATS_APIKEY   an API key llm-orch accepts
EOF
}

# ── jq rendering program ────────────────────────────────────────────────────
# Left-align helpers (pad with spaces to width $n; never truncate).
JQ=$(cat <<'EOF'
def lpad($n): . + (" " * ([$n - length, 0] | max));
def rpad($n): (" " * ([$n - length, 0] | max)) + .;
def commas: tostring | explode | reverse
  | [range(0; length; 3) as $i | .[$i:$i + 3] | reverse | implode]
  | reduce .[] as $g (""; if . == "" then $g else $g + "," + . end);
def pct($a; $b):
  if $a == null or $b == null or $b == 0 then null
  else (100 * $a / $b | floor) end;
def msfmt:
  if . == null then "-"
  elif . >= 60000 then "\((. / 6000 | floor) / 10) min"
  elif . >= 1000 then "\((. / 100 | floor) / 10) s"
  else "\(.) ms" end;
def dec2($x): (($x * 100) | floor) as $c
  | "\($c / 100 | floor).\(($c % 100 | tostring) | if length == 1 then "0" + . else . end)";
def ago($now):
  if . == null then "-"
  else (if $now > . then $now - . else 0 end) as $s
  | if $s < 60 then "\($s | floor) s"
    elif $s < 3600 then "\($s / 60 | floor) min"
    elif $s < 86400 then "\($s / 3600 | floor) h"
    else "\($s / 86400 | floor) d" end
  end;
def ts: if . == null or . == 0 then "—" else todate end;
def section($t): "── " + $t + " " + ("─" * 62);
def bar($p): ([([($p / 10) | floor, 10] | min), 0] | max) as $n
  | ("█" * $n) + ("░" * (10 - $n));

# Render a matrix of strings: column 0 left-aligned, the rest right-aligned,
# each column as wide as its widest cell.
def table($rows):
  ($rows | transpose | map(map(length) | max)) as $w
  | [ $rows[]
      | . as $cells
      | [ ($cells | length) as $nc
          | range(0; $nc) as $i
          | (if $i == 0
             then ($cells[$i] | lpad($w[$i]))
             else ($cells[$i] | rpad($w[$i])) end) ]
      | join("  ") ];

. as $doc
| now as $now
| ($doc.models // {}) as $models
| ($models | to_entries | sort_by(.key)) as $all
| ($all | map(select($pat == "" or (.key | test($pat)))) ) as $entries
| ($doc.gpus // []) as $gpus
| ($doc.rates // null) as $rates
| ($rates | if . == null then [] else (.models // {}) | to_entries | sort_by(.key) end) as $rentries
| ($entries | length) as $n
| ($all | length) as $an

# ── model rows ──
| ([[ "MODEL", "REQ TOT (BOOT)", "PROMPT TOK", "CACHE%", "COMPL TOK",
     "TOK/S", "TTFT AVG (MIN–MAX) MS", "DUR AVG", "LAST ACTIVE" ]]
   + [ $entries[]
       | .key as $k
       | .value as $m
       | $m.total as $t
       | ($m.boot // $t) as $b
       | (if $t.ttft_count > 0
          then ($t.ttft_sum_ms / $t.ttft_count | floor) else null end) as $ttft
       | (if $t.duration_sum_ms > 0
          then ($t.completion_tokens * 1000 / $t.duration_sum_ms | floor)
          else null end) as $tps
       | pct($t.cached_tokens; $t.prompt_tokens) as $cache
       | [ $k,
           "\($t.requests | commas) (\($b.requests | commas))",
           ($t.prompt_tokens | commas),
           (if $cache == null then "-" else "\($cache)%" end),
           ($t.completion_tokens | commas),
           (if $tps == null then "-" else ($tps | commas) end),
           (if $ttft == null then "-"
             else "\($ttft | commas) (\($t.ttft_min_ms // "-")–\($t.ttft_max_ms // "-"))" end),
           ($t.duration_avg_ms | msfmt),
           ($t.last_active | ago($now)) ] ]) as $model_matrix

# ── GPU rows ──
| ([[ "GPU", "VENDOR", "VRAM USED / TOTAL", "UTILISATION", "TEMP", "POWER",
     "BUSY", "CLOCKS" ]]
   + [ $gpus[]
       | . as $g
       | [ "\($g.node // "local"):\($g.index)",
           ($g.vram_vendor // "-"),
           "\(dec2($g.vram_used_bytes / 1073741824)) / \(dec2($g.vram_total_bytes / 1073741824)) GiB",
           "[\(bar($g.vram_util_pct))] \($g.vram_util_pct | floor)%",
           (if $g.temperature_c == null then "-" else "\($g.temperature_c | floor) °C" end),
           (if $g.power_w == null then "-" else "\($g.power_w | floor) W" end),
           (if $g.gpu_busy_pct == null then "-" else "\($g.gpu_busy_pct)%" end),
           (if $g.sclk_mhz == null then "-"
             else "\($g.sclk_mhz)\(if $g.mclk_mhz == null then "" else "/\($g.mclk_mhz)" end) MHz" end) ] ]) as $gpu_matrix

# ── rates rows ──
| ([[ "MODEL", "REQ/MIN", "PROMPT TOK/MIN", "COMPL TOK/MIN", "TOTAL TOK/MIN" ]]
   + [ $rentries[]
       | [ .key,
           (.value.requests_per_min | tostring),
           (.value.prompt_tokens_per_min | tostring),
           (.value.completion_tokens_per_min | tostring),
           (.value.tokens_per_min | tostring) ] ]) as $rates_matrix

# ── assemble ──
| [ "llm-orch stats — document v\($doc.version) — \(if $doc.enabled then "enabled" else "stats collection disabled" end)",
    (if $doc.enabled then empty
      else "(no stats: section in config.yaml — counters are not recorded)" end),
    "started \($doc.started_at | ts)   updated \($doc.updated_at | ts)   \($an) model\(if $an == 1 then "" else "s" end)\(if $pat != "" then ", \($n) matching \"\($pat)\"" else "" end), \($gpus | length) GPU\(if ($gpus | length) == 1 then "" else "s" end)" ]
  + [ "", section("Models") ]
  + (if $n == 0
     then [ if $an == 0 then "(no models recorded yet)" else "(no models match \"\($pat)\")" end ]
     else table($model_matrix) end)
  + (if ($gpus | length) == 0 then []
     else [ "", section("GPUs") ] + table($gpu_matrix) end)
  + (if ($rentries | length) == 0 then []
     else [ "", section("Rates — last \($rates.window_secs) s") ] + table($rates_matrix) end)
  | join("\n")
EOF
)

# ── options ────────────────────────────────────────────────────────────────
raw=0 watch=0 interval=5 src=- pattern=""
while (($#)); do
  case $1 in
    --raw|-r) raw=1 ;;
    --watch|-w)
      watch=1
      if [[ ${2:-} =~ ^[1-9][0-9]*$ ]]; then interval=$2; shift; fi ;;
    --file|-f)
      [[ -n ${2:-} ]] || die "--file needs a path"
      src=$2; shift ;;
    --model|-m)
      [[ -n ${2:-} ]] || die "--model needs a pattern"
      pattern=$2; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
  shift
done

if [[ $src == - ]]; then
  : "${LLM_STATS_URL:?LLM_STATS_URL is not set — set it to the llm-orch base URL (e.g. http://127.0.0.1:8888) or the full /v1/stats URL}"
  : "${LLM_STATS_APIKEY:?LLM_STATS_APIKEY is not set — set it to an API key llm-orch accepts}"
fi

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

# Fetch the document into $tmp; returns non-zero with a message on stderr.
fetch() {
  local url code
  url=${LLM_STATS_URL%/}
  [[ $url == */v1/stats ]] || url+=/v1/stats
  if ! code=$(curl -sS --max-time 10 -o "$tmp" -w '%{http_code}' \
                -H "Authorization: Bearer ${LLM_STATS_APIKEY}" "$url"); then
    return 1
  fi
  if [[ $code != 200 ]]; then
    printf 'HTTP %s from %s\n%s\n' "$code" "$url" "$(cat "$tmp")" >&2
    return 1
  fi
}

render() { # $1 = document file
  if ((raw)); then
    jq . "$1"
  else
    jq -r --arg pat "$pattern" "$JQ" "$1" || return 1
  fi
}

once() {
  if [[ $src == - ]]; then
    fetch || { printf 'request failed (is llm-orch up and LLM_STATS_APIKEY valid?)\n' >&2; return 1; }
    render "$tmp"
  else
    [[ -r $src ]] || { printf 'cannot read %s\n' "$src" >&2; return 1; }
    render "$src"
  fi
}

if ((watch)); then
  while true; do
    clear >/dev/null 2>&1 || true
    if ! once; then
      printf '\nretrying in %ss…\n' "$interval" >&2
    fi
    sleep "$interval"
  done
else
  once || exit 1
fi
