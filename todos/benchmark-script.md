# TODO

## Goal
Add `scripts/benchmark-llm.py` — a Python benchmark for OpenAI-compatible endpoints (llm-orch): pings the endpoint with a minimal query, then sends prompts of increasing size, each with a unique random prefix (no prefix-cache reuse), measuring prompt processing (TTFT via streaming, wall-clock with `max_tokens=1`, llama.cpp `timings` field when present) and optional generation throughput.

## Tasks

### 1. Script skeleton + CLI
- [x] Create `scripts/benchmark-llm.py` (Python 3, `httpx`): argparse CLI with `--model`, `--sizes` (token counts, default `256 1024 4096 16384`), `--repeats` (default 1), `--max-tokens` (default 1), `--stream/--no-stream` (default stream), `--prompts`, `--prompt-file`, `--url` (env `LLM_ORCH_URL`, default `http://127.0.0.1:8080`), `--api-key` (env `APIKEY_LLM_ORCH`; omit header if unset), `--csv FILE`, `--timeout`.
- [x] Fail with a clear message if `httpx` is not installed.

### 2. Ping step
- [x] Send minimal query (`"ping"`, `max_tokens=1`, non-streamed) first; check HTTP 200 + `model` field matches; print latency (includes model cold-start); exit non-zero with a helpful error if ping fails.

### 3. Prompt builder
- [x] For each size: content = unique random prefix (e.g. `<bench-<uuid>> `, different for every request) + filler = repeated single word ("apple " × N) as an approximation of N tokens; actual count comes from `usage.prompt_tokens`.
- [x] Support custom content via `--prompts` (list of raw strings, one query each) and `--prompt-file` (file per line, `--` = read file); unique prefix still prepended.

### 4. Request runner + measurement
- [x] Single request path used by ping/bench: stream mode measures TTFT (time to first SSE chunk with content delta) and total time; non-stream measures total time only.
- [x] Parse `usage` (prompt_tokens, completion_tokens) and llama.cpp `timings` extension (`prompt_ms`, `prompt_per_second`, `predicted_ms`, `predicted_per_second`) when present; PP time = `timings.prompt_ms` if present, else TTFT (stream) or total time (non-stream, max_tokens=1).
- [x] Decode metrics when `completion_tokens > 0`: gen tok/s = completion_tokens / (total − TTFT).

### 5. Reporting
- [x] Per-request results table: size, prompt_tokens, TTFT, total, PP ms, PP tok/s, gen tok/s; with `--repeats > 1`, add mean/median/min rows per size.
- [x] `--csv FILE` writes one row per request (append mode).

### 6. Devshell deps
- [x] Add `python3` and `python3Packages.httpx` to `buildInputs` in `flake.nix` devShell.

### 7. Smoke test
- [x] Run the script against a locally spawned `llm-orch` + `llm-orch-stub-backend` (test config in /tmp) with small sizes; verify ping, sizing ladder, unique prefixes, TTFT/timings/fallback paths, `--repeats`, `--csv`, and exit codes.

## Notes
- Follows existing scripts' env-var convention (`LLM_ORCH_URL`, `APIKEY_LLM_ORCH`).
- Every request gets a fresh random prefix so KV/prefix caches never apply — that is the point of the measurement.
- llama.cpp `timings` is a non-standard passthrough field; llm-orch forwards it untouched (per AGENTS.md forwarding philosophy).
- No new Rust code; script is standalone (stdlib + httpx only).
