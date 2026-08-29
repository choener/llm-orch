#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible chat-completions endpoint.

Flow
----
1. Ping: a minimal query ("ping", max_tokens=1, non-streamed) to verify the
   endpoint answers and the model is reachable.  The first request also pays
   the model cold-start (llm-orch spawns backends on demand).
2. Bench: prompts of increasing size (default 256 → 1024 → 4096 → 16384
   tokens).  Every request is prefixed with a fresh random token, so
   prefix/KV caches never apply — each request really processes its full
   prompt.

Prompt processing (PP) is measured three ways; the best available wins:
  * llama.cpp `timings` (prompt_ms / prompt_per_second) when the backend
    returns it — llm-orch passes non-standard fields through untouched;
  * time-to-first-token (TTFT) in stream mode — prefill + first decode
    (for reasoning models the first reasoning token counts);
  * total wall time with --max-tokens 1 — prefill dominates a 1-token
    completion.

Generation: --max-tokens caps the generated output (default 1 = pure
prompt-processing mode); with larger values the script also reports decode
tok/s — from llama.cpp `timings.predicted_ms` when present (includes
reasoning tokens), else completion_tokens / (total − TTFT).
--stream/--no-stream controls whether TTFT is measured.

Usage
-----
    scripts/benchmark-llm.py --model qwen                     # default ladder
    scripts/benchmark-llm.py --model qwen --sizes 512 2048 8192
    scripts/benchmark-llm.py --model qwen --max-tokens 512    # + decode stats
    scripts/benchmark-llm.py --model qwen --repeats 3 --csv results.csv
    scripts/benchmark-llm.py --model qwen --prompts "hello" "custom prompt"
    scripts/benchmark-llm.py --model qwen --prompt-file prompts.txt

Environment: LLM_ORCH_URL (default http://127.0.0.1:8080), APIKEY_LLM_ORCH
(no Authorization header when unset, for keyless OpenAI-compatible servers).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import uuid
from dataclasses import dataclass

try:
    import httpx
except ImportError:
    sys.exit("benchmark-llm.py needs httpx (pip install httpx, or `nix develop`)")

DEFAULT_URL = "http://127.0.0.1:8080"
DEFAULT_SIZES = [256, 1024, 4096, 16384]
FILLER_WORD = "apple"


@dataclass
class Result:
    label: str
    repeat: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    ttft_ms: float | None = None
    total_ms: float = 0.0
    pp_ms: float | None = None
    pp_tok_s: float | None = None
    pp_source: str = "-"  # "timings" | "ttft" | "total"
    gen_tok_s: float | None = None
    gen_source: str = "-"  # "timings" | "wall"
    error: str | None = None


# ── formatting helpers ──────────────────────────────────────────────────────

def _fmt(v, spec):
    return "-" if v is None else format(v, spec)


def fmt_ms(v):
    return _fmt(v, ".1f")


def fmt_int(v):
    return _fmt(v, "d")


def fmt_rate(v):
    return _fmt(v, ",.0f")


def fmt_csv(v):
    return "" if v is None else round(v, 1)


# ── HTTP ────────────────────────────────────────────────────────────────────

def send_request(client, url, payload, stream):
    """POST a chat completion.

    Returns (obj, ttft_ms, total_ms, error).  In stream mode `obj` holds the
    last-seen usage/timings from the SSE chunks; in non-stream mode it is the
    parsed response JSON.  ttft_ms is the time to the first content delta
    (stream mode only).
    """
    t0 = time.perf_counter()
    ttft = None
    try:
        if stream:
            usage, timings = None, None
            with client.stream("POST", url, json=payload) as resp:
                if resp.status_code >= 400:
                    body = resp.read().decode("utf-8", errors="replace")[:500]
                    return None, None, time.perf_counter() - t0, f"HTTP {resp.status_code}: {body}"
                for line in resp.iter_lines():
                    line = (line or "").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    if chunk.get("timings"):
                        timings = chunk["timings"]
                    if ttft is None:
                        choices = chunk.get("choices") or []
                        delta = (choices[0].get("delta") if choices else None) or {}
                        # First streamed token: visible content, or
                        # reasoning_content for reasoning models (Qwen3 ...).
                        if delta.get("content") or delta.get("reasoning_content"):
                            ttft = time.perf_counter() - t0
            total = time.perf_counter() - t0
            return {"usage": usage, "timings": timings}, ttft, total, None
        resp = client.post(url, json=payload)
        total = time.perf_counter() - t0
        if resp.status_code >= 400:
            return None, None, total, f"HTTP {resp.status_code}: {resp.text[:500]}"
        return resp.json(), None, total, None
    except httpx.TimeoutException as exc:
        return None, None, time.perf_counter() - t0, f"timeout: {exc}"
    except httpx.HTTPError as exc:
        return None, None, time.perf_counter() - t0, f"request failed: {exc}"


# ── ping ────────────────────────────────────────────────────────────────────

def ping(client, url, model):
    """Minimal non-streamed query; returns True if the endpoint + model work."""
    print(f"ping {url} (model {model!r}) ...", flush=True)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": False,
        "temperature": 0,
    }
    obj, _ttft, total_ms, err = send_request(client, url, payload, stream=False)
    if err:
        print(f"ping FAILED ({total_ms * 1000:.0f} ms): {err}", file=sys.stderr)
        return False
    returned = obj.get("model") if isinstance(obj, dict) else None
    if not isinstance(returned, str):
        print(f"ping FAILED: response has no `model` field: {json.dumps(obj)[:300]}", file=sys.stderr)
        return False
    note = "" if returned == model else f" (backend reports {returned!r} — alias?)"
    print(f"ping OK: {total_ms * 1000:.1f} ms (includes model cold-start){note}")
    return True


# ── prompt building ─────────────────────────────────────────────────────────

def make_unique_prefix():
    """Fresh random prefix per request so prefix/KV caches never apply."""
    return f"<bench-{uuid.uuid4().hex[:12]}> "


def filler_for_size(n):
    """A repeated single word tokenizes most predictably across tokenizers.

    Exact token counts come from usage.prompt_tokens in the response.
    """
    return (FILLER_WORD + " ") * n


def read_prompt_file(path):
    """One prompt per line; a line of exactly `--` splits multi-line blocks."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not any(l.strip() == "--" for l in lines):
        return [l for l in (x.strip() for x in lines) if l]
    blocks, cur = [], []
    for l in lines:
        if l.strip() == "--":
            blocks.append("\n".join(cur))
            cur = []
        else:
            cur.append(l)
    blocks.append("\n".join(cur))
    return [b.strip() for b in blocks if b.strip()]


def build_jobs(args, sizes):
    """Return [(label, content)]; the unique prefix is added per request."""
    if args.prompts or args.prompt_file:
        prompts = list(args.prompts or [])
        if args.prompt_file:
            prompts.extend(read_prompt_file(args.prompt_file))
        return [(f"custom-{i}", p) for i, p in enumerate(prompts, 1)]
    return [(f"{FILLER_WORD} x{n}", filler_for_size(n)) for n in sizes]


# ── bench request ───────────────────────────────────────────────────────────

def run_request(client, url, model, content, max_tokens, stream, label, repeat):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": make_unique_prefix() + content}],
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": 0,
    }
    # llama.cpp only sends usage in streams when asked; needed for
    # completion_tokens (and the wall-clock gen rate fallback).
    if stream:
        payload["stream_options"] = {"include_usage": True}
    obj, ttft_ms, total_ms, err = send_request(client, url, payload, stream)
    r = Result(label=label, repeat=repeat, ttft_ms=ttft_ms, total_ms=total_ms)
    if err:
        r.error = err
        return r
    usage = obj.get("usage") or {}
    timings = obj.get("timings") or {}
    r.prompt_tokens = usage.get("prompt_tokens")
    if r.prompt_tokens is None:
        r.prompt_tokens = timings.get("prompt_n")
    r.completion_tokens = usage.get("completion_tokens")
    if r.completion_tokens is None:
        r.completion_tokens = timings.get("predicted_n")
    r.pp_ms = timings.get("prompt_ms")
    if r.pp_ms is not None:
        r.pp_source = "timings"
    elif ttft_ms is not None:
        r.pp_ms, r.pp_source = ttft_ms, "ttft"
    else:
        r.pp_ms, r.pp_source = total_ms, "total"
    if r.pp_ms and r.pp_ms > 0 and r.prompt_tokens:
        r.pp_tok_s = r.prompt_tokens / (r.pp_ms / 1000.0)
    pred_ms = timings.get("predicted_ms")
    if r.completion_tokens and pred_ms and pred_ms > 0:
        # llama.cpp ground truth: total decode time for predicted_n tokens
        # (includes reasoning tokens).
        r.gen_tok_s = r.completion_tokens / (pred_ms / 1000.0)
        r.gen_source = "timings"
    elif r.completion_tokens and ttft_ms is not None and total_ms > ttft_ms:
        r.gen_tok_s = r.completion_tokens / ((total_ms - ttft_ms) / 1000.0)
        r.gen_source = "wall"
    return r


# ── reporting ───────────────────────────────────────────────────────────────

def print_results(results):
    header = (f"{'query':<16} {'rep':>3} {'prompt_tok':>10} {'TTFT ms':>9} "
              f"{'total ms':>9} {'PP ms':>8} {'PP tok/s':>10} {'gen tok/s':>9}  source")
    print(header)
    print("-" * len(header))
    by_label: dict[str, list[Result]] = {}
    for r in results:
        by_label.setdefault(r.label, []).append(r)
    for label, rs in by_label.items():
        for r in rs:
            print(f"{label:<16} {r.repeat:>3} {fmt_int(r.prompt_tokens):>10} "
                  f"{fmt_ms(r.ttft_ms):>9} {fmt_ms(r.total_ms):>9} {fmt_ms(r.pp_ms):>8} "
                  f"{fmt_rate(r.pp_tok_s):>10} {fmt_rate(r.gen_tok_s):>9}  "
                  f"{r.pp_source}/{r.gen_source}"
                  + (f"  ERROR: {r.error}" if r.error else ""))
        if len(rs) > 1:
            print_summary(rs)


def print_summary(rs):
    ok = [r for r in rs if r.error is None]
    if not ok:
        return

    def agg(key, spec):
        vals = [getattr(r, key) for r in ok if getattr(r, key) is not None]
        if not vals:
            return "-"
        return (f"{statistics.mean(vals):{spec}} / "
                f"{statistics.median(vals):{spec}} / "
                f"{min(vals):{spec}}")

    print(f"  summary (n={len(ok)}) mean/median/min: "
          f"total {agg('total_ms', '.1f')} ms, "
          f"TTFT {agg('ttft_ms', '.1f')} ms, "
          f"PP {agg('pp_ms', '.1f')} ms, "
          f"PP {agg('pp_tok_s', ',.0f')} tok/s, "
          f"gen {agg('gen_tok_s', ',.0f')} tok/s")


def write_csv(path, results, model, url, stream, max_tokens):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "model", "url", "query", "repeat", "prompt_tokens",
                        "completion_tokens", "ttft_ms", "total_ms", "pp_ms",
                        "pp_tok_s", "gen_tok_s", "pp_source", "gen_source",
                        "stream", "max_tokens", "error"])
        for r in results:
            w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), model, url, r.label,
                        r.repeat, r.prompt_tokens, r.completion_tokens,
                        fmt_csv(r.ttft_ms), fmt_csv(r.total_ms), fmt_csv(r.pp_ms),
                        fmt_csv(r.pp_tok_s), fmt_csv(r.gen_tok_s), r.pp_source,
                        r.gen_source, stream, max_tokens, r.error])


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, help="model name as the endpoint knows it")
    p.add_argument("--sizes", type=int, nargs="*", default=None,
                   help="token ladder (default: %s); ignored with --prompts/--prompt-file"
                        % " ".join(map(str, DEFAULT_SIZES)))
    p.add_argument("--repeats", type=int, default=1, help="runs per query (default 1)")
    p.add_argument("--max-tokens", type=int, default=1,
                   help="cap generated tokens (default 1: total time ≈ prompt processing)")
    p.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True,
                   help="stream to measure TTFT (default: on)")
    p.add_argument("--prompts", nargs="+", metavar="TEXT",
                   help="custom prompts, one query each")
    p.add_argument("--prompt-file", metavar="PATH",
                   help="custom prompts from file: one per line, `--` lines split blocks")
    p.add_argument("--url", default=None,
                   help=f"endpoint base URL (default: $LLM_ORCH_URL or {DEFAULT_URL})")
    p.add_argument("--api-key", default=None,
                   help="API key (default: $APIKEY_LLM_ORCH; no auth header if unset)")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="per-request timeout in seconds (default 300)")
    p.add_argument("--csv", metavar="PATH", help="append per-request results as CSV")
    return p.parse_args()


def main():
    args = parse_args()
    if args.repeats < 1:
        sys.exit("--repeats must be >= 1")
    if args.max_tokens < 1:
        sys.exit("--max-tokens must be >= 1")
    if args.timeout <= 0:
        sys.exit("--timeout must be > 0")
    sizes = args.sizes if args.sizes is not None else DEFAULT_SIZES
    if any(s < 1 for s in sizes):
        sys.exit("--sizes must be positive token counts")

    url = args.url or os.environ.get("LLM_ORCH_URL") or DEFAULT_URL
    chat_url = url.rstrip("/") + "/v1/chat/completions"
    key = args.api_key if args.api_key is not None else os.environ.get("APIKEY_LLM_ORCH")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    jobs = build_jobs(args, sizes)
    print(f"benchmark: model={args.model} url={url} max_tokens={args.max_tokens} "
          f"stream={'on' if args.stream else 'off'} queries={len(jobs)} repeats={args.repeats}")

    with httpx.Client(headers=headers, timeout=args.timeout) as client:
        if not ping(client, chat_url, args.model):
            sys.exit(1)

        results: list[Result] = []
        total = len(jobs) * args.repeats
        i = 0
        for label, content in jobs:
            for rep in range(1, args.repeats + 1):
                i += 1
                r = run_request(client, chat_url, args.model, content, args.max_tokens,
                                args.stream, label, rep)
                results.append(r)
                print(f"  [{i}/{total}] {label} rep {rep}: "
                      f"{fmt_int(r.prompt_tokens)} tok | TTFT {fmt_ms(r.ttft_ms)} ms | "
                      f"total {fmt_ms(r.total_ms)} ms | PP {fmt_ms(r.pp_ms)} ms "
                      f"({r.pp_source}) @ {fmt_rate(r.pp_tok_s)} tok/s | "
                      f"gen {fmt_rate(r.gen_tok_s)} tok/s"
                      + (f"  ERROR: {r.error}" if r.error else ""), flush=True)

        print()
        print_results(results)
        if args.csv:
            write_csv(args.csv, results, args.model, url, args.stream, args.max_tokens)
            print(f"\nCSV appended to {args.csv}")

    failed = [r for r in results if r.error]
    if failed:
        print(f"\n{len(failed)}/{len(results)} requests failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
