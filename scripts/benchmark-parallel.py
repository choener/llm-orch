#!/usr/bin/env python3
"""Throughput benchmark: fire P unique requests of ~K prompt tokens at once.

Companion to benchmark-llm.py, which measures a single request at a time.
This one answers "what happens under load": --parallel P requests are
in-flight simultaneously, each with a prompt of --depth K tokens (default
32k), so the orchestrator has to queue, scale out, or split — and the
numbers show which.

Every prompt is unique (random salt + shuffled word soup), so prefix/KV
caches never apply: all P requests really process their full K tokens.

`--depth` is calibrated, not guessed.  One probe request (at most 4096 filler
words, max_tokens=1) measures this model's real tokens-per-word, so K means
actual prompt tokens on any tokenizer; measured error stayed under 2% from
1k to 32k depth.  The filler is common short words because unusual ones split
into several tokens and drift further from K.  `--no-calibrate` skips the
probe.

Metrics
-------
Per request: prompt tokens (from usage), TTFT, total time, prompt-processing
rate (llama.cpp `timings` when the backend returns them, else TTFT).
Per batch: wall time, aggregate throughput (sum of prompt tokens / batch
wall time), aggregate decode tok/s, and the TTFT spread.  A large gap between
the fastest and the slowest TTFT means the requests were served one after
another (queueing on a single instance) instead of concurrently.

Flow: one ping (pays the model cold-start, like benchmark-llm.py), one
calibration probe, then the parallel batch.

Usage
-----
    scripts/benchmark-parallel.py --model "Qwen 27B"                   # 4 x 32k
    scripts/benchmark-parallel.py --model "Qwen 27B" --depth 8192 --parallel 8
    scripts/benchmark-parallel.py --model "Qwen 27B" --rounds 3
    scripts/benchmark-parallel.py --model "Qwen 27B" --max-tokens 128  # + decode

Quote the model name when it contains spaces, as llm-orch aliases can.

Environment: LLM_ORCH_URL (default http://127.0.0.1:8080), APIKEY_LLM_ORCH
(no Authorization header when unset, for keyless OpenAI-compatible servers).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

try:
    import httpx
except ImportError:
    sys.exit("benchmark-parallel.py needs httpx (pip install httpx, or `nix develop`)")

DEFAULT_URL = "http://127.0.0.1:8080"
DEFAULT_DEPTH = 32 * 1024
DEFAULT_PARALLEL = 4
CALIBRATION_MAX_WORDS = 4096
# Common short words: each stays one token at scale (~1.01 tok/word measured
# on Qwen), so --depth is nearly right even before calibration.  Rare words
# split into 2-3 tokens and their rate varies with prompt length.
FILLER_WORDS = ("the of and to in is was it for on with as at by this from or "
                "an are be have has had not but they you that his her our "
                "their who what when where which while into over under again "
                "there then than so if all one two new many most good very "
                "well can will said also would could should").split()


@dataclass
class Result:
    index: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    ttft_ms: float | None = None
    total_ms: float = 0.0
    pp_ms: float | None = None
    pp_source: str = "-"  # "timings" | "ttft" | "total"
    pp_tok_s: float | None = None
    gen_tok_s: float | None = None
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
    parsed response JSON.
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

def make_prompt(words: int, rng: random.Random) -> str:
    """Unique word soup of `words` words (~words * tokens_per_word tokens).

    A small repeated vocabulary tokenizes predictably, but P identical
    prompts would share the whole prefix cache.  Shuffling it per request
    makes every token position unique; the random tag at the front marks the
    request in server logs.  Exact counts come back in usage.prompt_tokens.
    """
    filler = [rng.choice(FILLER_WORDS) for _ in range(words)]
    return f"<bench-{uuid.uuid4().hex[:12]}> " + " ".join(filler)


def calibrate(client, url, model, depth):
    """Return the word count that puts a prompt at ~`depth` real tokens.

    One probe request, read back through usage.prompt_tokens: tokenizers
    differ too much for a fixed words-per-token guess to mean anything.  The
    probe runs at min(depth, 4096) words because the merge rate is not
    constant with prompt length — measuring at scale keeps the estimate
    within ~2%.
    """
    probe_words = max(min(depth, CALIBRATION_MAX_WORDS), 32)
    payload = {
        "model": model,
        "messages": [{"role": "user",
                      "content": make_prompt(probe_words, random.Random(0))}],
        "max_tokens": 1,
        "stream": False,
        "temperature": 0,
    }
    obj, _ttft, _total, err = send_request(client, url, payload, stream=False)
    tokens = None if err else (obj.get("usage") or {}).get("prompt_tokens")
    if not tokens:
        print(f"calibration probe failed ({err}); assuming 1 word = 1 token",
              file=sys.stderr)
        return depth
    # Chat templates add a fixed ~10-20 tokens; at hundreds of words that is
    # noise.
    rate = max(tokens / probe_words, 0.25)
    words = max(round(depth / rate), 1)
    print(f"calibrated: {tokens} prompt tokens from {probe_words} filler words "
          f"({rate:.3f} tok/word) -> {words} words per prompt at depth {depth}")
    return words



# ── one request of the batch ────────────────────────────────────────────────

def run_one(args, chat_url, headers, index, prompt, start_gate):
    """Worker body: wait for the gate, then send one request.

    One httpx.Client per thread; the gate releases all threads at the same
    instant so the batch really arrives simultaneously.
    """
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "stream": args.stream,
        "temperature": 0,
    }
    # llama.cpp only sends usage in streams when asked; needed for
    # prompt/completion tokens (and the wall-clock fallbacks).
    if args.stream:
        payload["stream_options"] = {"include_usage": True}
    r = Result(index=index)
    with httpx.Client(headers=headers, timeout=args.timeout) as client:
        # Connect before the gate releases: the batch should arrive at the
        # server at one instant, not staggered by TLS handshakes.
        try:
            client.get(chat_url.rsplit("/v1/", 1)[0] + "/v1/models")
        except httpx.HTTPError:
            pass
        start_gate.wait(timeout=args.timeout * 2)
        obj, ttft, total, err = send_request(client, chat_url, payload, args.stream)
    r.ttft_ms = None if ttft is None else ttft * 1000
    r.total_ms = total * 1000
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
    elif r.ttft_ms is not None:
        r.pp_ms, r.pp_source = r.ttft_ms, "ttft"
    else:
        r.pp_ms, r.pp_source = r.total_ms, "total"
    if r.pp_ms and r.pp_ms > 0 and r.prompt_tokens:
        r.pp_tok_s = r.prompt_tokens / (r.pp_ms / 1000.0)
    pred_ms = timings.get("predicted_ms")
    if r.completion_tokens and r.completion_tokens > 1:
        # A 1-token completion has no meaningful decode rate.
        if pred_ms and pred_ms > 0:
            r.gen_tok_s = r.completion_tokens / (pred_ms / 1000.0)
        elif r.ttft_ms is not None and r.total_ms > r.ttft_ms:
            r.gen_tok_s = r.completion_tokens / ((r.total_ms - r.ttft_ms) / 1000.0)
    return r


# ── reporting ───────────────────────────────────────────────────────────────

def print_round(results, batch_wall_s, words, parallel):
    header = (f"{'req':>4} {'prompt_tok':>10} {'TTFT ms':>10} {'total ms':>10} "
              f"{'PP ms':>9} {'PP tok/s':>10} {'gen tok/s':>9}  source")
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda r: r.index):
        print(f"{r.index:>4} {fmt_int(r.prompt_tokens):>10} {fmt_ms(r.ttft_ms):>10} "
              f"{fmt_ms(r.total_ms):>10} {fmt_ms(r.pp_ms):>9} "
              f"{fmt_rate(r.pp_tok_s):>10} {fmt_rate(r.gen_tok_s):>9}  "
              f"{r.pp_source}" + (f"  ERROR: {r.error}" if r.error else ""))

    ok = [r for r in results if r.error is None]
    failed = len(results) - len(ok)
    tokens = sum(r.prompt_tokens or 0 for r in ok)
    exact = all(r.prompt_tokens for r in ok)
    print(f"\nbatch: {len(ok)}/{len(results)} ok in {batch_wall_s:.2f} s wall | "
          f"prompts {'=' if exact else '~'} {fmt_int(tokens if exact else words * parallel)} tok | "
          f"aggregate {fmt_rate(tokens / batch_wall_s if batch_wall_s > 0 else None)} tok/s")
    if ok:
        ttfts = [r.ttft_ms for r in ok if r.ttft_ms is not None]
        if len(ttfts) > 1:
            spread = (max(ttfts) - min(ttfts)) / 1000.0
            print(f"TTFT min/median/max: {fmt_ms(min(ttfts))} / "
                  f"{fmt_ms(statistics.median(ttfts))} / {fmt_ms(max(ttfts))} ms "
                  f"(spread {spread:.1f} s — large = requests were served serially)")
        elif ttfts:
            print(f"TTFT: {fmt_ms(ttfts[0])} ms")
        # Batch decode figure: completions are produced concurrently, so the
        # batch total is the useful number, not the per-request mean.  With a
        # 1-token cap there is nothing to measure (timings for a single token
        # are noise), so skip the decode lines entirely.
        gen = sum(r.completion_tokens or 0 for r in ok)
        if gen > len(ok):
            if batch_wall_s > 0:
                print(f"gen: {fmt_int(gen)} tokens | batch aggregate "
                      f"{fmt_rate(gen / batch_wall_s)} tok/s")
            gens = [r.gen_tok_s for r in ok if r.gen_tok_s is not None]
            if gens:
                print(f"per-request gen: mean {fmt_rate(statistics.mean(gens))} "
                      f"tok/s (n={len(gens)}, each sharing the GPU with "
                      f"{max(len(gens) - 1, 0)} other request(s))")
    if failed:
        first_err = next(r.error for r in results if r.error)
        print(f"{failed} failed, first error: {first_err}")


def write_csv(path, rows, model, url, depth, words, parallel, batch_wall_s,
              stream, max_tokens):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "model", "url", "depth", "words", "parallel",
                        "batch_wall_s", "req", "prompt_tokens",
                        "completion_tokens", "ttft_ms", "total_ms", "pp_ms",
                        "pp_tok_s", "gen_tok_s", "pp_source", "stream",
                        "max_tokens", "error"])
        for r in rows:
            w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), model, url, depth,
                        words, parallel, round(batch_wall_s, 1), r.index,
                        r.prompt_tokens, r.completion_tokens, fmt_csv(r.ttft_ms),
                        fmt_csv(r.total_ms), fmt_csv(r.pp_ms), fmt_csv(r.pp_tok_s),
                        fmt_csv(r.gen_tok_s), r.pp_source, stream, max_tokens,
                        r.error])


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, help="model name as the endpoint knows it")
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                   help=f"prompt tokens per request, calibrated against the "
                        f"model (default {DEFAULT_DEPTH})")
    p.add_argument("--no-calibrate", action="store_true",
                   help="skip the calibration probe; assume 1 word = 1 token")
    p.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL,
                   help=f"requests in flight per batch (default {DEFAULT_PARALLEL})")
    p.add_argument("--rounds", type=int, default=1,
                   help="sequential batches of --parallel requests (default 1)")
    p.add_argument("--max-tokens", type=int, default=1,
                   help="cap generated tokens (default 1: total time ≈ prompt processing)")
    p.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True,
                   help="stream to measure TTFT (default: on)")
    p.add_argument("--url", default=None,
                   help=f"endpoint base URL (default: $LLM_ORCH_URL or {DEFAULT_URL})")
    p.add_argument("--api-key", default=None,
                   help="API key (default: $APIKEY_LLM_ORCH; no auth header if unset)")
    p.add_argument("--timeout", type=float, default=600.0,
                   help="per-request timeout in seconds (default 600)")
    p.add_argument("--csv", metavar="PATH", help="append per-request results as CSV")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for reproducible prompts (default: random)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.parallel < 1:
        sys.exit("--parallel must be >= 1")
    if args.depth < 1:
        sys.exit("--depth must be >= 1")
    if args.rounds < 1:
        sys.exit("--rounds must be >= 1")
    if args.max_tokens < 1:
        sys.exit("--max-tokens must be >= 1")
    if args.timeout <= 0:
        sys.exit("--timeout must be > 0")

    url = args.url or os.environ.get("LLM_ORCH_URL") or DEFAULT_URL
    chat_url = url.rstrip("/") + "/v1/chat/completions"
    key = args.api_key if args.api_key is not None else os.environ.get("APIKEY_LLM_ORCH")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    rng = random.Random(args.seed)
    print(f"benchmark: model={args.model} url={url} depth={args.depth} tok "
          f"parallel={args.parallel} rounds={args.rounds} "
          f"max_tokens={args.max_tokens} stream={'on' if args.stream else 'off'}")

    with httpx.Client(headers=headers, timeout=args.timeout) as client:
        if not ping(client, chat_url, args.model):
            sys.exit(1)
        words = args.depth if args.no_calibrate else calibrate(
            client, chat_url, args.model, args.depth)

    all_rows, batch_walls, failures = [], [], 0
    for round_i in range(1, args.rounds + 1):
        if args.rounds > 1:
            print(f"\n=== round {round_i}/{args.rounds} ===", flush=True)
        # Built per round, after calibration, and before the clock starts so
        # prompt construction never skews the batch.
        prompts = [make_prompt(words, rng) for _ in range(args.parallel)]
        gate = threading.Barrier(args.parallel)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = [pool.submit(run_one, args, chat_url, headers, i, p, gate)
                       for i, p in enumerate(prompts, 1)]
            results = [f.result() for f in futures]
        batch_wall = time.perf_counter() - t0
        batch_walls.append(batch_wall)
        failures += sum(1 for r in results if r.error)
        all_rows.extend(results)
        print()
        print_round(results, batch_wall, words, args.parallel)
        if args.csv:
            write_csv(args.csv, results, args.model, url, args.depth, words,
                      args.parallel, batch_wall, args.stream, args.max_tokens)

    if args.rounds > 1:
        print(f"\nrounds wall s: {' / '.join(f'{w:.1f}' for w in batch_walls)} "
              f"(mean {statistics.mean(batch_walls):.1f})")
    if args.csv:
        print(f"CSV appended to {args.csv}")
    if failures:
        print(f"\n{failures}/{len(all_rows)} requests failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
