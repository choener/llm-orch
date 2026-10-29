# 009 — Decision models (`POST /v1/systemone`)

Status: **implemented** (issue #9).

A *decision model* does not write. You hand it a **state** and some typed
**questions**, and in one forward pass it returns a probability for every
option you gave it. The format is TypeSafe's **System One** API, introduced
with the Jev models and now spoken by [llama.cpp's `/v1/systemone`
endpoint](https://github.com/ggml-org/llama.cpp/pull/29818) and by halogen's
NPU `decisions` endpoint, so an existing client only needs a new base URL.

Typical uses are the fast, boring decisions around a chat model: route this
request, is the agent's step done, is this input hostile, did the retrieval
answer the question. Milliseconds instead of a generation, which is why it
gets its own lane in llm-orch rather than riding `/v1/chat/completions`.

```sh
curl -s localhost:8080/v1/systemone -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" -d '{
  "model": "kev",
  "state": "Customer message: I was charged twice for my order and nobody replies.",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "payments, charges, refunds",
        "shipping": "delivery, tracking, lost parcels",
        "technical": "bugs, errors, login problems"
      }
    },
    "angry": {"type": "noul", "instructions": "Is the customer angry?"},
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this?",
      "criteria": ["can wait", "this week", "today", "right now"]
    }
  }
}'
```

```json
{
  "model": "kev",
  "answers": {
    "route":   {"type": "choice", "choice": "billing",
                "probabilities": {"billing": 0.90, "shipping": 0.03, "technical": 0.07},
                "confidence": 0.86},
    "angry":   {"type": "noul", "noul": 0.82},
    "urgency": {"type": "score", "score": 2.28,
                "legend": {"0": "can wait", "1": "this week", "2": "today", "3": "right now"},
                "probabilities": {"0": 0.04, "1": 0.19, "2": 0.22, "3": 0.55},
                "confidence": 0.28}
  },
  "usage": {"input_tokens": 130, "output_tokens": 0}
}
```

Three question types: `choice` (pick an option, `criteria` is an object of
`option → description`), `score` (expected level over 2–10 ordered levels,
`criteria` is an array), `noul` (probability of yes). `state` is a string or
a list of chat messages; `images` carries base64 data URLs for the models
that read them. See llama.cpp's server README and the [TypeSafe System One
docs](https://docs.typesafe.ai/concepts/system-one) for the full contract.

## Design decisions

- **llm-orch is the only gate.** Neither `llama-server` nor halogen requires a
  key on this endpoint, so an exposed decision model would otherwise be an
  anonymous oracle over your prompts. `/v1/systemone` sits behind the same
  `ApiKey` extractor as everything else: no apikeys configured ⇒ deny
  (`docs/001-plan.md`, default-deny), and an unauthenticated request never
  reaches the scheduler — it cannot even cause a spawn.
- **Passthrough, not typed.** The body is forwarded as a raw
  `serde_json::Value` (the rerank/responses pattern). The backend is the
  authority on what a valid question is: it answers a malformed question with
  400 and "this model cannot decide" with 501, and llm-orch passes both
  through with status and JSON body untouched. Only the `model` field is
  rewritten to the resolved model name before forwarding.
- **501 counts as an answer.** A 501 means the caller could have avoided it
  (wrong model for a decision), so it is not collapsed into the 502 that a
  broken backend gets. Other 5xx stay 502 and are not counted into stats.
- **No streaming.** Not in the upstream contract: the answer is a handful of
  numbers that only exist after the single forward pass.
- **Token accounting reflects the pass.** Nothing was sampled, so the whole
  batch — state, questions, options — is billed as prompt tokens
  (`usage.input_tokens`) with 0 generated, in `/v1/stats` and in the
  completion records behind `/admin/status`.
- **Everything else is an ordinary model.** Decision models are declared in
  `models:` with a normal `cmd`, and get the same spawn-on-demand, FIFO
  queue, autoscaling, `idle_ttl` eviction, aliases, `debug_log` JSONL,
  per-key accounting and peering as a chat model. Aliases resolve to the
  decision model like any other target.
- **Discovery is declared, never probed.** `capabilities` in config is
  published in `/v1/models` as llama.cpp's `architecture` object, so a client
  checks the same field against llm-orch as against a single `llama-server`.
  It has to come from config because the marker must be visible while the
  model is *not* loaded — that is exactly when a client picks a lane — and
  llm-orch synthesizes `/v1/models` from config. Probing a running backend
  would only work after the cost was already paid. Proxying the backend's
  `/v1/models` was rejected for the same reason.

| capability | maps to | meaning |
| --- | --- | --- |
| `decisions` | `output_modalities` | answers `POST /v1/systemone` |
| `image` / `audio` / `video` | `input_modalities` | reads that modality besides text |

`architecture` is emitted for every model and alias with llama.cpp's default
`{"text"}` on both sides; any value outside the table is a config error,
because a typo would silently hide a lane. Speech *output* is deliberately
not modelled — llm-orch routes TTS/ASR through `/v1/audio/*`, where clients
already find it. Cluster peering carries the advertisement in both
directions: a master republishes a leaf's `architecture` from its catalog, so
the decision lane is visible from the gateway while unloaded there (and while
the leaf is offline).

```console
$ curl -s localhost:8080/v1/models -H "Authorization: Bearer $KEY" \
    | jq '.data[] | select(.architecture.output_modalities | index("decisions")) | .id'
"kev"
"triage"          # an alias inherits its target's capabilities
```

## Configuring one

```yaml
models:
  - name: kev
    context_length: 8192
    max_concurrent: 8                # decisions are cheap: pack them
    idle_ttl: 300
    capabilities: [decisions]        # published as architecture
    # Kev-4B, lev and OpenJev evaluate state + question + options in one
    # batch, so size the context and batch for the whole prompt. llama.cpp
    # defaults the prompt cache to 0 for decision models.
    cmd: |
      {llama}
        --model /models/Kev-4B-Q4_K_M.gguf
        --ctx-size {context_length}
        --parallel {max_concurrent}
        --batch-size 1024 --ubatch-size 256

  # OpenJev reads images, so it must say so — a client that checks
  # input_modalities will otherwise never attach the page to be judged.
  - name: openjev
    context_length: 32768
    capabilities: [decisions, image]
    cmd: |
      {llama}
        --model /models/OpenJev-Q4_K_M.gguf
        --mmproj /models/OpenJev-mmproj.F16.gguf
        --ctx-size {context_length}
```

A halogen `decider`/`d3` model on a NPU is a remote model (`cmd:` an `http://`
URL) and needs the same `capabilities` line for llm-orch to advertise it.
Small models answer in single-digit milliseconds (Julia-1 144 M ≈ 3 ms, Kev-4B
≈ 12 ms, OpenJev 27B ≈ 43 ms on an RTX PRO 6000), so a single instance with a
high `max_concurrent` is usually enough; autoscale the same way as a chat
model if it is not.

## Out of scope

- **Content-based auto-routing.** The lane is chosen by model name/alias, not
  by llm-orch guessing from the prompt.
- **Question validation in llm-orch.** See passthrough above.
- **Registration with `GET /v1/requests/{id}`** and capabilities in `/v1/info`
  / `/admin/status` — rerank and embeddings do not do the first either, and
  the lane is advertised where clients look for it.

## Tests

The stub backend (`src/bin/llm-orch-stub-backend.rs`) implements the endpoint
with canned answers and llama.cpp's error split (400 malformed question, 501
unsupported question type). `tests/integration.rs` covers: end-to-end answers
and usage mapping, auth required with no spawn, 400/501 preserved verbatim,
5xx → 502, missing `model` field, alias resolution, stats and `/admin/status`
token accounting, debug-log request/response correlation, `/v1/models`
advertisement (including `input_modalities` and aliases) while unloaded, and a
peered decision model advertised and served through the master.
