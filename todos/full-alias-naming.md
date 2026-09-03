# TODO — Bidirectional model-name translation (`name` ↔ `model_name`)

## Problem

llm-orch forwards the **resolved config `name`** to the backend in the request
`model` field (e.g. `handlers.rs:705`, `handlers.rs:405`), on the rationale
that "strict backends validate the model field". That assumption holds for
vLLM but **not** for llama.cpp, which ignores the `model` field entirely
(single-model server). The consequence is an asymmetry:

- **llama.cpp backends**: the config `name` is a pure scheduler label; the
  operator can swap the underlying GGUF / quantization freely without
  touching the name. Clients see a stable public name.
- **Strict backends (vLLM, and any server that validates `model`)**: the
  forwarded name must match what the backend actually serves. When the
  config `name` differs from the model the container loads (e.g. vLLM's
  `--served_model_name /models/Qwen3.8-27B-MXFP4-mtpfp8`), every request
  404s with `The model '<config-name>' does not exist`.

The same asymmetry leaks into **aliases**: llm-orch resolves an alias to a
model name and forwards the *resolved* name, so the client-visible reporting
(alias in `/v1/models`, admin status, debug logs) and the backend-visible
name diverge, which is confusing to observe and debug.

### Immediate workaround (vLLM only)

vLLM's `--served-model-name` accepts **multiple** names and answers to any
of them, echoing the first. So for podman/vLLM backends the mismatch is
fixable today with zero code by adding the llm-orch config name to the
container args:

```
podman run --rm … localhost/vllm-radiance:0.11.0 \
  --model /models/Qwen3.8-27B-MXFP4-mtpfp8 \
  --served-model-name <config-name> [<extra-alias> …]
```

This unblocks the current deployment. The translation feature below is the
durable fix and is still wanted because (a) not all strict backends support
multiple served names (TGI/tei do not), and (b) it fixes the alias reporting
inconsistency generally.

## Goal

Give each model a stable public `name` and an optional backend-facing
`model_name`, and translate between the two transparently in both
directions, so the public name is decoupled from what the backend loads —
the way llama.cpp backends already behave by accident.

## Design (rough)

- `ModelConfig.model_name: Option<String>` — the name sent to the backend.
  Defaults to `name` when unset (backward compatible).
- **Request path**: after alias resolution picks a concrete model, rewrite
  the request `model` field to that model's `model_name` before forwarding.
- **Response path**: rewrite the response body's `model` field from
  `model_name` back to the name the client originally sent (the alias if one
  was used, else the public `name`), so the client sees a consistent name.
  For streaming, rewrite the `model` field inside every SSE `data:` chunk.
- **Reporting**: `/v1/models` lists public `name`s (current behavior);
  decide whether to expose `model_name` as an extra field. Admin status /
  debug logs should record both the requested (alias) name and the resolved
  model consistently.

## Tasks

### 1. Scope decision
- [ ] Decide whether the fix is podman-only or general. It is general
      unless the podman implementation turns out to be exactly the general
      fix (likely, since translation is name-based, not runner-based).
- [ ] Decide response-rewrite target: client's original name (alias-aware)
      vs resolved public `name`.

### 2. Config
- [ ] Add `model_name: Option<String>` to `ModelConfig` (serde default).
- [ ] Validate: `model_name` non-empty when present; document fallback.

### 3. Request translation
- [ ] Look up `model_name` for the resolved model and rewrite the outgoing
      `model` field in each forwarding handler:
      chat, completions, responses, embeddings, rerank, audio (speech /
      transcriptions / voices), images (generations / edits).
- [ ] Multipart handlers (audio transcriptions, image edits) need the
      `model` form field rewritten too.

### 4. Response translation
- [ ] Non-streaming: rewrite `model` in the aggregated JSON body back to the
      client-sent name.
- [ ] Streaming: rewrite `model` inside each SSE chunk's `data:` payload.
- [ ] Confirm error paths (404, backend error) still surface sensibly.

### 5. Alias consistency
- [ ] Make the alias → model → backend-name chain coherent in reporting:
      client sees the name it sent; backend sees `model_name`; logs record
      both.

### 6. Tests + docs
- [ ] Unit tests: config parse/fallback; per-endpoint request rewrite;
      non-stream and stream response rewrite; alias-aware rewrite.
- [ ] Integration test against the stub backend asserting the forwarded and
      echoed names.
- [ ] README + `config.example.yaml`: document `model_name` and the vLLM
      `--served-model-name` workaround.

## Notes

- Keep the passthrough philosophy: bodies are `serde_json::Value`, so the
  rewrite is a field assignment, not a typed-struct change.
- The vLLM `--served-model-name` workaround above is the zero-code fix for
  the current cheyenne deployment; implement this feature only if the
  general/alias benefits justify it.
