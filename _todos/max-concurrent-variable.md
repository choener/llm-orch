# TODO

## Goal
Support a `{max_concurrent}` runtime placeholder in model `cmd` stanzas (and `cmd_aliases`), substituted with the model's declared `max_concurrent` at spawn time — mirroring the existing `{context_length}` support.

## Tasks

### 1. Core substitution (src/scheduler.rs)
- [x] In `resolve_cmd` (~line 1430): add `.replace("{max_concurrent}", &cfg.max_concurrent.to_string())` alongside the existing `{port}` / `{context_length}` / `{name}` replacements
- [x] Update `resolve_cmd`'s doc comment to list `{max_concurrent}`

### 2. Reserved alias name (src/config.rs)
- [x] Add `"max_concurrent"` to the reserved `cmd_aliases` names list (line ~190, currently `["port", "context_length", "name"]`) so user aliases can't shadow the placeholder
- [x] Extend the `rejects_reserved_cmd_alias_names` test (line ~855) to cover `max_concurrent`

### 3. Documentation
- [x] `ModelConfig.cmd` field doc (config.rs ~line 449): mention `{max_concurrent}` substitution
- [x] `ModelConfig.max_concurrent` field doc: note the value is also used for `{max_concurrent}` cmd substitution at spawn time
- [x] `config.example.yaml`: add `{max_concurrent}` to the runtime-placeholder comment (lines 83–85) and show usage in example 1's `cmd` (`--parallel {max_concurrent}` with the same "substituted from …" comment style as `--ctx-size {context_length}`)

### 4. Unit test (src/scheduler.rs)
- [x] Add `resolve_cmd_substitutes_max_concurrent` test next to `resolve_cmd_substitutes_context_length_and_port` (~line 3898): a config with `max_concurrent: 8` and `cmd: "run --parallel {max_concurrent}"` resolves to `run --parallel 8`

### 5. Verification
- [x] `cargo build` + `cargo test` pass (145 lib + 18 integration tests, all green; new tests `resolve_cmd_substitutes_max_concurrent` and extended `rejects_reserved_cmd_alias_names` pass)
- [x] `cargo fmt` clean; `cargo clippy` — no warnings in changed code (15 pre-existing warnings in untouched files: backend.rs, handlers.rs, watcher.rs, gpu.rs, debug_log.rs, instance.rs, config.rs:54/222, scheduler.rs:1974/2391/2447/2537)

## Notes / Decisions
- **Fingerprint invariant preserved**: `max_concurrent` stays a routing-only field (excluded from `fingerprint_with_aliases`), so changing it on hot-reload does NOT retire running instances — they keep their old `--parallel` until idled out; new spawns use the new value. This matches the documented invariant in `fingerprint_with_aliases` ("changing them must not retire running instances") and the hot-reload semantics of every other routing field.
- Default `max_concurrent` is 4, so an undeclared value substitutes `4` — same behavior class as `context_length`'s default of 4096.
- llama-server queues internally when all `--parallel` slots are busy, so a temporary mismatch between a running instance's old `--parallel` and the newly declared `max_concurrent` degrades gracefully (no hard failures).
- The `{max_concurrent}` placeholder also works inside `cmd_aliases` values (resolution order: aliases first, then runtime placeholders — identical to how `{port}`/`{name}` already work there, e.g. the audiocpp example).
