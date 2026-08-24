# TODO

## Goal
CI for llm-orch building + testing via the Nix flake (no plain cargo builds in CI): Forgejo workflow (done, sections 1–3) and a GitHub Actions workflow doing the same (section 4).

## Tasks

### 1. Create the workflow
- [x] Create `.forgejo/workflows/build.yaml`:
  - `on: [push, pull_request]`
  - Single job `build-and-test`, `runs-on: native` (gitea-actions-runner on cheyenne; x86_64, matches flake's hardcoded `x86_64-linux`)
  - Steps: `actions/checkout@v4` → `nix build .#default` → `nix develop .#default --command cargo test`
  - (Fix discovered during task 2: flake ref must precede `--command`; `nix develop --command cargo test .#default` leaks `.#default` into cargo as a test-name filter and runs 0 tests)
- [x] Validate the workflow file parses as YAML (done via python3 + pyyaml from nix store; note: bare `on` key parses as YAML-1.1 boolean `True`, file itself is standard Actions syntax)

### 2. Verify the CI commands locally
- [x] Run `nix build .#default` — package builds (binary smoke-tested with `--help`)
- [x] Run `nix develop .#default --command cargo test` — 145 unit + 18 integration tests pass (stub backend, no GPU)

### 3. Commit
- [x] Ensure `.forgejo/workflows/build.yaml` and this todo file are in the jj working copy; describe the change with `jj describe --stdin` (done — change `owqwzkmw`; note: global gitignore `.*` blocked auto-tracking, needed `jj file track --include-ignored`)

### 4. GitHub Actions workflow (plain cargo — user switched from nix in CI)
- [x] Create `.github/workflows/build.yaml`: `on: [push, pull_request]`, job `build-and-test` on `ubuntu-latest`; steps: `actions/checkout@v4` → `dtolnay/rust-toolchain@1.96.1` (matches the flake's current rustc) → `Swatinem/rust-cache@v2` → `cargo build` → `cargo test`
- [x] Validate locally: YAML parse (python3 + pyyaml) + `actionlint` via `nix shell nixpkgs#actionlint` (exit 0, no findings); `cargo build` + `cargo test` via devshell pass (145 unit + 18 integration)
- [x] Track with `jj file track --include-ignored` (global gitignore `.*` blocks dotfile auto-tracking) and describe with `jj describe --stdin`
  - (Note: an external `jj new` ran mid-session — likely the user's editor jj integration — creating intermediate change `wluzrkxs` holding only a plan-file diff; left as-is, not restructured)

## Notes
- GitHub: plain cargo (not nix) per user decision — `dtolnay/rust-toolchain@1.96.1` (flake's current rustc via nixos-unstable; bump the pin when the flake's rustc changes) + `Swatinem/rust-cache@v2` for registry/target caching. Forgejo workflow stays nix-based (its `native` runner only has nix).
- Example followed: `~/git/nix/os-configuration/default/.forgejo/workflows/build.yaml` (`on: [push]`, `runs-on: native`, nix-based steps). Forgejo instance: cheyenne (`git.<fqdn>`), actions enabled, single `native` runner (hostPackages: bash, git, nix, nixos-rebuild, ...).
- User decisions: triggers = push + pull_request; scope = build + test only (no clippy/fmt steps for now); build mechanism = nix flake only, no plain cargo in CI.
- Flake: `packages.default` = `rustPlatform.buildRustPackage` (Cargo.lock committed); `devShells.default` has cargo/rustc/clippy/rustfmt.
- Repo currently has only a GitHub remote; workflow activates once pushed to Forgejo (no runner changes needed — `nix` is already in the runner's hostPackages).
- `nix build` locally updates the (gitignored-ish, untracked) `result` symlink — expected, not part of the commit.
