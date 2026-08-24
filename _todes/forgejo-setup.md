# TODO

## Goal
Add a Forgejo Actions workflow that builds and tests llm-orch on the self-hosted `native` runner, using the Nix flake (no plain cargo builds in CI).

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

## Notes
- Example followed: `~/git/nix/os-configuration/default/.forgejo/workflows/build.yaml` (`on: [push]`, `runs-on: native`, nix-based steps). Forgejo instance: cheyenne (`git.<fqdn>`), actions enabled, single `native` runner (hostPackages: bash, git, nix, nixos-rebuild, ...).
- User decisions: triggers = push + pull_request; scope = build + test only (no clippy/fmt steps for now); build mechanism = nix flake only, no plain cargo in CI.
- Flake: `packages.default` = `rustPlatform.buildRustPackage` (Cargo.lock committed); `devShells.default` has cargo/rustc/clippy/rustfmt.
- Repo currently has only a GitHub remote; workflow activates once pushed to Forgejo (no runner changes needed — `nix` is already in the runner's hostPackages).
- `nix build` locally updates the (gitignored-ish, untracked) `result` symlink — expected, not part of the commit.
