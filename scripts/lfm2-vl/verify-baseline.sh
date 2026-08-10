#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
cd -- "$REPO_ROOT"

# The verifier is intentionally offline, CPU-only, and locked. It never runs an example.
export CARGO_NET_OFFLINE=true
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES=""

if [[ ! -f Cargo.lock ]]; then
    printf '%s\n' \
        'error: Cargo.lock is required for locked verification.' \
        'Create the local-only lockfile in the Linux verification worktree, then record its SHA-256.' \
        >&2
    exit 2
fi

run_step() {
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
    "$@"
}

printf 'repo-root: %s\n' "$REPO_ROOT"
printf 'timestamp-start-utc: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf 'head: %s\n' "$(git rev-parse HEAD)"
printf 'platform: %s\n' "$(uname -srmo)"
printf 'policy: locked, offline, CPU-only, no model execution or download\n'
printf 'cargo-lock-sha256: %s\n' "$(sha256sum Cargo.lock | awk '{print $1}')"

run_step cargo fmt --all -- --check
run_step cargo check --locked --offline -p candle-core
run_step cargo check --locked --offline -p candle-nn
run_step cargo check --locked --offline -p candle-transformers
run_step cargo check --locked --offline -p candle-vlm
run_step cargo check --locked --offline -p candle-examples --example lfm2
run_step cargo check --locked --offline -p candle-examples --example quantized-lfm2
run_step cargo check --locked --offline -p candle-examples --example lfm2-vl
run_step git diff --check
run_step git diff --cached --check

printf 'timestamp-end-utc: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf 'baseline: passed\n'
