# LFM2.5-VL Current Entry Point

This is the live execution entry point for the Candle 0.11 LFM2.5-VL/MMProj extension. The original blank-folder and completed phase instructions are archived in `history/BOOTSTRAP_AND_PHASE_GUIDE.md`.

## Read Order

1. Read the repository `AGENTS.md`.
2. Read `SPEC.md` for the complete product contract.
3. Read `STATUS.md` for current truth and the exact worktree boundary.
4. Read `TODO.md` for active work only.
5. Select the narrowest relevant group from `/summary_bank.json`.
6. Read `PARITY.md`, `DECISIONS.md`, `SOURCES.md`, `TENSOR_MAP.md`, or `FAILURE_LOG.md` only when the task needs that authority.

`HISTORY.md` and `history/` are opt-in completed-work records. They are not part of normal task startup.

## Current Gate

The LFM2-VL implementation phases and the coordinated Candle/SnapFlash
framework primitives are complete. Their detailed lineage and proof belong in
`HISTORY.md`, `PARITY.md`, and `docs/FORK_OVERLAYS.md`; do not copy those
snapshots back into this entry point.

The active tree is the uncommitted combined-overlay 0.2.0 candidate based on
`origin/main` at `dca9849584e377cebc1da40de966d050733f3bbf`. Its tracked
lock/toolchain, local verification contract, overlay inventories, and external
identity-receipt gate are implemented. The only active product task is the
owner-authorized clean-head publication sequence in `TODO.md`.

Annotated tag `lfm2-vl-mvp-0.1.0` remains the immutable first-MVP snapshot at
`ff885586f6d44a3d9b9ac1724032cdf5f0155384`; never move or reuse it. The new
candidate uses the distinct `candle-overlays-mvp-0.2.0` namespace. Native
Windows/MSVC is release authority and WSL2/Linux is a secondary replay.
Production models and caches are external inputs and are currently absent
after operator cleanup; retained hash-bound parity remains valid, but no new
model run or download is implied.

## One-Task Contract

Before editing, state:

- Goal: one focused result.
- Boundary: exact behavior and files in scope.
- Baseline: the narrowest command that proves the starting state.
- Done when: one explicit acceptance gate.

Inspect the current implementation before changing it. Prefer controlled errors over panics on external input. Preserve public flags, paths, formats, and text-only LFM2 behavior. Do not add a generic VLM abstraction or speculative dependency.

After editing:

1. Run the nearest focused test.
2. Run the affected crate/example check.
3. Run `cargo fmt --all -- --check`.
4. Run `git diff --check` and inspect the complete diff.
5. Run the native Windows locked/offline gate when the slice reaches a phase or handoff boundary.
6. Replay `scripts/lfm2-vl/verify-baseline.sh` in WSL2/Linux when practical.
7. Update `STATUS.md`; update `DECISIONS.md` only for a material architecture or compatibility decision.
8. Run both the affected overlay verifier and
   `bash scripts/verify-fork-overlays.sh` when a publication path changes.

Use native Windows PowerShell and the MSVC Rust toolchain for the primary gate. Keep builds CPU-only and memory-bounded until CPU parity is green:

```powershell
$env:CARGO_NET_OFFLINE = 'true'
$env:CARGO_BUILD_JOBS = '2'
cargo fmt --all -- --check
cargo check --locked --offline -p candle-core -p candle-nn -p candle-transformers -p candle-vlm
cargo check --locked --offline -p candle-examples --example lfm2
cargo check --locked --offline -p candle-examples --example quantized-lfm2
cargo check --locked --offline -p candle-examples --example lfm2-vl
```

Then replay the portable baseline in WSL when its local cache is available:

```powershell
wsl.exe -d NVIDIA-Workbench --cd /mnt/c/DevStuff/candle-mods bash -lc "CARGO_TARGET_DIR=/home/workbench/code/candle-lfm2-vl/target bash scripts/lfm2-vl/verify-baseline.sh"
```

These commands perform compilation and tests only. They do not authorize dependency downloads, model execution, hosted CI, commits, pushes, or PRs. Hosted CI is not a release authority for this fork; required evidence is local native Windows proof plus an explicitly labeled WSL replay when practical. A missing offline dependency is a blocked lane to record in `STATUS.md`, not permission to fetch it implicitly.

## Production Parity Safety

Before any production-model run:

1. Run `scripts/lfm2-vl/preflight.ps1 -AsJson` and retain its read-only report outside Git; `review` still requires explicit owner approval, while `blocked` stops admission.
2. Run `tools/lfm2_vl/reference/inspect_artifact.py --model <450m|1.6b> --model-dir <regular-file-snapshot> --output <external-manifest.json> --allow-production` and verify the exact repository revision plus every model-snapshot config, tokenizer, processor, index, and weight identity. Pass that same `<regular-file-snapshot>` as Python oracle `--model-dir` and native Candle `--model-dir`; the Python trace refuses cache-only or download-backed model resolution, both lanes rehash their inputs after inference, and the comparator requires Candle's consumed-file sizes and hashes to match the oracle manifest. Record the source-image identity separately in each trace. This is a local-only hash pass; it never downloads or serializes weights.
3. Record physical memory, Windows committed bytes and limit, GPU memory, and all existing inference PIDs.
4. Refuse concurrent Python, build, llama, or model execution; `quiet_host`
   requires all dedicated workload sets to be empty.
5. For Transformers parity, run the bounded Python oracle first. Its `--prompt` is user text and its `metadata.json.prompt` is the exact official chat-template output containing the image sentinel. Pass that recorded prompt value plus the same image to Candle CPU F32 natively on Windows; do not pass the untemplated user text. Use WSL only as a later portability replay.
6. Launch every production Python, Candle, or llama.cpp inference executable through `scripts/lfm2-vl/run-bounded-oracle.ps1`; never wrap `cargo run`. Use name-wide concurrency for unique model tools, exact-executable concurrency for generic interpreters, and retain `-LogPath` so nonzero children preserve their combined stdout/stderr.
7. Verify the owner and complete process tree exit, then repeat the host/GPU census.
8. Stop if cleanup, memory recovery, tensor parity, or deterministic replay fails.

`FAILURE_LOG.md` F-0008 is the resource-containment authority.

## Git and Worktree Boundary

This particular Windows folder is a WSL-owned linked worktree attached to local
`main`. The historical feature branch remains checked out by
`/home/workbench/code/candle-lfm2-vl`; it is not a second publication line.
Windows Git cannot resolve the Linux absolute `.git` pointer, so WSL Git owns
all repository operations here. This is local Git topology, not a requirement
for building or using the fork on Windows.

- Read and edit here when requested.
- Use WSL Git for status, staging, commits, merges, and revision checks.
- Do not force-attach the same branch to two worktrees.
- Keep `main` as the single local and GitHub integration branch; no PR is
  required for owner-reviewed work.
- Fetch and review `origin/main`, preserve both histories without force, and
  rerun the local release gate after every integration.
- Invoke the ignored `.tools/gitpush.ps1` only after explicit approval and only
  from a clean named `main`; it verifies ancestry and remote identity and does
  not stage, commit, merge, delete refs, or force-push. Its optional guarded
  tag mode runs only after remote `main` equals local `HEAD` and publishes one
  annotated `lfm2-vl-mvp-X.Y.Z` tag that peels to that exact commit.
- Track the root `Cargo.lock` and `rust-toolchain.toml` as release inputs; keep
  `.tools/.secrets/`, models, caches, downloads, artifacts, and local logs out
  of publication.
- Stage only paths authorized by the affected overlay manifest and the root
  union verifier; never use broad staging.
- Keep committed LFM2-VL fixture JSON and Markdown LF-stable and fixture
  safetensors `-text` through root `.gitattributes`. Hash exact checkout bytes;
  never normalize line endings in a loader to hide a checkout-identity defect.

Gknome adoption is deferred outside the LFM2-VL product backlog. If revisited,
it must support ordinary native Windows repositories and recognize this
checkout's `.git` file, fail closed when its Git backend cannot operate through
the WSL pointer, and require a dry run with zero authority conflicts before any
generated file is accepted. Repair and additive template replacement remain
prohibited.

## Documentation Roles

- `STATUS.md`: compact current truth and most recent verification.
- `TODO.md`: active ordered work with completion conditions.
- `HISTORY.md`: detailed completed evidence formerly stored in status.
- `PARITY.md`: claim matrix and numerical boundaries.
- `FAILURE_LOG.md`: recurring hazards and prevention rules.
- `MOD_MANIFEST.md`: fork-versus-mod provenance and publication allowlist.
- `docs/FORK_OVERLAYS.md`: cross-overlay ownership, ordering, and shared paths.
- `summary_bank.json`: focused context routes, never a progress log.

---
AI-edited: 2026-08-13T20:08:58-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=ultra | task=repo-integrity | change=consolidated current orientation and removed duplicated release history
