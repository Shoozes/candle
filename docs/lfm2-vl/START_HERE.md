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

Phases 1 through 7 are checkpointed. The published review checkpoint is `c9b60f0b906fa8fe70423295e2e1164648a8fa53` on `Shoozes/candle:feat/lfm2-vl-mmproj`. Native Windows is the product and release-proof platform; WSL2/Linux is a secondary portability replay. NR-5B official 450M native Windows CPU-F32 component parity, P2 official-base GGUF same-artifact decoded-output comparison, and P3's no-model 1.6B admission forecast are green. The exact next task is a separately guarded acquisition of the absent 3,198,084,631-byte 1.6B regular snapshot; stop before model load.

Do not load the 1.6B checkpoint until its immutable inventory, resource forecast, reviewed Job ceiling, and fresh preflight are safe. Do not start CUDA inference before P3 is green, or any model while a prior inference/build process or host-memory pressure remains.

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

These commands perform compilation and tests only. They do not authorize dependency downloads, model execution, hosted CI, commits, pushes, or PRs. A missing offline dependency is a blocked lane to record in `STATUS.md`, not permission to fetch it implicitly.

## Production Parity Safety

Before any production-model run:

1. Run `scripts/lfm2-vl/preflight.ps1 -AsJson` and retain its read-only report outside Git; `review` still requires explicit owner approval, while `blocked` stops admission.
2. Run `tools/lfm2_vl/reference/inspect_artifact.py --model <450m|1.6b> --model-dir <regular-file-snapshot> --output <external-manifest.json> --allow-production` and verify the exact repository revision plus every model-snapshot config, tokenizer, processor, index, and weight identity. Pass that same `<regular-file-snapshot>` as Python oracle `--model-dir` and native Candle `--model-dir`; the Python trace refuses cache-only or download-backed model resolution, both lanes rehash their inputs after inference, and the comparator requires Candle's consumed-file sizes and hashes to match the oracle manifest. Record the source-image identity separately in each trace. This is a local-only hash pass; it never downloads or serializes weights.
3. Record physical memory, Windows committed bytes and limit, GPU memory, and all existing inference PIDs.
4. Refuse concurrent large-model execution.
5. For Transformers parity, run the bounded Python oracle first. Its `--prompt` is user text and its `metadata.json.prompt` is the exact official chat-template output containing the image sentinel. Pass that recorded prompt value plus the same image to Candle CPU F32 natively on Windows; do not pass the untemplated user text. Use WSL only as a later portability replay.
6. Launch every production Python, Candle, or llama.cpp inference executable through `scripts/lfm2-vl/run-bounded-oracle.ps1`; never wrap `cargo run`. Use name-wide concurrency for unique model tools, exact-executable concurrency for generic interpreters, and retain `-LogPath` so nonzero children preserve their combined stdout/stderr.
7. Verify the owner and complete process tree exit, then repeat the host/GPU census.
8. Stop if cleanup, memory recovery, tensor parity, or deterministic replay fails.

`FAILURE_LOG.md` F-0008 is the resource-containment authority.

## Git and Worktree Boundary

This particular Windows folder is a WSL-owned linked worktree. At the published checkpoint it is detached at `c9b60f0b906fa8fe70423295e2e1164648a8fa53`; the named feature branch is already checked out by `/home/workbench/code/candle-lfm2-vl`. Windows Git cannot resolve the Linux absolute `.git` pointer. This is a local Git topology, not a requirement for building or using the fork on Windows.

- Read and edit here when requested.
- Use WSL Git for status and diff inspection.
- Do not force-attach the same branch to two worktrees.
- Land changes from an intentionally named WSL branch/worktree or transfer the reviewed patch to the owning worktree.
- Keep `.tools/.secrets/`, the ignored `Cargo.lock`, models, caches, downloads, artifacts, and local logs out of publication.
- Stage only paths authorized by `MOD_MANIFEST.md`; never use broad staging.

Gknome adoption must support ordinary native Windows repositories and recognize this checkout's `.git` file. It must fail closed when its Git backend cannot operate through the WSL pointer. A dry run is mandatory before any generated file is accepted. The latest Candle dry run (`20260811T032224Z-4a87c2b8`) is inventory-clean but remains blocked on four project-authority conflicts; apply and additive repair are both prohibited until TODO C2's completion gates are met.

## Documentation Roles

- `STATUS.md`: compact current truth and most recent verification.
- `TODO.md`: active ordered work with completion conditions.
- `HISTORY.md`: detailed completed evidence formerly stored in status.
- `PARITY.md`: claim matrix and numerical boundaries.
- `FAILURE_LOG.md`: recurring hazards and prevention rules.
- `MOD_MANIFEST.md`: fork-versus-mod provenance and publication allowlist.
- `summary_bank.json`: focused context routes, never a progress log.

---
AI-edited: 2026-08-11T09:15:59-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=docs | change=fixed Gknome routing and checkpoint-neutral inspection guidance
