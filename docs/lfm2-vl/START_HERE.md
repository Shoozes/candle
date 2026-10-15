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

Maintenance follows the published independent five-overlay source MVP
[`candle-overlays-mvp-0.2.0`](https://github.com/Shoozes/candle/releases/tag/candle-overlays-mvp-0.2.0)
at `9a7c65c2ebca67e7997ae1a73b652e38d0a3781f`. Its source package,
annotated tag and four asset hashes are verified; the immutable
`lfm2-vl-mvp-0.1.0` object/target remain unchanged. Pinned upstream remains
`c68b24997319b8d76e5a4e775dab197c7f26322a`.

Use `STATUS.md` for the current working-tree repair and proof, `TODO.md` for
the selected bounded maintenance task, and `HISTORY.md` for completed release
records. The [release contract at the published revision](https://github.com/Shoozes/candle/blob/9a7c65c2ebca67e7997ae1a73b652e38d0a3781f/docs/releases/CANDLE_OVERLAYS_MVP_0.2.0.md)
and its matching identity receipt own the frozen release inputs. Later source
maintenance does not replace that tag, archive or numerical evidence.

Existing LFM/VL/d1, Q8Rows, diffusion, compatibility and Qwen behavior remains
within its source-bound support matrix. GPT-OSS is experimental; its completed
short qualification is not long-context, coding or route acceptance. Deferred
research needs separate selection and fresh allowances; no model task starts
automatically after release. See `D1_LOADING.md`, `MAINTAINED_MODELS.md`,
`PARITY.md` and the owning overlay status only when that behavior is selected.

Consumers keep their qualified graphs independently: Edge uses `a0c70e1d`
with Q8/Nomic defaults and disabled GPT profiles, GenEye keeps `3fa7075e`,
SnapFlash keeps `1660f9fc`, and Symbio keeps Codex 0.154.0. Their package/native
proof and publication are separately owned; Candle publication does not repin
or activate them. Native Windows/MSVC remains primary; WSL2 is secondary,
with missing offline dependencies reported rather than fetched.


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

Then replay the portable baseline in WSL when its local cache is available.
The installed distro observed on 2026-10-08 is `Codex-Compat`; its current
offline cache lacks `accelerate-src`, so this optional lane remains blocked:

```powershell
wsl.exe -d Codex-Compat --cd /mnt/c/DevStuff/candle bash -lc "CARGO_NET_OFFLINE=true CARGO_TARGET_DIR=/mnt/c/DevStuff/candle/artifacts/wsl-target bash scripts/lfm2-vl/verify-baseline.sh"
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

Canonical checkout is `C:\DevStuff\candle` on local `main`. The same tree is
`/mnt/c/DevStuff/candle` when WSL is used for an optional
`scripts/lfm2-vl/verify-baseline.sh` replay. WSL does not own `main`. Native
Windows Git owns status, staging, commits, merges, and publication.
`C:\DevStuff\candle-mods` is retired; do not recreate it. The historical
feature branch at `/home/workbench/code/candle-lfm2-vl` is not a second
publication line.

- Read and edit in `C:\DevStuff\candle`.
- Use native Windows Git for status, staging, commits, merges, and revision
  checks.
- Do not force-attach the same branch to two worktrees.
- Keep `main` as the single local and GitHub integration branch; no PR is
  required for owner-reviewed work.
- Fetch and review `origin/main`, preserve both histories without force, and
  rerun the local release gate after every integration.
- Publish with the Candle-owned `.tools/gitpush.ps1` plus ignored
  `.tools/.secrets/gt.txt` after explicit owner approval and from a reviewed,
  clean named `main`. The helper never stages or commits. Fast-forward only;
  never force-push or use another repository's helper.
  Annotated `lfm2-vl-mvp-X.Y.Z` tags remain a separately authorized action
  after remote `main` equals local `HEAD`.
- Track the root `Cargo.lock` and `rust-toolchain.toml` as release inputs; keep
  `.tools/.secrets/`, models, caches, downloads, artifacts, and local logs out
  of publication.
- Stage only paths authorized by the affected overlay manifest and the root
  union verifier; never use broad staging.
- Keep committed LFM2-VL fixture JSON and Markdown LF-stable and fixture
  safetensors `-text` through root `.gitattributes`. Hash exact checkout bytes;
  never normalize line endings in a loader to hide a checkout-identity defect.

Gknome adoption is deferred outside the LFM2-VL product backlog. If revisited,
it must support this ordinary native Windows repository, require a dry run
with zero authority conflicts before any generated file is accepted, and must
not revive a linked worktree. Repair and additive template replacement remain
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
AI-edited: 2026-10-14 | agent=Codex/root | model=unknown | effort=unknown | task=overlay-mvp-review | change=routed maintenance from the completed source release
