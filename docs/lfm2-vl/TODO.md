# LFM2.5-VL Active Backlog

Only incomplete work belongs here. Completed implementation and proof belong in `HISTORY.md`; recurring hazards belong in `FAILURE_LOG.md`. Execute these items in order unless an earlier gate exposes a correctness prerequisite.

## P3 — Official 1.6B Native CPU-F32 Component Parity

What: Repeat the proven NR-5B component gate for `LiquidAI/LFM2.5-VL-1.6B@919fde3d022e3f90a4716006f993938ee8c2eb97`.

Why: The larger checkpoint changes text width, layer inventory, tensor count, trace size, and peak-memory risk. The 450M result cannot establish its numerical or operational behavior.

When: Current product task. P2 and P3's no-model admission forecast are green; the guarded external artifact acquisition is next. Never overlap the 1.6B oracle/native runs with llama.cpp, Cargo, rustc, or another model.

Where: The NR-5B reference/native trace paths, `docs/lfm2-vl/PARITY.md`, and a separate external 1.6B artifact/evidence directory.

Completed admission: the exact Python 3.10.11/42-distribution environment is green; the pinned 589-tensor header is locked; official HEAD metadata fixes the eight-file snapshot at 3,198,084,631 bytes; the local cache is absent; the external 5,587-byte forecast (`0c8f3cd31cea807591356d90aa442a2a02421e86a58215c01b4bcecc12659a59`) defines stage-specific 16/24/12 GiB dry-load/Python-trace/native-trace ceilings; and the guarded acquisition plan is green without creating any path. No payload was downloaded or loaded.

How:

1. Re-run `acquire_snapshot.py --plan`, then, only with separate owner approval, invoke the exact `.venv` Python and `--allow-production-download` through `run-bounded-oracle.ps1` directly from the current PowerShell process. Use a 2 GiB Job ceiling, 7,200-second timeout, executable-scoped concurrency, and external log/owner evidence. Create the eight-file regular snapshot at the pinned revision; require 12 GiB free space first, use public/no-token serial Hub access with Xet disabled before import, hash every local file, and require `model.safetensors` to match expected 3,193,334,216 bytes and LFS SHA-256 `7fc7458e4382fc6e558cfdda45857fbf9ab5b40a8bf199c9cd073003b14ac26d` before any load. Snapshot and manifest publication must remain atomic and no-clobber. A killed transfer may resume from cache but cannot make a snapshot admissible; any leftover snapshot stage, manifest stage, output, or manifest blocks retry for inspection. Install the full CPU oracle lock only for later traces.
2. Before model load, run the stdlib-only config inspector against the acquired `config.json`, `processor_config.json`, and `tokenizer.json`. Require image token ID 396, at least one row/column marker, unique in-range wrapper/thumbnail/grid IDs, and retain the complete mapping. Reconcile the `image-wrapper-token-ids` source-lock candidate without hardcoding runtime IDs.
3. Run a fresh census before each process. Require at least 32 GiB available physical and commit headroom for the Python trace, and stop on any llama/model/Cargo/rustc process. Do not raise a Job ceiling automatically after a limit termination.
4. Run the pinned Python dry load under 16 GiB and deterministic trace under 24 GiB, verify child-tree exit, then run the already-built native CPU-F32 trace under 12 GiB with the oracle's exact rendered prompt.
5. Compare exact inputs, complete stage inventory, selected component tensors, cached decode, artifact identity, and reset replay; retain pre/post resource reports.
6. Diagnose and fix any mismatch before proceeding; do not relax tolerances merely to pass.

Done when: Every selected 1.6B tensor passes its recorded stage-specific tolerance, input and reset tensors are exact, all files are identified and reverified, both bounded processes are absent, and host/GPU memory recovers.

Verification: Exact environment lock; artifact/trace bundle validation; machine-readable comparator; focused reference and Rust tests; native locked/offline gate; process and memory evidence.

## P4 — CUDA and Distinct-Device Execution

What: Execute the existing native Windows CUDA-vision/CPU-text boundary, then add only measured, evidence-backed CUDA optimizations.

Why: The transfer boundary is source-complete but unexecuted. CPU proof cannot establish DLL/toolkit integration, device placement, kernel behavior, or accelerator memory safety.

When: After P3 is green and the owner confirms an existing compatible native Windows CUDA toolkit. Never install or update a toolkit implicitly.

Where: `candle-transformers/src/models/lfm2_vl/`, `candle-transformers/src/models/siglip2.rs`, focused distinct-device tests, native build configuration, and external evidence.

How:

1. Record MSVC, CUDA toolkit, driver, GPU, feature flags, executable/DLL identity, and a clear resource preflight.
2. Build once with bounded concurrency; verify the executable and dependency closure before loading a model.
3. Run the smallest distinct-device fixture, then the selected 450M production trace with explicit placement and containment.
4. Measure transfers and peaks; ensure only projected image features cross the device boundary.
5. Optimize only measured bottlenecks behind existing feature gates, replay CPU regressions, then repeat production parity.

Done when: The focused distinct-device test and selected production CUDA parity pass, CPU-only builds stay green, device transfers match policy, exact cleanup succeeds, and peak host/GPU memory is retained.

Verification: Feature-gated CUDA test; CPU regression replay; scoped Clippy; local Windows baseline; exact runtime/DLL and resource evidence; optional WSL replay labeled separately.

## C1 — Split Large Modules at Proven Seams

What: Reduce context and merge pressure in the largest LFM2-VL modules without changing public behavior or serialized evidence.

Why: `gguf.rs`, `weights.rs`, `runner.rs`, `processor.rs`, `model.rs`, `lfm2.rs`, `prompt.rs`, `siglip2.rs`, and `native_loading.rs` each exceed 1,200 lines. Splitting without measured seams would add indirection; leaving unrelated responsibilities together impedes review and scaling. The context bank now separates native model math from checkpoint loading, so a physical split is not an urgent substitute for the P3 product gate.

When: After P3, or earlier only when an active product fix already crosses one named seam.

Where: The named modules, their nearest tests, and the relevant `summary_bank.json` routes.

How:

1. Measure line/context size and call/test ownership before editing.
2. Split GGUF metadata/name normalization, tensor decode/layout, and model construction.
3. Split native weight inventory validation, canonical shape/layout conversion, and builder wiring.
4. Split runner artifact evidence, deterministic generation, trace serialization, and report emission.
5. Split processor/prompt only at image preprocessing versus token/span validation; retain shared checked-arithmetic helpers.
6. Re-measure routes and remove only proven duplication; do not create a generic VLM framework.

Done when: Each selected module has one clear responsibility, public APIs and evidence schemas are unchanged, routes load no more context than before, and every prior focused/full test remains green.

Verification: Before/after size report; import/compile checks; deterministic fixture and trace hashes where applicable; full local baseline; complete diff inspection.

## C2 — Adopt Gknome Safely Across Native Windows and This Linked Worktree

What: Integrate only useful Gknome project/context controls while preserving this mature repository's authority and safely refusing its Linux-absolute `.git` pointer.

Why: Native Windows is the normal product workflow, but this checkout remains a WSL-owned linked worktree even though it is now intentionally attached to local `main`. Windows Git still cannot resolve the Linux absolute pointer. The latest dry run correctly applied nothing and found four authority conflicts; bypassing them could overwrite project policy or context routing.

When: After Gknome produces a reviewed zero-conflict mature-repository plan. Do not use repair or implicit template replacement.

Where: Gknome at `C:\Users\jc816\OneDrive\Desktop\Gen-App\gknome`, this repo's non-secret generated controls, and `summary_bank.json`. Never read or replace `.tools/.secrets/`.

How:

1. Define a preserve-or-reviewed-merge contract for existing `.gitignore`, `AGENTS.md`, and `README.md`.
2. Map generated context routes onto existing groups or perform a staged reviewed merge; preserve the 256 KiB budget unless measured splitting proves a change.
3. Rerun the dry plan and require `existing_repository=true`, zero runtime/cache/secret inputs, explicit linked-Git refusal, byte-preserved authorities, and zero unresolved conflicts.
4. Apply only the reviewed plan as a separate action, then run generated project/context tests and add only accepted paths to the mod manifest and context bank.

Done when: Dry-run and adopted-project tests pass, no authority/runtime/secret path is treated as template input, the `.git` topology is reported truthfully, the context bank remains valid, and no commit/push/remote mutation occurs implicitly.

Verification: Gknome adoption/layout/inventory tests; zero-conflict plan JSON; before/after authority hashes; generated-file diff; context/project tests; WSL Git status when available; Candle summary-bank/mod-manifest verifiers; secret-tree exclusion with zero value exposure.

## C3 — Replay Linux Native-Trace Exclusive Publication

What: Run the exact native trace destination-race regression on a local
Linux/WSL Rust toolchain.

Why: Windows proves the user-facing no-clobber contract, and the Linux source
uses `renameat2(RENAME_NOREPLACE)`, but the installed `NVIDIA-Workbench` WSL
distribution currently has no `cargo`. The installed Linux Rust target is not
enough: offline cross-checking stops at `openssl-sys` without a Linux OpenSSL
sysroot. Source review is not a substitute for compiling and executing the
platform-specific branch.

When: After P3, or opportunistically when an existing local WSL Rust toolchain
is available. Do not install a toolchain or fetch dependencies implicitly.

Where: `candle-examples/examples/lfm2-vl/trace.rs` and the existing
`trace::tests::trace_publication_does_not_replace_a_racing_directory` test.

How:

1. Record the explicit WSL distribution, kernel, Cargo/Rust versions, and
   offline dependency availability.
2. Run the exact trace collision test with `--locked --offline -j 2`.
3. Confirm the competing owner directory remains intact and no staging
   directory survives.
4. Replay formatting and the native Windows example gate if a Linux-only fix
   is required.

Done when: The exact test compiles and passes on Linux without replacing the
racing directory, leaves no temporary output, and the Windows 29-test example
gate remains green.

Verification: Exact Linux test command and exit code; post-test temporary-path
inventory; native Windows example replay; `git diff --check`.

---
AI-edited: 2026-08-11T09:59:19-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=release | change=updated Gknome's linked-main boundary without changing active product priority
