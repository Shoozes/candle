# LFM2.5-VL Decisions

## D-0001: Direct Candle Fork

Status: Accepted

Decision:
Work directly from Candle 0.11.0 rather than building a wrapper around an unmodified dependency.

Why:
The planned implementation requires changes to LFM2 construction, embedding forwarding, model registration, examples, and quantized loading.

Consequences:
The repository retains upstream Candle history and should keep unrelated diffs minimal.

## D-0002: WSL2-First Development

Status: Accepted

Decision:
Use a Linux-home WSL2 checkout as the authoritative development and verification environment.

Why:
The execution plan calls for Linux filesystem behavior and avoids the permission, symlink, and performance problems of a Windows-mounted source tree.

Consequences:
The Windows-mounted edit worktree is not the build or verification authority.

## D-0003: CPU F32 Before CUDA

Status: Accepted

Decision:
All component parity must pass on CPU F32 before CUDA-specific work.

Why:
This separates model and preprocessing defects from accelerator precision and kernel defects.

Consequences:
Initial performance is not an acceptance criterion.

## D-0004: 450M Before 1.6B

Status: Accepted

Decision:
Use LFM2.5-VL-450M as the first production checkpoint, followed by 1.6B.

Why:
The 450M dimensions expose the current Candle normalization defect that the 1.6B dimensions can accidentally hide.

Consequences:
The 1.6B checkpoint remains a required second compatibility test.

## D-0005: Native Safetensors Before GGUF

Status: Accepted

Decision:
Prove native safetensors loading and CPU parity before adding quantized text, split mmproj, or direct GGUF mmproj support.

Why:
Starting with GGUF would combine model-math, preprocessing, tensor-name, layout, and quantization failures in one debugging surface.

Consequences:
GGUF work remains out of scope for Bootstrap Phase and the initial native parity gate.

## D-0006: Production Model Files Excluded From Git

Status: Accepted

Decision:
Do not commit production checkpoints, Hugging Face caches, generated runtime output, or local reference downloads.

Why:
These artifacts are large, mutable, and not part of the source or deterministic tiny-fixture contract.

Consequences:
Only reviewed deterministic tiny fixtures may be committed under `tests/fixtures/lfm2_vl_tiny/`.

## D-0007: Linked Edit and Linux Verification Worktrees

Status: Accepted

Decision:
Use `C:\DevStuff\candle-mods` only as the linked Windows edit worktree for Codex-authored changes. Keep the authoritative checkout and all verification work in Linux-home WSL2 worktrees. Builds and baseline checks never run from `/mnt/c` or `/mnt/d`.

Why:
The edit worktree is detached and linked to the authoritative WSL repository; the project execution plan requires Linux-home filesystem behavior.

Consequences:
Only Linux-home verification worktree evidence may be recorded as green. Cargo checks never run in the Windows edit worktree.

## D-0008: Local Verification Lockfile

Status: Accepted

Decision:
Keep `Cargo.lock` ignored and local to each Linux verification lane. Require it for every `--locked` check and record its SHA-256 with the retained proof.

Why:
Upstream Candle 0.11 intentionally ignores `Cargo.lock`, but the sprint requires locked local verification. Committing a workspace lockfile would change upstream repository policy before implementation evidence justifies it.

Consequences:
A fresh verification lane must resolve the lockfile deliberately, hydrate only the required local dependencies, and then run the phase verifier offline. Different lock hashes are different proof environments and must not be compared as identical baselines.

## D-0009: Immutable Authority Order

Status: Accepted

Decision:
Resolve every external source to an immutable revision and use the following authority order: official Transformers plus LiquidAI checkpoint files; mistral.rs as the Rust donor; llama.cpp for GGUF and independent parity; MLX-VLM and Transformers.js as secondary cross-checks; Candle only for local integration patterns.

Why:
The implementations evolve independently and have previously differed on image resizing, marker placement, positional interpolation, tiling, and config defaults.

Consequences:
A moving branch name is never sufficient evidence. When a secondary implementation disagrees with the pinned official config or Transformers output, the official behavior wins and the disagreement becomes a regression case.

## D-0010: Reference-First Adaptation Boundary

Status: Accepted

Decision:
Treat Transformers, llama.cpp, MLX-VLM, and Transformers.js as reference-only. Permit only narrow, explicitly attributed ports from the pinned MIT-licensed mistral.rs files, with the applicable notice retained.

Why:
Keeping the numerical oracle independent makes parity failures meaningful, while mistral.rs is the closest Candle-based donor and avoids re-inventing already-reviewed Rust structure.

Consequences:
Every directly adapted future file must cite repository, commit, path, license, and the parity test covering the port. No external implementation code was adapted during source locking.

## D-0011: Header-Only Production Tensor Inventory

Status: Accepted

Decision:
Use bounded safetensors HTTP Range reads to inspect production tensor headers without reading tensor payload bytes. Do not fetch production weights during source locking or reference-harness bootstrap.

Why:
The phase requires exact native names and shapes but explicitly excludes model downloads. Safetensors separates its JSON metadata header from tensor payloads.

Consequences:
`TENSOR_MAP.md` may record header-confirmed native shapes. Production numerics remain untested until the explicit production mode is invoked in a later phase.

## D-0012: Canonical LFM2 Text Normalization

Status: Accepted

Decision:
Normalize LFM2 text configuration at deserialization and checked conversion boundaries. When both spellings are supplied, pinned Transformers behavior gives legacy `block_ff_dim` precedence over `intermediate_size` and legacy `tie_embedding` precedence over `tie_word_embeddings`. Nested `rope_parameters.rope_theta` takes precedence over top-level `rope_theta`; absent FFN spellings retain the text compatibility fallback of `hidden_size * 4`. `full_attn_idxs` is accepted as an alias for the full-attention index list, while explicit `layer_types` remains authoritative. Preserve the public infallible `into_config` wrapper for valid legacy callers and provide `try_into_config` for fallible validation.

Why:
The official 450M and 1.6B text towers require effective FFN widths 4608 and 8192 respectively, while the VL checkpoints omit a separate tied `lm_head.weight`. A single normalized configuration path prevents alias drift between standalone and nested language roots.

Consequences:
Dense loading exposes `embed_tokens`, hidden-state forwarding, logit projection, and embedding-driven forwarding. Nested VL callers pass `model.language_model` directly through `new_from_parts`; standalone callers retain the `model` prefix. Quantized GGUF loading keeps its existing tensor aliases and adds only the embedding-driven route and cache reset API. The complete Phase 1 local gate is green.

## D-0013: Phase 1 Text Gate Evidence

Status: Accepted

Decision:
Accept the LFM2 text compatibility phase as green after its focused tests, broader library tests, existing examples, formatting, core crate checks, and staged/unstaged diff checks all passed in the Linux-home CPU/offline lane.

Why:
The 5 focused tests cover the required configuration aliases and precedence, 450M/1.6B effective FFN widths, legacy fallback, standalone and nested roots, tied and explicit heads, dense token-ID/embedding equivalence, committed-fixture merged prefill parity, three cached decode steps, cache-reset determinism, and quantized embedding-driven forwarding. All 18 `candle-transformers` library tests and the full locked/offline baseline also pass.

Consequences:
The implementation is checkpointed at `f660b8e3f2b4560f133356864e012be83f29d9c0` and tagged `lfm2-vl-phase-1-text`. The tiny deterministic fixture establishes the Phase 1 code path, but this decision does not claim production-checkpoint or GGUF numerical parity and does not cover any vision behavior.

## D-0014: SigLIP2 NaFlex CPU F32 Interpolation and Head Boundary

Status: Accepted

Decision:
Implement SigLIP2 NaFlex positional resizing as CPU F32 separable bilinear interpolation with `align_corners=false`, normalized antialiased triangular weights, and width-before-height composition. Cache resized position tensors per `(patch_rows, patch_cols)` behind a poison-safe `RwLock`. Keep attention score and softmax computation in F32, then cast softmax weights back to the original query/value dtype for the value matmul. Reject `vision_use_head=true` and unsupported pooling-head configurations.

Why:
The pinned Transformers behavior requires antialiased positional interpolation and a bidirectional key-padding mask, while the attached LFM2-VL specification explicitly requires the vision tower without a pooling head. A thread-safe cache preserves the immutable inference model's shareability, and the mixed-dtype attention order matches the official execution contract.

Consequences:
The focused Linux-home CPU F32 fixture gate is green at 7/7 tests, the broader `candle-transformers` library gate is green at 25 passed and 0 failed, and the full locked/offline baseline is green. Only the Phase 2 checkpoint/tag remains pending; this decision does not claim production-checkpoint, processor, projector, composite, CUDA, or GGUF parity.

---
AI-edited: 2026-08-10T01:15:43-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=siglip2-phase-2-decisions | change=recorded final Phase 2 gates with checkpoint pending
