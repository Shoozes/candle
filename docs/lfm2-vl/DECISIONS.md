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
The focused Linux-home CPU F32 fixture gate is green at 7/7 tests, the broader `candle-transformers` library gate is green at 25 passed and 0 failed, and the full locked/offline baseline is green. The Phase 2 checkpoint/tag is complete at commit `74e109aec5f9801cfead3eeb27fe3f93ac646b84`, annotated tag `lfm2-vl-phase-2-siglip2`; this decision does not claim production-checkpoint, processor, projector, composite, CUDA, or GGUF parity.

## D-0015: Native Packed-Tensor Projector and Composite Boundary

Status: Accepted

Decision:
Keep Phase 3 as a native tensor-level composition boundary: consume already-packed SigLIP2 crops, apply the config-driven factor-N projector, preserve crop/image ranges through `EncodedImages`, replace exactly one contiguous image-token span per image with an exact-length feature range, and run dense LFM2 prefill followed by ordinary cached decode.

Why:
This isolates projector and multimodal embedding composition from the later raw-image processor, tokenizer/chat-template, GGUF, CUDA, and CLI phases while proving the full merged embedding and cache path against the committed official tiny fixture.

Consequences:
Phase 3 focused proof passed 11/11. Projector-stage maximum absolute error was `5.960464478e-8`; encoded and merged embeddings were `6.519258022e-9`; prefill was `4.470348358e-8`; cached decode was `2.980232239e-8`. The checkpoint is complete at `37264b49cf74d0cf7697317eda0183f084db6ff8`, tagged `lfm2-vl-phase-3-native-composite`. Production-checkpoint parity, raw-image preprocessing, tokenizer/chat-template behavior, CUDA, GGUF, and CLI support remain unclaimed.

## D-0016: Contiguous Batched SigLIP2 Attention Layout

Status: Accepted

Decision:
Materialize the tensor returned by SigLIP2 `split_heads` after its transpose before batched attention matmul.

Why:
The real multi-crop Phase 3 path exposed `MatMulUnexpectedStriding` when batched attention received a non-contiguous transposed left-hand operand. Making the layout contiguous preserves values and resolves the runtime layout precondition without changing attention math.

Consequences:
The repeated-crop SigLIP2 regression passed 8/8, and the full `candle-transformers` library gate passed 37/37 in the manager's locked/offline CPU lane. The fix is a layout compatibility measure, not a CUDA optimization or a production-checkpoint parity claim.

## D-0017: Rust-Native Processor Crate and Official Oracle Boundary

Status: Accepted

Decision:
Place reusable image and prompt processing in the small `candle-vlm` workspace crate while keeping `candle-transformers` independent of image codecs and tokenizers. Resolve configuration as `explicit override > processor JSON > GGUF metadata > model config > architecture defaults`; use the one canonical projected-token function exported by the native model; resolve every marker ID through the tokenizer; and preserve one placeholder span per crop.

Why:
This removes the Python runtime dependency without mixing image decoding or chat-template concerns into model math. The boundary lets raw-image and tokenizer behavior be proved independently against the pinned Transformers/TorchVision oracle while the Phase 3 packed-tensor model remains reusable.

Consequences:
The Phase 4 Rust suite passes 24/24 across all required image modes, shapes, tiling cases, prompt placements, multiple images, and controlled failures. Packed integer metadata and prompt strings/IDs/spans are exact; normalized pixels differ by at most `1.192092896e-7`. Production checkpoint loading, GGUF/mmproj, CUDA, generated captions, and CLI integration remain separate gates.

## D-0018: Captured TorchVision Byte Resize Semantics

Status: Accepted

Decision:
Implement the pinned Torch `2.8.0+cpu` and TorchVision `0.23.0+cpu` bilinear-antialias path as separable F32 filtering followed by byte rounding. Use contracted accumulation for the short vertical support windows observed in the pinned CPU kernel, scalar accumulation for longer dynamic windows, and a bounded F64 shadow only to disambiguate exact half ties by a full F32 output ULP. Allocate internal resize and packed-processor buffers through checked, fallible reservation.

Why:
Generic image-library resize helpers and fixed-point approximations did not preserve the pinned uint8 processor's boundary behavior. The captured order matches the official fixture for normal, upscaled, tiled, and thumbnail paths while retaining deterministic debug/release bytes and controlled failure on impossible capacities.

Consequences:
All 12 full processor tensors pass with maximum normalized error `1.192092896e-7`, and the direct odd-size regression matches all 96 output RGB bytes exactly. The rule is pinned-oracle compatibility, not a claim that every future TorchVision version uses the same internal accumulation strategy.

## D-0019: Versioned Split-MMProj Boundary and Single-Buffer Loading

Status: Accepted

Decision:
Define Format B as a strict three-file bundle: dense canonical vision/projector tensors in `mmproj.safetensors`, a versioned `mmproj.json`, and canonical `processor_config.json`. Derive the only accepted tensor names and shapes from the embedded LFM2-VL config, require immutable source provenance and artifact hashes, and validate the complete inventory before export and load. Open the weights file once into a fallibly allocated buffer bounded by the validated payload plus the maximum safetensors header; hash, inspect, and construct the model from those same bytes. Keep processor metadata as neutral JSON in `candle-transformers`, with the typed conversion owned by downstream `candle-vlm`.

Why:
The split bundle must pair safely with independently sourced quantized text. Exact config-derived inventory prevents incomplete artifacts, immutable hashes make pairing diagnosable, and a single file identity removes replacement races between verification and construction. The crate boundary avoids reversing the existing `candle-vlm -> candle-transformers` dependency.

Consequences:
The loader rejects architecture, hidden-size, text-layer, patch/factor, image-token, tensor-count, name, shape, dtype, byte-count, offset, overlap, gap, payload, processor, provenance, and hash mismatches before inference. Buffered eager loading temporarily holds the dense MMProj file in host memory, but its allocation is checked and manifest-bounded. Direct GGUF MMProj remains a separate Phase 6 format.

## D-0020: Deterministic Real-GGUF Hybrid Proof and CUDA Skip Boundary

Status: Accepted

Decision:
Prove the Phase 5 text path with deterministic GGUF bytes written from the committed tiny language tensors and loaded through the public GGUF parser/constructor. Quantize every block-aligned tiny matrix to Q8_0 and retain F32 for small shapes that the format cannot represent as Q8_0. Require split/native image-feature equivalence, multimodal prefill, three cached decode steps, and cache-reset comparison. Keep a feature-gated CUDA-vision/CPU-text integration test in source, but record local execution as skipped when the owner machine lacks the Linux CUDA toolkit.

Why:
A synthetic in-memory quantized model would not prove file-format aliases, metadata, output tying, or the real GGUF constructor. The mixed tiny GGUF is deterministic and exercises Q8 while remaining format-valid. Local policy treats a missing optional toolchain as a truthful evidence gap, not permission to install dependencies or claim executed CUDA parity.

Consequences:
The pinned tiny GGUF SHA-256 is `8fbd510aeea4715547c57975a7adcb91c148a8bc5e8d869d9617b69af6a006b1`. CPU F32 split/native image features are exact; hybrid prefill and decode remain within `4.457309842e-5`. The source path transfers projected features only at merge. Executed CUDA parity and production-GGUF parity remain explicitly unclaimed.

---
AI-edited: 2026-08-10T06:04:00-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=lfm2-vl-phase-5-docs | change=recorded split artifact/load boundary and deterministic hybrid evidence policy
