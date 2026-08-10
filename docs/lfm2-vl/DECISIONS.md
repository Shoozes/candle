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

## D-0021: Strict Dense Compatibility Boundary for Direct GGUF MMProj

Status: Accepted

Decision:
Implement Format C first as a strict one-handle GGUF loader that reconstructs only the SigLIP2 tower and LFM2 projector, validates the complete metadata/tensor/range contract before construction, and dequantizes supported GGML tensors into F32, F16, or BF16 native operators. Require `general.architecture=clip`, `general.type=mmproj`, `clip.projector_type=lfm2`, the locked vision facts, and an exact config-derived inventory. Apply only the header-proven patch inverse `permute(0,2,3,1) -> contiguous -> reshape`; keep every other matrix in Candle `[out,in]` order.

Use caller-specific GGUF parser limits before allocation: at most 16,384 tensors, metadata records, and array elements; 1 MiB strings; a 16 MiB aligned header; an 8 GiB file; and checked 8 GiB retained-dense and conservative peak-allocation bounds. Reject duplicates, invalid alignment, missing/unexpected tensors, unsupported target dtypes, malformed offsets, overlaps, truncation, incomplete optional LayerNorm/bias pairs, and configuration mismatches before inference. When the official GGUF omits preprocessing metadata, retain pinned LFM2-VL architecture/processor defaults; resolve the image token ID from the tokenizer. Do not introduce a quantized vision operator in this phase.

Why:
The dense path separates format/orientation compatibility from Phase 7 operator design. Exact official headers resolve the prior orientation ambiguity without downloading production payloads, while narrow parser and memory limits keep untrusted artifact declarations from allocating according to generic GGUF maxima. Processor defaults and tokenizer-derived IDs are outside the official MMProj header and must not be silently replaced with false or hardcoded values.

Consequences:
The deterministic dense MMProj GGUF hash is `7361b57e6d9dbf2d7809d4f446944fdc7325b368e4444fee2bc3497376695256`; direct dense/native image features are exact. A synthetic Q8_0 MMProj dequantizes with image-feature max abs `8.463021368e-5`. Direct GGUF MMProj plus deterministic quantized text reproduces the Phase 5 hybrid prefill/decode errors and exact cache reset. The official F16/Q8_0 evidence remains header-only with zero retained payload bytes, so production numerical parity and native Q8 execution remain separate gates.

## D-0022: Native Q8_0 Linears with an Explicit Dense Fallback

Status: Accepted

Decision:
Route SigLIP2 Q/K/V/out, SigLIP2 MLP up/down, and projector linear 1/2 through a focused `LinearOp`. Dense construction keeps `candle_nn::Linear`; native Q8 construction retains the GGUF `QTensor` directly as `QMatMul::QTensor` and applies the dense bias after matmul. Keep patch projection, positions, LayerNorm parameters, biases, and any dense eligible matrix in the existing dense path.

Keep `load_gguf` and `from_gguf` as the Phase 6 dense compatibility APIs. Add explicit native-Q8 APIs that require F32 activations and at least one valid Q8_0 linear, plus automatic APIs that choose native Q8 for valid F32 Q8 artifacts. Validate exact inventory, tensor roles, GGML dtype, Q8 input-width alignment, ranges, and allocation bounds before reading tensor payloads. Reject lower-bit native weights and Q8 tensors in dense-only roles. Have the example use automatic selection and print its resolved execution mode and retained-Q8 count.

Why:
Direct `QMatMul::QTensor` construction guarantees that this path cannot be silently converted to dense by `CANDLE_DEQUANTIZE_ALL`. A separate dense API preserves Phase 6 behavior and supports F16/BF16 or compatibility-first callers. Role-aware mixed construction handles checkpoints where a dimension prevents Q8_0 without hard-coding the 450M inventory. Explicit diagnostics make fallback behavior observable.

Consequences:
The two-layer block-aligned fixture retains all 14 eligible linears and records native/dense projected-feature max abs `5.300968885e-3` with cosine `0.999923348`. The committed hybrid fixture records image-feature max abs `1.533385366e-4`, prefill `1.650899649e-4`, cached decode no worse than `7.853843272e-5`, and exact cache reset. Native F16/BF16 activations, lower-bit vision execution, production-payload comparison, llama.cpp runtime parity, and executed native-Q8 CUDA remain outside this decision.

## D-0023: Shared Request-Wide Vision Safety Limits

Status: Accepted

Decision:
Define `VisionLimits` in the dependency-neutral `candle-transformers` LFM2-VL module and re-export it from `candle-vlm`. Use the same limits at the raw-image processor, prompt/metadata, packed native model, split MMProj, direct GGUF MMProj, and quantized-text composition boundaries. Preserve the existing `encode_images` APIs with safe defaults and add explicit `encode_images_with_limits` variants.

Set the default and absolute implementation ceilings to 67,108,864 pixels per source or derived image surface, 16 images, 11 crops per image, 64 crops per request, 1,024 patches per crop, and 65,536 projected tokens per request. Processor/model/GGUF/explicit configuration may tighten these values through the normal precedence chain but cannot raise the hard ceilings. Treat `max_source_pixels` as a surface-allocation bound: it applies to source images, resized images, tiled canvases, crops, and packed `ImageMeta` surfaces.

Run raw-image request preflight before RGB conversion, resizing, crop extraction, patchification, or packed-tensor reservation. Revalidate externally supplied batches in prompt expansion and at the packed model boundary. Validate tensor shapes, image/crop ranges, resized surfaces, spatial values, masks, projected-token counts, and vision batch size before an MMProj moves tensors across devices. Use checked arithmetic and fallible reservations for prompt expansion and reject predicted context overflow before constructing the expanded image string.

Mark the two public `candle-vlm` processor configuration structs non-exhaustive. `candle-vlm` is new on this unreleased feature branch, so adding the initial safety contract is an intentional pre-release source-boundary change; no Candle 0.11 upstream public API is removed or changed.

Why:
Image dimensions, processor documents, packed tensors, and metadata can be untrusted. Checked arithmetic prevents integer overflow but does not stop a valid enormous allocation, and shallow shape checks do not protect a device transfer from malicious mask/spatial metadata. A shared, ceiling-bounded contract makes rejection order consistent and prevents individual entry points from bypassing request-wide budgets.

Consequences:
Exact-limit inputs remain valid; one-over, overflow, zero-limit, hard-ceiling, malformed metadata, and pre-transfer semantic cases return controlled errors. The existing tiny processor fixture remains numerically unchanged. The processor currently receives an already-decoded `DynamicImage`; a future file/CLI decoder must inspect dimensions before full decode where the codec permits it.

## D-0024: Explicit Example Dtype and MMProj Execution Policy

Status: Accepted

Decision:
Keep the `lfm2-vl` example dependency-free and move its fallible argument parser into a focused module. Preserve both the original three-position split-MMProj form and the explicit `--model-file` plus `--mmproj-file`/`--mmproj-dir` form, including `--processor-config`, `--cpu`, and `--vision-cpu`.

Add explicit `--dtype f32|bf16|f16` and `--mmproj-execution auto|dense|q8` policy flags. An omitted dtype preserves the existing device-dependent policy: F32 on CPU and BF16 on CUDA. Split safetensors MMProj input is always dense. Direct GGUF `auto`, `dense`, and `q8` requests route respectively to `load_gguf_auto`, `load_gguf`, and `load_gguf_q8`. Strict Q8 requires a direct GGUF MMProj and F32 activations; reject either policy violation before opening the text model or tokenizer.

Print requested/defaulted and resolved vision dtype separately, and print requested and resolved MMProj execution plus retained native-Q8 tensor count. Treat `float32`, `bfloat16`, and `float16` as dtype aliases; treat `dequantize`, `q8_0`, and `native-q8` as execution aliases. Keep device placement as a separately tested policy consumed by `main`.

Why:
The previous automatic path hid whether execution was selected by user intent or loader fallback. Explicit policy makes compatibility, benchmarking, and failure diagnosis reproducible while retaining every existing invocation form and avoiding a new CLI dependency.

Consequences:
Parser and load-plan tests cover defaults, every dtype spelling, aliases, conflicting path forms, missing/unknown values, CPU/vision-CPU placement, the complete input/execution routing matrix, and pre-I/O Q8 rejection. This decision changes only the local example interface and loader selection; model math, checkpoint formats, automatic GGUF semantics, production downloads, and CUDA claims are unchanged.

## D-0025: Bounded Unmodified Native Checkpoint Loading

Status: Accepted

Decision:
Add a local-only example loader for an unmodified Hugging Face LFM2-VL directory. Require exactly one unified `model.safetensors` or `model.safetensors.index.json`, canonicalize every referenced file beneath the model root, and bound JSON/header/file/aggregate bytes plus shard, tensor, shape, offset, and payload coverage before memory mapping. Validate the complete normalized configuration-derived inventory before constructing any model tensor.

Resolve the official `model.vision_tower.vision_model`, `model.multi_modal_projector`, and `model.language_model` namespaces directly while retaining the shorter vision root used only by the committed tiny fixture. Reuse `model.language_model.embed_tokens.weight` when output embeddings are tied; require `lm_head.weight` otherwise. Require and pair `config.json`, `processor_config.json`, and `tokenizer.json`, with an optional explicit processor override. Resolve default text and vision dtypes independently from their devices; an explicit dtype applies to both. Treat the model directory as an immutable local snapshot for the lifetime of the memory-mapped model.

Why:
Native safetensors is the first production artifact format in the required execution order. Loading renamed or pre-split files would avoid proving the official namespace and shard contract, while mapping before full inventory validation would turn malformed external configuration into large allocations or partial construction. Separate component builders preserve distinct-device execution without silently forcing CPU vision to determine CUDA text dtype.

Consequences:
The focused suite constructs real single-file and indexed tiny checkpoints and covers canonical/direct vision roots, tied/explicit heads, processor/tokenizer mismatches, traversal, missing files, wrong shard mappings, bad index sizes, duplicate tensors, malformed inventories, and independent BF16/F32 component loading. Header-only tests compare every expected official name, BF16 dtype, and exact shape through canonical SHA-256 inventories, derive exactly 349 and 589 tensors, and assert raw FFN 6,656/12,288 normalization to 4,608/8,192. Production payloads, inference, and numerical parity remain separate gates.

## D-0026: Local llama.cpp Is a Same-Artifact Parity Oracle

Status: Accepted

Decision:
Use `C:\llamacpp` read-only as an execution oracle only when Candle and llama.cpp receive the same text model, MMProj, tokenizer, processor policy, image bytes, prompt, context, and deterministic decode settings. Keep the pinned llama.cpp revision in `SOURCES.md` as the implementation authority; record the installed runtime build separately and do not imply commit identity when it is unproven. Never compare the discovered fine-tuned SFT pair against official-base Candle weights and call the result parity.

Build a deterministic Candle runner before executing comparisons. Compare exact preprocessing structure and prompt/token data where both runtimes expose it, then greedy decoded token sequences and output text. If the installed llama tools do not expose logits or intermediate tensors, mark those stages unavailable; captions or token agreement do not substitute for component-tensor parity.

Why:
Near-1:1 behavior is only meaningful under artifact identity. The installed b9981 runtime and local fine-tuned GGUF pair are useful independent evidence, but neither is the pinned source build nor the official base checkpoint. A strict same-artifact matrix prevents model, tokenizer, processor, sampling, or prompt differences from being misdiagnosed as an implementation defect.

Consequences:
No local llama.cpp inference runs until the GGUF/MMProj/tokenizer/processor pair is validated. Runtime comparisons will report the installed build, exact file identities, command lines, and available evidence stages. Official production parity still requires the pinned official artifacts and remains unclaimed.

## D-0027: Deterministic Inference Is a File-Identified Replay Contract

Status: Accepted

Decision:
Expose one bounded inference path for native safetensors and quantized-text hybrid models. Resolve and hash every exact file consumed by the loader rather than treating a model directory as artifact identity. Native evidence includes config, processor, tokenizer, optional index, every shard, and an optional processor override. Split-MMProj evidence includes its manifest, safetensors, processor document, text GGUF, and tokenizer. Direct-GGUF evidence includes the text GGUF, MMProj GGUF, tokenizer, and an explicit processor override when present. Directory-only evidence is rejected. Treat every local input as an immutable snapshot from loader open through report emission; mutable replacement during that interval is unsupported because native weights are memory-mapped and post-load hashing otherwise cannot identify the bytes already consumed.

Process each image and encode its vision features once, then execute deterministic greedy prefill and cached decode twice from a cleared text cache. Require complete trace equality across both runs, including generated IDs/tokens, full F32-logit SHA-256 values, stable top-k with lower-token-ID tie breaking, stop reason, and decoded forms. Reject empty/non-finite/oversized logit surfaces. Resolve EOS by `CLI > native model config or GGUF metadata > tokenizer candidate`, and record the source. Emit a versioned one-line JSON record without timings.

Why:
Decoded text alone cannot distinguish artifact drift, prompt mismatch, cache leakage, nondeterminism, or a plausible but numerically unrelated answer. Canonical paths plus file sizes and hashes make comparisons reproducible, while exact replay makes cache reset an observed invariant rather than a claim. Timings are excluded because they would make otherwise identical evidence records differ.

Consequences:
Committed tests cover native and real split-MMProj hybrid image prefill, cached decode, EOS provenance, all consumed-file hashes, exact direct/split/override source lists, and exact reset replay. The local fine-tuned text GGUF plus byte-identical official MMProj produces the same eight-token caption in Candle and llama.cpp under aligned deterministic settings. That result is same-artifact runtime evidence only: official-base text parity and llama.cpp component/logit equality remain separate gates.

## D-0028: Immutable llama.cpp Bundles and Suspended Job Containment

Status: Accepted

Decision:
Keep the incident b9981 install, the user-supplied current b10344 tools, and the exact pinned source build as separate immutable bundles. Identify each by source/release revision, build configuration, executable/DLL inventory, size, and SHA-256. Do not repair suspected mixing by deleting or updating in place; first prove the bundle's internal provenance and import resolution.

Run every expensive Windows llama.cpp build or oracle process through `scripts/lfm2-vl/run-bounded-oracle.ps1`. Serialize matching process names, create the child suspended, assign it before resume to a kill-on-close Job Object with per-process and per-job memory limits, enforce timeout, and require exact PID absence. Default CUDA graphs off for Windows CUDA/MTMD work while related upstream leak reports remain plausible; any override is explicit evidence. Admit a model only after a no-model identity probe and before/after physical, commit, and GPU memory census.

Why:
The completed b9981 parity run left approximately 131.5 GB of private allocation and could not be terminated through normal or elevated-looking paths. PID disappearance did not restore usable host performance; restart was required. Bundle audit found no DLL-mixing or Defender evidence, while WER recorded `RADAR_PRE_LEAK_64` and official llama.cpp reports describe related CUDA/MTMD memory growth. Containment is therefore required even though exact root cause is unresolved.

Consequences:
The harmless smoke suite proves normal exit, timeout plus descendant cleanup, owner-exit cleanup, concurrent-name refusal, suspended creation, assignment before resume, and exact PID absence. The legacy bundle remains evidence, b10344 is a comparison lane rather than the pinned parity authority, and the pinned source build is the preferred oracle. No real-model safety or numerical claim follows from the smoke test; the first bounded model run remains a separate gate.

---
AI-edited: 2026-08-10T15:34:55-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=bounded-llamacpp | change=made immutable bundles and suspended Job Object containment the Windows oracle contract
