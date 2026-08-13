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

Status: Superseded by D-0029

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

Status: Superseded by D-0029

Decision:
Use `C:\DevStuff\candle-mods` only as the linked Windows edit worktree for Codex-authored changes. Keep the authoritative checkout and all verification work in Linux-home WSL2 worktrees. Builds and baseline checks never run from `/mnt/c` or `/mnt/d`.

Why:
The edit worktree is detached and linked to the authoritative WSL repository; the project execution plan requires Linux-home filesystem behavior.

Consequences:
Only Linux-home verification worktree evidence may be recorded as green. Cargo checks never run in the Windows edit worktree.

## D-0008: Local Verification Lockfile

Status: Accepted

Decision:
Keep `Cargo.lock` ignored and local to each verification lane. Require it for every `--locked` check and record its SHA-256 with the retained proof.

Why:
Upstream Candle 0.11 intentionally ignores `Cargo.lock`, but the sprint requires locked local verification. Committing a workspace lockfile would change upstream repository policy before implementation evidence justifies it.

Consequences:
A fresh Windows or Linux verification lane must resolve the lockfile deliberately, hydrate only the required local dependencies, and then run the phase verifier offline. Different lock hashes or target caches are different proof environments and must not be compared as identical baselines.

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

Process each image and encode its vision features once, then execute deterministic greedy prefill and cached decode twice from a cleared text cache. Require complete trace equality across both runs, including generated IDs/tokens, full F32-logit SHA-256 values, stable top-k with lower-token-ID tie breaking, stop reason, and decoded forms. Reject empty/non-finite/oversized logit surfaces. Resolve EOS by `CLI > native model config or GGUF metadata > tokenizer candidate`, and record the source. Emit a versioned one-line JSON record without wall-clock timings.

Why:
Decoded text alone cannot distinguish artifact drift, prompt mismatch, cache leakage, nondeterminism, or a plausible but numerically unrelated answer. Canonical paths plus file sizes and hashes make comparisons reproducible, while exact replay makes cache reset an observed invariant rather than a claim. Timings are excluded because they would make otherwise identical evidence records differ.

Consequences:
Committed tests cover native and real split-MMProj hybrid image prefill, cached decode, EOS provenance, all consumed-file hashes, exact direct/split/override source lists, and exact reset replay. The local fine-tuned text GGUF plus byte-identical official MMProj produces the same eight-token caption in Candle and llama.cpp under aligned deterministic settings. That result is same-artifact runtime evidence only: official-base text parity and llama.cpp component/logit equality remain separate gates.

## D-0049: Timing and Benchmark Evidence Stay Outside Deterministic Inference

Status: Accepted

Decision:
Expose stage timing only through an opt-in `--timings` stderr diagnostic. Keep
the versioned JSON report free of wall-clock values so repeated evidence remains
byte-comparable. The diagnostic may report model load, image load, processor,
prompt, vision, first generation, cache-reset replay, and total inference, but
it must not alter execution order, cache-reset behavior, sampling, or report
schema. Expose the isolated generation lane separately through
`--benchmark-generation`: ten warm-ups, thirty measured direct
prefill/cached-decode iterations, device synchronization around the measured
region, exact generated-ID replay, and a versioned stderr-only benchmark
record. The benchmark and diagnostic/trace modes are mutually exclusive.

Why:
P4.4 needed measured performance data, while the existing JSON contract is a
deterministic parity artifact. Mixing timing into that artifact would create
false diffs and encourage consumers to treat noisy wall-clock values as model
correctness evidence.

Consequences:
Performance claims require bounded owner records, explicit median and median
absolute deviation, unchanged inference/parity output, and at most 5% relative
MAD. A source optimization is retained only after a quiet-host before/after
series proves at least a 10% improvement. The stable baseline closes PERF-1
without retaining a candidate because the earlier candidate did not meet that
contract.

## D-0028: Immutable llama.cpp Bundles and Suspended Job Containment

Status: Accepted

Decision:
Keep the incident b9981 install, the user-supplied current b10344 tools, and the exact pinned source build as separate immutable bundles. Identify each by source/release revision, build configuration, executable/DLL inventory, size, and SHA-256. Do not repair suspected mixing by deleting or updating in place; first prove the bundle's internal provenance and import resolution.

Run every expensive Windows llama.cpp build or oracle process through `scripts/lfm2-vl/run-bounded-oracle.ps1`. Serialize matching process names, create the child suspended, assign it before resume to a kill-on-close Job Object with per-process and per-job memory limits, enforce timeout, and require exact PID absence. Default CUDA graphs off for Windows CUDA/MTMD work while related upstream leak reports remain plausible; any override is explicit evidence. Admit a model only after a no-model identity probe and before/after physical, commit, and GPU memory census.

Why:
The completed b9981 parity run left approximately 131.5 GB of private allocation and could not be terminated through normal or elevated-looking paths. PID disappearance did not restore usable host performance; restart was required. Bundle audit found no DLL-mixing or Defender evidence, while WER recorded `RADAR_PRE_LEAK_64` and official llama.cpp reports describe related CUDA/MTMD memory growth. Containment is therefore required even though exact root cause is unresolved.

Consequences:
The harmless smoke suite proves normal exit, timeout plus descendant cleanup, owner-exit cleanup, concurrent-name refusal, suspended creation, assignment before resume, and exact PID absence. The legacy bundle remains evidence, b10344 is a comparison lane rather than the pinned parity authority, and the pinned source build is the preferred oracle. No real-model safety or numerical claim follows from the smoke test; the first bounded model run remains a separate gate.

## D-0029: Native Windows Is the Primary Product and Verification Lane

Status: Accepted

Decision:
Target native Windows first for user-facing execution, CPU-F32 production parity, and later CUDA proof, using the MSVC Rust toolchain and PowerShell process containment. Keep Candle's implementation OS-agnostic and replay relevant gates in WSL2/Linux as a secondary portability check when practical. Treat this folder's WSL-owned `.git` pointer and detached worktree state as local Git topology only; they do not make WSL a runtime or product requirement.

Why:
The fork will be used on Windows, and the independent llama.cpp oracle plus host-memory controls are Windows-native. A WSL-only green result cannot expose Windows toolchain, DLL, path, process-lifecycle, or accelerator integration defects. Retaining a Linux replay still protects portability without making it the release authority.

Consequences:
The next production checkpoint runs CPU F32 on native Windows first and may then be replayed in WSL2. Windows Cargo checks must be recorded independently; missing locked cache entries are a truthful blocked lane, not permission for an implicit network fetch. CUDA follows the same Windows-first order after CPU parity. D-0002 and D-0007 remain historical descriptions of the bootstrap topology and are superseded by this decision; D-0008 now applies per platform lane.

## D-0030: Pin the Python Oracle and Contain Every Inference Process

Status: Accepted

Decision:
Keep Python packages out of the Candle runtime and use them only in the separately managed reference/oracle lane. Require the exact Python, CPU Torch/TorchVision, safetensors, pinned Transformers commit, Hub, tokenizer, regex, and Pillow versions before importing an official model or processor; keep pytest test-only. Run every production Python or native inference command through `scripts/lfm2-vl/run-bounded-oracle.ps1` after building native binaries, never around `cargo run`.

Why:
The reference lane exists to reproduce the pinned official implementation and generate component tensors for parity; unpinned package drift can change preprocessing or model math while still producing plausible text. The previous Windows llama.cpp incident demonstrated that a long-lived model process can retain extreme private memory and survive ordinary cleanup paths, so inference must have owner-scoped memory, timeout, and descendant cleanup before it is allowed to run.

Consequences:
Config-only inspection and source-only Rust verification remain usable without the Python ML stack. Official tiny/production exports fail closed or skip when the oracle environment is wrong, and trace comparison remains a no-weights operation. The first real 450M trace still requires an owner-managed pinned CPU environment, a healthy resource preflight, exact artifact hashes, and post-run PID and memory evidence; no package installation or model inference is implicit.

## D-0031: Make Physical-Memory Preflight Independent of CIM Permissions

Status: Accepted

Decision:
Have the bounded Windows oracle wrapper try `Win32_ComputerSystem` first and
fall back to the kernel `GlobalMemoryStatusEx` API when CIM access is denied.
Record the selected probe as `physical_memory_source` alongside total physical
bytes, and fail closed if neither source returns a positive value.

Why:
Managed Windows sessions can deny CIM queries even when the process-containment
APIs and Job Object limits are available. A safe memory ceiling must not depend
on an unrelated administrative/provider permission, and it must never be
guessed from a partial or unavailable value.

Consequences:
The same 75%-of-physical-RAM ceiling and per-process/per-job Job Object limits
remain in force. The harmless wrapper suite covers the evidence field under
both Windows PowerShell 5.1 and PowerShell 7, and a restricted-host probe can
prove the native fallback without launching a model. This changes only host
preflight and evidence schema; it does not relax PID, timeout, child-tree, or
model-run containment.

## D-0032: Separate Host Census From Inference Containment

Status: Accepted

Decision:
Use `scripts/lfm2-vl/preflight.ps1` as a read-only admission report before any
large Python, native Candle, or llama.cpp process. Keep it independent from the
bounded launcher: the census records Git/worktree identity, physical and commit
memory, disk, optional NVIDIA state, and matching process/PID ancestry, while
`run-bounded-oracle.ps1` remains responsible for creating, limiting, and
cleaning up a child process tree. Omit command lines and never inspect secrets.

Why:
A bounded child can still be unsafe to start when another model or unexplained
worker already owns host memory. Conversely, a host census cannot prove that a
future child will be contained. Keeping the responsibilities separate makes
the admission evidence reusable for native Windows and WSL replay without
weakening Job Object, timeout, or exact-PID cleanup requirements.

Consequences:
`preflight.ps1` returns `blocked` when any recognized model process
(`llama`/MTMD/`lfm2-vl`), build process (Cargo/rustc/Ninja/CMake), Python
process, or required physical/committed-memory probe prevents quiet-host
admission. It returns `review` only when the model/build/Python sets are empty
and both memory measurements are complete; owner approval is still required.
Its smoke contract runs under
PowerShell 5.1 and 7, writes only an explicitly requested atomic report, and
reports linked-worktree Git failures as data rather than terminating on native
stderr. The general process evidence is capped, but the dedicated
model/build/Python
collections are complete before that cap. It is resource evidence, not model
or numerical parity evidence.

## D-0033: Make Pinned Artifact Identity a Separate, Hash-Only Admission Record

Status: Accepted

Decision:
Use the stdlib-only `tools/lfm2_vl/reference/inspect_artifact.py` command to
record the locked repository, revision, required filename, byte size, purpose,
and SHA-256 for a local regular-file model snapshot before NR-5B. Require an
explicit production opt-in, reject repository-local paths, symlinks, ambiguous
single/indexed safetensors layouts, changing files, and oversized inputs, and
write the small manifest outside the repository. Keep trace-bundle validation
equally strict: manifest, metadata, and tensor entries must be direct regular
files in the bundle root and JSON values must be objects.

Why:
The native runner already reports hashes after loading, but the oracle's
`from_pretrained` call otherwise hides which local cache files supplied the
model. A separate pre-run identity record lets both lanes be pointed at the
same immutable regular files without loading tensors into Python or embedding
production payloads in evidence. Path escape or symlink resolution would make
the recorded name different from the bytes actually consumed.

Consequences:
P1-B now has a bounded, testable tool and a clear admission contract, while the
official 450M manifest remains unclaimed until an owner-approved snapshot and
resource preflight are available. Tiny disposable snapshots prove the schema
and atomic write; no network, Torch, model load, or production bytes were used
by the implementation test.

## D-0034: Preserve Disk Evidence Across Windows PowerShell Versions

Status: Accepted

Decision:
Have `preflight.ps1` use `Get-PSDrive` only when it returns nonzero counters;
otherwise read the repository drive through `System.IO.DriveInfo`. Record the
source and require a positive free-space value in the cross-version smoke test.

Why:
The Windows resource contract supports both PowerShell 7 and the inbox
PowerShell 5.1. The latter can expose a valid-looking `PSDriveInfo` with zero
`Free`/`Used` values, which is not a trustworthy disk measurement and would
make later admission decisions ambiguous.

Consequences:
The report remains read-only and schema-compatible apart from the additive
`disk.source` field. Both shell lanes now retain real disk evidence without
weakening the fail-closed physical/commit-memory or PID rules.

## D-0035: Pin the Oracle Interpreter Per Supported Platform

Status: Accepted

Decision:
Require Python 3.10.11 for the native Windows oracle and retain Python 3.10.12
for the resolved Linux x86_64 oracle lock. Keep every shared package version
and the exact Transformers VCS commit identical. Record the selected platform,
full interpreter patch, installed versions, and mismatches before importing an
official model. Treat `requirements-reference.in` as the shared direct intent,
the existing `requirements-reference.txt` as Linux-only resolution evidence,
and require a separately proven Windows resolution before checking one in.

Why:
Python 3.10.12 is a source-only security release and Python.org provides no
Windows installer for it; Python 3.10.11 was the last Python 3.10 release with
an official Windows binary. A single exact 3.10.12 guard therefore contradicted
the Windows-first product and verification policy. Selecting the last official
Windows binary preserves an exact, supported interpreter identity without
weakening Torch, Transformers, processor, artifact, or tensor parity.

Consequences:
The committed Linux fixtures retain their original Python 3.10.12 provenance.
Native Windows production traces must use Python 3.10.11 and will record that
identity explicitly. Python 3.10.10, moving 3.10 ranges, unofficial 3.10.12
Windows builds, and global-package installation remain inadmissible. The first
Windows environment now has a checked-in resolved lock, a green import-light
pin verifier, and a 43/43 focused suite. Environment conformance alone does not
authorize a production run; the completed NR-5B gate separately records fresh
resource admission, bounded execution, exact cleanup, and numerical evidence.

## D-0036: Separate Model-Tool Name Concurrency From Exact Executable Concurrency

Status: Accepted

Decision:
Keep name-wide concurrency as the bounded owner's default for uniquely named
model tools such as `llama-mtmd-cli`. Add an explicit exact-executable mode for
generic hosts such as Python, comparing canonical executable paths and failing
closed when a same-name process path cannot be inspected. Retain optional
combined stdout/stderr in a caller-selected external log and record its byte
count and SHA-256 in wrapper evidence.

Why:
An unrelated bundled Python worker made name-wide refusal correct but too broad
for the pinned oracle interpreter. The first failed production trace also had
no actionable child error because output was not retained. Weakening all
concurrency to path matching would make unique model tools easier to overlap;
discarding logs would make safe bounded failure needlessly opaque.

Consequences:
Callers must select `Executable` explicitly for the pinned Python or native
Candle executable and retain `Name` for llama.cpp unless a reviewed reason says
otherwise. The mutex remains keyed by exact executable, path-inspection denial
is a refusal, and the evidence schema gains additive concurrency/log fields.
PowerShell 5.1 and 7 smoke tests cover same-name refusal, same-executable
refusal, unrelated same-name allowance, combined output, hashes, and cleanup.

## D-0037: Make Trace Evidence Semantics Cross-Runtime and Exit-Code Enforced

Status: Accepted

Decision:
Use framework-neutral canonical dtype labels in both trace manifests, serialize
the native loader's exact consumed-file evidence, and define
`input.projector_crop_ranges` as ranges over valid pre-projector vision patches.
The comparison CLI returns 0 only when its report has `passed=true`, 1 for a
valid comparison with failed tensors, and 2 for invalid input or invocation.

Why:
The first official comparison reached production bytes but failed validation
because the native manifest used Candle abbreviations, omitted already-computed
model input evidence, and reported post-projector token ranges under a
pre-projector range name. After those fixes, the report still exposed one
failed exact range while the CLI returned 0. A process exit alone therefore
could have produced a false green gate.

Consequences:
Native bundles are independently validatable by the Python safetensors reader,
artifact identity is checked before tensor math, exact-input names have one
stage meaning, and automation cannot treat `passed=false` as success. The tiny
native trace test asserts canonical dtypes, model-input hashes, and an 8-patch
rather than 2-projected-token range; the Python suite asserts failed reports
are written but return exit 1.

## D-0038: Inspect Local Full GGUFs Through a Payload-Free Bounded View

Status: Accepted

Decision:
Permit `inspect_gguf_header.py --full-file` only for an already-local complete
regular GGUF. Memory-map the file, cap parser access at 4 MiB, and hash exactly
through the computed aligned tensor-data offset. Report physical file bytes and
whether they equal the declared tensor extent separately. Full-file SHA-256 may
read every byte for artifact identity, but it must not decode or construct any
tensor. Retained full reports use UTF-8 files and quiet stdout.

Why:
P2's official Q4_0 text artifact was already present as a regular Hugging Face
cache blob, while the existing tool accepted only separately copied/downloaded
header prefixes. Creating an ad hoc prefix would weaken path identity and add a
temporary-copy step. Printing the complete tokenizer inventory also exposed a
CP1252 failure and produced an unnecessarily large wrapper log.

Consequences:
The official text GGUF is locked to the same immutable LiquidAI revision as the
MMProj, with an exact payload-free 2,388,128-byte header and separately verified
219,311,264-byte full-file hash. The inspector preserves prefix mode, escapes
console JSON for Windows code pages, and offers `--output ... --quiet` for full
UTF-8 evidence. This closes artifact discovery only; model loading and runtime
parity still require the bounded P2 execution gates.

## D-0039: Compare Only Stable Cross-Runtime Fields and Bound Context-Cap Differences

Status: Accepted

Decision:
For the official-base GGUF gate, require exact artifact, image, prompt-semantics,
deterministic decoded-output, and cleanup agreement. Record generated IDs,
preprocessing dimensions, projected-token counts, logits, intermediate tensors,
and reset replay as unavailable when the pinned llama.cpp CLI does not expose
them. Permit a smaller llama.cpp KV context ceiling only when the complete
observed sequence is strictly below both ceilings and the difference is retained
as a bounded operational fact.

Why:
The experimental `llama-mtmd-cli` provides a decoded stream and limited MTMD
progress logging, not a stable machine-readable tensor or token trace. Candle
reports the GGUF capacity of 128,000 tokens, while allocating that full KV range
in a cross-runtime smoke would add avoidable memory risk. Treating missing fields
as matches or claiming identical configured context would overstate parity;
requiring the same ceiling despite an 83-position sequence would add risk without
changing the exercised positions.

Consequences:
P2 is green at the fields both runtimes actually expose: identical official
artifacts, identical source image, equivalent official template framing, exact
three-token decoded output, bounded execution, and exact process cleanup. The
4,096-token llama.cpp ceiling is explicitly different from Candle's 128,000-token
capacity but contains all 83 used positions. Component-tensor parity remains
owned by the pinned Transformers oracle, and any future llama.cpp trace support
must be added as new evidence rather than retroactively inferred.

## D-0040: Admit the 1.6B Gate With Stage-Specific Ceilings Before Acquisition

Status: Accepted

Decision:
Forecast 1.6B memory from the exact official safetensors byte ratio and measured
450M Job peaks, then apply a 1.35 safety factor. Use separate first-attempt Job
ceilings of 16 GiB for Python dry load, 24 GiB for Python trace, and 12 GiB for
native trace. Require at least 32 GiB available physical and commit headroom for
the largest stage, zero model/build competitors, and a fresh preflight before
each process. Do not raise a ceiling automatically. Treat checkpoint acquisition
as a separate external action and locally hash every regular file before load.

Why:
The 1.6B model file is 3.558093732 times the 450M model file, while its selected
trace grows only 2.206735 times because processor and vocabulary shapes stay
fixed. One shared high ceiling would give the smaller native stage unnecessary
room and weaken OOM containment. Downloading first and planning later would also
commit disk/network resources before proving that sequential CPU-F32 execution
fits the host.

Consequences:
The no-model forecast admits a 3,198,084,631-byte regular snapshot and projects
about 182.53 MB per trace. The conservative cache/copy/two-trace workspace is
7.30 GiB, so acquisition requires 12 GiB free. A limit termination is a safe
failed measurement to investigate, not permission to increase memory. No
inference starts until the expected 3,193,334,216-byte model file is locally
rehashed to `7fc7458e4382fc6e558cfdda45857fbf9ab5b40a8bf199c9cd073003b14ac26d`.

## D-0041: Separate Resumable Acquisition From Clean Snapshot and Model Load

Status: Accepted

Decision:
Acquire only the exact direct files and immutable model revision recorded in
`reference-lock.json`. Keep resumable provider state in an external
caller-owned Hugging Face cache, stream verified bytes into a separate clean
staging directory, and publish the regular-file snapshot plus manifest through
atomic no-clobber operations. Use Windows no-replace rename or Linux
`renameat2(RENAME_NOREPLACE)` for the snapshot and a flushed sibling temporary
file plus hard-link publication for the manifest. Planning is stdlib-only;
downloading checks only the exact Hub
package, sets `HF_HUB_DISABLE_XET=1` before Hub import, refuses a process that
already imported Hub with Xet enabled, and always uses `token=False`. Model
loading remains a later, independently bounded action guarded by the complete
CPU oracle environment.

Why:
Transformers' cache can expose links and provider bookkeeping that are
unsuitable as the shared artifact identity. A direct local copy without a
pinned acquisition owner can mix revisions or leave a partial snapshot.
Conversely, requiring Torch, TorchVision, Transformers, Pillow, and every
oracle dependency merely to transfer bytes adds install weight without
improving download integrity. The pinned Hub installation also includes
`hf-xet`, whose own installed source documents parallel chunk downloads and
automatic activation when available; an outer serial file loop alone would not
bound transfer concurrency.

Consequences:
Interrupted transfers can resume without making a partial snapshot admissible;
every published file has a locked size and Git-blob/LFS identity; failed
identity or manifest checks roll publication back. A destination that appears
after planning is preserved rather than replaced, duplicate verifier paths are
rejected, and stale snapshot or manifest staging blocks retry. Every
Hub-returned source
must resolve inside the named caller-owned cache, and provider failure causes
are suppressed after retaining only filename and exception class. The
multi-gigabyte network
and disk action still requires separate owner approval. The retained
`transfer_policy` states `serial-files-resumable-http-xet-disabled`. Passing
acquisition does not claim that the model was imported, allocated, or
numerically correct. The production function exposes no alternate downloader
or artifact-builder parameter; test doubles are installed only by patching
private module boundaries during offline tests. The authorized transfer runs
through the existing Windows Job Object owner with a 2 GiB ceiling, two-hour
timeout, executable-scoped concurrency, retained log/evidence, and exact PID
cleanup; a terminated attempt may resume its cache but cannot make unmatched
staging/output/manifest state admissible.
Evidence schema 2 separately records network policy and observed use: planning
is disabled/false, while execution is permitted-cache-aware/unknown because an
immutable-revision cache pointer may satisfy the call without network traffic.

## D-0042: Make Durable Evidence Publication Exclusive by Default

Status: Accepted

Decision:
Publish standalone reports, JSON summaries, split-MMProj files, owner evidence,
and native trace directories without replacing an existing or racing target.
Python byte writers use a flushed sibling temporary plus a no-clobber hard
link; PowerShell uses `System.IO.File.Move` when force is absent; native trace
directories use Windows rename refusal or Linux
`renameat2(RENAME_NOREPLACE)`. Platforms without a proven exclusive directory
primitive fail closed. Replacement is available only through an explicit
`--overwrite`, `overwrite=True`, or force option on outputs whose contract
allows it; acquisition snapshots never gain an implicit replacement path.

Why:
The acquisition race review exposed a repository-wide assumption that an
absence check followed by an atomic replacement was exclusive. The same
check-then-publish shape existed in the trace comparator, config/GGUF reports,
split-MMProj exporter, native trace directory, and PowerShell evidence writers.
Leaving those paths inconsistent would preserve the original owner-data risk
outside the first code path that happened to reveal it.

Consequences:
Reusing an output path now returns a controlled failure unless the operator
explicitly authorizes replacement. Shared publication helpers remove repeated
temporary-file code and race behavior is covered at the helper, CLI, exporter,
PowerShell, and native Windows boundaries. Linux source uses the same
no-replace primitive as guarded acquisition; its exact Rust regression remains
a secondary TODO until a local WSL Rust toolchain is available.

## D-0043: Use One Owner-Reviewed Direct-Main Publication Line

Status: Accepted

Decision:
Use `main` in `Shoozes/candle` as the single integration and publication
branch. Review and verify changes locally, preserve any fetched `origin/main`
history through an explicit non-force integration, and push the already-clean
named branch without a pull request. Retain `feat/lfm2-vl-mmproj` only as a
historical checkpoint line.

Why:
The owner reviews code in the local Coding app and explicitly chose direct-main
publication. Maintaining a second active feature branch and PR ceremony would
create avoidable branch-state ambiguity. At the same time, GitHub main had
advanced nine upstream Candle commits beyond the pinned 0.11 implementation
base, so replacing it with the feature history would have discarded valid fork
updates.

Consequences:
The Windows folder remains a WSL-owned linked worktree but is now attached to
local `main`; WSL Git owns all repository operations. Before every push, fetch
and inspect `origin/main`, require a reviewed fast-forward ancestry result after
any merge, rerun local verification, stage only manifest-authorized paths, and
keep the worktree clean. The ignored `.tools/gitpush.ps1` reads the operator
token internally and may fetch, verify, push, and confirm the remote head. Once
remote `main` exactly equals reviewed local `HEAD`, its guarded tag mode may
publish one annotated `lfm2-vl-mvp-X.Y.Z` tag that peels to the same commit and
whose remote name is absent or already identical. It must not stage, commit,
merge, rebase, create a repository, delete refs, expose the token, force-push,
or publish unrelated refs. Hosted CI and PR status remain outside the
verification contract.

Evidence:
`origin/main` at `6f74e7c390c717f8fd34f23ce02aceb058173370`
diverged from historical mod checkpoint
`c9b60f0b906fa8fe70423295e2e1164648a8fa53` at Candle 0.11.0. Its 29 changed
paths did not overlap the mod's nine fork-origin files. Merge checkpoint
`2b1d9e80de06b251b2fe5f25e51c17d56db86591` preserved both histories without
conflict or force, and post-merge local Rust, Python, PowerShell, provenance,
and documentation gates remained authoritative.

## D-0044: Localize Stable SigLIP2 LayerNorm and Use Phase Contracts for CPU F32

Status: Accepted

Decision:
Route SigLIP2 encoder pre-norms through Candle's stable two-pass F32
`layer_norm_slow` implementation while leaving the global LayerNorm kernel and
post-layernorm contract unchanged. In the production trace comparator, keep
exact integer/input checks; require `<=2e-5` for resized positions, `<=1e-3`
for full prefill logits, and accept vision/projector/hidden-state stages when
their existing allclose passes or their cosine is at least `0.99999`; keep
the structural pixel-unshuffle stage on allclose.

Why:
The pinned 1.6B vision activations reach large offsets in later layers. The
fast one-pass `E[x²] - E[x]²` variance path produced cancellation: the first
native trace had six failures, while the localized stable F32 pre-norm reduced
that to three. A higher-precision F64 helper increased drift and was rejected.
The remaining 1.6B differences are small-magnitude CPU reduction differences:
vision layer 26 has max abs `0.022125244` but cosine above the target, and
prefill logits have max abs `0.0009407997131347656`, inside the written CPU-F32
bound. A single global elementwise allclose rule would reject those valid
phase-contract results while obscuring the stronger exact and directional
checks.

Consequences:
The 1.6B native comparison is green at 51/51 tensors with exact reset and
decoded IDs. The model-specific helper is covered by a large-offset regression
test, the comparator policy by a synthetic phase-contract regression, and the
full pinned reference suite remains local-only. This decision does not claim
CUDA, lower-bit production MMProj, or llama.cpp component-tensor parity.

## D-0045: Make CPU Text a Public Device-Policy Route

Status: Accepted

Decision:
Expose `--text-cpu` as the component-specific counterpart to the existing
`--vision-cpu` flag. Resolve the four placements as accelerator/accelerator by
default, accelerator/CPU with `--vision-cpu`, CPU/accelerator with
`--text-cpu`, and CPU/CPU with `--cpu`. Keep `--cpu` authoritative when it is
combined with either component flag; do not introduce a second device-policy
abstraction or change the existing loader/model APIs.

Why:
The native and hybrid model paths already accept separate vision and text
devices, and the CUDA-gated fixture already covered vision CUDA with text CPU,
but the public example could not select that supported boundary. A single
boolean route preserves existing command behavior while making the intended
placement explicit and testable.

Consequences:
The example reports both resolved devices using its existing report schema.
The trace lane remains explicitly CPU-only and rejects `--text-cpu` without
`--cpu`; CUDA execution remains a separate P4.2 runtime proof rather than an
implicit consequence of parsing the flag.

## D-0046: Pass the MSVC Conforming Preprocessor Through CUDA Kernel Builds

Status: Accepted

Decision:
When the CUDA build target is MSVC, pass `-Xcompiler /Zc:preprocessor` through
`candle-kernels/build.rs` for both PTX and static-library kernel compilation.
Keep the switch target-specific and leave non-MSVC CUDA flags unchanged.

Why:
CUDA 13.3's CCCL headers fail closed when nvcc invokes MSVC's traditional
preprocessor. The first bounded P4.2 build stopped at that header before any
test or runtime code executed. The switch is the documented compiler remedy,
and the same option is accepted by the existing MSVC toolchain used by this
repository.

Consequences:
The tiny native CUDA/distinct-device proof builds and passes on Windows while
preserving the CPU-only path and avoiding a global warning suppression. The
build remains explicitly toolchain-dependent; this decision does not claim
official production CUDA parity or lower-bit CUDA support.

## D-0047: Keep CPU Low-Precision Dtypes Explicitly Unsupported and Fail Early

Status: Accepted

Decision:
Reject explicit BF16 or F16 when either LFM2-VL component resolves to CPU,
before loading checkpoint weights. Keep CPU F32 as the supported fallback and
retain BF16/F16 for CUDA components.

Why:
Candle's CPU matmul path does not support these low-precision dtypes for this
model. Allowing the request through model loading produces a deep operator
error after allocation rather than an actionable placement message. Requested
flags are also insufficient because an accelerator helper can resolve to CPU;
the public placement contract must follow the actual devices.

Consequences:
The example reports a concise error for BF16/F16 on any resolved CPU text or
vision component; no production model load occurs. All-CUDA BF16 and F16 are
covered by the official 450M matrix. This is a capability boundary, not a
silent dtype conversion.

## D-0048: Materialize CUDA Matmul Inputs at the Narrow Dense Boundary

Status: Accepted

Decision:
Materialize dense LFM2-VL linear inputs as contiguous tensors immediately
before the CUDA dense matmul. Keep quantized paths on their existing explicit
contiguous boundary and do not add a global tensor-layout rewrite.

Why:
The projector can produce a broadcasted/non-contiguous activation layout that
is valid on CPU but rejected by the CUDA matmul implementation. The narrow
boundary fixes the proven production path while limiting allocation and
behavior changes elsewhere.

Consequences:
The all-CUDA and CPU-text/CUDA-vision 450M F32 routes pass; a focused CUDA
regression protects the non-contiguous input case. Future optimization must
measure this materialization before attempting to remove or fuse it.

## D-0050: Canonicalize Hashed Fixture Bytes at the Git Boundary

Status: Accepted.

Decision:
Use root `.gitattributes` to require LF checkout bytes for JSON and Markdown in
all three committed LFM2-VL fixture directories and to mark their safetensors
payloads `-text`. Keep manifests and runtime loaders byte-exact: do not
normalize, rewrite, or hash a transformed representation after checkout.

Why:
The fixture hashes identify exact consumed files. A native Windows Git 2.54
checkout with `core.autocrlf=true` changed `processor_config.json` from 524 LF
bytes and SHA-256
`97b79ebfc8eae3a5bcbeb8f1494c1decdbade5d20d3204739143d17b460906f2`
to 553 CRLF bytes and SHA-256
`09150e818ebe443d2df9009b78c46ef5aaa4aed17ebc4b20cf55eefb8f01e53f`.
Normalizing inside the loader would conceal a mutated artifact and weaken the
existing provenance contract. The other deterministic fixture manifests also
hash text metadata, so a split-bundle-only rule would leave the same defect in
two neighboring fixture families.

Consequences:
The mod-manifest verifier dynamically inventories the three fixture roots,
requires every committed JSON/Markdown file to resolve `text=set eol=lf` with
no carriage-return byte, and requires each safetensors file to resolve
`text=unset`. A fresh native Windows clone with `core.autocrlf=true` retained
all 10 text fixtures and all three split-bundle hashes exactly; the two exact
split-MMProj tests, full locked/offline workspace tests, and strict workspace
Clippy passed. Any new hashed fixture directory or text extension must extend
the attributes and verifier together. This decision adds no production
dependency and changes no runtime input format.

## D-0051: Make `candle-vlm` the Hybrid LFM2-VL Assembly Owner

Status: Accepted.

Decision:
Promote explicit local hybrid assembly from the LFM2-VL example into
`candle_vlm::lfm2_vl::load_lfm2_vl_hybrid`. The public API accepts the text
GGUF, tokenizer, optional processor configuration, split or GGUF MMProj source,
execution policy, dtype, and component devices. It returns the paired model,
processor, prompt contract, and exact local paths consumed. The example remains
a thin argument, device-policy, inference, and report adapter.

Why:
The model, processor, and prompt types were already reusable, but applications
would otherwise need to copy the only complete hybrid construction sequence
from `candle-examples`. Copying would duplicate pairing, image-token,
processor, and Q8 policy checks precisely where EdgeSymbio needs one stable
framework boundary. A path-explicit constructor preserves Candle's local-first
behavior without importing application policy.

Consequences:
The library performs no discovery, network access, download, fallback, hash
admission, retained-handle ownership, resource leasing, or proof publication.
Applications must bind the returned consumed-file list to their own admission
and evidence contracts. Deterministic tests construct split dense, direct dense
GGUF, and direct native-Q8 GGUF runtimes through the public API; invalid Q8
policy fails before path access. The change adds only a test-time SHA-256
dependency for fixture identity and leaves native and text-only paths intact.

## D-0052: Track Independent Fork Overlays and Verify Their Union

Status: Accepted.

Decision:
Keep LFM2-VL/MMProj and SnapFlash-derived diffusion work in separate overlay
manifests, with `docs/FORK_OVERLAYS.md` as the shared-path registry and
`scripts/verify-fork-overlays.sh` as the repository-wide union-completeness
gate. Permit duplicate manifest ownership only for an explicitly registered
shared path. Keep each overlay-specific verifier independently runnable.

Why:
The fork's first LFM2-VL snapshot is immutable, while future generic diffusion
primitives have a different donor, proof, and consumer sequence. One growing
manifest would let unfinished application-derived work appear inside an
unrelated model release and would make upstream reconciliation ambiguous.

Consequences:
Every baseline-to-current path must belong to at least one overlay, prohibited
local/runtime paths fail closed, and overlapping paths require declared shared
ownership plus both focused gates. Candle may absorb generic tensor, loader,
preprocessing, scheduler, or mutation primitives, but it must not depend on or
expose EdgeSymbio/SnapFlash request schemas, application names, filesystem
policy, queues, resource brokers, or proof records. A future composite tag is
eligible only after both consumers pin the same Candle revision and pass their
local gates; `lfm2-vl-mvp-0.1.0` remains unchanged.

## D-0053: Keep SDXL LoRA Tensor Semantics in Candle and Name Policy in Consumers

Status: Accepted.

Decision:
Expose validated SDXL LoRA pair parsing and a revision-bound
`VarMapSwapTransaction` from Candle. The transaction always owns three ordered
components (UNet, text encoder 1, text encoder 2), retains independent base
copies lazily, prepares and revalidates every target before the first write,
computes replacement adapters from base, restores exact base values, and rolls
back in reverse order after a later write failure. Require consumers to supply
`SdxlLoraTargetResolver`; do not put Kohya conversion, filenames, paths,
catalogs, license policy, or report schemas in Candle.

Require the consumer to hold its exclusive model execution/mutation lease
through plan preparation and application. Candle provides all-or-rollback
mutation inside that boundary, but cannot serialize inference or direct VarMap
mutation performed elsewhere. Check revision advancement before the first
write so exhausted bookkeeping cannot leave an applied-but-uncommitted state.

Define cross-consumer target evidence as SHA-256 over a fixed domain prefix,
tensor rank/shape, and finite canonical F32 values in little-endian order.
Record base, effective-delta, and merged hashes per applied target. Reject
unknown tensor names, incomplete or duplicate pairs, invalid alpha/rank,
unsupported shapes/dtypes, duplicate or unmatched targets, non-finite
strength, all-zero effective adapters, and stale plans.

Why:
SnapFlash-Server already proves three-component behavior while EdgeSymbio
independently proves immutable-base replacement and rollback, but duplicating
their tensor math would let the two applications drift. Key naming and product
policy genuinely differ and would make a framework API application-specific.
An opaque prepared plan plus exact live-snapshot check closes the gap between
validation and mutation without inventing a generic VLM/diffusion runtime.

Consequences:
SnapFlash-Server is the first consumer migration and EdgeSymbio follows on the
same exact Candle revision. Both can compare stable target/delta evidence while
retaining their own mapping and proof JSON. The canonical hash intentionally
normalizes supported floating dtypes to F32 and includes shape; it is an LoRA
comparison contract, not a general byte-identity hash. Adapter parsing accepts
rank-2 factors and rank-2 or 1x1 rank-4 model targets only. The implementation
is fresh Candle-native code based on behavior review; no application code was
copied, and the unlicensed SnapFlash donor remains a behavior reference only.

## D-0054: Harden Candle's Existing ControlNet Residual Hook Without Importing an Application Model

Status: Accepted.

Decision:
Retain `UNet2DConditionModel::forward_with_additional_residuals` as Candle's
generic ControlNet integration boundary. Validate the complete residual
inventory and every tensor before the first residual is added: exact down
count, exact shape, exact dtype, and exact device for all down residuals and
the mid residual. Derive the expected inventory from the configured UNet
topology with checked arithmetic. For the standard SDXL configuration of
three down blocks and two layers per block, the exact count is nine.

Do not add SnapFlash request types, ControlNet catalogs, retained-file policy,
queues, image preprocessing, model loading, or a second ControlNet model to
Candle. Keep the existing method signature, `None` fast path, and residual
application order. Treat model-level numerical parity as a separate
fixture-gated consumer task rather than inferring it from successful shape
admission.

Why:
Candle already exposes the only framework hook SnapFlash needs. The previous
implementation could fail late or panic when a caller supplied the stale
13-residual inventory used by some non-SDXL descriptions, or when a residual
matched in count but not shape, dtype, or device. Strengthening this boundary
closes a reusable tensor-contract gap without copying application orchestration
or creating a speculative framework abstraction.

Consequences:
Short, long, wrong-shape, wrong-dtype, and wrong-device inventories fail before
the corresponding addition; valid inputs and the no-residual path preserve
existing behavior. SnapFlash must independently prove that its ControlNet
topology emits the exact nine tensors in Candle order and must retain ownership
of input preprocessing, weight admission, resource limits, mutation
serialization, rollback, and artifact publication. Round 7 is therefore a
generic contract checkpoint, not a claim of full ControlNet or inpainting
numerical parity.

## D-0055: Add SDXL `text_time` as an Opt-In UNet Conditioning Primitive

Status: Accepted.

Decision:
Keep `UNet2DConditionModelConfig` source-compatible for downstream struct
literals. Add a separate checked `SdxlTextTimeAdditionConfig`, an opt-in
`new_with_added_conditioning` constructor, and a single
`forward_with_conditioning` route that composes optional SDXL `text_time` and
ControlNet residual inputs. Preserve `new`, `forward`, and
`forward_with_additional_residuals` as wrappers with their existing
signatures. Expose `StableDiffusionConfig::build_unet_from_vb` as the opt-in
high-level builder so mmap and retained-buffer consumers can use the built-in
private UNet topology without duplicating it.

For SDXL `text_time`, flatten `[batch, time_id_count]`, apply the existing
sinusoidal `Timesteps` projection, reshape to one vector per batch item,
concatenate with `[batch, pooled_text_width]`, project through the official
`add_embedding.linear_{1,2}` namespace, and add the result to the scalar
timestep embedding. Derive pooled width with checked arithmetic from the
configured projection width; reject zero/odd/overflowing dimensions and exact
rank, batch, width/count, dtype, or device mismatches before graph execution.
Accept only F32 for this first boundary because the pinned reference computes
the sinusoidal time projection in F32; reject lower precision until a future
parity test proves the cast order.

Do not put prompt encoding, CLIP projection, default size/crop policy,
ControlNet topology, application schemas, retained files, or runtime resource
ownership in Candle. Those remain consumer work in INT-5C.

Why:
Official SDXL UNet and ControlNet checkpoints use the same pooled-text plus six
size/crop time-ID addition. Adding cross-attention to only the application
ControlNet would still leave the base UNet mathematically incomplete. A
separate opt-in configuration avoids breaking every existing public
`UNet2DConditionModelConfig` literal while one structured forward path avoids
an expanding matrix of specialized methods.

Consequences:
Configured models fail closed without the required input, unconfigured models
reject unexpected `text_time`, and legacy calls remain exact. The primitive
loads no weights unless explicitly selected. Combined text-time plus zero
ControlNet residuals preserve the conditioned result. INT-5B proves
deterministic CPU-F32 input influence and contract validation, but does not
claim official full-UNet numerical parity; INT-5C must supply correct CLIP2
pooled projection, time-ID policy, attention graph, and retained application
ownership before the INT-5D differential fixture.

---
AI-edited: 2026-08-13T04:25:00-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=int-5b | change=accepted opt-in source-compatible SDXL text-time conditioning
