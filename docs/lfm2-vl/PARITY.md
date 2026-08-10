# LFM2.5-VL Parity

## Current State

The deterministic reference fixture, LFM2 text compatibility path, SigLIP2 NaFlex tensor path, native projector/composite path, Rust-native raw-image/prompt path, Phase 5 quantized-text plus split-dense-MMProj path, Phase 6 direct-GGUF-MMProj dense compatibility path, and Phase 7 CPU-F32 native-Q8 vision/projector path are established. No production-checkpoint numerical parity, production-GGUF payload execution, llama.cpp runtime parity, generated-text parity, or executed native-Q8 CUDA parity is claimed.

## Required Gates

| Gate | Required evidence | Phase status |
| --- | --- | --- |
| Workspace baseline | Locked CPU-only Candle checks and diff check from Linux home | Phase 7 staged baseline green; log SHA-256 `ff46cc0b23a28050ffe856be2cb81ef7144667977587021f1d3cd221e00ed330` |
| Reference fixture | Deterministic pinned-Python export with component and multimodal tensors | Green; 87 tensors, byte-identical independent exports; manifest SHA-256 `c5461dadb0edfd920b20f308650c59676977110a1cc2f199e317dea7d75bdd7b` |
| LFM2 text configuration | 450M effective FFN width `4608`; 1.6B width `8192` | Green in config tests and header evidence |
| Dense text forwarding | Token-ID and embedding-driven prefill plus incremental decode agree | Green on the committed fixture; maximum hidden-state error `2.38418579e-7`, maximum logit error `2.98023224e-8` |
| Quantized text forwarding | Token-ID and embedding-driven paths agree and cache can be reset | API/equivalence gate green; production GGUF numerical parity pending |
| SigLIP2 | Component tensor comparisons against pinned fixtures | Phase 2 checkpoint complete; Phase 3 repeated-crop regression 8/8 green |
| Projector | Exact pixel-unshuffle and stage-level comparisons | Phase 3 focused gate green; 11/11 total |
| Processor | Exact resize, crop, patch, mask, shape, and token metadata | Green on all 12 pinned cases; 24/24 crate tests; worst pixel max abs `1.192092896e-7`; integer/crop metadata exact |
| Prompt expander | Exact expanded strings, tokenizer IDs, marker placement, and one span per crop | Green on all 5 pinned prompt cases, including multiple images and tiled thumbnail markers |
| Composite model | Image-span replacement and prefill/decode parity | Phase 3 focused gate green; 11/11 total |
| Phase 3 library gate | Locked/offline `candle-transformers` library tests | 37/37 passed |
| Phase 4 fixture reproduction | Fresh pinned-oracle export matches checked-in bytes | Green; manifest, metadata, and safetensors hashes match exactly |
| Split MMProj artifact | Exact versioned inventory, hashes, immutable provenance, and processor pairing | Green on the deterministic 43-tensor fixture; exporter/reference suite 19/19 |
| Hybrid GGUF text + dense MMProj | Real GGUF parse/load, split/native image-feature equivalence, prefill/decode/cache comparison | Green on the committed deterministic fixture; image features exact, hybrid text-logit max abs `4.457309842e-5` |
| Direct GGUF MMProj dense compatibility | Strict metadata/inventory/range load, patch inverse, dequantization, image-feature and hybrid execution comparison | Green on deterministic GGUF fixtures; dense image features exact, Q8_0 dequantized max abs `8.463021368e-5`, direct hybrid errors equal Phase 5 |
| Native Q8_0 GGUF MMProj | Eligible weights remain QTensor, all vision/projector linear roles execute through QMatMul, dense fallback remains intact, and hybrid prefill/decode/cache stay within documented drift | Green on CPU F32 deterministic fixtures; 14/14 two-layer linear roles quantized, feature cosine `0.999923348`, prefill max abs `1.650899649e-4`, cache reset exact |
| Official MMProj header contract | Pinned F16/Q8_0 metadata, names, physical shapes, dtype placement, and zero-payload evidence | Green; 32 metadata records, 201 tensors, tensor-data offset 12,736, no retained payload bytes |
| Distinct devices | Vision and text may differ; only projected image features cross at merge | Source-complete CUDA-vision/CPU-text test committed; local execution skipped because Linux `nvcc`/toolkit is absent |
| Production checkpoints and GGUF | Native versus production and GGUF numerical validation | Not run; no production weights or GGUF files downloaded |

## Phase 2 Focused Evidence

The manager's Linux-home WSL2 CPU F32 verifier passed all 7 SigLIP2 tests. The exact maximum absolute errors and cosine similarities were: patch projection `5.960464478e-8` / `0.999999940`; resized positions `2.980232239e-8` / `0.999999940`; embedding sum `1.192092896e-7` / `1.000000119`; encoder layer 0 `4.768371582e-7` / `0.999999881`; encoder layer 1 `1.192092896e-6` / `0.999999881`; returned post-LN `7.152557373e-7` / `1.000000119`; and the post-LN hook matched the returned result. Padding-key isolation was exact: max absolute error `0`, cosine `1`.

This focused proof covers packed patch projection, CPU F32 separable antialiased positional interpolation with per-shape caching, bidirectional key masking, encoder stages, post-LN, malformed-input rejection, and the required no-pooling-head boundary. The Phase 2 checkpoint is complete at commit `74e109aec5f9801cfead3eeb27fe3f93ac646b84`, annotated tag `lfm2-vl-phase-2-siglip2`. The Phase 2-era broader and baseline logs remain historical; the final Phase 3 gate and pre-checkpoint baseline are recorded below.

## Phase 3 Focused Evidence

Phase 3 implements dynamic top-level configuration, factor-N official pixel-unshuffle, optional projector LayerNorm, linear/GELU/linear projection, crop unpadding/ranges/order, strict one-span-per-image exact-length merge, multimodal prefill, ordinary cached decode, cache reset, and `EncodedImages`.

The focused Phase 3 gate passed 11/11. Retained log: `artifacts/verification/native-composite/focused-tests.log`; SHA-256 `7d727e1b8558f1f242ce940c8af36d44a3e292f4ffa023d1ff124ccf2cc13638`. Maximum absolute errors were projector stages `<=5.960464478e-8`, encoded/merged embeddings `<=6.519258022e-9`, prefill `<=4.470348358e-8`, and cached decode `<=2.980232239e-8`.

The SigLIP2 repeated-crop regression passed 8/8. Retained log: `artifacts/verification/native-composite/siglip2-regression.log`; SHA-256 `5684568b060c6338f3e5d8bc94361d37bc64ddf84584ad4a5e05915acc275f38`. It protects a real runtime defect found during multi-crop execution: batched attention received a non-contiguous transposed left-hand operand and failed with `MatMulUnexpectedStriding`; `split_heads` now materializes a contiguous tensor.

The locked/offline `candle-transformers` library gate passed 37/37. Retained log: `artifacts/verification/native-composite/candle-transformers-lib.log`; SHA-256 `0f36d6a8d54f77abfe9c5031075b7174cff83859315d0997f60a1a399f475497`.

The full locked/offline CPU baseline passed `2026-08-10T05:48:07Z`–`2026-08-10T05:48:10Z` against pre-Phase-3-checkpoint HEAD `74e109aec5f9801cfead3eeb27fe3f93ac646b84`. Retained log: `artifacts/verification/native-composite/baseline-final.log`; SHA-256 `47d984dd3afe7b92b6a72bcdb93e7d9da99bd8673e5c1067b8f1fac7ed2b8b45`. Cargo.lock SHA-256 remains `4e059ffe6035520ca6553303932173eba562f4985f82931a90129eea9849ce54`.

The Phase 3 checkpoint is complete at `37264b49cf74d0cf7697317eda0183f084db6ff8`, tagged `lfm2-vl-phase-3-native-composite`. These results do not claim production-checkpoint parity, CUDA, GGUF, raw-image preprocessing, tokenizer/chat-template behavior, or CLI support.

## Phase 4 Processor and Prompt Evidence

The new `candle-vlm` crate implements the processor configuration precedence contract, RGB conversion, checked smart resize, TorchVision-compatible antialiased byte resize, tile-grid selection, row-major crops, optional thumbnails, fused rescale/normalization, patchification, fixed padding, masks/shapes, image/crop metadata, tokenizer-resolved special markers, sentinel-position preservation, and exact per-crop span recording.

The locked/offline `candle-vlm` suite passed 24/24. All 12 required image cases compare packed pixel tensors, masks, spatial shapes, image grids/sizes, crop ranges/order/kinds, and projected token counts. Maximum normalized pixel error was `1.192092896e-7` with cosine similarity `1.0`; all integer and structural metadata was exact. A direct 7×5 to 8×4 regression compares all 96 RGB channel bytes against pinned TorchVision output.

All 5 prompt oracles match exact expanded strings, tokenizer IDs, placeholder counts, and one span per crop, including tiled row/column markers, thumbnail markers, multiple images, image-first positioning, and images across turns. Controlled-error tests cover missing tokens, sentinel/image mismatch, projected-token mismatch, context overflow, malformed row-major crop metadata, inconsistent empty batches, packed allocation overflow, and encoded image ranges that split crop ranges.

All 10 real-dimension oracles assert smart dimensions, large-image classification, selected grid, tile canvas, and whole/tile/thumbnail order. A fresh pinned Python export reproduced the checked-in fixture byte-for-byte: manifest `2fb787e378f5fd1ddfa147913aadccd07add9a1045b8bb0f693ca2c2f564959c`, metadata `aca7f4d5e5e4ef0e4872adeb227b56cf3960d87b353c40162af97660783f2327`, and tensors `a25932fc57f3e78f48a1a8f558216521c7ae3e8659fcf0a389cd0a4ebe0ab3f6`.

The pinned Python reference tests passed 9/9, the full `candle-transformers` library regression remained green at 37/37, and the final staged baseline passed formatting, all required package/example checks, and both diff gates. Phase 4 is checkpointed at `8d1bbe471404848730685c98e7dd56b13a457eb4`, tagged `lfm2-vl-phase-4-native-e2e`. Production-checkpoint, GGUF/mmproj, CUDA, generated-caption, and CLI parity remained unclaimed at that gate.

## Phase 5 Hybrid MMProj Evidence

The split exporter emits only the config-derived canonical SigLIP2/projector inventory into `mmproj.safetensors`, plus a versioned manifest and canonical processor JSON. It rejects missing, unexpected, shape-incompatible, non-dense, duplicate-normalized, or incomplete tensors before writing, requires an immutable source revision, and produces byte-identical output from the committed tiny unified fixture.

The Rust loader validates the manifest and processor pair, exact tensor inventory, shape/dtype/byte counts, bounded safetensors header and tensor count, offsets, overlaps, gaps, payload coverage, and hashes. A single fallibly allocated buffer—bounded from the validated manifest payload and maximum header—is used for hash, inspection, and construction, removing path-replacement ambiguity. GGUF metadata also rejects malformed present RoPE values and bounds rotary-table allocation before construction.

The deterministic hybrid proof writes real GGUF bytes from committed text tensors, pins SHA-256 `8fbd510aeea4715547c57975a7adcb91c148a8bc5e8d869d9617b69af6a006b1`, parses them with `gguf_file::Content::read`, and loads them through `ModelWeights::from_gguf`. Q8_0 is used where tiny matrix widths meet its block constraint; small unalignable tensors remain F32. Split and unified image features are exact. Relative to the native dense model, maximum absolute errors are prefill `4.457309842e-5`, cached decode `2.650916576e-5`, `2.175569534e-5`, and `1.309439540e-5`, with exact cache reset.

Final retained evidence: Python 19/19, `candle-transformers` 42/42 plus its integration tests, `candle-vlm` 25/25, the `lfm2-vl` example check, scoped Clippy gates, and the staged locked/offline baseline all pass. The CUDA-vision/CPU-text test is source-complete and asserts device residency and `1e-4` prefill agreement, but local execution is truthfully skipped because WSL exposes the RTX 4090 driver without a Linux CUDA toolkit or `nvcc`. The assigned worker confirmed all nine audit findings resolved and no remaining code blocker. Production models and GGUF files were not downloaded.

## Phase 6 Direct GGUF MMProj Evidence

The direct loader opens one stable GGUF handle, applies phase-specific parser limits before allocation, validates exact metadata and tensor inventory, checks dtypes, element counts, alignment, offsets, overlaps, truncation, retained dense bytes, and conservative peak bytes, then dequantizes into the already proven native SigLIP2/projector path. It requires `general.type=mmproj`; optional projector LayerNorm and bias tensors must be complete pairs. The only layout transform is the header-proven inverse for `v.patch_embd.weight`.

Official header-only evidence at `LiquidAI/LFM2.5-VL-450M-GGUF@166cd80bbe157dc86d65f964eb8cc6a2cede62ca` fixes the 201-tensor name set, physical shapes, F16/F32 and Q8_0/F32 placement, and absent preprocessing keys. Both exact 12,736-byte prefixes end at the tensor-data boundary and contain zero payload bytes. The direct path therefore retains official processor defaults and resolves the image placeholder ID from the tokenizer rather than inventing GGUF metadata.

The deterministic dense GGUF has SHA-256 `7361b57e6d9dbf2d7809d4f446944fdc7325b368e4444fee2bc3497376695256` and matches native image features exactly. The Q8_0 compatibility fixture dequantizes with maximum image-feature error `8.463021368e-5`. Paired with the deterministic quantized text GGUF, direct-MMProj prefill max abs is `4.457309842e-5`; cached decode is `2.650916576e-5`, `2.175569534e-5`, and `1.309439540e-5`; cache reset is exact. These are deterministic fixture results, not production-payload or llama.cpp runtime parity.

Final local evidence is green: pinned Python 23/23; the complete offline core/transformer/VLM test command, including all integrations and doc tests; strict scoped Clippy with five documented pre-existing Rust 1.97 allowances; and the exact staged locked/offline baseline. Retained hashes are recorded in `STATUS.md`. The assigned worker's final static re-audit found no remaining P0/P1 defect. No production model or MMProj payload was downloaded.

## Phase 7 Native Q8 MMProj Evidence

`LinearOp` now covers every vision attention projection, both vision MLP linears, and both projector linears. Dense construction still stores `candle_nn::Linear`; native Q8 construction stores `QMatMul::QTensor` directly and adds the dense bias afterward. Patch projection, positions, LayerNorm parameters, and biases remain dense. Mixed checkpoints may retain dense eligible matrices, while explicit native mode rejects lower-bit weights, Q8 dense roles, and non-block-aligned Q8 input widths.

The two-layer block-aligned fixture quantizes all 14 eligible linears and has GGUF SHA-256 `241f59dc92c033c9877654261cf538dc107087eab5834920bd4b0e52cbdcc056`. Native versus dequantized-Q8 operator max abs is `3.734588623e-3`; native versus the dense source is `5.300968885e-3` with cosine `0.999923348`. This is a documented quantized drift gate, distinct from the earlier dense CPU-F32 target of cosine `>=0.99999`.

The committed hybrid fixture's native-Q8 GGUF SHA-256 is `225241e57bc84c62d097aab6daa9466a75e920dbb858daf4cba4cc18ef8bb3f0`. Its image-feature max abs is `1.533385366e-4`; multimodal prefill is `1.650899649e-4`; three cached decode comparisons are `7.853843272e-5`, `6.113573909e-5`, and `4.052370787e-5`; cache reset is exact. The full local gate passes 23 Python tests, all core/transformer/VLM tests, and strict scoped Clippy. The assigned worker's final audit reports no P0/P1 defect in the initial CPU-F32 scope.

The example automatically selects native Q8 for valid F32 Q8 artifacts and reports the selected execution mode/count. F16/BF16 automatic loading deliberately stays on the Phase 6 dense path. No production payload or llama.cpp runtime was used, so official-file numerical parity, top-k/token agreement, and native-Q8 CUDA remain evidence gaps rather than claims.

## Evidence Rules

- Plausible captions are not parity evidence.
- Component tensors and exact metadata must be compared at the applicable phase gate.
- Fixture evidence must not be described as production-checkpoint or production-GGUF parity.
- No result may be marked green until the exact command and result are recorded in `STATUS.md`.

## Next Parity Task

The Phase 7 CPU-F32 checkpoint and sprint audit are complete. Any production-payload llama.cpp comparison, native-Q8 CUDA execution, or lower-bit vision support requires a separately authorized follow-up.

---
AI-edited: 2026-08-10T08:56:00-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=lfm2-vl-phase-7 | change=closed the native Q8 checkpoint while preserving remaining evidence boundaries
