# LFM2.5-VL Parity

## Current State

The deterministic reference fixture, LFM2 text compatibility path, SigLIP2 NaFlex tensor path, Phase 3 native projector/composite path, and Phase 4 Rust-native raw-image/prompt path are established. No production-checkpoint, production GGUF, generated-text, CUDA, or CLI parity result is claimed.

## Required Gates

| Gate | Required evidence | Phase status |
| --- | --- | --- |
| Workspace baseline | Locked CPU-only Candle checks and diff check from Linux home | Phase 4 staged baseline green; log SHA-256 `fb16481302c9bdf15f4b04df0250e45bf8c9a2126b92b09a787f5360cc3a3140` |
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

The pinned Python reference tests passed 9/9, the full `candle-transformers` library regression remained green at 37/37, and the final staged baseline passed formatting, all required package/example checks, and both diff gates. The Phase 4 checkpoint remains pending manager commit/tag. Production-checkpoint, GGUF/mmproj, CUDA, generated-caption, and CLI parity remain unclaimed.

## Evidence Rules

- Plausible captions are not parity evidence.
- Component tensors and exact metadata must be compared at the applicable phase gate.
- Fixture evidence must not be described as production-checkpoint or production-GGUF parity.
- No result may be marked green until the exact command and result are recorded in `STATUS.md`.

## Next Parity Task

Create the Phase 4 checkpoint, then prove the Phase 5 hybrid quantized-text plus dense-mmproj loader before direct GGUF mmproj work.

---
AI-edited: 2026-08-10T04:35:42-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=lfm2-vl-phase-4-docs | change=recorded exact Rust processor, prompt, metadata, fixture, and baseline parity
