# LFM2.5-VL Parity

## Current State

The deterministic reference fixture, LFM2 text compatibility path, SigLIP2 NaFlex tensor path, and Phase 3 native projector/composite path are established. No production-checkpoint, production GGUF, generated-text, raw-image processor, tokenizer/chat-template, CUDA, or CLI parity result is claimed.

## Required Gates

| Gate | Required evidence | Phase status |
| --- | --- | --- |
| Workspace baseline | Locked CPU-only Candle checks and diff check from Linux home | Green; Phase 1 proof log SHA-256 `c72eccd8b77689878689f7e720c46a040c26f3cee8060b17727392f392862f46` |
| Reference fixture | Deterministic pinned-Python export with component and multimodal tensors | Green; 87 tensors, byte-identical independent exports; manifest SHA-256 `c5461dadb0edfd920b20f308650c59676977110a1cc2f199e317dea7d75bdd7b` |
| LFM2 text configuration | 450M effective FFN width `4608`; 1.6B width `8192` | Green in config tests and header evidence |
| Dense text forwarding | Token-ID and embedding-driven prefill plus incremental decode agree | Green on the committed fixture; maximum hidden-state error `2.38418579e-7`, maximum logit error `2.98023224e-8` |
| Quantized text forwarding | Token-ID and embedding-driven paths agree and cache can be reset | API/equivalence gate green; production GGUF numerical parity pending |
| SigLIP2 | Component tensor comparisons against pinned fixtures | Phase 2 checkpoint complete; Phase 3 repeated-crop regression 8/8 green |
| Projector | Exact pixel-unshuffle and stage-level comparisons | Phase 3 focused gate green; 11/11 total |
| Processor | Exact resize, crop, patch, mask, shape, and token metadata | Not implemented |
| Composite model | Image-span replacement and prefill/decode parity | Phase 3 focused gate green; 11/11 total |
| Phase 3 library gate | Locked/offline `candle-transformers` library tests | 37/37 passed |
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

The Phase 3 checkpoint is pending manager review/commit. These results do not claim production-checkpoint parity, CUDA, GGUF, raw-image preprocessing, tokenizer/chat-template behavior, or CLI support.

## Evidence Rules

- Plausible captions are not parity evidence.
- Component tensors and exact metadata must be compared at the applicable phase gate.
- Fixture evidence must not be described as production-checkpoint or production-GGUF parity.
- No result may be marked green until the exact command and result are recorded in `STATUS.md`.

## Next Parity Task

Create the Phase 3 checkpoint/commit after manager review. Keep production checkpoint, raw-image processor, tokenizer/chat-template, CUDA, GGUF, and CLI validation as separately labeled future evidence.

---
AI-edited: 2026-08-10T01:49:59-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=lfm2-vl-phase-3-docs | change=recorded Phase 3 native composite proof and remaining checkpoint state
