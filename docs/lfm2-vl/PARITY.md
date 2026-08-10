# LFM2.5-VL Parity

## Current State

The deterministic reference fixture, LFM2 text compatibility path, and focused SigLIP2 NaFlex tensor path are established. No production-checkpoint, production GGUF, generated-text, processor, projector, or composite-model parity result is claimed.

## Required Gates

| Gate | Required evidence | Phase status |
| --- | --- | --- |
| Workspace baseline | Locked CPU-only Candle checks and diff check from Linux home | Green; Phase 1 proof log SHA-256 `c72eccd8b77689878689f7e720c46a040c26f3cee8060b17727392f392862f46` |
| Reference fixture | Deterministic pinned-Python export with component and multimodal tensors | Green; 87 tensors, byte-identical independent exports; manifest SHA-256 `c5461dadb0edfd920b20f308650c59676977110a1cc2f199e317dea7d75bdd7b` |
| LFM2 text configuration | 450M effective FFN width `4608`; 1.6B width `8192` | Green in config tests and header evidence |
| Dense text forwarding | Token-ID and embedding-driven prefill plus incremental decode agree | Green on the committed fixture; maximum hidden-state error `2.38418579e-7`, maximum logit error `2.98023224e-8` |
| Quantized text forwarding | Token-ID and embedding-driven paths agree and cache can be reset | API/equivalence gate green; production GGUF numerical parity pending |
| SigLIP2 | Component tensor comparisons against pinned fixtures | Focused 7/7, broader 25-test library, and full baseline green; Phase 2 checkpoint/tag pending |
| Projector | Exact pixel-unshuffle and stage-level comparisons | Not implemented |
| Processor | Exact resize, crop, patch, mask, shape, and token metadata | Not implemented |
| Composite model | Image-span replacement and prefill/decode parity | Not implemented |
| Production checkpoints and GGUF | Native versus production and GGUF numerical validation | Not run; no production weights or GGUF files downloaded |

## Phase 2 Focused Evidence

The manager's Linux-home WSL2 CPU F32 verifier passed all 7 SigLIP2 tests. The exact maximum absolute errors and cosine similarities were: patch projection `5.960464478e-8` / `0.999999940`; resized positions `2.980232239e-8` / `0.999999940`; embedding sum `1.192092896e-7` / `1.000000119`; encoder layer 0 `4.768371582e-7` / `0.999999881`; encoder layer 1 `1.192092896e-6` / `0.999999881`; returned post-LN `7.152557373e-7` / `1.000000119`; and the post-LN hook matched the returned result. Padding-key isolation was exact: max absolute error `0`, cosine `1`.

This focused proof covers packed patch projection, CPU F32 separable antialiased positional interpolation with per-shape caching, bidirectional key masking, encoder stages, post-LN, malformed-input rejection, and the required no-pooling-head boundary. The broader command `cargo test --locked --offline -p candle-transformers --lib` passed 25 tests with 0 failures; log `artifacts/verification/siglip2/candle-transformers-lib.log`, SHA-256 `b4c91d4bd6a0c1850a66d9cc27d61776b5ec96c152783e6c13f23a0cfcdf5197`. The full baseline passed from `2026-08-10T05:14:56Z` to `2026-08-10T05:15:08Z`; log `artifacts/verification/siglip2/baseline-final.log`, SHA-256 `727e0d8a029f121a7225d3d35a53addd480791323b6e0c501576408cc6460d52`. Only the Phase 2 checkpoint/tag remains pending.

## Evidence Rules

- Plausible captions are not parity evidence.
- Component tensors and exact metadata must be compared at the applicable phase gate.
- Fixture evidence must not be described as production-checkpoint or production-GGUF parity.
- No result may be marked green until the exact command and result are recorded in `STATUS.md`.

## Next Parity Task

Create the Phase 2 checkpoint/tag after review. Keep production checkpoint, processor, projector, composite, CUDA, and GGUF validation as separately labeled future evidence.

---
AI-edited: 2026-08-10T01:15:43-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=siglip2-phase-2-parity | change=recorded final Phase 2 gates without production overclaim
