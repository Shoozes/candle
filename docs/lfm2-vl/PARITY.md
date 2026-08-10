# LFM2.5-VL Parity

## Current State

The deterministic reference fixture and LFM2 text compatibility path are established. No production-checkpoint, production GGUF, generated-text, SigLIP2, processor, projector, or composite-model parity result is claimed.

## Required Gates

| Gate | Required evidence | Phase status |
| --- | --- | --- |
| Workspace baseline | Locked CPU-only Candle checks and diff check from Linux home | Green; Phase 1 proof log SHA-256 `c72eccd8b77689878689f7e720c46a040c26f3cee8060b17727392f392862f46` |
| Reference fixture | Deterministic pinned-Python export with component and multimodal tensors | Green; 87 tensors, byte-identical independent exports; manifest SHA-256 `c5461dadb0edfd920b20f308650c59676977110a1cc2f199e317dea7d75bdd7b` |
| LFM2 text configuration | 450M effective FFN width `4608`; 1.6B width `8192` | Green in config tests and header evidence |
| Dense text forwarding | Token-ID and embedding-driven prefill plus incremental decode agree | Green on the committed fixture; maximum hidden-state error `2.38418579e-7`, maximum logit error `2.98023224e-8` |
| Quantized text forwarding | Token-ID and embedding-driven paths agree and cache can be reset | API/equivalence gate green; production GGUF numerical parity pending |
| SigLIP2 | Component tensor comparisons against pinned fixtures | Not implemented |
| Projector | Exact pixel-unshuffle and stage-level comparisons | Not implemented |
| Processor | Exact resize, crop, patch, mask, shape, and token metadata | Not implemented |
| Composite model | Image-span replacement and prefill/decode parity | Not implemented |
| Production checkpoints and GGUF | Native versus production and GGUF numerical validation | Not run; no production weights or GGUF files downloaded |

## Evidence Rules

- Plausible captions are not parity evidence.
- Component tensors and exact metadata must be compared at the applicable phase gate.
- Fixture evidence must not be described as production-checkpoint or production-GGUF parity.
- No result may be marked green until the exact command and result are recorded in `STATUS.md`.

## Next Parity Task

Checkpoint Phase 1 and stop for review. Begin SigLIP2 parity only after user continuation; keep production checkpoint and GGUF validation as separately labeled future evidence.

---
AI-edited: 2026-08-10T00:34:34-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=text-compatibility | change=recorded reference and Phase 1 text parity without production overclaim
