# LFM2.5-VL Parity

## Bootstrap State

No tensor, processor, model-output, or generated-text parity result has been established. This file is a bootstrap control document, not a parity claim.

## Required Gates

| Gate | Required evidence | Bootstrap status |
| --- | --- | --- |
| Workspace baseline | Locked CPU-only Candle checks and diff check from Linux home | Green; proof log SHA-256 `a4f77d1b007eb267865be01ef1c239754ac0e093dd1c27ad457d77242b614f22` |
| LFM2 text configuration | 450M effective FFN width `4608`; 1.6B width `8192` | Not tested |
| Text forwarding | Token-ID and embedding-driven prefill plus incremental decode agree | Not tested |
| SigLIP2 | Component tensor comparisons against pinned fixtures | Not implemented |
| Projector | Exact pixel-unshuffle and stage-level comparisons | Not implemented |
| Processor | Exact resize, crop, patch, mask, shape, and token metadata | Not implemented |
| Composite model | Image-span replacement and prefill/decode parity | Not implemented |
| GGUF | Native versus GGUF feature and pairing validation | Not implemented |

## Evidence Rules

- Plausible captions are not parity evidence.
- Component tensors and exact metadata must be compared at the applicable phase gate.
- No model downloads or production outputs are part of Bootstrap Phase.
- No result may be marked green until the exact command and result are recorded in `STATUS.md`.

## Next Parity Task

Perform the reference-source lock phase without changing Candle Rust source or downloading production weights.

---
AI-edited: 2026-08-09T22:35:40-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=docs | change=recorded green workspace baseline gate
