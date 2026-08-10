# LFM2.5-VL Sources

## Bootstrap State

This is an honest Bootstrap Phase inventory placeholder. No external source revision, model revision, processor revision, or license record has been pinned in this phase. The entries below are planned authorities only, not source-lock claims.

No implementation code was adapted from an external repository during this slice.

## Planned Authorities

| Source | Intended use | Revision or model pin | License and adaptation status |
| --- | --- | --- | --- |
| Hugging Face Transformers | Numerical model, processor, and image-processing oracle | Not pinned | To be recorded during Source Lock Phase; reference first |
| LiquidAI LFM2.5-VL model files | Official model and processor configuration | Not pinned | To be recorded during Source Lock Phase; reference first |
| mistral.rs | Primary Rust implementation reference | Not pinned | To be recorded during Source Lock Phase; adapt only the required math with notices preserved |
| llama.cpp | GGUF metadata, tensor layout, and independent parity reference | Not pinned | To be recorded during Source Lock Phase; reference first |
| MLX-VLM and Transformers.js | Secondary independent shape and processing references | Not pinned | To be recorded during Source Lock Phase; reference first |

## Bootstrap Repository

The local Candle 0.11 checkout is the implementation baseline for this task. Its current detached worktree commit is recorded in `STATUS.md`; this does not replace the external source-lock phase.

## Next Source Task

Pin exact revisions, paths, purposes, authority levels, licenses, and permitted adaptation boundaries before using external implementations as parity authorities.

---
AI-edited: 2026-08-09T22:35:40-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=docs | change=established honest bootstrap source inventory
