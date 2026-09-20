# Experimental GPT-OSS Candle Overlay Manifest

This manifest registers the experimental GPT-OSS support boundary separately
from the maintained LFM2-VL/Q8 product overlay. It owns only Candle-side
configuration admission, packed MXFP4 storage/loading, and deterministic CPU
reference code. It does not promote GPT-OSS to the product default.

## Current state

- C0 is delivered: local checkpoint admission rejects missing/model-free
  directories and never downloads weights.
- C1a is delivered: safetensors `*.blocks`/`*.scales` U8 payloads remain
  packed, and the CPU selected-expert reference decodes values on demand.
- C1b is delivered for synthetic weights: the CPU reference model proves
  cached and uncached final-token logits agree and reset replay is deterministic.
- C2a is delivered as a hash-pinned GGUF admission boundary: the exact owner
  SHA-256 is checked before parsing, GPT-OSS metadata is normalized into
  `GptOssConfig`, and the complete tensor inventory is owned by role, shape,
  dtype, and bounded raw byte range. Fused and current converter-style split
  expert layouts are fail-closed and covered by synthetic GGUF fixtures.
- No real GPT-OSS checkpoint, tokenizer, artifact manifest, production trace,
  or CUDA execution has been loaded or claimed; the owner-selected artifact
  is not stored in this checkout.
- Candle's maintained Q8 path and LFM2-VL release boundary remain unchanged.

The implementation remains execution-free at this boundary. The C2a loader
admits one exact GGUF byte identity and normalizes its directory without
dequantizing or constructing a live model. A future executor must publish its
external artifact receipt and independent numerical fixtures before any
runtime or parity claim is made.

## Overlay-owned additions

- `docs/gpt-oss/MOD_MANIFEST.md`
- `docs/gpt-oss/SOURCES.md`
- `docs/gpt-oss/STATUS.md`
- `candle-transformers/src/models/gpt_oss/mod.rs`
- `candle-transformers/src/models/gpt_oss/checkpoint.rs`
- `candle-transformers/src/models/gpt_oss/config.rs`
- `candle-transformers/src/models/gpt_oss/gguf.rs`
- `candle-transformers/src/models/gpt_oss/mxfp4.rs`
- `candle-transformers/src/models/gpt_oss/model.rs`
- `scripts/gpt-oss/verify-mod-manifest.sh`

## Shared overlay paths

These paths are also owned by an existing overlay and are listed in the
repository-wide shared-path registry:

- `candle-transformers/src/models/mod.rs`
- `docs/FORK_OVERLAYS.md`
- `docs/lfm2-vl/START_HERE.md`
- `docs/lfm2-vl/DECISIONS.md`
- `docs/lfm2-vl/STATUS.md`
- `docs/lfm2-vl/TODO.md`
- `scripts/verify-fork-overlays.sh`
- `summary_bank.json`

## Public Candle interfaces

- `candle_transformers::models::gpt_oss::GptOssCheckpoint` validates a local
  original-format checkpoint layout and rejects model-free input.
- `GptOssConfig` parses and validates architecture dimensions without
  hardcoding a checkpoint name.
- `PackedMxfp4` loads U8 blocks/scales while retaining only packed storage and
  exposes CPU reference matrix operations.
- `GptOssGgufArtifact` admits only
  `aab205256a9b6361e410c24de3086e30f907092ca6f9ba8cd4b22c8a2b025778`,
  normalizes GPT-OSS GGUF metadata, owns the full tensor inventory, and reads
  admitted raw payloads without dequantizing them.
- `Mxfp4ExpertOperation` applies the official FP4 lookup, E8M0 scale, SwiGLU,
  selected-expert routing, and residual reference operation.
- `GptOssModel`/`GptOssWeights` provide a synthetic CPU forward/cache proof
  boundary. `GptOssCheckpoint::open` is the only C0 admission boundary; no
  production model loader is exposed in C1.

## Completion boundary

The next dependency is Task 2: independent numerical fixtures plus explicit
token/cache byte bounds, cancellation/rollback, and no-duplicate/load-leak
evidence for the admitted GGUF boundary. Only after that gate is accepted
should a packed CUDA executor be proposed.

---

AI-edited: 2026-09-19; agent=Codex; task=gpt-oss-c0-c1; change=registered experimental packed-MXFP4 CPU boundary
