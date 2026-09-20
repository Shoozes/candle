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
- No real GPT-OSS checkpoint, tokenizer, artifact manifest, production trace,
  or CUDA execution has been loaded or claimed.
- Candle's maintained Q8 path and LFM2-VL release boundary remain unchanged.

The implementation is intentionally model-free at this boundary. A future
production loader must admit one exact checkpoint/config/tokenizer identity,
publish its external artifact receipt, and add a reference trace before any
live-model claim is made.

## Overlay-owned additions

- `docs/gpt-oss/MOD_MANIFEST.md`
- `docs/gpt-oss/SOURCES.md`
- `docs/gpt-oss/STATUS.md`
- `candle-transformers/src/models/gpt_oss/mod.rs`
- `candle-transformers/src/models/gpt_oss/checkpoint.rs`
- `candle-transformers/src/models/gpt_oss/config.rs`
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
- `Mxfp4ExpertOperation` applies the official FP4 lookup, E8M0 scale, SwiGLU,
  selected-expert routing, and residual reference operation.
- `GptOssModel`/`GptOssWeights` provide a synthetic CPU forward/cache proof
  boundary. `GptOssCheckpoint::open` is the only C0 admission boundary; no
  production model loader is exposed in C1.

## Completion boundary

The next dependency is an owner-admitted immutable GPT-OSS checkpoint snapshot
with its exact `config.json`, tokenizer identity, safetensors inventory, and
an independent reference trace. Only after that evidence exists should a
production CPU loader or CUDA implementation be proposed.

---

AI-edited: 2026-09-19; agent=Codex; task=gpt-oss-c0-c1; change=registered experimental packed-MXFP4 CPU boundary
