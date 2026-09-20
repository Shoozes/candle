# Experimental GPT-OSS Candle Overlay Manifest

This manifest registers the experimental GPT-OSS support boundary separately
from the maintained LFM2-VL/Q8 product overlay. It owns only Candle-side
configuration admission, packed MXFP4 storage/loading, deterministic CPU
reference code, and the bounded Task 2/Task 3 proof seams. It does not
promote GPT-OSS to the product default.

## Current state

- C0 is delivered: local checkpoint admission rejects missing/model-free
  directories and never downloads weights.
- C1a is delivered: safetensors `*.blocks`/`*.scales` U8 payloads remain
  packed, and the CPU selected-expert reference decodes values on demand.
- C1b is delivered for synthetic weights: the CPU reference model proves
  cached and uncached final-token logits agree and reset replay is deterministic.
- C2a is delivered as a hash-pinned GGUF admission and assembly boundary: the
  exact owner SHA-256 is checked before parsing, GPT-OSS metadata is normalized
  into `GptOssConfig`, and the complete tensor inventory is owned by role,
  shape, dtype, and bounded raw byte range. Fused and current converter-style
  split expert layouts are fail-closed and covered by synthetic GGUF fixtures;
  one retained bounded session now assembles both fixtures and the exact
  owner-selected product artifact into `GptOssWeights`.
- Task 2 is delivered for the synthetic CPU boundary: an independently
  generated packed/router/attention/forward/cache oracle is digest-pinned,
  token and logical KV-cache byte admission is checked before mutation,
  prefill/decode cancellation rolls back retained state, and cancellable RAII
  load leases prevent duplicates while releasing cancelled, failed, and
  completed ownership.
- Task 3 is delivered for the bounded native CUDA boundary: packed U8
  MXFP4 blocks/scales remain device-resident, a dedicated kernel covers both
  expert projections, and attention/router/cache/forward traces match the
  independent oracle at absolute tolerance `1e-4`. The expert helper now
  exposes a pure contribution contract, so CPU and CUDA both form the MoE
  residual as `post_attention_hidden + expert_contribution`. Static weight and
  logical cache admission precede device/forward allocation, and cancellation,
  rollback, eviction, and typed failure cases are covered.
- The exact owner-selected GGUF was read in place at the owner-provided path,
  admitted by its pinned SHA, and assembled into a CPU model object; it was
  not copied into this checkout. Tokenizer, forward-logit, and production
  numerical claims remain closed. CUDA execution evidence is still limited to
  the synthetic fixture on the named local RTX 4090 lane.
- Candle's maintained Q8 path and LFM2-VL release boundary remain unchanged.

The implementation remains production-execution-free at this boundary. The
C2a loader admits one exact GGUF byte identity, normalizes its directory, and
can assemble a live CPU model object through one bounded retained file
session; Q8_0 dense text dequantization and tokenizer/generation integration
remain deferred. Task 2 exercises the synthetic CPU reference and bounded
ownership contracts; Task 3 exercises an explicit opt-in CUDA executor over the
same bounded weights. Exact assembly is proven, but an inference receipt is
still required before any production parity claim.

## Overlay-owned additions

- `docs/gpt-oss/MOD_MANIFEST.md`
- `docs/gpt-oss/SOURCES.md`
- `docs/gpt-oss/STATUS.md`
- `candle-transformers/src/models/gpt_oss/mod.rs`
- `candle-transformers/src/models/gpt_oss/checkpoint.rs`
- `candle-transformers/src/models/gpt_oss/cuda.rs`
- `candle-transformers/src/models/gpt_oss/config.rs`
- `candle-transformers/src/models/gpt_oss/gguf.rs`
- `candle-transformers/src/models/gpt_oss/mxfp4.rs`
- `candle-transformers/src/models/gpt_oss/model.rs`
- `candle-transformers/src/models/gpt_oss/runtime.rs`
- `candle-kernels/src/ffi.rs`
- `candle-kernels/src/gpt_oss_mxfp4.cu`
- `tests/fixtures/gpt_oss_task2/README.md`
- `tests/fixtures/gpt_oss_task2/oracle.json`
- `tests/fixtures/gpt_oss_task3_two_layer/README.md`
- `tests/fixtures/gpt_oss_task3_two_layer/oracle.json`
- `scripts/gpt-oss/verify-mod-manifest.sh`
- `scripts/tests/test-verify-fork-overlays.sh`

## Shared overlay paths

These paths are also owned by an existing overlay and are listed in the
repository-wide shared-path registry:

- `candle-transformers/src/models/mod.rs`
- `candle-kernels/build.rs`
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
  admitted raw payloads without dequantizing them. Its bounded load methods
  assemble dense F32/BF16 tensors plus packed MXFP4 experts, and the combined
  loader retains one file session and one RAII registry lease.
- `Mxfp4ExpertOperation` applies the official FP4 lookup, E8M0 scale, SwiGLU,
  and selected-expert routing; `forward` retains its residual-returning API
  while `forward_contribution` returns only the weighted expert contribution.
- `GptOssModel`/`GptOssWeights` provide a synthetic CPU forward/cache proof
  boundary with explicit sequence/KV-byte limits, prefill/decode cancellation,
  rollback, and logical/capacity usage reporting.
- `GptOssLoadRegistry` and `GptOssGgufArtifact::open_with_registry` provide
  duplicate-load prevention plus RAII release on failed retry and completed
  load ownership. No worker process or hidden download is created.
- `GptOssCudaConfig`, `GptOssCudaModel`, and `GptOssCudaError` provide the
  feature-gated packed CUDA proof boundary with explicit device/dtype and
  static/cache admission, cancellation rollback, eviction, and typed kernel
  failures. The executor is not wired into product GGUF loading.

## Completion boundary

Task 3 remains complete for the bounded synthetic CPU/CUDA proof boundary, and
the uncommitted correction now includes narrowed-view CUDA equivalence and
synthetic/exact GGUF-to-weights assembly. The next dependency is quantized
GPT-OSS text plus split dense MMProj work; exact-model numerical parity remains
blocked until tokenizer and forward-logit evidence is produced.

---

AI-edited: 2026-09-20; agent=Codex; task=verify-fork-overlays; change=registered the isolated repository-wide overlay regression suite
