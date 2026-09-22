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
  serialized GGML MXFP4 bytes are normalized into Candle's internal packed
  layout, and one retained bounded session now assembles both fixtures and the
  exact owner-selected product artifact into `GptOssWeights`.
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
  rollback, eviction, typed failure cases, public-config validation, finite
  output checks, and final pre-commit cancellation are covered.
- The exact owner-selected GGUF was read in place at the owner-provided path,
  admitted by its pinned SHA, and assembled into a CPU model object; it was
  not copied into this checkout. Tokenizer, forward-logit, and production
  numerical claims remain closed. CUDA execution evidence is still limited to
  the synthetic fixture on the named local RTX 4090 lane.
- Candle's maintained Q8 path and LFM2-VL release boundary remain unchanged.

The implementation remains production-execution-free at this boundary. The
C2a loader admits one exact GGUF byte identity, normalizes its directory and
serialized MXFP4 wire layout, and can assemble a live CPU model object through
one retained file session without per-tensor reopen/rehash. The load lease is
held by the live model/runtime owner; Q8_0 dense text dequantization and
tokenizer/generation integration remain deferred. Task 2 exercises the
synthetic CPU reference and bounded ownership contracts; Task 3 exercises an
explicit opt-in CUDA executor over the same bounded weights. Exact assembly is
proven, but an inference receipt is still required before any production parity
claim. The bounded native performance packet now supplies an optimized-release
2080-total-token measurement envelope (2048 prompt tokens plus a 32-token
reserve) under the exact 20,000,000,000-byte Edge ceiling with a fail-closed
physical monitor; larger target contexts are intentionally refused by the
wrapper, and the 8160-prompt diagnostic is retained only as superseded evidence.

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
- `tests/fixtures/gpt_oss_task3_short_parity/README.md`
- `tests/fixtures/gpt_oss_task3_short_parity/reference.json`
- `candle-examples/examples/gpt-oss-short-parity.rs`
- `scripts/gpt-oss/run-short-parity.ps1`
- `candle-examples/examples/gpt-oss-performance.rs`
- `scripts/gpt-oss/run-performance.ps1`
- `scripts/gpt-oss/test-performance.ps1`
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
  normalize GGML MXFP4 wire bytes into adjacent-nibble packed tensors, assemble
  dense F32/BF16 tensors plus packed experts, and the combined loader retains
  one file session and one RAII registry lease.
- `Mxfp4ExpertOperation` applies the official FP4 lookup, E8M0 scale, SwiGLU,
  and selected-expert routing; `forward` retains its residual-returning API
  while `forward_contribution` returns only the weighted expert contribution.
- `GptOssModel`/`GptOssWeights` provide a synthetic CPU forward/cache proof
  boundary with explicit sequence/KV-byte limits, prefill/decode cancellation,
  rollback, and logical/capacity usage reporting.
- `GptOssLoadRegistry` and `GptOssGgufArtifact::open_with_registry` provide
  duplicate-load prevention plus RAII release on failed retry and completed
  load ownership. The combined load lease remains live through model/runtime
  teardown. No worker process or hidden download is created.
- `GptOssCudaConfig`, `GptOssCudaModel`, and `GptOssCudaError` provide the
  feature-gated packed CUDA proof boundary with explicit device/dtype and
  static/cache/total-device admission, cancellation rollback, eviction, and
  typed kernel failures. Runtime helpers expose checked static+KV+workspace
  accounting and exact-boundary search. The executor is not wired into product
  GGUF loading.

## Completion boundary

Task 3 remains complete for the bounded synthetic CPU/CUDA proof boundary, and
the current Task 3 correction adds GGML MXFP4 wire normalization,
retained-file identity/ownership, CUDA failure-atomic commit guards, and a
hash-bound short CUDA parity receipt on top of the narrowed-view and
synthetic/exact GGUF assembly proof. A separate performance harness now
qualifies an explicit autoregressive bounded native profile with optimized
release identity, exact 20,000,000,000-byte admission, checked static+KV+
workspace accounting, target refusal above 8192 total tokens, fail-closed
observed-device monitoring, deadline, cancellation, progress, cleanup, and
model-free failure tests; teacher-forced measurement remains explicitly
labeled. The accepted packet is the observed-safe 2080-total-token Candle
measurement only; the clean 8160-prompt diagnostic is superseded after a
physical-ceiling sample. The next dependency is quantized GPT-OSS text plus
split dense MMProj work; Q8/default and broad maintained production claims
remain outside this opt-in receipt.

---

AI-edited: 2026-09-20; agent=Codex; task=verify-fork-overlays; change=registered the isolated repository-wide overlay regression suite
