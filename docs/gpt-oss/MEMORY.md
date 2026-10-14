# GPT-OSS CUDA memory candidate

The selected Work repairs the native 32k capacity boundary while preserving
F32 activations, packed MXFP4 experts, CPU reference behavior and consumer holds.
The candidate is locally checkpointed with short numerical/runtime proof; long
capacity and consumer execution remain unqualified. Existing receipts remain
bound to their original source; no closed inference allowance is reused.

## Lossless table storage

CUDA embedding and output tables use BF16 storage only when every finite F32
value reconstructs with identical bits from BF16. Otherwise the entire table
retains F32 storage. Norms, biases and other dense projections remain F32.
Embedding lookup converts the selected row to F32. The output kernel reads BF16
weights and performs explicit F32 multiply/add and warp reduction with F32 inputs
and outputs. This storage policy does not enable BF16 activations or FastBf16.
The reduction order differs from the old dense output matmul and needs fresh
numerical qualification; the frozen component bound is absolute `1e-4`.

CPU source assembly remains bounded by `max_weight_bytes`. CUDA device admission
uses the exact selected table residency after assembly and before device creation,
instead of applying the CPU assembly byte count to its device ceiling.
`GptOssSession::static_weight_bytes` reports the selected backend's payload.
Temporary host conversion/upload storage remains part of measured Job admission.

A bounded header-only observation of the retained GGUF read 13,008,320 bytes,
SHA-256 `5c92e42711d0d30204668789a42713b83da35afb1d5f610f9d6df8552b3c385e`.
Its 459 descriptors contain 72 MXFP4, 289 F32 and 98 BF16 tensors; both
`token_embd.weight` and `output.weight` have BF16 shape `[201088,2880]` in
matrix order. No tensor payload or device was opened for that observation.
This header identity does not replace fresh whole-file admission in a proof Job.

## Cache and attention

CUDA caches use immutable pages of at most 64 rows. Transactions clone metadata
and share complete pages. Appending replaces only a partial tail with a private
allocation. A new page owns its row, avoiding retention of the larger QKV parent.
Failure/cancellation drops the candidate and preserves the committed root; only
the existing synchronize, finite-logit check and final cancellation boundary
commit it. The absolute sequence position remains monotonic for RoPE.

Full-attention layers retain all rows. Sliding layers retain at most their window
logically, and at most `window + 63` allocated rows because a partially evicted
front page may remain shared. `resource_usage` counts actual logical rows and
owned page capacity, rather than charging every layer for the full sequence.

Attention materializes one layer's page rows and uses batched grouped-query
matrix products. K/V are not expanded to query-head count and large elementwise
query/value products are eliminated. Materialization still copies a full layer
prefix; this candidate makes no claim that decode has eliminated that bandwidth.
CPU attention and its historical accounting helpers are unchanged.

## Backend-specific admission

The new public pure helpers are `cuda_cache_bytes_for_tokens`,
`cuda_cache_capacity_bytes_for_tokens`, `cuda_workspace_bytes_for_tokens` and
`cuda_peak_device_bytes_for_forward(config, static_bytes, committed, incoming)`.
The transaction bound counts candidate pages, old partial full-attention tails,
old sliding windows, one overlapping append tail, gathered/permuted K/V,
score/softmax buffers, fixed layer work and three output rows. Arithmetic is
checked. Empty incoming work fails. Session and CUDA admission use these same
bounds; sequence and cache limits remain independent.

An uncached evaluation also charges the original cache it leaves alive while
its temporary model runs. Caller-retained outputs or other models remain external
owners and require the physical launch ceiling.

For the frozen retained 20B dimensions, lossless table storage would save
2,316,533,760 payload bytes from the prior 17,370,183,936-byte F32 payload.
At 32,768 tokens the candidate cache-capacity bound is 1,620,000,768 bytes,
workspace is 305,643,520 bytes, and resumed one-token peak accounting is
16,992,041,216 bytes. The prior accounting was 24,903,656,192 bytes.
These are model-free arithmetic, not observed production memory.

The accounting excludes CUDA context, allocator retention, other processes and
runtime overhead. It is not a hard physical peak or a 32k qualification claim.
Fresh bounded whole-GPU/host measurements, numerical rows and recovery proof
remain necessary. Responses keeps 90 seconds; longer work uses WorkerRuntime's
600-second boundary. One route never qualifies the other.

## Verification and next gate

Model-free controls cover lossless selection/fallback, unchanged F32 input bits,
mixed-storage component parity, offset/permuted views, page boundaries, physical
window retention, snapshot failure/cancellation, exact recovery and admission.
CPU proof precedes CUDA fixtures. Production files are not used by these tests.

Observed local results: CPU57/CUDA69 pass with one production test ignored in each
lane; strict library Clippy on both devices, CUDA examples, affected routes and
inventories, and the full native locked/offline gate pass. CUDA fixture/Clippy
Jobs exit naturally and verify PID absence. Expanded test Clippy exposes nine
pre-existing style lints in unchanged GGUF/MXFP4 tests; it is not green. WSL
replay remains unrun. Compact proof is retained under
`artifacts/gpt-oss/memory-32k-20261013/` against the `48d81485` source baseline
and the reviewed candidate file identities.

The approved checkpoint is `3db05cd8`; separately approved packet
`c5c1fe4e…f8face4f` completes the frozen short CPU/CUDA proof: two loads,
44 forwards/74 steps/32 outputs, zero warmup/retry. CUDA selected static storage
is 15,053,650,176 bytes; sampled whole-GPU peak is 16,724,787,200 bytes, below 20e9.
Oracle bounds, exact replay, matching cold IDs, actual DLLs and all six owner
releases pass. [CHARACTERIZATION](CHARACTERIZATION.md) retains the source-bound
measurements and the post-run empty-console-log reporting failure separately.
The allowance is closed; this does not prove physical 32k or consumer deadlines.

The distinct [8k capacity runner](CAPACITY.md) and model-free controls are now
implemented; no capacity model run has occurred. Release drafts/startup Jobs pass.
Next obtain its scoped checkpoint after native closeout, then rebuild a clean
source-bound packet. The current short helper remains eight-token-only. New capacity execution, checkpoint and publication
need separate authority. Preserve Q8 defaults, GenEye `3fa7075e`, Codex `0.154.0` and
the disabled GPT-OSS provider until its consumer route is qualified.

---
AI-created: 2026-10-13 | agent=Codex/root | model=unknown | effort=unknown | task=gpt-oss-memory | change=recorded lossless table and paged GQA candidate boundary
