# GPT-OSS Experimental Candle Status

## Current disposition

Deferred experimental capability, outside the five-overlay MVP's completion
criteria. Review baseline is `63f0d2cd275e4653b268939b429cbb9af25aba1a`; latest
short model execution remains `2adfa4a337094d63900e03b0b5ef86130f5174b9`.
The release tooling batch changes no GPT model arithmetic. Completed narratives
and prior states are linked from [HISTORY](../lfm2-vl/HISTORY.md#2026-10-14---independent-mvp-scope-freeze).

## Proven evidence and limits

Fresh packet `0a244611` completed CPU then CUDA: two loads, six segments,
44 public forwards, 74 processed steps, 32 outputs, no warmups/retries.
Independent reference comparison, exact replay, actual DLLs, normal exits,
registry zero and seven-owner cleanup passed. All allowances are closed;
original receipts and earlier failed attempts remain unchanged in
[CHARACTERIZATION](CHARACTERIZATION.md#complete-short-qualification-at-2adfa4a3-2026-10-14).
The CUDA sample high 19,920,846,848 bytes is below 20e9 by only 79,153,152 bytes;
sampling can miss peaks and includes other GPU owners. Decode mean 78.59824ms
is a forward measurement, not end-to-end route or chat performance.

The [8k fixed-token stress](CAPACITY.md) completes but uses 637.218s of forward
work, exceeding both 90s Responses and 600s WorkerRuntime limits for that input.
No physical 16k/32k, long-context quality, maintained coding, generated-answer
route or consumer activation claim follows. Edge keeps GPT disabled and its
existing Q8/Nomic default graph; Symbio keeps Codex 0.154.0.

Model-free Short preparation is retained under the ignored
`artifacts/gpt-oss/practical-short-63f0d2cd-20261014/` root. Fifteen unpublished
source files are parked byte-for-byte with `parking.json`. Draft `34ce68aa`
has zero model work, no native generation runner and a fresh rejected GPU
admission; it is excluded from this release source. Its future actual launch
must bind GPU UUID, not only an ordinal.

## Last green verification

Baseline `63f0d2cd` native publication gate passed: maintained checks/tests,
strict Clippy, 461 model-free GPT controls and remote equality. The active
release-only verification is recorded by [Candle status](../lfm2-vl/STATUS.md).
No current-task production inference or CUDA model proof ran.

## Accepted capability rows

| Capability | State | Evidence boundary |
| --- | --- | --- |
| Exact artifact admission | Accepted | `GptOssGgufArtifact::open` hashes before parsing and admits only the selected SHA-256. |
| GGUF directory/config normalization | Accepted | Strict v2/v3 parser, `gpt-oss` to `gpt_oss` normalization, YaRN/context and dimension checks. |
| Packed MXFP4 ownership | Accepted | CUDA keeps expert blocks/scales as device U8 tensors and dispatches a dedicated packed kernel; no dense expert fallback exists. |
| Packed expert/router/attention execution | Accepted | CUDA component trace and direct packed output match the independent oracle at absolute tolerance `1e-4`. |
| Expert contribution/residual contract | Accepted | `Mxfp4ExpertOperation::forward_contribution` returns only weighted expert output; CPU and CUDA add it directly to the post-attention hidden state. |
| Post-attention/post-MoE traces | Accepted | CPU and CUDA expose both hidden states in the two-layer independent fixture, including sliding/full attention, sinks, unequal routing, and position 3. |
| CUDA forward/cache execution | Accepted | Uncached logits and cached prefill/decode match the oracle; exact logical cache usage is reported. |
| Explicit CUDA selection | Accepted | Public feature-gated config selects device index and rejects non-F32 activation dtype with typed errors. |
| Static/runtime resource admission | Accepted | Logical static weight bytes are checked before device creation; token and exact KV-byte admission happen before forward mutation. |
| Cancellation/rollback/eviction | Accepted | Cancellation before allocation and at admitted work boundaries leaves the retained cache unchanged; eviction releases logical cache state and retry succeeds. |
| CUDA narrowed-view equivalence | Accepted | Padded/narrowed CUDA views match the materialized-copy path; pre-launch device/dtype/shape checks run before dispatch and cudarc pointer guards remain alive through launch. |
| GGML MXFP4 wire normalization | Accepted | Real serialized fused and split GGUF fixtures normalize all 32 wire coordinates, preserve mixed signs/scales, and match an independent decoder before expert execution. |
| GGUF-to-weights assembly | Accepted | Fused and converter-style split synthetic fixtures assemble and execute on CPU; the exact owner-selected 13,792,638,656-byte artifact assembles under the 32 GiB resident bound. |
| Retained-file integrity and ownership | Accepted | Admission retains one identity-verified handle, Windows read-opens deny write/delete sharing, load reads avoid reopen/rehash, cancellation is chunk-aware, and the registry lease follows the live model/runtime owner. |
| CUDA validation and commit guards | Accepted | All public CUDA config fields validate before device setup; synchronization, finite-output validation, and a final cancellation checkpoint precede cache commit, with delayed failure/cancellation recovery tests. |
| Product artifact numerical parity | Accepted for bounded short CUDA gate | The exact short runner proves the pinned tokenizer IDs, real CUDA construction, cancellation rollback, prefill/decode parity, deterministic reset/replay, and registry teardown. The robust criterion requires exact top-1 agreement, top-k union log-probability error <= `0.25`, mean full-row error <= `0.01`, and probability-space total variation <= `0.01`; all three stages pass. This is not yet a cross-device, Q8, or maintained product-path claim. |
| Performance characterization harness | Accepted for bounded native packet | `gpt-oss-performance` defaults to optimized release, records build/model/tokenizer/GPU/runtime identities, enforces the exact 20,000,000,000-byte Edge ceiling with a fail-closed physical monitor, reports checked static+KV+workspace accounting, preserves cancellation/stop reasons, and writes cleanup evidence. The accepted packet passes 8/512/2048 prompt cases with 32 generated tokens; the 8160-prompt diagnostic is superseded after an observed physical-ceiling violation. No 32k or maintained production claim is made. |


## Deferred optional acceleration research

Reviewed 2026-10-07 against Candle `main` at
`b274902312a39cbe1b0028141276f43e64b56096`. This note records future research;
it establishes no speedup or new qualification. GenAce MVP/LTS and GenUni's
current foundation retain priority. The short-parity and observed-safe
2,080-token performance receipts above retain their executed source identities.
The [module header](../../candle-transformers/src/models/gpt_oss/mod.rs) and
manifest describe these separate evidence boundaries; documentation cleanup
does not renew the runtime receipts.

### Profile existing safe cases first

At this source, the [packed kernel](../../candle-kernels/src/gpt_oss_mxfp4.cu)
assigns each route/output to one thread and reduces K serially. The
[CUDA executor](../../candle-transformers/src/models/gpt_oss/cuda.rs) accepts F32
activations, processes prefill tokens individually, concatenates KV tensors,
and allocates routing/activation intermediates. These are source observations,
not measured bottlenecks. Candidate techniques are:

- Prefill: bounded token chunks and grouping by expert to expose larger GEMMs,
  preserving causal/sliding/sink attention and router ordering.
- Low-batch decode: cooperative Ada packed-MXFP4 reduction; large-M prefill
  throughput does not predict one-token latency.
- Both: fusion and scratch/KV buffer reuse while preserving staged cache commit,
  cancellation and packed expert ownership. Lower precision or activation
  quantization needs separate numerical proof.

Dense LFM2 has no GPT-OSS MoE routing requirement. Dense projection,
attention/convolution fusion and buffer reuse are separate hypotheses;
GGUF text and LFM2-VL vision/projector layouts need their own proof. Preserve
LFM convolution history across chunks and cache-reset behavior.

### Primary-source shortlist

The linked revisions are research snapshots checked on 2026-10-07, not new
dependency pins or replacement golden references. No library/kernel was built
or run for this note. Preserve upstream license notices if code is later adapted.

| Reference | Hardware, precision/layout and shape constraints | Build/runtime/license and evidence limit |
| --- | --- | --- |
| [DeepGEMM `057ca596`](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/README.md), [API exports](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/__init__.py) | Published CUDA kernels require SM90/SM100, excluding the recorded [RTX 4090's Ada SM89](https://developer.nvidia.com/cuda/gpus). FP8/FP4/BF16 APIs have distinct operand, scale, layout and alignment requirements. M-grouped contiguous prefill fixes N/K and aligns expert segments; masked decode is separate. SM90 FP32 versus SM100 packed UE8M0 scales are not Candle's U8 blocks/scales. cuBLASLt exports and legacy A100 Triton imports do not qualify Ada MXFP4. | [MIT](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/LICENSE). Python/PyTorch extension and runtime JIT; published minima: Python 3.8, PyTorch 2.3, C++20, CUDA 12.9, CUTLASS 4.0. Mega MoE additionally uses PyTorch 2.9 symmetric memory, multiple processes and NVLink dispatch/combine. Reference ideas only for this single-4090 lane; native Windows/Candle integration is unproven. |
| [CUTLASS 4.8.0 `0b55a2f6`](https://github.com/NVIDIA/cutlass/blob/0b55a2f691d69981583568fd9eb69687b1f0de8a/README.md), [functionality](https://github.com/NVIDIA/cutlass/blob/0b55a2f691d69981583568fd9eb69687b1f0de8a/media/docs/cpp/functionality.md) | SM80+ dense F16/BF16 and grouped-GEMM building blocks are candidates for Ada. Match each kernel's M/N/K, batch, layout, strides, alignment and tile limits. Dense or INT4 support does not establish packed MXFP4 support; changing Candle F32 activations requires parity. | [BSD-3-Clause](https://github.com/NVIDIA/cutlass/blob/0b55a2f691d69981583568fd9eb69687b1f0de8a/LICENSE.txt). C++17/CUDA templates need Candle launch integration. The README retains a C++ 3.x Windows-build warning despite newer CuTe DSL Windows support; prove the selected native MSVC/kernel slice. No Ada MXFP4 qualification or decode gain follows from library support. |
| [llama.cpp `88dcc460` MMVQ](https://github.com/ggml-org/llama.cpp/blob/88dcc460d628698bb8305b98c200c34f1edfdc04/ggml/src/ggml-cuda/mmvq.cu), [packed dot](https://github.com/ggml-org/llama.cpp/blob/88dcc460d628698bb8305b98c200c34f1edfdc04/ggml/src/ggml-cuda/vecdotq.cuh), [MMQ](https://github.com/ggml-org/llama.cpp/blob/88dcc460d628698bb8305b98c200c34f1edfdc04/ggml/src/ggml-cuda/mmq.cu) | Ada-aware MMVQ dispatch and MXFP4/Q8_1 dot techniques are relevant decode references; MMQ has separate batching criteria. Native FP4 instructions are Blackwell-gated. Dispatch depends on type, batch, expert count and architecture. GGML nibble/block layouts and quantized Q8_1 activations differ from Candle's adjacent-nibble/F32 executor. | [MIT](https://github.com/ggml-org/llama.cpp/blob/88dcc460d628698bb8305b98c200c34f1edfdc04/LICENSE); upstream documents [Windows CMake/MSVC and CUDA builds](https://github.com/ggml-org/llama.cpp/blob/88dcc460d628698bb8305b98c200c34f1edfdc04/docs/build.md). Study bounded techniques before adopting a runtime. Conversion and activation-quantization parity, memory and overhead need proof; source support supplies no Candle speedup. |
| [Triton `fa2e589b`](https://github.com/triton-lang/triton/blob/fa2e589b55706d0de73d25641840e134f20a2c65/README.md) | Upstream lists Linux and NVIDIA CC8.0+, making Ada a hardware candidate. Custom reduction/fusion kernels still need exact dtype, block/scale packing, strides, M/N/K and batch support. | [MIT](https://github.com/triton-lang/triton/blob/fa2e589b55706d0de73d25641840e134f20a2c65/LICENSE). Python/LLVM/JIT toolchain; at most a future Linux/WSL research path with compilation/cache costs included. No native Windows or Candle MXFP4 qualification. |

### Admission for a future optional path

1. Profile the existing admitted 8/512/2048-prompt plus 32-generated cases within
   the observed-safe 2,080-token envelope. Select one measured bottleneck and
   one optimization; report prefill throughput and low-batch decode latency
   separately, with GPT-OSS MoE and dense LFM2 evaluated independently.
2. Require exact device capability, driver/toolchain, compiled kernel, precision,
   blocks/scales/layout/strides, shape/batch and context support, plus explicit
   opt-in initially. Detection is not proof. Unsupported or unqualified cases
   retain the existing default fallback before mutable execution.
3. Bind independent component/router/expert/attention/logit, cache/reset and
   cancellation parity to the exact kernel/library/Candle/model/tokenizer
   versions. Keep existing tolerances. Require actual latency/throughput and
   physical peak-memory evidence on the same inputs, including conversion,
   packing, quantization, JIT, launches and allocations.
4. Existing physical VRAM/proof budgets, quiet-host checks, cancellation and
   process/lease ownership remain authoritative; this note adds no disk cap.
   On runtime failure preserve the failure/uncertain state and cache ownership;
   never silently retry another backend after cache mutation. Research does not
   enter the MVP release or open a product qualification gate.


## Exact next task

No GPT implementation task is selected during MVP closeout. Preserve source-
bound proof and disabled consumer profiles. The maintained-model, physical
16k/32k, speed/memory and coding/route contracts remain incomplete in the
[deferred backlog](../lfm2-vl/TODO.md#deferred-outside-the-mvp-release), with their
original numerical bounds and fresh-allowance requirements. The split-MMProj
carryover is withdrawn; GPT-OSS core remains text-only.

---
AI-edited: 2026-10-14 | agent=Codex/root | model=unknown | effort=unknown | task=overlay-mvp-release | change=parked research without relabeling source-bound qualification
