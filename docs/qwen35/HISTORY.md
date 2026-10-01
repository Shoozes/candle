# Qwen3.5 Compatibility History

## 2026-09-30: Exact 4B Q8_0/F32 CPU trace and reset gate

Added checked, caller-bounded GGUF admission and config-driven CPU
GatedDeltaNet/full-attention execution. The exact artifact's 426 tensors were
admitted after SHA-256 verification. The first native Windows trace gate
passed 49-token prefill, cached decode `[19, 248046]`, EOS, identical top-10
token sets, and maximum raw-logprob error `0.00005283917236376112` across 20
token IDs against the pinned reference. Reset after unrelated `[10,20,30]`
input replayed Candle's complete first raw-logit vector exactly.

The scoped GGML-compatible Q8_0 activation path leaves other Candle callers
unchanged. A pinned `ubatch=1` diagnostic capture matched all 32 layer
outputs at token 7 bit for bit after correcting GGML's grouped F32 dot,
softmax, value sum, convolution, recurrence, and RoPE arithmetic. Evidence and
resource bounds are recorded in `STATUS.md`. No model bytes or runtime output
are tracked, and no Edge activation or publication was performed.

## 2026-09-30: Release CPU baseline and first optimization evaluation

Added a bounded, hash-checking release benchmark runner and test-only
projection profiling. On the exact 4B Q8_0 artifact with four Candle threads,
the unprofiled three-repeat 16+16-token sample averaged 9.52 prefill and 9.45
cached-decode tokens/s. Profiled Q8_0 projections accounted for 7.417 of
about 10.09 inference seconds. Static scoped Q8_0 scheduling was slower in
the observed runs; a reused recurrent delta buffer did not show a repeatable
gain. Background game load was not recorded, so these timing comparisons are
not isolated causal estimates. Both candidate
source edits were reverted. `PERFORMANCE.md` records the inputs, resource
limits, full evidence, and a read-only RTX 4090 CUDA feasibility pass-down.

## 2026-10-01: Exact 4B CUDA projection and hybrid trace

Implemented an opt-in CUDA-resident projection/host-state hybrid for the
already admitted Q8_0/F32 model. The first GPU projection proof compared both
generic CUDA activation paths to CPU and separated weight H2D, activation
H2D, kernel, and output D2H timing. Whole-model parity initially failed;
layer-zero QKV exposed two activation quantization bytes and the CPU/GPU dot
reduction difference. A scoped nearest-even Q8_0 activation and CPU-order
eight-lane FMA CUDA kernel made the first token's 32 layer outputs bit exact.

The unchanged exact fixture then passed 49-token prefill, cached decode
`[19,248046]`, EOS, top-10 sets, and 0.00005269 maximum raw-logprob error
against the 0.01 limit. Reset and token-boundary cancellation recovery passed;
short-session CUDA free memory was unchanged at sampled points. The CPU gate
passed again on final source. Focused Qwen and GPT-OSS tests, native CPU/CUDA
checks, formatting, diff whitespace, and fork-overlay verification passed.
Accepted and failed evidence labels, limits, and remaining device-state and
consumer integration work are in `PERFORMANCE.md` and `STATUS.md`.

## 2026-10-01: Sealed 1,024-token latency and repeated-session proof

Profiled the exact Edge 1,024-token input against its retained gate and
optimized host attention value accumulation, recurrent state matvec,
intermediate prefill vocabulary work, and the scoped Q8_0 CUDA row mapping.
Three final-source prefill runs took 50.751, 50.630, and 50.621 seconds,
with the same selected tokens/top-10 sets and maximum checked raw-logprob
error 0.0002502. The 9,216-value projection remained bit exact. A separate
reset/cached-decode replay and post-token-512 cancellation replay matched
full raw-logit vectors exactly; sampled CUDA free memory remained unchanged.
The native CPU strict trace and shared GPT-OSS regression passed again.
`PERFORMANCE.md` records the matched stages, exact evidence paths, resource
bounds, and the still-required end-to-end Edge Responses gate.
