# Qwen3.5 Compatibility Decisions

## 2026-09-30: Own a separate, bounded Qwen3.5 overlay

Candle owns a `qwen35` module and keeps the exact Darkidol Ballad 4B reference
under an artifact-specific fixture directory. The model path reads architecture,
dimensions, layer schedule, and tensor layout from checked GGUF metadata. A
caller supplies explicit file, header, tensor, and allocation ceilings; no
format-wide unbounded loader defaults are used. The first loader accepts only
F32 dense roles and F32/Q8_0 matrix roles. Other dtypes, MTP, extra tensors,
missing tensors, invalid ranges, and incompatible dimensions fail admission.

The GGUF converter has already changed recurrent value-head ordering and
parameters. CPU execution must consume that layout as serialized and repeat
Q/K across value heads in tiled order. The first CPU trace is a bounded
artifact-specific milestone; it does not qualify 27B IQ, advertised context,
GPU, application activation, or full-vector numerical parity.

The pinned reference authority is the local llama.cpp source commit and the
hash-pinned Edge fixture listed in `SOURCES.md`. This decision keeps Qwen3.5
evidence independent of LFM2-VL and GPT-OSS.

## 2026-09-30: Scope ggml Q8_0 activation arithmetic to this model

The pinned llama.cpp x86 Q8_0 kernel quantizes activations with `127 / amax`
and nearest-even rounding. Candle's existing compact path uses `1 / (amax / 127)`
and its default rounding rule, while the x86 VNNI repack uses 256-value
activation blocks. Small differences at quantization thresholds changed the
first Qwen3.5 recurrent projection and later top-token scores. A scoped
`QMatMul::forward_ggml_q8_0` call uses ggml's 32-value arithmetic and bypasses
the x86 repack for Q8_0 on that call only. Existing Candle callers retain their
current path. No process-wide CPU setting is required for Qwen3.5.

The pinned GGML RMS kernel accumulates F32 products in a double sum before
forming its F32 scale. Its recurrent kernel uses fused grouped dot products and
fused state updates. The Zen4 reduction combines lanes 0+2 and 1+3 at the
final step. The Qwen3.5 CPU path follows those operations where the fixture is
sensitive. The exact fixture gate remains authoritative.

The pinned GGML RoPE CPU kernel advances each position angle by a single F32
`theta_scale` multiplication per rotary pair and computes sine and cosine
separately. Its Zen4 build contracts the rotation's first multiply and add.
The Qwen3.5 CPU path uses the same order. Recomputing each angle with a
separate power and omitting that fused operation changed attention Q/K by a few
F32 units in last place.

The full-attention score and value products use the pinned GGML F32 grouped
FMA dot and Zen4 reduction order. Attention softmax uses GGML's F32 vector
exponential, 16-lane tree sums, double-precision sum across chunks, then an
F32 reciprocal scale. At prompt token 7, layer 19 was the first attention
block that exposed a value-sum difference; matching that arithmetic made all
32 layer outputs for the token bit-identical to the pinned `ubatch=1` capture.
The exact two-position black-box fixture, not that single component capture,
is the acceptance authority for this milestone.

## 2026-09-30: Keep dynamic Q8_0 scheduling after release measurement

A bounded release benchmark of the exact 4B Q8_0 artifact attributed about
7.417 of 10.09 measured inference seconds to quantized projections. Static
row ranges for the scoped single-row Q8_0 path coincided with 7.774 seconds
of measured projection time and lower throughput. Unrelated game load was
not recorded, so this is insufficient evidence for a causal regression.
Reusing the recurrent delta buffer
removed repeated small allocations but showed no repeatable end-to-end gain.
Both edits were reverted. The existing shared-cursor Q8_0 scheduler and
per-head delta allocation remain until a narrower projection profile supports
a measurable improvement. Benchmark details and CUDA assessment are in
`PERFORMANCE.md`.

## 2026-10-01: Scope CUDA Q8_0 compatibility to this model call

The generic CUDA Q8_1 activation and quantized matvec path remains unchanged
for other Candle callers. A real layer-zero Q8_0 gate matched the CPU output
within 2.98e-7, but the first recurrent QKV differed by 0.001682: two of its
2,560 activation bytes differed between the pinned GGML CPU quantization and
the generic CUDA quantizer. Using GGML nearest-even quantization reduced that
QKV error to 4.77e-6, yet its later recurrent trace still exceeded 0.01
raw-logprob error. The remaining dot accumulation order was material.

`QMatMul::forward_ggml_q8_0` now scopes the CUDA F32 Q8_0 single-row path to
the GGML activation rule and an eight-lane FMA reduction matching Candle's
pinned AVX2 CPU reference. Unsupported CUDA shapes, dtypes, and the existing
forced-dequantization switch fail explicitly while in this scope. The fallback
is compared only through a generic unspecialized call. The CUDA FFI
declaration is shared with the GPT-OSS
overlay, but only the Qwen-scoped parameter and kernel are owned here; GPT-OSS
tests must remain green. The first token's 32 layer outputs became bit
identical to CPU, and the complete exact-artifact fixture passed its original
top-10 and 0.01 raw-logprob gates. This does not imply parity for other
hardware, quantizations, contexts, or Qwen-family members.

The first hybrid model keeps quantized projection weights on CUDA and retains
recurrent FIFO/matrix state, attention KV, and scalar mixing on the host.
Moving those states to CUDA without their operations would add transfers, so
device-state promotion waits for measured device kernels and component parity.
Cancellation is supported between prefill tokens and resets partial state;
mid-kernel cancellation is not claimed. EdgeSymbio and symbio-code require
separate model admission, tokenizer, profile, capacity, and receipt work before
activation.

## 2026-10-01: Preserve arithmetic while changing memory access and CUDA row mapping

The sealed 1,024-token Edge prompt exposed host attention value and recurrent
matrix-vector loops plus scoped Q8_0 CUDA projections as measured latency
costs. Attention values now traverse cached tokens row-major; recurrent
matrix-vector products traverse state key rows. Both retain each output's
original 64 FMA lanes and the same final reduction order. Unit tests compare
their full outputs bit for bit with the original strided dot calculations.
Prefill skips intermediate vocabulary projections because only the final
prompt token's logits are returned; cached decode still emits logits for
every call. Token-boundary cancellation continues to clear partial state.

The scoped CUDA Q8_0 kernel now assigns one thread to each of the eight
independent FMA lanes for a row, then shuffles those lane results into the
same final CPU-order reduction. This keeps the complete 9,216-value proof bit
exact while removing much of the serial work per CUDA thread. It changes only
the Qwen-scoped path; generic Q8_0 CUDA dispatch and other overlays retain
their existing behavior. The exact 49-token reference and the sealed
1,024-token Edge input passed their unchanged 0.01 raw-logprob ceiling.
Measured improvement and remaining application gate ownership are in
`PERFORMANCE.md` and `INTEGRATION.md`.
