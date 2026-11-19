# LFM2-MoE implementation decisions

The accepted November 10 plan owns this separate capability. D1, audio, frozen
releases and consumer pins keep their original identities.

- Reuse dense LFM2 attention/convolution/cache through opaque crate-private
  mixers. Existing public dense APIs and default behavior remain unchanged.
- Native configuration uses the published dense and expert widths directly.
  The native shard layout is per-expert `w1/w2/w3`, even though the numerical
  reference internally packs `gate_up_proj` and `down_proj`.
- Routing follows pinned Transformers: sigmoid, bias only for selection,
  unbiased gathered weights, optional normalization with `1e-6`, then scaling.
  Only route indices/control data are read on the host; expert computation stays
  on the selected device. There is no softmax substitution or device fallback.
- GGUF inspection validates every required tensor's role, dtype, dimensions,
  row alignment and checked byte extent. Physical EOF is checked before payload
  loading. Expert banks are read as checked individual matrices and remain
  quantized. CPU generic tiled dispatch avoids persistent repacked bank copies.
- The official GGUF uses tied embeddings and `token_embd_norm.weight`. Its
  omitted normalization/scaling controls follow the pinned `lfm2moe` defaults
  (normalized top-K, scale 1). F16/BF16 dense weights are widened to F32 and the
  admission estimate includes that storage; quantized banks are never widened.
- Native `conv_bias=true` and non-default RoPE are refused explicitly; neither
  appears in the selected official configuration. GGUF activations are F32;
  CPU/CUDA are supported, unsupported devices fail without CPU fallback.
- Edge's profile stays explicitly selected and held pending actual production
  qualification. Development uses a local Cargo patch, restoring its original
  lock after proof. Publication precedes normal reviewed pin/lock adoption.
- The official assistant prefix opens `<think>`. Edge parses its continuation
  from that state, refuses incomplete reasoning and unexpected control markers,
  and counts all generated tokens. Unmeasured reasoning/final token splits are
  reported as unmeasured rather than zero. Previous assistant reasoning is
  removed before a later user message, matching the pinned template default.

Native F32 oracle comparisons require `atol=1e-5`, `rtol=1e-4`. Quantized
matmul tests compare against the same dequantized weights with a separately
declared `0.02` absolute bound; they do not assert that Q4/Q8 equal unquantized
weights. Production llama.cpp differential proof remains a separate gate.

## Layer-2 routing comparison

Pinned Transformers `ddb849ab` governs Candle routing. Candle divides the
gathered unbiased weights by `sum + 1e-6`. The pinned llama.cpp `582eab71`
LFM graph enables normalization and divides by `max(sum, 6.103515625e-5)`.
The formulas can produce different F32 results even above the clamp threshold.

The quantized CPU router and the GGUF CUDA F32 route accumulate selected
weights in F64 and cast the raw sum back to the weight dtype before adding
the existing epsilon. This matched all 32 pinned raw sums in the Unicode
replay. Native safetensors keeps its existing weight-sum schedule. The
quantized CUDA router dot shares the CPU
eight-lane FMA and reduction order on aligned F32 matrices. A single
activation row uses the pinned four-accumulator F32 dot. GGUF CUDA F32
sigmoid evaluates the exponential in F64 and rounds it back to F32 before
adding one and taking the reciprocal, matching the retained CPU probabilities.

The layer-2 diagnostic compares logits, sigmoid probabilities, biased selection
scores, ordered integer IDs, unbiased gathered weights and raw sums before the
denominator. An expert-order mismatch retains the two ordered lists, their
scores and the ties at the reference cutoff. When the common inputs and sums
match exactly, the probe classifies a denominator difference only if both
observed values match their respective formulas. It records the first mismatch
and stops, including for a classified reference difference. Normalized weights
and later stages remain in the ordered contract. An unexplained difference
requires operand isolation. Any production compatibility change needs a
separate decision, and the production [qualification gate](QUALIFICATION.md)
still applies.

## Quantized MoE reference-order attention

The private MoE mixer selects pinned F32 attention on CUDA and on x86-64 CPU
with AVX2/FMA for head widths divisible by 32. Scores follow the four-vector Q·K
fold and softmax uses the pinned exponent and F32 group/F64 sum. Prompt
value reduction uses eight FMA lanes per channel. A single query uses the
plain four-accumulator value dot. A one-row Q6_K V projection uses the
plain Q8_K fold instead of the tiled fold. The implementation uses the
model's causal mask and grouped KV heads. Zero-only padded groups are implicit.
The shared attention layer owns RoPE, KV caching and output projection. Its
stock path serves dense models and configurations outside the ordered path.

For gathered experts with Q4_K gate/up and Q4_K or Q6_K down, CUDA uses
the CPU indexed folds. Q4_K uses the full-block Q8_K fold for every row.
Q6_K down selects the plain dot below eight gathered rows and the tiled
dot at and above eight. The ordered SiLU is shared with dense GGUF.

## Q4_K CUDA matrix-matrix activation sum

The Q4_K matrix-matrix dot product applies the packed weight minimum to
the same quantized activation bytes used by its integer product. Both
matrix-matrix paths now store the scale times the sum of those bytes in
the activation metadata. Fast MMQ selects the signed-byte sum DS4 launcher
for Q4_K. Other weight types retain their existing layouts and launchers.

The fast MMQ control reproduced all 227,328 stock CUDA projection values
bitwise on the shared layer-0 RMSNorm operands. The corrected DS4 launch
preserved every activation byte and scale and reduced the maximum projection
gap from 0.03963518 to 0.01320948. CPU and CUDA still use different activation
formats and accumulation schedules. Production token parity is measured separately.

## CUDA ShortConv Q4_K activation format

The paired position-2 trace showed matching embeddings and a layer-0
divergence before the output head. Swapping final-normalized operands
between the heads moved the `1113`/`6986` ranking with the operand.
On common retained layer-0 operands, the pinned 256-value Q8_K schedule
reproduced the CPU Q4_K input projection bitwise. The DS4 Q8_1 schedule
still differed on those operands.

Quantized MoE ShortConv uses Q8_K activations for its two Q4_K projections
on CUDA. It uses the paired subgroup fold for complete groups of four
rows and the full-block fold for the remaining rows independently per
sequence. The call keeps the packed weights and activation scratch on
device. Other projections still use their existing quantized dispatch.

## CUDA dense FFN gate and up

The dense gate and up replay reproduced the existing CUDA DS4 projections
on the retained CPU-normalized rows. Holding the F64 reduction fixed
isolated an activation-format difference. Holding all Q8_K blocks fixed
and choosing the wrong group or tail fold changed output bits.
The Q8_K group/tail schedule matched both pinned CPU projections
bitwise on all 37 rows.

The dense Q4_K gate and up now share the existing ShortConv projection
dispatch. CPU keeps its pinned group/tail fold and CUDA keeps packed
weights and activation scratch on device. Indexed experts retain their
separate fold and dispatch. The dense down path retains its Q6_K schedule.

## CUDA dense FFN down

The shared SwiGLU replay reproduced the existing Q6_K CUDA D4 output
bitwise on all 37 rows. Using Q8_K with the pinned tiled fold matched
all 75,776 CPU values. Holding those Q8_K bytes fixed and using a fused
fold changed 49,681 values.

Dense Q6_K down now uses the shared device-side Q8_K activation packer.
For multiple input rows, each superblock forms an integer dot, applies two
separately rounded F32 scale products, then adds the result in block order.
For a single dense FFN row, pinned CPU dispatches to the plain dot. CPU
and CUDA now match its eight FMA accumulators and final reduction order.
The single-row option belongs to dense FFN down. Attention and indexed
experts keep their separate dispatch.

## CUDA block RMSNorm

Layer 1 in the selected GGUF is ShortConv. Crossing the repaired layer-0
outputs separated inherited prefill drift from the first difference on
shared inputs, the block RMSNorm. Supplying CPU-normalized rows made the
CUDA input projection, cache, and mixer residual match bitwise.

The quantized MoE block operator and FFN RMSNorm on CUDA now use the
pinned CPU order. The kernel rounds each square in F32, sums in F64
in input order, and separately rounds the mean, square root, reciprocal,
and output products. The operation keeps input and weight on the GPU.
The dense model RMSNorm and the attention-head normalization retain
their existing dispatch.

---
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer2-inert | change=record routing authority and first-difference classification
AI-edited: 2026-11-14 | agent=Codex/root | task=lfm2-moe-maintained-reduction | change=record scoped CPU reduction and pending exact replay
AI-edited: 2026-11-17 | agent=Codex/root | task=moe-cuda-shortconv | change=bind the crossed replay and Q4_K activation sum correction
AI-edited: 2026-11-18 | agent=Codex/root | task=moe-unicode-position2 | change=identify the pinned plain dot for one dense FFN row
