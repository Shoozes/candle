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

---
AI-edited: 2026-11-13 | agent=Codex/root | task=lfm2-moe-layer2-inert | change=record routing authority and first-difference classification
