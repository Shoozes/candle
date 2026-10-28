# LFM3-Audio source authority

The implementation follows the original
[LiquidAI/LFM3-Audio-2B snapshot](https://huggingface.co/LiquidAI/LFM3-Audio-2B/tree/9d49701c372d6011d900f6abb40c6b296fb706dd),
revision `9d49701c372d6011d900f6abb40c6b296fb706dd`.
Read-only text sources/configs and their SHA-256 manifest are retained under
`artifacts/lfm3-audio/reference-9d49701c/`. They are reference inputs and are
never executed. Original weights and tokenizer are retained once on D:;
their `asset-manifest.json` binds names, sizes, hashes and revision.
The Hub connector supplied metadata but its source-fetch operation was
unavailable, so pinned source and asset acquisition used native public HTTP.

| Source file | Owning behavior |
| --- | --- |
| `configuration_lfm3.py`, `modeling_lfm3.py`, `lfm3_conv.py` | LFM3 dimensions, layer types, full-width frequency progression with zero-tail partial RoPE, gated attention, sandwich normalization, causal convolution/cache |
| `configuration_lfm3_audio.py`, `modeling_lfm3_audio.py` | Audio encoder/projector, shared input codebook table, tied depth heads, bidirectional codebook slots, MaskGIT schedule and EOS semantics |
| `feature_extraction_lfm3_audio.py`, `config.json` | 16 kHz periodic-Hann power spectrum, Slaney mel filters, base-10 floor/affine normalization and right padding |
| `configuration_liquid_codec.py`, `modeling_liquid_codec.py`, `audio_tokenizer/config.json` | Mixed-radix inverse FSQ, codec LFM architecture, nearest-exact frame repetition, predicted magnitude/phase, F32 inverse FFT and normalized overlap-add |
| `processing_lfm3_audio.py`, `generation_config.json`, model card | Audio-specific ChatML/BOS/task templates and finite generation defaults |

The reference implementation files carry Hugging Face's 2026 Apache-2.0
copyright/license notice. Rust model math is implemented with Candle operations;
source notices and the repository's Apache license apply to adapted algorithm
structure. The model weights are separately governed by LFM Open License v1.0;
this local implementation does not redistribute weights or grant distribution
approval. The acquired snapshot retains its original license.

Retained reference notice: "Copyright 2026 The HuggingFace Team. All rights
reserved." The reference implementation is licensed under the Apache License,
Version 2.0; the repository retains that license in `LICENSE-APACHE`.

The only new direct production dependency is `rustfft = 6.4.1`, already pinned
in the existing lock as a transitive dependency. It supplies native F32 forward
and inverse transforms for the released non-power-of-two FFT lengths (400 and
1280). No package versions are upgraded and no Python component is introduced.

The pinned audio model selects Transformers' `TopKLogitsWarper`. The cutoff
tie contract was independently checked against
[Transformers v4.57.1 source](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/generation/logits_process.py),
commit `8cb5963cc22174954e7dca2c0a3320b7dc2f4edc`: filtering removes scores
strictly below the kth threshold, preserving tied candidates. The installed
5.15.0.dev0 source agrees (file SHA-256
`c5b5d5666576b19e19a09b99b55806ea009dd8e999892f7a8d2d7876dbfc2002`).
This is read-only sampling-contract evidence, not a reference-model execution
or a change to the pinned model's package authority. No Python was executed.

The official ONNX snapshot `df2b3ed2ea10824030b3ff29ee46705754e72b99`,
Space `538b908a098c4796dd08d8e807a6e53ff80c99f4` and SymbioVoice's
llama.cpp `8ca4060b7e30e76baaa5670ee1ec1b760b7f8552` trial remain separate
references. Neither their output nor their runtime proves Candle compatibility.

---
AI-edited: 2026-10-28 | agent=Codex/root | task=lfm3-review-cleanup | change=record pinned top-k cutoff semantics without Python execution
