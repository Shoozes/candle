# LFM3-Audio source authority

## Standalone text qualification candidate

The selected standalone text checkpoint is
[LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B/tree/1da06bec6a819da10b54c67e5c90972b46d81871)
at `1da06bec6a819da10b54c67e5c90972b46d81871`. Public Hub metadata and
pinned source inspection identify the following inputs:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `model.safetensors` | 3,457,635,448 | `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330` |
| `tokenizer.json` | 17,908,635 | `7e226e34bface23ccf6e495d157bf77186c1c30a7659880e1a8994db528c3701` |
| `config.json` | 1,454 | `8c6ae8b8f6d88c87cfb98cf4b5e34e943d0ee07e24215c4121b3159a46e1399d` |

The released config declares BF16 and tied embeddings. Qualification calls
for F32 execution on both CPU and RTX 4090 from the same native weights.
The model, config-class and convolution Python sources match the retained
audio snapshot's corresponding source files byte for byte. Standalone
weights and tokenizer keep their independent identities.

Two HTTP range reads acquired the 32,376-byte safetensors header, including
its length prefix. All 286 BF16 tensor names and shapes matched the Candle
loader schema and the released Python model instantiated on the meta device.
Tensor offsets were contiguous and reconciled with the published file size.
The F32 parameters total 6,915,206,144 bytes before cache, scratch and loading
overhead. The range reads did not include tensor payloads.
[Header audit](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/standalone-header-audit.json).

[Transformers e7200cd9](https://github.com/huggingface/transformers/tree/e7200cd9a98f0c874edcdf8f7263b7d19530b403)
is a reference environment candidate. Its pinned source contains the
initialization, kernel fallback, accelerate hooks, and recurrent mask APIs
required by the model source. The offline import stopped before reaching the
model code: the local `huggingface-hub` 1.5.0 lacks `ResolvedRevision` and
the pin declares `huggingface-hub>=1.31.0,<3.0`. Local `tokenizers` 0.22.2
also falls below the declared `>=0.23.1,<0.24.0` range. The next reference
gate is a compatible isolated environment, followed by a tiny synthetic
prefill and cached-decode comparison using the released standalone source.
No weight payload was downloaded for this source intake.
[Pinned archive receipt](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/source.json).

PyPI metadata identifies four compatible Windows wheels for the local
CPython 3.13 AMD64 interpreter: `huggingface-hub` 1.31.0, `tokenizers`
0.23.1, `click` 8.4.2 and `hf-xet` 1.5.2. Declared wheel dependencies were
checked against these pins and the existing Python environment. The intake
records filenames, URLs, sizes and SHA-256 digests. The four downloaded
wheels matched those digests and were extracted into a process-local
`PYTHONPATH` directory. The system Python installation was untouched.
[Wheel intake](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/reference-wheel-intake.json).

The standalone source files were fetched at the model revision above and
matched the three previously recorded source hashes. The offline tiny replay
verified those hashes and the BF16 fixture inputs. Its full comparison pairs
95 F32 logits and five top IDs per lane for one-shot and cached prefill with
the retained Candle CPU reports. With the isolated wheels, the pinned
Transformers import and both comparisons passed. The fixture uses
special-token IDs within its 19-token vocabulary. Published source and
checkpoint configuration bytes were preserved. Transformers emitted cache
API deprecation and docstring diagnostics during the successful run.
[Tiny replay](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/replay_tiny.py).
[Replay log](D:/Compact_Evidence/candle/runs/lfm3-reference-e7200cd9/tiny-replay-result.log).

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
