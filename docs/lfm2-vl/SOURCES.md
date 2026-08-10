# LFM2.5-VL Sources

## Lock Boundary

The source lock was taken at `2026-08-10T02:56:01Z`. Every moving branch or model `main` reference below is resolved to an immutable commit. Production tensor payloads and GGUF files were not downloaded.

The two official safetensors files were inspected only through the bounded header ranges documented in `TENSOR_MAP.md`. This exposed tensor names, dtypes, shapes, and byte offsets without reading tensor payload bytes.

The machine-readable inventory is `tools/lfm2_vl/reference-lock.json`. It is the authority for exact paths and URLs; this document records the reasoning and adaptation boundary.

## Authority Order

| Rank | Authority | Use |
| --- | --- | --- |
| 1 | Pinned Transformers source plus pinned LiquidAI configs, processor, tokenizer, and safetensors headers | Numerical behavior, checkpoint dimensions, native names, prompt/image processing, and fixture generation |
| 2 | Pinned mistral.rs source | Candle-based Rust donor for narrowly ported model and processor logic |
| 3 | Pinned llama.cpp source and merged fix history | GGUF names/metadata, mmproj conversion, and independent parity cases |
| 4 | Pinned MLX-VLM and Transformers.js source | Independent shape, crop-order, processor, and browser-runtime cross-checks |
| 5 | Candle `0.11.0` source | Integration patterns and the implementation baseline, not the LFM2.5-VL numerical oracle |

When authorities disagree, the pinned official checkpoint files and Transformers behavior win. Secondary sources remain useful regression witnesses but cannot override the official config or processor.

## Hugging Face Transformers

- Repository: `huggingface/transformers`
- Revision: `fd12552d770f745fdbe41031ff4daa688f5ed57e`
- Authority: primary numerical oracle
- License: Apache-2.0
- Adaptation: reference-only by project policy; generate fixtures from it, but do not port Python implementation text into Candle

| Area | Pinned path | Purpose |
| --- | --- | --- |
| LFM2 | [`configuration_lfm2.py`](https://github.com/huggingface/transformers/blob/fd12552d770f745fdbe41031ff4daa688f5ed57e/src/transformers/models/lfm2/configuration_lfm2.py) | Config defaults and normalization fields |
| LFM2 | [`modular_lfm2.py`](https://github.com/huggingface/transformers/blob/fd12552d770f745fdbe41031ff4daa688f5ed57e/src/transformers/models/lfm2/modular_lfm2.py) | Authored modular model definition |
| LFM2 | [`modeling_lfm2.py`](https://github.com/huggingface/transformers/blob/fd12552d770f745fdbe41031ff4daa688f5ed57e/src/transformers/models/lfm2/modeling_lfm2.py) | Generated runtime graph and native tensor attributes |
| LFM2-VL | [`configuration_lfm2_vl.py`](https://github.com/huggingface/transformers/blob/fd12552d770f745fdbe41031ff4daa688f5ed57e/src/transformers/models/lfm2_vl/configuration_lfm2_vl.py) | Composite config and checkpoint defaults |
| LFM2-VL | [`modular_lfm2_vl.py`](https://github.com/huggingface/transformers/blob/fd12552d770f745fdbe41031ff4daa688f5ed57e/src/transformers/models/lfm2_vl/modular_lfm2_vl.py) | Projector, pixel-unshuffle, feature selection, and placeholder replacement |
| LFM2-VL | [`modeling_lfm2_vl.py`](https://github.com/huggingface/transformers/blob/fd12552d770f745fdbe41031ff4daa688f5ed57e/src/transformers/models/lfm2_vl/modeling_lfm2_vl.py) | Generated runtime model |
| LFM2-VL | [`processing_lfm2_vl.py`](https://github.com/huggingface/transformers/blob/fd12552d770f745fdbe41031ff4daa688f5ed57e/src/transformers/models/lfm2_vl/processing_lfm2_vl.py) | Prompt expansion and processor composition |
| LFM2-VL | [`image_processing_lfm2_vl.py`](https://github.com/huggingface/transformers/blob/fd12552d770f745fdbe41031ff4daa688f5ed57e/src/transformers/models/lfm2_vl/image_processing_lfm2_vl.py) | Slow image-processing oracle |
| LFM2-VL | [`image_processing_lfm2_vl_fast.py`](https://github.com/huggingface/transformers/blob/fd12552d770f745fdbe41031ff4daa688f5ed57e/src/transformers/models/lfm2_vl/image_processing_lfm2_vl_fast.py) | Checkpoint-selected fast image processor |
| SigLIP2 | [`configuration_siglip2.py`](https://github.com/huggingface/transformers/blob/fd12552d770f745fdbe41031ff4daa688f5ed57e/src/transformers/models/siglip2/configuration_siglip2.py) | Vision config |
| SigLIP2 | [`modular_siglip2.py`](https://github.com/huggingface/transformers/blob/fd12552d770f745fdbe41031ff4daa688f5ed57e/src/transformers/models/siglip2/modular_siglip2.py) | Authored SigLIP2 implementation |
| SigLIP2 | [`modeling_siglip2.py`](https://github.com/huggingface/transformers/blob/fd12552d770f745fdbe41031ff4daa688f5ed57e/src/transformers/models/siglip2/modeling_siglip2.py) | Packed patch embedding, positional interpolation, masks, encoder, and post-norm |

Pinned license: [`LICENSE`](https://github.com/huggingface/transformers/blob/fd12552d770f745fdbe41031ff4daa688f5ed57e/LICENSE).

## LiquidAI Checkpoints

Both model repositories identify their artifacts as LFM Open License v1.0. They are official config/tokenizer/processor authorities and reference inputs, not source-code donors. Full license handling is recorded in `LICENSE_NOTES.md`.

| Model | Immutable revision | Present small files | Absent expected file |
| --- | --- | --- | --- |
| `LiquidAI/LFM2.5-VL-450M` | [`fc6221ca597f3315e4f82fc2df606783267b34ba`](https://huggingface.co/LiquidAI/LFM2.5-VL-450M/tree/fc6221ca597f3315e4f82fc2df606783267b34ba) | `config.json`, `processor_config.json`, `tokenizer_config.json`, `tokenizer.json`, `chat_template.jinja`, `generation_config.json`, `LICENSE` | `special_tokens_map.json` |
| `LiquidAI/LFM2.5-VL-1.6B` | [`919fde3d022e3f90a4716006f993938ee8c2eb97`](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B/tree/919fde3d022e3f90a4716006f993938ee8c2eb97) | `config.json`, `processor_config.json`, `tokenizer_config.json`, `tokenizer.json`, `chat_template.jinja`, `generation_config.json`, `LICENSE` | `special_tokens_map.json` |

There is no separate tokenizer model or vocabulary file in either pinned tree. Direct immutable URLs for every present file are in `reference-lock.json`.

## mistral.rs

- Repository: `EricLBuehler/mistral.rs`
- Revision: `8010b6a0578e416120b590ed72fd46ed5f24ee85`
- Authority: primary Rust donor, subordinate to Transformers numerics
- License: MIT, copyright Eric Buehler
- Adaptation: narrow direct adaptation is permitted with explicit file/commit provenance and retained MIT notice; do not import mistral.rs pipeline, loader, cache, SDPA, device-map, or quantization abstractions

| Pinned path | Purpose |
| --- | --- |
| [`mistralrs-core/src/models/lfm2.rs`](https://github.com/EricLBuehler/mistral.rs/blob/8010b6a0578e416120b590ed72fd46ed5f24ee85/mistralrs-core/src/models/lfm2.rs) | Correct LFM2 FFN normalization and embedding-driven text path |
| [`mistralrs-core/src/vision_models/lfm2_vl/config.rs`](https://github.com/EricLBuehler/mistral.rs/blob/8010b6a0578e416120b590ed72fd46ed5f24ee85/mistralrs-core/src/vision_models/lfm2_vl/config.rs) | Rust config normalization |
| [`mistralrs-core/src/vision_models/lfm2_vl/mod.rs`](https://github.com/EricLBuehler/mistral.rs/blob/8010b6a0578e416120b590ed72fd46ed5f24ee85/mistralrs-core/src/vision_models/lfm2_vl/mod.rs) | Projector, pixel unshuffle, and multimodal composition |
| [`mistralrs-core/src/vision_models/lfm2_vl/vision.rs`](https://github.com/EricLBuehler/mistral.rs/blob/8010b6a0578e416120b590ed72fd46ed5f24ee85/mistralrs-core/src/vision_models/lfm2_vl/vision.rs) | Candle-based SigLIP2 NaFlex vision implementation |
| [`mistralrs-core/src/vision_models/lfm2_vl/inputs_processor.rs`](https://github.com/EricLBuehler/mistral.rs/blob/8010b6a0578e416120b590ed72fd46ed5f24ee85/mistralrs-core/src/vision_models/lfm2_vl/inputs_processor.rs) | Dynamic tiling, prompt markers, and packed processor flow |

Pinned license: [`LICENSE`](https://github.com/EricLBuehler/mistral.rs/blob/8010b6a0578e416120b590ed72fd46ed5f24ee85/LICENSE).

## llama.cpp

- Repository: `ggml-org/llama.cpp`
- Current revision: `74ce15741b420b8d6f12e720398458b576c51c2c`
- Authority: GGUF namespace/conversion authority and independent parity witness
- License: MIT, copyright the ggml authors
- Adaptation: reference-only unless a future file explicitly records a narrow MIT-derived port; never substitute llama.cpp behavior for a conflicting official processor result

| Pinned path | Purpose |
| --- | --- |
| [`conversion/lfm2.py`](https://github.com/ggml-org/llama.cpp/blob/74ce15741b420b8d6f12e720398458b576c51c2c/conversion/lfm2.py) | LFM2 text and LFM2-VL mmproj conversion, scale metadata, and patch-weight reshape |
| [`gguf-py/gguf/constants.py`](https://github.com/ggml-org/llama.cpp/blob/74ce15741b420b8d6f12e720398458b576c51c2c/gguf-py/gguf/constants.py) | GGUF metadata and tensor names |
| [`gguf-py/gguf/tensor_mapping.py`](https://github.com/ggml-org/llama.cpp/blob/74ce15741b420b8d6f12e720398458b576c51c2c/gguf-py/gguf/tensor_mapping.py) | Hugging Face-to-GGUF tensor aliases |
| [`src/models/lfm2.cpp`](https://github.com/ggml-org/llama.cpp/blob/74ce15741b420b8d6f12e720398458b576c51c2c/src/models/lfm2.cpp) | Independent LFM2 text runtime |
| [`tools/mtmd/clip.cpp`](https://github.com/ggml-org/llama.cpp/blob/74ce15741b420b8d6f12e720398458b576c51c2c/tools/mtmd/clip.cpp) | mmproj metadata loading and shared graph/preprocessor plumbing |
| [`tools/mtmd/clip-model.h`](https://github.com/ggml-org/llama.cpp/blob/74ce15741b420b8d6f12e720398458b576c51c2c/tools/mtmd/clip-model.h) | Projector and preprocessing parameter structures |
| [`tools/mtmd/models/siglip.cpp`](https://github.com/ggml-org/llama.cpp/blob/74ce15741b420b8d6f12e720398458b576c51c2c/tools/mtmd/models/siglip.cpp) | SigLIP/LFM2 position resize, pixel unshuffle, and projector graph |
| [`tools/mtmd/mtmd.cpp`](https://github.com/ggml-org/llama.cpp/blob/74ce15741b420b8d6f12e720398458b576c51c2c/tools/mtmd/mtmd.cpp) | LFM2 media-marker and image-position token handling |
| [`tools/mtmd/mtmd-image.cpp`](https://github.com/ggml-org/llama.cpp/blob/74ce15741b420b8d6f12e720398458b576c51c2c/tools/mtmd/mtmd-image.cpp) | Current LFM2 tiling and thumbnail preprocessing |

Pinned license: [`LICENSE`](https://github.com/ggml-org/llama.cpp/blob/74ce15741b420b8d6f12e720398458b576c51c2c/LICENSE).

Historical behavior locks:

| Role | Pull request | Immutable merged revision | Status at lock |
| --- | --- | --- | --- |
| Initial LFM2-VL support | [#15347](https://github.com/ggml-org/llama.cpp/pull/15347) | `65349f26f2299e06477ec8e85e46243046801358` | Merged |
| Smart resize, media wrappers, and antialiased positional interpolation parity | [#17577](https://github.com/ggml-org/llama.cpp/pull/17577) | `2ba719519d950c5a62c00cdb8b119cc0914c1fa3` | Merged |
| LFM2.5-VL tiling | [#19454](https://github.com/ggml-org/llama.cpp/pull/19454) | `262364e31d1da43596fe84244fba44e94a0de64e` | Merged |
| Read tiling parameters from GGUF metadata | [#25524](https://github.com/ggml-org/llama.cpp/pull/25524) | Head `e9d636c46df330e0372087d953b249c337898792` | Open and unmerged; not an authority |

The associated parity investigation is [issue #17290](https://github.com/ggml-org/llama.cpp/issues/17290). It is retained as a regression narrative, not as executable truth.

## Secondary Implementations

### MLX-VLM

- Repository: `Blaizzy/mlx-vlm`
- Revision: `ffd7aeff0bd213c31534a969e0003d49451eef39`
- Authority: secondary implementation cross-check
- License: MIT, copyright Prince Canuma
- Adaptation: reference-only
- Pinned paths: [`config.py`](https://github.com/Blaizzy/mlx-vlm/blob/ffd7aeff0bd213c31534a969e0003d49451eef39/mlx_vlm/models/lfm2_vl/config.py), [`language.py`](https://github.com/Blaizzy/mlx-vlm/blob/ffd7aeff0bd213c31534a969e0003d49451eef39/mlx_vlm/models/lfm2_vl/language.py), [`lfm2_vl.py`](https://github.com/Blaizzy/mlx-vlm/blob/ffd7aeff0bd213c31534a969e0003d49451eef39/mlx_vlm/models/lfm2_vl/lfm2_vl.py), [`processing_lfm2_vl.py`](https://github.com/Blaizzy/mlx-vlm/blob/ffd7aeff0bd213c31534a969e0003d49451eef39/mlx_vlm/models/lfm2_vl/processing_lfm2_vl.py), and [`vision.py`](https://github.com/Blaizzy/mlx-vlm/blob/ffd7aeff0bd213c31534a969e0003d49451eef39/mlx_vlm/models/lfm2_vl/vision.py)
- Relevant merged fixes: [#1162](https://github.com/Blaizzy/mlx-vlm/pull/1162) at `8113973482620b67f8b553196fa26ac18a07dea8`; [#1190](https://github.com/Blaizzy/mlx-vlm/pull/1190) at `3d0db8db79ba7fd19726ac379dda475bcbccfbe3`

### Transformers.js

- Repository: `huggingface/transformers.js`
- Revision: `353007be131c2e44d16d46ba49b9a56f2955dfd8`
- Authority: secondary browser-runtime cross-check
- License: Apache-2.0
- Adaptation: reference-only
- Pinned paths: [`modeling_lfm2.js`](https://github.com/huggingface/transformers.js/blob/353007be131c2e44d16d46ba49b9a56f2955dfd8/packages/transformers/src/models/lfm2/modeling_lfm2.js), [`modeling_lfm2_vl.js`](https://github.com/huggingface/transformers.js/blob/353007be131c2e44d16d46ba49b9a56f2955dfd8/packages/transformers/src/models/lfm2_vl/modeling_lfm2_vl.js), [`processing_lfm2_vl.js`](https://github.com/huggingface/transformers.js/blob/353007be131c2e44d16d46ba49b9a56f2955dfd8/packages/transformers/src/models/lfm2_vl/processing_lfm2_vl.js), [`image_processing_lfm2_vl.js`](https://github.com/huggingface/transformers.js/blob/353007be131c2e44d16d46ba49b9a56f2955dfd8/packages/transformers/src/models/lfm2_vl/image_processing_lfm2_vl.js), and their pinned tests listed in `reference-lock.json`

## Candle Integration References

All are locked by the Candle baseline commit `31f35b147389700ed2a178ee66a91c3cc25cc80d`:

- `candle-transformers/src/models/lfm2.rs`: dense text baseline and known FFN-width defect.
- `candle-transformers/src/models/quantized_lfm2.rs`: GGUF text loader and cache baseline.
- `candle-transformers/src/models/qwen3_vl/`: image-placeholder replacement and `forward_embeds` pattern.
- `candle-transformers/src/models/paligemma.rs`: simpler multimodal constructor pattern.
- `candle-transformers/src/models/siglip.rs`: fixed-grid vision baseline, not a SigLIP2 NaFlex implementation.

## Locked Conflicts and Gaps

- Both official configs state `max_position_embeddings: 128000`; the model cards advertise a 32,768-token context. The config value is the implementation input; production-context policy remains unresolved.
- `tie_word_embeddings` is absent from both checkpoint JSON files. Transformers defaults it to `true`, declares the tie, and both safetensors headers omit `lm_head.weight`; the effective tied behavior is source-and-header confirmed rather than checkpoint-explicit.
- The checkpoint files override generic Transformers projector defaults: hidden size `2048`, no projector LayerNorm, and EOS ID `7`.
- Only image placeholder ID `396` is explicit in `config.json`. The numeric IDs of image wrapper and tile-marker strings remain a tokenizer-harness output, not a source-lock assumption.
- llama.cpp PR #25524 was open and unmerged at the lock. Official `processor_config.json` values remain authoritative for min/max tiles and tile size.
- Exact physical GGUF matrix orientation remains a direct-GGUF fixture task. Only the converter-defined patch reshape and logical GGUF names are locked now; `TENSOR_MAP.md` marks this boundary explicitly.

---
AI-edited: 2026-08-09T22:56:01-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=source-lock | change=locked immutable implementation, model, license, and parity authorities
