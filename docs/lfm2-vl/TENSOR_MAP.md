# LFM2.5-VL Tensor Map

## Evidence Boundary

This map is locked to the official model revisions in `SOURCES.md`. Shapes and names were confirmed from bounded safetensors and GGUF headers using HTTP Range requests; no retained evidence includes tensor payload bytes.

| Model | Header ranges read | Header bytes | Tensor records | Payload bytes read |
| --- | --- | ---: | ---: | ---: |
| 450M at `fc6221ca597f3315e4f82fc2df606783267b34ba` | `0-7`, `8-46871` | 46,864 | 349 | 0 |
| 1.6B at `919fde3d022e3f90a4716006f993938ee8c2eb97` | `0-7`, `8-82407` | 82,400 | 589 | 0 |
| 450M F16 MMProj GGUF at `166cd80bbe157dc86d65f964eb8cc6a2cede62ca` | `0-12735` | 12,736 | 201 | 0 |
| 450M Q8_0 MMProj GGUF at `166cd80bbe157dc86d65f964eb8cc6a2cede62ca` | `0-12735` | 12,736 | 201 | 0 |

All inspected safetensors production tensors are BF16. The official MMProj headers contain F16/F32 or Q8_0/F32 tensors as recorded below. Shape notation follows Candle logical order: linear weights are `[out, in]`; embeddings are `[rows, hidden]`; the depthwise short-convolution kernel is `[channels, 1, kernel]`.

## Bounded Trace Stage Contract

The opt-in native `--trace-output` lane and the pinned Python production trace use the same external bundle names for the first parity checkpoint. Both are CPU F32, retain no weights, and deliberately require one non-tiled crop so projector stage shapes are deterministic. Inputs are `input.pixel_values`, `input.pixel_attention_mask`, `input.spatial_shapes`, `input.input_ids`, `input.projector_crop_ranges`, `input.image_rgb_u8`, and `input.decode_token_ids`. Vision stages are `stage.vision.patch_embedding`, `stage.vision.resized_position_embedding`, `stage.vision.embeddings_with_resized_position`, `stage.vision.encoder_layer.{i}`, `stage.vision.post_layernorm`, and `stage.vision.last_hidden_state`. Projector stages are `stage.projector.input`, `pixel_unshuffle`, optional `layer_norm`, `linear_1`, `activation`, `linear_2`, and `output`; language stages are `stage.text.embeddings`, `stage.multimodal.merged_embeddings`, `stage.language.hidden_states`, `stage.language.prefill_logits`, and `stage.language.decode_logits`. The decode input/logit tensors contain the fixed number of cached steps from the trace request, starting with the prefill-selected token, independent of the user-facing generated-token report.

## Normalized Dimensions

| Symbol or property | 450M | 1.6B | Evidence |
| --- | ---: | ---: | --- |
| Text hidden `H` | 1,024 | 2,048 | Config and header |
| Vocabulary `N` | 65,536 | 65,536 | Config and header |
| Text layers | 16 | 16 | Config and header namespace |
| Attention heads / KV heads | 16 / 8 | 32 / 8 | Config |
| Head dimension | 64 | 64 | Inferred as `H / attention_heads`, confirmed by K/V shapes |
| Raw FFN width | 6,656 | 12,288 | Config |
| Effective FFN `F` | 4,608 | 8,192 | Transformers normalization formula and header |
| Short-convolution cache/kernel `C` | 3 | 3 | Config and header |
| Vision hidden `V` | 768 | 1,152 | Config and header |
| Vision FFN `VF` | 3,072 | 4,304 | Config and header |
| Vision layers / heads | 12 / 12 | 27 / 16 | Config |
| Channels / patch size | 3 / 16 | 3 / 16 | Config |
| Packed patch width | 768 | 768 | `3 × 16 × 16`, confirmed by header |
| Learned base positions | 256 | 256 | Config and header |
| Downsample factor `D` | 2 | 2 | Config |
| Projector input `V × D²` | 3,072 | 4,608 | Inferred and header-confirmed |
| Projector hidden `P` | 2,048 | 2,048 | Config and header |
| Image placeholder ID | 396 | 396 | Config |

Full-attention layers are `[2, 5, 8, 10, 12, 14]`; the other ten layers use short convolution. Both checkpoints omit `lm_head.weight` and use the embedding matrix as the tied output projection.

## Text Tower

For native VL safetensors, Candle must point the LFM2 constructor directly at `model.language_model`; relative target names therefore remain `embed_tokens`, `layers`, and `embedding_norm`. Standalone LFM2 keeps the existing `model` root. GGUF names are from llama.cpp revision `74ce15741b420b8d6f12e720398458b576c51c2c`.

| Tensor role | Hugging Face native name | Candle target | llama.cpp GGUF name | 450M shape | 1.6B shape | Required transform |
| --- | --- | --- | --- | --- | --- | --- |
| Token embedding | `model.language_model.embed_tokens.weight` | Same native path; relative `embed_tokens.weight` | `token_embd.weight` | `[65536, 1024]` | `[65536, 2048]` | Native: none. GGUF: use the quantized embedding loader; no semantic transpose. |
| Final RMSNorm | `model.language_model.embedding_norm.weight` | Same native path | `output_norm.weight` | `[1024]` | `[2048]` | None |
| Tied output | `lm_head.weight` is absent | Reuse token embedding when tied | `output.weight` may be absent/tied | Logical `[65536, 1024]` | Logical `[65536, 2048]` | Do not require a separate tensor when tied. |
| Operator norm | `model.language_model.layers.{i}.operator_norm.weight` | Same native path | `blk.{i}.attn_norm.weight` | `[1024]` | `[2048]` | None |
| FFN norm | `model.language_model.layers.{i}.ffn_norm.weight` | Same native path | `blk.{i}.ffn_norm.weight` | `[1024]` | `[2048]` | None |
| Attention Q | `model.language_model.layers.{i}.self_attn.q_proj.weight` | Same native path | `blk.{i}.attn_q.weight` | `[1024, 1024]` | `[2048, 2048]` | Native: none; preserve `[out, in]`. |
| Attention K | `model.language_model.layers.{i}.self_attn.k_proj.weight` | Same native path | `blk.{i}.attn_k.weight` | `[512, 1024]` | `[512, 2048]` | Native: none; preserve GQA output width 512. |
| Attention V | `model.language_model.layers.{i}.self_attn.v_proj.weight` | Same native path | `blk.{i}.attn_v.weight` | `[512, 1024]` | `[512, 2048]` | Native: none; preserve GQA output width 512. |
| Attention output | `model.language_model.layers.{i}.self_attn.out_proj.weight` | Same native path | `blk.{i}.attn_output.weight` | `[1024, 1024]` | `[2048, 2048]` | Native: none. |
| FFN gate | `model.language_model.layers.{i}.feed_forward.w1.weight` | Same native path | `blk.{i}.ffn_gate.weight` | `[4608, 1024]` | `[8192, 2048]` | Native: none. The effective width must be normalized before loading. |
| FFN up | `model.language_model.layers.{i}.feed_forward.w3.weight` | Same native path | `blk.{i}.ffn_up.weight` | `[4608, 1024]` | `[8192, 2048]` | Native: none. |
| FFN down | `model.language_model.layers.{i}.feed_forward.w2.weight` | Same native path | `blk.{i}.ffn_down.weight` | `[1024, 4608]` | `[2048, 8192]` | Native: none. |
| Short-conv input | `model.language_model.layers.{i}.conv.in_proj.weight` | Same native path | `blk.{i}.shortconv.in_proj.weight` | `[3072, 1024]` | `[6144, 2048]` | Native: none; output is `3H`. |
| Short-conv depthwise kernel | `model.language_model.layers.{i}.conv.conv.weight` | Same native path | `blk.{i}.shortconv.conv.weight` | `[1024, 1, 3]` | `[2048, 1, 3]` | Preserve depthwise `[H, 1, C]`; no bias in these checkpoints. |
| Short-conv output | `model.language_model.layers.{i}.conv.out_proj.weight` | Same native path | `blk.{i}.shortconv.out_proj.weight` | `[1024, 1024]` | `[2048, 2048]` | Native: none. |

## Vision Tower and Projector

The production files use the nested vision namespace `model.vision_tower.vision_model`. This is header-confirmed; the shorter `model.vision_tower.embeddings` form is absent.

| Tensor role | Hugging Face native name | Candle target | llama.cpp GGUF name | 450M shape | 1.6B shape | Required transform |
| --- | --- | --- | --- | --- | --- | --- |
| Packed patch weight | `model.vision_tower.vision_model.embeddings.patch_embedding.weight` | Same native path | `v.patch_embd.weight` | `[768, 768]` | `[1152, 768]` | Native: none. GGUF conversion stores `[V, 3, 16, 16]`; reverse with `permute(0,2,3,1)`, contiguous, then reshape `[V,768]`. |
| Packed patch bias | `...patch_embedding.bias` | Same native path | `v.patch_embd.bias` | `[768]` | `[1152]` | None |
| Learned positions | `model.vision_tower.vision_model.embeddings.position_embedding.weight` | Same native path | `v.position_embd.weight` | `[256, 768]` | `[256, 1152]` | No weight transpose; runtime reshape to `16 × 16 × V` only for interpolation. |
| Vision LayerNorm 1 | `model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.{weight,bias}` | Same native path | `v.blk.{i}.ln1.{weight,bias}` | `[768]` | `[1152]` | None |
| Vision Q/K/V | `...encoder.layers.{i}.self_attn.{q_proj,k_proj,v_proj}.{weight,bias}` | Same native path | `v.blk.{i}.{attn_q,attn_k,attn_v}.{weight,bias}` | Weights `[768,768]`; bias `[768]` | Weights `[1152,1152]`; bias `[1152]` | Native: none. |
| Vision attention output | `...encoder.layers.{i}.self_attn.out_proj.{weight,bias}` | Same native path | `v.blk.{i}.attn_out.{weight,bias}` | Weight `[768,768]`; bias `[768]` | Weight `[1152,1152]`; bias `[1152]` | Native: none. |
| Vision LayerNorm 2 | `...encoder.layers.{i}.layer_norm2.{weight,bias}` | Same native path | `v.blk.{i}.ln2.{weight,bias}` | `[768]` | `[1152]` | None |
| Vision MLP up | `...encoder.layers.{i}.mlp.fc1.{weight,bias}` | Same native path | `v.blk.{i}.ffn_up.{weight,bias}` | Weight `[3072,768]`; bias `[3072]` | Weight `[4304,1152]`; bias `[4304]` | Native: none. |
| Vision MLP down | `...encoder.layers.{i}.mlp.fc2.{weight,bias}` | Same native path | `v.blk.{i}.ffn_down.{weight,bias}` | Weight `[768,3072]`; bias `[768]` | Weight `[1152,4304]`; bias `[1152]` | Native: none. |
| Vision post LayerNorm | `model.vision_tower.vision_model.post_layernorm.{weight,bias}` | Same native path | `v.post_ln.{weight,bias}` | `[768]` | `[1152]` | None |
| Optional projector LayerNorm | `model.multi_modal_projector.layer_norm.{weight,bias}` | Same native path when configured | `mm.input_norm.{weight,bias}` | Absent | Absent | Do not synthesize; both configs set `projector_use_layernorm=false`. |
| Projector linear 1 | `model.multi_modal_projector.linear_1.{weight,bias}` | Same native path | `mm.1.{weight,bias}` | Weight `[2048,3072]`; bias `[2048]` | Weight `[2048,4608]`; bias `[2048]` | Apply factor-2 pixel unshuffle before the linear layer; native weight needs no transpose. |
| Projector linear 2 | `model.multi_modal_projector.linear_2.{weight,bias}` | Same native path | `mm.2.{weight,bias}` | Weight `[1024,2048]`; bias `[1024]` | Weight `[2048,2048]`; bias `[2048]` | Native: none. Output width must equal text hidden `H`. |

## Official 450M MMProj GGUF Inventory

Both official headers contain 201 tensors: 9 fixed tensors plus 16 tensors for each of 12 vision blocks. The fixed set is `v.patch_embd.{weight,bias}`, `v.position_embd.weight`, `v.post_ln.{weight,bias}`, and `mm.{1,2}.{weight,bias}`. `mm.input_norm` is absent, matching `projector_use_layernorm=false`.

The F16 file contains 75 F16 and 126 F32 tensors. The Q8_0 file contains 74 Q8_0 and 127 F32 tensors. Both have tensor-name-set SHA-256 `45e3f6cf0b51dc9f5e458b8af3375d368cc59daff70b79e2938c7490a94df828`, header end 12,708, alignment 32, and tensor-data offset 12,736.

The 450M native-Q8 execution inventory is dimension-derived but resolves to exactly 74 matrices: 12 blocks × (Q/K/V/out + MLP up/down) = 72 vision linears, plus `mm.1.weight` and `mm.2.weight`. Their biases remain F32. Patch projection, positional embeddings, LayerNorm parameters, and other small tensors remain dense.

Header-confirmed representative Candle shapes are patch `[768,3,16,16]`, position `[256,768]`, `mm.1.weight` `[2048,3072]`, and `mm.2.weight` `[1024,2048]`. Every non-patch matrix already appears in Candle `[out,in]` order.

## Orientation and Loading Rules

1. Native safetensors weights are already in Candle linear order `[out, in]`; do not transpose them.
2. The only source-locked mmproj reshape is llama.cpp's patch conversion: Hugging Face `[V, 16 × 16 × 3]` becomes GGUF `[V, 3, 16, 16]`. The inverse must preserve the original pixel/channel ordering exactly.
3. The pinned official F16 and Q8_0 headers prove that Candle presents all non-patch GGUF matrices in `[out,in]` order. The direct loader applies no additional transpose.
4. LayerNorm weights/biases and embeddings are not transposed.
5. `lm_head.weight` is absent from both official safetensors headers. Native construction must honor tied output embeddings rather than report a missing tensor.
6. The 450M FFN shapes are the mandatory first text-only gate: any constructor that computes width 4,096 instead of 4,608 is wrong even if the 1.6B checkpoint loads.
7. Native Q8 execution retains only eligible Q8_0 rank-2 weights as `QMatMul::QTensor`; their input width must be divisible by the Q8_0 block size. Eligible F32/F16/BF16 weights use `LinearOp::Dense`, so mixed checkpoints remain valid.
8. Explicit native-Q8 loading rejects lower-bit weights and Q8_0 tensors assigned to patch, position, norm, or bias roles. The Phase 6 dense loader remains available as a separate compatibility API.

## Open Mapping Boundaries

- The config-only harness now reports tokenizer-derived image wrapper, row/column, and thumbnail IDs from an explicitly supplied local `tokenizer.json`. It requires at least one grid marker, distinct marker IDs, and IDs within the model vocabulary. P3 owns running it against the acquired pinned 1.6B snapshot and recording the official mapping; runtime code continues to resolve these IDs dynamically rather than hardcoding them.
- Run production-payload numerical parity only under a separately authorized model-download task; header evidence alone does not establish production numerical parity.
- Extend the native operator map for lower-bit vision formats only in a separately scoped follow-up; Phase 7 intentionally stops at Q8_0.

---
AI-edited: 2026-08-11T09:15:59-04:00 | agent=Codex/root | model=gpt-5.6-sol | effort=max | task=docs | change=recorded strict tokenizer marker inspection before P3 load
