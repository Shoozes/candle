# Candle 0.11 LFM2.5-VL/MMProj Extension Specification

## Verdict

This is feasible, but the real task is larger than adding a two-layer projector.

For LFM2.5-VL, “mmproj support” must include:

1. SigLIP2 NaFlex vision encoding.
2. Exact image resizing, tiling, patchification, masking, and positional interpolation.
3. Factor-2 pixel unshuffle.
4. The multimodal projector.
5. Prompt placeholder expansion.
6. Image-feature insertion into LFM2 input embeddings.
7. Multimodal prefill followed by ordinary cached text decoding.
8. Native safetensors, hybrid, and GGUF mmproj loading.

The current LFM2.5-VL checkpoints still identify themselves as `Lfm2VlForConditionalGeneration`. Both use factor-2 downsampling, a 2,048-wide projector, image token ID 396, patch size 16, and SigLIP2 vision encoders, but their text and vision dimensions differ substantially. The implementation must be config-driven rather than model-name-driven. ([Hugging Face][1])

The correct development order is:

1. Native unified safetensors.
2. Quantized GGUF text plus native or split safetensors mmproj.
3. Direct llama.cpp-compatible GGUF mmproj.
4. Quantized GGUF mmproj execution without dequantizing its major linear layers.

Starting with direct GGUF support would mix model-math bugs, processor bugs, tensor-name bugs, tensor-layout bugs, and quantization differences into one failure surface.

---

## 1. What Candle 0.11 already has

Candle 0.11 includes:

* Dense LFM2.
* Quantized GGUF LFM2.
* Fixed-grid SigLIP.
* PaLI-Gemma multimodal embedding composition.
* Qwen3-VL embedding replacement and multimodal forwarding.

It does not expose an `lfm2_vl` model module in its 0.11 model registry.

The closest existing Candle-native embedding insertion example is Qwen3-VL. It:

* Embeds text tokens.
* Encodes images.
* Verifies image feature counts against placeholder spans.
* Uses `slice_assign` to replace placeholder embeddings.
* Calls the language model through `forward_embeds`.

That is the internal pattern we should reuse for LFM2.5-VL.

PaLI-Gemma is still useful as a simpler example of exposing text embeddings and forwarding precomputed embeddings, although its image features are concatenated rather than inserted into sentinel positions.

---

## 2. Critical prerequisite: repair Candle’s dense LFM2 configuration

Candle 0.11 currently calculates the dense LFM2 feed-forward width from:

```text
hidden_size * 4 * multiplier
```

rounded to `block_multiple_of`.

That is not the configuration rule used by current LFM2 checkpoints. Candle’s current calculation is visible in `lfm2.rs`.

The correct calculation is:

```text
raw = block_ff_dim, otherwise intermediate_size

if block_auto_adjust_ff_dim:
    raw = floor(2 * raw / 3)
    raw = floor(block_ffn_dim_multiplier * raw)
    raw = ceil_to_multiple(raw, block_multiple_of)
```

### Why the 450M checkpoint must come first

| Checkpoint     | Hidden size | Raw FF width | Correct effective width | Candle 0.11 width |
| -------------- | ----------: | -----------: | ----------------------: | ----------------: |
| LFM2.5-VL-450M |       1,024 |        6,656 |                   4,608 |             4,096 |
| LFM2.5-VL-1.6B |       2,048 |       12,288 |                   8,192 |             8,192 |

The 1.6B checkpoint accidentally hides the bug because `2048 × 4` happens to equal its correct effective width. The 450M checkpoint exposes it immediately and should fail weight loading under the current calculation. The official configs provide the relevant widths and adjustment settings. ([Hugging Face][1])

### Required `Lfm2Config` additions

Candle’s raw LFM2 configuration should parse:

```rust
pub struct Lfm2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,

    pub intermediate_size: Option<usize>,
    pub block_ff_dim: Option<usize>,
    pub block_auto_adjust_ff_dim: bool,
    pub block_ffn_dim_multiplier: f64,
    pub block_multiple_of: usize,

    #[serde(alias = "conv_L_cache")]
    pub conv_l_cache: usize,

    pub conv_bias: bool,
    pub layer_types: Vec<LayerType>,

    pub tie_word_embeddings: Option<bool>,
    pub tie_embedding: Option<bool>,

    pub rope_theta: Option<f64>,
    pub rope_parameters: Option<RopeParameters>,

    pub max_position_embeddings: usize,
    pub norm_eps: f64,
}
```

The normalized calculation should be centralized:

```rust
fn effective_ffn_dim(cfg: &Lfm2Config) -> Result<usize> {
    let mut dim = cfg
        .block_ff_dim
        .or(cfg.intermediate_size)
        .unwrap_or(
            cfg.hidden_size
                .checked_mul(4)
                .ok_or_else(|| Error::Msg("LFM2 FFN size overflow".into()))?,
        );

    if cfg.block_auto_adjust_ff_dim {
        dim = dim
            .checked_mul(2)
            .ok_or_else(|| Error::Msg("LFM2 FFN size overflow".into()))?
            / 3;

        dim = (cfg.block_ffn_dim_multiplier * dim as f64) as usize;

        if cfg.block_multiple_of == 0 {
            candle::bail!("block_multiple_of cannot be zero");
        }

        dim = dim.div_ceil(cfg.block_multiple_of) * cfg.block_multiple_of;
    }

    Ok(dim)
}
```

Required tests:

```rust
assert_eq!(effective_ffn_dim(&lfm25_vl_450m_cfg)?, 4608);
assert_eq!(effective_ffn_dim(&lfm25_vl_16b_cfg)?, 8192);
```

### Constructor root problem

The current dense model constructor internally adds a `"model"` prefix. That works for standalone `Lfm2ForCausalLM`, but the VL checkpoint nests the text tower under:

```text
model.language_model
```

The constructor must be split:

```rust
impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Self::new_from_parts(
            cfg,
            vb.pp("model"),
            Some(vb.pp("lm_head")),
        )
    }

    pub fn new_from_parts(
        cfg: &Config,
        model_vb: VarBuilder,
        lm_head_vb: Option<VarBuilder>,
    ) -> Result<Self> {
        // model_vb points directly at embed_tokens/layers/embedding_norm.
    }
}
```

For LFM2.5-VL:

```rust
let language_model = lfm2::Model::new_from_parts(
    &cfg.text_config,
    vb.pp("model").pp("language_model"),
    Some(vb.pp("lm_head")),
)?;
```

The loader must support:

* Explicit `lm_head.weight`.
* Tied output embeddings.
* `tie_word_embeddings`.
* Older `tie_embedding`.
* Missing `lm_head.weight` when embeddings are tied.

---

## 3. Required dense and quantized LFM2 APIs

The current dense implementation accepts token IDs and projects only the final hidden state. It does not expose an embedding-driven path.

Add:

```rust
impl Model {
    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor>;

    pub fn forward_hidden(
        &self,
        input_embeds: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor>;

    pub fn project_logits(
        &self,
        hidden_states: &Tensor,
        logits_to_keep: usize,
    ) -> Result<Tensor>;

    pub fn forward_embeds(
        &self,
        input_embeds: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor>;

    pub fn forward(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let embeds = self.embed_tokens(input_ids)?;
        self.forward_embeds(&embeds, index_pos, cache)
    }
}
```

The existing `forward` behavior remains intact.

The quantized GGUF implementation also needs:

```rust
impl ModelWeights {
    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor>;

    pub fn forward_embeds(
        &mut self,
        input_embeds: &Tensor,
        index_pos: usize,
    ) -> Result<Tensor>;

    pub fn clear_cache(&mut self);
}
```

The quantized implementation already owns token embeddings, attention caches, convolution caches, norms, and the output projection; it only lacks a route that bypasses token lookup.

### Regression gate

Before adding vision:

1. Load the 450M language tower.
2. Run a text-only prompt in official Transformers.
3. Export final prefill logits.
4. Run the same IDs through Candle.
5. Compare logits and one-token incremental decode.
6. Repeat with 1.6B.
7. Repeat through the GGUF text backend.

Until this passes, multimodal failures will be impossible to localize reliably.

---

## 4. New SigLIP2 NaFlex implementation

Do not extend `siglip.rs` with a large collection of conditional branches.

Create:

```text
candle-transformers/src/models/siglip2.rs
```

Candle’s existing SigLIP implementation is built around a fixed-size vision grid and does not implement the packed-patch NaFlex input or per-image positional interpolation required here.

### Public input contract

```rust
pub struct PackedVisionInputs<'a> {
    // [crop_count, max_patches, channels * patch_size * patch_size]
    pub pixel_values: &'a Tensor,

    // [crop_count, max_patches], 1 for valid patches
    pub pixel_attention_mask: &'a Tensor,

    // [crop_count, 2], expressed as patch rows and patch columns
    pub spatial_shapes: &'a Tensor,
}
```

For the official defaults:

```text
patch_size = 16
channels = 3
patch_dimension = 16 × 16 × 3 = 768
max_patches = 1024
```

### Vision encoder data flow

```text
packed patches
    -> linear patch embedding with bias
    -> image-specific resized positional embedding
    -> transformer encoder with bidirectional padding mask
    -> final LayerNorm
    -> padded final hidden states
```

The official SigLIP2 implementation uses already-patchified pixel values, a linear patch projection, per-sample spatial shapes, a padding mask, bilinear positional interpolation with `align_corners=false` and antialiasing, and a final post-layer normalization.

### Required model components

```rust
pub struct Siglip2VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub num_patches: usize,
    pub patch_size: usize,
    pub hidden_act: Activation,
    pub layer_norm_eps: f64,
    pub attention_dropout: f64,
    pub vision_use_head: bool,
}
```

```rust
pub struct VisionEmbeddings {
    patch_embedding: Linear,
    position_embedding: Embedding,
    base_grid_side: usize,
}
```

```rust
pub struct VisionEncoderLayer {
    layer_norm1: LayerNorm,
    self_attn: Attention,
    layer_norm2: LayerNorm,
    mlp: Mlp,
}
```

```rust
pub struct Siglip2VisionModel {
    embeddings: VisionEmbeddings,
    encoder: Vec<VisionEncoderLayer>,
    post_layernorm: LayerNorm,
}
```

### Attention requirements

For parity:

* Bidirectional attention, not causal attention.
* Invalid keys receive negative infinity.
* Attention scores and softmax should run in F32.
* Softmax output may then be cast back to the original model dtype.
* Padded query outputs may exist but must be discarded before projection.
* `vision_use_head=false` means no pooling head is loaded.

### Positional interpolation

The base learned position table is:

```text
[num_patches, hidden_size]
```

For these checkpoints:

```text
num_patches = 256
base grid = 16 × 16
```

For each crop:

1. Reshape to `[16, 16, hidden]`.
2. Convert to `[1, hidden, 16, 16]`.
3. Interpolate to `[1, hidden, patch_h, patch_w]`.
4. Use bilinear interpolation.
5. Use `align_corners=false`.
6. Enable antialiasing.
7. Flatten row-major to `[patch_h * patch_w, hidden]`.
8. Pad to `max_patches`.
9. Fill unused position slots with the first resized positional vector, matching the reference behavior.
10. Add the result to patch embeddings.

The initial implementation should calculate resized positional embeddings on CPU in F32, cache them by shape, then transfer them to the vision device. The operation is small compared with vision inference, and this avoids adding new Candle CUDA or Metal kernels before parity is established.

Cache key:

```rust
struct PositionCacheKey {
    patch_rows: usize,
    patch_cols: usize,
    dtype: DType,
    device_location: DeviceLocation,
}
```

Llama.cpp’s first implementation required a substantial follow-up because missing positional antialiasing produced different outputs, especially when one image dimension was below 256 pixels. The same follow-up also corrected smart-resize rounding, stretching versus padding, image start/end tokens, and marker placement.

This is the highest-risk numerical component.

---

## 5. Multimodal projector specification

Create:

```text
candle-transformers/src/models/lfm2_vl/projector.rs
```

The official projector performs:

```text
vision features
    -> pixel unshuffle
    -> optional LayerNorm
    -> Linear 1
    -> exact GELU
    -> Linear 2
```

The official Transformers implementation unpads each crop, reshapes it to its original patch grid, applies the projector, flattens the projected grid, concatenates all crop features, and replaces only the `<image>` token embeddings.

### Configuration

```rust
pub struct ProjectorConfig {
    pub vision_hidden_size: usize,
    pub text_hidden_size: usize,
    pub projector_hidden_size: usize,
    pub downsample_factor: usize,
    pub use_layer_norm: bool,
    pub layer_norm_eps: f64,
    pub use_bias: bool,
    pub activation: Activation,
}
```

### Dimensions

For factor `f`:

```text
projector_input = vision_hidden × f²
```

For LFM2.5-VL-450M:

```text
768 × 4 = 3072
3072 -> 2048 -> 1024
```

For LFM2.5-VL-1.6B:

```text
1152 × 4 = 4608
4608 -> 2048 -> 2048
```

Those dimensions come directly from the current official configurations. ([Hugging Face][1])

### Pixel unshuffle

Input:

```text
[B, patch_rows, patch_cols, vision_hidden]
```

Output:

```text
[B, patch_rows / factor, patch_cols / factor, vision_hidden × factor²]
```

The implementation must reject patch grids not divisible by the factor.

Do not substitute a generic space-to-depth operation without proving that its channel ordering matches the official series of reshape and permute operations.

Required unit fixture:

```text
Input values: monotonically increasing sequence
Grid: 4 × 6
Channels: 2
Factor: 2
```

Export the official PyTorch output and compare every resulting value exactly.

---

## 6. Rust image processor specification

The model crate should accept packed tensors and remain independent of image codecs.

For our fork, the reusable processor belongs in a small workspace crate:

```text
candle-vlm/
  src/
    lib.rs
    image.rs
    lfm2_vl/
      mod.rs
      config.rs
      processor.rs
      prompt.rs
      types.rs
```

For a smaller upstreamable Candle patch, the first implementation can live under the LFM2-VL example and be extracted after parity.

### Processor configuration

```rust
pub struct Lfm2VlProcessorConfig {
    pub do_resize: bool,
    pub do_rescale: bool,
    pub rescale_factor: f32,
    pub do_normalize: bool,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub do_pad: bool,

    pub downsample_factor: usize,
    pub encoder_patch_size: usize,

    pub do_image_splitting: bool,
    pub min_tiles: usize,
    pub max_tiles: usize,
    pub use_thumbnail: bool,
    pub tile_size: usize,

    pub min_image_tokens: usize,
    pub max_image_tokens: usize,
    pub max_num_patches: Option<usize>,
    pub max_pixels_tolerance: f64,
}
```

The checked-in processor configuration currently specifies:

* Factor 2.
* Patch size 16.
* Mean and standard deviation of 0.5 per channel.
* Maximum 1,024 packed patches.
* 64 to 256 projected image tokens for a normal or thumbnail image.
* Up to 10 tiles.
* 512-pixel tiles.
* Thumbnail enabled. ([Hugging Face][2])

The 450M model card separately recommends `min_image_tokens=32`, while the shipped processor default is 64. Therefore the code must load the processor configuration and permit explicit overrides rather than hardcoding either value. ([Hugging Face][3])

### Configuration precedence

Use:

```text
explicit CLI or API override
    > processor_config.json
    > GGUF processor metadata
    > model config
    > architecture defaults
```

### Smart resize

Let:

```text
total_factor = encoder_patch_size × downsample_factor
min_pixels = min_image_tokens × total_factor²
max_pixels = max_image_tokens × total_factor²
```

For the official defaults:

```text
total_factor = 16 × 2 = 32
```

Both target dimensions must be divisible by 32.

The Rust implementation must reproduce:

1. Nearest multiple rounding for initial dimensions.
2. Downscaling if rounded area exceeds `max_pixels`.
3. Upscaling if rounded area is below `min_pixels`.
4. Aspect ratio preservation as closely as the reference permits.
5. Stretching to the calculated size, not resizing and padding to it.

### Tiling

When the image exceeds the large-image threshold:

1. Enumerate possible `(columns, rows)` grids whose tile count is between `min_tiles` and `max_tiles`.
2. Select the grid with the closest aspect ratio.
3. Apply the reference area-based tie-break.
4. Resize the source to:

```text
columns × tile_size
rows × tile_size
```

5. Split into non-overlapping tiles in row-major order.
6. Append a smart-resized thumbnail when enabled and the selected grid contains more than one tile.

Crop ordering must be:

```text
image 0:
    row 0 col 0
    row 0 col 1
    ...
    row N col M
    thumbnail

image 1:
    ...
```

### Patchification

For each crop:

```text
[C, H, W]
    -> [C, patch_rows, patch_h, patch_cols, patch_w]
    -> [patch_rows, patch_cols, patch_h, patch_w, C]
    -> [patch_rows * patch_cols, patch_h * patch_w * C]
```

Padding produces:

```text
pixel_values:         [crop_count, max_num_patches, patch_dimension]
pixel_attention_mask: [crop_count, max_num_patches]
spatial_shapes:       [crop_count, 2]
```

### Output metadata

```rust
pub struct ProcessedVisionBatch {
    pub pixel_values: Tensor,
    pub pixel_attention_mask: Tensor,
    pub spatial_shapes: Tensor,

    pub crops: Vec<CropMeta>,
    pub images: Vec<ImageMeta>,
}

pub struct CropMeta {
    pub image_index: usize,
    pub crop_index: usize,
    pub kind: CropKind,
    pub patch_rows: usize,
    pub patch_cols: usize,
    pub projected_tokens: usize,
}

pub enum CropKind {
    Whole,
    Tile { row: usize, col: usize },
    Thumbnail,
}

pub struct ImageMeta {
    pub crop_range: Range<usize>,
    pub rows: usize,
    pub cols: usize,
    pub resized_width: usize,
    pub resized_height: usize,
}
```

Use one canonical function for projected token counts:

```rust
fn projected_token_count(
    patch_rows: usize,
    patch_cols: usize,
    factor: usize,
) -> Result<usize> {
    if patch_rows % factor != 0 || patch_cols % factor != 0 {
        candle::bail!("vision patch grid is not divisible by projector factor");
    }

    Ok((patch_rows / factor) * (patch_cols / factor))
}
```

The prompt processor and projector must both consume this value. They must not maintain separate token-count formulas.

---

## 7. Prompt expansion

Create:

```text
candle-vlm/src/lfm2_vl/prompt.rs
```

The official processor replaces each `<image>` sentinel with:

```text
<|image_start|>
    [optional row/column marker]
    <image> repeated once per projected feature
    ...
    [optional thumbnail marker]
    <image> repeated once per thumbnail feature
<|image_end|>
```

For tiled images, each row/column marker remains an ordinary learned text token. Only token ID 396 placeholders are replaced by image vectors.

### Required token resolution

Resolve all special tokens from the tokenizer:

```rust
pub struct Lfm2VlSpecialTokens {
    pub image_token: String,
    pub image_token_id: u32,
    pub image_start_token: String,
    pub image_end_token: String,
    pub image_thumbnail_token: String,
    pub tile_tokens: HashMap<(usize, usize), String>,
}
```

Do not hardcode token IDs other than using the model config as a consistency check.

For backwards compatibility, config deserialization should accept both:

```text
image_token_id
image_token_index
```

It should also accept both older and newer model types:

```text
lfm2-vl
lfm2_vl
```

### Prompt positioning

The processor must preserve the location of each user-provided `<image>` sentinel. The default chat helper should put the image before its accompanying text:

```text
<image>Describe this image.
```

Current official examples follow that structure, and llama.cpp found LFM2-VL output to be sensitive to media marker placement. ([Hugging Face][4])

### Validation

Before tokenization:

* Number of sentinels must equal number of images.
* Every required row/column marker must exist in the tokenizer.
* Expanded text length must remain within the configured context budget.

After tokenization:

* Every textual `<image>` occurrence must map to exactly one image token ID.
* Placeholder spans must be recorded.
* Total placeholder count must equal total projected image features.

Never silently truncate, pad, duplicate, or discard image features to force a match.

---

## 8. Composite LFM2-VL model

Create:

```text
candle-transformers/src/models/lfm2_vl/
  mod.rs
  config.rs
  model.rs
  projector.rs
  weights.rs
```

Export it through:

```text
candle-transformers/src/models/mod.rs
```

### Config

```rust
pub struct Lfm2VlConfig {
    pub text_config: lfm2::Lfm2Config,
    pub vision_config: siglip2::Siglip2VisionConfig,

    pub image_token_id: u32,
    pub downsample_factor: usize,

    pub projector_hidden_size: usize,
    pub projector_hidden_act: Activation,
    pub projector_bias: bool,
    pub projector_use_layernorm: bool,

    pub use_image_special_tokens: bool,
}
```

Validation:

```text
vision hidden × factor² == projector input width
projector output width == text hidden width
vision hidden divisible by vision head count
text hidden divisible by text head count
text head count divisible by KV head count
base vision num_patches is a perfect square
patch dimension uses checked arithmetic
all crop grids are divisible by factor
```

### Model API

```rust
pub struct Lfm2VlModel {
    vision_tower: siglip2::Siglip2VisionModel,
    projector: Lfm2VlProjector,
    language_model: lfm2::Model,
    image_token_id: u32,
}
```

```rust
pub struct EncodedImages {
    // [total_projected_tokens, text_hidden]
    pub embeddings: Tensor,

    pub per_image_ranges: Vec<Range<usize>>,
    pub per_crop_ranges: Vec<Range<usize>>,
}
```

```rust
impl Lfm2VlModel {
    pub fn encode_images(
        &self,
        inputs: &ProcessedVisionBatch,
        vision_batch_size: usize,
    ) -> Result<EncodedImages>;

    pub fn merge_image_embeddings(
        &self,
        input_ids: &Tensor,
        input_embeds: &Tensor,
        image_spans: &[ImageTokenSpan],
        encoded_images: &EncodedImages,
    ) -> Result<Tensor>;

    pub fn prefill(
        &self,
        input_ids: &Tensor,
        image_spans: &[ImageTokenSpan],
        encoded_images: Option<&EncodedImages>,
        cache: &mut lfm2::Cache,
    ) -> Result<Tensor>;

    pub fn decode(
        &self,
        token_ids: &Tensor,
        index_pos: usize,
        cache: &mut lfm2::Cache,
    ) -> Result<Tensor>;
}
```

### Encode sequence

For each crop chunk:

1. Run SigLIP2.
2. Determine valid feature length from the mask.
3. Narrow to valid features.
4. Reshape to `[1, patch_rows, patch_cols, vision_hidden]`.
5. Pixel-unshuffle.
6. Apply optional LayerNorm.
7. Apply linear 1, GELU, and linear 2.
8. Flatten to `[projected_tokens, text_hidden]`.
9. Append in original crop order.

### Merge sequence

1. Embed all prompt IDs normally.
2. Verify each supplied span contains only `image_token_id`.
3. Verify the sum of span lengths equals the number of image vectors.
4. Use `slice_assign` for each contiguous span.
5. Leave start, end, tile, and thumbnail tokens untouched.
6. Run the full merged sequence through LFM2 once.
7. Populate attention and convolution caches.
8. Decode subsequent text tokens without rerunning the vision tower.

This is precisely where Candle’s Qwen3-VL span-replacement implementation is reusable.

### Batching scope

Functional MVP:

```text
text batch size = 1
images per prompt = multiple
crops per image = multiple
```

The internal metadata should still use image and crop ranges so true text batching can be added without changing the processor format.

---

## 9. MMProj format support

“MMProj” should be a format-neutral concept in our code.

```rust
pub struct Mmproj {
    pub vision_tower: Siglip2VisionModel,
    pub projector: Lfm2VlProjector,
    pub metadata: MmprojMetadata,
}
```

```rust
pub struct MmprojMetadata {
    pub architecture: String,
    pub vision_hidden_size: usize,
    pub text_hidden_size: usize,
    pub patch_size: usize,
    pub downsample_factor: usize,
    pub processor: Lfm2VlProcessorConfig,
    pub source_model: Option<String>,
    pub source_revision: Option<String>,
}
```

### Format A: unified Hugging Face safetensors

Required native prefixes:

```text
model.vision_tower.vision_model
model.multi_modal_projector
model.language_model
lm_head
```

Load the official repository without renaming files or modifying its config.

Loader output should report:

```text
loaded tensors
missing tensors
unexpected tensors
shape mismatches
tied tensor resolution
resolved model roots
```

This is the numerical reference path.

### Format B: split safetensors mmproj

Add a development exporter that extracts only:

```text
model.vision_tower.*
model.multi_modal_projector.*
```

Output:

```text
mmproj.safetensors
mmproj.json
processor_config.json
```

Manifest:

```json
{
  "format": "candle-mmproj",
  "version": 1,
  "architecture": "lfm2_vl",
  "source_model": "LiquidAI/LFM2.5-VL-450M",
  "source_revision": "<commit>",
  "expected_text_hidden_size": 1024,
  "image_token_id": 396,
  "tensor_namespace_version": 1
}
```

This enables the first practical hybrid:

```text
quantized LFM2 GGUF text
+
native BF16/F16 safetensors vision tower and projector
```

The exporter is a build/development tool, not a runtime Python dependency.

### Format C: llama.cpp GGUF mmproj

Llama.cpp already supports an LFM2 projector type with:

* SigLIP-style vision blocks.
* Dynamic positional interpolation.
* Pixel unshuffle.
* Optional projector input normalization.
* Two projector linear layers.
* A projector scale factor in GGUF metadata.

Its conversion code registers `Lfm2VlForConditionalGeneration`, records the LFM2 projector type, and converts the packed linear patch embedding into a Conv2d-style GGUF layout.

The GGUF loader must therefore normalize that patch tensor back into Candle’s packed-linear layout:

```text
GGUF: [vision_hidden, channels, patch_size, patch_size]
Candle SigLIP2: [vision_hidden, channels × patch_size × patch_size]
```

Do not assume it is already stored in Hugging Face orientation.

Relevant GGUF metadata includes:

```text
clip.projector_type
clip.has_vision_encoder
clip.vision.embedding_length
clip.vision.feed_forward_length
clip.vision.block_count
clip.vision.attention.head_count
clip.vision.attention.layer_norm_epsilon
clip.vision.image_size
clip.vision.patch_size
clip.vision.image_mean
clip.vision.image_std
clip.vision.projector.scale_factor
clip.vision.preproc_min_tiles
clip.vision.preproc_max_tiles
clip.vision.preproc_image_size
```

Llama.cpp defines these keys and the associated vision/projector tensor namespaces in its mtmd implementation.

### GGUF implementation stages

#### Stage C1: compatibility loader

* Read GGUF metadata.
* Read all mmproj tensors.
* Dequantize tensors into dense F32/BF16/F16 Candle tensors.
* Normalize orientations and names.
* Run through the already-proven dense SigLIP2/projector implementation.
* Compare against the native path.

This supports quantized files functionally without initially executing quantized vision matrix multiplications.

#### Stage C2: native quantized execution

Introduce:

```rust
pub enum LinearOp {
    Dense(candle_nn::Linear),
    Quantized(candle::quantized::QMatMul),
}
```

```rust
impl LinearOp {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}
```

Use `LinearOp` for:

* Vision Q/K/V/out projections.
* Vision MLP linear layers.
* Projector linear 1.
* Projector linear 2.

Keep these dense:

* Biases.
* LayerNorm weights and biases.
* Positional embeddings.
* Small metadata tensors.

Support progression:

```text
Q4/Q8 text + BF16 split mmproj
Q4/Q8 text + F16 GGUF mmproj
Q4/Q8 text + Q8_0 GGUF mmproj
lower-bit vision tensors only after Q8 parity
```

Official model cards currently publish GGUF variants for both LFM2.5-VL checkpoints, giving us a real compatibility target. ([Hugging Face][4])

### Pair validation

Before inference:

```text
mmproj projector output == text model hidden size
mmproj architecture == lfm2 or lfm2_vl compatible
patch size matches processor
downsample factor matches prompt token counts
tokenizer image ID matches model image ID
vision layer count matches loaded tensor table
```

A mismatch must fail with an actionable error.

---

## 10. Proposed repository layout

```text
candle-transformers/
  src/models/
    lfm2.rs
    quantized_lfm2.rs
    siglip2.rs
    lfm2_vl/
      mod.rs
      config.rs
      model.rs
      projector.rs
      weights.rs
      gguf.rs

candle-vlm/
  Cargo.toml
  src/
    lib.rs
    image.rs
    lfm2_vl/
      mod.rs
      config.rs
      processor.rs
      prompt.rs
      types.rs
      cache.rs

candle-examples/
  examples/
    lfm2-vl/
      main.rs
      args.rs
      loading.rs
      generation.rs

tools/
  export_lfm2_vl_fixtures.py
  export_lfm2_vl_mmproj.py
  inspect_mmproj_gguf.py

tests/
  fixtures/
    lfm2_vl_tiny/
      config.json
      processor_config.json
      weights.safetensors
      goldens.safetensors
      manifest.json
```

No Candle core modification should be required for the first correct implementation.

If positional interpolation eventually becomes a measurable bottleneck, it can later become a reusable Candle operation with CPU, CUDA, and Metal implementations.

---

## 11. Example CLI contract

```text
candle-lfm2-vl
    --model-id <HF model>
    --revision <revision>

    --model-file <text.gguf>
    --mmproj-file <mmproj.gguf>
    --mmproj-dir <split safetensors directory>

    --tokenizer <model id or local path>
    --processor-config <path>

    --image <path>              repeatable
    --prompt <text>

    --device cpu|cuda:0|metal
    --vision-device cpu|cuda:0|metal
    --dtype f32|f16|bf16
    --vision-batch-size <n>

    --min-image-tokens <n>
    --max-image-tokens <n>
    --max-tiles <n>
    --no-image-splitting
    --no-thumbnail

    --max-new-tokens <n>
    --temperature <f>
    --top-p <f>
    --seed <u64>

    --dump-intermediates <dir>
```

Supported loading modes:

```text
Native:
    --model-id LiquidAI/LFM2.5-VL-450M

Hybrid:
    --model-file text-q4.gguf
    --mmproj-dir extracted-mmproj/
    --tokenizer LiquidAI/LFM2.5-VL-450M

Direct GGUF:
    --model-file text-q4.gguf
    --mmproj-file mmproj-q8.gguf
    --tokenizer LiquidAI/LFM2.5-VL-450M
```

The explicit options are preferable to guessing solely from extensions.

---

## 12. Open-source implementations worth using

### Hugging Face Transformers: numerical oracle

Use for:

* Exact model graph.
* Pixel unshuffle ordering.
* Placeholder replacement.
* SigLIP2 packed-patch input.
* Positional interpolation.
* Processor and prompt expansion.
* Golden fixture generation.

The official model, SigLIP2, image processor, and prompt processor implementations provide the reference behavior.

### mistral.rs: primary Rust donor

This is the closest implementation to what we need. Its current tree contains:

```text
mistralrs-core/src/vision_models/lfm2_vl/mod.rs
mistralrs-core/src/vision_models/lfm2_vl/vision.rs
mistralrs-core/src/vision_models/lfm2_vl/config.rs
mistralrs-core/src/vision_models/lfm2_vl/inputs_processor.rs
```

It implements a Candle-based LFM2-VL path, including the projector, vision encoder, dynamic processor, prompt expansion, and embedding insertion.

It also implements the correct LFM2 feed-forward normalization logic.

Do not copy the whole module directly. It is coupled to:

* `ShardedVarBuilder`.
* mistral.rs quantization abstractions.
* Device mapping.
* Its own SDPA layer.
* Its own cache and request pipeline.
* Its multimodal runtime traits.

Port the model math and processor logic into Candle-native types, then validate every stage against Transformers.

### llama.cpp: GGUF and independent parity reference

Use for:

* GGUF metadata.
* Tensor naming and layout.
* Existing LFM2 mmproj files.
* Quantized mmproj behavior.
* Independent processor comparisons.
* CPU/CUDA test cases.

Its initial merged implementation tested dynamic resolutions across CPU and CUDA, including rectangular images and mixed text/mmproj quantization.

More importantly, its later parity fixes document exactly which apparently small details can break LFM2-VL output.

### MLX-VLM: secondary independent reference

MLX-VLM contains an LFM2-VL model, vision implementation, and processor. It is useful for checking shape transformations and crop ordering independently of PyTorch and llama.cpp.

### Candle Qwen3-VL and PaLI-Gemma: integration patterns

Use Qwen3-VL for placeholder span replacement and `forward_embeds`. Use PaLI-Gemma for simpler multimodal constructor and cache flow patterns.

---

## 13. Golden fixture and test strategy

### Python reference exporter

`tools/export_lfm2_vl_fixtures.py` should pin and record:

```text
model repository
model revision
Transformers version or commit
PyTorch version
processor config hash
tokenizer hash
input image SHA-256
dtype
device
seed
```

Export:

```text
normalized model config
resized image arrays
tile arrays
packed pixel_values
pixel_attention_mask
spatial_shapes
image rows and columns
expanded prompt
input IDs
image token spans
resized positional embeddings
patch embeddings
selected vision layer outputs
final vision output
pixel-unshuffle output
projector output
merged text/image embeddings
prefill logits
first several decode logits
```

Use safetensors for floating tensors and JSON for metadata.

### Tiny CI fixture

Do not commit official model weights into the Candle fork.

Create a miniature random model using the same operations:

```text
vision hidden = 16
vision heads = 4
vision layers = 2
vision FFN = 32
base position grid = 4 × 4
patch size = 2
factor = 2
projector hidden = 24
text hidden = 12
text layers = 2
```

Generate its weights and goldens from the official Python implementation.

This allows fast CPU CI coverage of every operation without downloading a production checkpoint.

### Required image fixtures

Synthetic:

```text
checkerboard
horizontal gradient
vertical gradient
single-pixel impulses
RGB color bars
grayscale source
RGBA source
odd-sized source
```

Real-dimension cases:

```text
256 × 256
277 × 512
512 × 277
512 × 384
384 × 512
512 × 512
128 × 512
512 × 128
1000 × 3000
3000 × 1000
```

The first five rectangular and square cases overlap llama.cpp’s original LFM2-VL validation set.

Prompt cases:

```text
one image before text
one image between text spans
two images in one user turn
images across multiple turns
tiled image with thumbnail
image sentinel with no image
image with no sentinel
feature-count mismatch
missing row/column token
```

### Component gates

#### Config

```text
450M effective FFN = 4608
1.6B effective FFN = 8192
RoPE theta = 1,000,000
layer type list length = 16
tied head behavior is correct
```

#### Processor

```text
target dimensions exact
tile grid exact
crop order exact
patch values within fixture tolerance
mask exact
spatial shapes exact
projected token counts exact
expanded special-token string exact
input IDs exact
```

#### Vision

```text
patch embedding parity
position interpolation parity
padding mask parity
each encoder layer parity
post-LN parity
padded patches do not affect valid outputs
```

#### Projector

```text
pixel-unshuffle exact
optional LayerNorm exact
linear 1 exact
GELU exact
linear 2 exact
flatten order exact
```

#### Composite

```text
placeholder spans exact
merged embeddings exact
prefill logits match
one-shot and incremental decode agree
vision tower executes once per encoded image
cache reset produces deterministic result
```

#### GGUF

```text
metadata parsed correctly
patch weight orientation normalized
native and dequantized-GGUF image features agree
Q8 path agrees within quantization tolerance
wrong text/mmproj pair fails before inference
```

### Initial tolerances

CPU F32:

```text
processor integer metadata: exact
token IDs and spans: exact
pixel unshuffle: exact
position interpolation max abs: target <= 2e-5
vision/projector cosine: target >= 0.99999
full prefill logits max abs: target <= 1e-3
```

BF16/F16:

```text
vision/projector cosine: target >= 0.999
greedy next-token agreement across fixed suite
no NaN or infinity
```

Quantized:

```text
compare against llama.cpp using the same GGUF files
top-1 agreement across fixed prompts
top-k overlap
cosine similarity for projected image features
document expected quantization drift
```

Output captions alone are not acceptable proof. A plausible caption can hide major numerical and prompt-processing defects.

---

## 14. Implementation pass-down

### Phase 0 — Freeze the reference baseline

**Task:** Build the fixture exporter and record official outputs.

**Why:** This prevents us from debugging Rust against subjective generated text.

**Where:**

```text
tools/export_lfm2_vl_fixtures.py
tests/fixtures/lfm2_vl_tiny/
```

**How:**

1. Pin model revisions.
2. Export tiny-model goldens.
3. Export 450M component outputs.
4. Export 1.6B configuration and selected outputs.
5. Record source versions and hashes.
6. Add a fixture manifest schema.

**Gate:** We can reproduce the same fixture twice byte-for-byte where deterministic.

---

### Phase 1 — Repair LFM2 text support

**Task:** Normalize current LFM2 configs and expose embedding forwarding.

**Why:** The 450M language tower cannot load correctly under Candle’s current FFN calculation.

**Where:**

```text
candle-transformers/src/models/lfm2.rs
candle-transformers/src/models/quantized_lfm2.rs
```

**How:**

1. Add missing configuration fields and aliases.
2. Implement exact FFN-width normalization.
3. Parse nested RoPE parameters.
4. Normalize tied-output behavior.
5. Split constructor roots.
6. Add dense `embed_tokens` and `forward_embeds`.
7. Add quantized `embed_tokens` and `forward_embeds`.
8. Preserve existing public `forward`.
9. Add text-only parity tests.

**Gate:** Both language towers match reference prefill and incremental decode without vision.

---

### Phase 2 — Implement SigLIP2 NaFlex math

**Task:** Add packed-patch SigLIP2.

**Why:** Existing Candle SigLIP is not the required vision architecture.

**Where:**

```text
candle-transformers/src/models/siglip2.rs
```

**How:**

1. Implement config parsing.
2. Add linear patch embeddings.
3. Add exact position resizing.
4. Add bidirectional padding masks.
5. Add F32 attention softmax.
6. Add encoder blocks.
7. Add post-LN.
8. Reject unsupported pooling heads.
9. Consume Python-preprocessed packed tensors first.
10. Compare every stage against goldens.

**Gate:** Final valid-patch outputs match the Python fixture before any Rust image processing exists.

---

### Phase 3 — Implement projector and native composite model

**Task:** Connect SigLIP2 to dense LFM2.

**Why:** This proves the model graph independently of Rust image resizing and GGUF.

**Where:**

```text
candle-transformers/src/models/lfm2_vl/
```

**How:**

1. Add top-level config.
2. Add exact pixel unshuffle.
3. Add optional projector LayerNorm.
4. Add two projector linear layers.
5. Unpad and reshape each crop.
6. Concatenate projected crop outputs.
7. Add placeholder count validation.
8. Replace contiguous spans.
9. Run multimodal prefill.
10. Decode through existing cache.

**Gate:** Native 450M produces matching prefill logits from Python-preprocessed tensors.

---

### Phase 4 — Implement the Rust processor and prompt expander

**Task:** Remove the Python preprocessing dependency.

**Why:** Runtime must remain Rust-native and reproducible.

**Where:**

```text
candle-vlm/src/lfm2_vl/
```

**How:**

1. Parse `processor_config.json`.
2. Implement checked smart resize.
3. Implement tile-grid selection.
4. Implement crop ordering.
5. Implement thumbnail generation.
6. Implement exact normalization.
7. Implement patchification and padding.
8. Build prompt marker sequences from actual crop metadata.
9. Tokenize and record placeholder spans.
10. Compare all outputs against the processor fixture.

**Gate:** Raw image plus text in Rust produces the same packed tensors and IDs as Transformers.

---

### Phase 5 — Add hybrid mmproj loading

**Task:** Run quantized GGUF text with split safetensors vision/projector.

**Why:** This gives us a useful quantized product before GGUF mmproj parsing is involved.

**Where:**

```text
tools/export_lfm2_vl_mmproj.py
candle-transformers/src/models/lfm2_vl/weights.rs
candle-examples/examples/lfm2-vl/loading.rs
```

**How:**

1. Define versioned `mmproj.json`.
2. Extract only vision/projector tensors.
3. Load quantized text.
4. Load dense mmproj.
5. Validate hidden-size pairing.
6. Transfer only projected image features to the text device.
7. Add mixed-backend tests.

**Gate:** Quantized text plus dense mmproj gives the same image features as the unified native model and reasonable text-logit agreement.

---

### Phase 6 — Add direct GGUF mmproj support

**Task:** Load existing llama.cpp-compatible mmproj files.

**Why:** This is the intended deployment compatibility target.

**Where:**

```text
candle-transformers/src/models/lfm2_vl/gguf.rs
```

**How:**

1. Snapshot the tensor table from real official mmproj files.
2. Implement metadata validation.
3. Build a canonical tensor-name map.
4. Reverse the GGUF patch-embedding layout.
5. Dequantize to dense tensors initially.
6. Compare against native safetensors.
7. Add `LinearOp`.
8. Keep eligible weights quantized.
9. Compare Q8 output against llama.cpp.
10. Add pairing and malformed-file tests.

**Gate:** Official text GGUF and mmproj GGUF pairs run directly in Candle with verified preprocessing, prefill, and decode.

---

### Phase 7 — Optimize and stabilize

**Task:** Improve throughput without changing outputs.

**Why:** Correctness comes first; then we remove avoidable work.

**Where:** All multimodal modules and benchmarks.

**How:**

1. Cache resized positional embeddings.
2. Cache projected image embeddings by image hash, processor config hash, and mmproj revision.
3. Chunk crop inference.
4. Group crops with identical spatial shapes.
5. Reuse scratch tensors.
6. Avoid full-sequence scatter when spans are contiguous.
7. Evaluate compact unpadded vision attention.
8. Add CUDA benchmarks.
9. Add Windows and Linux verification.
10. Document supported formats and limitations.

**Gate:** Optimization fixtures remain numerically equivalent and no benchmark regresses unexpectedly.

---

## 15. Safety and failure handling

External images and model files are untrusted inputs.

Required protections:

* Checked multiplication for every shape and allocation.
* Configurable maximum source pixels.
* Configurable maximum images per request.
* Configurable maximum crops and total projected tokens.
* Reject zero-sized images.
* Reject unsupported color conversions cleanly.
* Reject malformed spatial shapes.
* Reject model tensors whose dimensions disagree with metadata.
* Reject absurd GGUF counts before allocation.
* No `unwrap` or `expect` on external data.
* No silent fallback from an invalid mmproj to text-only mode.
* No remote network access inside the model or processor crates.
* Cache keys must include model revision and processor configuration.
* Clear all text caches when starting a new request.

Suggested hard limits:

```rust
pub struct VisionLimits {
    pub max_source_pixels: usize,
    pub max_images: usize,
    pub max_crops_per_image: usize,
    pub max_total_crops: usize,
    pub max_patches_per_crop: usize,
    pub max_total_projected_tokens: usize,
}
```

---

## 16. Explicit non-goals for the first release

Do not include these in the first functional implementation:

* Video processing.
* Training or fine-tuning.
* Arbitrary VLM abstraction covering every architecture.
* True multi-request text batching.
* Flash attention with padded NaFlex tensors.
* Custom CUDA image-resize kernels.
* A new GGUF converter.
* WASM or WebGPU execution.
* Lower-than-Q8 vision quantization.
* Automatic model downloading inside `candle-transformers`.

A giant generic `VisionLanguageModel` trait is premature. Build a correct typed LFM2-VL implementation first. Extract common interfaces after a second architecture proves which abstractions are actually shared.

---

## 17. Definition of done

The implementation is complete when all of the following are true:

* `LiquidAI/LFM2.5-VL-450M` loads unmodified from native safetensors.
* `LiquidAI/LFM2.5-VL-1.6B` loads unmodified from native safetensors.
* The 450M FFN width resolves to 4,608.
* The 1.6B FFN width resolves to 8,192.
* Text-only LFM2 behavior remains compatible with existing Candle examples.
* Single-image native-resolution input works.
* Rectangular input works.
* Large-image tiling works.
* Thumbnail processing works.
* Multiple images in one prompt work.
* Special image markers are generated correctly.
* Feature and placeholder mismatches fail.
* Vision runs only during prefill or explicit re-encoding.
* Dense prefill and incremental decode agree.
* Quantized text plus split safetensors mmproj works.
* Direct GGUF text plus GGUF mmproj works.
* CPU F32 fixture parity passes.
* CUDA BF16/F16 integration passes.
* Q8 mmproj agrees with llama.cpp within documented quantization tolerance.
* No model dimensions, tile counts, projector widths, or token counts are hardcoded to one checkpoint.
* Malformed images and model files return controlled errors rather than panics.

## TL;DR

We should not bolt an MLP onto Candle’s current LFM2 implementation. First repair dense LFM2 configuration and add `forward_embeds`; then add a separate SigLIP2 NaFlex model, exact processor, factor-2 pixel-unshuffle projector, and placeholder-span prefill path. Prove native safetensors parity on the 450M checkpoint first, because it exposes Candle’s current FFN-width bug. After that, add quantized GGUF text plus split mmproj, then direct llama.cpp-compatible GGUF mmproj loading and finally quantized vision execution. mistral.rs is the best Rust donor, Transformers is the numerical oracle, and llama.cpp is the GGUF and parity reference.

[1]: https://huggingface.co/LiquidAI/LFM2.5-VL-450M/blob/main/config.json "config.json · LiquidAI/LFM2.5-VL-450M at main"
[2]: https://huggingface.co/LiquidAI/LFM2.5-VL-450M/blob/main/processor_config.json "processor_config.json · LiquidAI/LFM2.5-VL-450M at main"
[3]: https://huggingface.co/LiquidAI/LFM2.5-VL-450M?utm_source=chatgpt.com "LiquidAI/LFM2.5-VL-450M · Hugging Face"
[4]: https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B?utm_source=chatgpt.com "LiquidAI/LFM2.5-VL-1.6B · Hugging Face"
