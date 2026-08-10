//! SigLIP2 NaFlex vision encoding for already-patchified inputs.
//!
//! This module intentionally stops at the padded vision hidden states. Image
//! resizing, tiling, normalization, patchification, and the LFM2.5-VL
//! projector are separate phases.

use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{layer_norm, linear, Activation, LayerNorm, LayerNormConfig, Linear, VarBuilder};
use std::collections::HashMap;
use std::sync::RwLock;

fn default_hidden_size() -> usize {
    768
}

fn default_intermediate_size() -> usize {
    3072
}

fn default_num_hidden_layers() -> usize {
    12
}

fn default_num_attention_heads() -> usize {
    12
}

fn default_num_channels() -> usize {
    3
}

fn default_patch_size() -> usize {
    16
}

fn default_num_patches() -> usize {
    256
}

fn default_hidden_act() -> Activation {
    Activation::GeluPytorchTanh
}

fn default_layer_norm_eps() -> f64 {
    1e-6
}

/// The vision-only fields consumed from a SigLIP2 configuration.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct Siglip2VisionConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_num_patches")]
    pub num_patches: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default)]
    pub attention_dropout: f64,
    #[serde(default)]
    pub vision_use_head: bool,
}

impl Siglip2VisionConfig {
    /// Deserialize and validate a standalone SigLIP2 vision JSON object.
    pub fn from_json(json: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(json)
            .map_err(|err| candle::Error::Msg(format!("invalid SigLIP2 config: {err}")))?;
        config.validate()?;
        Ok(config)
    }

    /// Validate dimensions and reject configuration features outside this phase.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 {
            candle::bail!("SigLIP2 hidden_size must be greater than zero")
        }
        if self.intermediate_size == 0 {
            candle::bail!("SigLIP2 intermediate_size must be greater than zero")
        }
        if self.num_hidden_layers == 0 {
            candle::bail!("SigLIP2 num_hidden_layers must be greater than zero")
        }
        if self.num_attention_heads == 0 || self.hidden_size % self.num_attention_heads != 0 {
            candle::bail!(
                "SigLIP2 hidden_size {} is not divisible by num_attention_heads {}",
                self.hidden_size,
                self.num_attention_heads
            )
        }
        if self.num_channels == 0 || self.patch_size == 0 {
            candle::bail!("SigLIP2 num_channels and patch_size must be greater than zero")
        }
        if self.num_patches == 0 {
            candle::bail!("SigLIP2 num_patches must be greater than zero")
        }
        let _ = self.base_grid_side()?;
        let _ = self.patch_dimension()?;
        if !self.layer_norm_eps.is_finite() || self.layer_norm_eps <= 0.0 {
            candle::bail!("SigLIP2 layer_norm_eps must be finite and positive")
        }
        if !self.attention_dropout.is_finite() || self.attention_dropout != 0.0 {
            candle::bail!(
                "SigLIP2 attention_dropout {} is unsupported; inference requires zero dropout",
                self.attention_dropout
            )
        }
        if self.vision_use_head {
            candle::bail!("SigLIP2 vision_use_head=true is unsupported in the NaFlex tensor phase")
        }
        Ok(())
    }

    fn base_grid_side(&self) -> Result<usize> {
        square_side(self.num_patches, "SigLIP2 num_patches")
    }

    fn patch_dimension(&self) -> Result<usize> {
        self.num_channels
            .checked_mul(self.patch_size)
            .and_then(|value| value.checked_mul(self.patch_size))
            .ok_or_else(|| candle::Error::Msg("SigLIP2 patch dimension overflow".into()))
    }
}

/// Packed, already-patchified input for the NaFlex vision model.
pub struct PackedVisionInputs<'a> {
    /// `[crop_count, max_patches, channels * patch_size * patch_size]`.
    pub pixel_values: &'a Tensor,
    /// `[crop_count, max_patches]`, with a one for every valid patch.
    pub pixel_attention_mask: &'a Tensor,
    /// `[crop_count, 2]`, containing `(patch_rows, patch_cols)`.
    pub spatial_shapes: &'a Tensor,
}

#[derive(Debug)]
struct VisionEmbeddings {
    patch_embedding: Linear,
    position_embedding: Tensor,
    base_grid_side: usize,
    hidden_size: usize,
    dtype: DType,
    device: Device,
    position_cache: RwLock<HashMap<(usize, usize), Tensor>>,
}

impl VisionEmbeddings {
    fn new(config: &Siglip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embedding = linear(
            config.patch_dimension()?,
            config.hidden_size,
            vb.pp("patch_embedding"),
        )?;
        let position_embedding = vb
            .pp("position_embedding")
            .get((config.num_patches, config.hidden_size), "weight")?;
        if position_embedding.dims() != [config.num_patches, config.hidden_size] {
            candle::bail!(
                "SigLIP2 position_embedding has shape {:?}, expected [{}, {}]",
                position_embedding.dims(),
                config.num_patches,
                config.hidden_size
            )
        }
        let dtype = patch_embedding.weight().dtype();
        let device = patch_embedding.weight().device().clone();
        Ok(Self {
            patch_embedding,
            position_embedding,
            base_grid_side: config.base_grid_side()?,
            hidden_size: config.hidden_size,
            dtype,
            device,
            position_cache: RwLock::new(HashMap::new()),
        })
    }

    fn forward(
        &self,
        inputs: &PackedVisionInputs<'_>,
        shapes: &[(usize, usize)],
    ) -> Result<EmbeddingStages> {
        let pixel_values = inputs.pixel_values.to_dtype(self.dtype)?;
        let patch_embedding = self.patch_embedding.forward(&pixel_values)?;
        let resized_position_embedding =
            self.position_embeddings(shapes, patch_embedding.dim(1)?)?;
        let embeddings_with_position =
            patch_embedding.broadcast_add(&resized_position_embedding)?;
        Ok(EmbeddingStages {
            patch_embedding,
            resized_position_embedding,
            embeddings_with_position,
        })
    }

    fn position_embeddings(&self, shapes: &[(usize, usize)], max_patches: usize) -> Result<Tensor> {
        let mut per_crop = Vec::with_capacity(shapes.len());
        for &(rows, cols) in shapes {
            let valid_patches = rows
                .checked_mul(cols)
                .ok_or_else(|| candle::Error::Msg("SigLIP2 spatial patch count overflow".into()))?;
            if valid_patches > max_patches {
                candle::bail!(
                    "SigLIP2 spatial shape [{rows}, {cols}] needs {valid_patches} patches, but max_patches is {max_patches}"
                )
            }
            let resized = self.resized_positions(rows, cols)?;
            let padded = if valid_patches == max_patches {
                resized
            } else {
                let first = resized.i(0)?.reshape((1, self.hidden_size))?;
                let padding =
                    first.broadcast_as((max_patches - valid_patches, self.hidden_size))?;
                Tensor::cat(&[&resized, &padding], 0)?
            };
            per_crop.push(padded.unsqueeze(0)?);
        }
        let per_crop: Vec<&Tensor> = per_crop.iter().collect();
        Tensor::cat(&per_crop, 0)
    }

    fn resized_positions(&self, rows: usize, cols: usize) -> Result<Tensor> {
        if let Some(cached) = self
            .position_cache
            .read()
            .map_err(|_| candle::Error::Msg("SigLIP2 position cache read lock poisoned".into()))?
            .get(&(rows, cols))
            .cloned()
        {
            return Ok(cached);
        }
        let base = self
            .position_embedding
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        let resized = resize_bilinear_antialias(
            &base,
            self.base_grid_side,
            self.base_grid_side,
            rows,
            cols,
            self.hidden_size,
        )?;
        let resized = Tensor::from_vec(
            resized,
            (
                rows.checked_mul(cols).ok_or_else(|| {
                    candle::Error::Msg("SigLIP2 resized position count overflow".into())
                })?,
                self.hidden_size,
            ),
            &self.device,
        )?
        .to_dtype(self.dtype)?;
        self.position_cache
            .write()
            .map_err(|_| candle::Error::Msg("SigLIP2 position cache write lock poisoned".into()))?
            .insert((rows, cols), resized.clone());
        Ok(resized)
    }
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug)]
struct EmbeddingStages {
    patch_embedding: Tensor,
    resized_position_embedding: Tensor,
    embeddings_with_position: Tensor,
}

#[derive(Clone, Debug)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(config: &Siglip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = config
            .hidden_size
            .checked_div(config.num_attention_heads)
            .ok_or_else(|| candle::Error::Msg("SigLIP2 attention head dimension is zero".into()))?;
        Ok(Self {
            q_proj: linear(config.hidden_size, config.hidden_size, vb.pp("q_proj"))?,
            k_proj: linear(config.hidden_size, config.hidden_size, vb.pp("k_proj"))?,
            v_proj: linear(config.hidden_size, config.hidden_size, vb.pp("v_proj"))?,
            out_proj: linear(config.hidden_size, config.hidden_size, vb.pp("out_proj"))?,
            num_heads: config.num_attention_heads,
            head_dim,
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (batch_size, sequence_length, _) = xs.dims3()?;
        let query = self.split_heads(&self.q_proj.forward(xs)?, batch_size, sequence_length)?;
        let key = self.split_heads(&self.k_proj.forward(xs)?, batch_size, sequence_length)?;
        let value = self.split_heads(&self.v_proj.forward(xs)?, batch_size, sequence_length)?;

        let query_f32 = query.to_dtype(DType::F32)?;
        let key_f32 = key.to_dtype(DType::F32)?;
        let scores = (query_f32.matmul(&key_f32.t()?)? / (self.head_dim as f64).sqrt())?;
        let mask = mask
            .reshape((batch_size, 1, 1, sequence_length))?
            .broadcast_as((batch_size, self.num_heads, sequence_length, sequence_length))?;
        let valid = mask.gt(0f32)?;
        let neg_inf =
            Tensor::new(f32::MIN, scores.device())?.broadcast_as(scores.shape().clone())?;
        let scores = valid.where_cond(&scores, &neg_inf)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(query.dtype())?;
        let output = weights.matmul(&value)?.transpose(1, 2)?.reshape((
            batch_size,
            sequence_length,
            self.num_heads * self.head_dim,
        ))?;
        self.out_proj.forward(&output.to_dtype(xs.dtype())?)
    }

    fn split_heads(
        &self,
        xs: &Tensor,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<Tensor> {
        xs.reshape((batch_size, sequence_length, self.num_heads, self.head_dim))?
            .transpose(1, 2)
    }
}

#[derive(Clone, Debug)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
    activation: Activation,
}

impl Mlp {
    fn new(config: &Siglip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(config.hidden_size, config.intermediate_size, vb.pp("fc1"))?,
            fc2: linear(config.intermediate_size, config.hidden_size, vb.pp("fc2"))?,
            activation: config.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.fc1)?
            .apply(&self.activation)?
            .apply(&self.fc2)
    }
}

#[derive(Clone, Debug)]
struct EncoderLayer {
    layer_norm1: LayerNorm,
    self_attn: Attention,
    layer_norm2: LayerNorm,
    mlp: Mlp,
}

impl EncoderLayer {
    fn new(config: &Siglip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        let layer_norm_config = LayerNormConfig {
            eps: config.layer_norm_eps,
            ..LayerNormConfig::default()
        };
        Ok(Self {
            layer_norm1: layer_norm(config.hidden_size, layer_norm_config, vb.pp("layer_norm1"))?,
            self_attn: Attention::new(config, vb.pp("self_attn"))?,
            layer_norm2: layer_norm(config.hidden_size, layer_norm_config, vb.pp("layer_norm2"))?,
            mlp: Mlp::new(config, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let attended = self
            .self_attn
            .forward(&xs.apply(&self.layer_norm1)?, mask)?;
        let xs = (residual + attended)?;
        let residual = &xs;
        let feed_forward = self.mlp.forward(&xs.apply(&self.layer_norm2)?)?;
        residual + feed_forward
    }
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug)]
struct ForwardStages {
    embeddings: EmbeddingStages,
    encoder_layers: Vec<Tensor>,
    post_layernorm: Tensor,
}

/// Candle SigLIP2 NaFlex vision encoder for packed patch tensors.
#[derive(Debug)]
pub struct Siglip2VisionModel {
    config: Siglip2VisionConfig,
    embeddings: VisionEmbeddings,
    encoder: Vec<EncoderLayer>,
    post_layernorm: LayerNorm,
}

impl Siglip2VisionModel {
    /// Load the model relative to the production `model.vision_tower.vision_model`
    /// namespace. Callers should pass a `VarBuilder` already positioned at
    /// `vision_model`.
    pub fn new(config: &Siglip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        config.validate()?;
        let embeddings = VisionEmbeddings::new(config, vb.pp("embeddings"))?;
        let encoder_vb = vb.pp("encoder").pp("layers");
        let mut encoder = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            encoder.push(EncoderLayer::new(config, encoder_vb.pp(index))?);
        }
        let post_layernorm = layer_norm(
            config.hidden_size,
            LayerNormConfig {
                eps: config.layer_norm_eps,
                ..LayerNormConfig::default()
            },
            vb.pp("post_layernorm"),
        )?;
        Ok(Self {
            config: config.clone(),
            embeddings,
            encoder,
            post_layernorm,
        })
    }

    /// Run packed vision inputs and return `[crop_count, max_patches, hidden_size]`.
    pub fn forward(&self, inputs: &PackedVisionInputs<'_>) -> Result<Tensor> {
        Ok(self.forward_stages(inputs)?.post_layernorm)
    }

    fn forward_stages(&self, inputs: &PackedVisionInputs<'_>) -> Result<ForwardStages> {
        let (crop_count, max_patches, patch_dimension) = inputs.pixel_values.dims3()?;
        if crop_count == 0 || max_patches == 0 {
            candle::bail!("SigLIP2 packed inputs must contain at least one crop and patch slot")
        }
        if inputs.pixel_attention_mask.dims() != [crop_count, max_patches] {
            candle::bail!(
                "SigLIP2 pixel_attention_mask has shape {:?}, expected [{crop_count}, {max_patches}]",
                inputs.pixel_attention_mask.dims()
            )
        }
        if inputs.spatial_shapes.dims() != [crop_count, 2] {
            candle::bail!(
                "SigLIP2 spatial_shapes has shape {:?}, expected [{crop_count}, 2]",
                inputs.spatial_shapes.dims()
            )
        }
        if patch_dimension != self.config.patch_dimension()? {
            candle::bail!(
                "SigLIP2 pixel_values patch dimension is {patch_dimension}, expected {}",
                self.config.patch_dimension()?
            )
        }
        let shapes = read_spatial_shapes(inputs.spatial_shapes)?;
        let mask = validate_attention_mask(inputs.pixel_attention_mask, &shapes, max_patches)?;
        let embeddings = self.embeddings.forward(inputs, &shapes)?;
        let mut hidden = embeddings.embeddings_with_position.clone();
        let mut encoder_layers = Vec::with_capacity(self.encoder.len());
        for layer in &self.encoder {
            hidden = layer.forward(&hidden, &mask)?;
            encoder_layers.push(hidden.clone());
        }
        let post_layernorm = self.post_layernorm.forward(&hidden)?;
        Ok(ForwardStages {
            embeddings,
            encoder_layers,
            post_layernorm,
        })
    }
}

fn square_side(value: usize, label: &str) -> Result<usize> {
    if value == 0 {
        candle::bail!("{label} must be a non-zero square")
    }
    let mut side = (value as f64).sqrt() as usize;
    while side
        .checked_mul(side)
        .map(|square| square < value)
        .unwrap_or(false)
    {
        side = side
            .checked_add(1)
            .ok_or_else(|| candle::Error::Msg(format!("{label} square root overflow")))?;
    }
    while side
        .checked_mul(side)
        .map(|square| square > value)
        .unwrap_or(true)
    {
        if side == 0 {
            candle::bail!("{label} is not a square")
        }
        side -= 1;
    }
    if side.checked_mul(side) != Some(value) {
        candle::bail!("{label}={value} must be a perfect square")
    }
    Ok(side)
}

fn read_spatial_shapes(spatial_shapes: &Tensor) -> Result<Vec<(usize, usize)>> {
    let values = read_integer_matrix(spatial_shapes, "spatial_shapes")?;
    values
        .into_iter()
        .map(|row| {
            if row.len() != 2 || row[0] == 0 || row[1] == 0 {
                candle::bail!("SigLIP2 spatial shape must contain two positive dimensions")
            }
            let rows = usize::try_from(row[0])
                .map_err(|_| candle::Error::Msg("SigLIP2 spatial row does not fit usize".into()))?;
            let cols = usize::try_from(row[1]).map_err(|_| {
                candle::Error::Msg("SigLIP2 spatial column does not fit usize".into())
            })?;
            rows.checked_mul(cols)
                .ok_or_else(|| candle::Error::Msg("SigLIP2 spatial patch count overflow".into()))?;
            Ok((rows, cols))
        })
        .collect()
}

fn read_integer_matrix(tensor: &Tensor, label: &str) -> Result<Vec<Vec<u64>>> {
    let values = match tensor.dtype() {
        DType::U8 => tensor
            .to_vec2::<u8>()?
            .into_iter()
            .map(|r| r.into_iter().map(u64::from).collect())
            .collect(),
        DType::U32 => tensor
            .to_vec2::<u32>()?
            .into_iter()
            .map(|r| r.into_iter().map(u64::from).collect())
            .collect(),
        DType::I16 => tensor
            .to_vec2::<i16>()?
            .into_iter()
            .map(|r| {
                r.into_iter()
                    .map(|v| {
                        u64::try_from(v).map_err(|_| {
                            candle::Error::Msg(format!("{label} contains a negative value"))
                        })
                    })
                    .collect()
            })
            .collect::<Result<Vec<Vec<u64>>>>()?,
        DType::I32 => tensor
            .to_vec2::<i32>()?
            .into_iter()
            .map(|r| {
                r.into_iter()
                    .map(|v| {
                        u64::try_from(v).map_err(|_| {
                            candle::Error::Msg(format!("{label} contains a negative value"))
                        })
                    })
                    .collect()
            })
            .collect::<Result<Vec<Vec<u64>>>>()?,
        DType::I64 => tensor
            .to_vec2::<i64>()?
            .into_iter()
            .map(|r| {
                r.into_iter()
                    .map(|v| {
                        u64::try_from(v).map_err(|_| {
                            candle::Error::Msg(format!("{label} contains a negative value"))
                        })
                    })
                    .collect()
            })
            .collect::<Result<Vec<Vec<u64>>>>()?,
        dtype => candle::bail!("SigLIP2 {label} must use an integer dtype, got {dtype:?}"),
    };
    Ok(values)
}

fn validate_attention_mask(
    mask: &Tensor,
    shapes: &[(usize, usize)],
    max_patches: usize,
) -> Result<Tensor> {
    let mask_f32 = mask.to_dtype(DType::F32)?;
    let values = mask_f32.to_vec2::<f32>()?;
    if values.len() != shapes.len() {
        candle::bail!("SigLIP2 attention mask crop count does not match spatial_shapes")
    }
    for (crop, (row, &(rows, cols))) in values.iter().zip(shapes).enumerate() {
        let valid_patches = rows
            .checked_mul(cols)
            .ok_or_else(|| candle::Error::Msg("SigLIP2 spatial patch count overflow".into()))?;
        if valid_patches > max_patches || row.len() != max_patches {
            candle::bail!("SigLIP2 attention mask length does not match packed input")
        }
        for (index, &value) in row.iter().enumerate() {
            if !value.is_finite() || (value != 0.0 && value != 1.0) {
                candle::bail!("SigLIP2 attention mask crop {crop} contains a non-binary value")
            }
            let expected = if index < valid_patches { 1.0 } else { 0.0 };
            if value != expected {
                candle::bail!(
                    "SigLIP2 attention mask crop {crop} is not a valid prefix for spatial shape [{rows}, {cols}]"
                )
            }
        }
    }
    Ok(mask_f32)
}

#[derive(Clone, Debug)]
struct ResizeWeights {
    indices: Vec<usize>,
    weights: Vec<f32>,
}

fn resize_weights(input: usize, output: usize, index: usize) -> Result<ResizeWeights> {
    if input == 0 || output == 0 || index >= output {
        candle::bail!("invalid SigLIP2 resize dimensions")
    }
    let scale = input as f32 / output as f32;
    let support = if scale >= 1.0 { scale } else { 1.0 };
    if !scale.is_finite() || !support.is_finite() {
        candle::bail!("SigLIP2 resize scale is not finite")
    }
    let max_length = (support.ceil() as usize)
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| candle::Error::Msg("SigLIP2 resize support overflow".into()))?;
    let max_length_i64 = i64::try_from(max_length)
        .map_err(|_| candle::Error::Msg("SigLIP2 resize support is too large".into()))?;
    let input_i64 = i64::try_from(input)
        .map_err(|_| candle::Error::Msg("SigLIP2 resize input dimension is too large".into()))?;
    let center = scale * (index as f32 + 0.5);
    let inv_scale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let start = ((center - support + 0.5) as i64).max(0);
    let end = ((center + support + 0.5) as i64).min(input_i64);
    let length_i64 = end
        .checked_sub(start)
        .ok_or_else(|| candle::Error::Msg("SigLIP2 resize range overflow".into()))?
        .clamp(0, max_length_i64);
    let length = usize::try_from(length_i64)
        .map_err(|_| candle::Error::Msg("SigLIP2 resize range is too large".into()))?;
    let mut indices = Vec::with_capacity(length);
    let mut weights = Vec::with_capacity(length);
    let mut total = 0f32;
    for offset in 0..length {
        let source = start as usize + offset;
        let argument = (source as f32 - center + 0.5) * inv_scale;
        let weight = (1.0 - argument.abs()).max(0.0);
        indices.push(source);
        weights.push(weight);
        total += weight;
    }
    if !total.is_finite() || total <= 0.0 {
        candle::bail!("SigLIP2 resize weights have zero or invalid normalization")
    }
    for weight in weights.iter_mut().take(length) {
        *weight /= total;
    }
    Ok(ResizeWeights { indices, weights })
}

fn resize_bilinear_antialias(
    input: &[Vec<f32>],
    input_height: usize,
    input_width: usize,
    output_height: usize,
    output_width: usize,
    channels: usize,
) -> Result<Vec<f32>> {
    let expected = input_height
        .checked_mul(input_width)
        .ok_or_else(|| candle::Error::Msg("SigLIP2 resize input size overflow".into()))?;
    if input.len() != expected || input.iter().any(|row| row.len() != channels) {
        candle::bail!("SigLIP2 positional table has an unexpected shape")
    }
    let horizontal: Vec<ResizeWeights> = (0..output_width)
        .map(|index| resize_weights(input_width, output_width, index))
        .collect::<Result<_>>()?;
    let vertical: Vec<ResizeWeights> = (0..output_height)
        .map(|index| resize_weights(input_height, output_height, index))
        .collect::<Result<_>>()?;

    // PyTorch's antialiased implementation is separable and processes the
    // contiguous (width) dimension before the height dimension.
    let horizontal_size = input_height
        .checked_mul(output_width)
        .and_then(|value| value.checked_mul(channels))
        .ok_or_else(|| candle::Error::Msg("SigLIP2 resize output size overflow".into()))?;
    let mut horizontal_output = vec![0f32; horizontal_size];
    for row in 0..input_height {
        for col in 0..output_width {
            let weights = &horizontal[col];
            for channel in 0..channels {
                let mut value = 0f32;
                for offset in 0..weights.indices.len() {
                    let source_col = weights.indices[offset];
                    value +=
                        input[row * input_width + source_col][channel] * weights.weights[offset];
                }
                let index = (row * output_width + col) * channels + channel;
                horizontal_output[index] = value;
            }
        }
    }

    let output_size = output_height
        .checked_mul(output_width)
        .and_then(|value| value.checked_mul(channels))
        .ok_or_else(|| candle::Error::Msg("SigLIP2 resize output size overflow".into()))?;
    let mut output = vec![0f32; output_size];
    for row in 0..output_height {
        let weights = &vertical[row];
        for col in 0..output_width {
            for channel in 0..channels {
                let mut value = 0f32;
                for offset in 0..weights.indices.len() {
                    let source_row = weights.indices[offset];
                    let index = (source_row * output_width + col) * channels + channel;
                    value += horizontal_output[index] * weights.weights[offset];
                }
                output[(row * output_width + col) * channels + channel] = value;
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    const TINY_FIXTURE: &[u8] =
        include_bytes!("../../../tests/fixtures/lfm2_vl_tiny/tensors.safetensors");

    fn tiny_config() -> Siglip2VisionConfig {
        Siglip2VisionConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_channels: 3,
            patch_size: 2,
            num_patches: 16,
            hidden_act: Activation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
            attention_dropout: 0.0,
            vision_use_head: false,
        }
    }

    #[test]
    fn serde_config_parses_and_validates_dynamic_fields() -> Result<()> {
        let config = Siglip2VisionConfig::from_json(
            r#"{
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_channels": 3,
                "patch_size": 2,
                "num_patches": 16,
                "hidden_act": "gelu_pytorch_tanh",
                "layer_norm_eps": 0.000001,
                "attention_dropout": 0.0,
                "vision_use_head": false
            }"#,
        )?;
        assert_eq!(config.hidden_size, 16);
        assert_eq!(config.intermediate_size, 32);
        assert_eq!(config.hidden_act, Activation::GeluPytorchTanh);
        assert_eq!(config.base_grid_side()?, 4);
        assert!(
            Siglip2VisionConfig::from_json(r#"{"hidden_size": 16, "num_patches": 15}"#).is_err()
        );
        assert!(Siglip2VisionConfig::from_json(
            r#"{"hidden_size": 16, "num_patches": 16, "vision_use_head": true}"#
        )
        .is_err());
        Ok(())
    }

    fn fixture_tensor<'a>(tensors: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
        tensors
            .get(name)
            .ok_or_else(|| candle::Error::Msg(format!("missing tiny fixture tensor {name}")))
    }

    fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32, label: &str) -> Result<()> {
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if actual.len() != expected.len() {
            candle::bail!("{label}: element count mismatch")
        }
        let mut max_abs = 0f32;
        let mut dot = 0f32;
        let mut actual_norm = 0f32;
        let mut expected_norm = 0f32;
        for (&lhs, &rhs) in actual.iter().zip(&expected) {
            max_abs = max_abs.max((lhs - rhs).abs());
            dot += lhs * rhs;
            actual_norm += lhs * lhs;
            expected_norm += rhs * rhs;
        }
        let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt());
        eprintln!("{label}: max_abs={max_abs:.9e}, cosine={cosine:.9}");
        assert!(
            max_abs <= tolerance,
            "{label}: max_abs={max_abs} > {tolerance}"
        );
        assert!(
            cosine.is_finite() && cosine >= 0.99999,
            "{label}: cosine={cosine} < 0.99999"
        );
        Ok(())
    }

    fn tiny_model() -> Result<(Siglip2VisionModel, HashMap<String, Tensor>)> {
        let device = Device::Cpu;
        let weights = VarBuilder::from_slice_safetensors(TINY_FIXTURE, DType::F32, &device)?
            .pp("weights")
            .pp("model")
            .pp("vision_tower");
        let model = Siglip2VisionModel::new(&tiny_config(), weights)?;
        let tensors = candle::safetensors::load_buffer(TINY_FIXTURE, &device)?;
        Ok((model, tensors))
    }

    #[test]
    fn tiny_fixture_matches_all_vision_stages() -> Result<()> {
        let (model, tensors) = tiny_model()?;
        let pixel_values = fixture_tensor(&tensors, "input.pixel_values")?;
        let mask = fixture_tensor(&tensors, "input.pixel_attention_mask")?;
        let shapes = fixture_tensor(&tensors, "input.spatial_shapes")?;
        let inputs = PackedVisionInputs {
            pixel_values,
            pixel_attention_mask: mask,
            spatial_shapes: shapes,
        };
        let stages = model.forward_stages(&inputs)?;
        assert_close(
            &stages.embeddings.patch_embedding,
            fixture_tensor(&tensors, "stage.vision.patch_embedding")?,
            2e-5,
            "patch projection",
        )?;
        assert_close(
            &stages.embeddings.resized_position_embedding,
            fixture_tensor(&tensors, "stage.vision.resized_position_embedding")?,
            2e-5,
            "resized positions",
        )?;
        assert_close(
            &stages.embeddings.embeddings_with_position,
            fixture_tensor(&tensors, "stage.vision.embeddings_with_resized_position")?,
            2e-5,
            "embedding plus positions",
        )?;
        for (index, actual) in stages.encoder_layers.iter().enumerate() {
            assert_close(
                actual,
                fixture_tensor(&tensors, &format!("stage.vision.encoder_layer.{index}"))?,
                2e-5,
                &format!("encoder layer {index}"),
            )?;
        }
        assert_close(
            &stages.post_layernorm,
            fixture_tensor(&tensors, "stage.vision.last_hidden_state")?,
            2e-5,
            "returned post layer norm",
        )?;
        assert_close(
            &stages.post_layernorm,
            fixture_tensor(&tensors, "stage.vision.post_layernorm")?,
            2e-5,
            "post layer norm",
        )?;
        Ok(())
    }

    #[test]
    fn tiny_fixture_repeat_is_deterministic() -> Result<()> {
        let (model, tensors) = tiny_model()?;
        let inputs = PackedVisionInputs {
            pixel_values: fixture_tensor(&tensors, "input.pixel_values")?,
            pixel_attention_mask: fixture_tensor(&tensors, "input.pixel_attention_mask")?,
            spatial_shapes: fixture_tensor(&tensors, "input.spatial_shapes")?,
        };
        let first = model.forward(&inputs)?;
        let second = model.forward(&inputs)?;
        assert_eq!(first.to_vec3::<f32>()?, second.to_vec3::<f32>()?);
        Ok(())
    }

    #[test]
    fn padding_keys_cannot_change_valid_patch_outputs() -> Result<()> {
        let (model, tensors) = tiny_model()?;
        let pixel_values = fixture_tensor(&tensors, "input.pixel_values")?;
        let mask = fixture_tensor(&tensors, "input.pixel_attention_mask")?;
        let shapes = fixture_tensor(&tensors, "input.spatial_shapes")?;
        let inputs = PackedVisionInputs {
            pixel_values,
            pixel_attention_mask: mask,
            spatial_shapes: shapes,
        };
        let baseline = model.forward(&inputs)?;
        let mut altered = pixel_values.to_vec3::<f32>()?;
        for value in altered[0][8..].iter_mut().flatten() {
            *value = 123.0;
        }
        let altered = Tensor::from_vec(
            altered.into_iter().flatten().flatten().collect(),
            (1, 10, 12),
            &Device::Cpu,
        )?;
        let altered_output = model.forward(&PackedVisionInputs {
            pixel_values: &altered,
            pixel_attention_mask: mask,
            spatial_shapes: shapes,
        })?;
        assert_close(
            &baseline.i((.., 0..8, ..))?,
            &altered_output.i((.., 0..8, ..))?,
            1e-6,
            "padding key isolation",
        )?;
        Ok(())
    }

    #[test]
    fn resize_weights_match_pinned_four_to_two_and_four_to_six_oracle() -> Result<()> {
        let expected_down = [
            [3.0 / 7.0, 3.0 / 7.0, 1.0 / 7.0, 0.0],
            [0.0, 1.0 / 7.0, 3.0 / 7.0, 3.0 / 7.0],
        ];
        for (index, expected) in expected_down.into_iter().enumerate() {
            let weights = resize_weights(4, 2, index)?;
            let mut actual = [0f32; 4];
            for offset in 0..weights.indices.len() {
                actual[weights.indices[offset]] = weights.weights[offset];
            }
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert!((actual - expected).abs() <= 1e-6);
            }
        }
        let expected_up = [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 5.0 / 6.0, 1.0 / 6.0, 0.0],
            [0.0, 1.0 / 6.0, 5.0 / 6.0, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ];
        for (index, expected) in expected_up.into_iter().enumerate() {
            let weights = resize_weights(4, 6, index)?;
            let mut actual = [0f32; 4];
            for offset in 0..weights.indices.len() {
                actual[weights.indices[offset]] = weights.weights[offset];
            }
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert!((actual - expected).abs() <= 1e-6);
            }
        }
        Ok(())
    }

    #[test]
    fn resize_composes_pinned_wide_and_tall_shapes() -> Result<()> {
        let width_only: Vec<Vec<f32>> = (0..16).map(|index| vec![(index % 4) as f32]).collect();
        let wide = resize_bilinear_antialias(&width_only, 4, 4, 2, 6, 1)?;
        let expected_wide = [0.0, 0.5, 7.0 / 6.0, 11.0 / 6.0, 2.5, 3.0];
        for row in 0..2 {
            for (column, expected) in expected_wide.into_iter().enumerate() {
                assert!((wide[row * 6 + column] - expected).abs() <= 1e-6);
            }
        }

        let height_only: Vec<Vec<f32>> = (0..16).map(|index| vec![(index / 4) as f32]).collect();
        let tall = resize_bilinear_antialias(&height_only, 4, 4, 6, 2, 1)?;
        let expected_tall = [0.0, 0.5, 7.0 / 6.0, 11.0 / 6.0, 2.5, 3.0];
        for (row, expected) in expected_tall.into_iter().enumerate() {
            for column in 0..2 {
                assert!((tall[row * 2 + column] - expected).abs() <= 1e-6);
            }
        }
        Ok(())
    }

    #[test]
    fn rejects_malformed_packed_inputs_and_config() -> Result<()> {
        let (model, tensors) = tiny_model()?;
        let mask = fixture_tensor(&tensors, "input.pixel_attention_mask")?;
        let shapes = fixture_tensor(&tensors, "input.spatial_shapes")?;
        let wrong_pixels = Tensor::zeros((1, 10, 11), DType::F32, &Device::Cpu)?;
        let error = model.forward(&PackedVisionInputs {
            pixel_values: &wrong_pixels,
            pixel_attention_mask: mask,
            spatial_shapes: shapes,
        });
        assert!(error.is_err());

        let bad_mask = Tensor::ones((1, 10), DType::F32, &Device::Cpu)?;
        let pixels = fixture_tensor(&tensors, "input.pixel_values")?;
        let error = model.forward(&PackedVisionInputs {
            pixel_values: pixels,
            pixel_attention_mask: &bad_mask,
            spatial_shapes: shapes,
        });
        assert!(error.is_err());

        let bad_shapes = Tensor::new(&[[3i64, 4i64]], &Device::Cpu)?;
        let error = model.forward(&PackedVisionInputs {
            pixel_values: pixels,
            pixel_attention_mask: mask,
            spatial_shapes: &bad_shapes,
        });
        assert!(error.is_err());

        let overflowing_shapes = Tensor::new(&[[i64::MAX, 2i64]], &Device::Cpu)?;
        let error = model.forward(&PackedVisionInputs {
            pixel_values: pixels,
            pixel_attention_mask: mask,
            spatial_shapes: &overflowing_shapes,
        });
        assert!(error.is_err());

        let mut nonsquare = tiny_config();
        nonsquare.num_patches = 15;
        assert!(nonsquare.validate().is_err());
        let mut head = tiny_config();
        head.vision_use_head = true;
        assert!(head.validate().is_err());
        Ok(())
    }
}
