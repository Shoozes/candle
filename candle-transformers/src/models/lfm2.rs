//! LFM2 (Liquid Foundation Model 2) implementation.
//!
//! LFM2 is a hybrid architecture that combines attention and short convolution layers.
//! See [LiquidAI](https://www.liquid.ai/) for more information.
//!
//! This implementation supports the LFM2ForCausalLM architecture from HuggingFace transformers.

use crate::models::with_tracing::{linear_no_bias as linear, Embedding, Linear, RmsNorm};
use crate::utils::repeat_kv;
use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    FullAttention,
    Conv,
}

#[derive(Debug, Clone)]
pub struct Lfm2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub conv_l_cache: usize,
    pub conv_bias: bool,
    pub layer_types: Vec<LayerType>,
    pub full_attention_layers: Option<Vec<usize>>,
    pub tie_word_embeddings: Option<bool>,
    pub tie_embedding: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub intermediate_size: Option<usize>,
    pub block_ff_dim: Option<usize>,
    pub block_auto_adjust_ff_dim: bool,
    pub block_ffn_dim_multiplier: f64,
    pub block_multiple_of: usize,
}

#[derive(Debug, serde::Deserialize)]
struct Lfm2ConfigSerde {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    num_key_value_heads: usize,
    #[serde(default = "default_norm_eps")]
    norm_eps: f64,
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    rope_parameters: Option<RopeParametersSerde>,
    #[serde(default = "default_max_position_embeddings")]
    max_position_embeddings: usize,
    #[serde(default = "default_conv_l_cache", alias = "conv_L_cache")]
    conv_l_cache: usize,
    #[serde(default)]
    conv_bias: bool,
    #[serde(default)]
    layer_types: Vec<LayerType>,
    #[serde(default, alias = "full_attn_idxs")]
    full_attention_layers: Option<Vec<usize>>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    tie_embedding: Option<bool>,
    #[serde(default)]
    bos_token_id: Option<u32>,
    #[serde(default)]
    eos_token_id: Option<u32>,
    #[serde(default)]
    intermediate_size: Option<usize>,
    #[serde(default)]
    block_ff_dim: Option<usize>,
    #[serde(default = "default_block_auto_adjust_ff_dim")]
    block_auto_adjust_ff_dim: bool,
    #[serde(default = "default_ffn_dim_multiplier")]
    block_ffn_dim_multiplier: f64,
    #[serde(default = "default_block_multiple_of")]
    block_multiple_of: usize,
}

#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
enum RopeParametersSerde {
    Scalar(f32),
    Object(RopeParametersObject),
}

#[derive(Debug, serde::Deserialize)]
struct RopeParametersObject {
    #[serde(default)]
    rope_theta: Option<f32>,
}

impl<'de> serde::Deserialize<'de> for Lfm2Config {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = <Lfm2ConfigSerde as serde::Deserialize>::deserialize(deserializer)?;
        let rope_theta = match raw.rope_parameters {
            Some(RopeParametersSerde::Scalar(value)) => Some(value),
            Some(RopeParametersSerde::Object(value)) => value.rope_theta,
            None => None,
        }
        .or(raw.rope_theta)
        .unwrap_or_else(default_rope_theta);

        // Transformers normalizes the legacy names in __post_init__: a
        // supplied block_ff_dim overrides intermediate_size, and a supplied
        // tie_embedding overrides tie_word_embeddings.
        let intermediate_size = raw.block_ff_dim.or(raw.intermediate_size);
        let tie_embedding = raw
            .tie_embedding
            .or(raw.tie_word_embeddings)
            .unwrap_or_else(default_tie_embedding);

        Ok(Self {
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            norm_eps: raw.norm_eps,
            rope_theta,
            max_position_embeddings: raw.max_position_embeddings,
            conv_l_cache: raw.conv_l_cache,
            conv_bias: raw.conv_bias,
            layer_types: raw.layer_types,
            full_attention_layers: raw.full_attention_layers,
            tie_word_embeddings: raw.tie_word_embeddings,
            tie_embedding,
            bos_token_id: raw.bos_token_id,
            eos_token_id: raw.eos_token_id,
            intermediate_size,
            block_ff_dim: raw.block_ff_dim,
            block_auto_adjust_ff_dim: raw.block_auto_adjust_ff_dim,
            block_ffn_dim_multiplier: raw.block_ffn_dim_multiplier,
            block_multiple_of: raw.block_multiple_of,
        })
    }
}

fn default_num_key_value_heads() -> usize {
    8
}

fn default_norm_eps() -> f64 {
    1e-5
}

fn default_rope_theta() -> f32 {
    1_000_000.0
}

fn default_block_auto_adjust_ff_dim() -> bool {
    true
}

fn default_tie_embedding() -> bool {
    true
}

fn default_max_position_embeddings() -> usize {
    128000
}

fn default_conv_l_cache() -> usize {
    3
}

fn default_ffn_dim_multiplier() -> f64 {
    1.0
}

fn default_block_multiple_of() -> usize {
    256
}

impl Lfm2Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Compute the normalized LFM2 feed-forward width.
    pub fn effective_ffn_dim(&self) -> Result<usize> {
        let mut dim = match self.block_ff_dim.or(self.intermediate_size) {
            Some(dim) => dim,
            None => self
                .hidden_size
                .checked_mul(4)
                .ok_or_else(|| candle::Error::Msg("LFM2 FFN size overflow".into()))?,
        };
        if dim == 0 {
            candle::bail!("LFM2 FFN width cannot be zero")
        }
        if self.block_auto_adjust_ff_dim {
            dim = dim
                .checked_mul(2)
                .ok_or_else(|| candle::Error::Msg("LFM2 FFN size overflow".into()))?
                / 3;
            if !self.block_ffn_dim_multiplier.is_finite() || self.block_ffn_dim_multiplier < 0.0 {
                candle::bail!("block_ffn_dim_multiplier must be finite and non-negative")
            }
            let scaled = (self.block_ffn_dim_multiplier * dim as f64).floor();
            if scaled > usize::MAX as f64 {
                candle::bail!("LFM2 FFN size overflow")
            }
            dim = scaled as usize;
            if self.block_multiple_of == 0 {
                candle::bail!("block_multiple_of cannot be zero")
            }
            dim = dim
                .div_ceil(self.block_multiple_of)
                .checked_mul(self.block_multiple_of)
                .ok_or_else(|| candle::Error::Msg("LFM2 FFN size overflow".into()))?;
        }
        if dim == 0 {
            candle::bail!("normalized LFM2 FFN width cannot be zero")
        }
        Ok(dim)
    }

    pub fn try_into_config(self, use_flash_attn: bool) -> Result<Config> {
        if let Some(full_attention_layers) = &self.full_attention_layers {
            for &layer_idx in full_attention_layers {
                if layer_idx >= self.num_hidden_layers {
                    candle::bail!(
                        "LFM2 full_attention_layers index {layer_idx} is outside num_hidden_layers {}",
                        self.num_hidden_layers
                    )
                }
            }
        }
        let intermediate_size = self.effective_ffn_dim()?;
        let layer_types = if !self.layer_types.is_empty() {
            if self.layer_types.len() != self.num_hidden_layers {
                candle::bail!(
                    "LFM2 layer_types length {} does not match num_hidden_layers {}",
                    self.layer_types.len(),
                    self.num_hidden_layers
                )
            }
            self.layer_types
        } else if let Some(full_attention_layers) = &self.full_attention_layers {
            let mut layer_types = Vec::with_capacity(self.num_hidden_layers);
            for layer_idx in 0..self.num_hidden_layers {
                layer_types.push(if full_attention_layers.contains(&layer_idx) {
                    LayerType::FullAttention
                } else {
                    LayerType::Conv
                });
            }
            layer_types
        } else {
            vec![LayerType::FullAttention; self.num_hidden_layers]
        };
        let config = Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            norm_eps: self.norm_eps,
            rope_theta: self.rope_theta,
            max_position_embeddings: self.max_position_embeddings,
            conv_l_cache: self.conv_l_cache,
            conv_bias: self.conv_bias,
            layer_types,
            tie_embedding: self.tie_embedding,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            use_flash_attn,
        };
        config.validate()?;
        Ok(config)
    }

    /// Convert a validated configuration using the legacy infallible API.
    ///
    /// New callers that need to report malformed configuration input should
    /// use [`Self::try_into_config`]. Existing model examples and integrations
    /// have historically called this method with configuration values that
    /// were already validated by their loader.
    pub fn into_config(self, use_flash_attn: bool) -> Config {
        match self.try_into_config(use_flash_attn) {
            Ok(config) => config,
            Err(err) => panic!("invalid LFM2 configuration: {err}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub conv_l_cache: usize,
    pub conv_bias: bool,
    pub layer_types: Vec<LayerType>,
    pub tie_embedding: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub use_flash_attn: bool,
}

impl Config {
    /// Validate dimensions and limits before constructing tensors or layers.
    ///
    /// `Config` is public for existing integrations, so callers can construct
    /// it without going through [`Lfm2Config::try_into_config`]. Constructors
    /// and caches call this method before using any dimension-derived shape.
    pub fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 {
            candle::bail!("LFM2 vocab_size must be greater than zero")
        }
        if self.hidden_size == 0 {
            candle::bail!("LFM2 hidden_size must be greater than zero")
        }
        if self.num_hidden_layers == 0 {
            candle::bail!("LFM2 num_hidden_layers must be greater than zero")
        }
        if self.num_attention_heads == 0 {
            candle::bail!("LFM2 num_attention_heads must be greater than zero")
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            candle::bail!(
                "LFM2 hidden_size {} must be divisible by num_attention_heads {}",
                self.hidden_size,
                self.num_attention_heads
            )
        }
        let head_dim = self.hidden_size / self.num_attention_heads;
        if head_dim == 0 || head_dim % 2 != 0 {
            candle::bail!("LFM2 attention head dimension {head_dim} must be a positive even number")
        }
        if self.num_key_value_heads == 0 {
            candle::bail!("LFM2 num_key_value_heads must be greater than zero")
        }
        if self.num_key_value_heads > self.num_attention_heads {
            candle::bail!(
                "LFM2 num_key_value_heads {} cannot exceed num_attention_heads {}",
                self.num_key_value_heads,
                self.num_attention_heads
            )
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            candle::bail!(
                "LFM2 num_attention_heads {} must be divisible by num_key_value_heads {}",
                self.num_attention_heads,
                self.num_key_value_heads
            )
        }
        if self.layer_types.len() != self.num_hidden_layers {
            candle::bail!(
                "LFM2 layer_types length {} does not match num_hidden_layers {}",
                self.layer_types.len(),
                self.num_hidden_layers
            )
        }
        if self.intermediate_size == 0 {
            candle::bail!("LFM2 intermediate_size must be greater than zero")
        }
        if self.conv_l_cache == 0 {
            candle::bail!("LFM2 conv_l_cache must be greater than zero")
        }
        if self.max_position_embeddings == 0 {
            candle::bail!("LFM2 max_position_embeddings must be greater than zero")
        }
        if self.max_position_embeddings > u32::MAX as usize {
            candle::bail!(
                "LFM2 max_position_embeddings {} exceeds the supported u32 position range",
                self.max_position_embeddings
            )
        }
        if !self.norm_eps.is_finite() || self.norm_eps <= 0.0 {
            candle::bail!("LFM2 norm_eps must be finite and greater than zero")
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            candle::bail!("LFM2 rope_theta must be finite and greater than zero")
        }
        self.hidden_size
            .checked_mul(3)
            .ok_or_else(|| candle::Error::Msg("LFM2 projection width overflow".into()))?;
        self.vocab_size
            .checked_mul(self.hidden_size)
            .ok_or_else(|| candle::Error::Msg("LFM2 embedding shape overflow".into()))?;
        self.hidden_size
            .checked_mul(self.intermediate_size)
            .ok_or_else(|| candle::Error::Msg("LFM2 FFN shape overflow".into()))?;
        self.max_position_embeddings
            .checked_mul(head_dim.div_ceil(2))
            .ok_or_else(|| candle::Error::Msg("LFM2 rotary cache shape overflow".into()))?;
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Cache for LFM2 model supporting both attention KV cache and convolution state cache.
#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<(usize, usize), Tensor>,
    pub use_kv_cache: bool,
    // KV cache for attention layers: (key, value) per layer
    kvs: Vec<Option<(Tensor, Tensor)>>,
    // Conv state cache for convolution layers
    conv_states: Vec<Option<Tensor>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.head_dim();
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

impl Cache {
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        config.validate()?;
        let theta = calculate_default_inv_freq(config);
        let theta = Tensor::new(theta, device)?;

        let max_position_embeddings = config.max_position_embeddings as u32;
        let idx_theta = Tensor::arange(0u32, max_position_embeddings, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;

        let num_layers = config.num_hidden_layers;
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; num_layers],
            conv_states: vec![None; num_layers],
            device: device.clone(),
            cos,
            sin,
        })
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize) -> Result<Tensor> {
        let kv_len = index_pos
            .checked_add(seq_len)
            .ok_or_else(|| candle::Error::Msg("LFM2 sequence position overflow".into()))?;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            Ok(mask.clone())
        } else {
            let mask = crate::utils::build_causal_mask(seq_len, index_pos, &self.device)?;
            self.masks.insert((seq_len, kv_len), mask.clone());
            Ok(mask)
        }
    }

    pub fn clear(&mut self) {
        self.masks.clear();
        self.kvs.iter_mut().for_each(|v| *v = None);
        self.conv_states.iter_mut().for_each(|v| *v = None);
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    candle::bail!(
        "LFM2 flash attention was requested, but candle-transformers was built without the 'flash-attn' feature"
    )
}

/// MLP layer with SwiGLU activation.
#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    span: tracing::Span,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;
        // LFM2 uses w1 (gate), w3 (up), w2 (down) naming convention
        let gate_proj = linear(hidden_size, intermediate_size, vb.pp("w1"))?;
        let up_proj = linear(hidden_size, intermediate_size, vb.pp("w3"))?;
        let down_proj = linear(intermediate_size, hidden_size, vb.pp("w2"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// Attention layer with per-head QK normalization and RoPE.
#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    use_flash_attn: bool,
    span: tracing::Span,
    span_rot: tracing::Span,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_attention_heads = cfg.num_attention_heads;
        let num_key_value_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();

        let q_proj = linear(hidden_size, num_attention_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_size, num_key_value_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_size, num_key_value_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear(
            num_attention_heads * head_dim,
            hidden_size,
            vb.pp("out_proj"),
        )?;

        let q_norm = RmsNorm::new(head_dim, cfg.norm_eps, vb.pp("q_layernorm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.norm_eps, vb.pp("k_layernorm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            use_flash_attn: cfg.use_flash_attn,
            span: tracing::span!(tracing::Level::TRACE, "attn"),
            span_rot: tracing::span!(tracing::Level::TRACE, "attn-rot"),
        })
    }

    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &Cache) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_, _, seq_len, _) = x.dims4()?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to (batch, seq, num_heads, head_dim) then transpose to (batch, num_heads, seq, head_dim)
        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply per-head QK normalization
        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;

        // Apply rotary embeddings
        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let k = self.apply_rotary_emb(&k, index_pos, cache)?;

        // Handle KV cache
        let (k, v) = if cache.use_kv_cache {
            match &cache.kvs[block_idx] {
                Some((k_cache, v_cache)) if index_pos > 0 => {
                    let k = Tensor::cat(&[k_cache, &k], 2)?.contiguous()?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?.contiguous()?;
                    (k, v)
                }
                _ => (k, v),
            }
        } else {
            (k, v)
        };

        if cache.use_kv_cache {
            cache.kvs[block_idx] = Some((k.clone(), v.clone()));
        }

        // Expand KV heads to match query heads
        let k = repeat_kv(k, self.num_attention_heads / self.num_key_value_heads)?;
        let v = repeat_kv(v, self.num_attention_heads / self.num_key_value_heads)?;

        let y = if self.use_flash_attn {
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?.transpose(1, 2)?
        } else {
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = if seq_len == 1 {
                att
            } else {
                let mask = cache.mask(seq_len, index_pos)?.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, f32::NEG_INFINITY)?
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?
        };

        let y = y.transpose(1, 2)?.reshape((
            b_sz,
            seq_len,
            self.num_attention_heads * self.head_dim,
        ))?;
        self.o_proj.forward(&y)
    }
}

/// Short convolution layer for efficient sequence processing.
#[derive(Debug, Clone)]
struct ShortConv {
    in_proj: Linear,
    out_proj: Linear,
    conv_weight: Tensor,
    l_cache: usize,
    hidden_size: usize,
    span: tracing::Span,
}

impl ShortConv {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let l_cache = cfg.conv_l_cache;

        // in_proj projects to 3 * hidden_size for B, C, X components
        let in_proj = linear(hidden_size, 3 * hidden_size, vb.pp("in_proj"))?;
        let out_proj = linear(hidden_size, hidden_size, vb.pp("out_proj"))?;

        // Conv weight shape: (hidden_size, 1, l_cache) or (hidden_size, l_cache)
        let conv_weight = vb.get((hidden_size, 1, l_cache), "conv.weight")?;

        Ok(Self {
            in_proj,
            out_proj,
            conv_weight,
            l_cache,
            hidden_size,
            span: tracing::span!(tracing::Level::TRACE, "shortconv"),
        })
    }

    fn forward(&self, x: &Tensor, block_idx: usize, cache: &mut Cache) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, _) = x.dims3()?;

        // Project input to B, C, X components
        let bcx = self.in_proj.forward(x)?.transpose(1, 2)?;
        let b = bcx.narrow(1, 0, self.hidden_size)?;
        let c = bcx.narrow(1, self.hidden_size, self.hidden_size)?;
        let x_proj = bcx.narrow(1, 2 * self.hidden_size, self.hidden_size)?;

        // Element-wise multiply B and X
        let bx = (b * &x_proj)?.contiguous()?;

        // Prepare conv weight: squeeze to (hidden_size, l_cache) for element-wise, or keep for Conv1d
        let conv_weight = self.conv_weight.squeeze(1)?;

        let conv_out = if seq_len == 1 {
            // Token-by-token generation: use cached state
            let mut state = match &cache.conv_states[block_idx] {
                Some(s) => s.clone(),
                None => Tensor::zeros(
                    (b_sz, self.hidden_size, self.l_cache),
                    bx.dtype(),
                    bx.device(),
                )?,
            };

            // Shift cache and add new token
            if self.l_cache > 1 {
                let tail = state.narrow(2, 1, self.l_cache - 1)?;
                state = Tensor::cat(&[tail, bx.clone()], 2)?;
            } else {
                state = bx.clone();
            }

            if cache.use_kv_cache {
                cache.conv_states[block_idx] = Some(state.clone());
            }

            // Apply convolution as element-wise multiply and sum
            (state * conv_weight.unsqueeze(0)?)?
                .sum_keepdim(2)?
                .contiguous()?
        } else {
            // Prefill: use Conv1d
            let conv = Conv1d::new(
                self.conv_weight.clone(),
                None,
                Conv1dConfig {
                    padding: self.l_cache.saturating_sub(1),
                    groups: self.hidden_size,
                    ..Default::default()
                },
            );
            let mut out = conv.forward(&bx)?;
            out = out.narrow(2, 0, seq_len)?;

            // Update cache with last l_cache tokens
            if cache.use_kv_cache && self.l_cache > 0 {
                let start = seq_len.saturating_sub(self.l_cache);
                let cache_len = seq_len - start;
                let mut cache_src = bx.narrow(2, start, cache_len)?;
                if cache_len < self.l_cache {
                    let pad = self.l_cache - cache_len;
                    let zeros = Tensor::zeros(
                        (b_sz, self.hidden_size, pad),
                        cache_src.dtype(),
                        cache_src.device(),
                    )?;
                    cache_src = Tensor::cat(&[zeros, cache_src], 2)?;
                }
                cache.conv_states[block_idx] = Some(cache_src);
            }

            out
        };

        // Multiply by C and project output
        let conv_out = (c * &conv_out)?;
        let conv_out = conv_out.transpose(1, 2)?.contiguous()?;
        self.out_proj.forward(&conv_out)
    }
}

/// Unified decoder layer supporting both attention and convolution.
#[derive(Debug, Clone)]
enum LayerKind {
    Attention(Box<Attention>),
    ShortConv(ShortConv),
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
    kind: LayerKind,
    span: tracing::Span,
}

impl DecoderLayer {
    fn new(cfg: &Config, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        // LFM2 uses operator_norm and ffn_norm naming
        let input_layernorm = RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("operator_norm"))?;
        let post_attention_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("ffn_norm"))?;
        // LFM2 uses feed_forward naming for MLP
        let mlp = Mlp::new(cfg, vb.pp("feed_forward"))?;

        let layer_type = cfg
            .layer_types
            .get(layer_idx)
            .copied()
            .unwrap_or(LayerType::FullAttention);
        let kind = match layer_type {
            LayerType::FullAttention => {
                LayerKind::Attention(Box::new(Attention::new(cfg, vb.pp("self_attn"))?))
            }
            LayerType::Conv => LayerKind::ShortConv(ShortConv::new(cfg, vb.pp("conv"))?),
        };

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            mlp,
            kind,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.input_layernorm.forward(x)?;

        let x = match &self.kind {
            LayerKind::Attention(attn) => attn.forward(&x, index_pos, block_idx, cache)?,
            LayerKind::ShortConv(conv) => conv.forward(&x, block_idx, cache)?,
        };

        let x = (x + residual)?;
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x + residual
    }
}

/// LFM2 model for causal language modeling.
#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    embedding_norm: RmsNorm,
    lm_head: Linear,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Self::new_from_parts(cfg, vb.pp("model"), Some(vb.pp("lm_head")))
    }

    /// Construct from the direct language-model variable root.
    ///
    /// `Model::new` is the standalone loader for checkpoints rooted at
    /// `model.*`. Nested multimodal checkpoints should pass
    /// `model.language_model` here and provide the separate `lm_head` root
    /// only when embeddings are not tied.
    pub fn new_from_parts(
        cfg: &Config,
        vb_m: VarBuilder,
        lm_head_vb: Option<VarBuilder>,
    ) -> Result<Self> {
        cfg.validate()?;
        let embed_tokens =
            Embedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(cfg, layer_idx, vb_l.pp(layer_idx))?;
            layers.push(layer);
        }

        let embedding_norm =
            RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb_m.pp("embedding_norm"))?;

        let lm_head = if cfg.tie_embedding {
            Linear::from_weights(embed_tokens.embeddings().clone(), None)
        } else {
            let lm_head_vb = match lm_head_vb {
                Some(lm_head_vb) => lm_head_vb,
                None => candle::bail!("untied LFM2 configuration requires an lm_head root"),
            };
            linear(cfg.hidden_size, cfg.vocab_size, lm_head_vb)?
        };

        Ok(Self {
            embed_tokens,
            layers,
            embedding_norm,
            lm_head,
            dtype: vb_m.dtype(),
        })
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn device(&self) -> &Device {
        self.embed_tokens.embeddings().device()
    }

    pub fn forward_hidden(
        &self,
        input_embeds: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (_, seq_len, _) = input_embeds.dims3()?;
        if seq_len == 0 {
            candle::bail!("LFM2 cannot forward an empty sequence")
        }
        let end_pos = index_pos
            .checked_add(seq_len)
            .ok_or_else(|| candle::Error::Msg("LFM2 sequence position overflow".into()))?;
        let max_position_embeddings = cache.cos.dim(0)?;
        if end_pos > max_position_embeddings {
            candle::bail!(
                "LFM2 sequence positions [{index_pos}, {end_pos}) exceed max_position_embeddings {max_position_embeddings}"
            )
        }

        let mut hidden_states = input_embeds.clone();
        for (block_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, index_pos, block_idx, cache)?;
        }
        self.embedding_norm.forward(&hidden_states)
    }

    pub fn project_logits(&self, hidden_states: &Tensor, logits_to_keep: usize) -> Result<Tensor> {
        let (_, seq_len, _) = hidden_states.dims3()?;
        if seq_len == 0 {
            candle::bail!("LFM2 cannot project logits for an empty sequence")
        }
        let keep = if logits_to_keep == 0 {
            seq_len
        } else {
            logits_to_keep
        };
        if keep > seq_len {
            candle::bail!("cannot keep {keep} LFM2 logits from a sequence of length {seq_len}")
        }
        let hidden_states = hidden_states.narrow(1, seq_len - keep, keep)?;
        self.lm_head.forward(&hidden_states)?.to_dtype(DType::F32)
    }

    pub fn forward_embeds(
        &self,
        input_embeds: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let hidden_states = self.forward_hidden(input_embeds, index_pos, cache)?;
        let logits = self.project_logits(&hidden_states, 1)?;
        logits.i((.., 0, ..))?.contiguous()
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        input_ids.dims2()?;
        let input_embeds = self.embed_tokens(input_ids)?;
        self.forward_embeds(&input_embeds, index_pos, cache)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device, IndexOp, Tensor};
    use candle_nn::VarBuilder;

    const TINY_FIXTURE: &[u8] =
        include_bytes!("../../../tests/fixtures/lfm2_vl_tiny/tensors.safetensors");

    fn tiny_config(tie_embedding: bool) -> Config {
        Config {
            vocab_size: 32,
            hidden_size: 12,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 3,
            num_key_value_heads: 1,
            norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            conv_l_cache: 3,
            conv_bias: false,
            layer_types: vec![LayerType::Conv, LayerType::FullAttention],
            tie_embedding,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            use_flash_attn: false,
        }
    }

    #[cfg(not(feature = "flash-attn"))]
    #[test]
    fn flash_attention_without_feature_returns_an_error() -> Result<()> {
        let tensor = Tensor::zeros((1, 1, 1, 1), DType::F32, &Device::Cpu)?;
        let err = flash_attn(&tensor, &tensor, &tensor, 1.0, false).unwrap_err();
        assert!(
            err.to_string()
                .contains("built without the 'flash-attn' feature"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32, label: &str) -> Result<()> {
        let max_abs = (actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        eprintln!("lfm2 {label}: max absolute error {max_abs:.8e}");
        assert!(
            max_abs <= tolerance,
            "{label}: max absolute error {max_abs} exceeds {tolerance}"
        );
        Ok(())
    }

    fn parse_config(value: serde_json::Value) -> Result<Lfm2Config> {
        serde_json::from_value(value)
            .map_err(|err| candle::Error::Msg(format!("invalid test LFM2 config: {err}")))
    }

    fn add_tiny_language_weights(
        weights: &mut HashMap<String, Tensor>,
        prefix: &str,
        cfg: &Config,
        device: &Device,
    ) -> Result<()> {
        let zero = |shape: &[usize]| Tensor::zeros(shape, DType::F32, device);
        weights.insert(
            format!("{prefix}.embed_tokens.weight"),
            zero(&[cfg.vocab_size, cfg.hidden_size])?,
        );
        weights.insert(
            format!("{prefix}.embedding_norm.weight"),
            Tensor::ones(cfg.hidden_size, DType::F32, device)?,
        );
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer_prefix = format!("{prefix}.layers.{layer_idx}");
            weights.insert(
                format!("{layer_prefix}.operator_norm.weight"),
                Tensor::ones(cfg.hidden_size, DType::F32, device)?,
            );
            weights.insert(
                format!("{layer_prefix}.ffn_norm.weight"),
                Tensor::ones(cfg.hidden_size, DType::F32, device)?,
            );
            weights.insert(
                format!("{layer_prefix}.feed_forward.w1.weight"),
                zero(&[cfg.intermediate_size, cfg.hidden_size])?,
            );
            weights.insert(
                format!("{layer_prefix}.feed_forward.w2.weight"),
                zero(&[cfg.hidden_size, cfg.intermediate_size])?,
            );
            weights.insert(
                format!("{layer_prefix}.feed_forward.w3.weight"),
                zero(&[cfg.intermediate_size, cfg.hidden_size])?,
            );
            match cfg.layer_types[layer_idx] {
                LayerType::Conv => {
                    weights.insert(
                        format!("{layer_prefix}.conv.in_proj.weight"),
                        zero(&[3 * cfg.hidden_size, cfg.hidden_size])?,
                    );
                    weights.insert(
                        format!("{layer_prefix}.conv.out_proj.weight"),
                        zero(&[cfg.hidden_size, cfg.hidden_size])?,
                    );
                    weights.insert(
                        format!("{layer_prefix}.conv.conv.weight"),
                        zero(&[cfg.hidden_size, 1, cfg.conv_l_cache])?,
                    );
                }
                LayerType::FullAttention => {
                    let head_dim = cfg.head_dim();
                    weights.insert(
                        format!("{layer_prefix}.self_attn.q_proj.weight"),
                        zero(&[cfg.num_attention_heads * head_dim, cfg.hidden_size])?,
                    );
                    weights.insert(
                        format!("{layer_prefix}.self_attn.k_proj.weight"),
                        zero(&[cfg.num_key_value_heads * head_dim, cfg.hidden_size])?,
                    );
                    weights.insert(
                        format!("{layer_prefix}.self_attn.v_proj.weight"),
                        zero(&[cfg.num_key_value_heads * head_dim, cfg.hidden_size])?,
                    );
                    weights.insert(
                        format!("{layer_prefix}.self_attn.out_proj.weight"),
                        zero(&[cfg.hidden_size, cfg.num_attention_heads * head_dim])?,
                    );
                    weights.insert(
                        format!("{layer_prefix}.self_attn.q_layernorm.weight"),
                        Tensor::ones(head_dim, DType::F32, device)?,
                    );
                    weights.insert(
                        format!("{layer_prefix}.self_attn.k_layernorm.weight"),
                        Tensor::ones(head_dim, DType::F32, device)?,
                    );
                }
            }
        }
        Ok(())
    }

    fn official_text_config(hidden_size: usize, intermediate_size: usize) -> serde_json::Value {
        serde_json::json!({
            "model_type": "lfm2",
            "vocab_size": 65536,
            "hidden_size": hidden_size,
            "num_hidden_layers": 16,
            "num_attention_heads": if hidden_size == 1024 { 16 } else { 32 },
            "num_key_value_heads": 8,
            "intermediate_size": intermediate_size,
            "block_ff_dim": intermediate_size,
            "block_auto_adjust_ff_dim": true,
            "block_ffn_dim_multiplier": 1.0,
            "block_multiple_of": 256,
            "conv_l_cache": 3,
            "full_attn_idxs": [2, 5, 8, 10, 12, 14],
            "rope_parameters": {"rope_theta": 1_000_000.0, "rope_type": "default"}
        })
    }

    #[test]
    fn config_aliases_and_official_ffn_widths() -> Result<()> {
        let legacy = parse_config(serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
            "layer_types": ["conv", "full_attention"],
            "intermediate_size": 64,
            "block_auto_adjust_ff_dim": false,
            "block_ffn_dim_multiplier": 1.0,
            "block_multiple_of": 16,
            "conv_L_cache": 5,
            "tie_embedding": false,
            "rope_theta": 42.0
        }))?;
        assert_eq!(legacy.conv_l_cache, 5);
        assert_eq!(legacy.intermediate_size, Some(64));
        assert_eq!(legacy.block_ff_dim, None);
        assert!(!legacy.tie_embedding);
        assert_eq!(legacy.rope_theta, 42.0);
        assert_eq!(legacy.effective_ffn_dim()?, 64);

        let precedence = parse_config(serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
            "layer_types": ["conv", "full_attention"],
            "intermediate_size": 4096,
            "block_ff_dim": 6656,
            "block_auto_adjust_ff_dim": true,
            "block_ffn_dim_multiplier": 1.0,
            "block_multiple_of": 256,
            "conv_l_cache": 7,
            "tie_word_embeddings": true,
            "tie_embedding": false,
            "rope_theta": 123.0,
            "rope_parameters": {"rope_theta": 456.0}
        }))?;
        assert_eq!(precedence.conv_l_cache, 7);
        assert_eq!(precedence.intermediate_size, Some(6656));
        assert_eq!(precedence.block_ff_dim, Some(6656));
        assert!(!precedence.tie_embedding);
        assert_eq!(precedence.rope_theta, 456.0);
        assert_eq!(precedence.effective_ffn_dim()?, 4608);

        let official_450 = parse_config(official_text_config(1024, 6656))?;
        assert_eq!(official_450.effective_ffn_dim()?, 4608);
        assert_eq!(official_450.tie_word_embeddings, None);
        let official_450_normalized = official_450.try_into_config(false)?;
        assert_eq!(official_450_normalized.intermediate_size, 4608);
        assert_eq!(official_450_normalized.layer_types.len(), 16);
        assert_eq!(
            official_450_normalized.layer_types[2],
            LayerType::FullAttention
        );
        assert_eq!(official_450_normalized.layer_types[1], LayerType::Conv);

        let official_16 = parse_config(official_text_config(2048, 12288))?;
        assert_eq!(official_16.effective_ffn_dim()?, 8192);
        assert!(official_16.tie_embedding);
        assert_eq!(official_16.try_into_config(false)?.intermediate_size, 8192);

        let current_untied = parse_config(serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
            "block_ff_dim": 32,
            "block_auto_adjust_ff_dim": false,
            "tie_word_embeddings": false
        }))?;
        assert_eq!(current_untied.tie_word_embeddings, Some(false));
        assert!(!current_untied.tie_embedding);

        let full_attention_alias = parse_config(serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
            "block_ff_dim": 32,
            "block_auto_adjust_ff_dim": false,
            "full_attn_idxs": [1]
        }))?;
        assert_eq!(
            full_attention_alias.try_into_config(false)?.layer_types,
            vec![LayerType::Conv, LayerType::FullAttention]
        );

        let layer_type_precedence = parse_config(serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
            "block_ff_dim": 32,
            "block_auto_adjust_ff_dim": false,
            "full_attn_idxs": [1],
            "layer_types": ["conv", "conv"]
        }))?;
        assert_eq!(
            layer_type_precedence.try_into_config(false)?.layer_types,
            vec![LayerType::Conv, LayerType::Conv]
        );

        Ok(())
    }

    #[test]
    fn missing_ffn_uses_legacy_hidden_width_fallback() -> Result<()> {
        let config = parse_config(serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 12,
            "num_hidden_layers": 1,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
            "block_auto_adjust_ff_dim": false
        }))?;
        assert_eq!(config.intermediate_size, None);
        assert_eq!(config.effective_ffn_dim()?, 48);
        assert!(config.tie_embedding);
        Ok(())
    }

    #[test]
    fn malformed_dimensions_are_rejected_before_model_construction() -> Result<()> {
        let base = serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
            "layer_types": ["conv", "full_attention"],
            "block_ff_dim": 32,
            "block_auto_adjust_ff_dim": false,
            "conv_l_cache": 3
        });
        let cases = [
            (
                "num_attention_heads",
                serde_json::json!(0),
                "num_attention_heads must be greater than zero",
            ),
            (
                "num_key_value_heads",
                serde_json::json!(2),
                "must be divisible by num_key_value_heads",
            ),
            (
                "hidden_size",
                serde_json::json!(15),
                "attention head dimension 5 must be a positive even number",
            ),
            (
                "conv_l_cache",
                serde_json::json!(0),
                "conv_l_cache must be greater than zero",
            ),
            (
                "max_position_embeddings",
                serde_json::json!(0),
                "max_position_embeddings must be greater than zero",
            ),
            (
                "norm_eps",
                serde_json::json!(0.0),
                "norm_eps must be finite and greater than zero",
            ),
            (
                "rope_theta",
                serde_json::json!(0.0),
                "rope_theta must be finite and greater than zero",
            ),
        ];
        for (field, value, expected) in cases {
            let mut raw = base.clone();
            raw[field] = value;
            let config = parse_config(raw)?;
            let error = config.try_into_config(false).unwrap_err().to_string();
            assert!(
                error.contains(expected),
                "{field}: expected error containing {expected:?}, got {error}"
            );
        }

        let mut out_of_range = base;
        out_of_range["full_attn_idxs"] = serde_json::json!([2]);
        let config = parse_config(out_of_range)?;
        let error = config.try_into_config(false).unwrap_err().to_string();
        assert!(error.contains("full_attention_layers index 2"));
        Ok(())
    }

    #[test]
    fn cache_rejects_unrepresentable_positions_and_index_overflow() -> Result<()> {
        let device = Device::Cpu;
        let mut invalid = tiny_config(true);
        invalid.max_position_embeddings = (u32::MAX as usize).saturating_add(1);
        let error = Cache::new(true, DType::F32, &invalid, &device)
            .unwrap_err()
            .to_string();
        assert!(error.contains("supported u32 position range"), "{error}");

        let mut cache = Cache::new(true, DType::F32, &tiny_config(true), &device)?;
        let error = cache.mask(1, usize::MAX).unwrap_err().to_string();
        assert!(error.contains("sequence position overflow"), "{error}");
        Ok(())
    }

    #[test]
    fn constructors_support_standalone_nested_and_explicit_heads() -> Result<()> {
        let device = Device::Cpu;
        let tied = tiny_config(true);
        let mut standalone_weights = HashMap::new();
        add_tiny_language_weights(&mut standalone_weights, "model", &tied, &device)?;
        let _standalone = Model::new(
            &tied,
            VarBuilder::from_tensors(standalone_weights, DType::F32, &device),
        )?;

        let mut nested_weights = HashMap::new();
        add_tiny_language_weights(&mut nested_weights, "model.language_model", &tied, &device)?;
        let nested_vb = VarBuilder::from_tensors(nested_weights, DType::F32, &device);
        let _nested =
            Model::new_from_parts(&tied, nested_vb.pp("model").pp("language_model"), None)?;

        let untied = tiny_config(false);
        let mut explicit_weights = HashMap::new();
        add_tiny_language_weights(&mut explicit_weights, "model", &untied, &device)?;
        explicit_weights.insert(
            "lm_head.weight".to_string(),
            Tensor::zeros((untied.vocab_size, untied.hidden_size), DType::F32, &device)?,
        );
        let _explicit = Model::new(
            &untied,
            VarBuilder::from_tensors(explicit_weights, DType::F32, &device),
        )?;
        Ok(())
    }

    #[test]
    fn fixture_proves_dense_embedding_and_cached_decode_parity() -> Result<()> {
        let device = Device::Cpu;
        let weights_vb = VarBuilder::from_slice_safetensors(TINY_FIXTURE, DType::F32, &device)?;
        let fixture_tensors = candle::safetensors::load_buffer(TINY_FIXTURE, &device)?;
        let fixture_tensor = |name: &str| match fixture_tensors.get(name) {
            Some(tensor) => Ok(tensor.clone()),
            None => candle::bail!("missing tiny fixture tensor {name}"),
        };
        let model_vb = weights_vb.pp("weights").pp("model").pp("language_model");
        let model = Model::new_from_parts(&tiny_config(true), model_vb, None)?;

        let input_ids = fixture_tensor("input.input_ids")?;
        let expected_embeddings = fixture_tensor("stage.text.embeddings")?;
        let embeddings = model.embed_tokens(&input_ids)?;
        assert_close(&embeddings, &expected_embeddings, 1e-6, "token embeddings")?;

        let merged_embeddings = fixture_tensor("stage.multimodal.merged_embeddings")?;
        let expected_hidden = fixture_tensor("stage.language.hidden_states")?;
        let expected_prefill = fixture_tensor("stage.language.prefill_logits")?;

        let mut token_cache = Cache::new(true, DType::F32, &tiny_config(true), &device)?;
        let token_logits = model.forward(&input_ids, 0, &mut token_cache)?;
        let mut embed_cache = Cache::new(true, DType::F32, &tiny_config(true), &device)?;
        let embed_logits = model.forward_embeds(&embeddings, 0, &mut embed_cache)?;
        assert_close(
            &token_logits,
            &embed_logits,
            1e-6,
            "token-ID vs embedding-driven forwarding",
        )?;

        let mut parity_cache = Cache::new(true, DType::F32, &tiny_config(true), &device)?;
        let hidden = model.forward_hidden(&merged_embeddings, 0, &mut parity_cache)?;
        assert_close(&hidden, &expected_hidden, 1e-3, "prefill hidden states")?;
        let prefill_logits = model.project_logits(&hidden, 0)?;
        assert_close(&prefill_logits, &expected_prefill, 1e-3, "prefill logits")?;

        let decode_ids = fixture_tensor("input.decode_token_ids")?;
        let expected_decode = fixture_tensor("stage.language.decode_logits")?;
        for step in 0..3 {
            let token = decode_ids.i((.., step..step + 1))?;
            let logits = model.forward(&token, 5 + step, &mut parity_cache)?;
            let expected = expected_decode.i((.., step, ..))?;
            assert_close(&logits, &expected, 1e-3, "cached decode logits")?;
        }

        parity_cache.clear();
        let hidden_after_reset = model.forward_hidden(&merged_embeddings, 0, &mut parity_cache)?;
        assert_close(
            &hidden_after_reset,
            &expected_hidden,
            1e-3,
            "cache-reset prefill hidden states",
        )?;
        let reset_decode = model.forward(&decode_ids.i((.., 0..1))?, 5, &mut parity_cache)?;
        assert_close(
            &reset_decode,
            &expected_decode.i((.., 0, ..))?,
            1e-3,
            "cache-reset decode logits",
        )?;

        Ok(())
    }
}
