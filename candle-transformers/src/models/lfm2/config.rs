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
