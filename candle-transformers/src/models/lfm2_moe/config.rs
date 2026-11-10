use crate::models::lfm2;
use candle::Result;

fn yes() -> bool {
    true
}
fn default_eps() -> f64 {
    1e-5
}
fn default_theta() -> f32 {
    1_000_000.
}
fn default_context() -> usize {
    128_000
}
fn default_conv() -> usize {
    3
}
fn default_dense() -> usize {
    2
}
fn default_topk() -> usize {
    4
}
fn default_scale() -> f64 {
    1.
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_type: Option<String>,
}

/// Explicit MoE widths are not subject to legacy dense-LFM2 FFN adjustment.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_experts: usize,
    #[serde(default = "default_topk")]
    pub num_experts_per_tok: usize,
    #[serde(default = "default_dense")]
    pub num_dense_layers: usize,
    #[serde(default = "default_eps")]
    pub norm_eps: f64,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default = "default_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_context")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_conv", alias = "conv_L_cache")]
    pub conv_l_cache: usize,
    #[serde(default)]
    pub conv_bias: bool,
    pub layer_types: Vec<lfm2::LayerType>,
    #[serde(default = "yes", alias = "tie_embedding")]
    pub tie_word_embeddings: bool,
    #[serde(default = "yes")]
    pub use_expert_bias: bool,
    #[serde(default = "yes")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_scale")]
    pub routed_scaling_factor: f64,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.num_experts == 0
            || self.num_experts > 4096
            || self.num_experts_per_tok == 0
            || self.num_experts_per_tok > self.num_experts
            || self.num_dense_layers > self.num_hidden_layers
            || self.moe_intermediate_size == 0
        {
            candle::bail!(
                "LFM2-MoE requires valid expert count, top-K, dense prefix and expert width"
            )
        }
        if !self.routed_scaling_factor.is_finite()
            || self.routed_scaling_factor <= 0.
            || !self.norm_eps.is_finite()
            || self.norm_eps <= 0.
        {
            candle::bail!("LFM2-MoE scaling and normalization epsilon must be positive and finite")
        }
        if self.conv_bias {
            candle::bail!("LFM2-MoE conv_bias=true is not supported by the shared LFM2 mixer")
        }
        if self
            .rope_parameters
            .as_ref()
            .and_then(|r| r.rope_type.as_deref())
            .is_some_and(|r| r != "default")
        {
            candle::bail!("LFM2-MoE supports default RoPE only")
        }
        for n in [self.bos_token_id, self.eos_token_id].into_iter().flatten() {
            if n as usize >= self.vocab_size {
                candle::bail!("LFM2-MoE special token {n} exceeds vocabulary")
            }
        }
        self.hidden_size
            .checked_mul(self.moe_intermediate_size)
            .and_then(|n| n.checked_mul(self.num_experts))
            .ok_or_else(|| candle::Error::Msg("LFM2-MoE expert dimensions overflow".into()))?;
        self.base().validate()
    }

    pub(crate) fn base(&self) -> lfm2::Config {
        lfm2::Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            norm_eps: self.norm_eps,
            rope_theta: self
                .rope_parameters
                .as_ref()
                .map_or(self.rope_theta, |r| r.rope_theta),
            max_position_embeddings: self.max_position_embeddings,
            conv_l_cache: self.conv_l_cache,
            conv_bias: self.conv_bias,
            layer_types: self.layer_types.clone(),
            tie_embedding: self.tie_word_embeddings,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            use_flash_attn: false,
        }
    }
}
