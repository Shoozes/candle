use candle::Result;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    Conv,
    FullAttention,
    SlidingAttention,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct RopeParameters {
    pub rope_type: String,
    pub rope_theta: f64,
    #[serde(default = "one")]
    pub partial_rotary_factor: f64,
}

fn one() -> f64 {
    1.0
}
fn yes() -> bool {
    true
}

/// Released LFM3 dimensions. FFN width is used directly, without LFM2 rounding.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub conv_bias: bool,
    #[serde(rename = "conv_L_cache")]
    pub conv_kernel_size: usize,
    pub layer_types: Vec<LayerType>,
    pub sliding_window: usize,
    pub use_sandwich_norm: bool,
    pub rope_parameters: HashMap<String, RopeParameters>,
    #[serde(default = "yes")]
    pub is_causal: bool,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if [
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.max_position_embeddings,
            self.conv_kernel_size,
        ]
        .contains(&0)
        {
            candle::bail!("LFM3 dimensions and context limits must be positive")
        }
        if !self.head_dim.is_multiple_of(2)
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            candle::bail!("LFM3 requires even head_dim and query heads divisible by KV heads")
        }
        if self.layer_types.len() != self.num_hidden_layers {
            candle::bail!("LFM3 layer_types length must equal num_hidden_layers")
        }
        if !(self.norm_eps as f32).is_finite() || (self.norm_eps as f32) <= 0.0 {
            candle::bail!("LFM3 norm_eps must be finite and positive")
        }
        if self.max_position_embeddings > u32::MAX as usize {
            candle::bail!("LFM3 context exceeds supported u32 positions")
        }
        if self.layer_types.contains(&LayerType::SlidingAttention) && self.sliding_window == 0 {
            candle::bail!("LFM3 sliding attention requires a positive window")
        }
        if !self.is_causal && self.layer_types.contains(&LayerType::Conv) {
            candle::bail!("bidirectional LFM3 currently supports attention-only layers")
        }
        for (kind, name) in [
            (LayerType::FullAttention, "full_attention"),
            (LayerType::SlidingAttention, "sliding_attention"),
        ] {
            if self.layer_types.contains(&kind) {
                let rope = self.rope_parameters.get(name).ok_or_else(|| {
                    candle::Error::Msg(format!("missing LFM3 {name} RoPE parameters"))
                })?;
                if rope.rope_type != "default" {
                    candle::bail!(
                        "unsupported LFM3 RoPE type {}; expected default",
                        rope.rope_type
                    )
                }
                if !(rope.rope_theta as f32).is_finite()
                    || (rope.rope_theta as f32) <= 0.0
                    || !rope.partial_rotary_factor.is_finite()
                    || rope.partial_rotary_factor <= 0.0
                    || rope.partial_rotary_factor > 1.0
                {
                    candle::bail!("invalid LFM3 {name} RoPE frequency parameters")
                }
            }
        }
        for shape in [
            vec![self.vocab_size, self.hidden_size],
            vec![self.hidden_size, self.intermediate_size],
            vec![self.hidden_size, 3, self.hidden_size],
            vec![self.hidden_size, self.conv_kernel_size],
            vec![self.num_attention_heads, self.head_dim, self.hidden_size],
            vec![self.num_key_value_heads, self.head_dim, self.hidden_size],
        ] {
            shape
                .iter()
                .try_fold(1usize, |n, d| n.checked_mul(*d))
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| candle::Error::Msg("LFM3 tensor byte size overflow".into()))?;
        }
        Ok(())
    }

    pub(crate) fn checked_cache_elements(&self, length: usize) -> Result<usize> {
        [
            length,
            self.num_key_value_heads,
            self.head_dim,
            self.num_hidden_layers,
            2,
            4,
        ]
        .into_iter()
        .try_fold(1usize, |n, d| n.checked_mul(d))
        .ok_or_else(|| candle::Error::Msg("LFM3 cache byte size overflow".into()))
    }
}
