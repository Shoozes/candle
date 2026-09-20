use candle::{bail, Result};
use serde::Deserialize;
use std::path::Path;

/// Configuration values used by the official GPT-OSS reference model.
///
/// The defaults describe the published 20B/120B family, but callers should
/// treat them as compatibility defaults only and validate any checkpoint
/// configuration before constructing a model.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct GptOssConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_experts")]
    pub num_experts: usize,
    #[serde(default = "default_experts_per_token")]
    pub experts_per_token: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_swiglu_limit")]
    pub swiglu_limit: f32,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    #[serde(default = "default_initial_context_length")]
    pub initial_context_length: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_rope_scaling_factor")]
    pub rope_scaling_factor: f32,
    #[serde(default = "default_rope_ntk_alpha")]
    pub rope_ntk_alpha: f32,
    #[serde(default = "default_rope_ntk_beta")]
    pub rope_ntk_beta: f32,
    #[serde(default)]
    pub architectures: Vec<String>,
}

impl GptOssConfig {
    pub fn from_json_str(json: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(json)
            .map_err(|error| candle::Error::Msg(format!("invalid GPT-OSS config JSON: {error}")))?;
        config.validate()?;
        Ok(config)
    }

    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path).map_err(|error| {
            candle::Error::Msg(format!("failed to read GPT-OSS config {:?}: {error}", path))
        })?;
        Self::from_json_str(&json)
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != "gpt_oss" {
            bail!(
                "unsupported GPT-OSS model_type {:?}; expected \"gpt_oss\"",
                self.model_type
            );
        }
        for (name, value) in [
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_experts", self.num_experts),
            ("experts_per_token", self.experts_per_token),
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("head_dim", self.head_dim),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("sliding_window", self.sliding_window),
            ("initial_context_length", self.initial_context_length),
        ] {
            if value == 0 {
                bail!("GPT-OSS config field {name} must be greater than zero");
            }
        }
        if self.experts_per_token > self.num_experts {
            bail!(
                "GPT-OSS experts_per_token {} exceeds num_experts {}",
                self.experts_per_token,
                self.num_experts
            );
        }
        if !self.intermediate_size.is_multiple_of(2) {
            bail!("GPT-OSS intermediate_size must be even for SwiGLU");
        }
        if !self.hidden_size.is_multiple_of(self.head_dim) {
            bail!("GPT-OSS hidden_size must be divisible by head_dim");
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            bail!("GPT-OSS attention heads must be divisible by KV heads");
        }
        if !self.head_dim.is_multiple_of(2) {
            bail!("GPT-OSS head_dim must be even for rotary embeddings");
        }
        for (name, value) in [
            ("swiglu_limit", self.swiglu_limit),
            ("rope_theta", self.rope_theta),
            ("rope_scaling_factor", self.rope_scaling_factor),
            ("rope_ntk_alpha", self.rope_ntk_alpha),
            ("rope_ntk_beta", self.rope_ntk_beta),
        ] {
            if !value.is_finite() || value <= 0.0 {
                bail!("GPT-OSS config field {name} must be finite and positive");
            }
        }
        Ok(())
    }
}

fn default_model_type() -> String {
    "gpt_oss".to_string()
}
const fn default_num_hidden_layers() -> usize {
    36
}
const fn default_num_experts() -> usize {
    128
}
const fn default_experts_per_token() -> usize {
    4
}
const fn default_vocab_size() -> usize {
    201_088
}
const fn default_hidden_size() -> usize {
    2_880
}
const fn default_intermediate_size() -> usize {
    2_880
}
const fn default_swiglu_limit() -> f32 {
    7.0
}
const fn default_head_dim() -> usize {
    64
}
const fn default_num_attention_heads() -> usize {
    64
}
const fn default_num_key_value_heads() -> usize {
    8
}
const fn default_sliding_window() -> usize {
    128
}
const fn default_initial_context_length() -> usize {
    4_096
}
const fn default_rope_theta() -> f32 {
    150_000.0
}
const fn default_rope_scaling_factor() -> f32 {
    32.0
}
const fn default_rope_ntk_alpha() -> f32 {
    1.0
}
const fn default_rope_ntk_beta() -> f32 {
    32.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_validates_tiny_synthetic_config() -> Result<()> {
        let config = GptOssConfig::from_json_str(
            r#"{
                "model_type": "gpt_oss",
                "num_hidden_layers": 1,
                "num_experts": 2,
                "experts_per_token": 1,
                "vocab_size": 16,
                "hidden_size": 8,
                "intermediate_size": 8,
                "head_dim": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "sliding_window": 4,
                "initial_context_length": 8,
                "rope_theta": 10000.0,
                "rope_scaling_factor": 1.0,
                "rope_ntk_alpha": 1.0,
                "rope_ntk_beta": 32.0
            }"#,
        )?;
        assert_eq!(config.hidden_size, 8);
        Ok(())
    }

    #[test]
    fn rejects_unsupported_model_type() {
        let error = GptOssConfig::from_json_str(r#"{"model_type":"llama"}"#)
            .expect_err("non GPT-OSS config must fail closed");
        assert!(error.to_string().contains("model_type"));
    }

    #[test]
    fn rejects_zero_sliding_window() {
        let error = GptOssConfig::from_json_str(r#"{"sliding_window":0}"#)
            .expect_err("zero sliding window must fail closed");
        assert!(error.to_string().contains("sliding_window"));
    }
}
