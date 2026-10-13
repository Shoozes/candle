use candle::{bail, Result};
use serde::Deserialize;
use std::path::Path;

/// Configuration values used by the official GPT-OSS reference model.
///
/// The defaults describe the published 20B/120B family, but callers should
/// treat them as compatibility defaults only and validate any checkpoint
/// configuration before constructing a model.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(try_from = "ConfigInput")]
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
    pub layer_types: Vec<GptOssAttentionType>,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: Option<usize>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GptOssAttentionType {
    SlidingAttention,
    FullAttention,
}

#[derive(Deserialize)]
struct ConfigInput {
    #[serde(default = "default_model_type")]
    model_type: String,
    #[serde(default = "default_num_hidden_layers")]
    num_hidden_layers: usize,
    num_experts: Option<usize>,
    num_local_experts: Option<usize>,
    experts_per_token: Option<usize>,
    num_experts_per_tok: Option<usize>,
    #[serde(default = "default_vocab_size")]
    vocab_size: usize,
    #[serde(default = "default_hidden_size")]
    hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    intermediate_size: usize,
    #[serde(default = "default_swiglu_limit")]
    swiglu_limit: f32,
    #[serde(default = "default_head_dim")]
    head_dim: usize,
    #[serde(default = "default_num_attention_heads")]
    num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    num_key_value_heads: usize,
    #[serde(default = "default_sliding_window")]
    sliding_window: usize,
    initial_context_length: Option<usize>,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    rope_scaling_factor: Option<f32>,
    rope_ntk_alpha: Option<f32>,
    rope_ntk_beta: Option<f32>,
    rope_scaling: Option<RopeInput>,
    #[serde(default)]
    architectures: Vec<String>,
    layer_types: Option<Vec<GptOssAttentionType>>,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f32,
    max_position_embeddings: Option<usize>,
    tie_word_embeddings: Option<bool>,
    attention_bias: Option<bool>,
    quantization_config: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct RopeInput {
    factor: f32,
    original_max_position_embeddings: Option<usize>,
    beta_slow: Option<f32>,
    beta_fast: Option<f32>,
    #[serde(alias = "type")]
    rope_type: String,
    #[serde(default)]
    truncate: bool,
}

fn reconcile<T: PartialEq>(
    name: &str,
    original: Option<T>,
    alternate: Option<T>,
    default: T,
) -> std::result::Result<T, String> {
    match (original, alternate) {
        (Some(a), Some(b)) if a != b => Err(format!("conflicting GPT-OSS config field {name}")),
        (Some(a), _) | (_, Some(a)) => Ok(a),
        _ => Ok(default),
    }
}

impl TryFrom<ConfigInput> for GptOssConfig {
    type Error = String;

    fn try_from(input: ConfigInput) -> std::result::Result<Self, Self::Error> {
        if input.tie_word_embeddings == Some(true) || input.attention_bias == Some(false) {
            return Err("GPT-OSS requires untied output weights and attention biases".into());
        }
        if let Some(quantization) = &input.quantization_config {
            if quantization
                .get("quant_method")
                .and_then(serde_json::Value::as_str)
                != Some("mxfp4")
            {
                return Err("GPT-OSS requires packed MXFP4 experts".into());
            }
        }
        if let Some(rope) = &input.rope_scaling {
            if rope.rope_type != "yarn" || rope.truncate {
                return Err("GPT-OSS supports untruncated YaRN rotary scaling".into());
            }
        }
        let rope = input.rope_scaling;
        let layer_types = match input.layer_types {
            Some(types) if types.len() != input.num_hidden_layers => {
                return Err("GPT-OSS layer_types count must match num_hidden_layers".into())
            }
            Some(types) => types,
            None => Vec::new(),
        };
        let config = Self {
            model_type: input.model_type,
            num_hidden_layers: input.num_hidden_layers,
            num_experts: reconcile(
                "num_experts",
                input.num_experts,
                input.num_local_experts,
                default_num_experts(),
            )?,
            experts_per_token: reconcile(
                "experts_per_token",
                input.experts_per_token,
                input.num_experts_per_tok,
                default_experts_per_token(),
            )?,
            vocab_size: input.vocab_size,
            hidden_size: input.hidden_size,
            intermediate_size: input.intermediate_size,
            swiglu_limit: input.swiglu_limit,
            head_dim: input.head_dim,
            num_attention_heads: input.num_attention_heads,
            num_key_value_heads: input.num_key_value_heads,
            sliding_window: input.sliding_window,
            initial_context_length: reconcile(
                "initial_context_length",
                input.initial_context_length,
                rope.as_ref()
                    .and_then(|r| r.original_max_position_embeddings),
                default_initial_context_length(),
            )?,
            rope_theta: input.rope_theta,
            rope_scaling_factor: reconcile(
                "rope_scaling_factor",
                input.rope_scaling_factor,
                rope.as_ref().map(|r| r.factor),
                default_rope_scaling_factor(),
            )?,
            rope_ntk_alpha: reconcile(
                "rope_ntk_alpha",
                input.rope_ntk_alpha,
                rope.as_ref().and_then(|r| r.beta_slow),
                default_rope_ntk_alpha(),
            )?,
            rope_ntk_beta: reconcile(
                "rope_ntk_beta",
                input.rope_ntk_beta,
                rope.as_ref().and_then(|r| r.beta_fast),
                default_rope_ntk_beta(),
            )?,
            architectures: input.architectures,
            layer_types,
            rms_norm_eps: input.rms_norm_eps,
            max_position_embeddings: input.max_position_embeddings,
        };
        config.validate().map_err(|error| error.to_string())?;
        Ok(config)
    }
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
            ("rms_norm_eps", self.rms_norm_eps),
            ("rope_theta", self.rope_theta),
            ("rope_scaling_factor", self.rope_scaling_factor),
            ("rope_ntk_alpha", self.rope_ntk_alpha),
            ("rope_ntk_beta", self.rope_ntk_beta),
        ] {
            if !value.is_finite() || value <= 0.0 {
                bail!("GPT-OSS config field {name} must be finite and positive");
            }
        }
        if !self.layer_types.is_empty() && self.layer_types.len() != self.num_hidden_layers {
            bail!("GPT-OSS layer_types count must match num_hidden_layers");
        }
        if self.max_position_embeddings == Some(0) {
            bail!("GPT-OSS max_position_embeddings must be positive");
        }
        Ok(())
    }

    pub fn attention_type(&self, layer: usize) -> Result<GptOssAttentionType> {
        if layer >= self.num_hidden_layers {
            bail!("GPT-OSS attention layer {layer} is out of range");
        }
        if self.layer_types.is_empty() {
            return Ok(if layer.is_multiple_of(2) {
                GptOssAttentionType::SlidingAttention
            } else {
                GptOssAttentionType::FullAttention
            });
        }
        self.layer_types.get(layer).copied().ok_or_else(|| {
            candle::Error::Msg("GPT-OSS layer_types count must match num_hidden_layers".into())
        })
    }

    /// Checkpoint context capacity; callers still supply smaller resource
    /// limits and must separately qualify their execution envelope.
    pub fn maximum_sequence_tokens(&self) -> Result<usize> {
        if let Some(maximum) = self.max_position_embeddings {
            if maximum == 0 {
                bail!("GPT-OSS max_position_embeddings must be positive");
            }
            return Ok(maximum);
        }
        let scaled = self.initial_context_length as f64 * f64::from(self.rope_scaling_factor);
        if !scaled.is_finite() || scaled < 1.0 || scaled >= usize::MAX as f64 {
            bail!("GPT-OSS scaled context capacity is outside usize range");
        }
        Ok(scaled.floor() as usize)
    }
}

const fn default_rms_norm_eps() -> f32 {
    1e-5
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
    fn normalizes_pinned_twenty_b_aliases_yarn_and_layer_types() -> Result<()> {
        let mut json = serde_json::json!({
            "model_type":"gpt_oss","num_hidden_layers":24,"num_local_experts":32,
            "experts_per_token":4,"num_experts_per_tok":4,"initial_context_length":4096,
            "max_position_embeddings":131072,"rms_norm_eps":0.00001,
            "rope_scaling":{"rope_type":"yarn","factor":32.0,"original_max_position_embeddings":4096,
                "beta_fast":32.0,"beta_slow":1.0,"truncate":false},
            "quantization_config":{"quant_method":"mxfp4"},"tie_word_embeddings":false,
            "attention_bias":true
        });
        json["layer_types"] = (0..24)
            .map(|i| {
                if i % 2 == 0 {
                    "sliding_attention"
                } else {
                    "full_attention"
                }
            })
            .collect::<Vec<_>>()
            .into();
        let config: GptOssConfig = serde_json::from_value(json).map_err(candle::Error::wrap)?;
        assert_eq!(
            (
                config.num_hidden_layers,
                config.num_experts,
                config.experts_per_token
            ),
            (24, 32, 4)
        );
        assert_eq!(config.rope_scaling_factor, 32.0);
        assert_eq!(config.maximum_sequence_tokens()?, 131072);
        assert_eq!(
            config.attention_type(1)?,
            GptOssAttentionType::FullAttention
        );
        let legacy = GptOssConfig::from_json_str("{}")?;
        assert_eq!((legacy.num_hidden_layers, legacy.num_experts), (36, 128));
        assert_eq!(legacy.maximum_sequence_tokens()?, 131072);
        Ok(())
    }

    #[test]
    fn rejects_conflicting_or_unsupported_checkpoint_fields() {
        for json in [
            r#"{"num_experts":64,"num_local_experts":32}"#,
            r#"{"experts_per_token":4,"num_experts_per_tok":2}"#,
            r#"{"rope_scaling_factor":16,"rope_scaling":{"rope_type":"yarn","factor":32}}"#,
            r#"{"rope_scaling":{"rope_type":"linear","factor":32}}"#,
            r#"{"layer_types":[]}"#,
            r#"{"rms_norm_eps":0}"#,
            r#"{"max_position_embeddings":0}"#,
            r#"{"tie_word_embeddings":true}"#,
            r#"{"quantization_config":{"quant_method":"other"}}"#,
        ] {
            assert!(GptOssConfig::from_json_str(json).is_err(), "{json}");
        }
    }

    #[test]
    fn explicit_attention_layout_controls_scheduling() -> Result<()> {
        let config = GptOssConfig::from_json_str(
            r#"{"num_hidden_layers":2,"layer_types":["full_attention","sliding_attention"]}"#,
        )?;
        assert_eq!(
            config.attention_type(0)?,
            GptOssAttentionType::FullAttention
        );
        assert_eq!(
            config.attention_type(1)?,
            GptOssAttentionType::SlidingAttention
        );
        assert!(config.attention_type(2).is_err());
        Ok(())
    }

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
