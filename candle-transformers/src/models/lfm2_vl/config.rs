use crate::models::{lfm2, siglip2};
use candle::Result;
use candle_nn::Activation;
use serde::de::Error as DeError;

fn default_image_token_id() -> u32 {
    396
}

fn default_projector_hidden_size() -> usize {
    2560
}

fn default_projector_hidden_act() -> Activation {
    Activation::Gelu
}

fn default_projector_bias() -> bool {
    true
}

fn default_projector_use_layernorm() -> bool {
    true
}

fn default_downsample_factor() -> usize {
    2
}

fn default_use_image_special_tokens() -> bool {
    false
}

fn default_tie_word_embeddings() -> bool {
    true
}

#[derive(Debug, Clone)]
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

#[derive(Debug, serde::Deserialize)]
struct Lfm2VlConfigSerde {
    #[serde(default)]
    model_type: Option<String>,
    text_config: lfm2::Lfm2Config,
    vision_config: siglip2::Siglip2VisionConfig,
    #[serde(default = "default_image_token_id", alias = "image_token_index")]
    image_token_id: u32,
    #[serde(default = "default_downsample_factor")]
    downsample_factor: usize,
    #[serde(default = "default_projector_hidden_size")]
    projector_hidden_size: usize,
    #[serde(default = "default_projector_hidden_act")]
    projector_hidden_act: Activation,
    #[serde(default = "default_projector_bias")]
    projector_bias: bool,
    #[serde(
        default = "default_projector_use_layernorm",
        alias = "projector_use_layer_norm"
    )]
    projector_use_layernorm: bool,
    #[serde(default = "default_use_image_special_tokens")]
    use_image_special_tokens: bool,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    tie_embedding: Option<bool>,
}

impl<'de> serde::Deserialize<'de> for Lfm2VlConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = <Lfm2VlConfigSerde as serde::Deserialize>::deserialize(deserializer)?;
        if let Some(model_type) = raw.model_type.as_deref() {
            if model_type != "lfm2_vl" && model_type != "lfm2-vl" {
                return Err(D::Error::custom(format!(
                    "unsupported LFM2-VL model_type {model_type:?}"
                )));
            }
        }

        // Transformers' top-level config gives the legacy spelling precedence
        // over the current spelling. When neither is present, both the
        // top-level and nested language configs default to tied embeddings.
        let tie_word_embeddings = match raw.tie_embedding.or(raw.tie_word_embeddings) {
            Some(value) => value,
            None => default_tie_word_embeddings(),
        };
        let mut text_config = raw.text_config;
        text_config.tie_embedding = tie_word_embeddings;
        text_config.tie_word_embeddings = Some(tie_word_embeddings);

        Ok(Self {
            text_config,
            vision_config: raw.vision_config,
            image_token_id: raw.image_token_id,
            downsample_factor: raw.downsample_factor,
            projector_hidden_size: raw.projector_hidden_size,
            projector_hidden_act: raw.projector_hidden_act,
            projector_bias: raw.projector_bias,
            projector_use_layernorm: raw.projector_use_layernorm,
            use_image_special_tokens: raw.use_image_special_tokens,
        })
    }
}

impl Lfm2VlConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(json)
            .map_err(|err| candle::Error::Msg(format!("invalid LFM2-VL config: {err}")))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        self.vision_config.validate()?;
        if self.downsample_factor == 0 {
            candle::bail!("LFM2-VL downsample_factor must be greater than zero")
        }
        if self.projector_hidden_size == 0 {
            candle::bail!("LFM2-VL projector_hidden_size must be greater than zero")
        }
        let factor_squared = self
            .downsample_factor
            .checked_mul(self.downsample_factor)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL downsample factor overflow".into()))?;
        let _projector_input = self
            .vision_config
            .hidden_size
            .checked_mul(factor_squared)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL projector input width overflow".into()))?;

        let text_config = self.text_config.clone().try_into_config(false)?;
        if text_config.num_hidden_layers == 0 {
            candle::bail!("LFM2-VL text model must contain at least one layer")
        }
        if text_config.num_attention_heads == 0
            || text_config.hidden_size % text_config.num_attention_heads != 0
        {
            candle::bail!(
                "LFM2-VL text hidden size {} is not divisible by attention heads {}",
                text_config.hidden_size,
                text_config.num_attention_heads
            )
        }
        if text_config.num_key_value_heads == 0
            || text_config.num_attention_heads % text_config.num_key_value_heads != 0
        {
            candle::bail!(
                "LFM2-VL text attention heads {} are not divisible by key/value heads {}",
                text_config.num_attention_heads,
                text_config.num_key_value_heads
            )
        }
        if self.image_token_id as usize >= text_config.vocab_size {
            candle::bail!(
                "LFM2-VL image_token_id {} is outside vocabulary size {}",
                self.image_token_id,
                text_config.vocab_size
            )
        }
        Ok(())
    }

    pub fn projector_input_size(&self) -> Result<usize> {
        let factor_squared = self
            .downsample_factor
            .checked_mul(self.downsample_factor)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL downsample factor overflow".into()))?;
        self.vision_config
            .hidden_size
            .checked_mul(factor_squared)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL projector input width overflow".into()))
    }

    pub fn projected_token_count(&self, patch_rows: usize, patch_cols: usize) -> Result<usize> {
        projected_token_count(patch_rows, patch_cols, self.downsample_factor)
    }

    pub fn text_model_config(&self) -> Result<lfm2::Config> {
        self.text_config.clone().try_into_config(false)
    }
}

/// Return the number of projected tokens for one valid patch grid.
///
/// This is the canonical checked count shared by the native projector and the
/// Rust image/prompt processor. A crop must be factor-aligned because the
/// official pixel-unshuffle cannot represent a partial factor block.
pub fn projected_token_count(
    patch_rows: usize,
    patch_cols: usize,
    downsample_factor: usize,
) -> Result<usize> {
    if patch_rows == 0 || patch_cols == 0 {
        candle::bail!("LFM2-VL crop patch dimensions must be positive")
    }
    if downsample_factor == 0 {
        candle::bail!("LFM2-VL downsample_factor must be greater than zero")
    }
    if patch_rows % downsample_factor != 0 || patch_cols % downsample_factor != 0 {
        candle::bail!(
            "LFM2-VL crop grid [{patch_rows}, {patch_cols}] is not divisible by projector factor {downsample_factor}"
        )
    }
    (patch_rows / downsample_factor)
        .checked_mul(patch_cols / downsample_factor)
        .ok_or_else(|| candle::Error::Msg("LFM2-VL projected token count overflow".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_json() -> String {
        serde_json::json!({
            "model_type": "lfm2-vl",
            "image_token_index": 3,
            "downsample_factor": 2,
            "projector_hidden_size": 24,
            "projector_hidden_act": "gelu",
            "projector_bias": true,
            "projector_use_layer_norm": true,
            "text_config": {
                "model_type": "lfm2",
                "vocab_size": 32,
                "hidden_size": 12,
                "num_hidden_layers": 2,
                "num_attention_heads": 3,
                "num_key_value_heads": 1,
                "intermediate_size": 32,
                "block_auto_adjust_ff_dim": false,
                "layer_types": ["conv", "full_attention"],
                "rope_theta": 10000.0
            },
            "vision_config": {
                "model_type": "siglip2_vision_model",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_channels": 3,
                "patch_size": 2,
                "num_patches": 16,
                "hidden_act": "gelu_pytorch_tanh",
                "vision_use_head": false
            }
        })
        .to_string()
    }

    #[test]
    fn parses_aliases_and_validates_relationships() -> Result<()> {
        let config = Lfm2VlConfig::from_json(&tiny_json())?;
        assert_eq!(config.image_token_id, 3);
        assert_eq!(config.projector_input_size()?, 64);
        assert_eq!(config.projected_token_count(2, 4)?, 2);
        assert_eq!(config.text_config.rope_theta, 10000.0);
        let factor_three = Lfm2VlConfig::from_json(
            &tiny_json().replace("\"downsample_factor\":2", "\"downsample_factor\":3"),
        )?;
        assert!(factor_three.projected_token_count(2, 4).is_err());
        assert!(Lfm2VlConfig::from_json(
            &tiny_json().replace("\"downsample_factor\":2", "\"downsample_factor\":0")
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn top_level_legacy_tie_alias_wins() -> Result<()> {
        let mut value: serde_json::Value = serde_json::from_str(&tiny_json())
            .map_err(|err| candle::Error::Msg(format!("invalid test JSON: {err}")))?;
        value["tie_word_embeddings"] = serde_json::Value::Bool(true);
        value["tie_embedding"] = serde_json::Value::Bool(false);
        let config = Lfm2VlConfig::from_json(&value.to_string())?;
        assert!(!config.text_config.tie_embedding);
        Ok(())
    }
}
