use crate::models::{lfm2, siglip2};
use candle::Result;
use candle_nn::Activation;
use serde::de::Error as DeError;

pub const DEFAULT_LFM2_VL_IMAGE_TOKEN_ID: u32 = 396;

/// Absolute safety ceilings for one LFM2-VL request. Configured limits may
/// tighten these values but cannot raise them.
pub const MAX_VISION_SOURCE_PIXELS: usize = 64 * 1024 * 1024;
pub const MAX_VISION_IMAGES: usize = 16;
pub const MAX_VISION_CROPS_PER_IMAGE: usize = 11;
pub const MAX_VISION_TOTAL_CROPS: usize = 64;
pub const MAX_VISION_PATCHES_PER_CROP: usize = 1024;
pub const MAX_VISION_TOTAL_PROJECTED_TOKENS: usize = 64 * 1024;

/// Request-wide allocation limits shared by raw-image and packed-tensor callers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionLimits {
    pub max_source_pixels: usize,
    pub max_images: usize,
    pub max_crops_per_image: usize,
    pub max_total_crops: usize,
    pub max_patches_per_crop: usize,
    pub max_total_projected_tokens: usize,
}

impl Default for VisionLimits {
    fn default() -> Self {
        Self {
            max_source_pixels: MAX_VISION_SOURCE_PIXELS,
            max_images: MAX_VISION_IMAGES,
            max_crops_per_image: MAX_VISION_CROPS_PER_IMAGE,
            max_total_crops: MAX_VISION_TOTAL_CROPS,
            max_patches_per_crop: MAX_VISION_PATCHES_PER_CROP,
            max_total_projected_tokens: MAX_VISION_TOTAL_PROJECTED_TOKENS,
        }
    }
}

impl VisionLimits {
    pub fn validate(&self) -> Result<()> {
        for (name, value, ceiling) in [
            (
                "max_source_pixels",
                self.max_source_pixels,
                MAX_VISION_SOURCE_PIXELS,
            ),
            ("max_images", self.max_images, MAX_VISION_IMAGES),
            (
                "max_crops_per_image",
                self.max_crops_per_image,
                MAX_VISION_CROPS_PER_IMAGE,
            ),
            (
                "max_total_crops",
                self.max_total_crops,
                MAX_VISION_TOTAL_CROPS,
            ),
            (
                "max_patches_per_crop",
                self.max_patches_per_crop,
                MAX_VISION_PATCHES_PER_CROP,
            ),
            (
                "max_total_projected_tokens",
                self.max_total_projected_tokens,
                MAX_VISION_TOTAL_PROJECTED_TOKENS,
            ),
        ] {
            if value == 0 {
                candle::bail!("LFM2-VL vision limit {name} must be positive")
            }
            if value > ceiling {
                candle::bail!(
                    "LFM2-VL vision limit {name}={value} exceeds implementation ceiling {ceiling}"
                )
            }
        }
        Ok(())
    }

    pub fn check_source_image(&self, width: usize, height: usize) -> Result<usize> {
        self.check_image_surface("source image", width, height)
    }

    pub fn check_image_surface(&self, label: &str, width: usize, height: usize) -> Result<usize> {
        self.validate()?;
        if width == 0 || height == 0 {
            candle::bail!("LFM2-VL {label} dimensions must be positive")
        }
        let pixels = width
            .checked_mul(height)
            .ok_or_else(|| candle::Error::Msg(format!("LFM2-VL {label} pixel count overflow")))?;
        if pixels > self.max_source_pixels {
            candle::bail!(
                "LFM2-VL {label} has {pixels} pixels, exceeding max_source_pixels {}",
                self.max_source_pixels
            )
        }
        Ok(pixels)
    }

    pub fn check_image_count(&self, image_count: usize) -> Result<()> {
        self.validate()?;
        if image_count == 0 {
            candle::bail!("LFM2-VL request must contain at least one image")
        }
        if image_count > self.max_images {
            candle::bail!(
                "LFM2-VL request has {image_count} images, exceeding limit {}",
                self.max_images
            )
        }
        Ok(())
    }

    pub fn check_crops_per_image(&self, crop_count: usize) -> Result<()> {
        self.validate()?;
        if crop_count == 0 {
            candle::bail!("LFM2-VL image must contain at least one crop")
        }
        if crop_count > self.max_crops_per_image {
            candle::bail!(
                "LFM2-VL image has {crop_count} crops, exceeding per-image limit {}",
                self.max_crops_per_image
            )
        }
        Ok(())
    }

    pub fn check_total_crops(&self, crop_count: usize) -> Result<()> {
        self.validate()?;
        if crop_count == 0 {
            candle::bail!("LFM2-VL request must contain at least one crop")
        }
        if crop_count > self.max_total_crops {
            candle::bail!(
                "LFM2-VL request has {crop_count} crops, exceeding total limit {}",
                self.max_total_crops
            )
        }
        Ok(())
    }

    pub fn check_crop(
        &self,
        patch_rows: usize,
        patch_cols: usize,
        projected_tokens: usize,
    ) -> Result<usize> {
        self.validate()?;
        if patch_rows == 0 || patch_cols == 0 || projected_tokens == 0 {
            candle::bail!("LFM2-VL crop dimensions and projected token count must be positive")
        }
        let patches = patch_rows
            .checked_mul(patch_cols)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL crop patch count overflow".into()))?;
        if patches > self.max_patches_per_crop {
            candle::bail!(
                "LFM2-VL crop has {patches} patches, exceeding limit {}",
                self.max_patches_per_crop
            )
        }
        if projected_tokens > self.max_total_projected_tokens {
            candle::bail!(
                "LFM2-VL crop has {projected_tokens} projected tokens, exceeding request limit {}",
                self.max_total_projected_tokens
            )
        }
        Ok(patches)
    }

    pub fn check_total_projected_tokens(&self, projected_tokens: usize) -> Result<()> {
        self.validate()?;
        if projected_tokens == 0 {
            candle::bail!("LFM2-VL request must contain projected image tokens")
        }
        if projected_tokens > self.max_total_projected_tokens {
            candle::bail!(
                "LFM2-VL request has {projected_tokens} projected tokens, exceeding limit {}",
                self.max_total_projected_tokens
            )
        }
        Ok(())
    }

    pub fn check_request(
        &self,
        image_count: usize,
        crop_count: usize,
        projected_tokens: usize,
    ) -> Result<()> {
        self.validate()?;
        self.check_image_count(image_count)?;
        self.check_total_crops(crop_count)?;
        self.check_total_projected_tokens(projected_tokens)
    }
}

fn default_image_token_id() -> u32 {
    DEFAULT_LFM2_VL_IMAGE_TOKEN_ID
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

/// Vision/projector configuration shared by native, split, and GGUF MMProj loaders.
///
/// A GGUF MMProj does not contain the complete language-model configuration, so
/// this intentionally carries only the fields required to construct and run the
/// vision tower and multimodal projector.
#[derive(Debug, Clone)]
pub struct Lfm2VlMmprojConfig {
    pub vision_config: siglip2::Siglip2VisionConfig,
    pub text_hidden_size: usize,
    pub image_token_id: u32,
    pub downsample_factor: usize,
    pub projector_hidden_size: usize,
    pub projector_hidden_act: Activation,
    pub projector_bias: bool,
    pub projector_use_layernorm: bool,
    pub use_image_special_tokens: bool,
}

impl From<&Lfm2VlConfig> for Lfm2VlMmprojConfig {
    fn from(config: &Lfm2VlConfig) -> Self {
        Self {
            vision_config: config.vision_config.clone(),
            text_hidden_size: config.text_config.hidden_size,
            image_token_id: config.image_token_id,
            downsample_factor: config.downsample_factor,
            projector_hidden_size: config.projector_hidden_size,
            projector_hidden_act: config.projector_hidden_act,
            projector_bias: config.projector_bias,
            projector_use_layernorm: config.projector_use_layernorm,
            use_image_special_tokens: config.use_image_special_tokens,
        }
    }
}

impl Lfm2VlMmprojConfig {
    pub fn validate(&self) -> Result<()> {
        self.vision_config.validate()?;
        if self.text_hidden_size == 0 {
            candle::bail!("LFM2-VL MMProj text hidden size must be greater than zero")
        }
        if self.downsample_factor == 0 {
            candle::bail!("LFM2-VL downsample_factor must be greater than zero")
        }
        if self.projector_hidden_size == 0 {
            candle::bail!("LFM2-VL projector_hidden_size must be greater than zero")
        }
        let _ = self.projector_input_size()?;
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
        Lfm2VlMmprojConfig::from(self).validate()?;

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
        Lfm2VlMmprojConfig::from(self).projector_input_size()
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

    #[test]
    fn vision_limits_accept_boundaries_and_reject_overages() -> Result<()> {
        let limits = VisionLimits {
            max_source_pixels: 6,
            max_images: 2,
            max_crops_per_image: 2,
            max_total_crops: 3,
            max_patches_per_crop: 6,
            max_total_projected_tokens: 5,
        };
        limits.validate()?;
        assert_eq!(limits.check_source_image(3, 2)?, 6);
        limits.check_crops_per_image(2)?;
        assert_eq!(limits.check_crop(2, 3, 5)?, 6);
        limits.check_request(2, 3, 5)?;

        assert!(limits.check_source_image(7, 1).is_err());
        assert!(limits.check_image_count(3).is_err());
        assert!(limits.check_crops_per_image(3).is_err());
        assert!(limits.check_total_crops(4).is_err());
        assert!(limits.check_crop(1, 7, 5).is_err());
        assert!(limits.check_total_projected_tokens(6).is_err());
        assert!(limits.check_source_image(usize::MAX, 2).is_err());
        Ok(())
    }

    #[test]
    fn vision_limits_reject_zero_caps() {
        assert!(VisionLimits {
            max_images: 0,
            ..VisionLimits::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn vision_limits_reject_configured_values_above_hard_ceilings() {
        assert!(VisionLimits {
            max_source_pixels: MAX_VISION_SOURCE_PIXELS + 1,
            ..VisionLimits::default()
        }
        .validate()
        .is_err());
        assert!(VisionLimits {
            max_total_projected_tokens: usize::MAX,
            ..VisionLimits::default()
        }
        .validate()
        .is_err());
    }
}
