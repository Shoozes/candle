//! Dynamic processor configuration and source-precedence resolution.

use candle::Result;
use candle_transformers::models::lfm2_vl::{Lfm2VlConfig, MmprojMetadata, VisionLimits};
use serde::Deserialize;

fn default_do_resize() -> bool {
    true
}

fn default_do_rescale() -> bool {
    true
}

fn default_rescale_factor() -> f32 {
    1.0 / 255.0
}

fn default_do_normalize() -> bool {
    true
}

fn default_mean() -> [f32; 3] {
    [0.5, 0.5, 0.5]
}

fn default_std() -> [f32; 3] {
    [0.5, 0.5, 0.5]
}

fn default_do_pad() -> bool {
    true
}

fn default_downsample_factor() -> usize {
    2
}

fn default_encoder_patch_size() -> usize {
    16
}

fn default_do_image_splitting() -> bool {
    true
}

fn default_min_tiles() -> usize {
    2
}

fn default_max_tiles() -> usize {
    10
}

fn default_use_thumbnail() -> bool {
    true
}

fn default_tile_size() -> usize {
    512
}

fn default_min_image_tokens() -> usize {
    64
}

fn default_max_image_tokens() -> usize {
    256
}

fn default_max_pixels_tolerance() -> f64 {
    2.0
}

/// Typed optional values from one configuration source.
///
/// The same patch type is used for explicit API overrides, processor JSON,
/// future GGUF metadata, and model-config hints. Resolution applies the
/// sources in increasing authority order.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct ProcessorConfigPatch {
    pub do_resize: Option<bool>,
    pub do_rescale: Option<bool>,
    pub rescale_factor: Option<f32>,
    pub do_normalize: Option<bool>,
    pub image_mean: Option<[f32; 3]>,
    pub image_std: Option<[f32; 3]>,
    pub do_pad: Option<bool>,

    pub downsample_factor: Option<usize>,
    #[serde(alias = "patch_size")]
    pub encoder_patch_size: Option<usize>,

    pub do_image_splitting: Option<bool>,
    pub min_tiles: Option<usize>,
    pub max_tiles: Option<usize>,
    pub use_thumbnail: Option<bool>,
    pub tile_size: Option<usize>,

    pub min_image_tokens: Option<usize>,
    pub max_image_tokens: Option<usize>,
    pub max_num_patches: Option<usize>,
    pub max_pixels_tolerance: Option<f64>,

    #[serde(alias = "max_context_length", alias = "max_position_embeddings")]
    pub context_length: Option<usize>,

    pub max_source_pixels: Option<usize>,
    pub max_images: Option<usize>,
    pub max_crops_per_image: Option<usize>,
    pub max_total_crops: Option<usize>,
    pub max_patches_per_crop: Option<usize>,
    pub max_total_projected_tokens: Option<usize>,
}

impl ProcessorConfigPatch {
    pub fn from_json(json: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(json)
            .map_err(|err| candle::Error::Msg(format!("invalid processor config: {err}")))?;
        Self::from_value(&value)
    }

    pub fn from_value(value: &serde_json::Value) -> Result<Self> {
        // Processor documents carry unrelated class metadata at the outer
        // level. The patch schema deliberately ignores unknown keys while
        // validating every known field during resolution.
        let source = value.get("image_processor").unwrap_or(value);
        serde_json::from_value(source.clone())
            .map_err(|err| candle::Error::Msg(format!("invalid processor config fields: {err}")))
    }

    pub fn from_model_config(config: &Lfm2VlConfig) -> Self {
        Self {
            downsample_factor: Some(config.downsample_factor),
            encoder_patch_size: Some(config.vision_config.patch_size),
            context_length: Some(config.text_config.max_position_embeddings),
            ..Self::default()
        }
    }

    pub fn from_gguf_metadata(metadata: Self) -> GgufProcessorMetadata {
        metadata
    }
}

/// Explicit API or CLI overrides. The alias documents the precedence layer
/// without giving GGUF loading an implementation dependency in this phase.
pub type ProcessorConfigOverride = ProcessorConfigPatch;

/// Typed future GGUF processor metadata. GGUF parsing remains out of scope.
pub type GgufProcessorMetadata = ProcessorConfigPatch;

/// Resolved processor settings. This fork-local crate is not yet a released
/// Candle API; the non-exhaustive boundary keeps future safety fields additive.
#[derive(Clone, Debug)]
#[non_exhaustive]
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
    pub context_length: Option<usize>,
    pub vision_limits: VisionLimits,
}

impl Default for Lfm2VlProcessorConfig {
    fn default() -> Self {
        Self {
            do_resize: default_do_resize(),
            do_rescale: default_do_rescale(),
            rescale_factor: default_rescale_factor(),
            do_normalize: default_do_normalize(),
            image_mean: default_mean(),
            image_std: default_std(),
            do_pad: default_do_pad(),
            downsample_factor: default_downsample_factor(),
            encoder_patch_size: default_encoder_patch_size(),
            do_image_splitting: default_do_image_splitting(),
            min_tiles: default_min_tiles(),
            max_tiles: default_max_tiles(),
            use_thumbnail: default_use_thumbnail(),
            tile_size: default_tile_size(),
            min_image_tokens: default_min_image_tokens(),
            max_image_tokens: default_max_image_tokens(),
            max_num_patches: None,
            max_pixels_tolerance: default_max_pixels_tolerance(),
            context_length: None,
            vision_limits: VisionLimits::default(),
        }
    }
}

impl Lfm2VlProcessorConfig {
    /// Parse a standalone `processor_config.json` over architecture defaults.
    pub fn from_json(json: &str) -> Result<Self> {
        let patch = ProcessorConfigPatch::from_json(json)?;
        Self::resolve(None, Some(&patch), None, None)
    }

    /// Resolve `explicit > processor > GGUF > model > architecture defaults`.
    pub fn resolve(
        explicit: Option<&ProcessorConfigOverride>,
        processor_json: Option<&ProcessorConfigPatch>,
        gguf_metadata: Option<&GgufProcessorMetadata>,
        model_config: Option<&ProcessorConfigPatch>,
    ) -> Result<Self> {
        let mut config = Self::default();
        if let Some(patch) = model_config {
            config.apply(patch);
        }
        if let Some(patch) = gguf_metadata {
            config.apply(patch);
        }
        if let Some(patch) = processor_json {
            config.apply(patch);
        }
        if let Some(patch) = explicit {
            config.apply(patch);
        }
        config.validate()?;
        Ok(config)
    }

    pub fn with_model_config(config: &Lfm2VlConfig) -> Result<Self> {
        let model_patch = ProcessorConfigPatch::from_model_config(config);
        Self::resolve(None, None, None, Some(&model_patch))
    }

    /// Resolve the typed processor bundled with a split dense MMProj.
    ///
    /// The model crate deliberately stores the processor document as neutral
    /// JSON to preserve the one-way `candle-vlm -> candle-transformers`
    /// dependency. This conversion applies the bundled processor over the
    /// embedded model hints and validates the resolved result.
    pub fn from_mmproj_metadata(metadata: &MmprojMetadata) -> Result<Self> {
        Self::from_mmproj_metadata_with_processor(metadata, None)
    }

    /// Resolve an MMProj's embedded processor facts with an optional explicit
    /// processor document. For GGUF, embedded facts occupy the GGUF precedence
    /// layer; for split bundles, the bundled JSON occupies the processor layer.
    pub fn from_mmproj_metadata_with_processor(
        metadata: &MmprojMetadata,
        processor: Option<&ProcessorConfigPatch>,
    ) -> Result<Self> {
        let processor_patch = ProcessorConfigPatch::from_value(&metadata.processor)?;
        let model_patch = ProcessorConfigPatch {
            downsample_factor: Some(metadata.downsample_factor),
            encoder_patch_size: Some(metadata.patch_size),
            ..ProcessorConfigPatch::default()
        };
        if metadata.gguf_metadata().is_some() {
            Self::resolve(None, processor, Some(&processor_patch), Some(&model_patch))
        } else {
            Self::resolve(processor, Some(&processor_patch), None, Some(&model_patch))
        }
    }

    fn apply(&mut self, patch: &ProcessorConfigPatch) {
        if let Some(value) = patch.do_resize {
            self.do_resize = value;
        }
        if let Some(value) = patch.do_rescale {
            self.do_rescale = value;
        }
        if let Some(value) = patch.rescale_factor {
            self.rescale_factor = value;
        }
        if let Some(value) = patch.do_normalize {
            self.do_normalize = value;
        }
        if let Some(value) = patch.image_mean {
            self.image_mean = value;
        }
        if let Some(value) = patch.image_std {
            self.image_std = value;
        }
        if let Some(value) = patch.do_pad {
            self.do_pad = value;
        }
        if let Some(value) = patch.downsample_factor {
            self.downsample_factor = value;
        }
        if let Some(value) = patch.encoder_patch_size {
            self.encoder_patch_size = value;
        }
        if let Some(value) = patch.do_image_splitting {
            self.do_image_splitting = value;
        }
        if let Some(value) = patch.min_tiles {
            self.min_tiles = value;
        }
        if let Some(value) = patch.max_tiles {
            self.max_tiles = value;
        }
        if let Some(value) = patch.use_thumbnail {
            self.use_thumbnail = value;
        }
        if let Some(value) = patch.tile_size {
            self.tile_size = value;
        }
        if let Some(value) = patch.min_image_tokens {
            self.min_image_tokens = value;
        }
        if let Some(value) = patch.max_image_tokens {
            self.max_image_tokens = value;
        }
        if let Some(value) = patch.max_num_patches {
            self.max_num_patches = Some(value);
        }
        if let Some(value) = patch.max_pixels_tolerance {
            self.max_pixels_tolerance = value;
        }
        if let Some(value) = patch.context_length {
            self.context_length = Some(value);
        }
        if let Some(value) = patch.max_source_pixels {
            self.vision_limits.max_source_pixels = value;
        }
        if let Some(value) = patch.max_images {
            self.vision_limits.max_images = value;
        }
        if let Some(value) = patch.max_crops_per_image {
            self.vision_limits.max_crops_per_image = value;
        }
        if let Some(value) = patch.max_total_crops {
            self.vision_limits.max_total_crops = value;
        }
        if let Some(value) = patch.max_patches_per_crop {
            self.vision_limits.max_patches_per_crop = value;
        }
        if let Some(value) = patch.max_total_projected_tokens {
            self.vision_limits.max_total_projected_tokens = value;
        }
    }

    pub fn validate(&self) -> Result<()> {
        if !self.rescale_factor.is_finite() || self.rescale_factor <= 0.0 {
            candle::bail!("processor rescale_factor must be finite and positive")
        }
        for (name, values) in [
            ("image_mean", self.image_mean),
            ("image_std", self.image_std),
        ] {
            for value in values {
                if !value.is_finite() {
                    candle::bail!("processor {name} values must be finite")
                }
            }
        }
        if self.image_std.iter().any(|&value| value == 0.0) {
            candle::bail!("processor image_std values must be non-zero")
        }
        if self.downsample_factor == 0 || self.encoder_patch_size == 0 {
            candle::bail!("processor patch and downsample factors must be positive")
        }
        let total_factor = self
            .encoder_patch_size
            .checked_mul(self.downsample_factor)
            .ok_or_else(|| candle::Error::Msg("processor resize factor overflow".into()))?;
        if total_factor == 0 {
            candle::bail!("processor total resize factor must be positive")
        }
        if self.min_tiles == 0
            || self.max_tiles == 0
            || self.min_tiles > self.max_tiles
            || self.max_tiles > 10
        {
            candle::bail!("processor tile range must be positive and ordered")
        }
        if self.tile_size == 0 || self.tile_size % self.encoder_patch_size != 0 {
            candle::bail!("processor tile_size must be divisible by encoder_patch_size")
        }
        let tile_patches = self.tile_size / self.encoder_patch_size;
        candle_transformers::models::lfm2_vl::projected_token_count(
            tile_patches,
            tile_patches,
            self.downsample_factor,
        )?;
        if self.min_image_tokens == 0 || self.max_image_tokens == 0 {
            candle::bail!("processor image token limits must be positive")
        }
        if self.min_image_tokens > self.max_image_tokens {
            candle::bail!("processor min_image_tokens must not exceed max_image_tokens")
        }
        if let Some(max_num_patches) = self.max_num_patches {
            if max_num_patches == 0 {
                candle::bail!("processor max_num_patches must be positive")
            }
        }
        if !self.max_pixels_tolerance.is_finite() || self.max_pixels_tolerance <= 0.0 {
            candle::bail!("processor max_pixels_tolerance must be finite and positive")
        }
        if self.context_length == Some(0) {
            candle::bail!("processor context_length must be positive when provided")
        }
        self.vision_limits.validate()?;
        let max_num_patches = self.effective_max_num_patches()?;
        if max_num_patches > self.vision_limits.max_patches_per_crop {
            candle::bail!(
                "processor packed maximum {max_num_patches} exceeds vision patch limit {}",
                self.vision_limits.max_patches_per_crop
            )
        }
        Ok(())
    }

    pub fn effective_max_num_patches(&self) -> Result<usize> {
        let factor_squared = self
            .downsample_factor
            .checked_mul(self.downsample_factor)
            .ok_or_else(|| candle::Error::Msg("processor factor square overflow".into()))?;
        let thumbnail = self
            .max_image_tokens
            .checked_mul(factor_squared)
            .ok_or_else(|| candle::Error::Msg("processor thumbnail patch limit overflow".into()))?;
        let tile = if self.do_image_splitting {
            let tile_patches = self.tile_size / self.encoder_patch_size;
            tile_patches
                .checked_mul(tile_patches)
                .ok_or_else(|| candle::Error::Msg("processor tile patch limit overflow".into()))?
        } else {
            0
        };
        let derived = thumbnail.max(tile);
        if let Some(configured) = self.max_num_patches {
            if configured < derived {
                candle::bail!(
                    "processor max_num_patches {configured} is smaller than required packed maximum {derived}"
                )
            }
            return Ok(configured);
        }
        Ok(derived)
    }

    pub fn total_factor(&self) -> Result<usize> {
        self.encoder_patch_size
            .checked_mul(self.downsample_factor)
            .ok_or_else(|| candle::Error::Msg("processor total factor overflow".into()))
    }

    pub fn projected_token_count(&self, patch_rows: usize, patch_cols: usize) -> Result<usize> {
        candle_transformers::models::lfm2_vl::projected_token_count(
            patch_rows,
            patch_cols,
            self.downsample_factor,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};
    use candle_transformers::models::lfm2_vl::Mmproj;
    use std::path::PathBuf;

    #[test]
    fn precedence_is_explicit_processor_gguf_model_defaults() -> Result<()> {
        let model = ProcessorConfigPatch {
            downsample_factor: Some(2),
            encoder_patch_size: Some(16),
            max_image_tokens: Some(32),
            ..ProcessorConfigPatch::default()
        };
        let gguf = ProcessorConfigPatch {
            max_image_tokens: Some(48),
            ..ProcessorConfigPatch::default()
        };
        let processor = ProcessorConfigPatch {
            max_image_tokens: Some(64),
            ..ProcessorConfigPatch::default()
        };
        let explicit = ProcessorConfigOverride {
            max_image_tokens: Some(96),
            ..ProcessorConfigPatch::default()
        };
        let resolved = Lfm2VlProcessorConfig::resolve(
            Some(&explicit),
            Some(&processor),
            Some(&gguf),
            Some(&model),
        )?;
        assert_eq!(resolved.max_image_tokens, 96);
        assert_eq!(resolved.downsample_factor, 2);
        Ok(())
    }

    #[test]
    fn sparse_official_gguf_metadata_retains_lfm2_vl_processor_defaults() -> Result<()> {
        let gguf = ProcessorConfigPatch {
            downsample_factor: Some(2),
            encoder_patch_size: Some(16),
            image_mean: Some([0.5; 3]),
            image_std: Some([0.5; 3]),
            ..ProcessorConfigPatch::default()
        };
        let resolved = Lfm2VlProcessorConfig::resolve(None, None, Some(&gguf), None)?;
        assert_eq!(resolved.min_tiles, 2);
        assert_eq!(resolved.max_tiles, 10);
        assert!(resolved.use_thumbnail);
        assert_eq!(resolved.tile_size, 512);
        assert_eq!(resolved.min_image_tokens, 64);
        assert_eq!(resolved.max_image_tokens, 256);
        assert_eq!(resolved.max_num_patches, None);
        assert_eq!(resolved.effective_max_num_patches()?, 1024);
        Ok(())
    }

    #[test]
    fn parses_dynamic_json_and_rejects_invalid_values() -> Result<()> {
        let config = Lfm2VlProcessorConfig::from_json(
            r#"{"downsample_factor":2,"encoder_patch_size":2,"tile_size":8,"min_tiles":1,"max_tiles":2,"min_image_tokens":4,"max_image_tokens":8}"#,
        )?;
        assert_eq!(config.effective_max_num_patches()?, 32);
        let configured = Lfm2VlProcessorConfig::from_json(
            r#"{"downsample_factor":2,"encoder_patch_size":2,"tile_size":8,"min_tiles":1,"max_tiles":2,"min_image_tokens":4,"max_image_tokens":8,"max_num_patches":40}"#,
        )?;
        assert_eq!(configured.effective_max_num_patches()?, 40);
        assert!(Lfm2VlProcessorConfig::from_json(r#"{"encoder_patch_size":0}"#).is_err());
        Ok(())
    }

    #[test]
    fn parses_the_pinned_wrapped_processor_document() -> Result<()> {
        let config = Lfm2VlProcessorConfig::from_json(
            r#"{
                "image_processor": {
                    "downsample_factor": 2,
                    "encoder_patch_size": 2,
                    "tile_size": 8,
                    "min_tiles": 1,
                    "max_tiles": 2,
                    "min_image_tokens": 4,
                    "max_image_tokens": 8,
                    "max_num_patches": 32
                },
                "processor_class": "Lfm2VlImageProcessor"
            }"#,
        )?;
        assert_eq!(config.encoder_patch_size, 2);
        assert_eq!(config.max_num_patches, Some(32));
        assert_eq!(config.max_image_tokens, 8);
        Ok(())
    }

    #[test]
    fn resolves_typed_config_from_split_mmproj_metadata() -> Result<()> {
        let bundle =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/lfm2_vl_mmproj_tiny");
        let mmproj = Mmproj::load(bundle, DType::F32, &Device::Cpu)?;
        let config = Lfm2VlProcessorConfig::from_mmproj_metadata(&mmproj.metadata)?;
        assert_eq!(config.encoder_patch_size, 2);
        assert_eq!(config.downsample_factor, 2);
        assert_eq!(config.max_num_patches, Some(64));
        assert_eq!(config.context_length, Some(128));
        Ok(())
    }

    #[test]
    fn vision_limit_overrides_follow_processor_precedence() -> Result<()> {
        let model = ProcessorConfigPatch {
            max_images: Some(2),
            ..ProcessorConfigPatch::default()
        };
        let gguf = ProcessorConfigPatch {
            max_images: Some(3),
            ..ProcessorConfigPatch::default()
        };
        let processor = ProcessorConfigPatch {
            max_images: Some(4),
            ..ProcessorConfigPatch::default()
        };
        let explicit = ProcessorConfigPatch {
            max_images: Some(5),
            max_source_pixels: Some(1234),
            ..ProcessorConfigPatch::default()
        };
        let resolved = Lfm2VlProcessorConfig::resolve(
            Some(&explicit),
            Some(&processor),
            Some(&gguf),
            Some(&model),
        )?;
        assert_eq!(resolved.vision_limits.max_images, 5);
        assert_eq!(resolved.vision_limits.max_source_pixels, 1234);

        let parsed = Lfm2VlProcessorConfig::from_json(
            r#"{"max_source_pixels":100,"max_images":2,"max_crops_per_image":5,"max_total_crops":10,"max_patches_per_crop":1024,"max_total_projected_tokens":200}"#,
        )?;
        assert_eq!(parsed.vision_limits.max_source_pixels, 100);
        assert_eq!(parsed.vision_limits.max_images, 2);
        assert_eq!(parsed.vision_limits.max_total_projected_tokens, 200);

        assert!(Lfm2VlProcessorConfig::from_json(r#"{"max_images":0}"#).is_err());
        let above_hard_ceiling = VisionLimits::default().max_source_pixels + 1;
        assert!(Lfm2VlProcessorConfig::from_json(&format!(
            r#"{{"max_source_pixels":{above_hard_ceiling}}}"#
        ))
        .is_err());
        assert!(Lfm2VlProcessorConfig::from_json(
            r#"{"max_num_patches":2048,"max_patches_per_crop":1024}"#
        )
        .is_err());
        Ok(())
    }
}
