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
