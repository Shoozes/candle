/// Candle SigLIP2 NaFlex vision encoder for packed patch tensors.
#[derive(Debug)]
pub struct Siglip2VisionModel {
    config: Siglip2VisionConfig,
    embeddings: VisionEmbeddings,
    encoder: Vec<EncoderLayer>,
    post_layernorm: LayerNorm,
}

impl Siglip2VisionModel {
    /// Load the model relative to the production `model.vision_tower.vision_model`
    /// namespace. Callers should pass a `VarBuilder` already positioned at
    /// `vision_model`.
    pub fn new(config: &Siglip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        config.validate()?;
        let embeddings = VisionEmbeddings::new(config, vb.pp("embeddings"))?;
        let encoder_vb = vb.pp("encoder").pp("layers");
        let mut encoder = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            encoder.push(EncoderLayer::new(config, encoder_vb.pp(index))?);
        }
        let post_layernorm = layer_norm(
            config.hidden_size,
            LayerNormConfig {
                eps: config.layer_norm_eps,
                ..LayerNormConfig::default()
            },
            vb.pp("post_layernorm"),
        )?;
        Ok(Self {
            config: config.clone(),
            embeddings,
            encoder,
            post_layernorm,
        })
    }

    pub fn device(&self) -> &Device {
        &self.embeddings.device
    }

    pub fn dtype(&self) -> DType {
        self.embeddings.dtype
    }

    pub(crate) fn new_with_quantized_linears(
        config: &Siglip2VisionConfig,
        vb: VarBuilder,
        mut quantized_weights: HashMap<String, QTensor>,
    ) -> Result<Self> {
        config.validate()?;
        let embeddings = VisionEmbeddings::new(config, vb.pp("embeddings"))?;
        let encoder_vb = vb.pp("encoder").pp("layers");
        let mut encoder = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            encoder.push(EncoderLayer::new_with_quantized_linears(
                config,
                encoder_vb.pp(index),
                &mut quantized_weights,
                index,
            )?);
        }
        if !quantized_weights.is_empty() {
            let mut names: Vec<_> = quantized_weights.into_keys().collect();
            names.sort();
            candle::bail!("unused SigLIP2 quantized linear weights: {names:?}")
        }
        let post_layernorm = layer_norm(
            config.hidden_size,
            LayerNormConfig {
                eps: config.layer_norm_eps,
                ..LayerNormConfig::default()
            },
            vb.pp("post_layernorm"),
        )?;
        Ok(Self {
            config: config.clone(),
            embeddings,
            encoder,
            post_layernorm,
        })
    }

    /// Run packed vision inputs and return `[crop_count, max_patches, hidden_size]`.
    pub fn forward(&self, inputs: &PackedVisionInputs<'_>) -> Result<Tensor> {
        Ok(self.forward_stages(inputs)?.post_layernorm)
    }

    pub(crate) fn forward_stages(&self, inputs: &PackedVisionInputs<'_>) -> Result<ForwardStages> {
        let (crop_count, max_patches, patch_dimension) = inputs.pixel_values.dims3()?;
        if crop_count == 0 || max_patches == 0 {
            candle::bail!("SigLIP2 packed inputs must contain at least one crop and patch slot")
        }
        if inputs.pixel_attention_mask.dims() != [crop_count, max_patches] {
            candle::bail!(
                "SigLIP2 pixel_attention_mask has shape {:?}, expected [{crop_count}, {max_patches}]",
                inputs.pixel_attention_mask.dims()
            )
        }
        if inputs.spatial_shapes.dims() != [crop_count, 2] {
            candle::bail!(
                "SigLIP2 spatial_shapes has shape {:?}, expected [{crop_count}, 2]",
                inputs.spatial_shapes.dims()
            )
        }
        if patch_dimension != self.config.patch_dimension()? {
            candle::bail!(
                "SigLIP2 pixel_values patch dimension is {patch_dimension}, expected {}",
                self.config.patch_dimension()?
            )
        }
        let shapes = read_spatial_shapes(inputs.spatial_shapes)?;
        let mask = validate_attention_mask(inputs.pixel_attention_mask, &shapes, max_patches)?;
        let embeddings = self.embeddings.forward(inputs, &shapes)?;
        let mut hidden = embeddings.embeddings_with_position.clone();
        let mut encoder_layers = Vec::with_capacity(self.encoder.len());
        for layer in &self.encoder {
            hidden = layer.forward(&hidden, &mask)?;
            encoder_layers.push(hidden.clone());
        }
        let post_layernorm = self.post_layernorm.forward(&hidden)?;
        Ok(ForwardStages {
            embeddings,
            encoder_layers,
            post_layernorm,
        })
    }
}
