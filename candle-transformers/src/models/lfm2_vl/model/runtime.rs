pub struct Lfm2VlModel {
    vision_tower: siglip2::Siglip2VisionModel,
    projector: Lfm2VlProjector,
    language_model: lfm2::Model,
    image_token_id: u32,
    config: Lfm2VlConfig,
}

impl Lfm2VlModel {
    /// Load a unified checkpoint rooted at `model.*` and `lm_head.*`.
    ///
    /// The caller may pass a VarBuilder at the safetensors root or already at
    /// `model`. The vision `vision_model` component is selected by inspecting
    /// the available tensor namespace, which also supports the direct
    /// vision-tower root used by the committed tiny fixture.
    pub fn new(config: &Lfm2VlConfig, vb: VarBuilder) -> Result<Self> {
        config.validate()?;
        let has_model_prefix = vb.contains_tensor("model.language_model.embed_tokens.weight")
            || vb.contains_tensor("model.multi_modal_projector.linear_1.weight")
            || vb.contains_tensor("model.vision_tower.embeddings.patch_embedding.weight")
            || vb.contains_tensor(
                "model.vision_tower.vision_model.embeddings.patch_embedding.weight",
            );
        let has_model_root = vb.contains_tensor("language_model.embed_tokens.weight")
            || vb.contains_tensor("multi_modal_projector.linear_1.weight")
            || vb.contains_tensor("vision_tower.embeddings.patch_embedding.weight")
            || vb.contains_tensor("vision_tower.vision_model.embeddings.patch_embedding.weight");
        let model_vb = if has_model_prefix {
            vb.pp("model")
        } else if has_model_root {
            vb.clone()
        } else {
            vb.pp("model")
        };

        let vision_tower_vb = model_vb.pp("vision_tower");
        let text_config = config.text_model_config()?;
        let lm_head_vb = if text_config.tie_embedding {
            None
        } else if has_model_prefix {
            Some(vb.pp("lm_head"))
        } else if has_model_root {
            candle::bail!(
                "untied LFM2-VL construction at model root requires explicit lm_head root; use new_from_parts"
            )
        } else {
            Some(vb.pp("lm_head"))
        };
        Self::new_from_parts(
            config,
            vision_tower_vb,
            model_vb.pp("multi_modal_projector"),
            model_vb.pp("language_model"),
            lm_head_vb,
        )
    }

    /// Construct from explicit component roots.
    ///
    /// `vision_tower_vb` is relative to `vision_tower` and may contain the
    /// official `vision_model` component or the direct vision-model tensors.
    /// The other builders are relative to `multi_modal_projector` and
    /// `language_model`; `lm_head_vb` is optional only for tied embeddings.
    pub fn new_from_parts(
        config: &Lfm2VlConfig,
        vision_tower_vb: VarBuilder,
        projector_vb: VarBuilder,
        language_vb: VarBuilder,
        lm_head_vb: Option<VarBuilder>,
    ) -> Result<Self> {
        config.validate()?;
        let vision_vb = if vision_tower_vb
            .contains_tensor("vision_model.embeddings.patch_embedding.weight")
            || !vision_tower_vb.contains_tensor("embeddings.patch_embedding.weight")
        {
            vision_tower_vb.pp("vision_model")
        } else {
            vision_tower_vb
        };
        let vision_tower = siglip2::Siglip2VisionModel::new(&config.vision_config, vision_vb)?;
        let projector = Lfm2VlProjector::new(config, projector_vb)?;
        let text_config = config.text_model_config()?;
        let language_model = lfm2::Model::new_from_parts(&text_config, language_vb, lm_head_vb)?;
        Ok(Self {
            vision_tower,
            projector,
            language_model,
            image_token_id: config.image_token_id,
            config: config.clone(),
        })
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.language_model.embed_tokens(input_ids)
    }

    pub fn config(&self) -> &Lfm2VlConfig {
        &self.config
    }

    pub fn vision_device(&self) -> &Device {
        self.vision_tower.device()
    }

    pub fn text_device(&self) -> &Device {
        self.language_model.device()
    }

    pub fn vision_dtype(&self) -> DType {
        self.vision_tower.dtype()
    }

    pub fn text_dtype(&self) -> DType {
        self.language_model.dtype()
    }

    pub fn encode_images(
        &self,
        inputs: &ProcessedVisionBatch,
        vision_batch_size: usize,
    ) -> Result<EncodedImages> {
        self.encode_images_with_limits(inputs, vision_batch_size, &VisionLimits::default())
    }

    pub fn encode_images_with_limits(
        &self,
        inputs: &ProcessedVisionBatch,
        vision_batch_size: usize,
        limits: &VisionLimits,
    ) -> Result<EncodedImages> {
        encode_images_with_parts(
            &self.vision_tower,
            &self.projector,
            &self.config.vision_config,
            self.config.downsample_factor,
            inputs,
            vision_batch_size,
            limits,
        )
    }

    /// Encode one crop while retaining the selected vision/projector stages.
    ///
    /// This is an explicit parity/debugging API. It is not used by ordinary
    /// inference and rejects multi-crop batches to keep activation retention
    /// bounded and deterministic.
    pub fn encode_images_with_trace(
        &self,
        inputs: &ProcessedVisionBatch,
        vision_batch_size: usize,
        limits: &VisionLimits,
    ) -> Result<(EncodedImages, Lfm2VlImageTrace)> {
        encode_images_with_parts_trace(
            &self.vision_tower,
            &self.projector,
            &self.config.vision_config,
            self.config.downsample_factor,
            inputs,
            vision_batch_size,
            limits,
        )
    }

    pub fn merge_image_embeddings(
        &self,
        input_ids: &Tensor,
        input_embeds: &Tensor,
        image_spans: &[ImageTokenSpan],
        encoded_images: &EncodedImages,
    ) -> Result<Tensor> {
        merge_projected_embeddings(
            input_ids,
            input_embeds,
            self.image_token_id,
            image_spans,
            encoded_images,
        )
    }

    pub fn prefill(
        &self,
        input_ids: &Tensor,
        image_spans: &[ImageTokenSpan],
        encoded_images: Option<&EncodedImages>,
        cache: &mut lfm2::Cache,
    ) -> Result<Tensor> {
        Ok(self
            .prefill_with_trace(input_ids, image_spans, encoded_images, cache)?
            .logits)
    }

    pub fn prefill_with_trace(
        &self,
        input_ids: &Tensor,
        image_spans: &[ImageTokenSpan],
        encoded_images: Option<&EncodedImages>,
        cache: &mut lfm2::Cache,
    ) -> Result<Lfm2VlPrefillTrace> {
        let input_ids_values = input_ids.to_dtype(DType::U32)?.to_vec2::<u32>()?;
        let image_token_count = input_ids_values
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&token_id| token_id == self.image_token_id)
            .count();
        let input_embeddings = self.language_model.embed_tokens(input_ids)?;
        let merged_embeddings = if image_token_count == 0 {
            if !image_spans.is_empty() || encoded_images.is_some() {
                candle::bail!("LFM2-VL image spans/features were supplied without image tokens")
            }
            input_embeddings.clone()
        } else {
            let encoded_images = encoded_images.ok_or_else(|| {
                candle::Error::Msg("LFM2-VL image tokens require encoded image features".into())
            })?;
            if image_spans.is_empty() {
                candle::bail!("LFM2-VL image tokens require explicit image spans")
            }
            self.merge_image_embeddings(input_ids, &input_embeddings, image_spans, encoded_images)?
        };
        let hidden = self
            .language_model
            .forward_hidden(&merged_embeddings, 0, cache)?;
        let logits = self.language_model.project_logits(&hidden, 0)?;
        Ok(Lfm2VlPrefillTrace {
            input_embeddings,
            merged_embeddings,
            hidden_states: hidden,
            logits,
        })
    }

    pub fn decode(
        &self,
        token_ids: &Tensor,
        index_pos: usize,
        cache: &mut lfm2::Cache,
    ) -> Result<Tensor> {
        token_ids.dims2()?;
        self.language_model.forward(token_ids, index_pos, cache)
    }

    pub fn decode_with_trace(
        &self,
        token_ids: &Tensor,
        index_pos: usize,
        cache: &mut lfm2::Cache,
    ) -> Result<Lfm2VlDecodeTrace> {
        token_ids.dims2()?;
        let input_embeddings = self.language_model.embed_tokens(token_ids)?;
        let hidden_states =
            self.language_model
                .forward_hidden(&input_embeddings, index_pos, cache)?;
        let logits = self.language_model.project_logits(&hidden_states, 1)?;
        Ok(Lfm2VlDecodeTrace {
            input_embeddings,
            hidden_states,
            logits,
        })
    }
}

pub(super) fn preflight_packed_vision_limits(
    inputs: &ProcessedVisionBatch,
    expected_patch_dimension: usize,
    downsample_factor: usize,
    vision_batch_size: usize,
    limits: &VisionLimits,
) -> Result<Vec<(usize, usize)>> {
    limits.validate()?;
    if vision_batch_size == 0 {
        candle::bail!("LFM2-VL vision_batch_size must be greater than zero")
    }
    let (crop_count, max_patches, patch_dimension) = inputs.pixel_values.dims3()?;
    if crop_count == 0 || max_patches == 0 {
        candle::bail!("LFM2-VL packed vision input must contain at least one crop")
    }
    if patch_dimension != expected_patch_dimension {
        candle::bail!(
            "LFM2-VL packed input patch dimension {patch_dimension} does not match model {expected_patch_dimension}"
        )
    }
    if inputs.pixel_attention_mask.dims() != [crop_count, max_patches] {
        candle::bail!("LFM2-VL pixel_attention_mask shape does not match pixel_values")
    }
    if inputs.spatial_shapes.dims() != [crop_count, 2] {
        candle::bail!("LFM2-VL spatial_shapes shape does not match pixel_values")
    }
    if inputs.crops.len() != crop_count {
        candle::bail!(
            "LFM2-VL crop metadata count {} does not match tensor crop count {crop_count}",
            inputs.crops.len()
        )
    }
    limits.check_image_count(inputs.images.len())?;
    limits.check_total_crops(crop_count)?;
    if max_patches > limits.max_patches_per_crop {
        candle::bail!(
            "LFM2-VL packed input has {max_patches} patch slots per crop, exceeding limit {}",
            limits.max_patches_per_crop
        )
    }
    validate_image_metadata(inputs, crop_count, limits)?;
    let shapes = read_spatial_shapes(&inputs.spatial_shapes)?;
    let mask_values = read_attention_mask(&inputs.pixel_attention_mask, &shapes, max_patches)?;
    let mut total_projected_tokens = 0usize;
    for (crop_index, crop) in inputs.crops.iter().enumerate() {
        let (rows, cols) = shapes[crop_index];
        if crop.patch_rows != rows || crop.patch_cols != cols {
            candle::bail!(
                "LFM2-VL crop {crop_index} metadata grid [{}, {}] does not match input [{rows}, {cols}]",
                crop.patch_rows,
                crop.patch_cols
            )
        }
        let projected_tokens = super::projected_token_count(rows, cols, downsample_factor)?;
        if crop.projected_tokens != projected_tokens {
            candle::bail!(
                "LFM2-VL crop {crop_index} metadata projects to {}, expected {projected_tokens}",
                crop.projected_tokens
            )
        }
        limits.check_crop(rows, cols, projected_tokens)?;
        total_projected_tokens = total_projected_tokens
            .checked_add(projected_tokens)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL projected token total overflow".into()))?;
        let valid = rows
            .checked_mul(cols)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL valid patch count overflow".into()))?;
        let mask_valid = mask_values[crop_index]
            .iter()
            .filter(|&&value| value == 1.0)
            .count();
        if mask_valid != valid {
            candle::bail!(
                "LFM2-VL crop {crop_index} mask has {mask_valid} valid patches, expected {valid}"
            )
        }
    }
    limits.check_total_projected_tokens(total_projected_tokens)?;
    Ok(shapes)
}

pub(super) fn encode_images_with_parts(
    vision_tower: &siglip2::Siglip2VisionModel,
    projector: &Lfm2VlProjector,
    vision_config: &siglip2::Siglip2VisionConfig,
    downsample_factor: usize,
    inputs: &ProcessedVisionBatch,
    vision_batch_size: usize,
    limits: &VisionLimits,
) -> Result<EncodedImages> {
    let (encoded, _) = encode_images_with_parts_internal(ImageEncodeRequest {
        vision_tower,
        projector,
        vision_config,
        downsample_factor,
        inputs,
        vision_batch_size,
        limits,
        capture_trace: false,
    })?;
    Ok(encoded)
}
