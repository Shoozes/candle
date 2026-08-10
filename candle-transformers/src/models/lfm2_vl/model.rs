use crate::models::lfm2_vl::config::{Lfm2VlConfig, VisionLimits};
use crate::models::lfm2_vl::projector::Lfm2VlProjector;
use crate::models::{lfm2, siglip2};
use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use std::ops::Range;

#[derive(Clone, Debug)]
pub enum CropKind {
    Whole,
    Tile { row: usize, col: usize },
    Thumbnail,
}

#[derive(Clone, Debug)]
pub struct CropMeta {
    pub image_index: usize,
    pub crop_index: usize,
    pub kind: CropKind,
    pub patch_rows: usize,
    pub patch_cols: usize,
    pub projected_tokens: usize,
}

#[derive(Clone, Debug)]
pub struct ImageMeta {
    pub crop_range: Range<usize>,
    pub rows: usize,
    pub cols: usize,
    pub resized_width: usize,
    pub resized_height: usize,
}

#[derive(Debug)]
pub struct ProcessedVisionBatch {
    pub pixel_values: Tensor,
    pub pixel_attention_mask: Tensor,
    pub spatial_shapes: Tensor,
    pub crops: Vec<CropMeta>,
    pub images: Vec<ImageMeta>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageTokenSpan {
    /// Spans are ordered one-per-crop, including a thumbnail crop.
    pub batch_index: usize,
    pub start: usize,
    pub end: usize,
}

impl ImageTokenSpan {
    pub fn new(batch_index: usize, start: usize, end: usize) -> Self {
        Self {
            batch_index,
            start,
            end,
        }
    }

    pub fn len(&self) -> Option<usize> {
        self.end.checked_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

#[derive(Debug)]
pub struct EncodedImages {
    /// `[total_projected_tokens, text_hidden]` in crop/image order.
    pub embeddings: Tensor,
    pub per_image_ranges: Vec<Range<usize>>,
    pub per_crop_ranges: Vec<Range<usize>>,
}

#[derive(Debug)]
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
        let input_ids_values = input_ids.to_dtype(DType::U32)?.to_vec2::<u32>()?;
        let image_token_count = input_ids_values
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&token_id| token_id == self.image_token_id)
            .count();
        let input_embeds = self.language_model.embed_tokens(input_ids)?;
        let input_embeds = if image_token_count == 0 {
            if !image_spans.is_empty() || encoded_images.is_some() {
                candle::bail!("LFM2-VL image spans/features were supplied without image tokens")
            }
            input_embeds
        } else {
            let encoded_images = encoded_images.ok_or_else(|| {
                candle::Error::Msg("LFM2-VL image tokens require encoded image features".into())
            })?;
            if image_spans.is_empty() {
                candle::bail!("LFM2-VL image tokens require explicit image spans")
            }
            self.merge_image_embeddings(input_ids, &input_embeds, image_spans, encoded_images)?
        };
        let hidden = self
            .language_model
            .forward_hidden(&input_embeds, 0, cache)?;
        self.language_model.project_logits(&hidden, 0)
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
    let expected_patch_dimension = vision_config.patch_dimension_for_vl()?;
    let shapes = preflight_packed_vision_limits(
        inputs,
        expected_patch_dimension,
        downsample_factor,
        vision_batch_size,
        limits,
    )?;
    let (crop_count, _, _) = inputs.pixel_values.dims3()?;

    let mut projected_crops = Vec::new();
    projected_crops
        .try_reserve_exact(crop_count)
        .map_err(|_| candle::Error::Msg("LFM2-VL projected crop allocation failed".into()))?;
    let mut per_crop_ranges = Vec::new();
    per_crop_ranges
        .try_reserve_exact(crop_count)
        .map_err(|_| candle::Error::Msg("LFM2-VL crop-range allocation failed".into()))?;
    let mut offset = 0usize;
    let mut crop_start = 0usize;
    while crop_start < crop_count {
        let remaining = crop_count - crop_start;
        let chunk_count = remaining.min(vision_batch_size);
        let pixel_values = inputs.pixel_values.narrow(0, crop_start, chunk_count)?;
        let pixel_attention_mask =
            inputs
                .pixel_attention_mask
                .narrow(0, crop_start, chunk_count)?;
        let spatial_shapes = inputs.spatial_shapes.narrow(0, crop_start, chunk_count)?;
        let hidden = vision_tower.forward(&siglip2::PackedVisionInputs {
            pixel_values: &pixel_values,
            pixel_attention_mask: &pixel_attention_mask,
            spatial_shapes: &spatial_shapes,
        })?;
        for local_index in 0..chunk_count {
            let crop_index = crop_start + local_index;
            let (rows, cols) = shapes[crop_index];
            let valid = rows
                .checked_mul(cols)
                .ok_or_else(|| candle::Error::Msg("LFM2-VL valid patch count overflow".into()))?;
            let crop_hidden = hidden.i((local_index, 0..valid, ..))?.reshape((
                1,
                rows,
                cols,
                vision_config.hidden_size,
            ))?;
            let projected = projector.forward(&crop_hidden)?.reshape((
                super::projected_token_count(rows, cols, downsample_factor)?,
                projector.output_size(),
            ))?;
            let token_count = projected.dim(0)?;
            let end = offset
                .checked_add(token_count)
                .ok_or_else(|| candle::Error::Msg("LFM2-VL image feature range overflow".into()))?;
            per_crop_ranges.push(offset..end);
            offset = end;
            projected_crops.push(projected);
        }
        crop_start += chunk_count;
    }
    let mut projected_refs = Vec::new();
    projected_refs
        .try_reserve_exact(projected_crops.len())
        .map_err(|_| {
            candle::Error::Msg("LFM2-VL projected tensor-reference allocation failed".into())
        })?;
    projected_refs.extend(projected_crops.iter());
    let embeddings = Tensor::cat(&projected_refs, 0)?;
    let per_image_ranges = image_ranges_from_crop_ranges(inputs, &per_crop_ranges)?;
    Ok(EncodedImages {
        embeddings,
        per_image_ranges,
        per_crop_ranges,
    })
}

/// Merge projected image features into explicit placeholder spans.
///
/// This is shared by dense native text and quantized GGUF text. The only
/// cross-device value is `encoded_images.embeddings`, transferred here to the
/// text embedding device and dtype immediately before the span replacement.
pub fn merge_projected_embeddings(
    input_ids: &Tensor,
    input_embeds: &Tensor,
    image_token_id: u32,
    image_spans: &[ImageTokenSpan],
    encoded_images: &EncodedImages,
) -> Result<Tensor> {
    let (batch_size, sequence_length) = input_ids.dims2()?;
    let (embed_batch, embed_sequence, hidden_size) = input_embeds.dims3()?;
    if embed_batch != batch_size || embed_sequence != sequence_length {
        candle::bail!("LFM2-VL input embeddings shape does not match input IDs")
    }
    validate_encoded_images(encoded_images)?;
    if image_spans.len() != encoded_images.per_crop_ranges.len() {
        candle::bail!(
            "LFM2-VL image span count {} does not match encoded crop count {}",
            image_spans.len(),
            encoded_images.per_crop_ranges.len()
        )
    }
    if input_embeds.dim(2)? != encoded_images.embeddings.dim(1)? {
        candle::bail!("LFM2-VL image feature width does not match text embedding width")
    }
    let input_ids = input_ids.to_dtype(DType::U32)?.to_vec2::<u32>()?;
    let mut total_span_tokens = 0usize;
    let mut previous: Option<(usize, usize)> = None;
    for (crop_index, span) in image_spans.iter().enumerate() {
        if span.batch_index >= batch_size {
            candle::bail!("LFM2-VL image span batch index is out of bounds")
        }
        if span.start >= span.end || span.end > sequence_length {
            candle::bail!("LFM2-VL image span must be a non-empty in-bounds range")
        }
        if let Some((previous_batch, previous_end)) = previous {
            if span.batch_index < previous_batch
                || (span.batch_index == previous_batch && span.start < previous_end)
            {
                candle::bail!("LFM2-VL image spans must be ordered and non-overlapping")
            }
        }
        for position in span.start..span.end {
            if input_ids[span.batch_index][position] != image_token_id {
                candle::bail!(
                        "LFM2-VL image span contains token {} at batch {}, position {}, expected image token {}",
                        input_ids[span.batch_index][position],
                        span.batch_index,
                        position,
                        image_token_id
                    )
            }
        }
        total_span_tokens = total_span_tokens
            .checked_add(span.end - span.start)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL image span count overflow".into()))?;
        let crop_range = &encoded_images.per_crop_ranges[crop_index];
        let expected_feature_count = crop_range
            .end
            .checked_sub(crop_range.start)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL encoded crop range is invalid".into()))?;
        if span.end - span.start != expected_feature_count {
            candle::bail!(
                    "LFM2-VL crop span {crop_index} has {} placeholders, but its encoded crop range has {expected_feature_count} features",
                    span.end - span.start
                )
        }
        previous = Some((span.batch_index, span.end));
    }

    let mut total_image_tokens = 0usize;
    for row in &input_ids {
        for &token_id in row {
            if token_id == image_token_id {
                total_image_tokens = total_image_tokens.checked_add(1).ok_or_else(|| {
                    candle::Error::Msg("LFM2-VL image token count overflow".into())
                })?;
            }
        }
    }
    if total_image_tokens != total_span_tokens {
        candle::bail!(
                "LFM2-VL image token count {total_image_tokens} does not match span count {total_span_tokens}"
            )
    }
    let feature_count = encoded_images.embeddings.dim(0)?;
    if feature_count != total_image_tokens {
        candle::bail!(
                "LFM2-VL image feature count {feature_count} does not match placeholder count {total_image_tokens}"
            )
    }
    let features = encoded_images
        .embeddings
        .to_device(input_embeds.device())?
        .to_dtype(input_embeds.dtype())?;
    let mut merged = input_embeds.clone();
    for (crop_index, span) in image_spans.iter().enumerate() {
        let span_len = span.end - span.start;
        let feature_range = &encoded_images.per_crop_ranges[crop_index];
        let chunk = features
            .narrow(0, feature_range.start, span_len)?
            .unsqueeze(0)?;
        merged = merged.slice_assign(
            &[
                span.batch_index..span.batch_index + 1,
                span.start..span.end,
                0..hidden_size,
            ],
            &chunk,
        )?;
    }
    Ok(merged)
}

fn validate_image_metadata(
    inputs: &ProcessedVisionBatch,
    crop_count: usize,
    limits: &VisionLimits,
) -> Result<()> {
    if inputs.images.is_empty() {
        candle::bail!("LFM2-VL vision metadata must contain at least one image")
    }
    let mut next_crop = 0usize;
    for (image_index, image) in inputs.images.iter().enumerate() {
        if image.crop_range.start != next_crop
            || image.crop_range.start >= image.crop_range.end
            || image.crop_range.end > crop_count
        {
            candle::bail!("LFM2-VL image crop ranges must be ordered, non-empty, and contiguous")
        }
        limits.check_crops_per_image(image.crop_range.len())?;
        limits.check_image_surface(
            "resized image metadata",
            image.resized_width,
            image.resized_height,
        )?;
        for (local_crop_index, crop_index) in image.crop_range.clone().enumerate() {
            if inputs.crops[crop_index].image_index != image_index
                || inputs.crops[crop_index].crop_index != local_crop_index
            {
                candle::bail!("LFM2-VL crop metadata image index does not match image ranges")
            }
        }
        next_crop = image.crop_range.end;
    }
    if next_crop != crop_count {
        candle::bail!("LFM2-VL image crop ranges do not cover every crop")
    }
    Ok(())
}

fn read_spatial_shapes(spatial_shapes: &Tensor) -> Result<Vec<(usize, usize)>> {
    let values: Vec<Vec<u64>> = match spatial_shapes.dtype() {
        DType::U8 => spatial_shapes
            .to_vec2::<u8>()?
            .into_iter()
            .map(|row| row.into_iter().map(u64::from).collect())
            .collect(),
        DType::U32 => spatial_shapes
            .to_vec2::<u32>()?
            .into_iter()
            .map(|row| row.into_iter().map(u64::from).collect())
            .collect(),
        DType::I16 => spatial_shapes
            .to_vec2::<i16>()?
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|value| {
                        u64::try_from(value).map_err(|_| {
                            candle::Error::Msg(
                                "LFM2-VL spatial shapes cannot contain negative values".into(),
                            )
                        })
                    })
                    .collect::<Result<Vec<u64>>>()
            })
            .collect::<Result<Vec<Vec<u64>>>>()?,
        DType::I32 => spatial_shapes
            .to_vec2::<i32>()?
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|value| {
                        u64::try_from(value).map_err(|_| {
                            candle::Error::Msg(
                                "LFM2-VL spatial shapes cannot contain negative values".into(),
                            )
                        })
                    })
                    .collect::<Result<Vec<u64>>>()
            })
            .collect::<Result<Vec<Vec<u64>>>>()?,
        DType::I64 => spatial_shapes
            .to_vec2::<i64>()?
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|value| {
                        u64::try_from(value).map_err(|_| {
                            candle::Error::Msg(
                                "LFM2-VL spatial shapes cannot contain negative values".into(),
                            )
                        })
                    })
                    .collect::<Result<Vec<u64>>>()
            })
            .collect::<Result<Vec<Vec<u64>>>>()?,
        dtype => candle::bail!("LFM2-VL spatial_shapes must use an integer dtype, got {dtype:?}"),
    };
    values
        .into_iter()
        .map(|row| {
            if row.len() != 2 || row[0] == 0 || row[1] == 0 {
                candle::bail!("LFM2-VL spatial shapes must contain two positive dimensions")
            }
            let rows = usize::try_from(row[0])
                .map_err(|_| candle::Error::Msg("LFM2-VL spatial row is too large".into()))?;
            let cols = usize::try_from(row[1])
                .map_err(|_| candle::Error::Msg("LFM2-VL spatial column is too large".into()))?;
            rows.checked_mul(cols)
                .ok_or_else(|| candle::Error::Msg("LFM2-VL spatial patch count overflow".into()))?;
            Ok((rows, cols))
        })
        .collect()
}

fn read_attention_mask(
    mask: &Tensor,
    shapes: &[(usize, usize)],
    max_patches: usize,
) -> Result<Vec<Vec<f32>>> {
    let values = mask.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    if values.len() != shapes.len() {
        candle::bail!("LFM2-VL attention mask crop count does not match spatial shapes")
    }
    for (crop_index, row) in values.iter().enumerate() {
        if row.len() != max_patches {
            candle::bail!("LFM2-VL attention mask length does not match packed input")
        }
        let valid = shapes[crop_index]
            .0
            .checked_mul(shapes[crop_index].1)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL valid patch count overflow".into()))?;
        if valid > max_patches {
            candle::bail!("LFM2-VL spatial shape exceeds max packed patches")
        }
        for (patch_index, &value) in row.iter().enumerate() {
            let expected = if patch_index < valid { 1.0 } else { 0.0 };
            if !value.is_finite() || value != expected {
                candle::bail!(
                    "LFM2-VL attention mask crop {crop_index} is not a binary valid-prefix mask"
                )
            }
        }
    }
    Ok(values)
}

fn image_ranges_from_crop_ranges(
    inputs: &ProcessedVisionBatch,
    per_crop_ranges: &[Range<usize>],
) -> Result<Vec<Range<usize>>> {
    let mut ranges = Vec::new();
    ranges
        .try_reserve_exact(inputs.images.len())
        .map_err(|_| candle::Error::Msg("LFM2-VL image-range allocation failed".into()))?;
    for image in &inputs.images {
        let first = per_crop_ranges.get(image.crop_range.start).ok_or_else(|| {
            candle::Error::Msg("LFM2-VL image crop range is out of bounds".into())
        })?;
        let last_index = image
            .crop_range
            .end
            .checked_sub(1)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL image crop range is empty".into()))?;
        let last = per_crop_ranges.get(last_index).ok_or_else(|| {
            candle::Error::Msg("LFM2-VL image crop range is out of bounds".into())
        })?;
        ranges.push(first.start..last.end);
    }
    Ok(ranges)
}

fn validate_encoded_images(encoded_images: &EncodedImages) -> Result<()> {
    let feature_count = encoded_images.embeddings.dim(0)?;
    let feature_width = encoded_images.embeddings.dim(1)?;
    if feature_count == 0 || feature_width == 0 {
        candle::bail!("LFM2-VL encoded image features must be non-empty")
    }
    let mut next = 0usize;
    for range in &encoded_images.per_crop_ranges {
        if range.start != next || range.start >= range.end || range.end > feature_count {
            candle::bail!("LFM2-VL encoded crop ranges are not contiguous")
        }
        next = range.end;
    }
    if next != feature_count {
        candle::bail!("LFM2-VL encoded crop ranges do not cover all features")
    }
    next = 0;
    for range in &encoded_images.per_image_ranges {
        if range.start != next || range.start >= range.end || range.end > feature_count {
            candle::bail!("LFM2-VL encoded image ranges are not contiguous")
        }
        next = range.end;
    }
    if next != feature_count {
        candle::bail!("LFM2-VL encoded image ranges do not cover all features")
    }
    let mut crop_index = 0usize;
    for image_range in &encoded_images.per_image_ranges {
        let first = encoded_images
            .per_crop_ranges
            .get(crop_index)
            .ok_or_else(|| {
                candle::Error::Msg("LFM2-VL encoded image range contains no crop ranges".into())
            })?;
        if first.start != image_range.start {
            candle::bail!("LFM2-VL encoded image ranges do not start on crop boundaries")
        }
        let mut union_end = image_range.start;
        while crop_index < encoded_images.per_crop_ranges.len()
            && encoded_images.per_crop_ranges[crop_index].start < image_range.end
        {
            let crop_range = &encoded_images.per_crop_ranges[crop_index];
            if crop_range.start != union_end || crop_range.end > image_range.end {
                candle::bail!("LFM2-VL encoded image ranges split a crop range")
            }
            union_end = crop_range.end;
            crop_index += 1;
        }
        if union_end != image_range.end {
            candle::bail!("LFM2-VL encoded image range is not the union of its crop ranges")
        }
    }
    if crop_index != encoded_images.per_crop_ranges.len() {
        candle::bail!("LFM2-VL encoded crop ranges are not assigned to an image")
    }
    Ok(())
}

impl siglip2::Siglip2VisionConfig {
    pub(super) fn patch_dimension_for_vl(&self) -> Result<usize> {
        self.num_channels
            .checked_mul(self.patch_size)
            .and_then(|value| value.checked_mul(self.patch_size))
            .ok_or_else(|| candle::Error::Msg("LFM2-VL patch dimension overflow".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    const TINY_FIXTURE: &[u8] =
        include_bytes!("../../../../tests/fixtures/lfm2_vl_tiny/tensors.safetensors");

    fn tiny_config() -> Lfm2VlConfig {
        Lfm2VlConfig::from_json(
            &serde_json::json!({
                "model_type": "lfm2_vl",
                "image_token_id": 3,
                "downsample_factor": 2,
                "projector_hidden_size": 24,
                "projector_hidden_act": "gelu",
                "projector_bias": true,
                "projector_use_layernorm": true,
                "text_config": {
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
            .to_string(),
        )
        .expect("fixed LFM2-VL config")
    }

    fn fixture_tensor<'a>(tensors: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
        tensors
            .get(name)
            .ok_or_else(|| candle::Error::Msg(format!("missing tiny fixture tensor {name}")))
    }

    fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32, label: &str) -> Result<()> {
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if actual.len() != expected.len() {
            candle::bail!("{label}: element count mismatch")
        }
        let mut max_abs = 0f32;
        let mut dot = 0f32;
        let mut actual_norm = 0f32;
        let mut expected_norm = 0f32;
        for (&lhs, &rhs) in actual.iter().zip(&expected) {
            max_abs = max_abs.max((lhs - rhs).abs());
            dot += lhs * rhs;
            actual_norm += lhs * lhs;
            expected_norm += rhs * rhs;
        }
        let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt());
        eprintln!("lfm2_vl composite {label}: max_abs={max_abs:.9e}, cosine={cosine:.9}");
        assert!(
            max_abs <= tolerance,
            "{label}: max_abs={max_abs} > {tolerance}"
        );
        assert!(
            cosine.is_finite() && cosine >= 0.99999,
            "{label}: cosine={cosine}"
        );
        Ok(())
    }

    fn fixture_batch(tensors: &HashMap<String, Tensor>) -> Result<ProcessedVisionBatch> {
        Ok(ProcessedVisionBatch {
            pixel_values: fixture_tensor(tensors, "input.pixel_values")?.clone(),
            pixel_attention_mask: fixture_tensor(tensors, "input.pixel_attention_mask")?.clone(),
            spatial_shapes: fixture_tensor(tensors, "input.spatial_shapes")?.clone(),
            crops: vec![CropMeta {
                image_index: 0,
                crop_index: 0,
                kind: CropKind::Whole,
                patch_rows: 2,
                patch_cols: 4,
                projected_tokens: 2,
            }],
            images: vec![ImageMeta {
                crop_range: 0..1,
                rows: 2,
                cols: 4,
                resized_width: 4,
                resized_height: 2,
            }],
        })
    }

    fn tiny_model(device: &Device) -> Result<(Lfm2VlModel, HashMap<String, Tensor>)> {
        let weights = VarBuilder::from_slice_safetensors(TINY_FIXTURE, DType::F32, device)?;
        let tensors = candle::safetensors::load_buffer(TINY_FIXTURE, device)?;
        let model = Lfm2VlModel::new(&tiny_config(), weights.pp("weights"))?;
        Ok((model, tensors))
    }

    fn image_spans(input_ids: &Tensor, image_token_id: u32) -> Result<Vec<ImageTokenSpan>> {
        let values = input_ids.to_dtype(DType::U32)?.to_vec2::<u32>()?;
        let mut spans = Vec::new();
        for (batch_index, row) in values.iter().enumerate() {
            let mut start = None;
            for (position, &token_id) in row.iter().enumerate() {
                if token_id == image_token_id && start.is_none() {
                    start = Some(position);
                }
                if token_id != image_token_id {
                    if let Some(start) = start.take() {
                        spans.push(ImageTokenSpan::new(batch_index, start, position));
                    }
                }
            }
            if let Some(start) = start {
                spans.push(ImageTokenSpan::new(batch_index, start, row.len()));
            }
        }
        Ok(spans)
    }

    #[test]
    fn fixture_matches_projector_merged_embeddings_and_prefill_decode() -> Result<()> {
        let device = Device::Cpu;
        let (model, tensors) = tiny_model(&device)?;
        let batch = fixture_batch(&tensors)?;
        let encoded = model.encode_images(&batch, 1)?;
        assert_close(
            &encoded.embeddings,
            fixture_tensor(&tensors, "stage.projector.output")?,
            2e-5,
            "encoded projector output",
        )?;
        assert_eq!(encoded.per_crop_ranges, vec![0..2]);
        assert_eq!(encoded.per_image_ranges, vec![0..2]);

        let input_ids = fixture_tensor(&tensors, "input.input_ids")?;
        let input_embeds = model.embed_tokens(input_ids)?;
        let spans = image_spans(input_ids, 3)?;
        let merged = model.merge_image_embeddings(input_ids, &input_embeds, &spans, &encoded)?;
        assert_close(
            &merged,
            fixture_tensor(&tensors, "stage.multimodal.merged_embeddings")?,
            1e-6,
            "merged embeddings",
        )?;

        let text_config = tiny_config().text_model_config()?;
        let mut cache = lfm2::Cache::new(true, DType::F32, &text_config, &device)?;
        let prefill = model.prefill(input_ids, &spans, Some(&encoded), &mut cache)?;
        assert_close(
            &prefill,
            fixture_tensor(&tensors, "stage.language.prefill_logits")?,
            1e-3,
            "multimodal prefill logits",
        )?;
        let decode_ids = fixture_tensor(&tensors, "input.decode_token_ids")?;
        let decode_expected = fixture_tensor(&tensors, "stage.language.decode_logits")?;
        for step in 0..3 {
            let token = decode_ids.i((.., step..step + 1))?;
            let logits = model.decode(&token, 5 + step, &mut cache)?;
            assert_close(
                &logits,
                &decode_expected.i((.., step, ..))?,
                1e-3,
                &format!("cached decode step {step}"),
            )?;
        }
        cache.clear();
        let reset = model.prefill(input_ids, &spans, Some(&encoded), &mut cache)?;
        assert_close(
            &reset,
            fixture_tensor(&tensors, "stage.language.prefill_logits")?,
            1e-3,
            "cache-reset prefill logits",
        )?;
        let reset_decode = model.decode(&decode_ids.i((.., 0..1))?, 5, &mut cache)?;
        assert_close(
            &reset_decode,
            &decode_expected.i((.., 0, ..))?,
            1e-3,
            "cache-reset decode logits",
        )?;
        Ok(())
    }

    #[test]
    fn packed_model_boundary_enforces_exact_and_one_over_limits() -> Result<()> {
        let device = Device::Cpu;
        let (model, tensors) = tiny_model(&device)?;
        let batch = fixture_batch(&tensors)?;
        let (_, max_patches, _) = batch.pixel_values.dims3()?;
        let source_pixels = batch.images[0]
            .resized_width
            .checked_mul(batch.images[0].resized_height)
            .ok_or_else(|| candle::Error::Msg("test image surface overflow".into()))?;
        let exact = VisionLimits {
            max_source_pixels: source_pixels,
            max_images: 1,
            max_crops_per_image: 1,
            max_total_crops: 1,
            max_patches_per_crop: max_patches,
            max_total_projected_tokens: 2,
        };
        let encoded = model.encode_images_with_limits(&batch, 1, &exact)?;
        assert_eq!(encoded.embeddings.dims(), [2, 12]);

        let token_over = VisionLimits {
            max_total_projected_tokens: 1,
            ..exact
        };
        let error = model
            .encode_images_with_limits(&batch, 1, &token_over)
            .expect_err("packed boundary must reject projected-token overage");
        assert!(error.to_string().contains("2 projected tokens"));

        let patch_over = VisionLimits {
            max_patches_per_crop: max_patches - 1,
            ..exact
        };
        let error = model
            .encode_images_with_limits(&batch, 1, &patch_over)
            .expect_err("packed boundary must reject padded patch overage");
        assert!(error.to_string().contains("patch slots per crop"));

        let surface_over = VisionLimits {
            max_source_pixels: exact.max_source_pixels - 1,
            ..exact
        };
        let error = model
            .encode_images_with_limits(&batch, 1, &surface_over)
            .expect_err("packed boundary must reject resized-surface overage");
        assert!(error.to_string().contains("resized image metadata"));

        let error = model
            .encode_images_with_limits(&batch, 0, &exact)
            .expect_err("packed boundary must reject a zero batch size before execution");
        assert!(error.to_string().contains("vision_batch_size"));
        Ok(())
    }

    #[test]
    fn multiple_crops_and_images_preserve_order_and_ranges() -> Result<()> {
        let device = Device::Cpu;
        let (model, tensors) = tiny_model(&device)?;
        let single = fixture_batch(&tensors)?;
        let pixel_values = Tensor::cat(
            &[
                &single.pixel_values,
                &single.pixel_values,
                &single.pixel_values,
            ],
            0,
        )?;
        let mask = Tensor::cat(
            &[
                &single.pixel_attention_mask,
                &single.pixel_attention_mask,
                &single.pixel_attention_mask,
            ],
            0,
        )?;
        let shapes = Tensor::cat(
            &[
                &single.spatial_shapes,
                &single.spatial_shapes,
                &single.spatial_shapes,
            ],
            0,
        )?;
        let batch = ProcessedVisionBatch {
            pixel_values,
            pixel_attention_mask: mask,
            spatial_shapes: shapes,
            crops: vec![
                CropMeta {
                    image_index: 0,
                    crop_index: 0,
                    kind: CropKind::Tile { row: 0, col: 0 },
                    patch_rows: 2,
                    patch_cols: 4,
                    projected_tokens: 2,
                },
                CropMeta {
                    image_index: 0,
                    crop_index: 1,
                    kind: CropKind::Thumbnail,
                    patch_rows: 2,
                    patch_cols: 4,
                    projected_tokens: 2,
                },
                CropMeta {
                    image_index: 1,
                    crop_index: 0,
                    kind: CropKind::Whole,
                    patch_rows: 2,
                    patch_cols: 4,
                    projected_tokens: 2,
                },
            ],
            images: vec![
                ImageMeta {
                    crop_range: 0..2,
                    rows: 2,
                    cols: 4,
                    resized_width: 8,
                    resized_height: 2,
                },
                ImageMeta {
                    crop_range: 2..3,
                    rows: 2,
                    cols: 4,
                    resized_width: 4,
                    resized_height: 2,
                },
            ],
        };
        let encoded = model.encode_images(&batch, 2)?;
        assert_eq!(encoded.embeddings.dims(), [6, 12]);
        assert_eq!(encoded.per_crop_ranges, vec![0..2, 2..4, 4..6]);
        assert_eq!(encoded.per_image_ranges, vec![0..4, 4..6]);
        assert_close(
            &encoded.embeddings.narrow(0, 0, 2)?,
            fixture_tensor(&tensors, "stage.projector.output")?,
            2e-5,
            "first crop order",
        )?;
        assert_close(
            &encoded.embeddings.narrow(0, 4, 2)?,
            fixture_tensor(&tensors, "stage.projector.output")?,
            2e-5,
            "second image order",
        )?;
        Ok(())
    }

    #[test]
    fn rejects_malformed_packed_shapes_and_masks() -> Result<()> {
        let device = Device::Cpu;
        let (model, tensors) = tiny_model(&device)?;
        let single = fixture_batch(&tensors)?;
        let bad_mask = ProcessedVisionBatch {
            pixel_values: single.pixel_values.clone(),
            pixel_attention_mask: single.pixel_attention_mask.narrow(1, 0, 9)?,
            spatial_shapes: single.spatial_shapes.clone(),
            crops: single.crops.clone(),
            images: single.images.clone(),
        };
        assert!(model.encode_images(&bad_mask, 1).is_err());

        let bad_spatial = ProcessedVisionBatch {
            pixel_values: single.pixel_values.clone(),
            pixel_attention_mask: single.pixel_attention_mask.clone(),
            spatial_shapes: Tensor::new(&[[-1i64, 4]], &device)?,
            crops: single.crops.clone(),
            images: single.images.clone(),
        };
        assert!(model.encode_images(&bad_spatial, 1).is_err());

        let nonbinary_mask = ProcessedVisionBatch {
            pixel_values: single.pixel_values,
            pixel_attention_mask: Tensor::ones((1, 10), DType::F32, &device)?,
            spatial_shapes: Tensor::new(&[[2i64, 4]], &device)?,
            crops: single.crops,
            images: single.images,
        };
        assert!(model.encode_images(&nonbinary_mask, 1).is_err());
        Ok(())
    }

    #[test]
    fn span_validation_rejects_bad_ranges_tokens_and_counts() -> Result<()> {
        let device = Device::Cpu;
        let (model, tensors) = tiny_model(&device)?;
        let batch = fixture_batch(&tensors)?;
        let encoded = model.encode_images(&batch, 1)?;
        let input_ids = fixture_tensor(&tensors, "input.input_ids")?;
        let embeds = model.embed_tokens(input_ids)?;
        let spans = image_spans(input_ids, 3)?;
        assert!(model
            .merge_image_embeddings(
                input_ids,
                &embeds,
                &[ImageTokenSpan::new(0, spans[0].start, spans[0].end + 1)],
                &encoded,
            )
            .is_err());
        assert!(model
            .merge_image_embeddings(
                input_ids,
                &embeds,
                &[
                    ImageTokenSpan::new(0, spans[0].start, spans[0].start + 1),
                    ImageTokenSpan::new(0, spans[0].start, spans[0].start + 1),
                ],
                &EncodedImages {
                    embeddings: encoded.embeddings.clone(),
                    per_image_ranges: vec![0..1, 1..2],
                    per_crop_ranges: vec![0..1, 1..2],
                },
            )
            .is_err());
        assert!(model
            .merge_image_embeddings(
                input_ids,
                &embeds,
                &[ImageTokenSpan::new(0, 0, input_ids.dim(1)? + 1)],
                &encoded,
            )
            .is_err());
        let wrong_token = ImageTokenSpan::new(0, 0, 1);
        assert!(model
            .merge_image_embeddings(input_ids, &embeds, &[wrong_token], &encoded)
            .is_err());
        let short = EncodedImages {
            embeddings: encoded.embeddings.narrow(0, 0, 1)?,
            per_image_ranges: std::iter::once(0..1).collect(),
            per_crop_ranges: std::iter::once(0..1).collect(),
        };
        assert!(model
            .merge_image_embeddings(input_ids, &embeds, &spans, &short)
            .is_err());
        Ok(())
    }

    #[test]
    fn multiple_spans_pair_with_exact_per_crop_feature_ranges() -> Result<()> {
        let device = Device::Cpu;
        let (model, tensors) = tiny_model(&device)?;
        let batch = fixture_batch(&tensors)?;
        let one_image = model.encode_images(&batch, 1)?;
        let three_crop_embeddings = Tensor::cat(
            &[
                &one_image.embeddings,
                &one_image.embeddings,
                &one_image.embeddings,
            ],
            0,
        )?;
        let encoded = EncodedImages {
            embeddings: three_crop_embeddings,
            per_image_ranges: vec![0..4, 4..6],
            per_crop_ranges: vec![0..2, 2..4, 4..6],
        };
        let input_ids = Tensor::new(&[[3i64, 3, 5, 3, 3, 5, 3, 3]], &device)?;
        let input_embeds = model.embed_tokens(&input_ids)?;
        let spans = [
            ImageTokenSpan::new(0, 0, 2),
            ImageTokenSpan::new(0, 3, 5),
            ImageTokenSpan::new(0, 6, 8),
        ];
        let merged = model.merge_image_embeddings(&input_ids, &input_embeds, &spans, &encoded)?;
        assert_close(
            &merged.i((.., 0..2, ..))?,
            &encoded.embeddings.i((0..2, ..))?.unsqueeze(0)?,
            0.0,
            "first crop span insertion",
        )?;
        assert_close(
            &merged.i((.., 3..5, ..))?,
            &encoded.embeddings.i((2..4, ..))?.unsqueeze(0)?,
            0.0,
            "second crop span insertion",
        )?;
        assert_close(
            &merged.i((.., 6..8, ..))?,
            &encoded.embeddings.i((4..6, ..))?.unsqueeze(0)?,
            0.0,
            "third crop span insertion",
        )?;
        assert_close(
            &merged.i((0, 2, ..))?,
            &input_embeds.i((0, 2, ..))?,
            0.0,
            "untouched tile marker embedding",
        )?;
        assert_close(
            &merged.i((0, 5, ..))?,
            &input_embeds.i((0, 5, ..))?,
            0.0,
            "untouched thumbnail marker embedding",
        )?;

        let mismatched_crop_lengths = EncodedImages {
            embeddings: encoded.embeddings.clone(),
            per_image_ranges: vec![0..2, 2..6],
            per_crop_ranges: vec![0..2, 2..5, 5..6],
        };
        assert!(model
            .merge_image_embeddings(&input_ids, &input_embeds, &spans, &mismatched_crop_lengths)
            .is_err());
        let image_range_splits_crop = EncodedImages {
            embeddings: encoded.embeddings.clone(),
            per_image_ranges: vec![0..3, 3..6],
            per_crop_ranges: vec![0..2, 2..4, 4..6],
        };
        assert!(model
            .merge_image_embeddings(&input_ids, &input_embeds, &spans, &image_range_splits_crop)
            .is_err());
        Ok(())
    }
}
