struct ImageEncodeRequest<'a> {
    vision_tower: &'a siglip2::Siglip2VisionModel,
    projector: &'a Lfm2VlProjector,
    vision_config: &'a siglip2::Siglip2VisionConfig,
    downsample_factor: usize,
    inputs: &'a ProcessedVisionBatch,
    vision_batch_size: usize,
    limits: &'a VisionLimits,
    capture_trace: bool,
}

fn encode_images_with_parts_trace(
    vision_tower: &siglip2::Siglip2VisionModel,
    projector: &Lfm2VlProjector,
    vision_config: &siglip2::Siglip2VisionConfig,
    downsample_factor: usize,
    inputs: &ProcessedVisionBatch,
    vision_batch_size: usize,
    limits: &VisionLimits,
) -> Result<(EncodedImages, Lfm2VlImageTrace)> {
    let (encoded, trace) = encode_images_with_parts_internal(ImageEncodeRequest {
        vision_tower,
        projector,
        vision_config,
        downsample_factor,
        inputs,
        vision_batch_size,
        limits,
        capture_trace: true,
    })?;
    Ok((
        encoded,
        trace.ok_or_else(|| {
            candle::Error::Msg("LFM2-VL trace capture unexpectedly returned no stages".into())
        })?,
    ))
}

fn encode_images_with_parts_internal(
    request: ImageEncodeRequest<'_>,
) -> Result<(EncodedImages, Option<Lfm2VlImageTrace>)> {
    let ImageEncodeRequest {
        vision_tower,
        projector,
        vision_config,
        downsample_factor,
        inputs,
        vision_batch_size,
        limits,
        capture_trace,
    } = request;
    let expected_patch_dimension = vision_config.patch_dimension_for_vl()?;
    let shapes = preflight_packed_vision_limits(
        inputs,
        expected_patch_dimension,
        downsample_factor,
        vision_batch_size,
        limits,
    )?;
    let (crop_count, _, _) = inputs.pixel_values.dims3()?;
    if capture_trace && crop_count != 1 {
        candle::bail!("LFM2-VL native trace requires exactly one image crop; received {crop_count}")
    }

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
    let mut trace: Option<Lfm2VlImageTrace> = None;
    while crop_start < crop_count {
        let remaining = crop_count - crop_start;
        let chunk_count = remaining.min(vision_batch_size);
        let pixel_values = inputs.pixel_values.narrow(0, crop_start, chunk_count)?;
        let pixel_attention_mask =
            inputs
                .pixel_attention_mask
                .narrow(0, crop_start, chunk_count)?;
        let spatial_shapes = inputs.spatial_shapes.narrow(0, crop_start, chunk_count)?;
        let vision_inputs = siglip2::PackedVisionInputs {
            pixel_values: &pixel_values,
            pixel_attention_mask: &pixel_attention_mask,
            spatial_shapes: &spatial_shapes,
        };
        let (hidden, vision_stages) = if capture_trace {
            let stages = vision_tower.forward_stages(&vision_inputs)?;
            (stages.post_layernorm.clone(), Some(stages))
        } else {
            (vision_tower.forward(&vision_inputs)?, None)
        };
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
            let (projected, projector_stages) = if capture_trace {
                let stages = projector.forward_stages(&crop_hidden)?;
                let projected = stages.output.reshape((
                    super::projected_token_count(rows, cols, downsample_factor)?,
                    projector.output_size(),
                ))?;
                (projected, Some(stages))
            } else {
                let projected = projector.forward(&crop_hidden)?.reshape((
                    super::projected_token_count(rows, cols, downsample_factor)?,
                    projector.output_size(),
                ))?;
                (projected, None)
            };
            let token_count = projected.dim(0)?;
            let end = offset
                .checked_add(token_count)
                .ok_or_else(|| candle::Error::Msg("LFM2-VL image feature range overflow".into()))?;
            per_crop_ranges.push(offset..end);
            offset = end;
            if let (Some(vision_stages), Some(projector_stages)) =
                (vision_stages.as_ref(), projector_stages.as_ref())
            {
                let input = crop_hidden.reshape((valid, vision_config.hidden_size))?;
                trace = Some(Lfm2VlImageTrace {
                    vision_patch_embedding: vision_stages.embeddings.patch_embedding.clone(),
                    vision_resized_position_embedding: vision_stages
                        .embeddings
                        .resized_position_embedding
                        .clone(),
                    vision_embeddings_with_position: vision_stages
                        .embeddings
                        .embeddings_with_position
                        .clone(),
                    vision_encoder_layers: vision_stages.encoder_layers.clone(),
                    vision_last_hidden_state: vision_stages.post_layernorm.clone(),
                    projector: Lfm2VlProjectorTrace {
                        input,
                        pixel_unshuffle: projector_stages.pixel_unshuffle.clone(),
                        layer_norm: projector_stages.layer_norm.clone(),
                        linear_1: projector_stages.linear_1.clone(),
                        activation: projector_stages.activation.clone(),
                        linear_2: projector_stages.linear_2.clone(),
                        output: projected.clone(),
                    },
                });
            }
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
    Ok((
        EncodedImages {
            embeddings,
            per_image_ranges,
            per_crop_ranges,
        },
        trace,
    ))
}

/// Merge projected image features into explicit placeholder spans.
///
/// This is shared by dense native text and quantized GGUF text. The only
/// cross-device value is `encoded_images.embeddings`, transferred here to the
/// text embedding device and dtype immediately before the span replacement.
