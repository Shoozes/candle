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
