fn square_side(value: usize, label: &str) -> Result<usize> {
    if value == 0 {
        candle::bail!("{label} must be a non-zero square")
    }
    let mut side = (value as f64).sqrt() as usize;
    while side
        .checked_mul(side)
        .map(|square| square < value)
        .unwrap_or(false)
    {
        side = side
            .checked_add(1)
            .ok_or_else(|| candle::Error::Msg(format!("{label} square root overflow")))?;
    }
    while side
        .checked_mul(side)
        .map(|square| square > value)
        .unwrap_or(true)
    {
        if side == 0 {
            candle::bail!("{label} is not a square")
        }
        side -= 1;
    }
    if side.checked_mul(side) != Some(value) {
        candle::bail!("{label}={value} must be a perfect square")
    }
    Ok(side)
}

fn read_spatial_shapes(spatial_shapes: &Tensor) -> Result<Vec<(usize, usize)>> {
    let values = read_integer_matrix(spatial_shapes, "spatial_shapes")?;
    values
        .into_iter()
        .map(|row| {
            if row.len() != 2 || row[0] == 0 || row[1] == 0 {
                candle::bail!("SigLIP2 spatial shape must contain two positive dimensions")
            }
            let rows = usize::try_from(row[0])
                .map_err(|_| candle::Error::Msg("SigLIP2 spatial row does not fit usize".into()))?;
            let cols = usize::try_from(row[1]).map_err(|_| {
                candle::Error::Msg("SigLIP2 spatial column does not fit usize".into())
            })?;
            rows.checked_mul(cols)
                .ok_or_else(|| candle::Error::Msg("SigLIP2 spatial patch count overflow".into()))?;
            Ok((rows, cols))
        })
        .collect()
}

fn read_integer_matrix(tensor: &Tensor, label: &str) -> Result<Vec<Vec<u64>>> {
    let values = match tensor.dtype() {
        DType::U8 => tensor
            .to_vec2::<u8>()?
            .into_iter()
            .map(|r| r.into_iter().map(u64::from).collect())
            .collect(),
        DType::U32 => tensor
            .to_vec2::<u32>()?
            .into_iter()
            .map(|r| r.into_iter().map(u64::from).collect())
            .collect(),
        DType::I16 => tensor
            .to_vec2::<i16>()?
            .into_iter()
            .map(|r| {
                r.into_iter()
                    .map(|v| {
                        u64::try_from(v).map_err(|_| {
                            candle::Error::Msg(format!("{label} contains a negative value"))
                        })
                    })
                    .collect()
            })
            .collect::<Result<Vec<Vec<u64>>>>()?,
        DType::I32 => tensor
            .to_vec2::<i32>()?
            .into_iter()
            .map(|r| {
                r.into_iter()
                    .map(|v| {
                        u64::try_from(v).map_err(|_| {
                            candle::Error::Msg(format!("{label} contains a negative value"))
                        })
                    })
                    .collect()
            })
            .collect::<Result<Vec<Vec<u64>>>>()?,
        DType::I64 => tensor
            .to_vec2::<i64>()?
            .into_iter()
            .map(|r| {
                r.into_iter()
                    .map(|v| {
                        u64::try_from(v).map_err(|_| {
                            candle::Error::Msg(format!("{label} contains a negative value"))
                        })
                    })
                    .collect()
            })
            .collect::<Result<Vec<Vec<u64>>>>()?,
        dtype => candle::bail!("SigLIP2 {label} must use an integer dtype, got {dtype:?}"),
    };
    Ok(values)
}

fn validate_attention_mask(
    mask: &Tensor,
    shapes: &[(usize, usize)],
    max_patches: usize,
) -> Result<Tensor> {
    let mask_f32 = mask.to_dtype(DType::F32)?;
    let values = mask_f32.to_vec2::<f32>()?;
    if values.len() != shapes.len() {
        candle::bail!("SigLIP2 attention mask crop count does not match spatial_shapes")
    }
    for (crop, (row, &(rows, cols))) in values.iter().zip(shapes).enumerate() {
        let valid_patches = rows
            .checked_mul(cols)
            .ok_or_else(|| candle::Error::Msg("SigLIP2 spatial patch count overflow".into()))?;
        if valid_patches > max_patches || row.len() != max_patches {
            candle::bail!("SigLIP2 attention mask length does not match packed input")
        }
        for (index, &value) in row.iter().enumerate() {
            if !value.is_finite() || (value != 0.0 && value != 1.0) {
                candle::bail!("SigLIP2 attention mask crop {crop} contains a non-binary value")
            }
            let expected = if index < valid_patches { 1.0 } else { 0.0 };
            if value != expected {
                candle::bail!(
                    "SigLIP2 attention mask crop {crop} is not a valid prefix for spatial shape [{rows}, {cols}]"
                )
            }
        }
    }
    Ok(mask_f32)
}

#[derive(Clone, Debug)]
struct ResizeWeights {
    indices: Vec<usize>,
    weights: Vec<f32>,
}

fn resize_weights(input: usize, output: usize, index: usize) -> Result<ResizeWeights> {
    if input == 0 || output == 0 || index >= output {
        candle::bail!("invalid SigLIP2 resize dimensions")
    }
    let scale = input as f32 / output as f32;
    let support = if scale >= 1.0 { scale } else { 1.0 };
    if !scale.is_finite() || !support.is_finite() {
        candle::bail!("SigLIP2 resize scale is not finite")
    }
    let max_length = (support.ceil() as usize)
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| candle::Error::Msg("SigLIP2 resize support overflow".into()))?;
    let max_length_i64 = i64::try_from(max_length)
        .map_err(|_| candle::Error::Msg("SigLIP2 resize support is too large".into()))?;
    let input_i64 = i64::try_from(input)
        .map_err(|_| candle::Error::Msg("SigLIP2 resize input dimension is too large".into()))?;
    let center = scale * (index as f32 + 0.5);
    let inv_scale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let start = ((center - support + 0.5) as i64).max(0);
    let end = ((center + support + 0.5) as i64).min(input_i64);
    let length_i64 = end
        .checked_sub(start)
        .ok_or_else(|| candle::Error::Msg("SigLIP2 resize range overflow".into()))?
        .clamp(0, max_length_i64);
    let length = usize::try_from(length_i64)
        .map_err(|_| candle::Error::Msg("SigLIP2 resize range is too large".into()))?;
    let mut indices = Vec::with_capacity(length);
    let mut weights = Vec::with_capacity(length);
    let mut total = 0f32;
    for offset in 0..length {
        let source = start as usize + offset;
        let argument = (source as f32 - center + 0.5) * inv_scale;
        let weight = (1.0 - argument.abs()).max(0.0);
        indices.push(source);
        weights.push(weight);
        total += weight;
    }
    if !total.is_finite() || total <= 0.0 {
        candle::bail!("SigLIP2 resize weights have zero or invalid normalization")
    }
    for weight in weights.iter_mut().take(length) {
        *weight /= total;
    }
    Ok(ResizeWeights { indices, weights })
}

fn resize_bilinear_antialias(
    input: &[Vec<f32>],
    input_height: usize,
    input_width: usize,
    output_height: usize,
    output_width: usize,
    channels: usize,
) -> Result<Vec<f32>> {
    let expected = input_height
        .checked_mul(input_width)
        .ok_or_else(|| candle::Error::Msg("SigLIP2 resize input size overflow".into()))?;
    if input.len() != expected || input.iter().any(|row| row.len() != channels) {
        candle::bail!("SigLIP2 positional table has an unexpected shape")
    }
    let horizontal: Vec<ResizeWeights> = (0..output_width)
        .map(|index| resize_weights(input_width, output_width, index))
        .collect::<Result<_>>()?;
    let vertical: Vec<ResizeWeights> = (0..output_height)
        .map(|index| resize_weights(input_height, output_height, index))
        .collect::<Result<_>>()?;

    // PyTorch's antialiased implementation is separable and processes the
    // contiguous (width) dimension before the height dimension.
    let horizontal_size = input_height
        .checked_mul(output_width)
        .and_then(|value| value.checked_mul(channels))
        .ok_or_else(|| candle::Error::Msg("SigLIP2 resize output size overflow".into()))?;
    let mut horizontal_output = vec![0f32; horizontal_size];
    for row in 0..input_height {
        for col in 0..output_width {
            let weights = &horizontal[col];
            for channel in 0..channels {
                let mut value = 0f32;
                for offset in 0..weights.indices.len() {
                    let source_col = weights.indices[offset];
                    value +=
                        input[row * input_width + source_col][channel] * weights.weights[offset];
                }
                let index = (row * output_width + col) * channels + channel;
                horizontal_output[index] = value;
            }
        }
    }

    let output_size = output_height
        .checked_mul(output_width)
        .and_then(|value| value.checked_mul(channels))
        .ok_or_else(|| candle::Error::Msg("SigLIP2 resize output size overflow".into()))?;
    let mut output = vec![0f32; output_size];
    for row in 0..output_height {
        let weights = &vertical[row];
        for col in 0..output_width {
            for channel in 0..channels {
                let mut value = 0f32;
                for offset in 0..weights.indices.len() {
                    let source_row = weights.indices[offset];
                    let index = (source_row * output_width + col) * channels + channel;
                    value += horizontal_output[index] * weights.weights[offset];
                }
                output[(row * output_width + col) * channels + channel] = value;
            }
        }
    }
    Ok(output)
}
