fn try_vec_with_capacity<T>(capacity: usize, label: &str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(capacity).map_err(|err| {
        candle::Error::Msg(format!(
            "failed to allocate {label} ({capacity} elements): {err}"
        ))
    })?;
    Ok(values)
}

fn dynamic_image_dimensions(image: &DynamicImage) -> Result<(usize, usize)> {
    let width = usize::try_from(image.width())
        .map_err(|_| candle::Error::Msg("LFM2-VL image width does not fit usize".into()))?;
    let height = usize::try_from(image.height())
        .map_err(|_| candle::Error::Msg("LFM2-VL image height does not fit usize".into()))?;
    if width == 0 || height == 0 {
        candle::bail!("LFM2-VL image dimensions must be positive")
    }
    Ok((width, height))
}

fn try_filled_vec<T: Clone>(length: usize, value: T, label: &str) -> Result<Vec<T>> {
    let mut values = try_vec_with_capacity(length, label)?;
    values.resize(length, value);
    Ok(values)
}

fn floor_multiple(value: f64, factor: usize) -> Result<usize> {
    if factor == 0 || !value.is_finite() || value < 0.0 {
        candle::bail!("LFM2-VL resize calculation is not finite")
    }
    let quotient = (value / factor as f64).floor();
    let quotient = usize::try_from(quotient as u128)
        .map_err(|_| candle::Error::Msg("LFM2-VL resize dimension is too large".into()))?;
    quotient
        .checked_mul(factor)
        .ok_or_else(|| candle::Error::Msg("LFM2-VL resize dimension overflow".into()))
}

fn ceil_multiple(value: f64, factor: usize) -> Result<usize> {
    if factor == 0 || !value.is_finite() || value < 0.0 {
        candle::bail!("LFM2-VL resize calculation is not finite")
    }
    let quotient = (value / factor as f64).ceil();
    let quotient = usize::try_from(quotient as u128)
        .map_err(|_| candle::Error::Msg("LFM2-VL resize dimension is too large".into()))?;
    quotient
        .checked_mul(factor)
        .ok_or_else(|| candle::Error::Msg("LFM2-VL resize dimension overflow".into()))
}
