fn crop_kind_matches(actual: &CropKind, expected: &CropKind) -> bool {
    match (actual, expected) {
        (CropKind::Whole, CropKind::Whole) | (CropKind::Thumbnail, CropKind::Thumbnail) => true,
        (
            CropKind::Tile {
                row: actual_row,
                col: actual_col,
            },
            CropKind::Tile {
                row: expected_row,
                col: expected_col,
            },
        ) => actual_row == expected_row && actual_col == expected_col,
        _ => false,
    }
}

fn tile_token_name(row: usize, col: usize) -> String {
    format!("<|img_row_{row}_col_{col}|>")
}

fn append_repeated(target: &mut String, token: &str, count: usize) -> Result<()> {
    let bytes = token
        .len()
        .checked_mul(count)
        .ok_or_else(|| candle::Error::Msg("LFM2-VL placeholder string size overflow".into()))?;
    target
        .try_reserve(bytes)
        .map_err(|_| candle::Error::Msg("LFM2-VL placeholder string allocation failed".into()))?;
    for _ in 0..count {
        target.push_str(token);
    }
    Ok(())
}

fn try_string_with_capacity(capacity: usize, label: &str) -> Result<String> {
    let mut value = String::new();
    value.try_reserve_exact(capacity).map_err(|err| {
        candle::Error::Msg(format!(
            "failed to allocate {label} ({capacity} bytes): {err}"
        ))
    })?;
    Ok(value)
}

fn try_push_str(target: &mut String, value: &str, label: &str) -> Result<()> {
    target.try_reserve(value.len()).map_err(|err| {
        candle::Error::Msg(format!(
            "failed to grow {label} by {} bytes: {err}",
            value.len()
        ))
    })?;
    target.push_str(value);
    Ok(())
}

fn try_vec_with_capacity<T>(capacity: usize, label: &str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(capacity).map_err(|err| {
        candle::Error::Msg(format!(
            "failed to allocate {label} ({capacity} elements): {err}"
        ))
    })?;
    Ok(values)
}

fn resolve_atomic_token(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    let id = tokenizer.token_to_id(token).ok_or_else(|| {
        candle::Error::Msg(format!("tokenizer is missing required token {token:?}"))
    })?;
    let encoding = tokenizer
        .encode(token, false)
        .map_err(|err| candle::Error::Msg(format!("failed to encode token {token:?}: {err}")))?;
    if encoding.get_ids() != [id] {
        candle::bail!("tokenizer does not encode {token:?} as one atomic token")
    }
    Ok(id)
}

fn resolve_atomic_token_if_present(tokenizer: &Tokenizer, token: &str) -> Result<Option<u32>> {
    match tokenizer.token_to_id(token) {
        Some(id) => {
            let resolved = resolve_atomic_token(tokenizer, token)?;
            if resolved != id {
                candle::bail!("tokenizer token resolution changed for {token:?}")
            }
            Ok(Some(id))
        }
        None => Ok(None),
    }
}

fn find_crop_spans(
    input_ids: &[u32],
    image_token_id: u32,
    expected_lengths: &[usize],
) -> Result<Vec<ImageTokenSpan>> {
    if expected_lengths.is_empty() {
        candle::bail!("LFM2-VL prompt contains no expected crop spans")
    }
    let mut spans = try_vec_with_capacity(expected_lengths.len(), "LFM2-VL crop span table")?;
    let mut cursor = 0usize;
    for &length in expected_lengths {
        if length == 0 {
            candle::bail!("LFM2-VL crop span length must be positive")
        }
        while cursor < input_ids.len() && input_ids[cursor] != image_token_id {
            cursor += 1;
        }
        let end = cursor
            .checked_add(length)
            .ok_or_else(|| candle::Error::Msg("LFM2-VL image span overflow".into()))?;
        if end > input_ids.len() {
            candle::bail!("LFM2-VL prompt has fewer image placeholders than expected")
        }
        if input_ids[cursor..end]
            .iter()
            .any(|&token_id| token_id != image_token_id)
        {
            candle::bail!("LFM2-VL image placeholders are not contiguous")
        }
        spans.push(ImageTokenSpan::new(0, cursor, end));
        cursor = end;
    }
    if input_ids[cursor..].contains(&image_token_id) {
        candle::bail!("LFM2-VL prompt contains unexpected image placeholders")
    }
    Ok(spans)
}
