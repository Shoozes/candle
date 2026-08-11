fn processor_metadata_json(metadata: &ParsedMetadata) -> serde_json::Value {
    let mut image_processor = serde_json::Map::new();
    image_processor.insert(
        "encoder_patch_size".into(),
        serde_json::Value::from(metadata.patch_size),
    );
    image_processor.insert(
        "downsample_factor".into(),
        serde_json::Value::from(metadata.downsample_factor),
    );
    image_processor.insert("image_mean".into(), serde_json::json!(metadata.image_mean));
    image_processor.insert("image_std".into(), serde_json::json!(metadata.image_std));
    if let Some((min_tiles, max_tiles, image_size)) = metadata.preproc {
        image_processor.insert("min_tiles".into(), serde_json::Value::from(min_tiles));
        image_processor.insert("max_tiles".into(), serde_json::Value::from(max_tiles));
        image_processor.insert("tile_size".into(), serde_json::Value::from(image_size));
    }
    serde_json::json!({ "image_processor": image_processor })
}

fn metadata_value<'a>(content: &'a gguf_file::Content, key: &str) -> Result<&'a Value> {
    content
        .metadata
        .get(key)
        .ok_or_else(|| candle::Error::Msg(format!("GGUF MMProj is missing metadata key {key:?}")))
}

fn required_string<'a>(content: &'a gguf_file::Content, key: &str) -> Result<&'a str> {
    match metadata_value(content, key)? {
        Value::String(value) => Ok(value),
        value => candle::bail!(
            "GGUF MMProj metadata {key:?} must be a string, got {:?}",
            value.value_type()
        ),
    }
}

fn optional_string<'a>(content: &'a gguf_file::Content, key: &str) -> Result<Option<&'a str>> {
    match content.metadata.get(key) {
        None => Ok(None),
        Some(Value::String(value)) => Ok(Some(value)),
        Some(value) => candle::bail!(
            "GGUF MMProj metadata {key:?} must be a string, got {:?}",
            value.value_type()
        ),
    }
}

fn required_bool(content: &gguf_file::Content, key: &str) -> Result<bool> {
    match metadata_value(content, key)? {
        Value::Bool(value) => Ok(*value),
        value => candle::bail!(
            "GGUF MMProj metadata {key:?} must be a boolean, got {:?}",
            value.value_type()
        ),
    }
}

fn positive_usize_value(value: &Value, key: &str) -> Result<usize> {
    let raw = match value {
        Value::U8(value) => *value as u64,
        Value::U16(value) => *value as u64,
        Value::U32(value) => *value as u64,
        Value::U64(value) => *value,
        Value::I8(value) if *value > 0 => *value as u64,
        Value::I16(value) if *value > 0 => *value as u64,
        Value::I32(value) if *value > 0 => *value as u64,
        Value::I64(value) if *value > 0 => *value as u64,
        _ => candle::bail!("GGUF MMProj metadata {key:?} must be a positive integer"),
    };
    if raw == 0 {
        candle::bail!("GGUF MMProj metadata {key:?} must be a positive integer")
    }
    usize::try_from(raw).map_err(|_| {
        candle::Error::Msg(format!(
            "GGUF MMProj metadata {key:?} does not fit this platform"
        ))
    })
}

fn required_positive_usize(content: &gguf_file::Content, key: &str) -> Result<usize> {
    positive_usize_value(metadata_value(content, key)?, key)
}

fn optional_positive_usize(content: &gguf_file::Content, key: &str) -> Result<Option<usize>> {
    content
        .metadata
        .get(key)
        .map(|value| positive_usize_value(value, key))
        .transpose()
}

fn required_positive_float(content: &gguf_file::Content, key: &str) -> Result<f64> {
    let value = match metadata_value(content, key)? {
        Value::F32(value) => *value as f64,
        Value::F64(value) => *value,
        value => candle::bail!(
            "GGUF MMProj metadata {key:?} must be an f32 or f64, got {:?}",
            value.value_type()
        ),
    };
    if !value.is_finite() || value <= 0.0 {
        candle::bail!("GGUF MMProj metadata {key:?} must be finite and positive")
    }
    Ok(value)
}

fn required_f32_triplet(content: &gguf_file::Content, key: &str) -> Result<[f32; 3]> {
    let values = match metadata_value(content, key)? {
        Value::Array(values) if values.len() == 3 => values,
        Value::Array(values) => candle::bail!(
            "GGUF MMProj metadata {key:?} must contain 3 values, got {}",
            values.len()
        ),
        value => candle::bail!(
            "GGUF MMProj metadata {key:?} must be an array, got {:?}",
            value.value_type()
        ),
    };
    let mut output = [0f32; 3];
    for (index, value) in values.iter().enumerate() {
        let value = match value {
            Value::F32(value) => *value,
            Value::F64(value) => *value as f32,
            value => candle::bail!(
                "GGUF MMProj metadata {key:?}[{index}] must be an f32 or f64, got {:?}",
                value.value_type()
            ),
        };
        if !value.is_finite() {
            candle::bail!("GGUF MMProj metadata {key:?}[{index}] must be finite")
        }
        output[index] = value;
    }
    Ok(output)
}

fn required_tensor_shape(content: &gguf_file::Content, name: &str) -> Result<Vec<usize>> {
    content
        .tensor_infos
        .get(name)
        .map(|info| info.shape.dims().to_vec())
        .ok_or_else(|| candle::Error::Msg(format!("GGUF MMProj is missing tensor {name:?}")))
}

fn checked_element_count(shape: &[usize], name: &str) -> Result<u64> {
    shape.iter().try_fold(1u64, |count, &dimension| {
        let dimension = u64::try_from(dimension).map_err(candle::Error::wrap)?;
        if dimension == 0 {
            candle::bail!("GGUF MMProj tensor {name:?} has a zero dimension")
        }
        count.checked_mul(dimension).ok_or_else(|| {
            candle::Error::Msg(format!(
                "GGUF MMProj tensor {name:?} element count overflowed"
            ))
        })
    })
}
