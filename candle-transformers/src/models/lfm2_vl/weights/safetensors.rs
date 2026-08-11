pub fn inspect_safetensors(
    weights_path: impl AsRef<Path>,
    manifest: &MmprojManifest,
    dtype: DType,
    device: &Device,
) -> Result<MmprojLoadReport> {
    let weights_path = weights_path.as_ref();
    let weights_bytes = read_weight_bytes(weights_path, "split MMProj safetensors", manifest)?;
    verify_bytes_sha256(
        &weights_bytes,
        &manifest.mmproj_safetensors_sha256,
        "split MMProj safetensors",
    )?;
    inspect_safetensors_bytes(&weights_bytes, manifest, dtype, device)
}

fn inspect_safetensors_bytes(
    weights_bytes: &[u8],
    manifest: &MmprojManifest,
    dtype: DType,
    device: &Device,
) -> Result<MmprojLoadReport> {
    let actual = safetensors_inventory(weights_bytes)?;
    let actual_names: BTreeSet<_> = actual.keys().cloned().collect();
    let manifest_names: BTreeSet<_> = manifest.tensor_inventory.keys().cloned().collect();
    let missing_tensors = manifest_names.difference(&actual_names).cloned().collect();
    let unexpected_tensors = actual_names.difference(&manifest_names).cloned().collect();
    let mut shape_or_dtype_mismatches = Vec::new();
    for name in manifest_names.intersection(&actual_names) {
        let expected = &manifest.tensor_inventory[name];
        let found = &actual[name];
        if expected != found {
            shape_or_dtype_mismatches.push(format!(
                "{name}: expected {} {:?} ({} bytes), found {} {:?} ({} bytes)",
                expected.dtype,
                expected.shape,
                expected.nbytes,
                found.dtype,
                found.shape,
                found.nbytes
            ));
        }
    }
    Ok(MmprojLoadReport {
        loaded_tensors: actual_names.into_iter().collect(),
        missing_tensors,
        unexpected_tensors,
        shape_or_dtype_mismatches,
        resolved_vision_root: VISION_ROOT.to_string(),
        resolved_projector_root: PROJECTOR_ROOT.to_string(),
        target_dtype: format!("{dtype:?}"),
        target_device: format!("{device:?}"),
    })
}

fn safetensors_inventory(weights_bytes: &[u8]) -> Result<BTreeMap<String, MmprojTensorInfo>> {
    let prefix = weights_bytes.get(..8).ok_or_else(|| {
        candle::Error::Msg("split MMProj safetensors is shorter than its header prefix".into())
    })?;
    let header_len = u64::from_le_bytes(prefix.try_into().map_err(candle::Error::wrap)?);
    let header_len = usize::try_from(header_len).map_err(candle::Error::wrap)?;
    if header_len == 0 || header_len > MAX_SAFETENSORS_HEADER_BYTES {
        candle::bail!(
            "split MMProj safetensors header length {header_len} is outside the supported range"
        )
    }
    let header_end = 8usize.checked_add(header_len).ok_or_else(|| {
        candle::Error::Msg("split MMProj safetensors header length overflows".into())
    })?;
    let header_bytes = weights_bytes.get(8..header_end).ok_or_else(|| {
        candle::Error::Msg("split MMProj safetensors header exceeds the file length".into())
    })?;
    let header_value: serde_json::Value = serde_json::from_slice(header_bytes).map_err(|err| {
        candle::Error::Msg(format!("invalid split MMProj safetensors header: {err}"))
    })?;
    let mut header = match header_value {
        serde_json::Value::Object(header) => header,
        _ => {
            candle::bail!("split MMProj safetensors header must be a JSON object")
        }
    };
    if let Some(metadata) = header.remove("__metadata__") {
        let metadata = metadata.as_object().ok_or_else(|| {
            candle::Error::Msg("split MMProj safetensors metadata must be an object".into())
        })?;
        if metadata
            .iter()
            .any(|(key, value)| key.is_empty() || !value.is_string())
        {
            candle::bail!("split MMProj safetensors metadata must contain only strings")
        }
    }
    let tensor_count = header.len();
    if tensor_count == 0 || tensor_count > MAX_MMPROJ_TENSORS {
        candle::bail!(
            "split MMProj safetensors tensor count {tensor_count} is outside the supported range"
        )
    }

    let mut actual = BTreeMap::new();
    let mut ranges = Vec::new();
    ranges.try_reserve_exact(tensor_count).map_err(|_| {
        candle::Error::Msg("split MMProj safetensors range allocation failed".into())
    })?;
    let data_size = weights_bytes.len() - header_end;
    for (name, value) in header {
        let info = value.as_object().ok_or_else(|| {
            candle::Error::Msg(format!(
                "split MMProj safetensors tensor {name:?} metadata must be an object"
            ))
        })?;
        let dtype = info
            .get("dtype")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "split MMProj safetensors tensor {name:?} lacks a string dtype"
                ))
            })?
            .to_string();
        let element_size = dense_dtype_size(&dtype).ok_or_else(|| {
            candle::Error::Msg(format!(
                "split MMProj safetensors tensor {name:?} has unsupported dense dtype {dtype:?}"
            ))
        })?;
        let raw_shape = info
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .filter(|shape| !shape.is_empty())
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "split MMProj safetensors tensor {name:?} has an invalid shape"
                ))
            })?;
        let mut shape = Vec::new();
        shape.try_reserve_exact(raw_shape.len()).map_err(|_| {
            candle::Error::Msg("split MMProj safetensors shape allocation failed".into())
        })?;
        for dimension in raw_shape {
            let dimension = dimension
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|&value| value > 0)
                .ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "split MMProj safetensors tensor {name:?} has an invalid shape"
                    ))
                })?;
            shape.push(dimension);
        }
        let raw_offsets = info
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .filter(|offsets| offsets.len() == 2)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "split MMProj safetensors tensor {name:?} has invalid data offsets"
                ))
            })?;
        let offset = |index: usize| -> Result<usize> {
            raw_offsets[index]
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "split MMProj safetensors tensor {name:?} has invalid data offsets"
                    ))
                })
        };
        let start = offset(0)?;
        let end = offset(1)?;
        if start > end || end > data_size {
            candle::bail!(
                "split MMProj safetensors tensor {name:?} has out-of-bounds data offsets [{start}, {end}]"
            )
        }
        let expected_nbytes = shape
            .iter()
            .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
            .and_then(|count| count.checked_mul(element_size))
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "split MMProj safetensors tensor {name:?} byte size overflows"
                ))
            })?;
        let nbytes = end - start;
        if nbytes != expected_nbytes {
            candle::bail!(
                "split MMProj safetensors tensor {name:?} stores {nbytes} bytes, expected {expected_nbytes}"
            )
        }
        ranges.push((start, end, name.clone()));
        actual.insert(
            name,
            MmprojTensorInfo {
                dtype,
                shape,
                nbytes,
            },
        );
    }

    ranges.sort_by_key(|(start, _, _)| *start);
    let mut previous_end = 0usize;
    for (start, end, name) in ranges {
        if start != previous_end {
            let relation = if start < previous_end {
                "overlaps another tensor"
            } else {
                "leaves a payload gap"
            };
            candle::bail!("split MMProj safetensors tensor {name:?} {relation}")
        }
        previous_end = end;
    }
    if previous_end != data_size {
        candle::bail!(
            "split MMProj safetensors has {} unclaimed payload bytes",
            data_size - previous_end
        )
    }
    Ok(actual)
}

fn validate_sha256(label: &str, value: &str) -> Result<()> {
    validate_lower_hex(label, value, &[64])
}

fn validate_lower_hex(label: &str, value: &str, allowed_lengths: &[usize]) -> Result<()> {
    if !allowed_lengths.contains(&value.len())
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        candle::bail!(
            "split MMProj {label} must be lowercase hexadecimal with length in {allowed_lengths:?}"
        )
    }
    Ok(())
}

fn verify_bytes_sha256(bytes: &[u8], expected: &str, label: &str) -> Result<()> {
    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual != expected {
        candle::bail!("{label} SHA-256 mismatch: expected {expected}, found {actual}")
    }
    Ok(())
}

fn read_bounded_text(path: &Path, label: &str) -> Result<String> {
    let bytes = read_bounded_bytes(path, label)?;
    String::from_utf8(bytes)
        .map_err(|err| candle::Error::Msg(format!("{label} is not UTF-8: {err}")))
}

fn read_bounded_bytes(path: &Path, label: &str) -> Result<Vec<u8>> {
    read_file_bytes(path, label, MAX_MANIFEST_BYTES)
}

fn read_weight_bytes(path: &Path, label: &str, manifest: &MmprojManifest) -> Result<Vec<u8>> {
    let payload_bytes = manifest
        .tensor_inventory
        .values()
        .try_fold(0u64, |total, info| {
            let nbytes = u64::try_from(info.nbytes).ok()?;
            total.checked_add(nbytes)
        })
        .ok_or_else(|| candle::Error::Msg("split MMProj payload size overflows".into()))?;
    let maximum_file_bytes = payload_bytes
        .checked_add(MAX_SAFETENSORS_HEADER_BYTES as u64)
        .and_then(|size| size.checked_add(8))
        .ok_or_else(|| candle::Error::Msg("split MMProj file size limit overflows".into()))?;
    read_file_bytes(path, label, maximum_file_bytes)
}

fn read_file_bytes(path: &Path, label: &str, max_bytes: u64) -> Result<Vec<u8>> {
    let mut file = File::open(path).map_err(|err| {
        candle::Error::Msg(format!("cannot open {label} at {}: {err}", path.display()))
    })?;
    let size = file
        .metadata()
        .map_err(|err| {
            candle::Error::Msg(format!(
                "cannot inspect {label} at {}: {err}",
                path.display()
            ))
        })?
        .len();
    if size == 0 {
        candle::bail!("{label} at {} is empty", path.display())
    }
    if size > max_bytes {
        candle::bail!("{label} size {size} is outside the supported range")
    }
    let size = usize::try_from(size).map_err(|_| {
        candle::Error::Msg(format!(
            "{label} at {} is too large for this platform",
            path.display()
        ))
    })?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(size).map_err(|_| {
        candle::Error::Msg(format!(
            "cannot allocate {size} bytes for {label} at {}",
            path.display()
        ))
    })?;
    bytes.resize(size, 0);
    file.read_exact(&mut bytes).map_err(|err| {
        candle::Error::Msg(format!("cannot read {label} at {}: {err}", path.display()))
    })?;
    let mut trailing = [0u8; 1];
    if file.read(&mut trailing).map_err(|err| {
        candle::Error::Msg(format!(
            "cannot finish reading {label} at {}: {err}",
            path.display()
        ))
    })? != 0
    {
        candle::bail!("{label} at {} changed while it was read", path.display())
    }
    Ok(bytes)
}

fn processor_pair_fields(processor: &serde_json::Value) -> Result<(usize, usize)> {
    let values = processor
        .get("image_processor")
        .unwrap_or(processor)
        .as_object()
        .ok_or_else(|| {
            candle::Error::Msg("split MMProj processor config must be a JSON object".into())
        })?;
    let positive_usize = |name: &str, aliases: &[&str]| -> Result<usize> {
        let value = aliases
            .iter()
            .find_map(|alias| values.get(*alias))
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&value| value > 0)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "split MMProj processor config lacks a positive {name}"
                ))
            })?;
        Ok(value)
    };
    Ok((
        positive_usize("encoder patch size", &["encoder_patch_size", "patch_size"])?,
        positive_usize("downsample factor", &["downsample_factor"])?,
    ))
}

fn dense_dtype_size(dtype: &str) -> Option<usize> {
    match dtype {
        "BF16" | "F16" => Some(2),
        "F32" => Some(4),
        "F64" => Some(8),
        _ => None,
    }
}

fn expected_tensor_shapes(config: &Lfm2VlConfig) -> Result<BTreeMap<String, Vec<usize>>> {
    let vision = &config.vision_config;
    if vision.num_hidden_layers > MAX_VISION_LAYERS {
        candle::bail!("split MMProj vision layer count exceeds {MAX_VISION_LAYERS}")
    }
    let patch_dimension = vision.patch_dimension_for_vl()?;
    let mut shapes = BTreeMap::new();
    let mut insert = |name: String, shape: Vec<usize>| {
        shapes.insert(name, shape);
    };
    insert(
        format!("{VISION_ROOT}.embeddings.patch_embedding.weight"),
        vec![vision.hidden_size, patch_dimension],
    );
    insert(
        format!("{VISION_ROOT}.embeddings.patch_embedding.bias"),
        vec![vision.hidden_size],
    );
    insert(
        format!("{VISION_ROOT}.embeddings.position_embedding.weight"),
        vec![vision.num_patches, vision.hidden_size],
    );
    for layer in 0..vision.num_hidden_layers {
        let root = format!("{VISION_ROOT}.encoder.layers.{layer}");
        for norm in ["layer_norm1", "layer_norm2"] {
            insert(format!("{root}.{norm}.weight"), vec![vision.hidden_size]);
            insert(format!("{root}.{norm}.bias"), vec![vision.hidden_size]);
        }
        for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            insert(
                format!("{root}.self_attn.{projection}.weight"),
                vec![vision.hidden_size, vision.hidden_size],
            );
            insert(
                format!("{root}.self_attn.{projection}.bias"),
                vec![vision.hidden_size],
            );
        }
        insert(
            format!("{root}.mlp.fc1.weight"),
            vec![vision.intermediate_size, vision.hidden_size],
        );
        insert(
            format!("{root}.mlp.fc1.bias"),
            vec![vision.intermediate_size],
        );
        insert(
            format!("{root}.mlp.fc2.weight"),
            vec![vision.hidden_size, vision.intermediate_size],
        );
        insert(format!("{root}.mlp.fc2.bias"), vec![vision.hidden_size]);
    }
    insert(
        format!("{VISION_ROOT}.post_layernorm.weight"),
        vec![vision.hidden_size],
    );
    insert(
        format!("{VISION_ROOT}.post_layernorm.bias"),
        vec![vision.hidden_size],
    );

    let projector_input = config.projector_input_size()?;
    if config.projector_use_layernorm {
        insert(
            format!("{PROJECTOR_ROOT}.layer_norm.weight"),
            vec![projector_input],
        );
        insert(
            format!("{PROJECTOR_ROOT}.layer_norm.bias"),
            vec![projector_input],
        );
    }
    insert(
        format!("{PROJECTOR_ROOT}.linear_1.weight"),
        vec![config.projector_hidden_size, projector_input],
    );
    insert(
        format!("{PROJECTOR_ROOT}.linear_2.weight"),
        vec![config.text_config.hidden_size, config.projector_hidden_size],
    );
    if config.projector_bias {
        insert(
            format!("{PROJECTOR_ROOT}.linear_1.bias"),
            vec![config.projector_hidden_size],
        );
        insert(
            format!("{PROJECTOR_ROOT}.linear_2.bias"),
            vec![config.text_config.hidden_size],
        );
    }
    Ok(shapes)
}
