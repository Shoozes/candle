fn expected_tensors(config: &Lfm2VlMmprojConfig) -> Result<BTreeMap<String, ExpectedTensor>> {
    let vision = &config.vision_config;
    let projector_input = config.projector_input_size()?;
    let mut expected = BTreeMap::new();
    let mut insert = |gguf: String, native: String, shape: Vec<usize>, patch_layout: bool| {
        let quantized_linear = is_quantized_linear_name(&gguf);
        expected.insert(
            gguf,
            ExpectedTensor {
                native_name: native,
                shape,
                patch_layout,
                quantized_linear,
            },
        );
    };
    insert(
        "v.patch_embd.weight".into(),
        format!("{NATIVE_VISION_ROOT}.embeddings.patch_embedding.weight"),
        vec![
            vision.hidden_size,
            vision.num_channels,
            vision.patch_size,
            vision.patch_size,
        ],
        true,
    );
    insert(
        "v.patch_embd.bias".into(),
        format!("{NATIVE_VISION_ROOT}.embeddings.patch_embedding.bias"),
        vec![vision.hidden_size],
        false,
    );
    insert(
        "v.position_embd.weight".into(),
        format!("{NATIVE_VISION_ROOT}.embeddings.position_embedding.weight"),
        vec![vision.num_patches, vision.hidden_size],
        false,
    );
    for layer in 0..vision.num_hidden_layers {
        let gguf_root = format!("v.blk.{layer}");
        let native_root = format!("{NATIVE_VISION_ROOT}.encoder.layers.{layer}");
        for (gguf_norm, native_norm) in [("ln1", "layer_norm1"), ("ln2", "layer_norm2")] {
            for suffix in ["weight", "bias"] {
                insert(
                    format!("{gguf_root}.{gguf_norm}.{suffix}"),
                    format!("{native_root}.{native_norm}.{suffix}"),
                    vec![vision.hidden_size],
                    false,
                );
            }
        }
        for (gguf_projection, native_projection) in [
            ("attn_q", "q_proj"),
            ("attn_k", "k_proj"),
            ("attn_v", "v_proj"),
            ("attn_out", "out_proj"),
        ] {
            insert(
                format!("{gguf_root}.{gguf_projection}.weight"),
                format!("{native_root}.self_attn.{native_projection}.weight"),
                vec![vision.hidden_size, vision.hidden_size],
                false,
            );
            insert(
                format!("{gguf_root}.{gguf_projection}.bias"),
                format!("{native_root}.self_attn.{native_projection}.bias"),
                vec![vision.hidden_size],
                false,
            );
        }
        insert(
            format!("{gguf_root}.ffn_up.weight"),
            format!("{native_root}.mlp.fc1.weight"),
            vec![vision.intermediate_size, vision.hidden_size],
            false,
        );
        insert(
            format!("{gguf_root}.ffn_up.bias"),
            format!("{native_root}.mlp.fc1.bias"),
            vec![vision.intermediate_size],
            false,
        );
        insert(
            format!("{gguf_root}.ffn_down.weight"),
            format!("{native_root}.mlp.fc2.weight"),
            vec![vision.hidden_size, vision.intermediate_size],
            false,
        );
        insert(
            format!("{gguf_root}.ffn_down.bias"),
            format!("{native_root}.mlp.fc2.bias"),
            vec![vision.hidden_size],
            false,
        );
    }
    for suffix in ["weight", "bias"] {
        insert(
            format!("v.post_ln.{suffix}"),
            format!("{NATIVE_VISION_ROOT}.post_layernorm.{suffix}"),
            vec![vision.hidden_size],
            false,
        );
    }
    if config.projector_use_layernorm {
        for suffix in ["weight", "bias"] {
            insert(
                format!("mm.input_norm.{suffix}"),
                format!("{NATIVE_PROJECTOR_ROOT}.layer_norm.{suffix}"),
                vec![projector_input],
                false,
            );
        }
    }
    insert(
        "mm.1.weight".into(),
        format!("{NATIVE_PROJECTOR_ROOT}.linear_1.weight"),
        vec![config.projector_hidden_size, projector_input],
        false,
    );
    insert(
        "mm.2.weight".into(),
        format!("{NATIVE_PROJECTOR_ROOT}.linear_2.weight"),
        vec![config.text_hidden_size, config.projector_hidden_size],
        false,
    );
    if config.projector_bias {
        insert(
            "mm.1.bias".into(),
            format!("{NATIVE_PROJECTOR_ROOT}.linear_1.bias"),
            vec![config.projector_hidden_size],
            false,
        );
        insert(
            "mm.2.bias".into(),
            format!("{NATIVE_PROJECTOR_ROOT}.linear_2.bias"),
            vec![config.text_hidden_size],
            false,
        );
    }
    Ok(expected)
}

fn is_quantized_linear_name(name: &str) -> bool {
    name == "mm.1.weight"
        || name == "mm.2.weight"
        || (name.contains(".attn_") && name.ends_with(".weight"))
        || (name.contains(".ffn_") && name.ends_with(".weight"))
}

fn inspect_inventory(
    content: &gguf_file::Content,
    expected: &BTreeMap<String, ExpectedTensor>,
    dtype: DType,
    device: &Device,
) -> Result<MmprojLoadReport> {
    let actual_names: BTreeSet<_> = content.tensor_infos.keys().cloned().collect();
    let expected_names: BTreeSet<_> = expected.keys().cloned().collect();
    let missing_tensors = expected_names.difference(&actual_names).cloned().collect();
    let unexpected_tensors = actual_names.difference(&expected_names).cloned().collect();
    let mut shape_or_dtype_mismatches = Vec::new();
    for name in expected_names.intersection(&actual_names) {
        let info = &content.tensor_infos[name];
        let found = info.shape.dims();
        let wanted = &expected[name].shape;
        if found != wanted {
            shape_or_dtype_mismatches.push(format!(
                "{name}: expected {wanted:?}, found {:?} ({:?})",
                found, info.ggml_dtype
            ));
        }
    }
    Ok(MmprojLoadReport {
        loaded_tensors: actual_names.into_iter().collect(),
        missing_tensors,
        unexpected_tensors,
        shape_or_dtype_mismatches,
        resolved_vision_root: NATIVE_VISION_ROOT.to_string(),
        resolved_projector_root: NATIVE_PROJECTOR_ROOT.to_string(),
        target_dtype: format!("{dtype:?}"),
        target_device: format!("{device:?}"),
    })
}

fn validate_ranges_and_sizes(
    content: &gguf_file::Content,
    expected: &BTreeMap<String, ExpectedTensor>,
    file_size: u64,
    dense_element_size: u64,
) -> Result<AllocationReport> {
    let alignment = optional_positive_usize(content, "general.alignment")?.unwrap_or(32) as u64;
    if !alignment.is_power_of_two() {
        candle::bail!("GGUF MMProj alignment {alignment} must be a power of two")
    }
    let mut ranges = Vec::new();
    ranges
        .try_reserve(expected.len())
        .map_err(|_| candle::Error::Msg("GGUF MMProj range allocation failed".into()))?;
    let mut dense_bytes = 0u64;
    let mut source_bytes_total = 0u64;
    let mut largest_transient_bytes = 0u64;
    for name in expected.keys() {
        let info = &content.tensor_infos[name];
        let element_count = checked_element_count(info.shape.dims(), name)?;
        let block_size = info.ggml_dtype.block_size() as u64;
        if element_count % block_size != 0 {
            candle::bail!(
                "GGUF MMProj tensor {name:?} element count {element_count} is not divisible by {:?} block size {block_size}",
                info.ggml_dtype
            )
        }
        let source_bytes = element_count
            .checked_div(block_size)
            .and_then(|blocks| blocks.checked_mul(info.ggml_dtype.type_size() as u64))
            .ok_or_else(|| {
                candle::Error::Msg(format!("GGUF MMProj tensor {name:?} byte size overflowed"))
            })?;
        source_bytes_total = source_bytes_total
            .checked_add(source_bytes)
            .ok_or_else(|| candle::Error::Msg("GGUF MMProj source byte total overflowed".into()))?;
        let target_bytes = element_count
            .checked_mul(dense_element_size)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "GGUF MMProj tensor {name:?} dense byte size overflowed"
                ))
            })?;
        dense_bytes = dense_bytes
            .checked_add(target_bytes)
            .ok_or_else(|| candle::Error::Msg("GGUF MMProj dense byte total overflowed".into()))?;
        if dense_bytes > MAX_DENSE_MMPROJ_BYTES {
            candle::bail!(
                "GGUF MMProj dense allocation {dense_bytes} exceeds {MAX_DENSE_MMPROJ_BYTES} bytes"
            )
        }
        // Loading can briefly hold both the input byte buffer and quantized
        // storage. F16/BF16 targets also coexist with the F32 dequantization
        // result, and layout conversion can retain one target-dtype scratch
        // tensor. Bound that conservative estimate alongside the retained map.
        let f32_dequant_scratch = if dense_element_size < 4 {
            element_count.checked_mul(4).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "GGUF MMProj tensor {name:?} F32 scratch size overflowed"
                ))
            })?
        } else {
            0
        };
        let transient_bytes = source_bytes
            .checked_mul(2)
            .and_then(|value| value.checked_add(f32_dequant_scratch))
            .and_then(|value| value.checked_add(target_bytes))
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "GGUF MMProj tensor {name:?} transient byte estimate overflowed"
                ))
            })?;
        largest_transient_bytes = largest_transient_bytes.max(transient_bytes);
        if info.offset % alignment != 0 {
            candle::bail!(
                "GGUF MMProj tensor {name:?} relative offset {} is not aligned to {alignment}",
                info.offset
            )
        }
        let start = content
            .tensor_data_offset
            .checked_add(info.offset)
            .ok_or_else(|| candle::Error::Msg("GGUF MMProj tensor offset overflowed".into()))?;
        let end = start.checked_add(source_bytes).ok_or_else(|| {
            candle::Error::Msg(format!("GGUF MMProj tensor {name:?} range overflowed"))
        })?;
        if end > file_size {
            candle::bail!(
                "GGUF MMProj tensor {name:?} ends at byte {end}, beyond file size {file_size}"
            )
        }
        ranges.push((start, end, name));
    }
    ranges.sort_by_key(|(start, _, _)| *start);
    let mut previous_end = content.tensor_data_offset;
    for (start, end, name) in ranges {
        if start < previous_end {
            candle::bail!("GGUF MMProj tensor {name:?} overlaps another tensor")
        }
        previous_end = end;
    }
    let estimated_peak_byte_count = dense_bytes
        .checked_add(largest_transient_bytes)
        .ok_or_else(|| candle::Error::Msg("GGUF MMProj peak byte estimate overflowed".into()))?;
    if estimated_peak_byte_count > MAX_ESTIMATED_MMPROJ_PEAK_BYTES {
        candle::bail!(
            "GGUF MMProj estimated peak allocation {estimated_peak_byte_count} exceeds {MAX_ESTIMATED_MMPROJ_PEAK_BYTES} bytes"
        )
    }
    Ok(AllocationReport {
        source_byte_count: source_bytes_total,
        dense_byte_count: dense_bytes,
        estimated_peak_byte_count,
    })
}
