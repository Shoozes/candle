fn inspect_native_checkpoint(
    checkpoint: &ResolvedCheckpoint,
    config: &Lfm2VlConfig,
    options: NativeLoadOptions<'_>,
) -> Result<NativeLoadReport> {
    config.validate()?;
    if config.text_config.num_hidden_layers > MAX_MODEL_LAYERS
        || config.vision_config.num_hidden_layers > MAX_MODEL_LAYERS
    {
        bail!(
            "native LFM2-VL layer counts text={} vision={} exceed {MAX_MODEL_LAYERS}",
            config.text_config.num_hidden_layers,
            config.vision_config.num_hidden_layers
        )
    }
    let vision_root = resolve_vision_root(&checkpoint.tensors)?;
    let expected = expected_tensor_shapes(config, vision_root)?;
    if expected.len() > MAX_EXPECTED_TENSORS {
        bail!(
            "native LFM2-VL expected tensor count {} exceeds {MAX_EXPECTED_TENSORS}",
            expected.len()
        )
    }
    let actual_names: BTreeSet<_> = checkpoint.tensors.keys().cloned().collect();
    let expected_names: BTreeSet<_> = expected.keys().cloned().collect();
    let missing_tensors = expected_names.difference(&actual_names).cloned().collect();
    let unexpected_tensors = actual_names.difference(&expected_names).cloned().collect();
    let loaded_tensors = expected_names
        .intersection(&actual_names)
        .cloned()
        .collect();
    let mut shape_or_dtype_mismatches = Vec::new();
    for name in expected_names.intersection(&actual_names) {
        let expected_shape = expected
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("expected tensor {name:?} disappeared"))?;
        let actual = checkpoint
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("native tensor {name:?} disappeared"))?;
        if &actual.shape != expected_shape {
            shape_or_dtype_mismatches.push(format!(
                "{name}: expected shape {expected_shape:?}, found {} {:?} in {}",
                actual.dtype, actual.shape, actual.shard
            ));
        }
    }
    let text_config = config.text_model_config()?;
    let tied_output_resolution = if text_config.tie_embedding {
        format!("tied:{LANGUAGE_ROOT}.embed_tokens.weight")
    } else {
        format!("explicit:{LM_HEAD_ROOT}.weight")
    };
    Ok(NativeLoadReport {
        loaded_tensors,
        missing_tensors,
        unexpected_tensors,
        shape_or_dtype_mismatches,
        resolved_vision_root: vision_root.to_owned(),
        resolved_projector_root: PROJECTOR_ROOT.to_owned(),
        resolved_language_root: LANGUAGE_ROOT.to_owned(),
        tied_output_resolution,
        shard_count: checkpoint.weight_files.len(),
        indexed: checkpoint.indexed,
        total_file_bytes: checkpoint.total_file_bytes,
        vision_dtype: format!("{:?}", options.vision_dtype),
        text_dtype: format!("{:?}", options.text_dtype),
        vision_device: format!("{:?}", options.vision_device),
        text_device: format!("{:?}", options.text_device),
    })
}

fn resolve_vision_root(
    tensors: &BTreeMap<String, crate::native_checkpoint::TensorInfo>,
) -> Result<&'static str> {
    let canonical_anchor = format!("{CANONICAL_VISION_ROOT}.embeddings.patch_embedding.weight");
    let direct_anchor = format!("{DIRECT_VISION_ROOT}.embeddings.patch_embedding.weight");
    match (
        tensors.contains_key(&canonical_anchor),
        tensors.contains_key(&direct_anchor),
    ) {
        (true, false) => Ok(CANONICAL_VISION_ROOT),
        (false, true) => Ok(DIRECT_VISION_ROOT),
        (true, true) => bail!("native checkpoint contains both supported vision tensor roots"),
        (false, false) => Ok(CANONICAL_VISION_ROOT),
    }
}

fn expected_tensor_shapes(
    config: &Lfm2VlConfig,
    vision_root: &str,
) -> Result<BTreeMap<String, Vec<usize>>> {
    let mut shapes = BTreeMap::new();
    let vision = &config.vision_config;
    let patch_dimension = vision
        .num_channels
        .checked_mul(vision.patch_size)
        .and_then(|value| value.checked_mul(vision.patch_size))
        .ok_or_else(|| anyhow::anyhow!("native SigLIP2 patch dimension overflow"))?;
    add_shape(
        &mut shapes,
        format!("{vision_root}.embeddings.patch_embedding.weight"),
        vec![vision.hidden_size, patch_dimension],
    )?;
    add_shape(
        &mut shapes,
        format!("{vision_root}.embeddings.patch_embedding.bias"),
        vec![vision.hidden_size],
    )?;
    add_shape(
        &mut shapes,
        format!("{vision_root}.embeddings.position_embedding.weight"),
        vec![vision.num_patches, vision.hidden_size],
    )?;
    for layer in 0..vision.num_hidden_layers {
        let root = format!("{vision_root}.encoder.layers.{layer}");
        for norm in ["layer_norm1", "layer_norm2"] {
            add_shape(
                &mut shapes,
                format!("{root}.{norm}.weight"),
                vec![vision.hidden_size],
            )?;
            add_shape(
                &mut shapes,
                format!("{root}.{norm}.bias"),
                vec![vision.hidden_size],
            )?;
        }
        for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            add_shape(
                &mut shapes,
                format!("{root}.self_attn.{projection}.weight"),
                vec![vision.hidden_size, vision.hidden_size],
            )?;
            add_shape(
                &mut shapes,
                format!("{root}.self_attn.{projection}.bias"),
                vec![vision.hidden_size],
            )?;
        }
        add_shape(
            &mut shapes,
            format!("{root}.mlp.fc1.weight"),
            vec![vision.intermediate_size, vision.hidden_size],
        )?;
        add_shape(
            &mut shapes,
            format!("{root}.mlp.fc1.bias"),
            vec![vision.intermediate_size],
        )?;
        add_shape(
            &mut shapes,
            format!("{root}.mlp.fc2.weight"),
            vec![vision.hidden_size, vision.intermediate_size],
        )?;
        add_shape(
            &mut shapes,
            format!("{root}.mlp.fc2.bias"),
            vec![vision.hidden_size],
        )?;
    }
    for parameter in ["weight", "bias"] {
        add_shape(
            &mut shapes,
            format!("{vision_root}.post_layernorm.{parameter}"),
            vec![vision.hidden_size],
        )?;
    }

    let projector_input = config.projector_input_size()?;
    if config.projector_use_layernorm {
        for parameter in ["weight", "bias"] {
            add_shape(
                &mut shapes,
                format!("{PROJECTOR_ROOT}.layer_norm.{parameter}"),
                vec![projector_input],
            )?;
        }
    }
    add_shape(
        &mut shapes,
        format!("{PROJECTOR_ROOT}.linear_1.weight"),
        vec![config.projector_hidden_size, projector_input],
    )?;
    add_shape(
        &mut shapes,
        format!("{PROJECTOR_ROOT}.linear_2.weight"),
        vec![config.text_config.hidden_size, config.projector_hidden_size],
    )?;
    if config.projector_bias {
        add_shape(
            &mut shapes,
            format!("{PROJECTOR_ROOT}.linear_1.bias"),
            vec![config.projector_hidden_size],
        )?;
        add_shape(
            &mut shapes,
            format!("{PROJECTOR_ROOT}.linear_2.bias"),
            vec![config.text_config.hidden_size],
        )?;
    }

    let text = config.text_model_config()?;
    add_shape(
        &mut shapes,
        format!("{LANGUAGE_ROOT}.embed_tokens.weight"),
        vec![text.vocab_size, text.hidden_size],
    )?;
    let head_dim = text.head_dim();
    let kv_width = text
        .num_key_value_heads
        .checked_mul(head_dim)
        .ok_or_else(|| anyhow::anyhow!("native LFM2 key/value width overflow"))?;
    let conv_projection = text
        .hidden_size
        .checked_mul(3)
        .ok_or_else(|| anyhow::anyhow!("native LFM2 convolution projection overflow"))?;
    for (layer, layer_type) in text.layer_types.iter().copied().enumerate() {
        let root = format!("{LANGUAGE_ROOT}.layers.{layer}");
        for norm in ["operator_norm", "ffn_norm"] {
            add_shape(
                &mut shapes,
                format!("{root}.{norm}.weight"),
                vec![text.hidden_size],
            )?;
        }
        add_shape(
            &mut shapes,
            format!("{root}.feed_forward.w1.weight"),
            vec![text.intermediate_size, text.hidden_size],
        )?;
        add_shape(
            &mut shapes,
            format!("{root}.feed_forward.w3.weight"),
            vec![text.intermediate_size, text.hidden_size],
        )?;
        add_shape(
            &mut shapes,
            format!("{root}.feed_forward.w2.weight"),
            vec![text.hidden_size, text.intermediate_size],
        )?;
        match layer_type {
            LayerType::FullAttention => {
                for (name, out_width) in [
                    ("q_proj", text.hidden_size),
                    ("k_proj", kv_width),
                    ("v_proj", kv_width),
                    ("out_proj", text.hidden_size),
                ] {
                    add_shape(
                        &mut shapes,
                        format!("{root}.self_attn.{name}.weight"),
                        vec![out_width, text.hidden_size],
                    )?;
                }
                for norm in ["q_layernorm", "k_layernorm"] {
                    add_shape(
                        &mut shapes,
                        format!("{root}.self_attn.{norm}.weight"),
                        vec![head_dim],
                    )?;
                }
            }
            LayerType::Conv => {
                add_shape(
                    &mut shapes,
                    format!("{root}.conv.in_proj.weight"),
                    vec![conv_projection, text.hidden_size],
                )?;
                add_shape(
                    &mut shapes,
                    format!("{root}.conv.out_proj.weight"),
                    vec![text.hidden_size, text.hidden_size],
                )?;
                add_shape(
                    &mut shapes,
                    format!("{root}.conv.conv.weight"),
                    vec![text.hidden_size, 1, text.conv_l_cache],
                )?;
            }
        }
    }
    add_shape(
        &mut shapes,
        format!("{LANGUAGE_ROOT}.embedding_norm.weight"),
        vec![text.hidden_size],
    )?;
    if !text.tie_embedding {
        add_shape(
            &mut shapes,
            format!("{LM_HEAD_ROOT}.weight"),
            vec![text.vocab_size, text.hidden_size],
        )?;
    }
    Ok(shapes)
}

fn add_shape(
    shapes: &mut BTreeMap<String, Vec<usize>>,
    name: String,
    shape: Vec<usize>,
) -> Result<()> {
    if shape.is_empty() || shape.contains(&0) {
        bail!("native LFM2-VL expected tensor {name:?} has an invalid shape")
    }
    if shapes.insert(name.clone(), shape).is_some() {
        bail!("native LFM2-VL expected tensor {name:?} is duplicated")
    }
    Ok(())
}
