fn parse_metadata(content: &gguf_file::Content) -> Result<ParsedMetadata> {
    let general_architecture = required_string(content, "general.architecture")?;
    if general_architecture != "clip" {
        candle::bail!(
            "GGUF MMProj general.architecture must be \"clip\", got {general_architecture:?}"
        )
    }
    let general_type = required_string(content, "general.type")?;
    if general_type != "mmproj" {
        candle::bail!("GGUF MMProj general.type must be \"mmproj\", got {general_type:?}")
    }
    let projector_type = required_string(content, "clip.projector_type")?;
    if projector_type != "lfm2" {
        candle::bail!("GGUF MMProj clip.projector_type must be \"lfm2\", got {projector_type:?}")
    }
    if !required_bool(content, "clip.has_vision_encoder")? {
        candle::bail!("GGUF MMProj clip.has_vision_encoder must be true")
    }
    if !required_bool(content, "clip.use_gelu")? {
        candle::bail!("GGUF MMProj clip.use_gelu must be true for the LFM2 projector")
    }

    let vision_layer_count = required_positive_usize(content, "clip.vision.block_count")?;
    if vision_layer_count > MAX_GGUF_VISION_LAYERS {
        candle::bail!(
            "GGUF MMProj vision layer count {vision_layer_count} exceeds {MAX_GGUF_VISION_LAYERS}"
        )
    }
    let image_std = required_f32_triplet(content, "clip.vision.image_std")?;
    if image_std.iter().any(|&value| value <= 0.0) {
        candle::bail!("GGUF MMProj clip.vision.image_std values must be positive")
    }
    let preproc_min = optional_positive_usize(content, "clip.vision.preproc_min_tiles")?;
    let preproc_max = optional_positive_usize(content, "clip.vision.preproc_max_tiles")?;
    let preproc_size = optional_positive_usize(content, "clip.vision.preproc_image_size")?;
    let preproc = match (preproc_min, preproc_max, preproc_size) {
        (None, None, None) => None,
        (Some(min_tiles), Some(max_tiles), Some(image_size)) => {
            if min_tiles > max_tiles {
                candle::bail!(
                    "GGUF MMProj preprocessor min tiles {min_tiles} exceeds max tiles {max_tiles}"
                )
            }
            Some((min_tiles, max_tiles, image_size))
        }
        _ => candle::bail!(
            "GGUF MMProj tiling metadata must provide min tiles, max tiles, and image size together"
        ),
    };
    Ok(ParsedMetadata {
        general_name: optional_string(content, "general.name")?.map(str::to_string),
        image_size: required_positive_usize(content, "clip.vision.image_size")?,
        vision_hidden_size: required_positive_usize(content, "clip.vision.embedding_length")?,
        vision_intermediate_size: required_positive_usize(
            content,
            "clip.vision.feed_forward_length",
        )?,
        vision_layer_count,
        vision_head_count: required_positive_usize(content, "clip.vision.attention.head_count")?,
        layer_norm_eps: required_positive_float(
            content,
            "clip.vision.attention.layer_norm_epsilon",
        )?,
        patch_size: required_positive_usize(content, "clip.vision.patch_size")?,
        image_mean: required_f32_triplet(content, "clip.vision.image_mean")?,
        image_std,
        downsample_factor: required_positive_usize(content, "clip.vision.projector.scale_factor")?,
        text_hidden_size: required_positive_usize(content, "clip.vision.projection_dim")?,
        preproc,
    })
}

fn reconstruct_config(
    content: &gguf_file::Content,
    metadata: &ParsedMetadata,
    image_token_id: u32,
) -> Result<Lfm2VlMmprojConfig> {
    if metadata.image_size % metadata.patch_size != 0 {
        candle::bail!(
            "GGUF MMProj image size {} is not divisible by patch size {}",
            metadata.image_size,
            metadata.patch_size
        )
    }
    let base_side = metadata.image_size / metadata.patch_size;
    let num_patches = base_side
        .checked_mul(base_side)
        .ok_or_else(|| candle::Error::Msg("GGUF MMProj base position count overflowed".into()))?;

    let patch_shape = required_tensor_shape(content, "v.patch_embd.weight")?;
    if patch_shape.len() != 4
        || patch_shape[0] != metadata.vision_hidden_size
        || patch_shape[2] != metadata.patch_size
        || patch_shape[3] != metadata.patch_size
    {
        candle::bail!(
            "GGUF MMProj tensor \"v.patch_embd.weight\" has shape {patch_shape:?}, expected [{}, channels, {}, {}]",
            metadata.vision_hidden_size,
            metadata.patch_size,
            metadata.patch_size
        )
    }
    let num_channels = patch_shape[1];
    if num_channels != metadata.image_mean.len() || num_channels != metadata.image_std.len() {
        candle::bail!(
            "GGUF MMProj patch channels {num_channels} do not match normalization metadata"
        )
    }

    let projector_input = metadata
        .vision_hidden_size
        .checked_mul(metadata.downsample_factor)
        .and_then(|value| value.checked_mul(metadata.downsample_factor))
        .ok_or_else(|| candle::Error::Msg("GGUF MMProj projector input overflowed".into()))?;
    let linear_1_shape = required_tensor_shape(content, "mm.1.weight")?;
    if linear_1_shape.len() != 2 || linear_1_shape[1] != projector_input {
        candle::bail!(
            "GGUF MMProj tensor \"mm.1.weight\" has shape {linear_1_shape:?}, expected [hidden, {projector_input}]"
        )
    }
    let projector_hidden_size = linear_1_shape[0];
    let linear_2_shape = required_tensor_shape(content, "mm.2.weight")?;
    if linear_2_shape != [metadata.text_hidden_size, projector_hidden_size] {
        candle::bail!(
            "GGUF MMProj tensor \"mm.2.weight\" has shape {linear_2_shape:?}, expected [{}, {projector_hidden_size}]",
            metadata.text_hidden_size
        )
    }

    let input_norm_weight = content.tensor_infos.contains_key("mm.input_norm.weight");
    let input_norm_bias = content.tensor_infos.contains_key("mm.input_norm.bias");
    if input_norm_weight != input_norm_bias {
        candle::bail!("GGUF MMProj optional input LayerNorm weight and bias must appear together")
    }
    let linear_1_bias = content.tensor_infos.contains_key("mm.1.bias");
    let linear_2_bias = content.tensor_infos.contains_key("mm.2.bias");
    if linear_1_bias != linear_2_bias {
        candle::bail!("GGUF MMProj projector linear biases must appear together")
    }

    let config = Lfm2VlMmprojConfig {
        vision_config: siglip2::Siglip2VisionConfig {
            hidden_size: metadata.vision_hidden_size,
            intermediate_size: metadata.vision_intermediate_size,
            num_hidden_layers: metadata.vision_layer_count,
            num_attention_heads: metadata.vision_head_count,
            num_channels,
            patch_size: metadata.patch_size,
            num_patches,
            hidden_act: Activation::GeluPytorchTanh,
            layer_norm_eps: metadata.layer_norm_eps,
            attention_dropout: 0.0,
            vision_use_head: false,
        },
        text_hidden_size: metadata.text_hidden_size,
        image_token_id,
        downsample_factor: metadata.downsample_factor,
        projector_hidden_size,
        projector_hidden_act: Activation::Gelu,
        projector_bias: linear_1_bias,
        projector_use_layernorm: input_norm_weight,
        use_image_special_tokens: true,
    };
    config.validate()?;
    Ok(config)
}
