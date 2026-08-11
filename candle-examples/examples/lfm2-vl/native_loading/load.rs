pub fn load_native(
    model_dir: impl AsRef<Path>,
    processor_override: Option<&Path>,
    options: NativeLoadOptions<'_>,
) -> Result<LoadedNative> {
    let checkpoint = resolve_checkpoint(model_dir)?;
    let config_path =
        checkpoint.required_file(CONFIG_FILE, "native model config", MAX_CONFIG_BYTES)?;
    let processor_path = checkpoint.required_file(
        PROCESSOR_FILE,
        "native processor config",
        MAX_PROCESSOR_CONFIG_BYTES,
    )?;
    let tokenizer_path =
        checkpoint.required_file(TOKENIZER_FILE, "native tokenizer", MAX_TOKENIZER_BYTES)?;

    let config_json = read_bounded_utf8(&config_path, "native model config", MAX_CONFIG_BYTES)?;
    let config = Lfm2VlConfig::from_json(&config_json).map_err(anyhow::Error::msg)?;
    let report = inspect_native_checkpoint(&checkpoint, &config, options)?;
    report.require_clean()?;

    let processor_json = read_bounded_utf8(
        &processor_path,
        "native processor config",
        MAX_PROCESSOR_CONFIG_BYTES,
    )?;
    let processor_patch =
        ProcessorConfigPatch::from_json(&processor_json).map_err(anyhow::Error::msg)?;
    let explicit_processor = processor_override
        .map(|path| {
            let path = inspect_bounded_file(
                path,
                "native processor override",
                MAX_PROCESSOR_CONFIG_BYTES,
            )?;
            let json = read_bounded_utf8(
                &path,
                "native processor override",
                MAX_PROCESSOR_CONFIG_BYTES,
            )?;
            let patch = ProcessorConfigPatch::from_json(&json).map_err(anyhow::Error::msg)?;
            Ok::<_, anyhow::Error>((path, patch))
        })
        .transpose()?;
    let model_patch = ProcessorConfigPatch::from_model_config(&config);
    let processor_config = Lfm2VlProcessorConfig::resolve(
        explicit_processor.as_ref().map(|(_, patch)| patch),
        Some(&processor_patch),
        None,
        Some(&model_patch),
    )?;
    if processor_config.encoder_patch_size != config.vision_config.patch_size
        || processor_config.downsample_factor != config.downsample_factor
    {
        bail!(
            "native processor/model mismatch: processor patch/factor [{}, {}], model [{}, {}]",
            processor_config.encoder_patch_size,
            processor_config.downsample_factor,
            config.vision_config.patch_size,
            config.downsample_factor
        )
    }
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(anyhow::Error::msg)?;
    let prompt = Lfm2VlPrompt::from_processor_config(
        tokenizer,
        Some(config.image_token_id),
        &processor_config,
        PromptOptions {
            use_image_special_tokens: config.use_image_special_tokens,
            context_length: Some(config.text_config.max_position_embeddings),
        },
    )?;
    let processor = Lfm2VlProcessor::from_config(&processor_config)?;

    // SAFETY: the resolver canonicalizes regular local files and validates every
    // safetensors header before mapping. The checkpoint files must not be
    // concurrently modified while the returned model is alive.
    let vision_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &checkpoint.weight_files,
            options.vision_dtype,
            options.vision_device,
        )
    }
    .context("mapping native LFM2-VL vision safetensors")?;
    let text_vb = if options.vision_device.same_device(options.text_device)
        && options.vision_dtype == options.text_dtype
    {
        vision_vb.clone()
    } else {
        // SAFETY: this maps the same already-resolved, already-validated files
        // for a distinct target device. They must remain immutable for the
        // lifetime of the returned model.
        unsafe {
            VarBuilder::from_mmaped_safetensors(
                &checkpoint.weight_files,
                options.text_dtype,
                options.text_device,
            )
        }
        .context("mapping native LFM2-VL text safetensors")?
    };
    let text_config = config.text_model_config()?;
    let lm_head_vb = (!text_config.tie_embedding).then(|| text_vb.pp(LM_HEAD_ROOT));
    let model = Lfm2VlModel::new_from_parts(
        &config,
        vision_vb.pp(DIRECT_VISION_ROOT),
        vision_vb.pp(PROJECTOR_ROOT),
        text_vb.pp(LANGUAGE_ROOT),
        lm_head_vb,
    )?;

    let mut source_files = Vec::new();
    let source_file_capacity = checkpoint
        .weight_files
        .len()
        .checked_add(5)
        .ok_or_else(|| anyhow::anyhow!("native source-file count overflow"))?;
    source_files
        .try_reserve_exact(source_file_capacity)
        .map_err(|_| anyhow::anyhow!("allocating native source-file evidence"))?;
    source_files.push(config_path);
    source_files.push(processor_path);
    source_files.push(tokenizer_path);
    if let Some(index_file) = &checkpoint.index_file {
        source_files.push(index_file.clone());
    }
    source_files.extend(checkpoint.weight_files.iter().cloned());
    if let Some((path, _)) = explicit_processor {
        if !source_files.contains(&path) {
            source_files.push(path);
        }
    }

    Ok(LoadedNative {
        model,
        processor,
        prompt,
        report,
        source_files,
    })
}
