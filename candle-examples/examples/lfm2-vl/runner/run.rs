pub fn run_native(
    model: &Lfm2VlModel,
    processor: &Lfm2VlProcessor,
    prompt: &Lfm2VlPrompt,
    request: InferenceRequest<'_>,
) -> Result<InferenceReport> {
    let mut runtime = NativeRuntime::new(model)?;
    run(&mut runtime, processor, prompt, request)
}

pub fn run_hybrid(
    model: &mut QuantizedLfm2VlModel,
    processor: &Lfm2VlProcessor,
    prompt: &Lfm2VlPrompt,
    request: InferenceRequest<'_>,
) -> Result<InferenceReport> {
    let mut runtime = HybridRuntime { model };
    run(&mut runtime, processor, prompt, request)
}

fn run(
    runtime: &mut impl Runtime,
    processor: &Lfm2VlProcessor,
    prompt: &Lfm2VlPrompt,
    request: InferenceRequest<'_>,
) -> Result<InferenceReport> {
    if request.vision_batch_size == 0 {
        bail!("vision batch size must be greater than zero")
    }
    let model_inputs = inspect_input_paths(request.model_inputs)?;
    if request.trace_output.is_some() && request.image_paths.len() != 1 {
        bail!("LFM2-VL native trace requires exactly one image input")
    }
    let limits = &processor.config().vision_limits;
    let (images, image_files) = load_images(request.image_paths, limits)?;
    let processed = if images.is_empty() {
        empty_processed_batch()?
    } else {
        processor
            .process(&images, runtime.vision_device())
            .map_err(anyhow::Error::from)?
    };
    let expanded = prompt
        .expand(request.prompt, &processed)
        .map_err(anyhow::Error::from)?;
    if expanded.input_ids.is_empty() {
        bail!("LFM2-VL inference prompt tokenized to an empty sequence")
    }
    let required_context = expanded
        .input_ids
        .len()
        .checked_add(request.max_new_tokens)
        .ok_or_else(|| anyhow::anyhow!("LFM2-VL requested context length overflow"))?;
    if required_context > runtime.context_length() {
        bail!(
            "LFM2-VL prompt tokens {} plus max new tokens {} exceed context length {}",
            expanded.input_ids.len(),
            request.max_new_tokens,
            runtime.context_length()
        )
    }

    let eos = resolve_eos(
        request.eos_token_id,
        runtime.default_eos_token_id(),
        runtime.default_eos_source(),
        prompt.tokenizer(),
    );
    let (encoded, image_trace) = if images.is_empty() {
        if request.trace_output.is_some() {
            bail!("LFM2-VL native trace requires one decoded image")
        }
        (None, None)
    } else if request.trace_output.is_some() {
        let (encoded, trace) =
            runtime.encode_images_with_trace(&processed, request.vision_batch_size, limits)?;
        (Some(encoded), trace)
    } else {
        (
            Some(runtime.encode_images(&processed, request.vision_batch_size, limits)?),
            None,
        )
    };
    let mut trace_capture = match (request.trace_output, image_trace) {
        (Some(_), Some(image)) => Some(NativeTraceCapture::new(image)),
        (Some(_), None) => bail!("LFM2-VL native trace is unavailable for the selected runtime"),
        (None, _) => None,
    };
    let input_ids =
        Tensor::new(expanded.input_ids.as_slice(), runtime.text_device())?.unsqueeze(0)?;
    let generation_inputs = GenerationInputs {
        tokenizer: prompt.tokenizer(),
        input_ids: &input_ids,
        image_spans: &expanded.image_spans,
        encoded_images: encoded.as_ref(),
        prompt_len: expanded.input_ids.len(),
        max_new_tokens: request.max_new_tokens,
        eos_token_id: eos.token_id,
    };
    let first = generate_once(runtime, &generation_inputs)?;
    if let Some(capture) = trace_capture.as_mut() {
        capture_trace(runtime, &generation_inputs, capture)?;
    }
    let second = generate_once(runtime, &generation_inputs)?;
    if first != second {
        bail!(
            "LFM2-VL cache reset replay diverged: first generated {:?}, second generated {:?}",
            first.generated_ids,
            second.generated_ids
        )
    }
    if request.trace_output.is_some() {
        verify_input_paths_unchanged(request.model_inputs, &model_inputs)?;
    }

    let report = InferenceReport {
        contract: CONTRACT,
        backend: request.backend.to_owned(),
        model_inputs,
        prompt: request.prompt.to_owned(),
        expanded_prompt: expanded.expanded_text.clone(),
        input_ids: expanded.input_ids.clone(),
        image_files,
        processed_images: processed_image_evidence(&processed),
        processed_crops: processed_crop_evidence(&processed),
        image_spans: image_span_evidence(&expanded)?,
        packed_tensors: packed_tensor_evidence(&processed),
        context_length: runtime.context_length(),
        vision_batch_size: request.vision_batch_size,
        max_new_tokens: request.max_new_tokens,
        eos,
        generation: first,
        cache_reset_exact: true,
    };
    if let Some(output) = request.trace_output {
        let capture = trace_capture
            .ok_or_else(|| anyhow::anyhow!("LFM2-VL native trace capture was not initialized"))?;
        trace::write_native_trace(
            output,
            &report,
            &processed,
            &input_ids,
            images.first(),
            capture,
        )?;
    }
    Ok(report)
}
