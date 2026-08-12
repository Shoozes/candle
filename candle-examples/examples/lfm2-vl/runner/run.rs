use std::time::Instant;

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
    let total_started = Instant::now();
    if request.vision_batch_size == 0 {
        bail!("vision batch size must be greater than zero")
    }
    if request.benchmark_generation && request.timings {
        bail!("generation benchmark and end-to-end timings are mutually exclusive")
    }
    if request.benchmark_generation && request.trace_output.is_some() {
        bail!("generation benchmark and native trace capture are mutually exclusive")
    }
    if request.benchmark_generation && request.max_new_tokens < 2 {
        bail!("generation benchmark requires at least two generated tokens")
    }
    let model_inputs = inspect_input_paths(request.model_inputs)?;
    if request.trace_output.is_some() && request.image_paths.len() != 1 {
        bail!("LFM2-VL native trace requires exactly one image input")
    }
    let limits = &processor.config().vision_limits;
    let image_load_started = Instant::now();
    let (images, image_files) = load_images(request.image_paths, limits)?;
    let image_load_ms = image_load_started.elapsed().as_secs_f64() * 1000.0;
    if request.timings {
        runtime.synchronize_devices()?;
    }
    let processor_started = Instant::now();
    let processed = if images.is_empty() {
        empty_processed_batch()?
    } else {
        processor
            .process(&images, runtime.vision_device())
            .map_err(anyhow::Error::from)?
    };
    if request.timings {
        runtime.synchronize_devices()?;
    }
    let processor_ms = processor_started.elapsed().as_secs_f64() * 1000.0;
    let prompt_started = Instant::now();
    let expanded = prompt
        .expand(request.prompt, &processed)
        .map_err(anyhow::Error::from)?;
    let prompt_ms = prompt_started.elapsed().as_secs_f64() * 1000.0;
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
    if request.timings {
        runtime.synchronize_devices()?;
    }
    let vision_started = Instant::now();
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
    if request.timings {
        runtime.synchronize_devices()?;
    }
    let vision_ms = vision_started.elapsed().as_secs_f64() * 1000.0;
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
    if request.timings {
        runtime.synchronize_devices()?;
    }
    let first_started = Instant::now();
    let first = generate_once(runtime, &generation_inputs)?;
    if request.timings {
        runtime.synchronize_devices()?;
    }
    let first_generation_ms = first_started.elapsed().as_secs_f64() * 1000.0;
    if let Some(capture) = trace_capture.as_mut() {
        capture_trace(runtime, &generation_inputs, capture)?;
    }
    if request.timings {
        runtime.synchronize_devices()?;
    }
    let reset_started = Instant::now();
    let second = generate_once(runtime, &generation_inputs)?;
    if request.timings {
        runtime.synchronize_devices()?;
    }
    let reset_generation_ms = reset_started.elapsed().as_secs_f64() * 1000.0;
    if first != second {
        bail!(
            "LFM2-VL cache reset replay diverged: first generated {:?}, second generated {:?}",
            first.generated_ids,
            second.generated_ids
        )
    }
    if request.benchmark_generation {
        let benchmark = run_generation_benchmark(
            runtime,
            &generation_inputs,
            request.backend,
            &first.generated_ids,
        )?;
        eprintln!(
            "lfm2-vl generation_benchmark_json {}",
            serde_json::to_string(&benchmark)?
        );
        if !benchmark.stable {
            bail!(
                "LFM2-VL generation benchmark relative MAD {:.6} exceeds {:.6}",
                benchmark.relative_mad,
                benchmark.max_relative_mad
            )
        }
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
    if request.timings {
        runtime.synchronize_devices()?;
        eprintln!(
            "lfm2-vl timings_ms image_load={image_load_ms:.3} processor={processor_ms:.3} prompt={prompt_ms:.3} vision={vision_ms:.3} generation_first={first_generation_ms:.3} generation_reset={reset_generation_ms:.3} inference_total={:.3} sync=cuda-device-complete",
            total_started.elapsed().as_secs_f64() * 1000.0
        );
    }
    Ok(report)
}
