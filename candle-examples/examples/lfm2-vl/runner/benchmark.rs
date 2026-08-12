const GENERATION_BENCHMARK_CONTRACT: &str = "candle-lfm2-vl-generation-benchmark-v1";
const GENERATION_BENCHMARK_WARMUPS: usize = 10;
const GENERATION_BENCHMARK_MEASUREMENTS: usize = 30;
const GENERATION_BENCHMARK_MAX_RELATIVE_MAD: f64 = 0.05;

#[derive(Debug, Serialize)]
struct GenerationBenchmarkReport {
    contract: &'static str,
    backend: String,
    text_device: String,
    vision_device: String,
    warmup_iterations: usize,
    measured_iterations: usize,
    max_new_tokens: usize,
    generated_ids: Vec<u32>,
    durations_ms: Vec<f64>,
    median_ms: f64,
    mad_ms: f64,
    relative_mad: f64,
    max_relative_mad: f64,
    stable: bool,
    timed_region: &'static str,
}

fn run_generation_benchmark(
    runtime: &mut impl Runtime,
    inputs: &GenerationInputs<'_>,
    backend: &str,
    expected_ids: &[u32],
) -> Result<GenerationBenchmarkReport> {
    if inputs.max_new_tokens < 2 {
        bail!("LFM2-VL generation benchmark requires at least two generated tokens")
    }
    for iteration in 0..GENERATION_BENCHMARK_WARMUPS {
        let (_, ids) = benchmark_generation_once(runtime, inputs)?;
        ensure_benchmark_ids(iteration, "warm-up", &ids, expected_ids)?;
    }

    let mut durations_ms = Vec::new();
    durations_ms
        .try_reserve_exact(GENERATION_BENCHMARK_MEASUREMENTS)
        .map_err(|_| anyhow::anyhow!("allocating generation benchmark durations"))?;
    for iteration in 0..GENERATION_BENCHMARK_MEASUREMENTS {
        let (duration_ms, ids) = benchmark_generation_once(runtime, inputs)?;
        ensure_benchmark_ids(iteration, "measured", &ids, expected_ids)?;
        durations_ms.push(duration_ms);
    }
    let (median_ms, mad_ms) = median_and_mad(&durations_ms)?;
    let relative_mad = if median_ms == 0.0 {
        0.0
    } else {
        mad_ms / median_ms
    };
    Ok(GenerationBenchmarkReport {
        contract: GENERATION_BENCHMARK_CONTRACT,
        backend: backend.to_owned(),
        text_device: format!("{:?}", runtime.text_device()),
        vision_device: format!("{:?}", runtime.vision_device()),
        warmup_iterations: GENERATION_BENCHMARK_WARMUPS,
        measured_iterations: GENERATION_BENCHMARK_MEASUREMENTS,
        max_new_tokens: inputs.max_new_tokens,
        generated_ids: expected_ids.to_vec(),
        durations_ms,
        median_ms,
        mad_ms,
        relative_mad,
        max_relative_mad: GENERATION_BENCHMARK_MAX_RELATIVE_MAD,
        stable: relative_mad <= GENERATION_BENCHMARK_MAX_RELATIVE_MAD,
        timed_region: "prefill+greedy-argmax+cached-decode+device-sync; excludes cache-reset, logits hashing/top-k, tokenizer decode, and evidence serialization",
    })
}

fn benchmark_generation_once(
    runtime: &mut impl Runtime,
    inputs: &GenerationInputs<'_>,
) -> Result<(f64, Vec<u32>)> {
    runtime.reset()?;
    runtime.synchronize_devices()?;
    let started = std::time::Instant::now();
    let prefill_logits =
        runtime.prefill(inputs.input_ids, inputs.image_spans, inputs.encoded_images)?;
    let mut next = benchmark_argmax(&prefill_logits)?;
    let mut generated_ids = Vec::new();
    generated_ids
        .try_reserve_exact(inputs.max_new_tokens)
        .map_err(|_| anyhow::anyhow!("allocating benchmark generated token buffer"))?;
    for step in 0..inputs.max_new_tokens {
        generated_ids.push(next);
        if inputs.eos_token_id == Some(next) || step + 1 == inputs.max_new_tokens {
            break;
        }
        let input_position = inputs
            .prompt_len
            .checked_add(step)
            .ok_or_else(|| anyhow::anyhow!("LFM2-VL benchmark decode position overflow"))?;
        let decode_ids = Tensor::new(&[next], runtime.text_device())?.unsqueeze(0)?;
        let decode_logits = runtime.decode(&decode_ids, input_position)?;
        next = benchmark_argmax(&decode_logits)?;
    }
    runtime.synchronize_devices()?;
    Ok((started.elapsed().as_secs_f64() * 1000.0, generated_ids))
}

fn benchmark_argmax(logits: &Tensor) -> Result<u32> {
    last_logits(logits)?
        .argmax(0)?
        .to_scalar::<u32>()
        .map_err(anyhow::Error::from)
}

fn ensure_benchmark_ids(
    iteration: usize,
    phase: &str,
    actual: &[u32],
    expected: &[u32],
) -> Result<()> {
    if actual != expected {
        bail!(
            "LFM2-VL generation benchmark {phase} iteration {iteration} diverged: expected {expected:?}, got {actual:?}"
        )
    }
    Ok(())
}

fn median_and_mad(values: &[f64]) -> Result<(f64, f64)> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite() || *value < 0.0) {
        bail!("generation benchmark durations must be non-empty, finite, and non-negative")
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = median_of_sorted(&sorted);
    let mut deviations = sorted
        .iter()
        .map(|value| (value - median).abs())
        .collect::<Vec<_>>();
    deviations.sort_by(f64::total_cmp);
    Ok((median, median_of_sorted(&deviations)))
}

fn median_of_sorted(values: &[f64]) -> f64 {
    let midpoint = values.len() / 2;
    if values.len().is_multiple_of(2) {
        values[midpoint - 1] + (values[midpoint] - values[midpoint - 1]) / 2.0
    } else {
        values[midpoint]
    }
}
