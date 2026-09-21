use candle::{Error, Result};
#[cfg(feature = "cuda")]
use candle_transformers::models::gpt_oss::{
    cache_bytes_for_tokens, cache_bytes_per_token, load_gpt_oss_weights_with_cancellation,
    GptOssCancellationToken, GptOssCudaConfig, GptOssCudaModel, GptOssLoadRegistry,
    GptOssResourceLimits,
};
#[cfg(feature = "cuda")]
use serde::Serialize;
#[cfg(feature = "cuda")]
use std::fs;
#[cfg(feature = "cuda")]
use std::path::{Path, PathBuf};
#[cfg(feature = "cuda")]
use std::thread;
#[cfg(feature = "cuda")]
use std::time::{Duration, Instant};
#[cfg(feature = "cuda")]
use tokenizers::Tokenizer;

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("gpt-oss-performance requires --features cuda");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct Args {
    model: PathBuf,
    tokenizer: PathBuf,
    output: PathBuf,
    lengths: String,
    target_context_tokens: Option<usize>,
    decode_tokens: usize,
    prompt: String,
    device_index: usize,
    max_weight_bytes: usize,
    post_unload_hold_ms: u64,
}

#[cfg(feature = "cuda")]
impl Args {
    fn parse() -> Result<Self> {
        let mut values = std::env::args().skip(1);
        let mut model = None;
        let mut tokenizer = None;
        let mut output = None;
        let mut lengths = None;
        let mut target_context_tokens = None;
        let mut decode_tokens = 4usize;
        let mut prompt = String::from(
            "The quick brown fox jumps over the lazy dog. Performance characterization prompt.",
        );
        let mut device_index = 0usize;
        let mut max_weight_bytes = usize::MAX;
        let mut post_unload_hold_ms = 10_000u64;
        while let Some(flag) = values.next() {
            let value =
                |name: &str, values: &mut std::iter::Skip<std::env::Args>| -> Result<String> {
                    values
                        .next()
                        .ok_or_else(|| Error::Msg(format!("missing value for {name}")))
                };
            match flag.as_str() {
                "--model" => model = Some(PathBuf::from(value("--model", &mut values)?)),
                "--tokenizer" => {
                    tokenizer = Some(PathBuf::from(value("--tokenizer", &mut values)?))
                }
                "--output" => output = Some(PathBuf::from(value("--output", &mut values)?)),
                "--lengths" => lengths = Some(value("--lengths", &mut values)?),
                "--target-context-tokens" => {
                    target_context_tokens = Some(
                        value("--target-context-tokens", &mut values)?
                            .parse()
                            .map_err(|error| {
                                Error::Msg(format!("invalid --target-context-tokens: {error}"))
                            })?,
                    )
                }
                "--decode-tokens" => {
                    decode_tokens = value("--decode-tokens", &mut values)?
                        .parse()
                        .map_err(|error| Error::Msg(format!("invalid --decode-tokens: {error}")))?
                }
                "--prompt" => prompt = value("--prompt", &mut values)?,
                "--device-index" => {
                    device_index = value("--device-index", &mut values)?
                        .parse()
                        .map_err(|error| Error::Msg(format!("invalid --device-index: {error}")))?
                }
                "--max-weight-bytes" => {
                    max_weight_bytes =
                        value("--max-weight-bytes", &mut values)?
                            .parse()
                            .map_err(|error| {
                                Error::Msg(format!("invalid --max-weight-bytes: {error}"))
                            })?
                }
                "--post-unload-hold-ms" => {
                    post_unload_hold_ms = value("--post-unload-hold-ms", &mut values)?
                        .parse()
                        .map_err(|error| {
                            Error::Msg(format!("invalid --post-unload-hold-ms: {error}"))
                        })?
                }
                "--help" | "-h" => {
                    println!(
                        "gpt-oss-performance --model PATH --tokenizer PATH --output PATH --lengths LIST [--target-context-tokens N] [--decode-tokens N] [--prompt TEXT] [--device-index N] [--max-weight-bytes N] [--post-unload-hold-ms N]"
                    );
                    std::process::exit(0);
                }
                other => return Err(Error::Msg(format!("unknown argument {other}"))),
            }
        }
        Ok(Self {
            model: required(model, "--model")?,
            tokenizer: required(tokenizer, "--tokenizer")?,
            output: required(output, "--output")?,
            lengths: required(lengths, "--lengths")?,
            target_context_tokens,
            decode_tokens,
            prompt,
            device_index,
            max_weight_bytes,
            post_unload_hold_ms,
        })
    }
}

#[cfg(feature = "cuda")]
fn required<T>(value: Option<T>, name: &str) -> Result<T> {
    value.ok_or_else(|| Error::Msg(format!("missing required argument {name}")))
}

#[cfg(feature = "cuda")]
#[derive(Debug, Serialize)]
struct CaseReport {
    actual_prompt_tokens: usize,
    decode_tokens: usize,
    prefill_ms: f64,
    prefill_tokens_per_second: f64,
    decode_ms: f64,
    decode_tokens_per_second: f64,
    total_inference_ms: f64,
    allocated_cache_bytes_after_prefill: usize,
    allocated_cache_bytes_after_decode: usize,
    cache_capacity_bytes_after_decode: usize,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Serialize)]
struct Report {
    schema: &'static str,
    status: &'static str,
    model_path: String,
    model_sha256: String,
    model_bytes: u64,
    device: String,
    device_index: usize,
    prompt: String,
    prompt_token_seed_length: usize,
    configured_context_length: usize,
    requested_target_context_length: Option<usize>,
    rope_initial_context_length: usize,
    cache_bytes_per_token: usize,
    configured_cache_bytes: usize,
    static_resident_bytes: usize,
    packed_resident_bytes: usize,
    tokenizer_load_ms: f64,
    cold_load_ms: f64,
    warmup_ms: f64,
    cases: Vec<CaseReport>,
    registry_loaded_count_after_teardown: usize,
}

#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    let args = Args::parse()?;
    let tokenizer_start = Instant::now();
    let tokenizer = Tokenizer::from_file(&args.tokenizer).map_err(|error| {
        Error::Msg(format!(
            "failed to load tokenizer {:?}: {error}",
            args.tokenizer
        ))
    })?;
    let tokenizer_load_ms = elapsed_ms(tokenizer_start);
    let seed = tokenizer
        .encode(args.prompt.as_str(), false)
        .map_err(|error| Error::Msg(format!("failed to tokenize benchmark prompt: {error}")))?
        .get_ids()
        .to_vec();
    if seed.is_empty() {
        return Err(Error::Msg(
            "benchmark prompt produced no tokens".to_string(),
        ));
    }
    println!("PERF_PHASE tokenizer_ready");
    flush_stdout();

    println!("PERF_PHASE cold_load_start");
    flush_stdout();
    let cold_start = Instant::now();
    let registry = GptOssLoadRegistry::default();
    let (artifact, weights, handle) = load_gpt_oss_weights_with_cancellation(
        &args.model,
        &registry,
        &GptOssCancellationToken::new(),
        args.max_weight_bytes,
    )?;
    let configured_context_length = artifact.context_length();
    let rope_initial_context_length = artifact.config().initial_context_length;
    let cache_per_token = cache_bytes_per_token(artifact.config())?;
    let configured_cache_bytes =
        cache_bytes_for_tokens(artifact.config(), configured_context_length)?;
    let limits = GptOssResourceLimits {
        max_sequence_tokens: configured_context_length,
        max_cache_bytes: configured_cache_bytes,
    };
    limits.validate(artifact.config())?;
    let cuda_config = GptOssCudaConfig::new(
        args.device_index,
        candle::DType::F32,
        args.max_weight_bytes,
        limits,
    )
    .map_err(|error| Error::Msg(error.to_string()))?;
    let mut model = GptOssCudaModel::new(artifact.config().clone(), weights, cuda_config)
        .map_err(|error| Error::Msg(error.to_string()))?;
    model
        .synchronize()
        .map_err(|error| Error::Msg(error.to_string()))?;
    let cold_load_ms = elapsed_ms(cold_start);
    println!("PERF_PHASE model_ready");
    flush_stdout();

    let lengths = resolve_lengths(
        &args.lengths,
        configured_context_length,
        args.target_context_tokens,
        args.decode_tokens,
    )?;
    let mut ids = Vec::new();
    let max_length = lengths.iter().copied().max().unwrap_or(0);
    ids.try_reserve_exact(
        max_length
            .checked_add(args.decode_tokens)
            .ok_or_else(|| Error::Msg("benchmark token length overflowed".to_string()))?,
    )
    .map_err(|error| Error::Msg(format!("benchmark token allocation failed: {error}")))?;
    for index in 0..(max_length + args.decode_tokens) {
        let token = seed
            .get(index % seed.len())
            .copied()
            .ok_or_else(|| Error::Msg("benchmark token seed indexing failed".to_string()))?;
        ids.push(token);
    }

    let warmup_start = Instant::now();
    println!("PERF_PHASE warmup_start");
    flush_stdout();
    let warmup_length = lengths
        .first()
        .copied()
        .ok_or_else(|| Error::Msg("benchmark requires at least one length".to_string()))?
        .min(8);
    model
        .reset_cache()
        .map_err(|error| Error::Msg(error.to_string()))?;
    model
        .prefill(&ids[..warmup_length], &GptOssCancellationToken::new())
        .map_err(|error| Error::Msg(error.to_string()))?;
    model
        .decode(ids[warmup_length], &GptOssCancellationToken::new())
        .map_err(|error| Error::Msg(error.to_string()))?;
    model
        .synchronize()
        .map_err(|error| Error::Msg(error.to_string()))?;
    model
        .reset_cache()
        .map_err(|error| Error::Msg(error.to_string()))?;
    let warmup_ms = elapsed_ms(warmup_start);
    println!("PERF_PHASE warmup_ready");
    flush_stdout();

    let mut cases = Vec::with_capacity(lengths.len());
    for length in lengths {
        println!("PERF_PHASE case_{length}_start");
        flush_stdout();
        model
            .reset_cache()
            .map_err(|error| Error::Msg(error.to_string()))?;
        let total_start = Instant::now();
        let prefill_start = Instant::now();
        model
            .prefill(&ids[..length], &GptOssCancellationToken::new())
            .map_err(|error| Error::Msg(error.to_string()))?;
        model
            .synchronize()
            .map_err(|error| Error::Msg(error.to_string()))?;
        let prefill_ms = elapsed_ms(prefill_start);
        let after_prefill = model
            .resource_usage()
            .map_err(|error| Error::Msg(error.to_string()))?;
        let decode_start = Instant::now();
        for index in 0..args.decode_tokens {
            model
                .decode(ids[length + index], &GptOssCancellationToken::new())
                .map_err(|error| Error::Msg(error.to_string()))?;
        }
        model
            .synchronize()
            .map_err(|error| Error::Msg(error.to_string()))?;
        let decode_ms = elapsed_ms(decode_start);
        let after_decode = model
            .resource_usage()
            .map_err(|error| Error::Msg(error.to_string()))?;
        let total_inference_ms = elapsed_ms(total_start);
        cases.push(CaseReport {
            actual_prompt_tokens: length,
            decode_tokens: args.decode_tokens,
            prefill_ms,
            prefill_tokens_per_second: rate(length, prefill_ms),
            decode_ms,
            decode_tokens_per_second: rate(args.decode_tokens, decode_ms),
            total_inference_ms,
            allocated_cache_bytes_after_prefill: after_prefill.cache_bytes,
            allocated_cache_bytes_after_decode: after_decode.cache_bytes,
            cache_capacity_bytes_after_decode: after_decode.cache_capacity_bytes,
        });
        println!("PERF_PHASE case_{length}_ready");
        flush_stdout();
    }

    let device = format!("{:?}", model.device());
    let static_resident_bytes = model.static_resident_bytes();
    let packed_resident_bytes = model.packed_resident_bytes();
    drop(model);
    drop(artifact);
    drop(handle);
    let registry_loaded_count_after_teardown = registry.loaded_count()?;
    if registry_loaded_count_after_teardown != 0 {
        return Err(Error::Msg(format!(
            "GPT-OSS registry retained {} entries after benchmark teardown",
            registry_loaded_count_after_teardown
        )));
    }
    println!("PERF_PHASE post_unload");
    flush_stdout();
    thread::sleep(Duration::from_millis(args.post_unload_hold_ms));

    let report = Report {
        schema: "candle.gpt_oss_performance.v1",
        status: "passed",
        model_path: args.model.to_string_lossy().into_owned(),
        model_sha256: candle_transformers::models::gpt_oss::SELECTED_GPT_OSS_GGUF_SHA256
            .to_string(),
        model_bytes: fs::metadata(&args.model)?.len(),
        device,
        device_index: args.device_index,
        prompt: args.prompt,
        prompt_token_seed_length: seed.len(),
        configured_context_length,
        requested_target_context_length: args.target_context_tokens,
        rope_initial_context_length,
        cache_bytes_per_token: cache_per_token,
        configured_cache_bytes,
        static_resident_bytes,
        packed_resident_bytes,
        tokenizer_load_ms,
        cold_load_ms,
        warmup_ms,
        cases,
        registry_loaded_count_after_teardown,
    };
    write_json(&args.output, &report)?;
    println!("PERF_PHASE report_written");
    flush_stdout();
    Ok(())
}

#[cfg(feature = "cuda")]
fn resolve_lengths(
    spec: &str,
    configured_context: usize,
    target_context: Option<usize>,
    decode_tokens: usize,
) -> Result<Vec<usize>> {
    let mut lengths = Vec::new();
    for item in spec.split(',') {
        let item = item.trim();
        let length = if item.eq_ignore_ascii_case("auto") {
            let target_context = target_context.ok_or_else(|| {
                Error::Msg(
                    "benchmark length 'auto' requires --target-context-tokens so the measurement target is explicit"
                        .to_string(),
                )
            })?;
            let target_context = target_context.min(configured_context);
            target_context.checked_sub(decode_tokens).ok_or_else(|| {
                Error::Msg("decode token count exceeds configured context".to_string())
            })?
        } else {
            item.parse::<usize>().map_err(|error| {
                Error::Msg(format!("invalid benchmark length {item:?}: {error}"))
            })?
        };
        if length == 0
            || length
                .checked_add(decode_tokens)
                .is_none_or(|value| value > configured_context)
        {
            return Err(Error::Msg(format!(
                "benchmark length {length} plus decode count {decode_tokens} exceeds configured context {configured_context}"
            )));
        }
        lengths.push(length);
    }
    if lengths.is_empty() {
        return Err(Error::Msg("benchmark length list is empty".to_string()));
    }
    Ok(lengths)
}

#[cfg(feature = "cuda")]
fn rate(tokens: usize, milliseconds: f64) -> f64 {
    if milliseconds > 0.0 {
        tokens as f64 * 1000.0 / milliseconds
    } else {
        0.0
    }
}

#[cfg(feature = "cuda")]
fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[cfg(feature = "cuda")]
fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| Error::Msg(format!("failed to serialize performance report: {error}")))?;
    fs::write(path, bytes)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn flush_stdout() {
    use std::io::Write;
    let _ = std::io::stdout().flush();
}
