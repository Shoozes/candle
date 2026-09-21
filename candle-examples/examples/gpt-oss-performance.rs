use candle::{Error, IndexOp, Result};
#[cfg(feature = "cuda")]
use candle_transformers::models::gpt_oss::{
    cache_bytes_for_tokens, cache_bytes_per_token, load_gpt_oss_weights_with_cancellation,
    max_supported_sequence_tokens, total_device_bytes_for_tokens, workspace_bytes_for_tokens,
    GptOssCancellationToken, GptOssCudaConfig, GptOssCudaError, GptOssCudaModel,
    GptOssLoadRegistry, GptOssResourceLimits, EDGE_DEVICE_CEILING_BYTES,
};
#[cfg(feature = "cuda")]
use serde::Serialize;
#[cfg(feature = "cuda")]
use std::fs;
#[cfg(feature = "cuda")]
use std::path::{Path, PathBuf};
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::thread;
#[cfg(feature = "cuda")]
use std::time::{Duration, Instant};
#[cfg(feature = "cuda")]
use tokenizers::Tokenizer;

#[cfg(feature = "cuda")]
const SMALL_TEST_CONTEXT_CEILING_TOKENS: usize = 8_192;

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("gpt-oss-performance requires --features cuda");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MeasurementMode {
    Autoregressive,
    TeacherForced,
}

#[cfg(feature = "cuda")]
impl MeasurementMode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "autoregressive" => Ok(Self::Autoregressive),
            "teacher-forced" => Ok(Self::TeacherForced),
            other => Err(Error::Msg(format!(
                "invalid --mode {other:?}; expected autoregressive or teacher-forced"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Autoregressive => "autoregressive",
            Self::TeacherForced => "teacher-forced",
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct Args {
    model: PathBuf,
    tokenizer: PathBuf,
    output: PathBuf,
    lengths: String,
    target_context_tokens: usize,
    decode_tokens: usize,
    prompt: String,
    device_index: usize,
    max_weight_bytes: usize,
    max_cache_bytes: usize,
    max_total_device_bytes: usize,
    overall_deadline_ms: u64,
    post_unload_hold_ms: u64,
    profile_id: String,
    build_identity: String,
    mode: MeasurementMode,
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
        let mut decode_tokens = 32usize;
        let mut prompt = String::from(
            "The quick brown fox jumps over the lazy dog. Performance characterization prompt.",
        );
        let mut device_index = 0usize;
        let mut max_weight_bytes = None;
        let mut max_cache_bytes = None;
        let mut max_total_device_bytes = None;
        let mut overall_deadline_ms = None;
        let mut post_unload_hold_ms = 10_000u64;
        let mut profile_id = None;
        let mut build_identity = None;
        let mut mode = MeasurementMode::Autoregressive;
        while let Some(flag) = values.next() {
            let value =
                |name: &str, values: &mut std::iter::Skip<std::env::Args>| -> Result<String> {
                    values
                        .next()
                        .ok_or_else(|| Error::Msg(format!("missing value for {name}")))
                };
            let parse_usize = |name: &str, values: &mut std::iter::Skip<std::env::Args>| {
                value(name, values)?
                    .parse::<usize>()
                    .map_err(|error| Error::Msg(format!("invalid {name}: {error}")))
            };
            let parse_u64 = |name: &str, values: &mut std::iter::Skip<std::env::Args>| {
                value(name, values)?
                    .parse::<u64>()
                    .map_err(|error| Error::Msg(format!("invalid {name}: {error}")))
            };
            match flag.as_str() {
                "--model" => model = Some(PathBuf::from(value("--model", &mut values)?)),
                "--tokenizer" => {
                    tokenizer = Some(PathBuf::from(value("--tokenizer", &mut values)?))
                }
                "--output" => output = Some(PathBuf::from(value("--output", &mut values)?)),
                "--lengths" => lengths = Some(value("--lengths", &mut values)?),
                "--target-context-tokens" => {
                    target_context_tokens =
                        Some(parse_usize("--target-context-tokens", &mut values)?)
                }
                "--decode-tokens" => decode_tokens = parse_usize("--decode-tokens", &mut values)?,
                "--prompt" => prompt = value("--prompt", &mut values)?,
                "--device-index" => device_index = parse_usize("--device-index", &mut values)?,
                "--max-weight-bytes" => {
                    max_weight_bytes = Some(parse_usize("--max-weight-bytes", &mut values)?)
                }
                "--max-cache-bytes" => {
                    max_cache_bytes = Some(parse_usize("--max-cache-bytes", &mut values)?)
                }
                "--max-total-device-bytes" => {
                    max_total_device_bytes =
                        Some(parse_usize("--max-total-device-bytes", &mut values)?)
                }
                "--overall-deadline-ms" => {
                    overall_deadline_ms = Some(parse_u64("--overall-deadline-ms", &mut values)?)
                }
                "--post-unload-hold-ms" => {
                    post_unload_hold_ms = parse_u64("--post-unload-hold-ms", &mut values)?
                }
                "--profile-id" => profile_id = Some(value("--profile-id", &mut values)?),
                "--build-identity" => {
                    build_identity = Some(value("--build-identity", &mut values)?)
                }
                "--mode" => mode = MeasurementMode::parse(&value("--mode", &mut values)?)?,
                "--help" | "-h" => {
                    println!(
                        "gpt-oss-performance --model PATH --tokenizer PATH --output PATH --lengths LIST --target-context-tokens N --max-weight-bytes N --max-cache-bytes N --max-total-device-bytes N --overall-deadline-ms N --profile-id ID --build-identity ID [--mode autoregressive|teacher-forced] [--decode-tokens N] [--prompt TEXT] [--device-index N] [--post-unload-hold-ms N]"
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
            target_context_tokens: required(target_context_tokens, "--target-context-tokens")?,
            decode_tokens,
            prompt,
            device_index,
            max_weight_bytes: required(max_weight_bytes, "--max-weight-bytes")?,
            max_cache_bytes: required(max_cache_bytes, "--max-cache-bytes")?,
            max_total_device_bytes: required(max_total_device_bytes, "--max-total-device-bytes")?,
            overall_deadline_ms: required(overall_deadline_ms, "--overall-deadline-ms")?,
            post_unload_hold_ms,
            profile_id: required(profile_id, "--profile-id")?,
            build_identity: required(build_identity, "--build-identity")?,
            mode,
        })
    }

    fn validate_pre_load(&self) -> Result<Vec<usize>> {
        if self.target_context_tokens == 0 {
            return Err(Error::Msg(
                "target context token count must be greater than zero".to_string(),
            ));
        }
        if self.target_context_tokens > SMALL_TEST_CONTEXT_CEILING_TOKENS {
            return Err(Error::Msg(format!(
                "bounded GPT-OSS qualification refuses target contexts above {} tokens",
                SMALL_TEST_CONTEXT_CEILING_TOKENS
            )));
        }
        if self.decode_tokens == 0 {
            return Err(Error::Msg(
                "generated/decode token count must be greater than zero".to_string(),
            ));
        }
        if self.overall_deadline_ms == 0 {
            return Err(Error::Msg(
                "overall deadline must be greater than zero".to_string(),
            ));
        }
        if self.max_weight_bytes == 0
            || self.max_cache_bytes == 0
            || self.max_total_device_bytes == 0
        {
            return Err(Error::Msg(
                "weight, cache, and total-device budgets must be greater than zero".to_string(),
            ));
        }
        if self.max_total_device_bytes != EDGE_DEVICE_CEILING_BYTES {
            return Err(Error::Msg(format!(
                "GPT-OSS Edge qualification requires max total-device budget exactly {} bytes",
                EDGE_DEVICE_CEILING_BYTES
            )));
        }
        if self.profile_id.trim().is_empty() || self.build_identity.trim().is_empty() {
            return Err(Error::Msg(
                "profile and build identities must be non-empty".to_string(),
            ));
        }
        if self.mode == MeasurementMode::Autoregressive && self.decode_tokens < 16 {
            return Err(Error::Msg(
                "autoregressive qualification requires at least 16 generated tokens".to_string(),
            ));
        }
        if !self.model.is_file() {
            return Err(Error::Msg(format!(
                "GPT-OSS model input is not a regular file: {:?}",
                self.model
            )));
        }
        if !self.tokenizer.is_file() {
            return Err(Error::Msg(format!(
                "tokenizer input is not a regular file: {:?}",
                self.tokenizer
            )));
        }
        validate_output_path(&self.output)?;
        parse_lengths(
            &self.lengths,
            self.target_context_tokens,
            self.decode_tokens,
        )
    }

    fn progress_path(&self) -> Result<PathBuf> {
        sidecar_path(&self.output, "runner-progress.json")
    }

    fn terminal_path(&self) -> Result<PathBuf> {
        sidecar_path(&self.output, "runner-terminal.json")
    }
}

#[cfg(feature = "cuda")]
fn required<T>(value: Option<T>, name: &str) -> Result<T> {
    value.ok_or_else(|| Error::Msg(format!("missing required argument {name}")))
}

#[cfg(feature = "cuda")]
fn validate_output_path(path: &Path) -> Result<()> {
    if path.exists() {
        return Err(Error::Msg(format!(
            "refusing to overwrite existing performance output: {:?}",
            path
        )));
    }
    let parent = path
        .parent()
        .ok_or_else(|| Error::Msg(format!("performance output has no parent: {path:?}")))?;
    if parent.exists() && !parent.is_dir() {
        return Err(Error::Msg(format!(
            "performance output parent is not a directory: {parent:?}"
        )));
    }
    fs::create_dir_all(parent)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn sidecar_path(output: &Path, name: &str) -> Result<PathBuf> {
    let parent = output
        .parent()
        .ok_or_else(|| Error::Msg(format!("performance output has no parent: {output:?}")))?;
    Ok(parent.join(name))
}

#[cfg(feature = "cuda")]
fn parse_lengths(spec: &str, target_context: usize, decode_tokens: usize) -> Result<Vec<usize>> {
    let available_prompt_tokens = target_context.checked_sub(decode_tokens).ok_or_else(|| {
        Error::Msg(format!(
            "target context {target_context} cannot fit {decode_tokens} generated/decode tokens"
        ))
    })?;
    if available_prompt_tokens == 0 {
        return Err(Error::Msg(
            "target context must leave at least one prompt token".to_string(),
        ));
    }
    let mut lengths = Vec::new();
    for item in spec.split(',') {
        let item = item.trim();
        if item.is_empty() {
            return Err(Error::Msg(
                "benchmark length list contains an empty item".to_string(),
            ));
        }
        let length = if item.eq_ignore_ascii_case("auto") {
            available_prompt_tokens
        } else {
            item.parse::<usize>().map_err(|error| {
                Error::Msg(format!("invalid benchmark length {item:?}: {error}"))
            })?
        };
        if length == 0 || length > available_prompt_tokens {
            return Err(Error::Msg(format!(
                "benchmark prompt length {length} exceeds the requested target: prompt + generated/decode tokens must be <= {target_context}"
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
#[derive(Debug, Serialize)]
struct CaseReport {
    mode: &'static str,
    requested_prompt_tokens: usize,
    actual_prompt_tokens: usize,
    requested_generated_tokens: usize,
    actual_generated_tokens: usize,
    actual_context_tokens_after_generation: usize,
    prefill_ms: f64,
    prefill_tokens_per_second: f64,
    first_token_ms: Option<f64>,
    steady_decode_ms: f64,
    steady_decode_tokens_per_second: f64,
    total_inference_ms: f64,
    allocated_cache_bytes_after_prefill: usize,
    allocated_cache_bytes_after_generation: usize,
    cache_capacity_bytes_after_generation: usize,
    workspace_bytes_after_generation: usize,
    accounted_device_bytes_after_generation: usize,
    generated_token_ids: Vec<u32>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Serialize)]
struct ProgressReport<'a> {
    schema: &'static str,
    status: &'static str,
    phase: &'static str,
    profile_id: &'a str,
    build_identity: &'a str,
    mode: &'static str,
    target_context_tokens: usize,
    generated_tokens: usize,
    max_weight_bytes: usize,
    max_cache_bytes: usize,
    max_total_device_bytes: usize,
    completed_cases: usize,
    total_cases: usize,
    cases: &'a [CaseReport],
}

#[cfg(feature = "cuda")]
#[derive(Debug, Serialize)]
struct TerminalRecord<'a> {
    schema: &'static str,
    status: &'static str,
    phase: &'static str,
    profile_id: &'a str,
    build_identity: &'a str,
    mode: &'static str,
    completed_cases: usize,
    total_cases: usize,
    report_written: bool,
    stop_reason: &'static str,
    error: Option<String>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Serialize)]
struct Report {
    schema: &'static str,
    status: &'static str,
    profile_id: String,
    build_identity: String,
    mode: &'static str,
    model_path: String,
    model_sha256: String,
    model_bytes: u64,
    device: String,
    device_index: usize,
    prompt: String,
    prompt_token_seed_length: usize,
    configured_context_length: usize,
    target_context_tokens: usize,
    rope_initial_context_length: usize,
    generated_tokens_requested: usize,
    cache_bytes_per_token: usize,
    configured_cache_bytes: usize,
    target_cache_bytes: usize,
    max_weight_bytes: usize,
    max_cache_bytes: usize,
    max_total_device_bytes: usize,
    edge_device_ceiling_bytes: usize,
    static_resident_bytes: usize,
    packed_resident_bytes: usize,
    tokenizer_load_ms: f64,
    cold_load_ms: f64,
    warmup_prompt_tokens: usize,
    warmup_ms: f64,
    max_supported_sequence_tokens: usize,
    max_supported_prompt_tokens: usize,
    workspace_definition: &'static str,
    unload_ms: f64,
    cases: Vec<CaseReport>,
    registry_loaded_count_after_teardown: usize,
}

#[cfg(feature = "cuda")]
struct DeadlineGuard {
    cancellation: Arc<GptOssCancellationToken>,
    deadline_hit: Arc<AtomicBool>,
    finished: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

#[cfg(feature = "cuda")]
impl DeadlineGuard {
    fn new(cancellation: Arc<GptOssCancellationToken>, deadline_ms: u64) -> Self {
        let deadline_hit = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let thread_cancellation = cancellation.clone();
        let thread_deadline_hit = deadline_hit.clone();
        let thread_finished = finished.clone();
        let worker = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_millis(deadline_ms);
            while !thread_finished.load(Ordering::Acquire) {
                let now = Instant::now();
                if now >= deadline {
                    thread_deadline_hit.store(true, Ordering::Release);
                    thread_cancellation.cancel();
                    return;
                }
                thread::sleep((deadline - now).min(Duration::from_millis(25)));
            }
        });
        Self {
            cancellation,
            deadline_hit,
            finished,
            worker: Some(worker),
        }
    }

    fn timed_out(&self) -> bool {
        self.deadline_hit.load(Ordering::Acquire)
    }

    fn check(&self) -> Result<()> {
        if self.timed_out() {
            return Err(Error::Msg(
                "performance overall deadline exceeded".to_string(),
            ));
        }
        if self.cancellation.is_cancelled() {
            return Err(Error::Msg("performance run cancelled".to_string()));
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Drop for DeadlineGuard {
    fn drop(&mut self) {
        self.finished.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(feature = "cuda")]
fn main() {
    let args = match Args::parse() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("gpt-oss-performance argument error: {error}");
            std::process::exit(2);
        }
    };
    let terminal_path = args.terminal_path().ok();
    let result = args.validate_pre_load().and_then(|_| run(&args));
    if let Some(path) = terminal_path {
        let (status, phase, stop_reason, error) = match &result {
            Ok(()) => ("success", "report_written", "completed", None),
            Err(error) if error.to_string().contains("deadline exceeded") => (
                "timeout",
                "deadline",
                "deadline_exceeded",
                Some(error.to_string()),
            ),
            Err(error) if error.to_string().contains("cancelled") => (
                "cancelled",
                "cancelled",
                "cancelled",
                Some(error.to_string()),
            ),
            Err(error) if error.to_string().contains("total device bytes") => (
                "resource_limit",
                "admission",
                "total_device_ceiling",
                Some(error.to_string()),
            ),
            Err(error) => (
                "model_error",
                "runner",
                "runner_error",
                Some(error.to_string()),
            ),
        };
        let progress_path = args.progress_path();
        let completed_cases = read_completed_cases(&progress_path);
        let total_cases = parse_lengths(
            &args.lengths,
            args.target_context_tokens,
            args.decode_tokens,
        )
        .map(|lengths| lengths.len())
        .unwrap_or(0);
        let record = TerminalRecord {
            schema: "candle.gpt_oss_performance_runner_terminal.v2",
            status,
            phase,
            profile_id: &args.profile_id,
            build_identity: &args.build_identity,
            mode: args.mode.as_str(),
            completed_cases,
            total_cases,
            report_written: result.is_ok(),
            stop_reason,
            error,
        };
        let _ = write_json(&path, &record);
    }
    if let Err(error) = result {
        eprintln!("gpt-oss-performance failed: {error}");
        std::process::exit(1);
    }
}

#[cfg(feature = "cuda")]
fn read_completed_cases(path: &Result<PathBuf>) -> usize {
    let Ok(path) = path else { return 0 };
    let Ok(bytes) = fs::read(path) else { return 0 };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return 0;
    };
    value
        .get("completed_cases")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(0)
}

#[cfg(feature = "cuda")]
fn run(args: &Args) -> Result<()> {
    let lengths = parse_lengths(
        &args.lengths,
        args.target_context_tokens,
        args.decode_tokens,
    )?;
    let progress_path = args.progress_path()?;
    let mut cases = Vec::new();
    write_progress(&progress_path, args, "starting", &lengths, &cases)?;

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

    let cancellation = Arc::new(GptOssCancellationToken::new());
    let deadline = DeadlineGuard::new(cancellation.clone(), args.overall_deadline_ms);
    println!("PERF_PHASE cold_load_start");
    flush_stdout();
    let cold_start = Instant::now();
    let registry = GptOssLoadRegistry::default();
    let loaded = load_gpt_oss_weights_with_cancellation(
        &args.model,
        &registry,
        cancellation.as_ref(),
        args.max_weight_bytes,
    );
    let (artifact, weights, handle) = match loaded {
        Ok(value) => value,
        Err(error) if deadline.timed_out() => {
            return Err(Error::Msg(format!(
                "performance overall deadline exceeded during model load: {error}"
            )))
        }
        Err(error) => return Err(error),
    };
    deadline.check()?;
    let configured_context_length = artifact.context_length();
    if args.target_context_tokens > configured_context_length {
        return Err(Error::Msg(format!(
            "requested target context {} exceeds model context {}",
            args.target_context_tokens, configured_context_length
        )));
    }
    let cache_per_token = cache_bytes_per_token(artifact.config())?;
    let target_cache_bytes = cache_bytes_for_tokens(artifact.config(), args.target_context_tokens)?;
    let configured_cache_bytes =
        cache_bytes_for_tokens(artifact.config(), configured_context_length)?;
    if target_cache_bytes > args.max_cache_bytes {
        return Err(Error::Msg(format!(
            "requested target cache {target_cache_bytes} exceeds explicit cache budget {}",
            args.max_cache_bytes
        )));
    }
    let weight_resident_bytes = artifact.weight_resident_bytes()?;
    if weight_resident_bytes > args.max_weight_bytes {
        return Err(Error::Msg(format!(
            "assembled static weight bytes {weight_resident_bytes} exceed explicit weight budget {}",
            args.max_weight_bytes
        )));
    }
    let limits = GptOssResourceLimits {
        max_sequence_tokens: args.target_context_tokens,
        max_cache_bytes: args.max_cache_bytes,
        max_total_device_bytes: args.max_total_device_bytes,
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
    let max_supported_sequence_tokens = max_supported_sequence_tokens(
        artifact.config(),
        weight_resident_bytes,
        args.target_context_tokens,
        args.max_total_device_bytes,
    )?;
    let max_supported_prompt_tokens =
        max_supported_sequence_tokens.saturating_sub(args.decode_tokens);
    cuda_call(model.synchronize(), &deadline)?;
    let cold_load_ms = elapsed_ms(cold_start);
    println!("PERF_PHASE model_ready");
    flush_stdout();
    write_progress(&progress_path, args, "model_ready", &lengths, &cases)?;

    let max_length = lengths.iter().copied().max().unwrap_or(0);
    let id_count = max_length
        .checked_add(if args.mode == MeasurementMode::TeacherForced {
            args.decode_tokens
        } else {
            0
        })
        .ok_or_else(|| Error::Msg("benchmark token length overflowed".to_string()))?;
    let mut ids = Vec::new();
    ids.try_reserve_exact(id_count)
        .map_err(|error| Error::Msg(format!("benchmark token allocation failed: {error}")))?;
    for index in 0..id_count {
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
        .iter()
        .copied()
        .min()
        .ok_or_else(|| Error::Msg("benchmark requires at least one length".to_string()))?
        .min(8);
    model
        .reset_cache()
        .map_err(|error| Error::Msg(error.to_string()))?;
    let warmup_logits = cuda_call(
        model.prefill(&ids[..warmup_length], cancellation.as_ref()),
        &deadline,
    )?;
    if args.mode == MeasurementMode::TeacherForced {
        for index in 0..args.decode_tokens {
            cuda_call(
                model.decode(ids[warmup_length + index], cancellation.as_ref()),
                &deadline,
            )?;
        }
    } else {
        let mut next = argmax(&warmup_logits)?;
        for _ in 1..args.decode_tokens {
            next = argmax(&cuda_call(
                model.decode(next, cancellation.as_ref()),
                &deadline,
            )?)?;
        }
    }
    cuda_call(model.synchronize(), &deadline)?;
    model
        .reset_cache()
        .map_err(|error| Error::Msg(error.to_string()))?;
    let warmup_ms = elapsed_ms(warmup_start);
    println!("PERF_PHASE warmup_ready");
    flush_stdout();

    for length in lengths.iter().copied() {
        deadline.check()?;
        println!("PERF_PHASE case_{length}_start");
        flush_stdout();
        let case = run_case(
            &mut model,
            args,
            &ids,
            length,
            cancellation.as_ref(),
            &deadline,
        )?;
        cases.push(case);
        write_progress(&progress_path, args, "case_ready", &lengths, &cases)?;
        println!("PERF_PHASE case_{length}_ready");
        flush_stdout();
    }

    let device = format!("{:?}", model.device());
    let static_resident_bytes = model.static_resident_bytes();
    let packed_resident_bytes = model.packed_resident_bytes();
    let rope_initial_context_length = artifact.config().initial_context_length;
    let unload_start = Instant::now();
    drop(model);
    drop(artifact);
    drop(handle);
    let registry_loaded_count_after_teardown = registry.loaded_count()?;
    if registry_loaded_count_after_teardown != 0 {
        return Err(Error::Msg(format!(
            "GPT-OSS registry retained {registry_loaded_count_after_teardown} entries after benchmark teardown"
        )));
    }
    let unload_ms = elapsed_ms(unload_start);
    drop(deadline);
    println!("PERF_PHASE post_unload");
    flush_stdout();
    thread::sleep(Duration::from_millis(args.post_unload_hold_ms));

    let report = Report {
        schema: "candle.gpt_oss_performance.v2",
        status: "passed",
        profile_id: args.profile_id.clone(),
        build_identity: args.build_identity.clone(),
        mode: args.mode.as_str(),
        model_path: args.model.to_string_lossy().into_owned(),
        model_sha256: candle_transformers::models::gpt_oss::SELECTED_GPT_OSS_GGUF_SHA256
            .to_string(),
        model_bytes: fs::metadata(&args.model)?.len(),
        device,
        device_index: args.device_index,
        prompt: args.prompt.clone(),
        prompt_token_seed_length: seed.len(),
        configured_context_length,
        target_context_tokens: args.target_context_tokens,
        rope_initial_context_length,
        generated_tokens_requested: args.decode_tokens,
        cache_bytes_per_token: cache_per_token,
        configured_cache_bytes,
        target_cache_bytes,
        max_weight_bytes: args.max_weight_bytes,
        max_cache_bytes: args.max_cache_bytes,
        max_total_device_bytes: args.max_total_device_bytes,
        edge_device_ceiling_bytes: EDGE_DEVICE_CEILING_BYTES,
        static_resident_bytes,
        packed_resident_bytes,
        tokenizer_load_ms,
        cold_load_ms,
        warmup_prompt_tokens: warmup_length,
        warmup_ms,
        max_supported_sequence_tokens,
        max_supported_prompt_tokens,
        workspace_definition: "checked peak one-layer F32 transient bound: 2x attention scores + 8x attention values + 2x QKV + 8x hidden + 4x router + 4x top-expert buffers; static weights and retained KV cache are separate",
        unload_ms,
        cases,
        registry_loaded_count_after_teardown,
    };
    write_json(&args.output, &report)?;
    println!("PERF_PHASE report_written");
    flush_stdout();
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_case(
    model: &mut GptOssCudaModel,
    args: &Args,
    ids: &[u32],
    length: usize,
    cancellation: &GptOssCancellationToken,
    deadline: &DeadlineGuard,
) -> Result<CaseReport> {
    model
        .reset_cache()
        .map_err(|error| Error::Msg(error.to_string()))?;
    let total_start = Instant::now();
    let prefill_start = Instant::now();
    let prefill_logits = cuda_call(model.prefill(&ids[..length], cancellation), deadline)?;
    cuda_call(model.synchronize(), deadline)?;
    let prefill_ms = elapsed_ms(prefill_start);
    let after_prefill = model
        .resource_usage()
        .map_err(|error| Error::Msg(error.to_string()))?;
    let mut generated_token_ids = Vec::new();
    let mut first_token_ms = None;
    let steady_decode_ms;
    let actual_generated_tokens;
    if args.mode == MeasurementMode::Autoregressive {
        let mut next = argmax(&prefill_logits)?;
        generated_token_ids.push(next);
        first_token_ms = Some(prefill_start.elapsed().as_secs_f64() * 1000.0);
        let decode_start = Instant::now();
        while generated_token_ids.len() < args.decode_tokens {
            deadline.check()?;
            next = argmax(&cuda_call(model.decode(next, cancellation), deadline)?)?;
            generated_token_ids.push(next);
        }
        cuda_call(model.synchronize(), deadline)?;
        steady_decode_ms = elapsed_ms(decode_start);
        actual_generated_tokens = generated_token_ids.len();
    } else {
        let decode_start = Instant::now();
        for index in 0..args.decode_tokens {
            deadline.check()?;
            let _ = cuda_call(model.decode(ids[length + index], cancellation), deadline)?;
        }
        cuda_call(model.synchronize(), deadline)?;
        steady_decode_ms = elapsed_ms(decode_start);
        actual_generated_tokens = args.decode_tokens;
    }
    let after_generation = model
        .resource_usage()
        .map_err(|error| Error::Msg(error.to_string()))?;
    let actual_context_tokens_after_generation = after_generation.sequence_tokens;
    let workspace_bytes_after_generation =
        workspace_bytes_for_tokens(model.config(), actual_context_tokens_after_generation)?;
    let accounted_device_bytes_after_generation = total_device_bytes_for_tokens(
        model.config(),
        model.static_resident_bytes(),
        actual_context_tokens_after_generation,
    )?;
    if actual_context_tokens_after_generation > args.target_context_tokens {
        return Err(Error::Msg(format!(
            "runtime context {} exceeded requested target {}",
            actual_context_tokens_after_generation, args.target_context_tokens
        )));
    }
    Ok(CaseReport {
        mode: args.mode.as_str(),
        requested_prompt_tokens: length,
        actual_prompt_tokens: length,
        requested_generated_tokens: args.decode_tokens,
        actual_generated_tokens,
        actual_context_tokens_after_generation,
        prefill_ms,
        prefill_tokens_per_second: rate(length, prefill_ms),
        first_token_ms,
        steady_decode_ms,
        steady_decode_tokens_per_second: rate(
            actual_generated_tokens.saturating_sub(1),
            steady_decode_ms,
        ),
        total_inference_ms: elapsed_ms(total_start),
        allocated_cache_bytes_after_prefill: after_prefill.cache_bytes,
        allocated_cache_bytes_after_generation: after_generation.cache_bytes,
        cache_capacity_bytes_after_generation: after_generation.cache_capacity_bytes,
        workspace_bytes_after_generation,
        accounted_device_bytes_after_generation,
        generated_token_ids,
    })
}

#[cfg(feature = "cuda")]
fn cuda_call<T>(
    result: std::result::Result<T, GptOssCudaError>,
    deadline: &DeadlineGuard,
) -> Result<T> {
    match result {
        Ok(value) => {
            deadline.check()?;
            Ok(value)
        }
        Err(error) if deadline.timed_out() => Err(Error::Msg(format!(
            "performance overall deadline exceeded: {error}"
        ))),
        Err(error) => Err(Error::Msg(error.to_string())),
    }
}

#[cfg(feature = "cuda")]
fn argmax(logits: &candle::Tensor) -> Result<u32> {
    let logits = match logits.rank() {
        1 => logits.clone(),
        2 => {
            let (batch, _) = logits.dims2()?;
            if batch != 1 {
                return Err(Error::Msg(format!(
                    "GPT-OSS performance logits batch {batch} is unsupported; expected 1"
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::Msg(format!(
                "GPT-OSS performance logits rank {rank} is unsupported; expected 1 or 2"
            )))
        }
    };
    logits.argmax(0)?.to_scalar::<u32>()
}

#[cfg(feature = "cuda")]
fn write_progress(
    path: &Path,
    args: &Args,
    phase: &'static str,
    lengths: &[usize],
    cases: &[CaseReport],
) -> Result<()> {
    let report = ProgressReport {
        schema: "candle.gpt_oss_performance_progress.v2",
        status: if cases.len() == lengths.len() {
            "complete"
        } else {
            "partial"
        },
        phase,
        profile_id: &args.profile_id,
        build_identity: &args.build_identity,
        mode: args.mode.as_str(),
        target_context_tokens: args.target_context_tokens,
        generated_tokens: args.decode_tokens,
        max_weight_bytes: args.max_weight_bytes,
        max_cache_bytes: args.max_cache_bytes,
        max_total_device_bytes: args.max_total_device_bytes,
        completed_cases: cases.len(),
        total_cases: lengths.len(),
        cases,
    };
    write_json(path, &report)
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
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| {
        Error::Msg(format!("failed to serialize performance evidence: {error}"))
    })?;
    fs::write(path, bytes)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn flush_stdout() {
    use std::io::Write;
    let _ = std::io::stdout().flush();
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_or_over_target_measurement_plan() {
        assert!(parse_lengths("0", 128, 16).is_err());
        assert!(parse_lengths("113", 128, 16).is_err());
        assert!(parse_lengths("auto", 128, 16).is_ok());
        assert!(parse_lengths("auto", 16, 16).is_err());
    }

    #[test]
    fn deadline_cancels_the_existing_token_cooperatively() {
        let token = Arc::new(GptOssCancellationToken::new());
        let deadline = DeadlineGuard::new(token.clone(), 10);
        thread::sleep(Duration::from_millis(40));
        assert!(deadline.timed_out());
        assert!(token.is_cancelled());
        assert!(deadline.check().is_err());
    }

    #[test]
    fn rank_one_logits_argmax_returns_a_scalar_token() -> Result<()> {
        let logits = candle::Tensor::from_slice(&[0.25f32, 2.0, -1.0], (3,), &candle::Device::Cpu)?;
        assert_eq!(argmax(&logits)?, 1);
        Ok(())
    }

    #[test]
    fn rank_two_batch_one_logits_argmax_returns_a_scalar_token() -> Result<()> {
        let logits =
            candle::Tensor::from_slice(&[0.25f32, 2.0, -1.0], (1, 3), &candle::Device::Cpu)?;
        assert_eq!(argmax(&logits)?, 1);
        Ok(())
    }

    #[test]
    fn output_validation_rejects_stale_report_without_overwrite() {
        let directory = std::env::temp_dir().join(format!(
            "candle-gpt-oss-performance-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).expect("test directory must be creatable");
        let output = directory.join("report.json");
        fs::write(&output, b"owner evidence").expect("test report must be writable");
        assert!(validate_output_path(&output).is_err());
        assert_eq!(
            fs::read(&output).expect("stale report must remain"),
            b"owner evidence"
        );
        let _ = fs::remove_dir_all(directory);
    }
}
