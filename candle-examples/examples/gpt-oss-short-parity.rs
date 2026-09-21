#[cfg(feature = "cuda")]
use candle::{DType, Error, Result, Tensor};
#[cfg(feature = "cuda")]
use candle_transformers::models::gpt_oss::{
    load_gpt_oss_weights_with_cancellation, GptOssCancellationToken, GptOssCudaConfig,
    GptOssCudaModel, GptOssLoadRegistry, GptOssResourceLimits,
};
#[cfg(feature = "cuda")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "cuda")]
use serde_json::json;
#[cfg(feature = "cuda")]
use sha2::{Digest, Sha256};
#[cfg(feature = "cuda")]
use std::cmp::Ordering;
#[cfg(feature = "cuda")]
use std::collections::BTreeSet;
#[cfg(feature = "cuda")]
use std::fs;
#[cfg(feature = "cuda")]
use std::io::{BufReader, Read};
#[cfg(feature = "cuda")]
use std::path::{Path, PathBuf};
#[cfg(feature = "cuda")]
use std::time::Instant;
#[cfg(feature = "cuda")]
use tokenizers::Tokenizer;

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("gpt-oss-short-parity requires --features cuda");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct Args {
    model: PathBuf,
    tokenizer: PathBuf,
    tokenizer_config: PathBuf,
    chat_template: PathBuf,
    fixture: PathBuf,
    reference_logits: PathBuf,
    output: PathBuf,
    prompt: String,
    baseline_commit: String,
    baseline_tree: String,
    candidate_tree: String,
    candidate_dirty: bool,
    device_index: usize,
    max_weight_bytes: usize,
}

#[cfg(feature = "cuda")]
impl Args {
    fn parse() -> Result<Self> {
        let mut values = std::env::args().skip(1);
        let mut model = None;
        let mut tokenizer = None;
        let mut tokenizer_config = None;
        let mut chat_template = None;
        let mut fixture = None;
        let mut reference_logits = None;
        let mut output = None;
        let mut prompt = None;
        let mut baseline_commit = None;
        let mut baseline_tree = None;
        let mut candidate_tree = None;
        let mut candidate_dirty = None;
        let mut device_index = 0usize;
        let mut max_weight_bytes = usize::MAX;
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
                "--tokenizer-config" => {
                    tokenizer_config =
                        Some(PathBuf::from(value("--tokenizer-config", &mut values)?))
                }
                "--chat-template" => {
                    chat_template = Some(PathBuf::from(value("--chat-template", &mut values)?))
                }
                "--fixture" => fixture = Some(PathBuf::from(value("--fixture", &mut values)?)),
                "--reference-logits" => {
                    reference_logits =
                        Some(PathBuf::from(value("--reference-logits", &mut values)?))
                }
                "--output" => output = Some(PathBuf::from(value("--output", &mut values)?)),
                "--prompt" => prompt = Some(value("--prompt", &mut values)?),
                "--baseline-commit" => {
                    baseline_commit = Some(value("--baseline-commit", &mut values)?)
                }
                "--baseline-tree" => baseline_tree = Some(value("--baseline-tree", &mut values)?),
                "--candidate-tree" => {
                    candidate_tree = Some(value("--candidate-tree", &mut values)?)
                }
                "--candidate-dirty" => {
                    candidate_dirty = Some(
                        value("--candidate-dirty", &mut values)?
                            .parse()
                            .map_err(|error| {
                                Error::Msg(format!("invalid --candidate-dirty: {error}"))
                            })?,
                    )
                }
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
                "--help" | "-h" => {
                    println!(
                        "gpt-oss-short-parity --model PATH --tokenizer PATH --tokenizer-config PATH --chat-template PATH --fixture PATH --reference-logits PATH --output PATH --prompt TEXT --baseline-commit SHA --baseline-tree TREE --candidate-tree TREE --candidate-dirty BOOL [--device-index N] [--max-weight-bytes N]"
                    );
                    std::process::exit(0);
                }
                other => return Err(Error::Msg(format!("unknown argument {other}"))),
            }
        }
        Ok(Self {
            model: required(model, "--model")?,
            tokenizer: required(tokenizer, "--tokenizer")?,
            tokenizer_config: required(tokenizer_config, "--tokenizer-config")?,
            chat_template: required(chat_template, "--chat-template")?,
            fixture: required(fixture, "--fixture")?,
            reference_logits: required(reference_logits, "--reference-logits")?,
            output: required(output, "--output")?,
            prompt: required(prompt, "--prompt")?,
            baseline_commit: required(baseline_commit, "--baseline-commit")?,
            baseline_tree: required(baseline_tree, "--baseline-tree")?,
            candidate_tree: required(candidate_tree, "--candidate-tree")?,
            candidate_dirty: required(candidate_dirty, "--candidate-dirty")?,
            device_index,
            max_weight_bytes,
        })
    }
}

#[cfg(feature = "cuda")]
fn required<T>(value: Option<T>, name: &str) -> Result<T> {
    value.ok_or_else(|| Error::Msg(format!("missing required argument {name}")))
}

#[cfg(feature = "cuda")]
#[derive(Debug, Deserialize)]
struct Fixture {
    fixture_id: String,
    prompt: String,
    tokenizer_add_bos: bool,
    tokenizer_add_eos: bool,
    token_ids: Vec<u32>,
    reference_prefix_token_ids: Vec<u32>,
    reference_context: usize,
    reference_vocab_size: usize,
    reference_rows: usize,
    reference_file_bytes: usize,
    reference_file_sha256: String,
    reference_executable: String,
    reference_command: String,
    reference_row_top1: Vec<ReferenceTop1>,
    comparison: Comparison,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Deserialize)]
struct ReferenceTop1 {
    token_id: u32,
    log_prob: f32,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Deserialize)]
struct Comparison {
    top_k: usize,
    top_k_log_probability_abs_tolerance: f32,
    mean_abs_log_probability_tolerance: f32,
    total_variation_probability_tolerance: f32,
    tolerance_rationale: String,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Serialize)]
struct StageReport {
    stage: String,
    max_abs_log_probability_error: f32,
    mean_abs_log_probability_error: f32,
    top_k_max_abs_log_probability_error: f32,
    total_variation_probability_error: f32,
    top_reference: Vec<TopLogProbability>,
    top_candle: Vec<TopLogProbability>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Serialize)]
struct TopLogProbability {
    token_id: usize,
    log_probability: f32,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Serialize)]
struct Receipt {
    status: &'static str,
    fixture_id: String,
    prompt: String,
    token_ids: Vec<u32>,
    model_path: String,
    model_bytes: u64,
    model_sha256: String,
    tokenizer_sha256: String,
    tokenizer_config_sha256: String,
    chat_template_sha256: String,
    baseline_commit: String,
    baseline_tree: String,
    candidate_tree: String,
    candidate_dirty: bool,
    publication_status: &'static str,
    candle_package_version: &'static str,
    feature: &'static str,
    device_index: usize,
    device: String,
    reference_executable: String,
    reference_command: String,
    reference_logits_sha256: String,
    reference_logits_bytes: usize,
    top_k_log_probability_tolerance: f32,
    mean_log_probability_tolerance: f32,
    total_variation_probability_tolerance: f32,
    tolerance_rationale: String,
    stages: Vec<StageReport>,
    reset_replay_max_abs_error: f32,
    cancellation_cleanup_cache_len: usize,
    registry_loaded_count_after_teardown: usize,
    elapsed_ms: u128,
}

#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    let args = Args::parse()?;
    let started = Instant::now();
    if !args.candidate_dirty && args.candidate_tree != args.baseline_tree {
        return Err(Error::Msg(
            "clean candidate tree must equal the baseline HEAD tree".into(),
        ));
    }
    let fixture: Fixture =
        serde_json::from_str(&fs::read_to_string(&args.fixture)?).map_err(Error::msg)?;
    if args.prompt != fixture.prompt {
        return Err(Error::Msg(
            "prompt does not match the pinned fixture".into(),
        ));
    }
    let tokenizer = Tokenizer::from_file(&args.tokenizer).map_err(Error::msg)?;
    let encoding = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(Error::msg)?;
    let token_ids = encoding.get_ids().to_vec();
    if token_ids != fixture.token_ids {
        return Err(Error::Msg(format!(
            "tokenizer IDs differ from fixture: actual={token_ids:?}, expected={:?}",
            fixture.token_ids
        )));
    }
    if fixture.tokenizer_add_bos || fixture.tokenizer_add_eos {
        return Err(Error::Msg(
            "fixture requires the no-BOS/no-EOS GPT-OSS tokenizer policy".into(),
        ));
    }
    if fixture.reference_context > token_ids.len()
        || fixture.reference_prefix_token_ids.len() != fixture.reference_context
        || fixture.reference_prefix_token_ids != token_ids[..fixture.reference_context]
    {
        return Err(Error::Msg(
            "reference prefix does not match tokenizer IDs".into(),
        ));
    }
    if fixture.reference_context < 3
        || fixture.reference_rows != 3
        || fixture.reference_row_top1.len() != 3
    {
        return Err(Error::Msg(
            "fixture must contain exactly three reference rows".into(),
        ));
    }

    eprintln!("short-parity: input hashes verified");
    let tokenizer_hash = hash_file(&args.tokenizer)?;
    let tokenizer_config_hash = hash_file(&args.tokenizer_config)?;
    let chat_template_hash = hash_file(&args.chat_template)?;
    let reference_hash = hash_file(&args.reference_logits)?;
    let reference_metadata = fs::metadata(&args.reference_logits)?;
    if reference_metadata.len() != fixture.reference_file_bytes as u64
        || reference_hash != fixture.reference_file_sha256
    {
        return Err(Error::Msg(format!(
            "reference logits identity mismatch: bytes={}, sha256={reference_hash}",
            reference_metadata.len()
        )));
    }
    let reference_rows = read_reference_logits(
        &args.reference_logits,
        &fixture.reference_prefix_token_ids,
        fixture.reference_context,
        fixture.reference_vocab_size,
        fixture.reference_rows,
    )?;
    eprintln!("short-parity: reference logits decoded");
    for (row, expected) in reference_rows.iter().zip(&fixture.reference_row_top1) {
        let top = top_log_probabilities(row, 1)
            .into_iter()
            .next()
            .ok_or_else(|| Error::Msg("reference row is empty".into()))?;
        if top.token_id != expected.token_id as usize {
            return Err(Error::Msg(format!(
                "reference fixture top-1 mismatch: actual={} expected={} log_prob={}",
                top.token_id, expected.token_id, expected.log_prob
            )));
        }
    }

    let registry = GptOssLoadRegistry::default();
    let cancellation = GptOssCancellationToken::new();
    eprintln!("short-parity: loading GGUF weights");
    let (artifact, weights, handle) = load_gpt_oss_weights_with_cancellation(
        &args.model,
        &registry,
        &cancellation,
        args.max_weight_bytes,
    )?;
    eprintln!("short-parity: GGUF weights loaded");
    let model_hash = artifact.sha256().to_owned();
    let limits = GptOssResourceLimits::for_config(artifact.config())?;
    let cuda_config =
        GptOssCudaConfig::new(args.device_index, DType::F32, args.max_weight_bytes, limits)
            .map_err(|error| Error::Msg(error.to_string()))?;
    let mut model = GptOssCudaModel::new(artifact.config().clone(), weights, cuda_config)
        .map_err(|error| Error::Msg(error.to_string()))?;
    eprintln!("short-parity: CUDA model constructed");

    let cancelled = GptOssCancellationToken::cancel_after_checks(0);
    let cancellation_error = model
        .prefill(&fixture.reference_prefix_token_ids[..1], &cancelled)
        .expect_err("cancelled parity prefill must fail before cache mutation");
    if !cancellation_error.to_string().contains("cancelled") || model.cache_len() != 0 {
        return Err(Error::Msg(format!(
            "cancellation cleanup failed: error={cancellation_error}, cache_len={}",
            model.cache_len()
        )));
    }
    let cancellation_cleanup_cache_len = model.cache_len();
    eprintln!("short-parity: cancellation rollback verified");

    let prefix_len = fixture.reference_context - 3;
    let prefill = tensor_to_vec(
        model
            .prefill(&token_ids[..prefix_len], &GptOssCancellationToken::new())
            .map_err(|error| Error::Msg(error.to_string()))?,
    )?;
    let decode_one = tensor_to_vec(
        model
            .decode(token_ids[prefix_len], &GptOssCancellationToken::new())
            .map_err(|error| Error::Msg(error.to_string()))?,
    )?;
    let decode_two = tensor_to_vec(
        model
            .decode(token_ids[prefix_len + 1], &GptOssCancellationToken::new())
            .map_err(|error| Error::Msg(error.to_string()))?,
    )?;
    let actual_rows = [prefill, decode_one, decode_two];
    eprintln!("short-parity: prefill and decode outputs captured");
    let mut stages = Vec::new();
    for (index, (actual, reference)) in actual_rows.iter().zip(&reference_rows).enumerate() {
        let stage = if index == 0 {
            "prefill".to_string()
        } else {
            format!("decode.step.{index}")
        };
        stages.push(compare_stage(
            &stage,
            actual,
            reference,
            &fixture.comparison,
        )?);
    }

    model
        .reset_cache()
        .map_err(|error| Error::Msg(error.to_string()))?;
    let replay_prefill = tensor_to_vec(
        model
            .prefill(&token_ids[..prefix_len], &GptOssCancellationToken::new())
            .map_err(|error| Error::Msg(error.to_string()))?,
    )?;
    let replay_decode_one = tensor_to_vec(
        model
            .decode(token_ids[prefix_len], &GptOssCancellationToken::new())
            .map_err(|error| Error::Msg(error.to_string()))?,
    )?;
    let replay_decode_two = tensor_to_vec(
        model
            .decode(token_ids[prefix_len + 1], &GptOssCancellationToken::new())
            .map_err(|error| Error::Msg(error.to_string()))?,
    )?;
    let reset_replay_max_abs_error = actual_rows
        .iter()
        .zip([replay_prefill, replay_decode_one, replay_decode_two])
        .map(|(first, replay)| max_abs_error(first, &replay))
        .fold(0.0f32, f32::max);
    if reset_replay_max_abs_error > 1e-5 {
        return Err(Error::Msg(format!(
            "reset/replay mismatch: max_abs_error={reset_replay_max_abs_error}"
        )));
    }

    let device = format!("{:?}", model.device());
    drop(model);
    drop(artifact);
    drop(handle);
    let registry_loaded_count_after_teardown = registry.loaded_count()?;
    if registry_loaded_count_after_teardown != 0 {
        return Err(Error::Msg(format!(
            "registry lease leaked after model teardown: {registry_loaded_count_after_teardown}"
        )));
    }
    let receipt = Receipt {
        status: "passed",
        fixture_id: fixture.fixture_id,
        prompt: args.prompt,
        token_ids,
        model_path: args.model.display().to_string(),
        model_bytes: fs::metadata(&args.model)?.len(),
        model_sha256: model_hash,
        tokenizer_sha256: tokenizer_hash,
        tokenizer_config_sha256: tokenizer_config_hash,
        chat_template_sha256: chat_template_hash,
        baseline_commit: args.baseline_commit,
        baseline_tree: args.baseline_tree,
        candidate_tree: args.candidate_tree,
        candidate_dirty: args.candidate_dirty,
        publication_status: if args.candidate_dirty {
            "uncommitted_candidate"
        } else {
            "committed_candidate"
        },
        candle_package_version: env!("CARGO_PKG_VERSION"),
        feature: "cuda",
        device_index: args.device_index,
        device,
        reference_executable: fixture.reference_executable,
        reference_command: fixture.reference_command,
        reference_logits_sha256: reference_hash,
        reference_logits_bytes: fixture.reference_file_bytes,
        top_k_log_probability_tolerance: fixture.comparison.top_k_log_probability_abs_tolerance,
        mean_log_probability_tolerance: fixture.comparison.mean_abs_log_probability_tolerance,
        total_variation_probability_tolerance: fixture
            .comparison
            .total_variation_probability_tolerance,
        tolerance_rationale: fixture.comparison.tolerance_rationale,
        stages,
        reset_replay_max_abs_error,
        cancellation_cleanup_cache_len,
        registry_loaded_count_after_teardown,
        elapsed_ms: started.elapsed().as_millis(),
    };
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let receipt_json = serde_json::to_vec_pretty(&json!(receipt)).map_err(Error::msg)?;
    fs::write(&args.output, receipt_json)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!(receipt)).map_err(Error::msg)?
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn hash_file(path: &Path) -> Result<String> {
    let file = fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(feature = "cuda")]
fn read_reference_logits(
    path: &Path,
    expected_tokens: &[u32],
    expected_context: usize,
    expected_vocab: usize,
    expected_rows: usize,
) -> Result<Vec<Vec<f32>>> {
    let bytes = fs::read(path)?;
    let mut offset = 0usize;
    let magic = take(&bytes, &mut offset, 8)?;
    if magic != b"_logits_" {
        return Err(Error::Msg(
            "reference logits magic does not match _logits_".into(),
        ));
    }
    let context = read_i32(&bytes, &mut offset)? as usize;
    let vocab = read_i32(&bytes, &mut offset)? as usize;
    let chunks = read_i32(&bytes, &mut offset)? as usize;
    if context != expected_context || vocab != expected_vocab || chunks != 1 {
        return Err(Error::Msg(format!(
            "reference logits header mismatch: context={context} vocab={vocab} chunks={chunks}"
        )));
    }
    let token_bytes = expected_tokens
        .len()
        .checked_mul(4)
        .ok_or_else(|| Error::Msg("reference token byte count overflowed".into()))?;
    let token_data = take(&bytes, &mut offset, token_bytes)?;
    let actual_tokens = token_data
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    if actual_tokens != expected_tokens {
        return Err(Error::Msg(format!(
            "reference token IDs differ: actual={actual_tokens:?}, expected={expected_tokens:?}"
        )));
    }
    let nv = 2 * vocab.div_ceil(2) + 4;
    let row_bytes = nv
        .checked_mul(2)
        .ok_or_else(|| Error::Msg("reference row byte count overflowed".into()))?;
    let mut rows = Vec::new();
    rows.try_reserve_exact(expected_rows)
        .map_err(|error| Error::Msg(format!("reference row allocation failed: {error}")))?;
    for _ in 0..expected_rows {
        let row = take(&bytes, &mut offset, row_bytes)?;
        let scale = f32::from_le_bytes(row[0..4].try_into().unwrap());
        let minimum = f32::from_le_bytes(row[4..8].try_into().unwrap());
        let values = row[8..]
            .chunks_exact(2)
            .take(vocab)
            .map(|chunk| minimum + scale * u16::from_le_bytes(chunk.try_into().unwrap()) as f32)
            .collect::<Vec<_>>();
        rows.push(values);
    }
    if offset != bytes.len() {
        return Err(Error::Msg(format!(
            "reference logits has {} trailing bytes",
            bytes.len() - offset
        )));
    }
    Ok(rows)
}

#[cfg(feature = "cuda")]
fn take<'a>(bytes: &'a [u8], offset: &mut usize, length: usize) -> Result<&'a [u8]> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| Error::Msg("reference byte range overflowed".into()))?;
    let slice = bytes
        .get(*offset..end)
        .ok_or_else(|| Error::Msg("reference logits file is truncated".into()))?;
    *offset = end;
    Ok(slice)
}

#[cfg(feature = "cuda")]
fn read_i32(bytes: &[u8], offset: &mut usize) -> Result<i32> {
    Ok(i32::from_le_bytes(
        take(bytes, offset, 4)?.try_into().unwrap(),
    ))
}

#[cfg(feature = "cuda")]
fn tensor_to_vec(tensor: Tensor) -> Result<Vec<f32>> {
    tensor.flatten_all()?.to_vec1::<f32>()
}

#[cfg(feature = "cuda")]
fn compare_stage(
    stage: &str,
    actual: &[f32],
    expected: &[f32],
    comparison: &Comparison,
) -> Result<StageReport> {
    if actual.len() != expected.len() {
        return Err(Error::Msg(format!(
            "parity mismatch at {stage}: output length {} != {} earliest_boundary=output_logits",
            actual.len(),
            expected.len()
        )));
    }
    for value in actual.iter().chain(expected) {
        if !value.is_finite() {
            return Err(Error::Msg(format!(
                "parity mismatch at {stage}: non-finite output earliest_boundary=output_logits"
            )));
        }
    }
    let actual_log_probabilities = log_probabilities(actual)?;
    let reference_floor = expected
        .iter()
        .copied()
        .reduce(f32::min)
        .ok_or_else(|| Error::Msg("reference log-probability row is empty".into()))?;
    if !reference_floor.is_finite() {
        return Err(Error::Msg(format!(
            "parity mismatch at {stage}: reference clamp floor is non-finite earliest_boundary=output_logits"
        )));
    }
    let projected_log_probabilities = actual_log_probabilities
        .iter()
        .map(|value| value.max(reference_floor))
        .collect::<Vec<_>>();
    let mut sum = 0.0f32;
    let mut max = 0.0f32;
    let mut max_token_id = 0usize;
    for (index, (actual, expected)) in projected_log_probabilities.iter().zip(expected).enumerate()
    {
        let error = (actual - expected).abs();
        sum += error;
        if error > max {
            max = error;
            max_token_id = index;
        }
    }
    let top_reference = top_log_probabilities(expected, comparison.top_k);
    let top_candle = top_log_probabilities(&projected_log_probabilities, comparison.top_k);
    let mut top_union = BTreeSet::new();
    top_union.extend(top_reference.iter().map(|entry| entry.token_id));
    top_union.extend(top_candle.iter().map(|entry| entry.token_id));
    let top_k_max = top_union
        .iter()
        .map(|&token_id| (projected_log_probabilities[token_id] - expected[token_id]).abs())
        .fold(0.0f32, f32::max);
    let normalized_reference = log_probabilities(expected)?;
    let normalized_candle = log_probabilities(&projected_log_probabilities)?;
    let total_variation = normalized_reference
        .iter()
        .zip(&normalized_candle)
        .map(|(reference, candle)| (reference.exp() - candle.exp()).abs())
        .sum::<f32>()
        * 0.5;
    let top1_matches = top_reference.first().map(|entry| entry.token_id)
        == top_candle.first().map(|entry| entry.token_id);
    let mean = sum / actual.len() as f32;
    if !top1_matches
        || top_k_max > comparison.top_k_log_probability_abs_tolerance
        || mean > comparison.mean_abs_log_probability_tolerance
        || total_variation > comparison.total_variation_probability_tolerance
    {
        return Err(Error::Msg(format!(
            "parity mismatch at {stage}: max_abs_log_probability_error={max}, mean_abs_log_probability_error={mean}, top_k_max_abs_log_probability_error={top_k_max}, total_variation_probability_error={total_variation}, top_k_tolerance={}, mean_tolerance={}, total_variation_tolerance={}, max_error_token_id={max_token_id}, candle_log_probability={}, reference_log_probability={}, reference_floor={reference_floor}, reference_top_k={top_reference:?}, candle_top_k={top_candle:?}, earliest_boundary=output_logits",
            comparison.top_k_log_probability_abs_tolerance,
            comparison.mean_abs_log_probability_tolerance,
            comparison.total_variation_probability_tolerance,
            projected_log_probabilities[max_token_id],
            expected[max_token_id],
        )));
    }
    Ok(StageReport {
        stage: stage.to_string(),
        max_abs_log_probability_error: max,
        mean_abs_log_probability_error: mean,
        top_k_max_abs_log_probability_error: top_k_max,
        total_variation_probability_error: total_variation,
        top_reference,
        top_candle,
    })
}

#[cfg(feature = "cuda")]
fn log_probabilities(logits: &[f32]) -> Result<Vec<f32>> {
    let maximum = logits
        .iter()
        .copied()
        .reduce(f32::max)
        .ok_or_else(|| Error::Msg("cannot normalize an empty logits vector".into()))?;
    if !maximum.is_finite() {
        return Err(Error::Msg("logits contain no finite maximum".into()));
    }
    let sum_exp = logits
        .iter()
        .map(|value| (value - maximum).exp())
        .sum::<f32>();
    if !sum_exp.is_finite() || sum_exp <= 0.0 {
        return Err(Error::Msg("logits could not be normalized".into()));
    }
    let log_sum_exp = sum_exp.ln();
    logits
        .iter()
        .map(|value| {
            let normalized = value - maximum - log_sum_exp;
            if normalized.is_finite() {
                Ok(normalized)
            } else {
                Err(Error::Msg(
                    "normalized log probability is non-finite".into(),
                ))
            }
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn top_log_probabilities(values: &[f32], count: usize) -> Vec<TopLogProbability> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|left, right| {
        values[*right]
            .partial_cmp(&values[*left])
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    indices
        .into_iter()
        .take(count)
        .map(|token_id| TopLogProbability {
            token_id,
            log_probability: values[token_id],
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn max_abs_error(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max)
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::{compare_stage, log_probabilities, Comparison};

    #[test]
    fn stable_log_probability_normalization() {
        let values = log_probabilities(&[1000.0, 999.0, 998.0]).unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
        let probability_sum = values.iter().map(|value| value.exp()).sum::<f32>();
        assert!((probability_sum - 1.0).abs() < 1e-6);
        assert!(values[0] > values[1] && values[1] > values[2]);
    }

    fn comparison() -> Comparison {
        Comparison {
            top_k: 2,
            top_k_log_probability_abs_tolerance: 0.25,
            mean_abs_log_probability_tolerance: 0.1,
            total_variation_probability_tolerance: 0.1,
            tolerance_rationale: "test".to_string(),
        }
    }

    #[test]
    fn clipped_tail_outlier_does_not_mask_head_agreement() {
        let expected =
            log_probabilities(&[0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -10.0, -10.0]).unwrap();
        let actual = [0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -10.5, -10.0];
        assert!(compare_stage("test", &actual, &expected, &comparison()).is_ok());
    }

    #[test]
    fn head_drift_fails_top_k_criterion() {
        let expected =
            log_probabilities(&[0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -10.0, -10.0]).unwrap();
        let actual = [0.0, -1.4, -2.0, -3.0, -4.0, -5.0, -10.0, -10.0];
        assert!(compare_stage("test", &actual, &expected, &comparison()).is_err());
    }
}
