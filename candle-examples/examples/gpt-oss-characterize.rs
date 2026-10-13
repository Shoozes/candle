//! Versioned, finite common-session timing/oracle/replay characterization.
//! Model execution needs a fresh clean-source/binary-bound owner allowance.
#[path = "gpt_oss_reference/mod.rs"]
#[allow(dead_code)] // The shared helper also serves the historical parity entrypoint.
mod reference;

use candle::{bail, Error, Result};
use candle_transformers::models::gpt_oss::{
    GptOssCancellationToken, GptOssDevice, GptOssFileIdentity, GptOssLoadOptions,
    GptOssLoadRegistry, GptOssResourceLimits, GptOssSession, GptOssSource,
    EDGE_DEVICE_CEILING_BYTES,
};
use clap::Parser;
use reference::{CapturedRow, Fixture, StageReport};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

const HOST_CEILING: usize = 32 * 1024 * 1024 * 1024;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    input: Option<PathBuf>,
    #[arg(long)]
    output: Option<PathBuf>,
    /// Model-free revalidation of captured rows, independent oracle and replay.
    #[arg(long)]
    verify_progress: Option<PathBuf>,
    /// Model-free codec/context/top-1 validation for a restored or regenerated oracle.
    #[arg(long)]
    inspect_reference: Option<PathBuf>,
}
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum Backend {
    Cpu,
    Cuda,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Identity {
    path: PathBuf,
    size_bytes: u64,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema: String,
    source_commit: String,
    backend: Backend,
    device_index: usize,
    model: Identity,
    reference: Identity,
    prompt_token_ids: Vec<u32>,
    generated_tokens: usize,
    segments: Vec<String>,
}
impl Input {
    fn validate(&self, f: &Fixture) -> Result<()> {
        if self.schema != "candle.gpt-oss.characterization-input.v1"
            || self.source_commit.len() != 40
            || !self.source_commit.bytes().all(|x| x.is_ascii_hexdigit())
        {
            bail!("characterization input version/source mismatch");
        }
        if self.prompt_token_ids != f.reference_prefix_token_ids
            || self.generated_tokens != 16
            || self.segments != ["cold", "numerical", "replay"]
        {
            bail!("characterization differs from frozen cold/oracle/replay workload");
        }
        if self.model.size_bytes != 13_792_638_656
            || self.model.sha256
                != "aab205256a9b6361e410c24de3086e30f907092ca6f9ba8cd4b22c8a2b025778"
        {
            bail!("characterization model identity is invalid");
        }
        if self.reference.size_bytes != f.reference_file_bytes as u64
            || self.reference.sha256 != f.reference_file_sha256
        {
            bail!("characterization reference identity is not the pinned oracle");
        }
        if matches!(self.backend, Backend::Cpu) && self.device_index != 0 {
            bail!("CPU requires device_index=0");
        }
        Ok(())
    }
}
#[derive(Default, Deserialize, Serialize)]
struct Call {
    segment: String,
    input_tokens: usize,
    committed_sequence_tokens: usize,
    forward_ms: f64,
    cuda_final_validation_ms: Option<f64>,
    caller_readback_ms: f64,
    argmax_ms: f64,
    row_capture_ms: f64,
    comparison_ms: f64,
}
#[derive(Deserialize, Serialize)]
struct Consumed {
    path: PathBuf,
    size_bytes: u64,
    sha256: String,
}
#[derive(Default, Deserialize, Serialize)]
struct Progress {
    schema: String,
    status: String,
    phase: String,
    source_commit: String,
    input_sha256: String,
    pid: u32,
    executable: PathBuf,
    consumed_files: Vec<Consumed>,
    loads_attempted: usize,
    requests_attempted: usize,
    requests_completed: usize,
    attempted_forward_calls: usize,
    completed_forward_calls: usize,
    attempted_token_steps_upper_bound: usize,
    completed_token_steps: usize,
    generated_token_ids: Vec<u32>,
    calls: Vec<Call>,
    rows: Vec<CapturedRow>,
    independent_comparisons: Vec<StageReport>,
    replay_max_abs_errors: Vec<f32>,
    reset_ms: Vec<f64>,
    load_ms: Option<f64>,
    static_weight_bytes: Option<usize>,
    cold_ms: Option<f64>,
    numerical_ms: Option<f64>,
    replay_ms: Option<f64>,
    unload_ms: Option<f64>,
    registry_entries_after_release: Option<usize>,
    total_ms: f64,
    error: Option<String>,
}
fn write_progress(path: &Path, p: &Progress, create: bool) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(create)
        .truncate(!create)
        .open(path)?;
    file.write_all(&serde_json::to_vec_pretty(p).map_err(Error::wrap)?)?;
    file.flush()?;
    Ok(())
}
fn read_input(path: &Path) -> Result<(Input, String)> {
    let mut bytes = Vec::new();
    fs::File::open(path)?.take(16385).read_to_end(&mut bytes)?;
    if bytes.len() > 16384 {
        bail!("characterization input exceeds 16 KiB");
    }
    Ok((
        serde_json::from_slice(&bytes).map_err(Error::wrap)?,
        format!("{:x}", Sha256::digest(&bytes)),
    ))
}
fn argmax(values: &[f32]) -> Result<u32> {
    let mut best = None;
    for (i, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            bail!("nonfinite characterization logits");
        }
        if best.is_none_or(|(_, old)| value > old) {
            best = Some((i, value));
        }
    }
    u32::try_from(
        best.ok_or_else(|| Error::Msg("missing characterization logits".into()))?
            .0,
    )
    .map_err(Error::wrap)
}
fn forward(
    session: &mut GptOssSession,
    tokens: &[u32],
    segment: &str,
    p: &mut Progress,
    path: &Path,
    cancel: &GptOssCancellationToken,
    prefill: bool,
) -> Result<Vec<f32>> {
    p.attempted_forward_calls += 1;
    p.attempted_token_steps_upper_bound += tokens.len();
    p.phase = segment.into();
    write_progress(path, p, false)?;
    let (logits, timing) = if prefill {
        session.prefill_profiled(tokens, cancel)?
    } else {
        session.decode_profiled(
            *tokens
                .first()
                .ok_or_else(|| Error::Msg("missing decode token".into()))?,
            cancel,
        )?
    };
    p.completed_forward_calls += 1;
    p.completed_token_steps += tokens.len();
    p.calls.push(Call {
        segment: segment.into(),
        input_tokens: tokens.len(),
        committed_sequence_tokens: session.resource_usage()?.sequence_tokens,
        forward_ms: timing.forward_ms,
        cuda_final_validation_ms: timing.cuda_final_validation_ms,
        ..Call::default()
    });
    let call = p
        .calls
        .last_mut()
        .ok_or_else(|| Error::Msg("missing completed call".into()))?;
    let start = Instant::now();
    let values = logits.into_cpu();
    call.caller_readback_ms = start.elapsed().as_secs_f64() * 1000.;
    let values = values?;
    if values.len() != session.config().vocab_size {
        bail!("characterization vocabulary length mismatch");
    }
    if values.iter().any(|x| !x.is_finite()) {
        bail!("nonfinite characterization logits");
    }
    Ok(values)
}
fn fixed_segment(
    session: &mut GptOssSession,
    input: &Input,
    f: &Fixture,
    reference_rows: (&[Vec<f32>], Option<&[Vec<f32>]>),
    p: &mut Progress,
    path: &Path,
    cancel: &GptOssCancellationToken,
) -> Result<Vec<Vec<f32>>> {
    let (oracle, prior) = reference_rows;
    let reset = Instant::now();
    session.reset_cache()?;
    p.reset_ms.push(reset.elapsed().as_secs_f64() * 1000.);
    if session.resource_usage()?.sequence_tokens != 0 {
        bail!("reset did not clear common-session cache");
    }
    let segment = if prior.is_some() {
        "replay"
    } else {
        "numerical"
    };
    let start = Instant::now();
    p.requests_attempted += 1;
    let prefixes = reference::fixed_prefixes(f)?;
    let mut rows = Vec::new();
    for (index, prefix) in prefixes.iter().enumerate() {
        let tokens = if index == 0 {
            prefix.as_slice()
        } else {
            prefix
                .get(prefix.len() - 1..)
                .ok_or_else(|| Error::Msg("missing fixed decode token".into()))?
        };
        let values = forward(session, tokens, segment, p, path, cancel, index == 0)?;
        if session.resource_usage()?.sequence_tokens != prefix.len() {
            bail!("numerical row cache position mismatch");
        }
        let call = p
            .calls
            .last_mut()
            .ok_or_else(|| Error::Msg("missing numerical call".into()))?;
        let capture = Instant::now();
        p.rows.push(CapturedRow::write(
            path.parent()
                .ok_or_else(|| Error::Msg("output has no parent".into()))?,
            &values,
            prefix,
            segment,
            index,
            &input.source_commit,
            &input.model.sha256,
        )?);
        call.row_capture_ms = capture.elapsed().as_secs_f64() * 1000.;
        let comparison = Instant::now();
        if let Some(prior) = prior {
            p.replay_max_abs_errors
                .push(reference::compare_replay(&prior[index], &values)?);
        } else {
            p.independent_comparisons.push(reference::compare_stage(
                &format!("context.{}", prefix.len()),
                &values,
                &oracle[index],
                &f.comparison,
            )?);
        }
        call.comparison_ms = comparison.elapsed().as_secs_f64() * 1000.;
        rows.push(values);
        write_progress(path, p, false)?;
    }
    p.requests_completed += 1;
    if prior.is_some() {
        p.replay_ms = Some(start.elapsed().as_secs_f64() * 1000.);
    } else {
        p.numerical_ms = Some(start.elapsed().as_secs_f64() * 1000.);
    }
    Ok(rows)
}
fn run(input: &Input, f: &Fixture, p: &mut Progress, path: &Path) -> Result<()> {
    // Oracle admission precedes weight loading and any state mutation.
    let oracle = reference::load_reference(&input.reference.path, f)?;
    let registry = GptOssLoadRegistry::default();
    let cancel = GptOssCancellationToken::new();
    let (device, budget) = match input.backend {
        Backend::Cpu => (GptOssDevice::Cpu, HOST_CEILING),
        Backend::Cuda => (
            GptOssDevice::Cuda {
                device_index: input.device_index,
            },
            EDGE_DEVICE_CEILING_BYTES,
        ),
    };
    let options = GptOssLoadOptions {
        source: GptOssSource::Gguf(GptOssFileIdentity {
            path: input.model.path.clone(),
            size_bytes: input.model.size_bytes,
            sha256: input.model.sha256.clone(),
        }),
        device,
        limits: GptOssResourceLimits {
            max_sequence_tokens: 24,
            max_cache_bytes: budget,
            max_total_device_bytes: budget,
        },
        max_weight_bytes: budget,
    };
    p.phase = "loading".into();
    p.loads_attempted = 1;
    write_progress(path, p, false)?;
    let start = Instant::now();
    let loaded = GptOssSession::load(&options, &registry, &cancel);
    p.load_ms = Some(start.elapsed().as_secs_f64() * 1000.);
    let result = loaded.and_then(|mut session| {
        p.consumed_files = session
            .consumed_files()
            .into_iter()
            .map(|x| Consumed {
                path: x.path,
                size_bytes: x.size_bytes,
                sha256: x.sha256,
            })
            .collect();
        p.static_weight_bytes = Some(session.static_weight_bytes());
        let result = (|| {
            if session.config().vocab_size != f.reference_vocab_size {
                bail!("model vocabulary differs from pinned oracle");
            }
            p.requests_attempted += 1;
            let cold = Instant::now();
            let mut token = None;
            for step in 0..16 {
                let decode;
                let tokens = if step == 0 {
                    input.prompt_token_ids.as_slice()
                } else {
                    decode = [
                        token.ok_or_else(|| Error::Msg("missing previous cold output".into()))?
                    ];
                    decode.as_slice()
                };
                let values = forward(&mut session, tokens, "cold", p, path, &cancel, step == 0)?;
                let sampling = Instant::now();
                let sampled = argmax(&values)?;
                p.calls
                    .last_mut()
                    .ok_or_else(|| Error::Msg("missing cold call".into()))?
                    .argmax_ms = sampling.elapsed().as_secs_f64() * 1000.;
                p.generated_token_ids.push(sampled);
                token = Some(sampled);
                write_progress(path, p, false)?;
            }
            p.cold_ms = Some(cold.elapsed().as_secs_f64() * 1000.);
            p.requests_completed += 1;
            let first = fixed_segment(&mut session, input, f, (&oracle, None), p, path, &cancel)?;
            fixed_segment(
                &mut session,
                input,
                f,
                (&oracle, Some(&first)),
                p,
                path,
                &cancel,
            )?;
            Ok(())
        })();
        p.phase = "unloading".into();
        let drop_start = Instant::now();
        drop(session);
        p.unload_ms = Some(drop_start.elapsed().as_secs_f64() * 1000.);
        result
    });
    p.registry_entries_after_release = Some(registry.loaded_count()?);
    if p.registry_entries_after_release != Some(0) {
        bail!("characterization registry did not release");
    }
    result
}
fn verify_artifacts(
    p: &Progress,
    input: &Input,
    input_hash: &str,
    path: &Path,
    f: &Fixture,
) -> Result<()> {
    if p.schema != "candle.gpt-oss.characterization-progress.v1"
        || p.status != "completed"
        || p.error.is_some()
        || p.source_commit != input.source_commit
        || p.input_sha256 != input_hash
        || p.loads_attempted != 1
        || p.requests_attempted != 3
        || p.requests_completed != 3
        || p.attempted_forward_calls != 22
        || p.completed_forward_calls != 22
        || p.attempted_token_steps_upper_bound != 37
        || p.completed_token_steps != 37
        || p.generated_token_ids.len() != 16
        || p.calls.len() != 22
        || p.rows.len() != 6
        || p.independent_comparisons.len() != 3
        || p.replay_max_abs_errors.len() != 3
        || p.reset_ms.len() != 2
        || p.registry_entries_after_release != Some(0)
        || p.consumed_files.len() != 1
        || p.consumed_files[0].sha256 != input.model.sha256
        || p.consumed_files[0].size_bytes != input.model.size_bytes
    {
        bail!("characterization progress is incomplete or has wrong work/source identity");
    }
    let oracle = reference::load_reference(&input.reference.path, f)?;
    let prefixes = reference::fixed_prefixes(f)?;
    let directory = path
        .parent()
        .ok_or_else(|| Error::Msg("progress has no parent".into()))?;
    for (index, prefix) in prefixes.iter().enumerate() {
        let first = p.rows[index].read_bound(
            directory,
            prefix,
            "numerical",
            index,
            f.reference_vocab_size,
            (&input.source_commit, &input.model.sha256),
        )?;
        let comparison = reference::compare_stage(
            &format!("context.{}", prefix.len()),
            &first,
            &oracle[index],
            &f.comparison,
        )?;
        if comparison != p.independent_comparisons[index] {
            bail!("reported independent comparison differs from captured rows");
        }
        let replay = p.rows[index + 3].read_bound(
            directory,
            prefix,
            "replay",
            index,
            f.reference_vocab_size,
            (&input.source_commit, &input.model.sha256),
        )?;
        if reference::compare_replay(&first, &replay)? != p.replay_max_abs_errors[index] {
            bail!("reported replay error differs from captured rows");
        }
    }
    Ok(())
}
fn main() -> Result<()> {
    let args = Args::parse();
    let f = reference::pinned_fixture()?;
    if let Some(path) = args.inspect_reference {
        let rows = reference::read_reference_logits(
            &path,
            &f.reference_prefix_token_ids,
            f.reference_context,
            f.reference_vocab_size,
            f.reference_rows,
        )?;
        let top = rows
            .iter()
            .map(|row| {
                reference::top_log_probabilities(row, 1)
                    .into_iter()
                    .next()
                    .ok_or_else(|| Error::Msg("reference row is empty".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        for (actual, expected) in top.iter().zip(&f.reference_row_top1) {
            if actual.token_id != expected.token_id as usize {
                bail!("regenerated reference top-1/prefix mismatch");
            }
        }
        println!("{}", serde_json::to_string(&serde_json::json!({"status":"verified-reference-structure",
            "rows":f.reference_rows,"vocabulary_size":f.reference_vocab_size,
            "prefix_token_ids":f.reference_prefix_token_ids,"prediction_contexts":[5,6,7],"top1":top})).map_err(Error::wrap)?);
        return Ok(());
    }
    let (input, input_hash) = read_input(
        &args
            .input
            .ok_or_else(|| Error::Msg("--input is required".into()))?,
    )?;
    input.validate(&f)?;
    if let Some(path) = args.verify_progress {
        let mut bytes = Vec::new();
        fs::File::open(&path)?
            .take(131073)
            .read_to_end(&mut bytes)?;
        if bytes.len() > 131072 {
            bail!("characterization progress exceeds 128 KiB");
        }
        let p: Progress = serde_json::from_slice(&bytes).map_err(Error::wrap)?;
        verify_artifacts(&p, &input, &input_hash, &path, &f)?;
        println!("{{\"status\":\"verified\",\"rows\":6,\"oracle_comparisons\":3,\"replay_comparisons\":3}}");
        return Ok(());
    }
    if cfg!(debug_assertions) {
        bail!("characterization requires an optimized release build");
    }
    require_execution_backend(input.backend)?;
    let path = args
        .output
        .ok_or_else(|| Error::Msg("--output is required for execution".into()))?;
    let mut p = Progress {
        schema: "candle.gpt-oss.characterization-progress.v1".into(),
        status: "running".into(),
        phase: "starting".into(),
        source_commit: input.source_commit.clone(),
        input_sha256: input_hash,
        pid: std::process::id(),
        executable: std::env::current_exe()?,
        ..Progress::default()
    };
    write_progress(&path, &p, true)?;
    let total = Instant::now();
    let result = run(&input, &f, &mut p, &path);
    p.total_ms = total.elapsed().as_secs_f64() * 1000.;
    p.status = if result.is_ok() {
        "completed"
    } else {
        "failed"
    }
    .into();
    p.error = result.as_ref().err().map(ToString::to_string);
    write_progress(&path, &p, false)?;
    result
}
fn require_execution_backend(_backend: Backend) -> Result<()> {
    #[cfg(not(feature = "cuda"))]
    if matches!(_backend, Backend::Cuda) {
        bail!("CUDA characterization requires cuda feature");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cpu_verifier_accepts_cuda_metadata_without_enabling_execution() -> Result<()> {
        let f = reference::pinned_fixture()?;
        let mut input = Input {
            schema: "candle.gpt-oss.characterization-input.v1".into(),
            source_commit: "a".repeat(40),
            backend: Backend::Cuda,
            device_index: 0,
            model: Identity {
                path: "never-opened.gguf".into(),
                size_bytes: 13_792_638_656,
                sha256: "aab205256a9b6361e410c24de3086e30f907092ca6f9ba8cd4b22c8a2b025778".into(),
            },
            reference: Identity {
                path: "never-opened.logits".into(),
                size_bytes: f.reference_file_bytes as u64,
                sha256: f.reference_file_sha256.clone(),
            },
            prompt_token_ids: f.reference_prefix_token_ids.clone(),
            generated_tokens: 16,
            segments: vec!["cold".into(), "numerical".into(), "replay".into()],
        };
        input.validate(&f)?;
        #[cfg(not(feature = "cuda"))]
        assert!(require_execution_backend(input.backend).is_err());
        input.model.sha256 = "0".repeat(64);
        assert!(input.validate(&f).is_err());
        require_execution_backend(Backend::Cpu)?;
        Ok(())
    }
    #[test]
    fn frozen_work_and_contexts() -> Result<()> {
        let f = reference::pinned_fixture()?;
        let prefixes = reference::fixed_prefixes(&f)?;
        let cold_steps = f.reference_prefix_token_ids.len() + 15;
        let fixed_steps = prefixes
            .first()
            .ok_or_else(|| Error::Msg("missing prefix".into()))?
            .len()
            + prefixes.len()
            - 1;
        assert_eq!(cold_steps + 2 * fixed_steps, 37);
        assert_eq!(16 + 2 * prefixes.len(), 22);
        assert_eq!(prefixes.iter().map(Vec::len).collect::<Vec<_>>(), [5, 6, 7]);
        Ok(())
    }
    #[test]
    fn raw_capture_binds_prefix_position_source_shape_and_digest() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "candle-characterization-row-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(Error::wrap)?
                .as_nanos()
        ));
        fs::create_dir(&root)?;
        let mut row = CapturedRow::write(
            &root,
            &[1., 2., 3.],
            &[4, 5],
            "numerical",
            0,
            "source",
            "model",
        )?;
        assert_eq!(
            row.read_bound(&root, &[4, 5], "numerical", 0, 3, ("source", "model"))?,
            [1., 2., 3.]
        );
        assert!(row
            .read_bound(&root, &[5, 4], "numerical", 0, 3, ("source", "model"))
            .is_err());
        assert!(row
            .read_bound(&root, &[4, 5], "numerical", 1, 3, ("source", "model"))
            .is_err());
        assert!(row
            .read_bound(&root, &[4, 5], "numerical", 0, 2, ("source", "model"))
            .is_err());
        row.sha256 = "bad".into();
        assert!(row
            .read_bound(&root, &[4, 5], "numerical", 0, 3, ("source", "model"))
            .is_err());
        fs::write(root.join(&row.file), [0_u8; 4])?;
        assert!(row
            .read_bound(&root, &[4, 5], "numerical", 0, 3, ("source", "model"))
            .is_err());
        fs::remove_dir_all(root)?;
        Ok(())
    }
}
