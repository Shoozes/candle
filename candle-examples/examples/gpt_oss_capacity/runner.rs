use super::input::{self, Backend, Identity, Input, CHUNK, DECODES, FORWARDS, PREFILL, TOTAL};
use candle::{bail, Error, Result};
use candle_transformers::models::gpt_oss::{
    GptOssCancellationToken, GptOssDevice, GptOssFileIdentity, GptOssForwardTimings,
    GptOssLoadOptions, GptOssLoadRegistry, GptOssResourceLimits, GptOssResourceUsage,
    GptOssSession, GptOssSource, EDGE_DEVICE_CEILING_BYTES,
};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

const HOST_CEILING: usize = 32 * 1024 * 1024 * 1024;

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Progress {
    pub schema: String,
    pub status: String,
    pub phase: String,
    pub source_commit: String,
    pub input_sha256: String,
    pub pid: u32,
    pub executable: PathBuf,
    pub consumed_files: Vec<Identity>,
    pub token_file: Option<Identity>,
    pub loads_attempted: usize,
    pub requests_attempted: usize,
    pub requests_completed: usize,
    pub attempted_forward_calls: usize,
    pub completed_forward_calls: usize,
    pub attempted_token_steps_upper_bound: usize,
    pub completed_token_steps: usize,
    pub completed_prefill_calls: usize,
    pub completed_decode_calls: usize,
    pub sampled_output_tokens: usize,
    pub committed_sequence_tokens: usize,
    pub cache_bytes: usize,
    pub cache_capacity_bytes: usize,
    pub peak_cache_capacity_bytes: usize,
    pub static_weight_bytes: Option<usize>,
    pub load_ms: Option<f64>,
    pub unload_ms: Option<f64>,
    pub prefill_forward_ms: f64,
    pub decode_forward_ms: f64,
    pub prefill_validation_ms: Option<f64>,
    pub decode_validation_ms: Option<f64>,
    pub registry_entries_after_release: Option<usize>,
    pub total_ms: f64,
    pub error: Option<String>,
}

fn add(counter: &mut usize, n: usize) -> Result<()> {
    *counter = counter
        .checked_add(n)
        .ok_or_else(|| Error::Msg("capacity work counter overflow".into()))?;
    Ok(())
}

pub(super) fn walk(
    tokens: &[u32],
    prefill: usize,
    chunk: usize,
    cancel: &GptOssCancellationToken,
    progress: &mut Progress,
    mut forward: impl FnMut(&[u32], bool) -> Result<(GptOssResourceUsage, GptOssForwardTimings)>,
    mut observe: impl FnMut(&Progress) -> Result<()>,
) -> Result<()> {
    if tokens.is_empty()
        || tokens.len() > TOTAL
        || prefill == 0
        || chunk == 0
        || !prefill.is_multiple_of(chunk)
        || prefill >= tokens.len()
    {
        bail!("capacity requires positive aligned prefill and nonempty fixed decode");
    }
    let prompts = tokens
        .get(..prefill)
        .ok_or_else(|| Error::Msg("capacity prefill is out of bounds".into()))?;
    let decode = tokens
        .get(prefill..)
        .ok_or_else(|| Error::Msg("capacity decode is out of bounds".into()))?;
    let spans = prompts
        .chunks(chunk)
        .map(|x| (x, true))
        .chain(decode.iter().map(|x| (std::slice::from_ref(x), false)));
    for (span, is_prefill) in spans {
        if cancel.is_cancelled() {
            bail!("capacity work cancelled");
        }
        progress.phase = if is_prefill {
            "prefill"
        } else {
            "fixed-decode"
        }
        .into();
        add(&mut progress.attempted_forward_calls, 1)?;
        add(&mut progress.attempted_token_steps_upper_bound, span.len())?;
        observe(progress)?;
        let (usage, timing) = forward(span, is_prefill)?;
        add(&mut progress.completed_forward_calls, 1)?;
        add(&mut progress.completed_token_steps, span.len())?;
        progress.committed_sequence_tokens = usage.sequence_tokens;
        progress.cache_bytes = usage.cache_bytes;
        progress.cache_capacity_bytes = usage.cache_capacity_bytes;
        progress.peak_cache_capacity_bytes = progress
            .peak_cache_capacity_bytes
            .max(usage.cache_capacity_bytes);
        if usage.sequence_tokens != progress.completed_token_steps
            || usage.cache_capacity_bytes < usage.cache_bytes
        {
            bail!("capacity committed cache/work count mismatch");
        }
        if !timing.forward_ms.is_finite()
            || timing.forward_ms < 0.
            || timing
                .cuda_final_validation_ms
                .is_some_and(|t| !t.is_finite() || t < 0. || t > timing.forward_ms + 0.002)
        {
            bail!("capacity forward returned malformed timing");
        }
        let (calls, wall, validation) = if is_prefill {
            (
                &mut progress.completed_prefill_calls,
                &mut progress.prefill_forward_ms,
                &mut progress.prefill_validation_ms,
            )
        } else {
            (
                &mut progress.completed_decode_calls,
                &mut progress.decode_forward_ms,
                &mut progress.decode_validation_ms,
            )
        };
        add(calls, 1)?;
        *wall += timing.forward_ms;
        if let Some(ms) = timing.cuda_final_validation_ms {
            *validation = Some(validation.unwrap_or(0.) + ms);
        }
        if !wall.is_finite() || validation.is_some_and(|ms| !ms.is_finite()) {
            bail!("capacity aggregate timing overflow");
        }
        observe(progress)?;
    }
    Ok(())
}

fn write_progress(path: &Path, p: &Progress, create: bool) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(create)
        .truncate(!create)
        .open(path)?;
    file.write_all(&serde_json::to_vec(p).map_err(Error::wrap)?)?;
    file.flush()?;
    Ok(())
}

fn run(input: &Input, tokens: &[u32], p: &mut Progress, path: &Path) -> Result<()> {
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
        max_weight_bytes: budget,
        limits: GptOssResourceLimits {
            max_sequence_tokens: TOTAL,
            max_cache_bytes: budget,
            max_total_device_bytes: budget,
        },
    };
    p.phase = "loading".into();
    p.loads_attempted = 1;
    write_progress(path, p, false)?;
    let load = Instant::now();
    let loaded = GptOssSession::load(&options, &registry, &cancel);
    p.load_ms = Some(load.elapsed().as_secs_f64() * 1000.);
    let result = loaded.and_then(|mut session| {
        p.consumed_files = session
            .consumed_files()
            .into_iter()
            .map(|x| Identity {
                path: x.path,
                size_bytes: x.size_bytes,
                sha256: x.sha256,
            })
            .collect();
        p.static_weight_bytes = Some(session.static_weight_bytes());
        p.requests_attempted = 1;
        let result = walk(
            tokens,
            PREFILL,
            CHUNK,
            &cancel,
            p,
            |span, prefill| {
                let (logits, timing) = if prefill {
                    session.prefill_profiled(span, &cancel)?
                } else {
                    session.decode_profiled(
                        *span
                            .first()
                            .ok_or_else(|| Error::Msg("missing fixed decode".into()))?,
                        &cancel,
                    )?
                };
                drop(logits);
                Ok((session.resource_usage()?, timing))
            },
            |state| write_progress(path, state, false),
        );
        if result.is_ok() {
            p.requests_completed = 1;
        }
        p.phase = "unloading".into();
        let drop_start = Instant::now();
        drop(session);
        p.unload_ms = Some(drop_start.elapsed().as_secs_f64() * 1000.);
        result
    });
    p.registry_entries_after_release = Some(registry.loaded_count()?);
    if p.registry_entries_after_release != Some(0) {
        bail!("capacity registry did not release");
    }
    result
}

pub(crate) fn execute(
    input: &Input,
    input_hash: String,
    tokens: &[u32],
    output: &Path,
) -> Result<()> {
    let mut p = Progress {
        schema: "candle.gpt-oss.capacity-progress.v1".into(),
        status: "running".into(),
        phase: "starting".into(),
        source_commit: input.source_commit.clone(),
        input_sha256: input_hash,
        pid: std::process::id(),
        executable: std::env::current_exe()?,
        token_file: Some(input.token_file.clone()),
        ..Progress::default()
    };
    write_progress(output, &p, true)?;
    let start = Instant::now();
    let result = run(input, tokens, &mut p, output);
    p.total_ms = start.elapsed().as_secs_f64() * 1000.;
    p.status = if result.is_ok() {
        "completed"
    } else {
        "failed"
    }
    .into();
    p.error = result.as_ref().err().map(ToString::to_string);
    write_progress(output, &p, false)?;
    result
}

pub(crate) fn verify_progress(path: &Path, input: &Input, input_hash: &str) -> Result<()> {
    let p: Progress =
        serde_json::from_slice(&input::read_bounded(path, 16384)?).map_err(Error::wrap)?;
    if p.pid == 0
        || p.executable.as_os_str().is_empty()
        || p.schema != "candle.gpt-oss.capacity-progress.v1"
        || p.status != "completed"
        || p.error.is_some()
        || p.source_commit != input.source_commit
        || p.input_sha256 != input_hash
        || p.loads_attempted != 1
        || p.requests_attempted != 1
        || p.requests_completed != 1
        || p.attempted_forward_calls != FORWARDS
        || p.completed_forward_calls != FORWARDS
        || p.attempted_token_steps_upper_bound != TOTAL
        || p.completed_token_steps != TOTAL
        || p.completed_prefill_calls != PREFILL / CHUNK
        || p.completed_decode_calls != DECODES
        || p.sampled_output_tokens != 0
        || p.committed_sequence_tokens != TOTAL
        || p.registry_entries_after_release != Some(0)
        || p.consumed_files.len() != 1
    {
        bail!("capacity progress has incomplete or mismatched work/source/release");
    }
    let consumed = p
        .consumed_files
        .first()
        .ok_or_else(|| Error::Msg("missing consumed model".into()))?;
    let token_file = p
        .token_file
        .as_ref()
        .ok_or_else(|| Error::Msg("missing token identity".into()))?;
    if !same_model_path(&consumed.path, &input.model.path)
        || consumed.sha256 != input.model.sha256
        || consumed.size_bytes != input.model.size_bytes
        || token_file.path != input.token_file.path
        || token_file.sha256 != input.token_file.sha256
        || token_file.size_bytes != input.token_file.size_bytes
        || p.cache_bytes == 0
        || p.cache_capacity_bytes < p.cache_bytes
        || p.peak_cache_capacity_bytes < p.cache_capacity_bytes
    {
        bail!("capacity storage/input identity mismatch");
    }
    for timing in [
        p.load_ms,
        p.unload_ms,
        Some(p.prefill_forward_ms),
        Some(p.decode_forward_ms),
        Some(p.total_ms),
    ] {
        if timing.is_none_or(|ms| !ms.is_finite() || ms < 0.) {
            bail!("capacity timing is missing or invalid");
        }
    }
    let budget = if matches!(input.backend, Backend::Cuda) {
        EDGE_DEVICE_CEILING_BYTES
    } else {
        HOST_CEILING
    };
    if p.static_weight_bytes.is_none_or(|n| n == 0 || n > budget)
        || p.peak_cache_capacity_bytes > budget
    {
        bail!("capacity payload exceeds selected ceiling");
    }
    for (wall, validation) in [
        (p.prefill_forward_ms, p.prefill_validation_ms),
        (p.decode_forward_ms, p.decode_validation_ms),
    ] {
        match input.backend {
            Backend::Cpu if validation.is_some() => {
                bail!("CPU capacity must not report CUDA validation")
            }
            Backend::Cuda
                if validation.is_none_or(|ms| {
                    !ms.is_finite() || ms < 0. || ms > wall + (FORWARDS as f64 * 0.002)
                }) =>
            {
                bail!("CUDA capacity validation timing mismatch")
            }
            _ => (),
        }
    }
    let observed = p.load_ms.unwrap_or(0.)
        + p.unload_ms.unwrap_or(0.)
        + p.prefill_forward_ms
        + p.decode_forward_ms;
    if !observed.is_finite() || observed > p.total_ms + (FORWARDS as f64 * 0.002) {
        bail!("capacity aggregate timing exceeds total");
    }
    Ok(())
}

fn same_model_path(actual: &Path, expected: &Path) -> bool {
    #[cfg(windows)]
    {
        let actual = actual.to_string_lossy();
        let expected = expected.to_string_lossy();
        actual
            .strip_prefix(r"\\?\")
            .unwrap_or(&actual)
            .eq_ignore_ascii_case(expected.strip_prefix(r"\\?\").unwrap_or(&expected))
    }
    #[cfg(not(windows))]
    {
        actual == expected
    }
}
