//! Bounded raw-token characterization of the public GPT-OSS F32 session.
//! Production execution needs a fresh owner-approved source/binary-bound packet.

use candle::{bail, Error, Result};
use candle_transformers::models::gpt_oss::{
    GptOssCancellationToken, GptOssDevice, GptOssFileIdentity, GptOssLoadOptions,
    GptOssLoadRegistry, GptOssLogits, GptOssResourceLimits, GptOssSession, GptOssSource,
    EDGE_DEVICE_CEILING_BYTES,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const HOST_CEILING: usize = 32 * 1024 * 1024 * 1024;

fn frozen_prompt() -> Result<Vec<u32>> {
    #[derive(Deserialize)]
    struct Fixture {
        reference_prefix_token_ids: Vec<u32>,
    }
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../tests/fixtures/gpt_oss_task3_short_parity/reference.json"
    ))
    .map_err(Error::wrap)?;
    Ok(fixture.reference_prefix_token_ids)
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum Backend {
    Cpu,
    Cuda,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelIdentity {
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
    model: ModelIdentity,
    prompt_token_ids: Vec<u32>,
    generated_tokens: usize,
}

impl Input {
    fn validate(&self) -> Result<usize> {
        if self.schema != "candle.gpt-oss.session-profile-input.v1"
            || self.source_commit.len() != 40
            || !self.source_commit.bytes().all(|b| b.is_ascii_hexdigit())
        {
            bail!("profile input requires its version and exact source commit");
        }
        // This first profiling gate deliberately stays small; larger studies
        // need a separately reviewed harness and fresh finite allowance.
        if self.prompt_token_ids.len() != 8 || self.generated_tokens != 16 {
            bail!("initial session profiling requires exactly 8 prompt and 16 output tokens");
        }
        if self.prompt_token_ids != frozen_prompt()? {
            bail!("initial profiling prompt differs from the pinned reference prefix");
        }
        if self.model.size_bytes == 0
            || self.model.sha256.len() != 64
            || !self.model.sha256.bytes().all(|b| b.is_ascii_hexdigit())
        {
            bail!("profile input requires an explicit model size and SHA-256");
        }
        if matches!(self.backend, Backend::Cpu) && self.device_index != 0 {
            bail!("CPU profiling requires device_index=0");
        }
        #[cfg(not(feature = "cuda"))]
        if matches!(self.backend, Backend::Cuda) {
            bail!("CUDA profiling requires the cuda feature");
        }
        self.prompt_token_ids
            .len()
            .checked_add(self.generated_tokens)
            .ok_or_else(|| Error::Msg("profile context count overflowed".into()))
    }
}

#[derive(Default, Serialize)]
struct Call {
    input_tokens: usize,
    forward_ms: f64,
    cuda_final_validation_ms: Option<f64>,
    caller_readback_ms: f64,
    argmax_ms: f64,
    committed_sequence_tokens: usize,
}

#[derive(Serialize)]
struct ConsumedFile {
    path: PathBuf,
    size_bytes: u64,
    sha256: String,
}

#[derive(Default, Serialize)]
struct Progress {
    schema: String,
    status: String,
    phase: String,
    source_commit: String,
    input_sha256: String,
    pid: u32,
    executable: PathBuf,
    consumed_files: Vec<ConsumedFile>,
    loads_attempted: usize,
    requests_attempted: usize,
    requests_completed: usize,
    attempted_forward_calls: usize,
    attempted_token_steps_upper_bound: usize,
    completed_forward_calls: usize,
    completed_token_steps: usize,
    generated_token_ids: Vec<u32>,
    calls: Vec<Call>,
    load_ms: Option<f64>,
    static_weight_bytes: Option<usize>,
    unload_ms: Option<f64>,
    registry_entries_after_release: Option<usize>,
    total_ms: f64,
    error: Option<String>,
}

fn argmax(values: &[f32]) -> Result<u32> {
    let mut best = None;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            bail!("profile sampling rejects non-finite logits");
        }
        if best.is_none_or(|(_, previous)| value > previous) {
            best = Some((index, value));
        }
    }
    let (index, _) = best.ok_or_else(|| Error::Msg("profile received no logits".into()))?;
    u32::try_from(index).map_err(|_| Error::Msg("profile token index overflowed".into()))
}

fn write_json(path: &Path, value: &Progress, create: bool) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).map_err(Error::wrap)?;
    let mut options = fs::OpenOptions::new();
    options.write(true);
    if create {
        options.create_new(true);
    } else {
        options.truncate(true);
    }
    let mut file = options.open(path)?;
    file.write_all(&bytes)?;
    file.flush()?;
    Ok(())
}

fn consume_logits(logits: GptOssLogits, call: &mut Call, vocab: usize) -> Result<u32> {
    let start = Instant::now();
    let values = logits.into_cpu()?;
    call.caller_readback_ms = start.elapsed().as_secs_f64() * 1000.;
    if values.len() != vocab {
        bail!(
            "profile logit row has {} values, expected {vocab}",
            values.len()
        );
    }
    let start = Instant::now();
    let token = argmax(&values)?;
    call.argmax_ms = start.elapsed().as_secs_f64() * 1000.;
    Ok(token)
}

fn run(input: &Input, progress: &mut Progress, path: &Path) -> Result<()> {
    let total_tokens = input.validate()?;
    let registry = GptOssLoadRegistry::default();
    let cancel = GptOssCancellationToken::new();
    let device = match input.backend {
        Backend::Cpu => GptOssDevice::Cpu,
        Backend::Cuda => GptOssDevice::Cuda {
            device_index: input.device_index,
        },
    };
    let budget = match input.backend {
        Backend::Cpu => HOST_CEILING,
        Backend::Cuda => EDGE_DEVICE_CEILING_BYTES,
    };
    let options = GptOssLoadOptions {
        source: GptOssSource::Gguf(GptOssFileIdentity {
            path: input.model.path.clone(),
            size_bytes: input.model.size_bytes,
            sha256: input.model.sha256.clone(),
        }),
        device,
        limits: GptOssResourceLimits {
            max_sequence_tokens: total_tokens,
            max_cache_bytes: budget,
            max_total_device_bytes: budget,
        },
        max_weight_bytes: budget,
    };
    progress.phase = "loading".into();
    progress.loads_attempted = 1;
    write_json(path, progress, false)?;
    let start = Instant::now();
    let loaded = GptOssSession::load(&options, &registry, &cancel);
    progress.load_ms = Some(start.elapsed().as_secs_f64() * 1000.);
    let result = loaded.and_then(|mut session| {
        progress.consumed_files = session
            .consumed_files()
            .into_iter()
            .map(|file| ConsumedFile {
                path: file.path,
                size_bytes: file.size_bytes,
                sha256: file.sha256,
            })
            .collect();
        progress.static_weight_bytes = Some(session.static_weight_bytes());
        progress.requests_attempted = 1;
        let mut token = None;
        for step in 0..input.generated_tokens {
            let incoming = if step == 0 {
                input.prompt_token_ids.len()
            } else {
                1
            };
            progress.phase = if step == 0 { "prefill" } else { "decode" }.into();
            progress.attempted_forward_calls += 1;
            progress.attempted_token_steps_upper_bound += incoming;
            write_json(path, progress, false)?;
            let (logits, timings) = if step == 0 {
                session.prefill_profiled(&input.prompt_token_ids, &cancel)?
            } else {
                session.decode_profiled(
                    token.ok_or_else(|| Error::Msg("profile has no prior token".into()))?,
                    &cancel,
                )?
            };
            progress.completed_forward_calls += 1;
            progress.completed_token_steps += incoming;
            let mut call = Call {
                input_tokens: incoming,
                forward_ms: timings.forward_ms,
                cuda_final_validation_ms: timings.cuda_final_validation_ms,
                committed_sequence_tokens: session.resource_usage()?.sequence_tokens,
                ..Call::default()
            };
            // Count the committed forward even if caller readback or sampling fails.
            let sampled = consume_logits(logits, &mut call, session.config().vocab_size);
            progress.calls.push(call);
            token = Some(sampled?);
            progress
                .generated_token_ids
                .push(token.ok_or_else(|| Error::Msg("profile sampling lost token".into()))?);
            write_json(path, progress, false)?;
        }
        progress.requests_completed = 1;
        progress.phase = "unloading".into();
        let start = Instant::now();
        drop(session);
        progress.unload_ms = Some(start.elapsed().as_secs_f64() * 1000.);
        Ok(())
    });
    progress.registry_entries_after_release = Some(registry.loaded_count()?);
    if progress.registry_entries_after_release != Some(0) {
        bail!("profile load registry retained an entry after release");
    }
    result
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if cfg!(debug_assertions) {
        anyhow::bail!("session profiling requires an optimized release build");
    }
    let mut bytes = Vec::new();
    fs::File::open(&args.input)?
        .take(16 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 16 * 1024 {
        anyhow::bail!("profile input exceeds 16 KiB");
    }
    let input: Input = serde_json::from_slice(&bytes)?;
    input.validate()?;
    let mut progress = Progress {
        schema: "candle.gpt-oss.session-profile-progress.v1".into(),
        status: "running".into(),
        phase: "starting".into(),
        source_commit: input.source_commit.clone(),
        input_sha256: format!("{:x}", Sha256::digest(&bytes)),
        pid: std::process::id(),
        executable: std::env::current_exe()?,
        ..Progress::default()
    };
    write_json(&args.output, &progress, true)?;
    let start = Instant::now();
    let result = run(&input, &mut progress, &args.output);
    progress.total_ms = start.elapsed().as_secs_f64() * 1000.;
    progress.status = if result.is_ok() {
        "completed"
    } else {
        "failed"
    }
    .into();
    progress.error = result.as_ref().err().map(ToString::to_string);
    write_json(&args.output, &progress, false)?;
    result.map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> Input {
        Input {
            schema: "candle.gpt-oss.session-profile-input.v1".into(),
            source_commit: "a".repeat(40),
            backend: Backend::Cpu,
            device_index: 0,
            model: ModelIdentity {
                path: "unused.gguf".into(),
                size_bytes: 1,
                sha256: "b".repeat(64),
            },
            prompt_token_ids: frozen_prompt().unwrap(),
            generated_tokens: 16,
        }
    }

    #[test]
    fn frozen_allowance_admits_only_selected_workload() -> Result<()> {
        let mut value = input();
        assert_eq!(value.validate()?, 24);
        value.generated_tokens = 17;
        assert!(value.validate().is_err());
        value.generated_tokens = 16;
        value.prompt_token_ids.pop();
        assert!(value.validate().is_err());
        value.prompt_token_ids = frozen_prompt()?;
        value.prompt_token_ids.swap(0, 1);
        assert!(value.validate().is_err());
        Ok(())
    }

    #[test]
    fn identities_and_cpu_device_validate_before_loading() {
        let mut value = input();
        value.device_index = 1;
        assert!(value.validate().is_err());
        value.device_index = 0;
        value.model.sha256 = "z".repeat(64);
        assert!(value.validate().is_err());
    }

    #[test]
    fn sampling_preserves_first_tie_and_rejects_invalid_rows() -> Result<()> {
        assert_eq!(argmax(&[1., 2., 2.])?, 1);
        assert!(argmax(&[]).is_err());
        assert!(argmax(&[1., f32::NAN]).is_err());
        assert!(argmax(&[f32::INFINITY]).is_err());
        Ok(())
    }
}
