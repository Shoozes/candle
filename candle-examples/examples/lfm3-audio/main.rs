//! Finite native CPU LFM3-Audio TTS; no downloads or alternate inference runtime.
use anyhow::{bail, Context, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{lfm3, lfm3_audio, liquid_codec};
use clap::Parser;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::OpenOptions,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

mod asr;

#[derive(Parser, Debug)]
struct Args {
    /// Explicit local snapshot with its hash-bound asset-manifest.json.
    #[arg(long)]
    model_dir: PathBuf,
    /// Transcribe a local 16 kHz PCM16 WAV instead of synthesizing speech.
    #[arg(long)]
    input: Option<PathBuf>,
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,
    /// Compare full codec synthesis with seven-frame chunks and final-tail flush.
    #[arg(long, conflicts_with_all = ["input", "inspect"])]
    verify_codec_chunks: bool,
    /// Parse and validate released configuration without loading weights.
    #[arg(long)]
    inspect: bool,
    #[arg(long, default_value = "Hello.")]
    text: String,
    #[arg(long, default_value = "American English")]
    language: String,
    #[arg(long, default_value = "female", value_parser = ["female", "male"])]
    voice: String,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    report: Option<PathBuf>,
    #[arg(long, default_value_t = 100)]
    max_frames: usize,
    #[arg(long, default_value_t = 1024)]
    context: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 1.)]
    audio_temperature: f32,
    #[arg(long, default_value_t = 128)]
    audio_top_k: usize,
}

#[derive(Deserialize)]
struct Asset {
    path: String,
    bytes: u64,
    sha256: String,
}
#[derive(Deserialize)]
struct Manifest {
    repo: String,
    revision: String,
    files: Vec<Asset>,
}

fn hash(path: &Path) -> Result<String> {
    let mut input = std::fs::File::open(path)?;
    let mut buffer = vec![0u8; 256 * 1024];
    let mut sha = Sha256::new();
    loop {
        let n = input.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        sha.update(&buffer[..n]);
    }
    Ok(format!("{:x}", sha.finalize()))
}

fn verify_assets(root: &Path, manifest: &Manifest) -> Result<()> {
    for required in [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "audio_tokenizer/config.json",
        "audio_tokenizer/model.safetensors",
    ] {
        let matches = manifest
            .files
            .iter()
            .filter(|a| a.path == required)
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            bail!("asset manifest must identify {required} exactly once");
        }
        let a = matches[0];
        let path = root.join(required);
        let metadata = path.symlink_metadata()?;
        if !metadata.is_file()
            || metadata.file_type().is_symlink()
            || metadata.len() != a.bytes
            || hash(&path)? != a.sha256
        {
            bail!("asset identity mismatch: {required}");
        }
    }
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_reader(std::fs::File::open(path)?)?)
}

/// Official audio-processor ChatML template, with the assistant's audio marker prefilled.
fn tts_prompt(text: &str, language: &str, voice: &str) -> Result<String> {
    if text.trim().is_empty() || text.len() > 16_384 || language.is_empty() || language.len() > 128
    {
        bail!("TTS requires nonempty bounded text and language");
    }
    if text.contains("<|") || language.contains("<|") || language.contains(['\n', '\r']) {
        bail!("TTS task text/language cannot inject special-token delimiters");
    }
    Ok(format!("<|startoftext|><|im_start|>system\nPerform TTS in {language} using a {voice} voice.<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<|audio_start|>"))
}

fn write_wav(path: &Path, audio: &[f32], sample_rate: u32) -> Result<(f32, f64, usize)> {
    if audio.is_empty() || audio.iter().any(|v| !v.is_finite()) {
        bail!("refusing empty or nonfinite PCM");
    }
    let peak = audio.iter().fold(0f32, |a, b| a.max(b.abs()));
    let rms = (audio.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / audio.len() as f64).sqrt();
    let saturated = audio.iter().filter(|v| v.abs() >= 1.).count();
    if rms <= 1e-6 || saturated > 0 {
        bail!("refusing silent or saturated PCM: rms={rms}, saturated={saturated}");
    }
    let bytes = audio
        .len()
        .checked_mul(2)
        .and_then(|n| u32::try_from(n).ok())
        .context("WAV size overflow")?;
    let riff = bytes.checked_add(36).context("RIFF size overflow")?;
    let byte_rate = sample_rate.checked_mul(2).context("WAV rate overflow")?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(b"RIFF")?;
    file.write_all(&riff.to_le_bytes())?;
    file.write_all(b"WAVEfmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?;
    file.write_all(&16u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&bytes.to_le_bytes())?;
    for &v in audio {
        file.write_all(&((v * 32767.).round() as i16).to_le_bytes())?;
    }
    file.sync_all()?;
    Ok((peak, rms, saturated))
}

#[derive(Serialize)]
struct Report<'a> {
    schema: &'static str,
    repo: &'a str,
    revision: &'a str,
    backend: &'static str,
    seed: u64,
    rng: &'static str,
    prompt_ids: Vec<u32>,
    frames: Vec<Vec<u32>>,
    termination: &'static str,
    cache_positions: usize,
    sample_rate: u32,
    samples: usize,
    peak: f32,
    rms: f64,
    saturated: usize,
    elapsed_ms: u128,
    wav_sha256: String,
    codec_chunk_max_abs: Option<f32>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cfg: lfm3_audio::Config = read_json(&args.model_dir.join("config.json"))?;
    cfg.validate()?;
    let codec_cfg: liquid_codec::Config =
        read_json(&args.model_dir.join("audio_tokenizer/config.json"))?;
    codec_cfg.validate()?;
    if cfg.num_codebooks != codec_cfg.n_q || cfg.audio_eos_code()? != codec_cfg.codebook_size()? {
        bail!("audio model and codec codebook identities do not match");
    }
    if args.inspect {
        println!(
            "{}",
            serde_json::json!({"status":"config_validated","text_layers":cfg.text_config.num_hidden_layers,"depth_layers":cfg.depth_config.num_hidden_layers,"codebooks":cfg.num_codebooks,"schedule":cfg.schedule()?,"sample_rate":codec_cfg.sampling_rate,"samples_per_frame":codec_cfg.frame_samples()?,"weights_loaded":false})
        );
        return Ok(());
    }
    if args.input.is_some() {
        return asr::run(&args, &cfg);
    }
    if args.max_frames == 0 || args.max_frames > 1000 || args.context > 4096 {
        bail!("example limits: 1..=1000 audio frames and at most 4096 context positions");
    }
    let output = args
        .output
        .as_ref()
        .context("--output is required for TTS")?;
    let report = args
        .report
        .as_ref()
        .context("--report is required for TTS")?;
    if output.exists() || report.exists() || output == report {
        bail!("output/report paths must be distinct and fresh");
    }
    let sampling = lfm3_audio::Sampling {
        temperature: args.audio_temperature,
        top_k: args.audio_top_k,
        ..Default::default()
    };
    sampling.validate()?;
    let prompt = tts_prompt(&args.text, &args.language, &args.voice)?;
    let manifest: Manifest = read_json(&args.model_dir.join("asset-manifest.json"))?;
    verify_assets(&args.model_dir, &manifest)?;
    let tokenizer = tokenizers::Tokenizer::from_file(args.model_dir.join("tokenizer.json"))
        .map_err(anyhow::Error::msg)?;
    for name in [
        "<|startoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|audio_start|>",
    ] {
        if tokenizer.token_to_id(name).is_none() {
            bail!("tokenizer missing required marker {name}");
        }
    }
    let ids = tokenizer
        .encode(prompt, false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    if ids
        .len()
        .checked_add(args.max_frames)
        .is_none_or(|n| n > args.context)
    {
        bail!("prompt plus frame allowance exceeds context budget");
    }
    let started = Instant::now();
    let device = Device::Cpu;
    // Files are explicit, hash-checked local inputs. The map is read-only and owned by VarBuilder.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[args.model_dir.join("model.safetensors")],
            DType::F32,
            &device,
        )?
    };
    let model = lfm3_audio::Model::new(&cfg, vb)?;
    eprintln!("native LFM3-Audio loaded; prefill {} tokens", ids.len());
    let mut cache = lfm3::Cache::new(&cfg.text_config, args.context)?;
    let mut hidden = model.prefill(
        &Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?,
        &mut cache,
    )?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    let mut frames = vec![];
    let mut termination = "frame_limit";
    for _ in 0..args.max_frames {
        let codes = model.sample_audio_frame(&hidden, &frames, sampling, &mut rng)?;
        if codes.contains(&cfg.audio_eos_code()?) {
            termination = "audio_eos";
            break;
        }
        frames.push(codes);
        if frames.len() < args.max_frames {
            hidden = model.advance_audio_frame(
                frames.last().context("missing generated frame")?,
                &mut cache,
            )?;
        }
        eprintln!(
            "audio frames {} elapsed {:.2}s",
            frames.len(),
            started.elapsed().as_secs_f64()
        );
    }
    if frames.is_empty() {
        bail!("model produced no nonterminal speech frames");
    }
    let positions = cache.position();
    drop(hidden);
    drop(cache);
    drop(model);
    let codec_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[args.model_dir.join("audio_tokenizer/model.safetensors")],
            DType::F32,
            &device,
        )?
    };
    let codec = liquid_codec::Decoder::new(&codec_cfg, codec_vb)?;
    let pcm = codec.decode(&frames, args.max_frames)?;
    if pcm.len() != frames.len() * codec_cfg.frame_samples()? {
        bail!("codec returned an unexpected sample count");
    }
    let codec_chunk_max_abs = if args.verify_codec_chunks {
        let mut state = codec.streaming_cache(args.max_frames)?;
        let mut chunked = Vec::new();
        for frames in frames.chunks(7) {
            chunked.extend(codec.decode_chunk(frames, &mut state, false)?);
        }
        chunked.extend(codec.decode_chunk(&[], &mut state, true)?);
        if !state.is_finished()
            || state.retained_overlap_samples() != 0
            || chunked.len() != pcm.len()
            || codec.decode_chunk(&[], &mut state, true).is_ok()
        {
            bail!("codec chunk/finalization contract failed");
        }
        let max_abs = pcm
            .iter()
            .zip(chunked)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        if max_abs > 1e-4 {
            bail!("codec chunk replay exceeds the local F32 1e-4 PCM tolerance: {max_abs}");
        }
        Some(max_abs)
    } else {
        None
    };
    drop(codec);
    verify_assets(&args.model_dir, &manifest)?;
    let (peak, rms, saturated) = write_wav(output, &pcm, codec_cfg.sampling_rate)?;
    let receipt = Report {
        schema: "candle-lfm3-audio-tts.v1",
        repo: &manifest.repo,
        revision: &manifest.revision,
        backend: "native-candle-cpu-f32",
        seed: args.seed,
        rng: "rand-0.9-StdRng (not PyTorch RNG)",
        prompt_ids: ids,
        frames,
        termination,
        cache_positions: positions,
        sample_rate: codec_cfg.sampling_rate,
        samples: pcm.len(),
        peak,
        rms,
        saturated,
        elapsed_ms: started.elapsed().as_millis(),
        wav_sha256: hash(output)?,
        codec_chunk_max_abs,
    };
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(report)?;
    serde_json::to_writer_pretty(&mut file, &receipt)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    println!("{}", serde_json::to_string(&receipt)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_verification_requires_tts_execution() {
        let base = ["lfm3-audio", "--model-dir", "unused"];
        let parse = |extra: &[&str]| Args::try_parse_from(base.iter().chain(extra.iter()));
        assert!(parse(&["--verify-codec-chunks"]).is_ok());
        assert!(parse(&["--input", "unused.wav"]).is_ok());
        assert!(parse(&["--inspect"]).is_ok());
        for extra in [
            vec!["--input", "unused.wav", "--verify-codec-chunks"],
            vec!["--inspect", "--verify-codec-chunks"],
        ] {
            assert_eq!(
                parse(&extra).unwrap_err().kind(),
                clap::error::ErrorKind::ArgumentConflict
            );
        }
    }

    #[test]
    fn task_template_matches_the_official_audio_processor_envelope() -> Result<()> {
        let prompt = tts_prompt("Hello.", "American English", "female")?;
        assert_eq!(prompt, "<|startoftext|><|im_start|>system\nPerform TTS in American English using a female voice.<|im_end|>\n<|im_start|>user\nHello.<|im_end|>\n<|im_start|>assistant\n<|audio_start|>");
        assert!(tts_prompt("<|audio|>", "English", "male").is_err());
        assert!(tts_prompt("", "English", "female").is_err());
        Ok(())
    }
}
