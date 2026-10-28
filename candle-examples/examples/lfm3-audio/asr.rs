use super::{hash, read_json, verify_assets, Args, Manifest};
use anyhow::{bail, Context, Result};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{lfm3, lfm3_audio};
use std::{fs::OpenOptions, io::Write, path::Path, time::Instant};

fn word(bytes: &[u8], offset: usize) -> Result<u16> {
    let end = offset.checked_add(2).context("WAV word offset overflow")?;
    let part: [u8; 2] = bytes
        .get(offset..end)
        .context("truncated WAV word")?
        .try_into()?;
    Ok(u16::from_le_bytes(part))
}
fn dword(bytes: &[u8], offset: usize) -> Result<u32> {
    let end = offset.checked_add(4).context("WAV dword offset overflow")?;
    let part: [u8; 4] = bytes
        .get(offset..end)
        .context("truncated WAV dword")?
        .try_into()?;
    Ok(u32::from_le_bytes(part))
}

/// PCM16 channels are mean-reduced; sample-rate conversion remains explicit host work.
fn read_wav(path: &Path, max_samples: usize) -> Result<Vec<f32>> {
    if std::fs::metadata(path)?.len() > 16 * 1024 * 1024 {
        bail!("WAV exceeds finite input byte limit");
    }
    let bytes = std::fs::read(path)?;
    parse_wav(&bytes, max_samples)
}

fn parse_wav(bytes: &[u8], max_samples: usize) -> Result<Vec<f32>> {
    if bytes.get(..4) != Some(b"RIFF") || bytes.get(8..12) != Some(b"WAVE") {
        bail!("expected RIFF/WAVE input");
    }
    let end = (dword(bytes, 4)? as usize)
        .checked_add(8)
        .context("WAV RIFF length overflow")?;
    if end != bytes.len() {
        bail!("WAV RIFF length mismatch");
    }
    let mut cursor = 12usize;
    let mut format = None;
    let mut data = None;
    while cursor < end {
        let chunk = bytes
            .get(cursor..cursor + 4)
            .context("truncated WAV chunk")?;
        let size = dword(bytes, cursor + 4)? as usize;
        let start = cursor.checked_add(8).context("WAV chunk offset overflow")?;
        let stop = start.checked_add(size).context("WAV chunk size overflow")?;
        let payload = bytes.get(start..stop).context("truncated WAV payload")?;
        if chunk == b"fmt " {
            if format.is_some() || size < 16 {
                bail!("duplicate or short WAV format");
            }
            let channels = word(payload, 2)? as usize;
            if word(payload, 0)? != 1
                || word(payload, 14)? != 16
                || !(1..=8).contains(&channels)
                || dword(payload, 4)? != 16000
                || word(payload, 12)? as usize != channels * 2
                || dword(payload, 8)? != 16000 * channels as u32 * 2
            {
                bail!("ASR example requires 16 kHz PCM16 WAV with 1..=8 channels");
            }
            format = Some(channels);
        } else if chunk == b"data" {
            if data.is_some() {
                bail!("multiple WAV data chunks are unsupported");
            }
            data = Some(payload);
        }
        cursor = stop.checked_add(size % 2).context("WAV padding overflow")?;
    }
    let channels = format.context("missing WAV format")?;
    let data = data.context("missing WAV data")?;
    let stride = channels * 2;
    if data.is_empty() || !data.len().is_multiple_of(stride) || data.len() / stride > max_samples {
        bail!("ASR input is empty, misaligned or exceeds the sample allowance");
    }
    let mut mono = Vec::with_capacity(data.len() / stride);
    for frame in data.chunks_exact(stride) {
        let mut sum = 0f32;
        for c in 0..channels {
            let sample = i16::from_le_bytes([frame[2 * c], frame[2 * c + 1]]);
            sum += sample as f32 / 32768.;
        }
        mono.push(sum / channels as f32);
    }
    Ok(mono)
}

pub(super) fn run(args: &Args, cfg: &lfm3_audio::Config) -> Result<()> {
    if args.max_new_tokens == 0
        || args.max_new_tokens > 512
        || args.context == 0
        || args.context > 4096
    {
        bail!("ASR example requires 1..=512 new tokens and 1..=4096 context positions");
    }
    let input = args.input.as_ref().context("ASR input missing")?;
    let report = args
        .report
        .as_ref()
        .context("--report is required for ASR")?;
    if report.exists() || args.output.is_some() {
        bail!("ASR requires a fresh report and does not write --output");
    }
    if args.language.is_empty()
        || args.language.len() > 128
        || args.language.contains(['\n', '\r'])
        || args.language.contains("<|")
    {
        bail!("invalid ASR language");
    }
    let initial_hash = hash(input)?;
    let pcm = read_wav(input, 160_000)?;
    let manifest: Manifest = read_json(&args.model_dir.join("asset-manifest.json"))?;
    verify_assets(&args.model_dir, &manifest)?;
    let raw: serde_json::Value = read_json(&args.model_dir.join("config.json"))?;
    let feature_cfg: lfm3_audio::features::Config = serde_json::from_value(
        raw.get("preprocessor")
            .context("missing preprocessor config")?
            .clone(),
    )?;
    let encoder_cfg: lfm3_audio::encoder::Config = serde_json::from_value(
        raw.get("encoder")
            .context("missing encoder config")?
            .clone(),
    )?;
    let device = Device::Cpu;
    let started = Instant::now();
    let features = lfm3_audio::features::extract(&feature_cfg, &pcm, 16000, 160_000, &device)?;
    let feature_frames = features.dim(2)?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[args.model_dir.join("model.safetensors")],
            DType::F32,
            &device,
        )?
    };
    let encoder =
        lfm3_audio::encoder::Encoder::new(&encoder_cfg, cfg.text_config.hidden_size, vb.clone())?;
    let encoded = encoder.encode(&features, 1000)?;
    let audio_tokens = encoded.dim(1)?;
    drop(encoder);
    drop(features);
    let tokenizer = tokenizers::Tokenizer::from_file(args.model_dir.join("tokenizer.json"))
        .map_err(anyhow::Error::msg)?;
    let marker = |name| {
        tokenizer
            .token_to_id(name)
            .with_context(|| format!("missing tokenizer marker {name}"))
    };
    let audio_id = marker("<|audio|>")?;
    let prefix = format!(
        "<|startoftext|><|im_start|>system\nPerform ASR in {}.<|im_end|>\n<|im_start|>user\n",
        args.language
    );
    let mut ids = tokenizer
        .encode(prefix, false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    ids.extend(std::iter::repeat_n(audio_id, audio_tokens));
    ids.extend(
        tokenizer
            .encode("<|im_end|>\n<|im_start|>assistant\n", false)
            .map_err(anyhow::Error::msg)?
            .get_ids(),
    );
    if ids
        .len()
        .checked_add(args.max_new_tokens)
        .is_none_or(|n| n > args.context)
    {
        bail!("ASR prompt and generation allowance exceed context");
    }
    let lm = lfm3::Model::new(
        &cfg.text_config,
        vb.pp("lfm.model"),
        Some(vb.pp("lfm.lm_head")),
    )?;
    let embed = lm.embed_tokens(&Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?)?;
    let merged = lfm3_audio::merge_audio_embeddings(&ids, &embed, audio_id, &encoded)?;
    let mut cache = lfm3::Cache::new(&cfg.text_config, args.context)?;
    let mut hidden = lm.backbone.forward_embeds(&merged, &mut cache)?;
    drop(encoded);
    drop(merged);
    drop(embed);
    let suppressed = [
        audio_id,
        marker("<|audio_out|>")?,
        marker("<|audio_start|>")?,
        marker("<|mixed_start|>")?,
    ];
    let stop = [marker("<|im_end|>")?, marker("<|text_end|>")?];
    let mut tokens = vec![];
    let mut termination = "token_limit";
    for _ in 0..args.max_new_tokens {
        let length = hidden.dim(1)?;
        let logits = lm
            .project_logits(&hidden.narrow(1, length - 1, 1)?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if logits.iter().any(|v| !v.is_finite()) {
            bail!("nonfinite ASR logits");
        }
        let id = logits
            .iter()
            .enumerate()
            .filter(|(i, _)| !suppressed.contains(&(*i as u32)))
            .max_by(|(a, x), (b, y)| x.total_cmp(y).then(b.cmp(a)))
            .map(|(i, _)| i as u32)
            .context("no ASR token candidates")?;
        if stop.contains(&id) {
            termination = "text_eos";
            break;
        }
        tokens.push(id);
        if tokens.len() < args.max_new_tokens {
            let input = lm.embed_tokens(&Tensor::new(&[[id]], &device)?)?;
            hidden = lm.backbone.forward_embeds(&input, &mut cache)?;
        }
    }
    let text = tokenizer
        .decode(&tokens, true)
        .map_err(anyhow::Error::msg)?;
    if text.trim().is_empty() {
        bail!("ASR produced no text");
    }
    let positions = cache.position();
    drop(hidden);
    drop(cache);
    drop(lm);
    verify_assets(&args.model_dir, &manifest)?;
    if hash(input)? != initial_hash {
        bail!("ASR audio input changed during execution");
    }
    let receipt = serde_json::json!({"schema":"candle-lfm3-audio-asr.v1","backend":"native-candle-cpu-f32","repo":manifest.repo,"revision":manifest.revision,"input_sha256":initial_hash,"input_samples":pcm.len(),"sample_rate":16000,"feature_frames":feature_frames,"audio_tokens":audio_tokens,"prompt_ids":ids,"generated_ids":tokens,"transcript":text,"termination":termination,"cache_positions":positions,"elapsed_ms":started.elapsed().as_millis()});
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
    fn wav_header_helpers_refuse_truncated_external_input() {
        assert!(word(&[1], 0).is_err());
        assert!(dword(&[1, 2, 3], 0).is_err());
        assert!(word(&[], usize::MAX).is_err());
        assert!(parse_wav(b"RIFF", 10).is_err());
    }

    #[test]
    fn pcm16_stereo_is_mean_reduced_and_wrong_rate_or_extent_is_rejected() -> Result<()> {
        let mut bytes = vec![];
        bytes.extend(b"RIFF");
        bytes.extend(44u32.to_le_bytes());
        bytes.extend(b"WAVEfmt ");
        bytes.extend(16u32.to_le_bytes());
        bytes.extend(1u16.to_le_bytes());
        bytes.extend(2u16.to_le_bytes());
        bytes.extend(16000u32.to_le_bytes());
        bytes.extend(64000u32.to_le_bytes());
        bytes.extend(4u16.to_le_bytes());
        bytes.extend(16u16.to_le_bytes());
        bytes.extend(b"data");
        bytes.extend(8u32.to_le_bytes());
        for sample in [32767i16, -32767, -32768, 0] {
            bytes.extend(sample.to_le_bytes());
        }
        assert_eq!(parse_wav(&bytes, 2)?, vec![0., -0.5]);
        assert!(parse_wav(&bytes, 1).is_err());
        let mut wrong_rate = bytes.clone();
        wrong_rate[24..28].copy_from_slice(&24000u32.to_le_bytes());
        assert!(parse_wav(&wrong_rate, 2).is_err());
        bytes.pop();
        assert!(parse_wav(&bytes, 2).is_err());
        Ok(())
    }
}
