//! Offline teacher-forced F32 LFM3 text probe.
use anyhow::{bail, Context, Result};
use candle::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::lfm3::{Cache, Config, Model};
use clap::Parser;
use serde::Serialize;
use std::{collections::HashMap, fs, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    /// Local standalone LFM3 configuration.
    #[arg(long)]
    config: PathBuf,
    /// Local safetensors shard. Repeat for a sharded checkpoint.
    #[arg(long, required = true)]
    weights: Vec<PathBuf>,
    /// Frozen input token IDs in their original order.
    #[arg(long, value_delimiter = ',', required = true)]
    tokens: Vec<u32>,
    /// Vocabulary columns to report at every input position.
    #[arg(long, value_delimiter = ',', required = true)]
    probes: Vec<u32>,
    #[arg(long)]
    context: usize,
    #[arg(long)]
    chunk: usize,
    /// CUDA ordinal. Omit for CPU.
    #[arg(long)]
    cuda: Option<usize>,
}

#[derive(Serialize)]
struct Report {
    device: String,
    dtype: &'static str,
    context: usize,
    chunk: usize,
    tokens: Vec<u32>,
    probes: Vec<u32>,
    top_ids: Vec<u32>,
    probe_logits: Vec<Vec<f32>>,
    load_ms: u128,
    forward_ms: u128,
}

fn run(args: Args) -> Result<Report> {
    let config: Config = serde_json::from_slice(&fs::read(&args.config)?)
        .with_context(|| format!("read LFM3 config {}", args.config.display()))?;
    config.validate()?;
    if !(1..=4096.min(config.max_position_embeddings)).contains(&args.context)
        || !(1..=128.min(args.context)).contains(&args.chunk)
        || !(1..=args.context).contains(&args.tokens.len())
        || !(1..=64).contains(&args.probes.len())
        || args
            .tokens
            .iter()
            .chain(&args.probes)
            .any(|&id| id as usize >= config.vocab_size)
    {
        bail!("invalid token IDs, probe IDs, context, or chunk limit");
    }
    let (device, device_label) = match args.cuda {
        Some(ordinal) => (Device::new_cuda(ordinal)?, format!("cuda:{ordinal}")),
        None => (Device::Cpu, "cpu".to_owned()),
    };
    let started = Instant::now();
    let mut tensors = HashMap::new();
    for path in &args.weights {
        let shard = candle::safetensors::load(path, &device)
            .with_context(|| format!("load local shard {}", path.display()))?;
        for (name, tensor) in shard {
            if tensors.contains_key(&name) {
                bail!("duplicate tensor {name}");
            }
            tensors.insert(name, tensor);
        }
    }
    let builder = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let head = (!config.tie_word_embeddings).then(|| builder.pp("lm_head"));
    let model = Model::new(&config, builder.pp("model"), head)?;
    device.synchronize()?;
    // Model owns its F32 tensors. Release the source BF16 map before prefill.
    drop(builder);
    let load_ms = started.elapsed().as_millis();

    let started = Instant::now();
    let mut cache = Cache::new(&config, args.context)?;
    let columns = Tensor::from_slice(&args.probes, args.probes.len(), &device)?;
    let mut top_ids = Vec::with_capacity(args.tokens.len());
    let mut probe_logits = Vec::with_capacity(args.tokens.len());
    for chunk in args.tokens.chunks(args.chunk) {
        let input = Tensor::from_slice(chunk, (1, chunk.len()), &device)?;
        let logits = model.forward(&input, &mut cache)?.squeeze(0)?;
        let values = logits.index_select(&columns, 1)?.to_vec2::<f32>()?;
        if values.iter().flatten().any(|value| !value.is_finite()) {
            bail!(
                "non-finite probe logit at chunk ending at position {}",
                cache.position()
            );
        }
        probe_logits.extend(values);
        top_ids.extend(logits.argmax(D::Minus1)?.to_vec1::<u32>()?);
    }
    device.synchronize()?;
    Ok(Report {
        device: device_label,
        dtype: "f32",
        context: args.context,
        chunk: args.chunk,
        tokens: args.tokens,
        probes: args.probes,
        top_ids,
        probe_logits,
        load_ms,
        forward_ms: started.elapsed().as_millis(),
    })
}

fn main() -> Result<()> {
    println!("{}", serde_json::to_string(&run(Args::parse())?)?);
    Ok(())
}
