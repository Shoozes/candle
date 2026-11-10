//! Local-only native/GGUF LFM2-MoE inference. No automatic asset downloads.
use anyhow::{bail, Context, Result};
use candle::quantized::{gguf_file, tokenizer::TokenizerFromGguf};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{lfm2_moe, quantized_lfm2_moe};
use clap::{ArgGroup, Parser};
use std::{
    collections::BTreeSet,
    fs::File,
    path::{Component, Path, PathBuf},
};
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(group(ArgGroup::new("model").required(true).args(["gguf", "native_dir"])))]
struct Args {
    #[arg(long)]
    gguf: Option<PathBuf>,
    #[arg(long)]
    native_dir: Option<PathBuf>,
    #[arg(long)]
    inspect: bool,
    #[arg(long)]
    tokenize_only: bool,
    #[arg(long, conflicts_with = "prompt_file")]
    prompt: Option<String>,
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    #[arg(long)]
    cuda: Option<usize>,
    #[arg(long, default_value = "f32")]
    dtype: String,
    #[arg(long, default_value_t = 12288)]
    context: usize,
    #[arg(long, default_value_t = 64)]
    max_new_tokens: usize,
    #[arg(long)]
    teacher_forced_ids: Option<PathBuf>,
    #[arg(long)]
    logit_probe_ids: Option<PathBuf>,
}

enum Backend {
    Native(lfm2_moe::Model, lfm2_moe::Cache),
    Gguf(quantized_lfm2_moe::ModelWeights),
}
enum Source {
    Gguf(gguf_file::Content, File),
    Native(lfm2_moe::Config, Vec<PathBuf>),
}
impl Backend {
    fn forward(&mut self, ids: &Tensor, position: usize) -> candle::Result<Tensor> {
        match self {
            Self::Native(model, cache) => model.forward(ids, position, cache),
            Self::Gguf(model) => model.forward(ids, position),
        }
    }
}

fn native_files(root: &Path) -> Result<Vec<PathBuf>> {
    let canonical = root.canonicalize()?;
    let index = root.join("model.safetensors.index.json");
    let names = if index.is_file() {
        let value: serde_json::Value = serde_json::from_reader(File::open(index)?)?;
        value
            .get("weight_map")
            .and_then(|v| v.as_object())
            .context("missing native weight_map")?
            .values()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .context("invalid shard filename")
            })
            .collect::<Result<BTreeSet<_>>>()?
    } else {
        BTreeSet::from(["model.safetensors".to_owned()])
    };
    if names.is_empty() {
        bail!("empty native shard index")
    }
    names
        .into_iter()
        .map(|name| {
            let relative = Path::new(&name);
            if relative
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
            {
                bail!("native shard path escapes model directory")
            }
            let path = root.join(relative).canonicalize()?;
            if !path.starts_with(&canonical) {
                bail!("native shard redirects outside model directory")
            }
            Ok(path)
        })
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.context == 0 || args.max_new_tokens == 0 || args.max_new_tokens > 2048 {
        bail!("require positive context and 1..=2048 output tokens")
    }
    let dtype = match args.dtype.as_str() {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        _ => bail!("dtype must be f32, f16 or bf16"),
    };
    let prompt = if let Some(path) = &args.prompt_file {
        std::fs::read_to_string(path)?
    } else {
        args.prompt.clone().unwrap_or_default()
    };
    let source;
    let tokenizer;
    let eos;
    let limit;
    let vocab;
    if let Some(path) = &args.gguf {
        let mut file = File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let metadata = quantized_lfm2_moe::inspect_gguf_metadata(&content)?;
        vocab = metadata.vocab_size;
        limit = args.context.min(metadata.context_length);
        if args.inspect {
            let memory = metadata.memory_estimate(limit, limit.min(128))?;
            println!(
                "{}",
                serde_json::json!({"architecture":metadata.architecture,"vocab_size":metadata.vocab_size,
                "context_capacity":metadata.context_length,"selected_context":limit,"tensor_bytes":metadata.tensor_bytes,
                "experts":metadata.expert_count,"topk":metadata.expert_used_count,"estimated_peak_bytes":memory.peak_bytes,
                "weight_payload_loaded":false})
            );
            return Ok(());
        }
        tokenizer = Tokenizer::from_gguf(&content)?;
        eos = metadata.eos_token_id();
        source = Source::Gguf(content, file);
    } else {
        let root = args
            .native_dir
            .as_deref()
            .context("missing native model directory")?;
        let mut cfg: lfm2_moe::Config =
            serde_json::from_reader(File::open(root.join("config.json"))?)?;
        cfg.validate()?;
        vocab = cfg.vocab_size;
        limit = args.context.min(cfg.max_position_embeddings);
        if args.inspect {
            println!(
                "{}",
                serde_json::json!({"architecture":"lfm2_moe","vocab_size":cfg.vocab_size,
                "context_capacity":cfg.max_position_embeddings,"experts":cfg.num_experts,"topk":cfg.num_experts_per_tok,
                "weight_payload_loaded":false})
            );
            return Ok(());
        }
        cfg.max_position_embeddings = limit;
        tokenizer =
            Tokenizer::from_file(root.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        eos = cfg.eos_token_id;
        source = Source::Native(cfg, native_files(root)?);
    }
    if prompt.is_empty() {
        bail!("provide a prompt or a UTF-8 prompt file, already chat-templated when required")
    }
    let input = tokenizer
        .encode(prompt.as_str(), false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    if args.tokenize_only {
        println!(
            "{}",
            serde_json::json!({"rendered":prompt,"token_ids":input,"weight_payload_loaded":false})
        );
        return Ok(());
    }
    let forced: Option<Vec<u32>> = args
        .teacher_forced_ids
        .as_ref()
        .map(|p| serde_json::from_reader(File::open(p)?).map_err(anyhow::Error::from))
        .transpose()?;
    let count = forced.as_ref().map_or(args.max_new_tokens, Vec::len);
    let probe_ids: Vec<u32> = args
        .logit_probe_ids
        .as_ref()
        .map(|p| serde_json::from_reader(File::open(p)?).map_err(anyhow::Error::from))
        .transpose()?
        .unwrap_or_default();
    if probe_ids.len() > 64
        || input
            .iter()
            .chain(forced.iter().flatten())
            .chain(probe_ids.iter())
            .any(|&id| id as usize >= vocab)
    {
        bail!("token or probe ID outside vocabulary, or more than 64 probes")
    }
    if count == 0
        || count > 2048
        || input.is_empty()
        || input.len().checked_add(count).is_none_or(|n| n > limit)
    {
        bail!("prompt plus output reservation exceeds selected context or output bound")
    }
    let device = match args.cuda {
        Some(index) => Device::new_cuda(index)?,
        None => Device::Cpu,
    };
    let mut model = match source {
        Source::Gguf(content, mut file) => {
            if dtype != DType::F32 {
                bail!("GGUF LFM2-MoE uses F32 activations")
            }
            Backend::Gguf(quantized_lfm2_moe::ModelWeights::from_gguf(
                content, &mut file, &device,
            )?)
        }
        Source::Native(cfg, files) => {
            // Immutable local caller-owned files; the VarBuilder retains its mappings.
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
            Backend::Native(
                lfm2_moe::Model::new(&cfg, vb)?,
                lfm2_moe::Cache::new(true, dtype, &cfg, &device)?,
            )
        }
    };
    let mut logits = None;
    for position in (0..input.len()).step_by(128) {
        let length = (input.len() - position).min(128);
        let chunk = input
            .get(position..position + length)
            .context("invalid prefill chunk")?;
        logits = Some(model.forward(&Tensor::new(chunk, &device)?.unsqueeze(0)?, position)?);
    }
    let mut logits = logits.context("empty prefill")?;
    let mut generated = Vec::new();
    let mut probes = Vec::new();
    let mut probe_logits = Vec::new();
    for step in 0..count {
        let next = logits
            .argmax(1)?
            .to_vec1::<u32>()?
            .into_iter()
            .next()
            .context("empty logits")?;
        probes.push(next);
        if !probe_ids.is_empty() {
            probe_logits.push(
                logits
                    .squeeze(0)?
                    .index_select(&Tensor::new(probe_ids.as_slice(), &device)?, 0)?
                    .to_vec1::<f32>()?,
            );
        }
        let token = match &forced {
            Some(ids) => *ids.get(step).context("missing forced token")?,
            None => next,
        };
        generated.push(token);
        if forced.is_none() && Some(token) == eos {
            break;
        }
        if step + 1 < count {
            logits = model.forward(&Tensor::new(&[[token]], &device)?, input.len() + step)?;
        }
    }
    let text = tokenizer
        .decode(&generated, false)
        .map_err(anyhow::Error::msg)?;
    println!(
        "{}",
        serde_json::json!({"input_tokens":input.len(),"input_token_ids":if forced.is_some() { Some(&input) } else { None },"output_tokens":generated.len(),
        "teacher_forced":forced.is_some(),"probe_top1":probes,"probe_token_ids":probe_ids,"probe_logits":probe_logits,"tokens":generated,"text":text})
    );
    Ok(())
}
