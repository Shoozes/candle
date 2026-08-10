//! Local-only loader for quantized LFM2 GGUF text plus split or GGUF MMProj.

use anyhow::{bail, Context, Result};
use candle::quantized::gguf_file;
use candle::{DType, Device};
use candle_transformers::models::lfm2_vl::{Mmproj, QuantizedLfm2VlModel};
use candle_transformers::models::quantized_lfm2::ModelWeights;
use candle_vlm::lfm2_vl::{
    Lfm2VlProcessor, Lfm2VlProcessorConfig, Lfm2VlPrompt, Lfm2VlSpecialTokens,
    ProcessorConfigPatch, PromptOptions,
};
use std::io::Read;
use std::path::Path;
use tokenizers::Tokenizer;

const MAX_PROCESSOR_CONFIG_BYTES: u64 = 16 * 1024 * 1024;

pub enum MmprojInput<'a> {
    SplitDirectory(&'a Path),
    GgufFile(&'a Path),
}

pub struct LoadedHybrid {
    pub model: QuantizedLfm2VlModel,
    pub processor: Lfm2VlProcessor,
    pub prompt: Lfm2VlPrompt,
}

pub fn load_hybrid(
    text_gguf: impl AsRef<Path>,
    mmproj_input: MmprojInput<'_>,
    tokenizer_path: impl AsRef<Path>,
    processor_config_path: Option<&Path>,
    vision_dtype: DType,
    vision_device: &Device,
    text_device: &Device,
) -> Result<LoadedHybrid> {
    let text_gguf = text_gguf.as_ref();
    let mut text_file = std::fs::File::open(text_gguf)
        .with_context(|| format!("opening text GGUF {}", text_gguf.display()))?;
    let content =
        gguf_file::Content::read(&mut text_file).map_err(|err| err.with_path(text_gguf))?;
    let text = ModelWeights::from_gguf(content, &mut text_file, text_device)?;

    let tokenizer_path = tokenizer_path.as_ref();
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
    let tokenizer_image_token_id =
        Lfm2VlSpecialTokens::resolve(&tokenizer, None, 0, false)?.image_token_id;

    let mmproj = match mmproj_input {
        MmprojInput::SplitDirectory(path) => Mmproj::load(path, vision_dtype, vision_device)?,
        MmprojInput::GgufFile(path) => {
            Mmproj::load_gguf_auto(path, vision_dtype, vision_device, tokenizer_image_token_id)?
        }
    };
    let processor_patch = processor_config_path
        .map(|path| {
            let json = read_processor_config(path)?;
            ProcessorConfigPatch::from_json(&json).map_err(anyhow::Error::msg)
        })
        .transpose()?;
    let processor_config = Lfm2VlProcessorConfig::from_mmproj_metadata_with_processor(
        &mmproj.metadata,
        processor_patch.as_ref(),
    )?;
    let prompt = Lfm2VlPrompt::from_processor_config(
        tokenizer,
        Some(mmproj.metadata.image_token_id),
        &processor_config,
        PromptOptions {
            use_image_special_tokens: mmproj.metadata.use_image_special_tokens,
            context_length: Some(text.metadata().context_length),
        },
    )?;
    let image_token_id = prompt.special_tokens().image_token_id;
    let model = QuantizedLfm2VlModel::new(
        text,
        mmproj,
        processor_config.encoder_patch_size,
        processor_config.downsample_factor,
        image_token_id,
    )?;
    let processor = Lfm2VlProcessor::from_config(&processor_config)?;
    Ok(LoadedHybrid {
        model,
        processor,
        prompt,
    })
}

fn read_processor_config(path: &Path) -> Result<String> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("opening processor config {}", path.display()))?;
    let declared = file
        .metadata()
        .with_context(|| format!("reading processor config metadata {}", path.display()))?
        .len();
    if declared > MAX_PROCESSOR_CONFIG_BYTES {
        bail!(
            "processor config {} is {declared} bytes, exceeding {MAX_PROCESSOR_CONFIG_BYTES}",
            path.display()
        )
    }
    let capacity =
        usize::try_from(declared).context("processor config length does not fit usize")?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| anyhow::anyhow!("processor config allocation failed"))?;
    file.take(MAX_PROCESSOR_CONFIG_BYTES + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("reading processor config {}", path.display()))?;
    if bytes.len() as u64 > MAX_PROCESSOR_CONFIG_BYTES {
        bail!(
            "processor config {} grew beyond {MAX_PROCESSOR_CONFIG_BYTES} bytes while reading",
            path.display()
        )
    }
    String::from_utf8(bytes)
        .with_context(|| format!("processor config {} is not UTF-8", path.display()))
}
