//! Local-only loader for quantized LFM2 GGUF text plus split dense MMProj.

use anyhow::{Context, Result};
use candle::quantized::gguf_file;
use candle::{DType, Device};
use candle_transformers::models::lfm2_vl::{Mmproj, QuantizedLfm2VlModel};
use candle_transformers::models::quantized_lfm2::ModelWeights;
use candle_vlm::lfm2_vl::{Lfm2VlProcessor, Lfm2VlProcessorConfig, Lfm2VlPrompt, PromptOptions};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct LoadedHybrid {
    pub model: QuantizedLfm2VlModel,
    pub processor: Lfm2VlProcessor,
    pub prompt: Lfm2VlPrompt,
}

pub fn load_hybrid(
    text_gguf: impl AsRef<Path>,
    mmproj_dir: impl AsRef<Path>,
    tokenizer_path: impl AsRef<Path>,
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

    let mmproj = Mmproj::load(mmproj_dir, vision_dtype, vision_device)?;
    let processor_config = Lfm2VlProcessorConfig::from_mmproj_metadata(&mmproj.metadata)?;
    let tokenizer_path = tokenizer_path.as_ref();
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
    let prompt = Lfm2VlPrompt::from_processor_config(
        tokenizer,
        Some(mmproj.metadata.image_token_id),
        &processor_config,
        PromptOptions {
            use_image_special_tokens: mmproj
                .metadata
                .manifest
                .model_config
                .use_image_special_tokens,
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
