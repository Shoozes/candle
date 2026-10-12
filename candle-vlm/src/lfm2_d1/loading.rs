use super::{D1Limits, D1PolicyV1, D1Session};
use crate::lfm2_vl::{
    load_lfm2_vl_hybrid_with_embedding_mode, Lfm2EmbeddingLoadMode, Lfm2VlHybridLoadOptions,
    Lfm2VlMmprojExecution,
};
use candle::quantized::gguf_file;
use candle::{DType, Device, Result};
use candle_transformers::models::quantized_lfm2::ModelWeights;
use std::{fs::File, path::Path};

pub struct D1LoadOptions<'a> {
    pub hybrid: Lfm2VlHybridLoadOptions<'a>,
    pub limits: D1Limits,
}

/// Explicit local language inputs. No processor or vision weights are consumed.
pub struct D1TextLoadOptions<'a> {
    pub text_gguf: &'a Path,
    pub tokenizer: &'a Path,
    pub text_device: &'a Device,
    pub limits: D1Limits,
}

pub fn load_d1_q8(options: D1LoadOptions<'_>) -> Result<D1Session> {
    load_d1_q8_with_embedding_mode(options, Lfm2EmbeddingLoadMode::DenseF32)
}

/// Opt in to retained Q8 token rows without changing decision policy or defaults.
pub fn load_d1_q8_with_embedding_mode(
    options: D1LoadOptions<'_>,
    embedding_mode: Lfm2EmbeddingLoadMode,
) -> Result<D1Session> {
    validate_limits(&options.limits)?;
    validate_devices(options.hybrid.text_device, options.hybrid.vision_device)?;
    validate_environment()?;
    if options.hybrid.mmproj_execution != Lfm2VlMmprojExecution::Q8
        || options.hybrid.vision_dtype != DType::F32
    {
        candle::bail!("d1 loading requires explicit Q8 MMProj execution and F32 activations")
    }
    D1Session::from_hybrid(
        load_lfm2_vl_hybrid_with_embedding_mode(options.hybrid, embedding_mode)?,
        D1PolicyV1,
        options.limits,
    )
}

/// Load a text-only native Q8 session with the existing DenseF32 lookup default.
pub fn load_d1_text_q8(options: D1TextLoadOptions<'_>) -> Result<D1Session> {
    load_d1_text_q8_with_embedding_mode(options, Lfm2EmbeddingLoadMode::DenseF32)
}

pub fn load_d1_text_q8_with_embedding_mode(
    options: D1TextLoadOptions<'_>,
    embedding_mode: Lfm2EmbeddingLoadMode,
) -> Result<D1Session> {
    validate_limits(&options.limits)?;
    validate_devices(options.text_device, options.text_device)?;
    validate_environment()?;
    let tokenizer = crate::lfm2_vl::loading::load_tokenizer(options.tokenizer)?;
    let tokenizer = prepare_tokenizer(tokenizer)?;
    let mut file = File::open(options.text_gguf).map_err(|error| {
        candle::Error::Msg(format!(
            "cannot open text GGUF {}: {error}",
            options.text_gguf.display()
        ))
    })?;
    let content =
        gguf_file::Content::read(&mut file).map_err(|error| error.with_path(options.text_gguf))?;
    let model = ModelWeights::from_gguf_with_embedding_mode(
        content,
        &mut file,
        options.text_device,
        embedding_mode,
    )?;
    D1Session::from_text(
        model,
        tokenizer,
        options.limits,
        vec![
            options.text_gguf.to_path_buf(),
            options.tokenizer.to_path_buf(),
        ],
    )
}

fn validate_environment() -> Result<()> {
    for name in ["CANDLE_DEQUANTIZE_ALL", "CANDLE_DEQUANTIZE_ALL_F16"] {
        if std::env::var_os(name).is_some_and(|value| !value.is_empty() && value != "0") {
            candle::bail!("d1 native Q8 loading rejects {name}")
        }
    }
    Ok(())
}

pub(super) fn validate_limits(limits: &D1Limits) -> Result<()> {
    if limits.max_questions == 0
        || limits.max_options == 0
        || limits.max_prompt_bytes == 0
        || limits.max_context_tokens == Some(0)
    {
        candle::bail!("d1 limits must be positive")
    }
    Ok(())
}

pub(super) fn validate_devices(text: &Device, vision: &Device) -> Result<()> {
    for device in [text, vision] {
        if !device.is_cpu() && !device.is_cuda() {
            candle::bail!("d1 supports CPU and CUDA devices")
        }
    }
    Ok(())
}

pub(super) fn prepare_tokenizer(
    mut tokenizer: tokenizers::Tokenizer,
) -> Result<tokenizers::Tokenizer> {
    tokenizer
        .with_truncation(None)
        .map_err(|error| candle::Error::Msg(error.to_string()))?;
    tokenizer.with_padding(None);
    for marker in ["<|im_start|>", "<|im_end|>"] {
        let token_id = tokenizer
            .token_to_id(marker)
            .ok_or_else(|| candle::Error::Msg(format!("d1 tokenizer is missing {marker}")))?;
        let ids = tokenizer
            .encode(marker, false)
            .map_err(|error| candle::Error::Msg(error.to_string()))?;
        if ids.get_ids() != [token_id] {
            candle::bail!("d1 tokenizer marker {marker} must be atomic")
        }
    }
    Ok(tokenizer)
}
