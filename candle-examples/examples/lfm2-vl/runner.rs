//! Deterministic, evidence-producing LFM2-VL inference.

use anyhow::{bail, Context, Result};
use candle::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::lfm2;
use candle_transformers::models::lfm2_vl::{
    CropKind, EncodedImages, ImageTokenSpan, Lfm2VlDecodeTrace, Lfm2VlImageTrace, Lfm2VlModel,
    Lfm2VlPrefillTrace, ProcessedVisionBatch, QuantizedLfm2VlModel, VisionLimits,
};
use candle_vlm::lfm2_vl::{ExpandedPrompt, Lfm2VlProcessor, Lfm2VlPrompt};
use image::{DynamicImage, GenericImageView, ImageReader};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;

use super::trace::{self, NativeTraceCapture};

const CONTRACT: &str = "candle-lfm2-vl-inference-v1";
const MAX_COMPRESSED_IMAGE_BYTES: u64 = 256 * 1024 * 1024;
const MAX_REPORTED_VOCAB: usize = 1 << 20;
const TOP_K: usize = 5;
const EOS_CANDIDATES: &[&str] = &[
    "</s>",
    "<|im_end|>",
    "<|eot_id|>",
    "<|end|>",
    "<|end_of_text|>",
    "<|endoftext|>",
];

pub struct InferenceRequest<'a> {
    pub backend: &'a str,
    pub model_inputs: &'a [PathBuf],
    pub prompt: &'a str,
    pub image_paths: &'a [PathBuf],
    pub max_new_tokens: usize,
    pub vision_batch_size: usize,
    pub eos_token_id: Option<u32>,
    pub timings: bool,
    pub trace_output: Option<&'a Path>,
}

#[derive(Debug, Serialize)]
pub struct InferenceReport {
    pub contract: &'static str,
    pub backend: String,
    pub model_inputs: Vec<InputPathEvidence>,
    pub prompt: String,
    pub expanded_prompt: String,
    pub input_ids: Vec<u32>,
    pub image_files: Vec<ImageFileEvidence>,
    pub processed_images: Vec<ProcessedImageEvidence>,
    pub processed_crops: Vec<ProcessedCropEvidence>,
    pub image_spans: Vec<ImageSpanEvidence>,
    pub packed_tensors: PackedTensorEvidence,
    pub context_length: usize,
    pub vision_batch_size: usize,
    pub max_new_tokens: usize,
    pub eos: EosEvidence,
    pub generation: GenerationTrace,
    pub cache_reset_exact: bool,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct InputPathEvidence {
    pub path: String,
    pub kind: String,
    pub bytes: Option<u64>,
    pub sha256: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ImageFileEvidence {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    pub format: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Serialize)]
pub struct ProcessedImageEvidence {
    pub image_index: usize,
    pub crop_start: usize,
    pub crop_end: usize,
    pub rows: usize,
    pub cols: usize,
    pub resized_width: usize,
    pub resized_height: usize,
}

#[derive(Debug, Serialize)]
pub struct ProcessedCropEvidence {
    pub image_index: usize,
    pub crop_index: usize,
    pub kind: String,
    pub tile_row: Option<usize>,
    pub tile_col: Option<usize>,
    pub patch_rows: usize,
    pub patch_cols: usize,
    pub projected_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ImageSpanEvidence {
    pub batch_index: usize,
    pub image_index: usize,
    pub crop_index: usize,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Serialize)]
pub struct PackedTensorEvidence {
    pub pixel_values: Vec<usize>,
    pub pixel_attention_mask: Vec<usize>,
    pub spatial_shapes: Vec<usize>,
}

#[derive(Debug, Serialize)]
pub struct EosEvidence {
    pub token_id: Option<u32>,
    pub source: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GenerationTrace {
    pub prefill_logits_sha256: String,
    pub prefill_top_k: Vec<RankedToken>,
    pub steps: Vec<GenerationStep>,
    pub generated_ids: Vec<u32>,
    pub generated_tokens: Vec<Option<String>>,
    pub decoded_skip_special_tokens: String,
    pub decoded_with_special_tokens: String,
    pub stop_reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GenerationStep {
    pub step: usize,
    pub sample_position: usize,
    pub model_input_position: Option<usize>,
    pub model_input_token_id: Option<u32>,
    pub selected_token_id: u32,
    pub selected_token: Option<String>,
    pub logits_sha256: String,
    pub top_k: Vec<RankedToken>,
    pub is_eos: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RankedToken {
    pub token_id: u32,
    pub token: Option<String>,
    pub logit: f32,
}

struct LogitsEvidence {
    sha256: String,
    top_k: Vec<RankedToken>,
}

struct GenerationInputs<'a> {
    tokenizer: &'a Tokenizer,
    input_ids: &'a Tensor,
    image_spans: &'a [ImageTokenSpan],
    encoded_images: Option<&'a EncodedImages>,
    prompt_len: usize,
    max_new_tokens: usize,
    eos_token_id: Option<u32>,
}

trait Runtime {
    fn context_length(&self) -> usize;
    fn default_eos_token_id(&self) -> Option<u32>;
    fn default_eos_source(&self) -> &'static str;
    fn vision_device(&self) -> &Device;
    fn text_device(&self) -> &Device;
    fn reset(&mut self) -> Result<()>;
    fn encode_images(
        &mut self,
        inputs: &ProcessedVisionBatch,
        vision_batch_size: usize,
        limits: &VisionLimits,
    ) -> Result<EncodedImages>;
    fn encode_images_with_trace(
        &mut self,
        inputs: &ProcessedVisionBatch,
        vision_batch_size: usize,
        limits: &VisionLimits,
    ) -> Result<(EncodedImages, Option<Lfm2VlImageTrace>)> {
        Ok((self.encode_images(inputs, vision_batch_size, limits)?, None))
    }
    fn prefill(
        &mut self,
        input_ids: &Tensor,
        image_spans: &[ImageTokenSpan],
        encoded_images: Option<&EncodedImages>,
    ) -> Result<Tensor>;
    fn prefill_with_trace(
        &mut self,
        input_ids: &Tensor,
        image_spans: &[ImageTokenSpan],
        encoded_images: Option<&EncodedImages>,
    ) -> Result<Option<Lfm2VlPrefillTrace>> {
        let _ = (input_ids, image_spans, encoded_images);
        Ok(None)
    }
    fn decode(&mut self, token_ids: &Tensor, index_pos: usize) -> Result<Tensor>;
    fn decode_with_trace(
        &mut self,
        token_ids: &Tensor,
        index_pos: usize,
    ) -> Result<Option<Lfm2VlDecodeTrace>> {
        let _ = (token_ids, index_pos);
        Ok(None)
    }
}

struct NativeRuntime<'a> {
    model: &'a Lfm2VlModel,
    text_config: lfm2::Config,
    cache: lfm2::Cache,
}

impl<'a> NativeRuntime<'a> {
    fn new(model: &'a Lfm2VlModel) -> Result<Self> {
        let text_config = model.config().text_model_config()?;
        let cache = lfm2::Cache::new(true, model.text_dtype(), &text_config, model.text_device())?;
        Ok(Self {
            model,
            text_config,
            cache,
        })
    }
}

impl Runtime for NativeRuntime<'_> {
    fn context_length(&self) -> usize {
        self.text_config.max_position_embeddings
    }

    fn default_eos_token_id(&self) -> Option<u32> {
        self.text_config.eos_token_id
    }

    fn default_eos_source(&self) -> &'static str {
        "model_config"
    }

    fn vision_device(&self) -> &Device {
        self.model.vision_device()
    }

    fn text_device(&self) -> &Device {
        self.model.text_device()
    }

    fn reset(&mut self) -> Result<()> {
        self.cache = lfm2::Cache::new(
            true,
            self.model.text_dtype(),
            &self.text_config,
            self.model.text_device(),
        )?;
        Ok(())
    }

    fn encode_images(
        &mut self,
        inputs: &ProcessedVisionBatch,
        vision_batch_size: usize,
        limits: &VisionLimits,
    ) -> Result<EncodedImages> {
        self.model
            .encode_images_with_limits(inputs, vision_batch_size, limits)
            .map_err(anyhow::Error::from)
    }

    fn encode_images_with_trace(
        &mut self,
        inputs: &ProcessedVisionBatch,
        vision_batch_size: usize,
        limits: &VisionLimits,
    ) -> Result<(EncodedImages, Option<Lfm2VlImageTrace>)> {
        let (encoded, trace) = self
            .model
            .encode_images_with_trace(inputs, vision_batch_size, limits)
            .map_err(anyhow::Error::from)?;
        Ok((encoded, Some(trace)))
    }

    fn prefill(
        &mut self,
        input_ids: &Tensor,
        image_spans: &[ImageTokenSpan],
        encoded_images: Option<&EncodedImages>,
    ) -> Result<Tensor> {
        self.model
            .prefill(input_ids, image_spans, encoded_images, &mut self.cache)
            .map_err(anyhow::Error::from)
    }

    fn prefill_with_trace(
        &mut self,
        input_ids: &Tensor,
        image_spans: &[ImageTokenSpan],
        encoded_images: Option<&EncodedImages>,
    ) -> Result<Option<Lfm2VlPrefillTrace>> {
        self.model
            .prefill_with_trace(input_ids, image_spans, encoded_images, &mut self.cache)
            .map(Some)
            .map_err(anyhow::Error::from)
    }

    fn decode(&mut self, token_ids: &Tensor, index_pos: usize) -> Result<Tensor> {
        self.model
            .decode(token_ids, index_pos, &mut self.cache)
            .map_err(anyhow::Error::from)
    }

    fn decode_with_trace(
        &mut self,
        token_ids: &Tensor,
        index_pos: usize,
    ) -> Result<Option<Lfm2VlDecodeTrace>> {
        self.model
            .decode_with_trace(token_ids, index_pos, &mut self.cache)
            .map(Some)
            .map_err(anyhow::Error::from)
    }
}

struct HybridRuntime<'a> {
    model: &'a mut QuantizedLfm2VlModel,
}

impl Runtime for HybridRuntime<'_> {
    fn context_length(&self) -> usize {
        self.model.context_length()
    }

    fn default_eos_token_id(&self) -> Option<u32> {
        self.model.eos_token_id()
    }

    fn default_eos_source(&self) -> &'static str {
        "gguf_metadata"
    }

    fn vision_device(&self) -> &Device {
        self.model.vision_device()
    }

    fn text_device(&self) -> &Device {
        self.model.text_device()
    }

    fn reset(&mut self) -> Result<()> {
        self.model.clear_cache();
        Ok(())
    }

    fn encode_images(
        &mut self,
        inputs: &ProcessedVisionBatch,
        vision_batch_size: usize,
        limits: &VisionLimits,
    ) -> Result<EncodedImages> {
        self.model
            .encode_images_with_limits(inputs, vision_batch_size, limits)
            .map_err(anyhow::Error::from)
    }

    fn prefill(
        &mut self,
        input_ids: &Tensor,
        image_spans: &[ImageTokenSpan],
        encoded_images: Option<&EncodedImages>,
    ) -> Result<Tensor> {
        self.model
            .prefill(input_ids, image_spans, encoded_images)
            .map_err(anyhow::Error::from)
    }

    fn decode(&mut self, token_ids: &Tensor, index_pos: usize) -> Result<Tensor> {
        self.model
            .decode(token_ids, index_pos)
            .map_err(anyhow::Error::from)
    }
}

pub fn run_native(
    model: &Lfm2VlModel,
    processor: &Lfm2VlProcessor,
    prompt: &Lfm2VlPrompt,
    request: InferenceRequest<'_>,
) -> Result<InferenceReport> {
    let mut runtime = NativeRuntime::new(model)?;
    run(&mut runtime, processor, prompt, request)
}

pub fn run_hybrid(
    model: &mut QuantizedLfm2VlModel,
    processor: &Lfm2VlProcessor,
    prompt: &Lfm2VlPrompt,
    request: InferenceRequest<'_>,
) -> Result<InferenceReport> {
    let mut runtime = HybridRuntime { model };
    run(&mut runtime, processor, prompt, request)
}

fn run(
    runtime: &mut impl Runtime,
    processor: &Lfm2VlProcessor,
    prompt: &Lfm2VlPrompt,
    request: InferenceRequest<'_>,
) -> Result<InferenceReport> {
    let total_started = Instant::now();
    if request.vision_batch_size == 0 {
        bail!("vision batch size must be greater than zero")
    }
    let model_inputs = inspect_input_paths(request.model_inputs)?;
    if request.trace_output.is_some() && request.image_paths.len() != 1 {
        bail!("LFM2-VL native trace requires exactly one image input")
    }
    let limits = &processor.config().vision_limits;
    let image_load_started = Instant::now();
    let (images, image_files) = load_images(request.image_paths, limits)?;
    let image_load_ms = image_load_started.elapsed().as_secs_f64() * 1000.0;
    let processor_started = Instant::now();
    let processed = if images.is_empty() {
        empty_processed_batch()?
    } else {
        processor
            .process(&images, runtime.vision_device())
            .map_err(anyhow::Error::from)?
    };
    let processor_ms = processor_started.elapsed().as_secs_f64() * 1000.0;
    let prompt_started = Instant::now();
    let expanded = prompt
        .expand(request.prompt, &processed)
        .map_err(anyhow::Error::from)?;
    let prompt_ms = prompt_started.elapsed().as_secs_f64() * 1000.0;
    if expanded.input_ids.is_empty() {
        bail!("LFM2-VL inference prompt tokenized to an empty sequence")
    }
    let required_context = expanded
        .input_ids
        .len()
        .checked_add(request.max_new_tokens)
        .ok_or_else(|| anyhow::anyhow!("LFM2-VL requested context length overflow"))?;
    if required_context > runtime.context_length() {
        bail!(
            "LFM2-VL prompt tokens {} plus max new tokens {} exceed context length {}",
            expanded.input_ids.len(),
            request.max_new_tokens,
            runtime.context_length()
        )
    }

    let eos = resolve_eos(
        request.eos_token_id,
        runtime.default_eos_token_id(),
        runtime.default_eos_source(),
        prompt.tokenizer(),
    );
    let vision_started = Instant::now();
    let (encoded, image_trace) = if images.is_empty() {
        if request.trace_output.is_some() {
            bail!("LFM2-VL native trace requires one decoded image")
        }
        (None, None)
    } else if request.trace_output.is_some() {
        let (encoded, trace) =
            runtime.encode_images_with_trace(&processed, request.vision_batch_size, limits)?;
        (Some(encoded), trace)
    } else {
        (
            Some(runtime.encode_images(&processed, request.vision_batch_size, limits)?),
            None,
        )
    };
    let vision_ms = vision_started.elapsed().as_secs_f64() * 1000.0;
    let mut trace_capture = match (request.trace_output, image_trace) {
        (Some(_), Some(image)) => Some(NativeTraceCapture::new(image)),
        (Some(_), None) => bail!("LFM2-VL native trace is unavailable for the selected runtime"),
        (None, _) => None,
    };
    let input_ids =
        Tensor::new(expanded.input_ids.as_slice(), runtime.text_device())?.unsqueeze(0)?;
    let generation_inputs = GenerationInputs {
        tokenizer: prompt.tokenizer(),
        input_ids: &input_ids,
        image_spans: &expanded.image_spans,
        encoded_images: encoded.as_ref(),
        prompt_len: expanded.input_ids.len(),
        max_new_tokens: request.max_new_tokens,
        eos_token_id: eos.token_id,
    };
    let first_started = Instant::now();
    let first = generate_once(runtime, &generation_inputs)?;
    let first_generation_ms = first_started.elapsed().as_secs_f64() * 1000.0;
    if let Some(capture) = trace_capture.as_mut() {
        capture_trace(runtime, &generation_inputs, capture)?;
    }
    let reset_started = Instant::now();
    let second = generate_once(runtime, &generation_inputs)?;
    let reset_generation_ms = reset_started.elapsed().as_secs_f64() * 1000.0;
    if first != second {
        bail!(
            "LFM2-VL cache reset replay diverged: first generated {:?}, second generated {:?}",
            first.generated_ids,
            second.generated_ids
        )
    }
    if request.trace_output.is_some() {
        verify_input_paths_unchanged(request.model_inputs, &model_inputs)?;
    }

    let report = InferenceReport {
        contract: CONTRACT,
        backend: request.backend.to_owned(),
        model_inputs,
        prompt: request.prompt.to_owned(),
        expanded_prompt: expanded.expanded_text.clone(),
        input_ids: expanded.input_ids.clone(),
        image_files,
        processed_images: processed_image_evidence(&processed),
        processed_crops: processed_crop_evidence(&processed),
        image_spans: image_span_evidence(&expanded)?,
        packed_tensors: packed_tensor_evidence(&processed),
        context_length: runtime.context_length(),
        vision_batch_size: request.vision_batch_size,
        max_new_tokens: request.max_new_tokens,
        eos,
        generation: first,
        cache_reset_exact: true,
    };
    if let Some(output) = request.trace_output {
        let capture = trace_capture
            .ok_or_else(|| anyhow::anyhow!("LFM2-VL native trace capture was not initialized"))?;
        trace::write_native_trace(
            output,
            &report,
            &processed,
            &input_ids,
            images.first(),
            capture,
        )?;
    }
    if request.timings {
        eprintln!(
            "lfm2-vl timings_ms image_load={image_load_ms:.3} processor={processor_ms:.3} prompt={prompt_ms:.3} vision={vision_ms:.3} generation_first={first_generation_ms:.3} generation_reset={reset_generation_ms:.3} inference_total={:.3}",
            total_started.elapsed().as_secs_f64() * 1000.0
        );
    }
    Ok(report)
}

fn generate_once(
    runtime: &mut impl Runtime,
    inputs: &GenerationInputs<'_>,
) -> Result<GenerationTrace> {
    runtime.reset()?;
    let prefill_logits =
        runtime.prefill(inputs.input_ids, inputs.image_spans, inputs.encoded_images)?;
    let prefill = analyze_logits(&last_logits(&prefill_logits)?, inputs.tokenizer)?;
    validate_eos_id(inputs.eos_token_id, &prefill_logits)?;
    let mut generated_ids = Vec::new();
    generated_ids
        .try_reserve_exact(inputs.max_new_tokens)
        .map_err(|_| anyhow::anyhow!("allocating generated token buffer"))?;
    let mut steps = Vec::new();
    steps
        .try_reserve_exact(inputs.max_new_tokens)
        .map_err(|_| anyhow::anyhow!("allocating generation trace"))?;
    let mut next = LogitsEvidence {
        sha256: prefill.sha256.clone(),
        top_k: prefill.top_k.clone(),
    };
    let mut model_input_position = None;
    let mut model_input_token_id = None;
    let mut stop_reason = "max_new_tokens".to_owned();

    for step in 0..inputs.max_new_tokens {
        let selected = next
            .top_k
            .first()
            .ok_or_else(|| anyhow::anyhow!("LFM2-VL logits did not contain any tokens"))?;
        let selected_token_id = selected.token_id;
        let is_eos = inputs.eos_token_id == Some(selected_token_id);
        generated_ids.push(selected_token_id);
        steps.push(GenerationStep {
            step,
            sample_position: inputs
                .prompt_len
                .checked_add(step)
                .ok_or_else(|| anyhow::anyhow!("LFM2-VL sample position overflow"))?,
            model_input_position,
            model_input_token_id,
            selected_token_id,
            selected_token: inputs.tokenizer.id_to_token(selected_token_id),
            logits_sha256: next.sha256,
            top_k: next.top_k,
            is_eos,
        });
        if is_eos {
            stop_reason = "eos".to_owned();
            break;
        }
        if step + 1 == inputs.max_new_tokens {
            break;
        }
        let input_position = inputs
            .prompt_len
            .checked_add(step)
            .ok_or_else(|| anyhow::anyhow!("LFM2-VL decode position overflow"))?;
        let decode_ids = Tensor::new(&[selected_token_id], runtime.text_device())?.unsqueeze(0)?;
        let decode_logits = runtime.decode(&decode_ids, input_position)?;
        next = analyze_logits(&last_logits(&decode_logits)?, inputs.tokenizer)?;
        model_input_position = Some(input_position);
        model_input_token_id = Some(selected_token_id);
    }

    let generated_tokens = generated_ids
        .iter()
        .map(|&token_id| inputs.tokenizer.id_to_token(token_id))
        .collect();
    let decoded_skip_special_tokens = inputs
        .tokenizer
        .decode(&generated_ids, true)
        .map_err(anyhow::Error::msg)?;
    let decoded_with_special_tokens = inputs
        .tokenizer
        .decode(&generated_ids, false)
        .map_err(anyhow::Error::msg)?;
    Ok(GenerationTrace {
        prefill_logits_sha256: prefill.sha256,
        prefill_top_k: prefill.top_k,
        steps,
        generated_ids,
        generated_tokens,
        decoded_skip_special_tokens,
        decoded_with_special_tokens,
        stop_reason,
    })
}

fn capture_trace(
    runtime: &mut impl Runtime,
    inputs: &GenerationInputs<'_>,
    capture: &mut NativeTraceCapture,
) -> Result<()> {
    runtime.reset()?;
    let prefill = runtime
        .prefill_with_trace(inputs.input_ids, inputs.image_spans, inputs.encoded_images)?
        .ok_or_else(|| anyhow::anyhow!("selected runtime did not return prefill trace"))?;
    let mut next = last_logits(&prefill.logits)?
        .argmax(0)?
        .to_scalar::<u32>()?;
    capture.prefill = Some(prefill);
    capture
        .decode_input_ids
        .try_reserve_exact(inputs.max_new_tokens)
        .map_err(|_| anyhow::anyhow!("allocating native trace decode input IDs"))?;
    capture
        .decode
        .try_reserve_exact(inputs.max_new_tokens)
        .map_err(|_| anyhow::anyhow!("allocating native trace decode tensors"))?;
    for step in 0..inputs.max_new_tokens {
        let input_position = inputs
            .prompt_len
            .checked_add(step)
            .ok_or_else(|| anyhow::anyhow!("LFM2-VL trace decode position overflow"))?;
        let decode_ids = Tensor::new(&[next], runtime.text_device())?.unsqueeze(0)?;
        let decode = runtime
            .decode_with_trace(&decode_ids, input_position)?
            .ok_or_else(|| anyhow::anyhow!("selected runtime did not return decode trace"))?;
        next = last_logits(&decode.logits)?.argmax(0)?.to_scalar::<u32>()?;
        capture.decode_input_ids.push(decode_ids);
        capture.decode.push(decode);
    }
    Ok(())
}

fn last_logits(logits: &Tensor) -> Result<Tensor> {
    match logits.rank() {
        3 => {
            let (_, sequence, _) = logits.dims3()?;
            if sequence == 0 {
                bail!("LFM2-VL prefill returned an empty logits sequence")
            }
            logits.i((0, sequence - 1)).map_err(anyhow::Error::from)
        }
        2 => {
            let (batch, _) = logits.dims2()?;
            if batch != 1 {
                bail!("LFM2-VL logits batch size {batch} is unsupported; expected 1")
            }
            logits.i(0).map_err(anyhow::Error::from)
        }
        rank => bail!("LFM2-VL logits rank {rank} is unsupported; expected 2 or 3"),
    }
}

#[cfg(test)]
fn top_k(logits: &Tensor, tokenizer: &Tokenizer) -> Result<Vec<RankedToken>> {
    Ok(analyze_logits(logits, tokenizer)?.top_k)
}

fn analyze_logits(logits: &Tensor, tokenizer: &Tokenizer) -> Result<LogitsEvidence> {
    let values = logits
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if values.is_empty() {
        bail!("LFM2-VL logits vocabulary is empty")
    }
    if values.len() > MAX_REPORTED_VOCAB {
        bail!(
            "LFM2-VL logits vocabulary {} exceeds evidence limit {MAX_REPORTED_VOCAB}",
            values.len()
        )
    }
    if let Some((token_id, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        bail!("LFM2-VL logit for token {token_id} is not finite")
    }
    let mut hasher = Sha256::new();
    for value in &values {
        hasher.update(value.to_bits().to_le_bytes());
    }
    let count = TOP_K.min(values.len());
    let mut selected = Vec::new();
    selected
        .try_reserve_exact(count)
        .map_err(|_| anyhow::anyhow!("allocating top-k evidence"))?;
    for _ in 0..count {
        let mut best: Option<(usize, f32)> = None;
        for (token_id, &logit) in values.iter().enumerate() {
            if selected
                .iter()
                .any(|entry: &RankedToken| entry.token_id as usize == token_id)
            {
                continue;
            }
            if best.is_none_or(|(best_id, best_logit)| {
                logit > best_logit || (logit == best_logit && token_id < best_id)
            }) {
                best = Some((token_id, logit));
            }
        }
        let (token_id, logit) =
            best.ok_or_else(|| anyhow::anyhow!("selecting LFM2-VL top-k logits"))?;
        let token_id =
            u32::try_from(token_id).map_err(|_| anyhow::anyhow!("LFM2-VL token ID exceeds u32"))?;
        selected.push(RankedToken {
            token_id,
            token: tokenizer.id_to_token(token_id),
            logit,
        });
    }
    Ok(LogitsEvidence {
        sha256: format!("{:x}", hasher.finalize()),
        top_k: selected,
    })
}

fn validate_eos_id(eos_token_id: Option<u32>, logits: &Tensor) -> Result<()> {
    if let Some(eos_token_id) = eos_token_id {
        let vocab = *logits
            .dims()
            .last()
            .ok_or_else(|| anyhow::anyhow!("LFM2-VL logits have no vocabulary dimension"))?;
        let eos_index = usize::try_from(eos_token_id)
            .map_err(|_| anyhow::anyhow!("EOS token ID cannot be represented as usize"))?;
        if eos_index >= vocab {
            bail!("EOS token ID {eos_token_id} is outside model vocabulary size {vocab}")
        }
    }
    Ok(())
}

fn resolve_eos(
    explicit: Option<u32>,
    model_default: Option<u32>,
    model_source: &'static str,
    tokenizer: &Tokenizer,
) -> EosEvidence {
    if let Some(token_id) = explicit {
        return EosEvidence {
            token_id: Some(token_id),
            source: "cli".to_owned(),
        };
    }
    if let Some(token_id) = model_default {
        return EosEvidence {
            token_id: Some(token_id),
            source: model_source.to_owned(),
        };
    }
    let vocab = tokenizer.get_vocab(true);
    if let Some(token_id) = EOS_CANDIDATES
        .iter()
        .find_map(|candidate| vocab.get(*candidate).copied())
    {
        return EosEvidence {
            token_id: Some(token_id),
            source: "tokenizer_guess".to_owned(),
        };
    }
    EosEvidence {
        token_id: None,
        source: "none".to_owned(),
    }
}

fn inspect_input_paths(paths: &[PathBuf]) -> Result<Vec<InputPathEvidence>> {
    let mut evidence = Vec::new();
    let mut seen = BTreeSet::new();
    evidence
        .try_reserve_exact(paths.len())
        .map_err(|_| anyhow::anyhow!("allocating model input evidence"))?;
    for path in paths {
        let canonical = std::fs::canonicalize(path)
            .with_context(|| format!("resolving model input {}", path.display()))?;
        if !seen.insert(canonical.clone()) {
            continue;
        }
        let metadata = std::fs::metadata(&canonical)
            .with_context(|| format!("inspecting model input {}", canonical.display()))?;
        if !metadata.is_file() {
            bail!(
                "model input evidence {} is not a regular file",
                canonical.display()
            )
        }
        evidence.push(InputPathEvidence {
            path: path_string(&canonical),
            kind: "file".to_owned(),
            bytes: Some(metadata.len()),
            sha256: Some(sha256_file(&canonical)?),
        });
    }
    Ok(evidence)
}

fn verify_input_paths_unchanged(paths: &[PathBuf], expected: &[InputPathEvidence]) -> Result<()> {
    let current = inspect_input_paths(paths)?;
    if current != expected {
        bail!("LFM2-VL model inputs changed during traced inference")
    }
    Ok(())
}

fn load_images(
    paths: &[PathBuf],
    limits: &VisionLimits,
) -> Result<(Vec<DynamicImage>, Vec<ImageFileEvidence>)> {
    limits.validate()?;
    if paths.len() > limits.max_images {
        bail!(
            "LFM2-VL request has {} images, exceeding limit {}",
            paths.len(),
            limits.max_images
        )
    }
    let mut images = Vec::new();
    let mut evidence = Vec::new();
    images
        .try_reserve_exact(paths.len())
        .map_err(|_| anyhow::anyhow!("allocating decoded image list"))?;
    evidence
        .try_reserve_exact(paths.len())
        .map_err(|_| anyhow::anyhow!("allocating image evidence"))?;
    for path in paths {
        let (image, item) = load_image(path, limits)?;
        images.push(image);
        evidence.push(item);
    }
    Ok((images, evidence))
}

fn load_image(path: &Path, limits: &VisionLimits) -> Result<(DynamicImage, ImageFileEvidence)> {
    let canonical = std::fs::canonicalize(path)
        .with_context(|| format!("resolving image {}", path.display()))?;
    let mut file =
        File::open(&canonical).with_context(|| format!("opening image {}", canonical.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("inspecting image {}", path.display()))?;
    if !metadata.is_file() {
        bail!("image input {} is not a regular file", path.display())
    }
    if metadata.len() > MAX_COMPRESSED_IMAGE_BYTES {
        bail!(
            "compressed image {} is {} bytes, exceeding {}",
            path.display(),
            metadata.len(),
            MAX_COMPRESSED_IMAGE_BYTES
        )
    }
    let read_limit = MAX_COMPRESSED_IMAGE_BYTES
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("compressed image read limit overflow"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(
            usize::try_from(metadata.len())
                .map_err(|_| anyhow::anyhow!("image file size exceeds address space"))?,
        )
        .map_err(|_| anyhow::anyhow!("allocating compressed image buffer"))?;
    file.by_ref()
        .take(read_limit)
        .read_to_end(&mut bytes)
        .with_context(|| format!("reading image {}", path.display()))?;
    if bytes.len() as u64 > MAX_COMPRESSED_IMAGE_BYTES {
        bail!(
            "compressed image {} grew beyond {} bytes while reading",
            path.display(),
            MAX_COMPRESSED_IMAGE_BYTES
        )
    }
    if bytes.is_empty() {
        bail!("image input {} is empty", path.display())
    }
    let byte_len = u64::try_from(bytes.len())
        .map_err(|_| anyhow::anyhow!("compressed image length exceeds u64"))?;
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .with_context(|| format!("detecting image format for {}", path.display()))?;
    let format = reader
        .format()
        .ok_or_else(|| anyhow::anyhow!("unsupported image format for {}", path.display()))?;
    let dimension_cap = u32::try_from(limits.max_source_pixels).unwrap_or(u32::MAX);
    let max_alloc = u64::try_from(limits.max_source_pixels)
        .ok()
        .and_then(|pixels| pixels.checked_mul(8))
        .ok_or_else(|| anyhow::anyhow!("image decoder allocation limit overflow"))?;
    let mut decoder_limits = image::Limits::default();
    decoder_limits.max_image_width = Some(dimension_cap);
    decoder_limits.max_image_height = Some(dimension_cap);
    decoder_limits.max_alloc = Some(max_alloc);
    reader.limits(decoder_limits);
    let image = reader
        .decode()
        .with_context(|| format!("decoding image {}", path.display()))?;
    let (width, height) = image.dimensions();
    limits.check_source_image(width as usize, height as usize)?;
    let item = ImageFileEvidence {
        path: path_string(&canonical),
        bytes: byte_len,
        sha256,
        format: format!("{format:?}").to_ascii_lowercase(),
        width,
        height,
    };
    Ok((image, item))
}

fn empty_processed_batch() -> Result<ProcessedVisionBatch> {
    Ok(ProcessedVisionBatch {
        pixel_values: Tensor::zeros((0usize, 0usize, 0usize), DType::F32, &Device::Cpu)?,
        pixel_attention_mask: Tensor::zeros((0usize, 0usize), DType::I32, &Device::Cpu)?,
        spatial_shapes: Tensor::zeros((0usize, 2usize), DType::I64, &Device::Cpu)?,
        crops: Vec::new(),
        images: Vec::new(),
    })
}

fn processed_image_evidence(batch: &ProcessedVisionBatch) -> Vec<ProcessedImageEvidence> {
    batch
        .images
        .iter()
        .enumerate()
        .map(|(image_index, image)| ProcessedImageEvidence {
            image_index,
            crop_start: image.crop_range.start,
            crop_end: image.crop_range.end,
            rows: image.rows,
            cols: image.cols,
            resized_width: image.resized_width,
            resized_height: image.resized_height,
        })
        .collect()
}

fn processed_crop_evidence(batch: &ProcessedVisionBatch) -> Vec<ProcessedCropEvidence> {
    batch
        .crops
        .iter()
        .map(|crop| {
            let (kind, tile_row, tile_col) = match crop.kind {
                CropKind::Whole => ("whole", None, None),
                CropKind::Tile { row, col } => ("tile", Some(row), Some(col)),
                CropKind::Thumbnail => ("thumbnail", None, None),
            };
            ProcessedCropEvidence {
                image_index: crop.image_index,
                crop_index: crop.crop_index,
                kind: kind.to_owned(),
                tile_row,
                tile_col,
                patch_rows: crop.patch_rows,
                patch_cols: crop.patch_cols,
                projected_tokens: crop.projected_tokens,
            }
        })
        .collect()
}

fn image_span_evidence(expanded: &ExpandedPrompt) -> Result<Vec<ImageSpanEvidence>> {
    if expanded.image_spans.len() != expanded.span_image_indices.len()
        || expanded.image_spans.len() != expanded.span_crop_indices.len()
    {
        bail!("LFM2-VL expanded prompt span provenance lengths do not match")
    }
    Ok(expanded
        .image_spans
        .iter()
        .zip(&expanded.span_image_indices)
        .zip(&expanded.span_crop_indices)
        .map(|((span, &image_index), &crop_index)| ImageSpanEvidence {
            batch_index: span.batch_index,
            image_index,
            crop_index,
            start: span.start,
            end: span.end,
        })
        .collect())
}

fn packed_tensor_evidence(batch: &ProcessedVisionBatch) -> PackedTensorEvidence {
    PackedTensorEvidence {
        pixel_values: batch.pixel_values.dims().to_vec(),
        pixel_attention_mask: batch.pixel_attention_mask.dims().to_vec(),
        spatial_shapes: batch.spatial_shapes.dims().to_vec(),
    }
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .with_context(|| format!("opening model input for hashing {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .with_context(|| format!("hashing model input {}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::MmprojExecutionArg;
    use crate::loading::{self, MmprojInput, MmprojLoadOptions};
    use candle::quantized::{gguf_file, GgmlDType, QTensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::{lfm2, lfm2_vl::Lfm2VlConfig};
    use candle_vlm::lfm2_vl::{Lfm2VlProcessorConfig, PromptOptions};
    use image::{ImageFormat, Rgb, RgbImage};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::AddedToken;

    const TINY_FIXTURE: &[u8] =
        include_bytes!("../../../tests/fixtures/lfm2_vl_tiny/tensors.safetensors");
    const TINY_CONFIG: &str =
        include_str!("../../../tests/fixtures/lfm2_vl_mmproj_tiny/source_model_config.json");

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Result<Self> {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let number = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "candle-lfm2-vl-runner-{}-{number}",
                std::process::id()
            ));
            std::fs::create_dir(&path)
                .with_context(|| format!("creating test directory {}", path.display()))?;
            Ok(Self(path))
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn tokenizer() -> Result<Tokenizer> {
        let vocab = (0..32u32)
            .map(|token_id| {
                let token = match token_id {
                    0 => "<unk>".to_owned(),
                    1 => "hello".to_owned(),
                    2 => "</s>".to_owned(),
                    3 => "<image>".to_owned(),
                    4 => "<|image_start|>".to_owned(),
                    5 => "<|image_end|>".to_owned(),
                    6 => "<|img_thumbnail|>".to_owned(),
                    7..=22 => {
                        let offset = token_id - 7;
                        format!("<|img_row_{}_col_{}|>", offset / 4 + 1, offset % 4 + 1)
                    }
                    _ => format!("token-{token_id}"),
                };
                (token, token_id)
            })
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".to_owned())
            .build()
            .map_err(anyhow::Error::msg)?;
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        let mut special_tokens = vec![
            AddedToken::from("<image>", true),
            AddedToken::from("<|image_start|>", true),
            AddedToken::from("<|image_end|>", true),
            AddedToken::from("<|img_thumbnail|>", true),
        ];
        for row in 1..=4 {
            for col in 1..=4 {
                special_tokens.push(AddedToken::from(
                    format!("<|img_row_{row}_col_{col}|>"),
                    true,
                ));
            }
        }
        tokenizer.add_special_tokens(&special_tokens);
        Ok(tokenizer)
    }

    fn fixture_tensor<'a>(tensors: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
        tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing tiny fixture tensor {name}"))
    }

    fn tiny_text_gguf(tensors: &HashMap<String, Tensor>, config: &Lfm2VlConfig) -> Result<Vec<u8>> {
        let text = config.text_model_config()?;
        let root = "weights.model.language_model";
        let mut names = vec![
            (
                "token_embd.weight".to_owned(),
                format!("{root}.embed_tokens.weight"),
            ),
            (
                "output_norm.weight".to_owned(),
                format!("{root}.embedding_norm.weight"),
            ),
        ];
        for (layer, layer_type) in text.layer_types.iter().enumerate() {
            let native = format!("{root}.layers.{layer}");
            let gguf = format!("blk.{layer}");
            names.extend([
                (
                    format!("{gguf}.attn_norm.weight"),
                    format!("{native}.operator_norm.weight"),
                ),
                (
                    format!("{gguf}.ffn_norm.weight"),
                    format!("{native}.ffn_norm.weight"),
                ),
                (
                    format!("{gguf}.ffn_gate.weight"),
                    format!("{native}.feed_forward.w1.weight"),
                ),
                (
                    format!("{gguf}.ffn_down.weight"),
                    format!("{native}.feed_forward.w2.weight"),
                ),
                (
                    format!("{gguf}.ffn_up.weight"),
                    format!("{native}.feed_forward.w3.weight"),
                ),
            ]);
            match layer_type {
                lfm2::LayerType::Conv => names.extend([
                    (
                        format!("{gguf}.shortconv.in_proj.weight"),
                        format!("{native}.conv.in_proj.weight"),
                    ),
                    (
                        format!("{gguf}.shortconv.out_proj.weight"),
                        format!("{native}.conv.out_proj.weight"),
                    ),
                    (
                        format!("{gguf}.shortconv.conv.weight"),
                        format!("{native}.conv.conv.weight"),
                    ),
                ]),
                lfm2::LayerType::FullAttention => names.extend([
                    (
                        format!("{gguf}.attn_q.weight"),
                        format!("{native}.self_attn.q_proj.weight"),
                    ),
                    (
                        format!("{gguf}.attn_k.weight"),
                        format!("{native}.self_attn.k_proj.weight"),
                    ),
                    (
                        format!("{gguf}.attn_v.weight"),
                        format!("{native}.self_attn.v_proj.weight"),
                    ),
                    (
                        format!("{gguf}.attn_output.weight"),
                        format!("{native}.self_attn.out_proj.weight"),
                    ),
                    (
                        format!("{gguf}.attn_q_norm.weight"),
                        format!("{native}.self_attn.q_layernorm.weight"),
                    ),
                    (
                        format!("{gguf}.attn_k_norm.weight"),
                        format!("{native}.self_attn.k_layernorm.weight"),
                    ),
                ]),
            }
        }

        let mut qtensors = Vec::new();
        for (gguf_name, native_name) in names {
            let tensor = fixture_tensor(tensors, &native_name)?.contiguous()?;
            let dtype = if tensor.rank() == 2
                && tensor.dim(1)?.is_multiple_of(GgmlDType::Q8_0.block_size())
            {
                GgmlDType::Q8_0
            } else {
                GgmlDType::F32
            };
            qtensors.push((gguf_name, QTensor::quantize(&tensor, dtype)?));
        }
        let to_u32 = |value: usize, label: &str| {
            u32::try_from(value).map_err(|_| anyhow::anyhow!("{label} exceeds u32"))
        };
        let metadata = vec![
            (
                "general.architecture".to_owned(),
                gguf_file::Value::String("lfm2".to_owned()),
            ),
            (
                "lfm2.attention.head_count".to_owned(),
                gguf_file::Value::U32(to_u32(text.num_attention_heads, "head count")?),
            ),
            (
                "lfm2.attention.head_count_kv".to_owned(),
                gguf_file::Value::Array(
                    text.layer_types
                        .iter()
                        .map(|kind| match kind {
                            lfm2::LayerType::FullAttention => {
                                to_u32(text.num_key_value_heads, "key/value head count")
                                    .map(gguf_file::Value::U32)
                            }
                            lfm2::LayerType::Conv => Ok(gguf_file::Value::U32(0)),
                        })
                        .collect::<Result<Vec<_>>>()?,
                ),
            ),
            (
                "lfm2.embedding_length".to_owned(),
                gguf_file::Value::U32(to_u32(text.hidden_size, "embedding length")?),
            ),
            (
                "lfm2.context_length".to_owned(),
                gguf_file::Value::U32(to_u32(text.max_position_embeddings, "context length")?),
            ),
            (
                "lfm2.block_count".to_owned(),
                gguf_file::Value::U32(to_u32(text.num_hidden_layers, "block count")?),
            ),
            (
                "lfm2.attention.layer_norm_rms_epsilon".to_owned(),
                gguf_file::Value::F32(text.norm_eps as f32),
            ),
            (
                "lfm2.rope.freq_base".to_owned(),
                gguf_file::Value::F32(text.rope_theta),
            ),
            (
                "lfm2.shortconv.l_cache".to_owned(),
                gguf_file::Value::U32(to_u32(text.conv_l_cache, "short-convolution cache")?),
            ),
            (
                "tokenizer.ggml.eos_token_id".to_owned(),
                gguf_file::Value::U32(31),
            ),
        ];
        let metadata_refs: Vec<_> = metadata
            .iter()
            .map(|(name, value)| (name.as_str(), value))
            .collect();
        let tensor_refs: Vec<_> = qtensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        let mut output = Cursor::new(Vec::new());
        gguf_file::write(&mut output, &metadata_refs, &tensor_refs)?;
        Ok(output.into_inner())
    }

    fn split_bundle_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/lfm2_vl_mmproj_tiny")
    }

    fn processor_and_prompt() -> Result<(Lfm2VlProcessor, Lfm2VlPrompt)> {
        let mut config = Lfm2VlProcessorConfig::default();
        config.do_resize = false;
        config.downsample_factor = 2;
        config.encoder_patch_size = 2;
        config.do_image_splitting = false;
        config.min_tiles = 1;
        config.max_tiles = 2;
        config.use_thumbnail = false;
        config.tile_size = 8;
        config.min_image_tokens = 1;
        config.max_image_tokens = 4;
        config.max_num_patches = Some(16);
        config.context_length = Some(64);
        let processor = Lfm2VlProcessor::from_config(&config)?;
        let prompt = Lfm2VlPrompt::new(
            tokenizer()?,
            Some(3),
            config,
            PromptOptions {
                use_image_special_tokens: false,
                context_length: Some(64),
            },
        )?;
        Ok((processor, prompt))
    }

    #[test]
    fn top_k_is_finite_and_breaks_ties_by_token_id() -> Result<()> {
        let tokenizer = Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default());
        let logits = Tensor::new(&[1f32, 3., 3., -2.], &Device::Cpu)?;
        let ranked = top_k(&logits, &tokenizer)?;
        assert_eq!(ranked[0].token_id, 1);
        assert_eq!(ranked[1].token_id, 2);
        let non_finite = Tensor::new(&[0f32, f32::NAN], &Device::Cpu)?;
        assert!(top_k(&non_finite, &tokenizer).is_err());
        Ok(())
    }

    #[test]
    fn native_fixture_runs_image_prefill_decode_and_exact_cache_replay() -> Result<()> {
        let config = Lfm2VlConfig::from_json(TINY_CONFIG)?;
        let weights = VarBuilder::from_slice_safetensors(TINY_FIXTURE, DType::F32, &Device::Cpu)?;
        let model = Lfm2VlModel::new(&config, weights.pp("weights"))?;
        let (processor, prompt) = processor_and_prompt()?;
        let dir = TestDir::new()?;
        let image_path = dir.path().join("fixture.png");
        let pixels = RgbImage::from_fn(8, 4, |x, y| {
            Rgb([(x * 17) as u8, (y * 41) as u8, ((x + y) * 13) as u8])
        });
        DynamicImage::ImageRgb8(pixels).save_with_format(&image_path, ImageFormat::Png)?;
        let trace_output = dir.path().join("trace");
        let model_inputs = vec![PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures/lfm2_vl_tiny/tensors.safetensors")];
        let image_paths = vec![image_path];
        let report = run_native(
            &model,
            &processor,
            &prompt,
            InferenceRequest {
                backend: "native-fixture",
                model_inputs: &model_inputs,
                prompt: "<image> hello",
                image_paths: &image_paths,
                max_new_tokens: 3,
                vision_batch_size: 1,
                eos_token_id: None,
                timings: false,
                trace_output: Some(trace_output.as_path()),
            },
        )?;

        assert!(report.cache_reset_exact);
        assert_eq!(report.contract, CONTRACT);
        assert_eq!(
            report.model_inputs[0].sha256.as_deref().map(str::len),
            Some(64)
        );
        assert_eq!(report.image_files.len(), 1);
        assert_eq!(report.image_files[0].width, 8);
        assert_eq!(report.image_files[0].height, 4);
        assert_eq!(report.image_files[0].sha256.len(), 64);
        assert_eq!(report.processed_crops.len(), 1);
        assert_eq!(report.processed_crops[0].patch_rows, 2);
        assert_eq!(report.processed_crops[0].patch_cols, 4);
        assert_eq!(report.image_spans.len(), 1);
        assert_eq!(report.image_spans[0].end - report.image_spans[0].start, 2);
        assert_eq!(report.generation.prefill_logits_sha256.len(), 64);
        assert!(!report.generation.generated_ids.is_empty());
        assert!(report.generation.generated_ids.len() <= 3);
        let json = serde_json::to_value(&report)?;
        assert_eq!(json["cache_reset_exact"], true);
        let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
            trace_output.join("manifest.json"),
        )?)?;
        assert_eq!(manifest["mode"], "native-trace");
        assert_eq!(manifest["weights_serialized"], false);
        assert_eq!(manifest["model_inputs_reverified"], true);
        assert!(manifest["tensor_inventory"]["stage.language.prefill_logits"].is_object());
        let trace_metadata: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
            trace_output.join("metadata.json"),
        )?)?;
        assert_eq!(
            trace_metadata["model_inputs"].as_array().map(Vec::len),
            Some(1)
        );
        assert_eq!(trace_metadata["model_inputs"][0]["kind"], "file");
        assert_eq!(
            trace_metadata["model_inputs"][0]["sha256"]
                .as_str()
                .map(str::len),
            Some(64)
        );
        assert_eq!(
            manifest["tensor_inventory"]["input.pixel_values"]["dtype"],
            "float32"
        );
        assert_eq!(
            manifest["tensor_inventory"]["input.pixel_attention_mask"]["dtype"],
            "int32"
        );
        assert_eq!(
            manifest["tensor_inventory"]["input.attention_mask"]["dtype"],
            "int64"
        );
        assert_eq!(
            manifest["tensor_inventory"]["input.image_rgb_u8"]["dtype"],
            "uint8"
        );
        let trace_tensors =
            candle::safetensors::load(trace_output.join("tensors.safetensors"), &Device::Cpu)?;
        assert!(trace_tensors.contains_key("stage.projector.output"));
        assert!(trace_tensors.contains_key("stage.vision.encoder_layer.0"));
        assert_eq!(
            trace_tensors["input.attention_mask"].dims(),
            trace_tensors["input.input_ids"].dims()
        );
        assert_eq!(
            trace_tensors["input.projector_crop_ranges"].to_vec2::<i64>()?,
            vec![vec![0, 8]]
        );
        assert_eq!(trace_tensors["input.decode_token_ids"].dims(), [1, 3]);
        assert_eq!(
            trace_tensors["stage.language.decode_logits"].dims(),
            [1, 3, 32]
        );
        Ok(())
    }

    #[test]
    fn hybrid_fixture_reports_exact_files_eos_and_cache_replay() -> Result<()> {
        let device = Device::Cpu;
        let config = Lfm2VlConfig::from_json(TINY_CONFIG)?;
        let tensors = candle::safetensors::load_buffer(TINY_FIXTURE, &device)?;
        let dir = TestDir::new()?;
        let text_path = dir.path().join("text.gguf");
        std::fs::write(&text_path, tiny_text_gguf(&tensors, &config)?)?;
        let tokenizer_path = dir.path().join("tokenizer.json");
        tokenizer()?
            .save(&tokenizer_path, false)
            .map_err(anyhow::Error::msg)?;
        let mut loaded = loading::load_hybrid(
            &text_path,
            MmprojInput::SplitDirectory(&split_bundle_dir()),
            &tokenizer_path,
            None,
            MmprojLoadOptions {
                execution: MmprojExecutionArg::Dense,
                dtype: DType::F32,
                device: &device,
            },
            &device,
        )?;
        let image_path = dir.path().join("fixture.png");
        let pixels = RgbImage::from_fn(8, 4, |x, y| {
            Rgb([(x * 17) as u8, (y * 41) as u8, ((x + y) * 13) as u8])
        });
        DynamicImage::ImageRgb8(pixels).save_with_format(&image_path, ImageFormat::Png)?;
        let image_paths = vec![image_path];
        let report = run_hybrid(
            &mut loaded.model,
            &loaded.processor,
            &loaded.prompt,
            InferenceRequest {
                backend: "hybrid-split-fixture",
                model_inputs: &loaded.source_files,
                prompt: "<image> hello",
                image_paths: &image_paths,
                max_new_tokens: 3,
                vision_batch_size: 1,
                eos_token_id: None,
                timings: false,
                trace_output: None,
            },
        )?;

        assert!(report.cache_reset_exact);
        assert_eq!(report.eos.token_id, Some(31));
        assert_eq!(report.eos.source, "gguf_metadata");
        assert_eq!(report.generation.steps.len(), 3);
        assert_eq!(report.model_inputs.len(), 5);
        assert!(report.model_inputs.iter().all(|input| {
            input.kind == "file"
                && input.bytes.is_some()
                && input.sha256.as_deref().is_some_and(|hash| hash.len() == 64)
        }));
        assert!(report
            .model_inputs
            .iter()
            .any(|input| input.path.ends_with("mmproj.safetensors")));
        let json = serde_json::to_string(&report)?;
        assert_eq!(json.lines().count(), 1);
        Ok(())
    }

    #[test]
    fn model_input_evidence_rejects_directories() -> Result<()> {
        let dir = TestDir::new()?;
        let error = inspect_input_paths(&[dir.path().to_path_buf()])
            .expect_err("directory evidence unexpectedly accepted");
        assert!(error.to_string().contains("not a regular file"));
        Ok(())
    }

    #[test]
    fn traced_model_input_evidence_detects_content_changes() -> Result<()> {
        let dir = TestDir::new()?;
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, b"before")?;
        let paths = vec![path.clone()];
        let expected = inspect_input_paths(&paths)?;
        verify_input_paths_unchanged(&paths, &expected)?;
        std::fs::write(path, b"after")?;
        let error = verify_input_paths_unchanged(&paths, &expected)
            .expect_err("changed trace input unexpectedly passed revalidation");
        assert!(error
            .to_string()
            .contains("changed during traced inference"));
        Ok(())
    }
}
