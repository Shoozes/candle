use super::backend::D1Backend;
use super::loading::{prepare_tokenizer, validate_devices, validate_limits};
use super::profiling::{measure, Phase};
use super::{
    D1Limits, D1PolicyV1, DecisionFailure, DecisionRequest, DecisionResponse, ExecutionReport,
    PreparedQuestion,
};
use super::{D1QuestionTiming, D1Timings};
use crate::lfm2_vl::{Lfm2EmbeddingStorage, LoadedLfm2VlHybrid};
use candle::quantized::with_native_q8_0;
use candle::{DType, IndexOp, Result, Tensor};
use image::DynamicImage;
use std::collections::HashSet;
use std::path::PathBuf;

fn is_image_marker(text: &str) -> bool {
    [
        "<image>",
        "<|image_start|>",
        "<|image_end|>",
        "<|img_thumbnail|>",
        "<|img_row_",
    ]
    .iter()
    .any(|marker| text.contains(marker))
}

/// Borrowed checkpoints for an explicitly requested numerical trace.
pub enum D1TraceEvent<'a> {
    ProcessedImages(&'a crate::lfm2_vl::ProcessedVisionBatch),
    ImageFeatures(&'a crate::lfm2_vl::EncodedImages),
    QuestionInput {
        index: usize,
        text: &'a str,
        ids: &'a [u32],
    },
    AnswerLogits {
        index: usize,
        logits: &'a Tensor,
    },
}

pub struct D1Session {
    backend: D1Backend,
    consumed_files: Vec<PathBuf>,
    policy: D1PolicyV1,
    limits: D1Limits,
    tokenizer: tokenizers::Tokenizer,
    text_quantized_linears: usize,
    image_token_ids: Vec<u32>,
}

impl D1Session {
    pub fn from_hybrid(
        mut loaded: LoadedLfm2VlHybrid,
        policy: D1PolicyV1,
        limits: D1Limits,
    ) -> Result<Self> {
        validate_limits(&limits)?;
        validate_devices(loaded.model.text_device(), loaded.model.vision_device())?;
        loaded.prompt.disable_tokenizer_controls()?;
        loaded.processor = loaded.processor.with_bicubic_resize();
        let text_quantized_linears = loaded.model.require_native_q8_text_linears()?;
        if loaded.model.mmproj().gguf_execution()
            != Some(candle_transformers::models::lfm2_vl::GgufMmprojExecution::Q8_0)
            || loaded.model.mmproj().native_quantized_tensor_count() == 0
            || loaded.model.mmproj().dtype() != DType::F32
        {
            candle::bail!("d1 session requires native Q8 MMProj storage with F32 activations")
        }
        let tokenizer = prepare_tokenizer(loaded.prompt.tokenizer().clone())?;
        let consumed_files = std::mem::take(&mut loaded.consumed_files);
        Ok(Self {
            backend: D1Backend::Hybrid(Box::new(loaded)),
            consumed_files,
            policy,
            limits,
            tokenizer,
            text_quantized_linears,
            image_token_ids: Vec::new(),
        })
    }

    pub(super) fn from_text(
        model: candle_transformers::models::quantized_lfm2::ModelWeights,
        tokenizer: tokenizers::Tokenizer,
        limits: D1Limits,
        consumed_files: Vec<PathBuf>,
    ) -> Result<Self> {
        validate_limits(&limits)?;
        validate_devices(model.device(), model.device())?;
        let text_quantized_linears = model.require_native_q8_linears()?;
        let tokenizer = prepare_tokenizer(tokenizer)?;
        let image_token_ids = [
            "<image>",
            "<|image_start|>",
            "<|image_end|>",
            "<|img_thumbnail|>",
        ]
        .iter()
        .filter_map(|marker| tokenizer.token_to_id(marker))
        .collect();
        Ok(Self {
            backend: D1Backend::Text(Box::new(model)),
            consumed_files,
            policy: D1PolicyV1,
            limits,
            tokenizer,
            text_quantized_linears,
            image_token_ids,
        })
    }

    /// Whether this session loaded the vision stack and permits image requests.
    pub fn supports_images(&self) -> bool {
        self.backend.supports_images() && self.limits.max_images > 0
    }

    pub fn consumed_files(&self) -> &[PathBuf] {
        &self.consumed_files
    }
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }
    pub fn limits(&self) -> &D1Limits {
        &self.limits
    }

    pub fn text_embedding_storage(&self) -> Result<Lfm2EmbeddingStorage> {
        self.backend.embedding_storage()
    }

    pub fn execution_details(&self) -> ExecutionReport {
        ExecutionReport {
            policy: D1PolicyV1::ID.into(),
            text_device: if self.backend.text_device().is_cpu() {
                "cpu"
            } else {
                "cuda"
            }
            .into(),
            vision_device: self
                .backend
                .vision_device()
                .map_or(
                    "none",
                    |device| if device.is_cpu() { "cpu" } else { "cuda" },
                )
                .into(),
            mmproj_quantized_tensors: self.backend.hybrid().map_or(0, |loaded| {
                loaded.model.mmproj().native_quantized_tensor_count()
            }),
            text_quantized_linears: self.text_quantized_linears,
            source_float_mmproj_linears: self.backend.hybrid().map_or_else(
                |_| Vec::new(),
                |loaded| loaded.model.mmproj().source_float_linear_names().to_vec(),
            ),
            ..Default::default()
        }
    }

    pub fn prepare(
        &self,
        request: &DecisionRequest,
        image_count: usize,
    ) -> Result<Vec<(String, PreparedQuestion, String)>> {
        if image_count > 0 && !self.backend.supports_images() {
            candle::bail!("text-only d1 session does not support images")
        }
        if request.questions.0.is_empty()
            || request.questions.0.len() > self.limits.max_questions
            || image_count > self.limits.max_images
        {
            candle::bail!("d1 request exceeds its question/image limits")
        }
        let mut names = HashSet::new();
        let mut plans = Vec::new();
        for (name, question) in &request.questions.0 {
            if name.trim().is_empty() || !names.insert(name) {
                candle::bail!("d1 question names must be nonempty and unique")
            }
            let plan = self
                .policy
                .prepare(&self.tokenizer, question, &self.limits)?;
            let rendered = self.policy.render(
                &self.tokenizer,
                &request.state,
                &plan,
                image_count,
                self.limits.max_prompt_bytes,
            )?;
            if !self.backend.supports_images() && is_image_marker(&rendered) {
                candle::bail!("text-only d1 prompt contains an image sentinel")
            }
            plans.push((name.clone(), plan, rendered));
        }
        Ok(plans)
    }

    pub fn decide(
        &mut self,
        request: &DecisionRequest,
        images: &[DynamicImage],
    ) -> std::result::Result<DecisionResponse, DecisionFailure> {
        self.decide_with_cancel(request, images, || false)
    }

    /// Cancellation is checked between synchronous forwards. The caller owns preemption.
    pub fn decide_with_cancel(
        &mut self,
        request: &DecisionRequest,
        images: &[DynamicImage],
        cancelled: impl Fn() -> bool,
    ) -> std::result::Result<DecisionResponse, DecisionFailure> {
        self.decide_traced(request, images, cancelled, |_| Ok(()))
    }

    pub fn decide_traced(
        &mut self,
        request: &DecisionRequest,
        images: &[DynamicImage],
        cancelled: impl Fn() -> bool,
        observe: impl FnMut(D1TraceEvent<'_>) -> Result<()>,
    ) -> std::result::Result<DecisionResponse, DecisionFailure> {
        self.decide_observed(request, images, cancelled, observe, None)
    }

    /// Reset and fill timings, including attempted phases on failure.
    /// Synchronization changes profiling latency; ordinary calls do not synchronize.
    pub fn decide_traced_profiled(
        &mut self,
        request: &DecisionRequest,
        images: &[DynamicImage],
        cancelled: impl Fn() -> bool,
        observe: impl FnMut(D1TraceEvent<'_>) -> Result<()>,
        timings: &mut D1Timings,
    ) -> std::result::Result<DecisionResponse, DecisionFailure> {
        *timings = D1Timings::default();
        self.decide_observed(request, images, cancelled, observe, Some(timings))
    }

    fn decide_observed(
        &mut self,
        request: &DecisionRequest,
        images: &[DynamicImage],
        cancelled: impl Fn() -> bool,
        mut observe: impl FnMut(D1TraceEvent<'_>) -> Result<()>,
        mut timings: Option<&mut D1Timings>,
    ) -> std::result::Result<DecisionResponse, DecisionFailure> {
        let started = timings.as_ref().map(|_| std::time::Instant::now());
        let text_device = self.backend.text_device().clone();
        let vision_device = self.backend.vision_device().cloned();
        let mut response = DecisionResponse {
            planned_questions: request.questions.0.len(),
            unattempted_questions: request.questions.0.len(),
            execution: self.execution_details(),
            ..Default::default()
        };
        let mut question_index = None;
        let result = (|| -> Result<()> {
            let plans = measure(&mut timings, Phase::Preparation, None, || {
                self.prepare(request, images.len())
            })?;
            if cancelled() {
                candle::bail!("d1 request cancelled before preparation")
            }
            let processed = if vision_device.is_none() {
                None
            } else {
                measure(
                    &mut timings,
                    Phase::ImageProcessing,
                    vision_device.as_ref(),
                    || {
                        Ok(if images.is_empty() {
                            None
                        } else {
                            let capped = images
                                .iter()
                                .map(|image| {
                                    super::image_cap::cap_pixels(
                                        image,
                                        super::image_cap::MAX_PIXELS,
                                    )
                                })
                                .collect::<Result<Vec<_>>>()?;
                            Some(self.backend.process_images(&capped)?)
                        })
                    },
                )?
            };
            if let Some(processed) = &processed {
                measure(&mut timings, Phase::Observation, None, || {
                    observe(D1TraceEvent::ProcessedImages(processed))
                })?;
            }
            let expanded = measure(&mut timings, Phase::Tokenization, None, || {
                let mut expanded = Vec::new();
                for (_, _, rendered) in &plans {
                    let (ids, spans) = match &processed {
                        Some(processed) => {
                            let prompt = self.backend.expand(rendered, processed)?;
                            (prompt.input_ids, prompt.image_spans)
                        }
                        None => {
                            let ids = self
                                .tokenizer
                                .encode(rendered.as_str(), false)
                                .map_err(|error| candle::Error::Msg(error.to_string()))?;
                            (ids.get_ids().to_vec(), Vec::new())
                        }
                    };
                    if !self.backend.supports_images()
                        && ids.iter().any(|id| {
                            self.image_token_ids.contains(id)
                                || self
                                    .tokenizer
                                    .id_to_token(*id)
                                    .is_some_and(|token| is_image_marker(&token))
                        })
                    {
                        candle::bail!("text-only d1 prompt contains an image sentinel")
                    }
                    let limit = self
                        .limits
                        .max_context_tokens
                        .unwrap_or(self.backend.context_length())
                        .min(self.backend.context_length());
                    if ids.is_empty() || ids.len() > limit {
                        candle::bail!("d1 prompt has {} tokens outside 1..={limit}", ids.len())
                    }
                    expanded.push((ids, spans));
                }
                Ok(expanded)
            })?;
            let encoded = match &processed {
                Some(processed) => {
                    if cancelled() {
                        candle::bail!("d1 request cancelled before vision forward")
                    }
                    let encoded =
                        measure(&mut timings, Phase::Vision, vision_device.as_ref(), || {
                            if cancelled() {
                                candle::bail!("d1 request cancelled before vision forward")
                            }
                            response.usage.vision_forwards = 1;
                            let (result, dispatch) =
                                with_native_q8_0(|| self.backend.encode_images(processed));
                            response.execution.accumulate(dispatch)?;
                            result
                        })?;
                    measure(&mut timings, Phase::Observation, None, || {
                        observe(D1TraceEvent::ImageFeatures(&encoded))
                    })?;
                    Some(encoded)
                }
                None => None,
            };
            for (index, ((name, plan, rendered), (ids, spans))) in
                plans.into_iter().zip(expanded).enumerate()
            {
                question_index = Some(index);
                if cancelled() {
                    candle::bail!("d1 request cancelled before question {index}")
                }
                if let Some(timings) = timings.as_deref_mut() {
                    timings.questions.push(D1QuestionTiming {
                        index,
                        ..Default::default()
                    });
                }
                measure(&mut timings, Phase::Observation, None, || {
                    observe(D1TraceEvent::QuestionInput {
                        index,
                        text: &rendered,
                        ids: &ids,
                    })
                })?;
                let logits = measure(
                    &mut timings,
                    Phase::Prefill(index),
                    Some(&text_device),
                    || {
                        if cancelled() {
                            candle::bail!("d1 request cancelled before question {index}")
                        }
                        let input = Tensor::new(ids.as_slice(), self.backend.text_device())?
                            .unsqueeze(0)?;
                        response.unattempted_questions -= 1;
                        response.usage.language_forwards += 1;
                        response.usage.input_tokens = response
                            .usage
                            .input_tokens
                            .checked_add(ids.len() as u64)
                            .ok_or_else(|| {
                                candle::Error::Msg("d1 input token counter overflow".into())
                            })?;
                        let (result, dispatch) = with_native_q8_0(|| {
                            self.backend.prefill(&input, &spans, encoded.as_ref())
                        });
                        response.execution.accumulate(dispatch)?;
                        result?.i(0)
                    },
                )?;
                measure(&mut timings, Phase::Observation, None, || {
                    observe(D1TraceEvent::AnswerLogits {
                        index,
                        logits: &logits,
                    })
                })?;
                let answer = measure(
                    &mut timings,
                    Phase::Readout(index),
                    Some(&text_device),
                    || plan.readout(&logits),
                )?;
                response.answers.0.push((name, answer));
            }
            Ok(())
        })();
        self.backend.clear_cache();
        if let (Some(timings), Some(started)) = (timings, started) {
            timings.total_ms = started.elapsed().as_secs_f64() * 1000.;
        }
        match result {
            Ok(()) => Ok(response),
            Err(error) => Err(DecisionFailure {
                message: error.to_string(),
                question_index,
                partial: Box::new(response),
            }),
        }
    }
}
