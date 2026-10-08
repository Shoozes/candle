use super::{
    D1Limits, D1PolicyV1, DecisionFailure, DecisionRequest, DecisionResponse, ExecutionReport,
    PreparedQuestion,
};
use crate::lfm2_vl::{
    load_lfm2_vl_hybrid, Lfm2VlHybridLoadOptions, Lfm2VlMmprojExecution, LoadedLfm2VlHybrid,
};
use candle::quantized::with_native_q8_0;
use candle::{DType, IndexOp, Result, Tensor};
use image::DynamicImage;
use std::collections::HashSet;
use std::path::PathBuf;

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

pub struct D1LoadOptions<'a> {
    pub hybrid: Lfm2VlHybridLoadOptions<'a>,
    pub limits: D1Limits,
}

pub struct D1Session {
    loaded: LoadedLfm2VlHybrid,
    policy: D1PolicyV1,
    limits: D1Limits,
    tokenizer: tokenizers::Tokenizer,
    text_quantized_linears: usize,
}

pub fn load_d1_q8(options: D1LoadOptions<'_>) -> Result<D1Session> {
    for name in ["CANDLE_DEQUANTIZE_ALL", "CANDLE_DEQUANTIZE_ALL_F16"] {
        if std::env::var_os(name).is_some_and(|value| !value.is_empty() && value != "0") {
            candle::bail!("d1 native Q8 loading rejects {name}")
        }
    }
    if options.hybrid.mmproj_execution != Lfm2VlMmprojExecution::Q8
        || options.hybrid.vision_dtype != DType::F32
    {
        candle::bail!("d1 loading requires explicit Q8 MMProj execution and F32 activations")
    }
    D1Session::from_hybrid(
        load_lfm2_vl_hybrid(options.hybrid)?,
        D1PolicyV1,
        options.limits,
    )
}

impl D1Session {
    pub fn from_hybrid(
        mut loaded: LoadedLfm2VlHybrid,
        policy: D1PolicyV1,
        limits: D1Limits,
    ) -> Result<Self> {
        if limits.max_questions == 0
            || limits.max_options == 0
            || limits.max_prompt_bytes == 0
            || limits.max_context_tokens == Some(0)
        {
            candle::bail!("d1 limits must be positive")
        }
        for device in [loaded.model.text_device(), loaded.model.vision_device()] {
            if !device.is_cpu() && !device.is_cuda() {
                candle::bail!("d1 supports CPU and CUDA devices")
            }
        }
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
        let mut tokenizer = loaded.prompt.tokenizer().clone();
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
        Ok(Self {
            loaded,
            policy,
            limits,
            tokenizer,
            text_quantized_linears,
        })
    }

    pub fn consumed_files(&self) -> &[PathBuf] {
        &self.loaded.consumed_files
    }
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }
    pub fn limits(&self) -> &D1Limits {
        &self.limits
    }

    pub fn execution_details(&self) -> ExecutionReport {
        ExecutionReport {
            policy: D1PolicyV1::ID.into(),
            text_device: if self.loaded.model.text_device().is_cpu() {
                "cpu"
            } else {
                "cuda"
            }
            .into(),
            vision_device: if self.loaded.model.vision_device().is_cpu() {
                "cpu"
            } else {
                "cuda"
            }
            .into(),
            mmproj_quantized_tensors: self.loaded.model.mmproj().native_quantized_tensor_count(),
            text_quantized_linears: self.text_quantized_linears,
            source_float_mmproj_linears: self
                .loaded
                .model
                .mmproj()
                .source_float_linear_names()
                .to_vec(),
            ..Default::default()
        }
    }

    pub fn prepare(
        &self,
        request: &DecisionRequest,
        image_count: usize,
    ) -> Result<Vec<(String, PreparedQuestion, String)>> {
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
        mut observe: impl FnMut(D1TraceEvent<'_>) -> Result<()>,
    ) -> std::result::Result<DecisionResponse, DecisionFailure> {
        let mut response = DecisionResponse {
            planned_questions: request.questions.0.len(),
            unattempted_questions: request.questions.0.len(),
            execution: self.execution_details(),
            ..Default::default()
        };
        let mut question_index = None;
        let result = (|| -> Result<()> {
            let plans = self.prepare(request, images.len())?;
            if cancelled() {
                candle::bail!("d1 request cancelled before preparation")
            }
            let processed = if images.is_empty() {
                None
            } else {
                let capped = images
                    .iter()
                    .map(|image| super::image_cap::cap_pixels(image, super::image_cap::MAX_PIXELS))
                    .collect::<Result<Vec<_>>>()?;
                Some(
                    self.loaded
                        .processor
                        .process(&capped, self.loaded.model.vision_device())?,
                )
            };
            if let Some(processed) = &processed {
                observe(D1TraceEvent::ProcessedImages(processed))?;
            }
            let mut expanded = Vec::new();
            for (_, _, rendered) in &plans {
                let (ids, spans) = match &processed {
                    Some(processed) => {
                        let prompt = self.loaded.prompt.expand(rendered, processed)?;
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
                let limit = self
                    .limits
                    .max_context_tokens
                    .unwrap_or(self.loaded.model.context_length())
                    .min(self.loaded.model.context_length());
                if ids.is_empty() || ids.len() > limit {
                    candle::bail!("d1 prompt has {} tokens outside 1..={limit}", ids.len())
                }
                expanded.push((ids, spans));
            }
            let encoded = match &processed {
                Some(processed) => {
                    if cancelled() {
                        candle::bail!("d1 request cancelled before vision forward")
                    }
                    response.usage.vision_forwards = 1;
                    let (result, dispatch) = with_native_q8_0(|| {
                        self.loaded.model.encode_images_with_limits(
                            processed,
                            1,
                            &self.loaded.processor.config().vision_limits,
                        )
                    });
                    response.execution.accumulate(dispatch)?;
                    let encoded = result?;
                    observe(D1TraceEvent::ImageFeatures(&encoded))?;
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
                observe(D1TraceEvent::QuestionInput {
                    index,
                    text: &rendered,
                    ids: &ids,
                })?;
                let input =
                    Tensor::new(ids.as_slice(), self.loaded.model.text_device())?.unsqueeze(0)?;
                response.unattempted_questions -= 1;
                response.usage.language_forwards += 1;
                response.usage.input_tokens = response
                    .usage
                    .input_tokens
                    .checked_add(ids.len() as u64)
                    .ok_or_else(|| candle::Error::Msg("d1 input token counter overflow".into()))?;
                let (result, dispatch) = with_native_q8_0(|| {
                    self.loaded.model.prefill(&input, &spans, encoded.as_ref())
                });
                response.execution.accumulate(dispatch)?;
                let logits = result?.i(0)?;
                observe(D1TraceEvent::AnswerLogits {
                    index,
                    logits: &logits,
                })?;
                response.answers.0.push((name, plan.readout(&logits)?));
            }
            Ok(())
        })();
        self.loaded.model.clear_cache();
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
