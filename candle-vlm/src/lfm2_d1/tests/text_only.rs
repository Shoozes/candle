use super::*;
use std::path::PathBuf;

struct Inputs(PathBuf);
impl Inputs {
    fn new() -> candle::Result<Self> {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!("candle-d1-text-{}-{id}", std::process::id()));
        std::fs::create_dir(&root)?;
        let inputs = Self(root);
        std::fs::write(inputs.0.join("text.gguf"), tiny_text_gguf()?)?;
        std::fs::copy(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../tests/fixtures/lfm2_d1_tiny/tokenizer.json"),
            inputs.0.join("tokenizer.json"),
        )?;
        Ok(inputs)
    }

    fn load(
        &self,
        device: &Device,
        mode: Lfm2EmbeddingLoadMode,
        limits: D1Limits,
    ) -> candle::Result<D1Session> {
        load_d1_text_q8_with_embedding_mode(
            D1TextLoadOptions {
                text_gguf: &self.0.join("text.gguf"),
                tokenizer: &self.0.join("tokenizer.json"),
                text_device: device,
                limits,
            },
            mode,
        )
    }
}
impl Drop for Inputs {
    fn drop(&mut self) {
        // This fresh test-owned directory contains only the two authored inputs.
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn d1_text_only_cpu_matches_hybrid_and_recovers() -> candle::Result<()> {
    check(&Device::Cpu)
}

#[cfg(feature = "cuda")]
#[test]
fn d1_text_only_cuda_matches_hybrid_and_recovers() -> candle::Result<()> {
    check(&Device::new_cuda(0)?)
}

fn check(device: &Device) -> candle::Result<()> {
    let inputs = Inputs::new()?;
    for mode in [
        Lfm2EmbeddingLoadMode::DenseF32,
        Lfm2EmbeddingLoadMode::Q8Rows,
    ] {
        let mut text = inputs.load(device, mode, D1Limits::default())?;
        let mut hybrid = tiny_session_with_embedding_mode(device, D1Limits::default(), mode)?;
        assert!(!text.supports_images());
        assert!(hybrid.supports_images());
        assert_eq!(
            text.consumed_files(),
            &[inputs.0.join("text.gguf"), inputs.0.join("tokenizer.json")]
        );
        assert_eq!(text.execution_details().vision_device, "none");
        assert_eq!(text.execution_details().mmproj_quantized_tensors, 0);
        assert!(text
            .execution_details()
            .source_float_mmproj_linears
            .is_empty());
        assert_eq!(
            text.text_embedding_storage()?,
            hybrid.text_embedding_storage()?
        );
        assert!(text.tokenizer().get_truncation().is_none());
        assert!(text.tokenizer().get_padding().is_none());
        let mut request = request();
        request.questions.0.push((
            "score".into(),
            Question::Score {
                instructions: "Rate clarity".into(),
                criteria: vec!["low".into(), "high".into()],
            },
        ));
        let mut expected = Vec::new();
        let mut actual = Vec::new();
        let mut expected_inputs = Vec::new();
        let mut actual_inputs = Vec::new();
        let a = hybrid
            .decide_traced(
                &request,
                &[],
                || false,
                |event| {
                    match event {
                        D1TraceEvent::AnswerLogits { logits, .. } => {
                            expected.push(logits.flatten_all()?.to_vec1::<f32>()?)
                        }
                        D1TraceEvent::QuestionInput { text, ids, .. } => {
                            expected_inputs.push((text.to_owned(), ids.to_vec()))
                        }
                        _ => panic!("text request emitted image trace"),
                    }
                    Ok(())
                },
            )
            .map_err(|e| candle::Error::Msg(e.to_string()))?;
        let b = text
            .decide_traced(
                &request,
                &[],
                || false,
                |event| {
                    match event {
                        D1TraceEvent::AnswerLogits { logits, .. } => {
                            actual.push(logits.flatten_all()?.to_vec1::<f32>()?)
                        }
                        D1TraceEvent::QuestionInput { text, ids, .. } => {
                            actual_inputs.push((text.to_owned(), ids.to_vec()))
                        }
                        _ => panic!("text session emitted image trace"),
                    }
                    Ok(())
                },
            )
            .map_err(|e| candle::Error::Msg(e.to_string()))?;
        assert_eq!(a.answers, b.answers);
        assert_eq!(a.usage, b.usage);
        assert_eq!(a.usage.language_forwards, 3);
        assert_eq!(b.usage.vision_forwards, 0);
        assert_eq!(b.usage.output_tokens, 0);
        assert_eq!(a.execution.cpu_q8_matmuls, b.execution.cpu_q8_matmuls);
        assert_eq!(a.execution.cuda_f32_q8, b.execution.cuda_f32_q8);
        assert_eq!(expected_inputs, actual_inputs);
        for (a, b) in expected.iter().zip(&actual) {
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        let partial = text
            .decide_traced(
                &request,
                &[],
                || false,
                |event| {
                    if matches!(event, D1TraceEvent::QuestionInput { index: 1, .. }) {
                        candle::bail!("authored second question failure")
                    }
                    Ok(())
                },
            )
            .expect_err("partial failure");
        assert_eq!(partial.partial.usage.language_forwards, 1);
        assert_eq!(partial.partial.unattempted_questions, 2);
        assert_eq!(
            text.decide_with_cancel(&request, &[], || true)
                .expect_err("cancel")
                .partial
                .usage
                .language_forwards,
            0
        );
        for _ in 0..3 {
            assert_eq!(
                text.decide(&request, &[])
                    .map_err(|e| candle::Error::Msg(e.to_string()))?
                    .answers,
                b.answers
            );
        }
        let completed = std::cell::Cell::new(false);
        let cancelled = text
            .decide_traced(
                &request,
                &[],
                || completed.get(),
                |event| {
                    if matches!(event, D1TraceEvent::AnswerLogits { index: 0, .. }) {
                        completed.set(true);
                    }
                    Ok(())
                },
            )
            .expect_err("cancel after first question");
        assert_eq!(cancelled.partial.usage.language_forwards, 1);
        assert_eq!(cancelled.partial.answers.0.len(), 1);
        assert_eq!(cancelled.partial.unattempted_questions, 2);
        assert_eq!(
            text.decide(&request, &[])
                .map_err(|e| candle::Error::Msg(e.to_string()))?
                .answers,
            b.answers
        );
        let boundary = actual_inputs
            .iter()
            .map(|(_, ids)| ids.len())
            .max()
            .expect("authored inputs");
        let mut at_limit = inputs.load(
            device,
            mode,
            D1Limits {
                max_context_tokens: Some(boundary),
                ..D1Limits::default()
            },
        )?;
        assert_eq!(
            at_limit
                .decide(&request, &[])
                .map_err(|e| candle::Error::Msg(e.to_string()))?
                .answers,
            b.answers
        );
        let mut below_limit = inputs.load(
            device,
            mode,
            D1Limits {
                max_context_tokens: Some(boundary - 1),
                ..D1Limits::default()
            },
        )?;
        assert_eq!(
            below_limit
                .decide(&request, &[])
                .expect_err("exact context boundary rejected")
                .partial
                .usage,
            DecisionUsage::default()
        );
        let image = image::DynamicImage::new_rgb8(1, 1);
        let failed = text.decide(&request, &[image]).expect_err("image rejected");
        assert!(failed.message.contains("text-only"));
        assert_eq!(failed.partial.usage, DecisionUsage::default());
        request.state = State::Text("<image>".into());
        let failed = text.decide(&request, &[]).expect_err("sentinel rejected");
        assert!(failed.message.contains("image sentinel"));
        assert_eq!(failed.partial.usage, DecisionUsage::default());
        let mut limited = inputs.load(
            device,
            mode,
            D1Limits {
                max_context_tokens: Some(1),
                ..D1Limits::default()
            },
        )?;
        assert_eq!(
            limited
                .decide(&super::request(), &[])
                .expect_err("context bound")
                .partial
                .usage,
            DecisionUsage::default()
        );
    }
    device.synchronize()?;
    Ok(())
}

#[test]
fn d1_text_only_rejects_nonatomic_image_markers() -> candle::Result<()> {
    let inputs = Inputs::new()?;
    let path = inputs.0.join("tokenizer.json");
    let mut tokenizer: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path)?).map_err(candle::Error::wrap)?;
    tokenizer["added_tokens"]
        .as_array_mut()
        .expect("fixture tokens")
        .retain(|token| {
            !token["content"]
                .as_str()
                .expect("fixture content")
                .contains("image")
                && !token["content"]
                    .as_str()
                    .expect("fixture content")
                    .contains("img_")
        });
    tokenizer["model"]["vocab"]
        .as_object_mut()
        .expect("fixture vocab")
        .retain(|token, _| !token.contains("image") && !token.contains("img_"));
    std::fs::write(
        path,
        serde_json::to_vec(&tokenizer).map_err(candle::Error::wrap)?,
    )?;
    let mut session = inputs.load(
        &Device::Cpu,
        Lfm2EmbeddingLoadMode::Q8Rows,
        D1Limits::default(),
    )?;
    for marker in [
        "<image>",
        "<|image_start|>",
        "<|image_end|>",
        "<|img_thumbnail|>",
        "<|img_row_1_col_1|>",
    ] {
        let mut request = request();
        request.state = State::Text(marker.into());
        let failure = session
            .decide(&request, &[])
            .expect_err("image sentinel rejected");
        assert!(failure.message.contains("image sentinel"));
        assert_eq!(failure.partial.usage, DecisionUsage::default());
    }
    Ok(())
}

#[test]
fn d1_text_only_rejects_resolved_tile_sentinel() -> candle::Result<()> {
    let inputs = Inputs::new()?;
    let path = inputs.0.join("tokenizer.json");
    let mut tokenizer: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path)?).map_err(candle::Error::wrap)?;
    tokenizer["normalizer"] = serde_json::json!({"type":"Replace","pattern":{"String":"tile"},"content":"<|img_row_1_col_1|>"});
    tokenizer["added_tokens"].as_array_mut().expect("fixture tokens").push(serde_json::json!({
        "id":22,"content":"<|img_row_1_col_1|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":true
    }));
    std::fs::write(
        path,
        serde_json::to_vec(&tokenizer).map_err(candle::Error::wrap)?,
    )?;
    let mut session = inputs.load(
        &Device::Cpu,
        Lfm2EmbeddingLoadMode::Q8Rows,
        D1Limits::default(),
    )?;
    assert_eq!(
        session
            .tokenizer()
            .encode("tile", false)
            .map_err(candle::Error::wrap)?
            .get_ids(),
        &[22]
    );
    let mut request = request();
    request.state = State::Text("tile".into());
    let failure = session
        .decide(&request, &[])
        .expect_err("resolved tile rejected");
    assert!(failure.message.contains("image sentinel"));
    assert_eq!(failure.partial.usage, DecisionUsage::default());
    Ok(())
}

#[test]
fn d1_text_only_loader_rejects_limits_and_malformed_inputs() -> candle::Result<()> {
    let inputs = Inputs::new()?;
    let device = Device::Cpu;
    let default = load_d1_text_q8(D1TextLoadOptions {
        text_gguf: &inputs.0.join("text.gguf"),
        tokenizer: &inputs.0.join("tokenizer.json"),
        text_device: &device,
        limits: D1Limits::default(),
    })?;
    assert_eq!(
        default.text_embedding_storage()?.mode,
        Lfm2EmbeddingLoadMode::DenseF32
    );
    let invalid = inputs
        .load(
            &device,
            Lfm2EmbeddingLoadMode::Q8Rows,
            D1Limits {
                max_questions: 0,
                ..D1Limits::default()
            },
        )
        .err()
        .expect("limits rejected");
    assert!(invalid.to_string().contains("limits must be positive"));
    std::fs::write(inputs.0.join("text.gguf"), b"invalid GGUF")?;
    assert!(inputs
        .load(
            &device,
            Lfm2EmbeddingLoadMode::DenseF32,
            D1Limits::default()
        )
        .is_err());
    std::fs::write(inputs.0.join("tokenizer.json"), b"{}")?;
    let error = inputs
        .load(
            &device,
            Lfm2EmbeddingLoadMode::DenseF32,
            D1Limits::default(),
        )
        .err()
        .expect("invalid tokenizer rejected");
    assert!(error.to_string().contains("tokenizer"));
    Ok(())
}
