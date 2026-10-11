use super::*;

#[test]
fn d1_q8_rows_cpu_preserves_logits_readout_images_and_counts() -> candle::Result<()> {
    check_session(&Device::Cpu)
}

#[cfg(feature = "cuda")]
#[test]
fn d1_q8_rows_cuda_preserves_logits_readout_images_and_counts() -> candle::Result<()> {
    check_session(&Device::new_cuda(0)?)
}

fn check_session(device: &Device) -> candle::Result<()> {
    let mut dense = tiny_session_with_embedding_mode(
        device,
        D1Limits::default(),
        Lfm2EmbeddingLoadMode::DenseF32,
    )?;
    let mut rows = tiny_session_with_embedding_mode(
        device,
        D1Limits::default(),
        Lfm2EmbeddingLoadMode::Q8Rows,
    )?;
    let dense_storage = dense.text_embedding_storage()?;
    let row_storage = rows.text_embedding_storage()?;
    assert_eq!(row_storage.mode, Lfm2EmbeddingLoadMode::Q8Rows);
    assert!(row_storage.table_bytes < dense_storage.table_bytes);
    assert_eq!(dense.consumed_files(), rows.consumed_files());
    let mut request = request();
    request.questions.0.push((
        "score".into(),
        Question::Score {
            instructions: "Rate clarity".into(),
            criteria: vec!["unclear".into(), "clear".into()],
        },
    ));
    let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        8,
        8,
        image::Rgb([64, 128, 192]),
    ));
    for images in [vec![], vec![image], vec![]] {
        let mut dense_logits = Vec::new();
        let mut row_logits = Vec::new();
        let expected = dense
            .decide_traced(
                &request,
                &images,
                || false,
                |event| {
                    if let D1TraceEvent::AnswerLogits { logits, .. } = event {
                        dense_logits.push(logits.flatten_all()?.to_vec1::<f32>()?);
                    }
                    Ok(())
                },
            )
            .map_err(|error| candle::Error::Msg(error.to_string()))?;
        let actual = rows
            .decide_traced(
                &request,
                &images,
                || false,
                |event| {
                    if let D1TraceEvent::AnswerLogits { logits, .. } = event {
                        row_logits.push(logits.flatten_all()?.to_vec1::<f32>()?);
                    }
                    Ok(())
                },
            )
            .map_err(|error| candle::Error::Msg(error.to_string()))?;
        assert_eq!(actual, expected);
        assert_eq!(actual.usage.language_forwards, 3);
        assert_eq!(actual.usage.vision_forwards, u64::from(!images.is_empty()));
        assert_eq!(actual.usage.output_tokens, 0);
        assert_eq!(dense_logits.len(), 3);
        for (expected, actual) in dense_logits.iter().zip(&row_logits) {
            assert_eq!(
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
    let failed = rows
        .decide_traced(
            &request,
            &[],
            || false,
            |event| {
                if matches!(event, D1TraceEvent::QuestionInput { index: 1, .. }) {
                    candle::bail!("authored second-question failure")
                }
                Ok(())
            },
        )
        .expect_err("authored partial failure");
    assert_eq!(failed.partial.usage.language_forwards, 1);
    assert_eq!(failed.partial.unattempted_questions, 2);
    assert_eq!(failed.partial.usage.output_tokens, 0);
    let cancelled = rows
        .decide_with_cancel(&request, &[], || true)
        .expect_err("cancelled");
    assert_eq!(cancelled.partial.usage.language_forwards, 0);
    assert_eq!(
        rows.decide(&request, &[])
            .map_err(|e| candle::Error::Msg(e.to_string()))?,
        dense
            .decide(&request, &[])
            .map_err(|e| candle::Error::Msg(e.to_string()))?
    );
    let mut limited = tiny_session_with_embedding_mode(
        device,
        D1Limits {
            max_context_tokens: Some(1),
            ..Default::default()
        },
        Lfm2EmbeddingLoadMode::Q8Rows,
    )?;
    let failed = limited
        .decide(&request, &[])
        .expect_err("context admission");
    assert_eq!(failed.partial.usage.language_forwards, 0);
    assert_eq!(failed.partial.usage.vision_forwards, 0);
    device.synchronize()?;
    Ok(())
}
