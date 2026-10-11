use super::*;

fn weight(device: &Device) -> Result<Arc<QTensor>> {
    let values: Vec<f32> = (0..8 * 256)
        .map(|i| ((i % 173) as f32 - 86.) / 37.)
        .collect();
    let dense = Tensor::from_vec(values, (8, 256), device)?;
    Ok(Arc::new(QTensor::quantize(&dense, GgmlDType::Q8_0)?))
}

fn check_rows(device: &Device) -> Result<()> {
    let weight = weight(device)?;
    let dense = TokenEmbedding::new(Arc::clone(&weight), Lfm2EmbeddingLoadMode::DenseF32)?;
    let rows = TokenEmbedding::new(Arc::clone(&weight), Lfm2EmbeddingLoadMode::Q8Rows)?;
    let ids = Tensor::new(&[[5u32, 3], [1, 3], [7, 0]], device)?;
    for ids in [
        ids.clone(),
        ids.t()?,
        ids.narrow(0, 1, 2)?,
        ids.to_dtype(DType::I64)?,
    ] {
        let actual = rows.forward(&ids)?;
        let expected = dense.forward(&ids)?;
        assert_eq!(actual.dtype(), DType::F32);
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(
            actual
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }
    let empty = Tensor::zeros((2, 0), DType::U32, device)?;
    assert_eq!(rows.forward(&empty)?.dims(), &[2, 0, 256]);
    for ids in [
        Tensor::new(&[8u32], device)?,
        Tensor::new(&[-1i64], device)?,
        Tensor::new(&[i64::MAX], device)?,
        Tensor::new(&[u32::MAX], device)?,
        Tensor::new(&[0f32], device)?,
    ] {
        assert!(rows.forward(&ids).is_err());
    }
    device.synchronize()?;
    let head = QMatMul::QTensor(Arc::clone(&weight));
    let storage = rows.storage(&head)?;
    assert_eq!(storage.mode, Lfm2EmbeddingLoadMode::Q8Rows);
    assert_eq!(storage.table_bytes, 8 * 256 / 32 * 34);
    assert!(storage.shared_with_output);
    assert_eq!(storage.additional_bytes, 0);
    let separate = QMatMul::QTensor(super::tests::weight(device)?);
    assert_eq!(
        rows.storage(&separate)?.additional_bytes,
        storage.table_bytes
    );
    let storage = dense.storage(&head)?;
    assert_eq!(storage.table_bytes, 8 * 256 * 4);
    assert_eq!(storage.additional_bytes, storage.table_bytes);
    assert!(!storage.shared_with_output);
    Ok(())
}

#[test]
fn q8_rows_cpu_match_dense_exactly_and_reject_bad_ids() -> Result<()> {
    check_rows(&Device::Cpu)
}

#[cfg(feature = "cuda")]
#[test]
fn q8_rows_cuda_match_dense_exactly_and_reject_bad_ids() -> Result<()> {
    check_rows(&Device::new_cuda(0)?)
}

#[test]
fn q8_rows_layout_rejects_invalid_storage_dimensions_and_overflow() {
    for shape in [
        Shape::from((0, 32)),
        Shape::from((8, 0)),
        Shape::from((8, 31)),
        Shape::from((8, usize::MAX - 31)),
        Shape::from((usize::MAX, 32)),
        Shape::from(32),
    ] {
        assert!(q8_layout(&shape, GgmlDType::Q8_0, &Device::Cpu).is_err());
    }
    assert!(q8_layout(&Shape::from((8, 32)), GgmlDType::F32, &Device::Cpu).is_err());
}

#[test]
fn q8_rows_header_rejects_before_payload_read() -> Result<()> {
    let bytes = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tests/fixtures/lfm2_vl_loader_tiny/text.gguf"
    ));
    let mut reader = std::io::Cursor::new(bytes.as_slice());
    let content = gguf_file::Content::read(&mut reader)?;
    let position = reader.position();
    assert!(
        crate::models::quantized_lfm2::ModelWeights::from_gguf_with_embedding_mode(
            content,
            &mut reader,
            &Device::Cpu,
            Lfm2EmbeddingLoadMode::Q8Rows
        )
        .is_err()
    );
    assert_eq!(reader.position(), position);
    let mut content = gguf_file::Content::read(&mut std::io::Cursor::new(bytes.as_slice()))?;
    let info = content
        .tensor_infos
        .get_mut(NAMES[0])
        .expect("embedding header");
    info.shape = Shape::from((8, 32));
    info.ggml_dtype = GgmlDType::Q8_0;
    assert!(validate_header(
        &content,
        &Device::Cpu,
        Lfm2EmbeddingLoadMode::Q8Rows,
        12,
        None
    )
    .is_err());
    assert!(validate_header(
        &content,
        &Device::Cpu,
        Lfm2EmbeddingLoadMode::Q8Rows,
        32,
        Some(8)
    )
    .is_err());
    assert!(validate_header(
        &content,
        &Device::Cpu,
        Lfm2EmbeddingLoadMode::Q8Rows,
        32,
        Some(7)
    )
    .is_ok());
    Ok(())
}

#[test]
fn embedding_modes_have_stable_explicit_names() -> Result<()> {
    assert_eq!(
        Lfm2EmbeddingLoadMode::default(),
        Lfm2EmbeddingLoadMode::DenseF32
    );
    for mode in [
        Lfm2EmbeddingLoadMode::DenseF32,
        Lfm2EmbeddingLoadMode::Q8Rows,
    ] {
        assert_eq!(mode.to_string().parse::<Lfm2EmbeddingLoadMode>()?, mode);
        assert_eq!(
            serde_json::to_value(mode).map_err(candle::Error::wrap)?,
            mode.to_string()
        );
    }
    assert!("auto".parse::<Lfm2EmbeddingLoadMode>().is_err());
    Ok(())
}

#[test]
fn q8_rows_rejects_malformed_buffer_before_lookup() -> Result<()> {
    use candle::quantized::{
        k_quants::{BlockQ8_0, GgmlType},
        QStorage,
    };
    let storage = QStorage::Cpu(Box::new(vec![BlockQ8_0::zeros()]));
    let malformed = Arc::new(QTensor::new(storage, (2, 32))?);
    assert!(TokenEmbedding::new(malformed, Lfm2EmbeddingLoadMode::Q8Rows).is_err());
    Ok(())
}
