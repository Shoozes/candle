impl Mmproj {
    /// Load a llama.cpp-compatible LFM2-VL MMProj GGUF from one stable file handle.
    pub fn load_gguf(
        path: impl AsRef<Path>,
        dtype: DType,
        device: &Device,
        image_token_id: u32,
    ) -> Result<Self> {
        Self::load_gguf_with_execution(
            path,
            dtype,
            device,
            image_token_id,
            RequestedExecution::Dense,
        )
    }

    /// Load a GGUF MMProj and retain supported Q8_0 linear weights for native matmul.
    pub fn load_gguf_q8(
        path: impl AsRef<Path>,
        dtype: DType,
        device: &Device,
        image_token_id: u32,
    ) -> Result<Self> {
        Self::load_gguf_with_execution(path, dtype, device, image_token_id, RequestedExecution::Q8)
    }

    /// Prefer native Q8_0 execution when the artifact and activation dtype support it.
    pub fn load_gguf_auto(
        path: impl AsRef<Path>,
        dtype: DType,
        device: &Device,
        image_token_id: u32,
    ) -> Result<Self> {
        Self::load_gguf_with_execution(
            path,
            dtype,
            device,
            image_token_id,
            RequestedExecution::Auto,
        )
    }

    fn load_gguf_with_execution(
        path: impl AsRef<Path>,
        dtype: DType,
        device: &Device,
        image_token_id: u32,
        requested_execution: RequestedExecution,
    ) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path).map_err(|error| {
            candle::Error::Msg(format!(
                "failed to open GGUF MMProj {}: {error}",
                path.display()
            ))
        })?;
        Self::from_gguf_with_execution(
            &mut file,
            dtype,
            device,
            image_token_id,
            requested_execution,
        )
        .map_err(|error| error.with_path(path))
    }

    /// Load a GGUF MMProj from a seekable reader using the dense compatibility path.
    pub fn from_gguf<R: Read + Seek>(
        reader: &mut R,
        dtype: DType,
        device: &Device,
        image_token_id: u32,
    ) -> Result<Self> {
        Self::from_gguf_with_execution(
            reader,
            dtype,
            device,
            image_token_id,
            RequestedExecution::Dense,
        )
    }

    /// Load a GGUF MMProj from a seekable reader using native Q8_0 linears.
    pub fn from_gguf_q8<R: Read + Seek>(
        reader: &mut R,
        dtype: DType,
        device: &Device,
        image_token_id: u32,
    ) -> Result<Self> {
        Self::from_gguf_with_execution(
            reader,
            dtype,
            device,
            image_token_id,
            RequestedExecution::Q8,
        )
    }

    /// Auto-select native Q8_0 execution or the dense compatibility path.
    pub fn from_gguf_auto<R: Read + Seek>(
        reader: &mut R,
        dtype: DType,
        device: &Device,
        image_token_id: u32,
    ) -> Result<Self> {
        Self::from_gguf_with_execution(
            reader,
            dtype,
            device,
            image_token_id,
            RequestedExecution::Auto,
        )
    }

    fn from_gguf_with_execution<R: Read + Seek>(
        reader: &mut R,
        dtype: DType,
        device: &Device,
        image_token_id: u32,
        requested_execution: RequestedExecution,
    ) -> Result<Self> {
        let dense_element_size = match dtype {
            DType::F32 => 4u64,
            DType::F16 | DType::BF16 => 2u64,
            _ => candle::bail!(
                "GGUF MMProj dense compatibility dtype must be F32, F16, or BF16, got {dtype:?}"
            ),
        };
        let file_size = reader.seek(SeekFrom::End(0))?;
        if file_size == 0 || file_size > MAX_GGUF_FILE_BYTES {
            candle::bail!(
                "GGUF MMProj file size {file_size} is outside 1..={MAX_GGUF_FILE_BYTES} bytes"
            )
        }
        reader.seek(SeekFrom::Start(0))?;
        let content = gguf_file::Content::read_with_limits(
            reader,
            gguf_file::ContentReadLimits {
                max_tensor_count: MAX_GGUF_MMPROJ_TENSORS as u64,
                max_metadata_count: MAX_GGUF_MMPROJ_METADATA,
                max_string_length: MAX_GGUF_MMPROJ_STRING_BYTES,
                max_array_elements: MAX_GGUF_MMPROJ_ARRAY_ELEMENTS,
                max_header_bytes: MAX_GGUF_MMPROJ_HEADER_BYTES,
            },
        )?;
        if content.tensor_infos.is_empty() || content.tensor_infos.len() > MAX_GGUF_MMPROJ_TENSORS {
            candle::bail!(
                "GGUF MMProj tensor count {} is outside 1..={MAX_GGUF_MMPROJ_TENSORS}",
                content.tensor_infos.len()
            )
        }

        let parsed = parse_metadata(&content)?;
        let config = reconstruct_config(&content, &parsed, image_token_id)?;
        let expected = expected_tensors(&config)?;
        let report = inspect_inventory(&content, &expected, dtype, device)?;
        report.require_clean()?;
        let (execution, native_quantized_tensor_count) =
            select_execution(&content, &expected, dtype, requested_execution)?;
        let allocations =
            validate_ranges_and_sizes(&content, &expected, file_size, dense_element_size)?;

        let mut native_tensors = HashMap::new();
        native_tensors.try_reserve(expected.len()).map_err(|_| {
            candle::Error::Msg("GGUF MMProj dense tensor-map allocation failed".into())
        })?;
        let mut vision_quantized = HashMap::new();
        let mut projector_quantized = HashMap::new();
        vision_quantized
            .try_reserve(native_quantized_tensor_count)
            .map_err(|_| {
                candle::Error::Msg("GGUF MMProj Q8 tensor-map allocation failed".into())
            })?;
        projector_quantized
            .try_reserve(native_quantized_tensor_count)
            .map_err(|_| {
                candle::Error::Msg("GGUF MMProj Q8 tensor-map allocation failed".into())
            })?;
        for (gguf_name, expected_tensor) in &expected {
            let quantized = content.tensor(reader, gguf_name, device).map_err(|error| {
                candle::Error::Msg(format!(
                    "failed to read GGUF MMProj tensor {gguf_name:?}: {error}"
                ))
            })?;
            if execution == GgufMmprojExecution::Q8_0
                && expected_tensor.quantized_linear
                && quantized.dtype() == GgmlDType::Q8_0
            {
                let (target, relative_name) = if let Some(name) =
                    relative_native_name(&expected_tensor.native_name, NATIVE_VISION_ROOT)
                {
                    (&mut vision_quantized, name)
                } else if let Some(name) =
                    relative_native_name(&expected_tensor.native_name, NATIVE_PROJECTOR_ROOT)
                {
                    (&mut projector_quantized, name)
                } else {
                    candle::bail!(
                        "GGUF MMProj Q8 tensor {:?} is outside the vision/projector roots",
                        expected_tensor.native_name
                    )
                };
                if target
                    .insert(relative_name.to_string(), quantized)
                    .is_some()
                {
                    candle::bail!(
                        "GGUF MMProj names normalize to duplicate Q8 tensor {relative_name:?}"
                    )
                }
                continue;
            }
            let mut tensor = quantized.dequantize(device)?.to_dtype(dtype)?;
            if expected_tensor.patch_layout {
                let [vision_hidden, channels, patch_rows, patch_cols] =
                    tensor.dims().try_into().map_err(|_| {
                        candle::Error::Msg(format!(
                            "GGUF MMProj patch tensor {gguf_name:?} must have rank 4"
                        ))
                    })?;
                let packed_width = channels
                    .checked_mul(patch_rows)
                    .and_then(|value| value.checked_mul(patch_cols))
                    .ok_or_else(|| {
                        candle::Error::Msg(
                            "GGUF MMProj patch packed-width calculation overflowed".into(),
                        )
                    })?;
                tensor = tensor
                    .permute((0, 2, 3, 1))?
                    .contiguous()?
                    .reshape((vision_hidden, packed_width))?;
            }
            if native_tensors
                .insert(expected_tensor.native_name.clone(), tensor)
                .is_some()
            {
                candle::bail!(
                    "GGUF MMProj names normalize to duplicate native tensor {:?}",
                    expected_tensor.native_name
                )
            }
        }

        let vb = VarBuilder::from_tensors(native_tensors, dtype, device);
        let (vision_tower, projector) = match execution {
            GgufMmprojExecution::DenseCompatibility => (
                siglip2::Siglip2VisionModel::new(&config.vision_config, vb.pp(NATIVE_VISION_ROOT))?,
                Lfm2VlProjector::from_mmproj_config(&config, vb.pp(NATIVE_PROJECTOR_ROOT))?,
            ),
            GgufMmprojExecution::Q8_0 => (
                siglip2::Siglip2VisionModel::new_with_quantized_linears(
                    &config.vision_config,
                    vb.pp(NATIVE_VISION_ROOT),
                    vision_quantized,
                )?,
                Lfm2VlProjector::from_mmproj_config_with_quantized_linears(
                    &config,
                    vb.pp(NATIVE_PROJECTOR_ROOT),
                    projector_quantized,
                )?,
            ),
        };
        let tensor_dtypes: BTreeMap<_, _> = content
            .tensor_infos
            .iter()
            .map(|(name, info)| (name.clone(), format!("{:?}", info.ggml_dtype)))
            .collect();
        let quantized_tensor_count = content
            .tensor_infos
            .values()
            .filter(|info| {
                !matches!(
                    info.ggml_dtype,
                    GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
                )
            })
            .count();
        let gguf_metadata = GgufMmprojMetadata {
            general_architecture: "clip".to_string(),
            projector_type: "lfm2".to_string(),
            general_name: parsed.general_name.clone(),
            vision_layer_count: parsed.vision_layer_count,
            image_size: parsed.image_size,
            image_mean: parsed.image_mean,
            image_std: parsed.image_std,
            tensor_count: content.tensor_infos.len(),
            quantized_tensor_count,
            source_byte_count: allocations.source_byte_count,
            dense_byte_count: allocations.dense_byte_count,
            estimated_peak_byte_count: allocations.estimated_peak_byte_count,
            tensor_dtypes,
        };
        let processor = processor_metadata_json(&parsed);
        let metadata = MmprojMetadata {
            architecture: "lfm2_vl".to_string(),
            vision_hidden_size: config.vision_config.hidden_size,
            text_hidden_size: config.text_hidden_size,
            patch_size: config.vision_config.patch_size,
            downsample_factor: config.downsample_factor,
            image_token_id: config.image_token_id,
            use_image_special_tokens: config.use_image_special_tokens,
            expected_text_layer_count: None,
            processor,
            source_model: parsed.general_name,
            source_revision: None,
            manifest: None,
            gguf: Some(gguf_metadata),
        };
        Ok(Mmproj::from_parts(
            vision_tower,
            projector,
            config,
            metadata,
            report,
            dtype,
            device,
            Some(execution),
            native_quantized_tensor_count,
        ))
    }
}

fn select_execution(
    content: &gguf_file::Content,
    expected: &BTreeMap<String, ExpectedTensor>,
    dtype: DType,
    requested: RequestedExecution,
) -> Result<(GgufMmprojExecution, usize)> {
    match requested {
        RequestedExecution::Dense => Ok((GgufMmprojExecution::DenseCompatibility, 0)),
        RequestedExecution::Q8 => {
            let count = validate_native_q8_tensors(content, expected, dtype)?;
            Ok((GgufMmprojExecution::Q8_0, count))
        }
        RequestedExecution::Auto => {
            let has_q8 = expected
                .keys()
                .any(|name| content.tensor_infos[name].ggml_dtype == GgmlDType::Q8_0);
            if dtype == DType::F32 && has_q8 {
                let count = validate_native_q8_tensors(content, expected, dtype)?;
                Ok((GgufMmprojExecution::Q8_0, count))
            } else {
                Ok((GgufMmprojExecution::DenseCompatibility, 0))
            }
        }
    }
}

fn validate_native_q8_tensors(
    content: &gguf_file::Content,
    expected: &BTreeMap<String, ExpectedTensor>,
    dtype: DType,
) -> Result<usize> {
    if dtype != DType::F32 {
        candle::bail!(
            "GGUF MMProj native Q8_0 execution currently requires F32 activations, got {dtype:?}"
        )
    }
    let mut count = 0usize;
    for (name, expected_tensor) in expected {
        let info = &content.tensor_infos[name];
        match info.ggml_dtype {
            GgmlDType::Q8_0 if expected_tensor.quantized_linear => {
                let input_width = *info.shape.dims().last().ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "GGUF MMProj Q8_0 linear tensor {name:?} has no input dimension"
                    ))
                })?;
                let block_size = GgmlDType::Q8_0.block_size();
                if !input_width.is_multiple_of(block_size) {
                    candle::bail!(
                        "GGUF MMProj Q8_0 linear tensor {name:?} input width {input_width} is not divisible by block size {block_size}"
                    )
                }
                count = count.checked_add(1).ok_or_else(|| {
                    candle::Error::Msg("GGUF MMProj native Q8 tensor count overflowed".into())
                })?;
            }
            GgmlDType::Q8_0 => {
                candle::bail!("GGUF MMProj tensor {name:?} is Q8_0 but its role must remain dense")
            }
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => {}
            other => candle::bail!(
                "GGUF MMProj native Q8_0 execution does not support {other:?} tensor {name:?}"
            ),
        }
    }
    if count == 0 {
        candle::bail!("GGUF MMProj native Q8_0 execution requires at least one Q8_0 linear tensor")
    }
    Ok(count)
}

fn relative_native_name<'a>(name: &'a str, root: &str) -> Option<&'a str> {
    name.strip_prefix(root)?.strip_prefix('.')
}
