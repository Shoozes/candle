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
