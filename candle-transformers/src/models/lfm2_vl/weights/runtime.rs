pub struct Mmproj {
    pub vision_tower: siglip2::Siglip2VisionModel,
    pub projector: Lfm2VlProjector,
    pub metadata: MmprojMetadata,
    pub report: MmprojLoadReport,
    config: Lfm2VlMmprojConfig,
    device: Device,
    dtype: DType,
    gguf_execution: Option<GgufMmprojExecution>,
    native_quantized_tensor_count: usize,
}

impl Mmproj {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_parts(
        vision_tower: siglip2::Siglip2VisionModel,
        projector: Lfm2VlProjector,
        config: Lfm2VlMmprojConfig,
        metadata: MmprojMetadata,
        report: MmprojLoadReport,
        dtype: DType,
        device: &Device,
        gguf_execution: Option<GgufMmprojExecution>,
        native_quantized_tensor_count: usize,
    ) -> Self {
        Self {
            vision_tower,
            projector,
            metadata,
            report,
            config,
            device: device.clone(),
            dtype,
            gguf_execution,
            native_quantized_tensor_count,
        }
    }

    pub fn load(bundle_dir: impl AsRef<Path>, dtype: DType, device: &Device) -> Result<Self> {
        let bundle_dir = bundle_dir.as_ref();
        Self::from_files(
            bundle_dir.join("mmproj.safetensors"),
            bundle_dir.join("mmproj.json"),
            bundle_dir.join("processor_config.json"),
            dtype,
            device,
        )
    }

    pub fn from_files(
        weights_path: impl AsRef<Path>,
        manifest_path: impl AsRef<Path>,
        processor_path: impl AsRef<Path>,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let weights_path = weights_path.as_ref();
        let manifest_path = manifest_path.as_ref();
        let processor_path = processor_path.as_ref();
        let manifest_json = read_bounded_text(manifest_path, "split MMProj manifest")?;
        let manifest = MmprojManifest::from_json(&manifest_json)?;
        let weights_bytes = read_weight_bytes(weights_path, "split MMProj safetensors", &manifest)?;
        verify_bytes_sha256(
            &weights_bytes,
            &manifest.mmproj_safetensors_sha256,
            "split MMProj safetensors",
        )?;
        let processor_bytes = read_bounded_bytes(processor_path, "split MMProj processor config")?;
        verify_bytes_sha256(
            &processor_bytes,
            &manifest.processor_config_sha256,
            "split MMProj processor config",
        )?;
        let processor: serde_json::Value =
            serde_json::from_slice(&processor_bytes).map_err(|err| {
                candle::Error::Msg(format!("invalid split MMProj processor config: {err}"))
            })?;
        let (processor_patch_size, processor_downsample_factor) =
            processor_pair_fields(&processor)?;
        if processor_patch_size != manifest.patch_size
            || processor_downsample_factor != manifest.downsample_factor
        {
            candle::bail!(
                "split MMProj processor/model mismatch: processor patch/factor [{processor_patch_size}, {processor_downsample_factor}], manifest [{}, {}]",
                manifest.patch_size,
                manifest.downsample_factor
            )
        }

        let report = inspect_safetensors_bytes(&weights_bytes, &manifest, dtype, device)?;
        report.require_clean()?;
        let vb = VarBuilder::from_buffered_safetensors(weights_bytes, dtype, device)?;
        let model_config = manifest.model_config.clone();
        let config = Lfm2VlMmprojConfig::from(&model_config);
        let vision_tower =
            siglip2::Siglip2VisionModel::new(&config.vision_config, vb.pp(VISION_ROOT))?;
        let projector = Lfm2VlProjector::from_mmproj_config(&config, vb.pp(PROJECTOR_ROOT))?;
        let metadata = MmprojMetadata {
            architecture: manifest.architecture.clone(),
            vision_hidden_size: manifest.vision_hidden_size,
            text_hidden_size: manifest.expected_text_hidden_size,
            patch_size: manifest.patch_size,
            downsample_factor: manifest.downsample_factor,
            image_token_id: manifest.image_token_id,
            use_image_special_tokens: model_config.use_image_special_tokens,
            expected_text_layer_count: Some(manifest.expected_text_layer_count),
            processor,
            source_model: Some(manifest.source_model.clone()),
            source_revision: Some(manifest.source_revision.clone()),
            manifest: Some(manifest),
            gguf: None,
        };
        Ok(Self {
            vision_tower,
            projector,
            metadata,
            report,
            config,
            device: device.clone(),
            dtype,
            gguf_execution: None,
            native_quantized_tensor_count: 0,
        })
    }

    pub fn encode_images(
        &self,
        inputs: &ProcessedVisionBatch,
        vision_batch_size: usize,
    ) -> Result<EncodedImages> {
        self.encode_images_with_limits(inputs, vision_batch_size, &VisionLimits::default())
    }

    pub fn encode_images_with_limits(
        &self,
        inputs: &ProcessedVisionBatch,
        vision_batch_size: usize,
        limits: &VisionLimits,
    ) -> Result<EncodedImages> {
        let patch_dimension = self.config.vision_config.patch_dimension_for_vl()?;
        preflight_packed_vision_limits(
            inputs,
            patch_dimension,
            self.config.downsample_factor,
            vision_batch_size,
            limits,
        )?;
        let device_inputs = ProcessedVisionBatch {
            pixel_values: inputs
                .pixel_values
                .to_device(&self.device)?
                .to_dtype(self.dtype)?,
            pixel_attention_mask: inputs.pixel_attention_mask.to_device(&self.device)?,
            spatial_shapes: inputs.spatial_shapes.to_device(&self.device)?,
            crops: inputs.crops.clone(),
            images: inputs.images.clone(),
        };
        encode_images_with_parts(
            &self.vision_tower,
            &self.projector,
            &self.config.vision_config,
            self.config.downsample_factor,
            &device_inputs,
            vision_batch_size,
            limits,
        )
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn gguf_execution(&self) -> Option<GgufMmprojExecution> {
        self.gguf_execution
    }

    pub fn native_quantized_tensor_count(&self) -> usize {
        self.native_quantized_tensor_count
    }
}

/// Quantized GGUF LFM2 text paired with a split or GGUF MMProj bundle.
pub struct QuantizedLfm2VlModel {
    text: quantized_lfm2::ModelWeights,
    mmproj: Mmproj,
    pairing: PairingReport,
}

impl QuantizedLfm2VlModel {
    pub fn new(
        text: quantized_lfm2::ModelWeights,
        mmproj: Mmproj,
        processor_patch_size: usize,
        processor_downsample_factor: usize,
        tokenizer_image_token_id: u32,
    ) -> Result<Self> {
        if text.hidden_size() != text.metadata().embedding_length {
            candle::bail!(
                "quantized LFM2 embedding tensor width {} does not match GGUF metadata {}",
                text.hidden_size(),
                text.metadata().embedding_length
            )
        }
        if mmproj.metadata.image_token_id as usize >= text.vocab_size() {
            candle::bail!(
                "MMProj image token ID {} is outside quantized text vocabulary size {}",
                mmproj.metadata.image_token_id,
                text.vocab_size()
            )
        }
        let pairing = mmproj.metadata.validate_pair(
            text.metadata(),
            processor_patch_size,
            processor_downsample_factor,
            tokenizer_image_token_id,
        )?;
        Ok(Self {
            text,
            mmproj,
            pairing,
        })
    }

    pub fn encode_images(
        &self,
        inputs: &ProcessedVisionBatch,
        vision_batch_size: usize,
    ) -> Result<EncodedImages> {
        self.mmproj.encode_images(inputs, vision_batch_size)
    }

    pub fn encode_images_with_limits(
        &self,
        inputs: &ProcessedVisionBatch,
        vision_batch_size: usize,
        limits: &VisionLimits,
    ) -> Result<EncodedImages> {
        self.mmproj
            .encode_images_with_limits(inputs, vision_batch_size, limits)
    }

    pub fn prefill(
        &mut self,
        input_ids: &Tensor,
        image_spans: &[ImageTokenSpan],
        encoded_images: Option<&EncodedImages>,
    ) -> Result<Tensor> {
        // A prefill always starts a new request. Clear every attention,
        // short-convolution, and mask cache even if subsequent validation
        // rejects the supplied multimodal inputs.
        self.text.clear_cache();
        let input_id_values = input_ids.to_dtype(DType::U32)?.to_vec2::<u32>()?;
        let image_token_count = input_id_values
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&token_id| token_id == self.pairing.image_token_id)
            .count();
        let input_embeds = self.text.embed_tokens(input_ids)?;
        let input_embeds = if image_token_count == 0 {
            if !image_spans.is_empty() || encoded_images.is_some() {
                candle::bail!("LFM2-VL image spans/features were supplied without image tokens")
            }
            input_embeds
        } else {
            let encoded_images = encoded_images.ok_or_else(|| {
                candle::Error::Msg("LFM2-VL image tokens require encoded image features".into())
            })?;
            if image_spans.is_empty() {
                candle::bail!("LFM2-VL image tokens require explicit image spans")
            }
            merge_projected_embeddings(
                input_ids,
                &input_embeds,
                self.pairing.image_token_id,
                image_spans,
                encoded_images,
            )?
        };
        self.text.forward_embeds(&input_embeds, 0)
    }

    pub fn decode(&mut self, token_ids: &Tensor, index_pos: usize) -> Result<Tensor> {
        token_ids.dims2()?;
        self.text.forward(token_ids, index_pos)
    }

    pub fn clear_cache(&mut self) {
        self.text.clear_cache();
    }

    pub fn pairing_report(&self) -> &PairingReport {
        &self.pairing
    }

    pub fn mmproj(&self) -> &Mmproj {
        &self.mmproj
    }

    pub fn vision_device(&self) -> &Device {
        self.mmproj.device()
    }

    pub fn text_device(&self) -> &Device {
        self.text.device()
    }

    pub fn context_length(&self) -> usize {
        self.text.metadata().context_length
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.text.metadata().eos_token_id
    }
}
