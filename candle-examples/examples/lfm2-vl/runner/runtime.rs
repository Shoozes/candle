trait Runtime {
    fn context_length(&self) -> usize;
    fn default_eos_token_id(&self) -> Option<u32>;
    fn default_eos_source(&self) -> &'static str;
    fn vision_device(&self) -> &Device;
    fn text_device(&self) -> &Device;
    fn synchronize_devices(&self) -> Result<()> {
        self.vision_device().synchronize()?;
        if !self.vision_device().same_device(self.text_device()) {
            self.text_device().synchronize()?;
        }
        Ok(())
    }
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
