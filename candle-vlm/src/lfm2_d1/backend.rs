use crate::lfm2_vl::{
    EncodedImages, ExpandedPrompt, ImageTokenSpan, Lfm2EmbeddingStorage, LoadedLfm2VlHybrid,
    ProcessedVisionBatch,
};
use candle::{Device, Result, Tensor};
use candle_transformers::models::quantized_lfm2::ModelWeights;
use image::DynamicImage;

pub(super) enum D1Backend {
    Hybrid(Box<LoadedLfm2VlHybrid>),
    Text(Box<ModelWeights>),
}

impl D1Backend {
    pub(super) fn supports_images(&self) -> bool {
        matches!(self, Self::Hybrid(_))
    }

    pub(super) fn text_device(&self) -> &Device {
        match self {
            Self::Hybrid(loaded) => loaded.model.text_device(),
            Self::Text(model) => model.device(),
        }
    }

    pub(super) fn vision_device(&self) -> Option<&Device> {
        match self {
            Self::Hybrid(loaded) => Some(loaded.model.vision_device()),
            Self::Text(_) => None,
        }
    }

    pub(super) fn context_length(&self) -> usize {
        match self {
            Self::Hybrid(loaded) => loaded.model.context_length(),
            Self::Text(model) => model.metadata().context_length,
        }
    }

    pub(super) fn embedding_storage(&self) -> Result<Lfm2EmbeddingStorage> {
        match self {
            Self::Hybrid(loaded) => loaded.model.text_embedding_storage(),
            Self::Text(model) => model.embedding_storage(),
        }
    }

    pub(super) fn hybrid(&self) -> Result<&LoadedLfm2VlHybrid> {
        match self {
            Self::Hybrid(loaded) => Ok(loaded),
            Self::Text(_) => candle::bail!("text-only d1 session does not support images"),
        }
    }

    pub(super) fn process_images(&self, images: &[DynamicImage]) -> Result<ProcessedVisionBatch> {
        let loaded = self.hybrid()?;
        loaded
            .processor
            .process(images, loaded.model.vision_device())
    }

    pub(super) fn expand(
        &self,
        text: &str,
        images: &ProcessedVisionBatch,
    ) -> Result<ExpandedPrompt> {
        self.hybrid()?.prompt.expand(text, images)
    }

    pub(super) fn encode_images(&self, images: &ProcessedVisionBatch) -> Result<EncodedImages> {
        let loaded = self.hybrid()?;
        loaded
            .model
            .encode_images_with_limits(images, 1, &loaded.processor.config().vision_limits)
    }

    pub(super) fn prefill(
        &mut self,
        input: &Tensor,
        spans: &[ImageTokenSpan],
        images: Option<&EncodedImages>,
    ) -> Result<Tensor> {
        match self {
            Self::Hybrid(loaded) => loaded.model.prefill(input, spans, images),
            Self::Text(model) => {
                model.clear_cache();
                if !spans.is_empty() || images.is_some() {
                    candle::bail!("text-only d1 prefill does not accept image features")
                }
                model.forward(input, 0)
            }
        }
    }

    pub(super) fn clear_cache(&mut self) {
        match self {
            Self::Hybrid(loaded) => loaded.model.clear_cache(),
            Self::Text(model) => model.clear_cache(),
        }
    }
}
