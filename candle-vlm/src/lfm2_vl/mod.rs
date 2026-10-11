//! LFM2.5-VL processor and prompt expansion.

pub mod config;
pub mod loading;
pub mod processor;
pub mod prompt;
pub mod types;

pub use candle_transformers::models::lfm2_vl::VisionLimits;
pub use config::{
    GgufProcessorMetadata, Lfm2VlProcessorConfig, ProcessorConfigOverride, ProcessorConfigPatch,
};
pub use loading::{
    load_lfm2_vl_hybrid, load_lfm2_vl_hybrid_with_embedding_mode, Lfm2EmbeddingLoadMode,
    Lfm2EmbeddingStorage, Lfm2VlHybridLoadOptions, Lfm2VlMmprojExecution, Lfm2VlMmprojSource,
    LoadedLfm2VlHybrid,
};
pub use processor::Lfm2VlProcessor;
pub use prompt::{ExpandedPrompt, Lfm2VlPrompt, Lfm2VlSpecialTokens, PromptOptions};
pub use types::{
    CropKind, CropMeta, EncodedImages, ImageMeta, ImageTokenSpan, ProcessedVisionBatch,
};
