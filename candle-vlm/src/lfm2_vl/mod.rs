//! LFM2.5-VL processor and prompt expansion.

pub mod config;
pub mod processor;
pub mod prompt;
pub mod types;

pub use config::{
    GgufProcessorMetadata, Lfm2VlProcessorConfig, ProcessorConfigOverride, ProcessorConfigPatch,
};
pub use processor::Lfm2VlProcessor;
pub use prompt::{ExpandedPrompt, Lfm2VlPrompt, Lfm2VlSpecialTokens, PromptOptions};
pub use types::{
    CropKind, CropMeta, EncodedImages, ImageMeta, ImageTokenSpan, ProcessedVisionBatch,
};
