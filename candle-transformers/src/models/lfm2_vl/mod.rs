//! LFM2.5-VL composition and MMProj loading.
//!
//! The model consumes already patchified SigLIP2 NaFlex tensors and supports
//! dense, split, direct-GGUF, and native-Q8 vision/projector construction. Raw
//! image processing and tokenization live in `candle-vlm`.

pub mod config;
pub mod gguf;
pub(crate) mod linear;
pub mod model;
pub mod projector;
pub mod weights;

pub use config::{
    projected_token_count, Lfm2VlConfig, Lfm2VlMmprojConfig, VisionLimits,
    DEFAULT_LFM2_VL_IMAGE_TOKEN_ID, MAX_VISION_CROPS_PER_IMAGE, MAX_VISION_IMAGES,
    MAX_VISION_PATCHES_PER_CROP, MAX_VISION_SOURCE_PIXELS, MAX_VISION_TOTAL_CROPS,
    MAX_VISION_TOTAL_PROJECTED_TOKENS,
};
pub use gguf::{GgufMmprojExecution, GgufMmprojMetadata};
pub use linear::LinearOp;
pub use model::{
    merge_projected_embeddings, CropKind, CropMeta, EncodedImages, ImageMeta, ImageTokenSpan,
    Lfm2VlDecodeTrace, Lfm2VlImageTrace, Lfm2VlModel, Lfm2VlPrefillTrace, Lfm2VlProjectorTrace,
    ProcessedVisionBatch,
};
pub use projector::Lfm2VlProjector;
pub use weights::{
    inspect_safetensors, Mmproj, MmprojLoadReport, MmprojManifest, MmprojMetadata,
    MmprojTensorInfo, PairingReport, QuantizedLfm2VlModel,
};
