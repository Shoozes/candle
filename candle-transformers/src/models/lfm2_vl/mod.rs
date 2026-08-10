//! Native tensor-level LFM2.5-VL composition.
//!
//! This phase consumes already patchified SigLIP2 NaFlex tensors. Raw image
//! processing, tokenization, GGUF, and quantized vision are intentionally
//! outside this module.

pub mod config;
pub mod model;
pub mod projector;
pub mod weights;

pub use config::{projected_token_count, Lfm2VlConfig};
pub use model::{
    merge_projected_embeddings, CropKind, CropMeta, EncodedImages, ImageMeta, ImageTokenSpan,
    Lfm2VlModel, ProcessedVisionBatch,
};
pub use projector::Lfm2VlProjector;
pub use weights::{
    inspect_safetensors, Mmproj, MmprojLoadReport, MmprojManifest, MmprojMetadata,
    MmprojTensorInfo, PairingReport, QuantizedLfm2VlModel,
};
