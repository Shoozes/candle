//! Experimental GPT-OSS building blocks.
//!
//! This module is intentionally separate from the maintained LFM2-VL/Q8
//! product path.  It currently provides configuration/checkpoint admission and
//! a CPU reference for the packed MXFP4 MoE weights.  It does not claim live
//! checkpoint, CUDA, or generation support.

mod checkpoint;
mod config;
mod gguf;
mod model;
pub mod mxfp4;

pub use checkpoint::GptOssCheckpoint;
pub use config::GptOssConfig;
pub use gguf::{
    GptOssGgufArtifact, GptOssGgufDType, GptOssGgufTensor, GptOssTensorRole,
    SELECTED_GPT_OSS_GGUF_SHA256,
};
pub use model::{DenseLinear, GptOssLayerWeights, GptOssModel, GptOssWeights};
