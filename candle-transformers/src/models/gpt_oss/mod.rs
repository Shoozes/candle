//! Experimental GPT-OSS building blocks.
//!
//! This module is intentionally separate from the maintained LFM2-VL/Q8
//! product path.  It currently provides configuration/checkpoint admission and
//! a CPU reference for the packed MXFP4 MoE weights.  The CUDA executor is an
//! explicit feature-gated synthetic proof boundary; no live checkpoint or
//! generation support is claimed.

mod checkpoint;
mod config;
#[cfg(feature = "cuda")]
mod cuda;
mod gguf;
mod model;
pub mod mxfp4;
mod runtime;

pub use checkpoint::GptOssCheckpoint;
pub use config::GptOssConfig;
#[cfg(feature = "cuda")]
pub use cuda::{GptOssCudaConfig, GptOssCudaError, GptOssCudaModel, GptOssCudaResult};
pub use gguf::{
    load_gpt_oss_weights, load_gpt_oss_weights_with_cancellation, GptOssGgufArtifact,
    GptOssGgufDType, GptOssGgufTensor, GptOssTensorRole, SELECTED_GPT_OSS_GGUF_SHA256,
};
pub use model::{DenseLinear, GptOssLayerWeights, GptOssModel, GptOssWeights};
pub use runtime::{
    cache_bytes_for_tokens, cache_bytes_per_token, GptOssCancellationToken, GptOssLoadLease,
    GptOssLoadRegistry, GptOssLoadedHandle, GptOssResourceLimits, GptOssResourceUsage,
};
