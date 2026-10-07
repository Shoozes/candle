//! Experimental GPT-OSS building blocks.
//!
//! This module is intentionally separate from the maintained LFM2-VL/Q8
//! product path. It provides checkpoint admission, packed MXFP4 ownership,
//! a CPU reference, and an opt-in, feature-gated CUDA executor. Component
//! fixtures and source-bound external short-parity/performance diagnostics
//! have separate evidence boundaries recorded in `docs/gpt-oss/STATUS.md`.
//! The maintained product default and broader qualification remain separate.

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
    cache_bytes_for_tokens, cache_bytes_per_token, max_supported_sequence_tokens,
    total_device_bytes_for_tokens, workspace_bytes_for_tokens, GptOssCancellationToken,
    GptOssLoadLease, GptOssLoadRegistry, GptOssLoadedHandle, GptOssResourceLimits,
    GptOssResourceUsage, EDGE_DEVICE_CEILING_BYTES,
};
