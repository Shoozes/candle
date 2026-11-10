//! Quantized LFM2-MoE with retained per-expert GGUF weights on CPU or CUDA.
mod metadata;
mod model;

pub use metadata::{inspect_gguf_metadata, Lfm2MoeGgufMetadata, MemoryEstimate};
pub use model::ModelWeights;

#[cfg(test)]
mod tests;
