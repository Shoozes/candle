//! Quantized LFM2-MoE with retained per-expert GGUF weights on CPU or CUDA.
use candle::{
    quantized::{GgmlDType, QMatMul},
    Result, Tensor,
};

pub(crate) mod cpu;
pub(crate) mod cpu_attention;
#[cfg(feature = "cuda")]
pub(crate) mod cuda;
#[cfg(feature = "cuda")]
pub(crate) mod cuda_attention;
mod metadata;
mod model;

pub use metadata::{inspect_gguf_metadata, Lfm2MoeGgufMetadata, MemoryEstimate};
pub use model::ModelWeights;

pub(crate) fn project_q4_gemm_gemv(
    weight: &QMatMul,
    input: &Tensor,
    fallback: impl FnOnce(&QMatMul, &Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    if matches!(weight, QMatMul::QTensor(q) if q.dtype() == GgmlDType::Q4K) {
        if input.device().is_cpu() {
            return cpu::project(weight, input, cpu::Projection::DenseGemmGemv);
        }
        #[cfg(feature = "cuda")]
        if input.device().is_cuda() {
            return weight.forward_cuda_q4k_q8k(&input.contiguous()?);
        }
    }
    fallback(weight, input)
}

#[cfg(test)]
mod tests;
