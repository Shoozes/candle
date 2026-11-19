//! Crate-private reuse of the existing quantized LFM2 token operators.
use super::*;
#[cfg(feature = "cuda")]
use crate::models::quantized_lfm2_moe::cuda;
use crate::models::quantized_lfm2_moe::{cpu, cpu_attention, project_q4_gemm_gemv};

pub(crate) struct Mixer(LayerKind);

impl Mixer {
    pub(crate) fn new<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
        layer: usize,
        metadata: &Lfm2GgufMetadata,
        rotary: &(Tensor, Tensor),
    ) -> Result<Self> {
        let prefix = format!("blk.{layer}");
        let mut linear = |name: &str| -> Result<AdaptedLinear> {
            let key = format!("{prefix}.{name}.weight");
            let weight = ct.tensor(reader, &key, device)?;
            let matrix = match weight.dtype() {
                candle::quantized::GgmlDType::F32
                | candle::quantized::GgmlDType::F16
                | candle::quantized::GgmlDType::BF16 => {
                    QMatMul::Tensor(weight.dequantize(device)?.to_dtype(DType::F32)?)
                }
                _ => QMatMul::QTensor(Arc::new(weight)),
            };
            AdaptedLinear::new(key, matrix)
        };
        let kv = *metadata
            .head_count_kv
            .get(layer)
            .ok_or_else(|| candle::Error::Msg("LFM2 mixer layer is outside metadata".into()))?;
        let kind = if kv == 0 {
            let in_proj = linear("shortconv.in_proj")?;
            let out_proj = linear("shortconv.out_proj")?;
            let conv = ct
                .tensor(reader, &format!("{prefix}.shortconv.conv.weight"), device)?
                .dequantize(device)?
                .to_dtype(DType::F32)?;
            LayerKind::ShortConv(ShortConvLayer {
                in_proj,
                out_proj,
                conv,
                l_cache: metadata.shortconv_l_cache,
                cache: None,
            })
        } else {
            let wq = linear("attn_q")?;
            let wk = linear("attn_k")?;
            let wv = linear("attn_v")?;
            let wo = linear("attn_output")?;
            let q_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device)?,
                metadata.rms_norm_eps,
            )?;
            let k_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device)?,
                metadata.rms_norm_eps,
            )?;
            LayerKind::Attention(AttentionLayer {
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                n_head: metadata.head_count,
                n_kv_head: kv,
                head_dim: metadata.embedding_length / metadata.head_count,
                cos: rotary.0.clone(),
                sin: rotary.1.clone(),
                neg_inf: Tensor::new(f32::NEG_INFINITY, device)?,
                kv_cache: None,
                span_attn: tracing::span!(tracing::Level::TRACE, "lfm2-moe-attn"),
                span_rot: tracing::span!(tracing::Level::TRACE, "lfm2-moe-rope"),
            })
        };
        Ok(Self(kind))
    }

    pub(crate) fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        position: usize,
    ) -> Result<Tensor> {
        match &mut self.0 {
            LayerKind::Attention(op) if x.device().is_cpu() => {
                let project = |weight: &QMatMul, input: &Tensor| {
                    cpu::project(weight, input, cpu::Projection::Dense)
                };
                let project_qkv = |weight: &QMatMul, input: &Tensor| {
                    cpu::project(weight, input, cpu::Projection::DenseQ6KSingleRow)
                };
                op.forward_with(
                    x,
                    mask,
                    position,
                    (project_qkv, project),
                    (
                        |norm: &RmsNorm, input: &Tensor| {
                            let (weight, epsilon) = norm.weight_and_epsilon();
                            cpu::rms_norm(input, weight, epsilon)
                        },
                        candle_nn::rotary_emb::rope,
                    ),
                    cpu_attention::forward,
                )
            }
            #[cfg(feature = "cuda")]
            LayerKind::Attention(op) if x.device().is_cuda() => op.forward_with(
                x,
                mask,
                position,
                (cuda::project_qkv, |weight, input| {
                    project_q4_gemm_gemv(weight, input, QMatMul::forward)
                }),
                (
                    |norm: &RmsNorm, input: &Tensor| {
                        let (weight, epsilon) = norm.weight_and_epsilon();
                        cuda::rms_norm(input, weight, epsilon)
                    },
                    cuda::rope,
                ),
                crate::models::quantized_lfm2_moe::cuda_attention::forward,
            ),
            LayerKind::Attention(op) => op.forward(x, mask, position),
            LayerKind::ShortConv(op) => op.forward_with(
                x,
                position,
                |weight, input| project_q4_gemm_gemv(weight, input, QMatMul::forward),
                true,
            ),
        }
    }

    pub(crate) fn clear(&mut self) {
        match &mut self.0 {
            LayerKind::Attention(op) => op.kv_cache = None,
            LayerKind::ShortConv(op) => op.cache = None,
        }
    }
}
