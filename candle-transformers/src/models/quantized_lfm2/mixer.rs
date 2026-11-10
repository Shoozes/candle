//! Crate-private reuse of the existing quantized LFM2 token operators.
use super::*;

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
            LayerKind::Attention(op) => op.forward(x, mask, position),
            LayerKind::ShortConv(op) => op.forward(x, position),
        }
    }

    pub(crate) fn clear(&mut self) {
        match &mut self.0 {
            LayerKind::Attention(op) => op.kv_cache = None,
            LayerKind::ShortConv(op) => op.cache = None,
        }
    }
}
