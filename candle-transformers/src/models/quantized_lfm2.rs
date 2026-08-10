use crate::quantized_nn::RmsNorm;
use crate::utils::repeat_kv;
use candle::quantized::gguf_file;
use candle::quantized::QMatMul;
use candle::{bail, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Embedding, Module};
use std::collections::HashMap;

fn get_qtensor<R: std::io::Seek + std::io::Read>(
    ct: &gguf_file::Content,
    reader: &mut R,
    device: &Device,
    names: &[String],
) -> Result<candle::quantized::QTensor> {
    for name in names {
        if ct.tensor_infos.contains_key(name) {
            return ct.tensor(reader, name, device);
        }
    }
    bail!("cannot find tensor info for {}", names.join(" | "))
}

fn get_dequantized<R: std::io::Seek + std::io::Read>(
    ct: &gguf_file::Content,
    reader: &mut R,
    device: &Device,
    names: &[String],
) -> Result<Tensor> {
    get_qtensor(ct, reader, device, names)?.dequantize(device)
}

#[derive(Debug, Clone)]
struct Mlp {
    w1: QMatMul,
    w2: QMatMul,
    w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.w1.forward(xs)?;
        let w3 = self.w3.forward(xs)?;
        self.w2.forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

#[derive(Debug, Clone)]
struct AttentionLayer {
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
    span_attn: tracing::Span,
    span_rot: tracing::Span,
}

#[derive(Debug, Clone)]
struct ShortConvLayer {
    in_proj: QMatMul,
    out_proj: QMatMul,
    conv: Tensor,
    l_cache: usize,
    cache: Option<Tensor>,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
enum LayerKind {
    Attention(AttentionLayer),
    ShortConv(ShortConvLayer),
}

#[derive(Debug, Clone)]
struct LayerWeights {
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
    mlp: Mlp,
    kind: LayerKind,
    span_mlp: tracing::Span,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, context_length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

impl AttentionLayer {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b, _n, seq_len, _d) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, n_embd) = xs.dims3()?;

        let q = self.wq.forward(xs)?;
        let k = self.wk.forward(xs)?;
        let v = self.wv.forward(xs)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = match mask {
            None => att,
            Some(mask) => {
                let mask = mask.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, &self.neg_inf)?
            }
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v.contiguous()?)?;

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        self.wo.forward(&y)
    }
}

impl ShortConvLayer {
    fn forward(&mut self, xs: &Tensor, _index_pos: usize) -> Result<Tensor> {
        let (b_sz, seq_len, hidden) = xs.dims3()?;
        let bcx = self.in_proj.forward(xs)?.transpose(1, 2)?;
        let b = bcx.narrow(1, 0, hidden)?;
        let c = bcx.narrow(1, hidden, hidden)?;
        let x = bcx.narrow(1, 2 * hidden, hidden)?;
        let bx = (b * &x)?.contiguous()?;

        // conv_weight shape -> [hidden, l_cache]
        let mut conv_weight = self.conv.clone();
        if conv_weight.dims().len() == 3 {
            conv_weight = conv_weight.squeeze(1)?;
        } else if conv_weight.dims().len() == 2 && conv_weight.dims2()? == (self.l_cache, hidden) {
            conv_weight = conv_weight.t()?.contiguous()?;
        }
        let conv_weight = conv_weight.contiguous()?;

        let mut conv_out = if seq_len == 1 {
            let mut state = if let Some(cache) = &self.cache {
                cache.clone()
            } else {
                Tensor::zeros((b_sz, hidden, self.l_cache), bx.dtype(), bx.device())?
            };

            if self.l_cache > 1 {
                let tail = state.narrow(2, 1, self.l_cache - 1)?;
                state = Tensor::cat(&[tail, bx.clone()], 2)?;
            } else {
                state = bx.clone();
            }
            self.cache = Some(state.clone());

            (state * &conv_weight.unsqueeze(0)?)?
                .sum_keepdim(2)?
                .contiguous()?
        } else {
            let conv = Conv1d::new(
                conv_weight
                    .reshape((hidden, 1, self.l_cache))?
                    .contiguous()?,
                None,
                Conv1dConfig {
                    padding: self.l_cache.saturating_sub(1),
                    groups: hidden,
                    ..Default::default()
                },
            );
            let mut out = conv.forward(&bx.contiguous()?)?;
            out = out.narrow(2, 0, seq_len)?;

            if self.l_cache > 0 {
                let (_, _, cur_len) = bx.dims3()?;
                let start = cur_len.saturating_sub(self.l_cache);
                let mut cache_src = bx.narrow(2, start, cur_len - start)?;
                if cache_src.dims3()?.2 < self.l_cache {
                    let pad = self.l_cache - cache_src.dims3()?.2;
                    let zeros =
                        Tensor::zeros((b_sz, hidden, pad), cache_src.dtype(), cache_src.device())?;
                    cache_src = Tensor::cat(&[zeros, cache_src], 2)?;
                }
                self.cache = Some(cache_src);
            }

            out
        };

        conv_out = (c * &conv_out)?;
        let conv_out = conv_out.transpose(1, 2)?.contiguous()?;
        self.out_proj.forward(&conv_out)
    }
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    metadata: Lfm2GgufMetadata,
    masks: HashMap<(usize, usize), Tensor>,
    span: tracing::Span,
    span_output: tracing::Span,
}

/// Header metadata required to construct and pair a quantized LFM2 text model.
#[derive(Debug, Clone, PartialEq)]
pub struct Lfm2GgufMetadata {
    pub architecture: String,
    pub embedding_length: usize,
    pub context_length: usize,
    pub block_count: usize,
    pub head_count: usize,
    pub head_count_kv: Vec<usize>,
    pub rms_norm_eps: f64,
    pub rope_freq_base: f32,
    pub shortconv_l_cache: usize,
    pub tied_output: bool,
}

const MAX_LFM2_GGUF_BLOCKS: usize = 512;
const MAX_LFM2_GGUF_EMBEDDING: usize = 1 << 16;
const MAX_LFM2_GGUF_CONTEXT: usize = 1 << 22;
const MAX_LFM2_GGUF_ROPE_ELEMENTS: usize = 1 << 26;

fn value_to_usize(v: &gguf_file::Value) -> Result<usize> {
    use gguf_file::Value::*;
    match v {
        U8(x) => Ok(*x as usize),
        I8(x) => usize::try_from(*x).map_err(candle::Error::wrap),
        U16(x) => Ok(*x as usize),
        I16(x) => usize::try_from(*x).map_err(candle::Error::wrap),
        U32(x) => usize::try_from(*x).map_err(candle::Error::wrap),
        I32(x) => usize::try_from(*x).map_err(candle::Error::wrap),
        U64(x) => usize::try_from(*x).map_err(candle::Error::wrap),
        I64(x) => usize::try_from(*x).map_err(candle::Error::wrap),
        F32(x) if x.is_finite() && *x >= 0.0 && x.fract() == 0.0 => {
            usize::try_from(*x as u64).map_err(candle::Error::wrap)
        }
        F64(x) if x.is_finite() && *x >= 0.0 && x.fract() == 0.0 => {
            usize::try_from(*x as u64).map_err(candle::Error::wrap)
        }
        F32(_) | F64(_) => bail!("metadata value is not a non-negative integer"),
        Bool(_) => bail!("unexpected boolean metadata"),
        String(_) => bail!("unexpected string metadata"),
        Array(_) => bail!("array should be handled separately"),
    }
}

fn read_usize_list(v: &gguf_file::Value, len: usize) -> Result<Vec<usize>> {
    use gguf_file::Value::Array;
    match v {
        Array(arr) => {
            if arr.len() > MAX_LFM2_GGUF_BLOCKS {
                bail!(
                    "quantized LFM2 metadata array length {} exceeds {MAX_LFM2_GGUF_BLOCKS}",
                    arr.len()
                )
            }
            let mut out = Vec::new();
            out.try_reserve_exact(arr.len()).map_err(|_| {
                candle::Error::Msg("quantized LFM2 metadata allocation failed".into())
            })?;
            for item in arr {
                out.push(value_to_usize(item)?);
            }
            if out.len() == len {
                Ok(out)
            } else if out.len() == 1 {
                Ok(vec![out[0]; len])
            } else {
                bail!(
                    "unexpected array length in metadata, expected {len} got {}",
                    out.len()
                )
            }
        }
        _ => Ok(vec![value_to_usize(v)?; len]),
    }
}

/// Inspect and validate quantized LFM2 GGUF metadata before tensor allocation.
pub fn inspect_gguf_metadata(ct: &gguf_file::Content) -> Result<Lfm2GgufMetadata> {
    let md_get = |name: &str| match ct.metadata.get(name) {
        None => bail!("cannot find {name} in metadata"),
        Some(value) => Ok(value),
    };
    let architecture = md_get("general.architecture")?.to_string()?.clone();
    if architecture != "lfm2" {
        bail!("unsupported quantized LFM2 GGUF architecture {architecture:?}")
    }
    let head_count = md_get("lfm2.attention.head_count")?.to_u32()? as usize;
    let embedding_length = md_get("lfm2.embedding_length")?.to_u32()? as usize;
    let context_length = md_get("lfm2.context_length")?.to_u32()? as usize;
    let block_count = md_get("lfm2.block_count")?.to_u32()? as usize;
    let rms_norm_eps = md_get("lfm2.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
    let rope_freq_base = match ct.metadata.get("lfm2.rope.freq_base") {
        Some(value) => value.to_f32()?,
        None => 1_000_000f32,
    };
    let shortconv_l_cache = md_get("lfm2.shortconv.l_cache")?.to_u32()? as usize;
    if head_count == 0 || embedding_length == 0 || embedding_length > MAX_LFM2_GGUF_EMBEDDING {
        bail!(
            "invalid quantized LFM2 embedding/head dimensions: embedding_length={embedding_length}, head_count={head_count}"
        )
    }
    if !embedding_length.is_multiple_of(head_count) {
        bail!(
            "quantized LFM2 embedding length {embedding_length} is not divisible by head count {head_count}"
        )
    }
    if block_count == 0 || block_count > MAX_LFM2_GGUF_BLOCKS {
        bail!("invalid quantized LFM2 block count {block_count}")
    }
    if context_length == 0 || context_length > MAX_LFM2_GGUF_CONTEXT {
        bail!("invalid quantized LFM2 context length {context_length}")
    }
    let rotary_width = (embedding_length / head_count).div_ceil(2);
    let rotary_elements = context_length
        .checked_mul(rotary_width)
        .ok_or_else(|| candle::Error::Msg("quantized LFM2 RoPE table size overflows".into()))?;
    if rotary_elements > MAX_LFM2_GGUF_ROPE_ELEMENTS {
        bail!(
            "quantized LFM2 RoPE table requires {rotary_elements} elements, limit is {MAX_LFM2_GGUF_ROPE_ELEMENTS}"
        )
    }
    if !rms_norm_eps.is_finite() || rms_norm_eps <= 0.0 {
        bail!("invalid quantized LFM2 RMS norm epsilon {rms_norm_eps}")
    }
    if !rope_freq_base.is_finite() || rope_freq_base <= 0.0 {
        bail!("invalid quantized LFM2 RoPE frequency base {rope_freq_base}")
    }
    if shortconv_l_cache == 0 || shortconv_l_cache > context_length {
        bail!(
            "invalid quantized LFM2 short-convolution cache length {shortconv_l_cache} for context {context_length}"
        )
    }
    let head_count_kv = read_usize_list(md_get("lfm2.attention.head_count_kv")?, block_count)?;
    for (layer, &kv_heads) in head_count_kv.iter().enumerate() {
        if kv_heads != 0 && (kv_heads > head_count || !head_count.is_multiple_of(kv_heads)) {
            bail!(
                "invalid quantized LFM2 key/value head count {kv_heads} at layer {layer} for {head_count} attention heads"
            )
        }
    }
    let tied_output = ![
        "output.weight",
        "lm_head.weight",
        "model.output.weight",
        "model.lm_head.weight",
    ]
    .iter()
    .any(|name| ct.tensor_infos.contains_key(*name));

    Ok(Lfm2GgufMetadata {
        architecture,
        embedding_length,
        context_length,
        block_count,
        head_count,
        head_count_kv,
        rms_norm_eps,
        rope_freq_base,
        shortconv_l_cache,
        tied_output,
    })
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let metadata = inspect_gguf_metadata(&ct)?;
        let head_count = metadata.head_count;
        let head_count_kv = &metadata.head_count_kv;
        let embedding_length = metadata.embedding_length;
        let context_length = metadata.context_length;
        let block_count = metadata.block_count;
        let rms_norm_eps = metadata.rms_norm_eps;
        let rope_freq_base = metadata.rope_freq_base;
        let l_cache = metadata.shortconv_l_cache;
        let head_dim = embedding_length / head_count;
        let (cos, sin) = precomput_freqs_cis(head_dim, rope_freq_base, context_length, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok_embeddings_q = get_qtensor(
            &ct,
            reader,
            device,
            &[
                "token_embd.weight",
                "tok_embeddings.weight",
                "model.embed_tokens.weight",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
        )?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        let (vocab_size, loaded_embedding_length) = tok_embeddings.dims2()?;
        if loaded_embedding_length != embedding_length {
            bail!(
                "quantized LFM2 token embedding width {loaded_embedding_length} does not match GGUF metadata {embedding_length}"
            )
        }
        tracing::debug!(
            tok_embd_shape = ?tok_embeddings.shape().dims(),
            "loaded lfm2 token embeddings"
        );

        let norm = RmsNorm::from_qtensor(
            get_qtensor(
                &ct,
                reader,
                device,
                &[
                    "output_norm.weight",
                    "embedding_norm.weight",
                    "model.embedding_norm.weight",
                    "model.embedding_norm",
                    "token_embd_norm.weight",
                ]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            )?,
            rms_norm_eps,
        )?;
        let output_q = if metadata.tied_output {
            tok_embeddings_q
        } else {
            get_qtensor(
                &ct,
                reader,
                device,
                &[
                    "output.weight",
                    "lm_head.weight",
                    "model.output.weight",
                    "model.lm_head.weight",
                ]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            )?
        };
        if output_q.shape().dims() != [vocab_size, embedding_length] {
            bail!(
                "quantized LFM2 output tensor shape {:?} does not match [{vocab_size}, {embedding_length}]",
                output_q.shape().dims()
            )
        }
        tracing::debug!(
            output_shape = ?output_q.shape().dims(),
            "loaded lfm2 output weight (using tok_embd if missing)"
        );

        let mut layers = Vec::new();
        layers
            .try_reserve_exact(block_count)
            .map_err(|_| candle::Error::Msg("quantized LFM2 layer allocation failed".into()))?;
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let is_attention = head_count_kv.get(layer_idx).copied().unwrap_or(head_count) > 0;

            let operator_norm = get_qtensor(
                &ct,
                reader,
                device,
                &[
                    format!("{prefix}.attn_norm.weight"),
                    format!("{prefix}.operator_norm.weight"),
                    format!("{prefix}.attention_norm.weight"),
                ],
            )?;
            let ffn_norm = get_qtensor(
                &ct,
                reader,
                device,
                &[
                    format!("{prefix}.ffn_norm.weight"),
                    format!("{prefix}.ffn_norm"),
                ],
            )?;
            let mlp = {
                let w1 = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.ffn_gate.weight"),
                        format!("{prefix}.feed_forward.w1.weight"),
                        format!("{prefix}.mlp.gate_proj.weight"),
                    ],
                )?;
                let w2 = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.ffn_down.weight"),
                        format!("{prefix}.feed_forward.w2.weight"),
                        format!("{prefix}.mlp.down_proj.weight"),
                    ],
                )?;
                let w3 = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.ffn_up.weight"),
                        format!("{prefix}.feed_forward.w3.weight"),
                        format!("{prefix}.mlp.up_proj.weight"),
                    ],
                )?;
                Mlp {
                    w1: QMatMul::from_qtensor(w1)?,
                    w2: QMatMul::from_qtensor(w2)?,
                    w3: QMatMul::from_qtensor(w3)?,
                }
            };

            let kind = if is_attention {
                let n_kv_head = head_count_kv[layer_idx];
                let wq = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.attn_q.weight"),
                        format!("{prefix}.self_attn.q_proj.weight"),
                    ],
                )?;
                let wk = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.attn_k.weight"),
                        format!("{prefix}.self_attn.k_proj.weight"),
                    ],
                )?;
                let wv = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.attn_v.weight"),
                        format!("{prefix}.self_attn.v_proj.weight"),
                    ],
                )?;
                let wo = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.attn_output.weight"),
                        format!("{prefix}.self_attn.out_proj.weight"),
                    ],
                )?;
                let q_norm = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.attn_q_norm.weight"),
                        format!("{prefix}.self_attn.q_layernorm.weight"),
                        format!("{prefix}.attention.q_norm.weight"),
                    ],
                )?;
                let k_norm = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.attn_k_norm.weight"),
                        format!("{prefix}.self_attn.k_layernorm.weight"),
                        format!("{prefix}.attention.k_norm.weight"),
                    ],
                )?;

                LayerKind::Attention(AttentionLayer {
                    wq: QMatMul::from_qtensor(wq)?,
                    wk: QMatMul::from_qtensor(wk)?,
                    wv: QMatMul::from_qtensor(wv)?,
                    wo: QMatMul::from_qtensor(wo)?,
                    q_norm: RmsNorm::from_qtensor(q_norm, rms_norm_eps)?,
                    k_norm: RmsNorm::from_qtensor(k_norm, rms_norm_eps)?,
                    n_head: head_count,
                    n_kv_head,
                    head_dim,
                    cos: cos.clone(),
                    sin: sin.clone(),
                    neg_inf: neg_inf.clone(),
                    kv_cache: None,
                    span_attn: tracing::span!(tracing::Level::TRACE, "attn"),
                    span_rot: tracing::span!(tracing::Level::TRACE, "attn-rot"),
                })
            } else {
                let in_proj = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.shortconv.in_proj.weight"),
                        format!("{prefix}.conv.in_proj.weight"),
                    ],
                )?;
                let out_proj = get_qtensor(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.shortconv.out_proj.weight"),
                        format!("{prefix}.conv.out_proj.weight"),
                    ],
                )?;
                let conv = get_dequantized(
                    &ct,
                    reader,
                    device,
                    &[
                        format!("{prefix}.shortconv.conv.weight"),
                        format!("{prefix}.conv.conv.weight"),
                        format!("{prefix}.shortconv.conv"),
                    ],
                )?;
                LayerKind::ShortConv(ShortConvLayer {
                    in_proj: QMatMul::from_qtensor(in_proj)?,
                    out_proj: QMatMul::from_qtensor(out_proj)?,
                    conv,
                    l_cache,
                    cache: None,
                })
            };

            layers.push(LayerWeights {
                operator_norm: RmsNorm::from_qtensor(operator_norm, rms_norm_eps)?,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                mlp,
                kind,
                span_mlp: tracing::span!(tracing::Level::TRACE, "ffn"),
            });
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: QMatMul::from_qtensor(output_q)?,
            metadata,
            masks: HashMap::new(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
            span_output: tracing::span!(tracing::Level::TRACE, "output"),
        })
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            Ok(mask.clone())
        } else {
            let mask = crate::utils::build_causal_mask(seq_len, index_pos, device)?;
            self.masks.insert((seq_len, kv_len), mask.clone());
            Ok(mask)
        }
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.tok_embeddings.forward(input_ids)
    }

    pub fn metadata(&self) -> &Lfm2GgufMetadata {
        &self.metadata
    }

    pub fn hidden_size(&self) -> usize {
        self.tok_embeddings.hidden_size()
    }

    pub fn vocab_size(&self) -> usize {
        self.tok_embeddings.embeddings().dims()[0]
    }

    pub fn device(&self) -> &Device {
        self.tok_embeddings.embeddings().device()
    }

    pub fn clear_cache(&mut self) {
        self.masks.clear();
        for layer in &mut self.layers {
            match &mut layer.kind {
                LayerKind::Attention(attn) => attn.kv_cache = None,
                LayerKind::ShortConv(conv) => conv.cache = None,
            }
        }
    }

    pub fn forward_embeds(&mut self, input_embeds: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len, _) = input_embeds.dims3()?;
        if seq_len == 0 {
            candle::bail!("quantized LFM2 cannot forward an empty sequence")
        }
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, index_pos, input_embeds.device())?)
        };

        let _enter = self.span.enter();
        let mut hidden = input_embeds.clone();
        for layer in self.layers.iter_mut() {
            let residual = hidden.clone();
            let normed = layer.operator_norm.forward(&hidden)?;
            hidden = match &mut layer.kind {
                LayerKind::Attention(attn) => attn.forward(&normed, mask.as_ref(), index_pos)?,
                LayerKind::ShortConv(conv) => conv.forward(&normed, index_pos)?,
            };
            hidden = (hidden + residual)?;

            let residual = hidden.clone();
            let ff = layer.ffn_norm.forward(&hidden)?;
            let _enter = layer.span_mlp.enter();
            let ff = layer.mlp.forward(&ff)?;
            hidden = (ff + residual)?;
        }
        let hidden = self.norm.forward(&hidden)?;
        let hidden = hidden.i((.., seq_len - 1, ..))?;
        let _enter = self.span_output.enter();
        self.output.forward(&hidden)
    }

    pub fn forward(&mut self, input_ids: &Tensor, index_pos: usize) -> Result<Tensor> {
        input_ids.dims2()?;
        let input_embeds = self.embed_tokens(input_ids)?;
        self.forward_embeds(&input_embeds, index_pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::lfm2_vl::{merge_projected_embeddings, EncodedImages, ImageTokenSpan};
    use candle::quantized::{GgmlDType, QMatMul, QTensor};
    use candle::{DType, Device, Tensor};
    use std::collections::HashMap;

    fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
        let max_abs = (actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(max_abs <= tolerance, "max absolute error {max_abs}");
        Ok(())
    }

    #[test]
    fn embedding_driven_forward_matches_quantized_token_forward() -> Result<()> {
        let device = Device::Cpu;
        let values: Vec<f32> = (0..(32 * 32)).map(|idx| (idx % 17) as f32 / 17.0).collect();
        let embedding_weights = Tensor::from_slice(&values, (32, 32), &device)?;
        let output_weights = Tensor::from_slice(&values, (32, 32), &device)?;
        let norm_weights = Tensor::ones(32, DType::F32, &device)?;

        let embedding = Embedding::new(embedding_weights, 32);
        let norm = RmsNorm::from_qtensor(QTensor::quantize(&norm_weights, GgmlDType::Q8_0)?, 1e-5)?;
        let output = QMatMul::from_qtensor(QTensor::quantize(&output_weights, GgmlDType::Q8_0)?)?;
        let mut model = ModelWeights {
            tok_embeddings: embedding,
            layers: Vec::new(),
            norm,
            output,
            metadata: Lfm2GgufMetadata {
                architecture: "lfm2".to_string(),
                embedding_length: 32,
                context_length: 32,
                block_count: 0,
                head_count: 1,
                head_count_kv: Vec::new(),
                rms_norm_eps: 1e-5,
                rope_freq_base: 10_000.0,
                shortconv_l_cache: 1,
                tied_output: false,
            },
            masks: HashMap::new(),
            span: tracing::span!(tracing::Level::TRACE, "test-model"),
            span_output: tracing::span!(tracing::Level::TRACE, "test-output"),
        };

        let input_ids = Tensor::from_slice(&[1u32, 2u32], (1, 2), &device)?;
        let token_logits = model.forward(&input_ids, 0)?;
        let input_embeds = model.embed_tokens(&input_ids)?;
        let embed_logits = model.forward_embeds(&input_embeds, 0)?;
        assert_close(&token_logits, &embed_logits, 1e-5)?;

        model.clear_cache();
        let reset_logits = model.forward_embeds(&input_embeds, 0)?;
        assert_close(&embed_logits, &reset_logits, 1e-5)?;

        let image_input_ids = Tensor::from_slice(&[1u32, 3u32, 3u32, 2u32], (1, 4), &device)?;
        let image_input_embeds = model.embed_tokens(&image_input_ids)?;
        let encoded = EncodedImages {
            embeddings: Tensor::ones((2, 32), DType::F32, &device)?,
            per_image_ranges: std::iter::once(0..2).collect(),
            per_crop_ranges: std::iter::once(0..2).collect(),
        };
        let merged = merge_projected_embeddings(
            &image_input_ids,
            &image_input_embeds,
            3,
            &[ImageTokenSpan::new(0, 1, 3)],
            &encoded,
        )?;
        assert_close(&merged.i((0, 1..3, ..))?, &encoded.embeddings, 0.0)?;
        model.clear_cache();
        let image_logits = model.forward_embeds(&merged, 0)?;
        assert_eq!(image_logits.dims(), [1, 32]);
        assert!(image_logits
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?
            .iter()
            .flatten()
            .all(|value| value.is_finite()));
        Ok(())
    }
}
