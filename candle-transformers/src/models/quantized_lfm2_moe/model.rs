use super::{inspect_gguf_metadata, Lfm2MoeGgufMetadata};
use crate::models::{lfm2_moe::routing, quantized_lfm2::mixer::Mixer};
use candle::{
    quantized::{gguf_file, GgmlDType, QMatMul, QTensor},
    DType, Device, IndexOp, Module, Result, Tensor,
};
use candle_nn::RmsNorm;
use std::{
    io::{Read, Seek, SeekFrom},
    sync::Arc,
};

const PREFILL_CHUNK: usize = 128;

fn matrix(weight: QTensor) -> Result<QMatMul> {
    Ok(match weight.dtype() {
        GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => {
            QMatMul::Tensor(weight.dequantize(&weight.device())?.to_dtype(DType::F32)?)
        }
        _ => QMatMul::QTensor(Arc::new(weight)),
    })
}

fn forward_matrix(weight: &QMatMul, x: &Tensor) -> Result<Tensor> {
    // Bound CPU cache ownership: avoid retaining an extra repacked expert bank.
    candle::quantized::with_cpu_quantized_matmul_mode(
        candle::quantized::CpuQuantizedMatmulMode::GenericTiled,
        || weight.forward(x),
    )
}

struct Expert {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}
impl Expert {
    fn quantized_matrices(&self) -> usize {
        [&self.gate, &self.up, &self.down]
            .iter()
            .filter(|w| matches!(w, QMatMul::QTensor(_)))
            .count()
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        forward_matrix(
            &self.down,
            &(candle_nn::ops::silu(&forward_matrix(&self.gate, x)?)?
                * forward_matrix(&self.up, x)?)?,
        )
    }
}

enum FeedForward {
    Dense(Expert),
    Sparse {
        router: Tensor,
        bias: Tensor,
        experts: Vec<Expert>,
        topk: usize,
    },
}
impl FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(expert) => expert.forward(x),
            Self::Sparse {
                router,
                bias,
                experts,
                topk,
            } => {
                let (batch, length, hidden) = x.dims3()?;
                let flat = x.reshape(((), hidden))?;
                let logits = flat.matmul(&router.t()?)?;
                let (ids, weights) = routing::route(&logits, Some(bias), *topk, true, 1.)?;
                let weights = weights.flatten_all()?;
                let mut output = flat.zeros_like()?;
                for (expert, (tokens, slots)) in
                    experts.iter().zip(routing::groups(&ids, experts.len())?)
                {
                    if tokens.is_empty() {
                        continue;
                    }
                    let tokens = Tensor::new(tokens.as_slice(), x.device())?;
                    let slots = Tensor::new(slots.as_slice(), x.device())?;
                    let values = expert.forward(&flat.index_select(&tokens, 0)?)?;
                    let weighted =
                        values.broadcast_mul(&weights.index_select(&slots, 0)?.unsqueeze(1)?)?;
                    output = output.index_add(&tokens, &weighted, 0)?;
                }
                output.reshape((batch, length, hidden))
            }
        }
    }
}

struct Layer {
    mixer: Mixer,
    norm: RmsNorm,
    ffn_norm: RmsNorm,
    ff: FeedForward,
}

pub struct ModelWeights {
    embeddings: Arc<QTensor>,
    layers: Vec<Layer>,
    norm: RmsNorm,
    head: QMatMul,
    metadata: Lfm2MoeGgufMetadata,
    position: usize,
    batch: Option<usize>,
}

fn read_expert<R: Read + Seek>(
    ct: &gguf_file::Content,
    reader: &mut R,
    name: &str,
    expert: usize,
    device: &Device,
) -> Result<QTensor> {
    let info = ct
        .tensor_infos
        .get(name)
        .ok_or_else(|| candle::Error::Msg(format!("missing expert bank {name}")))?;
    let (count, rows, cols) = info.shape.dims3()?;
    if expert >= count {
        candle::bail!("expert index is outside bank")
    }
    let bytes = rows
        .checked_mul(cols)
        .map(|n| n / info.ggml_dtype.block_size())
        .and_then(|n| n.checked_mul(info.ggml_dtype.type_size()))
        .ok_or_else(|| candle::Error::Msg("LFM2-MoE expert byte extent overflow".into()))?;
    let offset = expert
        .checked_mul(bytes)
        .and_then(|n| info.offset.checked_add(n as u64))
        .ok_or_else(|| candle::Error::Msg("LFM2-MoE expert byte offset overflow".into()))?;
    gguf_file::TensorInfo {
        ggml_dtype: info.ggml_dtype,
        shape: (rows, cols).into(),
        offset,
    }
    .read(reader, ct.tensor_data_offset, device)
}

impl ModelWeights {
    /// Retained quantized expert matrices; dense-prefix projections are excluded.
    pub fn quantized_expert_matrices(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| match &layer.ff {
                FeedForward::Sparse { experts, .. } => {
                    experts.iter().map(Expert::quantized_matrices).sum()
                }
                FeedForward::Dense(_) => 0,
            })
            .sum()
    }
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        if !device.is_cpu() && !device.is_cuda() {
            candle::bail!("LFM2-MoE supports CPU and CUDA devices only")
        }
        let metadata = inspect_gguf_metadata(&ct)?;
        let file_len = reader.seek(SeekFrom::End(0))?;
        for role in &metadata.roles {
            let info = ct
                .tensor_infos
                .get(&role.name)
                .ok_or_else(|| candle::Error::Msg("validated tensor disappeared".into()))?;
            let end = ct
                .tensor_data_offset
                .checked_add(info.offset)
                .and_then(|n| n.checked_add(role.bytes as u64))
                .ok_or_else(|| candle::Error::Msg("LFM2-MoE payload range overflow".into()))?;
            if end > file_len {
                candle::bail!("truncated LFM2-MoE tensor payload: {}", role.name)
            }
        }
        let h = metadata.embedding_length;
        let head_dim = h / metadata.shared.head_count;
        let theta = (0..head_dim)
            .step_by(2)
            .map(|i| {
                1f32 / metadata
                    .shared
                    .rope_freq_base
                    .powf(i as f32 / head_dim as f32)
            })
            .collect::<Vec<_>>();
        let theta = Tensor::new(theta.as_slice(), device)?.reshape((1, head_dim / 2))?;
        let positions = Tensor::arange(0u32, metadata.context_length as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((metadata.context_length, 1))?;
        let freqs = positions.matmul(&theta)?;
        let rotary = (freqs.cos()?, freqs.sin()?);
        let embeddings = Arc::new(ct.tensor(reader, "token_embd.weight", device)?);
        let head = if metadata.shared.tied_output {
            match embeddings.dtype() {
                GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => {
                    QMatMul::Tensor(embeddings.dequantize(device)?.to_dtype(DType::F32)?)
                }
                _ => QMatMul::QTensor(embeddings.clone()),
            }
        } else {
            matrix(ct.tensor(reader, "output.weight", device)?)?
        };
        let mut layers = Vec::with_capacity(metadata.block_count);
        for i in 0..metadata.block_count {
            let prefix = format!("blk.{i}");
            let mixer = Mixer::new(&ct, reader, device, i, &metadata.shared, &rotary)?;
            let norm = RmsNorm::new(
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?
                    .dequantize(device)?
                    .to_dtype(DType::F32)?,
                metadata.shared.rms_norm_eps,
            );
            let ffn_norm = RmsNorm::new(
                ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?
                    .dequantize(device)?
                    .to_dtype(DType::F32)?,
                metadata.shared.rms_norm_eps,
            );
            let ff = if i < metadata.leading_dense_block_count {
                FeedForward::Dense(Expert {
                    gate: matrix(ct.tensor(
                        reader,
                        &format!("{prefix}.ffn_gate.weight"),
                        device,
                    )?)?,
                    up: matrix(ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?)?,
                    down: matrix(ct.tensor(
                        reader,
                        &format!("{prefix}.ffn_down.weight"),
                        device,
                    )?)?,
                })
            } else {
                let router = ct
                    .tensor(reader, &format!("{prefix}.ffn_gate_inp.weight"), device)?
                    .dequantize(device)?
                    .to_dtype(DType::F32)?;
                let bias = ct
                    .tensor(reader, &format!("{prefix}.exp_probs_b.bias"), device)?
                    .dequantize(device)?
                    .to_dtype(DType::F32)?;
                let mut experts = Vec::with_capacity(metadata.expert_count);
                for expert in 0..metadata.expert_count {
                    experts.push(Expert {
                        gate: matrix(read_expert(
                            &ct,
                            reader,
                            &format!("{prefix}.ffn_gate_exps.weight"),
                            expert,
                            device,
                        )?)?,
                        up: matrix(read_expert(
                            &ct,
                            reader,
                            &format!("{prefix}.ffn_up_exps.weight"),
                            expert,
                            device,
                        )?)?,
                        down: matrix(read_expert(
                            &ct,
                            reader,
                            &format!("{prefix}.ffn_down_exps.weight"),
                            expert,
                            device,
                        )?)?,
                    });
                }
                FeedForward::Sparse {
                    router,
                    bias,
                    experts,
                    topk: metadata.expert_used_count,
                }
            };
            layers.push(Layer {
                mixer,
                norm,
                ffn_norm,
                ff,
            });
        }
        let norm = RmsNorm::new(
            ct.tensor(reader, "token_embd_norm.weight", device)?
                .dequantize(device)?
                .to_dtype(DType::F32)?,
            metadata.shared.rms_norm_eps,
        );
        Ok(Self {
            embeddings,
            layers,
            norm,
            head,
            metadata,
            position: 0,
            batch: None,
        })
    }
    pub fn metadata(&self) -> &Lfm2MoeGgufMetadata {
        &self.metadata
    }
    pub fn vocab_size(&self) -> usize {
        self.metadata.vocab_size
    }
    pub fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.mixer.clear();
        }
        self.position = 0;
        self.batch = None;
    }
    pub fn embed_tokens(&self, ids: &Tensor) -> Result<Tensor> {
        let (batch, length) = ids.dims2()?;
        if ids.dtype() != DType::U32
            || batch == 0
            || length == 0
            || ids
                .flatten_all()?
                .to_vec1::<u32>()?
                .iter()
                .any(|&id| id as usize >= self.metadata.vocab_size)
        {
            candle::bail!("quantized LFM2-MoE input requires nonempty U32 IDs within vocabulary")
        }
        self.embeddings
            .embedding(&ids.flatten_all()?.contiguous()?)?
            .reshape((batch, length, self.metadata.embedding_length))
    }
    fn chunk(&mut self, embeds: &Tensor, position: usize) -> Result<Tensor> {
        let (_, length, _) = embeds.dims3()?;
        let mask = if length == 1 {
            None
        } else {
            Some(crate::utils::build_causal_mask(
                length,
                position,
                embeds.device(),
            )?)
        };
        let mut x = embeds.clone();
        for layer in &mut self.layers {
            x = (&x
                + layer
                    .mixer
                    .forward(&layer.norm.forward(&x)?, mask.as_ref(), position)?)?;
            x = (&x + layer.ff.forward(&layer.ffn_norm.forward(&x)?)?)?;
        }
        let hidden = self
            .norm
            .forward(&x)?
            .i((.., length - 1, ..))?
            .contiguous()?;
        forward_matrix(&self.head, &hidden)
    }
    pub fn forward_embeds(&mut self, embeds: &Tensor, position: usize) -> Result<Tensor> {
        let result = (|| {
            let (batch, length, h) = embeds.dims3()?;
            let end = position
                .checked_add(length)
                .ok_or_else(|| candle::Error::Msg("LFM2-MoE context overflow".into()))?;
            if batch == 0
                || length == 0
                || h != self.metadata.embedding_length
                || end > self.metadata.context_length
                || (position != 0 && (position != self.position || self.batch != Some(batch)))
                || embeds.dtype() != DType::F32
                || !embeds.device().same_device(&self.embeddings.device())
            {
                candle::bail!("invalid quantized LFM2-MoE input, context, device or cache position")
            }
            if position == 0 {
                self.clear_cache();
            }
            let mut output = None;
            for start in (0..length).step_by(PREFILL_CHUNK) {
                output = Some(self.chunk(
                    &embeds.narrow(1, start, (length - start).min(PREFILL_CHUNK))?,
                    position + start,
                )?);
            }
            self.position = end;
            self.batch = Some(batch);
            output.ok_or_else(|| candle::Error::Msg("empty LFM2-MoE prefill".into()))
        })();
        if result.is_err() {
            self.clear_cache();
        }
        result
    }
    pub fn forward(&mut self, ids: &Tensor, position: usize) -> Result<Tensor> {
        let result = (|| {
            let (batch, length) = ids.dims2()?;
            let end = position
                .checked_add(length)
                .ok_or_else(|| candle::Error::Msg("LFM2-MoE context overflow".into()))?;
            if batch == 0 || length == 0 || end > self.metadata.context_length {
                candle::bail!(
                    "quantized LFM2-MoE input exceeds context before embedding allocation"
                )
            }
            self.forward_embeds(&self.embed_tokens(ids)?, position)
        })();
        if result.is_err() {
            self.clear_cache();
        }
        result
    }
}
