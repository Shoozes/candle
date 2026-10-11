use candle::quantized::{gguf_file, GgmlDType, QMatMul, QTensor};
use candle::{bail, DType, Device, Result, Shape, Tensor};
use candle_nn::{Embedding, Module};
use serde::Serialize;
use std::{fmt, str::FromStr, sync::Arc};

pub(super) const NAMES: [&str; 3] = [
    "token_embd.weight",
    "tok_embeddings.weight",
    "model.embed_tokens.weight",
];

/// Storage used for token lookup. Both modes return F32 activations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Lfm2EmbeddingLoadMode {
    #[default]
    DenseF32,
    /// Retain Q8_0 rows on CPU/CUDA, sharing a tied quantized output head.
    Q8Rows,
}

impl fmt::Display for Lfm2EmbeddingLoadMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::DenseF32 => "dense-f32",
            Self::Q8Rows => "q8-rows",
        })
    }
}

impl FromStr for Lfm2EmbeddingLoadMode {
    type Err = candle::Error;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "dense-f32" => Ok(Self::DenseF32),
            "q8-rows" => Ok(Self::Q8Rows),
            _ => bail!("unsupported LFM2 embedding mode {value:?}; expected dense-f32 or q8-rows"),
        }
    }
}

/// Lookup storage only; excludes temporary row results and all other model tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Lfm2EmbeddingStorage {
    pub mode: Lfm2EmbeddingLoadMode,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub table_bytes: usize,
    pub shared_with_output: bool,
    pub additional_bytes: usize,
}

#[derive(Debug, Clone)]
pub(super) enum TokenEmbedding {
    Dense(Embedding),
    Q8Rows {
        weight: Arc<QTensor>,
        device: Device,
    },
}

fn q8_layout(shape: &Shape, dtype: GgmlDType, device: &Device) -> Result<usize> {
    if !device.is_cpu() && !device.is_cuda() {
        bail!("LFM2 q8-rows embeddings support CPU and CUDA devices")
    }
    if dtype != GgmlDType::Q8_0 {
        bail!("LFM2 q8-rows embeddings require retained Q8_0 storage, received {dtype:?}")
    }
    let (rows, hidden) = shape.dims2()?;
    if rows == 0 || hidden == 0 || !hidden.is_multiple_of(dtype.block_size()) {
        bail!("LFM2 q8-rows embedding dimensions must be positive with block-aligned width")
    }
    if rows.saturating_sub(1) > u32::MAX as usize {
        bail!("LFM2 q8-rows vocabulary exceeds U32 token IDs")
    }
    rows.checked_mul(hidden)
        .and_then(|elements| elements.checked_mul(DType::F32.size_in_bytes()))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| candle::Error::Msg("LFM2 embedding size overflow".into()))?;
    rows.checked_mul(hidden / dtype.block_size())
        .and_then(|blocks| blocks.checked_mul(dtype.type_size()))
        .ok_or_else(|| candle::Error::Msg("LFM2 Q8 embedding storage size overflow".into()))
}

pub(super) fn validate_header(
    content: &gguf_file::Content,
    device: &Device,
    mode: Lfm2EmbeddingLoadMode,
    hidden_size: usize,
    eos: Option<u32>,
) -> Result<()> {
    if mode == Lfm2EmbeddingLoadMode::DenseF32 {
        return Ok(());
    }
    let info = NAMES
        .iter()
        .find_map(|name| content.tensor_infos.get(*name))
        .ok_or_else(|| candle::Error::Msg("cannot find LFM2 token embedding header".into()))?;
    q8_layout(&info.shape, info.ggml_dtype, device)?;
    let (rows, hidden) = info.shape.dims2()?;
    if hidden != hidden_size || eos.is_some_and(|id| id as usize >= rows) {
        bail!("LFM2 q8-rows embedding header disagrees with metadata width or EOS token ID")
    }
    Ok(())
}

impl TokenEmbedding {
    pub(super) fn new(weight: Arc<QTensor>, mode: Lfm2EmbeddingLoadMode) -> Result<Self> {
        let device = weight.device();
        let (_, hidden) = weight.shape().dims2()?;
        match mode {
            Lfm2EmbeddingLoadMode::DenseF32 => Ok(Self::Dense(Embedding::new(
                weight.dequantize(&device)?,
                hidden,
            ))),
            Lfm2EmbeddingLoadMode::Q8Rows => {
                let bytes = q8_layout(weight.shape(), weight.dtype(), &device)?;
                if bytes != weight.storage_size_in_bytes() {
                    bail!("LFM2 q8-rows embedding buffer does not match its declared shape")
                }
                Ok(Self::Q8Rows { weight, device })
            }
        }
    }

    fn shape(&self) -> &Shape {
        match self {
            Self::Dense(table) => table.embeddings().shape(),
            Self::Q8Rows { weight, .. } => weight.shape(),
        }
    }

    pub(super) fn hidden_size(&self) -> usize {
        self.shape().dims()[1]
    }
    pub(super) fn vocab_size(&self) -> usize {
        self.shape().dims()[0]
    }
    pub(super) fn device(&self) -> &Device {
        match self {
            Self::Dense(table) => table.embeddings().device(),
            Self::Q8Rows { device, .. } => device,
        }
    }
    pub(super) fn dtype(&self) -> DType {
        match self {
            Self::Dense(table) => table.embeddings().dtype(),
            Self::Q8Rows { .. } => DType::F32,
        }
    }

    pub(super) fn storage(&self, output: &QMatMul) -> Result<Lfm2EmbeddingStorage> {
        let (vocab_size, hidden_size) = self.shape().dims2()?;
        let (mode, table_bytes, shared_with_output) = match self {
            Self::Dense(table) => (
                Lfm2EmbeddingLoadMode::DenseF32,
                vocab_size
                    .checked_mul(hidden_size)
                    .and_then(|n| n.checked_mul(table.embeddings().dtype().size_in_bytes()))
                    .ok_or_else(|| {
                        candle::Error::Msg("LFM2 embedding byte count overflow".into())
                    })?,
                false,
            ),
            Self::Q8Rows { weight, .. } => (
                Lfm2EmbeddingLoadMode::Q8Rows,
                weight.storage_size_in_bytes(),
                matches!(output, QMatMul::QTensor(head) if Arc::ptr_eq(weight, head)),
            ),
        };
        Ok(Lfm2EmbeddingStorage {
            mode,
            vocab_size,
            hidden_size,
            table_bytes,
            shared_with_output,
            additional_bytes: if shared_with_output { 0 } else { table_bytes },
        })
    }
}

impl Module for TokenEmbedding {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let (weight, device) = match self {
            Self::Dense(table) => return table.forward(ids),
            Self::Q8Rows { weight, device } => (weight, device),
        };
        let mut shape = ids.dims().to_vec();
        shape.push(self.hidden_size());
        let elements = shape
            .iter()
            .try_fold(1usize, |n, &dim| n.checked_mul(dim))
            .and_then(|n| n.checked_mul(DType::F32.size_in_bytes()))
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or_else(|| candle::Error::Msg("LFM2 embedding output size overflow".into()))?;
        // CUDA row gather does not check IDs; validate before moving/dispatching them.
        let flat = ids.flatten_all()?.to_device(&Device::Cpu)?;
        let valid = match ids.dtype() {
            DType::U32 => flat
                .to_vec1::<u32>()?
                .iter()
                .all(|&id| (id as usize) < self.vocab_size()),
            DType::I64 => flat
                .to_vec1::<i64>()?
                .iter()
                .all(|&id| usize::try_from(id).is_ok_and(|id| id < self.vocab_size())),
            dtype => bail!("LFM2 q8-rows token IDs must be U32 or I64, received {dtype:?}"),
        };
        if !valid {
            bail!("LFM2 q8-rows token ID is negative or outside the vocabulary")
        }
        if elements == 0 {
            return Tensor::zeros(shape, DType::F32, device);
        }
        weight.embedding(ids)
    }
}

#[cfg(test)]
mod tests;
