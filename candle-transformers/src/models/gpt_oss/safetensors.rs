//! Explicit, hash-bound loading of local packed GPT-OSS checkpoints.

use super::gguf::{open_retained_file, sha256_reader};
use super::mxfp4::{Mxfp4ExpertOperation, PackedMxfp4};
use super::{
    DenseLinear, GptOssCancellationToken, GptOssConfig, GptOssLayerWeights, GptOssLoadRegistry,
    GptOssLoadedHandle, GptOssWeights,
};
use candle::{bail, DType, Device, Result, Tensor};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

const MAX_HEADER_BYTES: usize = 16 * 1024 * 1024;
const MAX_TOTAL_HEADER_BYTES: usize = 32 * 1024 * 1024;
const MAX_TOTAL_TENSORS: usize = 4096;
const READ_CHUNK_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GptOssFileIdentity {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GptOssSafetensorsLayout {
    Original,
    Transformers,
}

/// All paths and identities are supplied by the caller. No cache search or
/// download is performed. An indexed checkpoint consumes exactly its shards.
#[derive(Debug, Clone)]
pub struct GptOssSafetensorsSource {
    pub layout: GptOssSafetensorsLayout,
    pub config: GptOssFileIdentity,
    pub tokenizer: GptOssFileIdentity,
    pub index: Option<GptOssFileIdentity>,
    pub weights: Vec<GptOssFileIdentity>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

impl TensorInfo {
    fn elements(&self) -> Result<usize> {
        if self.shape.is_empty() || self.shape.len() > 4 || self.shape.contains(&0) {
            bail!("GPT-OSS safetensors requires positive, nonempty tensor shapes");
        }
        self.shape.iter().try_fold(1usize, |count, &dim| {
            count.checked_mul(dim).ok_or_else(|| {
                candle::Error::Msg("GPT-OSS safetensors element count overflowed".into())
            })
        })
    }

    fn dtype(&self) -> Result<DType> {
        match self.dtype.as_str() {
            "U8" => Ok(DType::U8),
            "F32" => Ok(DType::F32),
            "F16" => Ok(DType::F16),
            "BF16" => Ok(DType::BF16),
            other => bail!("unsupported GPT-OSS safetensors dtype {other}"),
        }
    }

    fn bytes(&self) -> Result<usize> {
        self.elements()?
            .checked_mul(self.dtype()?.size_in_bytes())
            .ok_or_else(|| candle::Error::Msg("GPT-OSS safetensors byte count overflowed".into()))
    }
}

struct RetainedFile {
    identity: GptOssFileIdentity,
    file: File,
}

impl RetainedFile {
    fn open(identity: &GptOssFileIdentity, cancel: &GptOssCancellationToken) -> Result<Self> {
        cancel.checkpoint()?;
        if identity.sha256.len() != 64 || !identity.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            bail!("GPT-OSS artifact requires an explicit SHA-256 digest");
        }
        let path = std::fs::canonicalize(&identity.path)?;
        let mut file = open_retained_file(&path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() || metadata.len() != identity.size_bytes {
            bail!("GPT-OSS artifact {:?} size or file kind mismatch", path);
        }
        let sha256 = sha256_reader(&mut file, Some(cancel))?;
        if !sha256.eq_ignore_ascii_case(&identity.sha256) {
            bail!("GPT-OSS artifact {:?} SHA-256 mismatch", path);
        }
        Ok(Self {
            identity: GptOssFileIdentity {
                path,
                size_bytes: metadata.len(),
                sha256,
            },
            file,
        })
    }

    fn bytes(
        &mut self,
        offset: u64,
        count: usize,
        cancel: &GptOssCancellationToken,
    ) -> Result<Vec<u8>> {
        let end = offset
            .checked_add(u64::try_from(count).map_err(|_| {
                candle::Error::Msg("GPT-OSS artifact read length overflowed".into())
            })?)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS artifact read range overflowed".into()))?;
        if end > self.identity.size_bytes {
            bail!("GPT-OSS artifact read is outside the retained file");
        }
        cancel.checkpoint()?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(count).map_err(|error| {
            candle::Error::Msg(format!("GPT-OSS artifact allocation failed: {error}"))
        })?;
        bytes.resize(count, 0);
        self.file.seek(SeekFrom::Start(offset))?;
        for chunk in bytes.chunks_mut(READ_CHUNK_BYTES) {
            cancel.checkpoint()?;
            self.file.read_exact(chunk)?;
        }
        cancel.checkpoint()?;
        Ok(bytes)
    }

    fn json(&mut self, cancel: &GptOssCancellationToken) -> Result<Vec<u8>> {
        let count = usize::try_from(self.identity.size_bytes)
            .map_err(|_| candle::Error::Msg("GPT-OSS JSON artifact is too large".into()))?;
        if count > MAX_HEADER_BYTES {
            bail!("GPT-OSS config/index exceeds the bounded JSON size");
        }
        self.bytes(0, count, cancel)
    }
}

struct Shard {
    retained: RetainedFile,
    tensors: BTreeMap<String, TensorInfo>,
    data_start: u64,
    header_bytes: usize,
}

struct HeaderVisitor {
    tensor_limit: usize,
}

impl<'de> serde::de::Visitor<'de> for HeaderVisitor {
    type Value = BTreeMap<String, TensorInfo>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a bounded safetensors header")
    }

    fn visit_map<M: serde::de::MapAccess<'de>>(
        self,
        mut map: M,
    ) -> std::result::Result<Self::Value, M::Error> {
        let mut tensors = BTreeMap::new();
        let mut metadata_seen = false;
        while let Some(name) = map.next_key::<String>()? {
            if name == "__metadata__" {
                if metadata_seen {
                    return Err(serde::de::Error::custom("duplicate safetensors metadata"));
                }
                metadata_seen = true;
                let _ = map.next_value::<BTreeMap<String, String>>()?;
                continue;
            }
            if name.is_empty() || name.len() > 1024 || tensors.len() >= self.tensor_limit {
                return Err(serde::de::Error::custom(
                    "safetensors tensor name/count exceeds its limit",
                ));
            }
            if tensors.contains_key(&name) {
                return Err(serde::de::Error::custom(
                    "duplicate safetensors tensor name",
                ));
            }
            tensors.insert(name, map.next_value::<TensorInfo>()?);
        }
        Ok(tensors)
    }
}

impl Shard {
    #[cfg(test)]
    fn open(identity: &GptOssFileIdentity, cancel: &GptOssCancellationToken) -> Result<Self> {
        Self::open_bounded(identity, cancel, MAX_TOTAL_HEADER_BYTES, MAX_TOTAL_TENSORS)
    }

    fn open_bounded(
        identity: &GptOssFileIdentity,
        cancel: &GptOssCancellationToken,
        remaining_header_bytes: usize,
        remaining_tensors: usize,
    ) -> Result<Self> {
        let mut retained = RetainedFile::open(identity, cancel)?;
        let mut prefix = [0u8; 8];
        retained.file.seek(SeekFrom::Start(0))?;
        retained.file.read_exact(&mut prefix)?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len == 0 || header_len > MAX_HEADER_BYTES.min(remaining_header_bytes) as u64 {
            bail!("GPT-OSS safetensors header exceeds the bounded size");
        }
        let header = retained.bytes(8, header_len as usize, cancel)?;
        let mut deserializer = serde_json::Deserializer::from_slice(&header);
        let tensors = serde::Deserializer::deserialize_map(
            &mut deserializer,
            HeaderVisitor {
                tensor_limit: remaining_tensors,
            },
        )
        .map_err(candle::Error::wrap)?;
        deserializer.end().map_err(candle::Error::wrap)?;
        if tensors.is_empty() {
            bail!("GPT-OSS safetensors tensor count is outside the bounded range");
        }
        let data_start = 8u64.checked_add(header_len).ok_or_else(|| {
            candle::Error::Msg("GPT-OSS safetensors data offset overflowed".into())
        })?;
        let mut ranges = Vec::new();
        for (name, info) in &tensors {
            let [start, end] = info.data_offsets;
            if end.checked_sub(start) != Some(info.bytes()? as u64) {
                bail!("GPT-OSS tensor {name:?} byte range does not match its shape/dtype");
            }
            ranges.push((start, end));
        }
        ranges.sort_unstable();
        let mut cursor = 0u64;
        for (start, end) in ranges {
            if start != cursor {
                bail!("GPT-OSS safetensors payload contains gaps or overlaps");
            }
            cursor = end;
        }
        if data_start.checked_add(cursor) != Some(retained.identity.size_bytes) {
            bail!("GPT-OSS safetensors payload is truncated or has trailing bytes");
        }
        Ok(Self {
            retained,
            tensors,
            data_start,
            header_bytes: header_len as usize,
        })
    }
}

/// Owns admitted file handles and the exact consumed-file inventory. Expert
/// tensor payloads are read once during assembly and remain packed.
pub struct GptOssSafetensorsArtifact {
    config: GptOssConfig,
    layout: GptOssSafetensorsLayout,
    files: Vec<RetainedFile>,
    shards: Vec<Shard>,
    locations: BTreeMap<String, usize>,
    consumed: BTreeSet<String>,
}

impl GptOssSafetensorsArtifact {
    pub fn open_with_registry(
        source: &GptOssSafetensorsSource,
        registry: &GptOssLoadRegistry,
        cancel: &GptOssCancellationToken,
    ) -> Result<(Self, GptOssLoadedHandle)> {
        cancel.checkpoint()?;
        let identity = std::fs::canonicalize(&source.config.path)?
            .to_string_lossy()
            .into_owned();
        let lease = registry.begin(identity)?;
        let artifact = Self::open(source, cancel)?;
        cancel.checkpoint()?;
        Ok((artifact, lease.commit()?))
    }

    pub fn load_weights_with_cancellation(
        &mut self,
        cancel: &GptOssCancellationToken,
        max_resident_bytes: usize,
    ) -> Result<GptOssWeights> {
        self.consumed.clear();
        self.assemble(cancel, max_resident_bytes)
    }

    pub fn config(&self) -> &GptOssConfig {
        &self.config
    }

    pub fn inventory(&self) -> Vec<GptOssFileIdentity> {
        self.files
            .iter()
            .map(|file| file.identity.clone())
            .chain(
                self.shards
                    .iter()
                    .map(|shard| shard.retained.identity.clone()),
            )
            .collect()
    }

    fn open(source: &GptOssSafetensorsSource, cancel: &GptOssCancellationToken) -> Result<Self> {
        if source.weights.is_empty() || source.weights.len() > 128 {
            bail!("GPT-OSS packed checkpoint requires 1..=128 explicit weight shards");
        }
        if source.weights.len() > 1 && source.index.is_none() {
            bail!("GPT-OSS multi-shard checkpoint requires its safetensors index");
        }
        let mut config_file = RetainedFile::open(&source.config, cancel)?;
        let config: GptOssConfig =
            serde_json::from_slice(&config_file.json(cancel)?).map_err(|error| {
                candle::Error::Msg(format!("invalid GPT-OSS checkpoint config: {error}"))
            })?;
        config.validate()?;
        let tokenizer = RetainedFile::open(&source.tokenizer, cancel)?;
        let mut files = vec![config_file, tokenizer];
        let mut shards = Vec::new();
        let mut locations = BTreeMap::new();
        let mut shard_names = BTreeMap::new();
        let mut remaining_header_bytes = MAX_TOTAL_HEADER_BYTES;
        let mut remaining_tensors = MAX_TOTAL_TENSORS;
        for identity in &source.weights {
            let shard =
                Shard::open_bounded(identity, cancel, remaining_header_bytes, remaining_tensors)?;
            remaining_header_bytes = remaining_header_bytes
                .checked_sub(shard.header_bytes)
                .ok_or_else(|| {
                    candle::Error::Msg("GPT-OSS aggregate header budget exceeded".into())
                })?;
            remaining_tensors = remaining_tensors
                .checked_sub(shard.tensors.len())
                .ok_or_else(|| {
                    candle::Error::Msg("GPT-OSS aggregate tensor budget exceeded".into())
                })?;
            let name = shard
                .retained
                .identity
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| candle::Error::Msg("GPT-OSS shard filename must be UTF-8".into()))?
                .to_owned();
            if shard_names.insert(name, shards.len()).is_some() {
                bail!("GPT-OSS checkpoint has duplicate shard filenames");
            }
            for name in shard.tensors.keys() {
                if locations.insert(name.clone(), shards.len()).is_some() {
                    bail!("GPT-OSS tensor {name:?} occurs in multiple shards");
                }
            }
            shards.push(shard);
        }
        if let Some(identity) = &source.index {
            #[derive(Deserialize)]
            struct Index {
                weight_map: BTreeMap<String, String>,
            }
            let mut file = RetainedFile::open(identity, cancel)?;
            let index: Index = serde_json::from_slice(&file.json(cancel)?).map_err(|error| {
                candle::Error::Msg(format!("invalid GPT-OSS safetensors index: {error}"))
            })?;
            if index.weight_map.len() != locations.len() {
                bail!("GPT-OSS safetensors index does not match the tensor inventory");
            }
            for (name, shard_name) in index.weight_map {
                let shard_index = shard_names.get(&shard_name).ok_or_else(|| {
                    candle::Error::Msg(format!("GPT-OSS index names absent shard {shard_name:?}"))
                })?;
                if locations.get(&name) != Some(shard_index) {
                    bail!("GPT-OSS index has mismatched tensor {name:?}");
                }
            }
            files.push(file);
        }
        Ok(Self {
            config,
            layout: source.layout,
            files,
            shards,
            locations,
            consumed: BTreeSet::new(),
        })
    }

    fn tensor(
        &mut self,
        name: &str,
        shape: &[usize],
        cancel: &GptOssCancellationToken,
    ) -> Result<(DType, Vec<u8>)> {
        let shard_index = *self
            .locations
            .get(name)
            .ok_or_else(|| candle::Error::Msg(format!("missing GPT-OSS tensor {name:?}")))?;
        let shard = self
            .shards
            .get_mut(shard_index)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS shard index is out of range".into()))?;
        let info = shard
            .tensors
            .get(name)
            .ok_or_else(|| candle::Error::Msg(format!("missing GPT-OSS tensor {name:?}")))?;
        if info.shape != shape {
            bail!(
                "GPT-OSS tensor {name:?} shape {:?}, expected {shape:?}",
                info.shape
            );
        }
        let dtype = info.dtype()?;
        let count = info.bytes()?;
        let offset = shard
            .data_start
            .checked_add(info.data_offsets[0])
            .ok_or_else(|| candle::Error::Msg("GPT-OSS tensor read offset overflowed".into()))?;
        let bytes = shard.retained.bytes(offset, count, cancel)?;
        self.consumed.insert(name.to_owned());
        Ok((dtype, bytes))
    }

    fn dense(
        &mut self,
        name: &str,
        shape: &[usize],
        cancel: &GptOssCancellationToken,
    ) -> Result<Vec<f32>> {
        let shard_index = *self
            .locations
            .get(name)
            .ok_or_else(|| candle::Error::Msg(format!("missing GPT-OSS tensor {name:?}")))?;
        let shard = self
            .shards
            .get_mut(shard_index)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS shard index is out of range".into()))?;
        let info = shard
            .tensors
            .get(name)
            .ok_or_else(|| candle::Error::Msg(format!("missing GPT-OSS tensor {name:?}")))?;
        if info.shape != shape {
            bail!(
                "GPT-OSS tensor {name:?} shape mismatch: {:?}, expected {shape:?}",
                info.shape
            );
        }
        let dtype = info.dtype()?;
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            bail!("GPT-OSS dense tensor {name:?} must be F32, F16 or BF16");
        }
        let elements = info.elements()?;
        let mut offset = shard
            .data_start
            .checked_add(info.data_offsets[0])
            .ok_or_else(|| candle::Error::Msg("GPT-OSS dense read offset overflowed".into()))?;
        let mut values = Vec::new();
        values.try_reserve_exact(elements).map_err(|error| {
            candle::Error::Msg(format!("GPT-OSS dense allocation failed: {error}"))
        })?;
        let element_bytes = dtype.size_in_bytes();
        while values.len() < elements {
            let count = (elements - values.len()).min(READ_CHUNK_BYTES / element_bytes);
            let byte_count = count
                .checked_mul(element_bytes)
                .ok_or_else(|| candle::Error::Msg("GPT-OSS dense chunk size overflowed".into()))?;
            let bytes = shard.retained.bytes(offset, byte_count, cancel)?;
            let chunk = Tensor::from_raw_buffer(&bytes, dtype, &[count], &Device::Cpu)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            values.extend(chunk);
            offset = offset
                .checked_add(byte_count as u64)
                .ok_or_else(|| candle::Error::Msg("GPT-OSS dense read offset overflowed".into()))?;
        }
        cancel.checkpoint()?;
        self.consumed.insert(name.to_owned());
        Ok(values)
    }

    fn packed(
        &mut self,
        blocks_name: &str,
        scales_name: &str,
        shape: &[usize; 3],
        cancel: &GptOssCancellationToken,
    ) -> Result<PackedMxfp4> {
        let [experts, rows, cols] = *shape;
        if !cols.is_multiple_of(32) {
            bail!("GPT-OSS packed input width must be divisible by 32");
        }
        let (blocks_dtype, blocks) =
            self.tensor(blocks_name, &[experts, rows, cols / 32, 16], cancel)?;
        let (scales_dtype, scales) =
            self.tensor(scales_name, &[experts, rows, cols / 32], cancel)?;
        if blocks_dtype != DType::U8 || scales_dtype != DType::U8 {
            bail!("GPT-OSS packed blocks and scales must be U8");
        }
        PackedMxfp4::from_parts(shape, blocks, scales)
    }

    pub fn weight_resident_bytes(&self) -> Result<usize> {
        let mut resident = 0usize;
        for shard in &self.shards {
            for info in shard.tensors.values() {
                let bytes = info
                    .elements()?
                    .checked_mul(if info.dtype()? == DType::U8 { 1 } else { 4 })
                    .ok_or_else(|| {
                        candle::Error::Msg("GPT-OSS weight residency overflowed".into())
                    })?;
                resident = resident.checked_add(bytes).ok_or_else(|| {
                    candle::Error::Msg("GPT-OSS weight residency overflowed".into())
                })?;
            }
        }
        Ok(resident)
    }

    fn assemble(
        &mut self,
        cancel: &GptOssCancellationToken,
        budget: usize,
    ) -> Result<GptOssWeights> {
        let resident = self.weight_resident_bytes()?;
        if resident > budget {
            bail!("GPT-OSS assembled weights require {resident} bytes, limit is {budget}");
        }
        let c = self.config.clone();
        let original = self.layout == GptOssSafetensorsLayout::Original;
        let embedding = self.dense(
            if original {
                "embedding.weight"
            } else {
                "model.embed_tokens.weight"
            },
            &[c.vocab_size, c.hidden_size],
            cancel,
        )?;
        let norm = self.dense(
            if original {
                "norm.scale"
            } else {
                "model.norm.weight"
            },
            &[c.hidden_size],
            cancel,
        )?;
        let head = self.dense(
            if original {
                "unembedding.weight"
            } else {
                "lm_head.weight"
            },
            &[c.vocab_size, c.hidden_size],
            cancel,
        )?;
        let head = DenseLinear::new(c.vocab_size, c.hidden_size, head, None)?;
        let q = c
            .num_attention_heads
            .checked_mul(c.head_dim)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS Q width overflowed".into()))?;
        let kv = c
            .num_key_value_heads
            .checked_mul(c.head_dim)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS KV width overflowed".into()))?;
        let qkv = kv
            .checked_mul(2)
            .and_then(|v| q.checked_add(v))
            .ok_or_else(|| candle::Error::Msg("GPT-OSS QKV width overflowed".into()))?;
        let up = c
            .intermediate_size
            .checked_mul(2)
            .ok_or_else(|| candle::Error::Msg("GPT-OSS expert width overflowed".into()))?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(c.num_hidden_layers)
            .map_err(|error| {
                candle::Error::Msg(format!("GPT-OSS layer allocation failed: {error}"))
            })?;
        for layer in 0..c.num_hidden_layers {
            cancel.checkpoint()?;
            let prefix = if original {
                format!("block.{layer}")
            } else {
                format!("model.layers.{layer}")
            };
            let name = |a: &str, b: &str| format!("{prefix}.{}", if original { a } else { b });
            let attention_norm = self.dense(
                &name("attn.norm.scale", "input_layernorm.weight"),
                &[c.hidden_size],
                cancel,
            )?;
            let (qkv_weights, qkv_bias) = if original {
                (
                    self.dense(&name("attn.qkv.weight", ""), &[qkv, c.hidden_size], cancel)?,
                    self.dense(&name("attn.qkv.bias", ""), &[qkv], cancel)?,
                )
            } else {
                let mut weights = Vec::new();
                let mut biases = Vec::new();
                for (projection, rows) in [("q", q), ("k", kv), ("v", kv)] {
                    let values = self.dense(
                        &name("", &format!("self_attn.{projection}_proj.weight")),
                        &[rows, c.hidden_size],
                        cancel,
                    )?;
                    weights.try_reserve_exact(values.len()).map_err(|error| {
                        candle::Error::Msg(format!("GPT-OSS QKV allocation failed: {error}"))
                    })?;
                    weights.extend(values);
                    let values = self.dense(
                        &name("", &format!("self_attn.{projection}_proj.bias")),
                        &[rows],
                        cancel,
                    )?;
                    biases.try_reserve_exact(values.len()).map_err(|error| {
                        candle::Error::Msg(format!("GPT-OSS QKV bias allocation failed: {error}"))
                    })?;
                    biases.extend(values);
                }
                (weights, biases)
            };
            let qkv_linear = DenseLinear::new(qkv, c.hidden_size, qkv_weights, Some(qkv_bias))?;
            let attention_out = DenseLinear::new(
                c.hidden_size,
                q,
                self.dense(
                    &name("attn.out.weight", "self_attn.o_proj.weight"),
                    &[c.hidden_size, q],
                    cancel,
                )?,
                Some(self.dense(
                    &name("attn.out.bias", "self_attn.o_proj.bias"),
                    &[c.hidden_size],
                    cancel,
                )?),
            )?;
            let sinks = self.dense(
                &name("attn.sinks", "self_attn.sinks"),
                &[c.num_attention_heads],
                cancel,
            )?;
            let moe_norm = self.dense(
                &name("mlp.norm.scale", "post_attention_layernorm.weight"),
                &[c.hidden_size],
                cancel,
            )?;
            let router = DenseLinear::new(
                c.num_experts,
                c.hidden_size,
                self.dense(
                    &name("mlp.gate.weight", "mlp.router.weight"),
                    &[c.num_experts, c.hidden_size],
                    cancel,
                )?,
                Some(self.dense(
                    &name("mlp.gate.bias", "mlp.router.bias"),
                    &[c.num_experts],
                    cancel,
                )?),
            )?;
            let mlp1 = self.packed(
                &name("mlp.mlp1_weight.blocks", "mlp.experts.gate_up_proj_blocks"),
                &name("mlp.mlp1_weight.scales", "mlp.experts.gate_up_proj_scales"),
                &[c.num_experts, up, c.hidden_size],
                cancel,
            )?;
            let mlp1_bias = self.dense(
                &name("mlp.mlp1_bias", "mlp.experts.gate_up_proj_bias"),
                &[c.num_experts, up],
                cancel,
            )?;
            let mlp2 = self.packed(
                &name("mlp.mlp2_weight.blocks", "mlp.experts.down_proj_blocks"),
                &name("mlp.mlp2_weight.scales", "mlp.experts.down_proj_scales"),
                &[c.num_experts, c.hidden_size, c.intermediate_size],
                cancel,
            )?;
            let mlp2_bias = self.dense(
                &name("mlp.mlp2_bias", "mlp.experts.down_proj_bias"),
                &[c.num_experts, c.hidden_size],
                cancel,
            )?;
            let experts =
                Mxfp4ExpertOperation::new(mlp1, mlp1_bias, mlp2, mlp2_bias, c.swiglu_limit)?;
            layers.push(GptOssLayerWeights::new(
                &c,
                attention_norm,
                qkv_linear,
                attention_out,
                sinks,
                moe_norm,
                router,
                experts,
            )?);
        }
        if self.consumed.len() != self.locations.len() {
            bail!("GPT-OSS checkpoint contains unconsumed weight tensors");
        }
        cancel.checkpoint()?;
        GptOssWeights::new(&c, embedding, layers, norm, head)
    }
}

/// Load one explicitly identified packed checkpoint, holding the registry
/// lease with the assembled weights through CPU/CUDA model ownership.
pub fn load_gpt_oss_safetensors(
    source: &GptOssSafetensorsSource,
    registry: &GptOssLoadRegistry,
    cancel: &GptOssCancellationToken,
    max_resident_bytes: usize,
) -> Result<(GptOssSafetensorsArtifact, GptOssWeights, GptOssLoadedHandle)> {
    cancel.checkpoint()?;
    let (mut artifact, handle) =
        GptOssSafetensorsArtifact::open_with_registry(source, registry, cancel)?;
    let mut weights = artifact.load_weights_with_cancellation(cancel, max_resident_bytes)?;
    cancel.checkpoint()?;
    weights.attach_load_handle(handle.clone());
    Ok((artifact, weights, handle))
}

#[cfg(test)]
#[path = "safetensors_tests.rs"]
mod tests;
